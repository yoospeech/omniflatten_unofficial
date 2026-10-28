"""Unmodified causal LLM with a unified text/speech vocabulary (paper §3)."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from omniflatten.tokens import SPECIAL_TOKENS, TokenLayout

@dataclass
class OmniFlattenConfig:
    backbone_name_or_path: str = 'Qwen/Qwen2-0.5B'
    num_speech_tokens: int = 4096
    freeze_backbone: bool = False
    torch_dtype: object = None


class OmniFlattenModel(nn.Module):
    def __init__(self, config=None, backbone=None):
        super().__init__()
        self.config = config or OmniFlattenConfig()
        kwargs = {} if self.config.torch_dtype is None else {'torch_dtype': self.config.torch_dtype}
        self.backbone = backbone if backbone is not None else AutoModelForCausalLM.from_pretrained(
            self.config.backbone_name_or_path, **kwargs)
        self.layout = None
        if self.config.freeze_backbone:
            raise ValueError('Paper-aligned training fine-tunes the entire backbone; freeze_backbone is unsupported')

    def prepare_tokenizer(self, tokenizer):
        # Qwen's embedding count can exceed len(tokenizer). Never infer IDs
        # from model.config.vocab_size or assume the added IDs start there.
        raw = tokenizer.init_kwargs.get('omniflatten_layout')
        if raw is not None:
            self.layout = TokenLayout(**raw)
            if self.layout.num_speech_tokens != self.config.num_speech_tokens:
                raise ValueError('Speech vocabulary size mismatch')
            if len(tokenizer) != self.layout.speech_offset + self.layout.num_speech_tokens:
                raise ValueError('Saved tokenizer vocabulary changed')
            if self.backbone.config.vocab_size != len(tokenizer):
                raise ValueError('Load the matching OmniFlatten model checkpoint')
        else:
            text_size = len(tokenizer)
            tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
            speech_offset = len(tokenizer)
            speech_tokens = [f'<|speech_{i}|>' for i in range(self.config.num_speech_tokens)]
            if tokenizer.add_tokens(speech_tokens) != self.config.num_speech_tokens:
                raise ValueError('Speech token strings already present without saved layout')
            self.layout = TokenLayout(text_size, speech_offset, self.config.num_speech_tokens,
                                      {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS})
            self.backbone.resize_token_embeddings(len(tokenizer))
            self.backbone.tie_weights()
            # Initialize every newly assigned ID, including IDs that previously
            # occupied Qwen's unused padded embedding rows.
            with torch.no_grad():
                weights = [self.backbone.get_input_embeddings().weight,
                           self.backbone.get_output_embeddings().weight]
                seen = set()
                for weight in weights:
                    if weight.data_ptr() not in seen:
                        weight[text_size:] = weight[:text_size].mean(0)
                        seen.add(weight.data_ptr())
            tokenizer.init_kwargs['omniflatten_layout'] = asdict(self.layout)
        self.special_token_ids = self.layout.special_ids
        return tokenizer

    def get_speech_token_offset(self):
        if self.layout is None:
            raise ValueError('Prepare the tokenizer first')
        return self.layout.speech_offset

    def speech_codebook_to_vocab_id(self, ids):
        if torch.any((ids < 0) | (ids >= self.config.num_speech_tokens)):
            raise ValueError('Invalid codebook IDs')
        return ids + self.get_speech_token_offset()

    def vocab_id_to_speech_codebook(self, ids):
        codes = ids - self.get_speech_token_offset()
        if torch.any((codes < 0) | (codes >= self.config.num_speech_tokens)):
            raise ValueError('Not speech vocabulary IDs')
        return codes

    def forward(self, input_ids, attention_mask=None, labels=None, loss_mask=None, loss_weight=None,
                past_key_values=None, use_cache=None, **kwargs):
        if labels is None:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                   past_key_values=past_key_values,
                                   use_cache=True if use_cache is None else use_cache,
                                   return_dict=True, **kwargs)
            return CausalLMOutputWithPast(logits=output.logits,
                                         past_key_values=output.past_key_values)
        if past_key_values is not None or use_cache:
            raise ValueError('Cached training with labels is not supported')
        # Do not create a (batch, sequence, vocabulary) logits tensor. Qwen2's
        # vocabulary exceeds 150k, so this can consume gigabytes by itself.
        output = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask,
                                     use_cache=False, return_dict=True, **kwargs)
        targets = labels[:, 1:].clone()
        if attention_mask is not None:
            targets.masked_fill_(attention_mask[:, 1:] == 0, -100)
        if loss_mask is not None:
            targets.masked_fill_(loss_mask[:, 1:] == 0, -100)
        hidden = output.last_hidden_state[:, :-1, :]
        valid = targets != -100
        selected_hidden = hidden[valid]
        selected_targets = targets[valid]
        selected_weights = loss_weight[:, 1:][valid] if loss_weight is not None else None
        head = self.backbone.get_output_embeddings()

        def chunk_loss(states, target, weights=None):
            logits = head(states).float()
            losses = F.cross_entropy(logits, target, reduction='none')
            return losses.sum() if weights is None else (losses * weights).sum()

        # Checkpoint each small projection: logits are recomputed during
        # backward instead of retained for all chunks simultaneously.
        loss = hidden.sum() * 0.0
        for start in range(0, selected_targets.numel(), 32):
            states = selected_hidden[start:start+32]
            target = selected_targets[start:start+32]
            weights = None if selected_weights is None else selected_weights[start:start+32]
            if torch.is_grad_enabled():
                loss = loss + checkpoint(chunk_loss, states, target, weights, use_reentrant=False)
            else:
                loss = loss + chunk_loss(states, target, weights)
        denominator = valid.sum() if selected_weights is None else selected_weights.sum()
        loss = loss / denominator.clamp_min(1)
        return CausalLMOutputWithPast(loss=loss, logits=None)

    def save_pretrained(self, path, tokenizer):
        if self.layout is None:
            raise ValueError('No token layout')
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path)
        tokenizer.save_pretrained(path)
        (path / 'omniflatten.json').write_text(json.dumps(asdict(self.layout), indent=2))

    @classmethod
    def from_pretrained(cls, path):
        from transformers import AutoTokenizer
        layout = TokenLayout(**json.loads((Path(path) / 'omniflatten.json').read_text()))
        model = cls(OmniFlattenConfig(str(path), layout.num_speech_tokens))
        # Transformers 4.57.6 misclassifies locally saved Qwen configs as
        # Mistral; preserve Qwen's original regex, do not apply that repair.
        tokenizer_kwargs = {'fix_mistral_regex': False} if model.backbone.config.model_type == 'qwen2' else {}
        tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
        model.prepare_tokenizer(tokenizer)
        if model.layout != layout:
            raise ValueError('Model/tokenizer metadata mismatch')
        return model, tokenizer

    def get_num_params(self, only_trainable=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)
