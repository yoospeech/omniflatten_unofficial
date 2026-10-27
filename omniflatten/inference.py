"""Cached autoregressive inference using the same formats as training.

stream_generate consumes an iterable of user codebook chunks and yields packets
immediately. full_duplex_generate is explicitly an offline audio-file wrapper.
"""
from dataclasses import dataclass
from itertools import chain
import time
import numpy as np
import torch
from omniflatten.tokens import layout_from_tokenizer
from omniflatten.data.data_utils import load_audio

@dataclass
class InferenceConfig:
    speech_chunk_size: int = 10
    text_chunk_size: int = 2
    mode: str = '3a'
    max_new_tokens: int = 4096
    max_response_chunks: int = 50
    stop_after_silent_chunks: int = 5
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    sample_rate: int = 16000
    device: str = 'cpu'

class OmniFlattenInference:
    def __init__(self, model, speech_tokenizer, speech_detokenizer, text_tokenizer, config=None):
        self.model, self.speech_tokenizer, self.speech_detokenizer = model, speech_tokenizer, speech_detokenizer
        self.text_tokenizer = text_tokenizer
        self.config = config or InferenceConfig()
        if self.config.mode not in ('3a','3b'):
            raise ValueError('mode must be 3a or 3b')
        if min(self.config.speech_chunk_size, self.config.text_chunk_size,
               self.config.max_new_tokens, self.config.stop_after_silent_chunks) <= 0:
            raise ValueError('Chunk sizes, generation budget and silence threshold must be positive')
        if self.config.max_response_chunks < 0 or not 0 < self.config.top_p <= 1 or self.config.temperature < 0:
            raise ValueError('Invalid inference configuration')
        self.layout = layout_from_tokenizer(text_tokenizer)
        if getattr(model, 'layout', None) != self.layout:
            raise ValueError('Model/tokenizer layouts differ')
        self.special_ids = {t.strip('[]'):i for t,i in self.layout.special_ids.items()}
        self.device = torch.device(self.config.device)
        self.model.to(self.device).eval()
        self.reset()

    def reset(self):
        self.cache, self.logits, self.context_length = None, None, 0

    @torch.inference_mode()
    def _feed(self, ids):
        if not ids:
            raise ValueError('Cannot feed empty context')
        limit = getattr(self.model.backbone.config, 'max_position_embeddings', None)
        if limit is not None and self.context_length+len(ids)>limit:
            raise ValueError('Context limit exceeded; reset between sessions')
        out = self.model(input_ids=torch.tensor([ids], device=self.device),
                         past_key_values=self.cache, use_cache=True)
        self.logits = out.logits[0,-1].float()
        self.cache = out.past_key_values
        self.context_length += len(ids)

    def _allowed(self, modality, stop=None, silent=False):
        allowed = torch.zeros(self.model.backbone.config.vocab_size, dtype=torch.bool, device=self.device)
        if modality=='speech':
            a = self.layout.speech_offset
            allowed[a:a+self.layout.num_speech_tokens] = True
        else:
            allowed[:self.layout.text_vocab_size] = True
            # Text content cannot emit backbone EOS/control tokens.
            for idx in self.text_tokenizer.all_special_ids:
                allowed[idx] = False
        if stop is not None:
            allowed[stop] = True
        if silent:
            allowed[self.special_ids['SILENT_SPEECH' if modality=='speech' else 'SILENT_TEXT']] = True
        return allowed

    @torch.inference_mode()
    def _generate(self, n, modality, stop=None, silent=False):
        if self.logits is None:
            raise ValueError('Feed a nonempty prompt before generation')
        allowed = self._allowed(modality, stop, silent)
        generated = []
        for _ in range(n):
            logits = self.logits.clone().masked_fill(~allowed, -torch.inf)
            if self.config.temperature==0:
                token = logits.argmax().item()
            else:
                logits /= self.config.temperature
                k = min(self.config.top_k, int(allowed.sum()))
                if k>0:
                    logits[logits < torch.topk(logits,k).values[-1]] = -torch.inf
                sorted_logits, indices = logits.sort(descending=True)
                remove = sorted_logits.softmax(-1).cumsum(-1) > self.config.top_p
                remove[1:] = remove[:-1].clone(); remove[0] = False
                logits[indices[remove]] = -torch.inf
                token = torch.multinomial(logits.softmax(-1),1).item()
            generated.append(token)
            # Feed even the final/stop token: cache always covers all emitted IDs.
            self._feed([token])
            if token==stop:
                break
        return generated

    def _tokenise_audio(self, path):
        if self.speech_tokenizer is None:
            raise ValueError('An audio codec is required for file input')
        return self.layout.speech_ids(self.speech_tokenizer.encode(load_audio(path, self.config.sample_rate)))

    def _detokenise_speech(self, ids):
        if self.speech_detokenizer is None:
            raise ValueError('An audio decoder is required')
        codes = self.layout.codebook_ids(ids)
        audio = self.speech_detokenizer.decode(codes)
        return torch.as_tensor(audio).detach().cpu().float().numpy().reshape(-1)

    def _decode_timeline(self, ids):
        """Keep explicit silence in place rather than collapsing it in time."""
        codec = self.speech_detokenizer
        if codec is None:
            return None
        rate = codec.output_sample_rate
        token_rate = codec.token_rate
        out = np.zeros(round(len(ids)*rate/token_rate), dtype=np.float32)
        silent = self.special_ids['SILENT_SPEECH']
        i = 0
        while i<len(ids):
            if ids[i]==silent:
                i+=1; continue
            j=i+1
            while j<len(ids) and ids[j]!=silent:
                j+=1
            audio = self._detokenise_speech(ids[i:j])
            start, end = round(i*rate/token_rate), round(j*rate/token_rate)
            length = min(end-start,len(audio))
            out[start:start+length] = audio[:length]
            i=j
        return out

    def stream_generate(self, user_speech_chunks, decode_audio=False):
        """Yield each response chunk before requesting the next user chunk.

        Input chunks are raw codebook IDs (not unified IDs). Short chunks are
        padded to preserve the training layout. After iterator exhaustion,
        bounded silence chunks allow the assistant to complete its response.
        decode_audio=True decodes chunks independently; continuous vocoder
        overlap/cache is a separate deployment concern.
        """
        self.reset()
        cs, ct = self.config.speech_chunk_size, self.config.text_chunk_size
        silent = self.special_ids['SILENT_SPEECH']
        budget = 0; silent_run = 0
        for codes in chain(user_speech_chunks, [None]*self.config.max_response_chunks):
            response_size = cs+(ct if self.config.mode=='3a' else 0)
            if budget+response_size > self.config.max_new_tokens:
                break
            flushing = codes is None
            user = [silent]*cs if flushing else self.layout.speech_ids(codes)
            if not 0<len(user)<=cs:
                raise ValueError(f'User chunk must contain 1..{cs} codebook IDs')
            user += [silent]*(cs-len(user))
            start = time.perf_counter()
            self._feed(user)
            text = self._generate(ct, 'text', silent=True) if self.config.mode=='3a' else []
            speech = self._generate(cs, 'speech', silent=True)
            budget += response_size
            audio = self._decode_timeline(speech) if decode_audio else None
            yield {'text_tokens':text, 'speech_tokens':speech, 'audio':audio,
                   'latency':time.perf_counter()-start, 'flushing':flushing}
            silent_run = silent_run+1 if all(t==silent for t in speech) else 0
            if flushing and silent_run>=self.config.stop_after_silent_chunks:
                break

    def full_duplex_generate(self, user_audio_path):
        start = time.perf_counter()
        codes = self.layout.codebook_ids(self._tokenise_audio(user_audio_path))
        cs = self.config.speech_chunk_size
        packets = list(self.stream_generate((codes[i:i+cs] for i in range(0,len(codes),cs))))
        text = [t for p in packets for t in p['text_tokens']]
        speech = [t for p in packets for t in p['speech_tokens']]
        audio = self._decode_timeline(speech)
        return {'assistant_audio':audio, 'assistant_text':self.text_tokenizer.decode(
                    [t for t in text if t!=self.special_ids['SILENT_TEXT']], skip_special_tokens=True),
                'assistant_text_tokens':text, 'assistant_speech_tokens':speech,
                'latency_per_chunk':[p['latency'] for p in packets],
                'total_latency':time.perf_counter()-start, 'num_chunks':len(packets)}

    def asr(self, audio_path):
        self.reset(); s=self.special_ids
        self._feed([s['ASR'],s['SOS']]+self._tokenise_audio(audio_path)+[s['EOS'],s['SOT']])
        ids = self._generate(self.config.max_new_tokens, 'text', stop=s['EOT'])
        return self.text_tokenizer.decode([i for i in ids if i!=s['EOT']], skip_special_tokens=True)

    def tts(self, text):
        self.reset(); s=self.special_ids
        self._feed([s['TTS'],s['SOT']]+self.text_tokenizer.encode(text,add_special_tokens=False)+[s['EOT'],s['SOS']])
        ids = self._generate(self.config.max_new_tokens, 'speech', stop=s['EOS'])
        return self._detokenise_speech([i for i in ids if i!=s['EOS']])

    def half_duplex_generate(self, user_audio_path):
        self.reset(); s=self.special_ids
        self._feed([s['SOS']]+self._tokenise_audio(user_audio_path)+[s['EOS'],s['SOT']])
        user_text = self._generate(self.config.max_new_tokens, 'text', stop=s['EOT'])
        if user_text[-1]!=s['EOT']:
            raise RuntimeError('User transcript did not finish within generation budget')
        self._feed([s['SOT']])
        text = self._generate(self.config.max_new_tokens, 'text', stop=s['EOT'])
        if text[-1]!=s['EOT']:
            raise RuntimeError('Assistant text did not finish within generation budget')
        self._feed([s['SOS']])
        speech = self._generate(self.config.max_new_tokens, 'speech', stop=s['EOS'])
        return {'assistant_text':self.text_tokenizer.decode(text[:-1],skip_special_tokens=True),
                'assistant_audio':self._detokenise_speech([i for i in speech if i!=s['EOS']]),
                'user_text':self.text_tokenizer.decode(user_text[:-1],skip_special_tokens=True)}
