import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from omniflatten.model import OmniFlattenModel, OmniFlattenConfig
from omniflatten.tokens import normalize_codes, layout_from_tokenizer
from omniflatten.data.modality_alignment_dataset import ModalityAlignmentDataset
from omniflatten.data.dialogue_dataset import HalfDuplexDialogueDataset, FullDuplexThreeStreamDataset, FullDuplexTwoStreamDataset
from omniflatten.data.formatting import collate, aligned_output
from omniflatten.inference import OmniFlattenInference, InferenceConfig
from omniflatten.data.simulate_dialogue import InteractionSimulator, DialogueSimulationPipeline, PipelineConfig
from omniflatten.speech_tokenizer import CosyVoiceCodec


def tiny_model():
    raw=Tokenizer(WordLevel({'hello':0,'world':1,'yes':2,'no':3,'[UNK]':4},unk_token='[UNK]'))
    raw.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]')
    backbone=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=512,
        tie_word_embeddings=True,attention_dropout=0.0))
    model=OmniFlattenModel(OmniFlattenConfig(num_speech_tokens=16),backbone=backbone)
    model.prepare_tokenizer(tokenizer)
    return model,tokenizer

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.model,self.tokenizer=tiny_model()
        self.s={k.strip('[]'):v for k,v in self.model.layout.special_ids.items()}

    def tearDown(self):
        self.temp.cleanup()

    def write(self,record,name='manifest.jsonl'):
        (self.root/name).write_text(json.dumps(record)+'\n')

    def test_vocabulary_and_checkpoint_roundtrip(self):
        layout=self.model.layout
        self.assertEqual(layout.speech_offset,13)
        self.assertEqual(layout.speech_ids(([0,15],0.2)),[13,28])
        self.assertEqual(normalize_codes(torch.tensor([[1,2]])),[1,2])
        with self.assertRaises(ValueError): layout.speech_ids([16])
        self.assertIs(self.model.backbone.get_input_embeddings().weight,self.model.backbone.get_output_embeddings().weight)
        self.model.save_pretrained(self.root,self.tokenizer)
        loaded,tok=OmniFlattenModel.from_pretrained(self.root)
        self.assertEqual(loaded.layout,layout_from_tokenizer(tok))
        self.model.eval(); loaded.eval()
        ids=torch.tensor([[0,1,13]])
        torch.testing.assert_close(self.model(ids).logits,loaded(ids).logits)

    def test_stage1_golden_sequences(self):
        self.write({'speech_tokens':[0,1], 'text':'hello world'})
        ds=ModalityAlignmentDataset(self.root,self.tokenizer,None)
        s=self.s
        expected=[s['ASR'],s['SOS'],13,14,s['EOS'],s['SOT'],0,1,s['EOT']]
        self.assertEqual(ds[0]['input_ids'].tolist(),expected)
        self.assertEqual(ds[0]['labels'].tolist(),[-100]*5+[s['SOT'],0,1,s['EOT']])
        self.assertEqual(ds[0]['task_id'].item(),0)
        self.assertEqual(ds[1]['input_ids'].tolist(),[s['TTS'],s['SOT'],0,1,s['EOT'],s['SOS'],13,14,s['EOS']])
        self.assertEqual(ds[1]['task_id'].item(),1)
        self.assertEqual(len(ds),2)

    def test_half_duplex_user_mask_and_zero_token_attention(self):
        self.write({'turns':[{'role':'user','text':'hello','speech_tokens':[0]},
                             {'role':'assistant','text':'yes','speech_tokens':[1]}]},'dialogue_manifest.jsonl')
        item=HalfDuplexDialogueDataset(self.root,self.tokenizer)[0]
        self.assertTrue(torch.all(item['labels'][:6]==-100))
        batch=collate([item,{'input_ids':torch.tensor([0,1]),'labels':torch.tensor([-100,1]),'attention_mask':torch.ones(2,dtype=torch.long)}])
        self.assertEqual(batch['attention_mask'][1].tolist(),[1,1]+[0]*10)

    def test_duplex_alignment_and_masks(self):
        record={'user_speech_tokens':[0]*20,'assistant_speech_tokens':[1]*20,
                'turns':[{'role':'assistant','start_time':0.2,'end_time':0.4,'text':'yes no'}]}
        self.write(record,'fullduplex_manifest.jsonl')
        ds=FullDuplexThreeStreamDataset(self.root,self.tokenizer)
        item=ds[0]
        self.assertEqual(item['input_ids'][10:12].tolist(),[self.s['SILENT_TEXT']]*2)
        self.assertEqual(item['input_ids'][12:22].tolist(),[self.s['SILENT_SPEECH']]*10)
        self.assertEqual(item['input_ids'][32:34].tolist(),[2,3])
        self.assertEqual(item['input_ids'][34:44].tolist(),[14]*10)
        self.assertEqual(item['loss_mask'].tolist(),([0]*10+[1]*12)*2)
        two=FullDuplexTwoStreamDataset(self.root,self.tokenizer,manifest_filename='fullduplex_manifest.jsonl')[0]
        self.assertEqual(len(two['input_ids']),40)
        self.assertEqual(two['loss_mask'].tolist(),([0]*10+[1]*10)*2)

    def test_truncated_text_requires_correction(self):
        turns=[{'role':'assistant','start_time':0.,'end_time':0.4,'text':'yes','truncated':True}]
        with self.assertRaisesRegex(ValueError,'spoken_text'):
            aligned_output([13]*20,turns,self.tokenizer,self.s)
        aligned_output([13]*20,turns,self.tokenizer,self.s,with_text=False)

    def test_loss_ignores_padding_and_masked_targets(self):
        self.model.eval()
        a={'input_ids':torch.tensor([0,1,2]),'labels':torch.tensor([-100,-100,2]),'attention_mask':torch.ones(3,dtype=torch.long)}
        one=self.model(**collate([a])).loss
        # Add padded positions to the same sample.
        padded={'input_ids':torch.tensor([[0,1,2,0,0]]), 'labels':torch.tensor([[-100,-100,2,-100,-100]]),
                'attention_mask':torch.tensor([[1,1,1,0,0]])}
        torch.testing.assert_close(one,self.model(**padded).loss)
        empty=self.model(input_ids=torch.tensor([[0,1]]),labels=torch.tensor([[-100,-100]])).loss
        self.assertEqual(empty.item(),0.)
        empty.backward()

    def test_chunked_loss_and_gradients_match_full_projection(self):
        import copy
        import torch.nn.functional as F
        ref = copy.deepcopy(self.model)
        self.model.eval(); ref.eval()
        ids = torch.tensor([[0, 13, 14, 1, 2, 15, 3, 0, 1, 2, 15, 13, 14, 1, 2] * 4])
        labels = ids.clone()
        labels[:, :3] = -100
        mask = torch.ones_like(ids)
        mask[:, 7] = 0
        actual = self.model(ids, labels=labels, loss_mask=mask).loss
        targets = labels[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -100
        logits = ref.backbone(ids, use_cache=False).logits[:, :-1].float()
        expected = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(actual, expected)
        actual.backward(); expected.backward()
        for (name, p), (_, q) in zip(self.model.named_parameters(), ref.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=lambda msg: name + ': ' + msg)

    def test_cache_matches_full_context_and_modality_constraints(self):
        engine=OmniFlattenInference(self.model,None,None,self.tokenizer,InferenceConfig(max_response_chunks=0))
        engine._feed([13,14])
        generated=engine._generate(2,'text',silent=True)
        generated+=engine._generate(10,'speech',silent=True)
        full=self.model(torch.tensor([[13,14]+generated]),use_cache=False).logits[0,-1].float()
        torch.testing.assert_close(engine.logits,full,rtol=1e-4,atol=1e-5)
        self.assertEqual(engine.cache.get_seq_length(),14)
        self.assertTrue(all(13<=i<29 or i==self.s['SILENT_SPEECH'] for i in generated[2:]))
        self.assertTrue(all(i<5 or i==self.s['SILENT_TEXT'] for i in generated[:2]))

    def test_stream_yields_before_consuming_next_input_and_two_stream(self):
        seen=[]
        def source():
            seen.append(1); yield [0]*10
            seen.append(2); yield [1]*3
        engine=OmniFlattenInference(self.model,None,None,self.tokenizer,InferenceConfig(mode='3b',max_response_chunks=1))
        iterator=engine.stream_generate(source())
        first=next(iterator)
        self.assertEqual(seen,[1])
        self.assertEqual(first['text_tokens'],[])
        packets=[first]+list(iterator)
        self.assertEqual(len(packets),3)
        self.assertEqual(engine.context_length,60)
        self.assertTrue(packets[-1]['flushing'])

    def test_silence_keeps_waveform_timing(self):
        decoder=type('Decoder',(),{'output_sample_rate':100,'token_rate':50,
            'decode':lambda _,codes:torch.ones(len(codes)*2)})()
        engine=OmniFlattenInference(self.model,None,decoder,self.tokenizer)
        audio=engine._decode_timeline([13,self.s['SILENT_SPEECH'],14])
        np.testing.assert_array_equal(audio,[1,1,0,0,1,1])

    def test_interruption_has_actual_overlap(self):
        simulator=InteractionSimulator(sample_rate=1000)
        arrays=[np.ones(1000,dtype=np.float32),np.ones(1000,dtype=np.float32)]
        texts=[{'text':'hello'},{'text':'yes'},{'text':'world'},{'text':'no'}]
        with patch.object(simulator,'_choose_interaction_type',return_value='user_interrupt'):
            result=simulator.simulate_interaction(arrays,arrays,texts)
        self.assertTrue(np.any((result['user_channel']>0)&(result['assistant_channel']>0)))
        turns=result['turn_timestamps']
        self.assertLess(turns[2]['start_time'],turns[1]['end_time'])
        self.assertTrue(turns[1]['truncated'])
        self.assertFalse(turns[-1]['truncated'])

    def test_simulation_writes_half_and_full_duplex_manifests(self):
        class Codec:
            def __init__(self, *args):
                self.cosyvoice=type('Voice',(),{'frontend':type('Frontend',(),{'spk2info':{'voice':{}}})()})()
            def _speaker(self): return 'voice'
            def synthesize(self,text,speaker_id=None): return np.ones(16000,dtype=np.float32)*0.1
        with patch('omniflatten.speech_tokenizer.CosyVoiceCodec',Codec):
            config=PipelineConfig(codec_dir='fixture',assistant_speaker_id='voice',
                                  dialogue_data_dir=str(self.root/'output'),interrupt_prob=0.0)
            pipeline=DialogueSimulationPipeline(config)
            source=self.root/'text.jsonl'
            source.write_text(json.dumps({'dialogue_id':'../untrusted-id','turns':[
                {'role':'user','text':'Hello there'},{'role':'assistant','text':'Hello friend'}]})+'\n')
            pipeline.run(str(source),str(self.root/'output/manifest.jsonl'))
        full=json.loads((self.root/'output/manifest.jsonl').read_text())
        half=json.loads((self.root/'output/half_duplex_manifest.jsonl').read_text())
        self.assertEqual(full['dialogue_id'],half['dialogue_id'])
        self.assertEqual(len(half['turns']),2)
        self.assertTrue(Path(half['turns'][0]['audio_path']).is_file())
        self.assertTrue(Path(full['assistant_audio_path']).is_relative_to(self.root/'output'))

    def test_no_fake_codec_or_tts_fallback(self):
        with self.assertRaises(ValueError): DialogueSimulationPipeline(PipelineConfig())
        with self.assertRaises(FileNotFoundError): CosyVoiceCodec(self.root)

if __name__=='__main__': unittest.main()
