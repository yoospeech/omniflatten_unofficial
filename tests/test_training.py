"""CPU end-to-end tests with a randomly initialized tiny Qwen2 (no downloads)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from test_core import tiny_model
from omniflatten.model import OmniFlattenModel

ROOT=Path(__file__).resolve().parents[1]

class TrainingTests(unittest.TestCase):
    def test_four_stage_curriculum_and_exact_mid_epoch_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            # Save a base model with an unextended tokenizer, as an actual LLM checkpoint would.
            model,extended=tiny_model()
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from transformers import PreTrainedTokenizerFast
            raw=Tokenizer(WordLevel({'hello':0,'world':1,'yes':2,'no':3,'[UNK]':4},unk_token='[UNK]'))
            raw.pre_tokenizer=Whitespace()
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]')
            base=root/'base'
            Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                max_position_embeddings=512,tie_word_embeddings=True)).save_pretrained(base)
            tokenizer.save_pretrained(base)
            data=root/'data'; data.mkdir()
            output=root/'out'
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
            common=[sys.executable,str(ROOT/'train.py'),'--data_dir',str(data),'--epochs','1',
                    '--gradient_accumulation_steps','3','--batch_size','1','--save_interval','1','--log_interval','1']
            def write(records):
                for name in ('train','val'):
                    (data/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            def run(stage,extra,out=output):
                result=subprocess.run(common+['--stage',stage,'--output_dir',str(out)]+extra,
                                      env=env,cwd=ROOT,text=True,capture_output=True,timeout=90)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                self.assertTrue((out/f'stage{stage}/best_model/model.safetensors').exists())
            write([{'speech_tokens':[0,1],'text':'hello'},{'speech_tokens':[2,3,4],'text':'hello world'}])
            run('1',['--backbone',str(base)])
            resumed=root/'resumed'
            run('1',['--resume_from_checkpoint',str(output/'stage1/step-1')],resumed)
            original,_=OmniFlattenModel.from_pretrained(output/'stage1/last_model')
            restored,_=OmniFlattenModel.from_pretrained(resumed/'stage1/last_model')
            for name,tensor in original.state_dict().items():
                torch.testing.assert_close(tensor,restored.state_dict()[name],rtol=0,atol=0)
            write([{'turns':[{'role':'user','text':'hello','speech_tokens':[0,1]},
                             {'role':'assistant','text':'yes','speech_tokens':[2,3]}]}])
            run('2',['--checkpoint',str(output/'stage1/best_model')])
            write([{'user_speech_tokens':[0]*20,'assistant_speech_tokens':[1]*20,
                    'turns':[{'role':'assistant','start_time':0.2,'end_time':0.4,'text':'yes no'}]}])
            run('3a',['--checkpoint',str(output/'stage2/best_model')])
            run('3b',['--checkpoint',str(output/'stage3a/best_model')])
            final,_=OmniFlattenModel.from_pretrained(output/'stage3b/best_model')
            self.assertEqual(final.layout.num_speech_tokens,4096)

if __name__=='__main__': unittest.main()
