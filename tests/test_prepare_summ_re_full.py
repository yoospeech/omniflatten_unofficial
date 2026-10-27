import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import soundfile as sf
import torch
from scripts.prepare_summ_re_full import segment_chunks, process_split

class FakeCodec:
    def encode(self, waveform):
        return torch.tensor([1, 2], dtype=torch.long)

class PreparationTests(unittest.TestCase):
    def test_long_segments_keep_words_and_report_long_silence(self):
        segment = {'start':0., 'end':26., 'transcript':'hello sil world', 'words':[
            {'start':0., 'end':1., 'word':'hello'},
            {'start':1., 'end':24., 'word':'sil'},
            {'start':24., 'end':26., 'word':'world'}]}
        chunks = segment_chunks(segment,12)
        self.assertEqual(' '.join(c[2] for c in chunks),segment['transcript'])
        self.assertEqual(sum(c[3] for c in chunks),1)
        self.assertTrue(all(b-a<=12 for a,b,_,_ in chunks))

    def test_resume_discards_uncommitted_tail_and_quarantines_bad_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            samples=np.zeros(16000,dtype=np.float32)
            wav=io.BytesIO()
            sf.write(wav,samples,16000,format='WAV')
            segments=[{'start':0.,'end':0.5,'transcript':'hello','words':[]},
                      {'start':0.5,'end':1.,'transcript':'world','words':[]},
                      {'start':0.,'end':15.,'transcript':'word',
                       'words':[{'start':0.,'end':15.,'word':'word'}]}]
            table=pa.Table.from_pylist([{'meeting_id':'meeting','speaker_id':'speaker',
                'audio_id':'audio','audio':{'bytes':wav.getvalue(),'path':None},'segments':segments}])
            shard=root/'sample-train-00000.arrow'
            with pa.OSFile(str(shard),'wb') as sink:
                with pa.ipc.new_stream(sink,table.schema) as writer:
                    writer.write_table(table)
            output=root/'output'
            first=process_split('train','train',root,output,FakeCodec(),'sig',12.,1)
            self.assertFalse(first['complete'])
            part=output/'train.jsonl.part'
            with part.open('ab') as stream: stream.write(b'uncommitted tail')
            final=process_split('train','train',root,output,FakeCodec(),'sig',12.,0)
            self.assertTrue(final['complete'])
            self.assertEqual(final['segments'],3)
            self.assertEqual(final['records'],2)
            self.assertEqual(final['review_segments'],1)
            rows=[json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
            self.assertEqual([r['text'] for r in rows],['hello','world'])
            self.assertEqual(json.loads((output/'train.review.jsonl').read_text())['segment']['transcript'],'word')
            again=process_split('train','train',root,output,FakeCodec(),'sig',12.,0)
            self.assertEqual(again['records'],2)

if __name__=='__main__': unittest.main()
