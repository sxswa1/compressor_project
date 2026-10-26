"""Verify planned inclusions/rejections and stored data before any reserved scoring."""
import pandas as pd
from common import ROOT,WORK,PROTOCOL,read,write,digest,assert_release
from generate import load_raw,controlled,stress,stress_blocks
from reserved_io import reserved_scenes

def main():
    assert_release();raw,meta=load_raw('test');directory=WORK/read(PROTOCOL)['dataset']['output_directory']
    ledger=[];counts={}
    for source,prefix,iterator in [('v2','controlled',controlled(raw,meta,'test',ledger)),('stress','stress',stress(raw,meta,'test',ledger))]:
        stored=reserved_scenes(prefix);scenes=rows=0
        for expected in iterator:
            actual=next(stored);assert expected[0]==actual[0]
            for a,b in zip(expected[1:],actual[1:]):
                a=a.copy()
                if 'risk_ref' in a:a['risk_ref']=a.risk_ref.astype(float)
                pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-10,atol=1e-9)
            scenes+=1;rows+=len(actual[1])
        assert next(stored,None) is None
        counts[source]=dict(scenes=scenes,rows=rows)
    assert ledger==read(directory/'inclusion_ledger.json') and len(ledger)==844
    plan=pd.read_csv(WORK/'final_evaluation_protocol_v1/results/reserved_stress_blocks.csv')
    blocks=stress_blocks(meta,'test')
    assert [(b['block_id'],b['first_row'],b['last_row']) for b in blocks]==list(plan[['block_id','first_source_row','last_source_row']].itertuples(index=False,name=None))
    write(ROOT/'results/reserved_dataset_verification.json',dict(status='PASS',counts=counts,planned=844,
        rejected=sum(r['status']=='rejected' for r in ledger),all_inclusions_and_rejections_reproduced=True,
        model_predictions_read=False,dataset_checksums_sha256=digest(directory/'checksums.json')))

if __name__=='__main__':main()
