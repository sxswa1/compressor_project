"""Support-only adaptation of a compatible saved process encoder.

The caller supplies labelled original-record IDs and an exact physical input
contract. No query table is accepted. This does not map unrelated sensors or
certify that caller-provided record IDs are independent fault incidents.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from common_algorithms.models.process_cnn import (validate_contract,ChannelStandardizer,ProcessEncoder,
    predict_process_resource,save_process_model)
from ..models.process_learning import train_process_classifier,process_prototypes
from .process_support import labelled_support_windows


def adapt_process_resource(df,source_path,output_path,input_contract,label_column,origin_record_column,
                           support_source_kind,method='prototype',window_id_column='window_id',
                           time_column='timestamp',seed=42,steps=250,class_order=None,learning_rate=.002):
    """Create one fixed resource; prototype/fine_tune/fine_tune_prototype.

    Prototype construction alone is not proof of episodic meta-training; retain
    and inspect source provenance. Fine-tuning uses a copied encoder and fresh
    target head. Counts distinguish windows from caller-provided origins.
    Only trusted torch resources are accepted. Existing resources never change.
    """
    contract=validate_contract(input_contract)
    source_path,output_path=Path(source_path),Path(output_path)
    if output_path.exists() or output_path.with_suffix(output_path.suffix+'.metadata.json').exists():
        raise FileExistsError('Choose a new resource path; existing source or target weights are preserved')
    if method not in {'prototype','fine_tune','fine_tune_prototype'} or support_source_kind not in {'public_measured','physics_simulation','field_labelled'}:
        raise ValueError('Declare the adaptation operation and measured/simulation support provenance')
    if isinstance(seed,bool) or not isinstance(seed,int) or isinstance(steps,bool) or not isinstance(steps,int) or steps<1:
        raise ValueError('Explicit integer seed and positive training steps are required')
    if isinstance(learning_rate,bool) or not isinstance(learning_rate,(int,float)) or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('learning_rate must be a finite positive number')
    columns=[c['name'] for c in contract['channels']]
    x,y,classes,counts,labelled,contract=labelled_support_windows(df,contract,label_column,origin_record_column,
        window_id_column,time_column,class_order)
    source=torch.load(source_path,map_location='cpu',weights_only=True)
    if source.get('contract')!=contract:
        raise ValueError('Support physical schema differs from the pretrained resource')
    if any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in source['state_dict'].values()):
        raise ValueError('Source encoder weights must be finite tensors')
    # Validate format, kind, architecture, scaler and old class/prototype layout.
    predict_process_resource(source,np.empty((0,len(columns),contract['window_points']),dtype=np.float32),contract)
    encoder=ProcessEncoder(**source['architecture'])
    state=source['state_dict'] if source['kind']=='prototype' else {
        k[len('encoder.'):]:v for k,v in source['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(state)
    scaler=ChannelStandardizer.from_state(source['scaler']);scaled=scaler.transform(x)
    provenance={'operation':method,'support_source_kind':support_source_kind,'classes':classes,
        'counts':counts,'query_records_used':0,'normalization_refitted':False,
        'origin_independence_not_certified_by_this_program':True,
        'source_resource_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'support_table_sha256':hashlib.sha256(labelled.to_csv(index=False).encode()).hexdigest(),
        'source_resource_kind':source['kind'],'source_training_provenance':source.get('provenance',{}),
        'seed':seed,'training_steps':0 if method=='prototype' else steps,
        'learning_rate':None if method=='prototype' else float(learning_rate)}
    json.dumps(provenance) # Reject nonserializable metadata before writing a resource.
    if method=='prototype':
        prototype=process_prototypes(encoder,scaled,y)
        save_process_model(output_path,encoder,scaler,contract,classes,'prototype',prototype,provenance)
    else:
        target,_=train_process_classifier(scaled,y,seed=seed,steps=steps,encoder=encoder,head_seed=seed+100000,lr=learning_rate)
        if method=='fine_tune':
            save_process_model(output_path,target,scaler,contract,classes,'transfer',provenance=provenance)
        else:
            prototype=process_prototypes(target.encoder,scaled,y)
            save_process_model(output_path,target.encoder,scaler,contract,classes,'prototype',prototype,provenance)
    output_path.with_suffix(output_path.suffix+'.metadata.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    return provenance
