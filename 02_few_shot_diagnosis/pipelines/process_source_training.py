"""Reusable supervised/episodic source training for a declared process schema.

The source table must already exclude final target queries. This API has no
query-table argument. The default frozen sampler needs one class per record;
explicit global_records mode assigns whole records across every episode class.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
from common_algorithms.models.process_cnn import ChannelStandardizer,save_process_model
from ..models.process_learning import train_process_classifier,train_process_protonet,process_prototypes
from ..models.global_record_meta import train_global_record_protonet
from .process_support import labelled_support_windows


def train_process_source(df,output_path,input_contract,label_column,origin_record_column,source_kind,
                         method='protonet',window_id_column='window_id',time_column='timestamp',
                         seed=42,steps=250,episodes=350,episode_support=2,episode_query=3,
                         episode_ways=None,learning_rate=.002,class_order=None,episode_grouping='single_class_records'):
    """Save a source resource usable by the existing support-only adapter.

    Default ProtoNet uses one class per original record. Global-record mode
    supports mixed-class records via verified whole-record role assignment.
    Counts refer to caller-supplied IDs, not certified independent incidents.
    CNN pretraining has no internal query split; final queries stay outside.
    """
    path=Path(output_path);audit=path.with_suffix(path.suffix+'.metadata.json')
    if path.exists() or audit.exists():
        raise FileExistsError('Choose a new source resource path; existing weights/audits are preserved')
    if path.suffix!='.pt' or method not in {'cnn','protonet'}:
        raise ValueError('Declare cnn/protonet and a new .pt source resource')
    if source_kind not in {'public_measured','physics_simulation','field_labelled'}:
        raise ValueError('Explicit measured/simulation/labelled-field source provenance is required')
    if episode_grouping not in {'single_class_records','global_records'}:
        raise ValueError('Declare single_class_records or global_records episode grouping')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:
        raise ValueError('An unsigned 32-bit integer seed is required')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in [steps,episodes,episode_support,episode_query]):
        raise ValueError('Steps, episodes and per-class episode counts must be positive integers')
    if isinstance(learning_rate,bool) or not isinstance(learning_rate,(int,float)) or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('learning_rate must be finite and positive')
    x,y,classes,counts,labelled,contract=labelled_support_windows(df,input_contract,label_column,
        origin_record_column,window_id_column,time_column,class_order)
    ways=min(5,len(classes)) if episode_ways is None else episode_ways
    if isinstance(ways,bool) or not isinstance(ways,int) or not 2<=ways<=len(classes):
        raise ValueError('Episode ways must be an integer from two to the source class count')
    # Same appearance order as the validated process_windows grouping. Keep
    # IDs as strings, allowing the CLI to preserve leading zeroes explicitly.
    window_ids=pd.unique(labelled[window_id_column])
    lookup=labelled.groupby(window_id_column,sort=False)[origin_record_column].first()
    groups=np.asarray([str(lookup.loc[key]) for key in window_ids])
    budget={c:{'source_windows':counts[c]['support_windows'],
               'caller_provided_origin_records':int(len(np.unique(groups[y==i])))}
            for i,c in enumerate(classes)}
    if method=='protonet':
        group_labels=pd.DataFrame({'record':groups,'label':y}).groupby('record',sort=False).label.nunique()
        if group_labels.gt(1).any() and episode_grouping=='single_class_records':
            raise ValueError('This episodic source sampler requires one class per original record. '
                'Use explicit global_records mode for mixed-class records; do not rename normal/fault segments as independent records.')
        required=episode_support+episode_query
        if any(item['caller_provided_origin_records']<required for item in budget.values()):
            raise ValueError('Not enough distinct original source records per class for episode support plus query; extra windows do not replace records')
    scaler=ChannelStandardizer().fit(x);scaled=scaler.transform(x)
    provenance={'operation':'source_pretraining','method':method,'source_kind':source_kind,
        'classes':classes,'counts':budget,'input_contract':contract,'seed':seed,
        'normalization_fit':'labelled_source_only','final_query_records_used':0,
        'final_source_support_query_split_must_be_verified_by_caller':True,
        'origin_independence_not_certified_by_this_program':True,
        'source_table_sha256':hashlib.sha256(labelled.to_csv(index=False).encode('utf-8')).hexdigest(),
        'learning_rate':float(learning_rate),'source_pretraining_used':True,
        'source_training_steps':steps if method=='cnn' else 0,
        'source_training_episodes':episodes if method=='protonet' else 0,
        'episode_support_per_class':episode_support if method=='protonet' else None,
        'episode_query_per_class':episode_query if method=='protonet' else None,
        'episode_ways':ways if method=='protonet' else None,
        'episode_grouping':episode_grouping if method=='protonet' else None,
        'episode_global_record_separation':method=='protonet',
        'episode_group_rule':('one class per original record; per-class disjoint IDs imply globally disjoint roles'
            if episode_grouping=='single_class_records' else 'whole original records have one support/query role across all selected episode classes') if method=='protonet' else None,
        'source_prototypes_are_new_target_support':False}
    if method=='cnn':
        model,history=train_process_classifier(scaled,y,seed=seed,steps=steps,lr=learning_rate)
        provenance['training_history']=history
        json.dumps(provenance)
        save_process_model(path,model,scaler,contract,classes,'cnn',provenance=provenance)
    else:
        if episode_grouping=='single_class_records':
            encoder,history=train_process_protonet(scaled,y,seed=seed,episodes=episodes,support=episode_support,
                query=episode_query,ways=ways,lr=learning_rate,record_groups=groups)
        else:
            encoder,history=train_global_record_protonet(scaled,y,groups,seed=seed,episodes=episodes,support=episode_support,
                query=episode_query,ways=ways,lr=learning_rate)
        provenance['training_history']=history
        # Source-class prototypes make the source artifact self-describing.
        # The adapter constructs fresh prototypes for new target classes.
        prototypes=process_prototypes(encoder,scaled,y)
        json.dumps(provenance)
        save_process_model(path,encoder,scaler,contract,classes,'prototype',prototypes,provenance)
    audit.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return provenance
