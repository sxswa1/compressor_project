"""Support-only RF/CNN fitting for a new, explicitly declared process schema.

No pretrained encoder is required. New sensor schemas require retraining;
this does not rename an old acoustic or public model to fit field channels.
"""
from pathlib import Path
import hashlib,json,math
import joblib
from common_algorithms.models.process_cnn import ChannelStandardizer,save_process_model
from common_algorithms.models.random_forest import train_random_forest
from common_algorithms.feature_extraction.process_features import process_features
from ..models.process_learning import train_process_classifier
from .process_support import labelled_support_windows
from .process_feature_selection import select_support_features


def fit_process_resource(df,output_path,input_contract,label_column,origin_record_column,support_source_kind,
                         method='random_forest',window_id_column='window_id',time_column='timestamp',
                         seed=42,steps=250,n_estimators=200,max_features=None,class_order=None,learning_rate=.002):
    """Create a new baseline resource using only the labelled support table.

    Normalization and optional RF ANOVA feature selection are support-only.
    This function neither accepts query data nor reports training accuracy as
    evidence of fault diagnosis. Window and original-record counts are distinct.
    Existing resource/audit files are never overwritten.
    """
    path=Path(output_path);audit=path.with_suffix(path.suffix+'.metadata.json')
    if path.exists() or audit.exists():
        raise FileExistsError('Choose a new resource path; existing weights and audit files are preserved')
    if method not in {'random_forest','cnn'} or support_source_kind not in {'public_measured','physics_simulation','field_labelled'}:
        raise ValueError('Declare RF/CNN and measured/simulation support provenance')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32 or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in [steps,n_estimators]):
        raise ValueError('An unsigned 32-bit seed and positive integer steps/tree count are required')
    if isinstance(learning_rate,bool) or not isinstance(learning_rate,(int,float)) or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('learning_rate must be a finite positive number')
    if method=='cnn' and max_features is not None:
        raise ValueError('Feature selection applies to the RF feature table, not CNN channels')
    if max_features is not None and (isinstance(max_features,bool) or not isinstance(max_features,int) or max_features<1):
        raise ValueError('max_features must be None or a positive integer')
    expected_suffix='.joblib' if method=='random_forest' else '.pt'
    if path.suffix!=expected_suffix:
        raise ValueError(f'{method} resources require the {expected_suffix} extension')
    x,y,classes,counts,labelled,contract=labelled_support_windows(df,input_contract,label_column,origin_record_column,
        window_id_column,time_column,class_order)
    scaler=ChannelStandardizer().fit(x);scaled=scaler.transform(x)
    provenance={'operation':'fit_from_labelled_support','method':method,'support_source_kind':support_source_kind,
        'classes':classes,'counts':counts,'query_records_used':0,'normalization_fit':'labelled_support_only',
        'origin_independence_not_certified_by_this_program':True,'source_pretraining_used':False,
        'support_table_sha256':hashlib.sha256(labelled.to_csv(index=False).encode()).hexdigest(),
        'input_contract':contract,'seed':seed,'training_steps':steps if method=='cnn' else 0,
        'n_estimators':n_estimators if method=='random_forest' else None,'learning_rate':float(learning_rate) if method=='cnn' else None,
        'feature_selection_fit':'labelled_support_only' if max_features else None}
    if method=='random_forest':
        features,names=process_features(scaled,[c['name'] for c in contract['channels']],contract['interval_seconds'])
        if max_features is not None and max_features>len(names):
            raise ValueError('max_features exceeds the canonical process feature dimension')
        indices=None
        if max_features is not None:
            indices,selection=select_support_features(features,y,max_features)
            provenance['feature_selection']=selection
        forest=train_random_forest(features if indices is None else features[:,indices],y,n_estimators=n_estimators,random_state=seed)
        resource={'model':forest,'feature_columns':names,'scaler':scaler.state(),'process_contract':contract,
            'class_names':classes,'provenance':provenance}
        if indices is not None:
            resource['selected_feature_indices']=indices.tolist()
            provenance['selected_feature_names']=[names[int(i)] for i in indices]
        json.dumps(provenance)
        path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(resource,path)
    else:
        model,history=train_process_classifier(scaled,y,seed=seed,steps=steps,head_seed=seed+100000,lr=learning_rate)
        provenance['training_history']=history
        json.dumps(provenance)
        save_process_model(path,model,scaler,contract,classes,'cnn',provenance=provenance)
    audit.write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    return provenance
