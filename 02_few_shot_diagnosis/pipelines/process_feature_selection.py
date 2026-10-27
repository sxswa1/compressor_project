"""Support-only diagnosis feature selection with an explicit K=1 fallback."""
import numpy as np


def select_support_features(features,labels,max_features=14):
    x=np.asarray(features,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or not len(x) or y.shape!=(len(x),) or not np.isfinite(x).all():
        raise ValueError('Finite support features and aligned labels are required')
    if isinstance(max_features,bool) or not isinstance(max_features,int) or max_features<1:
        raise ValueError('max_features must be a positive integer')
    classes,counts=np.unique(y,return_counts=True)
    if len(classes)<2:
        raise ValueError('At least two support classes are required')
    if counts.min()<2:
        return np.arange(x.shape[1]),{'method':'all_features','reason':'less_than_two_support_records_per_class'}
    grand=x.mean(axis=0)
    between=np.zeros(x.shape[1]);within=np.zeros(x.shape[1])
    for c in classes:
        part=x[y==c];mean=part.mean(axis=0)
        between+=len(part)*(mean-grand)**2
        within+=((part-mean)**2).sum(axis=0)
    score=np.divide(between/(len(classes)-1),within/(len(x)-len(classes)),out=np.zeros_like(between),where=within>1e-20)
    score[(within<=1e-20)&(between>1e-20)]=np.finfo(float).max
    ids=np.argsort(-score,kind='stable')[:min(max_features,x.shape[1])]
    ids=np.sort(ids)
    return ids,{'method':'support_only_anova','max_features':max_features,'scores':score.tolist(),
        'records_used':len(x),'query_records_used':0}
