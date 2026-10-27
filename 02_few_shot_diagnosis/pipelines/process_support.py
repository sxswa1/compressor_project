"""Validate labelled original-record windows before fitting or adaptation.

This checks the supplied evidence, not whether a human label is correct or
whether two caller-provided record IDs represent independent fault incidents.
There is deliberately no query/test-table argument.
"""
import numpy as np
import pandas as pd
from common_algorithms.data_processing.process_timeseries import process_windows
from common_algorithms.models.process_cnn import validate_contract


def labelled_support_windows(df,input_contract,label_column,origin_record_column,
                             window_id_column='window_id',time_column='timestamp',class_order=None):
    contract=validate_contract(input_contract)
    columns=[c['name'] for c in contract['channels']]
    metadata=[window_id_column,time_column,label_column,origin_record_column]
    if any(not isinstance(c,str) or not c for c in metadata):
        raise ValueError('Explicit support metadata column names are required')
    if set(columns)&set(metadata) or len({window_id_column,time_column,label_column})!=3 or origin_record_column in {time_column,label_column}:
        raise ValueError('Signal, label, clock and record metadata must be separate')
    required=set(columns+metadata)
    if not isinstance(df,pd.DataFrame) or df.empty or not df.columns.is_unique or not required.issubset(df.columns):
        raise ValueError('Nonempty labelled support table with original-record metadata is required')
    labelled=df.copy(deep=True)
    for name in [window_id_column,label_column,origin_record_column]:
        if labelled[name].isna().any() or labelled[name].astype(str).str.strip().eq('').any():
            raise ValueError('Support class names, window IDs and original-record IDs must be nonmissing and nonempty')
    labelled[label_column]=labelled[label_column].astype(str)
    groups=labelled.groupby(window_id_column,sort=False)
    for _,part in groups:
        if len(part)!=contract['window_points'] or part[label_column].nunique()!=1 or part[origin_record_column].nunique()!=1:
            raise ValueError('Every support window must be complete with exactly one class and original record')
    x,meta=process_windows(labelled,columns,time_column,[window_id_column],contract['interval_seconds'],
        contract['window_points'],contract['window_points'],contract.get('source_timezone'))
    if len(x)!=labelled[window_id_column].nunique():
        raise ValueError('A support window has missing values or discontinuous original timestamps')
    # Table windows originate from transposed time-major arrays. Canonicalize
    # NCT storage so seeded training uses the same reduction/kernel path as
    # the saved research arrays, even when every numeric value already agrees.
    x=np.ascontiguousarray(x)
    lookup=groups.first()
    if origin_record_column==window_id_column:
        lookup[origin_record_column]=lookup.index
    labels=[lookup.loc[key,label_column] for key in meta[window_id_column]]
    classes=sorted(set(labels)) if class_order is None else list(class_order)
    if len(classes)<2 or any(not isinstance(c,str) or not c.strip() for c in classes) or len(set(classes))!=len(classes) or set(classes)!=set(labels):
        raise ValueError('At least two observed classes and an exact unique class order are required')
    target_map={c:i for i,c in enumerate(classes)}
    y=np.asarray([target_map[c] for c in labels],dtype=np.int64)
    counts={c:{'support_windows':int(sum(v==c for v in labels)),
        'caller_provided_origin_records':int(lookup[lookup[label_column].eq(c)][origin_record_column].nunique())} for c in classes}
    return x,y,classes,counts,labelled,contract
