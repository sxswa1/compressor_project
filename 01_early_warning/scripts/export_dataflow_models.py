"""导出两个 DataFlow 算子所需的已训练模型，不重新训练。"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import shutil
import torch


def export_models(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'输出目录非空，请另选目录：{output}')
    fit = root/'final_model_fit_v1'
    source = root/'warning_lstm_budget_v1/results/bundle.pkl'
    meta_path = fit/'results/preprocessing.json'
    files = {'preprocessing.json': meta_path,
             'gradient_boosting_di.pkl': fit/'models/di/gradient_boosting_history__20261024.pkl',
             'reference.pkl': fit/'models/rul/reference.pkl',
             'interval_radii.json': fit/'models/rul/interval_radii.json'}
    for seed in [20261023, 20261024, 20261025]:
        files[f'di_{seed}.pt'] = fit/f'models/di/lstm_window__{seed}.pt'
        files[f'rul_{seed}.pkl'] = fit/f'models/rul/trees_history__residual_augmented__{seed}.pkl'
    for p in [source, *files.values()]:
        if not p.is_file():
            raise FileNotFoundError(f'缺少训练产物：{p}；先执行 train-warning 和 train-regression')
    with source.open('rb') as stream:
        bundle = pickle.load(stream)
    base = bundle['v3']['base']['base']
    warning = dict(format='rcsd_ewma_operator_v1', sensors=bundle['sensors'],
        targets=base['config']['targets'], reference_inputs=base['reference_inputs'],
        reference_model=base['reference_model'], change_scale=bundle['v3']['base']['scales']['change24'],
        speed_column='75SI865R.pv', speed_bounds=[6050, 8500],
        threshold=bundle['thresholds']['change_ewma'],
        config={k: bundle['cfg'][k] for k in ['alarm_consecutive_points',
                'recovery_threshold_ratio', 'recovery_consecutive_points']})
    output.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        shutil.copy2(path, output/name)
    with (output/'warning.pkl').open('wb') as stream:
        pickle.dump(warning, stream, protocol=5)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    units = {}
    for sensor in meta['sensors']:
        units[sensor] = ('rpm' if 'SI' in sensor else 'degC' if 'TI' in sensor else
                         'mm' if 'ZI' in sensor else 'um' if 'XI' in sensor else 'kg/cm2')
    for seed in [20261023,20261024,20261025]:
        model_path = output/f'di_{seed}.pt'
        payload = torch.load(model_path, map_location='cpu', weights_only=True)
        if (payload['window'],payload['features']) != (meta['history_points'],len(meta['sensors'])):
            raise ValueError('LSTM 权重与预处理描述不匹配')
        schema = dict(model_sha256=digest(model_path), feature_columns=meta['sensors'],
            feature_units=units, window_points=meta['history_points'], sampling_minutes=meta['sampling_minutes'],
            target_name='DI_raw', target_unit='dimensionless', running_column='75SI865R.pv',
            running_bounds=meta['running_speed_range'])
        (output/f'di_{seed}.schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    feature_names = [f'{recipe}__{sensor}' for recipe in meta['feature_order'] for sensor in meta['sensors']]
    feature_units = {f'{recipe}__{sensor}': units[sensor] for recipe in meta['feature_order'] for sensor in meta['sensors']}
    schema = dict(model_sha256=digest(output/'gradient_boosting_di.pkl'), feature_columns=feature_names,
        feature_units=feature_units, target_name='DI_raw', target_unit='dimensionless',
        feature_recipe='RCSD raw 96-point window, pipelines.baseline_data.tabular_features; float32 inputs',
        recipe_order=meta['feature_order'], sensors=meta['sensors'], window_points=96, sampling_minutes=15,
        running_column='75SI865R.pv', running_bounds=meta['running_speed_range'])
    (output/'gradient_boosting_di.schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest = dict(format='rcsd_dataflow_models_v1', speed_column='75SI865R.pv',
        seeds=[20261023, 20261024, 20261025],
        methods={'warning': 'change_ewma', 'di': 'lstm_window', 'rul': 'trees_history__residual_augmented'},
        files={p.name: digest(p) for p in sorted(output.iterdir())},
        training_sources={name: {'relative_path': str(p.relative_to(root)).replace('\\','/'),
                                 'sha256': digest(p)} for name, p in files.items()},
        warning_source_sha256=digest(source))
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'experiments/rcsd_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export_models(args.root, args.output)
    print(f"Exported {len(result['files'])} model files to {args.output}")
