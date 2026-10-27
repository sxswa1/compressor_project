"""将公共 LSTM、梯度提升及各自入口合并为可直接加载的单文件算子。"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(name):
    if name not in ['lstm','gradient_boosting']:
        raise ValueError(name)
    core = (ROOT/f'common_algorithms/models/{name}.py').read_text(encoding='utf-8')
    adapter = (ROOT/f'common_algorithms/models/{name}_operator.py').read_text(encoding='utf-8')
    lines = adapter.splitlines(keepends=True)
    for node in reversed(ast.parse(adapter).body):
        if isinstance(node, ast.ImportFrom) and node.module == f'common_algorithms.models.{name}':
            del lines[node.lineno-1:node.end_lineno]
    content = core+'\n'+''.join(lines)
    digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return f'# 高淏宇；由 tools/build_regression_operators.py 生成。\n# Source SHA256: {digest}\n'+content


if __name__ == '__main__':
    for name in ['lstm','gradient_boosting']:
        output = ROOT/f'dataflow_operators/predictive_analysis/{name}_regression.py'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build(name), encoding='utf-8')
        print(output.relative_to(ROOT))
