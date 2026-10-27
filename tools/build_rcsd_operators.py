"""Build the ten registered, self-contained process-data operators.

No data/model loading or training occurs here. Shared numerical cores stay
unchanged; only the exported execute signature and documentation are wrapped.
Run: python tools/build_rcsd_operators.py
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOCK = "common_algorithms/data_processing/process_clock.py"
FEATURES = "common_algorithms/feature_extraction/process_features.py"
CNN = "common_algorithms/models/process_cnn.py"

DESCRIPTIONS = {
    "anomaly_detection": """异常点标记与可选离线去噪；不产生故障标签。

    df: 输入测点表。value_columns 为数值测点列名或列名列表，原单位不变。
    group_columns 应包含设备、原始记录和数据划分，组间不共享统计量。
    time_column 可指定排序时钟；时间须有效且组内唯一。
    outlier_method: mad/none；threshold > 0；action: flag/replace_nan。
    denoise: none/median/mean；window 为正奇数。默认仅标记，不删除故障突变。
    range_limits: {列名: [下限或None, 上限或None]}，单位与测点一致。
    expected_interval > 0，日期时钟按秒，数值时钟沿用其单位。
    居中滤波和逐组MAD是离线处理；训练/支持/查询须分别调用，避免泄漏。
    out_1=result_df: 保留输入结构的处理表；out_2=info_df: 逐行质量标记。
    空表保留列结构；缺列/非法参数抛异常。输入不原地修改。依赖 numpy,pandas。
    """,
    "missing_value": """有界缺失补全和时间一致性检查；不自动创造缺失时刻。

    df 为数值测点表；value_columns 为测点列名或列表；group_columns 应隔离
    设备、原始记录和数据划分。time_column 可选；原物理单位不变。
    method: none/linear/ffill；max_gap 为非负整数，默认整段缺口超长则全部保留缺失。
    expected_interval > 0，日期时钟按秒，数值时钟按其原单位。
    duplicate_policy: raise/flag；source_timezone 仅在已知时区时填写。
    linear 及默认整段缺口判断使用未来上下文，仅供离线。
    在线需 causal=True、method=ffill/none、duplicate_policy=raise，提供时间列时
    必须显式给 expected_interval；只补到当前已出现的 max_gap 行。
    out_1=result_df: 补全后的表；out_2=info_df: 逐行缺失/补全/时钟标记。
    空表保留结构，非法输入抛异常，不修改原表。依赖 numpy,pandas。
    """,
    "process_timeseries": """按设备/记录分组对齐过程参数时钟，不插值、不改变单位。

    df 含 time_column 和数值 value_columns（字符串或列表）。group_columns
    应隔离设备、记录和数据划分。interval_seconds > 0（秒）；
    snap_tolerance_seconds >= 0，仅在容差内吸附，默认0；max_rows 为生成行数上限。
    source_timezone 可选，须为已知时区；朴素时间保留、带时区输入转UTC。
    缺失时刻新建空行并标记，重复或无法匹配的时钟拒绝；不会跨组补值。
    out_1=result_df: 对齐表；out_2=info_df: 逐行时钟质量标记。
    空表按原契约返回空表，非法参数抛异常，不修改原表。依赖 numpy,pandas。
    """,
    "process_window_table": """将连续过程长表切成模型窗口，并记录每个接受/拒绝原因。

    df 必须有 time_column、数值 value_columns。group_columns 必须显式非空，
    应包含设备、原始记录/事件和数据划分。不会自动猜测事件身份或训练/测试划分。
    interval_seconds > 0，window_points >= 2；stride_points 默认等于窗口长。
    quality_columns 可列出布尔坏点标记列；缺失/非有限/坏点窗口拒绝。
    断时钟分段、重复/无效时间拒绝、尾部不足窗记录原因；不插值、不拼接间断数据。
    window_id_prefix 是本批ID前缀；跨批应唯一。重叠窗口不算独立故障事件。
    max_output_rows/max_candidate_windows 是正整数资源上限。
    out_1=result_df: 接受窗口的长表，增加window_id/source_row_position；
    out_2=info_df: 所有候选及尾部的状态、来源、拒绝原因。
    空输入保持约定输出列，输入不修改。依赖 numpy,pandas。
    """,
    "process_features": """从完整的多测点低频窗口提取统计与趋势特征；不处理原始声波。

    df 为长表，含数值 value_columns、window_id_column、time_column。
    每个窗口ID只对应一个设备、记录与划分；组内时钟唯一且连续。
    interval_seconds > 0（秒）。每通道7特征：均值、标准差、最小、最大、
    末值、每小时斜率、相邻点绝对差均值。保留通道顺序和物理量级。
    仅接受有限值且每窗至少2点；不自动补齐、不做逐窗去均值或归一化。
    out_1=result_df: 每窗一行特征；out_2=info_df: feature_name特征字段表。
    空输入返回明确特征列的空表；非法输入抛异常。不修改原表。依赖 numpy,pandas。
    """,
}

MODEL_DOC = """{title}：使用显式挂载的可信模型进行逐窗推理，运行时不训练。

    df 为长表，含 window_id_column、time_column 和数值 value_columns。
    value_columns 必须与 input_contract['channels'] 的名称和顺序一致。
    input_contract 必须声明 signal_type=multivariate_process、channels中每列的
    name/unit/quantity/aggregation、interval_seconds、window_points，并与模型完全一致。
    单位/物理量/聚合/采样间隔不符，缺失、非有限、重复时间、不完整窗均拒绝。
    model_path 必填，由部署环境挂载，禁止使用研发电脑绝对路径。
    本仓RCSD资源是25通道、900秒间隔、20点窗的4种受控半合成状态，
    不可直接用于现场45测点/约6分钟数据，也不代表已验证8种真实机械故障。
    minimum_model_score 可选，范围[0,1]，低于阈值输出unresolved；阈值须另行验证，
    不保证未知故障检测。model_score 未校准，不能当作真实故障概率。
    out_1=result_df: 每窗一行，含窗口ID、start_time、end_time、predicted_label、
    model_score、diagnosis_status；out_2=info_df: 模型类型、窗口数及评分说明。
    空表须带必需列，仍校验模型契约并返回有结构空结果；不修改原表。
    {detail}
    """


def body(path: str, execute_name: str | None = None, drop_execute: bool = False):
    """Inline a source with no project imports or extra public execute entry."""
    text = (ROOT / path).read_text(encoding="utf-8-sig")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    for node in reversed(tree.body):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "__future__" or (node.module or "").startswith("common_algorithms")
        ):
            lines[node.lineno - 1:node.end_lineno] = ["\n"]
        elif isinstance(node, ast.FunctionDef) and node.name == "execute":
            if drop_execute:
                lines[node.lineno - 1:node.end_lineno] = ["\n"]
            elif execute_name:
                lines[node.lineno - 1] = lines[node.lineno - 1].replace("def execute(", f"def {execute_name}(", 1)
    return "".join(lines).strip()


def wrapper(path: str, description: str, kind: str | None = None):
    if kind:
        signature = ("df: pd.DataFrame, value_columns: str | list[str], model_path: str, "
                     "input_contract: dict, window_id_column: str = 'window_id', "
                     "time_column: str = 'timestamp', minimum_model_score: float | None = None, **kwargs")
        call = ("_algorithm_execute(df, value_columns, model_path, input_contract, "
                f"window_id_column, time_column, expected_kind={kind!r}, "
                "minimum_model_score=minimum_model_score, **kwargs)")
    else:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        signature = ast.unparse(function.args)
        args = [n.arg for n in function.args.posonlyargs + function.args.args]
        call = "_algorithm_execute(" + ", ".join(args) + ", **kwargs)"
    formatted_signature = signature.replace(", ", ",\n    ")
    return (f"def execute(\n    {formatted_signature},\n) -> dict[str, pd.DataFrame]:\n"
            '    """' + description.rstrip().removesuffix('"""') + '\n    """\n'
            f"    outputs = {call}\n"
            "    result_df = outputs['out_1']\n"
            "    info_df = outputs['out_2']\n"
            "    return {'out_1': result_df, 'out_2': info_df}\n")


def main():
    specs = []
    for name in ["anomaly_detection", "missing_value", "process_timeseries", "process_window_table"]:
        path = f"common_algorithms/data_processing/{name}.py"
        specs.append((f"data_processing/{name}.py", [CLOCK], path, DESCRIPTIONS[name], None))
    specs.append(("feature_extraction/process_features.py", [CLOCK], FEATURES, DESCRIPTIONS["process_features"], None))
    specs.append(("fault_diagnosis/process_random_forest.py", [CLOCK, FEATURES],
                  "common_algorithms/models/process_forest.py",
                  MODEL_DOC.format(title="随机森林过程诊断", detail="仅加载可信.joblib资源；依赖numpy,pandas,scikit-learn,joblib。"), None))
    for name, kind, title in [("process_cnn_diagnosis", "cnn", "CNN过程诊断"),
                              ("process_meta_learning", "prototype", "元学习编码器与适配原型诊断"),
                              ("process_transfer_learning", "transfer", "预训练后微调的迁移诊断")]:
        detail = "仅加载可信.pt资源，torch.load使用weights_only=True；依赖numpy,pandas,torch。"
        if kind == "prototype":
            detail += "原型推理本身不证明元学习，须核查资源provenance中的任务式训练来源。"
        specs.append((f"fault_diagnosis/{name}.py", [CLOCK], CNN, MODEL_DOC.format(title=title, detail=detail), kind))
    specs.append(("fault_diagnosis/process_seed_ensemble.py", [CLOCK, CNN],
                  "common_algorithms/models/process_seed_ensemble.py",
                  MODEL_DOC.format(title="三个训练种子的等权诊断集成", detail=
                      "仅加载可信.pt集成资源，weights_only=True；依赖numpy,pandas,torch。"
                      "三个成员必须同方法、同类别及物理契约；同一批支持标签复用，不增加真实事件数。"), None))
    for relative, helpers, core, description, kind in specs:
        pieces = [body(p, drop_execute=True) for p in helpers]
        pieces.append(body(core, execute_name="_algorithm_execute"))
        code = ("from __future__ import annotations\n\n" + "\n\n".join(pieces)
                + "\n\n" + wrapper(core, description, kind))
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        header = ("# UTF-8; generated by tools/build_rcsd_operators.py; do not edit generated bodies.\n"
                  "# Python 3.12 verified; imports require preinstalled third-party packages.\n"
                  "# Dependencies: numpy==2.3.5, pandas==3.0.1; model operators additionally\n"
                  "# torch==2.14.1 OR scikit-learn==1.9.1,joblib==1.6.0 as documented in execute.\n"
                  f"# Content SHA256: {digest}\n")
        target = ROOT / "dataflow_operators" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(header + code, encoding="utf-8", newline="\n")
        ast.parse(target.read_text(encoding="utf-8"))
        print(target.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
