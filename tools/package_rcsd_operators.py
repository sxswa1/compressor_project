"""Create an allowlisted, data-free operator/model handover ZIP.

Includes exact published operators, 39 trusted resources and synthetic demo.
No raw/derived sensor observations, Git history or private workspace paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing delivery; choose a new filename")
    namespace = {}
    exec((ROOT / "dataflow_operators/registry.py").read_text(encoding="utf-8"), namespace)
    files = ["dataflow_operators/registry.py", "dataflow_operators/README.md",
             "02_few_shot_diagnosis/scripts/run_rcsd_demo.py",
             "02_few_shot_diagnosis/requirements.txt", "tools/verify_rcsd_handover.py",
             "docs/凌晨/算子说明.md", "docs/凌晨/测试报告.md"]
    files += ["dataflow_operators/" + value["path"] for value in namespace["LINGCHEN_OPERATORS"].values()]
    resource = ROOT / "02_few_shot_diagnosis/resources/rcsd_v2"
    files += [path.relative_to(ROOT).as_posix() for path in resource.rglob("*") if path.is_file()]
    files += ["02_few_shot_diagnosis/results/verification/" + name for name in
              ["upload_operator_checks.json", "upload_regression.xml"]]
    data = {name: (ROOT / name).read_bytes() for name in files}
    for case in json.loads((resource / "resource_manifest.json").read_text(encoding="utf-8"))["cases"]:
        path = resource / case["model"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError("Model differs from published resource")
    data["README.md"] = """# 凌晨：DataFlow算子与模型交接

本包提供10个独立UTF-8算子.py和39份固定模型，无现场或公开传感器原值。
算子位于dataflow_operators，资源位于02_few_shot_diagnosis/resources/rcsd_v2。
按最新规范交接单个算子.py；推理算子还须挂载匹配的模型并传入路径。
不另交平台卡片JSON/YAML，卡片由维护方生成。registry.py是仓库清单，
不是平台数据库已注册的证明。平台依赖安装、资源挂载和实际联调仍待完成。

安装：python -m pip install -r 02_few_shot_diagnosis/requirements.txt
演示：python run_demo.py
检查：python tools/verify_rcsd_handover.py --output verification_local.json

演示使用人工数列，仅验证接口，不计算准确率。资源为RCSD25测点、15分钟、
20点窗口的半合成4状态模型，不可直接套用现场45测点/约6分钟数据。
model_score未校准为真实故障概率，真实故障覆盖及跨数据集验证尚未完成。
完整研究源码、实验报告、结果与复现说明请查看GitHub小样本诊断目录。
checksums.json可核对文件完整性。
""".encode("utf-8")
    data["run_demo.py"] = b"from pathlib import Path\nimport subprocess,sys\nroot=Path(__file__).resolve().parent\nsubprocess.run([sys.executable,'-I','-X','utf8',str(root/'02_few_shot_diagnosis/scripts/run_rcsd_demo.py')],cwd=root,check=True)\n"
    data["checksums.json"] = (json.dumps({name: hashlib.sha256(body).hexdigest() for name, body in data.items()},
                                         ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(data.items()):
            archive.writestr(name, body)
    print(json.dumps({"zip": str(args.output), "members": len(data), "operator_count": 10,
                      "resources": 39, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
