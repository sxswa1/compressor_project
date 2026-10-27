"""Summarize every fixed RCSD condition, errors, gains and bounded evidence."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
TITLES={'random_forest':'RF','random_forest_selected':'RF＋支持侧筛选','cnn_scratch':'从头CNN',
        'protonet':'ProtoNet元学习','transfer':'普通迁移','supervised_prototype':'普通预训练＋原型',
        'meta_adapted_prototype':'元学习适配＋原型','scratch_prototype':'从头CNN＋原型','source_cnn':'源CNN直接预测'}


def table(frame, columns, header):
    lines=['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(str(row[c]) for c in columns)+' |')
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study-dir',type=Path,default=ROOT/'02_few_shot_diagnosis/outputs/rcsd_diagnosis_v1')
    args=p.parse_args();out=args.study_dir
    complete=json.loads((out/'completion.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    if not complete['all_planned_conditions_completed']:raise ValueError('Finish every condition first')
    metrics=pd.read_csv(out/'metrics.csv');pred=pd.read_csv(out/'predictions.csv');agg=pd.read_csv(out/'aggregate_metrics.csv')
    if len(metrics)!=366 or len(pred)!=32208:raise ValueError('Unexpected full comparison budget')
    pred['correct']=pred.true_class.eq(pred.predicted_class)
    errors=[]
    for key,part in pred.groupby(['variant','k','method'],sort=False):
        normal=part.true_class.eq(0);fault=~normal
        errors.append(dict(zip(['variant','k','method'],key))|{
            'background_to_anomaly':float(part.loc[normal,'predicted_class'].ne(0).mean()),
            'anomaly_to_background':float(part.loc[fault,'predicted_class'].eq(0).mean()),
            'wrong_high_score_count':int((~part.correct & part.model_score.ge(.9)).sum()),
            'predictions':len(part),'unique_query_backgrounds':part.block_id.nunique()})
    pd.DataFrame(errors).to_csv(out/'error_summary.csv',index=False)
    gains=[];rng=np.random.default_rng(2026102602)
    comparisons=[('transfer','cnn_scratch'),('protonet','supervised_prototype'),
                 ('meta_adapted_prototype','scratch_prototype'),('random_forest_selected','random_forest')]
    for variant in agg.variant.unique():
        for k in [1,5,10]:
            for candidate,reference in comparisons:
                part=pred[pred.variant.eq(variant)&pred.k.eq(k)]
                a=part[part.method.eq(candidate)].groupby('block_id').correct.mean()
                b=part[part.method.eq(reference)].groupby('block_id').correct.mean()
                if len(a)!=22 or set(a.index)!=set(b.index):raise ValueError('Paired background coverage differs')
                diff=(a-b.reindex(a.index)).to_numpy()
                draws=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(axis=1)
                low,high=np.quantile(draws,[.025,.975])
                gains.append({'variant':variant,'k':k,'candidate':candidate,'reference':reference,
                    'accuracy_gain':float(diff.mean()),'conditional_ci_low':float(low),'conditional_ci_high':float(high),
                    'backgrounds':22,'bootstrap_draws':10000,'scope':'descriptive within one machine and fixed source/support seeds'})
    pd.DataFrame(gains).to_csv(out/'paired_background_gains.csv',index=False)
    # Figures are scientific summaries of saved predictions, not new training.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=ROOT/'docs/figures';figdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    methods=['random_forest','cnn_scratch','protonet','transfer','meta_adapted_prototype','random_forest_selected']
    english=['RF','CNN scratch','ProtoNet','Transfer','Meta adapted','RF selected']
    for ax,variant,title in zip(axes,['main','weak_lag_challenge'],['Primary simulated states','Weak amplitude + lag challenge']):
        for method,label in zip(methods,english):
            part=agg[agg.variant.eq(variant)&agg.method.eq(method)].sort_values('k')
            ax.plot(part.k,part.accuracy_mean*100,marker='o',label=label)
        source=agg[agg.variant.eq(variant)&agg.method.eq('source_cnn')].accuracy_mean.iloc[0]
        ax.axhline(source*100,color='grey',linestyle=':',label='Source CNN (no target labels)')
        ax.set(title=title,xlabel='K labelled windows per class',ylim=(0,102),xticks=[1,5,10]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Accuracy (%)');axes[1].legend(fontsize=8,loc='lower right')
    fig.suptitle('RCSD semi-synthetic diagnosis | 22 shared query backgrounds | not natural failures')
    fig.tight_layout();fig.savefig(figdir/'rcsd_diagnosis_v1.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,variant in zip(axes,['main','weak_lag_challenge']):
        row=metrics[metrics.variant.eq(variant)&metrics.method.eq('random_forest')&metrics.k.eq(5)&metrics.support_seed.eq(11)].iloc[0]
        cm=np.asarray(json.loads(row.confusion_matrix));ax.imshow(cm,cmap='Blues',vmin=0,vmax=22)
        for i in range(4):
            for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>11 else 'black')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=['BG','Temp','Oil-T','Vib-Ax'],yticklabels=['BG','Temp','Oil-T','Vib-Ax'],xlabel='Predicted',ylabel='True',title=variant)
    fig.suptitle('Predeclared RF delivery | K=5, support seed=11 | no post-test selection')
    fig.tight_layout();fig.savefig(figdir/'rcsd_default_confusion_v1.png',dpi=160);plt.close(fig)
    rows=[]
    for method in TITLES:
        if method=='source_cnn':continue
        item={'method':TITLES[method]}
        for k in [1,5,10]:
            r=agg[agg.variant.eq('main')&agg.method.eq(method)&agg.k.eq(k)].iloc[0]
            item[f'k{k}']=f'{r.accuracy_mean*100:.2f}%';item[f'n{k}']=int(r.n)
        r=agg[agg.variant.eq('main')&agg.method.eq(method)&agg.k.eq(5)].iloc[0]
        item['f1']=f'{r.macro_f1_mean:.3f}';rows.append(item)
    main_table=table(pd.DataFrame(rows),['method','k1','k5','k10','f1'],['方法','K=1准确率','K=5准确率','K=10准确率','K=5宏F1'])
    challenge=[]
    for method in TITLES:
        k=0 if method=='source_cnn' else 5
        r=agg[agg.variant.eq('weak_lag_challenge')&agg.method.eq(method)&agg.k.eq(k)].iloc[0]
        challenge.append({'method':TITLES[method],'k':k,'accuracy':f'{r.accuracy_mean*100:.2f}%',
                          'f1':f'{r.macro_f1_mean:.3f}','background':f'{r.background_recall*100:.2f}%'})
    challenge_table=table(pd.DataFrame(challenge),['method','k','accuracy','f1','background'],['方法','K','准确率','宏F1','背景正确召回'])
    gain_rows=[]
    for row in gains:
        if row['variant']=='main' and row['k']==5:
            gain_rows.append({'comparison':TITLES[row['candidate']]+' vs '+TITLES[row['reference']],
                'gain':f'{row["accuracy_gain"]*100:+.2f}个百分点',
                'ci':f'[{row["conditional_ci_low"]*100:+.2f}, {row["conditional_ci_high"]*100:+.2f}]'})
    gain_table=table(pd.DataFrame(gain_rows),['comparison','gain','ci'],['K=5主要配对比较','平均差','条件95%区间（百分点）'])
    source_main=agg[agg.variant.eq('main')&agg.method.eq('source_cnn')].iloc[0]
    prep=protocol['preparation'];cfg=prep['config']
    text=f'''# RCSD半合成小样本诊断技术与实验报告

日期：2026年10月26日。负责人：凌晨。已完成固定方案的模型训练、测试与资源导出。**这是公开压缩机真实背景上的半合成四状态实验，真实机械故障事件为0；不代表8类故障或现场85%已验证。**

## 任务与数据

本方向负责公共处理、诊断特征、RF/CNN、元学习、迁移及独立算子。复用高淏宇分支f731289a76e8d266a82e3672ee5586e8505150bc的背景筛选、三组变化定义与训练侧幅度规则，不重复开发其预警、DI、风险、RUL。原公开RCSD为离心压缩机，25测点、15分钟间隔；现场可能为往复式、45测点、约6分钟，模态相近不等于模型可直接互用。

原表SHA-256为{cfg['source_sha256']}，未改动。64个120小时背景块：1—6月31块源训练，7—8月11块目标支持池，9—12月22块最终查询。背景只是规则筛选候选，未确认健康。

四状态：未注入背景、模拟温度组升高、模拟油压下降伴轴承温度升高、模拟振动—轴位移变化。油压—温度组不是油温，不能改称已确认润滑失效；振动变化不能自动定位不平衡、不对中或轴承疲劳。曲线和严重度不另算故障类别。

每背景、每状态固定取第240—259点，共20点、首末4.75小时；在看模型分数前固定，全部异常窗在注入及滞后之后。没有将整条轨迹或注入前数据贴成异常。幅度只由源期候选估计，原值未改动的位置保持不变；没有AI自由编造机械标签。主数据256窗，挑战另88窗，总344窗；挑战对同一批查询背景施加半幅度、1.3次响应和8点延迟，单独报告，不新增事故。

源训练124个带构造标签窗口、31背景，目标每类K=1/5/10窗口，共4K个标签，但仅K个配对背景。支持从11块中固定3次抽取并随K嵌套；查询每状态22窗，共88，角色全局互斥。源与目标已知类别相同，是跨时段/背景适配，不称新类、独立设备或跨数据集验证。

## 实现与固定设置

过程统计每通道7项，共175特征；RF150棵树只用目标支持标签。另列支持侧ANOVA取35特征，K=1时无法估类内方差，退回全部特征。归一化只拟合源窗口，查询不拟合。所有方法共享该预处理，源数据和目标标签预算分别记录。

源CNN250步；ProtoNet实际350个4-way 2-support/3-query任务，以原始背景在全任务内分开支持/查询；不同类别版本不能串用同一背景到两边。目标从头CNN、普通迁移和元学习适配均250步、学习率0.002、匹配分类头初始化。源监督预训练与元训练用相同124个源标签，但任务更新不同，不称计算量匹配。

训练种子42/43/44×支持抽样11/22/33；RF不伪造神经初始化重复，只做3次支持抽取。源CNN直接预测不使用目标标签，K=0仅列3个源种子。共366条条件指标、32,208条重复预测；查询仍只有22个背景。源权重在全部目标适配后逐一校验未变。

固定协议、实现与输入哈希在训练前保存。原始表和同学测试结果曾被审阅，这不是全项目从未见过的外部盲测；本轮新诊断查询未用于训练、提前停止或改参数。以下保留全部方法，不按查询挑最高分替换默认资源。

## 主实验

{main_table}

源CNN不适配，K=0平均准确率{source_main.accuracy_mean*100:.2f}%、宏F1 {source_main.macro_f1_mean:.3f}，不与有K个目标标签的方法混淆预算。四类均衡，准确率等于平衡准确率；恒猜一类为25%。不同种子/支持抽样共用同一查询，不是9套独立设备测试。

![不同K的完整主要比较](figures/rcsd_diagnosis_v1.png)

## 弱幅度与滞后挑战

{challenge_table}

挑战与主实验共用22个背景；其中未注入对照值完全重复，不能当新增健康事件。减弱、非线性和滞后同时改变，不能仅凭本结果归因于某一因素。背景→异常与异常→背景另存error_summary.csv，均为本闭集窗口分布下的误分类比例，不是全天系统虚警率。

## 增益与证据边界

{gain_table}

条件区间按22个背景整体配对重采样10,000次，每个背景的状态、种子和支持抽样先聚合，仍仅是此机组、此源训练和此支持选择条件下的描述，不保证自然故障或新设备表现。纯元学习与普通监督预训练＋原型比较，适配元学习与同后端从头原型比较，不能把全部改善都归功于元训练。

本次数据的类别由三组变化模板规定，分类器可能主要识别模板。因此即使某格超过85%，也只证明该半合成设置下的结果；不能证明真实机械机理或项目最终验收完成。模型分数没有故障概率校准，不保证未知故障识别。

## 工程交付

导出25份预先指定资源：3个K值各8种目标方法的seed=42/support_seed=11资源，另有源CNN直接预测资源。默认RF、K=5、固定种子在打分前已规定，没有事后改成最高分方案。模型切换CLI为predict_rcsd_diagnosis.py；输入严格核对25列名称、原生单位、聚合、900秒和20点，缺测/错误契约拒绝。

已有8个独立算子覆盖公共清洗、补缺、对齐、特征及RF/CNN/原型/迁移推理。普通原型和从头原型复用原型推理文件，并不因此称作元学习；真正元学习的资源有训练来源标记。平台卡片、实际注册与运行仍由工程侧配合，当前未做；GitHub仍等用户审核。

数据入口prepare_rcsd_diagnosis.py；独立重建audit_rcsd_diagnosis.py；固定训练run_rcsd_diagnosis.py；分析analyze_rcsd_diagnosis.py。全部代码与配置保留，原始XLSX不放进模型交接包。完整结果在02_few_shot_diagnosis/outputs/rcsd_diagnosis_v1。不要覆盖冻结目录；重新研究应换新版本或新工作副本。

## 尚未完成的最终能力

真实8类故障及85%、真实现场故障精度、跨数据集/跨设备迁移和未知故障能力仍未证明。现场45列及mm/s振动不能改名后调用该25列/um资源；当前现场仅能继续作公共处理接入。半合成作为本阶段正式交付的口径仍待师兄/导师确认，没有将组内方案视为甲方批准。

来源：[RCSD](https://zenodo.org/records/14866092)、[同学数据说明](https://github.com/sxswa1/compressor_project/blob/f731289a76e8d266a82e3672ee5586e8505150bc/docs/高淏宇/数据与接口.md)。
'''
    report=ROOT/'docs/RCSD半合成小样本诊断技术与实验报告--凌晨.md'
    report.write_text(text,encoding='utf-8')
    verification={'all_fixed_conditions_retained':True,'metric_rows':len(metrics),'predictions':len(pred),
        'natural_fault_incidents':0,'query_backgrounds':22,'report_uses_all_methods_and_separates_challenge':True,
        'new_training_performed_in_analysis':False,'default_method_changed_after_test':False}
    (out/'analysis_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
