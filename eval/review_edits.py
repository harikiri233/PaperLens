"""人工审核记录：对 LLM 生成的候选题所做的修改（qa_draft.jsonl → qa.jsonl）。

原则：问题要能单独看懂（点明论文或方法）、不照抄原文、类别名副其实；参考答案已逐条对照来源块核实。
"""

import sys
from pathlib import Path

from paperlens.eval.dataset import load_items, save_items

EDITS = {
    # 没点明论文：ReAct 论文本身也做 ALFWorld，问题有歧义
    "q01": "In the Reflexion paper's ALFWorld analysis, at what hallucination rate does the ReAct-only baseline agent converge?",
    # "uses T5 的那篇" 指代不明
    "q07": "In the Fusion-in-Decoder paper, how many passages are retrieved per question by default, and how long are they truncated to?",
    "q22": "RETRO 的实验中，把评估时的检索库从 Wikipedia 扩大到整个 MassiveText，语言建模效果怎样变化？",
    # 原题问的是"为什么没评估困惑度"，不属于术语题；改成问论文里的数据集记号
    "q27": "Toolformer 论文的语言建模实验里，C* 这个数据集指的是什么？",
    "q31": "Reflexion 论文的代码生成实验里用到的 MultiPL-E 是什么？",
    # 跨语言题：补上中文描述的论文，去掉"数量级"这种不准确的说法
    "q36": "在那篇让模型学会按需检索、生成并用反思标记自我批判的论文里，分析训练数据规模时，作者从原始训练集中随机抽取了哪几种规模的子集来微调？",
    "q37": "在解码器端融合多段证据的开放域问答方法中，编码器逐段独立处理、解码器联合处理所有段落，这样设计在计算量上有什么好处？",
    "q38": "在我的世界游戏里做终身学习的“航行者”智能体中，一段程序通过自我验证并存入技能库之后，系统接下来做什么来确定下一个目标？",
    "q39": "在研究众包能否用于检索增强生成评估的论文里，作者用什么方法估计标注者能力并生成校正后的标准标签？",
    "q40": "训练阿特拉斯检索增强模型时，维基百科被切成了多少段？每段平均多少个词？",
}

# 问题改了，参考答案跟着改（均可在来源块中找到）
ANSWERS = {
    "q27": "C* 是用 API 调用标注过的数据集（在 CCNet 子集上插入 API 调用后得到），C 是未标注的原始数据。",
}


def main() -> None:
    items = load_items(Path("eval/qa_draft.jsonl"))
    for item in items:
        if item.id in EDITS:
            item.question = EDITS[item.id]
        if item.id in ANSWERS:
            item.answer = ANSWERS[item.id]
    save_items(Path("eval/qa.jsonl"), items)
    print(f"修改 {len(EDITS)} 题，共 {len(items)} 题", file=sys.stderr)


if __name__ == "__main__":
    main()
