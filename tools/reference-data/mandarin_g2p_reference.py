"""Mandarin G2P reference: Plachtaa/VALL-E-X's chinese_to_ipa (utils/g2p/mandarin.py) on a test corpus.

Runs the reference front end (pass a checkout with --vallex; it is not vendored here) on:
- hand-written sentences covering polyphones, numerals, Latin letters and punctuation;
- 2000 seeded sentences of random jieba dictionary words, single characters, numbers and punctuation, which exercise
  jieba's maximum-probability route and HMM and pypinyin's phrase matching across the dictionaries;
and an2cn on a set of numbers, recording tests/AiDotNet.Tests/TextToSpeech/ReferenceData/mandarin_g2p_reference.json.
Needs jieba 0.42.1, pypinyin 0.55.0 and cn2an 0.5.24 (the versions MandarinG2P's resources come from).
Default: verify; --write: regenerate.
"""
import argparse
import json
import os
import random
import sys

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/mandarin_g2p_reference.json"

HAND_WRITTEN = [
    "你好，世界。", "我们去银行取钱，然后在街上行走。", "这个问题很重要，请不要重复。", "他长大以后，量了量桌子的长度。",
    "今天是2023年10月9日，气温是23.5度。", "我有3个苹果和12个梨。", "价格是1001元，打折后是987元。",
    "第100000000位客人得到了奖品。", "请把AI和GPU的资料发给我。", "他说：“我们明天见！”", "你吃饭了吗？",
    "北京是中国的首都、也是一座古老的城市；有很多名胜古迹。", "春眠不觉晓，处处闻啼鸟。", "一二三四五六七八九十。",
    "我喜欢音乐，也喜欢快乐地唱歌。", "这本书的作者是谁？", "朝阳区的朝阳照在河上。", "他还没还钱。", "这件衣服太长了。",
    "大家都觉得他睡觉的样子很可爱。", "这个人姓单，他很单纯。", "银行行长正在行使他的权力。", "数一数有多少个数字。",
    "调查结果显示，空调需要调整。", "我们应该好好学习，天天向上。", "长江和黄河是中国最长的两条河。", "他得了第一名，我们都得努力。",
    "妈妈骂马，马慢。", "小明的爸爸是一位医生。", "今天的天气真好啊！", "我想喝一杯咖啡——不加糖。", "电话号码是13800138000。",
    "这道题的答案是3.14159。", "零点零五", "我在2008年去过北京奥运会。", "他跑了100米，用了9.58秒。", "Hello，你好！",
    "我们的CEO明天会来。", "这是一个测试句子，用来检查分词。", "研究生命的起源。", "结婚的和尚未结婚的人。", "南京市长江大桥。",
    "下雨天留客天留我不留。", "人工智能正在改变世界。", "她的名字叫李华。", "请在第3页的第2行签名。", "我不知道。",
    "这是一只可爱的小猫。", "老师说同学们要认真听讲。", "我们一起去公园散步吧。", "你会说中文吗？", "这里的风景非常美丽。",
    "他每天早上六点起床。", "火车站离这里不远。", "我买了两张电影票。", "这个周末你有空吗？", "鸟儿在树上唱歌。",
    "地球围绕太阳转。", "这是一件非常重要的事情。", "谢谢你的帮助！",
]


def load_reference(vallex):
    sys.path.insert(0, vallex)
    import logging
    logging.disable(logging.CRITICAL)
    import jieba
    jieba.setLogLevel(logging.CRITICAL)
    # utils/g2p/__init__.py imports the Japanese and English cleaners too; load the Mandarin module alone.
    import importlib.util
    spec = importlib.util.spec_from_file_location("vallex_mandarin", os.path.join(vallex, "utils", "g2p", "mandarin.py"))
    mandarin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mandarin)
    import cn2an
    return mandarin, cn2an


def random_sentences(count):
    import jieba
    words = []
    with open(os.path.join(os.path.dirname(jieba.__file__), "dict.txt"), encoding="utf-8") as handle:
        for line in handle:
            words.append(line.split(" ")[0])
    rng = random.Random(2294)
    punctuation = ["，", "。", "！", "？", "、", "；", "：", "—"]
    sentences = []
    for _ in range(count):
        parts = []
        for _ in range(rng.randint(2, 9)):
            choice = rng.random()
            if choice < 0.75:
                parts.append(rng.choice(words))
            elif choice < 0.85:
                word = rng.choice(words)
                parts.append(word[0])
            elif choice < 0.92:
                parts.append(str(rng.randint(0, 99999)) if rng.random() < 0.8 else f"{rng.randint(0, 999)}.{rng.randint(0, 99)}")
            else:
                parts.append(rng.choice(punctuation))
        sentences.append("".join(parts) + rng.choice(["。", "！", "？", ""]))
    return sentences


NUMBERS = ["0", "7", "10", "15", "20", "100", "101", "110", "1001", "1010", "10000", "10010", "100000", "1000000",
           "10000000", "100000000", "100000001", "123456789", "2023", "3.14", "12.0", "0.5", "007", "1000000000000",
           "9999999999999999", "200000010", "40000000001"]


def compute(vallex):
    mandarin, cn2an = load_reference(vallex)
    sentences = HAND_WRITTEN + random_sentences(2000)
    return {
        "sentences": sentences,
        "ipa": [mandarin.chinese_to_ipa(s) for s in sentences],
        "numbers": NUMBERS,
        "an2cn": [cn2an.an2cn(n) for n in NUMBERS],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vallex", required=True, help="a checkout of Plachtaa/VALL-E-X")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    path = os.path.join(root, *FIXTURE.split("/"))
    data = compute(args.vallex)
    data["generator"] = "tools/reference-data/" + os.path.basename(__file__)
    if args.write:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False)
        print("wrote " + FIXTURE)
        return
    with open(path, encoding="utf-8") as handle:
        committed = json.load(handle)
    if committed != data:
        raise SystemExit("MISMATCH: the committed fixture differs from the reference's output")
    print("Mandarin G2P: matches the reference implementation")


if __name__ == "__main__":
    main()
