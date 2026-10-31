"""Reference fixture for GPT-SoVITS's Chinese front end: text/cleaner.clean_text(text, "zh", version) for v1 and v2.

Runs GPT-SoVITS's own text package (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image (pypinyin 0.55.0,
jieba_fast 0.53, OpenCC 1.4.2, onnxruntime 1.23.2; see gptsovits_frontend_reference.py), with the G2PW model it downloads
(G2PWModel_1.1) and the chinese-roberta-wwm-ext-large tokenizer it reads (bert_path). Records, per sentence, the
normalized text, the phones and the phones per character of v1 (chinese.py) and v2 (chinese2.py), G2PW's raw readings
of each v2 segment, jieba_fast's part-of-speech cut of each segment, and OpenCC s2tw of each segment; G2PW's
probabilities for the first queries; zh_normalization's output on number cases; jieba_fast's lcut and lcut_for_search
of a spread of text; and the tone sandhi's readings of a spread of jieba's dictionary words.

Run from the GPT-SoVITS checkout (chinese2 opens GPT_SoVITS/text/G2PWModel relative to it):

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -v <G2PWModel_1.1>:/ref/GPT-SoVITS/GPT_SoVITS/text/G2PWModel \
        -v <roberta snapshot>:/roberta -e bert_path=/roberta -w /ref/GPT-SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_chinese_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_chinese_reference.json.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))
os.environ.setdefault("version", "v2")

from text import chinese, chinese2, cleaner  # noqa: E402
from text.symbols import punctuation  # noqa: E402
from text.zh_normalization.text_normlization import TextNormalizer  # noqa: E402
import jieba_fast  # noqa: E402
import jieba_fast.posseg as psg  # noqa: E402
from opencc import OpenCC  # noqa: E402

SENTENCES = [
    "你好，世界！",
    "这是一个示例文本：你好！这是一个测试....",
    "啊！但是《原神》是由,米哈游自主，[研发]的一款全新开放世界冒险游戏",
    "我有3个苹果和12.5公斤的米，还有2024年的1000000元。",
    "看一看，听一听，一天一夜，一段时间，第一名，一零零。",
    "不怕，不要，看不懂，不是不行，对不起。",
    "我很好，你也好吗？所有人都喜欢纸老虎和蒙古包。",
    "展览馆里有很多老虎，小李想买五把雨伞。",
    "奶奶和妈妈在桌上吃饭，孩子们上来下去。",
    "这个东西怎么样？我们的朋友在哪儿？",
    "老头儿在胡同儿里玩儿，一点儿也不累。",
    "嗯，呣，好吧，走吧，是呢，对啊，哈哈。",
    "他说：“明天见！”她回答：‘好的’——再见～",
    "Hello你好world世界，chemistry课程。",
    "我买了两个、三个和几个西瓜；每个都很甜。",
    "试试看，想想办法，研究研究，高高兴兴。",
    "电脑软件需要更新，手机屏幕坏了。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "早上好，今天天气真不错，我们去公园散步吧。",
    "请把窗户打开，让新鲜空气进来。",
    "一二三四五六七八九十，百千万亿。",
    "Ｗｉ－Ｆｉ信号很弱(真的)。",
    "他是一位非常有名的科学家，研究过很多问题。",
    "我想起来了，你说的那本书在书架上。",
    "能不能帮我一下？好不好？行不行？",
    "會議在2023年5月12日下午3:30開始，預計持續到5:45。",
    "溫度是-5°C到10℃之間，濕度百分之85%，風速12km每小時。",
    "电话号码是13812345678，座机010-12345678，客服400-123-4567。",
    "3/4的人同意，1+1=2，x²+y²=r²，版本号1.2.3.4。",
    "他走了，看着窗外，去过北京；长大了，银行行长重新说了一遍。",
    "我们一起去看看吧，女儿和孩儿们在花园儿里玩儿。",
    "你干嘛呢？朝阳区的重庆路上有一家银行，行人很多。",
    "这本书很好看，那个人很好，我喜欢好学的学生。",
    "我们在2019-08-15见过面，大约20~30个人参加，价格是3.5元。",
    "小明的作业做完了没有？还没呢，他还在着急地写着。",
    "＃＆＠这些符号会被删掉，但是逗号、句号。会保留。",
    "β版本的α测试，π约等于3.14159。",
    "一会儿，一下子，一模一样，一心一意，一五一十。",
    "长城很长，音乐让人快乐，还有重要的重复。",
    "那里有个人，这儿有两个人，还有半个月。",
    "……真的吗？！是的！！！",
    "给你，给我，供给，答应，回答，应该。",
]

NUMBER_CASES = [
    "2024年3月5日", "15:30", "08:05:09", "9:30~10:30", "2021/02/03", "2021-12-31", "-3.50", "0.5", ".25", "12.30",
    "-12", "007", "3个", "2条", "20多个", "5+人", "1.2.3", "3/4", "-1/2", "50%", "-3.5%", "25°C", "-10℃", "37.5度",
    "100摄氏度", "5cm2", "3kg", "12km", "100m", "2ml", "15s", "13912345678", "+86 13912345678", "021-87654321",
    "400-800-1234", "1+2=3", "a-b", "x²", "10³", "2ⁿ", "1-2", "3~5", "10%~20%", "①②③", "１２３ＡＢＣ", "1000000000",
    "100000001", "1010", "10", "110", "100", "0", "00", "3.0", "3.00", "10.10",
]


def opencc_cases():
    cc = OpenCC("s2tw")
    cases = SENTENCES + ["后来", "头发", "干净的干部", "皇后", "里面", "一只", "只有", "面条", "发现", "钟表", "台湾", "软件",
                         "鼠标", "打印机", "数据库", "网络", "出租车", "自行车", "冰淇淋", "信息", "⿰氵每", "㐀㐁", "𠀀𠀁"]
    return [{"text": t, "s2tw": cc.convert(t)} for t in cases]


def main():
    out = []
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    probability_queries = []
    for sentence in SENTENCES:
        entry = {"text": sentence}
        for version, module in (("v1", chinese), ("v2", chinese2)):
            norm = module.text_normalize(sentence)
            entry[f"{version}_normalized"] = norm
            try:
                phones, word2ph = module.g2p(norm)
                entry[f"{version}_phones"] = phones
                entry[f"{version}_word2ph"] = word2ph
            except Exception as error:  # noqa: BLE001
                entry[f"{version}_fails"] = type(error).__name__
            try:
                clean = cleaner.clean_text(sentence, "zh", version)
                entry[f"{version}_clean"] = clean[0]
            except Exception as error:  # noqa: BLE001
                entry[f"{version}_clean_fails"] = type(error).__name__
        norm = entry["v2_normalized"]
        segments = [re.sub("[a-zA-Z]+", "", s) for s in re.split(pattern, norm) if s.strip() != ""]
        batch = [s for s in segments if s]
        entry["segments"] = segments
        entry["g2pw"] = chinese2.g2pw._g2pw(batch) if batch else []
        entry["cuts"] = [[[w, p] for w, p in psg.lcut(s)] for s in segments]
        out.append(entry)
        if len(probability_queries) < 3 and batch:
            probability_queries.append(batch[0])

    # G2PW's probabilities: the converter's model input for a sentence, run through its session.
    from text.g2pw.onnx_api import prepare_onnx_input
    converter = chinese2.g2pw._g2pw
    probabilities = []
    for sentence in probability_queries:
        translated = converter.cc.convert(sentence)
        texts, query_ids, result_ids, _, _ = converter._prepare_data([translated])
        model_input = prepare_onnx_input(tokenizer=converter.tokenizer, labels=converter.labels,
                                         char2phonemes=converter.char2phonemes, chars=converter.chars, texts=texts,
                                         query_ids=query_ids, use_mask=converter.config.use_mask, window_size=None,
                                         char2id=converter.char2id, char_phoneme_masks=converter.char_phoneme_masks)
        probs = converter.session_g2pw.run([], {
            "input_ids": model_input["input_ids"], "token_type_ids": model_input["token_type_ids"],
            "attention_mask": model_input["attention_masks"], "phoneme_mask": model_input["phoneme_masks"],
            "char_ids": model_input["char_ids"], "position_ids": model_input["position_ids"]})[0]
        rows = []
        for i in range(len(texts)):
            labels = [j for j in range(probs.shape[1]) if model_input["phoneme_masks"][i][j] > 0]
            rows.append({"text": texts[i], "query": query_ids[i], "input_ids": model_input["input_ids"][i].tolist(),
                         "position": int(model_input["position_ids"][i]), "char_id": int(model_input["char_ids"][i]),
                         "labels": labels, "probabilities": [float(probs[i][j]) for j in labels]})
        probabilities.append({"sentence": sentence, "queries": rows})

    normalizer = TextNormalizer()
    numbers = [{"text": t, "sentences": normalizer.normalize(t)} for t in NUMBER_CASES]

    with open(os.path.join(os.path.dirname(jieba_fast.__file__), "dict.txt"), encoding="utf-8") as handle:
        entries = [line.split() for line in handle if line.strip()]
    words = []
    for word, _, tag in entries[::580]:
        initials, finals = chinese._get_initials_finals(word)
        record = {"word": word, "pos": tag, "initials": initials}
        try:
            record["finals"] = chinese.tone_modifier.modified_tone(word, tag, finals)
        except IndexError:
            record["fails"] = True
        record["search"] = jieba_fast.lcut_for_search(word)
        words.append(record)
    # jieba_fast's plain lcut over runs of dictionary words joined together (unknown-word HMM and route ties).
    joined = ["".join(e[0] for e in entries[i:i + 12]) for i in range(7, len(entries), 4111)]
    cuts = [{"text": t, "cut": jieba_fast.lcut(t), "search": jieba_fast.lcut_for_search(t)} for t in joined]
    cuts += [{"text": s, "cut": jieba_fast.lcut(s), "search": jieba_fast.lcut_for_search(s)} for s in SENTENCES]

    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "gptsovits_chinese_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"sentences": out, "probabilities": probabilities, "numbers": numbers, "opencc": opencc_cases(),
                   "words": words, "jieba": cuts}, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(out)} sentences, {len(numbers)} number cases, {len(words)} words, {len(cuts)} cuts")


if __name__ == "__main__":
    main()
