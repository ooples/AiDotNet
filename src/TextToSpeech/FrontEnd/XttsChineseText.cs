namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Coqui XTTS's zh_num2words.TextNorm followed by full pypinyin TONE3 transliteration.</summary>
internal static class XttsChineseText
{
    private const string Punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏·〈〉-!\"#$%&'()*+,-./:;<=>?@[\\]^_\x60{|}~";

    internal static string Normalize(string text)
    {
        string numbers = ChineseNswNormalizer.Normalize(text);
        var normalized = new string(numbers.Select(c => Punctuation.IndexOf(c) >= 0 ? ',' : c).ToArray());
        normalized = XttsLatinCleaner.ApplyLexicalRules(normalized, "zh");
        return string.Concat(PypinyinReader.XttsTone3.Read(normalized));
    }
}
