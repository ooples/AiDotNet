using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// Amphion's <c>g2p_</c> (MaskGCT's and Vevo's text front end, <c>models/tts/maskgct/g2p</c>): Chinese and English through
/// <c>chn_eng_g2p</c> (the polyphone model for Chinese), Japanese through Open JTalk, the rest through eSpeak NG, mapped to
/// the release's phone ids. The language front ends load on first use.
/// </summary>
internal sealed class AmphionPhonemizer<T>
{
    private readonly IEngine _engine;
    private readonly Func<string?> _polyphoneModelPath;
    private readonly Func<string?> _openJTalkDictionaryPath;
    private readonly string _owner;
    private MaskGctEspeakFrontEnd? _espeak;
    private MaskGctMandarinFrontEnd? _mandarin;
    private MaskGctJapaneseFrontEnd? _japanese;

    /// <param name="owner">The options class whose paths the errors name.</param>
    public AmphionPhonemizer(IEngine engine, Func<string?> polyphoneModelPath, Func<string?> openJTalkDictionaryPath, string owner)
    {
        _engine = engine;
        _polyphoneModelPath = polyphoneModelPath;
        _openJTalkDictionaryPath = openJTalkDictionaryPath;
        _owner = owner;
    }

    /// <summary>The phone string and the phone ids of <paramref name="text"/> in <paramref name="language"/>.</summary>
    public (string Phonemes, int[] Ids) Phonemize(string text, string language)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (language is null) throw new ArgumentNullException(nameof(language));
        _espeak ??= new MaskGctEspeakFrontEnd();
        Func<string, string, string>? chinese = null;
        Func<string, string>? japanese = null;
        if (language is "zh" or "en")
            chinese = (segment, _) => Mandarin().ChineseToIpa(segment);
        if (language == "ja") japanese = Japanese().JapaneseToIpa;
        return _espeak.G2P(text, language, chinese, japanese);
    }

    private MaskGctMandarinFrontEnd Mandarin()
    {
        if (_mandarin is not null) return _mandarin;
        string path = _polyphoneModelPath()
            ?? throw new InvalidOperationException($"Chinese text needs the release's polyphone model; set {_owner}.PolyphoneModelPath to poly_bert_model.onnx.");
        var model = new MaskGctPolyphoneModel<T>(_engine);
        model.Load(path);
        return _mandarin = new MaskGctMandarinFrontEnd(model.Predict);
    }

    private MaskGctJapaneseFrontEnd Japanese()
    {
        if (_japanese is not null) return _japanese;
        string path = _openJTalkDictionaryPath()
            ?? throw new InvalidOperationException($"Japanese text needs the Open JTalk dictionary; set {_owner}.OpenJTalkDictionaryPath to open_jtalk_dic_utf_8-1.11.");
        return _japanese = new MaskGctJapaneseFrontEnd(new OpenJTalk(path));
    }
}
