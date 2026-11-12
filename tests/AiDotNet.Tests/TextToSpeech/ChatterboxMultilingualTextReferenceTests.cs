using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.VoiceCloning;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>The multilingual Chatterbox's text front end against the release (tools/reference-data/
/// chatterbox_multilingual_text_reference.py, dicta_diacritizer_reference.py). pkuseg's cuts run with <c>PKUSEG_MODEL_DIR</c>
/// (the spacy_ontonotes model), the release's tokenization with <c>CHATTERBOX_MTL_DIR</c> too (grapheme_mtl_merged_expanded_v1.json
/// and Cangjie5_TC.json), Dicta with <c>DICTA_DIR</c> (dictabert-large-char-menaked).</summary>
public class ChatterboxMultilingualTextReferenceTests
{
    private static string? Directory(string variable) =>
        Environment.GetEnvironmentVariable(variable) is { Length: > 0 } path && System.IO.Directory.Exists(path) ? path : null;

    [Fact]
    public void PunctuationNormalization_MatchesPuncNorm()
    {
        var f = WhisperSpeechT2SReferenceTests.Load("chatterbox_multilingual_text_reference.json");
        foreach (var row in f["punc_norm"]!)
            Assert.Equal((string)row["normalized"]!, ChatterboxMultilingualText.NormalizePunctuation((string)row["text"]!));
    }

    [Fact]
    public void PythonLower_KeepsFinalSigmaAndDottedI()
    {
        // Python: "ΑΣ ΣΑ".lower() == "ας σα" (final sigma only at a word's end); "İ".lower() == "i̇".
        string sigma = ((char)0x03A3).ToString(), alpha = ((char)0x0391).ToString();
        string lower = PythonString.Lower(alpha + sigma + " " + sigma + alpha);
        Assert.Equal(new[] { 0x03B1, 0x03C2, 0x20, 0x03C3, 0x03B1 }, lower.Select(c => (int)c).ToArray());
        Assert.Equal(new[] { 0x69, 0x0307, 0x73 }, PythonString.Lower(((char)0x0130).ToString() + "S").Select(c => (int)c).ToArray());
    }

    [Fact]
    public void Pkuseg_CutsLikeSpacyPkuseg()
    {
        var model = Directory("PKUSEG_MODEL_DIR");
        if (model is null) return;
        var segmenter = Pkuseg.Load(model);
        var f = WhisperSpeechT2SReferenceTests.Load("chatterbox_multilingual_text_reference.json");
        foreach (var row in f["pkuseg"]!)
            Assert.Equal(row["words"]!.Values<string>().ToArray(), segmenter.Cut((string)row["text"]!).ToArray());
    }

    [Fact]
    public void Encode_MatchesMtlTokenizer()
    {
        var model = Directory("PKUSEG_MODEL_DIR");
        var release = Directory("CHATTERBOX_MTL_DIR");
        if (model is null || release is null) return;
        var segmenter = new Lazy<Pkuseg>(() => Pkuseg.Load(model));
        var text = new ChatterboxMultilingualText(ChatterboxTokenizer.Load(Path.Combine(release, "grapheme_mtl_merged_expanded_v1.json")),
            Path.Combine(release, "Cangjie5_TC.json"), () => segmenter.Value,
            _ => throw new InvalidOperationException("No Hebrew here."), _ => throw new InvalidOperationException("No Russian here."));
        var f = WhisperSpeechT2SReferenceTests.Load("chatterbox_multilingual_text_reference.json");
        foreach (var row in f["encode"]!)
        {
            string normalized = ChatterboxMultilingualText.NormalizePunctuation((string)row["text"]!);
            string language = (string)row["language"]!;
            Assert.True(row["ids"]!.Values<int>().SequenceEqual(text.Encode(normalized, language)), $"{language}: {normalized}");
            Assert.True(row["no_language"]!.Values<int>().SequenceEqual(text.Encode(normalized, null)), $"(none): {normalized}");
        }
    }

    [Fact]
    public void Dicta_DiacritizesLikeDictaOnnx()
    {
        var model = Directory("DICTA_DIR");
        if (model is null) return;
        var dicta = DictaDiacritizer.Load(model);
        var f = WhisperSpeechT2SReferenceTests.Load("dicta_diacritizer_reference.json");
        foreach (var row in f["sentences"]!)
        {
            string sentence = (string)row["text"]!;
            var (ids, offsets) = dicta.Tokenize(sentence);
            Assert.Equal(row["ids"]!.Values<int>().ToArray(), ids);
            Assert.Equal(row["offsets"]!.Select(o => (o[0]!.Value<int>(), o[1]!.Value<int>())).ToArray(), offsets.Select(o => (o.Start, o.End)).ToArray());
            Assert.Equal((string)row["diacritized"]!, dicta.AddDiacritics(sentence));
        }
    }
}
