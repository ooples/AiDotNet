using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.VoiceCloning;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>ChatterboxOptions.Variant chooses the release whose text front end and files the model uses.</summary>
public class ChatterboxVariantTests
{
    private static Chatterbox<double> Model(ChatterboxOptions options) =>
        new(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    [Fact]
    public void Multilingual_UsesTheReleaseVocabularyAndVersion2ByDefault()
    {
        var options = ChatterboxOptions.Multilingual();
        Assert.Equal(ChatterboxVariant.MultilingualV2, options.Variant);
        Assert.Equal(2454, options.VocabSize);
        Assert.Equal(ChatterboxVariant.MultilingualV3, ChatterboxOptions.Multilingual(ChatterboxVariant.MultilingualV3).Variant);
        Assert.Throws<ArgumentException>(() => ChatterboxOptions.Multilingual(ChatterboxVariant.English));
        Assert.Equal(ChatterboxVariant.English, new ChatterboxOptions().Variant);
    }

    [Fact]
    public void Variant_RoutesTokenizationToItsFrontEnd()
    {
        // The English model tokenizes with its BPE; a multilingual one needs the Cangjie mapping of its front end.
        Assert.NotEmpty(ChatterboxFixture.Create<double>().Tokenize("hello"));
        var options = ChatterboxFixture.Options();
        options.Variant = ChatterboxVariant.MultilingualV2;
        var error = Assert.Throws<InvalidOperationException>(() => Model(options).Tokenize("hello"));
        Assert.Contains("Cangjie5_TC.json", error.Message);
    }

    [Fact]
    public void Multilingual_TokenizesAsTheRelease()
    {
        string? release = Environment.GetEnvironmentVariable("CHATTERBOX_MTL_DIR");
        if (release is null || !Directory.Exists(release)) return;
        var options = ChatterboxFixture.Options();
        options.Variant = ChatterboxVariant.MultilingualV2;
        options.TokenizerPath = Path.Combine(release, "grapheme_mtl_merged_expanded_v1.json");
        options.CangjieMappingPath = Path.Combine(release, "Cangjie5_TC.json");
        var f = WhisperSpeechT2SReferenceTests.Load("chatterbox_multilingual_text_reference.json");
        foreach (var row in f["encode"]!.Where(r => (string)r["language"]! is "en" or "fr" or "ko"))
        {
            options.LanguageId = (string)row["language"]!;
            string normalized = ChatterboxMultilingualText.NormalizePunctuation((string)row["text"]!);
            Assert.Equal(row["ids"]!.Values<int>().ToArray(), Model(new ChatterboxOptions(options)).Tokenize(normalized));
        }
    }
}
