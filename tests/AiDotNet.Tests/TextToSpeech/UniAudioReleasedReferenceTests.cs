using System.IO.Compression;
using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class UniAudioReleasedReferenceTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", name);

    [Fact]
    public void ReleasedPreset_MatchesOriginalArchiveAxesAndCausality()
    {
        var options = UniAudioOptions.OfficialV1();
        Assert.Equal(16, options.NumHeads);
        Assert.Equal(4215, options.JointVocabularySize);
        Assert.Equal(4215, options.VocabSize);
        Assert.Equal(8000, options.MaxContextPatches);
        Assert.Equal(501, options.PhoneVocabularySize);
        Assert.Equal(256, options.CodecDimension);
        Assert.Equal(8, options.CodecTotalQuantizers);
        Assert.False(options.PrefixLanguageModel);
        Assert.True(options.UseReleasedTextFrontend);
        Assert.False(new UniAudioOptions().PrefixLanguageModel); // Table 8: both Transformers are causal.
        Assert.True(new UniAudioOptions(options).UseReleasedTextFrontend);
    }

    [Fact]
    public void ReleasedTextAndPhoneInventory_MatchUnchangedAuthorMethods()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Fixture("uniaudio_frontend_reference.json")));
        var root = document.RootElement;
        Assert.Equal(root.GetProperty("phone_inventory").EnumerateObject().Count(), UniAudioReleasedTokenizer.PhoneInventory.Count);
        foreach (var phone in root.GetProperty("phone_inventory").EnumerateObject())
            Assert.Equal(phone.Value.GetInt32(), UniAudioReleasedTokenizer.PhoneInventory[phone.Name]);
        foreach (var item in root.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()),
                UniAudioReleasedTokenizer.Tokenize(item.GetProperty("text").GetString()!, 500));
        foreach (var item in root.GetProperty("phone_cases").EnumerateArray())
            Assert.Equal(item.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()),
                UniAudioReleasedTokenizer.TokenizePhones(item.GetProperty("phones").GetString()!, 500));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CausalAndPrefixLoss_AllLogitGradientsMatchAuthorObjective(bool prefix)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Fixture("uniaudio_objective_reference.json")));
        var item = document.RootElement.GetProperty("cases").EnumerateArray()
            .Single(x => x.GetProperty("prefix_lm").GetBoolean() == prefix);
        var logits = new Tensor<double>(new[] { 5, 7 },
            new Vector<double>(item.GetProperty("logits").EnumerateArray().Select(x => x.GetDouble()).ToArray()));
        var labels = item.GetProperty("labels").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        using var tape = new GradientTape<double>();
        var loss = UniAudioOps<double>.TeacherForcedLoss(AiDotNetEngine.Current, logits, labels, 2, prefix, 0);
        Assert.Equal(item.GetProperty("loss").GetDouble(), loss[0], 12);
        var gradient = tape.ComputeGradients(loss, sources: new[] { logits })[logits];
        var expected = item.GetProperty("gradient").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], gradient[i], 12);
    }

    [Theory]
    [InlineData("", "g_emb.weight")]
    [InlineData("model.", "g_emb.weight")]
    [InlineData("codec_model.", "encoder.model.0.conv.conv.weight_v")]
    public void AuthorReader_HandlesReleasedWrappersAndRejectsWrongAxes(string prefix, string marker)
    {
        var tensor = new Tensor<float>(new[] { 2, 3 }, new Vector<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }));
        var reader = UniAudio<double>.AuthorStateReader(new Dictionary<string, Tensor<float>> { [prefix + marker] = tensor }, marker);
        Assert.Equal(new[] { 1d, 2d, 3d, 4d, 5d, 6d }, reader(marker, new[] { 2, 3 }));
        Assert.Throws<InvalidDataException>(() => reader(marker, new[] { 3, 2 }));
        Assert.Throws<InvalidDataException>(() => reader("missing", new[] { 2, 3 }));
    }

    [Fact]
    public void FilteredTorchImport_NeverReadsAnUnselectedOptimizerStorage()
    {
        string temporary = Path.Combine(Path.GetTempPath(), "uniaudio-filter-" + Guid.NewGuid().ToString("N") + ".pth");
        try
        {
            using (var source = ZipFile.OpenRead(Fixture("uniaudio_wrapped_lm_reference.pth")))
            using (var target = ZipFile.Open(temporary, ZipArchiveMode.Create))
            {
                var storages = source.Entries.Where(x => x.FullName.Contains("/data/"))
                    .OrderBy(x => int.Parse(x.FullName.Split('/').Last())).ToArray();
                Assert.Equal(59, storages.Length); // 58 separately saved author tensors, then optimizer momentum.
                string omitted = storages.Last().FullName;
                foreach (var entry in source.Entries.Where(x => x.FullName != omitted))
                {
                    using var input = entry.Open();
                    using var output = target.CreateEntry(entry.FullName).Open();
                    input.CopyTo(output);
                }
            }
            var loader = new WeightLoader();
            var state = loader.LoadWeights(temporary, name => name.StartsWith("model.", StringComparison.Ordinal));
            Assert.Equal(58, state.Count);
            Assert.All(state.Keys, key => Assert.StartsWith("model.", key));
            Assert.Equal(new[] { 64, 4 }, state["model.g_emb.weight"].Shape.ToArray());
            Assert.Throws<InvalidDataException>(() => loader.LoadWeights(temporary));
        }
        finally
        {
            File.Delete(temporary);
        }
    }
}
