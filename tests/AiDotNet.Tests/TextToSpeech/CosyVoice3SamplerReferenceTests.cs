using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class CosyVoice3SamplerReferenceTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GuidedCosineEulerTrajectory_MatchesReleasedSolverAndDiT(bool streaming)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "cosyvoice3_sampler_reference.json")));
        var reference = document.RootElement;
        double[] Values(JsonElement array) => array.EnumerateArray().Select(value => value.GetDouble()).ToArray();
        var options = new CosyVoice3Options
        {
            CodebookSize = 16, DiTInputDim = 80, SpeakerEmbeddingDim = 80, PreLookaheadChannels = 8,
            DiTDim = 32, DiTDepth = 1, DiTHeads = 2, DiTHeadDim = 16, DiTFeedForwardMultiplier = 2,
            MelChannels = 80, DropoutRate = 0, StaticChunkSize = 1, TokenMelRatio = 2,
            DiTTimesteps = 4, DiTInferenceCfgRate = .7
        };
        var flow = new CosyVoice3Flow<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options, new Random(1));
        flow.Estimator.Load(name => Values(reference.GetProperty("weights").GetProperty(name)), "");
        Tensor<double> Rows(string name) => new(new[] { 6, 80 }, new Vector<double>(Values(reference.GetProperty(name))));
        var speaker = new Tensor<double>(new[] { 80 }, new Vector<double>(Values(reference.GetProperty("speaker"))));
        var sampled = flow.Sample(Rows("mu"), speaker, Rows("condition"), Rows("noise"), streaming);
        var expected = Values(reference.GetProperty("outputs").GetProperty(streaming ? "true" : "false"));
        Assert.Equal(expected.Length, sampled.Length);
        double error = sampled.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 2e-6, $"Maximum solver/CFG error {error:R}; streaming={streaming}.");
    }
}
