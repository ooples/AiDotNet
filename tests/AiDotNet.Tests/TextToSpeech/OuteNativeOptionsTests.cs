using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class OuteNativeOptionsTests
{
    private static OuteTTSOptions Options() => new()
    {
        HiddenDim = 16, VocabSize = 64, Layers = 1, Heads = 2, KeyValueHeads = 1, HeadDim = 8, Intermediate = 32,
        CodecEncoderDim = 2, CodecEncoderRates = new[] { 4, 8, 10 }, CodecLatentDim = 16,
        CodecDecoderDim = 8, CodecDecoderRates = new[] { 10, 8, 4 }, CodecCodebookDim = 4,
        AlignerMels = 80, AlignerWidth = 16, AlignerHeads = 2, AlignerEncoderLayers = 1, AlignerDecoderLayers = 1,
        AlignerTemperatures = new[] { 0.0 },
    };

    private static OuteTTS<double> Create(OuteTTSOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
        { RandomSeed = 11 }, options);

    private static double Loss(OuteTTS<double> model)
    {
        using var _ = new NoGradScope<double>();
        int[] ids = { 1, 3, 5, 7, 9, 11, 13, 15 };
        return model.Loss(ids, ids)[0];
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void SharedHeadsResizeActualCausalAttention(int heads)
    {
        var options = Options(); options.NumHeads = heads;
        var model = Create(options);
        Assert.Equal(heads, model.LanguageModel.Configuration.Heads);
        Assert.Equal(heads * options.HeadDim * options.HiddenDim, model.LanguageModel.Layers[0].Query.ParameterCount);
        Assert.True(TtsPaperNumerics.IsFinite(Loss(model)));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void SharedDepthBuildsActualCausalBlocks(int depth)
    {
        var options = Options(); options.NumDecoderLayers = depth;
        var model = Create(options);
        Assert.Equal(depth, model.LanguageModel.Layers.Count);
        Assert.True(TtsPaperNumerics.IsFinite(Loss(model)));
    }

    [Fact]
    public void DropoutPreservesInitializationAndEvaluationButChangesNativeTrainingLoss()
    {
        var zero = Options(); var dropout = Options(); dropout.DropoutRate = .5;
        var a = Create(zero); var b = Create(dropout);
        Assert.Equal(a.GetParameters().ToArray(), b.GetParameters().ToArray());
        Assert.Equal(.5, b.LanguageModel.Configuration.AttentionDropout);
        a.SetTrainingMode(false); b.SetTrainingMode(false);
        double evaluation = Loss(a);
        Assert.Equal(evaluation, Loss(b));
        b.SetTrainingMode(true);
        Assert.NotEqual(evaluation, Loss(b));
        b.SetTrainingMode(false);
        Assert.Equal(evaluation, Loss(b));
    }

    [Theory]
    [InlineData("RopeLowFrequencyFactor")]
    [InlineData("RopeHighFrequencyFactor")]
    [InlineData("RopeOriginalMaxPositions")]
    public void LlamaScalingOptionsChangeNativeRotaryAttention(string option)
    {
        var a = Options(); var b = Options();
        typeof(OuteTTSOptions).GetProperty(option)!.SetValue(b, option == "RopeOriginalMaxPositions" ? (object)4096 : 2.0);
        var x = Create(a); var y = Create(b);
        Assert.False(x.LanguageModel.InverseFrequencies().SequenceEqual(y.LanguageModel.InverseFrequencies()));
        int[,] tokens = { { 1, 3, 5, 7 } };
        using var _ = new NoGradScope<double>();
        var before = x.LanguageModel.ForwardWithLogits(x.LanguageModel.Embed(tokens)).Hidden.ToArray();
        var after = y.LanguageModel.ForwardWithLogits(y.LanguageModel.Embed(tokens)).Hidden.ToArray();
        Assert.True(before.Zip(after, (u, v) => Math.Abs(u - v)).Max() > 1e-12);
    }

    [Theory]
    [InlineData("Chunked")]
    [InlineData("ChunkMinWords")]
    [InlineData("ChunkMaxWords")]
    public void SynthesisConsumesChangedNativeChunkPlan(string option)
    {
        var options = Options();
        var model = Create(options);
        string text = OuteOptionFixture.ChunkText(option);
        var before = model.SynthesisChunks(text);
        if (option == "Chunked") options.Chunked = false;
        if (option == "ChunkMinWords") options.ChunkMinWords = 11;
        if (option == "ChunkMaxWords") options.ChunkMaxWords = 31;
        var after = model.SynthesisChunks(text);
        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void SpeakerDurationBoundRejectsOrRunsTheRealWhisperAndCodecPipeline()
    {
        var options = Options(); options.MaxSpeakerSeconds = .05;
        var model = Create(options);
        Assert.Equal(0, OuteOptionFixture.Observe(model, "MaxSpeakerSeconds")[0]);
        options.MaxSpeakerSeconds = .1;
        Assert.Equal(1, OuteOptionFixture.Observe(model, "MaxSpeakerSeconds")[0]);
    }

    [Fact]
    public void SpeakerTemperaturesReachTheRealWhisperTranscriptionLoop()
    {
        var options = Options(); var model = Create(options);
        var greedy = OuteOptionFixture.Observe(model, "AlignerTemperatures");
        Assert.Equal(0, greedy[0]);
        options.AlignerTemperatures = new[] { 1.0 };
        var stochastic = OuteOptionFixture.Observe(model, "AlignerTemperatures");
        Assert.Equal(1, stochastic[0]);
        Assert.NotEqual(greedy[4], stochastic[4]); // The real sampled timestamp changes the segment end, beyond metadata.
        Assert.All(greedy.Concat(stochastic), v => Assert.True(TtsPaperNumerics.IsFinite(v)));
    }

    public sealed class DerivedOptions : OuteTTSOptions { public string Label { get; set; } = "derived"; }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeJsonDimensionsWinHistoricalSharedMetadataInEitherOrder(bool reverse)
    {
        string native = "\"Heads\":2,\"Layers\":3";
        string shared = "\"NumHeads\":8,\"NumDecoderLayers\":6,\"NumEncoderLayers\":6";
        var options = JsonConvert.DeserializeObject<DerivedOptions>("{" + (reverse ? native + "," + shared : shared + "," + native) + "}")!;
        Assert.Equal(2, options.NumHeads); Assert.Equal(3, options.NumDecoderLayers); Assert.Equal(0, options.NumEncoderLayers);
        Assert.Equal("derived", options.Label);
        var aliases = JsonConvert.DeserializeObject<OuteTTSOptions>("{\"NumHeads\":4,\"NumDecoderLayers\":2}")!;
        Assert.Equal(4, aliases.Heads); Assert.Equal(2, aliases.Layers);
        Assert.Equal(0, new OuteTTSOptions().NumEncoderLayers);
        Assert.Equal(32, new OuteTTSOptions().NumHeads); Assert.Equal(16, new OuteTTSOptions().NumDecoderLayers);
        Assert.Equal(16, OuteTTSOptions.Qwen3().NumHeads); Assert.Equal(28, OuteTTSOptions.Qwen3().NumDecoderLayers);
    }

    [Fact]
    public void DecoderOnlyReleaseRejectsAnInventedTextEncoder()
    {
        var options = Options(); options.NumEncoderLayers = 1;
        Assert.Throws<ArgumentException>(() => Create(options));
        var partial = JsonConvert.DeserializeObject<OuteTTSOptions>("{\"NumEncoderLayers\":1}")!;
        Assert.Equal(1, partial.NumEncoderLayers);
    }
}
