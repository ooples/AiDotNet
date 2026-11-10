using System;
using System.Linq;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class SPEARTTSOptionBindingTests
{
    public sealed class DerivedOptions : SPEARTTSOptions { }

    private static SPEARTTS<double> Create(SPEARTTSOptions options, int? seed = 7) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = seed }, options);

    [Theory]
    [InlineData("SampleRate", 32000)]
    [InlineData("HopSize", 800)]
    [InlineData("CodebookSize", 32)]
    [InlineData("NumCodebooks", 3)]
    public void CodecAliases_ConfigureTheNativeCodec_AndSurviveCopies(string name, int value)
    {
        var options = SPEARTTSPaperTests.Tiny();
        int selected = options.NumCodebooks;
        typeof(SPEARTTSOptions).GetProperty(name)!.SetValue(options, value);
        Assert.Equal(value, typeof(SPEARTTSOptions).GetProperty(name)!.GetValue(options));
        Assert.Equal(options.SampleRate, options.SoundStream.SampleRate);
        Assert.Equal(options.HopSize, options.SoundStream.Ratios.Aggregate(1, (a, b) => a * b));
        Assert.Equal(options.CodebookSize, options.SoundStream.CodebookSize);
        Assert.Equal(name == "NumCodebooks" ? value : selected, options.NumCodebooks);
        using var model = Create(options);
        Assert.Equal(options.NumCodebooks, model.NumCodebooks);
        var ids = new Tensor<double>(new[] { 2, options.NumCodebooks });
        var audio = model.DecodeFromTokens(ids);
        Assert.Equal(2 * options.HopSize, audio.Length);
        Assert.All(audio.ToArray(), x => Assert.True(double.IsFinite(x)));
        var copy = new SPEARTTSOptions(options);
        var restored = JsonConvert.DeserializeObject<SPEARTTSOptions>(JsonConvert.SerializeObject(options))!;
        foreach (var result in new[] { copy, restored })
        {
            Assert.Equal(options.SampleRate, result.SampleRate);
            Assert.Equal(options.HopSize, result.HopSize);
            Assert.Equal(options.NumCodebooks, result.NumCodebooks);
            Assert.Equal(options.CodebookSize, result.CodebookSize);
            Assert.NotSame(options.SoundStream, result.SoundStream);
            Assert.NotSame(options.SoundStream.Ratios, result.SoundStream.Ratios);
        }
    }

    [Fact]
    public void BaseTypedAliases_AndNativeChanges_AreVisibleInBothDirections()
    {
        var options = SPEARTTSPaperTests.Tiny();
        TtsModelOptions shared = options;
        shared.SampleRate = 32000;
        shared.HopSize = 800;
        Assert.Equal(32000, options.SoundStream.SampleRate);
        Assert.Equal(800, options.HopSize);
        Assert.Equal(2, options.NumCodebooks);
        options.SoundStream.SampleRate = 16000;
        options.SoundStream.Ratios = new[] { 8, 5, 4, 2 };
        options.SoundStream.CodebookSize = 32;
        options.SoundStream.NumQuantizers = 6;
        options.SoundStream.TargetBandwidthKbps = .75;
        Assert.Equal((16000, 320, 32, 3), (shared.SampleRate, shared.HopSize, options.CodebookSize, options.NumCodebooks));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CanonicalCodecJson_WinsOverLegacyAliases_InEitherOrder_AndDerivedTypes(bool nativeFirst)
    {
        const string native = "\"SoundStream\":{\"SampleRate\":16000,\"Ratios\":[8,5,5,4,2],\"NumQuantizers\":2,\"CodebookSize\":16,\"TargetBandwidthKbps\":0.08}";
        const string aliases = "\"SampleRate\":22050,\"HopSize\":256,\"NumCodebooks\":3,\"CodebookSize\":1024";
        string json = "{" + (nativeFirst ? native + "," + aliases : aliases + "," + native) + "}";
        foreach (var type in new[] { typeof(SPEARTTSOptions), typeof(DerivedOptions) })
        {
            var options = (SPEARTTSOptions)JsonConvert.DeserializeObject(json, type)!;
            Assert.Equal((16000, 1600, 2, 16), (options.SampleRate, options.HopSize, options.NumCodebooks, options.CodebookSize));
            Assert.Equal(.08, options.SoundStream.TargetBandwidthKbps);
            Assert.Equal(2, options.SoundStream.NumQuantizers);
        }
    }

    [Fact]
    public void PartialCodecJson_AppliesRemainingAliases_AfterTheNativeFields()
    {
        var options = JsonConvert.DeserializeObject<SPEARTTSOptions>("{\"HopSize\":1600,\"NumCodebooks\":2,\"SoundStream\":{\"SampleRate\":16000,\"CodebookSize\":16,\"NumQuantizers\":2}}")!;
        Assert.Equal((16000, 1600, 2, 16), (options.SampleRate, options.HopSize, options.NumCodebooks, options.CodebookSize));
        Assert.Equal(.08, options.SoundStream.TargetBandwidthKbps, 12);
        Assert.Throws<JsonSerializationException>(() => JsonConvert.DeserializeObject<SPEARTTSOptions>(
            "{\"NumCodebooks\":3,\"SoundStream\":{\"NumQuantizers\":2}}"));
    }

    [Theory]
    [InlineData(16001, 1600)]
    [InlineData(16000, 1000)]
    public void FractionalOrUnalignedClocks_AreRejectedBeforeAllocatingNetworks(int rate, int hop)
    {
        var options = SPEARTTSPaperTests.Tiny();
        options.SampleRate = rate;
        options.HopSize = hop;
        Assert.Throws<ArgumentException>(() => Create(options));
    }

    [Fact]
    public void ModelSeed_DrivesNativeWeights_WhenArchitectureDoesNotOverrideIt()
    {
        var a = SPEARTTSPaperTests.Tiny(); a.Seed = 17;
        var b = new SPEARTTSOptions(a) { Seed = 23 };
        using var first = Create(a, null);
        using var repeated = Create(new SPEARTTSOptions(a), null);
        using var different = Create(b, null);
        Assert.Equal(first.GetParameters().ToArray(), repeated.GetParameters().ToArray());
        Assert.NotEqual(first.ReadingModel!.Shared.GetParameters().ToArray(), different.ReadingModel!.Shared.GetParameters().ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExplicitSeeds_PreserveArchitectureAndChildCodecPrecedence(bool architectureOverride)
    {
        var a = SPEARTTSPaperTests.Tiny(); a.Seed = 17; a.SoundStream.Seed = 29;
        var b = new SPEARTTSOptions(a) { Seed = 23 };
        using var first = Create(a, architectureOverride ? 7 : null);
        using var second = Create(b, architectureOverride ? 7 : null);
        Assert.Equal(first.Codec.GetParameters().ToArray(), second.Codec.GetParameters().ToArray());
        bool sameT5 = first.ReadingModel!.Shared.GetParameters().ToArray()
            .SequenceEqual(second.ReadingModel!.Shared.GetParameters().ToArray());
        Assert.Equal(architectureOverride, sameT5);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void BacktranslationDepth_UpdatesBothEnds_AndFreezesTheRemainingMiddle(int depth)
    {
        var options = SPEARTTSPaperTests.Tiny();
        options.NumDecoderLayers = 4;
        options.BacktranslationTrainableDecoderLayers = depth;
        using var model = Create(options);
        model.CurrentStage = SPEARTTSTrainingStage.Backtranslation;
        var blocks = model.BacktranslationNetwork!.Decoder;
        var before = blocks.Select(b => b.SelfAttention.Query.GetParameters().ToArray()).ToArray();
        model.Train(SpearOptionFixture.Sample(model));
        for (int i = 0; i < blocks.Count; i++)
            Assert.Equal(i < depth || i >= blocks.Count - depth,
                !before[i].SequenceEqual(blocks[i].SelfAttention.Query.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData("PretrainingDeletionProbability")]
    [InlineData("PretrainingDropoutRate")]
    [InlineData("SpeakingDropoutRate")]
    [InlineData("PromptSeconds")]
    [InlineData("Temperature")]
    [InlineData("SamplingSeed")]
    [InlineData("MaxSemanticTokens")]
    public void StageAndSamplingOptions_ChangeTheActualNativeTask(string name)
    {
        var a = SPEARTTSPaperTests.Tiny();
        SpearOptionFixture.Prepare(a, name);
        var b = new SPEARTTSOptions(a);
        object value = name switch
        {
            "PretrainingDeletionProbability" => 0.0,
            "PretrainingDropoutRate" or "SpeakingDropoutRate" => .5,
            "PromptSeconds" => .8,
            "Temperature" => .1,
            "SamplingSeed" => (object)23,
            "MaxSemanticTokens" => (object)2,
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
        typeof(SPEARTTSOptions).GetProperty(name)!.SetValue(b, value);
        using var control = Create(a);
        using var candidate = Create(b);
        var baseline = SpearOptionFixture.Observe(control, name);
        var changed = SpearOptionFixture.Observe(candidate, name);
        Assert.All(baseline.Concat(changed), x => Assert.True(double.IsFinite(x)));
        Assert.False(baseline.SequenceEqual(changed), name + " must affect its native task.");
    }
}
