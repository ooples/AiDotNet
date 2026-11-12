using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class CosyVoice3OptionContextTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void PairedCausalHiftRatesAndMelHopControlDecodedLength(int rate)
    {
        var options = CosyVoice3Fixture.Options();
        options.HiftUpsampleRates = new[] { rate };
        options.HiftUpsampleKernels = new[] { 2 * rate };
        options.HopSize = 4 * rate;
        using var model = CosyVoice3Fixture.Create(options);
        var wave = model.CausalVocoder!.Forward(new Tensor<double>(new[] { 1, options.MelChannels, 4 }), new Random(7));
        Assert.Equal(4 * options.HopSize, wave.Length);
    }

    [Fact]
    public void UnpairedCausalHiftMelHopIsRejected()
    {
        var options = CosyVoice3Fixture.Options(); options.HopSize++;
        Assert.Throws<ArgumentException>(() => CosyVoice3Fixture.Create(options));
    }

    [Fact]
    public void NativeNucleusSamplerReadsConfiguredTopP()
    {
        var options = CosyVoice3Fixture.Options();
        options.TopP = 0.8;
        using var control = CosyVoice3Fixture.Create(options);
        using var changed = CosyVoice3Fixture.Create(new CosyVoice3Options(options) { TopP = 0.4 });
        long Observe(CosyVoice3<double> model) => CosyVoice3Fixture.ObserveTopP(model);
        Assert.NotEqual(Observe(control), Observe(changed));
    }

    [Fact]
    public void ReleasedOnnxFrontendEnforcesReferenceDurationBeforeExecutingTheGraph()
    {
        string path = Path.Combine(Path.GetTempPath(), "cosy3-frontend-contract-" + Guid.NewGuid().ToString("N") + ".onnx");
        try
        {
            ChatterboxFixture.WriteFrontendContractGraph(path);
            var options = CosyVoice3Fixture.Options();
            options.SpeechTokenizerPath = path;
            options.MaxReferenceSeconds = 1;
            using var limited = CosyVoice3Fixture.Create(options);
            using var extended = CosyVoice3Fixture.Create(new CosyVoice3Options(options) { MaxReferenceSeconds = 2 });
            var recording = ChatterboxFixture.Recording<double>(2);
            Assert.Throws<ArgumentException>(() => limited.SpeechTokens(recording, 24000));
            var tokens = extended.SpeechTokens(recording, 24000);
            Assert.Single(tokens);
            Assert.True(tokens[0] > 0);
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    public void VocoderSegmentControlsCausalHiftTargetsAndDecodedLength(int length)
    {
        var options = CosyVoice3Fixture.Options(); options.VocoderSegmentLength = length;
        using var model = CosyVoice3Fixture.Create(options);
        var targets = model.VocoderTargets(new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 1 }), Audio = ChatterboxFixture.Recording<double>(),
        }, new Random(7));
        Assert.Equal(length, targets.Audio.Length);
        Assert.Equal(targets.Mel.Shape[2], targets.Pitch.Length);
        Assert.Equal(targets.Mel.Shape[2] * options.HopSize, model.CausalVocoder!.Forward(targets.Mel, new Random(7)).Length);
    }

    [Theory]
    [InlineData("DropoutRate")]
    [InlineData("DiTTrainingCfgRate")]
    [InlineData("DiTSigmaMin")]
    public void NativeFlowOptionsChangeLossAndWeightUpdates(string option)
    {
        var options = CosyVoice3Fixture.Options();
        options.DiTTrainingCfgRate = 0;
        var alternative = new CosyVoice3Options(options);
        switch (option)
        {
            case "DropoutRate": alternative.DropoutRate = 0.5; break;
            case "DiTTrainingCfgRate": alternative.DiTTrainingCfgRate = 1; break;
            case "DiTSigmaMin": alternative.DiTSigmaMin = 0.25; break;
            default: throw new ArgumentOutOfRangeException(nameof(option));
        }
        using var control = CosyVoice3Fixture.Create(options);
        using var changed = CosyVoice3Fixture.Create(alternative);
        control.CurrentStage = changed.CurrentStage = CosyVoiceTrainingStage.Flow;
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        var mel = new Tensor<double>(new[] { 6, options.MelChannels });
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.2 * Math.Sin(i * 0.17);
        var speaker = new Tensor<double>(new[] { options.SpeakerEmbeddingDim }); speaker[0] = 1;
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 })),
            CodecTokens = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 })),
            Mel = mel, SpeakerReference = speaker,
        };
        var before = control.FlowLayers.SelectMany(layer => layer.GetParameters()).ToArray();
        for (int step = 0; step < 2; step++)
        {
            double first = control.Train(sample), second = changed.Train(sample);
            Assert.True(TtsPaperNumerics.IsFinite(first));
            Assert.True(TtsPaperNumerics.IsFinite(second));
            Assert.NotEqual(first, second);
        }
        Assert.False(before.SequenceEqual(control.FlowLayers.SelectMany(layer => layer.GetParameters())));
        Assert.False(control.FlowLayers.SelectMany(layer => layer.GetParameters()).SequenceEqual(
            changed.FlowLayers.SelectMany(layer => layer.GetParameters())));
    }

    [Fact]
    public void DitOptionsExcludeUnusedConformerAndNativeS3Controls()
    {
        foreach (string option in new[] { "FlowDim", "FlowHeads", "FlowFeedForwardDim", "FlowEncoderLayers",
            "FlowAttentionDropoutRate", "DecoderChannels", "DecoderHeads", "DecoderHeadDim", "DecoderBlocks",
            "DecoderMidBlocks", "FlowTimesteps", "TrainingCfgRate", "InferenceCfgRate", "FlowLoss",
            "SpeechTokenizerState", "SpeechTokenizerHeads", "SpeechTokenizerLayers" })
        {
            Assert.Null(typeof(CosyVoice3Options).GetProperty(option));
            Assert.NotNull(typeof(CosyVoiceOptions).GetProperty(option));
            Assert.NotNull(typeof(CosyVoice2Options).GetProperty(option));
            Assert.NotNull(typeof(AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions).GetProperty(option));
        }
    }

    [Theory]
    [InlineData(16, 2, 8)]
    [InlineData(32, 2, 16)]
    [InlineData(32, 4, 8)]
    public void PairedDitGeometryRunsNativeTrainingAndInference(int width, int heads, int headWidth)
    {
        var options = new CosyVoice3Options
        {
            CodebookSize = 8, MelChannels = 4, SpeakerEmbeddingDim = 4,
            DiTInputDim = 4, PreLookaheadChannels = 8, DiTDim = width, DiTDepth = 1,
            DiTHeads = heads, DiTHeadDim = headWidth, DiTFeedForwardMultiplier = 2,
            DiTTimesteps = 2, StaticChunkSize = 2, DropoutRate = 0,
        };
        var layers = new List<LayerBase<double>>();
        var flow = new CosyVoice3Flow<double>(AiDotNetEngine.Current, layers, options, new Random(3));
        var speaker = new Tensor<double>(new[] { 4 }); speaker[0] = 1;
        var mel = new Tensor<double>(new[] { 1, 4, 6 });
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.2 * Math.Sin(i);
        var loss = flow.TrainingLoss(new[] { 1, 2, 3 }, mel, speaker, training: true, new Random(4));
        Assert.True(TtsPaperNumerics.IsFinite(loss[0]));
        var generated = flow.Inference(new[] { 1, 2, 3 }, Array.Empty<int>(), null, speaker,
            new Tensor<double>(new[] { 1, 4, 6 }));
        Assert.Equal(new[] { 1, 4, 6 }, generated.Shape.ToArray());
        Assert.All(generated, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData("DiTDim")]
    [InlineData("DiTHeads")]
    [InlineData("DiTHeadDim")]
    [InlineData("TokenMelRatio")]
    public void UnpairedReleasedFlowGeometryIsRejected(string option)
    {
        var options = new CosyVoice3Options();
        switch (option)
        {
            case "DiTDim": options.DiTDim++; break;
            case "DiTHeads": options.DiTHeads++; break;
            case "DiTHeadDim": options.DiTHeadDim++; break;
            case "TokenMelRatio": options.TokenMelRatio = 1; break;
            default: throw new ArgumentOutOfRangeException(nameof(option));
        }
        Assert.Throws<ArgumentException>(() => new CosyVoice3<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 1), options));
    }
}
