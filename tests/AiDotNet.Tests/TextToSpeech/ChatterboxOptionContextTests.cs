using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class ChatterboxOptionContextTests
{
    [Fact]
    public void OptionalOnnxFrontendEnforcesItsConfiguredReferenceLimit()
    {
        string path = Path.Combine(Path.GetTempPath(), "chatter-frontend-contract-" + Guid.NewGuid().ToString("N") + ".onnx");
        try
        {
            // A contract graph returns the feature length to prove that accepted input reaches ONNX Runtime.
            // This graph tests frontend plumbing and duration validation, not released tokenizer fidelity.
            WriteFrontendContractGraph(path);
            var options = ChatterboxFixture.Options();
            options.SpeechTokenizerPath = path;
            options.MaxReferenceSeconds = 1;
            using var limited = Create(options);
            using var extended = Create(new ChatterboxOptions(options) { MaxReferenceSeconds = 2 });
            var recording = ChatterboxFixture.Recording<double>(2);
            var error = Assert.Throws<ArgumentException>(() => limited.EncodeToTokens(recording));
            Assert.Contains("at most 1 seconds", error.Message);
            var tokens = extended.EncodeToTokens(recording);
            Assert.Single(tokens);
            Assert.True(tokens[0] > 0);
        }
        finally { File.Delete(path); }
    }

    internal static void WriteFrontendContractGraph(string path)
    {
        var builder = new AiDotNet.Onnx.OnnxGraphBuilder(new AiDotNet.Onnx.OnnxExportOptions());
        builder.AddFloatInput("features", new[] { 1, 128, -1 });
        builder.AddFloatInput("length", new[] { 1 });
        var node = new AiDotNet.Onnx.Protobuf.NodeProto { OpType = "Identity" };
        node.Input.Add("length"); node.Output.Add("tokens"); builder.AddNode(node);
        builder.AddFloatOutput("tokens", new[] { 1 });
        using var stream = File.Create(path);
        builder.WriteTo(stream);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void PairedHiftRatesAndHopControlNativeWaveformLength(int rate)
    {
        var options = ChatterboxFixture.Options();
        options.HiftUpsampleRates = new[] { rate };
        options.HiftUpsampleKernels = new[] { 2 * rate };
        options.HopSize = 4 * rate;
        using var model = Create(options);
        var mel = new Tensor<double>(new[] { 1, options.MelChannels, 4 });
        Assert.Equal(4 * options.HopSize, model.Vocoder!.Forward(mel, new Random(7)).Length);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    public void VocoderSegmentControlsNativeTrainingTargetsAndDecodedLength(int segment)
    {
        var options = ChatterboxFixture.Options();
        options.VocoderSegmentLength = segment;
        using var model = Create(options);
        var targets = model.VocoderTargets(new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 1 }), Audio = ChatterboxFixture.Recording<double>(),
        }, new Random(7));
        Assert.Equal(segment, targets.Audio.Length);
        Assert.Equal(targets.Mel.Shape[2], targets.Pitch.Length);
        Assert.Equal(targets.Mel.Shape[2] * options.HopSize, model.Vocoder!.Forward(targets.Mel, new Random(7)).Length);
    }

    [Theory]
    [InlineData(64, 1)]
    [InlineData(128, 2)]
    public void PairedS3WidthsEncodeTheNativeReference(int state, int heads)
    {
        var options = ChatterboxFixture.Options();
        options.SpeechTokenizerState = state;
        options.SpeechTokenizerHeads = heads;
        using var model = Create(options);
        var tokens = model.EncodeToTokens(ChatterboxFixture.Recording<double>());
        Assert.True(tokens.Length > 0);
        foreach (double token in tokens) Assert.InRange(token, 0, 6560);
    }

    [Theory]
    [InlineData("CodebookSize")]
    [InlineData("SpeechTokenizerState")]
    [InlineData("SpeechTokenizerHeads")]
    public void UnpairedS3GeometryIsRejected(string option)
    {
        var options = ChatterboxFixture.Options();
        switch (option)
        {
            case "CodebookSize": options.CodebookSize = 6562; break;
            case "SpeechTokenizerState": options.SpeechTokenizerState = 128; break;
            case "SpeechTokenizerHeads": options.SpeechTokenizerHeads = 2; break;
            default: throw new ArgumentOutOfRangeException(nameof(option));
        }
        Assert.Throws<ArgumentException>(() => Create(options));
    }

    private static Chatterbox<double> Create(ChatterboxOptions options) => new(new NeuralNetworkArchitecture<double>(
        InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    private static TtsTrainingSample<double> Sample(Chatterbox<double> model)
    {
        var codes = new Tensor<double>(new[] { 8, 1 });
        for (int i = 0; i < codes.Length; i++) codes[i] = i;
        var mel = new Tensor<double>(new[] { 16, model.MelChannels });
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.3 * Math.Sin(i * 0.17);
        return new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 3.0, 4.0, 5.0 })),
            CodecTokens = codes, Mel = mel, SpeakerReference = ChatterboxFixture.Recording<double>(),
        };
    }

    private static IEnumerable<LayerBase<double>> VoiceLayers(Chatterbox<double> model) =>
        (IEnumerable<LayerBase<double>>)model.GetType().GetField("_voiceEncoderLayers",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;

    [Fact]
    public void NativeFlowStageTrainsWithConfiguredAdamWAndPreservesOtherComponents()
    {
        using var model = Create(ChatterboxFixture.Options());
        model.CurrentStage = CosyVoiceTrainingStage.Flow;
        var sample = Sample(model);
        // Resolve the frozen CAM++ frontend's lazy projection before comparing its parameter storage.
        Assert.True(TtsPaperNumerics.IsFinite(model.EvaluateTrainingObjective(sample)));
        var frozen = model.LanguageModelLayers.Concat(model.FrozenLayers).Concat(VoiceLayers(model)).ToArray();
        var before = frozen.Select(layer => layer.GetParameters().ToArray()).ToArray();
        var flowBefore = model.FlowLayers.SelectMany(layer => layer.GetParameters()).ToArray();
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Assert.False(flowBefore.SequenceEqual(model.FlowLayers.SelectMany(layer => layer.GetParameters())));
        for (int i = 0; i < frozen.Length; i++) Assert.Equal(before[i], frozen[i].GetParameters().ToArray());
    }

    [Fact]
    public void NativeT3WarmupAdvancesPerTrainingBatch()
    {
        var options = ChatterboxFixture.Options(); options.LearningRate = 0.001; options.WeightDecay = 0;
        using var control = Create(options);
        using var changed = Create(new ChatterboxOptions(options) { WarmupSteps = 4 });
        control.Voice = control.CreateVoice(ChatterboxFixture.Recording<double>(), 24000);
        changed.Voice = changed.CreateVoice(ChatterboxFixture.Recording<double>(), 24000);
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        for (int step = 0; step < 2; step++)
        {
            var before = control.LanguageModelLayers.SelectMany(layer => layer.GetParameters()).ToArray();
            Assert.True(TtsPaperNumerics.IsFinite(control.Train(Sample(control))));
            Assert.True(TtsPaperNumerics.IsFinite(changed.Train(Sample(changed))));
            Assert.False(before.SequenceEqual(control.LanguageModelLayers.SelectMany(layer => layer.GetParameters())), "The native T3 step must update its language-model parameters.");
        }
        Assert.False(control.LanguageModelLayers.SelectMany(layer => layer.GetParameters()).SequenceEqual(
            changed.LanguageModelLayers.SelectMany(layer => layer.GetParameters())));
    }

    [Theory]
    [InlineData("TrainingCfgRate")]
    [InlineData("FlowLoss")]
    [InlineData("FlowAttentionDropoutRate")]
    [InlineData("DropoutRate")]
    public void FlowTrainingOptionsChangeTheNativeUpdate(string option)
    {
        var options = ChatterboxFixture.Options();
        options.TrainingCfgRate = 0;
        var alternative = new ChatterboxOptions(options);
        switch (option)
        {
            case "TrainingCfgRate": alternative.TrainingCfgRate = 1; break;
            case "FlowLoss": alternative.FlowLoss = CosyVoiceFlowLoss.SquaredError; break;
            case "FlowAttentionDropoutRate": alternative.FlowAttentionDropoutRate = 0.5; break;
            case "DropoutRate": alternative.DropoutRate = 0.5; break;
            default: throw new ArgumentOutOfRangeException(nameof(option));
        }
        using var control = Create(options);
        using var changed = Create(alternative);
        control.CurrentStage = changed.CurrentStage = CosyVoiceTrainingStage.Flow;
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        double first = control.Train(Sample(control));
        double second = changed.Train(Sample(changed));
        Assert.True(TtsPaperNumerics.IsFinite(first));
        Assert.True(TtsPaperNumerics.IsFinite(second));
        Assert.NotEqual(first, second);
        Assert.False(control.FlowLayers.SelectMany(layer => layer.GetParameters()).SequenceEqual(
            changed.FlowLayers.SelectMany(layer => layer.GetParameters())));
    }

    [Fact]
    public void SharedSeedControlsNativeAndFrozenVoiceInitializationIndependentlyOfSampling()
    {
        var options = ChatterboxFixture.Options(); options.Seed = 11;
        using var first = Create(options);
        using var repeat = Create(new ChatterboxOptions(options) { SamplingSeed = 31 });
        using var changed = Create(new ChatterboxOptions(options) { Seed = 12 });
        Assert.Equal(first.GetParameters().ToArray(), repeat.GetParameters().ToArray());
        Assert.False(first.GetParameters().SequenceEqual(changed.GetParameters()));
        var voice = VoiceLayers(first).SelectMany(layer => layer.GetParameters()).ToArray();
        Assert.Equal(voice, VoiceLayers(repeat).SelectMany(layer => layer.GetParameters()).ToArray());
        Assert.False(voice.SequenceEqual(VoiceLayers(changed).SelectMany(layer => layer.GetParameters())));
    }

    [Fact]
    public void PartialRateChangesTheNativeVoiceEmbeddingWhenWindowsOverlap()
    {
        var options = ChatterboxFixture.Options();
        options.VoiceEncoderPartialFrames = 160;
        using var control = Create(options);
        using var changed = Create(new ChatterboxOptions(options) { VoiceEncoderPartialRate = 2 });
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        var recording = ChatterboxFixture.Recording<double>(2);
        var first = control.VoiceEncoderModel!.Embed(recording, 24000);
        var second = changed.VoiceEncoderModel!.Embed(recording, 24000);
        Assert.False(first.SequenceEqual(second));
    }

    [Fact]
    public void StaticChunksChangeStreamingFlowEncoding()
    {
        var options = ChatterboxFixture.Options();
        options.StaticChunkSize = 2;
        using var control = Create(options);
        using var changed = Create(new ChatterboxOptions(options) { StaticChunkSize = 3 });
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        var codes = Enumerable.Range(0, 12).ToArray();
        Assert.Equal(control.FlowModel!.Encode(codes, false, new Random(7)).ToArray(),
            changed.FlowModel!.Encode(codes, false, new Random(7)).ToArray());
        Assert.False(control.FlowModel.Encode(codes, false, new Random(7), streaming: true).SequenceEqual(
            changed.FlowModel.Encode(codes, false, new Random(7), streaming: true)));
    }
}
