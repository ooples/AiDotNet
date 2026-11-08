using System;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MARS5 end to end at test scale: deep and shallow cloning, and one training step of each network (the generated suite
/// trains only the default AR stage).
/// </summary>
public class Mars5ModelTests
{
    private static MARS5TTS<double> Create(MARS5TTSOptions options) => new(new NeuralNetworkArchitecture<double>(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);

    [Theory]
    [InlineData("SampleRate", 16000)]
    [InlineData("HopSize", 320)]
    [InlineData("NumCodebooks", 2)]
    [InlineData("CodebookSize", 16)]
    public void UnpairedCodecGeometryIsRejected(string property, int value)
    {
        var options = Tiny(Mars5Stage.Autoregressive);
        typeof(MARS5TTSOptions).GetProperty(property)!.SetValue(options, value);
        Assert.Throws<ArgumentException>(() => Create(options));
    }

    [Theory]
    [InlineData(24000, 320, 2, 16)]
    [InlineData(16000, 320, 4, 64)]
    [InlineData(24000, 640, 4, 16)]
    public void PairedCodecGeometryEncodesAndVocodesNativeAudio(int sampleRate, int hop, int codebooks, int vocabulary)
    {
        var options = Tiny(Mars5Stage.Autoregressive);
        options.SampleRate = options.Codec.SampleRate = sampleRate;
        options.HopSize = hop;
        options.Codec.Ratios = hop == 320 ? new[] { 8, 5, 4, 2 } : new[] { 8, 5, 4, 4 };
        options.NumCodebooks = options.Codec.NumQuantizers = codebooks;
        options.CodebookSize = options.Codec.CodebookSize = vocabulary;
        options.Codec.TargetBandwidthKbps = (double)sampleRate / hop * codebooks * Math.Log(vocabulary, 2) / 1000;
        options.VocosBandwidthId = codebooks == 2 ? 0 : 1;
        using var model = Create(options);
        var codes = model.Encode(Tone(6 * hop, 220));
        Assert.Equal(6, codes.GetLength(0));
        Assert.Equal(codebooks, codes.GetLength(1));
        var audio = model.Vocode(codes);
        Assert.Equal(6 * hop, audio.Length);
        Assert.All(audio.ToArray(), value => Assert.True(double.IsFinite(value)));
    }

    [Theory]
    [InlineData(-.1)]
    [InlineData(.1)]
    [InlineData(double.NaN)]
    public void ReleasedBlocksRejectUnsupportedActivationDropout(double value)
    {
        var options = Tiny(Mars5Stage.Autoregressive);
        options.DropoutRate = value;
        Assert.Throws<ArgumentOutOfRangeException>(() => Create(options));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void NativeSequenceCapacityIsEnforcedAtSlidingWindow(int window)
    {
        var options = Tiny(Mars5Stage.Autoregressive);
        options.SlidingWindow = options.MaxGenerationLength = window;
        using var model = Create(options);
        var speaker = model.Autoregressive.Speaker(new int[2, options.NumCodebooks]);
        Assert.Equal(window, model.Autoregressive.Logits(new int[window], speaker).Shape[0]);
        Assert.Throws<ArgumentException>(() => model.Autoregressive.Logits(new int[window + 1], speaker));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2048)]
    public void NativeByteTokenizerOwnsVocabularyInsteadOfInheritedMetadata(int metadata)
    {
        var options = Tiny(Mars5Stage.Autoregressive);
        options.VocabSize = metadata;
        using var model = Create(options);
        Assert.Equal(258, model.TextTokenizer.VocabSize);
        Assert.Equal(new[] { 104, 105 }, model.TextTokenizer.Encode("hi"));
    }

    [Fact]
    public void NativeTrimmingContextExercisesZeroDbBoundaryAndRestoresVoice()
    {
        var options = Tiny(Mars5Stage.Autoregressive);
        Mars5OptionFixture.PrepareOptions(options, "TrimDb");
        using var model = new MARS5TTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        var voice = model.Voice;
        var control = Mars5OptionFixture.Observe(model, "TrimDb");
        options.TrimDb = 0;
        Assert.NotNull(control);
        Assert.NotEqual(control, Mars5OptionFixture.Observe(model, "TrimDb"));
        Assert.Same(voice, model.Voice);
    }

    [Theory]
    [InlineData("ArBeta1", -.1)]
    [InlineData("ArBeta1", 1)]
    [InlineData("ArBeta1", double.NaN)]
    [InlineData("ArBeta1", double.PositiveInfinity)]
    [InlineData("ArBeta2", -.1)]
    [InlineData("ArBeta2", 1)]
    [InlineData("ArBeta2", double.NaN)]
    [InlineData("ArBeta2", double.PositiveInfinity)]
    [InlineData("NarBeta1", -.1)]
    [InlineData("NarBeta1", 1)]
    [InlineData("NarBeta1", double.NaN)]
    [InlineData("NarBeta1", double.PositiveInfinity)]
    [InlineData("NarBeta2", -.1)]
    [InlineData("NarBeta2", 1)]
    [InlineData("NarBeta2", double.NaN)]
    [InlineData("NarBeta2", double.PositiveInfinity)]
    public void InvalidStageAdamBetasAreRejectedBeforeTraining(string option, double value)
    {
        var options = Tiny(Mars5Stage.Autoregressive);
        typeof(MARS5TTSOptions).GetProperty(option)!.SetValue(options, value);
        Assert.Throws<ArgumentOutOfRangeException>(() => new MARS5TTS<double>(new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options));
    }

    private static MARS5TTSOptions Tiny(Mars5Stage stage) => new()
    {
        CurrentStage = stage, HiddenDim = 16, NumHeads = 2, ArLayers = 1, ArFeedForward = 32, ArSpeakerLayers = 1, SlidingWindow = 512,
        NarDim = 16, NarHeads = 2, NarEncoderLayers = 1, NarDecoderLayers = 1, NarSpeakerLayers = 1, TimeDim = 8, DiffusionSteps = 4,
        Q0OverrideSteps = 1, MaxGenerationLength = 64, NumCodebooks = 4, CodebookSize = 64, HopSize = 2560, DecoderDim = 16,
        DecoderIntermediateDim = 32, DecoderLayers = 1, WarmupSteps = 0,
        Codec = new EnCodecOptions
        {
            SampleRate = 24000, NumQuantizers = 4, CodebookSize = 64, Filters = 4, Ratios = new[] { 8, 5, 4, 4, 4 }, Dimension = 8,
            ResidualKernelSizes = new[] { 3, 1 }, TargetBandwidthKbps = 0.225,
        },
    };

    private static Tensor<double> Tone(int samples, double frequency)
    {
        var x = new double[samples];
        for (int i = 0; i < samples; i++) x[i] = 0.3 * Math.Sin(2 * Math.PI * frequency * i / 24000.0);
        return new Tensor<double>(new[] { samples }, new Vector<double>(x));
    }

    private static MARS5TTS<double> Model(Mars5Stage stage) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Tiny(stage));

    [Fact(Timeout = 900000)]
    public async Task DeepAndShallowClones_AndBothStagesTrain()
    {
        await Task.Yield();
        var model = Model(Mars5Stage.Autoregressive);
        model.Voice = model.CreateVoice(Tone(51200, 180), 24000, "hello there");
        var deep = model.Synthesize("good morning");
        model.Voice = model.CreateVoice(Tone(51200, 180), 24000);
        var shallow = model.Synthesize("good morning");
        foreach (var audio in new[] { deep, shallow })
            for (int i = 0; i < audio.Length; i++) Assert.False(double.IsNaN(audio.GetFlat(i)) || double.IsInfinity(audio.GetFlat(i)));

        var sample = new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Text = "a short test", Audio = Tone(25600, 220) };
        Assert.False(double.IsNaN(model.Train(sample)));
        var nar = Model(Mars5Stage.NonAutoregressive);
        double before = nar.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 3; i++) Assert.False(double.IsNaN(nar.Train(sample)));
        Assert.False(double.IsNaN(before));
    }
}
