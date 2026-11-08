using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Paper-level invariants and an executable miniature of DiTTo-TTS (Lee et al., 2024).</summary>
public sealed class DiTToTTSPaperTests
{
    [Theory]
    [InlineData("HiddenDim", 16)]
    [InlineData("EncoderDim", 16)]
    [InlineData("DecoderDim", 16)]
    [InlineData("NumDecoderLayers", 2)]
    public void SharedWidthAndDepthOptionsReachTheNativeTransformer(string option, int value)
    {
        using var control = Tiny();
        var options = TinyOptions();
        typeof(DiTToTTSOptions).GetProperty(option)!.SetValue(options, value);
        using var changed = new DiTToTTS<double>(new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        Assert.Equal(value, typeof(DiTToTTSOptions).GetProperty(option)!.GetValue(changed.GetOptions()));
        Assert.NotEqual(control.GetParameters().Length, changed.GetParameters().Length);
    }

    [Fact]
    public void MelVae_LaterInputsCannotChangeEarlierEncodedOrDecodedFrames()
    {
        var vae = new DiTToMelVae<double>(AiDotNetEngine.Current,
            new List<AiDotNet.NeuralNetworks.Layers.LayerBase<double>>(), TinyOptions(), new Random(7));
        var mel = new Tensor<double>(new[] { 16, 4 });
        for (int i = 0; i < mel.Length; i++) mel[i] = Math.Sin(i * 0.17);
        var changedMel = new Tensor<double>(mel.Shape.ToArray(), new Vector<double>(mel.ToArray()));
        for (int i = 8 * 4; i < changedMel.Length; i++) changedMel[i] += 10;
        var encoded = vae.Encode(mel);
        var encodedChanged = vae.Encode(changedMel);
        for (int i = 0; i < 4 * 8; i++) Assert.Equal(encoded[i], encodedChanged[i], 12);
        Assert.True(encoded.ToArray().Zip(encodedChanged.ToArray(), (a, b) => Math.Abs(a - b)).Max() > 1e-8);

        var changedLatents = new Tensor<double>(encoded.Shape.ToArray(), new Vector<double>(encoded.ToArray()));
        for (int i = 4 * 8; i < changedLatents.Length; i++) changedLatents[i] += 10;
        var decoded = vae.Decode(encoded, quantize: false);
        var decodedChanged = vae.Decode(changedLatents, quantize: false);
        Assert.Equal(16, decoded.Shape[0]);
        for (int i = 0; i < 8 * 4; i++) Assert.Equal(decoded[i], decodedChanged[i], 12);
        Assert.True(decoded.ToArray().Zip(decodedChanged.ToArray(), (a, b) => Math.Abs(a - b)).Max() > 1e-8);
    }

    internal static DiTToTTSOptions TinyOptions() => new()
    {
        HopSize = 4, FftSize = 16, MelChannels = 4,
        TextHiddenDim = 8, TextFeedForwardDim = 16, TextHeads = 2, TextEncoderLayers = 1, TextVocabulary = 81,
        MaxTextLength = 32, LatentDim = 8, MelVaeChannels = 8, MelVaeMultipliers = new[] { 1, 1 },
        MelVaeDownsampleFactor = 2, RvqDepth = 2, RvqVocabulary = 4,
        DiTHiddenDim = 8, DiTLayers = 1, DiTHeads = 2, DiTFeedForwardMultiplier = 2,
        LengthHiddenDim = 8, LengthLayers = 1, LengthHeads = 2, LengthFeedForwardDim = 16,
        MaxLatentLength = 16, LengthTopK = 4, TrainingDiffusionSteps = 10, InferenceSteps = 2,
        BigVganInitialChannels = 8, BigVganUpsampleRates = new[] { 2 }, BigVganUpsampleKernels = new[] { 4 },
        BigVganResidualKernels = new[] { 3 }, BigVganResidualDilations = new[] { 1 }, DropoutRate = 0
    };

    private static DiTToTTS<double> Tiny() => new(new NeuralNetworkArchitecture<double>(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), TinyOptions());

    [Fact]
    public void Defaults_AreTheEnglishBasePaperConfiguration()
    {
        var o = new DiTToTTSOptions();
        Assert.Equal((22050, 80, 256, 1024), (o.SampleRate, o.MelChannels, o.HopSize, o.FftSize));
        Assert.Equal((768, 12, 12), (o.DiTHiddenDim, o.DiTLayers, o.DiTHeads));
        Assert.Equal((512, 8, 32, 1024), (o.LatentDim, o.MelVaeDownsampleFactor, o.RvqDepth, o.RvqVocabulary));
        Assert.Equal(new[] { 1, 1, 2, 2 }, o.MelVaeMultipliers);
        Assert.Equal((1000, 25, 0.3, 5.0), (o.TrainingDiffusionSteps, o.InferenceSteps, o.NoiseScheduleScale, o.GuidanceScale));
        Assert.Equal((2048, 20), (o.MaxLatentLength, o.LengthTopK));
        Assert.Equal(new[] { 4, 4, 2, 2, 2, 2 }, o.BigVganUpsampleRates);
    }

    [Theory]
    [InlineData(DiTToModelSize.Small, 12, 384, 6)]
    [InlineData(DiTToModelSize.Base, 12, 768, 12)]
    [InlineData(DiTToModelSize.Large, 24, 1024, 16)]
    [InlineData(DiTToModelSize.ExtraLarge, 28, 1152, 16)]
    public void ScaleRows_MatchTableOne(DiTToModelSize size, int layers, int width, int heads)
    {
        var o = new DiTToTTSOptions().UseSize(size);
        Assert.Equal((layers, width, heads), (o.DiTLayers, o.DiTHiddenDim, o.DiTHeads));
    }

    [Fact]
    public void NativeGraph_HasEveryPublishedComponentAndNineWayAdaLn()
    {
        using var model = Tiny();
        Assert.NotNull(model.TextEncoder);
        Assert.NotNull(model.MelVae);
        Assert.NotNull(model.LengthPredictor);
        Assert.NotNull(model.DiffusionTransformer);
        Assert.NotNull(model.BigVgan);
        Assert.Equal(72, model.DiffusionTransformer!.ModulationWidth);
        Assert.Equal(2, model.MelVae!.Quantizer.Depth);
        Assert.Equal(new[] { 1.0, 0.5 }, model.MelVae.Quantizer.Scales(), new DoubleArrayComparer(1e-12));
    }

    [Fact]
    public void ShiftedCosineSchedule_AndLengthLabels_FollowThePaper()
    {
        using var model = Tiny();
        var atHalf = model.NoiseLevel(0.5);
        double expectedPower = 0.09 / 1.09; // cosine alpha/(1-alpha)=1, shifted by scale^2 = .3^2
        Assert.Equal(Math.Sqrt(expectedPower), atHalf.Signal, 12);
        Assert.Equal(Math.Sqrt(1 - expectedPower), atHalf.Noise, 12);
        Assert.Equal(new[] { 12, 11, 10, 9, 8 }, DiTToLengthPredictor<double>.RemainingLabels(12, 4));
    }

    [Fact(Timeout = 300000)]
    public async Task MiniatureModel_RunsSynthesisAndBothIndependentObjectives()
    {
        await Task.Yield();
        using var model = Tiny();
        var speech = model.Synthesize("hi");
        Assert.True(speech.Length > 0);
        Assert.All(speech.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));

        var tokens = model.TextToTokens("hi");
        var mel = new Tensor<double>(new[] { 8, 4 });
        for (int i = 0; i < mel.Length; i++) mel.SetFlat(i, -3.0 + 0.1 * Math.Sin(i));
        var audio = new Tensor<double>(new[] { 32 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.2 * Math.Sin(0.2 * i);
        var sample = new TtsTrainingSample<double> { Tokens = tokens, Text = "hi", Mel = mel, Audio = audio };

        foreach (var stage in new[] { DiTToTrainingStage.Diffusion, DiTToTrainingStage.LengthPredictor })
        {
            model.CurrentStage = stage;
            double objective = Convert.ToDouble(model.EvaluateTrainingObjective(sample));
            Assert.True(TtsPaperNumerics.IsFinite(objective) && objective > 0, $"{stage}: {objective}");
            double trained = Convert.ToDouble(model.Train(sample));
            Assert.True(TtsPaperNumerics.IsFinite(trained) && trained > 0, $"{stage}: {trained}");
        }
    }

    private sealed class DoubleArrayComparer(double tolerance) : IEqualityComparer<double>
    {
        public bool Equals(double x, double y) => Math.Abs(x - y) <= tolerance;
        public int GetHashCode(double obj) => 0;
    }
}
