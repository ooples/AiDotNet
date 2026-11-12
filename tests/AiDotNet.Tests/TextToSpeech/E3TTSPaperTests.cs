using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Paper invariants and an executable miniature of Gao et al.'s E3 TTS.</summary>
public sealed class E3TTSPaperTests
{
    [Theory]
    [InlineData(-1.0)]
    [InlineData(.3)]
    [InlineData(1.0)]
    public void EquationSix_WeightedLossAndAllPredictionVarianceDerivativesMatchClosedForm(double logOmega)
    {
        using var model = Tiny();
        double[] p = { .2, -.4, 1.2, .6 }, n = { -.1, .3, .2, -.5 }, w = { 1, 1, .1, .1 };
        var prediction = new Tensor<double>(new[] { 4 }, new Vector<double>(p));
        var noise = new Tensor<double>(new[] { 4 }, new Vector<double>(n));
        var weights = new Tensor<double>(new[] { 4 }, new Vector<double>(w));
        var variance = new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { logOmega }));
        double mse = Enumerable.Range(0, 4).Sum(i => w[i] * Math.Pow(p[i] - n[i], 2)) / w.Sum();
        using var tape = new GradientTape<double>();
        var loss = model.VarianceWeightedNoiseLoss(prediction, noise, weights, variance);
        Assert.Equal(Math.Exp(-logOmega) * mse + logOmega, loss[0], 13);
        var gradients = tape.ComputeGradients(loss, sources: new List<Tensor<double>> { prediction, variance });
        for (int i = 0; i < 4; i++)
            Assert.Equal(2 * Math.Exp(-logOmega) * w[i] * (p[i] - n[i]) / w.Sum(), gradients[prediction][i], 13);
        Assert.Equal(1 - Math.Exp(-logOmega) * mse, gradients[variance][0], 13);
    }

    internal static E3TTSOptions TinyOptions() => new()
    {
        BertHiddenDim = 8,
        BertLayers = 1,
        BertHeads = 2,
        BertIntermediateDim = 16,
        BertMaxPositions = 32,
        MaxTextLength = 32,
        BertDropout = 0,
        BlockChannels = [4, 4, 8, 8, 12, 16, 16, 16, 16, 16],
        BlockKernelSizes = [5, 5, 5, 5, 5, 3, 3, 3, 3, 3],
        BlockStrides = [2, 2, 2, 2, 4, 4, 2, 2, 2, 2],
        AdaptiveKernelCounts = [8, 8, 4, 4, 2, 0, 0, 0, 0, 0],
        AttentionHeads = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
        GroupNormGroups = 4,
        TimeEmbeddingDim = 8,
        ConditionDim = 16,
        SpeakerCount = 2,
        TrainingWaveformSamples = 4_096,
        InferenceWaveformSamples = 4_096,
        NumDiffusionSteps = 1,
        SilenceBlockSize = 128,
        SamplingSeed = 3,
    };

    private static E3TTS<double> Tiny() => new(new NeuralNetworkArchitecture<double>(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), TinyOptions());

    [Fact]
    public void Defaults_MatchThePublishedWaveformAndTableOneConfiguration()
    {
        var o = new E3TTSOptions();
        o.Validate();
        Assert.Equal(24_000, o.SampleRate);
        Assert.Equal(262_080, o.TrainingWaveformSamples);
        Assert.Equal(1_000, o.NumDiffusionSteps);
        Assert.Equal(1e-7, o.MinimumAlphaBar);
        Assert.Equal(0.1, o.PaddingLossWeight);
        Assert.Equal(new[] { 128, 128, 256, 256, 512, 1_024, 1_024, 1_024, 1_024, 1_024 }, o.BlockChannels);
        Assert.Equal(new[] { 5, 5, 5, 5, 5, 3, 3, 3, 3, 3 }, o.BlockKernelSizes);
        Assert.Equal(new[] { 2, 2, 2, 2, 4, 4, 2, 2, 2, 2 }, o.BlockStrides);
        Assert.Equal(4_096, o.BlockStrides.Aggregate(1, (a, b) => a * b));
        Assert.Equal(new[] { 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 }, o.BlockRepetitions);
        Assert.Equal(new[] { 8, 8, 4, 4, 2, 0, 0, 0, 0, 0 }, o.AdaptiveKernelCounts);
        Assert.Equal(new[] { 0, 0, 0, 0, 0, 8, 8, 8, 8, 8 }, o.AttentionHeads);
        Assert.Equal((30_522, 768, 12, 12, 3_072),
            (o.BertVocabularySize, o.BertHiddenDim, o.BertLayers, o.BertHeads, o.BertIntermediateDim));
    }

    [Fact]
    public void NativeGraph_ContainsBertAndAllPublishedDownUpBlocks()
    {
        using var model = Tiny();
        Assert.NotNull(model.TextEncoder);
        Assert.NotNull(model.WaveformUnet);
        Assert.Equal(10, model.WaveformUnet!.DownBlocks.Count);
        Assert.Equal(10, model.WaveformUnet.UpBlocks.Count);
        Assert.Equal(15, model.WaveformUnet.DownBlocks.Count + model.WaveformUnet.DownResiduals.Sum(x => x.Count));
        Assert.Equal(15, model.WaveformUnet.UpBlocks.Count + model.WaveformUnet.UpResiduals.Sum(x => x.Count));
        Assert.Equal(new[] { 8, 8, 4, 4, 2, 0, 0, 0, 0, 0 },
            model.WaveformUnet.DownBlocks.Select(b => b.AdaptiveKernelCount));
        Assert.Equal(5, model.WaveformUnet.DownBlocks.Count(b => b.AttentionHeads > 0));
        Assert.All(model.WaveformUnet.UpBlocks, b => Assert.True(b.IsTransposed));
    }

    [Fact]
    public void EquationSevenSchedule_HasThePublishedEndpointsAndShape()
    {
        var options = TinyOptions();
        options.NumDiffusionSteps = 1_000;
        using var model = new E3TTS<double>(new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        Assert.Equal(1.0, model.AlphaBar(0), 12);
        Assert.Equal(1e-7, model.AlphaBar(1_000), 15);
        double expected = Math.Exp(Math.Log(1e-7) * Math.Pow(1 - Math.Cos(Math.PI / 4), 1.5));
        Assert.Equal(expected, model.AlphaBar(500), 15);
    }

    [Fact]
    public void WaveformCanvas_NeedNotBeDivisibleByTheUnetStride()
    {
        using var model = Tiny();
        var audio = new Tensor<double>(new[] { 4032 });
        var text = model.TextEncoder!.Forward(new[] { 101, 102 });
        var result = model.WaveformUnet!.Forward(audio, text, 0.5, 0);
        Assert.Equal(4032, result.Epsilon.Length);
        Assert.All(result.Epsilon.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void Training_InterpolatesLinearBetaProductsRatherThanInferenceSchedule()
    {
        var options = TinyOptions();
        options.NumDiffusionSteps = 4;
        options.TrainingBetaStart = .01;
        options.TrainingBetaEnd = .04;
        var copy = new E3TTSOptions(options);
        Assert.Equal(.01, copy.TrainingBetaStart);
        Assert.Equal(.04, copy.TrainingBetaEnd);
        using var model = new E3TTS<double>(new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        double[] products = { 1, .99, .9702, .941094, .90345024 };
        for (int i = 0; i < products.Length; i++) Assert.Equal(products[i], model.TrainingAlphaBar(i), 14);
        Assert.True(Math.Abs(model.TrainingAlphaBar(4) - model.AlphaBar(4)) > .9);
        var sample = new TtsTrainingSample<double>
        {
            Tokens = model.TextToTokens("hello"), Audio = new Tensor<double>(new[] { 32 }),
        };
        var expectedRandom = new Random(2294);
        int interval = expectedRandom.Next(4);
        double expected = products[interval] + expectedRandom.NextDouble() * (products[interval + 1] - products[interval]);
        var prepare = typeof(E3TTS<double>).GetMethod("PrepareTrainingExample",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var example = prepare.Invoke(model, new object[] { sample, new Random(2294) })!;
        double actual = (double)example.GetType().GetProperty("AlphaBar")!.GetValue(example)!;
        Assert.Equal(expected, actual, 14);
    }

    [Fact]
    public void EquationSixVariance_DependsOnNoiseLevelRatherThanSpeakerIdentity()
    {
        using var model = Tiny();
        model.SetTrainingMode(false);
        var audio = new Tensor<double>(new[] { 4096 });
        var text = model.TextEncoder!.Forward(new[] { 101, 102 });
        var first = model.WaveformUnet!.Forward(audio, text, .25, 0);
        var otherSpeaker = model.WaveformUnet.Forward(audio, text, .25, 1);
        var otherTime = model.WaveformUnet.Forward(audio, text, .8, 0);
        Assert.Equal(first.LogVariance[0], otherSpeaker.LogVariance[0], 13);
        Assert.True(Math.Abs(first.LogVariance[0] - otherTime.LogVariance[0]) > 1e-8);
        Assert.False(first.Epsilon.ToArray().SequenceEqual(otherSpeaker.Epsilon.ToArray()));
    }

    [Fact(Timeout = 300000)]
    public async Task MiniatureModel_RunsWaveformSamplingPromptEditingAndKlObjective()
    {
        await Task.Yield();
        using var model = Tiny();
        var speech = model.Synthesize("hello");
        Assert.InRange(speech.Length, 1, 4_096);
        Assert.All(speech.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));

        var prompt = new Tensor<double>(new[] { 128 });
        for (int i = 0; i < prompt.Length; i++) prompt[i] = 0.1 * Math.Sin(i * 0.1);
        model.Voice = model.CreateVoice(prompt, 24_000, "hello");
        var continued = model.Synthesize("world");
        Assert.All(continued.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));

        var edited = model.Edit("hello", prompt, 32, 96, speakerId: 1);
        Assert.All(edited.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));

        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.2 * Math.Sin(i * 0.02);
        var sample = new TtsTrainingSample<double>
        {
            Tokens = model.TextToTokens("hello"), Text = "hello", Audio = audio, SpeakerId = 1,
        };
        double objective = Convert.ToDouble(model.EvaluateTrainingObjective(sample));
        Assert.True(TtsPaperNumerics.IsFinite(objective) && objective > 0, objective.ToString());
        double trained = Convert.ToDouble(model.Train(sample));
        Assert.True(TtsPaperNumerics.IsFinite(trained) && trained > 0, trained.ToString());
    }
}
