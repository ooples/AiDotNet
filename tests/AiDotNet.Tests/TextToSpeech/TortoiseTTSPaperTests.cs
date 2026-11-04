using System;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Regression tests for Betker's VQ-VAE -> GPT-2 -> CLVP -> diffusion -> UnivNet pipeline.</summary>
public class TortoiseTTSPaperTests
{
    [Fact]
    public void GenerationCap_RestrictsTheActualPipelineWithoutResizingCheckpointPositions()
    {
        var options = SmallOptions();
        options.MaxCodecFrames = 20;
        options.MaximumAutoregressiveTokens = 5;
        options.NumAutoregressiveSamples = 1;
        options.MinimumAutoregressiveTokens = 0;
        using var model = Create(options);
        model.Voice = model.CreateVoice(Audio(), options.SampleRate);
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var ar = (TortoiseAutoregressive<double>)typeof(TortoiseTTS<double>).GetField("_autoregressive", flags)!.GetValue(model)!;
        var head = (AiDotNet.NeuralNetworks.Layers.DenseLayer<double>)typeof(TortoiseAutoregressive<double>)
            .GetField("_speechHead", flags)!.GetValue(ar)!;
        var bias = new double[options.CodebookSize + 2];
        bias[2] = 100; // A non-silence token; no early EOS or silence cut.
        TorchParameters.Linear(AiDotNetEngine.Current, head, options.LLMDim, bias.Length,
            new double[options.LLMDim * bias.Length], bias);
        var wave = model.Synthesize("hi");
        Assert.Equal(5 * 4 * options.SampleRate / options.TokenizerSampleRate * options.HopSize, wave.Length);
        var positions = (AiDotNet.NeuralNetworks.Layers.LayerBase<double>)typeof(TortoiseAutoregressive<double>)
            .GetField("_speechPosition", flags)!.GetValue(ar)!;
        Assert.Equal((20 + 2 + options.MaxConditioningInputs) * options.LLMDim, positions.GetParameters().Length);
    }

    [Fact]
    public void ImmediateEos_UsesRepairedCandidateAndEightSilenceLatentsThroughTheRealPipeline()
    {
        var options = SmallOptions();
        options.MaxCodecFrames = 20;
        options.NumAutoregressiveSamples = 1;
        options.MinimumAutoregressiveTokens = 0;
        using var model = Create(options);
        model.Voice = model.CreateVoice(Audio(), options.SampleRate);
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var ar = (TortoiseAutoregressive<double>)typeof(TortoiseTTS<double>).GetField("_autoregressive", flags)!.GetValue(model)!;
        var head = (AiDotNet.NeuralNetworks.Layers.DenseLayer<double>)typeof(TortoiseAutoregressive<double>)
            .GetField("_speechHead", flags)!.GetValue(ar)!;
        var bias = new double[options.CodebookSize + 2];
        bias[options.SpeechStopToken] = 100;
        TorchParameters.Linear(AiDotNetEngine.Current, head, options.LLMDim, bias.Length,
            new double[options.LLMDim * bias.Length], bias);
        var wave = model.Synthesize("hi");
        // EOS is padded/repaired before ranking; nine silence codes trigger the
        // author's [:k] cut, retaining exactly eight latent frames for diffusion.
        int frames = 8 * 4 * options.SampleRate / options.TokenizerSampleRate;
        Assert.Equal(frames * options.HopSize, wave.Length);
        Assert.All(wave.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(0, true)]
    [InlineData(10, false)]
    public void UnivNetStage_UsesStftPretrainingAndSeparateAdversarialUpdates(int pretrainingSteps, bool criticChanges)
    {
        var options = SmallOptions(TortoiseTrainingComponent.UnivNet);
        options.VocoderTraining.PretrainingSteps = pretrainingSteps;
        using var model = Create(options);
        var sample = new TtsTrainingSample<double> { Tokens = Ids(3, 4), Text = "hi", Audio = Audio() };
        model.EvaluateTrainingObjective(sample);
        model.GetParameters();
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var generator = (TortoiseUnivNet<double>)typeof(TortoiseTTS<double>).GetField("_univNet", flags)!.GetValue(model)!;
        var critic = (AiDotNet.TextToSpeech.Vocoders.UnivNetTrainingLoss<double>)typeof(TortoiseTTS<double>).GetField("_vocoderTraining", flags)!.GetValue(model)!;
        var beforeGenerator = generator.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();
        var beforeCritic = critic.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Assert.False(beforeGenerator.SequenceEqual(generator.Layers.SelectMany(layer => layer.GetParameters().ToArray())));
        Assert.Equal(criticChanges, !beforeCritic.SequenceEqual(critic.Layers.SelectMany(layer => layer.GetParameters().ToArray())));
    }

    [Fact]
    public void Clvp_MatchesOfficialXtransformersScoresAndSymmetricBatchLoss()
    {
        using var reference = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "tortoise_clvp_reference.json")));
        var root = reference.RootElement;
        var clvp = new TortoiseClvp<double>(AiDotNetEngine.Current,
            new List<AiDotNet.NeuralNetworks.Layers.LayerBase<double>>(), new TortoiseTTSOptions
            { VocabSize = 16, CodebookSize = 8, ClvpDim = 8, ClvpHeads = 2, ClvpLayers = 1,
              ClvpTextLength = 8, ClvpSpeechLength = 8 });
        clvp.LoadTorchWeights((name, shape) => root.GetProperty("weights").GetProperty(name).EnumerateArray().Select(x => x.GetDouble()).ToArray());
        var texts = root.GetProperty("texts").EnumerateArray().Select(a => (IReadOnlyList<int>)a.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
        var speech = root.GetProperty("speech").EnumerateArray().Select(a => (IReadOnlyList<int>)a.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
        for (int i = 0; i < texts.Length; i++) for (int j = 0; j < speech.Length; j++)
        {
            double expected = root.GetProperty("scores")[i][j].GetDouble();
            double actual = clvp.Score(texts[i], speech[j]);
            Assert.True(Math.Abs(actual - expected) < 2e-6, $"CLVP score ({i},{j}): native={actual:R}, official={expected:R}.");
        }
        double loss = clvp.ContrastiveBatchLoss(texts, speech, new Random(1), training: false)[0];
        Assert.True(Math.Abs(loss - root.GetProperty("symmetric_loss").GetDouble()) < 2e-6);
        Assert.Equal(0, clvp.ContrastiveBatchLoss(texts.Take(1).ToArray(), speech.Take(1).ToArray(), new Random(1), training: false)[0]);
    }

    private static TortoiseTTSOptions SmallOptions(
        TortoiseTrainingComponent component = TortoiseTrainingComponent.Autoregressive) => new()
        {
            SampleRate = 1600,
            VocoderTraining = new AiDotNet.TextToSpeech.Vocoders.UnivNetOptions
            { StftFftSizes = [16, 32], StftHopSizes = [4, 8], StftWindowSizes = [16, 24],
                ResolutionDiscriminatorChannels = 2, DiscriminatorPeriods = [2, 3],
                PeriodDiscriminatorChannels = [2, 2, 2, 2, 2], PretrainingSteps = 0 },
            TokenizerSampleRate = 1600,
            MelChannels = 8,
            AutoregressiveMelChannels = 8,
            FftSize = 32,
            AutoregressiveFftSize = 32,
            HopSize = 8,
            AutoregressiveHopSize = 8,
            ConditioningSamples = 64,
            DiffusionConditioningSamples = 64,
            VocabSize = 256,
            MaxTextLength = 8,
            TextStartToken = 255,
            TextStopToken = 0,
            CodebookSize = 8,
            SpeechStartToken = 8,
            SpeechStopToken = 9,
            LLMDim = 8,
            TextEncoderDim = 8,
            NumLLMLayers = 1,
            NumEncoderLayers = 1,
            NumHeads = 2,
            AutoregressiveFeedForwardDim = 16,
            MaxCodecFrames = 2,
            NumAutoregressiveSamples = 2,
            MinimumAutoregressiveTokens = 1,
            ClvpDim = 8,
            ClvpLayers = 1,
            ClvpHeads = 2,
            ClvpTextLength = 8,
            ClvpSpeechLength = 8,
            DiffusionDim = 8,
            DiffusionLayers = 2,
            DiffusionHeads = 2,
            DiffusionTrainingSteps = 1000,
            DiffusionInferenceSteps = 2,
            TokenizerLatentDim = 8,
            TokenizerCodeDim = 2,
            UnivNetNoiseDim = 4,
            UnivNetChannels = 4,
            UnivNetUpsampleRates = new[] { 2, 2, 2 },
            UnivNetDilations = new[] { 1, 3 },
            UnivNetKernelPredictorHidden = 4,
            UnivNetPaddingFrames = 1,
            TrainingComponent = component,
        };

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 10);

    private static TortoiseTTS<double> Create(TortoiseTTSOptions? options = null) =>
        new(Architecture(), options ?? SmallOptions());

    private static Tensor<double> Audio(int samples = 96, double phase = 0)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.35 * Math.Sin(0.19 * i + phase) + 0.08 * Math.Cos(0.047 * i);
        return audio;
    }

    private static Tensor<double> Ids(params int[] ids)
    {
        var tensor = new Tensor<double>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = ids[i];
        return tensor;
    }

    [Fact]
    public void ReferenceDefaults_MatchReleasedTortoiseModels()
    {
        var options = new TortoiseTTSOptions();

        Assert.Equal(30, options.NumLLMLayers);
        Assert.Equal(1024, options.LLMDim);
        Assert.Equal(16, options.NumHeads);
        Assert.Equal(8192, options.CodebookSize);
        Assert.Equal(20, options.ClvpLayers);
        Assert.Equal(768, options.ClvpDim);
        Assert.Equal(10, options.DiffusionLayers);
        Assert.Equal(200, options.DiffusionInferenceSteps);
        Assert.Equal(256, options.UnivNetUpsampleRates.Aggregate(1, (product, rate) => product * rate));
    }

    [Fact]
    public void NativeModel_ExposesAllFivePaperComponents()
    {
        var model = Create();
        var metadata = model.GetModelMetadata();

        Assert.True(model.HasPaperPipeline);
        Assert.Equal("VQ-VAE -> GPT-2 candidates -> CLVP -> spaced DDPM -> UnivNet-c32",
            metadata.AdditionalInfo["Architecture"]);
        Assert.Equal(1, metadata.AdditionalInfo["AutoregressiveLayers"]);
        Assert.Equal(1, metadata.AdditionalInfo["ClvpLayersPerTower"]);
        Assert.Equal(true, metadata.AdditionalInfo["OfficialWeightsAvailable"]);
    }

    [Fact]
    public void SpeechTokenizer_ProducesOneBoundedCodeStreamAndDecodesThroughUnivNet()
    {
        var model = Create();

        var codes = model.EncodeToTokens(Audio());
        var wave = model.DecodeFromTokens(codes);

        Assert.NotEmpty(codes.ToArray());
        Assert.All(codes.ToArray(), code => Assert.InRange(code, 0, 7));
        Assert.Equal(codes.Length * 4 * 8, wave.Length);
        Assert.All(wave.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void Synthesis_RequiresAReferenceAndRunsCandidateRankingDiffusionAndVocoder()
    {
        var model = Create();
        Assert.Throws<InvalidOperationException>(() => model.Synthesize("hi"));

        model.Voice = model.CreateVoice(Audio(), 1600);
        var wave = model.Synthesize("hi");

        Assert.NotEmpty(wave.ToArray());
        Assert.All(wave.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        Assert.Contains(wave.ToArray(), value => Math.Abs(value) > 1e-10);
    }

    [Theory]
    [InlineData(TortoiseTrainingComponent.Autoregressive)]
    [InlineData(TortoiseTrainingComponent.Clvp)]
    [InlineData(TortoiseTrainingComponent.Diffusion)]
    [InlineData(TortoiseTrainingComponent.SpeechTokenizer)]
    [InlineData(TortoiseTrainingComponent.UnivNet)]
    public void DisclosedComponentObjectives_AreFinite(TortoiseTrainingComponent component)
    {
        var model = Create(SmallOptions(component));
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Ids(2, 3),
            Audio = Audio(),
            SpeakerReference = Audio(96, 0.7),
        };

        double objective = model.EvaluateTrainingObjective(sample);

        Assert.True(TtsPaperNumerics.IsFinite(objective), $"Objective was {objective}.");
        Assert.True(objective >= 0);
    }

    [Fact]
    public void AutoregressiveTraining_UpdatesNativeParameters()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double> { Text = "hi", Tokens = Ids(2, 3), Audio = Audio() };
        var before = model.GetParameters().ToArray();

        double loss = model.Train(sample);
        var after = model.GetParameters().ToArray();

        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
    }

    [Fact]
    public void ClvpScore_DependsOnTheSpeechCandidate()
    {
        var model = Create();

        double first = model.ScoreCandidate(Ids(2, 3), Ids(1, 2));
        double second = model.ScoreCandidate(Ids(2, 3), Ids(6, 7));

        Assert.True(TtsPaperNumerics.IsFinite(first));
        Assert.True(TtsPaperNumerics.IsFinite(second));
        Assert.NotEqual(first, second);
    }

    [Fact]
    public void ConvertedCheckpoint_IsStrictAndRestoresEveryComponent()
    {
        var source = Create();
        var expected = source.GetParameters().ToArray();
        var restored = Create();

        restored.LoadConvertedWeights(name => name == "tortoise.parameters" ? expected : null);

        Assert.Equal(expected, restored.GetParameters().ToArray());
        Assert.Throws<InvalidDataException>(() =>
            restored.LoadConvertedWeights(_ => new double[expected.Length - 1]));
    }
}
