using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class UniAudioCodecTrainingTests
{
    [Fact]
    public void SwitchingBackToLanguageModel_CopyRetainsCodecCriticGraph()
    {
        var options = Options(); using var model = Model(options);
        Assert.True(options.IncludeCodecDiscriminators);
        options.TrainingComponent = UniAudioTrainingComponent.LanguageModel;
        var copy = new UniAudioOptions(options);
        using var restored = Model(copy);
        Assert.True(copy.IncludeCodecDiscriminators);
        Assert.NotNull(Field<UniAudioCodecGan<double>>(restored, "_codecGan"));
        Assert.Equal(model.GetParameters().Length, restored.GetParameters().Length);
        restored.SetParameters(model.GetParameters());
        Assert.Equal(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
    }

    private static UniAudioOptions Options() => new()
    {
        TrainingComponent = UniAudioTrainingComponent.UniversalCodec,
        UseReleasedTextFrontend = false,
        SampleRate = 320, HopSize = 32, CodecFrameRate = 10,
        NumCodebooks = 3, CodebookSize = 8, CodecTotalQuantizers = 3,
        CodecFilters = 2, CodecDimension = 6, CodecRatios = [4, 2, 2, 2],
        CodecKMeansIterations = 0, CodecDeadCodeThreshold = 0,
        LLMDim = 12, TextEncoderDim = 12, NumHeads = 3, NumEncoderLayers = 1, NumLLMLayers = 1,
        GlobalFeedForwardDim = 24, LocalFeedForwardDim = 24, JointVocabularySize = 160, VocabSize = 160,
        AudioTokenOffset = 32, PhoneTokenOffset = 57, PhoneVocabularySize = 80,
        MaxTextLength = 8, MaxContextPatches = 64, MaxCodecFrames = 64, PromptSeconds = .1,
        CodecTraining = new UniAudioCodecTrainingOptions
        {
            Bandwidths = [.03, .06, .09], SegmentSize = 96,
            DiscriminatorHops = [4], DiscriminatorChannels = [32],
            FullFftSizes = [32, 64], FullWindowSizes = [24, 48], FullHopSizes = [8, 16],
            SubFftSizes = [8, 16], SubWindowSizes = [6, 12], SubHopSizes = [2, 4], Subbands = 6,
        },
    };

    private static UniAudio<double> Model(UniAudioOptions options)
    {
        var model = new UniAudio<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 160), options);
        model.DisableAutoStreaming();
        return model;
    }
    private static Tensor<double> Audio() => new([96], new Vector<double>(Enumerable.Range(0, 96)
        .Select(i => .2 * Math.Sin(i * .17) + .05 * Math.Cos(i * .037)).ToArray()));
    private static T Field<T>(UniAudio<double> model, string name) => (T)typeof(UniAudio<double>)
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
    private static IReadOnlyList<Tensor<double>> Parameters(IEnumerable<LayerBase<double>> layers) =>
        AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList(), -1);
    private static double[][] Snapshot(IReadOnlyList<Tensor<double>> parameters) => parameters.Select(x => x.ToArray()).ToArray();
    private static bool Changed(double[][] snapshot, IReadOnlyList<Tensor<double>> parameters) =>
        snapshot.Where((values, i) => !values.SequenceEqual(parameters[i].ToArray())).Any();
    private static TtsTrainingSample<double> CodecSample() => new()
    {
        Tokens = new Tensor<double>(new[] { 0 }), Audio = Audio(),
    };

    [Fact]
    public void CodecPretraining_UsesRecordingOnlyAndUpdatesCodecAndCriticsWhileFreezingLanguageModels()
    {
        var options = Options(); using var model = Model(options);
        var codec = Parameters(Field<List<LayerBase<double>>>(model, "_codecLayers"));
        var language = Parameters(Field<List<LayerBase<double>>>(model, "_languageLayers"));
        var critics = Parameters(Field<UniAudioCodecGan<double>>(model, "_codecGan").Layers);
        var codecBefore = Snapshot(codec); var languageBefore = Snapshot(language); var criticsBefore = Snapshot(critics);
        Assert.Equal(TtsSupervision.Recording, model.TrainingSupervision);
        Assert.Equal(0, model.MaxGradNormValue);
        for (int step = 0; step < 2; step++) Assert.True(TtsPaperNumerics.IsFinite(model.Train(CodecSample())));
        Assert.Equal(2, options.CodecTraining.CompletedSteps);
        Assert.True(Changed(codecBefore, codec));
        Assert.True(Changed(criticsBefore, critics));
        Assert.False(Changed(languageBefore, language));
        Assert.True(TtsPaperNumerics.IsFinite(model.LastCodecGeneratorLoss));
        Assert.True(TtsPaperNumerics.IsFinite(model.LastCodecDiscriminatorLoss));
    }

    [Fact]
    public void LanguageTraining_AfterCodecPretrainingFreezesCodecAndCritics()
    {
        var options = Options(); using var model = Model(options);
        model.Train(CodecSample());
        options.TrainingComponent = UniAudioTrainingComponent.LanguageModel;
        var codec = Parameters(Field<List<LayerBase<double>>>(model, "_codecLayers"));
        var language = Parameters(Field<List<LayerBase<double>>>(model, "_languageLayers"));
        var critics = Parameters(Field<UniAudioCodecGan<double>>(model, "_codecGan").Layers);
        var codecBefore = Snapshot(codec); var languageBefore = Snapshot(language); var criticsBefore = Snapshot(critics);
        var quantizer = Field<UniAudioCodec<double>>(model, "_codec").Quantizer;
        var codebooks = Enumerable.Range(0, options.CodecTotalQuantizers).Select(i => quantizer.Codebook(i).ToArray()).ToArray();
        var sample = new TtsTrainingSample<double>
        {
            Text = "hello",
            Tokens = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 1d, 2d })),
            Audio = Audio(), SpeakerReference = Audio(),
        };
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Assert.True(Changed(languageBefore, language));
        Assert.False(Changed(codecBefore, codec));
        Assert.False(Changed(criticsBefore, critics));
        for (int i = 0; i < codebooks.Length; i++) Assert.Equal(codebooks[i], quantizer.Codebook(i).ToArray());
    }

    [Fact]
    public void EpochDecay_AppliesActualAuthorSchedulerRatesToBothOptimizers()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "uniaudio_codec_training_reference.json")));
        var options = Options(); using var model = Model(options);
        var generator = Field<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>(model, "_codecGeneratorOptimizer");
        var discriminator = Field<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>(model, "_codecDiscriminatorOptimizer");
        int epoch = 0;
        foreach (var rate in document.RootElement.GetProperty("optimizer").GetProperty("epoch_rates").EnumerateArray())
        {
            model.SetCodecTrainingEpoch(epoch++);
            Assert.Equal(rate.GetDouble(), generator.GetCurrentLearningRate(), 15);
            Assert.Equal(rate.GetDouble(), discriminator.GetCurrentLearningRate(), 15);
        }
        double expected = generator.GetCurrentLearningRate();
        model.Train(CodecSample());
        Assert.Equal(expected, generator.GetCurrentLearningRate(), 15);
        Assert.Equal(expected, discriminator.GetCurrentLearningRate(), 15);
    }

    [Fact]
    public void CodecEvaluation_DoesNotUpdateWeightsCodebooksOrStepCount()
    {
        var options = Options(); options.CodecTraining.CompletedSteps = 1;
        using var model = Model(options);
        var parameters = Parameters(Field<List<LayerBase<double>>>(model, "_codecLayers"));
        var before = Snapshot(parameters);
        var quantizer = Field<UniAudioCodec<double>>(model, "_codec").Quantizer;
        var codebook = quantizer.Codebook(0).ToArray();
        bool mode = model.IsTrainingMode;
        Assert.True(TtsPaperNumerics.IsFinite(model.EvaluateTrainingObjective(CodecSample())));
        Assert.False(Changed(before, parameters));
        Assert.Equal(codebook, quantizer.Codebook(0).ToArray());
        Assert.Equal(1, options.CodecTraining.CompletedSteps);
        Assert.Equal(mode, model.IsTrainingMode);
    }
}
