using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoicePublicPipelineTests
{
    private static string FilePath(string name) => Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
    private static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(FilePath("metavoice_public_reference.json")));
    private static MetaVoice1BOptions Options() => new()
    {
        FirstStageDim = 8, NumFirstStageLayers = 2, NumHeads = 2, FirstStageContext = 16,
        SecondStageDim = 8, NumSecondStageLayers = 2, SecondStageHeads = 2, SecondStageContext = 16,
        SpeakerHiddenDim = 8, SpeakerEmbeddingDim = 3, SpeakerMelChannels = 5, SwiGLUMultipleOf = 4,
        MaximumNewTokens = 5, FirstStageTopP = 1, SpeakerGuidance = 2, PromptGuidance = 2,
        TrainingDropout = 0, FineTuneLastBlocks = 1, MinReferenceDurationSec = .001, RawDecoderOutput = true
    };
    private static MetaVoice1B<double> Model()
    {
        AiDotNetEngine.ResetToCpu();
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 2294 };
        var model = new MetaVoice1B<double>(architecture, Options());
        model.LoadLanguageModelCheckpoints(FilePath("metavoice_public_first.pt"), FilePath("metavoice_public_second.pt"));
        model.LoadSpeakerEncoderCheckpoint(FilePath("metavoice_public_speaker.pt"));
        return model;
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoadedTokenizersAndActorObjective_SurviveCloneAndFreshRestoration(bool serialize)
    {
        using var document = Reference();
        var reference = document.RootElement;
        using var model = Model();
        int[] expectedIds = model.EncodeText("hello");
        var tokens = reference.GetProperty("training_tokens").EnumerateArray().Select(x => (double)x.GetInt32()).ToArray();
        var codes = reference.GetProperty("training_codes").EnumerateArray()
            .SelectMany(row => row.EnumerateArray().Select(x => (double)x.GetInt32())).ToArray();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { tokens.Length }, new Vector<double>(tokens)),
            CodecTokens = new Tensor<double>(new[] { 8, 2 }, new Vector<double>(codes)),
            SpeakerReference = new Tensor<double>(new[] { 3 }, new Vector<double>(Values(reference.GetProperty("speaker"))))
        };
        double expectedLoss = model.EvaluateTrainingObjective(sample);
        var frameMajor = new double[codes.Length];
        for (int frame = 0; frame < 2; frame++) for (int book = 0; book < 8; book++)
            frameMajor[frame * 8 + book] = codes[book * 2 + frame];
        var canonicalSample = new TtsTrainingSample<double>
        {
            Tokens = sample.Tokens,
            CodecTokens = new Tensor<double>(new[] { 2, 8 }, new Vector<double>(frameMajor)),
            SpeakerReference = sample.SpeakerReference
        };
        Assert.Equal(expectedLoss, model.EvaluateTrainingObjective(canonicalSample), 12);
        using var restored = serialize
            ? new MetaVoice1B<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
                NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Options())
            : (MetaVoice1B<double>)model.DeepCopy();
        if (serialize) restored.Deserialize(model.Serialize());
        Assert.Equal(expectedIds, restored.EncodeText("hello"));
        Assert.Equal(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
        Assert.Equal(expectedLoss, restored.EvaluateTrainingObjective(sample), 12);
    }

    private static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteNativeCodecAndFourBandPipeline_SurvivesCloneAndFreshRestoration(bool serialize)
    {
        AiDotNetEngine.ResetToCpu();
        using var model = MetaVoiceNativeFixture.Create<double>();
        var recording = new Tensor<double>(new[] { 641 }, new Vector<double>(Enumerable.Range(0, 641)
            .Select(index => .1 * Math.Sin(index * .13)).ToArray()));
        var expectedSpeaker = model.ExtractSpeakerEmbedding(recording, 16000);
        ControlNativeSampling(model);
        var expected = model.Synthesize("hello world");
        Assert.True(expected.Length > 0);
        Assert.All(expected.ToArray(), value => Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        Assert.Contains(expected.ToArray(), value => Math.Abs(value) > 1e-12);
        using var copy = serialize ? MetaVoiceNativeFixture.CreateUnloaded<double>() : (MetaVoice1B<double>)model.DeepCopy();
        if (serialize) copy.Deserialize(model.Serialize());
        Assert.Equal(model.GetParameters().ToArray(), copy.GetParameters().ToArray());
        Assert.Equal(expectedSpeaker.ToArray(), copy.ExtractSpeakerEmbedding(recording, 16000).ToArray());
        ControlNativeSampling(copy);
        var actual = copy.Synthesize("hello world");
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        double waveformError = expected.ToArray().Zip(actual.ToArray(), (a, b) => Math.Abs(a - b)).Max();
        Assert.True(waveformError < 1e-10, $"Restored waveform error {waveformError:R}.");
        Assert.Equal(model.GetParameters().ToArray(), copy.GetParameters().ToArray());
    }

    private static void ControlNativeSampling(MetaVoice1B<double> model)
    {
        // Couple random draws across the comparison; actors, codec and all four
        // diffusion networks still execute their complete native forward paths.
        int draw = 0;
        model.SamplingUniformForAudit = () => draw++ % 2 == 0 ? .1 : .5;
        model.DiffusionGaussianForAudit = shape => new Tensor<double>(shape,
            new Vector<double>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
                .Select(index => .1 * Math.Sin(index * .13)).ToArray()));
    }

    [Fact]
    public void PublicTextTokensAndInputDomain_UseCheckpointByteBpeOffset()
    {
        using var model = Model();
        Assert.Equal(model.EncodeText("hello"), model.TextToTokens("hello").ToArray().Select(value => (int)value).ToArray());
        var domain = model.GetInputDomain(new[] { 4 });
        Assert.True(domain.IsIndices);
        Assert.Equal(2049, domain.MinInclusive);
        Assert.Equal(2562, domain.MaxExclusive);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PublicSynthesis_RunsOriginalSpeakerActorsAndAll101DrawsThenPassesExactCodesToDecoder(bool streamed)
    {
        using var document = Reference(); var reference = document.RootElement;
        using var model = Model(); Assert.True(model.HasPaperPipeline);
        var uniforms = Values(reference.GetProperty("uniforms")); int draws = 0;
        model.SamplingUniformForAudit = () => uniforms[draws++];
        var routedWaveform = new Tensor<double>(new[] { 1, 1, 3 }, new Vector<double>(new[] { .125, -.25, .5 }));
        bool decoded = false;
        model.DecoderForAudit = codes =>
        {
            decoded = true; var expected = reference.GetProperty("final_codes");
            Assert.Equal(8, codes.GetLength(0)); Assert.Equal(expected[0].GetArrayLength(), codes.GetLength(1));
            for (int q = 0; q < 8; q++) for (int t = 0; t < codes.GetLength(1); t++) Assert.Equal(expected[q][t].GetInt32(), codes[q, t]);
            return routedWaveform;
        };
        var audio = Values(reference.GetProperty("reference_audio"));
        var recording = new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio));
        var embedding = model.ExtractSpeakerEmbedding(recording, 16000);
        double error = embedding.ToArray().Zip(Values(reference.GetProperty("speaker")), (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 2e-7, $"Original raw speaker frontend error {error:R}.");
        string store = Path.Combine(Path.GetTempPath(), "metavoice-stream-" + Guid.NewGuid().ToString("N"));
        try
        {
            if (streamed) model.ConfigureWeightLifetime(new GpuOffloadOptions
            {
                StreamingBackingStorePath = store, StreamingPoolMaxResidentBytes = 1024,
                StreamingStoreDtype = StreamingStoreDtype.FullPrecision
            });
            var waveform = model.SynthesizeWithVoice(reference.GetProperty("text").GetString()!, recording, 16000);
            Assert.True(decoded); Assert.Equal(101, draws); Assert.Equal(new[] { 3 }, waveform.Shape.ToArray());
            Assert.Equal(routedWaveform.ToArray(), waveform.ToArray());
        }
        finally
        {
            if (streamed)
            {
                WeightRegistry.SetStreamingExecutionTraining(null);
                WeightRegistry.Reset();
                if (Directory.Exists(store)) Directory.Delete(store, recursive: true);
            }
        }
    }

    [Fact]
    public void PublicFineTuning_MatchesOriginalObjectiveAdamWStepAndEveryFrozenParameter()
    {
        using var document = Reference(); var reference = document.RootElement;
        using var model = Model(); model.TrainingSpeakerDrawForAudit = .5;
        var expected = reference.GetProperty("trained_first");
        var state = new WeightLoader().LoadWeights(FilePath("metavoice_public_first.pt"));
        var named = new Dictionary<Tensor<double>, (string Name, double[] After)>();
        var unmatched = expected.EnumerateObject().ToDictionary(pair => pair.Name, pair => pair.Value);
        foreach (var layer in model.Layers)
        {
            var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
            Assert.Single(parameters); var parameter = parameters[0];
            double[] Convert(double[] values, int[] shape)
            {
                if (layer is not BiasFreeLinearLayer<double>) return values;
                if (shape.Length != 2) return Array.Empty<double>();
                var transposed = new double[values.Length];
                for (int o = 0; o < shape[0]; o++) for (int i = 0; i < shape[1]; i++) transposed[i * shape[0] + o] = values[o * shape[1] + i];
                return transposed;
            }
            var candidates = unmatched.Where(pair =>
            {
                var original = state["model." + pair.Key];
                return original.Length == parameter.Length && Convert(original.ToArray().Select(x => (double)x).ToArray(), original.Shape.ToArray())
                    .Zip(parameter.ToArray(), (a, b) => Math.Abs(a - b)).DefaultIfEmpty(double.PositiveInfinity).Max() < 1e-12;
            }).ToArray();
            Assert.Single(candidates); var candidate = candidates[0];
            named.Add(parameter, (candidate.Key, Convert(Values(candidate.Value.GetProperty("values")),
                candidate.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray())));
            unmatched.Remove(candidate.Key);
        }
        Assert.Empty(unmatched);
        var tokens = reference.GetProperty("training_tokens").EnumerateArray().Select(x => (double)x.GetInt32()).ToArray();
        var codecValues = reference.GetProperty("training_codes").EnumerateArray().SelectMany(row => row.EnumerateArray().Select(x => (double)x.GetInt32())).ToArray();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { tokens.Length }, new Vector<double>(tokens)),
            CodecTokens = new Tensor<double>(new[] { 8, 2 }, new Vector<double>(codecValues)),
            SpeakerReference = new Tensor<double>(new[] { 3 }, new Vector<double>(Values(reference.GetProperty("speaker"))))
        };
        double evaluation = model.EvaluateTrainingObjective(sample);
        Assert.True(Math.Abs(evaluation - reference.GetProperty("evaluation_loss").GetDouble()) < 1e-9);
        var before = model.GetParameters().ToArray();
        int firstScalars = named.Keys.Sum(parameter => parameter.Length);
        double loss = model.Train(sample);
        Assert.True(Math.Abs(loss - reference.GetProperty("training_loss").GetDouble()) < 1e-9);
        foreach (var pair in named)
        {
            double error = pair.Key.ToArray().Zip(pair.Value.After, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 1e-9, $"Original MetaVoice AdamW parameter {pair.Value.Name} error {error:R}.");
        }
        var after = model.GetParameters().ToArray();
        Assert.Equal(before.Length, after.Length);
        Assert.Equal(before.Skip(firstScalars), after.Skip(firstScalars));
    }

    [Fact]
    public void Defaults_MatchBothActualCheckpointArchitecturesAndReleasedFineTuningRecipe()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(FilePath("metavoice_release_contracts.json")));
        var stages = document.RootElement.GetProperty("stages"); var first = stages.GetProperty("first").GetProperty("config");
        var second = stages.GetProperty("second").GetProperty("config"); var recipe = document.RootElement.GetProperty("fine_tuning");
        var options = new MetaVoice1BOptions();
        Assert.Equal(first.GetProperty("n_embd").GetInt32(), options.FirstStageDim);
        Assert.Equal(first.GetProperty("n_layer").GetInt32(), options.NumFirstStageLayers);
        Assert.Equal(first.GetProperty("n_head").GetInt32(), options.NumHeads);
        Assert.Equal(first.GetProperty("block_size").GetInt32(), options.FirstStageContext);
        Assert.Equal(second.GetProperty("n_embd").GetInt32(), options.SecondStageDim);
        Assert.Equal(second.GetProperty("n_layer").GetInt32(), options.NumSecondStageLayers);
        Assert.Equal(second.GetProperty("n_head").GetInt32(), options.SecondStageHeads);
        Assert.Equal(second.GetProperty("block_size").GetInt32(), options.SecondStageContext);
        Assert.Equal(recipe.GetProperty("last_n_blocks_to_finetune").GetInt32(), options.FineTuneLastBlocks);
        Assert.Equal(recipe.GetProperty("dropout").GetDouble(), options.TrainingDropout);
        Assert.Equal(24000, options.SampleRate); Assert.Equal(320, options.HopSize); Assert.Equal(75, options.CodecFrameRate);
    }

    [Fact]
    public void Copy_PreservesInheritedAndStageSpecificConfiguration()
    {
        var original = Options(); original.SampleRate = 32000; original.HopSize = 17; original.MelChannels = 13;
        original.DropoutRate = .37; original.FirstStageTopK = 7; original.FirstStageTopP = null;
        var copy = new MetaVoice1BOptions(original);
        Assert.Equal(original.SampleRate, copy.SampleRate); Assert.Equal(original.HopSize, copy.HopSize);
        Assert.Equal(original.MelChannels, copy.MelChannels); Assert.Equal(original.DropoutRate, copy.DropoutRate);
        Assert.Equal(original.SpeakerEmbeddingDim, copy.SpeakerEmbeddingDim);
        Assert.Equal(original.FirstStageContext, copy.FirstStageContext); Assert.Equal(original.SecondStageContext, copy.SecondStageContext);
        Assert.Equal(original.SecondStageHeads, copy.SecondStageHeads); Assert.Equal(original.MaximumNewTokens, copy.MaximumNewTokens);
        Assert.Equal(original.FirstStageTopK, copy.FirstStageTopK); Assert.Null(copy.FirstStageTopP);
        Assert.Equal(original.TrainingDropout, copy.TrainingDropout); Assert.Equal(original.FineTuneLastBlocks, copy.FineTuneLastBlocks);
    }
}
