using AiDotNet.Audio.TextToSpeech;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Audio;

public sealed class AlignedMelRuntimeTests
{
    private readonly ITestOutputHelper _output;
    public AlignedMelRuntimeTests(ITestOutputHelper output)
    {
        _output = output;
        TestModuleInitializer.EnsureInitialized();
    }

    [Fact]
    public void ExplicitDurationsExpandTokensAndKeepPaddingOutOfEveryLoss()
    {
        using var model = CreateModel();
        var batch = Batch();
        var result = model.EvaluateAlignedMel(batch);
        Assert.Equal(new[] { 2, 5, 4 }, result.Output.MelSpectrogram.Shape.ToArray());
        Assert.Equal(new[] { 5, 3 }, result.Output.MelLengths);
        Assert.Equal(new[] { 2, 3, 0, 1, 1, 1 }, Values(result.Output.TokenDurations));
        for (int feature = 0; feature < 4; feature++)
        {
            Assert.Equal(result.Output.MelSpectrogram[0, 0, feature], result.Output.MelSpectrogram[0, 1, feature]);
            Assert.Equal(result.Output.MelSpectrogram[0, 2, feature], result.Output.MelSpectrogram[0, 4, feature]);
            Assert.Equal(0, result.Output.MelSpectrogram[1, 3, feature]);
            Assert.Equal(0, result.Output.MelSpectrogram[1, 4, feature]);
        }
        Assert.Equal(0, result.Output.LogDurations[0, 2]);
        AssertFinite(result.TotalLoss);

        batch.Tokens[0, 2] = int.MaxValue;
        for (int frame = 3; frame < 5; frame++)
            for (int feature = 0; feature < 4; feature++) batch.TargetMels[1, frame, feature] = float.NaN;
        var changedPadding = model.EvaluateAlignedMel(batch);
        Assert.Equal(Values(result.TotalLoss), Values(changedPadding.TotalLoss));
        Assert.Equal(Values(result.Output.MelSpectrogram), Values(changedPadding.Output.MelSpectrogram));
    }

    [Fact]
    public void MaskedTargetsHaveExactlyZeroGradientWhileActiveTargetsAndEachBranchTrain()
    {
        using var model = CreateModel();
        var batch = Batch();
        _ = model.EvaluateAlignedMel(batch); // Materialize all real branch weights before the probe.
        using var tape = new GradientTape<float>();
        var objective = model.EvaluateAlignedMel(batch);
        var parameters = model.ParametersForTape();
        var requested = parameters.Concat(new[] { batch.TargetMels }).ToArray();
        var gradients = tape.ComputeGradients(objective.TotalLoss, requested);
        Assert.True(gradients.TryGetValue(batch.TargetMels, out var targetGradient));
        if (targetGradient is null) throw new Xunit.Sdk.XunitException("The active mel target has no gradient.");
        Assert.Contains(Values(targetGradient), value => Math.Abs(value) > 1e-7);
        for (int frame = 3; frame < 5; frame++)
            for (int feature = 0; feature < 4; feature++) Assert.Equal(0, targetGradient[1, frame, feature]);

        AssertBranchHasGradient(model.Extra<TokenDurationPredictorLayer<float>>(), gradients);
        AssertBranchHasGradient(model.Extra<EmbeddingLayer<float>>(), gradients);
        AssertBranchHasGradient(model.Extra<FullyConnectedLayer<float>>(), gradients);
        Assert.Contains(model.DecoderParameters(), parameter => gradients.TryGetValue(parameter, out var gradient)
            && gradient is not null && Values(gradient).Any(value => Math.Abs(value) > 1e-7));
    }

    [Fact]
    public void DurationLossDetachesEncoderButStillDifferentiatesTheDurationWeights()
    {
        using var model = CreateModel();
        var batch = Batch();
        _ = model.EvaluateAlignedMel(batch);
        using var tape = new GradientTape<float>();
        var objective = model.EvaluateAlignedMel(batch);
        var gradients = tape.ComputeGradients(objective.DurationLoss, model.ParametersForTape());
        AssertBranchHasGradient(model.Extra<TokenDurationPredictorLayer<float>>(), gradients);
        foreach (var parameter in model.Extra<EmbeddingLayer<float>>().GetTrainableParameters())
            Assert.False(gradients.TryGetValue(parameter, out var gradient) && gradient is not null
                && Values(gradient).Any(value => value != 0));
    }

    [Fact]
    public void LearnedAlignmentCoversEveryTokenAndTheCompleteMelSequence()
    {
        using var model = CreateModel();
        var explicitBatch = Batch();
        var batch = new AlignedMelBatch<float>(explicitBatch.Tokens, explicitBatch.TokenLengths,
            explicitBatch.TargetMels, explicitBatch.MelLengths);
        var result = model.EvaluateAlignedMel(batch);
        for (int row = 0; row < batch.TokenLengths.Count; row++)
        {
            int sum = 0;
            for (int token = 0; token < batch.TokenLengths[row]; token++)
            {
                Assert.True(result.Output.TokenDurations[row, token] >= 1);
                sum += result.Output.TokenDurations[row, token];
            }
            Assert.Equal(batch.MelLengths[row], sum);
        }
        AssertFinite(result.TotalLoss);
    }

    [Fact]
    public void LearnedAlignmentSelectsTheActualGaussianScoreOptimumNotUniformDurations()
    {
        using var model = new ControlledAlignmentModel();
        var tokens = new Tensor<int>(new[] { 1, 2 });
        tokens[0, 0] = 1; tokens[0, 1] = 2;
        var targets = new Tensor<float>(new[] { 1, 5, 2 });
        for (int frame = 2; frame < 5; frame++) targets[0, frame, 0] = 2;
        var objective = model.EvaluateAlignedMel(new AlignedMelBatch<float>(tokens, new[] { 2 }, targets, new[] { 5 }));
        Assert.Equal(new[] { 2, 3 }, Values(objective.Output.TokenDurations));
        Assert.Equal(0, objective.PriorLoss[0]);
        Assert.Equal(0, objective.MelLoss[0]);
        Assert.Equal(Values(targets), Values(objective.Output.MelSpectrogram));
    }

    [Fact]
    public void LearnedDurationsDriveInferenceAndSpeakingRateWithoutChangingTokenIds()
    {
        using var model = CreateModel();
        var tokens = new Tensor<int>(new[] { 1, 2 });
        tokens[0, 0] = 1;
        tokens[0, 1] = 2;
        SetConstantDuration(model, Math.Log(2.25));
        var slow = model.SynthesizeMel(tokens, new[] { 2 });
        var fast = model.SynthesizeMel(tokens, new[] { 2 }, speakingRate: 2);
        Assert.Equal(new[] { 6 }, slow.MelLengths);
        Assert.Equal(new[] { 4 }, fast.MelLengths);
        Assert.Equal(new[] { 3, 3 }, Values(slow.TokenDurations));
        Assert.Equal(new[] { 2, 2 }, Values(fast.TokenDurations));
        Assert.Equal(new[] { 1, 2 }, Values(tokens));
        Assert.Throws<InvalidOperationException>(() => model.SynthesizeMel(tokens, new[] { 2 }, maximumMelFrames: 3));
    }

    [Fact]
    public void RealAlignedTrainingUpdatesDurationWeightsAndRoundTripsAllBranches()
    {
        using var model = CreateModel();
        var batch = Batch();
        _ = model.EvaluateAlignedMel(batch);
        var duration = model.Extra<TokenDurationPredictorLayer<float>>();
        var before = duration.GetParameters().ToArray();
        float loss = model.Train(batch);
        Assert.False(float.IsNaN(loss) || float.IsInfinity(loss));
        Assert.False(before.SequenceEqual(duration.GetParameters().ToArray()));
        var updated = duration.GetParameters().ToArray();
        double durationDelta = Math.Sqrt(before.Zip(updated, (left, right) => Math.Pow(left - right, 2)).Sum());
        _output.WriteLine($"Actual aligned step loss={loss:R}; duration parameters={before.Length}; update L2={durationDelta:R}.");
        var expected = model.EvaluateAlignedMel(batch);
        byte[] serialized = model.Serialize();
        using var restored = CreateModel();
        restored.Deserialize(serialized);
        var actual = restored.EvaluateAlignedMel(batch);
        Assert.Equal(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
        Assert.Equal(model.ParameterCount, restored.ParameterCount);
        Assert.Equal(Values(expected.Output.MelSpectrogram), Values(actual.Output.MelSpectrogram));
        Assert.Equal(Values(expected.Output.LogDurations), Values(actual.Output.LogDurations));
        Assert.Equal(Values(expected.TotalLoss), Values(actual.TotalLoss));
    }

    [Fact]
    public void DeepCopyPreservesAlignedStateWithoutSharingMutableParameters()
    {
        using var model = new MatchaTTS<float>(new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 4) { RandomSeed = 739 }, new MatchaTTSOptions
        {
            NumMels = 4, PhonemeVocabSize = 32, TextEncoderDim = 8, NumTextEncoderLayers = 1,
            NumTextEncoderHeads = 2, DecoderDim = 8, NumDecoderLayers = 1,
            DurationPredictorDim = 8, NumDurationPredictorLayers = 1, DropoutRate = 0
        });
        var batch = Batch();
        _ = model.Train(batch);
        var expected = model.EvaluateAlignedMel(batch);
        using var copy = Assert.IsAssignableFrom<AlignedTextToMelModelBase<float>>(model.DeepCopy());
        var actual = copy.EvaluateAlignedMel(batch);
        Assert.Equal(Values(expected.Output.MelSpectrogram), Values(actual.Output.MelSpectrogram));
        Assert.Equal(Values(expected.Output.LogDurations), Values(actual.Output.LogDurations));
        var original = model.GetParameters().ToArray();
        var changed = copy.GetParameters();
        changed[changed.Length - 1] += 1;
        copy.SetParameters(changed);
        Assert.Equal(original, model.GetParameters().ToArray());
        Assert.NotEqual(model.GetParameters()[changed.Length - 1], copy.GetParameters()[changed.Length - 1]);
    }

    public enum InvalidBatch { NoTokens, OutOfVocabulary, TooFewFrames, WrongChannels, ZeroDuration, WrongDurationSum, WrongDurationCount }

    [Theory]
    [InlineData(InvalidBatch.NoTokens)]
    [InlineData(InvalidBatch.OutOfVocabulary)]
    [InlineData(InvalidBatch.TooFewFrames)]
    [InlineData(InvalidBatch.WrongChannels)]
    [InlineData(InvalidBatch.ZeroDuration)]
    [InlineData(InvalidBatch.WrongDurationSum)]
    [InlineData(InvalidBatch.WrongDurationCount)]
    public void InvalidActiveGeometryIsRejectedBeforeTraining(InvalidBatch invalid)
    {
        using var model = CreateModel();
        var tokens = new Tensor<int>(new[] { 1, 2 });
        tokens[0, 0] = 1;
        tokens[0, 1] = invalid == InvalidBatch.OutOfVocabulary ? 32 : 2;
        int tokenLength = invalid == InvalidBatch.NoTokens ? 0 : 2;
        int melLength = invalid == InvalidBatch.TooFewFrames ? 1 : 3;
        var target = new Tensor<float>(new[] { 1, 3, invalid == InvalidBatch.WrongChannels ? 3 : 4 });
        int[] durations = invalid switch
        {
            InvalidBatch.ZeroDuration => new[] { 0, 3 },
            InvalidBatch.WrongDurationSum => new[] { 1, 1 },
            InvalidBatch.WrongDurationCount => new[] { 3 },
            _ => new[] { 1, 2 }
        };
        var batch = new AlignedMelBatch<float>(tokens, new[] { tokenLength }, target, new[] { melLength }, new[] { durations });
        var before = model.GetParameters().ToArray();
        bool initialTrainingMode = model.IsTrainingMode;
        Assert.ThrowsAny<ArgumentException>(() => model.Train(batch));
        Assert.Equal(before, model.GetParameters().ToArray());
        Assert.Equal(initialTrainingMode, model.IsTrainingMode);
    }

    [Fact]
    public void BatchOwnsLengthsAndDurationMetadata()
    {
        var source = Batch();
        var tokenLengths = source.TokenLengths.ToArray();
        var melLengths = source.MelLengths.ToArray();
        int[][] durations = { new[] { 2, 3 }, new[] { 1, 1, 1 } };
        var batch = new AlignedMelBatch<float>(source.Tokens, tokenLengths, source.TargetMels, melLengths, durations);
        tokenLengths[0] = 0;
        melLengths[0] = 0;
        durations[0][0] = 999;
        using var model = CreateModel();
        var result = model.EvaluateAlignedMel(batch);
        Assert.Equal(new[] { 5, 3 }, result.Output.MelLengths);
        Assert.Equal(2, result.Output.TokenDurations[0, 0]);
    }

    private static void SetConstantDuration(InspectableMatcha model, double logDuration)
    {
        var head = model.Extra<TokenDurationPredictorLayer<float>>();
        var parameters = BranchParameters(head);
        foreach (var parameter in parameters)
            for (int i = 0; i < parameter.Length; i++) parameter[i] = 0;
        var projection = head.GetSubLayers().OfType<Conv1DLayer<float>>().Last();
        var bias = projection.GetTrainableParameters().Single(parameter => parameter.Rank == 1);
        bias[0] = (float)logDuration;
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity)]
    public void SynthesisRejectsNonfiniteLearnedLogDurations(double value)
    {
        using var model = CreateModel();
        SetConstantDuration(model, value);
        var tokens = new Tensor<int>(new[] { 1, 1 });
        tokens[0] = 1;
        Assert.Throws<InvalidOperationException>(() => model.SynthesizeMel(tokens, new[] { 1 }));
    }

    [Fact]
    public void SpeakingRateScalingAvoidsOverflowForFiniteRepresentableDurations()
    {
        using var model = CreateModel();
        SetConstantDuration(model, 710);
        var tokens = new Tensor<int>(new[] { 1, 1 });
        tokens[0] = 1;
        var result = model.SynthesizeMel(tokens, new[] { 1 }, speakingRate: 1e308, maximumMelFrames: 4);
        Assert.Equal(new[] { 3 }, result.MelLengths);
        Assert.Equal(3, result.TokenDurations[0]);
    }

    [Fact]
    public void SynthesisEntersInferenceModeForAdditionalStochasticBranches()
    {
        using var model = new InspectableMatcha(dropout: 0.5);
        SetConstantDuration(model, 0);
        var durationHead = model.Extra<TokenDurationPredictorLayer<float>>();
        durationHead.GetSubLayers().OfType<LayerNormalizationLayer<float>>().Single().GetBetaTensor()[0] = 1;
        var projection = durationHead.GetSubLayers().OfType<Conv1DLayer<float>>().Last();
        projection.GetTrainableParameters().Single(parameter => parameter.Rank == 4)[0] = 0.25f;
        var tokens = new Tensor<int>(new[] { 1, 2 });
        tokens[0] = 1; tokens[1] = 2;
        model.SetTrainingMode(true);
        // A real active dropout branch yields only 0 or 0.5 in training, never the
        // inference oracle 0.25. Zeroing the complete predictor would not prove its mode.
        var trainingLogs = durationHead.Forward(new Tensor<float>(new[] { 1, 2, 8 }));
        Assert.All(Values(trainingLogs), value => Assert.Contains(value, new[] { 0f, 0.5f }));
        var first = model.SynthesizeMel(tokens, new[] { 2 });
        var second = model.SynthesizeMel(tokens, new[] { 2 });
        Assert.All(Values(first.LogDurations), value => Assert.Equal(0.25f, value));
        Assert.Equal(Values(first.MelSpectrogram), Values(second.MelSpectrogram));
        Assert.Equal(Values(first.LogDurations), Values(second.LogDurations));
        Assert.True(model.IsTrainingMode);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SynthesisRestoresCallerModeAfterSuccess(bool training)
    {
        using var model = CreateModel();
        SetConstantDuration(model, 0);
        model.SetTrainingMode(training);
        var tokens = new Tensor<int>(new[] { 1, 1 });
        tokens[0] = 1;
        _ = model.SynthesizeMel(tokens, new[] { 1 });
        Assert.Equal(training, model.IsTrainingMode);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SynthesisRestoresCallerModeAfterPredictorFailure(bool training)
    {
        using var model = CreateModel();
        SetConstantDuration(model, double.NaN);
        model.SetTrainingMode(training);
        var tokens = new Tensor<int>(new[] { 1, 1 });
        tokens[0] = 1;
        Assert.Throws<InvalidOperationException>(() => model.SynthesizeMel(tokens, new[] { 1 }));
        Assert.Equal(training, model.IsTrainingMode);
    }

    [Fact]
    public void CallerOwnedNativeHiFiGanProducesTheDeclaredWaveformLength()
    {
        using var model = new InspectableMatcha(hopLength: 4);
        SetConstantDuration(model, 0);
        using var vocoder = new AiDotNet.TextToSpeech.Vocoders.HiFiGAN<float>(
            new NeuralNetworkArchitecture<float>(inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.Regression, inputHeight: 4, inputWidth: 2,
                inputDepth: 1, outputSize: 8) { RandomSeed = 47 },
            new AiDotNet.TextToSpeech.Vocoders.HiFiGANOptions
            {
                MelChannels = 4, SampleRate = 22050, HopSize = 4, DropoutRate = 0,
                UpsampleInitialChannels = 16, UpsampleRates = new[] { 2, 2 },
                UpsampleKernelSizes = new[] { 4, 4 }, ResblockKernelSizes = new[] { 3 },
                ResblockDilationSizes = new[] { new[] { 1 } }
            });
        var tokens = new Tensor<int>(new[] { 1, 2 });
        tokens[0] = 1; tokens[1] = 2;
        var waveforms = model.SynthesizeWithVocoder(tokens, new[] { 2 }, vocoder);
        Assert.Single(waveforms);
        Assert.Equal(new[] { 8 }, waveforms[0].Shape.ToArray());
        AssertFinite(waveforms[0]);
        model.Dispose();
        var direct = vocoder.MelToWaveform(new Tensor<float>(new[] { 1, 4, 2 }));
        Assert.Equal(8, direct.Length); // The supplied vocoder still belongs to its caller.
    }

    [Fact]
    public void DisposalReleasesTheOwnedDurationChildrenAndRejectsFurtherAlignedCalls()
    {
        var model = CreateModel();
        var batch = Batch();
        _ = model.EvaluateAlignedMel(batch);
        var child = model.Extra<TokenDurationPredictorLayer<float>>().GetSubLayers().OfType<Conv1DLayer<float>>().First();
        model.Dispose();
        model.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.EvaluateAlignedMel(batch));
        Assert.Throws<ObjectDisposedException>(() => child.Forward(new Tensor<float>(new[] { 1, 8, 2 })));
    }

    private static void AssertBranchHasGradient(LayerBase<float> branch, IReadOnlyDictionary<Tensor<float>, Tensor<float>> gradients)
    {
        Assert.Contains(BranchParameters(branch), parameter => gradients.TryGetValue(parameter, out var gradient)
            && gradient is not null && Values(gradient).Any(value => Math.Abs(value) > 1e-7));
    }

    private static IEnumerable<Tensor<float>> BranchParameters(AiDotNet.Interfaces.ILayer<float> branch)
    {
        if (branch is AiDotNet.Interfaces.ITrainableLayer<float> trainable)
            foreach (var parameter in trainable.GetTrainableParameters()) yield return parameter;
        foreach (var child in branch.GetSubLayers())
            foreach (var parameter in BranchParameters(child)) yield return parameter;
    }

    private static AlignedMelBatch<float> Batch()
    {
        var tokens = new Tensor<int>(new[] { 2, 3 });
        tokens[0, 0] = 1; tokens[0, 1] = 2;
        tokens[1, 0] = 3; tokens[1, 1] = 4; tokens[1, 2] = 5;
        var targets = new Tensor<float>(new[] { 2, 5, 4 });
        for (int i = 0; i < targets.Length; i++) targets[i] = 0.1f + 0.02f * i;
        return new AlignedMelBatch<float>(tokens, new[] { 2, 3 }, targets, new[] { 5, 3 },
            new[] { new[] { 2, 3 }, new[] { 1, 1, 1 } });
    }

    private static InspectableMatcha CreateModel() => new();
    private static TValue[] Values<TValue>(Tensor<TValue> tensor) => tensor.AsSpan().ToArray();
    private static void AssertFinite(Tensor<float> tensor) => Assert.All(Values(tensor), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));

    private sealed class InspectableMatcha : MatchaTTS<float>
    {
        internal InspectableMatcha(double dropout = 0, int hopLength = 256) : base(new NeuralNetworkArchitecture<float>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 4) { RandomSeed = 739 },
            new MatchaTTSOptions
            {
                NumMels = 4, HopLength = hopLength, PhonemeVocabSize = 32, TextEncoderDim = 8, NumTextEncoderLayers = 1,
                NumTextEncoderHeads = 2, DecoderDim = 8, NumDecoderLayers = 1,
                DurationPredictorDim = 8, NumDurationPredictorLayers = 1, DropoutRate = dropout
            }) { }
        internal TLayer Extra<TLayer>() where TLayer : LayerBase<float> => GetExtraTrainableLayers().OfType<TLayer>().Single();
        internal IReadOnlyList<Tensor<float>> ParametersForTape() => CollectModelTrainableTensors();
        internal IEnumerable<Tensor<float>> DecoderParameters() => Layers.Skip(Layers.Count - 6)
            .OfType<LayerBase<float>>().SelectMany(layer => layer.GetTrainableParameters());
    }

    private sealed class ControlledAlignmentModel : AlignedTextToMelModelBase<float>
    {
        internal ControlledAlignmentModel() : base(new NeuralNetworkArchitecture<float>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 2))
        {
            Layers.Add(new FullyConnectedLayer<float>(2, 2, new IdentityActivation<float>()));
            Layers.Add(new FullyConnectedLayer<float>(2, 2, new IdentityActivation<float>()));
            ConfigureAlignedMelPath(8, 2, 2, 256, 4, 1, 0, 0, 1);
            foreach (var layer in Layers.OfType<FullyConnectedLayer<float>>()
                         .Concat(GetExtraTrainableLayers().OfType<FullyConnectedLayer<float>>()))
            {
                var weights = layer.GetTrainableParameters().Single(parameter => parameter.Rank == 2);
                var bias = layer.GetTrainableParameters().Single(parameter => parameter.Rank == 1);
                for (int i = 0; i < weights.Length; i++) weights[i] = 0;
                for (int i = 0; i < bias.Length; i++) bias[i] = 0;
                weights[0, 0] = 1; weights[1, 1] = 1;
            }
            var embedding = GetExtraTrainableLayers().OfType<EmbeddingLayer<float>>().Single();
            var table = embedding.GetTrainableParameters().Single();
            for (int i = 0; i < table.Length; i++) table[i] = 0;
            table[2, 0] = 2;
        }
        protected override void InitializeLayers() { }
        protected override Tensor<float> PostprocessOutput(Tensor<float> output) => output;
        public override AiDotNet.Models.ModelMetadata<float> GetModelMetadata() => new() { Name = nameof(ControlledAlignmentModel) };
    }
}
