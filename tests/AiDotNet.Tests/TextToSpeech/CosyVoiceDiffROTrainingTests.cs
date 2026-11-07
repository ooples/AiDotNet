using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class CosyVoiceDiffROTrainingTests
{
    [Fact]
    public void PublicDiffRO_UpdatesOnlyPolicyLm_AndRestoresStageAndFrozenModes()
    {
        using var policy = Policy();
        using var reference = Policy();
        var reward = new CosyVoiceToken2TextReward<double>(5, 151936, 4, 8, 1, 1, 1);
        reference.SetTrainingMode(true);
        foreach (var layer in reward.Layers) layer.SetTrainingMode(true);
        policy.CurrentStage = CosyVoiceTrainingStage.Flow;
        var policyParameters = Parameters(policy.LanguageModelLayers);
        var before = Snapshot(policyParameters);
        var fixedParameters = Parameters(reference.LanguageModelLayers)
            .Concat(reward.Parameters).Concat(Parameters(policy.FlowLayers))
            .Concat(Parameters(policy.VocoderLayers)).ToArray();
        var frozenBefore = Snapshot(fixedParameters);
        var checkedReward = new CheckedReward(reward, () =>
        {
            Assert.False(reference.IsTrainingMode);
            Assert.All(reward.Layers, layer => Assert.False(layer.InTrainingMode));
        });
        for (int i = 0; i < 2; i++)
        {
            double loss = policy.TrainDifferentiableReward(Sample(), reference, checkedReward, random: new Fixed(.37));
            Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
        }
        Assert.Equal(2, checkedReward.Calls);
        Assert.Contains(Enumerable.Range(0, before.Length), i => !before[i].SequenceEqual(policyParameters[i].ToArray()));
        for (int i = 0; i < fixedParameters.Length; i++) Assert.Equal(frozenBefore[i], fixedParameters[i].ToArray());
        Assert.Equal(CosyVoiceTrainingStage.Flow, policy.CurrentStage);
        Assert.True(reference.IsTrainingMode);
        Assert.All(reward.Layers, layer => Assert.True(layer.InTrainingMode));
    }

    [Fact]
    public void Evaluation_IsDeterministic_AndDoesNotConsumeThePolicyTrainingNoise()
    {
        using var a = Policy();
        using var b = Policy();
        using var reference = Policy();
        var reward = new CosyVoiceToken2TextReward<double>(5, 151936, 4, 8, 1, 1, 1);
        Assert.Equal(a.TrainDifferentiableReward(Sample(), reference, reward), b.TrainDifferentiableReward(Sample(), reference, reward));
        double first = a.EvaluateDifferentiableReward(Sample(), reference, reward);
        Assert.Equal(first, a.EvaluateDifferentiableReward(Sample(), reference, reward));
        Assert.Equal(a.TrainDifferentiableReward(Sample(), reference, reward), b.TrainDifferentiableReward(Sample(), reference, reward));
        var pa = Parameters(a.LanguageModelLayers);
        var pb = Parameters(b.LanguageModelLayers);
        Assert.Equal(pa.Count, pb.Count);
        for (int i = 0; i < pa.Count; i++) Assert.Equal(pa[i].ToArray(), pb[i].ToArray());
    }

    [Fact]
    public void RewardFailure_RestoresStageAndFrozenModes_WithoutUpdatingWeights()
    {
        using var policy = Policy();
        using var reference = Policy();
        var reward = new CosyVoiceToken2TextReward<double>(5, 151936, 4, 8, 1, 1, 1);
        foreach (var layer in reward.Layers) layer.SetTrainingMode(true);
        policy.CurrentStage = CosyVoiceTrainingStage.Vocoder;
        var parameters = Parameters(policy.LanguageModelLayers);
        var before = Snapshot(parameters);
        var failing = new CheckedReward(reward, () => throw new InvalidOperationException("Reward failure"));
        Assert.Throws<InvalidOperationException>(() => policy.TrainDifferentiableReward(Sample(), reference, failing));
        for (int i = 0; i < parameters.Count; i++) Assert.Equal(before[i], parameters[i].ToArray());
        Assert.Equal(CosyVoiceTrainingStage.Vocoder, policy.CurrentStage);
        Assert.True(reference.IsTrainingMode);
        Assert.All(reward.Layers, layer => Assert.True(layer.InTrainingMode));
    }

    [Fact]
    public void FrozenReward_CannotAliasThePolicyLanguageModelParameters()
    {
        using var policy = Policy();
        using var reference = Policy();
        Assert.Throws<ArgumentException>(() => policy.TrainDifferentiableReward(Sample(), reference,
            new AliasedReward(policy.LanguageModelLayers)));
    }

    private static CosyVoice3<double> Policy() => new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
        NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), new CosyVoice3Options
        {
            CodebookSize = 5, ReservedSpeechTokens = 4, LLMDim = 4, NumLLMLayers = 1, NumHeads = 1,
            HopSize = 8, FftSize = 32, CodecFrameRate = 1500,
            KeyValueHeads = 1, LmFeedForwardDim = 8, MelChannels = 4, SpeakerEmbeddingDim = 4,
            SpeakerEncoderChannels = 8, SpeakerEncoderGrowthRate = 2,
            DiTInputDim = 4, PreLookaheadChannels = 8, DiTDim = 16, DiTDepth = 1, DiTHeads = 2,
            DiTHeadDim = 8, DiTFeedForwardMultiplier = 2, DiTTimesteps = 2, StaticChunkSize = 2,
            HiftBaseChannels = 8, HiftHarmonics = 1, HiftUpsampleRates = new[] { 2 },
            HiftUpsampleKernels = new[] { 4 }, HiftSourceKernels = new[] { 3 }, DropoutRate = 0,
            IncludeDiscriminators = false
        }, new GradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new GradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
            { InitialLearningRate = .001, UseAdaptiveLearningRate = false, InitialMomentum = 0 }));

    private static TtsTrainingSample<double> Sample() => new()
    {
        Tokens = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 4.0, 7.0, 3.0 })),
        CodecTokens = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, 3.0, 0.0, 2.0 }))
    };

    private static IReadOnlyList<Tensor<double>> Parameters(IEnumerable<LayerBase<double>> layers) =>
        TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList(), -1);
    private static double[][] Snapshot(IEnumerable<Tensor<double>> values) => values.Select(tensor => tensor.ToArray()).ToArray();

    private sealed class Fixed : Random
    {
        private readonly double _value;
        internal Fixed(double value) => _value = value;
        public override double NextDouble() => _value;
    }

    private sealed class CheckedReward : ICosyVoiceTokenReward<double>
    {
        private readonly ICosyVoiceTokenReward<double> _inner;
        private readonly Action _check;
        internal int Calls { get; private set; }
        internal CheckedReward(ICosyVoiceTokenReward<double> inner, Action check) { _inner = inner; _check = check; }
        public int SpeechVocabularySize => _inner.SpeechVocabularySize;
        public IReadOnlyList<LayerBase<double>> Layers => _inner.Layers;
        public Tensor<double> LogReward(Tensor<double> softSpeechTokens, IReadOnlyList<int> textTokens)
        {
            Calls++;
            _check();
            return _inner.LogReward(softSpeechTokens, textTokens);
        }
    }

    private sealed class AliasedReward : ICosyVoiceTokenReward<double>
    {
        internal AliasedReward(IReadOnlyList<LayerBase<double>> layers) => Layers = layers;
        public int SpeechVocabularySize => 5;
        public IReadOnlyList<LayerBase<double>> Layers { get; }
        public Tensor<double> LogReward(Tensor<double> softSpeechTokens, IReadOnlyList<int> textTokens)
            => throw new InvalidOperationException("Aliased reward must be rejected before forward.");
    }
}
