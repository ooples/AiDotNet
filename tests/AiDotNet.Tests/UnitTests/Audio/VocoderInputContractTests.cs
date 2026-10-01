using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

public sealed class VocoderInputContractTests
{
    public VocoderInputContractTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void DeclaredWaveformModelNormalizesUnbatchedMelWithoutChangingValues()
    {
        using var model = new DeclaredVocoder(declaresWaveform: true);
        var input = Input(new[] { 4, 3 });
        var output = model.MelToWaveform(input);
        Assert.Equal(new[] { 1, 1, 3 }, output.Shape.ToArray());
        var seen = model.LastInput;
        Assert.NotNull(seen);
        if (seen is null) throw new Xunit.Sdk.XunitException("No actual forward was executed.");
        Assert.Equal(new[] { 1, 4, 3 }, seen.Shape.ToArray());
        Assert.Equal(input.AsSpan().ToArray(), seen.AsSpan().ToArray());
        for (int frame = 0; frame < 3; frame++)
            Assert.Equal(Enumerable.Range(0, 4).Sum(channel => input[channel, frame]), output[0, 0, frame]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void ExistingBatchedMelKeepsInputIdentityAndEveryBatch(int batchSize)
    {
        using var model = new DeclaredVocoder(declaresWaveform: true);
        var input = Input(new[] { batchSize, 4, 3 });
        var output = model.MelToWaveform(input);
        Assert.Same(input, model.LastInput);
        Assert.Equal(new[] { batchSize, 1, 3 }, output.Shape.ToArray());
        for (int batch = 0; batch < batchSize; batch++)
            for (int frame = 0; frame < 3; frame++)
                Assert.Equal(Enumerable.Range(0, 4).Sum(channel => input[batch, channel, frame]), output[batch, 0, frame]);
    }

    [Fact]
    public void UndeclaredStepBasedModelKeepsItsExistingRankTwoSemantics()
    {
        using var model = new DeclaredVocoder(declaresWaveform: false);
        var input = Input(new[] { 2, 5 });
        var output = model.MelToWaveform(input);
        Assert.Same(input, model.LastInput);
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.Equal(input.AsSpan().ToArray().Select(value => value * 2), output.AsSpan().ToArray());
    }

    [Theory]
    [InlineData(3, 2)]
    [InlineData(4, 0)]
    [InlineData(0, 2)]
    public void DeclaredMelChannelsAndNonemptyFramesAreValidatedBeforeForward(int channels, int frames)
    {
        using var model = new DeclaredVocoder(declaresWaveform: true);
        Assert.ThrowsAny<ArgumentException>(() => model.MelToWaveform(new Tensor<double>(new[] { channels, frames })));
        Assert.Null(model.LastInput);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NormalizationPreservesTheActualInputGradient(bool strided)
    {
        using var model = new DeclaredVocoder(declaresWaveform: true);
        var input = strided ? model.Transpose(Input(new[] { 3, 4 })) : Input(new[] { 4, 3 });
        var expected = input.ToArray().Select(value => value * 2).ToArray();
        using var tape = new GradientTape<double>();
        var normalized = model.Normalize(input);
        var loss = model.SquaredSum(normalized);
        var gradients = tape.ComputeGradients(loss, new[] { input });
        Assert.Equal(new[] { 1, 4, 3 }, normalized.Shape.ToArray());
        Assert.True(gradients.TryGetValue(input, out var gradient));
        if (gradient is null) throw new Xunit.Sdk.XunitException("Normalization disconnected the input gradient.");
        Assert.Equal(expected, gradient.ToArray());
    }

    private static Tensor<double> Input(int[] shape)
    {
        var input = new Tensor<double>(shape);
        for (int i = 0; i < input.Length; i++) input[i] = i + 0.5;
        return input;
    }

    private sealed class DeclaredVocoder : VocoderBase<double>
    {
        private readonly bool _declaresWaveform;
        internal Tensor<double>? LastInput { get; private set; }

        internal DeclaredVocoder(bool declaresWaveform) : base(new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 4, inputWidth: 3, inputDepth: 1, outputSize: 3))
        {
            _declaresWaveform = declaresWaveform;
            MelChannels = 4;
            HopSize = 1;
        }

        public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int rank)
            => _declaresWaveform ? WaveformUpsampleContract(rank) : null;
        public override Tensor<double> MelToWaveform(Tensor<double> melSpectrogram) => Predict(melSpectrogram);
        internal Tensor<double> Normalize(Tensor<double> input) => NormalizeMelPredictionInput(input);
        internal Tensor<double> Transpose(Tensor<double> input) => Engine.TensorTranspose(input);
        internal Tensor<double> SquaredSum(Tensor<double> input)
            => Engine.ReduceSum(Engine.TensorMultiply(input, input), new[] { 0, 1, 2 }, keepDims: false);
        protected override Tensor<double> PredictCore(Tensor<double> input)
        {
            LastInput = input;
            return _declaresWaveform ? Engine.ReduceSum(input, new[] { 1 }, keepDims: true)
                : Engine.TensorMultiplyScalar(input, 2);
        }
        protected override void InitializeLayers() { }
        protected override Tensor<double> PreprocessText(string text) => new(new[] { 1 });
        protected override Tensor<double> PostprocessAudio(Tensor<double> modelOutput) => modelOutput;
        public override AiDotNet.Models.ModelMetadata<double> GetModelMetadata() => new() { Name = nameof(DeclaredVocoder) };
    }
}
