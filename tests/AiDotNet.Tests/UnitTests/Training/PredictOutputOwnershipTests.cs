using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// Predict skips its detach copy when the forward rented no host scratch (#2385). A forward that returns its input
/// unchanged (only an InputLayer) rents nothing either, and must still not hand the caller its own tensor back: the
/// caller would alias its input, and the streaming PredictInBatches path disposes each result after its callback.
/// Nor may it hand out a buffer a layer keeps and rewrites every call (DenseLayer's and FeedForwardLayer's
/// FusedLinearInto scratch): the caller's result would change under it on the next Predict.
/// </summary>
[Collection("NonParallelIntegration")]
public partial class PredictOutputOwnershipTests
{
    private static FeedForwardNeuralNetwork<float> PassThrough(bool reshape = false)
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 4,
            layers: reshape
                // The reshape returns a view of its input: an output distinct from the input that shares its storage.
                ? new List<ILayer<float>> { new InputLayer<float>(new[] { 4 }), new ReshapeLayer<float>(new[] { 2, 2 }) }
                : new List<ILayer<float>> { new InputLayer<float>(new[] { 4 }) });
        return new FeedForwardNeuralNetwork<float>(arch, lossFunction: new MeanSquaredErrorLoss<float>());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Predict_never_returns_the_callers_input_or_a_view_of_it(bool reshape)
    {
        // The detach this guards runs on the inference-arena path, so the arena is on whatever the environment says.
        var prior = AiDotNetEngine.Current;
        bool priorArena = InferenceArenaSettings.Enabled;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            InferenceArenaSettings.Enabled = true;
            var network = PassThrough(reshape);
            var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 2, 4 });
            network.Predict(input);   // warm: chain validation

            var output = network.Predict(input);
            Assert.NotSame(input, output);
            output[0] = 99f;
            Assert.Equal(1f, input[0]);

            // Neither writing to the result nor releasing it may reach the input's storage.
            output.Dispose();
            Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, input.ToArray());

            network.PredictInBatches(input, 2, (_, _) => { });
            Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, input.ToArray());
        }
        finally
        {
            InferenceArenaSettings.Enabled = priorArena;
            AiDotNetEngine.Current = prior;
        }
    }

    [Fact]
    public void Predict_never_returns_a_buffer_a_layer_rewrites_on_the_next_call()
    {
        var prior = AiDotNetEngine.Current;
        bool priorArena = InferenceArenaSettings.Enabled;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            InferenceArenaSettings.Enabled = true;
            var arch = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: Features,
                outputSize: Features,
                layers: new List<ILayer<float>> { new InputLayer<float>(new[] { Features }), new ResidentOutputLayer() });
            var network = new FeedForwardNeuralNetwork<float>(arch, lossFunction: new MeanSquaredErrorLoss<float>());
            var first = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 2, Features });
            var second = new Tensor<float>(new float[] { -1, -2, -3, -4, -5, -6, -7, -8 }, new[] { 2, Features });
            network.Predict(first);   // warm: chain validation

            var kept = network.Predict(first);
            var keptValues = kept.ToArray();
            Assert.Equal(new float[] { 2, 4, 6, 8, 10, 12, 14, 16 }, keptValues);

            // The layer writes this call's values into the same buffer it returned last time.
            var next = network.Predict(second);
            Assert.NotSame(kept, next);
            Assert.Equal(keptValues, kept.ToArray());
            Assert.Equal(new float[] { -2, -4, -6, -8, -10, -12, -14, -16 }, next.ToArray());
        }
        finally
        {
            InferenceArenaSettings.Enabled = priorArena;
            AiDotNetEngine.Current = prior;
        }
    }

    private const int Features = 4;

    /// <summary>
    /// Doubles its input into one buffer it keeps and overwrites on every call, renting nothing from the inference
    /// arena, as the FusedLinearInto scratch paths do.
    /// </summary>
    [TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    private sealed partial class ResidentOutputLayer : LayerBase<float>, IShapeContract
    {
        [Scratch]
        private Tensor<float>? _resident;

        public ResidentOutputLayer()
            : base([Features], [Features])
        {
        }

        public override long ParameterCount => 0;

        public override bool SupportsTraining => false;

        public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
            => inputRank == 1 ? [new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(Features))] : null;

        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            if (_resident is null || !_resident.Shape.ToArray().SequenceEqual(input.Shape.ToArray()))
                _resident = new Tensor<float>(input.Shape.ToArray());
            for (int i = 0; i < input.Length; i++) _resident[i] = 2f * input[i];
            return _resident;
        }

        public override void UpdateParameters(float learningRate) { }

        public override Vector<float> GetParameters() => new Vector<float>(0);

        public override void SetParameters(Vector<float> parameters) { }

        public override void ResetState() { }
    }
}
