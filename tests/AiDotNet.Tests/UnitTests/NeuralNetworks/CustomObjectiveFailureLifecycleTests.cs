using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class CustomObjectiveFailureLifecycleTests
{
    private readonly ITestOutputHelper _output;
    public CustomObjectiveFailureLifecycleTests(ITestOutputHelper output)
    {
        _output = output;
        TestModuleInitializer.EnsureInitialized();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PartialOptimizerFailureInvalidatesRealPackedInferenceWeights(bool retainBackingArray)
    {
        using var model = new CachedLinearNetwork();
        var input = Tensor<float>.CreateDefault(new[] { 16, 128 }, 1);
        var target = new Tensor<float>(new[] { 16, 128 });
        var optimizer = new PartialFailureSgd(model, retainBackingArray);
        float before = 0;
        for (int iteration = 0; iteration < 8; iteration++) before = model.Predict(input)[0];
        Assert.Equal(128, before);
        var error = Assert.Throws<InvalidOperationException>(() => model.Step(input, target, optimizer));
        Assert.Same(optimizer.Failure, error);
        Assert.Equal(5, model.Linear.Weights[0]);
        Assert.Equal(0, optimizer.BatchEnds);
        Assert.False(model.IsTrainingMode);

        float actual = model.Predict(input)[0];
        _output.WriteLine($"Real packed inference: before={before:R}; mutated live weight={model.Linear.Weights[0]:R}; expected=132; actual={actual:R}.");
        Assert.Equal(132, actual);

        optimizer.ThrowAfterPartialWrite = false;
        _ = model.Step(input, target, optimizer);
        Assert.Equal(1, optimizer.BatchEnds);
        Assert.False(model.IsTrainingMode);
    }

    private sealed class CachedLinearNetwork : NeuralNetworkBase<float>
    {
        internal CachedLinearLayer Linear => (CachedLinearLayer)Layers[0];
        internal CachedLinearNetwork() : base(new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 128, outputSize: 128), new MeanSquaredErrorLoss<float>()) { InitializeLayers(); }
        protected override void InitializeLayers() => Layers.Add(new CachedLinearLayer());
        public override AiDotNet.Models.ModelMetadata<float> GetModelMetadata() => new() { Name = nameof(CachedLinearNetwork) };
        internal float Step(Tensor<float> input, Tensor<float> target, PartialFailureSgd optimizer)
            => TrainWithCustomObjective(input, target, (current, expected)
                => new MeanSquaredErrorLoss<float>().ComputeTapeLoss(Linear.Forward(current), expected), optimizer);
    }

    private sealed class CachedLinearLayer : LayerBase<float>
    {
        internal Tensor<float> Weights { get; } = new(new[] { 128, 128 });
        internal CachedLinearLayer() : base(new[] { 128 }, new[] { 128 })
        {
            for (int i = 0; i < Weights.Length; i++) Weights[i] = 1;
            RegisterTrainableParameter(Weights, PersistentTensorRole.Weights);
        }
        public override bool SupportsTraining => true;
        protected override Tensor<float> ForwardTraced(Tensor<float> input) => Engine.TensorMatMul(input, Weights);
        public override void ResetState() { }
    }

    private sealed class PartialFailureSgd : StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>
    {
        internal InvalidOperationException Failure { get; } = new("Deliberate failure after the first live write.");
        internal bool ThrowAfterPartialWrite { get; set; } = true;
        internal int BatchEnds { get; private set; }
        private readonly float[]? _retainedWeights;
        internal PartialFailureSgd(CachedLinearNetwork model, bool retainBackingArray) : base(model)
        {
            if (retainBackingArray) _retainedWeights = model.Linear.Weights.GetDataArray();
        }
        protected override void StepCore(TapeStepContext<float> context)
        {
            if (ThrowAfterPartialWrite)
            {
                // The exception interrupts the update before its usual completion/version notification.
                // This is a real live parameter write, not a mocked cache result or optimizer callback.
                if (_retainedWeights is { } retained) retained[0] += 4;
                else context.Parameters[0].AsWritableSpan()[0] += 4;
                throw Failure;
            }
            base.StepCore(context);
        }
        public override void OnBatchEnd() { BatchEnds++; base.OnBatchEnd(); }
    }
}
