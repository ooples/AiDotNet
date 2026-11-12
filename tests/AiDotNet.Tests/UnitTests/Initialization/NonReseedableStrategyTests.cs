using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Initialization;

/// <summary>
/// A pinned layer seed must not clone or replace strategies that cannot be re-seeded: an external-weights strategy keeps
/// its per-layer cursor on the shared instance, and a custom <see cref="IInitializationStrategy{T}"/> keeps its own
/// distribution (#2294 review).
/// </summary>
public class NonReseedableStrategyTests
{
    // Counts weight loads on the shared instance and writes the load's index, as FromFile's weights_{index} cursor does.
    private sealed class ExternalStrategy : InitializationStrategyBase<double>
    {
        public ExternalStrategy() : base(null) { }
        public int Loads { get; private set; }
        public override bool IsLazy => false;
        public override bool LoadFromExternal => true;
        public override void InitializeWeights(Tensor<double> weights, int inputSize, int outputSize)
        {
            double index = Loads++;
            for (int i = 0; i < weights.Length; i++) weights.SetFlat(i, index);
        }
        public override void InitializeBiases(Tensor<double> biases) { }
    }

    private sealed class ConstantStrategy : IInitializationStrategy<double>
    {
        public bool IsLazy => false;
        public bool LoadFromExternal => false;
        public void InitializeWeights(Tensor<double> weights, int inputSize, int outputSize)
        {
            for (int i = 0; i < weights.Length; i++) weights.SetFlat(i, 0.25);
        }
        public void InitializeBiases(Tensor<double> biases) { }
    }

    private static DenseLayer<double> Seeded(IInitializationStrategy<double> strategy) =>
        new(3, (IActivationFunction<double>)new IdentityActivation<double>()) { InitializationStrategy = strategy, RandomSeed = 11 };

    private static Tensor<double> Weights(DenseLayer<double> layer)
    {
        layer.Forward(new Tensor<double>(new[] { 1, 4 }));
        return layer.GetWeights();
    }

    [Fact(Timeout = 60000)]
    public async Task ExternalStrategy_AdvancesItsSharedCursorUnderASeed()
    {
        await Task.Yield();
        var strategy = new ExternalStrategy();
        var first = Weights(Seeded(strategy));
        var second = Weights(Seeded(strategy));
        Assert.Equal(2, strategy.Loads);
        Assert.Equal(0.0, first.GetFlat(0));
        Assert.Equal(1.0, second.GetFlat(0));
    }

    [Fact(Timeout = 60000)]
    public async Task CustomInterfaceStrategy_IsUsedUnderASeed()
    {
        await Task.Yield();
        var weights = Weights(Seeded(new ConstantStrategy()));
        for (int i = 0; i < weights.Length; i++) Assert.Equal(0.25, weights.GetFlat(i));
    }
}
