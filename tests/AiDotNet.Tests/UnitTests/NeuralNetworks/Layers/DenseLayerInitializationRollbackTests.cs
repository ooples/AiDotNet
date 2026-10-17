using System;
using AiDotNet.Initialization;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// A <see cref="DenseLayer{T}"/>'s lazy initialization is all or nothing (#2087). A failure part-way used to leave the
/// allocated tensors behind with the layer still uninitialized, and the retry read them as restored parameters: a full
/// but unfilled set was adopted as trained weights, and a half set (the bias allocation running out of memory) threw
/// "parameters do not conform to the resolved shape" instead of the original error.
/// </summary>
public class DenseLayerInitializationRollbackTests
{
    /// <summary>
    /// Every fill must start from a freshly allocated, all-zero tensor. A tensor left over from the failed attempt
    /// (filled weights, sentinel biases) fails here, before the fill would overwrite the evidence.
    /// </summary>
    private static void AssertFresh(Tensor<double> tensor, string name)
    {
        for (int i = 0; i < tensor.Length; i++)
            Assert.True(tensor.GetFlat(i) == 0.0, $"{name}[{i}] = {tensor.GetFlat(i)} survived the failed initialization");
    }

    /// <summary>Throws on its first weight fill, then fills every weight with <see cref="Fill"/>.</summary>
    private sealed class FailsOnceStrategy : IInitializationStrategy<double>
    {
        public const double Fill = 0.25;
        private bool _failed;

        public bool IsLazy => false;
        public bool LoadFromExternal => false;

        public void InitializeWeights(Tensor<double> weights, int inputSize, int outputSize)
        {
            if (!_failed)
            {
                _failed = true;
                throw new InvalidOperationException("initializer failed");
            }
            AssertFresh(weights, "weights");
            for (int i = 0; i < weights.Length; i++) weights.SetFlat(i, Fill);
        }

        public void InitializeBiases(Tensor<double> biases)
        {
            AssertFresh(biases, "biases");
            for (int i = 0; i < biases.Length; i++) biases.SetFlat(i, 0.0);
        }
    }

    [Fact]
    public void FailedInitialization_IsRetriedFromScratch_NotAdoptedAsRestoredWeights()
    {
        var layer = new DenseLayer<double>(3, initializationStrategy: new FailsOnceStrategy());
        var input = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, 1.0);

        var first = Assert.Throws<InvalidOperationException>(() => layer.Forward(input));
        Assert.Equal("initializer failed", first.Message);

        // The retry initializes the layer: every output is the sum of four inputs times the filled weight.
        var output = layer.Forward(input);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(4 * FailsOnceStrategy.Fill, output.GetFlat(i), 12);
    }

    /// <summary>Fills every weight with <see cref="FailsOnceStrategy.Fill"/>; its first bias fill writes a sentinel and
    /// then throws, every later one zeroes the biases.</summary>
    private sealed class BiasFailsOnceStrategy : IInitializationStrategy<double>
    {
        public const double Sentinel = 99.0;
        private bool _failed;

        public bool IsLazy => false;
        public bool LoadFromExternal => false;

        public void InitializeWeights(Tensor<double> weights, int inputSize, int outputSize)
        {
            AssertFresh(weights, "weights");
            for (int i = 0; i < weights.Length; i++) weights.SetFlat(i, FailsOnceStrategy.Fill);
        }

        public void InitializeBiases(Tensor<double> biases)
        {
            if (!_failed)
            {
                _failed = true;
                for (int i = 0; i < biases.Length; i++) biases.SetFlat(i, Sentinel);
                throw new InvalidOperationException("bias initializer failed");
            }
            AssertFresh(biases, "biases");
            for (int i = 0; i < biases.Length; i++) biases.SetFlat(i, 0.0);
        }
    }

    /// <summary>
    /// A failure in the bias fill, after the weights are filled, rolls back BOTH tensors: the retry neither adopts the
    /// filled weights as restored parameters nor keeps the sentinel biases, and computes 4 x 0.25 + 0 = 1.
    /// </summary>
    [Fact]
    public void FailedBiasInitialization_RollsBackBothTensors()
    {
        var layer = new DenseLayer<double>(3, initializationStrategy: new BiasFailsOnceStrategy());
        var input = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, 1.0);

        var first = Assert.Throws<InvalidOperationException>(() => layer.Forward(input));
        Assert.Equal("bias initializer failed", first.Message);

        var output = layer.Forward(input);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(1.0, output.GetFlat(i), 12);
    }
}