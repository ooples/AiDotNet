using System;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// End-to-end proof that a layer clones correctly, on DenseLayer.
/// </summary>
/// <remarks>
/// <para>
/// This is the first layer cloned through the generated construction state rather than a
/// hand-written implementation, so it is checked property by property rather than assumed from the
/// fact that it compiles.
/// </para>
/// <para>
/// DenseLayer is a deliberate first case rather than a convenient one: it is lazily initialized
/// (its input width is resolved on the first forward pass, not in the constructor), and it declares
/// two constructors — one taking a scalar activation, one a vector activation. Both are exactly the
/// situations where a reconstruction that merely compiles would produce a subtly wrong layer.
/// </para>
/// </remarks>
public class DenseLayerCloneTests
{
    /// <summary>
    /// A trained layer's clone must produce identical output and share no parameter storage.
    /// </summary>
    /// <returns>A task representing the test.</returns>
    [Fact(Timeout = 120000)]
    public async Task TrainedLayer_Clone_MatchesOutputAndIsIndependent()
    {
        await Task.Yield();

        var original = new DenseLayer<double>(4, new ReLUActivation<double>() as IActivationFunction<double>);

        // Force lazy initialization: the input width is resolved on first forward, so a layer
        // cloned before this point and one cloned after are genuinely different situations.
        var input = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.25;
        var expected = original.Forward(input);

        // Move the parameters away from their initialized values, so carrying them is observable.
        var trained = original.GetParameters();

        // Without this the loops below are empty and every assertion holds vacuously -- a pass
        // that proves the clone carried nothing just as convincingly as one that proves it carried
        // everything. 4 outputs by 3 inputs plus 4 biases.
        Assert.True(trained.Length >= 16, $"expected a trained parameter vector, got {trained.Length}");

        for (int i = 0; i < trained.Length; i++) trained[i] += 0.125;
        original.UpdateParameters(trained);
        expected = original.Forward(input);

        var clone = (LayerBase<double>)original.Clone();

        Assert.NotSame(original, clone);
        Assert.Equal(original.GetType(), clone.GetType());
        Assert.Equal(original.ParameterCount, clone.ParameterCount);

        // The strongest statement available: same input, same output. This covers the learned
        // parameters and every constructor-derived structure at once, which a property-by-property
        // comparison would not.
        var actual = clone.Forward(input);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                Math.Abs(expected[i] - actual[i]) < 1e-12,
                $"output[{i}]: original {expected[i]:G17}, clone {actual[i]:G17}");
        }

        // Independence: training the clone must not move the original. Equal parameters that
        // change together is the bug a value comparison cannot see.
        var cloneParams = clone.GetParameters();
        for (int i = 0; i < cloneParams.Length; i++) cloneParams[i] += 1.0;
        clone.UpdateParameters(cloneParams);

        var originalAfter = original.GetParameters();
        for (int i = 0; i < originalAfter.Length; i++)
        {
            Assert.True(
                Math.Abs(originalAfter[i] - trained[i]) < 1e-12,
                $"parameter[{i}] moved when the CLONE was trained: {trained[i]:G17} -> {originalAfter[i]:G17}");
        }
    }

    /// <summary>
    /// Architecture-only cloning reproduces the shape without carrying what was learned.
    /// </summary>
    /// <returns>A task representing the test.</returns>
    /// <remarks>
    /// Matches scikit-learn's <c>clone()</c>, which returns an unfitted estimator with the same
    /// hyperparameters. The assertion is that the parameters DIFFER — a copy that carried them
    /// anyway would pass a shape check while ignoring the option entirely.
    /// </remarks>
    [Fact(Timeout = 120000)]
    public async Task ArchitectureClone_ReproducesShape_WithoutCarryingLearnedValues()
    {
        await Task.Yield();

        var original = new DenseLayer<double>(4, new ReLUActivation<double>() as IActivationFunction<double>);

        var input = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.25;
        original.Forward(input);

        var trained = original.GetParameters();
        for (int i = 0; i < trained.Length; i++) trained[i] = 0.75;
        original.UpdateParameters(trained);

        var fresh = (LayerBase<double>)original.Clone(CloneOptions.Architecture);

        Assert.Equal(original.GetType(), fresh.GetType());

        // A fresh layer is lazily initialized, so it has no parameters until it sees input.
        fresh.Forward(input);
        Assert.Equal(original.ParameterCount, fresh.ParameterCount);

        var freshParams = fresh.GetParameters();
        bool anyDiffer = false;
        for (int i = 0; i < freshParams.Length; i++)
        {
            if (Math.Abs(freshParams[i] - 0.75) > 1e-12) { anyDiffer = true; break; }
        }

        Assert.True(anyDiffer, "Architecture clone carried the learned parameters; it should be freshly initialized.");
    }
}
