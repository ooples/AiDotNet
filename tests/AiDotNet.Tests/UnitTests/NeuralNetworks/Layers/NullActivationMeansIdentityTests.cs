using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// A null activation means "no activation" on every dense-style layer.
/// </summary>
/// <remarks>
/// DenseLayer, GraphConvolutionalLayer and MultiHeadAttentionLayer read null as identity, but
/// FullyConnectedLayer and FeedForwardLayer read it as ReLU. Around fifty call sites pass an explicit
/// null meaning "none", so heads such as FTTransformerClassifier's clamped their logits at zero. The
/// layers now agree, and a caller that wants ReLU says so.
/// </remarks>
public class NullActivationMeansIdentityTests
{
    private static Tensor<double> NegativeInput()
    {
        var input = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < input.Length; i++) input[i] = -1.0 - i;
        return input;
    }

    /// <summary>
    /// Builds the layer twice, once with a null activation and once with an explicit IdentityActivation,
    /// gives both the same parameters and requires the same output on inputs whose pre-activations are all
    /// negative, which identity passes through and ReLU would zero.
    /// </summary>
    private static void AssertNullMatchesIdentity(System.Func<IActivationFunction<double>?, ILayer<double>> build)
    {
        var input = NegativeInput();
        var withNull = build(null);
        var withIdentity = build(new IdentityActivation<double>());
        _ = withNull.Forward(input);
        _ = withIdentity.Forward(input);
        // All-ones weights and biases on an all-negative input: every pre-activation is negative (row sums
        // of -1..-6 plus a bias of 1), so the comparison sees exactly the values ReLU would zero.
        var ones = withIdentity.GetParameters();
        for (int i = 0; i < ones.Length; i++) ones[i] = 1.0;
        withIdentity.SetParameters(ones);
        withNull.SetParameters(ones);

        var expected = withIdentity.Forward(input);
        var actual = withNull.Forward(input);

        bool anyNegative = false;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] < 0.0) anyNegative = true;
            Assert.Equal(expected[i], actual[i], 12);
        }
        Assert.True(anyNegative, "The comparison needs a negative pre-activation to tell identity from ReLU.");
    }

    [Fact]
    public void FullyConnectedLayer_NullActivation_IsIdentity()
        => AssertNullMatchesIdentity(activation => new FullyConnectedLayer<double>(3, activation));

    [Fact]
    public void FullyConnectedLayer_InputSizeConstructor_NullActivation_IsIdentity()
        => AssertNullMatchesIdentity(activation => new FullyConnectedLayer<double>(3, 3, activation));

    [Fact]
    public void FeedForwardLayer_NullActivation_IsIdentity()
        => AssertNullMatchesIdentity(activation => new FeedForwardLayer<double>(3, activation));

    [Fact]
    public void DenseLayer_NullActivation_IsIdentity()
        => AssertNullMatchesIdentity(activation => new DenseLayer<double>(3, activation));
    [Fact]
    public void FullyConnectedLayer_ExplicitReLU_StillClampsNegatives()
    {
        // The other half of the contract: a caller who asks for ReLU still gets it. All-ones parameters on
        // an all-negative input make every pre-activation negative, so ReLU must output exactly zero.
        var layer = new FullyConnectedLayer<double>(3, (IActivationFunction<double>)new ReLUActivation<double>());
        var input = NegativeInput();
        _ = layer.Forward(input);
        var parameters = layer.GetParameters();
        for (int i = 0; i < parameters.Length; i++) parameters[i] = 1.0;
        layer.SetParameters(parameters);

        var output = layer.Forward(input);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(0.0, output[i]);
    }
}
