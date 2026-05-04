using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.RadialBasisFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression coverage for issue #1213's lazy-ctor migration of the
/// remaining composite layers (ObliviousDecisionTree, RBF, CRF,
/// PrimaryCapsule, DigitCapsule, MoE, ExpertLayer). RBMLayer is covered
/// separately by <see cref="RBMLayerLazyCtorIssue1213Tests"/>.
///
/// Each test follows the same shape-resolve contract:
///   1. Construct via the architectural-only lazy ctor (no input-dim
///      args, just architectural params: capsule counts, output classes,
///      tree depth, etc.).
///   2. Assert <see cref="LayerBase{T}.IsShapeResolved"/> is false and
///      ParameterCount is 0 (no parameters allocated yet).
///   3. Run a Forward with a real input tensor.
///   4. Assert IsShapeResolved is now true, ParameterCount matches the
///      full materialized count, and the output shape matches the
///      architectural expectation.
/// </summary>
public class CompositeLayerLazyCtorIssue1213Tests
{
    [Fact]
    public void ObliviousDecisionTree_LazyCtor_ResolvesShape_OnFirstForward()
    {
        using var layer = new ObliviousDecisionTreeLayer<float>(depth: 3, outputDim: 2);

        Assert.False(layer.IsShapeResolved);
        Assert.Equal(-1, layer.GetInputShape()[0]);
        Assert.Equal(2, layer.GetOutputShape()[0]);
        Assert.Equal(0, layer.ParameterCount);

        var input = new Tensor<float>(new[] { 4, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)((i + 1) * 0.1);

        var output = layer.Forward(input);

        Assert.True(layer.IsShapeResolved);
        Assert.Equal(5, layer.GetInputShape()[0]);
        Assert.Equal(2, layer.GetOutputShape()[0]);
        // Expected: depth*inputDim + depth + numLeaves*outputDim
        //         = 3*5      +     3 + 8*2 = 34.
        Assert.Equal(3 * 5 + 3 + 8 * 2, layer.ParameterCount);
        Assert.Equal(2, output.Shape[1]);
    }

    [Fact]
    public void RBF_LazyCtor_ResolvesShape_OnFirstForward()
    {
        var rbf = new GaussianRBF<float>();
        using var layer = new RBFLayer<float>(outputSize: 4, rbf: rbf);

        Assert.False(layer.IsShapeResolved);
        Assert.Equal(-1, layer.GetInputShape()[0]);
        Assert.Equal(4, layer.GetOutputShape()[0]);
        // Pre-Forward: parameters not yet allocated. ParameterCount must
        // equal GetParameters().Length so the optimizer doesn't size its
        // bookkeeping for state that doesn't yet exist.
        Assert.Equal(0, layer.ParameterCount);
        Assert.Equal(0, layer.GetParameters().Length);

        var input = new Tensor<float>(new[] { 2, 6 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(i * 0.05);

        var output = layer.Forward(input);

        Assert.True(layer.IsShapeResolved);
        Assert.Equal(6, layer.GetInputShape()[0]);
        Assert.Equal(4, layer.GetOutputShape()[0]);
        // RBF parameters = numCenters * inputSize (centers) + numCenters (widths)
        Assert.Equal(4 * 6 + 4, layer.ParameterCount);
    }

    [Fact]
    public void CRF_LazyCtor_ResolvesShape_OnFirstForward()
    {
        using var layer = new ConditionalRandomFieldLayer<float>(numClasses: 5);

        Assert.False(layer.IsShapeResolved);
        Assert.Equal(-1, layer.GetInputShape()[0]);
        Assert.Equal(5, layer.GetInputShape()[1]);
        // CRF's parameters depend only on numClasses (architectural), so
        // they ARE allocated eagerly in the lazy ctor — only InputShape /
        // OutputShape are deferred. ParameterCount is the full numClasses-
        // based total before first forward.
        Assert.Equal(5 * 5 + 5 + 5, layer.ParameterCount);
        Assert.Equal(layer.ParameterCount, layer.GetParameters().Length);

        // CRF expects [seqLen, numClasses] per-batch input.
        var input = new Tensor<float>(new[] { 7, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(i * 0.01);

        var output = layer.Forward(input);

        Assert.True(layer.IsShapeResolved);
        Assert.Equal(7, layer.GetInputShape()[0]);
        Assert.Equal(5, layer.GetInputShape()[1]);
        // CRF parameters: numClasses^2 (transition) + numClasses (start) + numClasses (end)
        Assert.Equal(5 * 5 + 5 + 5, layer.ParameterCount);
    }

    [Fact]
    public void PrimaryCapsule_LazyCtor_ResolvesShape_OnFirstForward()
    {
        using var layer = new PrimaryCapsuleLayer<float>(
            capsuleChannels: 4,
            capsuleDimension: 2,
            kernelSize: 3,
            stride: 1);

        Assert.False(layer.IsShapeResolved);
        Assert.Equal(-1, layer.GetInputShape()[0]);
        Assert.Equal(4 * 2, layer.GetOutputShape()[0]);
        // Pre-Forward: conv weights / bias not yet allocated.
        Assert.Equal(0, layer.ParameterCount);
        Assert.Equal(0, layer.GetParameters().Length);

        // NCHW input: [batch, channels, h, w]
        var input = new Tensor<float>(new[] { 1, 3, 5, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(i * 0.001);

        var output = layer.Forward(input);

        Assert.True(layer.IsShapeResolved);
        Assert.Equal(3, layer.GetInputShape()[0]);
        // PrimaryCapsule conv: outputChannels * (inputChannels * kernelSize^2) + outputChannels
        Assert.Equal((4 * 2) * (3 * 3 * 3) + (4 * 2), layer.ParameterCount);
    }

    [Fact]
    public void DigitCapsule_LazyCtor_ResolvesShape_OnFirstForward()
    {
        using var layer = new DigitCapsuleLayer<float>(
            numClasses: 10,
            outputCapsuleDimension: 4,
            routingIterations: 2);

        Assert.False(layer.IsShapeResolved);
        Assert.Equal(-1, layer.GetInputShape()[0]);
        Assert.Equal(-1, layer.GetInputShape()[1]);
        // Pre-Forward: routing weights not yet allocated.
        Assert.Equal(0, layer.ParameterCount);
        Assert.Equal(0, layer.GetParameters().Length);

        // Input: [batch, inputCapsules, inputCapsuleDimension]
        var input = new Tensor<float>(new[] { 1, 6, 8 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(i * 0.001);

        var output = layer.Forward(input);

        Assert.True(layer.IsShapeResolved);
        Assert.Equal(6, layer.GetInputShape()[0]);
        Assert.Equal(8, layer.GetInputShape()[1]);
        // DigitCapsule weight: inputCapsules * numClasses * inputCapsuleDimension * outputCapsuleDimension
        Assert.Equal(6 * 10 * 8 * 4, layer.ParameterCount);
    }

    [Fact]
    public void ExpertLayer_LazyInnerLayers_ResolveOnFirstForward()
    {
        // ExpertLayer with lazy DenseLayer children — caller passes
        // [-1] as inputShape so chain-resolve in the ctor is skipped
        // and inner layers stay unresolved until first Forward.
        var dense1 = new DenseLayer<float>(outputSize: 16, activationFunction: new IdentityActivation<float>());
        var dense2 = new DenseLayer<float>(outputSize: 8, activationFunction: new IdentityActivation<float>());
        using var expert = new ExpertLayer<float>(
            new System.Collections.Generic.List<ILayer<float>> { dense1, dense2 },
            inputShape: new[] { -1 },
            outputShape: new[] { 8 });

        Assert.False(expert.IsShapeResolved);
        // Pre-Forward: inner Dense layers are unresolved, so each
        // reports ParameterCount=0; the outer Expert sums to 0.
        Assert.Equal(0, expert.ParameterCount);

        var input = new Tensor<float>(new[] { 2, 32 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(i * 0.01);

        var output = expert.Forward(input);

        Assert.True(expert.IsShapeResolved);
        Assert.Equal(32, expert.GetInputShape()[0]);
        Assert.Equal(8, expert.GetOutputShape()[0]);
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(8, output.Shape[1]);

        // Verify parameters were materialized post-forward, complementing
        // the pre-forward `ParameterCount == 0` assertion above. Without
        // this check the test would still pass even if lazy resolution
        // silently failed to allocate weights.
        // dense1: 32→16 (32*16 + 16 bias = 528). dense2: 16→8 (16*8 + 8 bias = 136).
        int expectedParams = (32 * 16 + 16) + (16 * 8 + 8);
        Assert.Equal(expectedParams, expert.ParameterCount);
    }

    [Fact]
    public void MoE_LazyCtor_ResolvesSubLayerShapes_OnFirstForward()
    {
        // Two trivial expert sub-layers, plus a router. All Dense, all
        // lazy. Caller passes [-1] as inputShape so the MoE ctor's
        // eager-resolve gate skips and sub-layers remain unresolved.
        var expert1 = new DenseLayer<float>(outputSize: 8, activationFunction: new IdentityActivation<float>());
        var expert2 = new DenseLayer<float>(outputSize: 8, activationFunction: new IdentityActivation<float>());
        var router = new DenseLayer<float>(outputSize: 2, activationFunction: new IdentityActivation<float>());

        using var moe = new MixtureOfExpertsLayer<float>(
            experts: new System.Collections.Generic.List<ILayer<float>> { expert1, expert2 },
            router: router,
            inputShape: new[] { -1 },
            outputShape: new[] { 8 });

        Assert.False(moe.IsShapeResolved);
        // Pre-Forward: router and experts unresolved; outer MoE sums to 0.
        Assert.Equal(0, moe.ParameterCount);

        // Vanilla MoE input: [batch, features].
        var input = new Tensor<float>(new[] { 1, 12 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(i * 0.01);

        var output = moe.Forward(input);

        Assert.True(moe.IsShapeResolved);
        Assert.Equal(12, moe.GetInputShape()[0]);
        Assert.Equal(1, output.Shape[0]);
        // Confirm the output's feature dim matches what the resolved
        // experts produce (8), not just any rank-2 tensor — the layer's
        // shape contract guarantees [batch, outputFeatures].
        Assert.Equal(8, output.Shape[1]);

        // Verify parameters were materialized post-forward. Pre-forward
        // ParameterCount==0 only confirms lazy resolution was deferred;
        // a positive count after the first Forward proves the chained
        // resolution actually allocated weights for router + experts.
        // expert1/2: 12→8 (12*8 + 8 bias = 104 each). router: 12→2 (12*2 + 2 = 26).
        int expectedParams = 2 * (12 * 8 + 8) + (12 * 2 + 2);
        Assert.Equal(expectedParams, moe.ParameterCount);
    }
}
