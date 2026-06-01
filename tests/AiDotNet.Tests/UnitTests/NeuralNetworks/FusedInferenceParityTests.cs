using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Parity tests for the fused inference path. The fused activation kernel routed
/// through IEngine.MlpForward must be numerically identical to the scalar
/// activation it claims to implement (the IFusedActivation contract), and the
/// FeedForwardNeuralNetwork.Predict fast path must not change inference results
/// or crash on a freshly-constructed (lazily-initialized) network.
/// </summary>
public class FusedInferenceParityTests
{
    public static IEnumerable<object[]> FusedActivations() => new List<object[]>
    {
        new object[] { "ReLU", new ReLUActivation<float>() },
        new object[] { "Sigmoid", new SigmoidActivation<float>() },
        new object[] { "Tanh", new TanhActivation<float>() },
        new object[] { "Identity", new IdentityActivation<float>() },
        new object[] { "GELU", new GELUActivation<float>() },
        new object[] { "Swish", new SwishActivation<float>() },
        new object[] { "SiLU", new SiLUActivation<float>() },
        new object[] { "LeakyReLU", new LeakyReLUActivation<float>() },
        new object[] { "Mish", new MishActivation<float>() },
        // ELU is intentionally excluded: the FusedLinear/MlpForward path still has
        // no ELU kernel (see Elu_ReportsNoFusedKernel below). Mish's kernel shipped
        // in Tensors #499 (0.90.0+), so it is now parity-checked above.
    };

    /// <summary>
    /// The fused kernel for each activation's declared FusedActivationType must
    /// equal applying that activation's own scalar Activate() to the same linear
    /// pre-activation. Both sides share the identical x·W (computed via MlpForward
    /// with no activation), so this isolates the activation formula.
    /// </summary>
    [Theory]
    [MemberData(nameof(FusedActivations))]
    public void FusedActivationKernel_MatchesScalarActivation(string name, IActivationFunction<float> activation)
    {
        AiDotNetEngine.ResetToCpu();
        var engine = AiDotNetEngine.Current;

        Assert.True(((AiDotNet.ActivationFunctions.Fused.IFusedActivation)activation)
            .TryGetFusedActivation(out var fusedType), $"{name} must declare a fused activation type");

        const int batch = 4, inF = 16, outF = 8;
        var rng = new Random(20260529);
        var wData = new float[inF * outF];
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var w = new Tensor<float>(wData, new[] { inF, outF });
        var xData = new float[batch * inF];
        for (int i = 0; i < xData.Length; i++) xData[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        var x = new Tensor<float>(xData, new[] { batch, inF });

        var weights = new List<Tensor<float>> { w };
        var noBias = new List<Tensor<float>?> { null };

        // Same linear pre-activation for both sides.
        var rawLinear = engine.MlpForward(x, weights, noBias,
            FusedActivationType.None, FusedActivationType.None);
        var eager = activation.Activate(rawLinear);                 // scalar activation
        var fused = engine.MlpForward(x, weights, noBias,
            FusedActivationType.None, fusedType);                   // fused-kernel activation

        Assert.Equal(eager.Length, fused.Length);
        for (int i = 0; i < eager.Length; i++)
        {
            double e = Convert.ToDouble(eager[i]);
            double f = Convert.ToDouble(fused[i]);
            Assert.True(Math.Abs(e - f) < 1e-4,
                $"{name}: fused kernel {f} != scalar Activate {e} at index {i}");
        }
    }

    /// <summary>
    /// A parametric activation whose parameter differs from the fused kernel's
    /// hardcoded value must report NO fused equivalent, so it stays on the exact
    /// generic path rather than silently getting the kernel's default parameter.
    /// </summary>
    [Fact]
    public void CustomParamLeakyReLU_DoesNotClaimFusedKernel()
    {
        var custom = new LeakyReLUActivation<float>(alpha: 0.25); // kernel hardcodes 0.01
        Assert.False(((AiDotNet.ActivationFunctions.Fused.IFusedActivation)custom)
            .TryGetFusedActivation(out _), "custom-slope LeakyReLU must not claim a fused kernel");

        var standard = new LeakyReLUActivation<float>(); // default 0.01 == kernel
        Assert.True(((AiDotNet.ActivationFunctions.Fused.IFusedActivation)standard)
            .TryGetFusedActivation(out var t) && t == FusedActivationType.LeakyReLU);
    }

    /// <summary>
    /// ELU must NOT advertise a fused kernel: the FusedLinear/MlpForward activation tables have no
    /// ELU kernel, so routing it through the fused path would throw. It stays on the exact generic
    /// path until a kernel is added. (Mish's kernel shipped in Tensors #499 / 0.90.0+, so Mish now
    /// DOES advertise a fused kernel and is parity-checked in
    /// <see cref="FusedActivationKernel_MatchesScalarActivation"/>.) This locks the contract so ELU
    /// isn't silently re-wired before its kernel exists.
    /// </summary>
    [Fact]
    public void Elu_ReportsNoFusedKernel()
    {
        Assert.False(new ELUActivation<float>() is AiDotNet.ActivationFunctions.Fused.IFusedActivation,
            "ELU must not claim a fused kernel — the FusedLinear path has none");
        // Mish, by contrast, now legitimately advertises a fused kernel (Tensors #499).
        Assert.True(new MishActivation<float>() is AiDotNet.ActivationFunctions.Fused.IFusedActivation,
            "Mish should claim a fused kernel — the FusedLinear Mish kernel shipped in Tensors #499 (0.90.0+)");
    }

    /// <summary>
    /// Predict on a freshly-constructed (lazily-initialized) network must not
    /// throw — the fused fast path bails to the generic Forward when weights
    /// aren't materialized yet, then engages on subsequent calls. Both calls must
    /// produce the same result.
    /// </summary>
    [Fact]
    public void Predict_FreshLazyNetwork_DoesNotThrow_AndIsStable()
    {
        AiDotNetEngine.ResetToCpu();
        const int inF = 16, batch = 4;
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inF, outputSize: 4,
            layers: new List<ILayer<float>>
            {
                new DenseLayer<float>(8, (IActivationFunction<float>)new ReLUActivation<float>()),
                new DenseLayer<float>(4, (IActivationFunction<float>?)null),
            });
        var net = new FeedForwardNeuralNetwork<float>(arch);

        var rng = new Random(7);
        var data = new float[batch * inF];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var input = new Tensor<float>(data, new[] { batch, inF });

        var first = net.Predict(input);   // lazy weights ⇒ fused path bails to Forward
        var second = net.Predict(input);  // weights now materialized ⇒ fused path engages

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.True(Math.Abs(Convert.ToDouble(first[i]) - Convert.ToDouble(second[i])) < 1e-4,
                $"fused vs fallback mismatch at {i}: {first[i]} vs {second[i]}");
    }
}
