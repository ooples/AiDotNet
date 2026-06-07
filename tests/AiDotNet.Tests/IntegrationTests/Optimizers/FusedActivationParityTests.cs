using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// Numerical-parity gate for activations wired onto the fused inference path
/// (#1447). An activation is only safe to map to a Tensors
/// <see cref="FusedActivationType"/> if the fused kernel produces the same
/// values as AiDotNet's eager scalar activation — otherwise <c>MlpForward</c>
/// would silently compute a different function than the per-layer path.
///
/// <para><b>Methodology:</b> isolate the activation by running it through
/// <c>IEngine.FusedLinear(x, I, null, type)</c> with an identity weight matrix
/// (so the linear part is a no-op and only the fused activation kernel applies),
/// and compare element-wise against the eager <c>activation.Activate(x)</c> over
/// a range of inputs spanning negatives, zero and positives. Only activations
/// that match here get <c>IFusedActivation</c> wired; this test is the gate.</para>
/// </summary>
public class FusedActivationParityTests
{
    private const double Tol = 1e-4;
    private readonly ITestOutputHelper _output;
    public FusedActivationParityTests(ITestOutputHelper output) => _output = output;

    public static IEnumerable<object[]> NonParametricActivations()
    {
        yield return new object[] { "Sigmoid", new SigmoidActivation<float>(), FusedActivationType.Sigmoid };
        yield return new object[] { "Tanh", new TanhActivation<float>(), FusedActivationType.Tanh };
        yield return new object[] { "Mish", new MishActivation<float>(), FusedActivationType.Mish };
        yield return new object[] { "SELU", new SELUActivation<float>(), FusedActivationType.SELU };
        yield return new object[] { "Softplus", new SoftPlusActivation<float>(), FusedActivationType.Softplus };
        yield return new object[] { "SoftSign", new SoftSignActivation<float>(), FusedActivationType.SoftSign };
        yield return new object[] { "Sign", new SignActivation<float>(), FusedActivationType.Sign };
        yield return new object[] { "BentIdentity", new BentIdentityActivation<float>(), FusedActivationType.BentIdentity };
        yield return new object[] { "Gaussian", new GaussianActivation<float>(), FusedActivationType.Gaussian };
        yield return new object[] { "LiSHT", new LiSHTActivation<float>(), FusedActivationType.LiSHT };
        yield return new object[] { "SQRBF", new SQRBFActivation<float>(), FusedActivationType.SQRBF };
        yield return new object[] { "ReLU6", new ReLU6Activation<float>(), FusedActivationType.ReLU6 };
        yield return new object[] { "HardSwish", new HardSwishActivation<float>(), FusedActivationType.HardSwish };
        // HardSigmoid intentionally NOT wired: AiDotNet's HardSigmoidActivation
        // uses slope 0.2 (clamp(0.2x+0.5,0,1)) while the fused kernel uses the
        // PyTorch form (x/6+0.5) — the parity gate measured a 0.333 divergence,
        // so it stays on the eager path. Reconcile the formula before wiring.
    }

    [Theory]
    [MemberData(nameof(NonParametricActivations))]
    public void FusedActivation_MatchesEagerActivate(string name, ActivationFunctionBase<float> activation, FusedActivationType type)
    {
        var engine = new CpuEngine();
        const int n = 5, d = 9;

        // Inputs spanning negative / zero / positive (and a couple of larger
        // magnitudes) so saturating regions are exercised.
        var x = new Tensor<float>(new[] { n, d });
        var samples = new float[] { -4f, -2f, -1f, -0.3f, 0f, 0.3f, 1f, 2f, 4f };
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                x[i, j] = samples[j] + i * 0.05f;

        // Identity weights → FusedLinear computes activation(x · I) = activation(x).
        var id = new Tensor<float>(new[] { d, d });
        for (int j = 0; j < d; j++) id[j, j] = 1f;

        var fused = engine.FusedLinear(x, id, null, type);
        var eager = activation.Activate(x);

        Assert.Equal(eager.Length, fused.Length);
        double maxAbs = 0;
        var ef = eager.ToArray();
        var ff = fused.ToArray();
        for (int i = 0; i < ff.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs((double)ff[i] - (double)ef[i]));

        _output.WriteLine($"{name}: maxAbsDiff(fused vs eager) = {maxAbs:E3}");
        Assert.True(maxAbs <= Tol,
            $"{name}: fused FusedActivationType.{type} diverges from eager {name}Activation by {maxAbs:E3} (> {Tol:E1}). " +
            "Do NOT wire IFusedActivation for this activation until the kernel and eager form agree.");
    }
}
