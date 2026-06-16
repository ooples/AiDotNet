using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Repro + regression test for #1629 — <see cref="DenseLayer{T}"/>'s constructor
/// silently substituted <see cref="ReLUActivation{T}"/> when callers passed
/// <c>activationFunction: null</c> (or omitted the argument).
///
/// <para><b>User-visible symptom:</b> A regression model written as
/// <c>new DenseLayer&lt;double&gt;(outputSize: 1)</c> with a <c>// linear output for regression</c>
/// comment would silently get a ReLU output layer. Once training nudged the
/// pre-activation negative, ReLU clamped every test prediction to exact 0.0 —
/// the gradient through dead ReLU is also 0, so the model never recovered.
/// On the AiDotNet facade walkthrough Example 2 (Sales Forecasting, Regression,
/// MSE loss) this produced <c>pred=(0.000000, 0.000000, 0.000000, 0.000000, 0.000000)</c>
/// with RMSE 0.685, which was originally mis-diagnosed as "FusedLinearGpu
/// broken for FusedActivationType.None" before re-tracing showed the layer was
/// actually running ReLU, not None.</para>
///
/// <para><b>Fix:</b> change the default in
/// <c>DenseLayer(outputSize, activationFunction = null, ...)</c> from
/// <c>?? new ReLUActivation&lt;T&gt;()</c> to <c>?? new IdentityActivation&lt;T&gt;()</c>.
/// Matches Keras / PyTorch <c>nn.Linear</c> / TensorFlow <c>Dense</c> conventions.</para>
/// </summary>
public class Issue1629_DenseLayerDefaultActivationTests
{
    private readonly ITestOutputHelper _output;
    public Issue1629_DenseLayerDefaultActivationTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// A DenseLayer constructed with no <c>activationFunction</c> argument should default
    /// to Identity (no activation / linear), NOT ReLU. Pre-fix this test fails because the
    /// default is ReLU. Post-fix it passes.
    /// </summary>
    [Fact]
    public void DenseLayer_WithoutExplicitActivation_DefaultsToIdentityNotReLU()
    {
        var layer = new DenseLayer<double>(outputSize: 1);

        _output.WriteLine($"ScalarActivation type = {layer.ScalarActivation?.GetType().Name ?? "null"}");

        Assert.IsType<IdentityActivation<double>>(layer.ScalarActivation);

        // Spell the regression out — if someone reverts the default and CI catches this,
        // the failure message should explain why the default matters.
        Assert.False(layer.ScalarActivation is ReLUActivation<double>,
            "DenseLayer's default activation must NOT be ReLU. Callers writing " +
            "`new DenseLayer<T>(outputSize: N)` expect no activation (linear output), " +
            "matching Keras / PyTorch / TF conventions. Silent ReLU substitution causes " +
            "regression models with a Dense output head to lock to all-zero predictions " +
            "via dying ReLU (issue #1629).");
    }

    /// <summary>
    /// Passing <c>null</c> explicitly should behave identically to omitting the argument —
    /// both should resolve to Identity, not ReLU.
    /// </summary>
    [Fact]
    public void DenseLayer_WithExplicitNullActivation_AlsoDefaultsToIdentity()
    {
        var layer = new DenseLayer<double>(outputSize: 1, activationFunction: null);

        Assert.IsType<IdentityActivation<double>>(layer.ScalarActivation);
    }

    /// <summary>
    /// Existing explicit-ReLU callers must still get ReLU. Sanity check that the change
    /// only affected the default, not the explicit path.
    /// </summary>
    [Fact]
    public void DenseLayer_WithExplicitReLU_StillGetsReLU()
    {
        var layer = new DenseLayer<double>(outputSize: 1, activationFunction: new ReLUActivation<double>());

        Assert.IsType<ReLUActivation<double>>(layer.ScalarActivation);
    }
}
