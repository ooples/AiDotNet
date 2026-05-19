using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SmokeTests;

/// <summary>
/// Release-gate smoke tests — selected by the automated-release pipeline's
/// pre-publish gate via the filter
/// <c>Category=Smoke|FullyQualifiedName~SmokeTests</c>.
/// </summary>
/// <remarks>
/// <para>
/// The release pipeline runs these tests after build, before
/// <c>dotnet pack</c>. If any of them fails the package is not published.
/// Keep them small and fast: every test in this class should run in well
/// under a second so the release gate stays cheap. Cover thin slices of
/// the public surface that, if broken, would render most user code
/// non-functional:
/// </para>
/// <list type="bullet">
///   <item>A loss function constructs and returns sensible values for
///         trivial inputs (smoke-tests the LinearAlgebra surface +
///         loss-function virtuals — the bottom of the gradient stack).</item>
///   <item>An optimizer constructs against default options (smoke-tests
///         the optimizer-options binding the regularization PR #1381
///         touched + the parameter-bridge surface).</item>
/// </list>
/// <para>
/// Bigger integration / model-training tests live in the IntegrationTests
/// project and are filtered OUT of the release gate to keep it fast.
/// </para>
/// </remarks>
[Trait("Category", "Smoke")]
public class ReleaseSmokeTests
{
    [Fact]
    public void MeanSquaredErrorLoss_OnIdenticalVectors_IsZero()
    {
        // Smoke: loss-function surface (constructor + CalculateLoss) is
        // reachable and produces the expected scalar for the trivial case
        // of predicted == actual. Verifies the LinearAlgebra Vector<T>
        // path resolved correctly through the typical
        // AiDotNet.LossFunctions namespace — a broken framework binding
        // here would block every consumer.
        var loss = new MeanSquaredErrorLoss<float>();
        var values = new Vector<float>(new float[] { 0.5f, -0.25f, 1.0f, 0.0f });

        float result = loss.CalculateLoss(values, values);

        Assert.Equal(0.0f, result, precision: 6);
    }

    [Fact]
    public void AdamOptimizerOptions_DefaultConstruction_HasSensibleDefaults()
    {
        // Smoke: optimizer-options ctor is reachable and the defaults
        // PR #1381 hardened (gradient clipping + L2-strength default)
        // round-trip through to a constructed options instance. A broken
        // default-binding here would surface as a NullReferenceException
        // in every gradient-based optimizer the consumer wires up.
        var opts = new AiDotNet.Models.Options.AdamOptimizerOptions<float, Vector<float>, Vector<float>>();

        Assert.Equal(32, opts.BatchSize);
        Assert.Equal(0.001, opts.InitialLearningRate);
        Assert.Equal(0.9, opts.Beta1);
        Assert.Equal(0.999, opts.Beta2);
        Assert.True(opts.EnableGradientClipping,
            "AdamOptimizerOptions ctor should enable gradient clipping by default (PR #1364).");
        Assert.Equal(1.0, opts.MaxGradientNorm);
    }

    [Fact]
    public void AdamOptimizer_NullModelConstruction_RegistersOptionsAndDeferredModel()
    {
        // Smoke: the null-model ctor path AiModelBuilder uses (construct
        // optimizer first, SetModel later) does not throw at construction
        // time AND lands in the expected initial state. PR #1380
        // root-caused as an InvalidCastException; this asserts the
        // build-and-construct sequence stays exception-free AND that the
        // optimizer holds onto the supplied options + leaves Model null
        // for a subsequent SetModel call (the contract the AiModelBuilder
        // facade relies on).
        var opts = new AiDotNet.Models.Options.AdamOptimizerOptions<float, Vector<float>, Vector<float>>();
        var optimizer = new AdamOptimizer<float, Vector<float>, Vector<float>>(model: null, options: opts);

        Assert.NotNull(optimizer);
        Assert.Null(optimizer.Model);
        Assert.Same(opts, optimizer.GetOptions());
    }
}
