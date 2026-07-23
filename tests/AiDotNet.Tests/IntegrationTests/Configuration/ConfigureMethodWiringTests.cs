using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Regression test for AiDotNet#1357 — <c>ConfigureAdversarialRobustness</c> was
/// stored but never consumed by any Build path, so the call had no observable
/// effect. The fix wires the stored configuration through to
/// <c>AiModelResult.AdversarialRobustnessOptions</c> via
/// <c>AttachAdversarialRobustness</c>.
///
/// <para>Tests assert THREE invariants the PR claims:</para>
/// <list type="number">
///   <item>Fluent API chaining (the returned builder is the same instance).</item>
///   <item>Configuration retention on the builder via the internal
///         <c>ConfiguredAdversarialRobustness</c> accessor — used by
///         <c>BuildAsync</c>'s <c>AttachAdversarialRobustness</c> path
///         and exposed via <c>InternalsVisibleTo</c> for tests.</item>
///   <item>Post-build propagation to <c>AiModelResult.AdversarialRobustness-
///         Options</c> — the core regression-test surface.</item>
/// </list>
/// </summary>
public class ConfigureMethodWiringTests
{
    private static (Matrix<double> x, Vector<double> y) BuildDataset(int rows = 20, int features = 3)
    {
        var rng = new System.Random(123);
        var xData = new double[rows, features];
        var yData = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < features; c++)
            {
                xData[r, c] = rng.NextDouble() * 2 - 1;
                sum += xData[r, c];
            }
            yData[r] = sum;
        }
        return (new Matrix<double>(xData), new Vector<double>(yData));
    }

    [Fact]
    public void ConfigureAdversarialRobustness_RetainsConfiguration_OnBuilder()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var configuration = AdversarialRobustnessConfiguration<double, Matrix<double>, Vector<double>>.BasicSafety();

        var returned = builder.ConfigureAdversarialRobustness(configuration);

        // 1. Fluent API still chains correctly.
        Assert.Same(builder, returned);

        // 2. Configuration is retained on the builder. Without this assertion
        //    the test only proved fluent chaining — a regression that dropped
        //    the field assignment would slip through (review #1361).
        Assert.NotNull(builder.ConfiguredAdversarialRobustness);
        Assert.Same(configuration, builder.ConfiguredAdversarialRobustness);
        Assert.True(builder.ConfiguredAdversarialRobustness.Enabled,
            "BasicSafety() must produce an Enabled=true configuration.");
    }

    [Fact]
    public void ConfigureAdversarialRobustness_DefaultArgument_StoresEnabledConfiguration()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        // null argument -> sensible default with Enabled=true (the documented contract).
        var returned = builder.ConfigureAdversarialRobustness(configuration: null);
        Assert.Same(builder, returned);

        // The documented null-arg contract: produce a default configuration
        // with Enabled=true (review #1361). The test name promised this; now
        // the assertions actually verify it.
        Assert.NotNull(builder.ConfiguredAdversarialRobustness);
        Assert.True(builder.ConfiguredAdversarialRobustness.Enabled,
            "ConfigureAdversarialRobustness(null) must default to Enabled=true per the documented contract.");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task ConfigureAdversarialRobustness_PropagatesToAiModelResult_OnBuildAsync()
    {
        // The core wiring under test: after BuildAsync, the returned
        // AiModelResult exposes the AdversarialRobustnessOptions that were
        // configured on the builder. Without this propagation the PR's
        // public-surface claim is unproven (review #1361 fix #3).
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);
        var configuration = AdversarialRobustnessConfiguration<double, Matrix<double>, Vector<double>>.BasicSafety();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureAdversarialRobustness(configuration)
            .BuildAsync();

        Assert.NotNull(result);
        Assert.NotNull(result.AdversarialRobustnessOptions);
        // The exact instance identity is up to AttachAdversarialRobustness'
        // wiring choice (it currently calls SetAdversarialRobustnessOptions
        // with the Options sub-object, not the full configuration). What
        // matters is that result observes the configuration's Options.
        Assert.Same(configuration.Options, result.AdversarialRobustnessOptions);
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task BuildAsync_WithoutConfigureAdversarialRobustness_LeavesOptionsNull()
    {
        // Sanity / negative test: without ConfigureAdversarialRobustness,
        // AiModelResult.AdversarialRobustnessOptions must NOT be populated
        // by BuildAsync.
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Null(result.AdversarialRobustnessOptions);
    }
}
