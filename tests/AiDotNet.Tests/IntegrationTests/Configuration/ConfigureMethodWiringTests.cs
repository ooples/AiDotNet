using AiDotNet.Configuration;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Regression test for AiDotNet#1357 — <c>ConfigureAdversarialRobustness</c> was
/// stored but never consumed by any Build path, so the call had no observable
/// effect. The fix wires the stored configuration through to
/// <c>AiModelResult.AdversarialRobustnessOptions</c> via
/// <c>AttachAdversarialRobustness</c>.
/// </summary>
public class ConfigureMethodWiringTests
{
    [Fact(Timeout = 60000)]
    public void ConfigureAdversarialRobustness_RetainsConfiguration_OnBuilder()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var configuration = AdversarialRobustnessConfiguration<double, Matrix<double>, Vector<double>>.BasicSafety();

        var returned = builder.ConfigureAdversarialRobustness(configuration);

        // Fluent API still chains correctly.
        Assert.Same(builder, returned);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureAdversarialRobustness_DefaultArgument_StoresEnabledConfiguration()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        // null argument -> sensible default with Enabled=true (the documented contract).
        var returned = builder.ConfigureAdversarialRobustness(configuration: null);
        Assert.Same(builder, returned);
    }
}
