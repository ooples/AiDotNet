using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class ParsedEvolutionConfigurationTests
{
    [Fact]
    public void ParsedSettingsAreAppliedWithoutReloadingAndEvolutionKeepsItsSnapshot()
    {
        var config = YamlConfigLoader.LoadFromString("evolution:\n  runId: parsed-once\n  seed: 7\n  maxEvaluationAttempts: 4\n");
        var builder = AiModelBuilder<double, Matrix<double>, Vector<double>>.FromConfiguration(config);
        config.Evolution!.RunId = "changed";
        config.Evolution.MaxEvaluationAttempts = 999;
        var configured = ((IConfiguredView<double, Matrix<double>, Vector<double>>)builder).ConfiguredEvolution!;
        Assert.Equal("parsed-once", configured.RunId);
        Assert.Equal(4, configured.MaxEvaluationAttempts);
        Assert.Equal(7UL, configured.Seed);
    }

    [Fact]
    public void NullParsedConfigurationIsRejected() =>
        Assert.Throws<ArgumentNullException>(() => AiModelBuilder<double, Matrix<double>, Vector<double>>.FromConfiguration(null!));
}
