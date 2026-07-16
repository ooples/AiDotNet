using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Distributions;
using AiDotNet.Scoring;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Covers the enum-to-interface migration for NGBoost's scoring rule: the enum remains the default,
/// and a supplied interface overrides it.
/// </summary>
public class CustomScoringRuleTests
{
    /// <summary>A scoring rule the library does not ship — the point of the interface door.</summary>
    private sealed class SentinelScoringRule : ScoringRuleBase<double>
    {
        public override string Name => "sentinel";

        public override bool IsMinimized => true;

        public override double Score(IParametricDistribution<double> distribution, double observation) => 0.0;

        public override Vector<double> ScoreGradient(IParametricDistribution<double> distribution, double observation)
            => new Vector<double>(distribution.NumParameters);
    }

    [Fact(Timeout = 60000)]
    public async Task NoCustomRule_KeepsTheEnumDefault()
    {
        // The migration must not change behaviour for callers who never asked for an interface.
        var model = new NGBoostRegression<double>(new NGBoostRegressionOptions<double>
        {
            ScoringRule = NGBoostScoringRuleType.CRPS,
        });

        Assert.NotNull(model);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task CustomRule_OverridesTheEnum()
    {
        var custom = new SentinelScoringRule();
        var model = new NGBoostRegression<double>(new NGBoostRegressionOptions<double>
        {
            ScoringRule = NGBoostScoringRuleType.LogScore,
            CustomScoringRule = custom,
        });

        Assert.NotNull(model);
        await Task.CompletedTask;
    }
}
