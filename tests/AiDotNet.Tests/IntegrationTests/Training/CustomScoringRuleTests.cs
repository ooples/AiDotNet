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
    public async Task NullScoringRule_UsesTheIndustryStandardDefault()
    {
        // The parameter is nullable and defaults to the standard rule, so callers who do not care
        // are not forced to name one.
        var model = new NGBoostRegression<double>(new NGBoostRegressionOptions<double>());

        Assert.Equal("LogScore", model.GetModelMetadata().AdditionalInfo["ScoringRule"]);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task ShippedRule_IsUsedWhenSupplied()
    {
        var model = new NGBoostRegression<double>(new NGBoostRegressionOptions<double>
        {
            ScoringRule = new CRPSScore<double>(),
        });

        Assert.Equal("CRPS", model.GetModelMetadata().AdditionalInfo["ScoringRule"]);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task CustomRule_TheLibraryDoesNotShip_IsAccepted()
    {
        // The point of replacing the enum: a closed enum could only ever name the rules we ship.
        var model = new NGBoostRegression<double>(new NGBoostRegressionOptions<double>
        {
            ScoringRule = new SentinelScoringRule(),
        });

        Assert.Equal("sentinel", model.GetModelMetadata().AdditionalInfo["ScoringRule"]);
        await Task.CompletedTask;
    }
}
