using System;
using AiDotNet.CausalInference;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.CausalInference;

/// <summary>
/// Pins that a seeded causal estimator reproduces its standard errors.
/// </summary>
/// <remarks>
/// <para>
/// Every standard error these models report comes from bootstrap resampling, and the resampler used to
/// build its own generator from <c>RandomHelper.CreateSecureRandom()</c> — ignoring the seed two of the
/// models already accepted, and offering none to the other five. Asking one fitted model for the same
/// standard error twice returned two different numbers, so a published confidence interval could not be
/// reproduced by anyone, including its author.
/// </para>
/// <para>
/// The estimate itself was never affected: it is computed directly from the data. Only the interval
/// around it moved, which is the part that looks stable and is not.
/// </para>
/// </remarks>
public class BootstrapStandardErrorSeedTests
{
    private static (Matrix<double> Covariates, Vector<int> Treatment, Vector<double> Outcome) Sample()
    {
        double[,] rows =
        {
            { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 },
            { 41, 1 }, { 58, 0 }, { 36, 1 }, { 63, 0 }
        };
        int[] treatment = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
        double[] outcome = { 3.1, 7.2, 2.8, 8.0, 3.4, 7.6, 3.0, 7.9, 2.6, 8.2 };

        var covariates = new Matrix<double>(rows.GetLength(0), rows.GetLength(1));
        for (int i = 0; i < rows.GetLength(0); i++)
        {
            for (int j = 0; j < rows.GetLength(1); j++)
            {
                covariates[i, j] = rows[i, j];
            }
        }

        return (covariates, new Vector<int>(treatment), new Vector<double>(outcome));
    }

    [Fact]
    public void ASeededEstimator_ReturnsTheSameStandardErrorEveryCall()
    {
        var (covariates, treatment, outcome) = Sample();
        var model = new InverseProbabilityWeighting<double>(randomSeed: 20260907);
        model.Fit(covariates, treatment);

        var first = model.EstimateATE(covariates, treatment, outcome);
        var second = model.EstimateATE(covariates, treatment, outcome);

        Assert.Equal(first.estimate, second.estimate);
        Assert.Equal(first.standardError, second.standardError);
    }

    [Fact]
    public void TwoEstimatorsWithTheSameSeed_AgreeOnTheStandardError()
    {
        var (covariates, treatment, outcome) = Sample();

        var a = new InverseProbabilityWeighting<double>(randomSeed: 7);
        var b = new InverseProbabilityWeighting<double>(randomSeed: 7);
        a.Fit(covariates, treatment);
        b.Fit(covariates, treatment);

        Assert.Equal(
            a.EstimateATE(covariates, treatment, outcome).standardError,
            b.EstimateATE(covariates, treatment, outcome).standardError);
    }

    /// <summary>
    /// A seed that changed nothing would satisfy the tests above just as well, so this pins that the
    /// resampling actually depends on it.
    /// </summary>
    [Fact]
    public void DifferentSeeds_GiveDifferentStandardErrors()
    {
        var (covariates, treatment, outcome) = Sample();

        var a = new InverseProbabilityWeighting<double>(randomSeed: 1);
        var b = new InverseProbabilityWeighting<double>(randomSeed: 999);
        a.Fit(covariates, treatment);
        b.Fit(covariates, treatment);

        var seA = a.EstimateATE(covariates, treatment, outcome).standardError;
        var seB = b.EstimateATE(covariates, treatment, outcome).standardError;

        Assert.NotEqual(seA, seB);

        // The point estimate is computed from the data, so the seed must not touch it.
        Assert.Equal(
            a.EstimateATE(covariates, treatment, outcome).estimate,
            b.EstimateATE(covariates, treatment, outcome).estimate);
    }

    [Fact]
    public void AnUnseededEstimator_StillDrawsFreshly()
    {
        var (covariates, treatment, outcome) = Sample();
        var model = new InverseProbabilityWeighting<double>();
        model.Fit(covariates, treatment);

        // Not asserting inequality: two unseeded draws could coincide, and a test that fails once in a
        // long while is worse than no test. What matters is that the unseeded path still works.
        var se = model.EstimateATE(covariates, treatment, outcome).standardError;
        Assert.False(double.IsNaN(se));
        Assert.True(se >= 0);
    }

    /// <summary>
    /// The four estimators whose standard error comes from resampling all take the seed. SLearner,
    /// TLearner and XLearner are deliberately absent: they compute the standard error in closed form
    /// from the spread of the per-row effects and draw no random numbers at all, so a seed on them
    /// would be a knob that changes nothing.
    /// </summary>
    [Theory]
    [InlineData("CausalForest")]
    [InlineData("DoublyRobustEstimator")]
    [InlineData("InverseProbabilityWeighting")]
    [InlineData("PropensityScoreMatching")]
    public void EveryResamplingEstimator_ReproducesItsStandardErrorFromASeed(string estimator)
    {
        var (covariates, treatment, outcome) = Sample();

        // Each estimator is fitted through its own two-argument propensity Fit, which the base does
        // not declare, so the build-and-fit stays inside the branch that knows the concrete type.
        double StandardError() => estimator switch
        {
            "CausalForest" => Fit(new CausalForest<double>(seed: 42)),
            "DoublyRobustEstimator" => Fit(new DoublyRobustEstimator<double>(randomSeed: 42)),
            "InverseProbabilityWeighting" => Fit(new InverseProbabilityWeighting<double>(randomSeed: 42)),
            "PropensityScoreMatching" => Fit(new PropensityScoreMatching<double>(seed: 42)),
            _ => throw new ArgumentOutOfRangeException(nameof(estimator))
        };

        double Fit<TModel>(TModel model) where TModel : CausalModelBase<double>
        {
            var treatmentAsT = new Vector<double>(treatment.Length);
            for (int i = 0; i < treatment.Length; i++)
            {
                treatmentAsT[i] = treatment[i];
            }

            model.Fit(covariates, treatmentAsT, outcome);
            return model.EstimateATE(covariates, treatment, outcome).standardError;
        }

        Assert.Equal(StandardError(), StandardError());
    }

    /// <summary>
    /// Propensity-score matching persists its seed, because a model that stops being reproducible the
    /// moment it is saved and loaded is not reproducible.
    /// </summary>
    [Fact]
    public void ASeedSurvivesASerializationRoundTrip()
    {
        var (covariates, treatment, outcome) = Sample();

        var original = new PropensityScoreMatching<double>(seed: 31337);
        original.Fit(covariates, treatment);
        var before = original.EstimateATT(covariates, treatment, outcome).standardError;

        var restored = new PropensityScoreMatching<double>();
        restored.Deserialize(original.Serialize());

        var after = restored.EstimateATT(covariates, treatment, outcome).standardError;

        Assert.Equal(before, after);
    }

    /// <summary>
    /// A payload written before the seed was persisted has no such field, and must still load.
    /// </summary>
    [Fact]
    public void APayloadWithoutASeedField_StillLoads()
    {
        var (covariates, treatment) = (Sample().Covariates, Sample().Treatment);

        var original = new PropensityScoreMatching<double>();
        original.Fit(covariates, treatment);

        var restored = new PropensityScoreMatching<double>();
        var ex = Record.Exception(() => restored.Deserialize(original.Serialize()));

        Assert.Null(ex);
    }
}
