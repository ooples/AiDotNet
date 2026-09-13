using System;
using AiDotNet.CausalInference;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.CausalInference;

/// <summary>
/// Covers <c>AiModelBuilder.Build(covariates, treatment, outcome)</c>, the three-argument overload that
/// keeps treatment assignment out of the covariate matrix.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="CausalModelBase{T}"/> reads column 0 of its design matrix as the treatment indicator,
/// which works but leaves callers assembling that matrix by hand and one mistake away from a covariate
/// being read as treatment. This overload hands the assignment to the model instead, the same way the
/// survival overload beside it hands over its censoring.
/// </para>
/// </remarks>
public class CausalBuildOverloadTests
{
    private const int Subjects = 6;

    private static Matrix<double> Covariates()
    {
        double[,] rows = { { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 } };
        var m = new Matrix<double>(Subjects, 2);
        for (int i = 0; i < Subjects; i++)
        {
            m[i, 0] = rows[i, 0];
            m[i, 1] = rows[i, 1];
        }

        return m;
    }

    private static Vector<int> Treatment() => new(new[] { 0, 1, 0, 1, 0, 1 });

    private static Vector<double> Outcome() =>
        new(new double[] { 3.1, 7.2, 2.8, 8.0, 3.4, 7.6 });

    private static Vector<double> TreatmentAsDouble()
    {
        var treatment = Treatment();
        var asT = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++)
        {
            asT[i] = treatment[i];
        }

        return asT;
    }

    /// <summary>
    /// The overload's whole job: reach the same fit as the model's own three-argument entry point, while
    /// the caller passes three named arguments and never sees a design matrix.
    /// </summary>
    [Fact]
    public void ItFitsWhatTheExplicitThreeArgumentCallFits()
    {
        var covariates = Covariates();

        var viaBuilder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new TLearner<double>())
            .Build(covariates, Treatment(), Outcome());

        var direct = new TLearner<double>();
        direct.Fit(covariates, TreatmentAsDouble(), Outcome());

        var a = viaBuilder.EstimateTreatmentEffect(covariates);
        var b = direct.EstimateTreatmentEffect(covariates);

        Assert.Equal(b.Length, a.Length);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], a[i]);
        }
    }

    /// <summary>
    /// The reason the overload hands the assignment over rather than folding it into X: the effect
    /// estimators take the covariates and the treatment separately, so the covariate matrix has to stay
    /// the covariate matrix on the way out as well as in.
    /// </summary>
    [Fact]
    public void TheEffectEstimatorsTakeTheSameCovariatesTrainingDid()
    {
        var covariates = Covariates();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new TLearner<double>())
            .Build(covariates, Treatment(), Outcome());

        var effects = result.EstimateTreatmentEffect(covariates);

        Assert.Equal(covariates.Rows, effects.Length);
    }

    /// <summary>
    /// If the assignment were dropped on the way through, who was treated would stop mattering.
    /// </summary>
    [Fact]
    public void TreatmentAssignmentReachesTheModel()
    {
        var covariates = Covariates();
        var flipped = new Vector<int>(new[] { 1, 0, 1, 0, 1, 0 });

        var asGiven = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new TLearner<double>())
            .Build(covariates, Treatment(), Outcome());

        var reversed = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new TLearner<double>())
            .Build(covariates, flipped, Outcome());

        var a = asGiven.EstimateTreatmentEffect(covariates);
        var b = reversed.EstimateTreatmentEffect(covariates);

        bool differs = false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                differs = true;
                break;
            }
        }

        Assert.True(differs, "Reversing who was treated changed nothing, so the assignment is lost.");
    }

    [Fact]
    public void ANonCausalModel_IsRejectedBeforeTraining()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(new SimpleRegression<double>())
                .Build(Covariates(), Treatment(), Outcome()));

        Assert.Contains("causal models", ex.Message, StringComparison.Ordinal);
        Assert.Contains(nameof(SimpleRegression<double>), ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NoModelConfigured_SaysWhatToConfigure()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .Build(Covariates(), Treatment(), Outcome()));

        Assert.Contains("TLearner", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MismatchedLengths_NameAllThreeCounts()
    {
        var shortTreatment = new Vector<int>(new[] { 0, 1 });

        var ex = Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(new TLearner<double>())
                .Build(Covariates(), shortTreatment, Outcome()));

        Assert.Contains("6 rows", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAssignmentThatIsNotZeroOrOne_IsRejected()
    {
        var bad = new Vector<int>(new[] { 0, 1, 3, 1, 0, 1 });

        var ex = Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(new TLearner<double>())
                .Build(Covariates(), bad, Outcome()));

        Assert.Contains("found 3 at index 2", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The survival overload takes the same three shapes in a different order. Which one a call reaches
    /// is decided by where the <c>Vector&lt;int&gt;</c> sits, so this pins that a causal-shaped call goes
    /// to the causal model and is not quietly read as survival data.
    /// </summary>
    [Fact]
    public void ACausalShapedCall_DoesNotReachTheSurvivalOverload()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(new global::AiDotNet.SurvivalAnalysis.KaplanMeierEstimator<double>())
                .Build(Covariates(), Treatment(), Outcome()));

        // Reached the causal overload, which rejected the survival model — rather than the survival
        // overload silently accepting a treatment vector as censoring.
        Assert.Contains("causal models", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("SLearner")]
    [InlineData("TLearner")]
    [InlineData("XLearner")]
    [InlineData("DoublyRobustEstimator")]
    [InlineData("InverseProbabilityWeighting")]
    public void EveryCausalModel_BuildsThroughTheOverload(string model)
    {
        CausalModelBase<double> Build() => model switch
        {
            "SLearner" => new SLearner<double>(),
            "TLearner" => new TLearner<double>(),
            "XLearner" => new XLearner<double>(),
            "DoublyRobustEstimator" => new DoublyRobustEstimator<double>(randomSeed: 42),
            "InverseProbabilityWeighting" => new InverseProbabilityWeighting<double>(randomSeed: 42),
            _ => throw new ArgumentOutOfRangeException(nameof(model))
        };

        var covariates = Covariates();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(Build())
            .Build(covariates, Treatment(), Outcome());

        var direct = Build();
        direct.Fit(covariates, TreatmentAsDouble(), Outcome());

        var a = result.EstimateTreatmentEffect(covariates);
        var b = direct.EstimateTreatmentEffect(covariates);

        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], a[i]);
        }
    }
}
