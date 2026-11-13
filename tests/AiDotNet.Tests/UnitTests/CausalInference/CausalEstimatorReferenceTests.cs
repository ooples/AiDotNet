using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.CausalInference;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.CausalInference;

/// <summary>
/// Pins the causal estimators against scikit-learn / DoWhy reference values on raw-scale covariates.
/// </summary>
/// <remarks>
/// <para>
/// The data are deterministic (no RNG) so the reference script reproduces them exactly:
/// age = 20 + (7i mod 41), income = (13i mod 17) / 4, treatment from a fixed hash of i and age, and
/// y = 2 + 0.05·age + 0.3·income + 1.5·t + 0.1·((29i mod 11) − 5), for i = 0..39.
/// </para>
/// <para>
/// References (scikit-learn 1.9.1): the propensity model is <c>LogisticRegression(C=1)</c> solved to
/// convergence (newton-cg, tol 1e-12), which is DoWhy's default propensity model; the meta-learner outcome
/// models are <c>Ridge(alpha = n·λ)</c> with λ = 0.01; matching follows DoWhy's
/// <c>PropensityScoreMatchingEstimator</c> (1 nearest neighbour with replacement, ATE = (ATT·n₁ + ATC·n₀)/n).
/// The old fixed-step gradient descent diverged on the age column and the old matching reported the ATT as the
/// ATE, so every test here fails on that code.
/// </para>
/// </remarks>
public class CausalEstimatorReferenceTests
{
    private const int N = 40;

    private static (Matrix<double> X, Vector<int> T, Vector<double> Y) Data()
    {
        var x = new Matrix<double>(N, 2);
        var t = new Vector<int>(N);
        var y = new Vector<double>(N);
        for (int i = 0; i < N; i++)
        {
            int age = 20 + (i * 7 % 41);
            double income = (i * 13 % 17) / 4.0;
            int treated = (i * 37 + age) % 10 < 4 + (age > 40 ? 2 : 0) ? 1 : 0;
            x[i, 0] = age;
            x[i, 1] = income;
            t[i] = treated;
            y[i] = 2 + 0.05 * age + 0.3 * income + 1.5 * treated + 0.1 * ((i * 29 % 11) - 5);
        }

        return (x, t, y);
    }

    private static Vector<double> AsDouble(Vector<int> t)
    {
        var v = new Vector<double>(t.Length);
        for (int i = 0; i < t.Length; i++)
        {
            v[i] = t[i];
        }

        return v;
    }

    [Fact]
    public void PropensityScores_MatchScikitLearnLogisticRegression()
    {
        var (x, t, _) = Data();
        var ipw = new InverseProbabilityWeighting<double>(trimMin: 1e-6, trimMax: 1 - 1e-6);
        ipw.Fit(x, t);

        var scores = ipw.EstimatePropensityScores(x);
        double[] expected = { 0.31924766, 0.47293633, 0.49261094, 0.51230846, 0.53196783, 0.68501523 };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], scores[i], 6);
        }
    }

    [Fact]
    public void SLearner_TreatmentEffect_MatchesScikitLearnRidge()
    {
        var (x, t, y) = Data();
        var learner = new SLearner<double>(lambda: 0.01);
        learner.Fit(x, AsDouble(t), y);

        // Ridge(alpha = 40 * 0.01) on [age, income, t]: coef_t = 1.23894571, intercept 2.13807211
        var effects = learner.EstimateTreatmentEffect(x);
        Assert.All(Enumerable.Range(0, N), i => Assert.Equal(1.23894571, effects[i], 6));

        var control = learner.PredictControl(x);
        Assert.Equal(2.13807211 + 0.04994186 * 20 + 0.30320339 * 0.0, control[0], 6);
    }

    [Fact]
    public void SLearner_NarrowerFeatures_Throws()
    {
        var (x, t, y) = Data();
        var learner = new SLearner<double>();
        learner.Fit(x, AsDouble(t), y);

        // One covariate fewer than the fit: the old code silently returned a covariate weight as the effect.
        var narrow = new Matrix<double>(3, 1);
        Assert.Throws<ArgumentException>(() => learner.EstimateTreatmentEffect(narrow));
    }

    [Fact]
    public void TLearner_OutcomeModels_MatchScikitLearnRidge()
    {
        var (x, t, y) = Data();
        var learner = new TLearner<double>(lambda: 0.01);
        learner.Fit(x, AsDouble(t), y);

        var treated = learner.PredictTreated(x);
        var control = learner.PredictControl(x);
        double[] expectedTreated = { 4.39352503, 5.78271584, 5.79734303, 5.81197022 };
        double[] expectedControl = { 3.17109303, 4.40484745, 4.48100064, 4.55715382 };
        for (int i = 0; i < expectedTreated.Length; i++)
        {
            Assert.Equal(expectedTreated[i], treated[i], 6);
            Assert.Equal(expectedControl[i], control[i], 6);
        }
    }

    [Fact]
    public void XLearner_Cate_MatchesKunzelReference()
    {
        var (x, t, y) = Data();
        var learner = new XLearner<double>(lambda: 0.01);
        learner.Fit(x, AsDouble(t), y);

        // τ(x) = e(x)·τ₀(x) + (1 − e(x))·τ₁(x) with Ridge base learners and the LogisticRegression propensity
        var cate = learner.EstimateTreatmentEffect(x);
        double[] expected = { 1.22266424, 1.37769234, 1.31622218, 1.25489655, 1.1937153 };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], cate[i], 6);
        }
    }

    [Fact]
    public void PropensityScoreMatching_AteAndAtt_MatchDoWhy()
    {
        var (x, t, y) = Data();
        var psm = new PropensityScoreMatching<double>(caliper: 1.0, withReplacement: true, matchRatio: 1, seed: 3);
        psm.Fit(x, t);

        var (ate, _) = psm.EstimateATE(x, t, y);
        var (att, _) = psm.EstimateATT(x, t, y);

        Assert.Equal(1.22875, ate, 9);
        Assert.Equal(1.2321428571428572, att, 9);
    }

    [Fact]
    public void DoublyRobust_SeededCrossFitting_IsReproducible()
    {
        var (x, t, y) = Data();
        var first = new DoublyRobustEstimator<double>(useCrossFitting: true, numFolds: 4, randomSeed: 11);
        var second = new DoublyRobustEstimator<double>(useCrossFitting: true, numFolds: 4, randomSeed: 11);
        first.Fit(x, t, y);
        second.Fit(x, t, y);

        // The fold assignment used to come from an unseeded generator, so two seeded fits disagreed.
        Assert.Equal(first.EstimateATE(x, t, y).estimate, second.EstimateATE(x, t, y).estimate, 12);
    }

    [Fact]
    public void CausalForest_HonestSamples_AreDisjointSubsamples()
    {
        var forest = new CausalForest<double>(numTrees: 1, honest: true, honestFraction: 0.5, seed: 5);

        for (int draw = 0; draw < 20; draw++)
        {
            var (structure, estimation) = forest.DrawTreeSamples(N);
            var structureRows = structure.ToArray();
            var estimationRows = estimation.ToArray();

            // Without replacement: no subject twice, and none in both halves (the bootstrap put copies in both).
            Assert.Equal(structureRows.Length, structureRows.Distinct().Count());
            Assert.Equal(estimationRows.Length, estimationRows.Distinct().Count());
            Assert.Empty(structureRows.Intersect(estimationRows));
            Assert.Equal(N / 2, structureRows.Length + estimationRows.Length);
        }
    }
}
