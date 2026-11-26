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

    [Fact]
    public void CausalForest_SameSeed_DrawsTheSameSamples()
    {
        // Disjointness alone would hold if the seed were ignored; two instances with one seed must agree draw by draw.
        var first = new CausalForest<double>(numTrees: 1, honest: true, honestFraction: 0.5, seed: 5);
        var second = new CausalForest<double>(numTrees: 1, honest: true, honestFraction: 0.5, seed: 5);

        for (int draw = 0; draw < 20; draw++)
        {
            var (firstStructure, firstEstimation) = first.DrawTreeSamples(N);
            var (secondStructure, secondEstimation) = second.DrawTreeSamples(N);
            Assert.Equal(firstStructure.ToArray(), secondStructure.ToArray());
            Assert.Equal(firstEstimation.ToArray(), secondEstimation.ToArray());
        }
    }

    [Fact]
    public void CausalForest_PartitionsTooSmallForBothGroups_AreRejected()
    {
        // 40 rows at sampleFraction 0.05 is a 2-row subsample, one row per honest partition: no partition can hold a
        // treated and a control row, and every tree used to predict exactly zero.
        var (x, t, y) = Data();
        var forest = new CausalForest<double>(numTrees: 2, honest: true, honestFraction: 0.5, seed: 5, sampleFraction: 0.05);

        Assert.Throws<ArgumentException>(() => forest.Fit(x, t, y));
    }

    [Fact]
    public void SLearner_FitsTheBinarizedIndicator()
    {
        // Treatment given as 0.2 / 0.8 is control / treated. The outcome jumps by 6 between the groups, so the
        // effect is 6; fitting the raw 0.2/0.8 magnitudes made the treatment coefficient 6 / 0.6 = 10.
        var x = new Matrix<double>(N, 1);
        var treatment = new Vector<double>(N);
        var outcome = new Vector<double>(N);
        for (int i = 0; i < N; i++)
        {
            x[i, 0] = i % 3;
            treatment[i] = i % 2 == 0 ? 0.2 : 0.8;
            outcome[i] = i % 2 == 0 ? 0.0 : 6.0;
        }

        var learner = new SLearner<double>(lambda: 1e-9);
        learner.Fit(x, treatment, outcome);

        Assert.Equal(6.0, learner.PredictTreatmentEffect(x).Average(), 4);
    }

    [Fact]
    public void CausalForest_SplitsBetweenAdjacentDoubles()
    {
        // Feature values a = 1 + 2^-52 and b = 1 + 2^-51 are adjacent doubles: (a + b) / 2 rounds to b, which puts
        // every row on the left and leaves no split, so every effect was 0. The threshold must fall back to a.
        // Rows at a have effect +10 and rows at b effect -10.
        double a = 1.0 + Math.Pow(2, -52), b = 1.0 + Math.Pow(2, -51);
        var x = new Matrix<double>(N, 1);
        var t = new Vector<int>(N);
        var y = new Vector<double>(N);
        for (int i = 0; i < N; i++)
        {
            bool low = i < N / 2;
            x[i, 0] = low ? a : b;
            t[i] = i % 2;
            y[i] = (low ? 10.0 : -10.0) * t[i];
        }

        var forest = new CausalForest<double>(numTrees: 1, maxDepth: 1, minSamplesLeaf: 1, maxFeatures: 1,
            honest: false, seed: 3, sampleFraction: 1.0);
        forest.Fit(x, t, y);
        var effects = forest.PredictTreatmentEffect(x);

        for (int i = 0; i < N; i++) Assert.Equal(i < N / 2 ? 10.0 : -10.0, effects[i], 9);
    }

    [Theory]
    [InlineData(typeof(SLearner<double>))]
    [InlineData(typeof(TLearner<double>))]
    [InlineData(typeof(XLearner<double>))]
    public void MetaLearners_RemovedIterationConstructor_IsACompileError(Type learner)
    {
        // new TLearner<double>(100) used to bind 100 to lambda; the guard constructor must be an error, not a warning.
        var constructor = learner.GetConstructor(new[] { typeof(int) });
        Assert.NotNull(constructor);
        var obsolete = Attribute.GetCustomAttribute(constructor, typeof(ObsoleteAttribute)) as ObsoleteAttribute;
        Assert.NotNull(obsolete);
        Assert.True(obsolete.IsError);
    }

    [Fact]
    public void CausalForest_HonestLeavesNeverKeepTheStructureEstimate()
    {
        // A twin forest with the same seed replays the first tree's structure/estimation draw, so the data can be
        // built around it. Structure rows split at x = 0.5 with effects +50 (x = 0) and -50 (x = 1); every
        // estimation row has x = 0 and y = 1 + 5t, so the right leaf receives no estimation row. Honest estimation
        // (grf) prunes that split and predicts the estimation contrast, 5, everywhere. The old leaf re-estimation
        // skipped the empty leaf, which kept the structure estimate -50 for x = 1.
        const int seed = 7;
        var twin = new CausalForest<double>(numTrees: 1, maxDepth: 1, minSamplesLeaf: 1, maxFeatures: 1,
            honest: true, honestFraction: 0.5, seed: seed, sampleFraction: 1.0);
        var (structure, estimation) = twin.DrawTreeSamples(N);

        var x = new Matrix<double>(N, 1);
        var t = new Vector<int>(N);
        var y = new Vector<double>(N);
        var structureRows = structure.ToArray().OrderBy(r => r).ToArray();
        for (int k = 0; k < structureRows.Length; k++)
        {
            int row = structureRows[k];
            bool right = k >= structureRows.Length / 2;
            x[row, 0] = right ? 1.0 : 0.0;
            t[row] = k % 2;
            y[row] = (right ? -50.0 : 50.0) * t[row];
        }

        var estimationRows = estimation.ToArray().OrderBy(r => r).ToArray();
        for (int k = 0; k < estimationRows.Length; k++)
        {
            int row = estimationRows[k];
            x[row, 0] = 0.0;
            t[row] = k % 2;
            y[row] = 1.0 + (5.0 * t[row]);
        }

        var forest = new CausalForest<double>(numTrees: 1, maxDepth: 1, minSamplesLeaf: 1, maxFeatures: 1,
            honest: true, honestFraction: 0.5, seed: seed, sampleFraction: 1.0);
        forest.Fit(x, t, y);

        var probe = new Matrix<double>(2, 1);
        probe[0, 0] = 0.0;
        probe[1, 0] = 1.0;
        var effects = forest.PredictTreatmentEffect(probe);
        Assert.Equal(5.0, effects[0], 9);
        Assert.Equal(5.0, effects[1], 9);
    }

    [Fact]
    public void DoublyRobust_TreatmentLeadingInput_PredictsFromTheCovariates()
    {
        // Train and Predict pass [treatment | covariates]. The outcome models have one slope per covariate, so the
        // leading treatment column must be dropped; truncating to the slope count (the old code) read the
        // treatment as age and age as income.
        var (x, t, y) = Data();
        var estimator = new DoublyRobustEstimator<double>();
        estimator.Fit(x, t, y);

        var withTreatment = new Matrix<double>(N, 3);
        for (int i = 0; i < N; i++)
        {
            withTreatment[i, 0] = t[i];
            withTreatment[i, 1] = x[i, 0];
            withTreatment[i, 2] = x[i, 1];
        }

        var expectedTreated = estimator.PredictTreated(x);
        var expectedControl = estimator.PredictControl(x);
        var treated = estimator.PredictTreated(withTreatment);
        var control = estimator.PredictControl(withTreatment);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(expectedTreated[i], treated[i], 12);
            Assert.Equal(expectedControl[i], control[i], 12);
        }

        // Any other width is rejected rather than truncated.
        Assert.Throws<ArgumentException>(() => estimator.PredictTreated(new Matrix<double>(N, 5)));
    }

    [Fact]
    public void DoublyRobust_ParameterVectorMustSplitIntoThreeBlocks()
    {
        // Propensity, treated-outcome and control-outcome blocks of NumFeatures + 1 each: 9 values here. Ten
        // values used to be zero-padded to 12, silently zeroing the trailing control-outcome coefficients.
        var (x, t, y) = Data();
        var estimator = new DoublyRobustEstimator<double>();
        estimator.Fit(x, t, y);
        var parameters = estimator.GetParameters();
        Assert.Equal(9, parameters.Length);

        var tooLong = new Vector<double>(10);
        for (int i = 0; i < 9; i++) tooLong[i] = parameters[i];
        Assert.Throws<ArgumentException>(() => estimator.SetParameters(tooLong));
    }
}
