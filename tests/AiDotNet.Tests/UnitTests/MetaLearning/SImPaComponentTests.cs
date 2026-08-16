using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Components;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Verifies SImPa's three components against the paper (Nguyen, Do and Carneiro, arXiv:2003.02455):
/// the implicit posterior generator, the compression-lemma KL estimator, and the PAC-Bayes bounds.
/// </summary>
/// <remarks>
/// <para>
/// These exist because the previous implementation of this citation passed every test it had while
/// implementing the OPPOSITE of the paper: a single point posterior with a closed-form diagonal-Gaussian
/// KL — the assumption the paper's abstract names as the thing its generative posterior improves on. No
/// finiteness, shape, or "loss decreases" test can tell the two apart, because the wrong one is a
/// perfectly well-behaved algorithm. Only a test of the distinguishing PROPERTY can.
/// </para>
/// <para>
/// The distinguishing property is <see cref="TwoDrawsFromTheImplicitPosteriorDiffer"/>: a point posterior
/// returns the same parameters every time, an implicit one does not.
/// </para>
/// </remarks>
public class SImPaComponentTests
{
    private static ImplicitPosteriorGenerator<double> SmallGenerator(int outputDim = 6, int seed = 5) =>
        new(outputDimension: outputDim, latentDimension: 8, firstHiddenWidth: 8, secondHiddenWidth: 8,
            rng: new Random(seed));

    private static double MaxAbsDiff(Vector<double> a, Vector<double> b)
    {
        double d = 0.0;
        for (int i = 0; i < a.Length; i++) d = Math.Max(d, Math.Abs(a[i] - b[i]));
        return d;
    }

    // ---------------- the implicit posterior ----------------

    [Fact]
    public void TwoDrawsFromTheImplicitPosteriorDiffer()
    {
        // THE test that separates this method from what it replaced. A point posterior — the old
        // implementation — yields one parameter vector, so two draws are bit-identical. An implicit
        // posterior is a DISTRIBUTION: each draw pushes fresh latent noise through the generator and
        // lands somewhere else. If this ever passes trivially, the method has silently collapsed back
        // into the baseline the paper was written to beat.
        var generator = SmallGenerator();
        var rng = new Random(1234);

        var first = generator.Sample(rng);
        var second = generator.Sample(rng);

        Assert.Equal(generator.OutputDimension, first.Length);
        Assert.Equal(generator.OutputDimension, second.Length);
        Assert.True(MaxAbsDiff(first, second) > 1e-9,
            "Two draws from the implicit posterior were identical, so it is behaving as a POINT estimate "
            + "and the method has degenerated into the diagonal-Gaussian baseline it replaces.");
    }

    [Fact]
    public void TheGeneratorIsDeterministicGivenItsLatent()
    {
        // The randomness must live entirely in z, not in the mapping. Otherwise the "posterior" is not a
        // fixed distribution at all and nothing about it can be optimized.
        var generator = SmallGenerator();
        var latent = new double[generator.LatentDimension];
        for (int i = 0; i < latent.Length; i++) latent[i] = i / (double)latent.Length;

        var a = generator.Generate(latent);
        var b = generator.Generate(latent);

        Assert.Equal(0.0, MaxAbsDiff(a, b), 12);
    }

    [Fact]
    public void GeneratedParametersAreBoundedByTheOutputTanh()
    {
        // The paper's stated tanh output activation bounds the generated parameters. Without it an
        // untrained generator can emit parameters large enough to make the base model diverge on its very
        // first forward pass, which would look like a broken model rather than an unconverged one.
        var generator = SmallGenerator(outputDim: 12);
        var rng = new Random(77);

        foreach (var sample in generator.SampleMany(24, rng))
        {
            for (int i = 0; i < sample.Length; i++)
            {
                Assert.InRange(sample[i], -1.0, 1.0);
                Assert.False(double.IsNaN(sample[i]), "A generated parameter is NaN.");
            }
        }
    }

    [Fact]
    public void ChangingTheGeneratorWeightsChangesTheDistribution()
    {
        // lambda is what the inner loop adapts, so it has to actually move the posterior. Held constant
        // latent, so the difference can only come from the weights.
        var generator = SmallGenerator();
        var latent = new double[generator.LatentDimension];
        for (int i = 0; i < latent.Length; i++) latent[i] = 0.25 + (0.5 * (i / (double)latent.Length));

        var before = generator.Generate(latent);

        var lambda = generator.GetParameters();
        for (int i = 0; i < lambda.Length; i++) lambda[i] = lambda[i] + 0.35;
        generator.SetParameters(lambda);

        var after = generator.Generate(latent);

        Assert.True(MaxAbsDiff(before, after) > 1e-9,
            "Perturbing the generator weights left the generated parameters unchanged, so the inner loop "
            + "cannot adapt the posterior.");
    }

    [Fact]
    public void TheGeneratorRejectsAMismatchedWeightVector()
    {
        var generator = SmallGenerator();
        Assert.Throws<ArgumentException>(() => generator.SetParameters(new Vector<double>(3)));
        Assert.Throws<ArgumentNullException>(() => generator.SetParameters(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ImplicitPosteriorGenerator<double>(0));
    }

    // ---------------- the compression-lemma KL estimator ----------------

    private static IReadOnlyList<Vector<double>> Gaussian(int count, int dim, double mean, double sd, int seed)
    {
        var rng = new Random(seed);
        var list = new List<Vector<double>>(count);
        for (int s = 0; s < count; s++)
        {
            var v = new Vector<double>(dim);
            for (int d = 0; d < dim; d++)
            {
                double u1 = Math.Max(1e-12, rng.NextDouble());
                double u2 = rng.NextDouble();
                v[d] = mean + (sd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            }
            list.Add(v);
        }
        return list;
    }

    [Fact]
    public void AnUntrainedPhiGivesExactlyTheTrivialZeroBound()
    {
        // The compression lemma's trivial point: phi identically 0 gives 0 - ln(1) = 0. Starting there
        // means the estimate can only ever be improved upward by training, never begin overstated — which
        // matters because the KL enters the bound positively.
        var estimator = new CompressionLemmaKLEstimator<double>(inputDimension: 4, hiddenWidth: 8, rng: new Random(3));
        var q = Gaussian(32, 4, mean: 2.0, sd: 0.5, seed: 11);
        var p = Gaussian(32, 4, mean: 0.0, sd: 1.0, seed: 12);

        // Weights are initialized so that the output layer is zero, hence phi == 0 regardless of input.
        Assert.Equal(0.0, estimator.Phi(q[0]), 12);
        Assert.Equal(0.0, estimator.Objective(q, p), 12);
    }

    [Fact]
    public void MaximizingSeparatesTwoClearlyDifferentDistributions()
    {
        // The estimator's whole job, and it uses SAMPLES ONLY — no density is ever evaluated, which is
        // what makes it usable with an implicit posterior. q is centred far from p, so a real KL exists
        // and the lower bound must climb above the trivial 0.
        var estimator = new CompressionLemmaKLEstimator<double>(inputDimension: 3, hiddenWidth: 12, rng: new Random(5));
        var q = Gaussian(128, 3, mean: 3.0, sd: 0.4, seed: 21);
        var p = Gaussian(128, 3, mean: 0.0, sd: 1.0, seed: 22);

        double estimate = estimator.EstimateKL(q, p, steps: 200, learningRate: 0.05, rng: new Random(23));

        Assert.True(estimate > 0.05,
            $"The KL estimate stayed at {estimate:E3} for two clearly separated distributions, so the "
            + "compression-lemma objective is not being maximized and the bound's KL term is vacuous.");
    }

    [Fact]
    public void TheEstimateStaysNearZeroForTheSameDistribution()
    {
        // The other side of the same claim: no separation means no divergence to find. Without this, a
        // test that only checks "the estimate goes up" would pass an estimator that inflates everything.
        var estimator = new CompressionLemmaKLEstimator<double>(inputDimension: 3, hiddenWidth: 12, rng: new Random(7));
        var q = Gaussian(128, 3, mean: 0.0, sd: 1.0, seed: 31);
        var p = Gaussian(128, 3, mean: 0.0, sd: 1.0, seed: 32);

        double same = estimator.EstimateKL(q, p, steps: 200, learningRate: 0.05, rng: new Random(33));

        var estimator2 = new CompressionLemmaKLEstimator<double>(inputDimension: 3, hiddenWidth: 12, rng: new Random(7));
        var far = Gaussian(128, 3, mean: 3.0, sd: 0.4, seed: 34);
        double different = estimator2.EstimateKL(far, p, steps: 200, learningRate: 0.05, rng: new Random(33));

        Assert.True(different > same,
            $"Two identical distributions scored {same:E3} while clearly different ones scored "
            + $"{different:E3}. The estimate does not track divergence.");
    }

    [Fact]
    public void TheEstimateIsNeverNegative()
    {
        // A KL divergence cannot be negative; a lower bound can be, and returning it raw would let the
        // PAC-Bayes bound shrink below its empirical loss.
        var estimator = new CompressionLemmaKLEstimator<double>(inputDimension: 2, hiddenWidth: 4, rng: new Random(9));
        var q = Gaussian(16, 2, mean: 0.0, sd: 1.0, seed: 41);
        var p = Gaussian(16, 2, mean: 0.0, sd: 1.0, seed: 42);

        Assert.True(estimator.EstimateKL(q, p, steps: 0) >= 0.0);
    }

    [Fact]
    public void TheEstimatorSurvivesLargePhiWithoutOverflowing()
    {
        // exp(phi) overflows for a well-trained phi, so the prior term is computed by log-sum-exp. Without
        // it the estimate becomes NaN exactly when the estimator starts working — a failure that looks
        // like instability rather than a missing numerical trick.
        var estimator = new CompressionLemmaKLEstimator<double>(inputDimension: 2, hiddenWidth: 4, rng: new Random(13));

        var omega = estimator.GetParameters();
        for (int i = 0; i < omega.Length; i++) omega[i] = 50.0;   // drives phi to a huge magnitude
        estimator.SetParameters(omega);

        var q = Gaussian(16, 2, mean: 1.0, sd: 0.2, seed: 51);
        var p = Gaussian(16, 2, mean: 0.0, sd: 1.0, seed: 52);

        double objective = estimator.Objective(q, p);
        Assert.False(double.IsNaN(objective), "The objective overflowed to NaN for a large phi.");
        Assert.False(double.IsInfinity(objective), "The objective overflowed to infinity for a large phi.");
    }

    // ---------------- the PAC-Bayes bounds ----------------

    [Fact]
    public void TheSingleTaskBoundMatchesTheoremOne()
    {
        // Theorem 1: empiricalLoss + sqrt((KL + ln(m / eps)) / (2 (m - 1))). Computed by hand so a change
        // to any term is caught rather than absorbed.
        const double loss = 0.25, kl = 2.0, eps = 0.1;
        const int m = 50;

        double expected = loss + Math.Sqrt((kl + Math.Log(m / eps)) / (2.0 * (m - 1)));

        Assert.Equal(expected, PacBayesMetaBound.SingleTask(loss, kl, m, eps), 12);
    }

    [Fact]
    public void TheMetaBoundMatchesTheoremTwoTermByTerm()
    {
        // Theorem 2's two square roots:
        //   sqrt((E[KL_task] + T^2 ln(m_v) / (eps (T-1))) / (2 (m_v - 1)))
        //   sqrt((KL_meta   + T ln(T) / eps)             / (2 (T - 1)))
        const double loss = 0.4, taskKL = 1.5, metaKL = 0.75, eps = 0.1;
        const int mv = 20, tasks = 8;

        double taskLog = tasks * (double)tasks * Math.Log(mv) / (eps * (tasks - 1));
        double taskTerm = Math.Sqrt((taskKL + taskLog) / (2.0 * (mv - 1)));
        double metaLog = tasks * Math.Log(tasks) / eps;
        double metaTerm = Math.Sqrt((metaKL + metaLog) / (2.0 * (tasks - 1)));

        double expected = loss + taskTerm + metaTerm;
        double actual = PacBayesMetaBound.MetaLearning(loss, taskKL, metaKL, mv, tasks, eps);

        Assert.Equal(expected, actual, 12);
        Assert.True(actual > loss, "A PAC-Bayes bound must exceed the empirical loss it bounds.");
    }

    [Fact]
    public void MoreValidationSamplesTightenTheBound()
    {
        // More data per task must buy a better guarantee: m_v enters as 1/(2(m_v - 1)) while only its
        // logarithm appears in the numerator, so the task term falls. A sign error in that denominator
        // inverts this while leaving every value finite.
        double few = PacBayesMetaBound.MetaLearning(0.3, 1.0, 0.5, validationSampleCount: 10, taskCount: 5);
        double many = PacBayesMetaBound.MetaLearning(0.3, 1.0, 0.5, validationSampleCount: 400, taskCount: 5);

        Assert.True(many < few,
            $"More validation samples must tighten the bound ({few:F4} -> {many:F4}).");
    }

    [Fact]
    public void MoreTasksLoosensTheBoundWhenTheKLTermsAreHeldFIxed()
    {
        // COUNTERINTUITIVE BUT CORRECT, and asserting the opposite is a mistake I made first. Holding both
        // KL terms fixed, growing T makes this bound LOOSER, not tighter:
        //   meta term -> sqrt(T ln(T) / eps / (2(T-1))) -> sqrt(ln(T) / (2 eps)), which grows with T;
        //   task term -> its T^2/(T-1) factor grows like T, so the term grows like sqrt(T).
        // Measured at eps=0.1, KL_task=1, KL_meta=0.5, m_v=10: T=5 gives 6.3188 and T=200 gives 21.4996.
        //
        // That is not a defect in the theorem. Those log terms are UNION-BOUND costs — the guarantee has
        // to hold simultaneously across every task, and more tasks means more to cover. The benefit of
        // more tasks arrives through the KL terms shrinking as the meta-parameters are better determined,
        // which is a property of TRAINING and is deliberately not asserted here, because this test holds
        // the KL fixed precisely to isolate the explicit T dependence.
        double few = PacBayesMetaBound.MetaLearning(0.3, 1.0, 0.5, validationSampleCount: 10, taskCount: 5);
        double many = PacBayesMetaBound.MetaLearning(0.3, 1.0, 0.5, validationSampleCount: 10, taskCount: 200);

        Assert.True(many > few,
            $"With the KL terms fixed, the explicit T dependence must grow ({few:F4} -> {many:F4}).");
    }

    [Fact]
    public void TheBoundsRefuseDegenerateInputs()
    {
        // One sample, or one task, means there is nothing to generalize over. The expressions divide by
        // (m - 1) and (T - 1), so returning a number there would be reporting a guarantee that does not
        // exist rather than declining to make one.
        Assert.Throws<ArgumentOutOfRangeException>(() => PacBayesMetaBound.SingleTask(0.1, 1.0, sampleCount: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => PacBayesMetaBound.MetaLearning(0.1, 1.0, 1.0, 1, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => PacBayesMetaBound.MetaLearning(0.1, 1.0, 1.0, 5, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => PacBayesMetaBound.SingleTask(0.1, -1.0, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => PacBayesMetaBound.SingleTask(0.1, 1.0, 10, epsilon: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PacBayesMetaBound.SingleTask(0.1, 1.0, 10, epsilon: 1.5));
    }
}
