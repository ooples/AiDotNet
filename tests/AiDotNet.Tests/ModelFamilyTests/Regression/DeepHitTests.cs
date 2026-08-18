using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class DeepHitTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new DeepHit<double>();

    /// <summary>
    /// DeepHit is a discrete-time survival model, so the identity-link invariants do not describe it.
    /// </summary>
    /// <remarks>
    /// The network's output is a probability mass function over (cause, time bin) cells, produced by one
    /// softmax. Prediction is the expectation over that distribution, so it is bounded by the observed
    /// time range: it cannot shift by an arbitrary constant when the targets are shifted, and it cannot
    /// extrapolate past the last bin. There is no intercept and no linear coefficient to recover a sign for.
    ///
    /// These invariants used to pass because TrainAsync set `_useOLS = true` and fitted ordinary least
    /// squares, which is an identity-link model; the DeepHit network was never trained. The
    /// survival-specific tests below check what actually characterizes this model.
    ///
    /// This matches the six existing non-identity-link regressions -- Poisson, Gamma, Inverse Gaussian,
    /// Negative Binomial, Tweedie and Beta -- which set the same flag.
    /// </remarks>
    protected override bool IdentityLinkInvariantsApplicable => false;

    /// <summary>
    /// Builds survival data where higher feature values mean higher hazard and therefore shorter survival.
    /// </summary>
    private static (Matrix<double> X, Vector<double> Times, Vector<double> Events) MakeSurvivalData(
        int n, double censorFraction, int seed)
    {
        var rng = ModelTestHelpers.CreateSeededRandom(seed);
        var x = new Matrix<double>(n, 2);
        var times = new Vector<double>(n);
        var events = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = rng.NextDouble() * 2.0;
            x[i, 1] = rng.NextDouble() * 2.0;

            double linear = 0.9 * x[i, 0] + 0.4 * x[i, 1];
            double rate = Math.Exp(linear);
            double u = Math.Max(1e-6, rng.NextDouble());
            times[i] = -Math.Log(u) / rate + 0.05;

            events[i] = rng.NextDouble() < censorFraction ? 0.0 : 1.0;
        }

        return (x, times, events);
    }

    private static double Concordance(Vector<double> predictedTimes, Vector<double> times, Vector<double> events)
    {
        int concordant = 0;
        int comparable = 0;

        for (int i = 0; i < times.Length; i++)
        {
            if (events[i] == 0.0) continue;

            for (int j = 0; j < times.Length; j++)
            {
                if (i == j) continue;
                if (times[j] <= times[i]) continue;

                comparable++;
                if (predictedTimes[i] < predictedTimes[j]) concordant++;
            }
        }

        return comparable == 0 ? 0.5 : (double)concordant / comparable;
    }

    private static DeepHitOptions<double> Options(int seed, int epochs = 150) => new DeepHitOptions<double>
    {
        Seed = seed,
        Epochs = epochs,
        NumTimeBins = 20,
        LearningRate = 0.01,
        DropoutRate = 0.0
    };

    [Fact(Timeout = 300000)]
    public async Task RiskOrdering_HigherHazardFeatures_PredictShorterTime()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(200, censorFraction: 0.0, seed: 7);
        var model = new DeepHit<double>(Options(11));

        await model.TrainAsync(x, times, events);
        var predicted = model.Predict(x);

        Assert.True(ModelTestHelpers.AllFinite(predicted), "Predicted event times must be finite.");

        // The bar is the oracle rather than a round number: the times are exponential, so even the true
        // risk function cannot order them perfectly, and the ceiling is set by that noise not by the model.
        var oracle = new Vector<double>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            oracle[i] = Math.Exp(-(0.9 * x[i, 0] + 0.4 * x[i, 1]));
        }

        double oracleC = Concordance(oracle, times, events);
        double c = Concordance(predicted, times, events);
        double bar = 0.5 + 0.70 * (oracleC - 0.5);

        Assert.True(c >= bar,
            $"Concordance was {c:F4}, below {bar:F4} = 70% of the oracle's excess over chance " +
            $"(oracle {oracleC:F4}). The network recovered too little of the available risk ordering.");
    }

    [Fact(Timeout = 300000)]
    public async Task PredictedPmf_IsAProperDistribution()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(120, censorFraction: 0.2, seed: 19);
        var model = new DeepHit<double>(Options(5, epochs: 80));

        await model.TrainAsync(x, times, events);
        var pmf = model.PredictPMF(x);

        // The single softmax spans every (cause, time bin) cell, so each subject's cells sum to one.
        for (int i = 0; i < x.Rows; i++)
        {
            double total = 0.0;
            for (int k = 0; k < 1; k++)
            {
                for (int t = 0; t < 20; t++)
                {
                    double p = pmf[i, k, t];
                    Assert.True(p >= -1e-12 && p <= 1.0 + 1e-9, $"PMF cell ({i},{k},{t}) = {p} is not a probability.");
                    total += p;
                }
            }

            Assert.True(Math.Abs(total - 1.0) < 1e-6, $"PMF for subject {i} sums to {total:G17}, not 1.");
        }
    }

    [Fact(Timeout = 300000)]
    public async Task Prediction_IsDeterministic_DropoutIsTrainingOnly()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(80, censorFraction: 0.0, seed: 29);

        // Dropout is deliberately ON here: it must still not perturb inference.
        var options = Options(13, epochs: 40);
        options.DropoutRate = 0.3;
        var model = new DeepHit<double>(options);

        await model.TrainAsync(x, times, events);

        var first = model.Predict(x);
        var second = model.Predict(x);

        // ApplyLayer used to drop units on every call, including from Predict, so the same model returned
        // a different answer each time it was asked.
        for (int i = 0; i < first.Length; i++)
        {
            Assert.True(Math.Abs(first[i] - second[i]) < 1e-12,
                $"Prediction {i} changed between two calls on the same model: {first[i]:G17} vs {second[i]:G17}.");
        }
    }

    [Fact(Timeout = 300000)]
    public async Task CompetingRisks_EachCauseGetsItsOwnDistribution()
    {
        await Task.Yield();
        var rng = ModelTestHelpers.CreateSeededRandom(37);
        int n = 200;
        var x = new Matrix<double>(n, 2);
        var times = new Vector<double>(n);
        var events = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = rng.NextDouble() * 2.0;
            x[i, 1] = rng.NextDouble() * 2.0;

            // Feature 0 drives cause 1, feature 1 drives cause 2.
            double rate1 = Math.Exp(1.0 * x[i, 0] - 0.5);
            double rate2 = Math.Exp(1.0 * x[i, 1] - 0.5);
            double t1 = -Math.Log(Math.Max(1e-6, rng.NextDouble())) / rate1;
            double t2 = -Math.Log(Math.Max(1e-6, rng.NextDouble())) / rate2;

            if (t1 <= t2) { times[i] = t1 + 0.05; events[i] = 1.0; }
            else { times[i] = t2 + 0.05; events[i] = 2.0; }
        }

        var options = Options(23, epochs: 150);
        options.NumRisks = 2;
        var model = new DeepHit<double>(options);

        await model.TrainAsync(x, times, events);

        var pmf = model.PredictPMF(x);

        // Total mass assigned to each cause should track which cause actually dominates that subject.
        int agree = 0;
        for (int i = 0; i < n; i++)
        {
            double mass1 = 0.0, mass2 = 0.0;
            for (int t = 0; t < 20; t++)
            {
                mass1 += pmf[i, 0, t];
                mass2 += pmf[i, 1, t];
            }

            bool cause1Favoured = mass1 > mass2;
            bool cause1Expected = x[i, 0] > x[i, 1];
            if (cause1Favoured == cause1Expected) agree++;
        }

        double rate = (double)agree / n;
        Assert.True(rate > 0.65,
            $"The predicted cause agreed with the dominant hazard on only {rate:P1} of subjects; " +
            "the cause-specific branches are not separating the two risks.");
    }

    [Fact(Timeout = 300000)]
    public async Task Serialization_RoundTrip_PreservesPredictions()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(100, censorFraction: 0.2, seed: 47);
        var model = new DeepHit<double>(Options(3, epochs: 50));
        await model.TrainAsync(x, times, events);

        var before = model.Predict(x);

        var restored = new DeepHit<double>(Options(3));
        restored.Deserialize(model.Serialize());
        var after = restored.Predict(x);

        // Feature standardization is a fitted parameter; if the round trip drops it the restored network
        // sees raw covariates on a different scale and these diverge.
        for (int i = 0; i < before.Length; i++)
        {
            Assert.True(Math.Abs(before[i] - after[i]) < 1e-8,
                $"Prediction {i} changed across serialization: {before[i]:G17} vs {after[i]:G17}.");
        }
    }

    [Fact(Timeout = 300000)]
    public async Task EventCodeBeyondNumRisks_Throws()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(40, censorFraction: 0.0, seed: 61);
        events[0] = 3.0; // NumRisks is 1, so cause 3 does not exist.

        var model = new DeepHit<double>(Options(3, epochs: 5));

        var ex = await Assert.ThrowsAsync<ArgumentException>(() => model.TrainAsync(x, times, events));
        Assert.Contains("NumRisks", ex.Message);
    }
}
