using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class DeepSurvTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new DeepSurv<double>();

    /// <summary>
    /// DeepSurv is a Cox proportional-hazards model, so the identity-link invariants do not describe it.
    /// </summary>
    /// <remarks>
    /// The Cox partial likelihood depends only on the ORDER of the observed times, so it is invariant to
    /// any monotone transformation of them -- shifting every time by +1000 leaves the fitted risk function
    /// unchanged, and translation equivariance cannot hold. There is no intercept to recover either: the
    /// baseline hazard absorbs it and is profiled out of the likelihood. The coefficient sign convention is
    /// also inverted relative to a least-squares fit on time, because a positive Cox coefficient raises the
    /// hazard and therefore SHORTENS survival.
    ///
    /// These invariants used to pass here, which was itself the bug: TrainAsync set `_useOLS = true` and
    /// fitted ordinary least squares, and least squares is an identity-link model. The Cox network that the
    /// class documents was never run. The survival-specific tests below check the behaviour that actually
    /// characterizes this model, and they are the ones that would have caught the substitution.
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

            // Hazard rises with both features, so survival time falls. Exponential survival with that rate.
            double linear = 0.9 * x[i, 0] + 0.4 * x[i, 1];
            double rate = Math.Exp(linear);
            double u = Math.Max(1e-6, rng.NextDouble());
            times[i] = -Math.Log(u) / rate + 0.05;

            bool censored = rng.NextDouble() < censorFraction;
            events[i] = censored ? 0.0 : 1.0;
        }

        return (x, times, events);
    }

    /// <summary>
    /// Concordance between predicted survival time and observed time, over comparable pairs.
    /// </summary>
    private static double Concordance(Vector<double> predictedTimes, Vector<double> times, Vector<double> events)
    {
        int concordant = 0;
        int comparable = 0;

        for (int i = 0; i < times.Length; i++)
        {
            // Only a subject whose event was observed can be known to have failed first.
            if (events[i] != 1.0) continue;

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

    [Fact(Timeout = 180000)]
    public async Task RiskOrdering_HigherHazardFeatures_PredictShorterSurvival()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(200, censorFraction: 0.0, seed: 7);
        var model = new DeepSurv<double>(new DeepSurvOptions<double> { Seed = 11, Epochs = 120 });

        model.TrainAsync(x, times, events).GetAwaiter().GetResult();

        var predicted = model.Predict(x);

        Assert.True(ModelTestHelpers.AllFinite(predicted), "Predicted survival times must be finite.");
        for (int i = 0; i < predicted.Length; i++)
        {
            Assert.True(predicted[i] > 0.0, $"Predicted survival time must be positive; got {predicted[i]} at {i}.");
        }

        // The whole point of the Cox fit: subjects who failed earlier should be predicted to survive less
        // long. An untrained or substituted model sits at chance, 0.5.
        //
        // The bar is the ORACLE rather than a round number. Survival times here are exponential, so even
        // the true risk function cannot order them perfectly -- the achievable concordance is capped by
        // that noise, not by the model. Comparing against the oracle measures how much of the available
        // signal the network actually recovered, and keeps the test meaningful if the data generator
        // is ever retuned.
        var oracle = new Vector<double>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            oracle[i] = Math.Exp(-(0.9 * x[i, 0] + 0.4 * x[i, 1]));
        }

        double oracleC = Concordance(oracle, times, events);
        double c = Concordance(predicted, times, events);
        double bar = 0.5 + 0.85 * (oracleC - 0.5);

        Assert.True(c >= bar,
            $"Concordance was {c:F4}, below {bar:F4} = 85% of the oracle's excess over chance " +
            $"(oracle {oracleC:F4}). The network recovered too little of the available risk ordering.");
    }

    [Fact(Timeout = 180000)]
    public async Task CensoredSubjects_AreUsedForTheRiskSet_NotDiscarded()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(200, censorFraction: 0.35, seed: 23);
        var model = new DeepSurv<double>(new DeepSurvOptions<double> { Seed = 11, Epochs = 120 });

        model.TrainAsync(x, times, events).GetAwaiter().GetResult();
        var predicted = model.Predict(x);

        Assert.True(ModelTestHelpers.AllFinite(predicted), "Predicted survival times must be finite under censoring.");

        double c = Concordance(predicted, times, events);
        Assert.True(c > 0.60, $"Concordance under 35% censoring was {c:F4}, no better than chance.");
    }

    [Fact(Timeout = 180000)]
    public async Task MonotoneTimeTransform_LeavesRiskOrderingUnchanged()
    {
        await Task.Yield();
        // The Cox partial likelihood depends only on the order of the times, so squaring every time --
        // a strictly increasing transform -- must not change which subjects the model ranks as riskier.
        var (x, times, events) = MakeSurvivalData(150, censorFraction: 0.0, seed: 31);
        var squared = new Vector<double>(times.Length);
        for (int i = 0; i < times.Length; i++) squared[i] = times[i] * times[i];

        var a = new DeepSurv<double>(new DeepSurvOptions<double> { Seed = 5, Epochs = 100, DropoutRate = 0.0 });
        var b = new DeepSurv<double>(new DeepSurvOptions<double> { Seed = 5, Epochs = 100, DropoutRate = 0.0 });

        a.TrainAsync(x, times, events).GetAwaiter().GetResult();
        b.TrainAsync(x, squared, events).GetAwaiter().GetResult();

        var riskA = a.PredictRiskScores(x);
        var riskB = b.PredictRiskScores(x);

        int agree = 0;
        int pairs = 0;
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = i + 1; j < x.Rows; j++)
            {
                pairs++;
                if ((riskA[i] < riskA[j]) == (riskB[i] < riskB[j])) agree++;
            }
        }

        double agreement = (double)agree / pairs;
        Assert.True(agreement > 0.90,
            $"Risk ordering agreed on only {agreement:P1} of pairs after a monotone transform of time; " +
            "the partial likelihood depends on time order alone, so it should be nearly unchanged.");
    }

    [Fact(Timeout = 180000)]
    public async Task Serialization_RoundTrip_PreservesPredictions()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(120, censorFraction: 0.2, seed: 47);
        var model = new DeepSurv<double>(new DeepSurvOptions<double> { Seed = 3, Epochs = 60 });
        model.TrainAsync(x, times, events).GetAwaiter().GetResult();

        var before = model.Predict(x);

        var restored = new DeepSurv<double>(new DeepSurvOptions<double> { Seed = 3 });
        restored.Deserialize(model.Serialize());
        var after = restored.Predict(x);

        // Batch-norm running statistics are model parameters; if the round trip drops them the restored
        // network normalizes with mean 0 / variance 1 and these diverge.
        for (int i = 0; i < before.Length; i++)
        {
            Assert.True(Math.Abs(before[i] - after[i]) < 1e-8,
                $"Prediction {i} changed across serialization: {before[i]:G17} vs {after[i]:G17}.");
        }
    }

    [Fact(Timeout = 180000)]
    public async Task AllCensored_Throws_RatherThanFittingNothing()
    {
        await Task.Yield();
        var (x, times, events) = MakeSurvivalData(50, censorFraction: 0.0, seed: 61);
        var allCensored = new Vector<double>(times.Length);

        var model = new DeepSurv<double>(new DeepSurvOptions<double> { Seed = 3 });

        var ex = Assert.Throws<ArgumentException>(() =>
        {
            try
            {
                model.TrainAsync(x, times, allCensored).GetAwaiter().GetResult();
            }
            catch (AggregateException agg) when (agg.InnerException is ArgumentException inner)
            {
                throw inner;
            }
        });

        Assert.Contains("censored", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
