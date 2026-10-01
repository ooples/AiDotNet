using System;
using System.Threading.Tasks;
using AiDotNet.Finance.Volatility;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for the classical GARCH-type volatility models (GARCH(1,1), EGARCH, GJR-GARCH).
/// </summary>
/// <remarks>
/// <para>
/// These models fit a return series by quasi-maximum likelihood and forecast conditional volatility;
/// <c>Train</c> fits the returns and ignores its target, and <c>Predict</c> is a one-step volatility
/// forecast. So the invariants are the properties of that estimator and forecast, and every one holds for
/// all three variants, not just GARCH(1,1):
/// </para>
/// <list type="bullet">
/// <item><description>Scale equivariance: returns scaled by c give variance forecasts scaled by c^2 (the
/// intercept absorbs the scale - omega * c^2 for GARCH and GJR, a log-intercept shift for EGARCH).</description></item>
/// <item><description>Volatility clustering: a large shock raises the next variance forecast. A negative
/// shock is used, so GJR responds through alpha + gamma and EGARCH through its asymmetry term.</description></item>
/// <item><description>Mean reversion: the multi-step forecast settles toward a long-run level.</description></item>
/// </list>
/// <para>
/// The data is one seeded GARCH(1,1) path (omega 0.05, alpha 0.08, beta 0.90, scaled to daily-return size),
/// so the fitted responses are clearly non-zero and every run sees the same series.
/// </para>
/// </remarks>
public abstract class VolatilityModelTestBase
{
    /// <summary>Length of the simulated return series.</summary>
    protected virtual int ReturnCount => 1500;

    /// <summary>Subclasses construct the model under test.</summary>
    protected abstract ClassicalVolatilityModelBase<double> CreateModel();

    private double[] SimulateReturns(double scale = 0.01)
    {
        var rng = RandomHelper.CreateSeededRandom(2024);
        const double omega = 0.05;
        const double alpha = 0.08;
        const double beta = 0.90;
        double variance = omega / (1.0 - alpha - beta);
        double previous = 0.0;
        var returns = new double[ReturnCount];
        for (int t = 0; t < returns.Length; t++)
        {
            variance = omega + (alpha * previous * previous) + (beta * variance);
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            previous = Math.Sqrt(variance) * z;
            returns[t] = previous * scale;
        }

        return returns;
    }

    private ClassicalVolatilityModelBase<double> FittedModel(double[] returns)
    {
        var model = CreateModel();
        model.FitReturns(returns);
        return model;
    }

    [Fact(Timeout = 120000)]
    public async Task Fit_ProducesFiniteParametersOfTheDeclaredCount()
    {
        await Task.Yield();
        var model = FittedModel(SimulateReturns());

        var parameters = model.GetParameters();

        Assert.Equal(model.ParameterCount, parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(!double.IsNaN(parameters[i]) && !double.IsInfinity(parameters[i]),
                $"Fitted parameter {i} is {parameters[i]}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Forecast_IsPositiveAndFinite()
    {
        await Task.Yield();
        var returns = SimulateReturns();
        var model = FittedModel(returns);

        double variance = model.ForecastNextVariance(returns);

        Assert.True(variance > 0 && !double.IsInfinity(variance),
            $"The one-step variance forecast is {variance}; a conditional variance is positive and finite.");
    }

    [Fact(Timeout = 120000)]
    public async Task Forecast_ScalesWithTheReturns()
    {
        await Task.Yield();
        var returns = SimulateReturns(scale: 0.01);
        var scaled = SimulateReturns(scale: 0.10);

        double baseVariance = FittedModel(returns).ForecastNextVariance(returns);
        double scaledVariance = FittedModel(scaled).ForecastNextVariance(scaled);

        // Returns scaled by 10 must give a variance forecast scaled by 100. The tolerance covers the
        // simplex optimizer converging along a different path, not a different optimum.
        double ratio = scaledVariance / baseVariance;
        Assert.InRange(ratio, 95.0, 105.0);
    }

    [Fact(Timeout = 120000)]
    public async Task Forecast_RespondsToAShock()
    {
        await Task.Yield();
        var returns = SimulateReturns();
        var model = FittedModel(returns);

        double spread = 0;
        foreach (double r in returns)
        {
            spread += r * r / returns.Length;
        }

        var calm = new double[returns.Length + 1];
        var shocked = new double[returns.Length + 1];
        Array.Copy(returns, calm, returns.Length);
        Array.Copy(returns, shocked, returns.Length);
        calm[returns.Length] = 0.0;
        shocked[returns.Length] = -5.0 * Math.Sqrt(spread);

        double afterCalm = model.ForecastNextVariance(calm);
        double afterShock = model.ForecastNextVariance(shocked);

        Assert.True(afterShock > afterCalm,
            $"A five-sigma shock left the next variance forecast at {afterShock} against {afterCalm} after a " +
            "calm day. Volatility clustering is the behaviour these models exist to capture.");
    }

    [Fact(Timeout = 120000)]
    public async Task MultiStepForecast_SettlesTowardALongRunLevel()
    {
        await Task.Yield();
        var returns = SimulateReturns();
        var model = FittedModel(returns);
        var tensor = new Tensor<double>(new[] { returns.Length }, new Vector<double>(returns));

        var path = model.ForecastVolatility(tensor, 400);

        double firstStep = Math.Abs(path[1] - path[0]);
        double lastStep = Math.Abs(path[path.Length - 1] - path[path.Length - 2]);
        Assert.True(lastStep <= firstStep + 1e-12,
            $"The forecast is still moving by {lastStep} after 400 steps against {firstStep} at the start; " +
            "a stationary model's multi-step forecast settles toward its long-run level.");
    }

    [Fact(Timeout = 120000)]
    public async Task Fit_IsDeterministic()
    {
        await Task.Yield();
        var returns = SimulateReturns();

        double first = FittedModel(returns).ForecastNextVariance(returns);
        double second = FittedModel(returns).ForecastNextVariance(returns);

        Assert.Equal(first, second, 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Serialize_RoundTripsTheForecast()
    {
        await Task.Yield();
        var returns = SimulateReturns();
        var model = FittedModel(returns);

        var restored = CreateModel();
        restored.Deserialize(model.Serialize());

        // Forecast on a DIFFERENT series than the fit. ForecastNextVariance refits when a model has no
        // fitted parameters, so on the fitting series a restore that lost them would refit to the same
        // answer and pass. Here a lost restore refits to the new series and disagrees.
        var other = SimulateReturns(scale: 0.02);
        Assert.Equal(model.ForecastNextVariance(other), restored.ForecastNextVariance(other), 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Fit_RejectsTooFewReturns()
    {
        await Task.Yield();
        var model = CreateModel();

        Assert.Throws<ArgumentException>(() => model.FitReturns(new double[5]));
    }
}
