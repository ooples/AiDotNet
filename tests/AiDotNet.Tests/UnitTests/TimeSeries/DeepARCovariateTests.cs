using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Tests for DeepAR covariate consumption (DeepAROptions.CovariateSize > 0): the model must actually
/// condition on the covariates (not silently ignore x), validate the covariate contract, and reject the
/// univariate predict path once trained with covariates.
/// </summary>
public sealed class DeepARCovariateTests
{
    private static DeepAROptions<double> CovOptions() => new()
    {
        LookbackWindow = 10,
        ForecastHorizon = 3,
        HiddenSize = 20,
        NumLayers = 1,
        Epochs = 60,
        BatchSize = 8,
        NumSamples = 100,
        LikelihoodType = "Gaussian",
        CovariateSize = 1,
    };

    // net471-safe finiteness check (double.IsFinite is .NET Core / netstandard2.1+ only).
    private static bool IsFinite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);

    // Series whose one-step change is DRIVEN by a covariate: y[t+1] - y[t] ≈ covariate[t]. A covariate-blind
    // model cannot fit this; a covariate-aware one can, so the trained model's next-step forecast must move
    // with the last covariate.
    private static (Matrix<double> x, Vector<double> y, Matrix<double> cov) CovariateDrivenSeries(int n)
    {
        var rng = new Random(11);
        var y = new Vector<double>(n);
        var cov = new Matrix<double>(n, 1);
        double level = 0.0;
        for (var i = 0; i < n; i++)
        {
            double c = Math.Sin(i * 0.3) + (rng.NextDouble() - 0.5) * 0.1; // covariate at index i
            cov[i, 0] = c;
            y[i] = level;
            level += c; // next level moves by the current covariate
        }

        // x doubles as the covariate matrix (the covariate contract: x is [N, CovariateSize] aligned to y).
        var x = new Matrix<double>(n, 1);
        for (var i = 0; i < n; i++)
            x[i, 0] = cov[i, 0];
        return (x, y, cov);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Covariate_model_trains_and_forecasts_finite()
    {
        var options = CovOptions();
        var model = new DeepARModel<double>(options);
        var (x, y, cov) = CovariateDrivenSeries(160);
        model.Train(x, y);

        var history = new Vector<double>(options.LookbackWindow);
        var histCov = new Matrix<double>(options.LookbackWindow, 1);
        for (var j = 0; j < options.LookbackWindow; j++)
        {
            history[j] = y[100 + j];
            histCov[j, 0] = cov[100 + j, 0];
        }

        var future = new Matrix<double>(options.ForecastHorizon, 1);
        for (var h = 0; h < options.ForecastHorizon; h++)
            future[h, 0] = 0.2;

        var bands = model.ForecastWithCovariates(history, histCov, future, new[] { 0.1, 0.5, 0.9 });
        for (var h = 0; h < options.ForecastHorizon; h++)
        {
            Assert.True(IsFinite(bands[0.5][h]), $"non-finite covariate forecast at h={h}");
            Assert.True(bands[0.1][h] <= bands[0.5][h] + 1e-9 && bands[0.5][h] <= bands[0.9][h] + 1e-9, "non-monotone band");
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void The_covariate_actually_moves_the_forecast()
    {
        var options = CovOptions();
        var model = new DeepARModel<double>(options);
        var (x, y, cov) = CovariateDrivenSeries(160);
        model.Train(x, y);

        // Same series window; two covariate windows that differ ONLY in the last row (the covariate that drives
        // the next step). If x were ignored, the two point forecasts would be identical.
        var history = new Vector<double>(options.LookbackWindow);
        var covLow = new Matrix<double>(options.LookbackWindow, 1);
        var covHigh = new Matrix<double>(options.LookbackWindow, 1);
        for (var j = 0; j < options.LookbackWindow; j++)
        {
            history[j] = y[100 + j];
            covLow[j, 0] = cov[100 + j, 0];
            covHigh[j, 0] = cov[100 + j, 0];
        }
        covLow[options.LookbackWindow - 1, 0] = -1.0;
        covHigh[options.LookbackWindow - 1, 0] = 1.0;

        double fLow = model.PredictNextWithCovariates(history, covLow);
        double fHigh = model.PredictNextWithCovariates(history, covHigh);

        Assert.True(IsFinite(fLow) && IsFinite(fHigh), "non-finite covariate point forecast");
        // A covariate that drives the next step must change the forecast, and in the right direction
        // (higher covariate → higher next value on this series).
        Assert.True(Math.Abs(fHigh - fLow) > 1e-4, $"covariate had no effect on the forecast ({fLow} vs {fHigh}) — x was ignored");
        Assert.True(fHigh > fLow, $"forecast did not move with the covariate sign ({fLow} -> {fHigh})");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Covariate_contract_is_validated()
    {
        var options = CovOptions();
        options.CovariateSize = 2; // require 2 covariate columns
        var model = new DeepARModel<double>(options);

        var y = new Vector<double>(40);
        var xTooFew = new Matrix<double>(40, 1); // only 1 column — violates the contract
        Assert.Throws<ArgumentException>(() => model.Train(xTooFew, y));
    }

    [Fact]
    [Trait("category", "unit")]
    public void Univariate_predict_is_rejected_for_a_covariate_model()
    {
        var options = CovOptions();
        var model = new DeepARModel<double>(options);
        var (x, y, _) = CovariateDrivenSeries(120);
        model.Train(x, y);

        var window = new Vector<double>(options.LookbackWindow);
        Assert.Throws<InvalidOperationException>(() => model.PredictSingle(window));
    }
}
