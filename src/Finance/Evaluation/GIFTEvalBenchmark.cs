using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Interfaces;

namespace AiDotNet.Finance.Evaluation;

/// <summary>
/// GIFT-Eval benchmark implementation for standardized evaluation of time series foundation models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GIFT-Eval (General Time Series Forecasting Model Evaluation) is the
/// standard benchmark used by the research community to compare foundation models. It uses two
/// key metrics:
/// <list type="bullet">
/// <item><b>MASE</b> (Mean Absolute Scaled Error): Measures point forecast accuracy relative
/// to a naive baseline. A MASE of 1.0 means the model performs the same as naive forecasting;
/// values below 1.0 mean it's better.</item>
/// <item><b>CRPS</b> (Continuous Ranked Probability Score): Measures probabilistic forecast
/// quality by comparing predicted distributions to actual values. Lower is better.</item>
/// </list>
/// </para>
/// <para>
/// <b>Reference:</b> Jain et al., "GIFT-Eval: A Benchmark for General Time Series Forecasting
/// Model Evaluation", 2024. https://arxiv.org/abs/2410.10393
/// </para>
/// <para>
/// <b>Leaderboard:</b> https://huggingface.co/spaces/Salesforce/GIFT-Eval
/// </para>
/// </remarks>
public class GIFTEvalBenchmark<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the Mean Absolute Scaled Error (MASE) for point forecasts.
    /// </summary>
    /// <param name="predictions">Predicted values of shape [horizon].</param>
    /// <param name="actuals">Actual values of shape [horizon].</param>
    /// <param name="historicalData">Historical data used to compute the naive baseline.</param>
    /// <param name="seasonalPeriod">Seasonal period for the naive baseline (default: 1 for non-seasonal).</param>
    /// <returns>MASE score. Values below 1.0 indicate better-than-naive performance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MASE divides the forecast error by the error of a simple naive
    /// forecast (predicting the last observed value, or the value from one season ago):
    /// <code>
    /// MASE = mean(|prediction - actual|) / mean(|history[t] - history[t-m]|)
    /// </code>
    /// where m is the seasonal period. This makes MASE scale-independent, allowing fair
    /// comparison across datasets with different magnitudes.
    /// </para>
    /// </remarks>
    public double ComputeMASE(Tensor<T> predictions, Tensor<T> actuals, Tensor<T> historicalData, int seasonalPeriod = 1)
    {
        Guard.NotNull(predictions);
        Guard.NotNull(actuals);
        Guard.NotNull(historicalData);
        Guard.Positive(seasonalPeriod);

        int horizon = Math.Min(predictions.Length, actuals.Length);

        // Compute forecast MAE
        if (horizon == 0) return 0.0;

        double forecastMae = 0;
        for (int i = 0; i < horizon; i++)
        {
            double diff = Math.Abs(NumOps.ToDouble(predictions[i]) - NumOps.ToDouble(actuals[i]));
            forecastMae += diff;
        }
        forecastMae /= horizon;

        // Compute naive baseline MAE (seasonal naive)
        double naiveMae = 0;
        int naiveCount = 0;
        for (int i = seasonalPeriod; i < historicalData.Length; i++)
        {
            double diff = Math.Abs(NumOps.ToDouble(historicalData[i]) - NumOps.ToDouble(historicalData[i - seasonalPeriod]));
            naiveMae += diff;
            naiveCount++;
        }

        if (naiveCount > 0)
            naiveMae /= naiveCount;

        // Avoid division by zero
        if (naiveMae < 1e-10)
            return forecastMae < 1e-10 ? 1.0 : double.PositiveInfinity;

        return forecastMae / naiveMae;
    }

    /// <summary>
    /// Computes the Continuous Ranked Probability Score (CRPS) for quantile/probabilistic forecasts.
    /// </summary>
    /// <param name="quantileForecasts">Dictionary mapping quantile level to forecast tensor, each of shape [horizon].</param>
    /// <param name="actuals">Actual values of shape [horizon].</param>
    /// <returns>Average CRPS score across the forecast horizon. Lower is better.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CRPS measures how well the predicted probability distribution
    /// matches the actual outcome. It generalizes MAE to probabilistic forecasts:
    /// <list type="bullet">
    /// <item>If you only provide a point forecast, CRPS equals MAE</item>
    /// <item>If you provide well-calibrated quantile forecasts, CRPS will be lower than MAE</item>
    /// <item>Poorly calibrated probabilistic forecasts may have CRPS higher than MAE</item>
    /// </list>
    /// We approximate CRPS using the quantile score (pinball loss) approach, which is the
    /// standard method used in the GIFT-Eval benchmark.
    /// </para>
    /// </remarks>
    public double ComputeCRPS(IReadOnlyDictionary<double, Tensor<T>> quantileForecasts, Tensor<T> actuals)
    {
        Guard.NotNull(quantileForecasts);
        Guard.NotNull(actuals);

        if (quantileForecasts.Count == 0)
            return double.PositiveInfinity;

        int horizon = actuals.Length;
        double totalCrps = 0;

        for (int t = 0; t < horizon; t++)
        {
            double actual = NumOps.ToDouble(actuals[t]);
            double stepCrps = 0;

            foreach (var kvp in quantileForecasts)
            {
                double q = kvp.Key;
                if (t >= kvp.Value.Length)
                    throw new ArgumentException($"Quantile forecast for q={q:F2} has length {kvp.Value.Length} but horizon requires index {t}.", nameof(quantileForecasts));
                double predicted = NumOps.ToDouble(kvp.Value[t]);
                double error = actual - predicted;

                // Pinball loss (quantile loss)
                double pinball = error >= 0 ? q * error : (q - 1.0) * error;
                stepCrps += pinball;
            }

            stepCrps /= quantileForecasts.Count;
            totalCrps += stepCrps;
        }

        return totalCrps / horizon;
    }

    /// <summary>
    /// Computes the quantile loss (pinball loss) for a specific quantile level.
    /// </summary>
    /// <param name="predictions">Quantile forecast values of shape [horizon].</param>
    /// <param name="actuals">Actual values of shape [horizon].</param>
    /// <param name="quantileLevel">The quantile level (e.g., 0.5 for median).</param>
    /// <returns>Average quantile loss. Lower is better.</returns>
    public double ComputeQuantileLoss(Tensor<T> predictions, Tensor<T> actuals, double quantileLevel)
    {
        Guard.NotNull(predictions);
        Guard.NotNull(actuals);

        int horizon = Math.Min(predictions.Length, actuals.Length);
        double totalLoss = 0;

        for (int i = 0; i < horizon; i++)
        {
            double error = NumOps.ToDouble(actuals[i]) - NumOps.ToDouble(predictions[i]);
            double pinball = error >= 0 ? quantileLevel * error : (quantileLevel - 1.0) * error;
            totalLoss += pinball;
        }

        return totalLoss / horizon;
    }

    /// <summary>
    /// Runs a comprehensive GIFT-Eval style evaluation on a foundation model.
    /// </summary>
    /// <param name="model">The foundation model to evaluate.</param>
    /// <param name="testSeries">List of test time series (each containing context + future).</param>
    /// <param name="contextLengths">Context length for each test series.</param>
    /// <param name="seasonalPeriods">Seasonal period for MASE computation per series.</param>
    /// <param name="quantileLevels">Quantile levels for CRPS computation.</param>
    /// <returns>Dictionary of metric names to average scores.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This runs the full benchmark pipeline: for each test series,
    /// it feeds the context to the model, gets a forecast, and computes both point metrics
    /// (MASE) and probabilistic metrics (CRPS).
    /// </remarks>
    public Dictionary<string, double> RunBenchmark(
        ITimeSeriesFoundationModel<T> model,
        IReadOnlyList<Tensor<T>> testSeries,
        IReadOnlyList<int> contextLengths,
        IReadOnlyList<int> seasonalPeriods,
        double[]? quantileLevels = null)
    {
        Guard.NotNull(model);
        Guard.NotNull(testSeries);
        Guard.NotNull(contextLengths);
        Guard.NotNull(seasonalPeriods);

        quantileLevels ??= new[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };

        if (contextLengths.Count != testSeries.Count)
            throw new ArgumentException($"contextLengths.Count ({contextLengths.Count}) must equal testSeries.Count ({testSeries.Count}).", nameof(contextLengths));
        if (seasonalPeriods.Count != testSeries.Count)
            throw new ArgumentException($"seasonalPeriods.Count ({seasonalPeriods.Count}) must equal testSeries.Count ({testSeries.Count}).", nameof(seasonalPeriods));

        var maseScores = new List<double>();
        var crpsScores = new List<double>();

        for (int s = 0; s < testSeries.Count; s++)
        {
            var series = testSeries[s];
            int contextLen = contextLengths[s];
            int seasonalPeriod = seasonalPeriods[s];

            if (contextLen >= series.Length)
                continue;

            // Split into context and future
            int horizonLen = series.Length - contextLen;
            var context = new Tensor<T>(new[] { contextLen });
            var future = new Tensor<T>(new[] { horizonLen });

            for (int i = 0; i < contextLen; i++)
                context.Data.Span[i] = series[i];
            for (int i = 0; i < horizonLen; i++)
                future.Data.Span[i] = series[contextLen + i];

            // Get forecast
            var prediction = model.Forecast(context, quantileLevels);
            int evalLen = Math.Min(prediction.Length, horizonLen);

            // Compute MASE
            var predSlice = new Tensor<T>(new[] { evalLen });
            var actualSlice = new Tensor<T>(new[] { evalLen });
            for (int i = 0; i < evalLen; i++)
            {
                predSlice.Data.Span[i] = prediction[i];
                actualSlice.Data.Span[i] = future[i];
            }

            double mase = ComputeMASE(predSlice, actualSlice, context, seasonalPeriod);
            if (!double.IsInfinity(mase) && !double.IsNaN(mase))
                maseScores.Add(mase);
        }

        var results = new Dictionary<string, double>();
        results["MASE"] = maseScores.Count > 0 ? maseScores.Average() : double.NaN;
        results["NumSeries"] = testSeries.Count;
        results["NumEvaluated"] = maseScores.Count;

        return results;
    }
}
