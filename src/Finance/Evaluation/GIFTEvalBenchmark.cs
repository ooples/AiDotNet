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

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Elements [start, start + count) of <paramref name="tensor"/> in row-major order, as a rank-1 tensor.</summary>
    private static Tensor<T> FlatRange(Tensor<T> tensor, int start, int count)
    {
        var flat = Engine.Reshape(tensor, new[] { tensor.Length });
        return start == 0 && count == tensor.Length ? flat : Engine.TensorSlice(flat, new[] { start }, new[] { count });
    }

    /// <summary>Sum of the pinball (quantile) loss at level <paramref name="quantile"/> over error = actual - predicted.</summary>
    /// <remarks>error &gt;= 0 ? q * error : (q - 1) * error, which is q * error + max(-error, 0).</remarks>
    private static T PinballSum(Tensor<T> error, double quantile)
        => NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(quantile), Engine.TensorSum(error)),
            Engine.TensorSum(Engine.TensorReLU(Engine.TensorNegate(error))));

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
    public T ComputeMASE(Tensor<T> predictions, Tensor<T> actuals, Tensor<T> historicalData, int seasonalPeriod = 1)
    {
        Guard.NotNull(predictions);
        Guard.NotNull(actuals);
        Guard.NotNull(historicalData);
        Guard.Positive(seasonalPeriod);

        int horizon = Math.Min(predictions.Length, actuals.Length);

        // Compute forecast MAE
        if (horizon == 0) return NumOps.Zero;

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        T forecastMae = NumOps.Divide(
            Engine.TensorSum(Engine.TensorAbs(Engine.TensorSubtract(
                FlatRange(predictions, 0, horizon), FlatRange(actuals, 0, horizon)))),
            NumOps.FromDouble(horizon));

        // Compute naive baseline MAE (seasonal naive): |h[i] - h[i - period]| for i in [period, length).
        T naiveMae = NumOps.Zero;
        int naiveCount = Math.Max(0, historicalData.Length - seasonalPeriod);
        if (naiveCount > 0)
        {
            naiveMae = NumOps.Divide(
                Engine.TensorSum(Engine.TensorAbs(Engine.TensorSubtract(
                    FlatRange(historicalData, seasonalPeriod, naiveCount), FlatRange(historicalData, 0, naiveCount)))),
                NumOps.FromDouble(naiveCount));
        }

        // Avoid division by zero
        T tiny = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(naiveMae, tiny))
            return NumOps.LessThan(forecastMae, tiny) ? NumOps.One : NumOps.FromDouble(double.PositiveInfinity);

        return NumOps.Divide(forecastMae, naiveMae);
    }

    /// <summary>
    /// Computes the Continuous Ranked Probability Score (CRPS) for quantile/probabilistic forecasts.
    /// </summary>
    /// <param name="quantileForecasts">Dictionary mapping quantile level to forecast tensor, each of shape [horizon].</param>
    /// <param name="actuals">Actual values of shape [horizon].</param>
    /// <returns>
    /// The GIFT-Eval CRPS of one series: the mean weighted quantile loss
    /// <c>mean_q [2 * sum_t pinball_q(y_t, yhat_q,t) / sum_t |y_t|]</c>. Lower is better.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CRPS measures how well the predicted probability distribution
    /// matches the actual outcome. It generalizes MAE to probabilistic forecasts:
    /// <list type="bullet">
    /// <item>If you only provide a median forecast, CRPS equals MAE divided by the mean absolute target</item>
    /// <item>If you provide well-calibrated quantile forecasts, CRPS will be lower than MAE</item>
    /// <item>Poorly calibrated probabilistic forecasts may have CRPS higher than MAE</item>
    /// </list>
    /// We approximate CRPS using the quantile score (pinball loss) approach, which is the
    /// standard method used in the GIFT-Eval benchmark: GIFT-Eval reports the gluonts
    /// <c>MeanWeightedSumQuantileLoss</c>, where each quantile's summed loss is doubled and scaled by
    /// the summed absolute target, then the quantiles are averaged.
    /// </para>
    /// </remarks>
    public T ComputeCRPS(IReadOnlyDictionary<double, Tensor<T>> quantileForecasts, Tensor<T> actuals)
    {
        Guard.NotNull(quantileForecasts);
        Guard.NotNull(actuals);

        if (quantileForecasts.Count == 0)
            return NumOps.FromDouble(double.PositiveInfinity);

        int horizon = actuals.Length;

        // The step-by-step scan reported the first quantile (in enumeration order) that runs out at the earliest
        // step; that is the shortest forecast, at the step equal to its length.
        if (horizon > 0)
        {
            int shortest = int.MaxValue;
            foreach (var kvp in quantileForecasts)
                shortest = Math.Min(shortest, kvp.Value.Length);
            if (shortest < horizon)
            {
                foreach (var kvp in quantileForecasts)
                {
                    if (kvp.Value.Length == shortest)
                        throw new ArgumentException($"Quantile forecast for q={kvp.Key:F2} has length {kvp.Value.Length} but horizon requires index {shortest}.", nameof(quantileForecasts));
                }
            }
        }

        // GIFT-Eval CRPS (gluonts MeanWeightedSumQuantileLoss):
        // (1 / Q) * sum_q [2 * sum_t pinball_q / sum_t |y_t|] = 2 * sum_q sum_t pinball_q / (Q * sum_t |y_t|).
        T totalLoss = NumOps.Zero;
        T absoluteTarget = NumOps.Zero;
        if (horizon > 0)
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var actual = FlatRange(actuals, 0, horizon);
            absoluteTarget = Engine.TensorSum(Engine.TensorAbs(actual));
            foreach (var kvp in quantileForecasts)
            {
                var error = Engine.TensorSubtract(actual, FlatRange(kvp.Value, 0, horizon));
                totalLoss = NumOps.Add(totalLoss, PinballSum(error, kvp.Key));
            }
        }

        return NumOps.Divide(
            NumOps.Multiply(NumOps.FromDouble(2.0), totalLoss),
            NumOps.Multiply(NumOps.FromDouble(quantileForecasts.Count), absoluteTarget));
    }

    /// <summary>
    /// Computes the quantile loss (pinball loss) for a specific quantile level.
    /// </summary>
    /// <param name="predictions">Quantile forecast values of shape [horizon].</param>
    /// <param name="actuals">Actual values of shape [horizon].</param>
    /// <param name="quantileLevel">The quantile level (e.g., 0.5 for median).</param>
    /// <returns>Average quantile loss. Lower is better.</returns>
    public T ComputeQuantileLoss(Tensor<T> predictions, Tensor<T> actuals, double quantileLevel)
    {
        Guard.NotNull(predictions);
        Guard.NotNull(actuals);

        int horizon = Math.Min(predictions.Length, actuals.Length);
        T totalLoss = NumOps.Zero;
        if (horizon > 0)
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var error = Engine.TensorSubtract(FlatRange(actuals, 0, horizon), FlatRange(predictions, 0, horizon));
            totalLoss = PinballSum(error, quantileLevel);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(horizon));
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
    public Dictionary<string, T> RunBenchmark(
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

        var maseScores = new List<T>();

        // GIFT-Eval aggregates the weighted quantile loss over the whole dataset: each level's pinball
        // losses and the absolute targets are summed across every scored series before dividing.
        var levelLosses = new List<Tensor<T>>();
        var absoluteTargets = new List<T>();
        int numQ = quantileLevels.Length;
        var levelRow = new Tensor<double>(new[] { 1, numQ }, new Vector<double>(quantileLevels.ToArray())).Cast<T>();

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

            series.AsSpan().Slice(0, contextLen).CopyTo(context.Data.Span);
            series.AsSpan().Slice(contextLen, horizonLen).CopyTo(future.Data.Span);

            // Get point forecast (no quantiles) for MASE
            var pointPrediction = model.Forecast(context, null);
            int evalLen = Math.Min(pointPrediction.Length, horizonLen);

            // Compute MASE from point forecast
            var predSlice = new Tensor<T>(new[] { evalLen });
            var actualSlice = new Tensor<T>(new[] { evalLen });
            pointPrediction.AsSpan().Slice(0, evalLen).CopyTo(predSlice.Data.Span);
            future.AsSpan().Slice(0, evalLen).CopyTo(actualSlice.Data.Span);

            T mase = ComputeMASE(predSlice, actualSlice, context, seasonalPeriod);
            maseScores.Add(mase);

            // Compute CRPS from quantile forecasts (single call with all quantile levels)
            var allQuantileForecast = model.Forecast(context, quantileLevels);

            // Output layout: [horizon * numQuantiles] with quantile as inner dimension. A forecast that does
            // not cover the whole [evalLen, numQ] grid is not scored: GIFT-Eval needs every level at every
            // step, and filling the gap with zeros would score a forecast that was never made.
            int needed = evalLen * numQ;
            if (evalLen <= 0 || numQ <= 0 || allQuantileForecast.Length < needed)
                continue;

            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            {
                var gridShape = new[] { evalLen, numQ };
                var grid = Engine.Reshape(FlatRange(allQuantileForecast, 0, needed), gridShape);
                var actualColumn = Engine.TensorBroadcastTo(Engine.Reshape(actualSlice, new[] { evalLen, 1 }), gridShape);
                var error = Engine.TensorSubtract(actualColumn, grid);

                // pinball_q(e) = q * e + max(-e, 0), summed down the steps for each level.
                var pinball = Engine.TensorAdd(
                    Engine.TensorMultiply(Engine.TensorBroadcastTo(levelRow, gridShape), error),
                    Engine.TensorReLU(Engine.TensorNegate(error)));
                levelLosses.Add(Engine.ReduceSum(pinball, new[] { 0 }, false));
                absoluteTargets.Add(Engine.TensorSum(Engine.TensorAbs(actualSlice)));
            }
        }

        var results = new Dictionary<string, T>();
        var (maseMean, maseFinite) = FiniteMeanOrNaN(maseScores);
        results["MASE"] = maseMean;
        results["CRPS"] = DatasetWeightedQuantileLoss(levelLosses, absoluteTargets, numQ);
        results["NumCRPSEvaluated"] = NumOps.FromDouble(levelLosses.Count);
        results["NumSeries"] = NumOps.FromDouble(testSeries.Count);
        results["NumEvaluated"] = NumOps.FromDouble(maseFinite);

        return results;
    }

    /// <summary>
    /// GIFT-Eval's dataset CRPS: (1 / Q) * sum_q [2 * (sum over series and steps of pinball_q) / (sum of |y|)];
    /// NaN when no series was scored.
    /// </summary>
    private static T DatasetWeightedQuantileLoss(List<Tensor<T>> levelLosses, List<T> absoluteTargets, int numQ)
    {
        if (levelLosses.Count == 0 || numQ == 0)
            return NumOps.FromDouble(double.NaN);
        var perLevel = Engine.ReduceSum(Engine.TensorStack(levelLosses.ToArray(), 0), new[] { 0 }, false);
        T absoluteTotal = Engine.TensorSum(
            new Tensor<T>(new[] { absoluteTargets.Count }, new Vector<T>(absoluteTargets.ToArray())));
        return NumOps.Divide(
            NumOps.Multiply(NumOps.FromDouble(2.0), Engine.TensorSum(perLevel)),
            NumOps.Multiply(NumOps.FromDouble(numQ), absoluteTotal));
    }

    /// <summary>
    /// Mean of the finite scores (NaN and infinite ones are skipped) and how many were finite;
    /// NaN when none are.
    /// </summary>
    private static (T Mean, int Finite) FiniteMeanOrNaN(List<T> scores)
    {
        if (scores.Count == 0)
            return (NumOps.FromDouble(double.NaN), 0);
        var all = new Tensor<T>(new[] { scores.Count }, new Vector<T>(scores.ToArray()));
        // x - x is 0 exactly when x is finite (NaN and +/-infinity give NaN).
        var finite = Engine.TensorEquals(Engine.TensorSubtract(all, all), NumOps.Zero);
        int count = Engine.TensorCountNonzero(finite);
        if (count == 0)
            return (NumOps.FromDouble(double.NaN), 0);
        var kept = Engine.TensorWhere(finite, all, Engine.TensorZerosLike(all));
        T countT = NumOps.FromDouble(count);
        T mean = NumOps.Divide(Engine.TensorSum(kept), countT);
        // The sum of finite scores can overflow T (two float scores near 3e38) while their mean is finite:
        // then divide each score by the count before summing.
        if (NumOps.IsInfinity(mean))
            mean = Engine.TensorSum(Engine.TensorDivideScalar(kept, countT));
        return (mean, count);
    }
}
