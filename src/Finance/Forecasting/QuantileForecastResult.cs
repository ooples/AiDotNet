using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Finance.Forecasting;

/// <summary>
/// Represents the result of a probabilistic/quantile forecast from a time series foundation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A quantile forecast doesn't just predict a single value — it predicts
/// a range of possible values with associated probabilities. For example:
/// <list type="bullet">
/// <item>10th percentile: There's a 10% chance the actual value will be below this</item>
/// <item>50th percentile (median): The "middle" forecast — equally likely to be above or below</item>
/// <item>90th percentile: There's a 90% chance the actual value will be below this</item>
/// </list>
/// This gives you prediction intervals (confidence bands) instead of just point predictions.
/// </para>
/// <para>
/// <b>Supported Models:</b> Chronos-2, Moirai 2.0, TimesFM 2.5, Chronos-Bolt, Lag-Llama,
/// and diffusion-based models (TimeGrad, CSDI, TSDiff) that produce sample-based forecasts.
/// </para>
/// </remarks>
public class QuantileForecastResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the point forecast (median or mean prediction) of shape [horizon].
    /// </summary>
    public Tensor<T> PointForecast { get; }

    /// <summary>
    /// Gets the quantile levels (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]).
    /// </summary>
    public IReadOnlyList<double> QuantileLevels { get; }

    /// <summary>
    /// Gets the quantile forecasts as a dictionary mapping quantile level to forecast tensor.
    /// Each tensor has shape [horizon].
    /// </summary>
    public IReadOnlyDictionary<double, Tensor<T>> QuantileForecasts { get; }

    /// <summary>
    /// Gets the forecast horizon (number of predicted time steps).
    /// </summary>
    public int Horizon => PointForecast.Length;

    /// <summary>
    /// Creates a quantile forecast result from precomputed quantile tensors.
    /// </summary>
    /// <param name="pointForecast">The point (median/mean) forecast.</param>
    /// <param name="quantileForecasts">Mapping from quantile level to forecast tensor.</param>
    public QuantileForecastResult(Tensor<T> pointForecast, Dictionary<double, Tensor<T>> quantileForecasts)
    {
        Guard.NotNull(pointForecast);
        Guard.NotNull(quantileForecasts);

        PointForecast = pointForecast;
        QuantileForecasts = quantileForecasts;
        QuantileLevels = quantileForecasts.Keys.OrderBy(q => q).ToList();
    }

    /// <summary>
    /// Creates a quantile forecast result from sample trajectories.
    /// </summary>
    /// <param name="samples">List of sample trajectories, each of shape [horizon].</param>
    /// <param name="quantileLevels">Quantile levels to compute (e.g., [0.1, 0.5, 0.9]).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Diffusion-based models (TimeGrad, CSDI) produce many sample
    /// trajectories. This constructor converts those samples into quantile forecasts by
    /// sorting the samples at each time step and picking the appropriate percentiles.
    /// </remarks>
    public QuantileForecastResult(IReadOnlyList<Tensor<T>> samples, double[] quantileLevels)
    {
        Guard.NotNull(samples);
        Guard.NotNull(quantileLevels);

        if (samples.Count == 0)
            throw new ArgumentException("At least one sample is required.", nameof(samples));

        int horizon = samples[0].Length;
        for (int i = 1; i < samples.Count; i++)
        {
            if (samples[i].Length != horizon)
                throw new ArgumentException($"All samples must have equal length. Sample 0 has length {horizon}, but sample {i} has length {samples[i].Length}.", nameof(samples));
        }
        int numSamples = samples.Count;

        // Compute point forecast as mean of samples
        var pointData = new T[horizon];
        for (int t = 0; t < horizon; t++)
        {
            T sum = NumOps.Zero;
            for (int s = 0; s < numSamples; s++)
                sum = NumOps.Add(sum, samples[s][t]);
            pointData[t] = NumOps.Divide(sum, NumOps.FromDouble(numSamples));
        }
        PointForecast = new Tensor<T>(new[] { horizon });
        for (int t = 0; t < horizon; t++)
            PointForecast.Data.Span[t] = pointData[t];

        // Compute quantile forecasts
        var qForecasts = new Dictionary<double, Tensor<T>>();
        foreach (double q in quantileLevels)
        {
            var qTensor = new Tensor<T>(new[] { horizon });
            for (int t = 0; t < horizon; t++)
            {
                var values = new double[numSamples];
                for (int s = 0; s < numSamples; s++)
                    values[s] = NumOps.ToDouble(samples[s][t]);
                Array.Sort(values);

                int idx = (int)Math.Floor(q * (numSamples - 1));
                idx = Math.Max(0, Math.Min(idx, numSamples - 1));
                qTensor.Data.Span[t] = NumOps.FromDouble(values[idx]);
            }
            qForecasts[q] = qTensor;
        }

        QuantileForecasts = qForecasts;
        QuantileLevels = quantileLevels.OrderBy(q => q).ToList();
    }

    /// <summary>
    /// Gets the prediction interval at the specified confidence level.
    /// </summary>
    /// <param name="confidenceLevel">Confidence level (e.g., 0.9 for 90% interval).</param>
    /// <returns>Tuple of (lower bound tensor, upper bound tensor) each of shape [horizon].</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A 90% prediction interval means there's roughly a 90% chance
    /// the actual value falls within [lower, upper] at each time step.
    /// </remarks>
    public (Tensor<T> Lower, Tensor<T> Upper) GetPredictionInterval(double confidenceLevel = 0.9)
    {
        double lowerQ = (1.0 - confidenceLevel) / 2.0;
        double upperQ = 1.0 - lowerQ;

        var lower = GetClosestQuantile(lowerQ);
        var upper = GetClosestQuantile(upperQ);

        return (lower, upper);
    }

    private Tensor<T> GetClosestQuantile(double target)
    {
        double bestDist = double.MaxValue;
        Tensor<T>? bestTensor = null;

        foreach (var kvp in QuantileForecasts)
        {
            double dist = Math.Abs(kvp.Key - target);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestTensor = kvp.Value;
            }
        }

        return bestTensor ?? PointForecast;
    }
}
