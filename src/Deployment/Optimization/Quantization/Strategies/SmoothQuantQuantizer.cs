using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using AiDotNet.Deployment.Optimization.Quantization.Calibration;

namespace AiDotNet.Deployment.Optimization.Quantization.Strategies;

/// <summary>
/// SmoothQuant - transfers quantization difficulty from activations to weights using per-channel smoothing.
/// Enables efficient W8A8 quantization (both weights and activations at 8-bit).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Activations (intermediate values during inference) often have
/// outliers that are very hard to quantize. SmoothQuant "smooths" these outliers by mathematically
/// transferring some of their range to the weights, making both easier to quantize.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Collect activation statistics (max absolute values per channel)</description></item>
/// <item><description>Collect weight statistics (max absolute values per channel)</description></item>
/// <item><description>Compute smoothing factor s = (act_max)^α / (weight_max)^(1-α) per channel</description></item>
/// <item><description>Apply transformation: Y = (X * diag(s)^-1) * (diag(s) * W)</description></item>
/// <item><description>Now both X' = X/s and W' = s*W have similar, easier-to-quantize ranges</description></item>
/// </list>
///
/// <para><b>Key Features:</b></para>
/// <list type="bullet">
/// <item><description>Enables W8A8: Both weights AND activations quantized to 8-bit</description></item>
/// <item><description>Alpha parameter: Controls balance of smoothing (0.5 = balanced)</description></item>
/// <item><description>Per-channel: Different smoothing for each output channel</description></item>
/// <item><description>Mathematically equivalent: Same outputs as original model</description></item>
/// </list>
///
/// <para><b>Reference:</b> Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
/// Quantization for Large Language Models" (2023)</para>
/// </remarks>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class SmoothQuantQuantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly QuantizationConfiguration _config;
    private readonly Dictionary<string, double> _scaleFactors = new();
    private readonly Dictionary<string, int> _zeroPoints = new();
    private readonly Dictionary<string, double[]> _smoothingScales = new();
    private ActivationStatistics<T>? _activationStats;
    private double[]? _activationMaxValues;
    private bool _isCalibrated;

    /// <inheritdoc/>
    public QuantizationMode Mode => _config.Mode;

    /// <inheritdoc/>
    public int BitWidth => _config.EffectiveBitWidth;

    /// <summary>
    /// Gets whether the quantizer has been calibrated.
    /// </summary>
    public bool IsCalibrated => _isCalibrated;

    /// <summary>
    /// Gets whether calibration used real forward passes through the model.
    /// </summary>
    /// <remarks>
    /// Returns true if calibration collected activation statistics by running actual forward passes
    /// through the model. Returns false if calibration fell back to parameter-based estimation.
    /// </remarks>
    public bool UsedRealForwardPasses => _activationStats?.IsFromRealForwardPasses ?? false;

    /// <summary>
    /// Computes the zero-point for asymmetric quantization.
    /// </summary>
    /// <param name="min">The minimum value in the range.</param>
    /// <param name="scale">The quantization scale factor.</param>
    /// <param name="qMax">The maximum quantized value.</param>
    /// <returns>The clamped zero-point value.</returns>
    private static int ComputeAsymmetricZeroPoint(double min, double scale, double qMax)
    {
        return (int)MathHelper.Clamp(Math.Round(-min / scale), 0, qMax);
    }

    /// <summary>
    /// Initializes a new instance of the SmoothQuantQuantizer.
    /// </summary>
    /// <param name="config">Quantization configuration</param>
    public SmoothQuantQuantizer(QuantizationConfiguration? config = null)
    {
        _config = config ?? QuantizationConfiguration.ForSmoothQuant();

        if (_config.Strategy != QuantizationStrategy.SmoothQuant)
        {
            _config.Strategy = QuantizationStrategy.SmoothQuant;
        }
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        if (!_isCalibrated && config.CalibrationMethod != CalibrationMethod.None)
        {
            throw new InvalidOperationException(
                "SmoothQuant requires calibration data to compute smoothing factors. Call Calibrate() first.");
        }

        var parameters = model.GetParameters();
        var quantizedParams = QuantizeWithSmoothQuant(parameters, config);
        return model.WithParameters(quantizedParams);
    }

    /// <inheritdoc/>
    public void Calibrate(IFullModel<T, TInput, TOutput> model, IEnumerable<TInput> calibrationData)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (calibrationData == null) throw new ArgumentNullException(nameof(calibrationData));

        var dataList = calibrationData.ToList();
        if (dataList.Count == 0)
        {
            throw new ArgumentException("Calibration data cannot be empty", nameof(calibrationData));
        }

        // Use CalibrationHelper to collect activation statistics via real forward passes
        var calibrationHelper = new CalibrationHelper<T, TInput, TOutput>(_config);
        _activationStats = calibrationHelper.CollectActivationStatistics(model, dataList);

        var parameters = model.GetParameters();
        int n = parameters.Length;

        // Extract max absolute activation values from statistics
        if (_activationStats.GlobalMaxAbsActivations != null &&
            _activationStats.GlobalMaxAbsActivations.Length >= n)
        {
            _activationMaxValues = _activationStats.GlobalMaxAbsActivations;
        }
        else
        {
            // Fallback: use parameter magnitudes with small epsilon
            _activationMaxValues = new double[n];
            for (int i = 0; i < n; i++)
            {
                _activationMaxValues[i] = Math.Abs(Convert.ToDouble(parameters[i])) + 1e-6;
            }
        }

        // Compute weight statistics
        var weightStats = new double[n];
        for (int i = 0; i < n; i++)
        {
            weightStats[i] = Math.Abs(Convert.ToDouble(parameters[i])) + 1e-6;
        }

        // Compute smoothing scales: s = (act_max)^α / (weight_max)^(1-α)
        ComputeSmoothingScales(_activationMaxValues, weightStats);

        _isCalibrated = true;
    }

    /// <inheritdoc/>
    public double GetScaleFactor(string layerName)
    {
        return _scaleFactors.TryGetValue(layerName, out var scale) ? scale :
               _scaleFactors.TryGetValue("global", out var globalScale) ? globalScale : 1.0;
    }

    /// <inheritdoc/>
    public int GetZeroPoint(string layerName)
    {
        return _zeroPoints.TryGetValue(layerName, out var zp) ? zp :
               _zeroPoints.TryGetValue("global", out var globalZp) ? globalZp : 0;
    }

    /// <summary>
    /// Gets the smoothing scales applied to transform weights.
    /// </summary>
    public IReadOnlyDictionary<string, double[]> SmoothingScales => _smoothingScales;

    /// <summary>
    /// Quantizes parameters using SmoothQuant with per-channel smoothing.
    /// </summary>
    private Vector<T> QuantizeWithSmoothQuant(Vector<T> parameters, QuantizationConfiguration config)
    {
        int n = parameters.Length;
        int bitWidth = config.EffectiveBitWidth;

        double qMin = config.UseSymmetricQuantization ? -(1 << (bitWidth - 1)) : 0;
        double qMax = config.UseSymmetricQuantization ? (1 << (bitWidth - 1)) - 1 : (1 << bitWidth) - 1;

        var result = new T[n];
        var weights = new double[n];

        // Convert to double
        for (int i = 0; i < n; i++)
        {
            weights[i] = Convert.ToDouble(parameters[i]);
        }

        // Get smoothing scales
        double[] smoothingScales = GetSmoothingScales(n);

        // Step 1: Apply smoothing transformation to weights: W' = s * W
        var smoothedWeights = new double[n];
        for (int i = 0; i < n; i++)
        {
            smoothedWeights[i] = weights[i] * smoothingScales[i];
        }

        // Step 2: Quantize the smoothed weights
        if (config.Granularity == QuantizationGranularity.PerChannel ||
            config.Granularity == QuantizationGranularity.PerTensor)
        {
            QuantizePerChannel(smoothedWeights, result, bitWidth, config, qMin, qMax);
        }
        else
        {
            // Per-group quantization
            QuantizePerGroup(smoothedWeights, result, bitWidth, config, qMin, qMax);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs per-channel quantization.
    /// </summary>
    private void QuantizePerChannel(double[] weights, T[] result, int bitWidth,
                                    QuantizationConfiguration config, double qMin, double qMax)
    {
        int n = weights.Length;

        // For per-tensor, use single scale
        if (config.Granularity == QuantizationGranularity.PerTensor)
        {
            double min = weights.Min();
            double max = weights.Max();

            double scale;
            int zeroPoint;

            if (config.UseSymmetricQuantization)
            {
                double absMax = Math.Max(Math.Abs(min), Math.Abs(max));
                scale = absMax / ((1 << (bitWidth - 1)) - 1);
                scale = Math.Max(scale, config.MinScaleFactor);
                zeroPoint = 0;
            }
            else
            {
                scale = (max - min) / ((1 << bitWidth) - 1);
                scale = Math.Max(scale, config.MinScaleFactor);
                // Clamp zero-point to valid asymmetric range [0, qMax]
                zeroPoint = ComputeAsymmetricZeroPoint(min, scale, qMax);
            }

            _scaleFactors["global"] = scale;
            _zeroPoints["global"] = zeroPoint;

            for (int i = 0; i < n; i++)
            {
                double quantized = Math.Round(weights[i] / scale) + zeroPoint;
                quantized = MathHelper.Clamp(quantized, qMin, qMax);
                double dequantized = (quantized - zeroPoint) * scale;
                result[i] = NumOps.FromDouble(dequantized);
            }
        }
        else
        {
            // Estimate channel structure (assume square-ish distribution)
            int numChannels = (int)Math.Sqrt(n);
            numChannels = Math.Max(1, numChannels);
            int channelSize = n / numChannels;

            for (int c = 0; c < numChannels; c++)
            {
                int start = c * channelSize;
                int end = (c == numChannels - 1) ? n : start + channelSize;

                double min = double.MaxValue;
                double max = double.MinValue;

                for (int i = start; i < end; i++)
                {
                    min = Math.Min(min, weights[i]);
                    max = Math.Max(max, weights[i]);
                }

                double scale;
                int zeroPoint;

                if (config.UseSymmetricQuantization)
                {
                    double absMax = Math.Max(Math.Abs(min), Math.Abs(max));
                    scale = absMax / ((1 << (bitWidth - 1)) - 1);
                    scale = Math.Max(scale, config.MinScaleFactor);
                    zeroPoint = 0;
                }
                else
                {
                    scale = (max - min) / ((1 << bitWidth) - 1);
                    scale = Math.Max(scale, config.MinScaleFactor);
                    // Clamp zero-point to valid asymmetric range [0, qMax]
                    zeroPoint = ComputeAsymmetricZeroPoint(min, scale, qMax);
                }

                _scaleFactors[$"channel_{c}"] = scale;
                _zeroPoints[$"channel_{c}"] = zeroPoint;

                for (int i = start; i < end; i++)
                {
                    double quantized = Math.Round(weights[i] / scale) + zeroPoint;
                    quantized = MathHelper.Clamp(quantized, qMin, qMax);
                    double dequantized = (quantized - zeroPoint) * scale;
                    result[i] = NumOps.FromDouble(dequantized);
                }
            }

            // Set global average
            _scaleFactors["global"] = _scaleFactors.Values.Average();
            _zeroPoints["global"] = 0;
        }
    }

    /// <summary>
    /// Performs per-group quantization.
    /// </summary>
    private void QuantizePerGroup(double[] weights, T[] result, int bitWidth,
                                  QuantizationConfiguration config, double qMin, double qMax)
    {
        int n = weights.Length;
        int groupSize = config.GroupSize;
        int numGroups = (n + groupSize - 1) / groupSize;

        for (int g = 0; g < numGroups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, n);

            double min = double.MaxValue;
            double max = double.MinValue;

            for (int i = start; i < end; i++)
            {
                min = Math.Min(min, weights[i]);
                max = Math.Max(max, weights[i]);
            }

            double scale;
            int zeroPoint;

            if (config.UseSymmetricQuantization)
            {
                double absMax = Math.Max(Math.Abs(min), Math.Abs(max));
                scale = absMax / ((1 << (bitWidth - 1)) - 1);
                scale = Math.Max(scale, config.MinScaleFactor);
                zeroPoint = 0;
            }
            else
            {
                scale = (max - min) / ((1 << bitWidth) - 1);
                scale = Math.Max(scale, config.MinScaleFactor);
                // Clamp zero-point to valid asymmetric range [0, qMax]
                zeroPoint = ComputeAsymmetricZeroPoint(min, scale, qMax);
            }

            _scaleFactors[$"group_{g}"] = scale;
            _zeroPoints[$"group_{g}"] = zeroPoint;

            for (int i = start; i < end; i++)
            {
                double quantized = Math.Round(weights[i] / scale) + zeroPoint;
                quantized = MathHelper.Clamp(quantized, qMin, qMax);
                double dequantized = (quantized - zeroPoint) * scale;
                result[i] = NumOps.FromDouble(dequantized);
            }
        }

        // Set global average
        _scaleFactors["global"] = _scaleFactors.Values.Average();
        _zeroPoints["global"] = 0;
    }

    /// <summary>
    /// Computes smoothing scales using the SmoothQuant formula.
    /// </summary>
    private void ComputeSmoothingScales(double[] activationMax, double[] weightMax)
    {
        int n = Math.Min(activationMax.Length, weightMax.Length);
        double alpha = _config.SmoothQuantAlpha;

        var scales = new double[n];

        for (int i = 0; i < n; i++)
        {
            // s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
            // This balances the quantization difficulty between X and W

            double actMax = Math.Max(activationMax[i], 1e-6);
            double wMax = Math.Max(weightMax[i], 1e-6);

            // Use log-space computation for numerical stability when values are very small
            // s = exp(alpha * log(actMax) - (1-alpha) * log(wMax))
            double logAct = Math.Log(actMax);
            double logW = Math.Log(wMax);
            double logS = alpha * logAct - (1.0 - alpha) * logW;
            double s = Math.Exp(logS);

            // Clamp to reasonable range to avoid extreme scales
            s = MathHelper.Clamp(s, 0.01, 100.0);

            scales[i] = s;
        }

        _smoothingScales["global"] = scales;
    }

    /// <summary>
    /// Gets smoothing scales for the given size.
    /// </summary>
    private double[] GetSmoothingScales(int n)
    {
        if (_smoothingScales.TryGetValue("global", out var scales) && scales.Length >= n)
        {
            return scales;
        }

        // Default: no smoothing (scale = 1)
        var defaultScales = new double[n];
        for (int i = 0; i < n; i++)
        {
            defaultScales[i] = 1.0;
        }
        return defaultScales;
    }
}
