using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using AiDotNet.Deployment.Optimization.Quantization.Calibration;

namespace AiDotNet.Deployment.Optimization.Quantization.Strategies;

/// <summary>
/// AWQ (Activation-aware Weight Quantization) - protects important weights based on activation magnitudes.
/// Particularly effective for very large models (70B+ parameters).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> AWQ observes which weights are "activated" most strongly during
/// inference and protects those from aggressive quantization. It's like knowing which roads
/// are most traveled and keeping those in better condition.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Run calibration data through the model to collect activation statistics</description></item>
/// <item><description>Compute importance scores for each weight based on activation magnitudes</description></item>
/// <item><description>Apply per-channel scaling to protect important weights before quantization</description></item>
/// <item><description>Compensate by scaling the next layer's weights in the opposite direction</description></item>
/// </list>
///
/// <para><b>Key Features:</b></para>
/// <list type="bullet">
/// <item><description>Activation-aware: Protects weights that matter most for model outputs</description></item>
/// <item><description>Stable at 4-bit: Particularly good for 70B+ parameter models</description></item>
/// <item><description>Per-channel scaling: Adjusts quantization sensitivity per output channel</description></item>
/// <item><description>Search-based optimization: Finds optimal scaling factors</description></item>
/// </list>
///
/// <para><b>Reference:</b> Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
/// Compression and Acceleration" (2024)</para>
/// </remarks>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class AWQQuantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly QuantizationConfiguration _config;
    private readonly Dictionary<string, double> _scaleFactors = new();
    private readonly Dictionary<string, int> _zeroPoints = new();
    private readonly Dictionary<string, double[]> _activationScales = new();
    private ActivationStatistics<T>? _activationStats;
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
    /// through the model. Returns false if calibration fell back to parameter-based estimation
    /// (which is less accurate but works for all model types).
    /// </remarks>
    public bool UsedRealForwardPasses => _activationStats?.IsFromRealForwardPasses ?? false;

    /// <summary>
    /// Initializes a new instance of the AWQQuantizer.
    /// </summary>
    /// <param name="config">Quantization configuration</param>
    public AWQQuantizer(QuantizationConfiguration? config = null)
    {
        _config = config ?? QuantizationConfiguration.ForAWQ();

        if (_config.Strategy != QuantizationStrategy.AWQ)
        {
            _config.Strategy = QuantizationStrategy.AWQ;
        }
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        if (!_isCalibrated && config.CalibrationMethod != CalibrationMethod.None)
        {
            throw new InvalidOperationException(
                "AWQ requires calibration data to compute activation importance. Call Calibrate() first.");
        }

        var parameters = model.GetParameters();
        var quantizedParams = QuantizeWithAWQ(parameters, config);
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

        // Compute activation-based importance scales from the collected statistics
        ComputeActivationScalesFromStats();

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
    /// Quantizes parameters using the AWQ algorithm with activation-aware scaling.
    /// </summary>
    private Vector<T> QuantizeWithAWQ(Vector<T> parameters, QuantizationConfiguration config)
    {
        int n = parameters.Length;
        int groupSize = config.Granularity == QuantizationGranularity.PerGroup ? config.GroupSize : n;
        int numGroups = (n + groupSize - 1) / groupSize;
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

        // Get activation-based importance scales
        double[] activationScales = GetActivationScales(n);

        // Compute protection mask (top X% of important weights)
        bool[] protectedMask = ComputeProtectionMask(weights, activationScales, config.AWQProtectionPercentage);

        // Process each group
        for (int g = 0; g < numGroups; g++)
        {
            int groupStart = g * groupSize;
            int groupEnd = Math.Min(groupStart + groupSize, n);

            // Compute optimal scaling for this group using grid search
            double optimalScale = FindOptimalScale(weights, activationScales, protectedMask,
                                                   groupStart, groupEnd, bitWidth, config);

            // Apply AWQ scaling: scale important weights up, then quantize, then scale back down
            double[] scaledWeights = new double[groupEnd - groupStart];
            for (int i = groupStart; i < groupEnd; i++)
            {
                double importance = activationScales[i];
                double scaleFactor = 1.0 + (importance * optimalScale);

                // Scale up important weights before quantization
                scaledWeights[i - groupStart] = weights[i] * scaleFactor;
            }

            // Compute quantization scale for this group
            double groupMin = scaledWeights.Min();
            double groupMax = scaledWeights.Max();

            double scale;
            int zeroPoint;

            if (config.UseSymmetricQuantization)
            {
                double absMax = Math.Max(Math.Abs(groupMin), Math.Abs(groupMax));
                scale = absMax / ((1 << (bitWidth - 1)) - 1);
                scale = Math.Max(scale, config.MinScaleFactor);
                zeroPoint = 0;
            }
            else
            {
                scale = (groupMax - groupMin) / ((1 << bitWidth) - 1);
                scale = Math.Max(scale, config.MinScaleFactor);
                zeroPoint = (int)Math.Round(-groupMin / scale);
            }

            _scaleFactors[$"group_{g}"] = scale;
            _zeroPoints[$"group_{g}"] = zeroPoint;

            // Quantize and dequantize
            for (int i = groupStart; i < groupEnd; i++)
            {
                double importance = activationScales[i];
                double scaleFactor = 1.0 + (importance * optimalScale);

                // Scaled weight
                double scaledW = scaledWeights[i - groupStart];

                // Quantize
                double quantized = Math.Round(scaledW / scale) + zeroPoint;
                quantized = MathHelper.Clamp(quantized, qMin, qMax);

                // Dequantize
                double dequantized = (quantized - zeroPoint) * scale;

                // Scale back down
                dequantized /= scaleFactor;

                result[i] = NumOps.FromDouble(dequantized);
            }
        }

        // Store global scale
        if (_scaleFactors.Count > 0)
        {
            _scaleFactors["global"] = _scaleFactors.Values.Average();
            _zeroPoints["global"] = 0;
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes activation-based importance scales from the collected activation statistics.
    /// </summary>
    /// <remarks>
    /// This method processes the activation statistics collected by CalibrationHelper
    /// (which uses real forward passes when possible) to compute importance scales.
    /// </remarks>
    private void ComputeActivationScalesFromStats()
    {
        if (_activationStats == null) return;

        // Use the global activation magnitudes computed by CalibrationHelper
        if (_activationStats.GlobalActivationMagnitudes != null &&
            _activationStats.GlobalActivationMagnitudes.Length > 0)
        {
            _activationScales["global"] = _activationStats.GlobalActivationMagnitudes;
            return;
        }

        // Fallback: compute from layer statistics if global not available
        if (_activationStats.LayerStats.Count > 0)
        {
            // Estimate parameter count from first layer with stats
            int estimatedSize = _activationStats.LayerStats.Values
                .Select(s => s.PerChannelMaxAbs?.Length ?? 0)
                .Where(n => n > 0)
                .FirstOrDefault();

            if (estimatedSize == 0) estimatedSize = 1000; // Default estimate

            var avgActivations = new double[estimatedSize];
            double maxActivation = 0;

            // Combine layer statistics
            int layerIndex = 0;
            foreach (var layerStat in _activationStats.LayerStats.Values)
            {
                // Use per-channel max abs if available, otherwise use layer max
                if (layerStat.PerChannelMaxAbs != null)
                {
                    int channelCount = layerStat.PerChannelMaxAbs.Length;
                    int startIdx = (layerIndex * estimatedSize) / Math.Max(1, _activationStats.LayerStats.Count);

                    for (int c = 0; c < channelCount && startIdx + c < estimatedSize; c++)
                    {
                        avgActivations[startIdx + c] = layerStat.PerChannelMaxAbs[c];
                        maxActivation = Math.Max(maxActivation, layerStat.PerChannelMaxAbs[c]);
                    }
                }
                else
                {
                    // Fill with layer average
                    maxActivation = Math.Max(maxActivation, layerStat.MaxAbsValue);
                }
                layerIndex++;
            }

            // Normalize to [0, 1]
            if (maxActivation > 0)
            {
                for (int i = 0; i < estimatedSize; i++)
                {
                    avgActivations[i] /= maxActivation;
                }
            }

            _activationScales["global"] = avgActivations;
        }
    }

    /// <summary>
    /// Gets activation scales for quantization.
    /// </summary>
    private double[] GetActivationScales(int n)
    {
        if (_activationScales.TryGetValue("global", out var scales) && scales.Length >= n)
        {
            return scales;
        }

        // Default: uniform importance
        var defaultScales = new double[n];
        for (int i = 0; i < n; i++)
        {
            defaultScales[i] = 1.0;
        }
        return defaultScales;
    }

    /// <summary>
    /// Computes which weights should be protected based on activation importance.
    /// </summary>
    private bool[] ComputeProtectionMask(double[] weights, double[] activationScales, double protectionPercentage)
    {
        int n = weights.Length;
        var mask = new bool[n];

        // Compute importance = |weight| * activation_scale
        var importance = new (int index, double score)[n];
        for (int i = 0; i < n; i++)
        {
            double actScale = i < activationScales.Length ? activationScales[i] : 1.0;
            importance[i] = (i, Math.Abs(weights[i]) * actScale);
        }

        // Sort by importance descending
        var sorted = importance.OrderByDescending(x => x.score).ToArray();

        // Mark top X% as protected
        int numProtected = (int)Math.Ceiling(n * protectionPercentage / 100.0);
        for (int i = 0; i < numProtected && i < sorted.Length; i++)
        {
            mask[sorted[i].index] = true;
        }

        return mask;
    }

    /// <summary>
    /// Finds the optimal AWQ scaling factor using grid search.
    /// </summary>
    private double FindOptimalScale(double[] weights, double[] activationScales, bool[] protectedMask,
                                    int start, int end, int bitWidth, QuantizationConfiguration config)
    {
        double qMin = config.UseSymmetricQuantization ? -(1 << (bitWidth - 1)) : 0;
        double qMax = config.UseSymmetricQuantization ? (1 << (bitWidth - 1)) - 1 : (1 << bitWidth) - 1;

        double bestScale = 0;
        double bestError = double.MaxValue;

        // Grid search over scaling factors (configurable via AWQScaleSearchOptions)
        double[] scaleOptions = config.AWQScaleSearchOptions.Length > 0
            ? config.AWQScaleSearchOptions
            : [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0];

        foreach (double s in scaleOptions)
        {
            double totalError = 0;

            // Compute scaled weights
            var scaledWeights = new double[end - start];
            for (int i = start; i < end; i++)
            {
                double importance = i < activationScales.Length ? activationScales[i] : 1.0;
                double scaleFactor = 1.0 + (importance * s);
                scaledWeights[i - start] = weights[i] * scaleFactor;
            }

            // Compute quantization scale
            double groupMin = scaledWeights.Min();
            double groupMax = scaledWeights.Max();

            double scale;
            int zeroPoint;

            if (config.UseSymmetricQuantization)
            {
                double absMax = Math.Max(Math.Abs(groupMin), Math.Abs(groupMax));
                scale = absMax / ((1 << (bitWidth - 1)) - 1);
                scale = Math.Max(scale, config.MinScaleFactor);
                zeroPoint = 0;
            }
            else
            {
                scale = (groupMax - groupMin) / ((1 << bitWidth) - 1);
                scale = Math.Max(scale, config.MinScaleFactor);
                zeroPoint = (int)Math.Round(-groupMin / scale);
            }

            // Compute reconstruction error
            for (int i = start; i < end; i++)
            {
                double importance = i < activationScales.Length ? activationScales[i] : 1.0;
                double scaleFactor = 1.0 + (importance * s);

                double scaledW = scaledWeights[i - start];

                // Quantize
                double quantized = Math.Round(scaledW / scale) + zeroPoint;
                quantized = MathHelper.Clamp(quantized, qMin, qMax);

                // Dequantize
                double dequantized = (quantized - zeroPoint) * scale;
                dequantized /= scaleFactor;

                // Weight error by activation importance (AWQ key insight)
                double error = Math.Abs(weights[i] - dequantized);
                if (protectedMask[i])
                {
                    error *= 10.0; // Higher penalty for protected weights
                }
                error *= (importance + 0.1); // Weight by activation

                totalError += error * error;
            }

            if (totalError < bestError)
            {
                bestError = totalError;
                bestScale = s;
            }
        }

        return bestScale;
    }
}
