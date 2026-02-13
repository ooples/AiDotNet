using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Deployment.Optimization.Quantization.Training;

/// <summary>
/// Quantization-Aware Training (QAT) hook that applies fake quantization during training.
/// Simulates quantization effects in the forward pass while allowing gradients to flow through.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> QAT trains the model with quantization simulation so it learns
/// to be robust to low-precision inference. This hook inserts "fake quantization" operations
/// that quantize and immediately dequantize values, simulating the precision loss.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Forward pass: Apply fake quantization (quantize then dequantize)</description></item>
/// <item><description>Backward pass: Use Straight-Through Estimator (STE) - pass gradients unchanged</description></item>
/// <item><description>The model learns to produce outputs that are robust to quantization</description></item>
/// </list>
///
/// <para><b>Key Components:</b></para>
/// <list type="bullet">
/// <item><description><b>Fake Quantization:</b> Quantize then immediately dequantize to simulate precision loss</description></item>
/// <item><description><b>Straight-Through Estimator:</b> Pass gradients through the non-differentiable round() operation</description></item>
/// <item><description><b>Learnable scale:</b> Optionally learn optimal quantization scales during training</description></item>
/// </list>
///
/// <para><b>Reference:</b> Jacob et al., "Quantization and Training of Neural Networks for
/// Efficient Integer-Arithmetic-Only Inference" (2018)</para>
/// </remarks>
/// <typeparam name="T">The numeric type used in the model</typeparam>
public class QATTrainingHook<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly QuantizationConfiguration _config;
    private readonly Dictionary<string, QuantizationState> _layerStates = new();
    private int _currentEpoch;
    private bool _quantizationEnabled;

    /// <summary>
    /// Gets whether quantization is currently enabled.
    /// </summary>
    public bool IsQuantizationEnabled => _quantizationEnabled;

    /// <summary>
    /// Gets the current epoch number.
    /// </summary>
    public int CurrentEpoch => _currentEpoch;

    /// <summary>
    /// Initializes a new instance of the QATTrainingHook.
    /// </summary>
    /// <param name="config">Quantization configuration</param>
    public QATTrainingHook(QuantizationConfiguration config)
    {
        Guard.NotNull(config);
        _config = config;
        _quantizationEnabled = config.QATWarmupEpochs == 0;
    }

    /// <summary>
    /// Called at the start of each training epoch to manage warmup and quantization state.
    /// </summary>
    /// <param name="epoch">Current epoch number (0-indexed)</param>
    public void OnEpochStart(int epoch)
    {
        _currentEpoch = epoch;

        // Enable quantization after warmup epochs
        if (!_quantizationEnabled && epoch >= _config.QATWarmupEpochs)
        {
            _quantizationEnabled = true;
        }
    }

    /// <summary>
    /// Applies fake quantization to weights during the forward pass.
    /// </summary>
    /// <param name="weights">Original weights</param>
    /// <param name="layerName">Name of the layer for tracking state</param>
    /// <returns>Fake-quantized weights (quantized then dequantized)</returns>
    public Vector<T> ApplyFakeQuantization(Vector<T> weights, string layerName)
    {
        if (!_quantizationEnabled)
        {
            return weights; // During warmup, return original weights
        }

        // Get or create state for this layer
        if (!_layerStates.TryGetValue(layerName, out var state))
        {
            state = InitializeLayerState(weights, layerName);
            _layerStates[layerName] = state;
        }

        return FakeQuantize(weights, state);
    }

    /// <summary>
    /// Applies fake quantization to activations during the forward pass.
    /// </summary>
    /// <param name="activations">Original activations</param>
    /// <param name="layerName">Name of the layer</param>
    /// <returns>Fake-quantized activations</returns>
    public Vector<T> ApplyFakeQuantizationToActivations(Vector<T> activations, string layerName)
    {
        if (!_quantizationEnabled || !_config.QuantizeActivations)
        {
            return activations;
        }

        string activationKey = $"{layerName}_activation";
        if (!_layerStates.TryGetValue(activationKey, out var state))
        {
            // Use ActivationBitWidth for activations, not EffectiveBitWidth
            state = InitializeLayerState(activations, activationKey, _config.ActivationBitWidth);
            _layerStates[activationKey] = state;
        }

        // Update activation statistics (running min/max for calibration)
        UpdateActivationStatistics(activations, state);

        return FakeQuantize(activations, state);
    }

    /// <summary>
    /// Applies the Straight-Through Estimator for gradient computation.
    /// The gradient passes through the quantization operation unchanged.
    /// </summary>
    /// <param name="gradOutput">Gradient from the next layer</param>
    /// <param name="quantizedWeights">The fake-quantized weights from forward pass</param>
    /// <param name="originalWeights">Original weights before quantization</param>
    /// <returns>Gradient to pass to previous layer</returns>
    public Vector<T> StraightThroughEstimator(Vector<T> gradOutput, Vector<T> quantizedWeights, Vector<T> originalWeights)
    {
        // STE: Pass gradient through unchanged
        // This is the key insight - the rounding operation is non-differentiable,
        // but we pretend it's the identity function for gradients
        return gradOutput;
    }

    /// <summary>
    /// Updates quantization scales based on observed statistics (for learnable scales).
    /// </summary>
    /// <param name="layerName">Layer name</param>
    /// <param name="learningRate">Learning rate for scale updates</param>
    public void UpdateScales(string layerName, double learningRate = 0.001)
    {
        if (!_layerStates.TryGetValue(layerName, out var state))
        {
            return;
        }

        // LSQ-style scale update: scale gradient = sum(grad * round_error * sign)
        // For simplicity, we use exponential moving average of observed ranges
        double observedRange = state.MaxValue - state.MinValue;
        double targetRange = state.QuantMax - state.QuantMin;

        if (observedRange > 0 && targetRange > 0)
        {
            double optimalScale = observedRange / targetRange;
            state.Scale = state.Scale * (1 - learningRate) + optimalScale * learningRate;
            state.Scale = Math.Max(state.Scale, _config.MinScaleFactor);
        }
    }

    /// <summary>
    /// Gets the current quantization state for a layer.
    /// </summary>
    public QuantizationState? GetLayerState(string layerName)
    {
        return _layerStates.TryGetValue(layerName, out var state) ? state : null;
    }

    /// <summary>
    /// Resets the quantization state for all layers.
    /// </summary>
    public void Reset()
    {
        _layerStates.Clear();
        _currentEpoch = 0;
        _quantizationEnabled = _config.QATWarmupEpochs == 0;
    }

    /// <summary>
    /// Initializes quantization state for a layer.
    /// </summary>
    /// <param name="weights">Weight or activation values to analyze</param>
    /// <param name="layerName">Layer name for identification</param>
    /// <param name="bitWidthOverride">Optional bit width override (for activations using ActivationBitWidth)</param>
    private QuantizationState InitializeLayerState(Vector<T> weights, string layerName, int? bitWidthOverride = null)
    {
        int bitWidth = bitWidthOverride ?? _config.EffectiveBitWidth;

        // Compute initial scale from weights
        double minVal = double.MaxValue;
        double maxVal = double.MinValue;

        for (int i = 0; i < weights.Length; i++)
        {
            double val = Convert.ToDouble(weights[i]);
            minVal = Math.Min(minVal, val);
            maxVal = Math.Max(maxVal, val);
        }

        double scale;
        int zeroPoint;
        double qMin, qMax;

        if (_config.UseSymmetricQuantization)
        {
            double absMax = Math.Max(Math.Abs(minVal), Math.Abs(maxVal));
            qMin = -(1 << (bitWidth - 1));
            qMax = (1 << (bitWidth - 1)) - 1;
            scale = absMax / qMax;
            zeroPoint = 0;
        }
        else
        {
            qMin = 0;
            qMax = (1 << bitWidth) - 1;
            scale = (maxVal - minVal) / qMax;
            scale = Math.Max(scale, _config.MinScaleFactor);
            zeroPoint = (int)Math.Round(-minVal / scale);
        }

        if (_config.UseSymmetricQuantization)
        {
            scale = Math.Max(scale, _config.MinScaleFactor);
        }

        return new QuantizationState
        {
            LayerName = layerName,
            Scale = scale,
            ZeroPoint = zeroPoint,
            BitWidth = bitWidth,
            MinValue = minVal,
            MaxValue = maxVal,
            QuantMin = qMin,
            QuantMax = qMax,
            IsSymmetric = _config.UseSymmetricQuantization,
            SamplesObserved = 1
        };
    }

    /// <summary>
    /// Updates activation statistics using exponential moving average.
    /// </summary>
    private void UpdateActivationStatistics(Vector<T> activations, QuantizationState state)
    {
        double minVal = double.MaxValue;
        double maxVal = double.MinValue;

        for (int i = 0; i < activations.Length; i++)
        {
            double val = Convert.ToDouble(activations[i]);
            minVal = Math.Min(minVal, val);
            maxVal = Math.Max(maxVal, val);
        }

        // Exponential moving average for stability
        double momentum = 0.1;
        state.MinValue = state.MinValue * (1 - momentum) + minVal * momentum;
        state.MaxValue = state.MaxValue * (1 - momentum) + maxVal * momentum;
        state.SamplesObserved++;

        // Update scale based on observed range
        double range = state.MaxValue - state.MinValue;
        double quantRange = state.QuantMax - state.QuantMin;

        if (range > 0 && quantRange > 0)
        {
            state.Scale = range / quantRange;
            state.Scale = Math.Max(state.Scale, _config.MinScaleFactor);

            if (!state.IsSymmetric)
            {
                state.ZeroPoint = (int)Math.Round(-state.MinValue / state.Scale);
            }
        }
    }

    /// <summary>
    /// Applies fake quantization to a vector.
    /// </summary>
    private Vector<T> FakeQuantize(Vector<T> input, QuantizationState state)
    {
        int n = input.Length;
        var result = new T[n];

        double scale = state.Scale;
        int zeroPoint = state.ZeroPoint;
        double qMin = state.QuantMin;
        double qMax = state.QuantMax;

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(input[i]);

            // Quantize: q = clamp(round(x / scale) + zero_point, qMin, qMax)
            double quantized = Math.Round(val / scale) + zeroPoint;
            quantized = MathHelper.Clamp(quantized, qMin, qMax);

            // Dequantize: x' = (q - zero_point) * scale
            double dequantized = (quantized - zeroPoint) * scale;

            result[i] = NumOps.FromDouble(dequantized);
        }

        return new Vector<T>(result);
    }
}

/// <summary>
/// Stores quantization state for a layer during QAT.
/// </summary>
public class QuantizationState
{
    /// <summary>
    /// Name of the layer.
    /// </summary>
    public string LayerName { get; set; } = string.Empty;

    /// <summary>
    /// Quantization scale factor.
    /// </summary>
    public double Scale { get; set; } = 1.0;

    /// <summary>
    /// Zero point for asymmetric quantization.
    /// </summary>
    public int ZeroPoint { get; set; }

    /// <summary>
    /// Bit width for quantization.
    /// </summary>
    public int BitWidth { get; set; } = 8;

    /// <summary>
    /// Observed minimum value.
    /// </summary>
    public double MinValue { get; set; }

    /// <summary>
    /// Observed maximum value.
    /// </summary>
    public double MaxValue { get; set; }

    /// <summary>
    /// Minimum quantized value (e.g., -128 for INT8 symmetric).
    /// </summary>
    public double QuantMin { get; set; }

    /// <summary>
    /// Maximum quantized value (e.g., 127 for INT8 symmetric).
    /// </summary>
    public double QuantMax { get; set; }

    /// <summary>
    /// Whether using symmetric quantization.
    /// </summary>
    public bool IsSymmetric { get; set; }

    /// <summary>
    /// Number of samples observed for statistics.
    /// </summary>
    public long SamplesObserved { get; set; }
}
