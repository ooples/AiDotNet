using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Deployment.Optimization.Quantization.Calibration;

/// <summary>
/// Holds activation statistics collected during calibration forward passes.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When quantizing a model, we need to know the typical range of values
/// that flow through each layer during inference. This class stores those statistics, such as the
/// minimum, maximum, mean, and standard deviation of activations.</para>
///
/// <para><b>Why This Matters:</b></para>
/// <list type="bullet">
/// <item><description>AWQ needs activation magnitudes to identify important weights</description></item>
/// <item><description>SmoothQuant needs activation ranges to balance quantization difficulty</description></item>
/// <item><description>GPTQ uses activation statistics to minimize reconstruction error</description></item>
/// </list>
///
/// <para><b>Usage:</b> These statistics are collected by running calibration data through the model
/// and observing the intermediate activations at each layer.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations</typeparam>
public class ActivationStatistics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Per-layer activation statistics. Key is layer name, value is statistics for that layer.
    /// </summary>
    public Dictionary<string, LayerActivationStats<T>> LayerStats { get; } = new();

    /// <summary>
    /// Global (flattened) activation magnitudes across all layers, normalized to [0,1].
    /// Used by AWQ for importance scoring.
    /// </summary>
    public double[]? GlobalActivationMagnitudes { get; set; }

    /// <summary>
    /// Global maximum absolute activation values per parameter position.
    /// Used by SmoothQuant for smoothing factor computation.
    /// </summary>
    public double[]? GlobalMaxAbsActivations { get; set; }

    /// <summary>
    /// Number of calibration samples processed.
    /// </summary>
    public int SampleCount { get; set; }

    /// <summary>
    /// Whether the statistics were collected from actual forward passes (true)
    /// or estimated from parameter magnitudes (false).
    /// </summary>
    public bool IsFromRealForwardPasses { get; set; }

    /// <summary>
    /// Warnings generated during calibration (e.g., high failure rate).
    /// </summary>
    public List<string> CalibrationWarnings { get; } = new();
}

/// <summary>
/// Activation statistics for a single layer.
/// </summary>
/// <typeparam name="T">The numeric type for calculations</typeparam>
public class LayerActivationStats<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Layer name or identifier.
    /// </summary>
    public string LayerName { get; set; } = string.Empty;

    /// <summary>
    /// Minimum activation value observed.
    /// </summary>
    public double MinValue { get; set; } = double.MaxValue;

    /// <summary>
    /// Maximum activation value observed.
    /// </summary>
    public double MaxValue { get; set; } = double.MinValue;

    /// <summary>
    /// Maximum absolute activation value observed.
    /// </summary>
    public double MaxAbsValue { get; set; }

    /// <summary>
    /// Running mean of activation values.
    /// </summary>
    public double Mean { get; set; }

    /// <summary>
    /// Running variance of activation values (for standard deviation).
    /// </summary>
    public double Variance { get; set; }

    /// <summary>
    /// Per-channel maximum absolute values (for per-channel quantization).
    /// </summary>
    public double[]? PerChannelMaxAbs { get; set; }

    /// <summary>
    /// Number of samples accumulated.
    /// </summary>
    public int SampleCount { get; set; }

    /// <summary>
    /// Updates statistics with a new batch of activations.
    /// </summary>
    /// <param name="activations">Tensor of activations from forward pass</param>
    public void Update(Tensor<T> activations)
    {
        var dataSpan = activations.Data.Span;
        int n = dataSpan.Length;

        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(dataSpan[i]);
            double absVal = Math.Abs(val);

            MinValue = Math.Min(MinValue, val);
            MaxValue = Math.Max(MaxValue, val);
            MaxAbsValue = Math.Max(MaxAbsValue, absVal);

            // Welford's online algorithm for mean/variance
            SampleCount++;
            double delta = val - Mean;
            Mean += delta / SampleCount;
            double delta2 = val - Mean;
            Variance += delta * delta2;
        }

        // Update per-channel max abs (assume last dimension is channels)
        UpdatePerChannelStats(activations);
    }

    /// <summary>
    /// Updates per-channel maximum absolute values.
    /// </summary>
    private void UpdatePerChannelStats(Tensor<T> activations)
    {
        var shape = activations.Shape;
        if (shape.Length < 2) return;

        int numChannels = shape[^1]; // Last dimension is typically channels
        var dataSpan = activations.Data.Span;

        if (PerChannelMaxAbs == null || PerChannelMaxAbs.Length != numChannels)
        {
            PerChannelMaxAbs = new double[numChannels];
        }

        for (int i = 0; i < dataSpan.Length; i++)
        {
            int channel = i % numChannels;
            double absVal = Math.Abs(NumOps.ToDouble(dataSpan[i]));
            PerChannelMaxAbs[channel] = Math.Max(PerChannelMaxAbs[channel], absVal);
        }
    }

    /// <summary>
    /// Gets the standard deviation of activations.
    /// </summary>
    public double StandardDeviation => SampleCount > 1 ? Math.Sqrt(Variance / (SampleCount - 1)) : 0;
}
