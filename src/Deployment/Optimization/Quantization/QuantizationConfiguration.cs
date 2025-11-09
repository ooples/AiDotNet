using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// Configuration for model quantization.
/// </summary>
public class QuantizationConfiguration
{
    /// <summary>
    /// Gets or sets the quantization mode.
    /// </summary>
    public QuantizationMode Mode { get; set; } = QuantizationMode.Int8;

    /// <summary>
    /// Gets or sets whether to use symmetric quantization (default: true).
    /// </summary>
    public bool UseSymmetricQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use per-channel quantization (default: false).
    /// </summary>
    public bool UsePerChannelQuantization { get; set; } = false;

    /// <summary>
    /// Gets or sets the calibration method.
    /// </summary>
    public CalibrationMethod CalibrationMethod { get; set; } = CalibrationMethod.MinMax;

    /// <summary>
    /// Gets or sets the number of calibration samples to use.
    /// </summary>
    public int NumCalibrationSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets layers to skip during quantization.
    /// </summary>
    public HashSet<string> SkipLayers { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to quantize only weights or both weights and activations.
    /// </summary>
    public bool QuantizeActivations { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum scale factor to prevent underflow.
    /// </summary>
    public double MinScaleFactor { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum scale factor to prevent overflow.
    /// </summary>
    public double MaxScaleFactor { get; set; } = 1e6;

    /// <summary>
    /// Gets or sets whether to use quantization-aware training (QAT).
    /// </summary>
    public bool UseQuantizationAwareTraining { get; set; } = false;

    /// <summary>
    /// Gets or sets the percentile to use for histogram-based calibration.
    /// </summary>
    public double HistogramPercentile { get; set; } = 99.99;

    /// <summary>
    /// Gets or sets custom quantization parameters per layer.
    /// </summary>
    public Dictionary<string, LayerQuantizationParams> CustomLayerParams { get; set; } = new();

    /// <summary>
    /// Creates a configuration for INT8 quantization.
    /// </summary>
    public static QuantizationConfiguration ForInt8(CalibrationMethod method = CalibrationMethod.MinMax)
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            CalibrationMethod = method,
            UseSymmetricQuantization = true,
            QuantizeActivations = true
        };
    }

    /// <summary>
    /// Creates a configuration for FP16 quantization.
    /// </summary>
    public static QuantizationConfiguration ForFloat16()
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Float16,
            CalibrationMethod = CalibrationMethod.None,
            UseSymmetricQuantization = false,
            QuantizeActivations = true
        };
    }

    /// <summary>
    /// Creates a configuration for dynamic quantization (weights only).
    /// </summary>
    public static QuantizationConfiguration ForDynamic()
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Dynamic,
            CalibrationMethod = CalibrationMethod.None,
            UseSymmetricQuantization = true,
            QuantizeActivations = false
        };
    }
}
