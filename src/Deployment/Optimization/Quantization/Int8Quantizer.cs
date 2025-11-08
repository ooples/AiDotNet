using AiDotNet.Interfaces;
using AiDotNet.Deployment.Export;

namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// INT8 quantizer for model optimization.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class Int8Quantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput> where T : struct
{
    private readonly Dictionary<string, double> _scaleFactors = new();
    private readonly Dictionary<string, int> _zeroPoints = new();
    private bool _isCalibrated = false;

    /// <inheritdoc/>
    public QuantizationMode Mode => QuantizationMode.Int8;

    /// <inheritdoc/>
    public int BitWidth => 8;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (!_isCalibrated && config.CalibrationMethod != CalibrationMethod.None)
        {
            throw new InvalidOperationException(
                "Quantizer must be calibrated before quantizing. Call Calibrate() with sample data first.");
        }

        // Get current parameters via IParameterizable<T, TInput, TOutput>
        var parameters = model.GetParameters();

        // Quantize the parameters
        var quantizedParams = QuantizeParameters(parameters, config);

        // Create new model with quantized parameters using WithParameters
        var quantizedModel = model.WithParameters(quantizedParams);

        return quantizedModel;
    }

    /// <inheritdoc/>
    public void Calibrate(IEnumerable<TInput> calibrationData)
    {
        if (calibrationData == null || !calibrationData.Any())
            throw new ArgumentException("Calibration data cannot be null or empty", nameof(calibrationData));

        // Note: In production, this would run forward passes through the model
        // to collect activation statistics for proper INT8 calibration.
        // For now, we set a default scale to prevent divide-by-zero.

        // Set default calibration to prevent zero-scale errors
        var scaleFactor = 0.01; // Default non-zero scale factor
        _scaleFactors["global"] = Math.Max(scaleFactor, 1e-6); // Prevent zero-scale
        _zeroPoints["global"] = 0; // Symmetric quantization uses zero point of 0
        _isCalibrated = true;
    }

    /// <inheritdoc/>
    public double GetScaleFactor(string layerName)
    {
        if (_scaleFactors.TryGetValue(layerName, out var scale))
            return scale;

        return _scaleFactors.GetValueOrDefault("global", 1.0);
    }

    /// <inheritdoc/>
    public int GetZeroPoint(string layerName)
    {
        if (_zeroPoints.TryGetValue(layerName, out var zeroPoint))
            return zeroPoint;

        return _zeroPoints.GetValueOrDefault("global", 0);
    }

    private Vector<T> QuantizeParameters(Vector<T> parameters, QuantizationConfiguration config)
    {
        var scaleFactor = _scaleFactors.GetValueOrDefault("global", 1.0);
        var zeroPoint = _zeroPoints.GetValueOrDefault("global", 0);

        var quantizedValues = new T[parameters.Length];

        for (int i = 0; i < parameters.Length; i++)
        {
            var value = Convert.ToDouble(parameters[i]);

            // Quantize: q = round(value / scale) + zero_point
            var quantizedValue = Math.Round(value / scaleFactor) + zeroPoint;

            // Clamp to INT8 range
            quantizedValue = Math.Clamp(quantizedValue, -128, 127);

            // Dequantize back to original type for storage
            var dequantizedValue = (quantizedValue - zeroPoint) * scaleFactor;

            quantizedValues[i] = (T)Convert.ChangeType(dequantizedValue, typeof(T));
        }

        return new Vector<T>(quantizedValues);
    }
}
