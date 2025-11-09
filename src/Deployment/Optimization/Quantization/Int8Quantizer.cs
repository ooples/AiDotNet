using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// INT8 quantizer for model optimization.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
public class Int8Quantizer<T> : IQuantizer<T> where T : struct
{
    private readonly Dictionary<string, double> _scaleFactors = new();
    private readonly Dictionary<string, int> _zeroPoints = new();
    private bool _isCalibrated = false;

    /// <inheritdoc/>
    public QuantizationMode Mode => QuantizationMode.Int8;

    /// <inheritdoc/>
    public int BitWidth => 8;

    /// <inheritdoc/>
    public object Quantize(object model, QuantizationConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (!_isCalibrated && config.CalibrationMethod != CalibrationMethod.None)
        {
            throw new InvalidOperationException(
                "Quantizer must be calibrated before quantizing. Call Calibrate() with sample data first.");
        }

        // Clone the model to avoid modifying the original
        var quantizedModel = CloneModel(model);

        // Quantize parameters
        if (model is IParameterizable<T> paramModel)
        {
            var parameters = paramModel.GetParameters();
            var quantizedParams = QuantizeParameters(parameters, config);

            if (quantizedModel is IParameterizable<T> quantizedParamModel)
            {
                quantizedParamModel.SetParameters(quantizedParams);
            }
        }

        return quantizedModel;
    }

    /// <inheritdoc/>
    public void Calibrate(IEnumerable<T[]> calibrationData)
    {
        if (calibrationData == null || !calibrationData.Any())
            throw new ArgumentException("Calibration data cannot be null or empty", nameof(calibrationData));

        // Collect statistics from calibration data
        var samples = calibrationData.ToList();

        // Find min and max values across all samples
        double globalMin = double.MaxValue;
        double globalMax = double.MinValue;

        foreach (var sample in samples)
        {
            foreach (var value in sample)
            {
                var doubleValue = Convert.ToDouble(value);
                if (doubleValue < globalMin) globalMin = doubleValue;
                if (doubleValue > globalMax) globalMax = doubleValue;
            }
        }

        // Calculate scale factor and zero point for symmetric quantization
        var absMax = Math.Max(Math.Abs(globalMin), Math.Abs(globalMax));
        var scaleFactor = absMax / 127.0; // INT8 range: -128 to 127

        // Prevent zero scale when all calibration values are zero or very small
        if (scaleFactor < 1e-10)
            scaleFactor = 1.0;

        // Store calibration results
        _scaleFactors["global"] = scaleFactor;
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

    private T[] QuantizeParameters(T[] parameters, QuantizationConfiguration config)
    {
        var scaleFactor = _scaleFactors.GetValueOrDefault("global", 1.0);
        var zeroPoint = _zeroPoints.GetValueOrDefault("global", 0);

        var quantized = new T[parameters.Length];

        for (int i = 0; i < parameters.Length; i++)
        {
            var value = Convert.ToDouble(parameters[i]);

            // Quantize: q = round(value / scale) + zero_point
            var quantizedValue = Math.Round(value / scaleFactor) + zeroPoint;

            // Clamp to INT8 range
            quantizedValue = Math.Clamp(quantizedValue, -128, 127);

            // Dequantize back to original type for storage
            var dequantizedValue = (quantizedValue - zeroPoint) * scaleFactor;

            quantized[i] = (T)Convert.ChangeType(dequantizedValue, typeof(T));
        }

        return quantized;
    }

    private object CloneModel(object model)
    {
        // If model implements ICloneable, use it
        if (model is ICloneable cloneable)
        {
            return cloneable.Clone();
        }

        // If model implements IModelSerializer, use serialization for cloning
        if (model is IModelSerializer serializer)
        {
            var data = serializer.Serialize();
            var clone = Activator.CreateInstance(model.GetType());

            if (clone is IModelSerializer cloneSerializer)
            {
                cloneSerializer.Deserialize(data);
                return clone;
            }
        }

        throw new NotSupportedException($"Model type {model.GetType().Name} does not support cloning");
    }
}
