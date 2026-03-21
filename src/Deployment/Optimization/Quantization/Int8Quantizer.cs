using AiDotNet.Enums;
using AiDotNet.Interfaces;


namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// INT8 quantizer for model optimization.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Int8Quantizer provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class Int8Quantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
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
    public void Calibrate(IFullModel<T, TInput, TOutput> model, IEnumerable<TInput> calibrationData)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (calibrationData == null || !calibrationData.Any())
            throw new ArgumentException("Calibration data cannot be null or empty", nameof(calibrationData));

        // Collect parameter statistics
        var parameters = model.GetParameters();
        var paramValues = new List<double>();

        for (int i = 0; i < parameters.Length; i++)
        {
            paramValues.Add(Convert.ToDouble(parameters[i]));
        }

        // Compute scale and zero point using MinMax method
        double min = paramValues.Min();
        double max = paramValues.Max();

        // Ensure min and max are different
        if (Math.Abs(max - min) < 1e-10)
        {
            min = min - 0.1;
            max = max + 0.1;
        }

        // For symmetric quantization (INT8: -128 to 127)
        double absMax = Math.Max(Math.Abs(min), Math.Abs(max));
        double scale = absMax / 127.0;

        // Prevent zero scale
        scale = Math.Max(scale, 1e-6);

        _scaleFactors["global"] = scale;
        _zeroPoints["global"] = 0; // Symmetric quantization

        // If model supports inference (IModel interface), collect activation statistics
        if (model is IModel<TInput, TOutput, object> inferenceModel)
        {
            var activations = new List<double>();
            int samplesProcessed = 0;

            foreach (var sample in calibrationData.Take(100)) // Limit calibration samples
            {
                try
                {
                    var output = inferenceModel.Predict(sample);

                    // If output is an array type, collect statistics from it
                    if (output is T[] outputArray)
                    {
                        foreach (var val in outputArray)
                        {
                            activations.Add(Convert.ToDouble(val));
                        }
                    }

                    samplesProcessed++;
                }
                catch
                {
                    // Skip samples that fail inference
                    continue;
                }
            }

            // Compute activation statistics if we collected any
            if (activations.Count > 0)
            {
                double actMin = activations.Min();
                double actMax = activations.Max();
                double actAbsMax = Math.Max(Math.Abs(actMin), Math.Abs(actMax));
                double actScale = actAbsMax / 127.0;

                // Use average of parameter and activation scales
                _scaleFactors["global"] = (scale + Math.Max(actScale, 1e-6)) / 2.0;
            }
        }

        _isCalibrated = true;
    }

    /// <inheritdoc/>
    public double GetScaleFactor(string layerName)
    {
        if (_scaleFactors.TryGetValue(layerName, out var scale))
            return scale;

        return (_scaleFactors.TryGetValue("global", out var sf1) ? sf1 : 1.0);
    }

    /// <inheritdoc/>
    public int GetZeroPoint(string layerName)
    {
        if (_zeroPoints.TryGetValue(layerName, out var zeroPoint))
            return zeroPoint;

        return (_zeroPoints.TryGetValue("global", out var zp1) ? zp1 : 0);
    }

    private Vector<T> QuantizeParameters(Vector<T> parameters, QuantizationConfiguration config)
    {
        var scaleFactor = (_scaleFactors.TryGetValue("global", out var sf1) ? sf1 : 1.0);
        var zeroPoint = (_zeroPoints.TryGetValue("global", out var zp1) ? zp1 : 0);

        var quantizedValues = new T[parameters.Length];

        for (int i = 0; i < parameters.Length; i++)
        {
            var value = Convert.ToDouble(parameters[i]);

            // Quantize: q = round(value / scale) + zero_point
            var quantizedValue = Math.Round(value / scaleFactor) + zeroPoint;

            // Clamp to INT8 range
            quantizedValue = MathHelper.Clamp(quantizedValue, -128, 127);

            // Dequantize back to original type for storage
            var dequantizedValue = (quantizedValue - zeroPoint) * scaleFactor;

            quantizedValues[i] = (T)Convert.ChangeType(dequantizedValue, typeof(T));
        }

        return new Vector<T>(quantizedValues);
    }
}
