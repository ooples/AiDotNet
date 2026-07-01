using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;


namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// INT4 (4-bit) quantizer for model optimization — the simple MinMax counterpart to
/// <see cref="Int8Quantizer{T, TInput, TOutput}"/>, used for QLoRA-style low-bit base weights.
/// </summary>
/// <typeparam name="T">The numeric type used in the model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> 4-bit quantization snaps each weight to one of only 16 levels, so the
/// base model is ~8x smaller than FP32. That loses precision, which is exactly why QLoRA keeps a
/// small full-precision LoRA adapter on top — the adapter absorbs what the 4-bit base can't represent.</para>
/// <para>Symmetric quantization on the signed 4-bit grid [-8, 7] with <c>scale = absMax / 7</c>. Like
/// <see cref="Int8Quantizer{T, TInput, TOutput}"/> this is simulated (fake) quantization: values are
/// rounded onto the 4-bit grid and stored dequantized in <typeparamref name="T"/>, so the precision
/// loss is modeled without changing the parameter container.</para>
/// </remarks>
public class Int4Quantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
{
    private readonly Dictionary<string, double> _scaleFactors = new();
    private readonly Dictionary<string, int> _zeroPoints = new();
    private bool _isCalibrated = false;

    // Signed symmetric 4-bit grid: 16 levels, magnitude up to 7 (mirrors INT8's 127).
    private const int MaxLevel = 7;
    private const int MinLevel = -8;

    /// <inheritdoc/>
    public QuantizationMode Mode => QuantizationMode.Int4;

    /// <inheritdoc/>
    public int BitWidth => 4;

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

        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();
        var quantizedParams = QuantizeParameters(parameters, config);
        var quantizedModel = InterfaceGuard.Parameterizable(model).WithParameters(quantizedParams);

        return quantizedModel;
    }

    /// <inheritdoc/>
    public void Calibrate(IFullModel<T, TInput, TOutput> model, IEnumerable<TInput> calibrationData)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (calibrationData == null || !calibrationData.Any())
            throw new ArgumentException("Calibration data cannot be null or empty", nameof(calibrationData));

        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();
        var paramValues = new List<double>();
        for (int i = 0; i < parameters.Length; i++)
        {
            paramValues.Add(Convert.ToDouble(parameters[i]));
        }

        double min = paramValues.Min();
        double max = paramValues.Max();
        if (Math.Abs(max - min) < 1e-10)
        {
            min -= 0.1;
            max += 0.1;
        }

        // Symmetric quantization on the 4-bit grid.
        double absMax = Math.Max(Math.Abs(min), Math.Abs(max));
        double scale = Math.Max(absMax / MaxLevel, 1e-6);

        _scaleFactors["global"] = scale;
        _zeroPoints["global"] = 0; // symmetric

        if (model is IModel<TInput, TOutput, object> inferenceModel)
        {
            var activations = new List<double>();
            foreach (var sample in calibrationData.Take(100))
            {
                try
                {
                    var output = inferenceModel.Predict(sample);
                    if (output is T[] outputArray)
                    {
                        foreach (var val in outputArray)
                            activations.Add(Convert.ToDouble(val));
                    }
                }
                catch
                {
                    continue;
                }
            }

            if (activations.Count > 0)
            {
                double actAbsMax = Math.Max(Math.Abs(activations.Min()), Math.Abs(activations.Max()));
                double actScale = Math.Max(actAbsMax / MaxLevel, 1e-6);
                _scaleFactors["global"] = (scale + actScale) / 2.0;
            }
        }

        _isCalibrated = true;
    }

    /// <inheritdoc/>
    public double GetScaleFactor(string layerName) =>
        _scaleFactors.TryGetValue(layerName, out var scale) ? scale
            : (_scaleFactors.TryGetValue("global", out var sf1) ? sf1 : 1.0);

    /// <inheritdoc/>
    public int GetZeroPoint(string layerName) =>
        _zeroPoints.TryGetValue(layerName, out var zeroPoint) ? zeroPoint
            : (_zeroPoints.TryGetValue("global", out var zp1) ? zp1 : 0);

    private Vector<T> QuantizeParameters(Vector<T> parameters, QuantizationConfiguration config)
    {
        double scaleFactor;
        if (_scaleFactors.TryGetValue("global", out var sf1))
        {
            scaleFactor = sf1;
        }
        else
        {
            // No external calibration — derive a symmetric scale from the weights themselves
            // (weight-only MinMax), so a bare Mode=Int4 still quantizes sensibly instead of using 1.0.
            double absMax = 1e-6;
            for (int i = 0; i < parameters.Length; i++)
                absMax = Math.Max(absMax, Math.Abs(Convert.ToDouble(parameters[i])));
            scaleFactor = absMax / MaxLevel;
        }
        var zeroPoint = _zeroPoints.TryGetValue("global", out var zp1) ? zp1 : 0;

        var quantizedValues = new T[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            var value = Convert.ToDouble(parameters[i]);

            // Quantize onto the 4-bit grid, then dequantize for storage (simulated quantization).
            var quantizedValue = Math.Round(value / scaleFactor) + zeroPoint;
            quantizedValue = MathHelper.Clamp(quantizedValue, MinLevel, MaxLevel);
            var dequantizedValue = (quantizedValue - zeroPoint) * scaleFactor;

            quantizedValues[i] = (T)Convert.ChangeType(dequantizedValue, typeof(T));
        }

        return new Vector<T>(quantizedValues);
    }
}
