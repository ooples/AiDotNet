using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// FP16 (half-precision) quantizer for model optimization.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Float16Quantizer provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class Float16Quantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public QuantizationMode Mode => QuantizationMode.Float16;

    /// <inheritdoc/>
    public int BitWidth => 16;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        // Get current parameters via IParameterizable<T, TInput, TOutput>
        var parameters = model.GetParameters();

        // Quantize to FP16 and back
        var quantizedParams = QuantizeParametersToFp16(parameters);

        // Create new model with quantized parameters using WithParameters
        var quantizedModel = model.WithParameters(quantizedParams);

        return quantizedModel;
    }

    /// <inheritdoc/>
    public void Calibrate(IFullModel<T, TInput, TOutput> model, IEnumerable<TInput> calibrationData)
    {
        // FP16 quantization doesn't require calibration as it's a direct type conversion
        // This is a no-op
    }

    /// <inheritdoc/>
    public double GetScaleFactor(string layerName)
    {
        // FP16 doesn't use scale factors
        return 1.0;
    }

    /// <inheritdoc/>
    public int GetZeroPoint(string layerName)
    {
        // FP16 doesn't use zero points
        return 0;
    }

    private Vector<T> QuantizeParametersToFp16(Vector<T> parameters)
    {
        var quantizedValues = new T[parameters.Length];

        for (int i = 0; i < parameters.Length; i++)
        {
            var value = Convert.ToDouble(parameters[i]);

            // Convert to FP16 (Half) and back to simulate precision loss
            var fp16Value = ConvertToFloat16(value);
            var fp32Value = ConvertFromFloat16(fp16Value);

            quantizedValues[i] = (T)Convert.ChangeType(fp32Value, typeof(T));
        }

        return new Vector<T>(quantizedValues);
    }

    private ushort ConvertToFloat16(double value)
    {
        // Convert double to float first
        var floatValue = (float)value;

        // Get the bits
        var bytes = BitConverter.GetBytes(floatValue);
        var bits = BitConverter.ToInt32(bytes, 0);

        // Extract sign, exponent, and mantissa
        var sign = (bits >> 31) & 0x1;
        var exponent = (bits >> 23) & 0xFF;
        var mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if (exponent == 0xFF)
        {
            // Infinity or NaN - preserve mantissa bits for NaN
            ushort nanMantissa = (ushort)(mantissa != 0 ? ((mantissa >> 13) | 0x200) : 0);
            return (ushort)((sign << 15) | 0x7C00 | nanMantissa);
        }

        if (exponent == 0 && mantissa == 0)
        {
            // Zero
            return (ushort)(sign << 15);
        }

        // Adjust exponent for FP16 bias
        var fp16Exponent = exponent - 127 + 15;

        // Handle overflow/underflow
        if (fp16Exponent >= 31)
        {
            // Overflow to infinity
            return (ushort)((sign << 15) | 0x7C00);
        }

        if (fp16Exponent <= 0)
        {
            // Underflow to zero or denormal
            return (ushort)(sign << 15);
        }

        // Convert mantissa (23 bits to 10 bits)
        var fp16Mantissa = mantissa >> 13;

        // Combine into FP16
        return (ushort)((sign << 15) | (fp16Exponent << 10) | fp16Mantissa);
    }

    private float ConvertFromFloat16(ushort fp16)
    {
        // Extract components
        var sign = (fp16 >> 15) & 0x1;
        var exponent = (fp16 >> 10) & 0x1F;
        var mantissa = fp16 & 0x3FF;

        // Handle special cases
        if (exponent == 0x1F)
        {
            // Infinity or NaN
            return mantissa != 0 ? float.NaN : (sign != 0 ? float.NegativeInfinity : float.PositiveInfinity);
        }

        if (exponent == 0 && mantissa == 0)
        {
            // Zero
            return sign != 0 ? -0.0f : 0.0f;
        }

        // Adjust exponent for FP32 bias
        var fp32Exponent = exponent - 15 + 127;

        // Expand mantissa (10 bits to 23 bits)
        var fp32Mantissa = mantissa << 13;

        // Combine into FP32
        var bits = (sign << 31) | (fp32Exponent << 23) | fp32Mantissa;
        var bytes = BitConverter.GetBytes(bits);
        return BitConverter.ToSingle(bytes, 0);
    }
}
