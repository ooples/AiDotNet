using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Deployment.Optimization.Quantization.Formats;

/// <summary>
/// FP8 (8-bit Floating Point) quantizer supporting E4M3 and E5M2 formats.
/// Provides better outlier handling than INT8 while maintaining 8-bit efficiency.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FP8 is a newer 8-bit format that uses floating-point representation
/// instead of integers. It's better at handling outliers (extreme values) and requires less
/// calibration than INT8.</para>
///
/// <para><b>FP8 Formats:</b></para>
/// <list type="bullet">
/// <item><description><b>E4M3:</b> 4 exponent bits, 3 mantissa bits. Better precision, smaller range. Best for weights.</description></item>
/// <item><description><b>E5M2:</b> 5 exponent bits, 2 mantissa bits. Larger range, less precision. Best for gradients/activations.</description></item>
/// </list>
///
/// <para><b>Key Features:</b></para>
/// <list type="bullet">
/// <item><description>Native support on H100, MI300x, and newer GPUs</description></item>
/// <item><description>Better outlier handling than INT8 (no clipping needed)</description></item>
/// <item><description>Less calibration sensitive than integer quantization</description></item>
/// <item><description>99-100% accuracy preservation even at 405B+ scale</description></item>
/// </list>
///
/// <para><b>Reference:</b> NVIDIA FP8 specification and various hardware vendor implementations</para>
/// </remarks>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class FP8Quantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly QuantizationConfiguration _config;
    private readonly FP8Format _format;
    private readonly Dictionary<string, double> _scaleFactors = new();
    private readonly Dictionary<string, int> _zeroPoints = new();
    private bool _isCalibrated;

    // FP8 E4M3 constants (for weights)
    private const int E4M3_EXPONENT_BITS = 4;
    private const int E4M3_MANTISSA_BITS = 3;
    private const int E4M3_EXPONENT_BIAS = 7;
    private const double E4M3_MAX_VALUE = 448.0;    // Max representable value
    private const double E4M3_MIN_SUBNORMAL = 0.001953125; // Smallest subnormal

    // FP8 E5M2 constants (for gradients/activations)
    private const int E5M2_EXPONENT_BITS = 5;
    private const int E5M2_MANTISSA_BITS = 2;
    private const int E5M2_EXPONENT_BIAS = 15;
    private const double E5M2_MAX_VALUE = 57344.0;  // Max representable value
    private const double E5M2_MIN_SUBNORMAL = 0.0000152587890625;

    /// <inheritdoc/>
    public QuantizationMode Mode => QuantizationMode.Float16; // Closest mode

    /// <inheritdoc/>
    public int BitWidth => 8;

    /// <summary>
    /// Gets the FP8 format being used (E4M3 or E5M2).
    /// </summary>
    public FP8Format Format => _format;

    /// <summary>
    /// Initializes a new instance of the FP8Quantizer.
    /// </summary>
    /// <param name="format">FP8 format to use (default: E4M3 for weights)</param>
    /// <param name="config">Quantization configuration</param>
    public FP8Quantizer(FP8Format format = FP8Format.E4M3, QuantizationConfiguration? config = null)
    {
        _format = format;
        _config = config ?? new QuantizationConfiguration
        {
            Mode = QuantizationMode.Float16,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseSymmetricQuantization = true
        };
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        var parameters = model.GetParameters();
        var quantizedParams = QuantizeToFP8(parameters, config);
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

        // Collect parameter statistics for optimal scaling
        var parameters = model.GetParameters();
        double maxAbs = 0;

        for (int i = 0; i < parameters.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(Convert.ToDouble(parameters[i])));
        }

        // Compute optimal scale to fit values within FP8 range
        double fp8Max = _format == FP8Format.E4M3 ? E4M3_MAX_VALUE : E5M2_MAX_VALUE;
        double scale = maxAbs > 0 ? fp8Max / maxAbs : 1.0;

        _scaleFactors["global"] = scale;
        _zeroPoints["global"] = 0; // FP8 uses symmetric representation

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
        return 0; // FP8 doesn't use zero point
    }

    /// <summary>
    /// Quantizes parameters to FP8 format.
    /// </summary>
    private Vector<T> QuantizeToFP8(Vector<T> parameters, QuantizationConfiguration config)
    {
        int n = parameters.Length;
        var result = new T[n];

        // Get or compute scale factor
        double scale = _scaleFactors.TryGetValue("global", out var s) ? s : 1.0;

        double fp8Max = _format == FP8Format.E4M3 ? E4M3_MAX_VALUE : E5M2_MAX_VALUE;

        // If not calibrated, compute scale now
        if (!_isCalibrated)
        {
            double maxAbs = 0;
            for (int i = 0; i < n; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(Convert.ToDouble(parameters[i])));
            }
            scale = maxAbs > 0 ? fp8Max / maxAbs : 1.0;
            _scaleFactors["global"] = scale;
        }

        for (int i = 0; i < n; i++)
        {
            double value = Convert.ToDouble(parameters[i]);

            // Scale value to FP8 range
            double scaled = value * scale;

            // Convert to FP8 representation and back
            double fp8Value = ToFP8(scaled);

            // Scale back
            double dequantized = fp8Value / scale;

            result[i] = NumOps.FromDouble(dequantized);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Converts a double to FP8 representation and back (simulating FP8 precision loss).
    /// </summary>
    private double ToFP8(double value)
    {
        if (double.IsNaN(value))
            // Note: IEEE FP8 E4M3 can represent NaN, but for ML quantization we map NaN to 0
            // to avoid propagating invalid values through the model (safer for inference)
            return 0;

        return _format == FP8Format.E4M3 ? ToE4M3(value) : ToE5M2(value);
    }

    /// <summary>
    /// Converts to E4M3 format (4 exponent bits, 3 mantissa bits).
    /// </summary>
    private double ToE4M3(double value)
    {
        if (value == 0) return 0;

        bool negative = value < 0;
        double absValue = Math.Abs(value);

        // Clamp to E4M3 range
        if (absValue > E4M3_MAX_VALUE)
        {
            absValue = E4M3_MAX_VALUE;
        }

        if (absValue < E4M3_MIN_SUBNORMAL)
        {
            return 0; // Underflow to zero
        }

        // Extract exponent and mantissa
        int exponent = (int)Math.Floor(Math.Log(absValue) / Math.Log(2));
        exponent = MathHelper.Clamp(exponent, -6, 8); // E4M3 exponent range with bias 7

        // Compute mantissa
        double mantissa = absValue / Math.Pow(2, exponent);

        // Quantize mantissa to 3 bits (8 levels: 1.0 to 1.875)
        int mantissaInt = (int)Math.Round((mantissa - 1.0) * 8);
        mantissaInt = MathHelper.Clamp(mantissaInt, 0, 7);

        // Reconstruct
        double quantizedMantissa = 1.0 + mantissaInt / 8.0;
        double result = quantizedMantissa * Math.Pow(2, exponent);

        return negative ? -result : result;
    }

    /// <summary>
    /// Converts to E5M2 format (5 exponent bits, 2 mantissa bits).
    /// </summary>
    private double ToE5M2(double value)
    {
        if (value == 0) return 0;

        bool negative = value < 0;
        double absValue = Math.Abs(value);

        // Handle infinity
        if (double.IsInfinity(absValue))
        {
            return negative ? -E5M2_MAX_VALUE : E5M2_MAX_VALUE;
        }

        // Clamp to E5M2 range
        if (absValue > E5M2_MAX_VALUE)
        {
            absValue = E5M2_MAX_VALUE;
        }

        if (absValue < E5M2_MIN_SUBNORMAL)
        {
            return 0; // Underflow to zero
        }

        // Extract exponent and mantissa
        int exponent = (int)Math.Floor(Math.Log(absValue) / Math.Log(2));
        exponent = MathHelper.Clamp(exponent, -14, 15); // E5M2 exponent range with bias 15

        // Compute mantissa
        double mantissa = absValue / Math.Pow(2, exponent);

        // Quantize mantissa to 2 bits (4 levels: 1.0, 1.25, 1.5, 1.75)
        int mantissaInt = (int)Math.Round((mantissa - 1.0) * 4);
        mantissaInt = MathHelper.Clamp(mantissaInt, 0, 3);

        // Reconstruct
        double quantizedMantissa = 1.0 + mantissaInt / 4.0;
        double result = quantizedMantissa * Math.Pow(2, exponent);

        return negative ? -result : result;
    }

    /// <summary>
    /// Converts a byte to E4M3 double value.
    /// </summary>
    public static double ByteToE4M3(byte b)
    {
        if (b == 0) return 0;

        bool sign = (b & 0x80) != 0;
        int exp = (b >> 3) & 0x0F;
        int mantissa = b & 0x07;

        double value;
        if (exp == 0)
        {
            // Subnormal
            value = mantissa / 8.0 * Math.Pow(2, -6);
        }
        else if (exp == 15)
        {
            // E4M3 doesn't have inf/nan, max value instead
            value = E4M3_MAX_VALUE;
        }
        else
        {
            // Normal
            value = (1.0 + mantissa / 8.0) * Math.Pow(2, exp - E4M3_EXPONENT_BIAS);
        }

        return sign ? -value : value;
    }

    /// <summary>
    /// Converts an E4M3 double value to byte.
    /// </summary>
    public static byte E4M3ToByte(double value)
    {
        if (value == 0) return 0;

        bool sign = value < 0;
        double absValue = Math.Abs(value);

        if (absValue >= E4M3_MAX_VALUE)
        {
            return (byte)(sign ? 0xFF : 0x7F); // Max value
        }

        if (absValue < E4M3_MIN_SUBNORMAL)
        {
            return 0; // Zero
        }

        int exp = (int)Math.Floor(Math.Log(absValue) / Math.Log(2)) + E4M3_EXPONENT_BIAS;

        if (exp <= 0)
        {
            // Subnormal
            int mantissa = (int)Math.Round(absValue / Math.Pow(2, -6) * 8);
            mantissa = MathHelper.Clamp(mantissa, 0, 7);
            return (byte)((sign ? 0x80 : 0) | mantissa);
        }

        exp = MathHelper.Clamp(exp, 1, 14);
        double mantissaD = absValue / Math.Pow(2, exp - E4M3_EXPONENT_BIAS) - 1.0;
        int mant = (int)Math.Round(mantissaD * 8);
        mant = MathHelper.Clamp(mant, 0, 7);

        return (byte)((sign ? 0x80 : 0) | (exp << 3) | mant);
    }

    /// <summary>
    /// Converts a byte to E5M2 double value.
    /// </summary>
    public static double ByteToE5M2(byte b)
    {
        if (b == 0) return 0;

        bool sign = (b & 0x80) != 0;
        int exp = (b >> 2) & 0x1F;
        int mantissa = b & 0x03;

        double value;
        if (exp == 0)
        {
            // Subnormal
            value = mantissa / 4.0 * Math.Pow(2, -14);
        }
        else if (exp == 31)
        {
            // Infinity or NaN
            value = mantissa == 0 ? double.PositiveInfinity : double.NaN;
        }
        else
        {
            // Normal
            value = (1.0 + mantissa / 4.0) * Math.Pow(2, exp - E5M2_EXPONENT_BIAS);
        }

        return sign ? -value : value;
    }

    /// <summary>
    /// Converts an E5M2 double value to byte.
    /// </summary>
    public static byte E5M2ToByte(double value)
    {
        if (value == 0) return 0;
        if (double.IsNaN(value)) return 0x7F; // NaN pattern
        if (double.IsPositiveInfinity(value)) return 0x7C;
        if (double.IsNegativeInfinity(value)) return 0xFC;

        bool sign = value < 0;
        double absValue = Math.Abs(value);

        if (absValue >= E5M2_MAX_VALUE)
        {
            return (byte)(sign ? 0xFB : 0x7B); // Max finite value
        }

        if (absValue < E5M2_MIN_SUBNORMAL)
        {
            return 0;
        }

        int exp = (int)Math.Floor(Math.Log(absValue) / Math.Log(2)) + E5M2_EXPONENT_BIAS;

        if (exp <= 0)
        {
            // Subnormal
            int mantissa = (int)Math.Round(absValue / Math.Pow(2, -14) * 4);
            mantissa = MathHelper.Clamp(mantissa, 0, 3);
            return (byte)((sign ? 0x80 : 0) | mantissa);
        }

        exp = MathHelper.Clamp(exp, 1, 30);
        double mantissaD = absValue / Math.Pow(2, exp - E5M2_EXPONENT_BIAS) - 1.0;
        int mant = (int)Math.Round(mantissaD * 4);
        mant = MathHelper.Clamp(mant, 0, 3);

        return (byte)((sign ? 0x80 : 0) | (exp << 2) | mant);
    }
}

/// <summary>
/// Specifies the FP8 format variant to use.
/// </summary>
public enum FP8Format
{
    /// <summary>
    /// E4M3: 4 exponent bits, 3 mantissa bits. Better precision, smaller range.
    /// Best for weights. Max value: 448.
    /// </summary>
    E4M3,

    /// <summary>
    /// E5M2: 5 exponent bits, 2 mantissa bits. Larger range, less precision.
    /// Best for gradients and activations. Max value: 57344.
    /// </summary>
    E5M2
}
