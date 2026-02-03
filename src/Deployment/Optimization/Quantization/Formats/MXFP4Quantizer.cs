using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Deployment.Optimization.Quantization.Formats;

/// <summary>
/// MXFP4 (Microscaling FP4) quantizer - uses shared exponents for efficient 4-bit floating point.
/// Part of the OCP (Open Compute Project) Microscaling specification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MXFP4 is a 4-bit floating point format where groups of numbers
/// share a common scale (exponent). This allows better representation of values across
/// different magnitudes while staying compact.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Groups of values (typically 32) share a single 8-bit scale</description></item>
/// <item><description>Each value uses 4 bits: 1 sign bit + 3 bits for the mantissa</description></item>
/// <item><description>The shared scale adjusts all values in the group</description></item>
/// </list>
///
/// <para><b>Format Details:</b></para>
/// <list type="bullet">
/// <item><description>E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit per element</description></item>
/// <item><description>Shared 8-bit scale per block (typically 32 elements)</description></item>
/// <item><description>Effective storage: 4 bits per element + amortized scale overhead</description></item>
/// </list>
///
/// <para><b>Key Features:</b></para>
/// <list type="bullet">
/// <item><description>Hardware support on latest accelerators (NVIDIA H100, AMD MI300)</description></item>
/// <item><description>Better dynamic range than fixed-point INT4</description></item>
/// <item><description>Efficient block-wise computation</description></item>
/// <item><description>Part of OCP Microscaling standard</description></item>
/// </list>
///
/// <para><b>Reference:</b> OCP Microscaling Formats Specification (2023)</para>
/// </remarks>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class MXFP4Quantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly QuantizationConfiguration _config;
    private readonly Dictionary<string, double> _scaleFactors = new();
    private readonly int _blockSize;

    // MXFP4 (E2M1) codebook: 16 values
    // Format: 1 sign, 2 exponent bits, 1 mantissa bit
    // Values: sign * (1 + M/2) * 2^(E-1) for normal, with special handling for subnormal
    private static readonly double[] MXFP4Positive = new double[]
    {
        0.0,     // 0 00 0 - Zero
        0.5,     // 0 00 1 - Subnormal: 0.5 * 2^0
        1.0,     // 0 01 0 - Normal: 1.0 * 2^0
        1.5,     // 0 01 1 - Normal: 1.5 * 2^0
        2.0,     // 0 10 0 - Normal: 1.0 * 2^1
        3.0,     // 0 10 1 - Normal: 1.5 * 2^1
        4.0,     // 0 11 0 - Normal: 1.0 * 2^2
        6.0      // 0 11 1 - Normal: 1.5 * 2^2
    };

    /// <inheritdoc/>
    public QuantizationMode Mode => _config.Mode;

    /// <inheritdoc/>
    public int BitWidth => 4;

    /// <summary>
    /// Gets whether the quantizer has been calibrated.
    /// </summary>
    public bool IsCalibrated { get; private set; }

    /// <summary>
    /// Gets the block size used for shared scaling.
    /// </summary>
    public int BlockSize => _blockSize;

    /// <summary>
    /// Initializes a new instance of the MXFP4Quantizer.
    /// </summary>
    /// <param name="config">Quantization configuration</param>
    /// <param name="blockSize">Block size for shared scaling (default: 32)</param>
    public MXFP4Quantizer(QuantizationConfiguration? config = null, int blockSize = 32)
    {
        _config = config ?? new QuantizationConfiguration
        {
            Mode = QuantizationMode.Float16,
            TargetBitWidth = 4,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = blockSize
        };

        _blockSize = Math.Max(1, blockSize);
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        var parameters = model.GetParameters();
        var quantizedParams = QuantizeWithMXFP4(parameters);

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

        // Pre-compute block scales
        var parameters = model.GetParameters();
        ComputeBlockScales(parameters);

        IsCalibrated = true;
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
        return 0; // MXFP4 uses floating point representation, no zero point
    }

    /// <summary>
    /// Computes shared scales for each block of parameters.
    /// </summary>
    private void ComputeBlockScales(Vector<T> parameters)
    {
        int n = parameters.Length;
        int numBlocks = (n + _blockSize - 1) / _blockSize;

        for (int b = 0; b < numBlocks; b++)
        {
            int start = b * _blockSize;
            int end = Math.Min(start + _blockSize, n);

            double maxAbs = 0;
            for (int i = start; i < end; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(Convert.ToDouble(parameters[i])));
            }

            // Compute shared exponent (scale factor)
            // Scale = 2^E where E is chosen so max value fits in MXFP4 range
            double maxMXFP4 = 6.0; // Maximum positive MXFP4 value
            double scale = maxAbs > 0 ? maxAbs / maxMXFP4 : 1.0;

            _scaleFactors[$"block_{b}"] = scale;
        }
    }

    /// <summary>
    /// Quantizes parameters using MXFP4 format with microscaling.
    /// </summary>
    private Vector<T> QuantizeWithMXFP4(Vector<T> parameters)
    {
        int n = parameters.Length;
        var result = new T[n];
        int numBlocks = (n + _blockSize - 1) / _blockSize;

        for (int b = 0; b < numBlocks; b++)
        {
            int start = b * _blockSize;
            int end = Math.Min(start + _blockSize, n);

            // Use calibrated scale if available, otherwise compute
            double scale;
            string blockKey = $"block_{b}";
            if (_scaleFactors.TryGetValue(blockKey, out var calibratedScale))
            {
                scale = calibratedScale;
            }
            else
            {
                // Compute shared scale for this block
                double maxAbs = 0;
                for (int i = start; i < end; i++)
                {
                    maxAbs = Math.Max(maxAbs, Math.Abs(Convert.ToDouble(parameters[i])));
                }

                double maxMXFP4 = 6.0;
                scale = maxAbs > 0 ? maxAbs / maxMXFP4 : 1.0;
                _scaleFactors[blockKey] = scale;
            }

            // Quantize each value in the block
            for (int i = start; i < end; i++)
            {
                double value = Convert.ToDouble(parameters[i]);
                double scaled = value / scale;

                // Find nearest MXFP4 value
                double quantized = FindNearestMXFP4Value(scaled);

                // Dequantize
                double dequantized = quantized * scale;
                result[i] = NumOps.FromDouble(dequantized);
            }
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Finds the nearest MXFP4 representation for a value.
    /// </summary>
    private static double FindNearestMXFP4Value(double value)
    {
        bool isNegative = value < 0;
        double absValue = Math.Abs(value);

        // Find nearest positive MXFP4 value
        double bestValue = MXFP4Positive[0];
        double bestDistance = Math.Abs(absValue - bestValue);

        for (int i = 1; i < MXFP4Positive.Length; i++)
        {
            double distance = Math.Abs(absValue - MXFP4Positive[i]);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestValue = MXFP4Positive[i];
            }
        }

        return isNegative ? -bestValue : bestValue;
    }

    /// <summary>
    /// Encodes a value to its 4-bit MXFP4 representation.
    /// </summary>
    /// <param name="value">Value to encode (should be pre-scaled)</param>
    /// <returns>4-bit encoding (0-15)</returns>
    public static int EncodeToMXFP4(double value)
    {
        bool isNegative = value < 0;
        double absValue = Math.Abs(value);

        // Find nearest index
        int bestIndex = 0;
        double bestDistance = Math.Abs(absValue - MXFP4Positive[0]);

        for (int i = 1; i < MXFP4Positive.Length; i++)
        {
            double distance = Math.Abs(absValue - MXFP4Positive[i]);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        // Add sign bit (bit 3)
        return isNegative ? (bestIndex | 0x8) : bestIndex;
    }

    /// <summary>
    /// Decodes a 4-bit MXFP4 representation to its value.
    /// </summary>
    /// <param name="encoded">4-bit encoding (0-15)</param>
    /// <returns>Decoded value</returns>
    public static double DecodeFromMXFP4(int encoded)
    {
        bool isNegative = (encoded & 0x8) != 0;
        int index = encoded & 0x7;

        double value = MXFP4Positive[index];
        return isNegative ? -value : value;
    }
}
