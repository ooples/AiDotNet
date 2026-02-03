using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using AiDotNet.Deployment.Optimization.Quantization.Calibration;

namespace AiDotNet.Deployment.Optimization.Quantization.Formats;

/// <summary>
/// NF4 (4-bit NormalFloat) quantizer - optimal for normally distributed weights.
/// Used by QLoRA for efficient 4-bit base model quantization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> NF4 is a special 4-bit format where the 16 possible values
/// are chosen to be optimal for weights that follow a normal distribution (bell curve).
/// This makes it perfect for neural network weights, which are typically normally distributed.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Uses 16 asymmetric quantization levels from the normal distribution</description></item>
/// <item><description>Levels are placed at quantiles of N(0,1): each covers equal probability mass</description></item>
/// <item><description>Combined with block-wise quantization for better accuracy</description></item>
/// </list>
///
/// <para><b>Key Features:</b></para>
/// <list type="bullet">
/// <item><description>Information-theoretically optimal for normal distributions</description></item>
/// <item><description>Used in QLoRA for 4-bit base model quantization</description></item>
/// <item><description>8x compression (4-bit vs 32-bit)</description></item>
/// <item><description>Block-wise scaling for per-block accuracy</description></item>
/// </list>
///
/// <para><b>Reference:</b> Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)</para>
/// </remarks>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class NF4Quantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly QuantizationConfiguration _config;
    private readonly Dictionary<string, double> _scaleFactors = new();
    private readonly int _blockSize;

    // NF4 codebook: 16 values used in QLoRA for N(0,1)-distributed weights.
    // These are empirically optimized representative values for equal-probability bins of N(0,1),
    // not exact analytical quantiles. The interval comments below show approximate quantile ranges
    // for each bin. See Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023),
    // Appendix / NF4 codebook definition, for the original table of values.
    private static readonly double[] NF4Codebook = new double[]
    {
        -1.0,         // approx. -∞ to -1.18
        -0.6961928,   // approx. -1.18 to -0.86
        -0.5250730,   // approx. -0.86 to -0.61
        -0.3949460,   // approx. -0.61 to -0.39
        -0.2844714,   // approx. -0.39 to -0.20
        -0.1828020,   // approx. -0.20 to -0.02
        -0.0911346,   // approx. -0.02 to 0.08
        0.0,          // approx. 0.08 to 0.18
        0.0796089,    // approx. 0.18 to 0.29
        0.1609563,    // approx. 0.29 to 0.40
        0.2461107,    // approx. 0.40 to 0.53
        0.3379640,    // approx. 0.53 to 0.67
        0.4407326,    // approx. 0.67 to 0.84
        0.5626170,    // approx. 0.84 to 1.07
        0.7229568,    // approx. 1.07 to 1.41
        1.0           // approx. 1.41 to +∞
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
    /// Initializes a new instance of the NF4Quantizer.
    /// </summary>
    /// <param name="config">Quantization configuration</param>
    /// <param name="blockSize">Block size for block-wise quantization (default: 64)</param>
    public NF4Quantizer(QuantizationConfiguration? config = null, int blockSize = 64)
    {
        _config = config ?? new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8, // Use Int8 mode as base
            TargetBitWidth = 4,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = blockSize,
            UseSymmetricQuantization = false // NF4 is asymmetric
        };

        _blockSize = Math.Max(1, blockSize);
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        var parameters = model.GetParameters();
        var quantizedParams = QuantizeWithNF4(parameters);

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

        // Collect activation statistics using calibration data
        var calibrationHelper = new CalibrationHelper<T, TInput, TOutput>(_config);
        var activationStats = calibrationHelper.CollectActivationStatistics(model, dataList);

        // Pre-compute block scales
        var parameters = model.GetParameters();
        int numBlocks = (parameters.Length + _blockSize - 1) / _blockSize;

        // Get activation magnitudes if available
        double[]? activationMagnitudes = activationStats.GlobalMaxAbsActivations;

        for (int b = 0; b < numBlocks; b++)
        {
            int start = b * _blockSize;
            int end = Math.Min(start + _blockSize, parameters.Length);

            double maxAbs = 0;
            for (int i = start; i < end; i++)
            {
                // Consider both parameter values and activation magnitudes for scaling
                double paramAbs = Math.Abs(Convert.ToDouble(parameters[i]));
                double actAbs = (activationMagnitudes != null && i < activationMagnitudes.Length)
                    ? activationMagnitudes[i]
                    : 0;
                maxAbs = Math.Max(maxAbs, Math.Max(paramAbs, actAbs));
            }

            _scaleFactors[$"block_{b}"] = maxAbs > 0 ? maxAbs : 1.0;
        }

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
        return 0; // NF4 doesn't use zero point offset
    }

    /// <summary>
    /// Quantizes parameters using NF4 format.
    /// </summary>
    private Vector<T> QuantizeWithNF4(Vector<T> parameters)
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
            if (IsCalibrated && _scaleFactors.TryGetValue(blockKey, out var calibratedScale))
            {
                scale = calibratedScale;
            }
            else
            {
                // Compute block scale (absmax)
                double maxAbs = 0;
                for (int i = start; i < end; i++)
                {
                    maxAbs = Math.Max(maxAbs, Math.Abs(Convert.ToDouble(parameters[i])));
                }
                scale = maxAbs > 0 ? maxAbs : 1.0;
                _scaleFactors[blockKey] = scale;
            }

            // Quantize each value in the block
            for (int i = start; i < end; i++)
            {
                double value = Convert.ToDouble(parameters[i]);
                double normalized = value / scale; // Normalize to [-1, 1]

                // Find nearest NF4 codebook value
                double quantized = FindNearestNF4Value(normalized);

                // Dequantize
                double dequantized = quantized * scale;
                result[i] = NumOps.FromDouble(dequantized);
            }
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Finds the nearest value in the NF4 codebook.
    /// </summary>
    private static double FindNearestNF4Value(double value)
    {
        // Clamp to codebook range
        value = MathHelper.Clamp(value, NF4Codebook[0], NF4Codebook[15]);

        double bestValue = NF4Codebook[0];
        double bestDistance = Math.Abs(value - bestValue);

        for (int i = 1; i < NF4Codebook.Length; i++)
        {
            double distance = Math.Abs(value - NF4Codebook[i]);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestValue = NF4Codebook[i];
            }
        }

        return bestValue;
    }

    /// <summary>
    /// Converts a 4-bit index to its NF4 value.
    /// </summary>
    public static double IndexToNF4(int index)
    {
        if (index < 0 || index >= 16)
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be 0-15");
        return NF4Codebook[index];
    }

    /// <summary>
    /// Converts a value to its nearest 4-bit index.
    /// </summary>
    public static int NF4ToIndex(double value)
    {
        value = MathHelper.Clamp(value, NF4Codebook[0], NF4Codebook[15]);

        int bestIndex = 0;
        double bestDistance = Math.Abs(value - NF4Codebook[0]);

        for (int i = 1; i < NF4Codebook.Length; i++)
        {
            double distance = Math.Abs(value - NF4Codebook[i]);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
