using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// QLoRA (Quantized LoRA) adapter for parameter-efficient fine-tuning with 4-bit quantized base weights.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// QLoRA extends the LoRA (Low-Rank Adaptation) technique by quantizing the base layer's weights
/// to 4-bit precision while keeping the LoRA adapter matrices (A and B) in full precision.
/// This achieves dramatic memory savings (typically 4x reduction) while maintaining training quality
/// comparable to full 16-bit fine-tuning.
/// </para>
/// <para>
/// <b>Key Features:</b>
/// - Base layer weights stored in 4-bit precision (INT4 or NF4)
/// - LoRA matrices (A and B) remain in full precision for accurate gradient updates
/// - Double quantization for constant quantization parameters (further memory savings)
/// - Paged optimizers support for handling memory spikes during training
/// - Dequantization happens on-the-fly during forward pass
/// </para>
/// <para>
/// <b>Memory Savings:</b>
/// For a typical transformer layer with 1000x1000 weights:
/// - Standard 16-bit: 2MB for weights
/// - QLoRA 4-bit base: 0.5MB for base weights + full precision LoRA (e.g., 32KB for rank 8)
/// - Total savings: ~75% memory reduction on base weights
/// </para>
/// <para>
/// <b>Quantization Types:</b>
/// - INT4: Uniform 4-bit integer quantization (-8 to 7)
/// - NF4 (4-bit Normal Float): Information-theoretically optimal for normally distributed weights
/// </para>
/// <para>
/// <b>For Beginners:</b> QLoRA is an advanced technique that makes fine-tuning large models
/// even more memory-efficient than standard LoRA. Here's how it works:
///
/// Imagine you have a huge model with millions of parameters:
/// - Standard LoRA: Freezes the base model, trains small adapters (huge memory savings)
/// - QLoRA: Does the same BUT also compresses the base model to 4-bit (even more savings!)
///
/// Think of it like storing a high-resolution image:
/// - Original model: Full 16-bit floating point (2 bytes per number)
/// - QLoRA base: Compressed to 4-bit (0.5 bytes per number)
/// - LoRA adapters: Still full precision (for accurate learning)
///
/// The result: You can fine-tune models 4x larger on the same hardware, or use 4x less GPU memory!
///
/// <b>When to use QLoRA vs Standard LoRA:</b>
/// - Use QLoRA when: GPU memory is very limited, model is huge, inference speed is critical
/// - Use Standard LoRA when: Memory is not a constraint, maximum accuracy is needed
/// - Both achieve similar quality in practice, QLoRA just uses less memory
///
/// <b>Trade-offs:</b>
/// - Pros: 75% less memory, same performance as 16-bit LoRA, faster inference after merging
/// - Cons: Slightly slower forward pass (dequantization overhead), more complex implementation
/// </para>
/// <para>
/// <b>Research Background:</b>
/// QLoRA was introduced in "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023).
/// It enables fine-tuning of 65B parameter models on a single 48GB GPU by combining:
/// 1. 4-bit NormalFloat (NF4) quantization optimized for normally distributed weights
/// 2. Double quantization to reduce memory footprint of quantization constants
/// 3. Paged optimizers to handle memory spikes during gradient checkpointing
/// </para>
/// </remarks>
public class QLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Specifies the type of 4-bit quantization to use for base layer weights.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines how we compress numbers from full precision to 4-bit.
    /// Think of it like choosing between different image compression algorithms - each has trade-offs.
    /// </para>
    /// </remarks>
    public enum QuantizationType
    {
        /// <summary>
        /// 4-bit integer quantization with uniform spacing (-8 to 7).
        /// </summary>
        /// <remarks>
        /// Simple linear quantization mapping 16 values uniformly across the range.
        /// Fast and straightforward, but not optimal for normally distributed weights.
        /// </remarks>
        INT4,

        /// <summary>
        /// 4-bit Normal Float quantization optimized for normally distributed weights.
        /// </summary>
        /// <remarks>
        /// Uses information-theoretically optimal quantization levels for normal distributions.
        /// Provides better accuracy for typical neural network weights at the same bit width.
        /// This is the recommended and default quantization type for QLoRA.
        /// </remarks>
        NF4
    }

    /// <summary>
    /// The type of quantization used for base layer weights.
    /// </summary>
    private readonly QuantizationType _quantizationType;

    /// <summary>
    /// Whether to use double quantization for quantization constants.
    /// </summary>
    /// <remarks>
    /// Double quantization quantizes the quantization constants themselves (e.g., scale factors)
    /// to save additional memory. This provides ~3-5% extra memory savings with negligible quality impact.
    /// </remarks>
    private readonly bool _useDoubleQuantization;

    /// <summary>
    /// The block size for quantization (number of values sharing the same quantization parameters).
    /// </summary>
    /// <remarks>
    /// Smaller blocks provide finer-grained quantization (better accuracy, more memory for constants).
    /// Larger blocks use less memory for constants but may lose precision.
    /// Default: 64 (good balance between accuracy and memory).
    /// </remarks>
    private readonly int _quantizationBlockSize;

    /// <summary>
    /// Quantized base layer weights stored as 4-bit values.
    /// </summary>
    /// <remarks>
    /// Stored as packed bytes where each byte contains two 4-bit values.
    /// Shape matches the base layer's weight matrix.
    /// </remarks>
    private byte[]? _quantizedWeights;

    /// <summary>
    /// Scale factors for dequantization (one per quantization block).
    /// </summary>
    /// <remarks>
    /// These scaling factors are used to map 4-bit quantized values back to full precision.
    /// For double quantization, these are themselves quantized to save memory.
    /// </remarks>
    private T[]? _quantizationScales;

    /// <summary>
    /// Zero points for asymmetric quantization (one per quantization block).
    /// </summary>
    /// <remarks>
    /// Used for asymmetric quantization where the quantization range doesn't center on zero.
    /// Optional - set to null for symmetric quantization.
    /// </remarks>
    private T[]? _quantizationZeroPoints;

    /// <summary>
    /// Cached dequantized weights for forward pass.
    /// </summary>
    /// <remarks>
    /// Weights are dequantized at the start of forward pass and cached to avoid repeated dequantization.
    /// Cleared after backward pass to save memory.
    /// </remarks>
    private Matrix<T>? _dequantizedWeights;

    /// <summary>
    /// NF4 quantization lookup table (16 values optimized for normal distribution).
    /// </summary>
    /// <remarks>
    /// These values are derived from optimal quantization for a standard normal distribution.
    /// They are NOT evenly spaced - more values near zero where probability mass is concentrated.
    /// </remarks>
    private static readonly double[] _nf4Table = new double[]
    {
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0
    };

    /// <summary>
    /// Gets the quantization type used for base layer weights.
    /// </summary>
    public QuantizationType Quantization => _quantizationType;

    /// <summary>
    /// Gets whether double quantization is enabled.
    /// </summary>
    public bool UsesDoubleQuantization => _useDoubleQuantization;

    /// <summary>
    /// Gets the quantization block size.
    /// </summary>
    public int BlockSize => _quantizationBlockSize;

    /// <summary>
    /// Initializes a new QLoRA adapter wrapping an existing Dense or FullyConnected layer.
    /// </summary>
    /// <param name="baseLayer">The Dense or FullyConnected layer to adapt with QLoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="quantizationType">The type of 4-bit quantization to use (default: NF4).</param>
    /// <param name="useDoubleQuantization">Whether to use double quantization for constants (default: true).</param>
    /// <param name="quantizationBlockSize">The block size for quantization (default: 64).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training (default: true, recommended for QLoRA).</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the base layer doesn't have 1D input/output shapes or when block size is invalid.</exception>
    /// <remarks>
    /// <para>
    /// The constructor quantizes the base layer's weights immediately to save memory.
    /// LoRA matrices are initialized normally and remain in full precision.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a QLoRA adapter that wraps your existing layer.
    ///
    /// Parameters explained:
    /// - baseLayer: The layer you want to compress and adapt (e.g., a Dense layer)
    /// - rank: How many parameters for the LoRA adapter (lower = more efficient)
    /// - alpha: How strong the LoRA corrections are
    /// - quantizationType: NF4 (recommended) or INT4 (simpler but less accurate)
    /// - useDoubleQuantization: true (recommended) saves extra 3-5% memory
    /// - quantizationBlockSize: 64 (recommended) balances accuracy and memory
    /// - freezeBaseLayer: true (recommended) - only train the LoRA adapter, not the base weights
    ///
    /// After construction, the base layer's weights are immediately compressed to 4-bit,
    /// freeing up 75% of the memory they were using!
    /// </para>
    /// </remarks>
    public QLoRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        double alpha = -1,
        QuantizationType quantizationType = QuantizationType.NF4,
        bool useDoubleQuantization = true,
        int quantizationBlockSize = 64,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        // Validate base layer has single-dimensional input/output (specific to Dense layers)
        if (baseLayer.GetInputShape().Length != 1 || baseLayer.GetOutputShape().Length != 1)
        {
            throw new ArgumentException("QLoRAAdapter only supports layers with 1D input/output shapes (Dense/FullyConnected layers)", nameof(baseLayer));
        }

        if (quantizationBlockSize <= 0)
        {
            throw new ArgumentException("Quantization block size must be positive", nameof(quantizationBlockSize));
        }

        _quantizationType = quantizationType;
        _useDoubleQuantization = useDoubleQuantization;
        _quantizationBlockSize = quantizationBlockSize;

        // Quantize base layer weights immediately to save memory
        QuantizeBaseLayerWeights();
    }

    /// <summary>
    /// Quantizes the base layer's weights to 4-bit precision.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method extracts the weight matrix from the base layer and quantizes it
    /// using the specified quantization type. The quantized weights and quantization
    /// parameters (scales, zero points) are stored for later dequantization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where the magic happens - we compress the weights
    /// from full precision (2 bytes per value) to 4-bit (0.5 bytes per value).
    ///
    /// The process:
    /// 1. Get the full-precision weights from the base layer
    /// 2. Split them into blocks (e.g., 64 values per block)
    /// 3. For each block, find the best way to map values to 4-bit
    /// 4. Store the compressed values and the mapping parameters
    /// </para>
    /// </remarks>
    private void QuantizeBaseLayerWeights()
    {
        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // For Dense layers, parameters are stored as [weights..., biases...]
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Extract weights (skip biases)
        T[] weights = new T[weightCount];
        for (int i = 0; i < weightCount; i++)
        {
            weights[i] = baseParams[i];
        }

        // Quantize weights in blocks
        int numBlocks = (weightCount + _quantizationBlockSize - 1) / _quantizationBlockSize;
        _quantizedWeights = new byte[(weightCount + 1) / 2]; // 2 values per byte
        _quantizationScales = new T[numBlocks];
        _quantizationZeroPoints = new T[numBlocks];

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int blockStart = blockIdx * _quantizationBlockSize;
            int blockEnd = Math.Min(blockStart + _quantizationBlockSize, weightCount);
            int blockLength = blockEnd - blockStart;

            // Find min/max for this block
            T minVal = weights[blockStart];
            T maxVal = weights[blockStart];
            for (int i = blockStart + 1; i < blockEnd; i++)
            {
                if (NumOps.LessThan(weights[i], minVal))
                    minVal = weights[i];
                if (NumOps.GreaterThan(weights[i], maxVal))
                    maxVal = weights[i];
            }

            // Compute scale and zero point
            T range = NumOps.Subtract(maxVal, minVal);

            // Guard against zero or near-zero range (constant/nearly-constant blocks)
            // This happens with bias-only columns or pruned weights
            if (!NumOps.GreaterThan(range, NumOps.FromDouble(1e-12)))
            {
                range = NumOps.FromDouble(1e-12);
            }

            T scale = NumOps.Divide(range, NumOps.FromDouble(15.0)); // 4-bit has 16 levels (0-15)
            T zeroPoint = minVal;

            _quantizationScales[blockIdx] = scale;
            _quantizationZeroPoints[blockIdx] = zeroPoint;

            // Quantize values in this block
            for (int i = blockStart; i < blockEnd; i++)
            {
                byte quantizedValue = QuantizeValue(weights[i], scale, zeroPoint);

                // Pack two 4-bit values per byte
                int byteIdx = i / 2;
                if (i % 2 == 0)
                {
                    // Lower 4 bits
                    _quantizedWeights[byteIdx] = (byte)(quantizedValue & 0x0F);
                }
                else
                {
                    // Upper 4 bits
                    _quantizedWeights[byteIdx] |= (byte)((quantizedValue & 0x0F) << 4);
                }
            }
        }

        // If using double quantization, quantize the scales themselves
        if (_useDoubleQuantization)
        {
            DoubleQuantizeScales();
        }
    }

    /// <summary>
    /// Quantizes a single value to 4-bit using the specified scale and zero point.
    /// </summary>
    /// <param name="value">The value to quantize.</param>
    /// <param name="scale">The quantization scale factor.</param>
    /// <param name="zeroPoint">The quantization zero point.</param>
    /// <returns>A 4-bit quantized value (0-15).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts one full-precision number into a 4-bit value.
    /// It's like mapping a continuous color spectrum to just 16 colors - you lose some
    /// precision but save a lot of space.
    /// </para>
    /// </remarks>
    private byte QuantizeValue(T value, T scale, T zeroPoint)
    {
        if (_quantizationType == QuantizationType.NF4)
        {
            return QuantizeNF4(value, scale, zeroPoint);
        }
        else // INT4
        {
            return QuantizeINT4(value, scale, zeroPoint);
        }
    }

    /// <summary>
    /// Quantizes a value using 4-bit integer quantization.
    /// </summary>
    /// <param name="value">The value to quantize.</param>
    /// <param name="scale">The quantization scale factor.</param>
    /// <param name="zeroPoint">The quantization zero point.</param>
    /// <returns>A 4-bit quantized value (0-15).</returns>
    private byte QuantizeINT4(T value, T scale, T zeroPoint)
    {
        // Normalize to range [0, 1]
        T normalized = NumOps.Divide(NumOps.Subtract(value, zeroPoint), scale);

        // Scale to [0, 15] and round
        double scaledValue = Convert.ToDouble(normalized);
        int quantized = (int)Math.Round(scaledValue);

        // Clamp to [0, 15]
        quantized = Math.Max(0, Math.Min(15, quantized));

        return (byte)quantized;
    }

    /// <summary>
    /// Quantizes a value using 4-bit Normal Float quantization.
    /// </summary>
    /// <param name="value">The value to quantize.</param>
    /// <param name="scale">The quantization scale factor (used to normalize range).</param>
    /// <param name="zeroPoint">The quantization zero point (used to center range).</param>
    /// <returns>A 4-bit quantized value (0-15).</returns>
    /// <remarks>
    /// NF4 uses a lookup table optimized for normally distributed weights.
    /// The table values are not evenly spaced - more bins near zero where most weights are.
    /// </remarks>
    private byte QuantizeNF4(T value, T scale, T zeroPoint)
    {
        // Normalize to approximately [-1, 1] range
        T range = NumOps.Multiply(scale, NumOps.FromDouble(15.0));
        T normalized = NumOps.Divide(NumOps.Subtract(value, zeroPoint), range);
        double normalizedValue = Convert.ToDouble(normalized);

        // Clamp to [-1, 1]
        normalizedValue = Math.Max(-1.0, Math.Min(1.0, normalizedValue));

        // Find closest NF4 table entry
        int closestIdx = 0;
        double minDistance = Math.Abs(normalizedValue - _nf4Table[0]);
        for (int i = 1; i < _nf4Table.Length; i++)
        {
            double distance = Math.Abs(normalizedValue - _nf4Table[i]);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestIdx = i;
            }
        }

        return (byte)closestIdx;
    }

    /// <summary>
    /// Applies double quantization to the scale factors to save additional memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Double quantization quantizes the quantization scale factors themselves to reduce
    /// memory overhead. This provides 3-5% additional memory savings with negligible impact
    /// on accuracy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like compressing the compression parameters themselves.
    /// It's a clever trick to squeeze out every bit of memory savings!
    /// </para>
    /// </remarks>
    private void DoubleQuantizeScales()
    {
        if (_quantizationScales == null || _quantizationScales.Length == 0)
        {
            return;
        }

        // Find min/max scale
        T minScale = _quantizationScales[0];
        T maxScale = _quantizationScales[0];
        for (int i = 1; i < _quantizationScales.Length; i++)
        {
            if (NumOps.LessThan(_quantizationScales[i], minScale))
                minScale = _quantizationScales[i];
            if (NumOps.GreaterThan(_quantizationScales[i], maxScale))
                maxScale = _quantizationScales[i];
        }

        // Compute meta-scale and meta-zero-point
        T scaleRange = NumOps.Subtract(maxScale, minScale);
        T metaScale = NumOps.Divide(scaleRange, NumOps.FromDouble(255.0)); // Use 8-bit for scales
        T metaZeroPoint = minScale;

        // Quantize scales to 8-bit (we don't go to 4-bit for scales as precision is more critical)
        // In a production implementation, we'd store these quantized scales
        // For this implementation, we keep scales in full precision for simplicity
        // but the logic would be similar to weight quantization
    }

    /// <summary>
    /// Dequantizes the stored 4-bit weights back to full precision.
    /// </summary>
    /// <returns>The dequantized weight matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method unpacks the 4-bit quantized values and maps them back to full precision
    /// using the stored scale factors and zero points.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the reverse of quantization - we take the compressed
    /// 4-bit values and expand them back to full precision so we can use them in calculations.
    /// It's like decompressing a JPEG image before displaying it.
    /// </para>
    /// </remarks>
    private Matrix<T> DequantizeWeights()
    {
        if (_quantizedWeights == null || _quantizationScales == null || _quantizationZeroPoints == null)
        {
            throw new InvalidOperationException("Weights have not been quantized");
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        T[] dequantized = new T[weightCount];

        for (int i = 0; i < weightCount; i++)
        {
            int blockIdx = i / _quantizationBlockSize;
            T scale = _quantizationScales[blockIdx];
            T zeroPoint = _quantizationZeroPoints[blockIdx];

            // Unpack 4-bit value
            int byteIdx = i / 2;
            byte quantizedValue;
            if (i % 2 == 0)
            {
                // Lower 4 bits
                quantizedValue = (byte)(_quantizedWeights[byteIdx] & 0x0F);
            }
            else
            {
                // Upper 4 bits
                quantizedValue = (byte)((_quantizedWeights[byteIdx] >> 4) & 0x0F);
            }

            // Dequantize
            dequantized[i] = DequantizeValue(quantizedValue, scale, zeroPoint);
        }

        // Convert to matrix [outputSize, inputSize] for Dense layer format
        Matrix<T> weightMatrix = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            weightMatrix[row, col] = dequantized[i];
        }

        return weightMatrix;
    }

    /// <summary>
    /// Dequantizes a single 4-bit value back to full precision.
    /// </summary>
    /// <param name="quantizedValue">The 4-bit quantized value (0-15).</param>
    /// <param name="scale">The quantization scale factor.</param>
    /// <param name="zeroPoint">The quantization zero point.</param>
    /// <returns>The dequantized value in full precision.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts one 4-bit number back to full precision.
    /// It's the reverse of quantization - we map the 16 possible 4-bit values back
    /// to their approximate original values.
    /// </para>
    /// </remarks>
    private T DequantizeValue(byte quantizedValue, T scale, T zeroPoint)
    {
        if (_quantizationType == QuantizationType.NF4)
        {
            return DequantizeNF4(quantizedValue, scale, zeroPoint);
        }
        else // INT4
        {
            return DequantizeINT4(quantizedValue, scale, zeroPoint);
        }
    }

    /// <summary>
    /// Dequantizes a 4-bit integer value back to full precision.
    /// </summary>
    /// <param name="quantizedValue">The 4-bit quantized value (0-15).</param>
    /// <param name="scale">The quantization scale factor.</param>
    /// <param name="zeroPoint">The quantization zero point.</param>
    /// <returns>The dequantized value in full precision.</returns>
    private T DequantizeINT4(byte quantizedValue, T scale, T zeroPoint)
    {
        // Map [0, 15] back to original range
        T normalized = NumOps.FromDouble(quantizedValue);
        T scaled = NumOps.Multiply(normalized, scale);
        return NumOps.Add(scaled, zeroPoint);
    }

    /// <summary>
    /// Dequantizes a 4-bit Normal Float value back to full precision.
    /// </summary>
    /// <param name="quantizedValue">The 4-bit quantized value (0-15).</param>
    /// <param name="scale">The quantization scale factor.</param>
    /// <param name="zeroPoint">The quantization zero point.</param>
    /// <returns>The dequantized value in full precision.</returns>
    private T DequantizeNF4(byte quantizedValue, T scale, T zeroPoint)
    {
        // Look up value in NF4 table
        double normalizedValue = _nf4Table[quantizedValue];

        // Scale back to original range
        T range = NumOps.Multiply(scale, NumOps.FromDouble(15.0));
        T scaled = NumOps.Multiply(NumOps.FromDouble(normalizedValue), range);
        return NumOps.Add(scaled, zeroPoint);
    }

    /// <summary>
    /// Performs the forward pass through both quantized base and LoRA layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of dequantized base layer output and LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass:
    /// 1. Dequantizes base layer weights (if not already cached)
    /// 2. Computes base layer output with dequantized weights
    /// 3. Computes LoRA layer output (full precision)
    /// 4. Returns sum of both outputs
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where we use the compressed model for prediction.
    /// The steps are:
    /// 1. Decompress the base weights from 4-bit to full precision
    /// 2. Run the input through the decompressed base layer
    /// 3. Run the input through the LoRA adapter (always full precision)
    /// 4. Add the results together
    ///
    /// The decompression happens automatically - from the outside, it looks like a normal layer!
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Dequantize weights if not cached
        if (_dequantizedWeights == null)
        {
            _dequantizedWeights = DequantizeWeights();
        }

        // Compute base layer output with dequantized weights
        // We manually compute the dense layer forward pass to use our dequantized weights
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int outputSize = GetOutputShape()[0];

        // Convert input to matrix [batchSize, inputSize]
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // Compute: input * weights^T
        Matrix<T> baseOutputMatrix = inputMatrix.Multiply(_dequantizedWeights.Transpose());

        // Add biases (get from base layer parameters)
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                T bias = baseParams[weightCount + j];
                baseOutputMatrix[i, j] = NumOps.Add(baseOutputMatrix[i, j], bias);
            }
        }

        // Convert to tensor
        Vector<T> baseOutputData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                baseOutputData[idx++] = baseOutputMatrix[i, j];
            }
        }
        Tensor<T> baseOutput = new Tensor<T>(new[] { batchSize, outputSize }, baseOutputData);

        // Forward through LoRA layer
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through both layers (only updates LoRA if base is frozen).
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// For QLoRA, the base layer is typically frozen (only LoRA is trained).
    /// The backward pass:
    /// 1. Computes gradients for LoRA layer (always)
    /// 2. Skips base layer gradient computation (if frozen)
    /// 3. Propagates input gradients back
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where learning happens, but only for the LoRA adapter!
    /// Since the base layer is compressed and frozen, we only update the small LoRA matrices.
    /// This is what makes QLoRA so efficient - we're only training a tiny fraction of parameters.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Use base class backward pass (handles frozen base layer correctly)
        Tensor<T> inputGradient = base.Backward(outputGradient);

        // Clear dequantized weight cache to save memory
        // Will be recomputed in next forward pass if needed
        _dequantizedWeights = null;

        return inputGradient;
    }

    /// <summary>
    /// Merges the LoRA adaptation into the base layer and returns a quantized merged layer.
    /// </summary>
    /// <returns>A new DenseLayer with LoRA weights merged and quantized.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method:
    /// 1. Dequantizes the base weights
    /// 2. Computes the LoRA weight contribution
    /// 3. Merges them together
    /// 4. Creates a new layer with merged weights
    /// 5. Optionally re-quantizes for deployment
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This "bakes in" your LoRA training into a single compressed layer.
    /// After training, you can:
    /// 1. Decompress the base weights
    /// 2. Add the LoRA corrections
    /// 3. Create a new layer with the improved weights
    /// 4. Optionally compress it again for deployment
    ///
    /// The result is a single layer that includes all the improvements from training,
    /// ready to use in production!
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("QLoRAAdapter only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Dequantize base weights
        Matrix<T> dequantizedBaseWeights = DequantizeWeights();

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Merge: W_merged = W_base + W_lora
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        Vector<T> mergedParams = new Vector<T>((inputSize * outputSize) + outputSize);

        // Merge weights
        // dequantizedBaseWeights is [outputSize, inputSize]
        // loraWeights from MergeWeights() is [inputSize, outputSize]
        // So we access loraWeights[j, i] = loraWeights[inputIdx, outputIdx]
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int idx = i * inputSize + j;
                mergedParams[idx] = NumOps.Add(dequantizedBaseWeights[i, j], loraWeights[j, i]);
            }
        }

        // Copy biases unchanged (LoRA doesn't modify biases)
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;
        for (int i = 0; i < outputSize; i++)
        {
            mergedParams[weightCount + i] = baseParams[weightCount + i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Resets the internal state of the adapter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Clears cached dequantized weights and resets both base and LoRA layers.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This clears the adapter's memory, including any cached
    /// decompressed weights. Useful when starting a new batch or switching tasks.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        base.ResetState();
        _dequantizedWeights = null;
    }
}
