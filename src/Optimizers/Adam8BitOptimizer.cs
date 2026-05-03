using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements an 8-bit quantized Adam optimizer that reduces memory usage by storing optimizer states in 8-bit format.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// 8-bit Adam provides the same optimization algorithm as standard Adam but uses quantized 8-bit representations
/// for storing the first moment (m) and second moment (v) estimates. This reduces memory usage by approximately
/// 4x for optimizer states, which is particularly beneficial when training large models.
/// </para>
/// <para><b>For Beginners:</b> When training a neural network, the optimizer needs to remember information about
/// past gradients. Standard Adam stores two numbers per parameter (momentum and variance), which can use a lot of
/// memory for large models. 8-bit Adam compresses these numbers, similar to how images are compressed, reducing
/// memory usage while maintaining training quality.
/// </para>
/// <para><b>How It Works:</b>
/// <list type="bullet">
/// <item>Optimizer states are divided into blocks (default 2048 elements each)</item>
/// <item>Each block has its own scaling factor for accurate quantization</item>
/// <item>States are dequantized before computing updates, then requantized after</item>
/// <item>The actual parameter updates use full precision for accuracy</item>
/// </list>
/// </para>
/// <para><b>When to Use:</b>
/// <list type="bullet">
/// <item>Training large models where optimizer memory is a bottleneck</item>
/// <item>GPU training with limited VRAM</item>
/// <item>Distributed training where memory per GPU is constrained</item>
/// </list>
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class Adam8BitOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the 8-bit Adam optimizer.
    /// </summary>
    private Adam8BitOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Quantized first moment vector (moving average of gradients).
    /// Span-optimized <see cref="Vector{T}"/> over <c>byte</c>; backed by
    /// span-aware memory the engine can address without extra copies.
    /// </summary>
    private Vector<byte>? _mQuantized;

    /// <summary>
    /// Quantized second moment vector (moving average of squared gradients).
    /// </summary>
    private Vector<byte>? _vQuantized;

    /// <summary>
    /// Scaling factors for first moment quantization blocks.
    /// </summary>
    private Vector<double>? _mScales;

    /// <summary>
    /// Scaling factors for second moment quantization blocks.
    /// </summary>
    private Vector<double>? _vScales;

    /// <summary>
    /// Full-precision first moment vector (used when CompressBothMoments is false).
    /// </summary>
    private Vector<T>? _mFullPrecision;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// The current value of beta1 (exponential decay rate for first moment estimates).
    /// </summary>
    private T _currentBeta1;

    /// <summary>
    /// The current value of beta2 (exponential decay rate for second moment estimates).
    /// </summary>
    private T _currentBeta2;

    /// <summary>
    /// Number of quantization blocks.
    /// </summary>
    private int _numBlocks;

    /// <summary>
    /// Length of the parameter vector.
    /// </summary>
    private int _parameterLength;

    /// <summary>
    /// Random number generator for stochastic rounding.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the Adam8BitOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the 8-bit Adam optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the 8-bit Adam optimizer with its initial configuration.
    /// The optimizer will use quantized storage for momentum and variance estimates, reducing memory usage.
    /// </para>
    /// </remarks>
    public Adam8BitOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        Adam8BitOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _t = 0;
        _options = options ?? new();
        _currentBeta1 = NumOps.Zero;
        _currentBeta2 = NumOps.Zero;
        _random = RandomHelper.CreateSeededRandom(42);

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the Adam optimizer.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        _currentBeta1 = NumOps.FromDouble(_options.Beta1);
        _currentBeta2 = NumOps.FromDouble(_options.Beta2);
    }

    /// <summary>
    /// Initializes the quantized optimizer state buffers.
    /// </summary>
    /// <param name="length">The number of parameters to optimize.</param>
    private void InitializeQuantizedState(int length)
    {
        _parameterLength = length;
        _numBlocks = (length + _options.BlockSize - 1) / _options.BlockSize;

        // Always-quantized second moment. Vector<byte> is the span-aware
        // wrapper over the byte buffer the engine kernels can address
        // without extra copies.
        _vQuantized = new Vector<byte>(length);
        _vScales = new Vector<double>(_numBlocks);

        if (_options.CompressBothMoments)
        {
            _mQuantized = new Vector<byte>(length);
            _mScales = new Vector<double>(_numBlocks);
            _mFullPrecision = null;

            // For signed quantization, 128 represents 0 (since we map
            // [-127, 127] to [1, 255] with 128 = 0).
            for (int i = 0; i < length; i++) _mQuantized[i] = 128;
        }
        else
        {
            _mQuantized = null;
            _mScales = null;
            _mFullPrecision = new Vector<T>(length);
        }

        // Initialize scales (scale of 1.0 works with the zero-initialized state).
        for (int b = 0; b < _numBlocks; b++)
        {
            if (_mScales is not null) _mScales[b] = 1.0;
            _vScales[b] = 1.0;
        }
    }

    /// <summary>
    /// Quantizes a full-precision vector to 8-bit representation.
    /// </summary>
    /// <param name="values">The full-precision values to quantize.</param>
    /// <param name="quantized">The output quantized byte vector (span-backed).</param>
    /// <param name="scales">The output scaling factors per block.</param>
    /// <param name="isSigned">Whether to use signed quantization (for m) or unsigned (for v).</param>
    private void Quantize(Vector<T> values, Vector<byte> quantized, Vector<double> scales, bool isSigned)
    {
        for (int b = 0; b < _numBlocks; b++)
        {
            int blockStart = b * _options.BlockSize;
            int blockEnd = Math.Min(blockStart + _options.BlockSize, _parameterLength);

            // Find the scale for this block
            double maxAbs = 0;
            if (_options.QuantizationPercentile >= 100)
            {
                // Use absolute maximum
                for (int i = blockStart; i < blockEnd; i++)
                {
                    double val = Math.Abs(NumOps.ToDouble(values[i]));
                    if (val > maxAbs) maxAbs = val;
                }
            }
            else
            {
                // Use percentile-based scale (collect values, sort, take percentile)
                var absValues = new List<double>(blockEnd - blockStart);
                for (int i = blockStart; i < blockEnd; i++)
                {
                    absValues.Add(Math.Abs(NumOps.ToDouble(values[i])));
                }
                absValues.Sort();
                int percentileIdx = (int)((absValues.Count - 1) * _options.QuantizationPercentile / 100.0);
                maxAbs = absValues[percentileIdx];
            }

            // Compute scale (with small epsilon to avoid division by zero)
            double scale = maxAbs / (isSigned ? 127.0 : 255.0);
            if (scale < 1e-10) scale = 1e-10;
            scales[b] = scale;

            // Quantize values in this block
            for (int i = blockStart; i < blockEnd; i++)
            {
                double val = NumOps.ToDouble(values[i]);
                double scaled = val / scale;

                // Apply rounding
                int quantizedVal;
                if (_options.UseStochasticRounding)
                {
                    double floor = Math.Floor(scaled);
                    double frac = scaled - floor;
                    quantizedVal = (int)(floor + (_random.NextDouble() < frac ? 1 : 0));
                }
                else
                {
                    quantizedVal = (int)Math.Round(scaled);
                }

                // Clamp to valid range
                if (isSigned)
                {
                    quantizedVal = MathHelper.Clamp(quantizedVal, -127, 127);
                    quantized[i] = (byte)(quantizedVal + 128); // Map [-127, 127] to [1, 255], with 128 representing 0 (0 is unused in the stored range)
                }
                else
                {
                    quantizedVal = MathHelper.Clamp(quantizedVal, 0, 255);
                    quantized[i] = (byte)quantizedVal;
                }
            }
        }
    }

    /// <summary>
    /// Dequantizes an 8-bit representation back to full precision.
    /// </summary>
    /// <param name="quantized">The quantized byte vector.</param>
    /// <param name="scales">The scaling factors per block.</param>
    /// <param name="isSigned">Whether the quantization used signed format.</param>
    /// <returns>The dequantized full-precision vector.</returns>
    private Vector<T> Dequantize(Vector<byte> quantized, Vector<double> scales, bool isSigned)
    {
        var result = new Vector<T>(_parameterLength);

        for (int b = 0; b < _numBlocks; b++)
        {
            int blockStart = b * _options.BlockSize;
            int blockEnd = Math.Min(blockStart + _options.BlockSize, _parameterLength);
            double scale = scales[b];

            for (int i = blockStart; i < blockEnd; i++)
            {
                double quantizedVal;
                if (isSigned)
                {
                    quantizedVal = (int)quantized[i] - 128; // Map [1, 255] back to [-127, 127]
                }
                else
                {
                    quantizedVal = quantized[i];
                }

                result[i] = NumOps.FromDouble(quantizedVal * scale);
            }
        }

        return result;
    }

    /// <summary>
    /// Performs the optimization process using the 8-bit Adam algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();

        InitializeQuantizedState(parameters.Length);
        _t = 0;

        InitializeAdaptiveParameters();

        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);

            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _t++;
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                var newSolution = UpdateSolution(currentSolution, gradient);
                currentSolution = newSolution;
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(
                NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)),
                NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Per-parameter quantized Adam state for the tape training path.
    /// Each registered parameter tensor gets its own block-quantized m and v
    /// estimates stored as byte[] (signed [-127, 127] mapped to [1, 255] for m,
    /// unsigned [0, 255] for v) plus per-block scaling factors. When
    /// <see cref="Adam8BitOptimizerOptions{T,TInput,TOutput}.CompressBothMoments"/>
    /// is false, m is kept as a full-precision Tensor instead — matching the
    /// legacy <see cref="UpdateSolution"/> path's contract.
    /// </summary>
    /// <remarks>
    /// Allocated lazily on the first Step() that sees the parameter; the byte[]
    /// pair plus block scales replaces what would have been
    /// 2 × (parameter.Length × sizeof(T)) bytes of full-precision Tensor state.
    /// For a 300 M-parameter foundation model at fp64 this drops the optimizer's
    /// resident state from ~4.8 GB to ~600 MB (the 8× reduction the class name
    /// promised but was not delivering before this fix).
    /// </remarks>
    private sealed class QuantizedTapeState
    {
        public int Length;
        public int NumBlocks;
        public Vector<byte>? MQuantized;        // null when CompressBothMoments == false
        public Tensor<T>? MFullPrecision;       // null when CompressBothMoments == true
        public Vector<byte> VQuantized = new(0);
        public Vector<double>? MScales;         // null when CompressBothMoments == false
        public Vector<double> VScales = new(0);
    }

    private readonly Dictionary<Tensor<T>, QuantizedTapeState> _tapeStates =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private int _tapeStep;

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        _tapeStep++;

        T beta1 = _currentBeta1;
        T beta2 = _currentBeta2;
        T oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        T oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(Convert.ToDouble(beta1), _tapeStep));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(Convert.ToDouble(beta2), _tapeStep));

        foreach (var param in context.Parameters)
        {
            if (!context.Gradients.TryGetValue(param, out var grad))
                continue;

            // Look up or lazily allocate the per-parameter quantized state. The
            // byte[] storage replaces the full-precision Tensor pair the original
            // Step path was holding, which is the whole point of Adam8Bit — see
            // QuantizedTapeState's remarks for the memory math.
            //
            // Shape-mismatch guard mirrors AdamOptimizer.Step's: if the
            // parameter was first seen at a lazy-init placeholder shape
            // (e.g., a MultiHeadAttentionLayer that hadn't yet seen its
            // first Forward), our cached state's byte[] / scale buffers
            // were sized for the placeholder. Once the real weights
            // materialize the parameter length grows; without a re-alloc
            // here, DequantizeTensor / QuantizeTensor would index past
            // the end of the stored arrays.
            if (!_tapeStates.TryGetValue(param, out var state) || state.Length != param.Length)
            {
                state = AllocateTapeState(param.Length);
                _tapeStates[param] = state;
            }

            // Reshape gradient to match parameter shape when element counts
            // match — same fix as AdamOptimizer.Step. Reshape() adds/removes
            // batch dimensions in some forward paths, leaving grad and param
            // with different _shape arrays but identical Length. The math
            // ops below assume shape compatibility; without this guard,
            // TensorAdd would throw on a length-equal-but-shape-different
            // pair.
            if (!param._shape.SequenceEqual(grad._shape) && param.Length == grad.Length)
            {
                grad = Engine.Reshape(grad, param._shape);
            }

            // Dequantize moments into transient Tensors for the math path. These
            // are scoped to this iteration only — when Step returns, the
            // engine's TensorArena reclaims them, leaving only the byte[] state
            // in resident memory.
            Tensor<T> m;
            if (_options.CompressBothMoments)
            {
                m = DequantizeTensor(state.MQuantized!, state.MScales!, param._shape, state.NumBlocks, isSigned: true);
            }
            else
            {
                // Lazy-allocate the full-precision m on the first Step that
                // sees this parameter — AllocateTapeState intentionally
                // leaves state.MFullPrecision null because the parameter's
                // _shape isn't known until Step actually runs (the
                // optimizer's tape state is keyed by Tensor reference, not
                // by an a-priori-known shape). Zero-initialization on first
                // alloc matches Adam's m_0 = 0 initial condition.
                state.MFullPrecision ??= new Tensor<T>(param._shape);
                m = state.MFullPrecision;
            }
            Tensor<T> v = DequantizeTensor(state.VQuantized, state.VScales, param._shape, state.NumBlocks, isSigned: false);

            // Update biased first / second moments — same recurrences the legacy
            // UpdateSolution path uses, expressed against engine Tensor ops:
            //     m_t = beta1·m_{t-1} + (1-beta1)·g
            //     v_t = beta2·v_{t-1} + (1-beta2)·g²
            var newM = Engine.TensorAdd(Engine.TensorMultiplyScalar(m, beta1),
                                        Engine.TensorMultiplyScalar(grad, oneMinusBeta1));
            var newV = Engine.TensorAdd(Engine.TensorMultiplyScalar(v, beta2),
                                        Engine.TensorMultiplyScalar(Engine.TensorMultiply(grad, grad), oneMinusBeta2));

            // Re-quantize the updated moments back into the byte[] state. After
            // this the transient newM / newV Tensors are no longer reachable
            // and the arena will reclaim their backing memory on Step exit;
            // only state.MQuantized, state.VQuantized, and (if applicable)
            // state.MFullPrecision remain resident.
            if (_options.CompressBothMoments)
            {
                QuantizeTensor(newM, state.MQuantized!, state.MScales!, state.NumBlocks, isSigned: true);
            }
            else
            {
                // Copy newM's values into the persistent state.MFullPrecision
                // tensor in place rather than replacing the reference. The
                // engine's tensor ops (TensorAdd / TensorMultiplyScalar) return
                // arena-allocated tensors that the arena will reclaim on Step
                // exit — assigning newM to state.MFullPrecision would either
                // (a) retain a tensor backed by reclaimed memory, or (b) keep
                // the arena allocation alive across Step calls and bypass the
                // arena's per-iteration recycling. TensorCopy keeps the
                // long-lived state on a stable backing buffer.
                Engine.TensorCopy(newM, state.MFullPrecision!);
            }
            QuantizeTensor(newV, state.VQuantized, state.VScales, state.NumBlocks, isSigned: false);

            // Apply the bias-corrected Adam update directly to the parameter.
            //     update = lr · (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)
            var mHat = Engine.TensorDivideScalar(newM, biasCorrection1);
            var vHat = Engine.TensorDivideScalar(newV, biasCorrection2);
            var denom = Engine.TensorAddScalar(Engine.TensorSqrt(vHat), epsilon);
            var update = Engine.TensorMultiplyScalar(Engine.TensorDivide(mHat, denom), CurrentLearningRate);
            Engine.TensorSubtractInPlace(param, update);
        }
    }

    /// <summary>
    /// Allocates a freshly-zeroed <see cref="QuantizedTapeState"/> sized for a
    /// parameter tensor of the given length. Block count is derived from
    /// <see cref="Adam8BitOptimizerOptions{T,TInput,TOutput}.BlockSize"/>; each
    /// block carries its own scale so per-block magnitude variation doesn't get
    /// crushed into a single global scale.
    /// </summary>
    private QuantizedTapeState AllocateTapeState(int paramLength)
    {
        int blockSize = _options.BlockSize;
        int numBlocks = (paramLength + blockSize - 1) / blockSize;

        var state = new QuantizedTapeState
        {
            Length = paramLength,
            NumBlocks = numBlocks,
            VQuantized = new Vector<byte>(paramLength),
            VScales = new Vector<double>(numBlocks),
        };
        // v starts at zero. Unsigned byte 0 maps to the zero quantization
        // bucket, so default-init is correct.
        for (int b = 0; b < numBlocks; b++) state.VScales[b] = 1.0;

        if (_options.CompressBothMoments)
        {
            state.MQuantized = new Vector<byte>(paramLength);
            state.MScales = new Vector<double>(numBlocks);
            // m starts at zero. For signed quantization 0 is encoded as 128
            // (the [-127, 127] → [1, 255] offset), so initialize to 128.
            for (int i = 0; i < paramLength; i++) state.MQuantized[i] = 128;
            for (int b = 0; b < numBlocks; b++) state.MScales[b] = 1.0;
        }
        else
        {
            // Full-precision m allocated on first Step iteration once the
            // parameter's shape is observed.
            state.MFullPrecision = null;
        }

        return state;
    }

    /// <summary>
    /// Block-quantizes a tensor's values into a pre-allocated byte buffer.
    /// Each block of <see cref="Adam8BitOptimizerOptions{T,TInput,TOutput}.BlockSize"/>
    /// elements gets its own scale (max-abs or percentile-based) so per-block
    /// magnitude variation is preserved. Mirrors the legacy
    /// <see cref="Quantize"/> Vector path but works against a Tensor without
    /// requiring shared instance state, so it can run from the tape Step where
    /// many parameters of different sizes coexist.
    /// </summary>
    private void QuantizeTensor(Tensor<T> values, Vector<byte> quantized, Vector<double> scales, int numBlocks, bool isSigned)
    {
        int blockSize = _options.BlockSize;
        int totalLength = values.Length;

        // Reuse a single rented buffer across all blocks for the percentile
        // path. Previously each block allocated a fresh List<double> and
        // fully sorted it — at default QuantizationPercentile=99.9 with a
        // foundation-scale model (300 M params, blockSize=64 → ~4.7 M
        // blocks per Step) this was the dominant allocator hotspot. ArrayPool
        // amortizes the allocation across the whole tensor; we still pay a
        // sort per block but no GC pressure for the buffer itself.
        double[]? rentedBuffer = null;
        try
        {
        for (int b = 0; b < numBlocks; b++)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, totalLength);

            double maxAbs = 0;
            if (_options.QuantizationPercentile >= 100)
            {
                for (int i = blockStart; i < blockEnd; i++)
                {
                    double val = Math.Abs(NumOps.ToDouble(values[i]));
                    if (val > maxAbs) maxAbs = val;
                }
            }
            else
            {
                int blockLen = blockEnd - blockStart;
                rentedBuffer ??= System.Buffers.ArrayPool<double>.Shared.Rent(blockSize);
                for (int i = 0; i < blockLen; i++)
                    rentedBuffer[i] = Math.Abs(NumOps.ToDouble(values[blockStart + i]));
                Array.Sort(rentedBuffer, 0, blockLen);
                int percentileIdx = (int)((blockLen - 1) * _options.QuantizationPercentile / 100.0);
                maxAbs = rentedBuffer[percentileIdx];
            }

            double scale = maxAbs / (isSigned ? 127.0 : 255.0);
            if (scale < 1e-10) scale = 1e-10;
            scales[b] = scale;

            for (int i = blockStart; i < blockEnd; i++)
            {
                double val = NumOps.ToDouble(values[i]);
                double scaled = val / scale;

                int quantizedVal;
                if (_options.UseStochasticRounding)
                {
                    double floor = Math.Floor(scaled);
                    double frac = scaled - floor;
                    quantizedVal = (int)(floor + (_random.NextDouble() < frac ? 1 : 0));
                }
                else
                {
                    quantizedVal = (int)Math.Round(scaled);
                }

                if (isSigned)
                {
                    quantizedVal = MathHelper.Clamp(quantizedVal, -127, 127);
                    quantized[i] = (byte)(quantizedVal + 128);
                }
                else
                {
                    quantizedVal = MathHelper.Clamp(quantizedVal, 0, 255);
                    quantized[i] = (byte)quantizedVal;
                }
            }
        }
        }
        finally
        {
            if (rentedBuffer is not null)
                System.Buffers.ArrayPool<double>.Shared.Return(rentedBuffer);
        }
    }

    /// <summary>
    /// Block-dequantizes an 8-bit byte buffer into a freshly-allocated tensor
    /// of the supplied shape. The transient tensor is intended to be consumed
    /// by Adam's compute path within a single Step iteration and then released
    /// to the engine arena.
    /// </summary>
    private Tensor<T> DequantizeTensor(Vector<byte> quantized, Vector<double> scales, int[] paramShape, int numBlocks, bool isSigned)
    {
        var result = new Tensor<T>(paramShape);
        int blockSize = _options.BlockSize;
        int totalLength = result.Length;
        for (int b = 0; b < numBlocks; b++)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, totalLength);
            double scale = scales[b];

            for (int i = blockStart; i < blockEnd; i++)
            {
                double quantizedVal = isSigned ? (int)quantized[i] - 128 : (int)quantized[i];
                result[i] = NumOps.FromDouble(quantizedVal * scale);
            }
        }
        return result;
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            CurrentLearningRate = MathHelper.Max(NumOps.FromDouble(_options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(_options.MaxLearningRate), CurrentLearningRate));
        }

        if (_options.UseAdaptiveBetas)
        {
            _currentBeta1 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta1),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta1), _currentBeta1));
            _currentBeta2 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta2),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta2), _currentBeta2));
        }
    }

    /// <summary>
    /// Updates the current solution using the 8-bit Adam update rule.
    /// </summary>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();

        if (_mQuantized == null && _mFullPrecision == null)
        {
            InitializeQuantizedState(parameters.Length);
        }

        // Dequantize current moment estimates
        Vector<T> m;
        if (_options.CompressBothMoments)
        {
            m = Dequantize(_mQuantized!, _mScales!, isSigned: true);
        }
        else
        {
            m = _mFullPrecision!;
        }
        var v = Dequantize(_vQuantized!, _vScales!, isSigned: false);

        // Compute Adam update using full precision
        T beta1 = _currentBeta1;
        T beta2 = _currentBeta2;
        T oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        T oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        T biasCorrection1 = NumOps.Subtract(NumOps.One, NumOps.Power(beta1, NumOps.FromDouble(_t)));
        T biasCorrection2 = NumOps.Subtract(NumOps.One, NumOps.Power(beta2, NumOps.FromDouble(_t)));
        T epsilon = NumOps.FromDouble(_options.Epsilon);

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Re-quantize the updated moments
        if (_options.CompressBothMoments)
        {
            Quantize(m, _mQuantized!, _mScales!, isSigned: true);
        }
        else
        {
            _mFullPrecision = m;
        }
        Quantize(v, _vQuantized!, _vScales!, isSigned: false);

        // Compute bias-corrected first moment: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(m, biasCorrection1);

        // Compute bias-corrected second moment: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(v, biasCorrection2);

        // Compute update: update = learningRate * mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var updateDiv = (Vector<T>)Engine.Divide(mHat, denominator);
        var update = (Vector<T>)Engine.Multiply(updateDiv, CurrentLearningRate);

        // Apply update: parameters = parameters - update
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, update);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(updatedParams);
    }

    /// <summary>
    /// Updates a vector of parameters using the 8-bit Adam optimization algorithm.
    /// </summary>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_mQuantized == null && _mFullPrecision == null || _parameterLength != parameters.Length)
        {
            InitializeQuantizedState(parameters.Length);
        }

        _t++;

        // Dequantize current moment estimates
        Vector<T> m;
        if (_options.CompressBothMoments)
        {
            m = Dequantize(_mQuantized!, _mScales!, isSigned: true);
        }
        else
        {
            m = _mFullPrecision!;
        }
        var v = Dequantize(_vQuantized!, _vScales!, isSigned: false);

        // Compute Adam update using full precision
        // Use adaptive betas (consistent with UpdateSolution when UseAdaptiveBetas is enabled)
        T beta1 = _currentBeta1;
        T beta2 = _currentBeta2;
        T oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        T oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        double beta1Double = Convert.ToDouble(beta1);
        double beta2Double = Convert.ToDouble(beta2);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(beta1Double, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(beta2Double, _t));

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Re-quantize the updated moments
        if (_options.CompressBothMoments)
        {
            Quantize(m, _mQuantized!, _mScales!, isSigned: true);
        }
        else
        {
            _mFullPrecision = m;
        }
        Quantize(v, _vQuantized!, _vScales!, isSigned: false);

        // Compute bias-corrected moments
        var mHat = (Vector<T>)Engine.Divide(m, biasCorrection1);
        var vHat = (Vector<T>)Engine.Divide(v, biasCorrection2);

        // Compute update
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var update = (Vector<T>)Engine.Divide(mHat, denominator);
        var scaledUpdate = (Vector<T>)Engine.Multiply(update, CurrentLearningRate);

        // Apply update
        return (Vector<T>)Engine.Subtract(parameters, scaledUpdate);
    }

    /// <summary>
    /// Updates a matrix of parameters using the 8-bit Adam optimization algorithm.
    /// </summary>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        int totalSize = parameters.Rows * parameters.Columns;

        // Flatten matrix to vector
        var paramVec = new Vector<T>(totalSize);
        var gradVec = new Vector<T>(totalSize);
        int idx = 0;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                paramVec[idx] = parameters[i, j];
                gradVec[idx] = gradient[i, j];
                idx++;
            }
        }

        // Update using vector method
        var updatedVec = UpdateParameters(paramVec, gradVec);

        // Unflatten vector back to matrix
        var updatedMatrix = new Matrix<T>(parameters.Rows, parameters.Columns);
        idx = 0;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                updatedMatrix[i, j] = updatedVec[idx];
                idx++;
            }
        }

        return updatedMatrix;
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _mQuantized = null;
        _vQuantized = null;
        _mScales = null;
        _vScales = null;
        _mFullPrecision = null;
        _t = 0;
        _parameterLength = 0;
        _numBlocks = 0;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is Adam8BitOptimizerOptions<T, TInput, TOutput> adamOptions)
        {
            _options = adamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected Adam8BitOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Gets the memory usage statistics for this optimizer.
    /// </summary>
    /// <returns>A dictionary containing memory usage information.</returns>
    public Dictionary<string, long> GetMemoryUsage()
    {
        var stats = new Dictionary<string, long>();

        // Quantized state memory
        long quantizedStateMemory = 0;
        if (_mQuantized != null) quantizedStateMemory += _mQuantized.Length;
        if (_vQuantized != null) quantizedStateMemory += _vQuantized.Length;

        // Scaling factors memory (8 bytes per double)
        long scalesMemory = 0;
        if (_mScales != null) scalesMemory += _mScales.Length * 8;
        if (_vScales != null) scalesMemory += _vScales.Length * 8;

        // Type size for T (float = 4 bytes, double = 8 bytes)
        int bytesPerElement = typeof(T) == typeof(float) ? 4 : 8;

        // Full precision state memory (if used)
        long fullPrecisionMemory = 0;
        if (_mFullPrecision != null)
        {
            fullPrecisionMemory += _mFullPrecision.Length * bytesPerElement;
        }

        stats["QuantizedStateBytes"] = quantizedStateMemory;
        stats["ScalingFactorBytes"] = scalesMemory;
        stats["FullPrecisionStateBytes"] = fullPrecisionMemory;
        stats["TotalBytes"] = quantizedStateMemory + scalesMemory + fullPrecisionMemory;

        // Calculate savings compared to standard Adam
        long standardAdamMemory = _parameterLength * 2 * bytesPerElement; // m and v at full precision
        stats["StandardAdamBytes"] = standardAdamMemory;
        stats["MemorySavingsBytes"] = standardAdamMemory - stats["TotalBytes"];

        return stats;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize options
            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            // Serialize 8-bit Adam-specific state
            writer.Write(_t);
            writer.Write(_parameterLength);
            writer.Write(_numBlocks);

            // Serialize quantized first moment (if used)
            writer.Write(_options.CompressBothMoments);
            if (_options.CompressBothMoments && _mQuantized is not null)
            {
                writer.Write(_mQuantized.Length);
                for (int i = 0; i < _mQuantized.Length; i++) writer.Write(_mQuantized[i]);
                foreach (var scale in _mScales!)
                {
                    writer.Write(scale);
                }
            }
            else if (_mFullPrecision is not null)
            {
                writer.Write(_mFullPrecision.Length);
                foreach (var value in _mFullPrecision)
                {
                    writer.Write(Convert.ToDouble(value));
                }
            }

            // Serialize quantized second moment
            writer.Write(_vQuantized is not null);
            if (_vQuantized is not null)
            {
                writer.Write(_vQuantized.Length);
                for (int i = 0; i < _vQuantized.Length; i++) writer.Write(_vQuantized[i]);
                foreach (var scale in _vScales!)
                {
                    writer.Write(scale);
                }
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize options
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<Adam8BitOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize state
            _t = reader.ReadInt32();
            _parameterLength = reader.ReadInt32();
            _numBlocks = reader.ReadInt32();

            // Deserialize first moment
            bool compressBothMoments = reader.ReadBoolean();
            if (compressBothMoments)
            {
                int mLength = reader.ReadInt32();
                var mBytes = reader.ReadBytes(mLength);
                _mQuantized = new Vector<byte>(mLength);
                for (int i = 0; i < mLength; i++) _mQuantized[i] = mBytes[i];
                _mScales = new Vector<double>(_numBlocks);
                for (int i = 0; i < _numBlocks; i++)
                {
                    _mScales[i] = reader.ReadDouble();
                }
            }
            else
            {
                int mLength = reader.ReadInt32();
                _mFullPrecision = new Vector<T>(mLength);
                for (int i = 0; i < mLength; i++)
                {
                    _mFullPrecision[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }

            // Deserialize second moment
            bool hasVQuantized = reader.ReadBoolean();
            if (hasVQuantized)
            {
                int vLength = reader.ReadInt32();
                var vBytes = reader.ReadBytes(vLength);
                _vQuantized = new Vector<byte>(vLength);
                for (int i = 0; i < vLength; i++) _vQuantized[i] = vBytes[i];
                _vScales = new Vector<double>(_numBlocks);
                for (int i = 0; i < _numBlocks; i++)
                {
                    _vScales[i] = reader.ReadDouble();
                }
            }

            InitializeAdaptiveParameters();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Adam8Bit_{_options.InitialLearningRate}_{_options.MaxIterations}_{_options.BlockSize}";
    }
}
