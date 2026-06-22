using AiDotNet.Helpers;
using System.Buffers;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.MixedPrecision;

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
    /// Magic header for the v2 checkpoint format ("A8B1" in ASCII LE).
    /// Written immediately after the options JSON in <see cref="Serialize"/>
    /// and validated as the first read in <see cref="Deserialize"/> so v1
    /// payloads (which wrote <c>_t</c> at this position) can't be silently
    /// mis-detected as v2. See Serialize/Deserialize for design notes.
    /// </summary>
    private const int Adam8BitV2Magic = unchecked((int)0x31423841);

    /// <summary>
    /// Current checkpoint format version. Bumped whenever the byte layout
    /// after the magic header changes in a non-backward-compatible way;
    /// readers reject mismatched versions with a clear migration message.
    /// </summary>
    private const int StateFormatVersion = 2;

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
        // Blocks are independent (disjoint slices + own scale[b]) — parallelize.
        // Stochastic rounding uses RandomHelper.ThreadSafeRandom (per-thread
        // LockedRandom) so it stays thread-safe; exact seed reproducibility can't
        // survive parallel work-stealing regardless, and the default path is
        // deterministic round-to-nearest (UseStochasticRounding == false).
        int blockSize = _options.BlockSize;
        int length = _parameterLength;
        CpuParallelSettings.ParallelForOrSerial(0, _numBlocks, (long)length, b =>
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, length);

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
                // Use percentile-based scale (collect values, sort, take percentile).
                int count = blockEnd - blockStart;
                var absValues = ArrayPool<double>.Shared.Rent(count);
                try
                {
                    for (int i = blockStart; i < blockEnd; i++)
                    {
                        absValues[i - blockStart] = Math.Abs(NumOps.ToDouble(values[i]));
                    }

                    Array.Sort(absValues, 0, count);
                    int percentileIdx = (int)((count - 1) * _options.QuantizationPercentile / 100.0);
                    maxAbs = absValues[percentileIdx];
                }
                finally
                {
                    ArrayPool<double>.Shared.Return(absValues);
                }
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
                    quantizedVal = (int)(floor + (RandomHelper.ThreadSafeRandom.NextDouble() < frac ? 1 : 0));
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
        });
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

        // Blocks are independent (disjoint slices + own scale) — parallelize over
        // them; the grain gate keeps small parameters serial.
        int blockSize = _options.BlockSize;
        int length = _parameterLength;
        CpuParallelSettings.ParallelForOrSerial(0, _numBlocks, (long)length, b =>
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, length);
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
        });

        return result;
    }

    /// <summary>
    /// Performs the optimization process using the 8-bit Adam algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
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

            if (IsConvergedAgainstPreviousEpoch(epoch, currentStepData, previousStepData, _options.Tolerance))
            {
                // H6 convergence fix (PR #1364): compare CURRENT vs PREVIOUS
                // epoch (not bestStepData — UpdateBestSolution copies
                // currentStepData into bestStepData on epoch 0, so |best -
                // current| = 0 < tolerance would falsely converge). Skip
                // check on epoch 0 where previousStepData is the pre-training
                // baseline. Helper is on GradientBasedOptimizerBase.
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Per-parameter quantized Adam state for the tape training path.
    /// Each registered parameter tensor gets its own block-quantized m
    /// and v estimates stored in span-backed <see cref="Vector{T}"/>
    /// over <c>byte</c> buffers (signed [-127, 127] mapped to [1, 255]
    /// for m, unsigned [0, 255] for v) plus per-block scaling factors
    /// in span-backed <see cref="Vector{T}"/> over <c>double</c>. When
    /// <see cref="Adam8BitOptimizerOptions{T,TInput,TOutput}.CompressBothMoments"/>
    /// is false, m is kept as a full-precision Tensor instead — matching the
    /// legacy <see cref="UpdateSolution"/> path's contract.
    /// </summary>
    /// <remarks>
    /// Allocated lazily on the first Step() that sees the parameter. The
    /// moment storage is a <see cref="Vector{T}"/> over <c>byte</c> (the
    /// span-backed wrapper this codebase uses for all optimizer state)
    /// plus a per-block <see cref="Vector{T}"/> over <c>double</c> for
    /// scales. Together these replace what would have been
    /// 2 × (parameter.Length × sizeof(T)) bytes of full-precision Tensor
    /// state. For a 300 M-parameter foundation model at fp64 this drops
    /// the optimizer's resident state from ~4.8 GB to ~600 MB (the 8×
    /// reduction the class name promised but was not delivering before
    /// this fix).
    /// </remarks>
    private sealed class QuantizedTapeState
    {
        public int Length;
        public int NumBlocks;
        public Vector<byte>? MQuantized;        // null when CompressBothMoments == false
        public Tensor<T>? MFullPrecision;       // null when CompressBothMoments == true
        // Initialized to null! — AllocateTapeState always overwrites these
        // before the state is reachable from anywhere else, so the
        // immediate-discard `new(0)` defaults were just GC pressure.
        public Vector<byte> VQuantized = null!;
        public Vector<double>? MScales;         // null when CompressBothMoments == false
        public Vector<double> VScales = null!;

        // BF16 moment storage (UseBFloat16MomentStorage == true): 2 bytes/element, no per-block
        // scales. Mutually exclusive with the byte-quantized fields above — only one set is allocated.
        public Vector<ushort>? MBf16;
        public Vector<ushort>? VBf16;

        // GPU-resident 8-bit state (AIDOTNET_GPU_ADAM=1, CUDA): int8 m/v + per-block
        // double scales kept on the device across steps so the adam8bit_update kernel
        // runs the whole dequant→Adam→requant cycle with no host download. Allocated
        // lazily on the first GPU step for this parameter; null on the CPU path.
        public AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer? GpuMQ;
        public AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer? GpuVQ;
        public AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer? GpuMScales;
        public AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer? GpuVScales;
        public bool GpuResident;
    }

    private readonly ConcurrentDictionary<Tensor<T>, QuantizedTapeState> _tapeStates =
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

        // GPU-resident 8-bit Adam (AIDOTNET_GPU_ADAM=1, CUDA): the adam8bit_update
        // kernel does the whole blockwise dequant→Adam→requant on the device with no
        // host download. Only the kernel-matched config (both moments compressed,
        // absolute-max scale, deterministic rounding) is eligible; otherwise the CPU
        // path runs. Quantized state is kept GPU-resident per parameter across steps.
        bool gpu8 = typeof(T) == typeof(float)
            && !_options.UseBFloat16MomentStorage
            && System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_ADAM") == "1"
            && AiDotNet.Tensors.Engines.AiDotNetEngine.Current is AiDotNet.Tensors.Engines.DirectGpuTensorEngine
            && _options.CompressBothMoments
            && _options.QuantizationPercentile >= 100
            && !_options.UseStochasticRounding;

        foreach (var param in context.Parameters)
        {
            // True sparse scatter Adam8Bit: dequant + Adam + requant only on the
            // BLOCKS that contain touched indices. The block granularity is
            // necessary because changing a block's per-block scale re-interprets
            // every byte in that block — we can't touch one byte without
            // re-encoding the rest at the new scale. Only the most-common
            // configuration is eligible (compressBothMoments=true,
            // percentile>=100, no stochastic rounding); other configs fall
            // through to the dense ToDense path so quantization semantics stay
            // bit-identical with the dense code.
            if (!gpu8 && !_options.UseBFloat16MomentStorage && SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(param))
            {
                // Lazily allocate quantized state at the parameter's actual length —
                // mirroring the same shape-mismatch handling as the dense path below
                // so a lazy-init shape change is caught BEFORE the sparse helper runs
                // (the helper assumes Length matches between param and state).
                if (!_tapeStates.TryGetValue(param, out var stateSp) || stateSp.Length != param.Length)
                {
                    if (stateSp is not null && stateSp.GpuResident)
                    {
                        AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.FreeGpuBuffer(stateSp.GpuMQ);
                        AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.FreeGpuBuffer(stateSp.GpuVQ);
                        AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.FreeGpuBuffer(stateSp.GpuMScales);
                        AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.FreeGpuBuffer(stateSp.GpuVScales);
                    }
                    stateSp = AllocateTapeState(param.Length);
                    _tapeStates[param] = stateSp;
                }
                if (SparseEmbeddingOptimizerHelpers.TryApplyAdam8BitSparse(
                        param,
                        stateSp.MQuantized, stateSp.MScales,
                        stateSp.VQuantized, stateSp.VScales,
                        _options.BlockSize, stateSp.NumBlocks,
                        NumOps.ToDouble(CurrentLearningRate),
                        NumOps.ToDouble(beta1), NumOps.ToDouble(beta2),
                        NumOps.ToDouble(biasCorrection1), NumOps.ToDouble(biasCorrection2),
                        _options.Epsilon,
                        _options.CompressBothMoments,
                        _options.QuantizationPercentile,
                        _options.UseStochasticRounding))
                {
                    continue;
                }
            }

            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            // Look up or lazily allocate the per-parameter quantized state. The
            // byte[] storage replaces the full-precision Tensor pair the original
            // Step path was holding, which is the whole point of Adam8Bit — see
            // QuantizedTapeState's remarks for the memory math.
            //
            // Shape-mismatch guard mirrors AdamOptimizer.Step's: if the
            // parameter was first seen at a lazy-init placeholder shape
            // (e.g., a MultiHeadAttentionLayer that hadn't yet seen its
            // first Forward), our cached state's Vector<byte> /
            // Vector<double> scale buffers were sized for the placeholder.
            // Once the real weights materialize the parameter length grows;
            // without a re-alloc here, DequantizeTensor / QuantizeTensor
            // would index past the end of the stored vectors.
            if (!_tapeStates.TryGetValue(param, out var state) || state.Length != param.Length)
            {
                // Free any GPU-resident quant state from the stale (wrong-length) entry
                // before dropping it, so a shape change doesn't leak device buffers.
                if (state is not null && state.GpuResident)
                {
                    AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.FreeGpuBuffer(state.GpuMQ);
                    AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.FreeGpuBuffer(state.GpuVQ);
                    AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.FreeGpuBuffer(state.GpuMScales);
                    AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.FreeGpuBuffer(state.GpuVScales);
                }
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

            // BF16 moment storage: expand the 2-byte moments to a transient full-precision tensor,
            // run the identical Adam recurrence + update, then re-pack to BF16. Only this parameter's
            // moments are materialized at a time (per-parameter loop), so the resident footprint stays
            // at 2 bytes/element while the math runs at full precision.
            if (_options.UseBFloat16MomentStorage)
            {
                Tensor<T> mB = Bf16ToTensor(state.MBf16!, param._shape);
                Tensor<T> vB = Bf16ToTensor(state.VBf16!, param._shape);

                var newMB = Engine.TensorAdd(Engine.TensorMultiplyScalar(mB, beta1),
                                             Engine.TensorMultiplyScalar(grad, oneMinusBeta1));
                var newVB = Engine.TensorAdd(Engine.TensorMultiplyScalar(vB, beta2),
                                             Engine.TensorMultiplyScalar(Engine.TensorMultiply(grad, grad), oneMinusBeta2));

                TensorToBf16(newMB, state.MBf16!);
                TensorToBf16(newVB, state.VBf16!);

                var mHatB = Engine.TensorDivideScalar(newMB, biasCorrection1);
                var vHatB = Engine.TensorDivideScalar(newVB, biasCorrection2);
                var denomB = Engine.TensorAddScalar(Engine.TensorSqrt(vHatB), epsilon);
                var updateB = Engine.TensorMultiplyScalar(Engine.TensorDivide(mHatB, denomB), CurrentLearningRate);
                Engine.TensorSubtractInPlace(param, updateB);
                continue;
            }

            // GPU-resident 8-bit step: lazily allocate the device quant state on
            // first sight of this parameter, then run the in-place kernel. Skips the
            // CPU dequant/quant path entirely when param/grad resolve to GPU buffers.
            if (gpu8 && param.Length == grad.Length)
            {
                if (!state.GpuResident
                    && AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAllocAdam8BitState(param.Length, _options.BlockSize,
                        out state.GpuMQ, out state.GpuVQ, out state.GpuMScales, out state.GpuVScales))
                {
                    state.GpuResident = true;
                }
                if (state.GpuResident
                    && AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdam8BitStep(
                        (Tensor<float>)(object)param, (Tensor<float>)(object)grad,
                        state.GpuMQ, state.GpuVQ, state.GpuMScales, state.GpuVScales,
                        (float)NumOps.ToDouble(CurrentLearningRate), (float)NumOps.ToDouble(beta1), (float)NumOps.ToDouble(beta2),
                        (float)NumOps.ToDouble(epsilon), (float)NumOps.ToDouble(biasCorrection1), (float)NumOps.ToDouble(biasCorrection2),
                        _options.BlockSize))
                {
                    continue; // weights + quantized moments updated in place on the GPU
                }
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
                //
                // Shape-rebuild guard: the outer Length-mismatch branch only
                // triggers when element counts differ. A parameter can keep
                // the same Length but switch its _shape (e.g., reshape from
                // [B, F] to [F, B], or a lazy-init layer migrating from
                // placeholder rank to its resolved rank with the same
                // element count). MFullPrecision is allocated against a
                // fixed _shape and the math ops below assume shape
                // compatibility with `param`/`grad`. Reallocate when the
                // cached tensor's shape no longer matches the parameter's
                // — preserve numeric content by copying into the
                // freshly-shaped tensor since Adam's m_t is the running
                // first moment of gradients and zeroing it on shape change
                // would produce a transient gradient-flow stall.
                if (state.MFullPrecision is null)
                {
                    state.MFullPrecision = new Tensor<T>(param._shape);
                }
                else if (!state.MFullPrecision._shape.SequenceEqual(param._shape))
                {
                    var rebuilt = new Tensor<T>(param._shape);
                    // Element counts match (we're in the Length-equal branch),
                    // so a flat copy preserves the moment values across the
                    // shape change. The rank/axes can differ — only the
                    // element count must match for a valid in-place reshape.
                    Engine.TensorCopy(state.MFullPrecision, rebuilt);
                    state.MFullPrecision = rebuilt;
                }
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
        if (_options.UseBFloat16MomentStorage)
        {
            // BF16 moments: 2 bytes/element, zero-initialized (BF16 0x0000 == +0.0), no scales/blocks.
            return new QuantizedTapeState
            {
                Length = paramLength,
                NumBlocks = 0,
                MBf16 = new Vector<ushort>(paramLength),
                VBf16 = new Vector<ushort>(paramLength),
            };
        }

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
    /// <summary>
    /// Serializes a <see cref="Vector{T}"/> of <see cref="byte"/> to a
    /// <see cref="BinaryWriter"/> in fixed-size chunks rather than allocating
    /// a full-size scratch <c>byte[]</c>. For a 300 M-parameter checkpoint
    /// the previous implementation doubled resident quantized state during
    /// the copy (300 MB live + 300 MB scratch = 600 MB peak); the chunked
    /// path caps the scratch overhead at <see cref="ChunkBytes"/> regardless
    /// of vector length.
    /// </summary>
    private const int ChunkBytes = 64 * 1024;
    private static void WriteVectorBytesChunked(BinaryWriter writer, Vector<byte> v)
    {
        int total = v.Length;
        if (total == 0) return;
        var chunk = new byte[Math.Min(total, ChunkBytes)];
        int offset = 0;
        while (offset < total)
        {
            int n = Math.Min(chunk.Length, total - offset);
            for (int i = 0; i < n; i++) chunk[i] = v[offset + i];
            writer.Write(chunk, 0, n);
            offset += n;
        }
    }

    private void QuantizeTensor(Tensor<T> values, Vector<byte> quantized, Vector<double> scales, int numBlocks, bool isSigned)
    {
        int blockSize = _options.BlockSize;
        int totalLength = values.Length;

        // Blocks are independent — parallelize over them (grain-gated for small
        // tensors). The percentile path needs a per-block sort scratch; rather
        // than allocate one List per block (the dominant allocator hotspot at
        // foundation scale), each worker lazily rents ONE ArrayPool buffer via
        // localInit and reuses it across the blocks it processes, returning it in
        // localFinally. Stochastic rounding uses the thread-safe per-thread RNG.
        CpuParallelSettings.ParallelForOrSerial<double[]?>(
            0, numBlocks, (long)totalLength,
            () => null,
            (b, _, rentedBuffer) =>
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
                        quantizedVal = (int)(floor + (RandomHelper.ThreadSafeRandom.NextDouble() < frac ? 1 : 0));
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

                return rentedBuffer;
            },
            rentedBuffer =>
            {
                if (rentedBuffer is not null)
                    System.Buffers.ArrayPool<double>.Shared.Return(rentedBuffer);
            });
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
        // Blocks are independent — parallelize (grain-gated for small tensors).
        CpuParallelSettings.ParallelForOrSerial(0, numBlocks, (long)totalLength, b =>
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, totalLength);
            double scale = scales[b];

            for (int i = blockStart; i < blockEnd; i++)
            {
                double quantizedVal = isSigned ? (int)quantized[i] - 128 : (int)quantized[i];
                result[i] = NumOps.FromDouble(quantizedVal * scale);
            }
        });
        return result;
    }

    /// <summary>
    /// Expands a BF16 (2 bytes/element) moment buffer into a freshly-allocated full-precision tensor of
    /// the given shape. Transient — consumed within a single Step iteration and released to the arena.
    /// </summary>
    private Tensor<T> Bf16ToTensor(Vector<ushort> bf16, int[] paramShape)
    {
        var result = new Tensor<T>(paramShape);
        int length = result.Length;
        CpuParallelSettings.ParallelForOrSerial(0, length, (long)length, i =>
        {
            result[i] = NumOps.FromDouble(BitConverterHelper.Bf16BitsToFloat(bf16[i]));
        });
        return result;
    }

    /// <summary>
    /// Packs a full-precision tensor's values back into a pre-allocated BF16 (2 bytes/element) buffer
    /// with round-to-nearest-even. BF16 keeps the float32 exponent, so no scale factor is needed.
    /// </summary>
    private void TensorToBf16(Tensor<T> values, Vector<ushort> bf16)
    {
        int length = values.Length;
        CpuParallelSettings.ParallelForOrSerial(0, length, (long)length, i =>
        {
            bf16[i] = BitConverterHelper.FloatToBf16Bits((float)NumOps.ToDouble(values[i]));
        });
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
        // #1413 CONSOLIDATION: NN solutions go through base.UpdateSolution
        // which synthesizes a TapeStepContext and delegates to Step
        // (one source of truth, matches PyTorch/TF/JAX). Non-NN solutions
        // (regression, clustering, classical models) keep the legacy
        // flat-vector path below for backward compatibility.
        if (currentSolution is AiDotNet.Interfaces.INeuralNetwork<T>)
        {
            return base.UpdateSolution(currentSolution, gradient);
        }
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
        // Legacy flat-state path (Step(IFullModel) / UpdateSolution).
        _mQuantized = null;
        _vQuantized = null;
        _mScales = null;
        _vScales = null;
        _mFullPrecision = null;
        _t = 0;
        _parameterLength = 0;
        _numBlocks = 0;
        // Tape-state path (Step(TapeStepContext)). Without these clears,
        // a fresh Reset() leaves stale per-parameter moments + bias-
        // correction step counter in place — the next training run
        // would resume from old state instead of cold-starting.
        _tapeStates.Clear();
        _tapeStep = 0;
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

        // Tape-mode state memory: Step(TapeStepContext<T>) writes its
        // per-parameter Adam moments into _tapeStates rather than the
        // legacy flat _m*/_v* fields. After a tape-only run those legacy
        // fields are still null and the dictionary holds the actual byte/
        // scale buffers — which is the resident optimizer memory the
        // 8× saving claim is measured against. Walk the dictionary and
        // attribute each QuantizedTapeState's contribution to the same
        // category (quantized / scales / full-precision) so the
        // saving math stays apples-to-apples regardless of which Step
        // path the optimizer drove.
        long tapeStateCount = 0;
        long tapeParameterLength = 0;
        foreach (var kvp in _tapeStates)
        {
            var tapeState = kvp.Value;
            tapeStateCount++;
            tapeParameterLength += tapeState.Length;
            if (tapeState.MQuantized != null) quantizedStateMemory += tapeState.MQuantized.Length;
            // VQuantized/VScales are null after a BF16 run (UseBFloat16MomentStorage) — the V moment
            // lives in VBf16 instead — so guard the deref to avoid a NullReferenceException, and
            // attribute the BF16 buffers (2 bytes/element, no per-block scales) so the savings math
            // stays correct regardless of which storage mode the run used.
            if (tapeState.VQuantized != null) quantizedStateMemory += tapeState.VQuantized.Length;
            if (tapeState.MScales != null) scalesMemory += tapeState.MScales.Length * 8;
            if (tapeState.VScales != null) scalesMemory += tapeState.VScales.Length * 8;
            if (tapeState.MBf16 != null) quantizedStateMemory += tapeState.MBf16.Length * 2;
            if (tapeState.VBf16 != null) quantizedStateMemory += tapeState.VBf16.Length * 2;
            if (tapeState.MFullPrecision != null)
            {
                fullPrecisionMemory += tapeState.MFullPrecision.Length * bytesPerElement;
            }
        }

        stats["QuantizedStateBytes"] = quantizedStateMemory;
        stats["ScalingFactorBytes"] = scalesMemory;
        stats["FullPrecisionStateBytes"] = fullPrecisionMemory;
        stats["TotalBytes"] = quantizedStateMemory + scalesMemory + fullPrecisionMemory;
        stats["TapeStateCount"] = tapeStateCount;

        // Calculate savings compared to standard Adam. Standard Adam's m
        // and v are both at full precision, so its baseline is
        // 2 × paramLength × bytesPerElement. For a legacy-Step run that's
        // _parameterLength; for a tape-Step run it's the sum of every
        // tape state's Length (each tape entry corresponds to a distinct
        // model parameter the tape touched). For a mixed run, both add
        // — the optimizer is bookkeeping for both populations.
        long totalParamLength = _parameterLength + tapeParameterLength;
        long standardAdamMemory = totalParamLength * 2 * bytesPerElement;
        stats["StandardAdamBytes"] = standardAdamMemory;
        stats["MemorySavingsBytes"] = standardAdamMemory - stats["TotalBytes"];

        return stats;
    }

    /// <summary>
    /// Test-only snapshot of one tape-state entry. Exposes the structural
    /// fields downstream tests need (lengths, presence of m-quantized vs
    /// m-fullprecision, scale block counts) without forcing tests to
    /// reach into private state via reflection. The fields are public
    /// readonly because the type itself is internal — only assemblies
    /// listed in <c>InternalsVisibleTo</c> on AiDotNet.csproj see it.
    /// </summary>
    internal sealed class TapeStateInfo
    {
        public int Length { get; init; }
        public int NumBlocks { get; init; }
        public bool HasMQuantized { get; init; }
        public int MQuantizedLength { get; init; }
        public bool HasMScales { get; init; }
        public int MScalesLength { get; init; }
        public bool HasMFullPrecision { get; init; }
        public int MFullPrecisionLength { get; init; }
        public int VQuantizedLength { get; init; }
        public int VScalesLength { get; init; }
    }

    /// <summary>
    /// Test hook: returns a structural snapshot of every tape-state entry.
    /// Tests use this to assert per-parameter quantization layout
    /// (block count, presence of m vs m-fullprecision, etc.) without
    /// reflecting into private fields. Snapshot is a copy, not a live
    /// view — mutating the returned dictionary or its values does not
    /// affect optimizer state.
    /// </summary>
    internal IReadOnlyDictionary<Tensor<T>, TapeStateInfo> GetTapeStateSnapshotForTests()
    {
        // Use the same reference-identity comparer the live state uses, so a
        // hypothetical Tensor<T>.Equals override that compares by value
        // doesn't merge distinct parameter tensors in the snapshot.
        var snapshot = new Dictionary<Tensor<T>, TapeStateInfo>(
            _tapeStates.Count,
            TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var kvp in _tapeStates)
        {
            var s = kvp.Value;
            snapshot[kvp.Key] = new TapeStateInfo
            {
                Length = s.Length,
                NumBlocks = s.NumBlocks,
                HasMQuantized = s.MQuantized is not null,
                MQuantizedLength = s.MQuantized?.Length ?? 0,
                HasMScales = s.MScales is not null,
                MScalesLength = s.MScales?.Length ?? 0,
                HasMFullPrecision = s.MFullPrecision is not null,
                MFullPrecisionLength = s.MFullPrecision?.Length ?? 0,
                // Null after a BF16 run (the V moment is in VBf16) — guard so the test snapshot
                // doesn't throw a NullReferenceException.
                VQuantizedLength = s.VQuantized?.Length ?? 0,
                VScalesLength = s.VScales?.Length ?? 0,
            };
        }
        return snapshot;
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

            // Magic header + format version. Pre-#1240 (v1) checkpoints
            // wrote `_t` (the Adam step counter) immediately after the
            // options JSON. Writing a bare version int here would be
            // ambiguous: an older checkpoint with `_t == 2` (after just
            // two training steps) would be mis-detected as v2 format and
            // every field that follows would parse with the wrong layout
            // (corrupted lengths, multi-GB phantom allocations on
            // ReadBytes, confusing failures deep in the call stack).
            //
            // Write a distinctive 4-byte ASCII magic before the version
            // so we disambiguate v2 from v1 by signature, not by guess.
            // Magic = "A8B1" (Adam-8-Bit v1-of-versioned-format) which
            // BinaryWriter writes as bytes 0x41 0x38 0x42 0x31 in stream
            // order — visible in hex dumps. Probability that a v1 _t
            // ever equals this 32-bit value is 1/2^32 (~2e-10), and
            // because v1's _t is monotonic from 0 it'd take 0.83 billion
            // steps to first hit the value — well past any realistic
            // training run length. Independent of probability, the
            // semantic check is unambiguous: v1 wrote a step counter,
            // not this magic, so a match here is a deliberate v2 marker.
            // Constants are class-level (Adam8BitV2Magic / StateFormatVersion)
            // so Serialize/Deserialize can't drift out of sync.
            writer.Write(Adam8BitV2Magic);
            writer.Write(StateFormatVersion);

            // Serialize 8-bit Adam-specific state
            writer.Write(_t);
            writer.Write(_parameterLength);
            writer.Write(_numBlocks);

            // Serialize quantized first moment (if used). Always emit a
            // hasMState flag BEFORE the conditional payload so Deserialize
            // doesn't blindly read length+data when the optimizer was never
            // initialized (legacy UpdateSolution path) or when only the
            // tape Step has run (no _mQuantized / _mFullPrecision yet).
            // The previous serialization wrote the data conditionally but
            // Deserialize unconditionally read it, producing
            // EndOfStreamException on uninitialized state.
            // The compressBothMoments flag is written for cross-checking on
            // load — _options.CompressBothMoments is the authoritative source
            // of truth (it round-trips through the options JSON above), but
            // emitting it here lets Deserialize fail fast on a tampered or
            // mode-mismatched payload before allocating the wrong moment
            // representation.
            writer.Write(_options.CompressBothMoments);
            bool hasMState = _options.CompressBothMoments
                ? _mQuantized is not null
                : _mFullPrecision is not null;
            writer.Write(hasMState);
            if (hasMState)
            {
                if (_options.CompressBothMoments)
                {
                    writer.Write(_mQuantized!.Length);
                    WriteVectorBytesChunked(writer, _mQuantized);
                    foreach (var scale in _mScales!)
                    {
                        writer.Write(scale);
                    }
                }
                else
                {
                    writer.Write(_mFullPrecision!.Length);
                    foreach (var value in _mFullPrecision)
                    {
                        writer.Write(Convert.ToDouble(value));
                    }
                }
            }

            // Serialize quantized second moment
            writer.Write(_vQuantized is not null);
            if (_vQuantized is not null)
            {
                writer.Write(_vQuantized.Length);
                WriteVectorBytesChunked(writer, _vQuantized);
                foreach (var scale in _vScales!)
                {
                    writer.Write(scale);
                }
            }

            // Tape-state checkpoint partial: persist the global step counter
            // (_tapeStep) so bias-correction terms resume correctly after
            // load. Per-parameter tape moments (_tapeStates) are NOT
            // persisted because the dictionary is keyed by Tensor<T>
            // reference — those references don't survive a process restart,
            // and there's no stable parameter-id mapping to re-key them on
            // load. Warn loudly so users know mid-training Adam-state
            // checkpoint/resume is partial: the bias-correction step
            // counter resumes (so update magnitudes match), but per-
            // parameter moments cold-start (first few steps after resume
            // see a small spike before momentum re-accumulates). Full
            // tape-state checkpoint is a larger architectural change
            // tracked separately.
            if (_tapeStates.Count > 0)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"Adam8BitOptimizer.Serialize: {_tapeStates.Count} per-parameter " +
                    $"tape Adam moments are NOT persisted (dictionary keyed by Tensor " +
                    $"reference, not stable across serialize/deserialize). The global " +
                    $"step counter _tapeStep={_tapeStep} IS persisted for bias-correction " +
                    $"continuity. Mid-training resume will cold-start moments but match " +
                    $"the bias-correction trajectory.");
            }
            writer.Write(_tapeStep);

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
            // Deserialize base class data. The first ReadBytes is the only
            // pre-magic-check allocation in the wire format, so guard it
            // against malformed/tampered checkpoints that could otherwise
            // request an arbitrarily large allocation. Cap against the
            // remaining stream length: a baseDataLength larger than what's
            // actually present in the buffer is unambiguously invalid.
            int baseDataLength = reader.ReadInt32();
            if (baseDataLength < 0)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: invalid baseDataLength={baseDataLength} in checkpoint header.");
            }
            long remainingBytes = ms.Length - ms.Position;
            if (baseDataLength > remainingBytes)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: declared baseDataLength={baseDataLength} exceeds remaining " +
                    $"stream bytes ({remainingBytes}). Checkpoint is truncated or malformed.");
            }
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize options
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<Adam8BitOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Read magic header + format version (added in #1240 follow-
            // up). Pre-#1240 (v1) payloads wrote _t immediately after the
            // options JSON; using a bare version int as the discriminator
            // would mis-detect any v1 checkpoint whose _t happens to equal
            // the version number. Read the magic first — the magic value
            // is a fixed marker (class-level constant Adam8BitV2Magic)
            // that v1 never wrote at this position, so a match here is
            // unambiguous evidence of v2 format.
            int magic = reader.ReadInt32();
            if (magic != Adam8BitV2Magic)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: incompatible checkpoint format. Expected v2 " +
                    $"magic header 0x{Adam8BitV2Magic:X8} ('A8B1' in ASCII LE) immediately " +
                    $"after the options JSON; got 0x{magic:X8}. Older checkpoints " +
                    $"(format v1) wrote the Adam step counter (_t) at that position and " +
                    $"the byte layout that follows is incompatible with v2. Re-serialize " +
                    $"this checkpoint with a build that writes the v2 byte-quantized state " +
                    $"format. If you authored a custom serializer, write " +
                    $"BinaryWriter.Write(0x{Adam8BitV2Magic:X8}) followed by " +
                    $"BinaryWriter.Write((int){StateFormatVersion}) immediately after the " +
                    $"options JSON.");
            }
            int stateFormatVersion = reader.ReadInt32();
            if (stateFormatVersion != StateFormatVersion)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: unrecognized format version {stateFormatVersion} " +
                    $"after valid v2 magic. Expected version {StateFormatVersion}; this " +
                    $"build does not yet support reading newer formats. Upgrade to a build " +
                    $"that recognizes version {stateFormatVersion} or re-serialize from " +
                    $"this build.");
            }

            // Deserialize state
            _t = reader.ReadInt32();
            _parameterLength = reader.ReadInt32();
            _numBlocks = reader.ReadInt32();

            // Bounds-check ALL structural fields before allocating anything
            // sized off them. Untrusted/tampered checkpoints could otherwise:
            //   (a) force multi-GB phantom allocations on the ReadBytes calls
            //       below by claiming impossibly large lengths,
            //   (b) trigger DivideByZeroException via BlockSize <= 0,
            //   (c) overflow (_parameterLength + blockSize - 1) when both
            //       are near int.MaxValue and the addition wraps to negative,
            //   (d) skip the consistency check via negative _numBlocks that
            //       happen to satisfy `_numBlocks != expectedNumBlocks`
            //       being false (it isn't, but defensive belt-and-suspenders).
            // All checks happen before any allocation downstream.
            if (_parameterLength < 0)
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: invalid _parameterLength={_parameterLength} in checkpoint.");
            int blockSize = _options.BlockSize;
            if (blockSize <= 0)
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: invalid BlockSize={blockSize} in checkpoint options. " +
                    $"BlockSize must be positive (typical values: 64, 128, 256, 2048).");
            if (_numBlocks < 0)
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: invalid _numBlocks={_numBlocks} in checkpoint.");
            // Compute expected blocks in long arithmetic to avoid int
            // overflow on hostile _parameterLength near int.MaxValue.
            long expectedNumBlocksLong = _parameterLength == 0 ? 0L
                : ((long)_parameterLength + blockSize - 1L) / blockSize;
            if (expectedNumBlocksLong > int.MaxValue)
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: _parameterLength={_parameterLength} and BlockSize=" +
                    $"{blockSize} produce {expectedNumBlocksLong} blocks, exceeding int.MaxValue. " +
                    $"Checkpoint is malformed or out of supported range.");
            if (_numBlocks != (int)expectedNumBlocksLong)
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: _numBlocks={_numBlocks} inconsistent with " +
                    $"_parameterLength={_parameterLength} and BlockSize={blockSize} " +
                    $"(expected {expectedNumBlocksLong}). Checkpoint may be corrupted.");

            // The m-quantized and v-quantized read branches below each
            // cap their declared length against the remaining stream
            // bytes (ms.Length - ms.Position) so a payload claiming
            // mLength=int.MaxValue can't force a 2 GB ReadBytes
            // allocation before the truncation check fires.

            // Deserialize first moment. The hasMState flag (added in #1240
            // follow-up) tells us whether m was actually initialized at
            // serialize time. If false, leave the m fields null so the
            // first Step / UpdateSolution call after deser allocates them
            // freshly — matches the contract of an optimizer that was
            // serialized before any training had run.
            //
            // The streamed compressBothMoments flag is cross-checked
            // against _options.CompressBothMoments (the authoritative
            // value, just deserialized from the options JSON). A
            // mismatch indicates a tampered payload, manual format
            // surgery, or a bug — fail fast rather than allocate the
            // wrong moment representation and silently produce wrong
            // updates downstream.
            bool streamedCompressBothMoments = reader.ReadBoolean();
            if (streamedCompressBothMoments != _options.CompressBothMoments)
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: checkpoint compressBothMoments flag " +
                    $"({streamedCompressBothMoments}) does not match the value in the " +
                    $"deserialized options ({_options.CompressBothMoments}). The options " +
                    $"JSON is the source of truth — a mismatch here means the payload's " +
                    $"m-state layout is inconsistent with the options that were " +
                    $"serialized alongside it. Re-serialize from a consistent build.");
            bool hasMState = reader.ReadBoolean();
            if (hasMState)
            {
                if (_options.CompressBothMoments)
                {
                    int mLength = reader.ReadInt32();
                    if (mLength != _parameterLength)
                        throw new InvalidOperationException(
                            $"Adam8BitOptimizer: m-quantized length {mLength} does not " +
                            $"match _parameterLength={_parameterLength}.");
                    // Pre-check: payload can't exceed the remaining stream
                    // bytes — protects against a malformed payload whose
                    // declared length passes the _parameterLength check but
                    // the actual data was truncated upstream. Without this,
                    // ReadBytes would allocate a full-sized array and only
                    // then notice the truncation.
                    long mAfter = ms.Position + mLength;
                    if (mLength < 0 || mAfter > ms.Length)
                        throw new InvalidOperationException(
                            $"Adam8BitOptimizer: m-quantized declared length {mLength} exceeds " +
                            $"remaining stream bytes ({ms.Length - ms.Position}). Checkpoint truncated.");
                    // Bulk read — per-element copy was O(N) writer touches
                    // and unnecessarily slow for large checkpoints. ReadBytes
                    // returns a single contiguous byte[] which we copy into
                    // the Vector<byte>. The bounds check above caps the
                    // allocation at remaining stream bytes so a tampered
                    // length can't force a multi-GB phantom allocation.
                    var mBytes = reader.ReadBytes(mLength);
                    if (mBytes.Length != mLength)
                        throw new InvalidOperationException(
                            $"Adam8BitOptimizer: m-quantized truncated (expected {mLength} " +
                            $"bytes, got {mBytes.Length}). Checkpoint is corrupted.");
                    _mQuantized = new Vector<byte>(mLength);
                    for (int i = 0; i < mLength; i++) _mQuantized[i] = mBytes[i];
                    _mScales = new Vector<double>(_numBlocks);
                    for (int i = 0; i < _numBlocks; i++)
                    {
                        _mScales[i] = reader.ReadDouble();
                    }
                    // Clear stale full-precision m on mode switch — see
                    // OzYc: deserializing a CompressBothMoments=true payload
                    // into an instance that previously held _mFullPrecision
                    // would otherwise leave that buffer resident, inflating
                    // GetMemoryUsage and breaking the 8x savings claim.
                    _mFullPrecision = null;
                }
                else
                {
                    int mLength = reader.ReadInt32();
                    if (mLength != _parameterLength)
                        throw new InvalidOperationException(
                            $"Adam8BitOptimizer: m-fullprecision length {mLength} does not " +
                            $"match _parameterLength={_parameterLength}.");
                    _mFullPrecision = new Vector<T>(mLength);
                    for (int i = 0; i < mLength; i++)
                    {
                        _mFullPrecision[i] = NumOps.FromDouble(reader.ReadDouble());
                    }
                    // Clear stale quantized m on mode switch (symmetric
                    // with the compressBothMoments branch above).
                    _mQuantized = null;
                    _mScales = null;
                }
            }
            else
            {
                _mQuantized = null;
                _mFullPrecision = null;
                _mScales = null;
            }

            // Deserialize second moment
            bool hasVQuantized = reader.ReadBoolean();
            if (hasVQuantized)
            {
                int vLength = reader.ReadInt32();
                if (vLength != _parameterLength)
                    throw new InvalidOperationException(
                        $"Adam8BitOptimizer: v-quantized length {vLength} does not " +
                        $"match _parameterLength={_parameterLength}.");
                // Pre-check declared length against remaining stream — see
                // the m-quantized branch for rationale.
                long vAfter = ms.Position + vLength;
                if (vLength < 0 || vAfter > ms.Length)
                    throw new InvalidOperationException(
                        $"Adam8BitOptimizer: v-quantized declared length {vLength} exceeds " +
                        $"remaining stream bytes ({ms.Length - ms.Position}). Checkpoint truncated.");
                var vBytes = reader.ReadBytes(vLength);
                if (vBytes.Length != vLength)
                    throw new InvalidOperationException(
                        $"Adam8BitOptimizer: v-quantized truncated (expected {vLength} " +
                        $"bytes, got {vBytes.Length}). Checkpoint is corrupted.");
                _vQuantized = new Vector<byte>(vLength);
                for (int i = 0; i < vLength; i++) _vQuantized[i] = vBytes[i];
                _vScales = new Vector<double>(_numBlocks);
                for (int i = 0; i < _numBlocks; i++)
                {
                    _vScales[i] = reader.ReadDouble();
                }
            }
            else
            {
                // Clear stale v state when deserializing into a reused
                // optimizer instance. Without this, an instance that
                // previously held _vQuantized / _vScales from an earlier
                // load would carry that state forward when a fresh,
                // never-stepped checkpoint is loaded — silently producing
                // wrong updates. Symmetric with the m-state else branch
                // above.
                _vQuantized = null;
                _vScales = null;
            }

            // Tape-state checkpoint partial (matches Serialize): read the
            // global step counter so bias-correction continues from the
            // saved trajectory. Per-parameter tape moments cold-start —
            // _tapeStates is left empty and the next tape Step lazily
            // re-allocates entries on first touch.
            //
            // Defensive try/catch: a valid v2 payload always contains
            // _tapeStep (the magic-header check above already rejects v1
            // payloads, so format-migration is not the concern here).
            // The catch handles a different failure mode — a v2 payload
            // truncated mid-write (disk full, process killed during
            // serialize, partial network transfer). Falling back to
            // _tapeStep=0 lets the optimizer resume with cold-started
            // bias correction rather than throwing on a recoverable
            // truncation. Cold-start matches the contract of an
            // optimizer that was checkpointed before any training had
            // run, so the resumed run sees one step's worth of slightly-
            // overconfident updates before bias correction stabilizes.
            try
            {
                _tapeStep = reader.ReadInt32();
            }
            catch (EndOfStreamException)
            {
                _tapeStep = 0;
            }
            _tapeStates.Clear();

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
