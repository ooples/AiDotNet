using AiDotNet.Helpers;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using SimdVector = System.Numerics.Vector;
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
public partial class Adam8BitOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <inheritdoc/>
    /// <remarks>
    /// #1745: in BFloat16 moment-storage mode the optimizer state is plain Adam
    /// with bf16 m/v — which the fused CPU kernel now supports directly via
    /// <c>ICompiledTrainingPlan.RequestBf16MomentStorage</c>. So this optimizer
    /// keeps the fused fast path AND the halved moment footprint, instead of
    /// falling back to the eager tape. The true 8-bit block-quantized mode
    /// (UseBFloat16MomentStorage == false) has no fused kernel yet and stays on
    /// the eager tape (returns false). Adaptive LR / AMSGrad — which the bf16
    /// Adam/AdamW kernels don't model — also fall back.
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        // Only BF16 moment storage maps to a fused kernel today; the int8
        // block-quant path changes the update enough to need its own kernel.
        if (!_options.UseBFloat16MomentStorage) return false;
        // Adaptive LR mutates the rate between steps and AMSGrad needs the
        // max-second-moment variant — neither is modeled by the bf16 Adam/AdamW
        // kernels, so fall back to eager for those.
        if (_options.UseAdaptiveLearningRate || _options.UseAMSGrad) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.Adam,
            (float)GetCurrentLearningRate(),
            (float)_options.Beta1, (float)_options.Beta2, (float)_options.Epsilon,
            0f, schedule)
        { UseBf16Moments = true };
        return true;
    }

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
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<byte>? _mQuantized;

    /// <summary>
    /// Quantized second moment vector (moving average of squared gradients).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<byte>? _vQuantized;

    /// <summary>
    /// Scaling factors for first moment quantization blocks.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<double>? _mScales;

    /// <summary>
    /// Scaling factors for second moment quantization blocks.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<double>? _vScales;

    /// <summary>
    /// Full-precision first moment vector (used when CompressBothMoments is false).
    /// </summary>
    [AiDotNet.Attributes.Buffer]
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

            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);

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
        public ushort[]? MBf16;
        public ushort[]? VBf16;

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
    private readonly Dictionary<int, QuantizedTapeState> _pendingTapeStatesByParameterIndex = new();
    // Guards _pendingTapeStatesByParameterIndex (a plain, non-concurrent Dictionary): Serialize's
    // WriteTapeStates enumerates it while Deserialize/restore/Reset mutate it, so without this lock a
    // checkpoint taken concurrently with training could throw "collection was modified" or snapshot a
    // torn state.
    private readonly object _pendingTapeStatesLock = new();
    private int _tapeStep;

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        PrepareTapeState(context);

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

        int parameterIndex = -1;
        foreach (var param in context.Parameters)
        {
            parameterIndex++;
            // Mirror PrepareTapeState: tolerate null parameter slots — skip restoring/updating their
            // tape state but keep advancing parameterIndex so the stable parameter ordering is preserved.
            if (param is null) continue;
            RestorePendingTapeState(parameterIndex, param);

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
                // Moments expanded to transient full-precision tensors; update them IN PLACE to avoid
                // ~7 extra full-size scratch allocations per parameter (#1688 Fix 2). Math identical.
                // AllocateTapeState always allocates both BF16 buffers when UseBFloat16MomentStorage is
                // set, so they are non-null on this path; capture into locals (no null-forgiving operator).
                ushort[] mBf16 = state.MBf16 ?? throw new InvalidOperationException("BF16 moment buffer M was not allocated.");
                ushort[] vBf16 = state.VBf16 ?? throw new InvalidOperationException("BF16 moment buffer V was not allocated.");

                Tensor<T> mB = Bf16ToTensor(mBf16, param._shape);
                Tensor<T> vB = Bf16ToTensor(vBf16, param._shape);

                var gradScaledB = Engine.TensorMultiplyScalar(grad, oneMinusBeta1); // (1-beta1)·g
                Engine.TensorMultiplyScalarInPlace(mB, beta1);                      // mB *= beta1
                Engine.TensorAddInPlace(mB, gradScaledB);                           // mB := m_t
                var gradSqB = Engine.TensorMultiply(grad, grad);                    // g²
                Engine.TensorMultiplyScalarInPlace(gradSqB, oneMinusBeta2);         // (1-beta2)·g²
                Engine.TensorMultiplyScalarInPlace(vB, beta2);                      // vB *= beta2
                Engine.TensorAddInPlace(vB, gradSqB);                               // vB := v_t

                // mB/vB were updated in place above, so they hold m_t/v_t — pack them straight back into
                // the null-safe BF16 buffers (no separate newMB/newVB tensors, no null-forgiving operator).
                TensorToBf16(mB, mBf16);
                TensorToBf16(vB, vBf16);

                var mHatB = Engine.TensorDivideScalar(mB, biasCorrection1);
                var vHatB = Engine.TensorDivideScalar(vB, biasCorrection2);
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

            // Update biased first / second moments IN PLACE to minimize per-parameter transient
            // allocations (#1688 Fix 2). Identical recurrences to the legacy path, but expressed with
            // in-place ops so a step over a large parameter no longer allocates ~7 full-size scratch
            // tensors for the moment update (plus it removes the prior TensorCopy):
            //     m_t = beta1·m_{t-1} + (1-beta1)·g
            //     v_t = beta2·v_{t-1} + (1-beta2)·g²
            // `m` is either the persistent state.MFullPrecision (CompressBothMoments == false) or a
            // transient dequant — either way updating it in place IS the moment update, so when
            // CompressBothMoments is false no separate copy into state.MFullPrecision is needed. `v` is
            // always a transient dequant. `grad` is tape-owned/shared, so it is never mutated.
            var gradScaled = Engine.TensorMultiplyScalar(grad, oneMinusBeta1); // (1-beta1)·g
            Engine.TensorMultiplyScalarInPlace(m, beta1);                      // m *= beta1
            Engine.TensorAddInPlace(m, gradScaled);                            // m := m_t
            var gradSq = Engine.TensorMultiply(grad, grad);                    // g²
            Engine.TensorMultiplyScalarInPlace(gradSq, oneMinusBeta2);         // (1-beta2)·g²
            Engine.TensorMultiplyScalarInPlace(v, beta2);                      // v *= beta2
            Engine.TensorAddInPlace(v, gradSq);                                // v := v_t

            // Re-quantize the updated moments back into the byte[] state. When CompressBothMoments is
            // false, m IS state.MFullPrecision and was updated in place above — no copy needed.
            if (_options.CompressBothMoments)
            {
                QuantizeTensor(m, state.MQuantized!, state.MScales!, state.NumBlocks, isSigned: true);
            }
            QuantizeTensor(v, state.VQuantized, state.VScales, state.NumBlocks, isSigned: false);

            // Apply the bias-corrected Adam update directly to the parameter.
            //     update = lr · (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)
            var mHat = Engine.TensorDivideScalar(m, biasCorrection1);
            var vHat = Engine.TensorDivideScalar(v, biasCorrection2);
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
                MBf16 = new ushort[paramLength],
                VBf16 = new ushort[paramLength],
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
    private Tensor<T> Bf16ToTensor(ushort[] bf16, int[] paramShape)
    {
        var result = new Tensor<T>(paramShape);
        int length = result.Length;
        // Raw-array dequant (no Tensor/Vector indexer, no NumOps dispatch on the float fast path).
        // Profiled as ~half the NN-training managed hot path (ViT) before this; the typeof(T) branch
        // folds at JIT. BF16→float is a pure widening, no scale.
        var dst = result.GetCpuData();
        if (typeof(T) == typeof(float))
        {
            var f = (float[])(object)dst;
            // Chunked range loop (not per-element delegate) so the inner loop is tight and can run a
            // SIMD bulk widen on NET7+. Bit-identical to BitConverterHelper.Bf16BitsToFloat (the scalar
            // tail uses it directly, and the SIMD body reproduces the same high-16-bits placement).
            CpuParallelSettings.ParallelForChunks(length, Bf16BulkChunkGrain, (chunkStart, chunkCount) =>
            {
                int i = chunkStart;
                int chunkEnd = chunkStart + chunkCount;
#if NET7_0_OR_GREATER
                i = ConvertBf16ToFloatSimd(bf16, f, chunkStart, chunkCount);
#endif
                for (; i < chunkEnd; i++)
                    f[i] = BitConverterHelper.Bf16BitsToFloat(bf16[i]);
            });
        }
        else
        {
            CpuParallelSettings.ParallelForOrSerial(0, length, (long)length, i => dst[i] = NumOps.FromDouble(BitConverterHelper.Bf16BitsToFloat(bf16[i])));
        }
        return result;
    }

    /// <summary>
    /// Packs a full-precision tensor's values back into a pre-allocated BF16 (2 bytes/element) buffer
    /// with round-to-nearest-even. BF16 keeps the float32 exponent, so no scale factor is needed.
    /// </summary>
    private void TensorToBf16(Tensor<T> values, ushort[] bf16)
    {
        int length = values.Length;
        // Raw-array quant (no Tensor/Vector indexer, no NumOps dispatch on the float fast path).
        var src = values.GetCpuData();
        if (typeof(T) == typeof(float))
        {
            var f = (float[])(object)src;
            // Chunked range loop + SIMD bulk round-to-nearest-even on NET7+. Bit-identical to
            // BitConverterHelper.FloatToBf16Bits (the scalar tail calls it; the SIMD body reproduces
            // the same RNE add + NaN-preserving path lane-for-lane).
            CpuParallelSettings.ParallelForChunks(length, Bf16BulkChunkGrain, (chunkStart, chunkCount) =>
            {
                int i = chunkStart;
                int chunkEnd = chunkStart + chunkCount;
#if NET7_0_OR_GREATER
                i = ConvertFloatToBf16Simd(f, bf16, chunkStart, chunkCount);
#endif
                for (; i < chunkEnd; i++)
                    bf16[i] = BitConverterHelper.FloatToBf16Bits(f[i]);
            });
        }
        else
        {
            CpuParallelSettings.ParallelForOrSerial(0, length, (long)length, i => bf16[i] = BitConverterHelper.FloatToBf16Bits((float)NumOps.ToDouble(src[i])));
        }
    }

    /// <summary>
    /// Minimum elements per parallel chunk for the BF16 bulk conversions. Below this the work runs
    /// serially (single chunk); above it the work is split across cores, and within each chunk the
    /// inner loop runs a SIMD bulk convert (NET7+) followed by a scalar tail.
    /// </summary>
    private const int Bf16BulkChunkGrain = 8192;

#if NET7_0_OR_GREATER
    /// <summary>
    /// SIMD bulk float→BF16 with round-to-nearest-even, bit-identical to
    /// <see cref="BitConverterHelper.FloatToBf16Bits(float)"/>. Processes whole <see cref="Vector{T}"/>
    /// blocks in <paramref name="src"/>[<paramref name="start"/>, start+count); returns the first index
    /// not covered (the caller finishes the &lt; one-vector tail with the scalar helper).
    /// </summary>
    private static int ConvertFloatToBf16Simd(float[] src, ushort[] dst, int start, int count)
    {
        int laneU = System.Numerics.Vector<uint>.Count;        // == Vector<float>.Count
        int stride = System.Numerics.Vector<ushort>.Count;     // == 2 * laneU ushorts produced per iteration
        var c7fff = new System.Numerics.Vector<uint>(0x7FFFu);
        var cInf = new System.Numerics.Vector<uint>(0x7F800000u);
        var cAbs = new System.Numerics.Vector<uint>(0x7FFFFFFFu);
        var cNan = new System.Numerics.Vector<uint>(0x0040u);
        var cOne = new System.Numerics.Vector<uint>(1u);
        int i = start;
        int end = start + count;
        for (; i + stride <= end; i += stride)
        {
            System.Numerics.Vector<uint> lo = FloatLaneToBf16(new System.Numerics.Vector<float>(src, i), c7fff, cInf, cAbs, cNan, cOne);
            System.Numerics.Vector<uint> hi = FloatLaneToBf16(new System.Numerics.Vector<float>(src, i + laneU), c7fff, cInf, cAbs, cNan, cOne);
            SimdVector.Narrow(lo, hi).CopyTo(dst, i);
        }
        return i;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static System.Numerics.Vector<uint> FloatLaneToBf16(System.Numerics.Vector<float> value,
        System.Numerics.Vector<uint> c7fff, System.Numerics.Vector<uint> cInf,
        System.Numerics.Vector<uint> cAbs, System.Numerics.Vector<uint> cNan, System.Numerics.Vector<uint> cOne)
    {
        System.Numerics.Vector<uint> bits = SimdVector.AsVectorUInt32(value);
        System.Numerics.Vector<uint> high = SimdVector.ShiftRightLogical(bits, 16);
        System.Numerics.Vector<uint> isNan = SimdVector.GreaterThan(bits & cAbs, cInf);   // (bits & 0x7FFFFFFF) > 0x7F800000
        System.Numerics.Vector<uint> nanResult = high | cNan;                              // (bits>>16) | 0x40
        System.Numerics.Vector<uint> lsb = high & cOne;                                    // (bits>>16) & 1
        System.Numerics.Vector<uint> rounded = SimdVector.ShiftRightLogical(bits + c7fff + lsb, 16);
        // Narrow takes the low 16 bits, matching the scalar (ushort) truncation.
        return SimdVector.ConditionalSelect(isNan, nanResult, rounded);
    }

    /// <summary>
    /// SIMD bulk BF16→float (pure widening, bit-identical to
    /// <see cref="BitConverterHelper.Bf16BitsToFloat(ushort)"/>). Returns the first index not covered.
    /// </summary>
    private static int ConvertBf16ToFloatSimd(ushort[] src, float[] dst, int start, int count)
    {
        int laneU = System.Numerics.Vector<uint>.Count;
        int stride = System.Numerics.Vector<ushort>.Count;
        int i = start;
        int end = start + count;
        for (; i + stride <= end; i += stride)
        {
            SimdVector.Widen(new System.Numerics.Vector<ushort>(src, i), out System.Numerics.Vector<uint> lo, out System.Numerics.Vector<uint> hi);
            SimdVector.AsVectorSingle(SimdVector.ShiftLeft(lo, 16)).CopyTo(dst, i);
            SimdVector.AsVectorSingle(SimdVector.ShiftLeft(hi, 16)).CopyTo(dst, i + laneU);
        }
        return i;
    }

    /// <summary>
    /// Test-only hook: runs the exact production float→BF16→float path (SIMD block + scalar tail) over
    /// whole arrays so a unit test can assert bit-identity against the scalar
    /// <see cref="BitConverterHelper"/> reference. Not part of the optimizer's runtime behavior.
    /// </summary>
    internal static void Bf16RoundTripForTest(float[] src, ushort[] bf16, float[] back)
    {
        int i = ConvertFloatToBf16Simd(src, bf16, 0, src.Length);
        for (; i < src.Length; i++)
            bf16[i] = BitConverterHelper.FloatToBf16Bits(src[i]);
        int j = ConvertBf16ToFloatSimd(bf16, back, 0, bf16.Length);
        for (; j < bf16.Length; j++)
            back[j] = BitConverterHelper.Bf16BitsToFloat(bf16[j]);
    }
#endif

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
        if (parameters.Length != gradient.Length)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) must match gradient vector length ({gradient.Length}).",
                nameof(gradient));
        }

        if (_mQuantized == null && _mFullPrecision == null || _parameterLength != parameters.Length)
        {
            InitializeQuantizedState(parameters.Length);
            _t = 0;
        }

        _t++;

        T beta1 = _currentBeta1;
        T beta2 = _currentBeta2;
        T oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        T oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        double beta1Double = Convert.ToDouble(beta1);
        double beta2Double = Convert.ToDouble(beta2);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(beta1Double, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(beta2Double, _t));

        var updatedParameters = new Vector<T>(parameters.Length, skipZeroInit: true);
        var parameterSpan = parameters.AsSpan();
        var gradientSpan = gradient.AsSpan();
        var updatedSpan = updatedParameters.AsWritableSpan();
        Span<T> fullPrecisionMSpan = _options.CompressBothMoments
            ? Span<T>.Empty
            : _mFullPrecision!.AsWritableSpan();
        int blockSize = _options.BlockSize;

        T[] mBlock = ArrayPool<T>.Shared.Rent(blockSize);
        T[] vBlock = ArrayPool<T>.Shared.Rent(blockSize);
        double[] absoluteValues = ArrayPool<double>.Shared.Rent(blockSize);
        try
        {
            for (int block = 0; block < _numBlocks; block++)
            {
                int blockStart = block * blockSize;
                int blockLength = Math.Min(blockSize, parameters.Length - blockStart);
                double oldMScale = _options.CompressBothMoments ? _mScales![block] : 0.0;
                double oldVScale = _vScales![block];

                for (int j = 0; j < blockLength; j++)
                {
                    int i = blockStart + j;
                    T g = gradientSpan[i];
                    T oldM = _options.CompressBothMoments
                        ? NumOps.FromDouble(((int)_mQuantized![i] - 128) * oldMScale)
                        : fullPrecisionMSpan[i];
                    T oldV = NumOps.FromDouble(_vQuantized![i] * oldVScale);
                    T newM = NumOps.Add(
                        NumOps.Multiply(beta1, oldM),
                        NumOps.Multiply(oneMinusBeta1, g));
                    T newV = NumOps.Add(
                        NumOps.Multiply(beta2, oldV),
                        NumOps.Multiply(oneMinusBeta2, NumOps.Multiply(g, g)));

                    mBlock[j] = newM;
                    vBlock[j] = newV;
                    if (!_options.CompressBothMoments)
                    {
                        fullPrecisionMSpan[i] = newM;
                    }

                    T mHat = NumOps.Divide(newM, biasCorrection1);
                    T vHat = NumOps.Divide(newV, biasCorrection2);
                    T denominator = NumOps.Add(NumOps.Sqrt(vHat), epsilon);
                    T update = NumOps.Multiply(
                        CurrentLearningRate,
                        NumOps.Divide(mHat, denominator));
                    updatedSpan[i] = NumOps.Subtract(parameterSpan[i], update);
                }

                if (_options.CompressBothMoments)
                {
                    QuantizeBlock(mBlock, blockLength, blockStart, block, _mQuantized!, _mScales!, true, absoluteValues);
                }
                QuantizeBlock(vBlock, blockLength, blockStart, block, _vQuantized!, _vScales!, false, absoluteValues);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(mBlock, clearArray: true);
            ArrayPool<T>.Shared.Return(vBlock, clearArray: true);
            ArrayPool<double>.Shared.Return(absoluteValues);
        }

        return updatedParameters;
    }

    private void QuantizeBlock(
        T[] values,
        int count,
        int targetOffset,
        int block,
        Vector<byte> quantized,
        Vector<double> scales,
        bool isSigned,
        double[] absoluteValues)
    {
        double maxAbs = 0.0;
        if (_options.QuantizationPercentile >= 100)
        {
            for (int i = 0; i < count; i++)
            {
                double absoluteValue = Math.Abs(NumOps.ToDouble(values[i]));
                if (absoluteValue > maxAbs) maxAbs = absoluteValue;
            }
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                absoluteValues[i] = Math.Abs(NumOps.ToDouble(values[i]));
            }
            Array.Sort(absoluteValues, 0, count);
            int percentileIndex = (int)((count - 1) * _options.QuantizationPercentile / 100.0);
            maxAbs = absoluteValues[percentileIndex];
        }

        double scale = maxAbs / (isSigned ? 127.0 : 255.0);
        if (scale < 1e-10) scale = 1e-10;
        scales[block] = scale;

        for (int i = 0; i < count; i++)
        {
            double scaled = NumOps.ToDouble(values[i]) / scale;
            int quantizedValue;
            if (_options.UseStochasticRounding)
            {
                double floor = Math.Floor(scaled);
                double fraction = scaled - floor;
                quantizedValue = (int)(floor + (RandomHelper.ThreadSafeRandom.NextDouble() < fraction ? 1 : 0));
            }
            else
            {
                quantizedValue = (int)Math.Round(scaled);
            }

            if (isSigned)
            {
                quantizedValue = MathHelper.Clamp(quantizedValue, -127, 127);
                quantized[targetOffset + i] = (byte)(quantizedValue + 128);
            }
            else
            {
                quantizedValue = MathHelper.Clamp(quantizedValue, 0, 255);
                quantized[targetOffset + i] = (byte)quantizedValue;
            }
        }
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
        // Clear BOTH maps under _pendingTapeStatesLock so this reset is atomic
        // with RestorePendingTapeState (which does its pending lookup and the
        // matching _tapeStates insert under the same lock). Otherwise a restore
        // running concurrently could read a pending entry, we clear both maps,
        // and the restore then writes a stale checkpoint moment back into the
        // freshly-reset optimizer.
        lock (_pendingTapeStatesLock)
        {
            _tapeStates.Clear();
            _pendingTapeStatesByParameterIndex.Clear();
        }
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

    private void WriteTapeStates(BinaryWriter writer)
    {
        var entries = new SortedDictionary<int, QuantizedTapeState>();

        // Snapshot the pending map under the lock so a concurrent Deserialize/restore/Reset can't mutate
        // it mid-enumeration (see _pendingTapeStatesLock).
        lock (_pendingTapeStatesLock)
        {
            foreach (var entry in _pendingTapeStatesByParameterIndex)
            {
                entries[entry.Key] = entry.Value;
            }
        }

        foreach (var entry in _tapeStates)
        {
            if (TryGetTapeParameterIndex(entry.Key, out int parameterIndex))
            {
                entries[parameterIndex] = entry.Value;
            }
        }

        writer.Write(entries.Count);
        foreach (var entry in entries)
        {
            writer.Write(entry.Key);
            WriteTapeState(writer, entry.Value);
        }
    }

    private void ReadTapeStates(BinaryReader reader)
    {
        // Parse into a LOCAL map first so the stream I/O runs outside _pendingTapeStatesLock, then swap the
        // contents in atomically under the lock (WriteTapeStates snapshots the shared map under the same
        // lock, so a concurrent Serialize sees either the old or the new full state, never a torn one).
        var pending = new Dictionary<int, QuantizedTapeState>();

        int count = reader.ReadInt32();
        // Each entry is at minimum a 4-byte parameter index, so a valid count can't exceed the bytes
        // remaining. This rejects a negative count (which would silently restore zero entries) and an
        // absurd positive count before the loop.
        ValidateDeclaredCount(reader, count, sizeof(int), "tape-state table");
        for (int i = 0; i < count; i++)
        {
            int parameterIndex = reader.ReadInt32();
            if (parameterIndex < 0)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: invalid tape-state parameter index {parameterIndex}.");
            }
            // ContainsKey (not Dictionary.TryAdd, which is unavailable on net471) before reading the
            // state, so a duplicate index throws instead of silently overwriting an earlier entry.
            if (pending.ContainsKey(parameterIndex))
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: duplicate tape-state parameter index {parameterIndex} in checkpoint.");
            }
            pending[parameterIndex] = ReadTapeState(reader);
        }

        lock (_pendingTapeStatesLock)
        {
            _pendingTapeStatesByParameterIndex.Clear();
            foreach (var entry in pending)
                _pendingTapeStatesByParameterIndex[entry.Key] = entry.Value;
        }
    }

    private void WriteTapeState(BinaryWriter writer, QuantizedTapeState state)
    {
        // The GPU 8-bit step (AIDOTNET_GPU_ADAM=1) updates GpuMQ/GpuVQ (and the device scale buffers) in
        // place and never writes back to the host MQuantized/VQuantized/scale fields this method serializes.
        // Persisting a GPU-resident state here would therefore checkpoint STALE host moments — silent
        // corruption that resumes with wrong optimizer state. The device->host readback belongs in the
        // Tensors GpuOptimizer layer (which owns the device buffers) and has to be validated on real GPU
        // hardware, so it is a tracked enhancement rather than something this diffusion-training PR ships
        // unvalidated. Until then we fail fast with actionable guidance instead of writing wrong data:
        // to checkpoint an 8-bit Adam run, train with AIDOTNET_GPU_ADAM unset (or CompressBothMoments off)
        // so the moments stay host-resident and serialize correctly. Higher-level checkpoint code may catch
        // this to degrade to a model-only save with a clear status.
        if (state.GpuResident)
        {
            throw new InvalidOperationException(
                "Adam8BitOptimizer: cannot serialize a GPU-resident 8-bit tape state — the device moment " +
                "buffers have no host-readback path yet, so persisting would checkpoint stale host moments. " +
                "To checkpoint, run without AIDOTNET_GPU_ADAM (host-resident moments serialize normally), or " +
                "handle this exception at the checkpoint layer to save model-only.");
        }

        writer.Write(state.Length);
        writer.Write(state.NumBlocks);
        WriteByteVector(writer, state.MQuantized);
        WriteTensor(writer, state.MFullPrecision);
        WriteByteVector(writer, state.VQuantized);
        WriteDoubleVector(writer, state.MScales);
        WriteDoubleVector(writer, state.VScales);
        WriteUShortArray(writer, state.MBf16);
        WriteUShortArray(writer, state.VBf16);
    }

    private QuantizedTapeState ReadTapeState(BinaryReader reader)
    {
        var state = new QuantizedTapeState
        {
            Length = reader.ReadInt32(),
            NumBlocks = reader.ReadInt32(),
            MQuantized = ReadByteVector(reader),
            MFullPrecision = ReadTensor(reader),
            VQuantized = ReadByteVector(reader) ?? null!,
            MScales = ReadDoubleVector(reader),
            VScales = ReadDoubleVector(reader) ?? null!,
            MBf16 = ReadUShortArray(reader),
            VBf16 = ReadUShortArray(reader),
            GpuResident = false
        };

        if (state.Length < 0)
        {
            throw new InvalidOperationException($"Adam8BitOptimizer: invalid tape-state length {state.Length}.");
        }

        // Validate that each payload buffer is not just PRESENT but the RIGHT length for this state's
        // element count — a malformed checkpoint with a short/long moment buffer would otherwise crash
        // later or silently apply partial moment data. Quantized moments are one byte per element;
        // per-block scales are one double per block; BF16 moments are one ushort per element.
        int expectedBlocks = state.Length == 0 ? 0 : (state.Length + _options.BlockSize - 1) / _options.BlockSize;

        if (_options.UseBFloat16MomentStorage)
        {
            if (state.MBf16 is null || state.VBf16 is null)
            {
                throw new InvalidOperationException("Adam8BitOptimizer: BF16 tape-state payload is incomplete.");
            }
            if (state.MBf16.Length != state.Length || state.VBf16.Length != state.Length)
            {
                throw new InvalidOperationException("Adam8BitOptimizer: BF16 tape-state payload has inconsistent lengths.");
            }
        }
        else
        {
            if (state.VQuantized is null || state.VScales is null)
            {
                throw new InvalidOperationException("Adam8BitOptimizer: V tape-state payload is incomplete.");
            }
            if (state.VQuantized.Length != state.Length || state.VScales.Length != expectedBlocks)
            {
                throw new InvalidOperationException("Adam8BitOptimizer: V tape-state payload has inconsistent lengths.");
            }

            if (_options.CompressBothMoments)
            {
                if (state.MQuantized is null || state.MScales is null)
                {
                    throw new InvalidOperationException("Adam8BitOptimizer: quantized M tape-state payload is incomplete.");
                }
                if (state.MQuantized.Length != state.Length || state.MScales.Length != expectedBlocks)
                {
                    throw new InvalidOperationException("Adam8BitOptimizer: quantized M tape-state payload has inconsistent lengths.");
                }
            }
            else if (state.MFullPrecision is null)
            {
                throw new InvalidOperationException("Adam8BitOptimizer: full-precision M tape-state payload is incomplete.");
            }
            else if (state.MFullPrecision.Length != state.Length)
            {
                throw new InvalidOperationException("Adam8BitOptimizer: full-precision M tape-state payload has inconsistent length.");
            }
        }

        if (!_options.UseBFloat16MomentStorage)
        {
            if (state.NumBlocks != expectedBlocks)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: tape-state block count {state.NumBlocks} does not match " +
                    $"length {state.Length} and BlockSize {_options.BlockSize} (expected {expectedBlocks}).");
            }
        }

        return state;
    }

    private void RestorePendingTapeState(int parameterIndex, Tensor<T> parameter)
    {
        // Do the pending lookup, the _tapeStates insert, and the pending remove
        // all under _pendingTapeStatesLock (and Reset() clears both maps under the
        // same lock). Releasing the lock between the lookup and the insert let a
        // concurrent Reset() clear both maps in the gap, after which this method
        // wrote a stale checkpoint moment back into the freshly-reset optimizer.
        lock (_pendingTapeStatesLock)
        {
            if (!_pendingTapeStatesByParameterIndex.TryGetValue(parameterIndex, out var state))
            {
                return;
            }

            if (state.Length != parameter.Length)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer checkpoint tape state for parameter {parameterIndex} has length " +
                    $"{state.Length}, but the current parameter has length {parameter.Length}.");
            }

            if (state.MFullPrecision is not null && !state.MFullPrecision._shape.SequenceEqual(parameter._shape))
            {
                var reshaped = new Tensor<T>(parameter._shape);
                state.MFullPrecision.AsSpan().CopyTo(reshaped.AsWritableSpan());
                state.MFullPrecision = reshaped;
            }

            _tapeStates[parameter] = state;
            _pendingTapeStatesByParameterIndex.Remove(parameterIndex);
        }
    }

    private static void WriteByteVector(BinaryWriter writer, Vector<byte>? vector)
    {
        writer.Write(vector is not null);
        if (vector is null) return;

        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            writer.Write(vector[i]);
        }
    }

    // Reject a stream-declared element count that is negative OR larger than the bytes physically
    // remaining in the (seekable) checkpoint stream, BEFORE allocating. A malformed/truncated payload
    // could otherwise declare billions of elements and force a multi-GB allocation (OOM) before the
    // read loop ever fails. elementSize is the on-wire bytes per element.
    private static void ValidateDeclaredCount(BinaryReader reader, int length, int elementSize, string what)
    {
        if (length < 0)
        {
            throw new InvalidOperationException($"Adam8BitOptimizer: invalid {what} length {length}.");
        }

        var stream = reader.BaseStream;
        if (stream.CanSeek)
        {
            long remaining = stream.Length - stream.Position;
            if ((long)length * elementSize > remaining)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: {what} declares {length} elements ({(long)length * elementSize} bytes) " +
                    $"but only {remaining} bytes remain in the checkpoint stream.");
            }
        }
    }

    private static Vector<byte>? ReadByteVector(BinaryReader reader)
    {
        if (!reader.ReadBoolean()) return null;

        int length = reader.ReadInt32();
        ValidateDeclaredCount(reader, length, sizeof(byte), "byte-vector");

        var vector = new Vector<byte>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = reader.ReadByte();
        }

        return vector;
    }

    private static void WriteDoubleVector(BinaryWriter writer, Vector<double>? vector)
    {
        writer.Write(vector is not null);
        if (vector is null) return;

        writer.Write(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            writer.Write(vector[i]);
        }
    }

    private static Vector<double>? ReadDoubleVector(BinaryReader reader)
    {
        if (!reader.ReadBoolean()) return null;

        int length = reader.ReadInt32();
        ValidateDeclaredCount(reader, length, sizeof(double), "double-vector");

        var vector = new Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = reader.ReadDouble();
        }

        return vector;
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T>? tensor)
    {
        writer.Write(tensor is not null);
        if (tensor is null) return;

        writer.Write(tensor._shape.Length);
        foreach (int dimension in tensor._shape)
        {
            writer.Write(dimension);
        }

        var span = tensor.AsSpan();
        writer.Write(span.Length);
        for (int i = 0; i < span.Length; i++)
        {
            writer.Write(NumOps.ToDouble(span[i]));
        }
    }

    private Tensor<T>? ReadTensor(BinaryReader reader)
    {
        if (!reader.ReadBoolean()) return null;

        int rank = reader.ReadInt32();
        // Each rank dimension is a 4-byte int on the wire; reject an absurd rank before allocating shape.
        ValidateDeclaredCount(reader, rank, sizeof(int), "tensor rank");

        var shape = new int[rank];
        long declaredElements = rank == 0 ? 0 : 1;
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
            if (shape[i] < 0)
            {
                throw new InvalidOperationException($"Adam8BitOptimizer: invalid tensor dimension {shape[i]} at axis {i}.");
            }
            // checked: an unchecked long product can silently overflow/wrap for a malicious shape,
            // producing a small count that bypasses the byte-remaining guard and forces a huge alloc.
            try
            {
                declaredElements = checked(declaredElements * shape[i]);
            }
            catch (OverflowException)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: tensor shape overflows its element count at axis {i}. Checkpoint is corrupted or malicious.");
            }
        }

        // A single in-memory tensor cannot exceed int.MaxValue elements (Tensor.Length is Int32); reject
        // an out-of-range count so a wrapped/absurd shape can't slip past the byte-bound check.
        if (declaredElements > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"Adam8BitOptimizer: tensor shape declares {declaredElements} elements, exceeding the maximum supported tensor size.");
        }

        // Bound the element count (each element is an 8-byte double on the wire) against the bytes that
        // physically remain BEFORE allocating the tensor, so a malformed shape can't force an OOM.
        var stream = reader.BaseStream;
        if (stream.CanSeek)
        {
            long remaining = stream.Length - stream.Position;
            if (declaredElements * sizeof(double) > remaining)
            {
                throw new InvalidOperationException(
                    $"Adam8BitOptimizer: tensor shape declares {declaredElements} elements " +
                    $"({declaredElements * sizeof(double)} bytes) but only {remaining} bytes remain in the checkpoint stream.");
            }
        }

        int length = reader.ReadInt32();
        var tensor = new Tensor<T>(shape);
        if (tensor.Length != length)
        {
            throw new InvalidOperationException(
                $"Adam8BitOptimizer: serialized tensor length {length} does not match shape length {tensor.Length}.");
        }

        var span = tensor.AsWritableSpan();
        for (int i = 0; i < length; i++)
        {
            span[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        return tensor;
    }

    private static void WriteUShortArray(BinaryWriter writer, ushort[]? values)
    {
        writer.Write(values is not null);
        if (values is null) return;

        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            writer.Write(values[i]);
        }
    }

    private static ushort[]? ReadUShortArray(BinaryReader reader)
    {
        if (!reader.ReadBoolean()) return null;

        int length = reader.ReadInt32();
        ValidateDeclaredCount(reader, length, sizeof(ushort), "ushort-array");

        var values = new ushort[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = reader.ReadUInt16();
        }

        return values;
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
