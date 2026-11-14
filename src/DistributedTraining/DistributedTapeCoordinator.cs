using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MixedPrecision;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
#if !AIDN_LEGACY_TFM
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;
#endif

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Coordinates multi-rank tape-based training across <see cref="DistributedStrategy.DDP"/>,
/// <see cref="DistributedStrategy.ZeRO1"/>, <see cref="DistributedStrategy.ZeRO2"/>, and
/// <see cref="DistributedStrategy.FSDP"/> strategies, including initial parameter broadcast, token-weighted
/// step metadata reduction, overlapped bucketed gradient reduction, mixed-precision unscaling with synchronized
/// overflow detection, global-norm gradient clipping, sharded optimizer updates with <c>AllGather</c>, and
/// cross-world-size optimizer checkpoint consolidation.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="SynchronizeUnscaleClipAndStep"/> enforces the canonical distributed mixed-precision training sequence:
/// (1) flush and await in-flight gradient buckets (<c>AllReduce</c> for DDP/ZeRO-1, <c>ReduceScatter</c> for ZeRO-2/FSDP),
/// (2) unscale reduced gradients by <c>1 / LossScaler.Scale</c> and agree across all ranks on <c>NaN</c>/<c>Inf</c> overflow
/// before restoring FP32 master weights,
/// (3) apply parameter regularization and exact global L2 norm clipping prior to the optimizer step,
/// (4) execute either the full-parameter step (DDP) or the owned contiguous shard step followed by <c>AllGather</c>
/// (ZeRO-1/ZeRO-2/FSDP), and
/// (5) synchronize working weights from updated FP32 master weights and advance the dynamic loss scaler.
/// </para>
/// <para><b>For Beginners:</b> Think of <see cref="DistributedTapeCoordinator{T}"/> as the conductor that keeps multiple
/// GPUs or processes in lockstep on every training batch—making sure every rank starts from identical weights, weights
/// its gradients fairly when the final batch has an uneven number of tokens, clips gradients using the true global norm
/// across all ranks, and saves checkpoints that can be resumed on any number of GPUs.</para>
/// </remarks>
/// <example>
/// <code>
/// using var coordinator = new DistributedTapeCoordinator&lt;float&gt;(backend, DistributedStrategy.ZeRO2);
/// coordinator.EnsureInitialParametersSynchronized(trainableParameters);
/// var (rankWeight, globalLoss, globalTokens) = coordinator.SynchronizeStepMetadata(localTokens, localLoss);
/// coordinator.BeginBackwardStep(trainableParameters, rankWeight);
/// // During or after backward:
/// coordinator.SynchronizeUnscaleClipAndStep(
///     trainableParameters, gradients, optimizer,
///     mixedPrecisionContext: null, restoreFp32Masters: null, syncFp32MastersFromWorking: null,
///     regularizeGradients: null, maxGradNorm: 1.0, executeFullParameterStep: ctx =&gt; optimizer.Step(ctx));
/// </code>
/// </example>
/// <typeparam name="T">The numeric element type.</typeparam>
internal sealed class DistributedTapeCoordinator<T> : IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Vector<T> _metadataBuffer = new Vector<T>(2);
    private readonly Vector<T> _singleElementBuffer = new Vector<T>(1);
    private readonly Dictionary<Tensor<T>, Tensor<T>> _persistentGradSlots =
        new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly object _slotSync = new object();
    private float[]? _slotDownloadScratch;

    private GradientBucketReducer<T>? _reducer;
    private Tensor<T>?[]? _cachedInputParameters;
    private Tensor<T>[]? _cachedUniqueParameters;
    private int[]? _cachedUniqueParameterLengths;
    private int _cachedTotalElements;
    private Tensor<T>[]? _syncedParamSnapshot;
    private Tensor<T>? _ownedParamShard;
    private Tensor<T>[]? _ownedParamShardList;
    private Tensor<T>? _shardGradTensor;
    private Dictionary<Tensor<T>, Tensor<T>>? _shardGradDict;
    private Dictionary<Tensor<T>, Tensor<T>>? _shardRegDict;
    private Tensor<T>[]? _lastShardedParamsRef;
    private int[]? _lastShardedParamVersions;
    private int[]? _lastShardedParamGpuVersions;
    private int _lastShardedStart = -1;
    private int _lastShardedLength = -1;
    private Vector<T>? _gatherSendBuffer;
    private Dictionary<Tensor<float>, (IGpuBuffer Buffer, Tensor<float> BoundTensor)>? _persistentGradBuffers;
    private (IGpuBuffer Buffer, Tensor<float> BoundTensor)? _shardGradGpuBinding;
    private IDirectGpuBackend? _gpuBufferBackend;

    private long _stepStartTimestamp;
    private double _stepCommMilliseconds;
    private double _stepExposedCommMilliseconds;
    private bool _stepMetadataRecorded;
    private bool _backwardStepBegun;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="DistributedTapeCoordinator{T}"/> class.
    /// </summary>
    /// <param name="backend">The distributed communication backend.</param>
    /// <param name="strategy">The distributed execution and sharding strategy.</param>
    /// <param name="configuration">Optional sharding configuration controlling communication overlap and bucket size.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="backend"/> is null.</exception>
    /// <exception cref="NotSupportedException">Thrown when <paramref name="strategy"/> requires pipeline or tensor model-parallel execution not handled by data/ZeRO/FSDP tape coordination.</exception>
    public DistributedTapeCoordinator(
        ICommunicationBackend<T> backend,
        DistributedStrategy strategy,
        IShardingConfiguration<T>? configuration = null)
    {
        Backend = backend ?? throw new ArgumentNullException(nameof(backend));
        ValidateStrategy(strategy);
        Strategy = strategy;

        OverlapCommunication = configuration?.OverlapCommunication ?? true;
        GradientBucketSizeBytes = configuration is not null && configuration.GradientBucketSizeBytes > 0
            ? configuration.GradientBucketSizeBytes
            : 25 * 1024 * 1024;

        if (!Backend.IsInitialized)
        {
            Backend.Initialize();
        }
    }

    /// <summary>
    /// Gets the underlying communication backend.
    /// </summary>
    public ICommunicationBackend<T> Backend { get; }

    /// <summary>
    /// Gets the active distributed training strategy.
    /// </summary>
    public DistributedStrategy Strategy { get; }

    /// <summary>
    /// Gets this process's zero-based rank within the communication group.
    /// </summary>
    public int Rank => Backend.Rank;

    /// <summary>
    /// Gets the total number of participating ranks in the communication group.
    /// </summary>
    public int WorldSize => Backend.WorldSize;

    /// <summary>
    /// Gets whether asynchronous bucketed communication overlaps with backward propagation.
    /// </summary>
    public bool OverlapCommunication { get; }

    /// <summary>
    /// Gets the target capacity in bytes of each gradient communication bucket.
    /// </summary>
    public int GradientBucketSizeBytes { get; }

    /// <summary>
    /// Gets the exact token-weighted global loss computed during the most recent <see cref="SynchronizeStepMetadata"/> call.
    /// </summary>
    public double LastGlobalLoss { get; private set; }

    /// <summary>
    /// Gets the global valid token/element count summed across all ranks during the most recent <see cref="SynchronizeStepMetadata"/> call.
    /// </summary>
    public long LastGlobalTokenCount { get; private set; }

    /// <summary>
    /// Gets the raw wall-clock duration in milliseconds of the most recently completed distributed step.
    /// </summary>
    public double LastStepTotalMilliseconds { get; private set; }

    /// <summary>
    /// Gets the raw cumulative milliseconds spent in collective communication (including background overlapped collectives)
    /// during the most recently completed distributed step.
    /// </summary>
    public double LastStepCommunicationMilliseconds { get; private set; }

    /// <summary>
    /// Gets the raw non-overlapped wall-clock milliseconds spent blocked on collective communication on the main training
    /// thread during the most recently completed distributed step.
    /// </summary>
    public double LastStepExposedCommunicationMilliseconds { get; private set; }

    /// <summary>
    /// Gets the unclamped ratio <c>LastStepCommunicationMilliseconds / LastStepTotalMilliseconds</c> for the most recent step
    /// (<c>0.0</c> when <see cref="LastStepTotalMilliseconds"/> is zero).
    /// </summary>
    public double LastStepCommunicationShare { get; private set; }

    private bool IsShardedStrategy =>
        Strategy == DistributedStrategy.ZeRO1 ||
        Strategy == DistributedStrategy.ZeRO2 ||
        Strategy == DistributedStrategy.ZeRO3 ||
        Strategy == DistributedStrategy.FSDP;

    private bool UseReduceScatter =>
        Strategy == DistributedStrategy.ZeRO2 ||
        Strategy == DistributedStrategy.ZeRO3 ||
        Strategy == DistributedStrategy.FSDP;

    private static void ValidateStrategy(DistributedStrategy strategy)
    {
        switch (strategy)
        {
            case DistributedStrategy.None:
            case DistributedStrategy.DDP:
            case DistributedStrategy.ZeRO1:
            case DistributedStrategy.ZeRO2:
            case DistributedStrategy.ZeRO3:
            case DistributedStrategy.FSDP:
            case DistributedStrategy.Hybrid:
                return;
            case DistributedStrategy.PipelineParallel:
            case DistributedStrategy.TensorParallel:
                throw new NotSupportedException(
                    $"DistributedStrategy.{strategy} requires stage/tensor graph partitioning and is not supported by DistributedTapeCoordinator. " +
                    "Supported strategies are DDP, ZeRO1, ZeRO2, ZeRO3, FSDP, and Hybrid.");
            default:
                throw new NotSupportedException($"Unsupported DistributedStrategy value '{strategy}'.");
        }
    }

    /// <summary>
    /// Gets a value indicating whether initial parameters have already been synchronized at least once since creation or reset.
    /// </summary>
    public bool HasSynchronizedInitialParameters => _syncedParamSnapshot is not null;

    /// <summary>
    /// Clears the cached initial-parameter broadcast marker so the next call to
    /// <see cref="EnsureInitialParametersSynchronized"/> re-broadcasts Rank 0's parameters.
    /// </summary>
    public void ResetParameterSync()
    {
        ThrowIfDisposed();
        _syncedParamSnapshot = null;
        _lastShardedParamsRef = null;
    }

    /// <summary>
    /// Coordinates lazy-layer warmup shapes across ranks so idle ranks (receiving an empty batch on the first step)
    /// can materialize lazy parameter shapes identically to active ranks before <see cref="EnsureInitialParametersSynchronized"/>.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("Exchanges tensor rank shape metadata (1-8 dimensions) across ranks for lazy warmup")]
    internal bool CoordinateLazyWarmupShapes(
        bool localNeedsWarmup,
        int[]? localInputShape,
        int[]? localTargetShape,
        out int[]? donorInputShape,
        out int[]? donorTargetShape)
    {
        ThrowIfDisposed();
        donorInputShape = null;
        donorTargetShape = null;

        bool hasNonEmptyLocal = localInputShape is { Length: > 0 } && localTargetShape is { Length: > 0 };
        if (WorldSize <= 1)
        {
            if (localNeedsWarmup && hasNonEmptyLocal)
            {
                donorInputShape = localInputShape;
                donorTargetShape = localTargetShape;
                return true;
            }

            return false;
        }

        EnsureBackendInitialized();
        var flags = new Vector<T>(2);
        {
            Span<T> flagSpan = flags.AsWritableSpan();
            flagSpan[0] = NumOps.FromDouble(localNeedsWarmup ? 1.0 : 0.0);
            flagSpan[1] = NumOps.FromDouble(hasNonEmptyLocal ? (WorldSize - Rank) : 0.0);
        }

        Backend.AllReduce(flags, ReductionOperation.Max);
        ReadOnlySpan<T> flagRead = flags.AsSpan();
        bool anyNeedsWarmup = NumOps.ToDouble(flagRead[0]) > 0.5;
        int donorScore = (int)Math.Round(NumOps.ToDouble(flagRead[1]));
        if (!anyNeedsWarmup || donorScore <= 0)
        {
            return false;
        }

        int donorRank = WorldSize - donorScore;
        var rankDims = new Vector<T>(2);
        if (Rank == donorRank && localInputShape is not null && localTargetShape is not null)
        {
            Span<T> dimSpan = rankDims.AsWritableSpan();
            dimSpan[0] = NumOps.FromDouble(localInputShape.Length);
            dimSpan[1] = NumOps.FromDouble(localTargetShape.Length);
        }

        Backend.AllReduce(rankDims, ReductionOperation.Max);
        ReadOnlySpan<T> dimRead = rankDims.AsSpan();
        int inRank = (int)Math.Round(NumOps.ToDouble(dimRead[0]));
        int tgtRank = (int)Math.Round(NumOps.ToDouble(dimRead[1]));
        if (inRank <= 0 || tgtRank <= 0)
        {
            return false;
        }

        var shapeValues = new Vector<T>(inRank + tgtRank);
        if (Rank == donorRank && localInputShape is not null && localTargetShape is not null)
        {
            Span<T> valSpan = shapeValues.AsWritableSpan();
            for (int i = 0; i < inRank; i++)
            {
                valSpan[i] = NumOps.FromDouble(localInputShape[i]);
            }

            for (int i = 0; i < tgtRank; i++)
            {
                valSpan[inRank + i] = NumOps.FromDouble(localTargetShape[i]);
            }
        }

        Backend.AllReduce(shapeValues, ReductionOperation.Max);
        ReadOnlySpan<T> valRead = shapeValues.AsSpan();
        var resolvedIn = new int[inRank];
        var resolvedTgt = new int[tgtRank];
        for (int i = 0; i < inRank; i++)
        {
            resolvedIn[i] = (int)Math.Round(NumOps.ToDouble(valRead[i]));
        }

        for (int i = 0; i < tgtRank; i++)
        {
            resolvedTgt[i] = (int)Math.Round(NumOps.ToDouble(valRead[inRank + i]));
        }

        donorInputShape = resolvedIn;
        donorTargetShape = resolvedTgt;
        return true;
    }

    /// <summary>
    /// Broadcasts Rank 0's initial trainable parameters to all other ranks once per distinct parameter list.
    /// Subsequent calls with the identical parameter tensor list are idempotent no-ops until <see cref="ResetParameterSync"/> is called.
    /// </summary>
    /// <param name="trainableParameters">The ordered list of trainable parameter tensors.</param>
    /// <param name="invalidateCaches">Optional callback invoked after unpacking broadcast weights to invalidate cached views or buffers.</param>
    /// <exception cref="ObjectDisposedException">Thrown if this coordinator has been disposed.</exception>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="trainableParameters"/> is null.</exception>
    public void EnsureInitialParametersSynchronized(
        IReadOnlyList<Tensor<T>> trainableParameters,
        Action? invalidateCaches = null)
    {
        ThrowIfDisposed();
        if (trainableParameters is null)
        {
            throw new ArgumentNullException(nameof(trainableParameters));
        }

        var uniqueParams = GetOrDeduplicateParameters(trainableParameters, out int totalElements);
        if (MatchesSnapshot(_syncedParamSnapshot, uniqueParams))
        {
            return;
        }

        if (WorldSize <= 1 || totalElements <= 0)
        {
            _syncedParamSnapshot = uniqueParams;
            _lastShardedParamsRef = null;
            return;
        }

        EnsureBackendInitialized();

        var flat = new Vector<T>(totalElements);
        if (Rank == 0)
        {
            Span<T> dst = flat.AsWritableSpan();
            int offset = 0;
            for (int i = 0; i < uniqueParams.Length; i++)
            {
                var p = uniqueParams[i];
                p.AsSpan().CopyTo(dst.Slice(offset, p.Length));
                offset += p.Length;
            }
        }

        flat = Backend.Broadcast(flat, root: 0);
        UnpackVectorIntoParameters(flat.AsSpan().Slice(0, totalElements), uniqueParams, preserveGpuWeightBuffers: false);
        invalidateCaches?.Invoke();
        _syncedParamSnapshot = uniqueParams;
        _lastShardedParamsRef = null;
    }

    /// <summary>
    /// Returns the total number of valid target elements in <paramref name="targetBatch"/> (<c>0</c> when <paramref name="targetBatch"/> is null or empty).
    /// </summary>
    /// <param name="targetBatch">The local rank's target batch tensor, or <c>null</c> for an idle tail-batch rank.</param>
    /// <returns>The non-negative element count of <paramref name="targetBatch"/>.</returns>
    public static long CountValidElements(Tensor<T>? targetBatch)
    {
        return targetBatch is null ? 0L : targetBatch.Length;
    }

    /// <summary>
    /// Synchronizes local valid token counts and local unscaled mean losses across all ranks via a single 2-element
    /// <c>AllReduce(Sum)</c>, returning this rank's exact gradient weight <c>localTokenCount / GlobalTokens</c>,
    /// the global token-weighted mean loss, and the global valid token count.
    /// </summary>
    /// <param name="localTokenCount">Number of valid tokens/elements processed on this rank (<c>0</c> if idle on an uneven tail batch).</param>
    /// <param name="localUnscaledLoss">This rank's local unscaled mean loss over <paramref name="localTokenCount"/> elements.</param>
    /// <returns>A tuple containing <c>(RankWeight, GlobalLoss, GlobalTokens)</c>.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if this coordinator has been disposed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="localTokenCount"/> is negative.</exception>
    public (double RankWeight, double GlobalLoss, long GlobalTokens) SynchronizeStepMetadata(
        long localTokenCount,
        double localUnscaledLoss)
    {
        ThrowIfDisposed();
        if (localTokenCount < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(localTokenCount),
                localTokenCount,
                "Local token count cannot be negative.");
        }

        _stepStartTimestamp = Stopwatch.GetTimestamp();
        _stepCommMilliseconds = 0.0;
        _stepExposedCommMilliseconds = 0.0;
        _stepMetadataRecorded = true;

        double sanitizedLocalLoss = localTokenCount > 0 ? localUnscaledLoss : 0.0;

        if (WorldSize <= 1)
        {
            double singleRankWeight = localTokenCount > 0 ? 1.0 : 0.0;
            LastGlobalTokenCount = localTokenCount;
            LastGlobalLoss = sanitizedLocalLoss;
            return (singleRankWeight, sanitizedLocalLoss, localTokenCount);
        }

        EnsureBackendInitialized();

        {
            Span<T> writeSpan = _metadataBuffer.AsWritableSpan();
            writeSpan[0] = NumOps.FromDouble(localTokenCount);
            writeSpan[1] = NumOps.FromDouble(localTokenCount > 0 ? localUnscaledLoss * localTokenCount : 0.0);
        }

        long t0 = Stopwatch.GetTimestamp();
        Backend.AllReduce(_metadataBuffer, ReductionOperation.Sum);
        long t1 = Stopwatch.GetTimestamp();
        double elapsedMs = (t1 - t0) * 1000.0 / Stopwatch.Frequency;
        _stepCommMilliseconds += elapsedMs;
        _stepExposedCommMilliseconds += elapsedMs;

        ReadOnlySpan<T> readSpan = _metadataBuffer.AsSpan();
        double globalTokensDouble = NumOps.ToDouble(readSpan[0]);
        long globalTokens = globalTokensDouble > 0.0 ? (long)Math.Round(globalTokensDouble) : 0L;
        double weightedLossSum = NumOps.ToDouble(readSpan[1]);
        double globalLoss = globalTokens > 0 ? weightedLossSum / globalTokens : 0.0;
        double rankWeight = globalTokens > 0 ? (double)localTokenCount / globalTokens : 0.0;

        LastGlobalTokenCount = globalTokens;
        LastGlobalLoss = globalLoss;
        return (rankWeight, globalLoss, globalTokens);
    }

    /// <summary>
    /// Prepares the bucketed gradient reducer for the upcoming backward pass with the specified token-weighted <paramref name="rankWeight"/>.
    /// </summary>
    /// <param name="trainableParameters">The ordered list of trainable parameter tensors.</param>
    /// <param name="rankWeight">This rank's gradient scaling weight from <see cref="SynchronizeStepMetadata"/>.</param>
    /// <exception cref="ObjectDisposedException">Thrown if this coordinator has been disposed.</exception>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="trainableParameters"/> is null.</exception>
    public void BeginBackwardStep(IReadOnlyList<Tensor<T>> trainableParameters, double rankWeight)
    {
        ThrowIfDisposed();
        if (trainableParameters is null)
        {
            throw new ArgumentNullException(nameof(trainableParameters));
        }

        if (!_stepMetadataRecorded)
        {
            _stepStartTimestamp = Stopwatch.GetTimestamp();
            _stepCommMilliseconds = 0.0;
            _stepExposedCommMilliseconds = 0.0;
        }

        var reducer = EnsureReducer(trainableParameters);
        reducer.BeginStep(rankWeight);
        _backwardStepBegun = true;
    }

    /// <summary>
    /// Notifies the active <see cref="GradientBucketReducer{T}"/> that <paramref name="parameter"/>'s local backward
    /// gradient is ready so completed buckets can be reduced asynchronously while earlier layers finish differentiating.
    /// </summary>
    /// <param name="parameter">The trainable parameter tensor.</param>
    /// <param name="gradient">The parameter's local backward gradient.</param>
    public void OnParameterGradientReady(Tensor<T> parameter, Tensor<T> gradient)
    {
        ThrowIfDisposed();
        _reducer?.NotifyGradientReady(parameter, gradient);
    }

    /// <summary>
    /// Returns a persistent reusable host-backed destination gradient tensor for <paramref name="parameter"/>,
    /// eliminating per-step LOH allocations and GPU copy snapshots during overlapped distributed backward passes.
    /// </summary>
    internal Tensor<T> GetOrCreateGradientSlot(Tensor<T> parameter)
    {
        if (parameter is null)
        {
            throw new ArgumentNullException(nameof(parameter));
        }

        lock (_slotSync)
        {
            if (!_persistentGradSlots.TryGetValue(parameter, out var slot) ||
                slot is null ||
                slot.Length != parameter.Length)
            {
                slot = new Tensor<T>(parameter._shape);
                _persistentGradSlots[parameter] = slot;
            }

            return slot;
        }
    }

    /// <summary>
    /// Copies <paramref name="sourceGradient"/> into the persistent reusable host-backed gradient slot
    /// for <paramref name="parameter"/> without per-step LOH allocations or stale GPU host-sync downloads.
    /// </summary>
    internal Tensor<T> CopyIntoGradientSlot(Tensor<T> parameter, Tensor<T> sourceGradient)
    {
        if (sourceGradient is null)
        {
            throw new ArgumentNullException(nameof(sourceGradient));
        }

        var slot = GetOrCreateGradientSlot(parameter);
        int length = Math.Min(slot.Length, sourceGradient.Length);
        if (length <= 0)
        {
            return slot;
        }

        var gpuEngine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        if (slot._storageOffset == 0 && slot._storage.Length == slot.Length)
        {
            if (slot.GetBackingArrayForCacheLookupUnsafe() is { } slotBacking)
            {
                AiDotNet.Tensors.Helpers.HostSync.Remove(slotBacking);
            }

            AiDotNet.Tensors.Helpers.HostSync.Remove(slot.DataVector);
        }

        gpuEngine?.InvalidateResidentWeightBuffer(slot);

        if (typeof(T) == typeof(float) &&
            (object)sourceGradient is Tensor<float> srcFloat &&
            (object)slot is Tensor<float> slotFloat &&
            srcFloat.HasPendingGpuData &&
            srcFloat._gpuBuffer is { } srcGpuBuf &&
            srcFloat._gpuBackend is { } gpuBack &&
            srcGpuBuf.Handle != IntPtr.Zero &&
            slotFloat._storageOffset == 0 &&
            slotFloat.GetBackingArrayForCacheLookupUnsafe() is { } slotArr)
        {
            int srcOffset = srcFloat._storageOffset;
            int required = Math.Max(srcOffset + length, srcGpuBuf.Size);
            if (srcOffset == 0 && slotArr.Length >= required)
            {
                gpuBack.DownloadBuffer(srcGpuBuf, slotArr);
                slot.IncrementVersion();
                return slot;
            }

            lock (_slotSync)
            {
                if (_slotDownloadScratch is null || _slotDownloadScratch.Length < required)
                {
                    _slotDownloadScratch = new float[required];
                }

                gpuBack.DownloadBuffer(srcGpuBuf, _slotDownloadScratch);
                _slotDownloadScratch.AsSpan(srcOffset, length).CopyTo(slotArr.AsSpan(0, length));
            }

            slot.IncrementVersion();
            return slot;
        }

        sourceGradient.AsSpan().Slice(0, length).CopyTo(slot.AsWritableSpan().Slice(0, length));
        slot.IncrementVersion();
        return slot;
    }

    /// <summary>
    /// Completes gradient synchronization, unscales mixed-precision gradients with cross-rank overflow consensus,
    /// restores FP32 master weights, applies regularization and global-norm clipping prior to the optimizer step,
    /// executes the DDP or ZeRO/FSDP sharded parameter update (with <c>AllGather</c>), and synchronizes working weights.
    /// </summary>
    /// <param name="trainableParameters">The ordered list of trainable model parameters.</param>
    /// <param name="gradients">Dictionary of local/reduced parameter gradients.</param>
    /// <param name="optimizer">The gradient-based optimizer instance.</param>
    /// <param name="mixedPrecisionContext">Optional mixed-precision context providing dynamic loss scaling.</param>
    /// <param name="restoreFp32Masters">Optional callback restoring FP32 master weights into <paramref name="trainableParameters"/> before regularization and stepping.</param>
    /// <param name="syncFp32MastersFromWorking">Optional callback refreshing FP32 master snapshots and recasting working weights after stepping.</param>
    /// <param name="regularizeGradients">Optional callback adding explicit L1/L2/coupled regularization terms to gradients prior to global-norm clipping.</param>
    /// <param name="maxGradNorm">Maximum global L2 gradient norm (<c>&lt;= 0</c> disables model-level norm clipping).</param>
    /// <param name="executeFullParameterStep">Callback executing the full-parameter optimizer update on DDP (preserving GPU-resident optimizer paths).</param>
    /// <param name="invalidateLayerCaches">Optional callback invalidating cached layer views after parameter <c>AllGather</c>.</param>
    /// <returns><c>true</c> if the optimizer step executed; <c>false</c> if mixed-precision gradient overflow was detected and the step was skipped.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if this coordinator has been disposed.</exception>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="trainableParameters"/> or <paramref name="gradients"/> is null.</exception>
    public bool SynchronizeUnscaleClipAndStep(
        IReadOnlyList<Tensor<T>> trainableParameters,
        Dictionary<Tensor<T>, Tensor<T>> gradients,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        MixedPrecisionContext? mixedPrecisionContext,
        Action? restoreFp32Masters,
        Action? syncFp32MastersFromWorking,
        Action<IReadOnlyList<Tensor<T>>, Dictionary<Tensor<T>, Tensor<T>>>? regularizeGradients,
        double maxGradNorm,
        Action<TapeStepContext<T>>? executeFullParameterStep,
        Action? invalidateLayerCaches = null)
    {
        ThrowIfDisposed();
        if (trainableParameters is null)
        {
            throw new ArgumentNullException(nameof(trainableParameters));
        }

        if (gradients is null)
        {
            throw new ArgumentNullException(nameof(gradients));
        }

        if (optimizer is null && executeFullParameterStep is null)
        {
            throw new ArgumentException(
                "Either optimizer or executeFullParameterStep must be provided to SynchronizeUnscaleClipAndStep.",
                nameof(optimizer));
        }

        if (_stepStartTimestamp == 0L)
        {
            _stepStartTimestamp = Stopwatch.GetTimestamp();
            _stepCommMilliseconds = 0.0;
            _stepExposedCommMilliseconds = 0.0;
        }

        var reducer = EnsureReducer(trainableParameters);
        if (!_backwardStepBegun)
        {
            double defaultWeight = WorldSize <= 1 ? 1.0 : 1.0 / WorldSize;
            reducer.BeginStep(defaultWeight);
        }

        // (a) Flush & synchronize GradientBucketReducer (AllReduce for DDP/ZeRO1, ReduceScatter for ZeRO2/FSDP).
        reducer.FlushAndSynchronize(gradients, out Vector<T>? reducedLocalShard);
        _stepCommMilliseconds += reducer.LastTotalCollectiveMilliseconds;
        _stepExposedCommMilliseconds += reducer.LastExposedWaitMilliseconds;

        var uniqueParams = GetOrDeduplicateParameters(trainableParameters, out int totalElements);
        int worldSize = Math.Max(1, WorldSize);
        int shardSize = totalElements > 0 ? (int)(((long)totalElements + worldSize - 1L) / worldSize) : 0;
        int myStart = Rank * shardSize;
        int myValidLen = Math.Max(0, Math.Min(shardSize, totalElements - myStart));

        Tensor<T>? shardGradTensor = null;
        if (UseReduceScatter && myValidLen > 0)
        {
            shardGradTensor = EnsureShardGradTensor(myValidLen);
            PrepareTensorForFullHostOverwrite(shardGradTensor, AiDotNetEngine.Current as DirectGpuTensorEngine);
            if (reducedLocalShard is not null && reducedLocalShard.Length >= myValidLen)
            {
                reducedLocalShard.AsSpan().Slice(0, myValidLen).CopyTo(shardGradTensor.AsWritableSpan());
            }
            else
            {
                shardGradTensor.AsWritableSpan().Clear();
            }

            shardGradTensor.IncrementVersion();
        }

        // (b) Mixed-precision unscale (1.0 / scale) + cross-rank NaN/Inf overflow consensus.
        if (mixedPrecisionContext is not null)
        {
            double scale = mixedPrecisionContext.LossScaler?.Scale ?? 1.0;
            double invScale = !IsZeroDouble(scale) ? 1.0 / scale : 1.0;
            bool hasOverflow = false;

            if (UseReduceScatter)
            {
                if (shardGradTensor is not null && shardGradTensor.Length > 0)
                {
                    hasOverflow = UnscaleAndDetectOverflow(shardGradTensor, invScale);
                }
            }
            else
            {
                for (int i = 0; i < uniqueParams.Length; i++)
                {
                    if (gradients.TryGetValue(uniqueParams[i], out var grad) &&
                        grad is not null &&
                        grad.Length > 0 &&
                        UnscaleAndDetectOverflow(grad, invScale))
                    {
                        hasOverflow = true;
                    }
                }
            }

            if (WorldSize > 1)
            {
                _singleElementBuffer.AsWritableSpan()[0] = NumOps.FromDouble(hasOverflow ? 1.0 : 0.0);

                long t0 = Stopwatch.GetTimestamp();
                Backend.AllReduce(_singleElementBuffer, ReductionOperation.Max);
                long t1 = Stopwatch.GetTimestamp();
                double flagMs = (t1 - t0) * 1000.0 / Stopwatch.Frequency;
                _stepCommMilliseconds += flagMs;
                _stepExposedCommMilliseconds += flagMs;

                hasOverflow = NumOps.ToDouble(_singleElementBuffer.AsSpan()[0]) > 0.0;
            }

            if (hasOverflow)
            {
                restoreFp32Masters?.Invoke();
                mixedPrecisionContext.LossScaler?.RecordOverflow();
                FinalizeStepDiagnostics();
                return false;
            }
        }

        // Restore FP32 master weights BEFORE regularization and optimizer update so coupled weight decay
        // and moment updates operate on full-precision weights rather than rounded working weights.
        restoreFp32Masters?.Invoke();

        var effectiveOptimizer = UnwrapGradientOptimizer(optimizer);
        var gradOptBase = effectiveOptimizer as GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>>;

        // (c) & (d) Regularization + Global-Norm Clipping BEFORE optimizer step, followed by DDP or sharded step.
        if (!IsShardedStrategy)
        {
            if (gradOptBase is { RegularizesAfterClip: true })
            {
                if (maxGradNorm > 0.0)
                {
                    ClipGradientsByGlobalNorm(uniqueParams, gradients, maxGradNorm);
                }

                gradOptBase.ClipTapeGradientsAhead(
                    uniqueParams,
                    gradients,
                    canReuseNormInStep: regularizeGradients is null);
                regularizeGradients?.Invoke(trainableParameters, gradients);
            }
            else
            {
                regularizeGradients?.Invoke(trainableParameters, gradients);
                if (maxGradNorm > 0.0)
                {
                    ClipGradientsByGlobalNorm(uniqueParams, gradients, maxGradNorm);
                }
            }

            if (typeof(T) == typeof(float) &&
                AiDotNetEngine.Current is DirectGpuTensorEngine preStepGpuEngine &&
                preStepGpuEngine.GetBackend() is { } preStepGpuBackend)
            {
                for (int i = 0; i < uniqueParams.Length; i++)
                {
                    var p = uniqueParams[i];
                    if (gradients.TryGetValue(p, out var grad) && grad is not null && grad.Length > 0)
                    {
                        BindHostSynchronizedGpuGradient(
                            preStepGpuEngine,
                            preStepGpuBackend,
                            (Tensor<float>)(object)p,
                            (Tensor<float>)(object)grad,
                            isShardGrad: false);
                    }
                }
            }

            var fullContext = new TapeStepContext<T>(
                trainableParameters,
                gradients,
                NumOps.FromDouble(LastGlobalLoss));

            if (executeFullParameterStep is not null)
            {
                executeFullParameterStep(fullContext);
            }
            else if (effectiveOptimizer is not null)
            {
                effectiveOptimizer.Step(fullContext);
            }

            var ddpGpuEngine = AiDotNetEngine.Current as DirectGpuTensorEngine;
            for (int i = 0; i < uniqueParams.Length; i++)
            {
                var p = uniqueParams[i];
                if (!p.HasPendingGpuData && (ddpGpuEngine is null || !ddpGpuEngine.IsDeviceAuthoritative(p)))
                {
                    p.IncrementVersion();
                    ddpGpuEngine?.InvalidateResidentWeightBuffer(p);
                }
            }

            invalidateLayerCaches?.Invoke();
        }
        else
        {
            if (myValidLen > 0)
            {
                var (ownedShard, _, shardCreated) = EnsureOwnedParamShard(myValidLen);
                bool mpRestoredBeforeStep = mixedPrecisionContext is not null && restoreFp32Masters is not null;
                if (shardCreated || mpRestoredBeforeStep || !IsShardedParameterSliceUpToDate(uniqueParams, myStart, myValidLen))
                {
                    PackParameterSliceIntoShard(uniqueParams, myStart, myValidLen, ownedShard);
                }

                EnsureOwnedParamShardGpuResident(ownedShard);
            }

            double optClipNorm = gradOptBase is not null && !double.IsNaN(gradOptBase.TapeStepGradientClipNorm)
                ? gradOptBase.TapeStepGradientClipNorm
                : 0.0;

            if (Strategy == DistributedStrategy.ZeRO1)
            {
                // ZeRO-1 reduced full gradients via AllReduce: regularize and clip globally on full gradients,
                // then slice this rank's owned contiguous shard for the sharded optimizer step.
                if (gradOptBase is { RegularizesAfterClip: true })
                {
                    if (maxGradNorm > 0.0)
                    {
                        ClipGradientsByGlobalNorm(uniqueParams, gradients, maxGradNorm);
                    }

                    if (optClipNorm > 0.0)
                    {
                        ClipGradientsByGlobalNorm(uniqueParams, gradients, optClipNorm);
                    }

                    regularizeGradients?.Invoke(trainableParameters, gradients);
                }
                else
                {
                    regularizeGradients?.Invoke(trainableParameters, gradients);
                    if (maxGradNorm > 0.0)
                    {
                        ClipGradientsByGlobalNorm(uniqueParams, gradients, maxGradNorm);
                    }

                    if (optClipNorm > 0.0)
                    {
                        ClipGradientsByGlobalNorm(uniqueParams, gradients, optClipNorm);
                    }
                }

                if (myValidLen > 0)
                {
                    shardGradTensor = EnsureShardGradTensor(myValidLen);
                    PackGradientSliceIntoShard(uniqueParams, gradients, myStart, myValidLen, shardGradTensor);
                }
            }
            else
            {
                // ZeRO-2 / FSDP: each rank holds only its ReduceScattered owned gradient shard.
                if (gradOptBase is { RegularizesAfterClip: true })
                {
                    ApplyDistributedShardGlobalClip(shardGradTensor, maxGradNorm, optClipNorm);
                    shardGradTensor = ApplyShardRegularization(shardGradTensor, regularizeGradients);
                }
                else
                {
                    shardGradTensor = ApplyShardRegularization(shardGradTensor, regularizeGradients);
                    ApplyDistributedShardGlobalClip(shardGradTensor, maxGradNorm, optClipNorm);
                }
            }

            if (myValidLen > 0 &&
                _ownedParamShard is not null &&
                shardGradTensor is not null &&
                typeof(T) == typeof(float) &&
                AiDotNetEngine.Current is DirectGpuTensorEngine shardedGpuEngine &&
                shardedGpuEngine.GetBackend() is { } shardedGpuBackend)
            {
                BindHostSynchronizedGpuGradient(
                    shardedGpuEngine,
                    shardedGpuBackend,
                    (Tensor<float>)(object)_ownedParamShard,
                    (Tensor<float>)(object)shardGradTensor,
                    isShardGrad: true);
            }

            if (myValidLen > 0 &&
                effectiveOptimizer is not null &&
                _ownedParamShard is not null &&
                _ownedParamShardList is not null &&
                shardGradTensor is not null)
            {
                var shardGradDict = PopulateShardGradDict(_ownedParamShard, shardGradTensor);

                if (gradOptBase is not null && optClipNorm > 0.0)
                {
                    gradOptBase.MarkGradientsClippedAhead(shardGradDict);
                }

                effectiveOptimizer.Step(new TapeStepContext<T>(
                    _ownedParamShardList,
                    shardGradDict,
                    NumOps.FromDouble(LastGlobalLoss)));
            }
            else if (myValidLen > 0 &&
                     effectiveOptimizer is null &&
                     executeFullParameterStep is not null &&
                     _ownedParamShard is not null &&
                     _ownedParamShardList is not null &&
                     shardGradTensor is not null)
            {
                var shardGradDict = PopulateShardGradDict(_ownedParamShard, shardGradTensor);

                executeFullParameterStep(new TapeStepContext<T>(
                    _ownedParamShardList,
                    shardGradDict,
                    NumOps.FromDouble(LastGlobalLoss)));
            }

            if (shardSize > 0 && totalElements > 0)
            {
                if (WorldSize <= 1)
                {
                    if (_ownedParamShard is not null)
                    {
                        if (_gatherSendBuffer is null || _gatherSendBuffer.Length < totalElements)
                        {
                            _gatherSendBuffer = new Vector<T>(totalElements);
                        }

                        CopyOwnedParamShardToVector(_ownedParamShard, totalElements, _gatherSendBuffer);
                        UnpackVectorIntoParameters(
                            _gatherSendBuffer.AsSpan().Slice(0, totalElements),
                            uniqueParams,
                            preserveGpuWeightBuffers: true);
                        RecordShardedParameterVersions(uniqueParams, myStart, myValidLen);
                    }
                }
                else
                {
                    if (_gatherSendBuffer is null || _gatherSendBuffer.Length != shardSize)
                    {
                        _gatherSendBuffer = new Vector<T>(shardSize);
                    }

                    if (myValidLen > 0 && _ownedParamShard is not null)
                    {
                        CopyOwnedParamShardToVector(_ownedParamShard, myValidLen, _gatherSendBuffer);
                    }

                    if (myValidLen < shardSize)
                    {
                        _gatherSendBuffer.AsWritableSpan().Slice(myValidLen, shardSize - myValidLen).Clear();
                    }

                    long t0 = Stopwatch.GetTimestamp();
                    Vector<T> gathered = Backend.AllGather(_gatherSendBuffer);
                    long t1 = Stopwatch.GetTimestamp();
                    double gatherMs = (t1 - t0) * 1000.0 / Stopwatch.Frequency;
                    _stepCommMilliseconds += gatherMs;
                    _stepExposedCommMilliseconds += gatherMs;

                    UnpackVectorIntoParameters(
                        gathered.AsSpan().Slice(0, totalElements),
                        uniqueParams,
                        preserveGpuWeightBuffers: true);
                    RecordShardedParameterVersions(uniqueParams, myStart, myValidLen);
                }

                invalidateLayerCaches?.Invoke();
            }
        }

        // (e) Sync working weights from updated FP32 master weights and record successful scaler update.
        syncFp32MastersFromWorking?.Invoke();
        mixedPrecisionContext?.LossScaler?.RecordSuccess();
        FinalizeStepDiagnostics();
        return true;
    }

    /// <summary>
    /// Consolidates optimizer state across all ranks into a canonical 1-rank-compatible byte payload on Rank 0
    /// (returning <see cref="Array.Empty{T}"/> on non-zero ranks when <see cref="WorldSize"/> &gt; 1), enabling seamless
    /// checkpoint save/resume across different world sizes (<c>N</c>-rank to <c>1</c>-rank or <c>M</c>-rank).
    /// </summary>
    /// <param name="optimizer">The optimizer whose state should be consolidated and serialized.</param>
    /// <param name="trainableParameters">The ordered list of trainable model parameters defining the canonical per-parameter layout.</param>
    /// <returns>Canonical serialized optimizer bytes on Rank 0; empty byte array on non-zero ranks.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if this coordinator has been disposed.</exception>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="optimizer"/> or <paramref name="trainableParameters"/> is null.</exception>
    public byte[] ConsolidateOptimizerStateBytes(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        IReadOnlyList<Tensor<T>> trainableParameters)
    {
        ThrowIfDisposed();
        if (optimizer is null)
        {
            throw new ArgumentNullException(nameof(optimizer));
        }

        if (trainableParameters is null)
        {
            throw new ArgumentNullException(nameof(trainableParameters));
        }

        var effectiveOptimizer = UnwrapGradientOptimizer(optimizer) ?? optimizer;

        if (!IsShardedStrategy)
        {
            return WorldSize <= 1 || Rank == 0
                ? effectiveOptimizer.Serialize()
                : Array.Empty<byte>();
        }

        var uniqueParams = GetOrDeduplicateParameters(trainableParameters, out int totalElements);
        if (uniqueParams.Length == 0 || totalElements <= 0)
        {
            return WorldSize <= 1 || Rank == 0
                ? effectiveOptimizer.Serialize()
                : Array.Empty<byte>();
        }

        if (effectiveOptimizer is not GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> gradOpt ||
            !gradOpt.SupportsShardedTapeStateConsolidation)
        {
            if (WorldSize <= 1)
            {
                return effectiveOptimizer.Serialize();
            }

            throw new NotSupportedException(
                $"Sharded optimizer state consolidation across {WorldSize} ranks under {Strategy} is not supported for optimizer type '{effectiveOptimizer.GetType().Name}'. " +
                "Use an element-aligned GradientBasedOptimizerBase (such as AdamOptimizer, AdamWOptimizer with Float32/BFloat16 moments, SGD, Lion, or RMSProp), or use DistributedStrategy.DDP.");
        }

        EnsureBackendInitialized();

        int worldSize = Math.Max(1, WorldSize);
        int shardSize = (int)(((long)totalElements + worldSize - 1L) / worldSize);
        int myStart = Rank * shardSize;
        int myValidLen = Math.Max(0, Math.Min(shardSize, totalElements - myStart));

        string[] stateKeys = gradOpt.GetSortedTapeStateKeyNames();
        var consolidatedPerField = Rank == 0
            ? new Dictionary<string, Dictionary<int, Tensor<T>>>(stateKeys.Length, StringComparer.Ordinal)
            : null;

        for (int f = 0; f < stateKeys.Length; f++)
        {
            string stateKey = stateKeys[f];
            Tensor<T>? localShardState = gradOpt.TryExtractShardTapeStateTensor(
                stateKey,
                _ownedParamShard,
                myValidLen,
                out bool disposeLocal);
            try
            {
                bool hasLocal = localShardState is not null;
                bool anyHasState = WorldSize > 1 ? AllReduceAnyRankFlag(hasLocal) : hasLocal;

                if (!anyHasState)
                {
                    continue;
                }

                Vector<T> fullMoment;
                if (WorldSize <= 1)
                {
                    fullMoment = new Vector<T>(totalElements);
                    if (localShardState is not null)
                    {
                        int copyLen = Math.Min(totalElements, localShardState.Length);
                        localShardState.AsSpan().Slice(0, copyLen).CopyTo(fullMoment.AsWritableSpan().Slice(0, copyLen));
                    }
                }
                else
                {
                    var paddedShard = new Vector<T>(shardSize);
                    Span<T> sendSpan = paddedShard.AsWritableSpan();
                    if (localShardState is not null && myValidLen > 0)
                    {
                        int copyLen = Math.Min(myValidLen, localShardState.Length);
                        localShardState.AsSpan().Slice(0, copyLen).CopyTo(sendSpan.Slice(0, copyLen));
                        if (copyLen < shardSize)
                        {
                            sendSpan.Slice(copyLen, shardSize - copyLen).Clear();
                        }
                    }
                    else
                    {
                        sendSpan.Clear();
                    }

                    fullMoment = Backend.AllGather(paddedShard);
                }

                if (Rank == 0 && consolidatedPerField is not null)
                {
                    ReadOnlySpan<T> gatheredSpan = fullMoment.AsSpan();
                    var perParamDict = new Dictionary<int, Tensor<T>>(uniqueParams.Length);
                    int offset = 0;
                    for (int pIdx = 0; pIdx < uniqueParams.Length; pIdx++)
                    {
                        var param = uniqueParams[pIdx];
                        var paramMoment = new Tensor<T>(param._shape);
                        gatheredSpan.Slice(offset, param.Length).CopyTo(paramMoment.AsWritableSpan());
                        perParamDict[pIdx] = paramMoment;
                        offset += param.Length;
                    }

                    consolidatedPerField[stateKey] = perParamDict;
                }
            }
            finally
            {
                if (disposeLocal && localShardState is not null)
                {
                    localShardState.Dispose();
                }
            }
        }

        if (WorldSize > 1 && Rank != 0)
        {
            return Array.Empty<byte>();
        }

        return gradOpt.SerializeWithConsolidatedTapeStates(consolidatedPerField);
    }

    private bool AllReduceAnyRankFlag(bool hasLocal)
    {
        _singleElementBuffer.AsWritableSpan()[0] = NumOps.FromDouble(hasLocal ? 1.0 : 0.0);
        Backend.AllReduce(_singleElementBuffer, ReductionOperation.Max);
        return NumOps.ToDouble(_singleElementBuffer.AsSpan()[0]) > 0.0;
    }

    /// <summary>
    /// Broadcasts consolidated 1-rank-format optimizer checkpoint bytes from Rank 0 to all ranks, deserializes the
    /// optimizer state, and—for sharded strategies (<see cref="DistributedStrategy.ZeRO1"/>, <see cref="DistributedStrategy.ZeRO2"/>,
    /// <see cref="DistributedStrategy.FSDP"/>)—reshards canonical per-parameter moment tensors onto each rank's owned
    /// parameter shard <c>[myStart .. myStart + myValidLen)</c>.
    /// </summary>
    /// <param name="optimizer">The optimizer instance to restore into on this rank.</param>
    /// <param name="trainableParameters">The ordered list of trainable model parameters defining the canonical layout.</param>
    /// <param name="rank0Bytes">Consolidated optimizer state bytes on Rank 0 (ignored on non-zero ranks when <see cref="WorldSize"/> &gt; 1).</param>
    /// <exception cref="ObjectDisposedException">Thrown if this coordinator has been disposed.</exception>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="optimizer"/> or <paramref name="trainableParameters"/> is null.</exception>
    public void RestoreOptimizerStateFromConsolidatedBytes(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        IReadOnlyList<Tensor<T>> trainableParameters,
        byte[]? rank0Bytes)
    {
        ThrowIfDisposed();
        if (optimizer is null)
        {
            throw new ArgumentNullException(nameof(optimizer));
        }

        if (trainableParameters is null)
        {
            throw new ArgumentNullException(nameof(trainableParameters));
        }

        var effectiveOptimizer = UnwrapGradientOptimizer(optimizer) ?? optimizer;
        int totalElements = 0;
        var uniqueParams = IsShardedStrategy
            ? GetOrDeduplicateParameters(trainableParameters, out totalElements)
            : Array.Empty<Tensor<T>>();

        if (IsShardedStrategy &&
            WorldSize > 1 &&
            uniqueParams.Length > 0 &&
            totalElements > 0 &&
            (effectiveOptimizer is not GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> checkGradOpt ||
             !checkGradOpt.SupportsShardedTapeStateConsolidation))
        {
            throw new NotSupportedException(
                $"Sharded optimizer state restoration across {WorldSize} ranks under {Strategy} is not supported for optimizer type '{effectiveOptimizer.GetType().Name}'. " +
                "Use an element-aligned GradientBasedOptimizerBase (such as AdamOptimizer, AdamWOptimizer with Float32/BFloat16 moments, SGD, Lion, or RMSProp), or use DistributedStrategy.DDP.");
        }

        byte[] payload = BroadcastBytePayload(rank0Bytes, root: 0);
        if (payload.Length == 0)
        {
            return;
        }

        effectiveOptimizer.Deserialize(payload);

        if (!IsShardedStrategy ||
            uniqueParams.Length == 0 ||
            totalElements <= 0 ||
            effectiveOptimizer is not GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> gradOpt)
        {
            return;
        }

        int worldSize = Math.Max(1, WorldSize);
        int shardSize = (int)(((long)totalElements + worldSize - 1L) / worldSize);
        int myStart = Rank * shardSize;
        int myValidLen = Math.Max(0, Math.Min(shardSize, totalElements - myStart));

        if (myValidLen > 0)
        {
            // Allocate a fresh shard tensor reference on checkpoint restore so any per-tensor
            // backing-array or compressed-slot cache in the optimizer cannot alias pre-restore moments.
            _ownedParamShard = new Tensor<T>(new[] { myValidLen });
            _ownedParamShardList = new[] { _ownedParamShard };
            _lastShardedParamsRef = null;
            PackParameterSliceIntoShard(uniqueParams, myStart, myValidLen, _ownedParamShard);
        }

        gradOpt.ReshardPendingTapeStatesForLocalSlice(uniqueParams, myStart, myValidLen);
    }

    private byte[] BroadcastBytePayload(byte[]? rank0Bytes, int root)
    {
        if (WorldSize <= 1)
        {
            return rank0Bytes ?? Array.Empty<byte>();
        }

        EnsureBackendInitialized();

        int localLen = Rank == root && rank0Bytes is not null ? rank0Bytes.Length : 0;
        var header = new Vector<T>(4);
        if (Rank == root)
        {
            Span<T> hSpan = header.AsWritableSpan();
            hSpan[0] = NumOps.FromDouble(localLen & 0xFF);
            hSpan[1] = NumOps.FromDouble((localLen >> 8) & 0xFF);
            hSpan[2] = NumOps.FromDouble((localLen >> 16) & 0xFF);
            hSpan[3] = NumOps.FromDouble((localLen >> 24) & 0xFF);
        }

        header = Backend.Broadcast(header, root);
        ReadOnlySpan<T> hRead = header.AsSpan();
        int b0 = (int)Math.Round(NumOps.ToDouble(hRead[0])) & 0xFF;
        int b1 = (int)Math.Round(NumOps.ToDouble(hRead[1])) & 0xFF;
        int b2 = (int)Math.Round(NumOps.ToDouble(hRead[2])) & 0xFF;
        int b3 = (int)Math.Round(NumOps.ToDouble(hRead[3])) & 0xFF;
        int byteLength = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);

        if (byteLength <= 0)
        {
            return Array.Empty<byte>();
        }

        var encoded = new Vector<T>(byteLength);
        if (Rank == root && rank0Bytes is not null)
        {
            PackBytesIntoSpan(rank0Bytes, encoded.AsWritableSpan());
        }

        encoded = Backend.Broadcast(encoded, root);
        if (Rank == root && rank0Bytes is not null)
        {
            return rank0Bytes;
        }

        var decoded = new byte[byteLength];
        UnpackBytesFromSpan(encoded.AsSpan(), decoded);
        return decoded;
    }

    [AiDotNet.Attributes.ScalarKernel("One-time checkpoint byte-to-T packing (exact 0..255 integer values across all numeric types).")]
    private static void PackBytesIntoSpan(byte[] source, Span<T> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = NumOps.FromDouble(source[i]);
        }
    }

    [AiDotNet.Attributes.ScalarKernel("One-time checkpoint T-to-byte unpacking (exact 0..255 integer values across all numeric types).")]
    private static void UnpackBytesFromSpan(ReadOnlySpan<T> source, byte[] destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (byte)Math.Round(NumOps.ToDouble(source[i]));
        }
    }

    private Dictionary<Tensor<T>, Tensor<T>> PopulateShardGradDict(Tensor<T> paramShard, Tensor<T> gradShard)
    {
        _shardGradDict ??= new Dictionary<Tensor<T>, Tensor<T>>(1, TensorReferenceComparer<Tensor<T>>.Instance);
        _shardGradDict.Clear();
        _shardGradDict[paramShard] = gradShard;
        return _shardGradDict;
    }

    private Tensor<T>? ApplyShardRegularization(
        Tensor<T>? shardGradTensor,
        Action<IReadOnlyList<Tensor<T>>, Dictionary<Tensor<T>, Tensor<T>>>? regularizeGradients)
    {
        if (regularizeGradients is null ||
            shardGradTensor is null ||
            shardGradTensor.Length == 0 ||
            _ownedParamShard is null ||
            _ownedParamShardList is null)
        {
            return shardGradTensor;
        }

        _shardRegDict ??= new Dictionary<Tensor<T>, Tensor<T>>(1, TensorReferenceComparer<Tensor<T>>.Instance);
        _shardRegDict.Clear();
        _shardRegDict[_ownedParamShard] = shardGradTensor;

        regularizeGradients(_ownedParamShardList, _shardRegDict);
        return _shardRegDict.TryGetValue(_ownedParamShard, out var updated) && updated is not null
            ? updated
            : shardGradTensor;
    }

    private void ApplyDistributedShardGlobalClip(Tensor<T>? shardGradTensor, double maxGradNorm, double optClipNorm)
    {
        if (maxGradNorm <= 0.0 && optClipNorm <= 0.0)
        {
            return;
        }

        double localNormSq = shardGradTensor is not null && shardGradTensor.Length > 0
            ? ComputeSumOfSquares(shardGradTensor)
            : 0.0;

        double globalNormSq = localNormSq;
        if (WorldSize > 1)
        {
            _singleElementBuffer.AsWritableSpan()[0] = NumOps.FromDouble(localNormSq);

            long t0 = Stopwatch.GetTimestamp();
            Backend.AllReduce(_singleElementBuffer, ReductionOperation.Sum);
            long t1 = Stopwatch.GetTimestamp();
            double normMs = (t1 - t0) * 1000.0 / Stopwatch.Frequency;
            _stepCommMilliseconds += normMs;
            _stepExposedCommMilliseconds += normMs;

            globalNormSq = NumOps.ToDouble(_singleElementBuffer.AsSpan()[0]);
        }

        if (double.IsNaN(globalNormSq) || double.IsInfinity(globalNormSq) || globalNormSq <= 0.0)
        {
            return;
        }

        double globalNorm = Math.Sqrt(globalNormSq);
        if (double.IsNaN(globalNorm) || double.IsInfinity(globalNorm))
        {
            return;
        }

        if (maxGradNorm > 0.0 && globalNorm > maxGradNorm)
        {
            double s1 = maxGradNorm / (globalNorm + 1e-6);
            if (shardGradTensor is not null && shardGradTensor.Length > 0)
            {
                ScaleTensorInPlace(shardGradTensor, s1);
            }

            globalNorm *= s1;
        }

        if (optClipNorm > 0.0 && globalNorm > optClipNorm)
        {
            double s2 = optClipNorm / (globalNorm + 1e-6);
            if (shardGradTensor is not null && shardGradTensor.Length > 0)
            {
                ScaleTensorInPlace(shardGradTensor, s2);
            }
        }
    }

    private static void ClipGradientsByGlobalNorm(
        Tensor<T>[] uniqueParams,
        Dictionary<Tensor<T>, Tensor<T>> gradients,
        double maxNorm)
    {
        if (maxNorm <= 0.0 || double.IsNaN(maxNorm) || double.IsInfinity(maxNorm))
        {
            return;
        }

        double totalNormSq = 0.0;
        for (int i = 0; i < uniqueParams.Length; i++)
        {
            if (gradients.TryGetValue(uniqueParams[i], out var grad) && grad is not null && grad.Length > 0)
            {
                totalNormSq += ComputeSumOfSquares(grad);
            }
        }

        if (double.IsNaN(totalNormSq) || double.IsInfinity(totalNormSq) || totalNormSq <= 0.0)
        {
            return;
        }

        double globalNorm = Math.Sqrt(totalNormSq);
        if (double.IsNaN(globalNorm) || double.IsInfinity(globalNorm) || globalNorm <= maxNorm)
        {
            return;
        }

        double clipScale = maxNorm / (globalNorm + 1e-6);
        for (int i = 0; i < uniqueParams.Length; i++)
        {
            if (gradients.TryGetValue(uniqueParams[i], out var grad) && grad is not null && grad.Length > 0)
            {
                ScaleTensorInPlace(grad, clipScale);
            }
        }
    }

    private static double ComputeSumOfSquares(Tensor<T> tensor)
    {
        if ((object)tensor is Tensor<float> tf)
        {
            ReadOnlySpan<float> span = tf.AsSpan();
#if !AIDN_LEGACY_TFM
            float sumF = TensorPrimitives.SumOfSquares(span);
            if (!float.IsInfinity(sumF))
            {
                return sumF;
            }
#endif
            return ComputeSumOfSquaresFloatWideSimd(span);
        }

        if ((object)tensor is Tensor<double> td)
        {
            ReadOnlySpan<double> span = td.AsSpan();
#if !AIDN_LEGACY_TFM
            return TensorPrimitives.SumOfSquares(span);
#else
            return ComputeSumOfSquaresDoubleSimdLegacy(span);
#endif
        }

        return ComputeSumOfSquaresGeneric(tensor.AsSpan());
    }

    private static double ComputeSumOfSquaresFloatWideSimd(ReadOnlySpan<float> span)
    {
        int length = span.Length;
        int i = 0;
        double sum = 0.0;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<float>.Count)
        {
            int width = System.Numerics.Vector<float>.Count;
            int simdLen = length - (length % width);
            var vecs = MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(span.Slice(0, simdLen));
            var accLo = System.Numerics.Vector<double>.Zero;
            var accHi = System.Numerics.Vector<double>.Zero;
            for (int v = 0; v < vecs.Length; v++)
            {
                System.Numerics.Vector.Widen(vecs[v], out var lo, out var hi);
                accLo += lo * lo;
                accHi += hi * hi;
            }

            var totalVec = accLo + accHi;
            sum += System.Numerics.Vector.Dot(totalVec, System.Numerics.Vector<double>.One);
            i = simdLen;
        }

        for (; i < length; i++)
        {
            double val = span[i];
            sum += val * val;
        }

        return sum;
    }

#if AIDN_LEGACY_TFM
    private static double ComputeSumOfSquaresDoubleSimdLegacy(ReadOnlySpan<double> span)
    {
        int length = span.Length;
        int i = 0;
        double sum = 0.0;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<double>.Count)
        {
            int width = System.Numerics.Vector<double>.Count;
            int simdLen = length - (length % width);
            var vecs = MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(span.Slice(0, simdLen));
            var acc = System.Numerics.Vector<double>.Zero;
            for (int v = 0; v < vecs.Length; v++)
            {
                var dv = vecs[v];
                acc += dv * dv;
            }

            sum += System.Numerics.Vector.Dot(acc, System.Numerics.Vector<double>.One);
            i = simdLen;
        }

        for (; i < length; i++)
        {
            double val = span[i];
            sum += val * val;
        }

        return sum;
    }
#endif

    [AiDotNet.Attributes.ScalarKernel("Generic T L2 norm sum-of-squares fallback after float and double fast paths.")]
    private static double ComputeSumOfSquaresGeneric(ReadOnlySpan<T> span)
    {
        double sum = 0.0;
        for (int i = 0; i < span.Length; i++)
        {
            double v = NumOps.ToDouble(span[i]);
            sum += v * v;
        }

        return sum;
    }

    private static bool IsZeroDouble(double value) => !(Math.Abs(value) > 0.0);

    private static bool IsUnitDouble(double value) => !(value < 1.0 || value > 1.0);

    private static bool IsUnitFloat(float value) => !(value < 1.0f || value > 1.0f);

    private static void ScaleTensorInPlace(Tensor<T> tensor, double scale)
    {
        if (tensor.Length == 0 || IsUnitDouble(scale))
        {
            return;
        }

        if ((object)tensor is Tensor<float> tf)
        {
            float factor = (float)scale;
            Span<float> span = tf.AsWritableSpan();
#if !AIDN_LEGACY_TFM
            TensorPrimitives.Multiply(span, factor, span);
#else
            ScaleFloatSimdLegacy(span, factor);
#endif
        }
        else if ((object)tensor is Tensor<double> td)
        {
            Span<double> span = td.AsWritableSpan();
#if !AIDN_LEGACY_TFM
            TensorPrimitives.Multiply(span, scale, span);
#else
            ScaleDoubleSimdLegacy(span, scale);
#endif
        }
        else
        {
            ScaleSpanGenericInPlace(tensor.AsWritableSpan(), scale);
        }

        tensor.IncrementVersion();
        if (AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)
        {
            gpuEngine.InvalidateResidentWeightBuffer(tensor);
        }
    }

#if AIDN_LEGACY_TFM
    private static void ScaleFloatSimdLegacy(Span<float> span, float factor)
    {
        int length = span.Length;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<float>.Count)
        {
            int width = System.Numerics.Vector<float>.Count;
            int simdLen = length - (length % width);
            var factorVec = new System.Numerics.Vector<float>(factor);
            var vecs = MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(span.Slice(0, simdLen));
            for (int v = 0; v < vecs.Length; v++)
            {
                vecs[v] *= factorVec;
            }

            i = simdLen;
        }

        for (; i < length; i++)
        {
            span[i] *= factor;
        }
    }

    private static void ScaleDoubleSimdLegacy(Span<double> span, double factor)
    {
        int length = span.Length;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<double>.Count)
        {
            int width = System.Numerics.Vector<double>.Count;
            int simdLen = length - (length % width);
            var factorVec = new System.Numerics.Vector<double>(factor);
            var vecs = MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(span.Slice(0, simdLen));
            for (int v = 0; v < vecs.Length; v++)
            {
                vecs[v] *= factorVec;
            }

            i = simdLen;
        }

        for (; i < length; i++)
        {
            span[i] *= factor;
        }
    }
#endif

    [AiDotNet.Attributes.ScalarKernel("Generic T in-place gradient scaling fallback after float and double fast paths.")]
    private static void ScaleSpanGenericInPlace(Span<T> span, double scale)
    {
        T factor = NumOps.FromDouble(scale);
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = NumOps.Multiply(span[i], factor);
        }
    }

    private static bool UnscaleAndDetectOverflow(Tensor<T> tensor, double invScale)
    {
        bool overflow;
        if ((object)tensor is Tensor<float> tf)
        {
            float factor = (float)invScale;
            Span<float> span = tf.AsWritableSpan();
            if (!IsUnitFloat(factor))
            {
#if !AIDN_LEGACY_TFM
                TensorPrimitives.Multiply(span, factor, span);
#else
                ScaleFloatSimdLegacy(span, factor);
#endif
            }

            overflow = ContainsNonFiniteFloatSimd(span);
        }
        else if ((object)tensor is Tensor<double> td)
        {
            Span<double> span = td.AsWritableSpan();
            if (!IsUnitDouble(invScale))
            {
#if !AIDN_LEGACY_TFM
                TensorPrimitives.Multiply(span, invScale, span);
#else
                ScaleDoubleSimdLegacy(span, invScale);
#endif
            }

            overflow = ContainsNonFiniteDoubleSimd(span);
        }
        else
        {
            overflow = UnscaleAndDetectOverflowGeneric(tensor.AsWritableSpan(), invScale);
        }

        tensor.IncrementVersion();
        if (AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)
        {
            gpuEngine.InvalidateResidentWeightBuffer(tensor);
        }

        return overflow;
    }

    private static bool ContainsNonFiniteFloatSimd(ReadOnlySpan<float> span)
    {
        ReadOnlySpan<int> bits = MemoryMarshal.Cast<float, int>(span);
        int length = bits.Length;
        int i = 0;
        const int expMaskScalar = 0x7F800000;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<int>.Count)
        {
            int width = System.Numerics.Vector<int>.Count;
            int simdLen = length - (length % width);
            var expMask = new System.Numerics.Vector<int>(expMaskScalar);
            var vecs = MemoryMarshal.Cast<int, System.Numerics.Vector<int>>(bits.Slice(0, simdLen));
            for (int v = 0; v < vecs.Length; v++)
            {
                if (System.Numerics.Vector.EqualsAny(vecs[v] & expMask, expMask))
                {
                    return true;
                }
            }

            i = simdLen;
        }

        for (; i < length; i++)
        {
            if ((bits[i] & expMaskScalar) == expMaskScalar)
            {
                return true;
            }
        }

        return false;
    }

    private static bool ContainsNonFiniteDoubleSimd(ReadOnlySpan<double> span)
    {
        ReadOnlySpan<long> bits = MemoryMarshal.Cast<double, long>(span);
        int length = bits.Length;
        int i = 0;
        const long expMaskScalar = 0x7FF0000000000000L;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<long>.Count)
        {
            int width = System.Numerics.Vector<long>.Count;
            int simdLen = length - (length % width);
            var expMask = new System.Numerics.Vector<long>(expMaskScalar);
            var vecs = MemoryMarshal.Cast<long, System.Numerics.Vector<long>>(bits.Slice(0, simdLen));
            for (int v = 0; v < vecs.Length; v++)
            {
                if (System.Numerics.Vector.EqualsAny(vecs[v] & expMask, expMask))
                {
                    return true;
                }
            }

            i = simdLen;
        }

        for (; i < length; i++)
        {
            if ((bits[i] & expMaskScalar) == expMaskScalar)
            {
                return true;
            }
        }

        return false;
    }

    [AiDotNet.Attributes.ScalarKernel("Generic T mixed-precision unscale and NaN/Inf overflow detection fallback.")]
    private static bool UnscaleAndDetectOverflowGeneric(Span<T> span, double invScale)
    {
        T factor = NumOps.FromDouble(invScale);
        bool overflow = false;
        for (int i = 0; i < span.Length; i++)
        {
            T unscaled = NumOps.Multiply(span[i], factor);
            span[i] = unscaled;
            if (!overflow && (NumOps.IsNaN(unscaled) || NumOps.IsInfinity(unscaled)))
            {
                overflow = true;
            }
        }

        return overflow;
    }

    private GradientBucketReducer<T> EnsureReducer(IReadOnlyList<Tensor<T>> trainableParameters)
    {
        if (_reducer is { } existing &&
            existing.MatchesConfiguration(
                trainableParameters,
                GradientBucketSizeBytes,
                OverlapCommunication,
                UseReduceScatter))
        {
            existing.GradientSlotProvider ??= GetOrCreateGradientSlot;
            return existing;
        }

        _reducer?.Dispose();
        var created = new GradientBucketReducer<T>(
            Backend,
            trainableParameters,
            GradientBucketSizeBytes,
            OverlapCommunication,
            UseReduceScatter)
        {
            GradientSlotProvider = GetOrCreateGradientSlot
        };
        _reducer = created;
        return created;
    }

    private (Tensor<T> Shard, Tensor<T>[] ShardList, bool Created) EnsureOwnedParamShard(int validLength)
    {
        if (_ownedParamShard is { } existingShard &&
            existingShard.Length == validLength &&
            _ownedParamShardList is { } existingList)
        {
            return (existingShard, existingList, false);
        }

        var createdShard = new Tensor<T>(new[] { validLength });
        var createdList = new[] { createdShard };
        _ownedParamShard = createdShard;
        _ownedParamShardList = createdList;
        _lastShardedParamsRef = null;
        return (createdShard, createdList, true);
    }

    private Tensor<T> EnsureShardGradTensor(int validLength)
    {
        if (_shardGradTensor is null || _shardGradTensor.Length != validLength)
        {
            _shardGradTensor = new Tensor<T>(new[] { validLength });
        }

        return _shardGradTensor;
    }

    private void CopyOwnedParamShardToVector(Tensor<T> ownedParamShard, int validLength, Vector<T> destination)
    {
        if (validLength <= 0)
        {
            return;
        }

        if (typeof(T) == typeof(float) &&
            (object)ownedParamShard is Tensor<float> pf &&
            (object)destination is Vector<float> dstFloat &&
            pf.HasPendingGpuData &&
            pf._gpuBuffer is { } shardBuf &&
            shardBuf.Handle != IntPtr.Zero &&
            pf._gpuBackend is { } gpuBack &&
            pf._gpuBufferVersion == pf.Version)
        {
            int srcOffset = pf._storageOffset;
            int required = Math.Max(srcOffset + validLength, shardBuf.Size);
            lock (_slotSync)
            {
                if (_slotDownloadScratch is null || _slotDownloadScratch.Length < required)
                {
                    _slotDownloadScratch = new float[required];
                }

                gpuBack.DownloadBuffer(shardBuf, _slotDownloadScratch);
                _slotDownloadScratch.AsSpan(srcOffset, validLength).CopyTo(dstFloat.AsWritableSpan().Slice(0, validLength));
            }

            return;
        }

        ownedParamShard.AsSpan().Slice(0, validLength).CopyTo(destination.AsWritableSpan().Slice(0, validLength));
    }

    private bool IsShardedParameterSliceUpToDate(Tensor<T>[] uniqueParams, int shardStart, int shardLength)
    {
        if (!ReferenceEquals(_lastShardedParamsRef, uniqueParams) ||
            _lastShardedParamVersions is null ||
            _lastShardedParamGpuVersions is null ||
            _lastShardedParamVersions.Length != uniqueParams.Length ||
            _lastShardedParamGpuVersions.Length != uniqueParams.Length ||
            _lastShardedStart != shardStart ||
            _lastShardedLength != shardLength)
        {
            return false;
        }

        for (int i = 0; i < uniqueParams.Length; i++)
        {
            var p = uniqueParams[i];
            if (p.Version != _lastShardedParamVersions[i] ||
                p.GpuCacheVersion != _lastShardedParamGpuVersions[i])
            {
                return false;
            }
        }

        return true;
    }

    private void RecordShardedParameterVersions(Tensor<T>[] uniqueParams, int shardStart, int shardLength)
    {
        if (_lastShardedParamVersions is null || _lastShardedParamVersions.Length != uniqueParams.Length)
        {
            _lastShardedParamVersions = new int[uniqueParams.Length];
        }

        if (_lastShardedParamGpuVersions is null || _lastShardedParamGpuVersions.Length != uniqueParams.Length)
        {
            _lastShardedParamGpuVersions = new int[uniqueParams.Length];
        }

        for (int i = 0; i < uniqueParams.Length; i++)
        {
            var p = uniqueParams[i];
            _lastShardedParamVersions[i] = p.Version;
            _lastShardedParamGpuVersions[i] = p.GpuCacheVersion;
        }

        _lastShardedParamsRef = uniqueParams;
        _lastShardedStart = shardStart;
        _lastShardedLength = shardLength;
    }

    private static void EnsureOwnedParamShardGpuResident(Tensor<T> ownedParamShard)
    {
        if (typeof(T) != typeof(float) ||
            AiDotNetEngine.Current is not DirectGpuTensorEngine gpuEngine ||
            gpuEngine.GetBackend() is not { } gpuBackend)
        {
            return;
        }

        var pf = (Tensor<float>)(object)ownedParamShard;
        if (!pf.IsContiguous || pf._storageOffset != 0 || pf._storage.Length != pf.Length)
        {
            return;
        }

        if (pf.HasPendingGpuData || gpuEngine.IsDeviceAuthoritative(pf))
        {
            return;
        }

        if (pf._gpuBuffer is { } existingBuf &&
            existingBuf.Handle != IntPtr.Zero &&
            ReferenceEquals(pf._gpuBackend, gpuBackend) &&
            pf.GetLiveBackingArrayOrNull() is { } hostArr &&
            hostArr.Length <= existingBuf.Size)
        {
            gpuBackend.UploadBufferInPlace(hostArr, existingBuf);
            gpuEngine.CommitParameterUpdatedOnDevice(pf, existingBuf, gpuBackend, syncPoint: null);
            return;
        }

        if (gpuEngine.AcquireParameterBuffer(pf, out _) is { } acquiredBuf)
        {
            gpuEngine.CommitParameterUpdatedOnDevice(pf, acquiredBuf, gpuBackend, syncPoint: null);
        }
    }

    private void BindHostSynchronizedGpuGradient(
        DirectGpuTensorEngine gpuEngine,
        IDirectGpuBackend gpuBackend,
        Tensor<float> keyOwner,
        Tensor<float> gradFloat,
        bool isShardGrad)
    {
        if (_gpuBufferBackend is { } existingBackend && !ReferenceEquals(existingBackend, gpuBackend))
        {
            ReleasePersistentGpuBuffers();
        }

        _gpuBufferBackend = gpuBackend;

        if (gradFloat.TryGetGpuBuffer() is not null &&
            ReferenceEquals(gradFloat._gpuBackend, gpuBackend) &&
            gradFloat._gpuBufferVersion == gradFloat.Version)
        {
            return;
        }

        if (!gradFloat.IsContiguous ||
            gradFloat._storageOffset != 0 ||
            gradFloat._storage.Length != gradFloat.Length ||
            gradFloat.HasPendingGpuData ||
            gradFloat.GetLiveBackingArrayOrNull() is not { } hostArr ||
            hostArr.Length != gradFloat.Length)
        {
            return;
        }

        IGpuBuffer persistentBuf;
        if (isShardGrad)
        {
            if (_shardGradGpuBinding is { } existingShard &&
                existingShard.Buffer.Handle != IntPtr.Zero &&
                existingShard.Buffer.Size == gradFloat.Length)
            {
                persistentBuf = existingShard.Buffer;
            }
            else
            {
                if (_shardGradGpuBinding is { } oldShard)
                {
                    DetachAndDisposeGpuBinding(oldShard.Buffer, oldShard.BoundTensor);
                }

                persistentBuf = gpuBackend.AllocateBuffer(gradFloat.Length);
            }

            _shardGradGpuBinding = (persistentBuf, gradFloat);
        }
        else
        {
            _persistentGradBuffers ??= new Dictionary<Tensor<float>, (IGpuBuffer Buffer, Tensor<float> BoundTensor)>(
                TensorReferenceComparer<Tensor<float>>.Instance);

            if (_persistentGradBuffers.TryGetValue(keyOwner, out var existingEntry) &&
                existingEntry.Buffer.Handle != IntPtr.Zero &&
                existingEntry.Buffer.Size == gradFloat.Length)
            {
                persistentBuf = existingEntry.Buffer;
            }
            else
            {
                if (existingEntry.Buffer is not null)
                {
                    DetachAndDisposeGpuBinding(existingEntry.Buffer, existingEntry.BoundTensor);
                }

                persistentBuf = gpuBackend.AllocateBuffer(gradFloat.Length);
            }

            _persistentGradBuffers[keyOwner] = (persistentBuf, gradFloat);
        }

        gpuBackend.UploadBufferInPlace(hostArr, persistentBuf);
        gpuEngine.BindResidentBuffer(gradFloat, persistentBuf, gpuBackend);

        // Both hostArr and persistentBuf hold the identical reduced gradient; clear the pending D2H flag
        // armed by BindResidentBuffer while preserving ResidentHostMirror metadata so host reads (such as
        // GlobalGradientSquaredNorm or PublishParameterGradients) do not re-download 50 MB over PCIe.
        AiDotNet.Tensors.Helpers.HostSync.Remove(hostArr);
        gradFloat.Device = AiDotNet.Tensors.TensorDevice.CPU;
    }

    private static void DetachAndDisposeGpuBinding(IGpuBuffer buffer, Tensor<float>? boundTensor)
    {
        if (boundTensor is not null)
        {
            if (boundTensor.GetBackingArrayForCacheLookupUnsafe() is { } backing)
            {
                AiDotNet.Tensors.Helpers.HostSync.Remove(backing);
            }

            AiDotNet.Tensors.Helpers.HostSync.Remove(boundTensor.DataVector);
            if (ReferenceEquals(boundTensor._gpuBuffer, buffer))
            {
                boundTensor._gpuBuffer = null;
                boundTensor._gpuBackend = null;
                boundTensor._gpuBufferVersion = -1;
                boundTensor.Device = AiDotNet.Tensors.TensorDevice.CPU;
            }
        }

        if (buffer.Handle != IntPtr.Zero)
        {
            buffer.Dispose();
        }
    }

    private void ReleasePersistentGpuBuffers()
    {
        if (_shardGradGpuBinding is { } shardBinding)
        {
            DetachAndDisposeGpuBinding(shardBinding.Buffer, shardBinding.BoundTensor);
            _shardGradGpuBinding = null;
        }

        if (_persistentGradBuffers is not null)
        {
            foreach (var kvp in _persistentGradBuffers)
            {
                DetachAndDisposeGpuBinding(kvp.Value.Buffer, kvp.Value.BoundTensor);
            }

            _persistentGradBuffers.Clear();
            _persistentGradBuffers = null;
        }

        _gpuBufferBackend = null;
    }

    private static void PrepareTensorForFullHostOverwrite(Tensor<T> tensor, DirectGpuTensorEngine? gpuEngine)
    {
        if (tensor._storageOffset == 0 && tensor._storage.Length == tensor.Length)
        {
            if (tensor.GetBackingArrayForCacheLookupUnsafe() is { } backing)
            {
                AiDotNet.Tensors.Helpers.HostSync.Remove(backing);
            }

            AiDotNet.Tensors.Helpers.HostSync.Remove(tensor.DataVector);
        }

        gpuEngine?.InvalidateResidentWeightBuffer(tensor);
    }

    private static void PackParameterSliceIntoShard(
        Tensor<T>[] uniqueParams,
        int shardStart,
        int shardLength,
        Tensor<T> destinationShard)
    {
        if (destinationShard._storageOffset == 0 && destinationShard._storage.Length == destinationShard.Length)
        {
            if (destinationShard.GetBackingArrayForCacheLookupUnsafe() is { } shardBacking)
            {
                AiDotNet.Tensors.Helpers.HostSync.Remove(shardBacking);
            }

            AiDotNet.Tensors.Helpers.HostSync.Remove(destinationShard.DataVector);
            destinationShard.Device = AiDotNet.Tensors.TensorDevice.CPU;
        }

        Span<T> dstSpan = destinationShard.AsWritableSpan();
        int shardEnd = shardStart + shardLength;
        int paramOffset = 0;

        for (int i = 0; i < uniqueParams.Length; i++)
        {
            var param = uniqueParams[i];
            int paramLen = param.Length;
            int paramEnd = paramOffset + paramLen;

            int overlapStart = Math.Max(paramOffset, shardStart);
            int overlapEnd = Math.Min(paramEnd, shardEnd);
            if (overlapEnd > overlapStart)
            {
                int count = overlapEnd - overlapStart;
                param.AsSpan()
                    .Slice(overlapStart - paramOffset, count)
                    .CopyTo(dstSpan.Slice(overlapStart - shardStart, count));
            }

            paramOffset = paramEnd;
        }

        destinationShard.IncrementVersion();
        if (typeof(T) != typeof(float) && AiDotNetEngine.Current is DirectGpuTensorEngine gpuEngine)
        {
            gpuEngine.InvalidateResidentWeightBuffer(destinationShard);
        }
    }

    private static void PackGradientSliceIntoShard(
        Tensor<T>[] uniqueParams,
        Dictionary<Tensor<T>, Tensor<T>> gradients,
        int shardStart,
        int shardLength,
        Tensor<T> destinationShard)
    {
        var gradGpuEngine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        PrepareTensorForFullHostOverwrite(destinationShard, gradGpuEngine);

        Span<T> dstSpan = destinationShard.AsWritableSpan();
        int shardEnd = shardStart + shardLength;
        int paramOffset = 0;

        for (int i = 0; i < uniqueParams.Length; i++)
        {
            var param = uniqueParams[i];
            int paramLen = param.Length;
            int paramEnd = paramOffset + paramLen;

            int overlapStart = Math.Max(paramOffset, shardStart);
            int overlapEnd = Math.Min(paramEnd, shardEnd);
            if (overlapEnd > overlapStart)
            {
                int count = overlapEnd - overlapStart;
                Span<T> targetSlice = dstSpan.Slice(overlapStart - shardStart, count);
                if (gradients.TryGetValue(param, out var grad) && grad is not null && grad.Length == paramLen)
                {
                    grad.AsSpan().Slice(overlapStart - paramOffset, count).CopyTo(targetSlice);
                }
                else
                {
                    targetSlice.Clear();
                }
            }

            paramOffset = paramEnd;
        }

        destinationShard.IncrementVersion();
    }

    private static void UnpackVectorIntoParameters(
        ReadOnlySpan<T> flatSource,
        Tensor<T>[] uniqueParams,
        bool preserveGpuWeightBuffers)
    {
        var gpuEngine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var gpuBackend = typeof(T) == typeof(float) ? gpuEngine?.GetBackend() : null;
        int offset = 0;
        for (int i = 0; i < uniqueParams.Length; i++)
        {
            var param = uniqueParams[i];
            int len = param.Length;

            if (gpuEngine is not null &&
                gpuBackend is not null &&
                (object)param is Tensor<float> pf &&
                pf.IsContiguous &&
                pf._storageOffset == 0 &&
                pf._storage.Length == pf.Length &&
                !DirectGpuTensorEngine.IsFlatParameterView(pf))
            {
                IGpuBuffer? existingPBuf = pf._gpuBuffer is { } b && b.Handle != IntPtr.Zero && ReferenceEquals(pf._gpuBackend, gpuBackend)
                    ? b
                    : (preserveGpuWeightBuffers ? gpuEngine.AcquireParameterBuffer(pf, out _) : null);

                if (pf.GetBackingArrayForCacheLookupUnsafe() is { } pArr)
                {
                    AiDotNet.Tensors.Helpers.HostSync.Remove(pArr);
                }

                AiDotNet.Tensors.Helpers.HostSync.Remove(pf.DataVector);
                pf.Device = AiDotNet.Tensors.TensorDevice.CPU;

                flatSource.Slice(offset, len).CopyTo(param.AsWritableSpan());

                if (existingPBuf is not null &&
                    existingPBuf.Handle != IntPtr.Zero &&
                    pf.GetLiveBackingArrayOrNull() is { } pHostArr &&
                    pHostArr.Length <= existingPBuf.Size)
                {
                    gpuBackend.UploadBufferInPlace(pHostArr, existingPBuf);
                    gpuEngine.CommitParameterUpdatedOnDevice(pf, existingPBuf, gpuBackend, syncPoint: null);
                    offset += len;
                    continue;
                }
            }

            PrepareTensorForFullHostOverwrite(param, gpuEngine);
            flatSource.Slice(offset, len).CopyTo(param.AsWritableSpan());
            param.IncrementVersion();
            offset += len;
        }
    }

    private Tensor<T>[] GetOrDeduplicateParameters(IReadOnlyList<Tensor<T>> parameters, out int totalElements)
    {
        var cachedInputs = _cachedInputParameters;
        var cachedUnique = _cachedUniqueParameters;
        var cachedLengths = _cachedUniqueParameterLengths;
        if (cachedInputs is not null &&
            cachedUnique is not null &&
            cachedLengths is not null &&
            parameters.Count == cachedInputs.Length)
        {
            bool match = true;
            for (int i = 0; i < cachedInputs.Length; i++)
            {
                if (!ReferenceEquals(parameters[i], cachedInputs[i]))
                {
                    match = false;
                    break;
                }
            }

            if (match)
            {
                for (int u = 0; u < cachedUnique.Length; u++)
                {
                    if (cachedUnique[u].Length != cachedLengths[u])
                    {
                        match = false;
                        break;
                    }
                }

                if (match)
                {
                    totalElements = _cachedTotalElements;
                    return cachedUnique;
                }
            }
        }

        var unique = DeduplicateParameters(parameters, out totalElements);
        var inputSnapshot = new Tensor<T>?[parameters.Count];
        for (int i = 0; i < parameters.Count; i++)
        {
            inputSnapshot[i] = parameters[i];
        }

        var uniqueLengths = new int[unique.Length];
        for (int u = 0; u < unique.Length; u++)
        {
            uniqueLengths[u] = unique[u].Length;
        }

        _cachedInputParameters = inputSnapshot;
        _cachedUniqueParameters = unique;
        _cachedUniqueParameterLengths = uniqueLengths;
        _cachedTotalElements = totalElements;
        return unique;
    }

    private static Tensor<T>[] DeduplicateParameters(IReadOnlyList<Tensor<T>> parameters, out int totalElements)
    {
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        var unique = new List<Tensor<T>>(parameters.Count);
        long sum = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            var p = parameters[i];
            if (p is not null && p.Length > 0 && seen.Add(p))
            {
                unique.Add(p);
                sum = checked(sum + p.Length);
            }
        }

        if (sum > int.MaxValue)
        {
            throw new OverflowException(
                $"Total trainable parameter element count ({sum}) exceeds single-vector indexing capacity ({int.MaxValue}).");
        }

        totalElements = (int)sum;
        return unique.ToArray();
    }

    private static bool MatchesSnapshot(Tensor<T>[]? snapshot, Tensor<T>[] current)
    {
        if (snapshot is null || snapshot.Length != current.Length)
        {
            return false;
        }

        if (ReferenceEquals(snapshot, current))
        {
            return true;
        }

        for (int i = 0; i < snapshot.Length; i++)
        {
            if (!ReferenceEquals(snapshot[i], current[i]) || snapshot[i].Length != current[i].Length)
            {
                return false;
            }
        }

        return true;
    }

    private static IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? UnwrapGradientOptimizer(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
    {
        var current = optimizer;
        while (current is IShardedOptimizer<T, Tensor<T>, Tensor<T>> sharded &&
               sharded.WrappedOptimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> inner)
        {
            current = inner;
        }

        return current;
    }

    private void EnsureBackendInitialized()
    {
        if (!Backend.IsInitialized)
        {
            Backend.Initialize();
        }
    }

    private void FinalizeStepDiagnostics()
    {
        long endTimestamp = Stopwatch.GetTimestamp();
        double totalMs = _stepStartTimestamp > 0L
            ? (endTimestamp - _stepStartTimestamp) * 1000.0 / Stopwatch.Frequency
            : 0.0;

        LastStepTotalMilliseconds = totalMs;
        LastStepCommunicationMilliseconds = _stepCommMilliseconds;
        LastStepExposedCommunicationMilliseconds = _stepExposedCommMilliseconds;
        LastStepCommunicationShare = totalMs > 0.0 ? _stepCommMilliseconds / totalMs : 0.0;

        _stepStartTimestamp = 0L;
        _stepMetadataRecorded = false;
        _backwardStepBegun = false;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DistributedTapeCoordinator<T>));
        }
    }

    /// <summary>
    /// Disposes owned bucket reducer buffers and releases temporary shard resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _reducer?.Dispose();
        _reducer = null;
        ReleasePersistentGpuBuffers();
        lock (_slotSync)
        {
            _persistentGradSlots.Clear();
            _slotDownloadScratch = null;
        }

        _ownedParamShard = null;
        _ownedParamShardList = null;
        _shardGradTensor = null;
        _shardGradDict = null;
        _shardRegDict = null;
        _lastShardedParamsRef = null;
        _lastShardedParamVersions = null;
        _lastShardedParamGpuVersions = null;
        _cachedInputParameters = null;
        _cachedUniqueParameters = null;
        _cachedUniqueParameterLengths = null;
        _gatherSendBuffer = null;
        _syncedParamSnapshot = null;
    }
}
