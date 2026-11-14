using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
#if !AIDN_LEGACY_TFM
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;
#endif

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Partitions parameter gradients into fixed-capacity communication buckets, scales each rank's local gradients
/// by its token-weighted contribution, and overlaps asynchronous <c>AllReduce</c> or <c>ReduceScatter</c> collectives
/// with backward propagation.
/// </summary>
/// <remarks>
/// <para>
/// In data-parallel (<see cref="DistributedStrategy.DDP"/>) and optimizer-sharded (<see cref="DistributedStrategy.ZeRO1"/>)
/// modes (<c>useReduceScatter == false</c>), parameters are grouped in reverse parameter-list order so output-layer
/// gradients—which finish earliest during reverse-mode automatic differentiation—fill bucket 0 first and immediately
/// launch <see cref="IAsyncCommunicationBackend{T}.AllReduceAsync"/> while earlier layers are still differentiating.
/// </para>
/// <para>
/// In gradient-sharded (<see cref="DistributedStrategy.ZeRO2"/> and <see cref="DistributedStrategy.FSDP"/>) modes
/// (<c>useReduceScatter == true</c>), parameters are packed into canonical global parameter order
/// <c>[P_0 || P_1 || ... || P_{K-1}]</c> padded to <c>WorldSize * ceil(totalElements / WorldSize)</c> so that
/// <see cref="IAsyncCommunicationBackend{T}.ReduceScatterAsync"/> delivers Rank <c>r</c>'s exact contiguous owned
/// gradient shard <c>[r * shardSize .. (r + 1) * shardSize)</c>.
/// </para>
/// <para><b>For Beginners:</b> Instead of sending thousands of tiny network messages (one per weight tensor) or waiting
/// until the entire backward pass finishes before starting network communication, <see cref="GradientBucketReducer{T}"/>
/// packs gradients into ~25 MiB buckets and sends each bucket across ranks in the background as soon as its gradients
/// are ready.</para>
/// </remarks>
/// <example>
/// <code>
/// using var reducer = new GradientBucketReducer&lt;float&gt;(backend, trainableParameters);
/// reducer.BeginStep(rankWeight: 0.5);
/// foreach (var kvp in localGradients)
/// {
///     reducer.NotifyGradientReady(kvp.Key, kvp.Value);
/// }
/// reducer.FlushAndSynchronize(localGradients, out _);
/// </code>
/// </example>
/// <typeparam name="T">The numeric element type.</typeparam>
internal sealed class GradientBucketReducer<T> : IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ICommunicationBackend<T> _backend;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly int _bucketSizeBytes;
    private readonly bool _overlapCommunication;
    private readonly bool _useReduceScatter;
    private readonly Tensor<T>?[] _inputParameterSnapshot;
    private readonly Tensor<T>[] _uniqueParameters;
    private readonly int[] _uniqueParameterLengths;
    private readonly int _totalElements;
    private readonly int _shardSize;
    private readonly Bucket[] _buckets;
    private readonly Dictionary<Tensor<T>, ParamSlot> _paramLocations;
    private readonly Vector<T>? _emptyScatterShard;
    private readonly object _syncLock = new object();

    private Dictionary<Tensor<T>, Tensor<T>>? _ownedUnpackGradients;
    private float[]? _gpuDownloadScratch;
    private double _rankWeight = 1.0;
    private int _nextBucketToDispatch;
    private bool _disposed;

    internal Func<Tensor<T>, Tensor<T>>? GradientSlotProvider { get; set; }

    private readonly struct BucketEntry
    {
        public BucketEntry(Tensor<T> parameter, int offset, int length)
        {
            Parameter = parameter;
            Offset = offset;
            Length = length;
        }

        public Tensor<T> Parameter { get; }
        public int Offset { get; }
        public int Length { get; }
    }

    private readonly struct ParamSlot
    {
        public ParamSlot(int bucketIndex, int entryIndex, int offset, int length)
        {
            BucketIndex = bucketIndex;
            EntryIndex = entryIndex;
            Offset = offset;
            Length = length;
        }

        public int BucketIndex { get; }
        public int EntryIndex { get; }
        public int Offset { get; }
        public int Length { get; }
    }

    private sealed class Bucket
    {
        public Bucket(BucketEntry[] entries, int bufferLength)
        {
            Entries = entries;
            Buffer = new Vector<T>(bufferLength);
            SlotReady = new bool[entries.Length];
        }

        public BucketEntry[] Entries { get; }
        public Vector<T> Buffer { get; }
        public bool[] SlotReady { get; }
        public int ReadyCount { get; set; }
        public bool Dispatched { get; set; }
        public IAsyncCollectiveHandle? Handle { get; set; }
        public IAsyncCollectiveResultHandle<T>? ResultHandle { get; set; }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GradientBucketReducer{T}"/> class.
    /// </summary>
    /// <param name="backend">The distributed communication backend.</param>
    /// <param name="parameters">The ordered list of trainable model parameters.</param>
    /// <param name="bucketSizeBytes">Target bucket capacity in bytes for <c>AllReduce</c> bucketing (default: 25 MiB).</param>
    /// <param name="overlapCommunication">Whether to dispatch asynchronous collectives during backward propagation when <paramref name="backend"/> implements <see cref="IAsyncCommunicationBackend{T}"/>.</param>
    /// <param name="useReduceScatter">When <c>true</c>, packs parameters in canonical forward order padded to <c>WorldSize * shardSize</c> and executes <c>ReduceScatter(Sum)</c>; otherwise partitions in reverse order and executes <c>AllReduce(Sum)</c>.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="backend"/> or <paramref name="parameters"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="bucketSizeBytes"/> is not positive.</exception>
    public GradientBucketReducer(
        ICommunicationBackend<T> backend,
        IReadOnlyList<Tensor<T>> parameters,
        int bucketSizeBytes = 25 * 1024 * 1024,
        bool overlapCommunication = true,
        bool useReduceScatter = false)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        if (bucketSizeBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(bucketSizeBytes),
                bucketSizeBytes,
                "Gradient bucket size in bytes must be greater than zero.");
        }

        _worldSize = Math.Max(1, backend.WorldSize);
        _rank = backend.Rank;
        _bucketSizeBytes = bucketSizeBytes;
        _overlapCommunication = overlapCommunication;
        _useReduceScatter = useReduceScatter;
        _emptyScatterShard = useReduceScatter ? new Vector<T>(0) : null;

        _inputParameterSnapshot = new Tensor<T>?[parameters.Count];
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        var unique = new List<Tensor<T>>(parameters.Count);
        long totalElementsLong = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            var param = parameters[i];
            _inputParameterSnapshot[i] = param;
            if (param is not null && param.Length > 0 && seen.Add(param))
            {
                unique.Add(param);
                totalElementsLong = checked(totalElementsLong + param.Length);
            }
        }

        if (totalElementsLong > int.MaxValue)
        {
            throw new OverflowException(
                $"Total trainable parameter count ({totalElementsLong}) exceeds single-vector indexing capacity ({int.MaxValue}).");
        }

        _uniqueParameters = unique.ToArray();
        _uniqueParameterLengths = new int[_uniqueParameters.Length];
        for (int i = 0; i < _uniqueParameters.Length; i++)
        {
            _uniqueParameterLengths[i] = _uniqueParameters[i].Length;
        }
        _totalElements = (int)totalElementsLong;
        _shardSize = _totalElements > 0
            ? (int)(((long)_totalElements + _worldSize - 1L) / _worldSize)
            : 0;

        _paramLocations = new Dictionary<Tensor<T>, ParamSlot>(
            _uniqueParameters.Length,
            TensorReferenceComparer<Tensor<T>>.Instance);

        if (_uniqueParameters.Length == 0)
        {
            _buckets = Array.Empty<Bucket>();
            return;
        }

        if (_useReduceScatter)
        {
            long paddedLong = checked((long)_worldSize * _shardSize);
            if (paddedLong > int.MaxValue)
            {
                throw new OverflowException(
                    $"Padded ReduceScatter buffer length ({paddedLong}) exceeds maximum vector length ({int.MaxValue}).");
            }

            var entries = new BucketEntry[_uniqueParameters.Length];
            int offset = 0;
            for (int i = 0; i < _uniqueParameters.Length; i++)
            {
                var param = _uniqueParameters[i];
                int len = param.Length;
                entries[i] = new BucketEntry(param, offset, len);
                _paramLocations[param] = new ParamSlot(0, i, offset, len);
                offset += len;
            }

            _buckets = new[] { new Bucket(entries, (int)paddedLong) };
        }
        else
        {
            int elementBytes = GetElementSizeBytes();
            long maxElementsPerBucket = Math.Max(1L, bucketSizeBytes / elementBytes);

            var bucketList = new List<Bucket>();
            var currentEntries = new List<BucketEntry>();
            int currentOffset = 0;

            for (int i = _uniqueParameters.Length - 1; i >= 0; i--)
            {
                var param = _uniqueParameters[i];
                int len = param.Length;

                if (currentEntries.Count > 0 && (long)currentOffset + len > maxElementsPerBucket)
                {
                    AddAllReduceBucket(bucketList, currentEntries, currentOffset);
                    currentEntries = new List<BucketEntry>();
                    currentOffset = 0;
                }

                currentEntries.Add(new BucketEntry(param, currentOffset, len));
                currentOffset = checked(currentOffset + len);
            }

            if (currentEntries.Count > 0)
            {
                AddAllReduceBucket(bucketList, currentEntries, currentOffset);
            }

            _buckets = bucketList.ToArray();
        }
    }

    private void AddAllReduceBucket(List<Bucket> bucketList, List<BucketEntry> entries, int bufferLength)
    {
        int bucketIndex = bucketList.Count;
        var entryArray = entries.ToArray();
        for (int s = 0; s < entryArray.Length; s++)
        {
            var e = entryArray[s];
            _paramLocations[e.Parameter] = new ParamSlot(bucketIndex, s, e.Offset, e.Length);
        }

        bucketList.Add(new Bucket(entryArray, bufferLength));
    }

    /// <summary>
    /// Gets the number of communication buckets constructed for the parameter set.
    /// </summary>
    public int BucketCount => _buckets.Length;

    /// <summary>
    /// Gets the raw wall-clock milliseconds spent blocked inside <see cref="FlushAndSynchronize"/> waiting for
    /// collective handles (or executing non-overlapped synchronous collectives) after backward propagation completed.
    /// </summary>
    public double LastExposedWaitMilliseconds { get; private set; }

    /// <summary>
    /// Gets the raw cumulative milliseconds spent executing transport collectives across all buckets during the
    /// most recent step (including overlapped background collectives).
    /// </summary>
    public double LastTotalCollectiveMilliseconds { get; private set; }

    /// <summary>
    /// Checks whether this reducer instance matches the supplied parameter list and configuration so the coordinator
    /// can reuse bucket staging buffers across steps without reallocation.
    /// </summary>
    internal bool MatchesConfiguration(
        IReadOnlyList<Tensor<T>> parameters,
        int bucketSizeBytes,
        bool overlapCommunication,
        bool useReduceScatter)
    {
        if (_disposed ||
            _bucketSizeBytes != bucketSizeBytes ||
            _overlapCommunication != overlapCommunication ||
            _useReduceScatter != useReduceScatter)
        {
            return false;
        }

        if (parameters.Count == _inputParameterSnapshot.Length)
        {
            bool exactSnapshotMatch = true;
            for (int i = 0; i < _inputParameterSnapshot.Length; i++)
            {
                if (!ReferenceEquals(parameters[i], _inputParameterSnapshot[i]))
                {
                    exactSnapshotMatch = false;
                    break;
                }
            }

            if (exactSnapshotMatch)
            {
                for (int u = 0; u < _uniqueParameters.Length; u++)
                {
                    if (_uniqueParameters[u].Length != _uniqueParameterLengths[u])
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        int uniqueIdx = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            var p = parameters[i];
            if (p is null || p.Length <= 0 || !seen.Add(p))
            {
                continue;
            }

            if (uniqueIdx >= _uniqueParameters.Length ||
                !ReferenceEquals(_uniqueParameters[uniqueIdx], p) ||
                _uniqueParameters[uniqueIdx].Length != p.Length)
            {
                return false;
            }

            uniqueIdx++;
        }

        return uniqueIdx == _uniqueParameters.Length;
    }

    /// <summary>
    /// Resets bucket readiness state for a new backward step and sets this rank's token-weighted gradient scale factor.
    /// </summary>
    /// <param name="rankWeight">This rank's contribution weight (<c>localTokens / globalTokens</c>, or <c>0.0</c> for an idle tail-batch rank).</param>
    /// <exception cref="ObjectDisposedException">Thrown if this reducer has been disposed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="rankWeight"/> is <c>NaN</c> or infinite.</exception>
    public void BeginStep(double rankWeight)
    {
        ThrowIfDisposed();
        if (double.IsNaN(rankWeight) || double.IsInfinity(rankWeight))
        {
            throw new ArgumentOutOfRangeException(
                nameof(rankWeight),
                rankWeight,
                "Rank weight must be a finite number.");
        }

        lock (_syncLock)
        {
            _rankWeight = rankWeight;
            _nextBucketToDispatch = 0;
            LastExposedWaitMilliseconds = 0.0;
            LastTotalCollectiveMilliseconds = 0.0;

            for (int b = 0; b < _buckets.Length; b++)
            {
                var bucket = _buckets[b];
                WaitAndDisposeBucketHandles(bucket);
                bucket.ReadyCount = 0;
                bucket.Dispatched = false;
                bucket.SlotReady.AsSpan().Clear();

                if (_useReduceScatter && _totalElements < bucket.Buffer.Length)
                {
                    bucket.Buffer.AsWritableSpan().Slice(_totalElements).Clear();
                }
            }
        }
    }

    /// <summary>
    /// Records that <paramref name="parameter"/>'s local gradient has finished computing, scales it by the step's
    /// <c>rankWeight</c> into its staging bucket, and dispatches any consecutive ready buckets asynchronously when
    /// communication overlap is enabled.
    /// </summary>
    /// <param name="parameter">The trainable parameter tensor.</param>
    /// <param name="gradient">The local unscaled gradient computed for <paramref name="parameter"/>.</param>
    /// <exception cref="ObjectDisposedException">Thrown if this reducer has been disposed.</exception>
    /// <exception cref="ArgumentException">Thrown if <paramref name="gradient"/> length does not match <paramref name="parameter"/> length.</exception>
    public void NotifyGradientReady(Tensor<T> parameter, Tensor<T> gradient)
    {
        ThrowIfDisposed();
        if (parameter is null || gradient is null)
        {
            return;
        }

        if (!_paramLocations.TryGetValue(parameter, out ParamSlot slot))
        {
            return;
        }

        if (gradient.Length != slot.Length)
        {
            throw new ArgumentException(
                $"Gradient length ({gradient.Length}) does not match parameter length ({slot.Length}).",
                nameof(gradient));
        }

        lock (_syncLock)
        {
            var bucket = _buckets[slot.BucketIndex];
            if (bucket.Dispatched)
            {
                throw new InvalidOperationException(
                    "NotifyGradientReady received a duplicate gradient notification for a parameter whose communication bucket has already been dispatched. " +
                    "Disable OverlapCommunication when a model contains shared/tied parameter references or recomputed subgraphs that invoke backward hooks multiple times for the same parameter.");
            }

            if (!bucket.SlotReady[slot.EntryIndex])
            {
                WriteScaledToVectorSlice(parameter, gradient, bucket.Buffer, slot.Offset, slot.Length, _rankWeight);
                bucket.SlotReady[slot.EntryIndex] = true;
                bucket.ReadyCount++;
            }
            else
            {
                AccumulateScaledToVectorSlice(parameter, gradient, bucket.Buffer, slot.Offset, slot.Length, _rankWeight);
            }

            if (_worldSize > 1 && _overlapCommunication && _backend is IAsyncCommunicationBackend<T> asyncBackend)
            {
                if (!_backend.IsInitialized)
                {
                    _backend.Initialize();
                }

                while (_nextBucketToDispatch < _buckets.Length)
                {
                    var candidate = _buckets[_nextBucketToDispatch];
                    if (candidate.ReadyCount < candidate.Entries.Length)
                    {
                        break;
                    }

                    DispatchBucketAsync(asyncBackend, candidate);
                    _nextBucketToDispatch++;
                }
            }
        }
    }

    /// <summary>
    /// Packs any remaining gradients not yet delivered via <see cref="NotifyGradientReady"/> (filling missing
    /// parameters or <c>rankWeight == 0</c> ranks with zeros so all ranks participate in every collective without
    /// hanging), dispatches remaining collectives, waits for completion in strict bucket order, and unpacks the
    /// reduced gradients.
    /// </summary>
    /// <param name="gradients">Dictionary of parameter gradients; populated in-place with reduced gradients when <c>useReduceScatter == false</c>.</param>
    /// <param name="reducedLocalShard">When <c>useReduceScatter == true</c>, receives this rank's contiguous reduced gradient shard of length <c>myValidLen</c>; otherwise <c>null</c>.</param>
    /// <exception cref="ObjectDisposedException">Thrown if this reducer has been disposed.</exception>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="gradients"/> is null.</exception>
    public void FlushAndSynchronize(IDictionary<Tensor<T>, Tensor<T>> gradients, out Vector<T>? reducedLocalShard)
    {
        ThrowIfDisposed();
        if (gradients is null)
        {
            throw new ArgumentNullException(nameof(gradients));
        }

        if (_buckets.Length == 0)
        {
            LastExposedWaitMilliseconds = 0.0;
            LastTotalCollectiveMilliseconds = 0.0;
            reducedLocalShard = _useReduceScatter ? (_emptyScatterShard ?? new Vector<T>(0)) : null;
            return;
        }

        lock (_syncLock)
        {
            for (int b = 0; b < _buckets.Length; b++)
            {
                var bucket = _buckets[b];
                if (bucket.Dispatched)
                {
                    continue;
                }

                for (int s = 0; s < bucket.Entries.Length; s++)
                {
                    if (bucket.SlotReady[s])
                    {
                        continue;
                    }

                    var entry = bucket.Entries[s];
                    if (!IsZeroScale(_rankWeight) &&
                        gradients.TryGetValue(entry.Parameter, out var grad) &&
                        grad is not null)
                    {
                        if (grad.Length != entry.Length)
                        {
                            throw new ArgumentException(
                                $"Gradient length ({grad.Length}) does not match parameter length ({entry.Length}).",
                                nameof(gradients));
                        }

                        WriteScaledToVectorSlice(entry.Parameter, grad, bucket.Buffer, entry.Offset, entry.Length, _rankWeight);
                    }
                    else
                    {
                        bucket.Buffer.AsWritableSpan().Slice(entry.Offset, entry.Length).Clear();
                        if (_useReduceScatter && GradientSlotProvider is not null)
                        {
                            ClearSlotForReduceScatterLocked(entry.Parameter, entry.Length);
                        }
                    }

                    bucket.SlotReady[s] = true;
                    bucket.ReadyCount++;
                }

                if (_useReduceScatter && _totalElements < bucket.Buffer.Length)
                {
                    bucket.Buffer.AsWritableSpan().Slice(_totalElements).Clear();
                }
            }
        }

        if (_worldSize <= 1)
        {
            LastExposedWaitMilliseconds = 0.0;
            LastTotalCollectiveMilliseconds = 0.0;

            if (_useReduceScatter)
            {
                reducedLocalShard = _buckets[0].Buffer.Length == _totalElements
                    ? _buckets[0].Buffer
                    : ExtractVectorSlice(_buckets[0].Buffer, 0, _totalElements);
            }
            else
            {
                reducedLocalShard = null;
                UnpackAllReduceBuckets(gradients);
            }

            return;
        }

        if (!_backend.IsInitialized)
        {
            _backend.Initialize();
        }

        long exposedStart = Stopwatch.GetTimestamp();
        double totalCollectiveMs = 0.0;
        Vector<T>? scatterShard = null;

        if (_overlapCommunication && _backend is IAsyncCommunicationBackend<T> asyncBackend)
        {
            lock (_syncLock)
            {
                while (_nextBucketToDispatch < _buckets.Length)
                {
                    DispatchBucketAsync(asyncBackend, _buckets[_nextBucketToDispatch]);
                    _nextBucketToDispatch++;
                }
            }

            for (int b = 0; b < _buckets.Length; b++)
            {
                var bucket = _buckets[b];
                if (_useReduceScatter)
                {
                    using var resultHandle = bucket.ResultHandle
                        ?? throw new InvalidOperationException("ReduceScatter bucket was marked dispatched without a result handle.");
                    bucket.ResultHandle = null;
                    scatterShard = resultHandle.GetResult();
                    totalCollectiveMs += resultHandle.Elapsed.TotalMilliseconds;
                }
                else
                {
                    using var handle = bucket.Handle
                        ?? throw new InvalidOperationException("AllReduce bucket was marked dispatched without a collective handle.");
                    bucket.Handle = null;
                    handle.Wait();
                    totalCollectiveMs += handle.Elapsed.TotalMilliseconds;
                }
            }

            long exposedEnd = Stopwatch.GetTimestamp();
            LastExposedWaitMilliseconds = (exposedEnd - exposedStart) * 1000.0 / Stopwatch.Frequency;
            LastTotalCollectiveMilliseconds = totalCollectiveMs;
        }
        else
        {
            long syncCollectiveTicks = 0L;
            for (int b = 0; b < _buckets.Length; b++)
            {
                var bucket = _buckets[b];
                bucket.Dispatched = true;

                long t0 = Stopwatch.GetTimestamp();
                if (_useReduceScatter)
                {
                    scatterShard = _backend.ReduceScatter(bucket.Buffer, ReductionOperation.Sum);
                }
                else
                {
                    _backend.AllReduce(bucket.Buffer, ReductionOperation.Sum);
                }

                long t1 = Stopwatch.GetTimestamp();
                syncCollectiveTicks += t1 - t0;
            }

            long exposedEnd = Stopwatch.GetTimestamp();
            LastExposedWaitMilliseconds = (exposedEnd - exposedStart) * 1000.0 / Stopwatch.Frequency;
            LastTotalCollectiveMilliseconds = syncCollectiveTicks * 1000.0 / Stopwatch.Frequency;
        }

        if (_useReduceScatter)
        {
            int myStart = _rank * _shardSize;
            int myValidLen = Math.Max(0, Math.Min(_shardSize, _totalElements - myStart));
            if (scatterShard is null || myValidLen == 0)
            {
                reducedLocalShard = _emptyScatterShard ?? new Vector<T>(0);
            }
            else if (scatterShard.Length == myValidLen)
            {
                reducedLocalShard = scatterShard;
            }
            else
            {
                reducedLocalShard = ExtractVectorSlice(scatterShard, 0, myValidLen);
            }
        }
        else
        {
            reducedLocalShard = null;
            UnpackAllReduceBuckets(gradients);
        }
    }

    private void DispatchBucketAsync(IAsyncCommunicationBackend<T> asyncBackend, Bucket bucket)
    {
        if (bucket.Dispatched)
        {
            return;
        }

        bucket.Dispatched = true;
        if (_useReduceScatter)
        {
            bucket.ResultHandle = asyncBackend.ReduceScatterAsync(bucket.Buffer, ReductionOperation.Sum);
        }
        else
        {
            bucket.Handle = asyncBackend.AllReduceAsync(bucket.Buffer, ReductionOperation.Sum);
        }
    }

    private Tensor<T> GetOrCreateSlotLocked(Tensor<T> parameter)
    {
        if (GradientSlotProvider is { } provider)
        {
            return provider(parameter);
        }

        _ownedUnpackGradients ??= new Dictionary<Tensor<T>, Tensor<T>>(
            _uniqueParameters.Length,
            TensorReferenceComparer<Tensor<T>>.Instance);

        if (!_ownedUnpackGradients.TryGetValue(parameter, out var slot) ||
            slot is null ||
            slot.Length != parameter.Length)
        {
            slot = new Tensor<T>(parameter._shape);
            _ownedUnpackGradients[parameter] = slot;
        }

        return slot;
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

    private void UnpackAllReduceBuckets(IDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        var gpuEngine = AiDotNetEngine.Current as DirectGpuTensorEngine;

        for (int b = 0; b < _buckets.Length; b++)
        {
            var bucket = _buckets[b];
            ReadOnlySpan<T> bucketSpan = bucket.Buffer.AsSpan();
            var entries = bucket.Entries;

            for (int s = 0; s < entries.Length; s++)
            {
                var entry = entries[s];
                if (!gradients.TryGetValue(entry.Parameter, out var destTensor) ||
                    destTensor is null ||
                    destTensor.Length != entry.Length ||
                    destTensor.DataVector.IsLazyAllocated)
                {
                    lock (_syncLock)
                    {
                        destTensor = GetOrCreateSlotLocked(entry.Parameter);
                    }

                    gradients[entry.Parameter] = destTensor;
                }

                PrepareTensorForFullHostOverwrite(destTensor, gpuEngine);
                bucketSpan.Slice(entry.Offset, entry.Length).CopyTo(destTensor.AsWritableSpan());
                destTensor.IncrementVersion();
            }
        }
    }

    private static Vector<T> ExtractVectorSlice(Vector<T> source, int offset, int length)
    {
        if (length <= 0)
        {
            return new Vector<T>(0);
        }

        var result = new Vector<T>(length);
        source.AsSpan().Slice(offset, length).CopyTo(result.AsWritableSpan());
        return result;
    }

    private void ClearSlotForReduceScatterLocked(Tensor<T> parameter, int length)
    {
        var slot = GetOrCreateSlotLocked(parameter);
        PrepareTensorForFullHostOverwrite(slot, AiDotNetEngine.Current as DirectGpuTensorEngine);
        int clearLen = Math.Min(length, slot.Length);
        if (clearLen > 0)
        {
            slot.AsWritableSpan().Slice(0, clearLen).Clear();
        }
    }

    private ReadOnlySpan<float> GetFloatSourceSpanLocked(
        Tensor<T> parameter,
        Tensor<float> srcFloat,
        int length,
        bool allowDirectSlotWrite,
        out bool downloadedDirectlyIntoSlot)
    {
        downloadedDirectlyIntoSlot = false;
        if (srcFloat.HasPendingGpuData &&
            srcFloat._gpuBuffer is { } srcGpuBuf &&
            srcGpuBuf.Handle != IntPtr.Zero &&
            srcFloat._gpuBackend is { } gpuBack &&
            srcFloat._gpuBufferVersion == srcFloat.Version &&
            srcFloat._storageOffset == 0 &&
            srcFloat.IsContiguous)
        {
            int requiredSize = Math.Max(length, srcGpuBuf.Size);
            if (allowDirectSlotWrite &&
                (object)GetOrCreateSlotLocked(parameter) is Tensor<float> slotFloat &&
                slotFloat.IsContiguous &&
                slotFloat._storageOffset == 0 &&
                slotFloat._storage.Length == slotFloat.Length &&
                !ReferenceEquals(slotFloat, srcFloat))
            {
                if (slotFloat.GetBackingArrayForCacheLookupUnsafe() is { } slotBacking)
                {
                    AiDotNet.Tensors.Helpers.HostSync.Remove(slotBacking);
                }

                AiDotNet.Tensors.Helpers.HostSync.Remove(slotFloat.DataVector);
                slotFloat.Device = AiDotNet.Tensors.TensorDevice.CPU;
                float[] slotArr = slotFloat.GetDataArray();
                if (slotArr.Length >= requiredSize)
                {
                    gpuBack.DownloadBuffer(srcGpuBuf, slotArr);
                    downloadedDirectlyIntoSlot = true;
                    return slotArr.AsSpan(0, length);
                }
            }

            if (_gpuDownloadScratch is null || _gpuDownloadScratch.Length < requiredSize)
            {
                _gpuDownloadScratch = new float[requiredSize];
            }

            gpuBack.DownloadBuffer(srcGpuBuf, _gpuDownloadScratch);
            return _gpuDownloadScratch.AsSpan(0, length);
        }

        return srcFloat.AsSpan().Slice(0, length);
    }

    private void WriteScaledToVectorSlice(
        Tensor<T> parameter,
        Tensor<T> source,
        Vector<T> destination,
        int offset,
        int length,
        double scale)
    {
        if (length <= 0)
        {
            return;
        }

        if (IsZeroScale(scale))
        {
            // Explicitly zero rather than multiplying by 0.0 so an idle rank whose dummy forward produced
            // NaN/Inf gradients never contributes IEEE 754 (0.0 * NaN == NaN) into AllReduce/ReduceScatter.
            destination.AsWritableSpan().Slice(offset, length).Clear();
            if (_useReduceScatter && GradientSlotProvider is not null)
            {
                ClearSlotForReduceScatterLocked(parameter, length);
            }
            return;
        }

        if ((object)source is Tensor<float> srcFloat && (object)destination is Vector<float> dstFloat)
        {
            ReadOnlySpan<float> srcSpan = GetFloatSourceSpanLocked(
                parameter,
                srcFloat,
                length,
                allowDirectSlotWrite: true,
                out bool downloadedDirectlyIntoSlot);
            Span<float> dstSpan = dstFloat.AsWritableSpan().Slice(offset, length);
            if (IsUnitScale(scale))
            {
                srcSpan.CopyTo(dstSpan);
            }
            else
            {
                float factor = (float)scale;
#if !AIDN_LEGACY_TFM
                TensorPrimitives.Multiply(srcSpan, factor, dstSpan);
#else
                WriteScaledFloatSimdLegacy(srcSpan, factor, dstSpan);
#endif
            }

            if (_useReduceScatter && GradientSlotProvider is not null)
            {
                var slot = GetOrCreateSlotLocked(parameter);
                var gpuEngine = AiDotNetEngine.Current as DirectGpuTensorEngine;
                if (downloadedDirectlyIntoSlot)
                {
                    gpuEngine?.InvalidateResidentWeightBuffer(slot);
                    slot.IncrementVersion();
                }
                else if (!ReferenceEquals(slot, source) && (object)slot is Tensor<float> slotFloat)
                {
                    PrepareTensorForFullHostOverwrite(slot, gpuEngine);
                    srcSpan.CopyTo(slotFloat.AsWritableSpan().Slice(0, Math.Min(length, slotFloat.Length)));
                }
            }
            return;
        }

        ReadOnlySpan<T> hostSrcSpan = source.AsSpan().Slice(0, length);
        if (_useReduceScatter && GradientSlotProvider is not null)
        {
            var slot = GetOrCreateSlotLocked(parameter);
            if (!ReferenceEquals(slot, source))
            {
                PrepareTensorForFullHostOverwrite(slot, null);
                hostSrcSpan.CopyTo(slot.AsWritableSpan().Slice(0, Math.Min(length, slot.Length)));
            }
        }

        if (IsUnitScale(scale))
        {
            hostSrcSpan.CopyTo(destination.AsWritableSpan().Slice(offset, length));
            return;
        }

        if ((object)source is Tensor<double> srcDouble && (object)destination is Vector<double> dstDouble)
        {
            ReadOnlySpan<double> srcSpan = srcDouble.AsSpan().Slice(0, length);
            Span<double> dstSpan = dstDouble.AsWritableSpan().Slice(offset, length);
#if !AIDN_LEGACY_TFM
            TensorPrimitives.Multiply(srcSpan, scale, dstSpan);
#else
            WriteScaledDoubleSimdLegacy(srcSpan, scale, dstSpan);
#endif
            return;
        }

        WriteScaledGenericSpan(
            hostSrcSpan,
            destination.AsWritableSpan().Slice(offset, length),
            scale);
    }

    [AiDotNet.Attributes.ScalarKernel("Generic T span scaling fallback after float and double fast paths.")]
    private static void WriteScaledGenericSpan(ReadOnlySpan<T> source, Span<T> destination, double scale)
    {
        T factor = NumOps.FromDouble(scale);
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = NumOps.Multiply(source[i], factor);
        }
    }

    private void AccumulateScaledToVectorSlice(
        Tensor<T> parameter,
        Tensor<T> source,
        Vector<T> destination,
        int offset,
        int length,
        double scale)
    {
        if (length <= 0 || IsZeroScale(scale))
        {
            return;
        }

        if ((object)source is Tensor<float> srcFloat && (object)destination is Vector<float> dstFloat)
        {
            float factor = (float)scale;
            ReadOnlySpan<float> srcSpan = GetFloatSourceSpanLocked(
                parameter,
                srcFloat,
                length,
                allowDirectSlotWrite: false,
                out _);
            Span<float> dstSpan = dstFloat.AsWritableSpan().Slice(offset, length);
#if !AIDN_LEGACY_TFM
            TensorPrimitives.MultiplyAdd(srcSpan, factor, dstSpan, dstSpan);
#else
            AccumulateScaledFloatSimdLegacy(srcSpan, factor, dstSpan);
#endif
            if (_useReduceScatter && GradientSlotProvider is not null)
            {
                var slot = GetOrCreateSlotLocked(parameter);
                if (!ReferenceEquals(slot, source) && (object)slot is Tensor<float> slotFloat)
                {
                    int slotLen = Math.Min(length, slotFloat.Length);
                    Span<float> slotSpan = slotFloat.AsWritableSpan().Slice(0, slotLen);
#if !AIDN_LEGACY_TFM
                    TensorPrimitives.Add(slotSpan, srcSpan.Slice(0, slotLen), slotSpan);
#else
                    AccumulateScaledFloatSimdLegacy(srcSpan.Slice(0, slotLen), 1.0f, slotSpan);
#endif
                }
            }
            return;
        }

        if ((object)source is Tensor<double> srcDouble && (object)destination is Vector<double> dstDouble)
        {
            ReadOnlySpan<double> srcSpan = srcDouble.AsSpan().Slice(0, length);
            Span<double> dstSpan = dstDouble.AsWritableSpan().Slice(offset, length);
#if !AIDN_LEGACY_TFM
            TensorPrimitives.MultiplyAdd(srcSpan, scale, dstSpan, dstSpan);
#else
            AccumulateScaledDoubleSimdLegacy(srcSpan, scale, dstSpan);
#endif
            if (_useReduceScatter && GradientSlotProvider is not null)
            {
                var slot = GetOrCreateSlotLocked(parameter);
                if (!ReferenceEquals(slot, source) && (object)slot is Tensor<double> slotDouble)
                {
                    int slotLen = Math.Min(length, slotDouble.Length);
                    Span<double> slotSpan = slotDouble.AsWritableSpan().Slice(0, slotLen);
#if !AIDN_LEGACY_TFM
                    TensorPrimitives.Add(slotSpan, srcSpan.Slice(0, slotLen), slotSpan);
#else
                    AccumulateScaledDoubleSimdLegacy(srcSpan.Slice(0, slotLen), 1.0, slotSpan);
#endif
                }
            }
            return;
        }

        ReadOnlySpan<T> genericSrc = source.AsSpan().Slice(0, length);
        AccumulateScaledGenericSpan(
            genericSrc,
            destination.AsWritableSpan().Slice(offset, length),
            scale);
        if (_useReduceScatter && GradientSlotProvider is not null)
        {
            var slot = GetOrCreateSlotLocked(parameter);
            if (!ReferenceEquals(slot, source))
            {
                int slotLen = Math.Min(length, slot.Length);
                AccumulateScaledGenericSpan(genericSrc.Slice(0, slotLen), slot.AsWritableSpan().Slice(0, slotLen), 1.0);
            }
        }
    }

#if AIDN_LEGACY_TFM
    private static void WriteScaledFloatSimdLegacy(ReadOnlySpan<float> srcSpan, float factor, Span<float> dstSpan)
    {
        int length = srcSpan.Length;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<float>.Count)
        {
            int width = System.Numerics.Vector<float>.Count;
            int simdLen = length - (length % width);
            var factorVec = new System.Numerics.Vector<float>(factor);
            var srcVecs = MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(srcSpan.Slice(0, simdLen));
            var dstVecs = MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(dstSpan.Slice(0, simdLen));
            for (int v = 0; v < srcVecs.Length; v++)
            {
                dstVecs[v] = srcVecs[v] * factorVec;
            }

            i = simdLen;
        }

        for (; i < length; i++)
        {
            dstSpan[i] = srcSpan[i] * factor;
        }
    }

    private static void WriteScaledDoubleSimdLegacy(ReadOnlySpan<double> srcSpan, double factor, Span<double> dstSpan)
    {
        int length = srcSpan.Length;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<double>.Count)
        {
            int width = System.Numerics.Vector<double>.Count;
            int simdLen = length - (length % width);
            var factorVec = new System.Numerics.Vector<double>(factor);
            var srcVecs = MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(srcSpan.Slice(0, simdLen));
            var dstVecs = MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(dstSpan.Slice(0, simdLen));
            for (int v = 0; v < srcVecs.Length; v++)
            {
                dstVecs[v] = srcVecs[v] * factorVec;
            }

            i = simdLen;
        }

        for (; i < length; i++)
        {
            dstSpan[i] = srcSpan[i] * factor;
        }
    }

    private static void AccumulateScaledFloatSimdLegacy(ReadOnlySpan<float> srcSpan, float factor, Span<float> dstSpan)
    {
        int length = srcSpan.Length;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<float>.Count)
        {
            int width = System.Numerics.Vector<float>.Count;
            int simdLen = length - (length % width);
            var factorVec = new System.Numerics.Vector<float>(factor);
            var srcVecs = MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(srcSpan.Slice(0, simdLen));
            var dstVecs = MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(dstSpan.Slice(0, simdLen));
            for (int v = 0; v < srcVecs.Length; v++)
            {
                dstVecs[v] += srcVecs[v] * factorVec;
            }

            i = simdLen;
        }

        for (; i < length; i++)
        {
            dstSpan[i] += srcSpan[i] * factor;
        }
    }

    private static void AccumulateScaledDoubleSimdLegacy(ReadOnlySpan<double> srcSpan, double factor, Span<double> dstSpan)
    {
        int length = srcSpan.Length;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && length >= System.Numerics.Vector<double>.Count)
        {
            int width = System.Numerics.Vector<double>.Count;
            int simdLen = length - (length % width);
            var factorVec = new System.Numerics.Vector<double>(factor);
            var srcVecs = MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(srcSpan.Slice(0, simdLen));
            var dstVecs = MemoryMarshal.Cast<double, System.Numerics.Vector<double>>(dstSpan.Slice(0, simdLen));
            for (int v = 0; v < srcVecs.Length; v++)
            {
                dstVecs[v] += srcVecs[v] * factorVec;
            }

            i = simdLen;
        }

        for (; i < length; i++)
        {
            dstSpan[i] += srcSpan[i] * factor;
        }
    }
#endif

    [AiDotNet.Attributes.ScalarKernel("Generic T span scaled-accumulation fallback after float and double fast paths.")]
    private static void AccumulateScaledGenericSpan(ReadOnlySpan<T> source, Span<T> destination, double scale)
    {
        T factor = NumOps.FromDouble(scale);
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = NumOps.Add(destination[i], NumOps.Multiply(source[i], factor));
        }
    }

    private static int GetElementSizeBytes()
    {
        if (typeof(T) == typeof(float) || typeof(T) == typeof(int) || typeof(T) == typeof(uint))
        {
            return 4;
        }

        if (typeof(T) == typeof(double) || typeof(T) == typeof(long) || typeof(T) == typeof(ulong))
        {
            return 8;
        }

        if (typeof(T) == typeof(decimal))
        {
            return 16;
        }

        if (typeof(T) == typeof(short) || typeof(T) == typeof(ushort))
        {
            return 2;
        }

#if !AIDN_LEGACY_TFM
        if (typeof(T) == typeof(Half))
        {
            return 2;
        }
#endif

        if (typeof(T) == typeof(byte) || typeof(T) == typeof(sbyte))
        {
            return 1;
        }

        return 8;
    }

    private static bool IsZeroScale(double scale) => !(Math.Abs(scale) > 0.0);

    private static bool IsUnitScale(double scale) => !(scale < 1.0 || scale > 1.0);

    private static void WaitAndDisposeBucketHandles(Bucket bucket)
    {
        if (bucket.Handle is { } handle)
        {
            bucket.Handle = null;
            try
            {
                handle.Wait();
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Debug.WriteLine($"GradientBucketReducer handle wait during cleanup faulted: {ex.Message}");
            }
            finally
            {
                handle.Dispose();
            }
        }

        if (bucket.ResultHandle is { } resultHandle)
        {
            bucket.ResultHandle = null;
            try
            {
                _ = resultHandle.GetResult();
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Debug.WriteLine($"GradientBucketReducer result-handle wait during cleanup faulted: {ex.Message}");
            }
            finally
            {
                resultHandle.Dispose();
            }
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientBucketReducer<T>));
        }
    }

    /// <summary>
    /// Releases any pending asynchronous collective handles owned by this reducer.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        lock (_syncLock)
        {
            _disposed = true;
            for (int b = 0; b < _buckets.Length; b++)
            {
                WaitAndDisposeBucketHandles(_buckets[b]);
            }

            _ownedUnpackGradients?.Clear();
            _ownedUnpackGradients = null;
            _gpuDownloadScratch = null;
            GradientSlotProvider = null;
        }
    }
}
