using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;

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
public sealed class GradientBucketReducer<T> : IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ICommunicationBackend<T> _backend;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly int _bucketSizeBytes;
    private readonly bool _overlapCommunication;
    private readonly bool _useReduceScatter;
    private readonly Tensor<T>[] _uniqueParameters;
    private readonly int _totalElements;
    private readonly int _shardSize;
    private readonly Bucket[] _buckets;
    private readonly Dictionary<Tensor<T>, ParamSlot> _paramLocations;
    private readonly object _syncLock = new object();

    private double _rankWeight = 1.0;
    private int _nextBucketToDispatch;
    private bool _disposed;

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

        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        var unique = new List<Tensor<T>>(parameters.Count);
        long totalElementsLong = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            var param = parameters[i];
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
                DisposeBucketHandles(bucket);
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
                return;
            }

            if (!bucket.SlotReady[slot.EntryIndex])
            {
                WriteScaledToVectorSlice(gradient, bucket.Buffer, slot.Offset, slot.Length, _rankWeight);
                bucket.SlotReady[slot.EntryIndex] = true;
                bucket.ReadyCount++;
            }
            else
            {
                AccumulateScaledToVectorSlice(gradient, bucket.Buffer, slot.Offset, slot.Length, _rankWeight);
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
            reducedLocalShard = _useReduceScatter ? new Vector<T>(0) : null;
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
                    if (_rankWeight != 0.0 &&
                        gradients.TryGetValue(entry.Parameter, out var grad) &&
                        grad is not null)
                    {
                        if (grad.Length != entry.Length)
                        {
                            throw new ArgumentException(
                                $"Gradient length ({grad.Length}) does not match parameter length ({entry.Length}).",
                                nameof(gradients));
                        }

                        WriteScaledToVectorSlice(grad, bucket.Buffer, entry.Offset, entry.Length, _rankWeight);
                    }
                    else
                    {
                        bucket.Buffer.AsWritableSpan().Slice(entry.Offset, entry.Length).Clear();
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
                reducedLocalShard = ExtractVectorSlice(_buckets[0].Buffer, 0, _totalElements);
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
                    var resultHandle = bucket.ResultHandle
                        ?? throw new InvalidOperationException("ReduceScatter bucket was marked dispatched without a result handle.");
                    try
                    {
                        scatterShard = resultHandle.GetResult();
                        totalCollectiveMs += resultHandle.Elapsed.TotalMilliseconds;
                    }
                    finally
                    {
                        resultHandle.Dispose();
                        bucket.ResultHandle = null;
                    }
                }
                else
                {
                    var handle = bucket.Handle
                        ?? throw new InvalidOperationException("AllReduce bucket was marked dispatched without a collective handle.");
                    try
                    {
                        handle.Wait();
                        totalCollectiveMs += handle.Elapsed.TotalMilliseconds;
                    }
                    finally
                    {
                        handle.Dispose();
                        bucket.Handle = null;
                    }
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
                reducedLocalShard = new Vector<T>(0);
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
                    destTensor.Length != entry.Length)
                {
                    destTensor = new Tensor<T>(entry.Parameter._shape);
                    gradients[entry.Parameter] = destTensor;
                }

                bucketSpan.Slice(entry.Offset, entry.Length).CopyTo(destTensor.AsWritableSpan());
                destTensor.IncrementVersion();
                gpuEngine?.InvalidateResidentWeightBuffer(destTensor);
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

    private static void WriteScaledToVectorSlice(
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

        if (scale == 0.0)
        {
            // Explicitly zero rather than multiplying by 0.0 so an idle rank whose dummy forward produced
            // NaN/Inf gradients never contributes IEEE 754 (0.0 * NaN == NaN) into AllReduce/ReduceScatter.
            destination.AsWritableSpan().Slice(offset, length).Clear();
            return;
        }

        if (scale == 1.0)
        {
            source.AsSpan().Slice(0, length).CopyTo(destination.AsWritableSpan().Slice(offset, length));
            return;
        }

        if ((object)source is Tensor<float> srcFloat && (object)destination is Vector<float> dstFloat)
        {
            float factor = (float)scale;
            ReadOnlySpan<float> srcSpan = srcFloat.AsSpan().Slice(0, length);
            Span<float> dstSpan = dstFloat.AsWritableSpan().Slice(offset, length);
            for (int i = 0; i < length; i++)
            {
                dstSpan[i] = srcSpan[i] * factor;
            }

            return;
        }

        if ((object)source is Tensor<double> srcDouble && (object)destination is Vector<double> dstDouble)
        {
            ReadOnlySpan<double> srcSpan = srcDouble.AsSpan().Slice(0, length);
            Span<double> dstSpan = dstDouble.AsWritableSpan().Slice(offset, length);
            for (int i = 0; i < length; i++)
            {
                dstSpan[i] = srcSpan[i] * scale;
            }

            return;
        }

        WriteScaledGenericSpan(
            source.AsSpan().Slice(0, length),
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

    private static void AccumulateScaledToVectorSlice(
        Tensor<T> source,
        Vector<T> destination,
        int offset,
        int length,
        double scale)
    {
        if (length <= 0 || scale == 0.0)
        {
            return;
        }

        if ((object)source is Tensor<float> srcFloat && (object)destination is Vector<float> dstFloat)
        {
            float factor = (float)scale;
            ReadOnlySpan<float> srcSpan = srcFloat.AsSpan().Slice(0, length);
            Span<float> dstSpan = dstFloat.AsWritableSpan().Slice(offset, length);
            for (int i = 0; i < length; i++)
            {
                dstSpan[i] += srcSpan[i] * factor;
            }

            return;
        }

        if ((object)source is Tensor<double> srcDouble && (object)destination is Vector<double> dstDouble)
        {
            ReadOnlySpan<double> srcSpan = srcDouble.AsSpan().Slice(0, length);
            Span<double> dstSpan = dstDouble.AsWritableSpan().Slice(offset, length);
            for (int i = 0; i < length; i++)
            {
                dstSpan[i] += srcSpan[i] * scale;
            }

            return;
        }

        AccumulateScaledGenericSpan(
            source.AsSpan().Slice(0, length),
            destination.AsWritableSpan().Slice(offset, length),
            scale);
    }

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

    private static void DisposeBucketHandles(Bucket bucket)
    {
        if (bucket.Handle is not null)
        {
            bucket.Handle.Dispose();
            bucket.Handle = null;
        }

        if (bucket.ResultHandle is not null)
        {
            bucket.ResultHandle.Dispose();
            bucket.ResultHandle = null;
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
                DisposeBucketHandles(_buckets[b]);
            }
        }
    }
}
