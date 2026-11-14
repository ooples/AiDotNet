using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides a high-throughput, low-latency inter-process shared-memory communication backend for multi-GPU
/// training on a single machine (specifically designed for Windows where native NCCL is unavailable, with
/// cross-platform POSIX shared-file memory mapping on Linux and macOS).
/// </summary>
/// <remarks>
/// <para><b>Architecture:</b>
/// Processes on the same host map a common shared-memory region (<see cref="MemoryMappedFile"/>) partitioned into:
/// <list type="bullet">
/// <item><description>Cache-line-aligned (64-byte) per-rank sense-reversing phase headers to avoid false sharing.</description></item>
/// <item><description>Dedicated per-rank collective staging slots supporting pipelined multi-chunk <c>AllReduce</c>,
/// <c>ReduceScatter</c>, <c>AllGather</c> (both uniform and uneven shard sizes), <c>Broadcast</c>, and <c>Scatter</c>.</description></item>
/// <item><description>Dedicated per-pair point-to-point mailboxes for tagged <c>Send</c>/<c>Receive</c>.</description></item>
/// <item><description>OS process-ID liveness monitoring so if any peer rank crashes or is killed mid-run, surviving
/// ranks fail fast with an <see cref="IOException"/> instead of hanging indefinitely.</description></item>
/// </list>
/// </para>
/// <para><b>Determinism:</b>
/// Reductions combine peer contributions in strict canonical rank order (<c>0, 1, ..., WorldSize - 1</c>) on every
/// rank, guaranteeing bit-identical floating-point results across all participating processes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric element type.</typeparam>
public sealed class SharedMemoryCommunicationBackend<T> : CommunicationBackendBase<T>, IDisposable
{
    private const int MagicValue = 0x4149444E; // "AIDN"
    private const int ProtocolVersion = 1;
    private const int MaxSupportedRanks = 64;
    private const int GlobalHeaderSize = 64;
    private const int RankHeaderStride = 64;
    private const int HeaderBytes = GlobalHeaderSize + (MaxSupportedRanks * RankHeaderStride); // 4160 bytes -> padded to 8192
    private const int CollectiveRegionOffset = 8192;
    private const int P2PHeaderStride = 64;
    private const int P2PChunkCapacityBytes = 256 * 1024; // 256 KiB per ordered (src, dst) pair

    private const int OpInit = 1;
    private const int OpBarrier = 2;
    private const int OpAllReduceData = 3;
    private const int OpAllReduceAck = 4;
    private const int OpAllGatherMeta = 5;
    private const int OpAllGatherMetaAck = 6;
    private const int OpAllGatherData = 7;
    private const int OpAllGatherAck = 8;
    private const int OpBroadcastMeta = 9;
    private const int OpBroadcastMetaAck = 10;
    private const int OpBroadcastData = 11;
    private const int OpBroadcastAck = 12;
    private const int OpScatterMeta = 13;
    private const int OpScatterMetaAck = 14;
    private const int OpScatterData = 15;
    private const int OpScatterAck = 16;
    private const int OpReduceScatterData = 17;
    private const int OpReduceScatterAck = 18;

    private const int RankStateUninitialized = 0;
    private const int RankStateActive = 1;
    private const int RankStateShutdown = 2;

    private readonly int _rank;
    private readonly int _worldSize;
    private readonly string _sessionName;
    private readonly long _slotCapacityBytes;
    private readonly int _timeoutMilliseconds;
    private readonly object _collectiveLock = new object();

    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private FileStream? _backingFileStream;
    private string? _backingFilePath;
    private EventWaitHandle? _wakeEvent;
    private Process?[]? _peerProcesses;
    private long _localPhase;
    private long _p2pBaseOffset;
    private long[]? _p2pSendSeq;
    private long[]? _p2pRecvSeq;
    private bool _disposed;

    /// <summary>
    /// Initializes a new <see cref="SharedMemoryCommunicationBackend{T}"/> configured from environment variables
    /// (<c>AIDOTNET_RANK</c> or <c>RANK</c>, <c>AIDOTNET_WORLD_SIZE</c> or <c>WORLD_SIZE</c>, and
    /// <c>AIDOTNET_SHM_NAME</c>).
    /// </summary>
    public SharedMemoryCommunicationBackend()
        : this(
            ReadEnvInt("AIDOTNET_RANK", "RANK", defaultValue: 0),
            ReadEnvInt("AIDOTNET_WORLD_SIZE", "WORLD_SIZE", defaultValue: 1),
            Environment.GetEnvironmentVariable("AIDOTNET_SHM_NAME") ?? "aidotnet_shm_default")
    {
    }

    /// <summary>
    /// Initializes a new <see cref="SharedMemoryCommunicationBackend{T}"/> for a specific rank and world size.
    /// </summary>
    /// <param name="rank">This process's rank in <c>[0, worldSize - 1]</c>.</param>
    /// <param name="worldSize">Total number of participating processes (<c>1..64</c>).</param>
    /// <param name="sessionName">Unique shared-memory session name shared by all ranks in the job.</param>
    /// <param name="slotCapacityBytes">Per-rank collective staging buffer capacity in bytes (default 16 MiB).</param>
    /// <param name="timeoutMilliseconds">Maximum wait time in milliseconds before throwing a timeout error.</param>
    public SharedMemoryCommunicationBackend(
        int rank,
        int worldSize,
        string sessionName = "aidotnet_shm_default",
        long slotCapacityBytes = 16L * 1024L * 1024L,
        int timeoutMilliseconds = 120_000)
    {
        if (worldSize < 1 || worldSize > MaxSupportedRanks)
        {
            throw new ArgumentOutOfRangeException(
                nameof(worldSize),
                $"WorldSize must be between 1 and {MaxSupportedRanks}, got {worldSize}.");
        }

        if (rank < 0 || rank >= worldSize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(rank),
                $"Rank must be between 0 and {worldSize - 1}, got {rank}.");
        }

        if (string.IsNullOrWhiteSpace(sessionName))
        {
            throw new ArgumentException("Session name cannot be null or whitespace.", nameof(sessionName));
        }

        if (slotCapacityBytes < 4096)
        {
            throw new ArgumentOutOfRangeException(nameof(slotCapacityBytes), "Slot capacity must be at least 4096 bytes.");
        }

        if (timeoutMilliseconds <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(timeoutMilliseconds), "Timeout must be positive.");
        }

        _rank = rank;
        _worldSize = worldSize;
        _sessionName = SanitizeSessionName(sessionName);
        _slotCapacityBytes = AlignUp(slotCapacityBytes, 64);
        _timeoutMilliseconds = timeoutMilliseconds;
    }

    /// <inheritdoc/>
    public override int Rank => _rank;

    /// <inheritdoc/>
    public override int WorldSize => _worldSize;

    /// <summary>
    /// Gets the sanitized shared-memory session identifier used by this backend.
    /// </summary>
    public string SessionName => _sessionName;

    /// <inheritdoc/>
    protected override void OnInitialize()
    {
        if (_worldSize == 1)
        {
            _localPhase = 0;
            return;
        }

        long pairCount = (long)_worldSize * _worldSize;
        long p2pSlotStride = P2PHeaderStride + P2PChunkCapacityBytes;
        _p2pBaseOffset = CollectiveRegionOffset + (_worldSize * _slotCapacityBytes);
        long totalCapacityBytes = _p2pBaseOffset + (pairCount * p2pSlotStride);

#if AIDN_LEGACY_TFM
        if (Environment.OSVersion.Platform == PlatformID.Win32NT)
        {
            InitializeWindowsSharedMemory(totalCapacityBytes);
        }
        else
        {
            InitializeFileBackedSharedMemory(totalCapacityBytes);
        }
#else
        if (OperatingSystem.IsWindows())
        {
            InitializeWindowsSharedMemory(totalCapacityBytes);
        }
        else
        {
            InitializeFileBackedSharedMemory(totalCapacityBytes);
        }
#endif

        var accessor = _accessor ?? throw new InvalidOperationException("Shared-memory accessor failed to initialize.");
        _p2pSendSeq = new long[_worldSize];
        _p2pRecvSeq = new long[_worldSize];
        _peerProcesses = new Process?[_worldSize];
        _localPhase = 0;

        int currentPid = Process.GetCurrentProcess().Id;
        long myHeader = RankHeaderOffset(_rank);

        // Register this rank's metadata in its dedicated cache line before publishing active state.
        accessor.Write(myHeader + 4, currentPid);
        accessor.Write(myHeader + 8, 0L);
        accessor.Write(myHeader + 16, 0);
        accessor.Write(myHeader + 20, 0);
        accessor.Write(myHeader + 24, Stopwatch.GetTimestamp());
        accessor.Write(myHeader + 32, 0);
        accessor.Write(myHeader + 40, 0);
        accessor.Write(myHeader + 44, 0);
        Thread.MemoryBarrier();
        accessor.Write(myHeader + 0, RankStateActive);
        Thread.MemoryBarrier();

        if (_rank == 0)
        {
            accessor.Write(4, _worldSize);
            accessor.Write(8, 0); // AbortFlag = 0
            Thread.MemoryBarrier();
            accessor.Write(0, (ProtocolVersion << 16) ^ MagicValue);
            Thread.MemoryBarrier();
        }

        SignalWakeEvent();

        // Wait for all ranks to register and bind peer Process handles for instant crash detection.
        long startTicks = Stopwatch.GetTimestamp();
        int expectedHeader = (ProtocolVersion << 16) ^ MagicValue;
        for (int peer = 0; peer < _worldSize; peer++)
        {
            long peerHeader = RankHeaderOffset(peer);
            int spins = 0;
            while (true)
            {
                Thread.MemoryBarrier();
                int hdrMagic = accessor.ReadInt32(0);
                int state = accessor.ReadInt32(peerHeader + 0);
                if (hdrMagic == expectedHeader && state == RankStateActive)
                {
                    int savedWorld = accessor.ReadInt32(4);
                    if (savedWorld != _worldSize)
                    {
                        throw new InvalidOperationException(
                            $"Shared-memory session '{_sessionName}' was initialized with WorldSize={savedWorld}, " +
                            $"but rank {_rank} configured WorldSize={_worldSize}.");
                    }

                    int peerPid = accessor.ReadInt32(peerHeader + 4);
                    if (peer != _rank && peerPid > 0 && peerPid != currentPid)
                    {
                        try
                        {
                            _peerProcesses[peer] = Process.GetProcessById(peerPid);
                        }
                        catch (ArgumentException)
                        {
                            throw new IOException(
                                $"Distributed peer rank {peer} (PID {peerPid}) exited during shared-memory handshake.");
                        }
                    }

                    break;
                }

                CheckTimeout(startTicks, $"initializing shared-memory session '{_sessionName}' waiting for rank {peer}");
                SpinOrYield(ref spins);
            }
        }

        // Initial barrier confirms every rank has bound peer PIDs before any collective begins.
        AdvancePhaseAndWait(OpInit, 0);
    }

    /// <inheritdoc/>
    protected override void OnShutdown()
    {
        if (_accessor is not null)
        {
            try
            {
                long myHeader = RankHeaderOffset(_rank);
                _accessor.Write(myHeader + 0, RankStateShutdown);
                Thread.MemoryBarrier();
                SignalWakeEvent();
            }
            catch (ObjectDisposedException)
            {
            }
        }

        ReleaseResources();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (IsInitialized)
        {
            Shutdown();
        }
        else
        {
            ReleaseResources();
        }
    }

    /// <inheritdoc/>
    public override void Barrier()
    {
        EnsureInitialized();
        DrainAsyncCollectives();
        long start = Stopwatch.GetTimestamp();
        if (_worldSize <= 1)
        {
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return;
        }

        lock (_collectiveLock)
        {
            AdvancePhaseAndWait(OpBarrier, 0);
        }

        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
    }

    /// <inheritdoc/>
    public override void AllReduce(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));
        DrainAsyncCollectives();

        long start = Stopwatch.GetTimestamp();
        if (_worldSize <= 1 || data.Length == 0)
        {
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return;
        }

        lock (_collectiveLock)
        {
            int elemSize = ElementSizeBytes();
            int maxChunkElements = (int)Math.Min(int.MaxValue, _slotCapacityBytes / elemSize);
            var span = data.AsWritableSpan();
            int offset = 0;
            while (offset < span.Length)
            {
                int count = Math.Min(maxChunkElements, span.Length - offset);
                WriteSpanToSlot(_rank, span.Slice(offset, count));
                AdvancePhaseAndWait(OpAllReduceData, count);
                ReduceSlotsIntoSpan(span.Slice(offset, count), count, operation);
                AdvancePhaseAndWait(OpAllReduceAck, count);
                offset += count;
            }
        }

        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
    }

    /// <inheritdoc/>
    public override Vector<T> AllGather(Vector<T> sendData)
    {
        EnsureInitialized();
        ValidateData(sendData, nameof(sendData));
        DrainAsyncCollectives();

        long start = Stopwatch.GetTimestamp();
        if (_worldSize <= 1)
        {
            var single = sendData.Clone();
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return single;
        }

        lock (_collectiveLock)
        {
            var accessor = _accessor!;
            accessor.Write(RankHeaderOffset(_rank) + 32, sendData.Length);
            AdvancePhaseAndWait(OpAllGatherMeta, 0);

            var counts = new int[_worldSize];
            var offsets = new int[_worldSize];
            long totalLenLong = 0;
            int maxPeerLen = 0;
            int invalidRank = -1;
            int invalidCount = 0;
            for (int r = 0; r < _worldSize; r++)
            {
                int c = accessor.ReadInt32(RankHeaderOffset(r) + 32);
                if (c < 0 && invalidRank < 0)
                {
                    invalidRank = r;
                    invalidCount = c;
                }

                counts[r] = c;
                offsets[r] = checked((int)totalLenLong);
                if (c > 0)
                {
                    totalLenLong += c;
                    if (c > maxPeerLen)
                    {
                        maxPeerLen = c;
                    }
                }
            }

            AdvancePhaseAndWait(OpAllGatherMetaAck, 0);

            if (invalidRank >= 0)
            {
                throw new InvalidOperationException($"Rank {invalidRank} reported negative AllGather count {invalidCount}.");
            }

            int totalLength = checked((int)totalLenLong);
            var gathered = new Vector<T>(totalLength);
            if (totalLength == 0)
            {
                RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
                return gathered;
            }

            int elemSize = ElementSizeBytes();
            int maxChunkElements = (int)Math.Min(int.MaxValue, _slotCapacityBytes / elemSize);
            var sendSpan = sendData.AsSpan();
            var dstSpan = gathered.AsWritableSpan();

            int cursor = 0;
            while (cursor < maxPeerLen)
            {
                int myChunk = cursor < sendSpan.Length
                    ? Math.Min(maxChunkElements, sendSpan.Length - cursor)
                    : 0;
                if (myChunk > 0)
                {
                    WriteReadOnlySpanToSlot(_rank, sendSpan.Slice(cursor, myChunk));
                }

                AdvancePhaseAndWait(OpAllGatherData, myChunk);

                for (int r = 0; r < _worldSize; r++)
                {
                    int peerTotal = counts[r];
                    if (cursor >= peerTotal)
                    {
                        continue;
                    }

                    int peerChunk = Math.Min(maxChunkElements, peerTotal - cursor);
                    ReadSlotIntoSpan(r, dstSpan.Slice(offsets[r] + cursor, peerChunk), peerChunk);
                }

                AdvancePhaseAndWait(OpAllGatherAck, 0);
                cursor += maxChunkElements;
            }

            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return gathered;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Broadcast(Vector<T> data, int root = 0)
    {
        EnsureInitialized();
        ValidateRoot(root);
        if (_rank == root)
        {
            ValidateData(data, nameof(data));
        }

        DrainAsyncCollectives();
        long start = Stopwatch.GetTimestamp();
        if (_worldSize <= 1)
        {
            var single = data.Clone();
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return single;
        }

        lock (_collectiveLock)
        {
            var accessor = _accessor!;
            int myCount = _rank == root ? data.Length : 0;
            accessor.Write(RankHeaderOffset(_rank) + 32, myCount);
            AdvancePhaseAndWait(OpBroadcastMeta, 0);

            int totalLength = accessor.ReadInt32(RankHeaderOffset(root) + 32);
            AdvancePhaseAndWait(OpBroadcastMetaAck, 0);

            if (totalLength < 0)
            {
                throw new InvalidOperationException($"Broadcast root rank {root} reported negative length {totalLength}.");
            }

            var result = new Vector<T>(totalLength);
            if (totalLength == 0)
            {
                RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
                return result;
            }

            if (_rank == root)
            {
                data.AsSpan().CopyTo(result.AsWritableSpan());
            }

            int elemSize = ElementSizeBytes();
            int maxChunkElements = (int)Math.Min(int.MaxValue, _slotCapacityBytes / elemSize);
            var dstSpan = result.AsWritableSpan();
            int offset = 0;
            while (offset < totalLength)
            {
                int count = Math.Min(maxChunkElements, totalLength - offset);
                if (_rank == root)
                {
                    WriteReadOnlySpanToSlot(root, dstSpan.Slice(offset, count));
                }

                AdvancePhaseAndWait(OpBroadcastData, count);

                if (_rank != root)
                {
                    ReadSlotIntoSpan(root, dstSpan.Slice(offset, count), count);
                }

                AdvancePhaseAndWait(OpBroadcastAck, count);
                offset += count;
            }

            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return result;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Scatter(Vector<T> sendData, int root = 0)
    {
        EnsureInitialized();
        ValidateRoot(root);
        if (_rank == root)
        {
            ValidateData(sendData, nameof(sendData));
            if (sendData.Length % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"Scatter input length ({sendData.Length}) must be divisible by WorldSize ({_worldSize}).",
                    nameof(sendData));
            }
        }

        DrainAsyncCollectives();
        long start = Stopwatch.GetTimestamp();
        if (_worldSize <= 1)
        {
            var single = sendData.Clone();
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return single;
        }

        lock (_collectiveLock)
        {
            var accessor = _accessor!;
            int myShardLength = _rank == root ? sendData.Length / _worldSize : 0;
            accessor.Write(RankHeaderOffset(_rank) + 32, myShardLength);
            AdvancePhaseAndWait(OpScatterMeta, 0);

            int shardLength = accessor.ReadInt32(RankHeaderOffset(root) + 32);
            AdvancePhaseAndWait(OpScatterMetaAck, 0);

            if (shardLength < 0)
            {
                throw new InvalidOperationException($"Scatter root rank {root} reported negative shard length {shardLength}.");
            }

            var result = new Vector<T>(shardLength);
            if (shardLength == 0)
            {
                RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
                return result;
            }

            int elemSize = ElementSizeBytes();
            int maxChunkElements = (int)Math.Min(int.MaxValue, _slotCapacityBytes / elemSize);
            var dstSpan = result.AsWritableSpan();
            int offset = 0;
            while (offset < shardLength)
            {
                int count = Math.Min(maxChunkElements, shardLength - offset);
                if (_rank == root)
                {
                    var rootSpan = sendData.AsSpan();
                    for (int r = 0; r < _worldSize; r++)
                    {
                        WriteReadOnlySpanToSlot(r, rootSpan.Slice((r * shardLength) + offset, count));
                    }
                }

                AdvancePhaseAndWait(OpScatterData, count);
                ReadSlotIntoSpan(_rank, dstSpan.Slice(offset, count), count);
                AdvancePhaseAndWait(OpScatterAck, count);
                offset += count;
            }

            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return result;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));
        if (data.Length % _worldSize != 0)
        {
            throw new ArgumentException(
                $"ReduceScatter input length ({data.Length}) must be divisible by WorldSize ({_worldSize}).",
                nameof(data));
        }

        DrainAsyncCollectives();
        long start = Stopwatch.GetTimestamp();
        if (_worldSize <= 1)
        {
            var single = data.Clone();
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return single;
        }

        int shardLength = data.Length / _worldSize;
        var shard = new Vector<T>(shardLength);
        if (shardLength == 0)
        {
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
            return shard;
        }

        lock (_collectiveLock)
        {
            int elemSize = ElementSizeBytes();
            int maxTotalChunkPerRank = (int)Math.Min(int.MaxValue, _slotCapacityBytes / (elemSize * _worldSize));
            if (maxTotalChunkPerRank < 1)
            {
                maxTotalChunkPerRank = 1;
            }

            var srcSpan = data.AsSpan();
            var dstSpan = shard.AsWritableSpan();
            int shardOffset = 0;
            while (shardOffset < shardLength)
            {
                int count = Math.Min(maxTotalChunkPerRank, shardLength - shardOffset);

                // Pack sub-slices [0..W-1] of length `count` contiguously into this rank's slot so each peer r
                // reads only sub-slice `r` from all slots (O(N/W) reduction work per rank instead of O(N)).
                for (int targetRank = 0; targetRank < _worldSize; targetRank++)
                {
                    long byteSubOffset = (long)targetRank * count * elemSize;
                    WriteReadOnlySpanToSlotOffset(
                        _rank,
                        byteSubOffset,
                        srcSpan.Slice((targetRank * shardLength) + shardOffset, count));
                }

                AdvancePhaseAndWait(OpReduceScatterData, count);

                long myByteSubOffset = (long)_rank * count * elemSize;
                ReduceSlotsAtSubOffsetIntoSpan(myByteSubOffset, dstSpan.Slice(shardOffset, count), count, operation);

                AdvancePhaseAndWait(OpReduceScatterAck, count);
                shardOffset += count;
            }
        }

        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
        return shard;
    }

    /// <inheritdoc/>
    public override void Send(Vector<T> data, int destinationRank, int tag = 0)
    {
        EnsureInitialized();
        ValidateRank(destinationRank, nameof(destinationRank));
        ValidateData(data, nameof(data));

        long start = Stopwatch.GetTimestamp();
        var accessor = _accessor!;
        int pairIdx = (_rank * _worldSize) + destinationRank;
        long pairOffset = _p2pBaseOffset + ((long)pairIdx * (P2PHeaderStride + P2PChunkCapacityBytes));
        long dataOffset = pairOffset + P2PHeaderStride;
        long nextSeq = (_p2pSendSeq![destinationRank] + 1);

        int elemSize = ElementSizeBytes();
        int maxChunkElements = P2PChunkCapacityBytes / elemSize;
        var span = data.AsSpan();
        int sent = 0;
        do
        {
            int chunk = Math.Min(maxChunkElements, span.Length - sent);
            long waitStart = Stopwatch.GetTimestamp();
            int spins = 0;
            while (true)
            {
                Thread.MemoryBarrier();
                long ack = accessor.ReadInt64(pairOffset + 8);
                if (ack == nextSeq - 1)
                {
                    break;
                }

                CheckPeerAlive(destinationRank);
                CheckTimeout(waitStart, $"Send from rank {_rank} to rank {destinationRank}");
                SpinOrYield(ref spins);
            }

            if (chunk > 0)
            {
                WriteReadOnlySpanToRawOffset(dataOffset, span.Slice(sent, chunk));
            }

            accessor.Write(pairOffset + 16, chunk);
            accessor.Write(pairOffset + 20, tag);
            accessor.Write(pairOffset + 24, span.Length);
            Thread.MemoryBarrier();
            accessor.Write(pairOffset + 0, nextSeq);
            Thread.MemoryBarrier();
            SignalWakeEvent();

            // Wait for receiver acknowledgment before advancing.
            waitStart = Stopwatch.GetTimestamp();
            spins = 0;
            while (true)
            {
                Thread.MemoryBarrier();
                long ack = accessor.ReadInt64(pairOffset + 8);
                if (ack == nextSeq)
                {
                    break;
                }

                CheckPeerAlive(destinationRank);
                CheckTimeout(waitStart, $"Send ack from rank {destinationRank} to rank {_rank}");
                SpinOrYield(ref spins);
            }

            _p2pSendSeq[destinationRank] = nextSeq;
            nextSeq++;
            sent += chunk;
        }
        while (sent < span.Length);

        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
    }

    /// <inheritdoc/>
    public override Vector<T> Receive(int sourceRank, int count, int tag = 0)
    {
        EnsureInitialized();
        ValidateRank(sourceRank, nameof(sourceRank));
        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Receive count cannot be negative.");
        }

        long start = Stopwatch.GetTimestamp();
        var accessor = _accessor!;
        int pairIdx = (sourceRank * _worldSize) + _rank;
        long pairOffset = _p2pBaseOffset + ((long)pairIdx * (P2PHeaderStride + P2PChunkCapacityBytes));
        long dataOffset = pairOffset + P2PHeaderStride;
        long expectedSeq = _p2pRecvSeq![sourceRank] + 1;

        var result = new Vector<T>(count);
        var dstSpan = result.AsWritableSpan();
        int received = 0;
        do
        {
            long waitStart = Stopwatch.GetTimestamp();
            int spins = 0;
            while (true)
            {
                Thread.MemoryBarrier();
                long seq = accessor.ReadInt64(pairOffset + 0);
                if (seq == expectedSeq)
                {
                    break;
                }

                CheckPeerAlive(sourceRank);
                CheckTimeout(waitStart, $"Receive on rank {_rank} from rank {sourceRank}");
                SpinOrYield(ref spins);
            }

            int msgTag = accessor.ReadInt32(pairOffset + 20);
            int chunk = accessor.ReadInt32(pairOffset + 16);
            if (msgTag != tag)
            {
                throw new InvalidOperationException(
                    $"SharedMemory Receive tag mismatch from rank {sourceRank}: expected tag {tag}, received tag {msgTag}.");
            }

            if (received + chunk > count)
            {
                throw new InvalidOperationException(
                    $"SharedMemory Receive count overflow from rank {sourceRank}: expected {count} elements, got at least {received + chunk}.");
            }

            if (chunk > 0)
            {
                ReadRawOffsetIntoSpan(dataOffset, dstSpan.Slice(received, chunk), chunk);
            }

            Thread.MemoryBarrier();
            accessor.Write(pairOffset + 8, expectedSeq);
            Thread.MemoryBarrier();
            SignalWakeEvent();

            _p2pRecvSeq[sourceRank] = expectedSeq;
            expectedSeq++;
            received += chunk;
        }
        while (received < count);

        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
        return result;
    }

    private void AdvancePhaseAndWait(int opKind, int elementCount)
    {
        var accessor = _accessor!;
        long nextPhase = ++_localPhase;
        int parity = (int)(nextPhase & 1L);
        int countOffset = parity == 0 ? 16 : 40;
        int opOffset = parity == 0 ? 20 : 44;
        long myHeader = RankHeaderOffset(_rank);

        accessor.Write(myHeader + countOffset, elementCount);
        accessor.Write(myHeader + opOffset, opKind);
        accessor.Write(myHeader + 24, Stopwatch.GetTimestamp());
        Thread.MemoryBarrier();
        accessor.Write(myHeader + 8, nextPhase);
        Thread.MemoryBarrier();
        SignalWakeEvent();

        long waitStart = Stopwatch.GetTimestamp();
        for (int peer = 0; peer < _worldSize; peer++)
        {
            if (peer == _rank)
            {
                continue;
            }

            long peerHeader = RankHeaderOffset(peer);
            int spins = 0;
            while (true)
            {
                Thread.MemoryBarrier();
                if (accessor.ReadInt32(8) != 0)
                {
                    throw new IOException(
                        $"Shared-memory session '{_sessionName}' was aborted because a peer rank failed or disconnected.");
                }

                long peerPhase = accessor.ReadInt64(peerHeader + 8);
                if (peerPhase >= nextPhase)
                {
                    Thread.MemoryBarrier();
                    int peerOp = accessor.ReadInt32(peerHeader + opOffset);
                    if (peerOp != opKind)
                    {
                        accessor.Write(8, 1);
                        Thread.MemoryBarrier();
                        throw new InvalidOperationException(
                            $"Collective mismatch at phase {nextPhase}: rank {_rank} entered op {opKind}, " +
                            $"but rank {peer} entered op {peerOp}.");
                    }

                    break;
                }

                int peerState = accessor.ReadInt32(peerHeader + 0);
                if (peerState == RankStateShutdown)
                {
                    accessor.Write(8, 1);
                    Thread.MemoryBarrier();
                    throw new IOException(
                        $"Distributed peer rank {peer} shut down before completing collective phase {nextPhase}.");
                }

                CheckPeerAlive(peer);
                CheckTimeout(waitStart, $"collective op {opKind} at phase {nextPhase} waiting on rank {peer}");
                SpinOrYield(ref spins);
            }
        }

        Thread.MemoryBarrier();
    }

    private void CheckPeerAlive(int peer)
    {
        var proc = _peerProcesses?[peer];
        if (proc is null)
        {
            return;
        }

        bool exited;
        try
        {
            exited = proc.HasExited;
        }
        catch (Exception)
        {
            exited = false;
        }

        if (exited)
        {
            if (_accessor is not null)
            {
                _accessor.Write(8, 1);
                Thread.MemoryBarrier();
            }

            SignalWakeEvent();
            throw new IOException(
                $"Distributed peer rank {peer} (PID {proc.Id}) terminated unexpectedly during distributed training.");
        }
    }

    private void CheckTimeout(long startTimestamp, string context)
    {
        double elapsedMs = (Stopwatch.GetTimestamp() - startTimestamp) * 1000.0 / Stopwatch.Frequency;
        if (elapsedMs > _timeoutMilliseconds)
        {
            throw new TimeoutException(
                $"Rank {_rank} timed out after {elapsedMs:F0} ms (limit {_timeoutMilliseconds} ms) while {context}.");
        }
    }

    private void SpinOrYield(ref int spins)
    {
        spins++;
        if (spins <= 2048)
        {
            Thread.SpinWait(16);
        }
        else if (spins <= 4096)
        {
            Thread.Yield();
        }
        else if (_wakeEvent is not null)
        {
            _wakeEvent.WaitOne(millisecondsTimeout: 1);
        }
        else
        {
            Thread.Sleep(1);
        }
    }

    private void SignalWakeEvent()
    {
        try
        {
            _wakeEvent?.Set();
        }
        catch (ObjectDisposedException)
        {
        }
    }

#if !AIDN_LEGACY_TFM
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
#endif
    private void InitializeWindowsSharedMemory(long totalCapacityBytes)
    {
        string mapName = $"Local\\AiDotNet_Shm_{_sessionName}";
        string eventName = $"Local\\AiDotNet_Evt_{_sessionName}";
        _mmf = MemoryMappedFile.CreateOrOpen(mapName, totalCapacityBytes, MemoryMappedFileAccess.ReadWrite);
        _accessor = _mmf.CreateViewAccessor(0, totalCapacityBytes, MemoryMappedFileAccess.ReadWrite);
        _wakeEvent = new EventWaitHandle(false, EventResetMode.AutoReset, eventName);
    }

    private void InitializeFileBackedSharedMemory(long totalCapacityBytes)
    {
        string shmDir = Directory.Exists("/dev/shm") ? "/dev/shm" : Path.GetTempPath();
        _backingFilePath = Path.Combine(shmDir, $"aidotnet_shm_{_sessionName}.bin");
        _backingFileStream = new FileStream(
            _backingFilePath,
            FileMode.OpenOrCreate,
            FileAccess.ReadWrite,
            FileShare.ReadWrite | FileShare.Delete);
        if (_backingFileStream.Length < totalCapacityBytes)
        {
            try
            {
                _backingFileStream.SetLength(totalCapacityBytes);
            }
            catch (IOException)
            {
                // Another rank resized concurrently.
            }
        }

        _mmf = MemoryMappedFile.CreateFromFile(
            _backingFileStream,
            mapName: null,
            capacity: totalCapacityBytes,
            MemoryMappedFileAccess.ReadWrite,
            HandleInheritability.None,
            leaveOpen: true);
        _accessor = _mmf.CreateViewAccessor(0, totalCapacityBytes, MemoryMappedFileAccess.ReadWrite);
    }

    private void ReleaseResources()
    {
        if (_peerProcesses is not null)
        {
            for (int i = 0; i < _peerProcesses.Length; i++)
            {
                _peerProcesses[i]?.Dispose();
                _peerProcesses[i] = null;
            }

            _peerProcesses = null;
        }

        _wakeEvent?.Dispose();
        _wakeEvent = null;
        _accessor?.Dispose();
        _accessor = null;
        _mmf?.Dispose();
        _mmf = null;

        if (_backingFileStream is not null)
        {
            _backingFileStream.Dispose();
            _backingFileStream = null;
        }

        if (_rank == 0 && _backingFilePath is not null)
        {
            try
            {
                if (File.Exists(_backingFilePath))
                {
                    File.Delete(_backingFilePath);
                }
            }
            catch (IOException)
            {
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long RankHeaderOffset(int rank) => GlobalHeaderSize + ((long)rank * RankHeaderStride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private long SlotByteOffset(int rank) => CollectiveRegionOffset + ((long)rank * _slotCapacityBytes);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ElementSizeBytes()
    {
        if (typeof(T) == typeof(float) || typeof(T) == typeof(int))
        {
            return sizeof(float);
        }

        if (typeof(T) == typeof(double) || typeof(T) == typeof(long))
        {
            return sizeof(double);
        }

        if (typeof(T) == typeof(byte))
        {
            return 1;
        }

        return sizeof(double); // Generic fallback serializes via IEEE-754 double.
    }

    private void WriteSpanToSlot(int rank, Span<T> span)
        => WriteReadOnlySpanToRawOffset(SlotByteOffset(rank), span);

    private void WriteReadOnlySpanToSlot(int rank, ReadOnlySpan<T> span)
        => WriteReadOnlySpanToRawOffset(SlotByteOffset(rank), span);

    private void WriteReadOnlySpanToSlotOffset(int rank, long byteSubOffset, ReadOnlySpan<T> span)
        => WriteReadOnlySpanToRawOffset(SlotByteOffset(rank) + byteSubOffset, span);

    [AiDotNet.Attributes.ScalarKernel("Generic T fallback converts elements to double for shared-memory staging; float and double use bulk array block copies.")]
    private void WriteReadOnlySpanToRawOffset(long byteOffset, ReadOnlySpan<T> span)
    {
        int count = span.Length;
        if (count == 0)
        {
            return;
        }

        var accessor = _accessor!;
        if (typeof(T) == typeof(float))
        {
            float[] rented = ArrayPool<float>.Shared.Rent(count);
            try
            {
                CopyGenericSpanToFloatArray(span, rented, count);
                accessor.WriteArray(byteOffset, rented, 0, count);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }

            return;
        }

        if (typeof(T) == typeof(double))
        {
            double[] rented = ArrayPool<double>.Shared.Rent(count);
            try
            {
                CopyGenericSpanToDoubleArray(span, rented, count);
                accessor.WriteArray(byteOffset, rented, 0, count);
            }
            finally
            {
                ArrayPool<double>.Shared.Return(rented);
            }

            return;
        }

        double[] fallback = ArrayPool<double>.Shared.Rent(count);
        try
        {
            for (int i = 0; i < count; i++)
            {
                fallback[i] = NumOps.ToDouble(span[i]);
            }

            accessor.WriteArray(byteOffset, fallback, 0, count);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(fallback);
        }
    }

    private void ReadSlotIntoSpan(int rank, Span<T> destination, int count)
        => ReadRawOffsetIntoSpan(SlotByteOffset(rank), destination, count);

    [AiDotNet.Attributes.ScalarKernel("Generic T fallback converts staged doubles back to T; float and double use bulk array block copies.")]
    private void ReadRawOffsetIntoSpan(long byteOffset, Span<T> destination, int count)
    {
        if (count == 0)
        {
            return;
        }

        var accessor = _accessor!;
        if (typeof(T) == typeof(float))
        {
            float[] rented = ArrayPool<float>.Shared.Rent(count);
            try
            {
                accessor.ReadArray(byteOffset, rented, 0, count);
                CopyFloatArrayToGenericSpan(rented, destination, count);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }

            return;
        }

        if (typeof(T) == typeof(double))
        {
            double[] rented = ArrayPool<double>.Shared.Rent(count);
            try
            {
                accessor.ReadArray(byteOffset, rented, 0, count);
                CopyDoubleArrayToGenericSpan(rented, destination, count);
            }
            finally
            {
                ArrayPool<double>.Shared.Return(rented);
            }

            return;
        }

        double[] fallback = ArrayPool<double>.Shared.Rent(count);
        try
        {
            accessor.ReadArray(byteOffset, fallback, 0, count);
            for (int i = 0; i < count; i++)
            {
                destination[i] = NumOps.FromDouble(fallback[i]);
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(fallback);
        }
    }

    private void ReduceSlotsIntoSpan(Span<T> destination, int count, ReductionOperation operation)
        => ReduceSlotsAtSubOffsetIntoSpan(0L, destination, count, operation);

    [AiDotNet.Attributes.ScalarKernel("Generic T reduction fallback; float and double use vectorized array kernels in canonical rank order.")]
    private void ReduceSlotsAtSubOffsetIntoSpan(long byteSubOffset, Span<T> destination, int count, ReductionOperation operation)
    {
        var accessor = _accessor!;
        if (typeof(T) == typeof(float))
        {
            float[] acc = ArrayPool<float>.Shared.Rent(count);
            float[] peer = ArrayPool<float>.Shared.Rent(count);
            try
            {
                accessor.ReadArray(SlotByteOffset(0) + byteSubOffset, acc, 0, count);
                for (int r = 1; r < _worldSize; r++)
                {
                    accessor.ReadArray(SlotByteOffset(r) + byteSubOffset, peer, 0, count);
                    CombineFloatArraysInPlace(acc, peer, count, operation);
                }

                if (operation == ReductionOperation.Average)
                {
                    ScaleFloatArrayInPlace(acc, count, 1.0f / _worldSize);
                }

                CopyFloatArrayToGenericSpan(acc, destination, count);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(peer);
                ArrayPool<float>.Shared.Return(acc);
            }

            return;
        }

        if (typeof(T) == typeof(double))
        {
            double[] acc = ArrayPool<double>.Shared.Rent(count);
            double[] peer = ArrayPool<double>.Shared.Rent(count);
            try
            {
                accessor.ReadArray(SlotByteOffset(0) + byteSubOffset, acc, 0, count);
                for (int r = 1; r < _worldSize; r++)
                {
                    accessor.ReadArray(SlotByteOffset(r) + byteSubOffset, peer, 0, count);
                    CombineDoubleArraysInPlace(acc, peer, count, operation);
                }

                if (operation == ReductionOperation.Average)
                {
                    ScaleDoubleArrayInPlace(acc, count, 1.0 / _worldSize);
                }

                CopyDoubleArrayToGenericSpan(acc, destination, count);
            }
            finally
            {
                ArrayPool<double>.Shared.Return(peer);
                ArrayPool<double>.Shared.Return(acc);
            }

            return;
        }

        double[] genericAcc = ArrayPool<double>.Shared.Rent(count);
        double[] genericPeer = ArrayPool<double>.Shared.Rent(count);
        try
        {
            accessor.ReadArray(SlotByteOffset(0) + byteSubOffset, genericAcc, 0, count);
            for (int i = 0; i < count; i++)
            {
                destination[i] = NumOps.FromDouble(genericAcc[i]);
            }

            for (int r = 1; r < _worldSize; r++)
            {
                accessor.ReadArray(SlotByteOffset(r) + byteSubOffset, genericPeer, 0, count);
                for (int i = 0; i < count; i++)
                {
                    destination[i] = ApplyReductionOperation(destination[i], NumOps.FromDouble(genericPeer[i]), operation);
                }
            }

            if (operation == ReductionOperation.Average)
            {
                T invWorld = NumOps.FromDouble(1.0 / _worldSize);
                for (int i = 0; i < count; i++)
                {
                    destination[i] = NumOps.Multiply(destination[i], invWorld);
                }
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(genericPeer);
            ArrayPool<double>.Shared.Return(genericAcc);
        }
    }

    private static void CombineFloatArraysInPlace(float[] acc, float[] peer, int count, ReductionOperation op)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && (op == ReductionOperation.Sum || op == ReductionOperation.Average || op == ReductionOperation.Product || op == ReductionOperation.Min || op == ReductionOperation.Max))
        {
            int width = System.Numerics.Vector<float>.Count;
            int limit = count - width;
            for (; i <= limit; i += width)
            {
                var va = new System.Numerics.Vector<float>(acc, i);
                var vb = new System.Numerics.Vector<float>(peer, i);
                var vr = op switch
                {
                    ReductionOperation.Sum or ReductionOperation.Average => va + vb,
                    ReductionOperation.Product => va * vb,
                    ReductionOperation.Min => System.Numerics.Vector.Min(va, vb),
                    ReductionOperation.Max => System.Numerics.Vector.Max(va, vb),
                    _ => va + vb,
                };
                vr.CopyTo(acc, i);
            }
        }

        for (; i < count; i++)
        {
            acc[i] = op switch
            {
                ReductionOperation.Sum or ReductionOperation.Average => acc[i] + peer[i],
                ReductionOperation.Product => acc[i] * peer[i],
                ReductionOperation.Min => peer[i] < acc[i] ? peer[i] : acc[i],
                ReductionOperation.Max => peer[i] > acc[i] ? peer[i] : acc[i],
                _ => throw new NotSupportedException($"Operation {op} is not supported."),
            };
        }
    }

    private static void ScaleFloatArrayInPlace(float[] acc, int count, float factor)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int width = System.Numerics.Vector<float>.Count;
            int limit = count - width;
            var vf = new System.Numerics.Vector<float>(factor);
            for (; i <= limit; i += width)
            {
                (new System.Numerics.Vector<float>(acc, i) * vf).CopyTo(acc, i);
            }
        }

        for (; i < count; i++)
        {
            acc[i] *= factor;
        }
    }

    private static void CombineDoubleArraysInPlace(double[] acc, double[] peer, int count, ReductionOperation op)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && (op == ReductionOperation.Sum || op == ReductionOperation.Average || op == ReductionOperation.Product || op == ReductionOperation.Min || op == ReductionOperation.Max))
        {
            int width = System.Numerics.Vector<double>.Count;
            int limit = count - width;
            for (; i <= limit; i += width)
            {
                var va = new System.Numerics.Vector<double>(acc, i);
                var vb = new System.Numerics.Vector<double>(peer, i);
                var vr = op switch
                {
                    ReductionOperation.Sum or ReductionOperation.Average => va + vb,
                    ReductionOperation.Product => va * vb,
                    ReductionOperation.Min => System.Numerics.Vector.Min(va, vb),
                    ReductionOperation.Max => System.Numerics.Vector.Max(va, vb),
                    _ => va + vb,
                };
                vr.CopyTo(acc, i);
            }
        }

        for (; i < count; i++)
        {
            acc[i] = op switch
            {
                ReductionOperation.Sum or ReductionOperation.Average => acc[i] + peer[i],
                ReductionOperation.Product => acc[i] * peer[i],
                ReductionOperation.Min => peer[i] < acc[i] ? peer[i] : acc[i],
                ReductionOperation.Max => peer[i] > acc[i] ? peer[i] : acc[i],
                _ => throw new NotSupportedException($"Operation {op} is not supported."),
            };
        }
    }

    private static void ScaleDoubleArrayInPlace(double[] acc, int count, double factor)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int width = System.Numerics.Vector<double>.Count;
            int limit = count - width;
            var vf = new System.Numerics.Vector<double>(factor);
            for (; i <= limit; i += width)
            {
                (new System.Numerics.Vector<double>(acc, i) * vf).CopyTo(acc, i);
            }
        }

        for (; i < count; i++)
        {
            acc[i] *= factor;
        }
    }

    private static void CopyGenericSpanToFloatArray(ReadOnlySpan<T> src, float[] dst, int count)
    {
#if NET7_0_OR_GREATER
        System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(
            ref Unsafe.As<T, float>(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(src)),
            count).CopyTo(dst.AsSpan(0, count));
#else
        for (int i = 0; i < count; i++)
        {
            T val = src[i];
            dst[i] = Unsafe.As<T, float>(ref val);
        }
#endif
    }

    private static void CopyFloatArrayToGenericSpan(float[] src, Span<T> dst, int count)
    {
#if NET7_0_OR_GREATER
        src.AsSpan(0, count).CopyTo(
            System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
                ref Unsafe.As<T, float>(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(dst)),
                count));
#else
        for (int i = 0; i < count; i++)
        {
            float val = src[i];
            dst[i] = Unsafe.As<float, T>(ref val);
        }
#endif
    }

    private static void CopyGenericSpanToDoubleArray(ReadOnlySpan<T> src, double[] dst, int count)
    {
#if NET7_0_OR_GREATER
        System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(
            ref Unsafe.As<T, double>(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(src)),
            count).CopyTo(dst.AsSpan(0, count));
#else
        for (int i = 0; i < count; i++)
        {
            T val = src[i];
            dst[i] = Unsafe.As<T, double>(ref val);
        }
#endif
    }

    private static void CopyDoubleArrayToGenericSpan(double[] src, Span<T> dst, int count)
    {
#if NET7_0_OR_GREATER
        src.AsSpan(0, count).CopyTo(
            System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
                ref Unsafe.As<T, double>(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(dst)),
                count));
#else
        for (int i = 0; i < count; i++)
        {
            double val = src[i];
            dst[i] = Unsafe.As<double, T>(ref val);
        }
#endif
    }

    private static string SanitizeSessionName(string name)
    {
        var sb = new StringBuilder(name.Length);
        for (int i = 0; i < name.Length; i++)
        {
            char c = name[i];
            if (char.IsLetterOrDigit(c) || c == '_' || c == '-')
            {
                sb.Append(c);
            }
            else
            {
                sb.Append('_');
            }
        }

        return sb.Length > 0 ? sb.ToString() : "default";
    }

    private static long AlignUp(long value, long alignment)
        => ((value + alignment - 1) / alignment) * alignment;

    private static int ReadEnvInt(string primaryVar, string fallbackVar, int defaultValue)
    {
        string? raw = Environment.GetEnvironmentVariable(primaryVar) ?? Environment.GetEnvironmentVariable(fallbackVar);
        return !string.IsNullOrWhiteSpace(raw) && int.TryParse(raw, out int parsed) ? parsed : defaultValue;
    }
}
