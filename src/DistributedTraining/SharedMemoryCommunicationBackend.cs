using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
#if !AIDN_LEGACY_TFM
using System.Numerics.Tensors;
#endif
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
/// <item><description>Dedicated per-rank collective staging slots supporting single-pass zero-copy SIMD and pipelined
/// multi-chunk <c>AllReduce</c>, <c>ReduceScatter</c>, <c>AllGather</c> (both uniform and uneven shard sizes),
/// <c>Broadcast</c>, and <c>Scatter</c>.</description></item>
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
    private const int ProtocolVersion = 2;
    private const int MaxSupportedRanks = 64;
    private const int GlobalHeaderSize = 64;
    private const int RankHeaderStride = 64;
    private const int CollectiveRegionOffset = 8192;
    private const int P2PHeaderStride = 64;
    private const int P2PChunkCapacityBytes = 256 * 1024; // 256 KiB per ordered (src, dst) pair

    private const int GlobalMagicOffset = 0;
    private const int GlobalWorldSizeOffset = 4;
    private const int GlobalAbortFlagOffset = 8;
    private const int GlobalWaiterCountOffset = 12;
    private const int GlobalSlotCapacityOffset = 16; // int64 (bytes 16..23)
    private const int GlobalWireTypeOffset = 24;     // int32 (bytes 24..27)
    private const int GlobalRootPidOffset = 28;      // int32 (bytes 28..31)
    private const int GlobalSessionTokenOffset = 32; // int64 (bytes 32..39)

    private const int SpinCountBeforePark = 16384;
    private const int FastSpinThreshold = 256;

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
    private readonly long _timeoutStopwatchTicks;
    private readonly long _livenessPollIntervalTicks;
    private readonly object _collectiveLock = new object();

    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private nint _basePtr;
    private long _mappedCapacityBytes;
    private bool _viewHandleAcquired;
    private FileStream? _backingFileStream;
    private string? _backingFilePath;
    private EventWaitHandle? _wakeEvent;
    private Process?[]? _peerProcesses;
    private int[]? _peerExited;
    private long[]? _lastAliveCheckTicks;
    private int[]? _gatherCounts;
    private int[]? _gatherOffsets;
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
    /// <param name="slotCapacityBytes">Per-rank collective staging buffer capacity in bytes (default 64 MiB).</param>
    /// <param name="timeoutMilliseconds">Maximum wait time in milliseconds before throwing a timeout error.</param>
    public SharedMemoryCommunicationBackend(
        int rank,
        int worldSize,
        string sessionName = "aidotnet_shm_default",
        long slotCapacityBytes = 64L * 1024L * 1024L,
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
        _timeoutStopwatchTicks = Math.Max(1L, (long)Math.Ceiling(timeoutMilliseconds * (Stopwatch.Frequency / 1000.0)));
        _livenessPollIntervalTicks = Math.Max(1L, Stopwatch.Frequency / 10); // 100 ms fallback poll interval
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
        _p2pBaseOffset = checked(CollectiveRegionOffset + ((long)_worldSize * _slotCapacityBytes));
        long totalCapacityBytes = checked(_p2pBaseOffset + (pairCount * p2pSlotStride));

        try
        {
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
            AcquireViewPointer(accessor, totalCapacityBytes);

            _p2pSendSeq = new long[_worldSize];
            _p2pRecvSeq = new long[_worldSize];
            _peerProcesses = new Process?[_worldSize];
            _peerExited = new int[_worldSize];
            _lastAliveCheckTicks = new long[_worldSize];
            _gatherCounts = new int[_worldSize];
            _gatherOffsets = new int[_worldSize];
            _localPhase = 0;

            int currentPid = Process.GetCurrentProcess().Id;
            int myWireType = WireElementTypeCode();
            long myHeader = RankHeaderOffset(_rank);
            int expectedHeader = (ProtocolVersion << 16) ^ MagicValue;
            long startTicks = Stopwatch.GetTimestamp();
            long sessionToken;

            if (_rank == 0)
            {
                int existingMagic = ReadInt32Volatile(GlobalMagicOffset);
                int existingRootState = ReadInt32Volatile(RankHeaderOffset(0) + 0);
                int existingRootPid = ReadInt32Volatile(GlobalRootPidOffset);
                if (existingMagic == expectedHeader &&
                    existingRootState == RankStateActive &&
                    existingRootPid > 0 &&
                    existingRootPid != currentPid &&
                    IsOsProcessAlive(existingRootPid))
                {
                    throw new InvalidOperationException(
                        $"Shared-memory session '{_sessionName}' is already owned by active rank-0 process PID {existingRootPid}.");
                }

                sessionToken = GenerateSessionToken(currentPid);

                // Invalidate global magic while rank 0 scrubs any stale headers/P2P state from a crashed prior session.
                WriteInt32Volatile(GlobalMagicOffset, 0);
                Thread.MemoryBarrier();

                for (int r = 0; r < _worldSize; r++)
                {
                    long rHdr = RankHeaderOffset(r);
                    WriteInt32Volatile(rHdr + 0, RankStateUninitialized);
                    WriteInt32Volatile(rHdr + 4, 0);
                    WriteInt64Volatile(rHdr + 8, 0L);
                    WriteInt32Volatile(rHdr + 16, 0);
                    WriteInt32Volatile(rHdr + 20, 0);
                    WriteInt64Volatile(rHdr + 24, 0L);
                    WriteInt32Volatile(rHdr + 32, 0);
                    WriteInt32Volatile(rHdr + 40, 0);
                    WriteInt32Volatile(rHdr + 44, 0);
                }

                for (long pair = 0; pair < pairCount; pair++)
                {
                    long pOffset = _p2pBaseOffset + (pair * p2pSlotStride);
                    WriteInt64Volatile(pOffset + 0, 0L);
                    WriteInt64Volatile(pOffset + 8, 0L);
                    WriteInt32Volatile(pOffset + 16, 0);
                    WriteInt32Volatile(pOffset + 20, 0);
                    WriteInt32Volatile(pOffset + 24, 0);
                }

                WriteInt32Volatile(myHeader + 4, currentPid);
                WriteInt64Volatile(myHeader + 8, 0L);
                WriteInt32Volatile(myHeader + 16, 0);
                WriteInt32Volatile(myHeader + 20, 0);
                WriteInt64Volatile(myHeader + 24, sessionToken);
                WriteInt32Volatile(myHeader + 32, 0);
                WriteInt32Volatile(myHeader + 40, 0);
                WriteInt32Volatile(myHeader + 44, 0);
                Thread.MemoryBarrier();
                WriteInt32Volatile(myHeader + 0, RankStateActive);

                WriteInt32Volatile(GlobalWorldSizeOffset, _worldSize);
                WriteInt32Volatile(GlobalAbortFlagOffset, 0);
                WriteInt32Volatile(GlobalWaiterCountOffset, 0);
                WriteInt64Volatile(GlobalSlotCapacityOffset, _slotCapacityBytes);
                WriteInt32Volatile(GlobalWireTypeOffset, myWireType);
                WriteInt32Volatile(GlobalRootPidOffset, currentPid);
                WriteInt64Volatile(GlobalSessionTokenOffset, sessionToken);
                Thread.MemoryBarrier();
                WriteInt32Volatile(GlobalMagicOffset, expectedHeader);
                Thread.MemoryBarrier();
                SignalWakeEventUnconditional();
            }
            else
            {
                int rootSpins = 0;
                while (true)
                {
                    int hdrMagic = ReadInt32Volatile(GlobalMagicOffset);
                    long candidateToken = ReadInt64Volatile(GlobalSessionTokenOffset);
                    int rootPid = ReadInt32Volatile(GlobalRootPidOffset);
                    int rootState = ReadInt32Volatile(RankHeaderOffset(0) + 0);
                    long rootRankToken = ReadInt64Volatile(RankHeaderOffset(0) + 24);

                    if (hdrMagic == expectedHeader &&
                        candidateToken != 0L &&
                        rootState == RankStateActive &&
                        rootRankToken == candidateToken &&
                        rootPid > 0 &&
                        (rootPid == currentPid || IsOsProcessAlive(rootPid)))
                    {
                        if (ReadInt32Volatile(GlobalAbortFlagOffset) != 0)
                        {
                            ThrowSessionAborted();
                        }

                        ValidateSessionHeaderOrAbort(myWireType);
                        sessionToken = candidateToken;
                        break;
                    }

                    if (rootSpins < SpinCountBeforePark)
                    {
                        Thread.SpinWait(rootSpins < FastSpinThreshold ? 1 : 4);
                        rootSpins++;
                        if ((rootSpins & 255) == 0 && IsTimedOut(startTicks, out double elapsedMs))
                        {
                            ThrowHandshakeTimeout(elapsedMs, peer: 0);
                        }
                    }
                    else
                    {
                        WaitParkSlice();
                        if (IsTimedOut(startTicks, out double elapsedMs))
                        {
                            ThrowHandshakeTimeout(elapsedMs, peer: 0);
                        }
                    }
                }

                WriteInt32Volatile(myHeader + 4, currentPid);
                WriteInt64Volatile(myHeader + 8, 0L);
                WriteInt32Volatile(myHeader + 16, 0);
                WriteInt32Volatile(myHeader + 20, 0);
                WriteInt64Volatile(myHeader + 24, sessionToken);
                WriteInt32Volatile(myHeader + 32, 0);
                WriteInt32Volatile(myHeader + 40, 0);
                WriteInt32Volatile(myHeader + 44, 0);
                Thread.MemoryBarrier();
                WriteInt32Volatile(myHeader + 0, RankStateActive);
                Thread.MemoryBarrier();
                SignalWakeEventUnconditional();
            }

            // Wait for all ranks to register under this session token and bind peer Process handles for crash detection.
            for (int peer = 0; peer < _worldSize; peer++)
            {
                long peerHeader = RankHeaderOffset(peer);
                int spins = 0;
                while (true)
                {
                    if (ReadInt32Volatile(GlobalAbortFlagOffset) != 0)
                    {
                        ThrowSessionAborted();
                    }

                    int hdrMagic = ReadInt32Volatile(GlobalMagicOffset);
                    int state = ReadInt32Volatile(peerHeader + 0);
                    long peerToken = ReadInt64Volatile(peerHeader + 24);
                    if (hdrMagic == expectedHeader && state == RankStateActive && peerToken == sessionToken)
                    {
                        ValidateSessionHeaderOrAbort(myWireType);

                        int peerPid = ReadInt32Volatile(peerHeader + 4);
                        if (peer != _rank && peerPid > 0 && peerPid != currentPid)
                        {
                            Process peerProc;
                            try
                            {
                                peerProc = Process.GetProcessById(peerPid);
                            }
                            catch (ArgumentException)
                            {
                                throw new IOException(
                                    $"Distributed peer rank {peer} (PID {peerPid}) exited during shared-memory handshake.");
                            }

                            _peerProcesses[peer] = peerProc;
                            _lastAliveCheckTicks[peer] = Stopwatch.GetTimestamp();
                            int capturedPeer = peer;
                            try
                            {
                                peerProc.EnableRaisingEvents = true;
                                peerProc.Exited += (_, _) =>
                                {
                                    var exitedFlags = _peerExited;
                                    if (exitedFlags is not null)
                                    {
                                        Volatile.Write(ref exitedFlags[capturedPeer], 1);
                                    }

                                    SignalWakeEventUnconditional();
                                };
                                if (peerProc.HasExited)
                                {
                                    Volatile.Write(ref _peerExited[peer], 1);
                                }
                            }
                            catch (InvalidOperationException ex)
                            {
                                Debug.WriteLine($"Peer PID {peerPid} exited while subscribing Exited event: {ex.Message}");
                                Volatile.Write(ref _peerExited[peer], 1);
                            }
                            catch (System.ComponentModel.Win32Exception ex)
                            {
                                Debug.WriteLine($"Peer PID {peerPid} restricted Exited subscription, using fallback polling: {ex.Message}");
                            }
                            catch (NotSupportedException ex)
                            {
                                Debug.WriteLine($"Peer PID {peerPid} does not support Exited events, using fallback polling: {ex.Message}");
                            }
                        }

                        break;
                    }

                    if (spins < SpinCountBeforePark)
                    {
                        Thread.SpinWait(spins < FastSpinThreshold ? 1 : 4);
                        spins++;
                        if ((spins & 255) == 0 && IsTimedOut(startTicks, out double elapsedMs))
                        {
                            ThrowHandshakeTimeout(elapsedMs, peer);
                        }
                    }
                    else
                    {
                        WaitParkSlice();
                        if (IsTimedOut(startTicks, out double elapsedMs))
                        {
                            ThrowHandshakeTimeout(elapsedMs, peer);
                        }
                    }
                }
            }

            // Initial barrier confirms every rank has bound peer PIDs before any collective begins.
            AdvancePhaseAndWait(OpInit, 0);
        }
        catch
        {
            ReleaseResources();
            throw;
        }
    }

    /// <inheritdoc/>
    protected override void OnShutdown()
    {
        if (_viewHandleAcquired && _basePtr != 0)
        {
            long myHeader = RankHeaderOffset(_rank);
            WriteInt32Volatile(myHeader + 0, RankStateShutdown);
            Thread.MemoryBarrier();
            SignalWakeEventUnconditional();
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
            WriteInt32Volatile(RankHeaderOffset(_rank) + 32, sendData.Length);
            AdvancePhaseAndWait(OpAllGatherMeta, 0);

            int[] counts = _gatherCounts ?? throw new InvalidOperationException("Shared-memory gather buffers are not initialized.");
            int[] offsets = _gatherOffsets ?? throw new InvalidOperationException("Shared-memory gather buffers are not initialized.");
            long totalLenLong = 0;
            int maxPeerLen = 0;
            int invalidRank = -1;
            int invalidCount = 0;
            for (int r = 0; r < _worldSize; r++)
            {
                int c = ReadInt32Volatile(RankHeaderOffset(r) + 32);
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
            var gathered = AllocateExactOutputVector(totalLength);
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
            int myCount = _rank == root ? data.Length : 0;
            WriteInt32Volatile(RankHeaderOffset(_rank) + 32, myCount);
            AdvancePhaseAndWait(OpBroadcastMeta, 0);

            int totalLength = ReadInt32Volatile(RankHeaderOffset(root) + 32);
            AdvancePhaseAndWait(OpBroadcastMetaAck, 0);

            if (totalLength < 0)
            {
                throw new InvalidOperationException($"Broadcast root rank {root} reported negative length {totalLength}.");
            }

            var result = AllocateExactOutputVector(totalLength);
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
            int myShardLength = _rank == root ? sendData.Length / _worldSize : 0;
            WriteInt32Volatile(RankHeaderOffset(_rank) + 32, myShardLength);
            AdvancePhaseAndWait(OpScatterMeta, 0);

            int shardLength = ReadInt32Volatile(RankHeaderOffset(root) + 32);
            AdvancePhaseAndWait(OpScatterMetaAck, 0);

            if (shardLength < 0)
            {
                throw new InvalidOperationException($"Scatter root rank {root} reported negative shard length {shardLength}.");
            }

            var result = AllocateExactOutputVector(shardLength);
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
        var shard = AllocateExactOutputVector(shardLength);
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
        DrainAsyncCollectives();

        long start = Stopwatch.GetTimestamp();
        int pairIdx = (_rank * _worldSize) + destinationRank;
        long pairOffset = _p2pBaseOffset + ((long)pairIdx * (P2PHeaderStride + P2PChunkCapacityBytes));
        long dataOffset = pairOffset + P2PHeaderStride;
        long peerHeader = RankHeaderOffset(destinationRank);
        var sendSeqTable = _p2pSendSeq ?? throw new InvalidOperationException("Shared-memory point-to-point state is not initialized.");
        long nextSeq = sendSeqTable[destinationRank] + 1;

        int elemSize = ElementSizeBytes();
        int maxChunkElements = P2PChunkCapacityBytes / elemSize;
        var span = data.AsSpan();
        int sent = 0;
        do
        {
            int chunk = Math.Min(maxChunkElements, span.Length - sent);
            WaitForP2PAck(pairOffset, peerHeader, destinationRank, nextSeq - 1, isPreSendSlotWait: true);

            if (chunk > 0)
            {
                WriteReadOnlySpanToRawOffset(dataOffset, span.Slice(sent, chunk));
            }

            WriteInt32Volatile(pairOffset + 16, chunk);
            WriteInt32Volatile(pairOffset + 20, tag);
            WriteInt32Volatile(pairOffset + 24, span.Length);
            WriteInt64Volatile(pairOffset + 0, nextSeq);
            Thread.MemoryBarrier();
            SignalWakeEvent();

            // Wait for receiver acknowledgment before advancing.
            WaitForP2PAck(pairOffset, peerHeader, destinationRank, nextSeq, isPreSendSlotWait: false);

            sendSeqTable[destinationRank] = nextSeq;
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

        DrainAsyncCollectives();

        long start = Stopwatch.GetTimestamp();
        int pairIdx = (sourceRank * _worldSize) + _rank;
        long pairOffset = _p2pBaseOffset + ((long)pairIdx * (P2PHeaderStride + P2PChunkCapacityBytes));
        long dataOffset = pairOffset + P2PHeaderStride;
        long peerHeader = RankHeaderOffset(sourceRank);
        var recvSeqTable = _p2pRecvSeq ?? throw new InvalidOperationException("Shared-memory point-to-point state is not initialized.");
        long expectedSeq = recvSeqTable[sourceRank] + 1;

        int elemSize = ElementSizeBytes();
        int maxChunkElements = P2PChunkCapacityBytes / elemSize;
        var result = AllocateExactOutputVector(count);
        var dstSpan = result.AsWritableSpan();
        int received = 0;
        do
        {
            WaitForP2PSeq(pairOffset, peerHeader, sourceRank, expectedSeq);

            int msgTag = ReadInt32Volatile(pairOffset + 20);
            int chunk = ReadInt32Volatile(pairOffset + 16);
            int totalElements = ReadInt32Volatile(pairOffset + 24);
            if (msgTag != tag)
            {
                PublishAbortAndThrowP2PProtocolMismatch(
                    $"SharedMemory Receive tag mismatch from rank {sourceRank}: expected tag {tag}, received tag {msgTag}.");
            }

            if (totalElements != count)
            {
                PublishAbortAndThrowP2PProtocolMismatch(
                    $"SharedMemory Receive length mismatch from rank {sourceRank}: expected {count} elements, sender transmitted {totalElements} elements.");
            }

            if (chunk < 0 || chunk > maxChunkElements)
            {
                PublishAbortAndThrowP2PProtocolMismatch(
                    $"SharedMemory Receive invalid chunk size {chunk} from rank {sourceRank} (max {maxChunkElements}).");
            }

            if (received + chunk > count)
            {
                PublishAbortAndThrowP2PProtocolMismatch(
                    $"SharedMemory Receive count overflow from rank {sourceRank}: expected {count} elements, got at least {received + chunk}.");
            }

            if (chunk > 0)
            {
                ReadRawOffsetIntoSpan(dataOffset, dstSpan.Slice(received, chunk), chunk);
            }

            WriteInt64Volatile(pairOffset + 8, expectedSeq);
            Thread.MemoryBarrier();
            SignalWakeEvent();

            recvSeqTable[sourceRank] = expectedSeq;
            expectedSeq++;
            received += chunk;
        }
        while (received < count);

        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - start);
        return result;
    }

    private void WaitForP2PAck(long pairOffset, long peerHeader, int destinationRank, long expectedAck, bool isPreSendSlotWait)
    {
        long waitStart = 0;
        int spins = 0;
        while (true)
        {
            if (ReadInt32Volatile(GlobalAbortFlagOffset) != 0)
            {
                ThrowSessionAborted();
            }

            if (ReadInt64Volatile(pairOffset + 8) == expectedAck)
            {
                return;
            }

            if (ReadInt32Volatile(peerHeader + 0) == RankStateShutdown)
            {
                PublishAbortAndThrowPeerShutdownP2P(destinationRank);
            }

            if (waitStart == 0)
            {
                waitStart = Stopwatch.GetTimestamp();
            }

            if (spins < SpinCountBeforePark)
            {
                Thread.SpinWait(spins < FastSpinThreshold ? 1 : 4);
                spins++;
                if ((spins & 255) == 0 && IsTimedOut(waitStart, out double elapsedMs))
                {
                    ThrowP2PSendTimeout(elapsedMs, destinationRank, isPreSendSlotWait);
                }

                if ((spins & 2047) == 0)
                {
                    CheckPeerAlive(destinationRank);
                }
            }
            else
            {
                bool ready = false;
                ref int waiterCountRef = ref GetGlobalWaiterCountRef();
                Interlocked.Increment(ref waiterCountRef);
                try
                {
                    Thread.MemoryBarrier();
                    if (ReadInt32Volatile(GlobalAbortFlagOffset) != 0)
                    {
                        ThrowSessionAborted();
                    }

                    if (ReadInt64Volatile(pairOffset + 8) == expectedAck)
                    {
                        ready = true;
                    }
                    else if (ReadInt32Volatile(peerHeader + 0) == RankStateShutdown)
                    {
                        PublishAbortAndThrowPeerShutdownP2P(destinationRank);
                    }
                    else
                    {
                        WaitParkSlice();
                    }
                }
                finally
                {
                    int remaining = Interlocked.Decrement(ref waiterCountRef);
                    if (ready && remaining > 0)
                    {
                        SignalWakeEventUnconditional();
                    }
                }

                if (ready)
                {
                    return;
                }

                CheckPeerAlive(destinationRank);
                if (IsTimedOut(waitStart, out double parkedElapsedMs))
                {
                    ThrowP2PSendTimeout(parkedElapsedMs, destinationRank, isPreSendSlotWait);
                }
            }
        }
    }

    private void WaitForP2PSeq(long pairOffset, long peerHeader, int sourceRank, long expectedSeq)
    {
        long waitStart = 0;
        int spins = 0;
        while (true)
        {
            if (ReadInt32Volatile(GlobalAbortFlagOffset) != 0)
            {
                ThrowSessionAborted();
            }

            if (ReadInt64Volatile(pairOffset + 0) == expectedSeq)
            {
                return;
            }

            if (ReadInt32Volatile(peerHeader + 0) == RankStateShutdown)
            {
                PublishAbortAndThrowPeerShutdownP2P(sourceRank);
            }

            if (waitStart == 0)
            {
                waitStart = Stopwatch.GetTimestamp();
            }

            if (spins < SpinCountBeforePark)
            {
                Thread.SpinWait(spins < FastSpinThreshold ? 1 : 4);
                spins++;
                if ((spins & 255) == 0 && IsTimedOut(waitStart, out double elapsedMs))
                {
                    ThrowP2PReceiveTimeout(elapsedMs, sourceRank);
                }

                if ((spins & 2047) == 0)
                {
                    CheckPeerAlive(sourceRank);
                }
            }
            else
            {
                bool ready = false;
                ref int waiterCountRef = ref GetGlobalWaiterCountRef();
                Interlocked.Increment(ref waiterCountRef);
                try
                {
                    Thread.MemoryBarrier();
                    if (ReadInt32Volatile(GlobalAbortFlagOffset) != 0)
                    {
                        ThrowSessionAborted();
                    }

                    if (ReadInt64Volatile(pairOffset + 0) == expectedSeq)
                    {
                        ready = true;
                    }
                    else if (ReadInt32Volatile(peerHeader + 0) == RankStateShutdown)
                    {
                        PublishAbortAndThrowPeerShutdownP2P(sourceRank);
                    }
                    else
                    {
                        WaitParkSlice();
                    }
                }
                finally
                {
                    int remaining = Interlocked.Decrement(ref waiterCountRef);
                    if (ready && remaining > 0)
                    {
                        SignalWakeEventUnconditional();
                    }
                }

                if (ready)
                {
                    return;
                }

                CheckPeerAlive(sourceRank);
                if (IsTimedOut(waitStart, out double parkedElapsedMs))
                {
                    ThrowP2PReceiveTimeout(parkedElapsedMs, sourceRank);
                }
            }
        }
    }

    private void AdvancePhaseAndWait(int opKind, int elementCount)
    {
        long nextPhase = ++_localPhase;
        int parity = (int)(nextPhase & 1L);
        int countOffset = parity == 0 ? 16 : 40;
        int opOffset = parity == 0 ? 20 : 44;
        long myHeader = RankHeaderOffset(_rank);

        WriteInt32Volatile(myHeader + countOffset, elementCount);
        WriteInt32Volatile(myHeader + opOffset, opKind);
        WriteInt64Volatile(myHeader + 8, nextPhase);
        Thread.MemoryBarrier();
        SignalWakeEvent();

        long waitStart = 0;
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
                if (ReadInt32Volatile(GlobalAbortFlagOffset) != 0)
                {
                    ThrowSessionAborted();
                }

                long peerPhase = ReadInt64Volatile(peerHeader + 8);
                if (peerPhase >= nextPhase)
                {
                    int peerOp = ReadInt32Volatile(peerHeader + opOffset);
                    if (peerOp != opKind)
                    {
                        PublishAbortAndThrowMismatch(nextPhase, opKind, peer, peerOp);
                    }

                    break;
                }

                int peerState = ReadInt32Volatile(peerHeader + 0);
                if (peerState == RankStateShutdown)
                {
                    PublishAbortAndThrowPeerShutdown(peer, nextPhase);
                }

                if (waitStart == 0)
                {
                    waitStart = Stopwatch.GetTimestamp();
                }

                if (spins < SpinCountBeforePark)
                {
                    Thread.SpinWait(spins < FastSpinThreshold ? 1 : 4);
                    spins++;
                    if ((spins & 255) == 0 && IsTimedOut(waitStart, out double elapsedMs))
                    {
                        ThrowCollectiveTimeout(elapsedMs, opKind, nextPhase, peer);
                    }

                    if ((spins & 2047) == 0)
                    {
                        CheckPeerAlive(peer);
                    }
                }
                else
                {
                    bool ready = false;
                    ref int waiterCountRef = ref GetGlobalWaiterCountRef();
                    Interlocked.Increment(ref waiterCountRef);
                    try
                    {
                        Thread.MemoryBarrier();
                        if (ReadInt32Volatile(GlobalAbortFlagOffset) != 0)
                        {
                            ThrowSessionAborted();
                        }

                        long recheckPhase = ReadInt64Volatile(peerHeader + 8);
                        if (recheckPhase >= nextPhase)
                        {
                            int peerOp = ReadInt32Volatile(peerHeader + opOffset);
                            if (peerOp != opKind)
                            {
                                PublishAbortAndThrowMismatch(nextPhase, opKind, peer, peerOp);
                            }

                            ready = true;
                        }
                        else if (ReadInt32Volatile(peerHeader + 0) == RankStateShutdown)
                        {
                            PublishAbortAndThrowPeerShutdown(peer, nextPhase);
                        }
                        else
                        {
                            WaitParkSlice();
                        }
                    }
                    finally
                    {
                        int remaining = Interlocked.Decrement(ref waiterCountRef);
                        if (ready && remaining > 0)
                        {
                            SignalWakeEventUnconditional();
                        }
                    }

                    if (ready)
                    {
                        break;
                    }

                    CheckPeerAlive(peer);
                    if (IsTimedOut(waitStart, out double parkedElapsedMs))
                    {
                        ThrowCollectiveTimeout(parkedElapsedMs, opKind, nextPhase, peer);
                    }
                }
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

        var exitedFlags = _peerExited;
        bool exited = exitedFlags is not null && Volatile.Read(ref exitedFlags[peer]) != 0;
        if (!exited)
        {
            var lastChecks = _lastAliveCheckTicks;
            long now = Stopwatch.GetTimestamp();
            if (lastChecks is not null && now - Volatile.Read(ref lastChecks[peer]) < _livenessPollIntervalTicks)
            {
                return;
            }

            if (lastChecks is not null)
            {
                Volatile.Write(ref lastChecks[peer], now);
            }

            try
            {
                exited = proc.HasExited;
                if (exited && exitedFlags is not null)
                {
                    Volatile.Write(ref exitedFlags[peer], 1);
                }
            }
            catch (InvalidOperationException)
            {
                exited = true;
                if (exitedFlags is not null)
                {
                    Volatile.Write(ref exitedFlags[peer], 1);
                }
            }
            catch (System.ComponentModel.Win32Exception ex)
            {
                Debug.WriteLine($"HasExited query restricted for peer {peer}: {ex.Message}");
                exited = false;
            }
            catch (NotSupportedException ex)
            {
                Debug.WriteLine($"HasExited query unsupported for peer {peer}: {ex.Message}");
                exited = false;
            }
        }

        if (exited)
        {
            if (_viewHandleAcquired && _basePtr != 0)
            {
                WriteInt32Volatile(GlobalAbortFlagOffset, 1);
                Thread.MemoryBarrier();
            }

            SignalWakeEventUnconditional();
            throw new IOException(
                $"Distributed peer rank {peer} (PID {proc.Id}) terminated unexpectedly during distributed training.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool IsTimedOut(long startTimestamp, out double elapsedMs)
    {
        long elapsedTicks = Stopwatch.GetTimestamp() - startTimestamp;
        if (elapsedTicks > _timeoutStopwatchTicks)
        {
            elapsedMs = elapsedTicks * 1000.0 / Stopwatch.Frequency;
            return true;
        }

        elapsedMs = 0.0;
        return false;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowSessionAborted()
    {
        throw new IOException(
            $"Shared-memory session '{_sessionName}' was aborted because a peer rank failed or disconnected.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void PublishAbortAndThrowMismatch(long nextPhase, int opKind, int peer, int peerOp)
    {
        WriteInt32Volatile(GlobalAbortFlagOffset, 1);
        Thread.MemoryBarrier();
        SignalWakeEventUnconditional();
        throw new InvalidOperationException(
            $"Collective mismatch at phase {nextPhase}: rank {_rank} entered op {opKind}, " +
            $"but rank {peer} entered op {peerOp}.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void PublishAbortAndThrowPeerShutdown(int peer, long nextPhase)
    {
        WriteInt32Volatile(GlobalAbortFlagOffset, 1);
        Thread.MemoryBarrier();
        SignalWakeEventUnconditional();
        throw new IOException(
            $"Distributed peer rank {peer} shut down before completing collective phase {nextPhase}.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void PublishAbortAndThrowPeerShutdownP2P(int peer)
    {
        WriteInt32Volatile(GlobalAbortFlagOffset, 1);
        Thread.MemoryBarrier();
        SignalWakeEventUnconditional();
        throw new IOException(
            $"Distributed peer rank {peer} shut down before completing point-to-point transfer.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void PublishAbortAndThrowP2PProtocolMismatch(string message)
    {
        WriteInt32Volatile(GlobalAbortFlagOffset, 1);
        Thread.MemoryBarrier();
        SignalWakeEventUnconditional();
        throw new InvalidOperationException(message);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowCollectiveTimeout(double elapsedMs, int opKind, long nextPhase, int peer)
    {
        throw new TimeoutException(
            $"Rank {_rank} timed out after {elapsedMs:F0} ms (limit {_timeoutMilliseconds} ms) while collective op {opKind} at phase {nextPhase} waiting on rank {peer}.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowHandshakeTimeout(double elapsedMs, int peer)
    {
        throw new TimeoutException(
            $"Rank {_rank} timed out after {elapsedMs:F0} ms (limit {_timeoutMilliseconds} ms) while initializing shared-memory session '{_sessionName}' waiting for rank {peer}.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowP2PSendTimeout(double elapsedMs, int destinationRank, bool isPreSendSlotWait)
    {
        string action = isPreSendSlotWait
            ? $"Send from rank {_rank} to rank {destinationRank}"
            : $"Send ack from rank {destinationRank} to rank {_rank}";
        throw new TimeoutException(
            $"Rank {_rank} timed out after {elapsedMs:F0} ms (limit {_timeoutMilliseconds} ms) while {action}.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowP2PReceiveTimeout(double elapsedMs, int sourceRank)
    {
        throw new TimeoutException(
            $"Rank {_rank} timed out after {elapsedMs:F0} ms (limit {_timeoutMilliseconds} ms) while Receive on rank {_rank} from rank {sourceRank}.");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void WaitParkSlice()
    {
        if (_wakeEvent is not null)
        {
            _wakeEvent.WaitOne(millisecondsTimeout: 1);
        }
        else
        {
            Thread.Sleep(1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SignalWakeEvent()
    {
        if (ReadInt32Volatile(GlobalWaiterCountOffset) > 0)
        {
            SignalWakeEventUnconditional();
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void SignalWakeEventUnconditional()
    {
        try
        {
            _wakeEvent?.Set();
        }
        catch (ObjectDisposedException ex)
        {
            Debug.WriteLine($"Shared-memory wake event already disposed: {ex.Message}");
        }
    }

    private void AcquireViewPointer(MemoryMappedViewAccessor accessor, long totalCapacityBytes)
    {
        bool acquired = false;
        accessor.SafeMemoryMappedViewHandle.DangerousAddRef(ref acquired);
        if (!acquired)
        {
            throw new InvalidOperationException("Failed to acquire shared-memory view handle.");
        }

        _basePtr = accessor.SafeMemoryMappedViewHandle.DangerousGetHandle() + (nint)accessor.PointerOffset;
        Volatile.Write(ref _mappedCapacityBytes, totalCapacityBytes);
        _viewHandleAcquired = true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> AllocateExactOutputVector(int length)
    {
        if (length == 0)
        {
            return new Vector<T>(Array.Empty<T>());
        }

#if !AIDN_LEGACY_TFM
        if (typeof(T) == typeof(float) ||
            typeof(T) == typeof(double) ||
            typeof(T) == typeof(int) ||
            typeof(T) == typeof(long) ||
            typeof(T) == typeof(byte))
        {
            return new Vector<T>(GC.AllocateUninitializedArray<T>(length));
        }
#endif
        return new Vector<T>(length);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ref byte GetMappedRef(long byteOffset, int byteLength)
    {
        EnsureRangeInBounds(byteOffset, byteLength);
        nint ptr = _basePtr;
        if (ptr == 0)
        {
            ThrowViewNotMapped();
        }

        ref byte baseRef = ref Unsafe.AddByteOffset(ref Unsafe.NullRef<byte>(), ptr);
        return ref Unsafe.AddByteOffset(ref baseRef, (IntPtr)byteOffset);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ref int GetGlobalWaiterCountRef()
        => ref Unsafe.As<byte, int>(ref GetMappedRef(GlobalWaiterCountOffset, sizeof(int)));

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowViewNotMapped()
    {
        throw new ObjectDisposedException(
            nameof(SharedMemoryCommunicationBackend<T>),
            $"Shared-memory session '{_sessionName}' view has already been unmapped.");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int ReadInt32Volatile(long byteOffset)
        => Volatile.Read(ref Unsafe.As<byte, int>(ref GetMappedRef(byteOffset, sizeof(int))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void WriteInt32Volatile(long byteOffset, int value)
        => Volatile.Write(ref Unsafe.As<byte, int>(ref GetMappedRef(byteOffset, sizeof(int))), value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private long ReadInt64Volatile(long byteOffset)
        => Volatile.Read(ref Unsafe.As<byte, long>(ref GetMappedRef(byteOffset, sizeof(long))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void WriteInt64Volatile(long byteOffset, long value)
        => Volatile.Write(ref Unsafe.As<byte, long>(ref GetMappedRef(byteOffset, sizeof(long))), value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EnsureRangeInBounds(long byteOffset, int byteLength)
    {
        long capacity = Volatile.Read(ref _mappedCapacityBytes);
        if (byteLength < 0 ||
            (ulong)byteOffset > (ulong)capacity ||
            (ulong)byteLength > (ulong)(capacity - byteOffset))
        {
            ThrowMappedRangeOutOfBounds(byteOffset, byteLength, capacity);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowMappedRangeOutOfBounds(long byteOffset, int byteLength, long capacity)
    {
        throw new ArgumentOutOfRangeException(
            nameof(byteOffset),
            $"Shared-memory range [{byteOffset}, {byteOffset + byteLength}) exceeds mapped capacity {capacity}.");
    }

#if !AIDN_LEGACY_TFM
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
#endif
    private void InitializeWindowsSharedMemory(long totalCapacityBytes)
    {
        string mapName = $"Local\\AiDotNet_Shm_{_sessionName}";
        string eventName = $"Local\\AiDotNet_Evt_{_sessionName}";
        try
        {
            _mmf = MemoryMappedFile.CreateOrOpen(mapName, totalCapacityBytes, MemoryMappedFileAccess.ReadWrite);
            _accessor = _mmf.CreateViewAccessor(0, totalCapacityBytes, MemoryMappedFileAccess.ReadWrite);
            _wakeEvent = new EventWaitHandle(false, EventResetMode.AutoReset, eventName);
        }
        catch (UnauthorizedAccessException ex)
        {
            TrySignalExistingWindowsSessionAbort(mapName, eventName);
            throw new InvalidOperationException(
                $"Failed to open Windows shared-memory session '{_sessionName}' with requested capacity {totalCapacityBytes} bytes. " +
                "Another rank may have created the session with a smaller slotCapacityBytes or worldSize, or with incompatible access permissions.",
                ex);
        }
    }

#if !AIDN_LEGACY_TFM
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
#endif
    private static void TrySignalExistingWindowsSessionAbort(string mapName, string eventName)
    {
        try
        {
            using var existing = MemoryMappedFile.OpenExisting(mapName);
            using var headerView = existing.CreateViewAccessor(0, GlobalHeaderSize, MemoryMappedFileAccess.ReadWrite);
            int expectedHeader = (ProtocolVersion << 16) ^ MagicValue;
            if (headerView.ReadInt32(GlobalMagicOffset) == expectedHeader)
            {
                headerView.Write(GlobalAbortFlagOffset, 1);
                headerView.Flush();
                Thread.MemoryBarrier();
                using var evt = new EventWaitHandle(false, EventResetMode.AutoReset, eventName);
                evt.Set();
            }
        }
        catch (Exception abortEx) when (abortEx is IOException or UnauthorizedAccessException or ObjectDisposedException)
        {
            Debug.WriteLine($"Best-effort Windows shared-memory abort signal skipped: {abortEx.Message}");
        }
    }

    private void InitializeFileBackedSharedMemory(long totalCapacityBytes)
    {
        string shmDir = Directory.Exists("/dev/shm") ? "/dev/shm" : Path.GetTempPath();
        string safeFileName = Path.GetFileName($"aidotnet_shm_{_sessionName}.bin");
        _backingFilePath = Path.Combine(shmDir, safeFileName);
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
            catch (IOException ex)
            {
                Debug.WriteLine($"Concurrent shared-memory file resize observed: {ex.Message}");
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
        if (_viewHandleAcquired && _basePtr != 0)
        {
            long myHeader = RankHeaderOffset(_rank);
            WriteInt32Volatile(myHeader + 0, RankStateShutdown);
            Thread.MemoryBarrier();
        }

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

        bool wasViewAcquired = _viewHandleAcquired;
        _viewHandleAcquired = false;
        Volatile.Write(ref _mappedCapacityBytes, 0L);
        _basePtr = 0;
        Thread.MemoryBarrier();

        if (wasViewAcquired && _accessor is not null)
        {
            _accessor.SafeMemoryMappedViewHandle.DangerousRelease();
        }

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
            catch (IOException ex)
            {
                Debug.WriteLine($"Unable to unlink shared-memory backing file '{_backingFilePath}': {ex.Message}");
            }
        }
    }

    private void ValidateSessionHeaderOrAbort(int expectedWireType)
    {
        int savedWorld = ReadInt32Volatile(GlobalWorldSizeOffset);
        if (savedWorld != _worldSize)
        {
            WriteInt32Volatile(GlobalAbortFlagOffset, 1);
            Thread.MemoryBarrier();
            SignalWakeEventUnconditional();
            throw new InvalidOperationException(
                $"Shared-memory session '{_sessionName}' was initialized with WorldSize={savedWorld}, " +
                $"but rank {_rank} configured WorldSize={_worldSize}.");
        }

        long savedSlotCapacity = ReadInt64Volatile(GlobalSlotCapacityOffset);
        if (savedSlotCapacity != _slotCapacityBytes)
        {
            WriteInt32Volatile(GlobalAbortFlagOffset, 1);
            Thread.MemoryBarrier();
            SignalWakeEventUnconditional();
            throw new InvalidOperationException(
                $"Shared-memory session '{_sessionName}' was initialized with SlotCapacityBytes={savedSlotCapacity}, " +
                $"but rank {_rank} configured SlotCapacityBytes={_slotCapacityBytes}.");
        }

        int savedWireType = ReadInt32Volatile(GlobalWireTypeOffset);
        if (savedWireType != expectedWireType)
        {
            WriteInt32Volatile(GlobalAbortFlagOffset, 1);
            Thread.MemoryBarrier();
            SignalWakeEventUnconditional();
            throw new InvalidOperationException(
                $"Shared-memory session '{_sessionName}' element wire type code ({savedWireType}) does not match " +
                $"rank {_rank}'s element type '{typeof(T).Name}' ({expectedWireType}).");
        }
    }

    private static bool IsOsProcessAlive(int pid)
    {
        if (pid <= 0)
        {
            return false;
        }

        try
        {
            using var proc = Process.GetProcessById(pid);
            return !proc.HasExited;
        }
        catch (ArgumentException)
        {
            return false;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return true;
        }
    }

    private static long GenerateSessionToken(int pid)
    {
        byte[] guidBytes = Guid.NewGuid().ToByteArray();
        long guidPart = BitConverter.ToInt64(guidBytes, 0);
        long token = (Stopwatch.GetTimestamp() << 16) ^ ((long)pid & 0xFFFFL) ^ guidPart;
        return token == 0L ? 1L : token;
    }

    private static int WireElementTypeCode()
    {
        if (typeof(T) == typeof(float))
        {
            return 1;
        }

        if (typeof(T) == typeof(double))
        {
            return 2;
        }

        if (typeof(T) == typeof(int))
        {
            return 3;
        }

        if (typeof(T) == typeof(long))
        {
            return 4;
        }

        if (typeof(T) == typeof(byte))
        {
            return 5;
        }

        // Deterministic FNV-1a 32-bit hash across processes for non-primitive numeric types.
        string name = typeof(T).FullName ?? typeof(T).Name;
        uint hash = 2166136261u;
        for (int i = 0; i < name.Length; i++)
        {
            hash ^= name[i];
            hash *= 16777619u;
        }

        int code = unchecked((int)(hash | 0x40000000u));
        return code == 0 ? 6 : code;
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

#if !AIDN_LEGACY_TFM
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Span<byte> GetMappedByteSpan(long byteOffset, int byteLength)
        => MemoryMarshal.CreateSpan(ref GetMappedRef(byteOffset, byteLength), byteLength);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ReadOnlySpan<byte> GetMappedReadOnlyByteSpan(long byteOffset, int byteLength)
        => MemoryMarshal.CreateReadOnlySpan(ref GetMappedRef(byteOffset, byteLength), byteLength);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Span<TElem> GetMappedSpan<TElem>(long byteOffset, int elementCount)
        where TElem : unmanaged
    {
        int byteLength = checked(elementCount * Unsafe.SizeOf<TElem>());
        return MemoryMarshal.Cast<byte, TElem>(GetMappedByteSpan(byteOffset, byteLength));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private ReadOnlySpan<TElem> GetMappedReadOnlySpan<TElem>(long byteOffset, int elementCount)
        where TElem : unmanaged
    {
        int byteLength = checked(elementCount * Unsafe.SizeOf<TElem>());
        return MemoryMarshal.Cast<byte, TElem>(GetMappedReadOnlyByteSpan(byteOffset, byteLength));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ReadOnlySpan<TElem> AsReadOnlyTypedSpan<TElem>(ReadOnlySpan<T> span, int count)
        where TElem : unmanaged
        => MemoryMarshal.CreateReadOnlySpan(
            ref Unsafe.As<T, TElem>(ref MemoryMarshal.GetReference(span)),
            count);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Span<TElem> AsWritableTypedSpan<TElem>(Span<T> span, int count)
        where TElem : unmanaged
        => MemoryMarshal.CreateSpan(
            ref Unsafe.As<T, TElem>(ref MemoryMarshal.GetReference(span)),
            count);
#endif

    private void WriteSpanToSlot(int rank, Span<T> span)
        => WriteReadOnlySpanToRawOffset(SlotByteOffset(rank), span);

    private void WriteReadOnlySpanToSlot(int rank, ReadOnlySpan<T> span)
        => WriteReadOnlySpanToRawOffset(SlotByteOffset(rank), span);

    private void WriteReadOnlySpanToSlotOffset(int rank, long byteSubOffset, ReadOnlySpan<T> span)
        => WriteReadOnlySpanToRawOffset(SlotByteOffset(rank) + byteSubOffset, span);

    private void WriteReadOnlySpanToRawOffset(long byteOffset, ReadOnlySpan<T> span)
    {
        int count = span.Length;
        if (count == 0)
        {
            return;
        }

#if !AIDN_LEGACY_TFM
        if (typeof(T) == typeof(float))
        {
            MemoryMarshal.AsBytes(AsReadOnlyTypedSpan<float>(span, count))
                .CopyTo(GetMappedByteSpan(byteOffset, checked(count * sizeof(float))));
            return;
        }

        if (typeof(T) == typeof(double))
        {
            MemoryMarshal.AsBytes(AsReadOnlyTypedSpan<double>(span, count))
                .CopyTo(GetMappedByteSpan(byteOffset, checked(count * sizeof(double))));
            return;
        }

        if (typeof(T) == typeof(int))
        {
            MemoryMarshal.AsBytes(AsReadOnlyTypedSpan<int>(span, count))
                .CopyTo(GetMappedByteSpan(byteOffset, checked(count * sizeof(int))));
            return;
        }

        if (typeof(T) == typeof(long))
        {
            MemoryMarshal.AsBytes(AsReadOnlyTypedSpan<long>(span, count))
                .CopyTo(GetMappedByteSpan(byteOffset, checked(count * sizeof(long))));
            return;
        }

        if (typeof(T) == typeof(byte))
        {
            AsReadOnlyTypedSpan<byte>(span, count).CopyTo(GetMappedByteSpan(byteOffset, count));
            return;
        }

        WriteGenericSpanToMappedDoubles(GetMappedSpan<double>(byteOffset, count), span, count);
#else
        WriteReadOnlySpanToRawOffsetLegacy(byteOffset, span, count);
#endif
    }

    private void ReadSlotIntoSpan(int rank, Span<T> destination, int count)
        => ReadRawOffsetIntoSpan(SlotByteOffset(rank), destination, count);

    private void ReadRawOffsetIntoSpan(long byteOffset, Span<T> destination, int count)
    {
        if (count == 0)
        {
            return;
        }

#if !AIDN_LEGACY_TFM
        if (typeof(T) == typeof(float))
        {
            GetMappedReadOnlySpan<float>(byteOffset, count)
                .CopyTo(AsWritableTypedSpan<float>(destination, count));
            return;
        }

        if (typeof(T) == typeof(double))
        {
            GetMappedReadOnlySpan<double>(byteOffset, count)
                .CopyTo(AsWritableTypedSpan<double>(destination, count));
            return;
        }

        if (typeof(T) == typeof(int))
        {
            GetMappedReadOnlySpan<int>(byteOffset, count)
                .CopyTo(AsWritableTypedSpan<int>(destination, count));
            return;
        }

        if (typeof(T) == typeof(long))
        {
            GetMappedReadOnlySpan<long>(byteOffset, count)
                .CopyTo(AsWritableTypedSpan<long>(destination, count));
            return;
        }

        if (typeof(T) == typeof(byte))
        {
            GetMappedReadOnlyByteSpan(byteOffset, count)
                .CopyTo(AsWritableTypedSpan<byte>(destination, count));
            return;
        }

        ReadMappedDoublesIntoGenericSpan(GetMappedReadOnlySpan<double>(byteOffset, count), destination, count);
#else
        ReadRawOffsetIntoSpanLegacy(byteOffset, destination, count);
#endif
    }

    private void ReduceSlotsIntoSpan(Span<T> destination, int count, ReductionOperation operation)
        => ReduceSlotsAtSubOffsetIntoSpan(0L, destination, count, operation);

    private void ReduceSlotsAtSubOffsetIntoSpan(long byteSubOffset, Span<T> destination, int count, ReductionOperation operation)
    {
        if (count == 0)
        {
            return;
        }

#if !AIDN_LEGACY_TFM
        if (typeof(T) == typeof(float))
        {
            Span<float> dest = AsWritableTypedSpan<float>(destination, count);
            GetMappedReadOnlySpan<float>(SlotByteOffset(0) + byteSubOffset, count).CopyTo(dest);
            for (int r = 1; r < _worldSize; r++)
            {
                ReadOnlySpan<float> peer = GetMappedReadOnlySpan<float>(SlotByteOffset(r) + byteSubOffset, count);
                CombineFloatSpanInPlace(dest, peer, operation);
            }

            if (operation == ReductionOperation.Average)
            {
                TensorPrimitives.Multiply(dest, 1.0f / _worldSize, dest);
            }

            return;
        }

        if (typeof(T) == typeof(double))
        {
            Span<double> dest = AsWritableTypedSpan<double>(destination, count);
            GetMappedReadOnlySpan<double>(SlotByteOffset(0) + byteSubOffset, count).CopyTo(dest);
            for (int r = 1; r < _worldSize; r++)
            {
                ReadOnlySpan<double> peer = GetMappedReadOnlySpan<double>(SlotByteOffset(r) + byteSubOffset, count);
                CombineDoubleSpanInPlace(dest, peer, operation);
            }

            if (operation == ReductionOperation.Average)
            {
                TensorPrimitives.Multiply(dest, 1.0 / _worldSize, dest);
            }

            return;
        }

        if (typeof(T) == typeof(int))
        {
            Span<int> dest = AsWritableTypedSpan<int>(destination, count);
            GetMappedReadOnlySpan<int>(SlotByteOffset(0) + byteSubOffset, count).CopyTo(dest);
            for (int r = 1; r < _worldSize; r++)
            {
                ReadOnlySpan<int> peer = GetMappedReadOnlySpan<int>(SlotByteOffset(r) + byteSubOffset, count);
                CombineInt32SpanInPlace(dest, peer, operation);
            }

            if (operation == ReductionOperation.Average)
            {
                for (int i = 0; i < count; i++)
                {
                    dest[i] /= _worldSize;
                }
            }

            return;
        }

        if (typeof(T) == typeof(long))
        {
            Span<long> dest = AsWritableTypedSpan<long>(destination, count);
            GetMappedReadOnlySpan<long>(SlotByteOffset(0) + byteSubOffset, count).CopyTo(dest);
            for (int r = 1; r < _worldSize; r++)
            {
                ReadOnlySpan<long> peer = GetMappedReadOnlySpan<long>(SlotByteOffset(r) + byteSubOffset, count);
                CombineInt64SpanInPlace(dest, peer, operation);
            }

            if (operation == ReductionOperation.Average)
            {
                for (int i = 0; i < count; i++)
                {
                    dest[i] /= _worldSize;
                }
            }

            return;
        }

        if (typeof(T) == typeof(byte))
        {
            Span<byte> dest = AsWritableTypedSpan<byte>(destination, count);
            GetMappedReadOnlyByteSpan(SlotByteOffset(0) + byteSubOffset, count).CopyTo(dest);
            for (int r = 1; r < _worldSize; r++)
            {
                ReadOnlySpan<byte> peer = GetMappedReadOnlyByteSpan(SlotByteOffset(r) + byteSubOffset, count);
                for (int i = 0; i < count; i++)
                {
                    dest[i] = operation switch
                    {
                        ReductionOperation.Sum or ReductionOperation.Average => (byte)(dest[i] + peer[i]),
                        ReductionOperation.Product => (byte)(dest[i] * peer[i]),
                        ReductionOperation.Min => peer[i] < dest[i] ? peer[i] : dest[i],
                        ReductionOperation.Max => peer[i] > dest[i] ? peer[i] : dest[i],
                        _ => throw new NotSupportedException($"Operation {operation} is not supported."),
                    };
                }
            }

            if (operation == ReductionOperation.Average)
            {
                for (int i = 0; i < count; i++)
                {
                    dest[i] = (byte)(dest[i] / _worldSize);
                }
            }

            return;
        }

        ReduceGenericMappedSlotsIntoSpan(byteSubOffset, destination, count, operation);
#else
        ReduceSlotsAtSubOffsetIntoSpanLegacy(byteSubOffset, destination, count, operation);
#endif
    }

#if !AIDN_LEGACY_TFM
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CombineFloatSpanInPlace(Span<float> dest, ReadOnlySpan<float> peer, ReductionOperation op)
    {
        switch (op)
        {
            case ReductionOperation.Sum:
            case ReductionOperation.Average:
                TensorPrimitives.Add(dest, peer, dest);
                break;
            case ReductionOperation.Product:
                TensorPrimitives.Multiply(dest, peer, dest);
                break;
            case ReductionOperation.Min:
                TensorPrimitives.Min(dest, peer, dest);
                break;
            case ReductionOperation.Max:
                TensorPrimitives.Max(dest, peer, dest);
                break;
            default:
                throw new NotSupportedException($"Operation {op} is not supported.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CombineDoubleSpanInPlace(Span<double> dest, ReadOnlySpan<double> peer, ReductionOperation op)
    {
        switch (op)
        {
            case ReductionOperation.Sum:
            case ReductionOperation.Average:
                TensorPrimitives.Add(dest, peer, dest);
                break;
            case ReductionOperation.Product:
                TensorPrimitives.Multiply(dest, peer, dest);
                break;
            case ReductionOperation.Min:
                TensorPrimitives.Min(dest, peer, dest);
                break;
            case ReductionOperation.Max:
                TensorPrimitives.Max(dest, peer, dest);
                break;
            default:
                throw new NotSupportedException($"Operation {op} is not supported.");
        }
    }

    private static void CombineInt32SpanInPlace(Span<int> dest, ReadOnlySpan<int> peer, ReductionOperation op)
    {
        switch (op)
        {
            case ReductionOperation.Sum:
            case ReductionOperation.Average:
                TensorPrimitives.Add(dest, peer, dest);
                break;
            case ReductionOperation.Product:
                TensorPrimitives.Multiply(dest, peer, dest);
                break;
            case ReductionOperation.Min:
                TensorPrimitives.Min(dest, peer, dest);
                break;
            case ReductionOperation.Max:
                TensorPrimitives.Max(dest, peer, dest);
                break;
            default:
                throw new NotSupportedException($"Operation {op} is not supported.");
        }
    }

    private static void CombineInt64SpanInPlace(Span<long> dest, ReadOnlySpan<long> peer, ReductionOperation op)
    {
        switch (op)
        {
            case ReductionOperation.Sum:
            case ReductionOperation.Average:
                TensorPrimitives.Add(dest, peer, dest);
                break;
            case ReductionOperation.Product:
                TensorPrimitives.Multiply(dest, peer, dest);
                break;
            case ReductionOperation.Min:
                TensorPrimitives.Min(dest, peer, dest);
                break;
            case ReductionOperation.Max:
                TensorPrimitives.Max(dest, peer, dest);
                break;
            default:
                throw new NotSupportedException($"Operation {op} is not supported.");
        }
    }

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE shared-memory write fallback converts T elements to IEEE-754 double.")]
    private void WriteGenericSpanToMappedDoubles(Span<double> mapped, ReadOnlySpan<T> src, int count)
    {
        for (int i = 0; i < count; i++)
        {
            mapped[i] = NumOps.ToDouble(src[i]);
        }
    }

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE shared-memory read fallback converts IEEE-754 doubles back to T.")]
    private void ReadMappedDoublesIntoGenericSpan(ReadOnlySpan<double> mapped, Span<T> dst, int count)
    {
        for (int i = 0; i < count; i++)
        {
            dst[i] = NumOps.FromDouble(mapped[i]);
        }
    }

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE shared-memory reduction fallback in canonical rank order.")]
    private void ReduceGenericMappedSlotsIntoSpan(long byteSubOffset, Span<T> destination, int count, ReductionOperation operation)
    {
        ReadOnlySpan<double> rank0 = GetMappedReadOnlySpan<double>(SlotByteOffset(0) + byteSubOffset, count);
        for (int i = 0; i < count; i++)
        {
            destination[i] = NumOps.FromDouble(rank0[i]);
        }

        for (int r = 1; r < _worldSize; r++)
        {
            ReadOnlySpan<double> peer = GetMappedReadOnlySpan<double>(SlotByteOffset(r) + byteSubOffset, count);
            for (int i = 0; i < count; i++)
            {
                destination[i] = ApplyReductionOperation(destination[i], NumOps.FromDouble(peer[i]), operation);
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
#else
    [AiDotNet.Attributes.ScalarKernel("Legacy TFM shared-memory write fallback for generic T; float/double/int/long/byte use bulk array block copies.")]
    private void WriteReadOnlySpanToRawOffsetLegacy(long byteOffset, ReadOnlySpan<T> span, int count)
    {
        var accessor = _accessor ?? throw new InvalidOperationException("Shared-memory accessor is not initialized.");
        if (typeof(T) == typeof(float))
        {
            float[] rented = ArrayPool<float>.Shared.Rent(count);
            try
            {
                for (int i = 0; i < count; i++)
                {
                    T val = span[i];
                    rented[i] = Unsafe.As<T, float>(ref val);
                }

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
                for (int i = 0; i < count; i++)
                {
                    T val = span[i];
                    rented[i] = Unsafe.As<T, double>(ref val);
                }

                accessor.WriteArray(byteOffset, rented, 0, count);
            }
            finally
            {
                ArrayPool<double>.Shared.Return(rented);
            }

            return;
        }

        if (typeof(T) == typeof(int))
        {
            int[] rented = ArrayPool<int>.Shared.Rent(count);
            try
            {
                for (int i = 0; i < count; i++)
                {
                    T val = span[i];
                    rented[i] = Unsafe.As<T, int>(ref val);
                }

                accessor.WriteArray(byteOffset, rented, 0, count);
            }
            finally
            {
                ArrayPool<int>.Shared.Return(rented);
            }

            return;
        }

        if (typeof(T) == typeof(long))
        {
            long[] rented = ArrayPool<long>.Shared.Rent(count);
            try
            {
                for (int i = 0; i < count; i++)
                {
                    T val = span[i];
                    rented[i] = Unsafe.As<T, long>(ref val);
                }

                accessor.WriteArray(byteOffset, rented, 0, count);
            }
            finally
            {
                ArrayPool<long>.Shared.Return(rented);
            }

            return;
        }

        if (typeof(T) == typeof(byte))
        {
            byte[] rented = ArrayPool<byte>.Shared.Rent(count);
            try
            {
                for (int i = 0; i < count; i++)
                {
                    T val = span[i];
                    rented[i] = Unsafe.As<T, byte>(ref val);
                }

                accessor.WriteArray(byteOffset, rented, 0, count);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
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

    [AiDotNet.Attributes.ScalarKernel("Legacy TFM shared-memory read fallback for generic T; float/double/int/long/byte use bulk array block copies.")]
    private void ReadRawOffsetIntoSpanLegacy(long byteOffset, Span<T> destination, int count)
    {
        var accessor = _accessor ?? throw new InvalidOperationException("Shared-memory accessor is not initialized.");
        if (typeof(T) == typeof(float))
        {
            float[] rented = ArrayPool<float>.Shared.Rent(count);
            try
            {
                accessor.ReadArray(byteOffset, rented, 0, count);
                for (int i = 0; i < count; i++)
                {
                    float val = rented[i];
                    destination[i] = Unsafe.As<float, T>(ref val);
                }
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
                for (int i = 0; i < count; i++)
                {
                    double val = rented[i];
                    destination[i] = Unsafe.As<double, T>(ref val);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(rented);
            }

            return;
        }

        if (typeof(T) == typeof(int))
        {
            int[] rented = ArrayPool<int>.Shared.Rent(count);
            try
            {
                accessor.ReadArray(byteOffset, rented, 0, count);
                for (int i = 0; i < count; i++)
                {
                    int val = rented[i];
                    destination[i] = Unsafe.As<int, T>(ref val);
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(rented);
            }

            return;
        }

        if (typeof(T) == typeof(long))
        {
            long[] rented = ArrayPool<long>.Shared.Rent(count);
            try
            {
                accessor.ReadArray(byteOffset, rented, 0, count);
                for (int i = 0; i < count; i++)
                {
                    long val = rented[i];
                    destination[i] = Unsafe.As<long, T>(ref val);
                }
            }
            finally
            {
                ArrayPool<long>.Shared.Return(rented);
            }

            return;
        }

        if (typeof(T) == typeof(byte))
        {
            byte[] rented = ArrayPool<byte>.Shared.Rent(count);
            try
            {
                accessor.ReadArray(byteOffset, rented, 0, count);
                for (int i = 0; i < count; i++)
                {
                    byte val = rented[i];
                    destination[i] = Unsafe.As<byte, T>(ref val);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
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

    [AiDotNet.Attributes.ScalarKernel("Legacy TFM generic reduction fallback; float and double use System.Numerics.Vector SIMD in canonical rank order.")]
    private void ReduceSlotsAtSubOffsetIntoSpanLegacy(long byteSubOffset, Span<T> destination, int count, ReductionOperation operation)
    {
        var accessor = _accessor ?? throw new InvalidOperationException("Shared-memory accessor is not initialized.");
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
                    CombineFloatArraysInPlaceLegacy(acc, peer, count, operation);
                }

                if (operation == ReductionOperation.Average)
                {
                    ScaleFloatArrayInPlaceLegacy(acc, count, 1.0f / _worldSize);
                }

                for (int i = 0; i < count; i++)
                {
                    float val = acc[i];
                    destination[i] = Unsafe.As<float, T>(ref val);
                }
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
                    CombineDoubleArraysInPlaceLegacy(acc, peer, count, operation);
                }

                if (operation == ReductionOperation.Average)
                {
                    ScaleDoubleArrayInPlaceLegacy(acc, count, 1.0 / _worldSize);
                }

                for (int i = 0; i < count; i++)
                {
                    double val = acc[i];
                    destination[i] = Unsafe.As<double, T>(ref val);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(peer);
                ArrayPool<double>.Shared.Return(acc);
            }

            return;
        }

        if (typeof(T) == typeof(int) || typeof(T) == typeof(long) || typeof(T) == typeof(byte))
        {
            T[] peerBuf = ArrayPool<T>.Shared.Rent(count);
            try
            {
                ReadRawOffsetIntoSpanLegacy(SlotByteOffset(0) + byteSubOffset, destination, count);
                for (int r = 1; r < _worldSize; r++)
                {
                    ReadRawOffsetIntoSpanLegacy(SlotByteOffset(r) + byteSubOffset, peerBuf.AsSpan(0, count), count);
                    for (int i = 0; i < count; i++)
                    {
                        destination[i] = ApplyReductionOperation(destination[i], peerBuf[i], operation);
                    }
                }

                if (operation == ReductionOperation.Average)
                {
                    T divisor = NumOps.FromDouble(_worldSize);
                    for (int i = 0; i < count; i++)
                    {
                        destination[i] = NumOps.Divide(destination[i], divisor);
                    }
                }
            }
            finally
            {
                ArrayPool<T>.Shared.Return(peerBuf);
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

    private static void CombineFloatArraysInPlaceLegacy(float[] acc, float[] peer, int count, ReductionOperation op)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated &&
            (op == ReductionOperation.Sum || op == ReductionOperation.Average || op == ReductionOperation.Product ||
             op == ReductionOperation.Min || op == ReductionOperation.Max))
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

    private static void ScaleFloatArrayInPlaceLegacy(float[] acc, int count, float factor)
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

    private static void CombineDoubleArraysInPlaceLegacy(double[] acc, double[] peer, int count, ReductionOperation op)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated &&
            (op == ReductionOperation.Sum || op == ReductionOperation.Average || op == ReductionOperation.Product ||
             op == ReductionOperation.Min || op == ReductionOperation.Max))
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

    private static void ScaleDoubleArrayInPlaceLegacy(double[] acc, int count, double factor)
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
#endif

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
