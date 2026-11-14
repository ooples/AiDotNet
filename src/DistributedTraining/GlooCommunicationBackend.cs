using System;
using System.Buffers;
#if !AIDN_LEGACY_TFM
using System.Buffers.Binary;
using System.Numerics.Tensors;
#endif
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Gloo-based communication backend for CPU-based collective operations.
/// </summary>
/// <remarks>
/// <para><b>Overview:</b>
/// Gloo is Facebook's collective communications library optimized for both CPUs and GPUs.
/// It provides efficient implementations of collective operations for CPU-based training
/// or heterogeneous environments. Gloo is particularly well-suited for training on CPUs
/// or mixed CPU/GPU clusters where NCCL may not be available or optimal.
/// </para>
/// <para><b>Features:</b>
/// - CPU-optimized collective operations
/// - Supports TCP, InfiniBand via ibverbs
/// - Works on both CPUs and GPUs
/// - Cross-platform (Linux, macOS, Windows)
/// - Used by PyTorch's distributed package
/// </para>
/// <para><b>Use Cases:</b>
/// - CPU-based distributed training
/// - Heterogeneous clusters (mixed CPU/GPU)
/// - When NCCL is not available (non-NVIDIA hardware, macOS, etc.)
/// - Development and testing on laptops/workstations
/// - Production training on CPU clusters
/// </para>
/// <para><b>Requirements:</b>
/// - Gloo library (C++)
/// - .NET bindings for Gloo (custom P/Invoke or wrapper library)
/// - Network connectivity between workers (TCP/IP or InfiniBand)
/// </para>
/// <para><b>Architecture:</b>
/// This backend supports two modes of operation:
///
/// 1. **Native Gloo Mode (Optional):**
///    Requires GlooSharp package (separate NuGet) which provides .NET bindings for the
///    native Gloo C++ library. Gloo offers optimized collective operations for CPU and GPU.
///    To use: Install the GlooSharp package separately.
///
/// 2. **Built-in TCP Mode (Default, Production-Ready):**
///    Production-ready TCP-based implementation using industry-standard ring algorithms
///    (ring-allreduce, ring-allgather, ring-reduce-scatter). Provides full multi-process
///    functionality without external dependencies.
///
/// The TCP implementation features:
/// - Automatic TCP connection initialization with retry logic and handshakes
/// - Ring-based collective operations for optimal bandwidth utilization
/// - Zero-LOH pooled buffers, direct span socket I/O, and SIMD reductions
/// - Proper error handling, validation, and timeout mechanisms
/// - Environment-based rendezvous (AIDOTNET_MASTER_ADDR, AIDOTNET_MASTER_PORT)
/// - Support for arbitrary world sizes and fault-tolerant connection establishment
///
/// **Environment Variables:**
/// - AIDOTNET_GLOO_TRANSPORT: Transport to use ("tcp" or "ib"/"infiniband"). Default: "tcp".
/// - AIDOTNET_GLOO_IB_DEVICE: InfiniBand device name (only when transport is "ib").
/// - AIDOTNET_GLOO_STORE_PATH: Filesystem path for Gloo rendezvous/store coordination.
/// - AIDOTNET_MASTER_ADDR: IP address of rank 0 for TCP rendezvous.
/// - AIDOTNET_MASTER_PORT: Base port number for TCP connections (each rank uses port + rank).
///
/// **Recommendation:** Use TCP mode for most scenarios. Add GlooSharp only if you need
/// specialized hardware support (InfiniBand) or have specific Gloo optimizations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> GlooCommunicationBackend provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class GlooCommunicationBackend<T> : CommunicationBackendBase<T>
{
    private const int TcpSocketBufferBytes = 4 * 1024 * 1024; // 4 MiB send/receive buffers
    private const int MaxDirectSpanWriteBytes = 64 * 1024 * 1024; // 64 MiB per span chunk (avoids int32 byte-length overflow)
    private const int LegacyStagingBufferBytes = 256 * 1024; // 256 KiB staging on legacy TFMs
    private const int GenericDoubleBatchElements = 32768; // 256 KiB pooled double[] batch for generic T

    private readonly int _rank;
    private readonly int _worldSize;
    private bool _useNativeTCP;
    private bool _useNativeGloo;
    private Dictionary<int, TcpClient>? _tcpConnections;
    private TcpListener? _tcpListener;
    private readonly object _connectionLock = new object();
    private object[]? _peerSendLocks;
    private object[]? _peerReceiveLocks;
    private readonly T[] _barrierSendSlot;
    private readonly T[] _barrierReceiveSlot;
    private readonly T[] _scatterLengthSendSlot;
    private readonly T[] _scatterLengthReceiveSlot;
    private T[]? _ringRecvScratch;
    private T[]? _reduceScatterWorkScratch;
    private PersistentSendWorker? _sendWorker;

    // Native Gloo objects (loaded via reflection from GlooSharp package)
    private IDisposable? _nativeGlooContext;
    private Type? _glooCollectivesType;
    private Type? _glooTransportType;
    private MethodInfo? _glooAllReduceDouble;
    private MethodInfo? _glooBroadcastDouble;
    private MethodInfo? _glooAllGatherDouble;
    private MethodInfo? _glooBarrier;
    private MethodInfo? _glooReduceScatterDouble;
    private IReadOnlyDictionary<ReductionOperation, object>? _glooReduceOpCache;

    // Strongly-typed delegates for hot-path collective operations (avoids MethodInfo.Invoke overhead)
    private Action<object, double[], object>? _allReduceDelegate;
    private Action<object, double[], int>? _broadcastDelegate;
    private Func<object, double[], double[]?>? _allGatherDelegate;
    private Action<object>? _barrierDelegate;
    private Func<object, double[], object, double[]?>? _reduceScatterDelegate;

    /// <summary>
    /// Creates a new Gloo communication backend using production-ready TCP implementation.
    /// </summary>
    /// <param name="rank">This process's rank</param>
    /// <param name="worldSize">Total number of processes</param>
    /// <remarks>
    /// This backend uses TCP-based collective operations via ring algorithms.
    /// For native Gloo library support with InfiniBand, see GitHub issue #461.
    /// </remarks>
    public GlooCommunicationBackend(int rank = 0, int worldSize = 1)
    {
        _rank = rank;
        _worldSize = worldSize;
        _useNativeTCP = false;
        _barrierSendSlot = new[] { NumOps.One };
        _barrierReceiveSlot = new T[1];
        _scatterLengthSendSlot = new T[1];
        _scatterLengthReceiveSlot = new T[1];
    }

    /// <inheritdoc/>
    public override int Rank => _rank;

    /// <inheritdoc/>
    public override int WorldSize => _worldSize;

    /// <inheritdoc/>
    protected override void OnInitialize()
    {
        _tcpConnections = new Dictionary<int, TcpClient>();

        // Single-process mode: No communication infrastructure needed
        if (_worldSize == 1)
        {
            _useNativeTCP = false;
            _useNativeGloo = false;
            Console.WriteLine("GlooCommunicationBackend: Single-process mode (worldSize=1).");
            return;
        }

        // Try to detect GlooSharp native package
        if (TryInitializeNativeGloo())
        {
            _useNativeGloo = true;
            // Also initialize TCP for point-to-point Send/Receive (Gloo only handles collectives)
            _useNativeTCP = true;
            try
            {
                InitializeTCPConnections();
                Console.WriteLine($"GlooCommunicationBackend: Using native Gloo via GlooSharp for collectives, TCP for point-to-point ({_worldSize} processes).");
                return;
            }
            catch (SocketException)
            {
                // Clean up ALL resources on failure: native Gloo context + TCP connections
                CleanupOnInitFailure();
                throw;
            }
            catch (InvalidOperationException)
            {
                // Clean up ALL resources on failure: native Gloo context + TCP connections
                CleanupOnInitFailure();
                throw;
            }
        }

        // Fall back to production-ready TCP implementation
        _useNativeGloo = false;
        _useNativeTCP = true;
        Console.WriteLine($"GlooCommunicationBackend: Using production-ready TCP implementation for {_worldSize} processes.");
        InitializeTCPConnections();
    }

    /// <summary>
    /// Attempts to detect and initialize the GlooSharp native package via reflection.
    /// </summary>
    /// <remarks>
    /// Uses the same pattern as NCCL/MPI backends: load GlooSharp types via reflection
    /// so AiDotNet does not have a hard dependency on the GlooSharp NuGet package.
    /// If GlooSharp is not installed, this method returns false and the backend falls
    /// back to the built-in TCP implementation.
    /// </remarks>
    /// <returns><c>true</c> if GlooSharp was detected and initialized successfully.</returns>
    private bool TryInitializeNativeGloo()
    {
        // Pre-validate all required types BEFORE creating any IDisposable objects
        var glooContextType = Type.GetType("GlooSharp.GlooContext, GlooSharp");
        if (glooContextType is null)
        {
            return false;
        }

        _glooCollectivesType = Type.GetType("GlooSharp.GlooCollectives, GlooSharp");
        _glooTransportType = Type.GetType("GlooSharp.GlooTransport, GlooSharp");
        if (_glooCollectivesType is null || _glooTransportType is null)
        {
            return false;
        }

        var glooReduceOpType = Type.GetType("GlooSharp.GlooReduceOp, GlooSharp");
        if (glooReduceOpType is null)
        {
            Console.WriteLine("GlooSharp detection failed: GlooReduceOp type not found.");
            return false;
        }

        // Create GlooContext(rank, worldSize) — this may return an IDisposable
        var contextInstance = Activator.CreateInstance(glooContextType, _rank, _worldSize);
        if (contextInstance is null)
        {
            Console.WriteLine("GlooSharp detection failed: Activator.CreateInstance returned null for GlooContext.");
            return false;
        }

        // Safe cast to IDisposable — if GlooContext doesn't implement IDisposable,
        // fall back gracefully instead of throwing InvalidCastException.
        if (contextInstance is not IDisposable context)
        {
            Console.WriteLine("GlooSharp detection failed: GlooContext does not implement IDisposable. Incompatible GlooSharp version.");
            return false;
        }

        // From here on, 'context' is IDisposable and MUST be disposed on all error paths
        return TryConfigureNativeGloo(context, glooContextType, glooReduceOpType);
    }

    /// <summary>
    /// Configures native Gloo transport, caches method handles, and transfers ownership of context on success.
    /// Disposes context on any failure path.
    /// </summary>
    private bool TryConfigureNativeGloo(IDisposable context, Type glooContextType, Type glooReduceOpType)
    {
        try
        {
            // Set up transport based on environment variable
            string transportEnv = Environment.GetEnvironmentVariable("AIDOTNET_GLOO_TRANSPORT") ?? "tcp";
            string? storePath = Environment.GetEnvironmentVariable("AIDOTNET_GLOO_STORE_PATH");

            if (string.Equals(transportEnv, "ib", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(transportEnv, "infiniband", StringComparison.OrdinalIgnoreCase))
            {
                string ibDevice = Environment.GetEnvironmentVariable("AIDOTNET_GLOO_IB_DEVICE") ?? "";
                var createIB = _glooTransportType?.GetMethod("CreateInfiniBand",
                    new[] { glooContextType, typeof(string), typeof(string) });
                if (createIB is null)
                {
                    Console.WriteLine("GlooSharp detection failed: CreateInfiniBand method not found.");
                    return false;
                }
                createIB.Invoke(null, new object?[] { context, ibDevice, storePath });
                Console.WriteLine($"GlooCommunicationBackend: InfiniBand transport initialized (device={ibDevice}).");
            }
            else
            {
                string hostname = Environment.GetEnvironmentVariable("AIDOTNET_MASTER_ADDR") ?? "localhost";
                string portStr = Environment.GetEnvironmentVariable("AIDOTNET_MASTER_PORT") ?? "29500";
                int port = int.TryParse(portStr, out int p) ? p : 29500;

                var createTCP = _glooTransportType?.GetMethod("CreateTCP",
                    new[] { glooContextType, typeof(string), typeof(int), typeof(string) });
                if (createTCP is null)
                {
                    Console.WriteLine("GlooSharp detection failed: CreateTCP method not found.");
                    return false;
                }
                createTCP.Invoke(null, new object?[] { context, hostname, port, storePath });
            }

            // Cache reflection MethodInfo for collective operations (double[] overloads)
            _glooAllReduceDouble = _glooCollectivesType?.GetMethod("AllReduce", new[]
            {
                glooContextType, typeof(double[]), glooReduceOpType
            });
            _glooBroadcastDouble = _glooCollectivesType?.GetMethod("Broadcast", new[]
            {
                glooContextType, typeof(double[]), typeof(int)
            });
            _glooAllGatherDouble = _glooCollectivesType?.GetMethod("AllGather", new[]
            {
                glooContextType, typeof(double[])
            });
            _glooBarrier = _glooCollectivesType?.GetMethod("Barrier", new[] { glooContextType });
            _glooReduceScatterDouble = _glooCollectivesType?.GetMethod("ReduceScatter", new[]
            {
                glooContextType, typeof(double[]), glooReduceOpType
            });

            // Validate all required collective methods were found
            if (_glooAllReduceDouble is null || _glooBroadcastDouble is null ||
                _glooAllGatherDouble is null || _glooBarrier is null ||
                _glooReduceScatterDouble is null)
            {
                Console.WriteLine("GlooSharp detected but required collective methods not found. Falling back to TCP.");
                return false;
            }

            // Cache GlooReduceOp enum values to avoid per-call reflection
            _glooReduceOpCache = CreateGlooReduceOpMap(glooReduceOpType);

            // Create strongly-typed delegates from MethodInfo to avoid Invoke overhead on hot path
            _allReduceDelegate = (ctx, data, op) =>
                _glooAllReduceDouble.Invoke(null, new[] { ctx, data, op });
            _broadcastDelegate = (ctx, data, root) =>
                _glooBroadcastDouble.Invoke(null, new object[] { ctx, data, root });
            _allGatherDelegate = (ctx, data) =>
                _glooAllGatherDouble.Invoke(null, new[] { ctx, data }) as double[];
            _barrierDelegate = (ctx) =>
                _glooBarrier.Invoke(null, new[] { ctx });
            _reduceScatterDelegate = (ctx, data, op) =>
                _glooReduceScatterDouble.Invoke(null, new[] { ctx, data, op }) as double[];

            // Success — transfer ownership to the field (do NOT dispose)
            _nativeGlooContext = context;
            return true;
        }
        catch (TargetInvocationException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (invocation error): {(ex.InnerException ?? ex)}");
            return false;
        }
        catch (TypeLoadException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (type load error): {ex}");
            return false;
        }
        catch (InvalidOperationException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed: {ex}");
            return false;
        }
        catch (MissingMethodException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (API signature mismatch): {ex}");
            return false;
        }
        catch (MemberAccessException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (access error): {ex}");
            return false;
        }
        catch (ArgumentException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (enum mismatch): {ex}");
            return false;
        }
        catch (SocketException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (transport error): {ex}");
            return false;
        }
        catch (TimeoutException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (transport timeout): {ex}");
            return false;
        }
        catch (IOException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (transport I/O error): {ex}");
            return false;
        }
        catch (NotSupportedException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"GlooSharp detection failed (not supported): {ex}");
            return false;
        }
        finally
        {
            // If we didn't transfer ownership (i.e., _nativeGlooContext was not set), dispose
            if (_nativeGlooContext != context)
            {
                context.Dispose();
            }
        }
    }

    /// <summary>
    /// Creates a cached mapping from <see cref="ReductionOperation"/> to GlooSharp GlooReduceOp enum values.
    /// </summary>
    private static IReadOnlyDictionary<ReductionOperation, object> CreateGlooReduceOpMap(Type glooReduceOpType)
    {
        // Note: Average is NOT mapped here because the call sites already handle Average
        // by mapping it to Sum and dividing by worldSize after the collective operation.
        return new Dictionary<ReductionOperation, object>
        {
            { ReductionOperation.Sum, Enum.Parse(glooReduceOpType, "Sum", ignoreCase: true) },
            { ReductionOperation.Product, Enum.Parse(glooReduceOpType, "Product", ignoreCase: true) },
            { ReductionOperation.Min, Enum.Parse(glooReduceOpType, "Min", ignoreCase: true) },
            { ReductionOperation.Max, Enum.Parse(glooReduceOpType, "Max", ignoreCase: true) },
        };
    }

    /// <summary>
    /// Maps an AiDotNet <see cref="ReductionOperation"/> to the cached GlooSharp GlooReduceOp enum value.
    /// Uses the pre-computed cache populated during <see cref="TryInitializeNativeGloo"/> to avoid
    /// per-call reflection overhead on the hot path.
    /// </summary>
    private object MapToGlooReduceOp(ReductionOperation operation)
    {
        if (_glooReduceOpCache == null)
        {
            throw new InvalidOperationException("Gloo reduce op cache not initialized. Ensure TryInitializeNativeGloo() succeeded.");
        }

        if (!_glooReduceOpCache.TryGetValue(operation, out var glooValue))
        {
            throw new ArgumentException($"Unsupported reduction operation: {operation}");
        }

        return glooValue;
    }

    /// <summary>
    /// Initializes TCP connections between all ranks.
    /// </summary>
    /// <remarks>
    /// Uses environment variables for rendezvous:
    /// - AIDOTNET_MASTER_ADDR: IP address of rank 0
    /// - AIDOTNET_MASTER_PORT: Base port for rank 0
    /// Each rank listens on MASTER_PORT + rank
    /// </remarks>
    private void InitializeTCPConnections()
    {
        string? masterAddr = Environment.GetEnvironmentVariable("AIDOTNET_MASTER_ADDR");
        string? masterPortStr = Environment.GetEnvironmentVariable("AIDOTNET_MASTER_PORT");

        if (string.IsNullOrEmpty(masterAddr) || string.IsNullOrEmpty(masterPortStr))
        {
            throw new InvalidOperationException(
                "GlooCommunicationBackend TCP mode requires environment variables:\n" +
                "- AIDOTNET_MASTER_ADDR: IP address of rank 0 (e.g., 192.168.1.10 or localhost)\n" +
                "- AIDOTNET_MASTER_PORT: Base port number (e.g., 29500)\n" +
                "Each rank will use port = MASTER_PORT + rank");
        }

        if (!int.TryParse(masterPortStr, out int basePort))
        {
            throw new InvalidOperationException($"Invalid AIDOTNET_MASTER_PORT: {masterPortStr}");
        }

        _peerSendLocks = new object[_worldSize];
        _peerReceiveLocks = new object[_worldSize];
        for (int r = 0; r < _worldSize; r++)
        {
            _peerSendLocks[r] = new object();
            _peerReceiveLocks[r] = new object();
        }

        // Start TCP listener on this rank's port.
        // Set SO_RCVBUF / SO_SNDBUF on the listening socket BEFORE Start() (listen()) so Windows Winsock
        // negotiates RFC 1323 TCP Window Scaling during the SYN/SYN-ACK handshake on accepted sockets.
        int myPort = basePort + _rank;
        _tcpListener = new TcpListener(IPAddress.Any, myPort);
        _tcpListener.Server.ReceiveBufferSize = TcpSocketBufferBytes;
        _tcpListener.Server.SendBufferSize = TcpSocketBufferBytes;
        _tcpListener.Start();
        Console.WriteLine($"Rank {_rank}: TCP listener started on port {myPort}");

        // Connect to all ranks with lower rank numbers (they are already listening)
        for (int otherRank = 0; otherRank < _rank; otherRank++)
        {
            ConnectToRank(otherRank, masterAddr, basePort);
        }

        // Accept connections from all ranks with higher rank numbers
        // IMPORTANT: We cannot predict the order in which higher ranks will connect,
        // so we accept ANY connection and verify the rank from the handshake.
        int numExpectedConnections = _worldSize - _rank - 1;
        for (int i = 0; i < numExpectedConnections; i++)
        {
            AcceptConnectionFromAnyRank();
        }

        _sendWorker?.Dispose();
        _sendWorker = new PersistentSendWorker(this);

        Console.WriteLine($"Rank {_rank}: All TCP connections established ({_tcpConnections?.Count ?? 0} peers)");
    }

    /// <summary>
    /// Cleans up all resources (native Gloo context + TCP connections) on initialization failure.
    /// Prevents resource leaks when InitializeTCPConnections() throws after partial setup.
    /// </summary>
    private void CleanupOnInitFailure()
    {
        _useNativeGloo = false;
        _useNativeTCP = false;

        _nativeGlooContext?.Dispose();
        _nativeGlooContext = null;

        if (_tcpConnections is not null)
        {
            foreach (var connection in _tcpConnections.Values)
            {
                try { connection.Close(); }
                catch (ObjectDisposedException ex)
                {
                    Debug.WriteLine($"TCP connection already disposed during cleanup: {ex.Message}");
                }
            }
            _tcpConnections.Clear();
        }

        if (_tcpListener is not null)
        {
            try { _tcpListener.Stop(); }
            catch (ObjectDisposedException ex)
            {
                Debug.WriteLine($"TCP listener already disposed during cleanup: {ex.Message}");
            }
            _tcpListener = null;
        }

        Interlocked.Exchange(ref _sendWorker, null)?.Dispose();
    }

    /// <summary>
    /// Connects to a specific rank (active connection).
    /// </summary>
    private void ConnectToRank(int targetRank, string masterAddr, int basePort)
    {
        int targetPort = basePort + targetRank;
        int maxRetries = 10;
        int retryDelayMs = 1000;

        for (int attempt = 0; attempt < maxRetries; attempt++)
        {
            TcpClient? client = null;
            try
            {
                client = new TcpClient
                {
                    NoDelay = true,
                    SendBufferSize = TcpSocketBufferBytes,
                    ReceiveBufferSize = TcpSocketBufferBytes,
                };
                client.Connect(masterAddr, targetPort);

                // Send handshake: my rank (4-byte little-endian int32, without BinaryWriter stream buffering)
                var stream = client.GetStream();
                WriteInt32LittleEndian(stream, _rank);
                stream.Flush();
                lock (_connectionLock)
                {
                    if (_tcpConnections != null)
                    {
                        _tcpConnections[targetRank] = client;
                    }
                }

                Console.WriteLine($"Rank {_rank}: Connected to rank {targetRank} at {masterAddr}:{targetPort}");
                return;
            }
            catch (SocketException)
            {
                client?.Close();
                if (attempt < maxRetries - 1)
                {
                    Thread.Sleep(retryDelayMs);
                }
                else
                {
                    throw new InvalidOperationException(
                        $"Rank {_rank}: Failed to connect to rank {targetRank} at {masterAddr}:{targetPort} after {maxRetries} attempts");
                }
            }
        }
    }

    /// <summary>
    /// Accepts connection from any rank (passive connection).
    /// The actual rank is determined from the handshake, not predetermined.
    /// This avoids connection-order race conditions where ranks connect in unpredictable order.
    /// </summary>
    private void AcceptConnectionFromAnyRank()
    {
        if (_tcpListener == null)
        {
            throw new InvalidOperationException("TCP listener not initialized");
        }

        // Accept incoming connection from ANY rank
        var client = _tcpListener.AcceptTcpClient();
        client.NoDelay = true;
        client.SendBufferSize = TcpSocketBufferBytes;
        client.ReceiveBufferSize = TcpSocketBufferBytes;

        // Read handshake to identify the connecting rank (exact 4 bytes, zero read-ahead buffering)
        var stream = client.GetStream();
        int receivedRank = ReadInt32LittleEndian(stream);
        // Verify the connecting rank is valid (should be a higher rank than us)
        if (receivedRank <= _rank)
        {
            client.Close();
            throw new InvalidOperationException(
                $"Rank {_rank}: Received unexpected connection from rank {receivedRank}. " +
                $"Expected connections only from ranks {_rank + 1} to {_worldSize - 1}.");
        }

        if (receivedRank >= _worldSize)
        {
            client.Close();
            throw new InvalidOperationException(
                $"Rank {_rank}: Received connection from invalid rank {receivedRank}. " +
                $"World size is {_worldSize} (valid ranks: 0 to {_worldSize - 1}).");
        }

        lock (_connectionLock)
        {
            if (_tcpConnections != null)
            {
                // Check if we already have a connection from this rank (duplicate connection)
                if (_tcpConnections.ContainsKey(receivedRank))
                {
                    client.Close();
                    throw new InvalidOperationException(
                        $"Rank {_rank}: Duplicate connection attempt from rank {receivedRank}");
                }

                _tcpConnections[receivedRank] = client;
            }
        }

        Console.WriteLine($"Rank {_rank}: Accepted connection from rank {receivedRank}");
    }

    /// <inheritdoc/>
    protected override void OnShutdown()
    {
        // Dispose native Gloo context if active
        if (_useNativeGloo && _nativeGlooContext != null)
        {
            try
            {
                _nativeGlooContext.Dispose();
            }
            catch (ObjectDisposedException)
            {
                // Already disposed, safe to ignore
            }
            catch (InvalidOperationException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Warning: Error disposing native Gloo context: {ex}");
            }
            _nativeGlooContext = null;
            _useNativeGloo = false;
        }

        if (_useNativeTCP && _tcpConnections != null)
        {
            lock (_connectionLock)
            {
                foreach (var connection in _tcpConnections.Values)
                {
                    try
                    {
                        connection.Close();
                    }
                    catch (ObjectDisposedException)
                    {
                        // Already disposed, safe to ignore
                    }
                    catch (IOException ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Warning: Error closing TCP connection: {ex}");
                    }
                }
                _tcpConnections.Clear();
            }

            if (_tcpListener != null)
            {
                try
                {
                    _tcpListener.Stop();
                }
                catch (SocketException ex)
                {
                    System.Diagnostics.Trace.TraceWarning($"Warning: Error stopping TCP listener: {ex}");
                }
                catch (ObjectDisposedException)
                {
                    // Already disposed, safe to ignore
                }
                _tcpListener = null;
            }
        }

        Interlocked.Exchange(ref _sendWorker, null)?.Dispose();
        Interlocked.Exchange(ref _ringRecvScratch, null);
        Interlocked.Exchange(ref _reduceScatterWorkScratch, null);
    }

    private static T[] AcquirePersistentScratch(ref T[]? slot, int minimumLength)
    {
        T[]? existing = Interlocked.Exchange(ref slot, null);
        if (existing is not null && existing.Length >= minimumLength)
        {
            return existing;
        }

        return AllocateExactOutputArray(Math.Max(1, minimumLength));
    }

    private static void ReleasePersistentScratch(ref T[]? slot, T[] buffer)
    {
        while (true)
        {
            T[]? current = Volatile.Read(ref slot);
            if (current is not null && current.Length >= buffer.Length)
            {
                return;
            }

            if (Interlocked.CompareExchange(ref slot, buffer, current) == current)
            {
                return;
            }
        }
    }

    /// <inheritdoc/>
    public override void Barrier()
    {
        EnsureInitialized();
        DrainAsyncCollectives();
        long startTicks = Stopwatch.GetTimestamp();

        if (_worldSize == 1)
        {
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return;
        }

        if (_useNativeGloo)
        {
            if (_barrierDelegate == null || _nativeGlooContext == null)
            {
                throw new InvalidOperationException("Native Gloo Barrier method is not available.");
            }
            _barrierDelegate(_nativeGlooContext);
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return;
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("Barrier requires TCP mode to be initialized");
        }

        for (int otherRank = 0; otherRank < _worldSize; otherRank++)
        {
            if (otherRank != _rank)
            {
                SendArraySlice(otherRank, _barrierSendSlot, 0, 1);
            }
        }

        for (int otherRank = 0; otherRank < _worldSize; otherRank++)
        {
            if (otherRank != _rank)
            {
                ReceiveArrayInto(otherRank, _barrierReceiveSlot, 0, expectedLength: 1);
            }
        }

        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
    }

    /// <inheritdoc/>
    public override void AllReduce(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));
        DrainAsyncCollectives();
        long startTicks = Stopwatch.GetTimestamp();

        if (_worldSize == 1)
        {
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return;
        }

        if (_useNativeGloo)
        {
            PerformNativeGlooAllReduce(data, operation);
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return;
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("AllReduce requires TCP mode to be initialized");
        }

        PerformRingAllReduce(data, operation);
        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
    }

    /// <inheritdoc/>
    public override Vector<T> AllGather(Vector<T> sendData)
    {
        EnsureInitialized();
        ValidateData(sendData, nameof(sendData));
        DrainAsyncCollectives();
        long startTicks = Stopwatch.GetTimestamp();

        if (_worldSize == 1)
        {
            var single = sendData.Clone();
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return single;
        }

        if (_useNativeGloo)
        {
            var nativeRes = PerformNativeGlooAllGather(sendData);
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return nativeRes;
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("AllGather requires TCP mode to be initialized");
        }

        var gathered = PerformRingAllGather(sendData);
        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
        return gathered;
    }

    /// <inheritdoc/>
    public override Vector<T> Broadcast(Vector<T> data, int root = 0)
    {
        EnsureInitialized();
        ValidateRoot(root);
        if (_rank == root || _worldSize == 1)
        {
            ValidateData(data, nameof(data));
        }

        DrainAsyncCollectives();
        long startTicks = Stopwatch.GetTimestamp();

        if (_worldSize == 1)
        {
            var single = data.Clone();
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return single;
        }

        if (_useNativeGloo)
        {
            var nativeRes = PerformNativeGlooBroadcast(data, root);
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return nativeRes;
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("Broadcast requires TCP mode to be initialized");
        }

        var broadcasted = PerformTreeBroadcast(data, root);
        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
        return broadcasted;
    }

    /// <inheritdoc/>
    public override Vector<T> Scatter(Vector<T> sendData, int root = 0)
    {
        EnsureInitialized();
        ValidateRoot(root);
        DrainAsyncCollectives();
        long startTicks = Stopwatch.GetTimestamp();

        if (_worldSize == 1)
        {
            if (Rank == root)
            {
                ValidateData(sendData, nameof(sendData));
                var single = sendData.Clone();
                RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
                return single;
            }
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return new Vector<T>(Array.Empty<T>());
        }

        if (_useNativeGloo)
        {
            int totalLength;
            if (Rank == root)
            {
                ValidateData(sendData, nameof(sendData));
                if (sendData.Length % _worldSize != 0)
                {
                    throw new ArgumentException(
                        $"Data length {sendData.Length} must be divisible by world size {_worldSize}.");
                }
                totalLength = sendData.Length;
            }
            else
            {
                totalLength = 0;
            }

            var lengthVector = PerformNativeGlooBroadcast(
                new Vector<T>(new[] { NumOps.FromDouble(totalLength) }),
                root);
            totalLength = (int)NumOps.ToDouble(lengthVector[0]);

            if (totalLength % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"Data length {totalLength} must be divisible by world size {_worldSize}.");
            }

            var fullBuffer = Rank == root ? sendData : new Vector<T>(new T[totalLength]);
            var broadcasted = PerformNativeGlooBroadcast(fullBuffer, root);
            int chunkSize = broadcasted.Length / _worldSize;
            var chunk = new T[chunkSize];
            broadcasted.AsSpanUnmarked().Slice(_rank * chunkSize, chunkSize).CopyTo(chunk);
            GC.KeepAlive(broadcasted);
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return new Vector<T>(chunk);
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("Scatter requires TCP mode to be initialized");
        }

        var scattered = PerformTreeScatter(sendData, root);
        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
        return scattered;
    }

    /// <inheritdoc/>
    public override Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));

        if (data.Length % _worldSize != 0)
        {
            throw new ArgumentException(
                $"Data length {data.Length} must be divisible by world size {_worldSize}.");
        }

        DrainAsyncCollectives();
        long startTicks = Stopwatch.GetTimestamp();

        if (_worldSize == 1)
        {
            var single = data.Clone();
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return single;
        }

        if (_useNativeGloo)
        {
            var nativeRes = PerformNativeGlooReduceScatter(data, operation);
            RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
            return nativeRes;
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("ReduceScatter requires TCP mode to be initialized");
        }

        var reducedShard = PerformRingReduceScatter(data, operation);
        RecordCollectiveDurationTicks(Stopwatch.GetTimestamp() - startTicks);
        return reducedShard;
    }

    #region Native Gloo Collective Implementations

    /// <summary>
    /// Converts a <see cref="Vector{T}"/> to a <c>double[]</c> for native Gloo calls.
    /// </summary>
    private double[] VectorToDoubleArray(Vector<T> vector)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = NumOps.ToDouble(vector[i]);
        }
        return result;
    }

    /// <summary>
    /// Converts a <c>double[]</c> back to a <see cref="Vector{T}"/>.
    /// </summary>
    private Vector<T> DoubleArrayToVector(double[] array)
    {
        var result = new T[array.Length];
        for (int i = 0; i < array.Length; i++)
        {
            result[i] = NumOps.FromDouble(array[i]);
        }
        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs AllReduce via native Gloo (GlooSharp).
    /// </summary>
    private void PerformNativeGlooAllReduce(Vector<T> data, ReductionOperation operation)
    {
        if (_allReduceDelegate == null || _nativeGlooContext == null)
        {
            throw new InvalidOperationException("Native Gloo AllReduce method is not available.");
        }

        var doubleData = VectorToDoubleArray(data);
        var glooOp = MapToGlooReduceOp(operation == ReductionOperation.Average ? ReductionOperation.Sum : operation);
        _allReduceDelegate(_nativeGlooContext, doubleData, glooOp);

        // Write results back to the vector
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = NumOps.FromDouble(doubleData[i]);
        }

        // Apply averaging using NumOps (consistent with TCP ring-allreduce path)
        if (operation == ReductionOperation.Average)
        {
            T divisor = NumOps.FromDouble(_worldSize);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = NumOps.Divide(data[i], divisor);
            }
        }
    }

    /// <summary>
    /// Performs Broadcast via native Gloo (GlooSharp).
    /// </summary>
    private Vector<T> PerformNativeGlooBroadcast(Vector<T> data, int root)
    {
        if (_broadcastDelegate == null || _nativeGlooContext == null)
        {
            throw new InvalidOperationException("Native Gloo Broadcast method is not available.");
        }

        // Non-root ranks may pass no buffer, and the native call fills the array it is given in place, so the root's
        // length goes first and every other rank sizes its receive buffer from it.
        double[] doubleData;
        if (_rank == root)
        {
            if (data is null)
            {
                throw new ArgumentNullException(nameof(data), "The broadcast root must supply the data to send.");
            }

            _broadcastDelegate(_nativeGlooContext, new double[] { data.Length }, root);
            doubleData = VectorToDoubleArray(data);
        }
        else
        {
            var length = new double[1];
            _broadcastDelegate(_nativeGlooContext, length, root);
            doubleData = new double[checked((int)length[0])];
        }

        _broadcastDelegate(_nativeGlooContext, doubleData, root);
        return DoubleArrayToVector(doubleData);
    }

    /// <summary>
    /// Performs AllGather via native Gloo (GlooSharp).
    /// </summary>
    private Vector<T> PerformNativeGlooAllGather(Vector<T> sendData)
    {
        if (_allGatherDelegate == null || _nativeGlooContext == null)
        {
            throw new InvalidOperationException("Native Gloo AllGather method is not available.");
        }

        var doubleData = VectorToDoubleArray(sendData);
        var gathered = _allGatherDelegate(_nativeGlooContext, doubleData);
        if (gathered is not null)
        {
            return DoubleArrayToVector(gathered);
        }
        throw new InvalidOperationException("Native Gloo AllGather invocation did not return a double[] result.");
    }

    /// <summary>
    /// Performs ReduceScatter via native Gloo (GlooSharp).
    /// </summary>
    private Vector<T> PerformNativeGlooReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        if (_reduceScatterDelegate == null || _nativeGlooContext == null)
        {
            throw new InvalidOperationException("Native Gloo ReduceScatter method is not available.");
        }

        var doubleData = VectorToDoubleArray(data);
        var glooOp = MapToGlooReduceOp(operation == ReductionOperation.Average ? ReductionOperation.Sum : operation);
        var scattered = _reduceScatterDelegate(_nativeGlooContext, doubleData, glooOp);

        if (scattered is null)
        {
            throw new InvalidOperationException("Native Gloo ReduceScatter invocation did not return a double[] result.");
        }

        var result = DoubleArrayToVector(scattered);

        // Apply averaging using NumOps (consistent with TCP ring-reduce-scatter path)
        if (operation == ReductionOperation.Average)
        {
            T divisor = NumOps.FromDouble(_worldSize);
            var resultArray = result.ToArray();
            for (int i = 0; i < resultArray.Length; i++)
            {
                resultArray[i] = NumOps.Divide(resultArray[i], divisor);
            }
            result = new Vector<T>(resultArray);
        }

        return result;
    }

    #endregion

    /// <summary>
    /// Performs ring-based AllReduce operation directly in-place on <paramref name="data"/>'s backing buffer
    /// using a single pooled receive scratch buffer and SIMD reductions (zero LOH allocations).
    /// </summary>
    private void PerformRingAllReduce(Vector<T> data, ReductionOperation operation)
    {
        int totalLength = data.Length;
        if (_worldSize == 1 || totalLength == 0)
        {
            return;
        }

        int chunkSize = (totalLength + _worldSize - 1) / _worldSize;
        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var writableMemory = data.AsWritableMemoryUnmarked();
        T[] workArray;
        int workBaseOffset;
        T[]? rentedWorkArray = null;
        if (MemoryMarshal.TryGetArray((ReadOnlyMemory<T>)writableMemory, out var segment) && segment.Array is not null)
        {
            workArray = segment.Array;
            workBaseOffset = segment.Offset;
        }
        else
        {
            rentedWorkArray = ArrayPool<T>.Shared.Rent(totalLength);
            writableMemory.Span.CopyTo(rentedWorkArray.AsSpan(0, totalLength));
            workArray = rentedWorkArray;
            workBaseOffset = 0;
        }

        T[] recvScratch = AcquirePersistentScratch(ref _ringRecvScratch, chunkSize);
        try
        {
            // Phase 1: ReduceScatter - reduce chunks in ring pattern
            for (int step = 0; step < _worldSize - 1; step++)
            {
                int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
                int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

                int sendStart = sendChunkIdx * chunkSize;
                int sendCount = Math.Max(0, Math.Min(chunkSize, totalLength - sendStart));
                int recvStart = recvChunkIdx * chunkSize;
                int recvCount = Math.Max(0, Math.Min(chunkSize, totalLength - recvStart));

                SendAndReceiveConcurrent(
                    nextRank, workArray, workBaseOffset + sendStart, sendCount,
                    prevRank, recvScratch, 0, recvCount);

                if (recvCount > 0)
                {
                    ReduceArrayInPlace(workArray, workBaseOffset + recvStart, recvScratch, 0, recvCount, operation);
                }
            }

            // Phase 2: AllGather - distribute reduced chunks directly into non-overlapping workArray slices
            for (int step = 0; step < _worldSize - 1; step++)
            {
                int sendChunkIdx = (_rank - step + 1 + _worldSize) % _worldSize;
                int recvChunkIdx = (_rank - step + _worldSize) % _worldSize;

                int sendStart = sendChunkIdx * chunkSize;
                int sendCount = Math.Max(0, Math.Min(chunkSize, totalLength - sendStart));
                int recvStart = recvChunkIdx * chunkSize;
                int recvCount = Math.Max(0, Math.Min(chunkSize, totalLength - recvStart));

                SendAndReceiveConcurrent(
                    nextRank, workArray, workBaseOffset + sendStart, sendCount,
                    prevRank, workArray, workBaseOffset + recvStart, recvCount);
            }

            if (operation == ReductionOperation.Average)
            {
                ScaleArrayByReciprocalWorldSize(workArray, workBaseOffset, totalLength, _worldSize);
            }

            if (rentedWorkArray is not null)
            {
                rentedWorkArray.AsSpan(0, totalLength).CopyTo(data.AsWritableSpanUnmarked());
            }
        }
        finally
        {
            ReleasePersistentScratch(ref _ringRecvScratch, recvScratch);
            if (rentedWorkArray is not null)
            {
                ArrayPool<T>.Shared.Return(rentedWorkArray);
            }

            GC.KeepAlive(data);
        }
    }

    private PersistentSendWorker GetOrCreateSendWorker()
    {
        var worker = Volatile.Read(ref _sendWorker);
        if (worker is not null)
        {
            return worker;
        }

        var created = new PersistentSendWorker(this);
        var existing = Interlocked.CompareExchange(ref _sendWorker, created, null);
        if (existing is not null)
        {
            created.Dispose();
            return existing;
        }

        return created;
    }

    private sealed class PersistentSendWorker : IDisposable
    {
        private readonly GlooCommunicationBackend<T> _owner;
        private readonly object _dispatchLock = new object();
        private readonly AutoResetEvent _workAvailable = new AutoResetEvent(false);
        private readonly AutoResetEvent _workCompleted = new AutoResetEvent(false);
        private readonly Thread _thread;
        private volatile bool _stop;
        private int _disposed;
        private int _destRank;
        private T[]? _buffer;
        private int _offset;
        private int _count;
        private Exception? _exception;

        public PersistentSendWorker(GlooCommunicationBackend<T> owner)
        {
            _owner = owner;
            _thread = new Thread(RunLoop)
            {
                IsBackground = true,
                Name = $"AiDotNet.GlooSend.{owner._rank}",
            };
            _thread.Start();
        }

        public void BeginSend(int destRank, T[] buffer, int offset, int count)
        {
            Monitor.Enter(_dispatchLock);
            try
            {
                if (_stop)
                {
                    throw new ObjectDisposedException(nameof(PersistentSendWorker), "Gloo background send worker has been shut down.");
                }

                _destRank = destRank;
                _buffer = buffer;
                _offset = offset;
                _count = count;
                _exception = null;
                _workAvailable.Set();
            }
            catch
            {
                Monitor.Exit(_dispatchLock);
                throw;
            }
        }

        public void EndSend(bool suppressException)
        {
            Exception? sendEx;
            try
            {
                _workCompleted.WaitOne();
                sendEx = _exception;
                _exception = null;
            }
            finally
            {
                Monitor.Exit(_dispatchLock);
            }

            if (sendEx is not null)
            {
                if (suppressException)
                {
                    Debug.WriteLine($"Concurrent send faulted after receive error: {sendEx.Message}");
                }
                else
                {
                    ExceptionDispatchInfo.Capture(sendEx).Throw();
                }
            }
        }

        private void RunLoop()
        {
            while (true)
            {
                _workAvailable.WaitOne();
                if (_stop)
                {
                    _workCompleted.Set();
                    return;
                }

                T[]? buf = _buffer;
                int dest = _destRank;
                int off = _offset;
                int cnt = _count;
                _buffer = null;

                Exception? err = null;
                try
                {
                    if (buf is not null)
                    {
                        _owner.SendArraySlice(dest, buf, off, cnt);
                    }
                }
                catch (Exception ex)
                {
                    err = ex;
                }

                _exception = err;
                _workCompleted.Set();
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            _stop = true;
            try
            {
                _workAvailable.Set();
            }
            catch (ObjectDisposedException)
            {
            }

            bool joined = !_thread.IsAlive || _thread.Join(millisecondsTimeout: 5000);
            if (joined)
            {
                _workAvailable.Dispose();
                _workCompleted.Dispose();
            }
        }
    }

    private void SendAndReceiveConcurrent(
        int sendRank,
        T[] sendBuffer,
        int sendOffset,
        int sendCount,
        int receiveRank,
        T[] receiveBuffer,
        int receiveOffset,
        int receiveCount)
    {
        var sendWorker = GetOrCreateSendWorker();
        sendWorker.BeginSend(sendRank, sendBuffer, sendOffset, sendCount);
        Exception? receiveException = null;
        try
        {
            ReceiveArrayInto(receiveRank, receiveBuffer, receiveOffset, expectedLength: receiveCount);
        }
        catch (Exception ex)
        {
            receiveException = ex;
        }
        finally
        {
            sendWorker.EndSend(suppressException: receiveException is not null);
        }

        if (receiveException is not null)
        {
            ExceptionDispatchInfo.Capture(receiveException).Throw();
        }
    }

    private void ReduceArrayInPlace(T[] dst, int dstOffset, T[] src, int srcOffset, int count, ReductionOperation operation)
    {
        if (count <= 0)
        {
            return;
        }

        if (typeof(T) == typeof(float))
        {
            var dstF = (float[])(object)dst;
            var srcF = (float[])(object)src;
#if !AIDN_LEGACY_TFM
            var dstSpan = dstF.AsSpan(dstOffset, count);
            ReadOnlySpan<float> srcSpan = srcF.AsSpan(srcOffset, count);
            switch (operation)
            {
                case ReductionOperation.Sum:
                case ReductionOperation.Average:
                    TensorPrimitives.Add(dstSpan, srcSpan, dstSpan);
                    return;
                case ReductionOperation.Product:
                    TensorPrimitives.Multiply(dstSpan, srcSpan, dstSpan);
                    return;
                case ReductionOperation.Min:
                    TensorPrimitives.Min(dstSpan, srcSpan, dstSpan);
                    return;
                case ReductionOperation.Max:
                    TensorPrimitives.Max(dstSpan, srcSpan, dstSpan);
                    return;
            }
#else
            ReduceFloatSimdLegacy(dstF, dstOffset, srcF, srcOffset, count, operation);
            return;
#endif
        }
        else if (typeof(T) == typeof(double))
        {
            var dstD = (double[])(object)dst;
            var srcD = (double[])(object)src;
#if !AIDN_LEGACY_TFM
            var dstSpan = dstD.AsSpan(dstOffset, count);
            ReadOnlySpan<double> srcSpan = srcD.AsSpan(srcOffset, count);
            switch (operation)
            {
                case ReductionOperation.Sum:
                case ReductionOperation.Average:
                    TensorPrimitives.Add(dstSpan, srcSpan, dstSpan);
                    return;
                case ReductionOperation.Product:
                    TensorPrimitives.Multiply(dstSpan, srcSpan, dstSpan);
                    return;
                case ReductionOperation.Min:
                    TensorPrimitives.Min(dstSpan, srcSpan, dstSpan);
                    return;
                case ReductionOperation.Max:
                    TensorPrimitives.Max(dstSpan, srcSpan, dstSpan);
                    return;
            }
#else
            ReduceDoubleSimdLegacy(dstD, dstOffset, srcD, srcOffset, count, operation);
            return;
#endif
        }

        ReduceArrayInPlaceGeneric(dst, dstOffset, src, srcOffset, count, operation);
    }

#if AIDN_LEGACY_TFM
    private static void ReduceFloatSimdLegacy(float[] dst, int dstOffset, float[] src, int srcOffset, int count, ReductionOperation operation)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && count >= System.Numerics.Vector<float>.Count)
        {
            int vecLen = System.Numerics.Vector<float>.Count;
            int limit = count - vecLen;
            switch (operation)
            {
                case ReductionOperation.Sum:
                case ReductionOperation.Average:
                    for (; i <= limit; i += vecLen)
                    {
                        var vDst = new System.Numerics.Vector<float>(dst, dstOffset + i);
                        var vSrc = new System.Numerics.Vector<float>(src, srcOffset + i);
                        (vDst + vSrc).CopyTo(dst, dstOffset + i);
                    }
                    break;
                case ReductionOperation.Product:
                    for (; i <= limit; i += vecLen)
                    {
                        var vDst = new System.Numerics.Vector<float>(dst, dstOffset + i);
                        var vSrc = new System.Numerics.Vector<float>(src, srcOffset + i);
                        (vDst * vSrc).CopyTo(dst, dstOffset + i);
                    }
                    break;
                case ReductionOperation.Min:
                    for (; i <= limit; i += vecLen)
                    {
                        var vDst = new System.Numerics.Vector<float>(dst, dstOffset + i);
                        var vSrc = new System.Numerics.Vector<float>(src, srcOffset + i);
                        System.Numerics.Vector.Min(vDst, vSrc).CopyTo(dst, dstOffset + i);
                    }
                    break;
                case ReductionOperation.Max:
                    for (; i <= limit; i += vecLen)
                    {
                        var vDst = new System.Numerics.Vector<float>(dst, dstOffset + i);
                        var vSrc = new System.Numerics.Vector<float>(src, srcOffset + i);
                        System.Numerics.Vector.Max(vDst, vSrc).CopyTo(dst, dstOffset + i);
                    }
                    break;
                default:
                    throw new ArgumentException($"Unsupported reduction operation: {operation}");
            }
        }

        switch (operation)
        {
            case ReductionOperation.Sum:
            case ReductionOperation.Average:
                for (; i < count; i++) dst[dstOffset + i] += src[srcOffset + i];
                break;
            case ReductionOperation.Product:
                for (; i < count; i++) dst[dstOffset + i] *= src[srcOffset + i];
                break;
            case ReductionOperation.Min:
                for (; i < count; i++) if (src[srcOffset + i] < dst[dstOffset + i]) dst[dstOffset + i] = src[srcOffset + i];
                break;
            case ReductionOperation.Max:
                for (; i < count; i++) if (src[srcOffset + i] > dst[dstOffset + i]) dst[dstOffset + i] = src[srcOffset + i];
                break;
            default:
                throw new ArgumentException($"Unsupported reduction operation: {operation}");
        }
    }

    private static void ReduceDoubleSimdLegacy(double[] dst, int dstOffset, double[] src, int srcOffset, int count, ReductionOperation operation)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && count >= System.Numerics.Vector<double>.Count)
        {
            int vecLen = System.Numerics.Vector<double>.Count;
            int limit = count - vecLen;
            switch (operation)
            {
                case ReductionOperation.Sum:
                case ReductionOperation.Average:
                    for (; i <= limit; i += vecLen)
                    {
                        var vDst = new System.Numerics.Vector<double>(dst, dstOffset + i);
                        var vSrc = new System.Numerics.Vector<double>(src, srcOffset + i);
                        (vDst + vSrc).CopyTo(dst, dstOffset + i);
                    }
                    break;
                case ReductionOperation.Product:
                    for (; i <= limit; i += vecLen)
                    {
                        var vDst = new System.Numerics.Vector<double>(dst, dstOffset + i);
                        var vSrc = new System.Numerics.Vector<double>(src, srcOffset + i);
                        (vDst * vSrc).CopyTo(dst, dstOffset + i);
                    }
                    break;
                case ReductionOperation.Min:
                    for (; i <= limit; i += vecLen)
                    {
                        var vDst = new System.Numerics.Vector<double>(dst, dstOffset + i);
                        var vSrc = new System.Numerics.Vector<double>(src, srcOffset + i);
                        System.Numerics.Vector.Min(vDst, vSrc).CopyTo(dst, dstOffset + i);
                    }
                    break;
                case ReductionOperation.Max:
                    for (; i <= limit; i += vecLen)
                    {
                        var vDst = new System.Numerics.Vector<double>(dst, dstOffset + i);
                        var vSrc = new System.Numerics.Vector<double>(src, srcOffset + i);
                        System.Numerics.Vector.Max(vDst, vSrc).CopyTo(dst, dstOffset + i);
                    }
                    break;
                default:
                    throw new ArgumentException($"Unsupported reduction operation: {operation}");
            }
        }

        switch (operation)
        {
            case ReductionOperation.Sum:
            case ReductionOperation.Average:
                for (; i < count; i++) dst[dstOffset + i] += src[srcOffset + i];
                break;
            case ReductionOperation.Product:
                for (; i < count; i++) dst[dstOffset + i] *= src[srcOffset + i];
                break;
            case ReductionOperation.Min:
                for (; i < count; i++) if (src[srcOffset + i] < dst[dstOffset + i]) dst[dstOffset + i] = src[srcOffset + i];
                break;
            case ReductionOperation.Max:
                for (; i < count; i++) if (src[srcOffset + i] > dst[dstOffset + i]) dst[dstOffset + i] = src[srcOffset + i];
                break;
            default:
                throw new ArgumentException($"Unsupported reduction operation: {operation}");
        }
    }
#endif

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE ring reduction fallback when T is neither float nor double")]
    private void ReduceArrayInPlaceGeneric(T[] dst, int dstOffset, T[] src, int srcOffset, int count, ReductionOperation operation)
    {
        for (int i = 0; i < count; i++)
        {
            dst[dstOffset + i] = PerformReduction(dst[dstOffset + i], src[srcOffset + i], operation);
        }
    }

    private void ScaleArrayByReciprocalWorldSize(T[] dst, int dstOffset, int count, int worldSize)
    {
        if (count <= 0)
        {
            return;
        }

        if (typeof(T) == typeof(float))
        {
            var dstF = (float[])(object)dst;
            float inv = 1.0f / worldSize;
#if !AIDN_LEGACY_TFM
            var span = dstF.AsSpan(dstOffset, count);
            TensorPrimitives.Multiply(span, inv, span);
#else
            int i = 0;
            if (System.Numerics.Vector.IsHardwareAccelerated && count >= System.Numerics.Vector<float>.Count)
            {
                int vecLen = System.Numerics.Vector<float>.Count;
                int limit = count - vecLen;
                var vInv = new System.Numerics.Vector<float>(inv);
                for (; i <= limit; i += vecLen)
                {
                    (new System.Numerics.Vector<float>(dstF, dstOffset + i) * vInv).CopyTo(dstF, dstOffset + i);
                }
            }

            for (; i < count; i++)
            {
                dstF[dstOffset + i] *= inv;
            }
#endif
            return;
        }

        if (typeof(T) == typeof(double))
        {
            var dstD = (double[])(object)dst;
            double inv = 1.0 / worldSize;
#if !AIDN_LEGACY_TFM
            var span = dstD.AsSpan(dstOffset, count);
            TensorPrimitives.Multiply(span, inv, span);
#else
            int i = 0;
            if (System.Numerics.Vector.IsHardwareAccelerated && count >= System.Numerics.Vector<double>.Count)
            {
                int vecLen = System.Numerics.Vector<double>.Count;
                int limit = count - vecLen;
                var vInv = new System.Numerics.Vector<double>(inv);
                for (; i <= limit; i += vecLen)
                {
                    (new System.Numerics.Vector<double>(dstD, dstOffset + i) * vInv).CopyTo(dstD, dstOffset + i);
                }
            }

            for (; i < count; i++)
            {
                dstD[dstOffset + i] *= inv;
            }
#endif
            return;
        }

        ScaleArrayByReciprocalWorldSizeGeneric(dst, dstOffset, count, worldSize);
    }

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE ring average division fallback")]
    private void ScaleArrayByReciprocalWorldSizeGeneric(T[] dst, int dstOffset, int count, int worldSize)
    {
        T divisor = NumOps.FromDouble(worldSize);
        for (int i = 0; i < count; i++)
        {
            dst[dstOffset + i] = NumOps.Divide(dst[dstOffset + i], divisor);
        }
    }

    /// <summary>
    /// Performs a reduction operation on two values.
    /// </summary>
    private T PerformReduction(T a, T b, ReductionOperation operation)
    {
        return operation switch
        {
            ReductionOperation.Sum or ReductionOperation.Average => NumOps.Add(a, b),
            ReductionOperation.Max => NumOps.GreaterThan(a, b) ? a : b,
            ReductionOperation.Min => NumOps.LessThan(a, b) ? a : b,
            ReductionOperation.Product => NumOps.Multiply(a, b),
            _ => throw new ArgumentException($"Unsupported reduction operation: {operation}")
        };
    }

    /// <summary>
    /// Performs ring-based AllGather operation supporting both uniform and uneven shard sizes across ranks
    /// while pooling all transit shard buffers in <see cref="ArrayPool{T}"/>.
    /// </summary>
    private Vector<T> PerformRingAllGather(Vector<T> sendData)
    {
        if (_worldSize == 1)
        {
            return sendData.Clone();
        }

        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var shardBuffers = new T[]?[_worldSize];
        var shardOffsets = new int[_worldSize];
        var shardLengths = new int[_worldSize];
        var shardRented = new bool[_worldSize];

        GetReadOnlyArraySegmentOrRent(
            sendData,
            out T[] localBuf,
            out int localOffset,
            out bool localRented);
        shardBuffers[_rank] = localBuf;
        shardOffsets[_rank] = localOffset;
        shardLengths[_rank] = sendData.Length;
        shardRented[_rank] = localRented;

        try
        {
            // Ring AllGather: each rank forwards its current shard while receiving a variable-length shard into a pooled buffer.
            for (int step = 0; step < _worldSize - 1; step++)
            {
                int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
                int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

                T[] outgoingBuf = shardBuffers[sendChunkIdx] ?? Array.Empty<T>();
                int outgoingOffset = shardOffsets[sendChunkIdx];
                int outgoingLength = shardLengths[sendChunkIdx];

                var sendWorker = GetOrCreateSendWorker();
                sendWorker.BeginSend(nextRank, outgoingBuf, outgoingOffset, outgoingLength);
                Exception? receiveException = null;
                T[]? incomingRented = null;
                int incomingLength = 0;
                try
                {
                    incomingRented = ReceiveRentedArray(prevRank, out incomingLength);
                }
                catch (Exception ex)
                {
                    receiveException = ex;
                }
                finally
                {
                    sendWorker.EndSend(suppressException: receiveException is not null);
                }

                if (receiveException is not null)
                {
                    if (incomingRented is not null && incomingLength > 0)
                    {
                        ArrayPool<T>.Shared.Return(incomingRented);
                    }

                    ExceptionDispatchInfo.Capture(receiveException).Throw();
                }

                shardBuffers[recvChunkIdx] = incomingRented;
                shardOffsets[recvChunkIdx] = 0;
                shardLengths[recvChunkIdx] = incomingLength;
                shardRented[recvChunkIdx] = incomingRented is not null && incomingLength > 0;
            }

            int totalLength = 0;
            for (int r = 0; r < _worldSize; r++)
            {
                totalLength = checked(totalLength + shardLengths[r]);
            }

            var result = AllocateExactOutputArray(totalLength);
            int writeOffset = 0;
            for (int r = 0; r < _worldSize; r++)
            {
                int len = shardLengths[r];
                if (len > 0 && shardBuffers[r] is { } src)
                {
                    Array.Copy(src, shardOffsets[r], result, writeOffset, len);
                    writeOffset += len;
                }
            }

            return new Vector<T>(result);
        }
        finally
        {
            for (int r = 0; r < _worldSize; r++)
            {
                if (shardRented[r] && shardBuffers[r] is { } rented)
                {
                    ArrayPool<T>.Shared.Return(rented);
                }
            }

            GC.KeepAlive(sendData);
        }
    }

    /// <summary>
    /// Performs tree-based Broadcast operation without intermediate staging allocations.
    /// </summary>
    private Vector<T> PerformTreeBroadcast(Vector<T> data, int root)
    {
        if (_worldSize == 1)
        {
            return data.Clone();
        }

        int relativeRank = (_rank - root + _worldSize) % _worldSize;

        T[] dataArray;
        if (relativeRank == 0)
        {
            dataArray = AllocateExactOutputArray(data.Length);
            if (data.Length > 0)
            {
                data.AsSpanUnmarked().CopyTo(dataArray);
                GC.KeepAlive(data);
            }
        }
        else
        {
            int parentRelative = (relativeRank - 1) / 2;
            int parentAbsolute = (parentRelative + root) % _worldSize;
            dataArray = ReceiveExactNewArray(parentAbsolute, expectedLength: null, hasTag: false, expectedTag: 0);
        }

        int leftChildRelative = 2 * relativeRank + 1;
        int rightChildRelative = 2 * relativeRank + 2;

        if (leftChildRelative < _worldSize)
        {
            int leftChildAbsolute = (leftChildRelative + root) % _worldSize;
            SendArraySlice(leftChildAbsolute, dataArray, 0, dataArray.Length);
        }

        if (rightChildRelative < _worldSize)
        {
            int rightChildAbsolute = (rightChildRelative + root) % _worldSize;
            SendArraySlice(rightChildAbsolute, dataArray, 0, dataArray.Length);
        }

        return new Vector<T>(dataArray);
    }

    /// <summary>
    /// Performs tree-based Scatter operation without full-buffer <c>ToArray()</c> or temporary chunk copies.
    /// </summary>
    private Vector<T> PerformTreeScatter(Vector<T> sendData, int root)
    {
        if (_worldSize == 1)
        {
            if (Rank == root)
            {
                ValidateData(sendData, nameof(sendData));
                return sendData.Clone();
            }
            return new Vector<T>(Array.Empty<T>());
        }

        T[] myChunk;

        if (Rank == root)
        {
            ValidateData(sendData, nameof(sendData));

            if (sendData.Length % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"Data length {sendData.Length} must be divisible by world size {_worldSize}.");
            }

            int chunkSize = sendData.Length / _worldSize;
            GetReadOnlyArraySegmentOrRent(sendData, out T[] allData, out int baseOffset, out bool rented);
            try
            {
                myChunk = AllocateExactOutputArray(chunkSize);
                if (chunkSize > 0)
                {
                    Array.Copy(allData, baseOffset + (root * chunkSize), myChunk, 0, chunkSize);
                }

                _scatterLengthSendSlot[0] = NumOps.FromDouble(chunkSize);
                for (int r = 0; r < _worldSize; r++)
                {
                    if (r != root)
                    {
                        SendArraySlice(r, _scatterLengthSendSlot, 0, 1);
                        SendArraySlice(r, allData, baseOffset + (r * chunkSize), chunkSize);
                    }
                }
            }
            finally
            {
                if (rented)
                {
                    ArrayPool<T>.Shared.Return(allData);
                }

                GC.KeepAlive(sendData);
            }
        }
        else
        {
            ReceiveArrayInto(root, _scatterLengthReceiveSlot, 0, expectedLength: 1);
            int chunkSize = _scatterLengthReceiveSlot[0] != null ? (int)NumOps.ToDouble(_scatterLengthReceiveSlot[0]) : 0;
            myChunk = AllocateExactOutputArray(chunkSize);
            ReceiveArrayInto(root, myChunk, 0, expectedLength: chunkSize);
        }

        return new Vector<T>(myChunk);
    }

    /// <summary>
    /// Performs ring-based ReduceScatter operation using pooled work and receive buffers and SIMD reductions.
    /// </summary>
    private Vector<T> PerformRingReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        if (_worldSize == 1)
        {
            return data.Clone();
        }

        int totalLength = data.Length;
        int chunkSize = totalLength / _worldSize;
        if (chunkSize == 0)
        {
            return new Vector<T>(Array.Empty<T>());
        }

        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        T[] workArray = AcquirePersistentScratch(ref _reduceScatterWorkScratch, totalLength);
        T[] recvScratch = AcquirePersistentScratch(ref _ringRecvScratch, chunkSize);
        try
        {
            data.AsSpanUnmarked().CopyTo(workArray.AsSpan(0, totalLength));
            GC.KeepAlive(data);

            for (int step = 0; step < _worldSize - 1; step++)
            {
                int sendChunkIdx = (_rank - 1 - step + (2 * _worldSize)) % _worldSize;
                int recvChunkIdx = (_rank - 2 - step + (2 * _worldSize)) % _worldSize;

                int sendStart = sendChunkIdx * chunkSize;
                int sendCount = Math.Max(0, Math.Min(chunkSize, totalLength - sendStart));
                int recvStart = recvChunkIdx * chunkSize;
                int recvCount = Math.Max(0, Math.Min(chunkSize, totalLength - recvStart));

                SendAndReceiveConcurrent(
                    nextRank, workArray, sendStart, sendCount,
                    prevRank, recvScratch, 0, recvCount);

                if (recvCount > 0)
                {
                    ReduceArrayInPlace(workArray, recvStart, recvScratch, 0, recvCount, operation);
                }
            }

            var myChunk = AllocateExactOutputArray(chunkSize);
            Array.Copy(workArray, _rank * chunkSize, myChunk, 0, chunkSize);

            if (operation == ReductionOperation.Average)
            {
                ScaleArrayByReciprocalWorldSize(myChunk, 0, chunkSize, _worldSize);
            }

            return new Vector<T>(myChunk);
        }
        finally
        {
            ReleasePersistentScratch(ref _ringRecvScratch, recvScratch);
            ReleasePersistentScratch(ref _reduceScatterWorkScratch, workArray);
        }
    }

    /// <inheritdoc/>
    public override void Send(Vector<T> data, int destinationRank, int tag = 0)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));
        ValidateRank(destinationRank, nameof(destinationRank));

        if (tag < 0)
        {
            throw new ArgumentException("Tag must be non-negative.", nameof(tag));
        }

        // Single-process mode: cannot send to self
        if (_worldSize == 1)
        {
            throw new InvalidOperationException("Cannot send in single-process mode.");
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("Send requires TCP mode to be initialized");
        }

        SendDataWithTag(destinationRank, data, tag);
    }

    /// <inheritdoc/>
    public override Vector<T> Receive(int sourceRank, int count, int tag = 0)
    {
        EnsureInitialized();
        ValidateRank(sourceRank, nameof(sourceRank));

        if (count <= 0)
        {
            throw new ArgumentException("Count must be positive.", nameof(count));
        }

        if (tag < 0)
        {
            throw new ArgumentException("Tag must be non-negative.", nameof(tag));
        }

        // Single-process mode: cannot receive from self
        if (_worldSize == 1)
        {
            throw new InvalidOperationException("Cannot receive in single-process mode.");
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("Receive requires TCP mode to be initialized");
        }

        return ReceiveDataWithTag(sourceRank, count, tag);
    }

    /// <summary>
    /// Sends data to a specific rank via TCP with message tag directly from the vector's backing memory without copying.
    /// </summary>
    private void SendDataWithTag(int destRank, Vector<T> data, int tag)
    {
        GetReadOnlyArraySegmentOrRent(data, out T[] buffer, out int offset, out bool rented);
        try
        {
            SendArraySliceCore(destRank, buffer, offset, data.Length, hasTag: true, tag: tag);
        }
        finally
        {
            if (rented)
            {
                ArrayPool<T>.Shared.Return(buffer);
            }

            GC.KeepAlive(data);
        }
    }

    /// <summary>
    /// Receives data from a specific rank via TCP with message tag directly into the returned vector's backing array.
    /// </summary>
    private Vector<T> ReceiveDataWithTag(int sourceRank, int expectedLength, int expectedTag)
    {
        var result = ReceiveExactNewArray(sourceRank, expectedLength, hasTag: true, expectedTag: expectedTag);
        return new Vector<T>(result);
    }

    private static void GetReadOnlyArraySegmentOrRent(
        Vector<T> vector,
        out T[] array,
        out int offset,
        out bool rented)
    {
        int length = vector.Length;
        if (length == 0)
        {
            array = Array.Empty<T>();
            offset = 0;
            rented = false;
            return;
        }

        ReadOnlyMemory<T> memory = vector.AsMemoryUnmarked();
        if (MemoryMarshal.TryGetArray(memory, out var segment) && segment.Array is not null)
        {
            array = segment.Array;
            offset = segment.Offset;
            rented = false;
            return;
        }

        array = ArrayPool<T>.Shared.Rent(length);
        memory.Span.CopyTo(array.AsSpan(0, length));
        offset = 0;
        rented = true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T[] AllocateExactOutputArray(int length)
    {
        if (length == 0)
        {
            return Array.Empty<T>();
        }

#if !AIDN_LEGACY_TFM
        if (typeof(T) == typeof(float) ||
            typeof(T) == typeof(double) ||
            typeof(T) == typeof(int) ||
            typeof(T) == typeof(long) ||
            typeof(T) == typeof(byte))
        {
            return GC.AllocateUninitializedArray<T>(length);
        }
#endif
        return new T[length];
    }

    private TcpClient GetConnectedPeerClient(int peerRank)
    {
        lock (_connectionLock)
        {
            if (_tcpConnections == null || !_tcpConnections.TryGetValue(peerRank, out var client))
            {
                throw new InvalidOperationException(
                    $"No TCP connection to rank {peerRank}. Ensure Initialize() was called and connections were established.");
            }

            return client;
        }
    }

    private void SendArraySlice(int destRank, T[] data, int offset, int count)
    {
        SendArraySliceCore(destRank, data, offset, count, hasTag: false, tag: 0);
    }

    private void SendArraySliceCore(int destRank, T[] data, int offset, int count, bool hasTag, int tag)
    {
        var client = GetConnectedPeerClient(destRank);
        object sendLock = _peerSendLocks is not null ? _peerSendLocks[destRank] : _connectionLock;

        lock (sendLock)
        {
            var stream = client.GetStream();
#if !AIDN_LEGACY_TFM
            if (hasTag)
            {
                Span<byte> hdr = stackalloc byte[8];
                BinaryPrimitives.WriteInt32LittleEndian(hdr.Slice(0, 4), tag);
                BinaryPrimitives.WriteInt32LittleEndian(hdr.Slice(4, 4), count);
                stream.Write(hdr);
            }
            else
            {
                Span<byte> hdr = stackalloc byte[4];
                BinaryPrimitives.WriteInt32LittleEndian(hdr, count);
                stream.Write(hdr);
            }
#else
            if (hasTag)
            {
                WriteInt32PairLittleEndian(stream, tag, count);
            }
            else
            {
                WriteInt32LittleEndian(stream, count);
            }
#endif

            if (count > 0)
            {
                WriteElementsBulk(stream, data, offset, count);
            }

            stream.Flush();
        }
    }

    private void ReceiveArrayInto(int sourceRank, T[] destination, int dstOffset, int expectedLength)
    {
        ReceiveArrayIntoCore(sourceRank, destination, dstOffset, expectedLength, hasTag: false, expectedTag: 0);
    }

    private void ReceiveArrayIntoCore(
        int sourceRank,
        T[] destination,
        int dstOffset,
        int expectedLength,
        bool hasTag,
        int expectedTag)
    {
        var client = GetConnectedPeerClient(sourceRank);
        object recvLock = _peerReceiveLocks is not null ? _peerReceiveLocks[sourceRank] : _connectionLock;

        lock (recvLock)
        {
            var stream = client.GetStream();
            int length = ReadHeaderAndValidate(stream, sourceRank, expectedLength, hasTag, expectedTag);
            if (length == 0)
            {
                return;
            }

            ReadElementsBulkInto(stream, destination, dstOffset, length);
        }
    }

    private T[]? ReceiveRentedArray(int sourceRank, out int length)
    {
        var client = GetConnectedPeerClient(sourceRank);
        object recvLock = _peerReceiveLocks is not null ? _peerReceiveLocks[sourceRank] : _connectionLock;

        lock (recvLock)
        {
            var stream = client.GetStream();
            length = ReadHeaderAndValidate(stream, sourceRank, expectedLength: null, hasTag: false, expectedTag: 0);
            if (length == 0)
            {
                return null;
            }

            T[] rented = ArrayPool<T>.Shared.Rent(length);
            try
            {
                ReadElementsBulkInto(stream, rented, 0, length);
                return rented;
            }
            catch
            {
                ArrayPool<T>.Shared.Return(rented);
                throw;
            }
        }
    }

    private T[] ReceiveExactNewArray(int sourceRank, int? expectedLength, bool hasTag, int expectedTag)
    {
        var client = GetConnectedPeerClient(sourceRank);
        object recvLock = _peerReceiveLocks is not null ? _peerReceiveLocks[sourceRank] : _connectionLock;

        lock (recvLock)
        {
            var stream = client.GetStream();
            int length = ReadHeaderAndValidate(stream, sourceRank, expectedLength, hasTag, expectedTag);
            if (length == 0)
            {
                return Array.Empty<T>();
            }

            T[] result = AllocateExactOutputArray(length);
            ReadElementsBulkInto(stream, result, 0, length);
            return result;
        }
    }

    private int ReadHeaderAndValidate(
        Stream stream,
        int sourceRank,
        int? expectedLength,
        bool hasTag,
        int expectedTag)
    {
        int length;
#if !AIDN_LEGACY_TFM
        if (hasTag)
        {
            Span<byte> hdr = stackalloc byte[8];
            stream.ReadExactly(hdr);
            int receivedTag = BinaryPrimitives.ReadInt32LittleEndian(hdr.Slice(0, 4));
            if (receivedTag != expectedTag)
            {
                throw new InvalidOperationException(
                    $"Rank {_rank}: Expected tag {expectedTag} from rank {sourceRank}, but received tag {receivedTag}");
            }

            length = BinaryPrimitives.ReadInt32LittleEndian(hdr.Slice(4, 4));
        }
        else
        {
            Span<byte> hdr = stackalloc byte[4];
            stream.ReadExactly(hdr);
            length = BinaryPrimitives.ReadInt32LittleEndian(hdr);
        }
#else
        if (hasTag)
        {
            int receivedTag = ReadInt32LittleEndian(stream);
            if (receivedTag != expectedTag)
            {
                throw new InvalidOperationException(
                    $"Rank {_rank}: Expected tag {expectedTag} from rank {sourceRank}, but received tag {receivedTag}");
            }
        }

        length = ReadInt32LittleEndian(stream);
#endif

        if (length < 0)
        {
            throw new InvalidOperationException(
                $"Rank {_rank}: Received negative payload length {length} from rank {sourceRank}");
        }

        if (expectedLength.HasValue && length != expectedLength.Value)
        {
            throw new InvalidOperationException(
                $"Rank {_rank}: Expected {expectedLength.Value} elements from rank {sourceRank}, but received {length}");
        }

        return length;
    }

    private static void WriteInt32LittleEndian(Stream stream, int value)
    {
#if !AIDN_LEGACY_TFM
        Span<byte> hdr = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(hdr, value);
        stream.Write(hdr);
#else
        byte[] hdr = ArrayPool<byte>.Shared.Rent(4);
        try
        {
            hdr[0] = (byte)value;
            hdr[1] = (byte)(value >> 8);
            hdr[2] = (byte)(value >> 16);
            hdr[3] = (byte)(value >> 24);
            stream.Write(hdr, 0, 4);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(hdr);
        }
#endif
    }

#if AIDN_LEGACY_TFM
    private static void WriteInt32PairLittleEndian(Stream stream, int first, int second)
    {
        byte[] hdr = ArrayPool<byte>.Shared.Rent(8);
        try
        {
            hdr[0] = (byte)first;
            hdr[1] = (byte)(first >> 8);
            hdr[2] = (byte)(first >> 16);
            hdr[3] = (byte)(first >> 24);
            hdr[4] = (byte)second;
            hdr[5] = (byte)(second >> 8);
            hdr[6] = (byte)(second >> 16);
            hdr[7] = (byte)(second >> 24);
            stream.Write(hdr, 0, 8);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(hdr);
        }
    }
#endif

    private static int ReadInt32LittleEndian(Stream stream)
    {
#if !AIDN_LEGACY_TFM
        Span<byte> hdr = stackalloc byte[4];
        stream.ReadExactly(hdr);
        return BinaryPrimitives.ReadInt32LittleEndian(hdr);
#else
        byte[] hdr = ArrayPool<byte>.Shared.Rent(4);
        try
        {
            ReadExactBytes(stream, hdr, 0, 4);
            return hdr[0]
                | (hdr[1] << 8)
                | (hdr[2] << 16)
                | (hdr[3] << 24);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(hdr);
        }
#endif
    }

#if AIDN_LEGACY_TFM
    private static void ReadExactBytes(Stream stream, byte[] buffer, int offset, int count)
    {
        int totalRead = 0;
        while (totalRead < count)
        {
            int read = stream.Read(buffer, offset + totalRead, count - totalRead);
            if (read <= 0)
            {
                throw new EndOfStreamException(
                    $"Unexpected EOF on Gloo TCP transport after reading {totalRead} of {count} bytes (peer rank closed or crashed).");
            }

            totalRead += read;
        }
    }
#endif

    private void WriteElementsBulk(Stream stream, T[] source, int offset, int count)
    {
        if (typeof(T) == typeof(byte))
        {
            var srcB = (byte[])(object)source;
            stream.Write(srcB, offset, count);
            return;
        }

#if !AIDN_LEGACY_TFM
        if (typeof(T) == typeof(float))
        {
            var srcF = (float[])(object)source;
            const int maxElems = MaxDirectSpanWriteBytes / sizeof(float);
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                stream.Write(MemoryMarshal.AsBytes(srcF.AsSpan(offset + pos, batch)));
                pos += batch;
            }

            return;
        }

        if (typeof(T) == typeof(double))
        {
            var srcD = (double[])(object)source;
            const int maxElems = MaxDirectSpanWriteBytes / sizeof(double);
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                stream.Write(MemoryMarshal.AsBytes(srcD.AsSpan(offset + pos, batch)));
                pos += batch;
            }

            return;
        }

        if (typeof(T) == typeof(int))
        {
            var srcI = (int[])(object)source;
            const int maxElems = MaxDirectSpanWriteBytes / sizeof(int);
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                stream.Write(MemoryMarshal.AsBytes(srcI.AsSpan(offset + pos, batch)));
                pos += batch;
            }

            return;
        }

        if (typeof(T) == typeof(long))
        {
            var srcL = (long[])(object)source;
            const int maxElems = MaxDirectSpanWriteBytes / sizeof(long);
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                stream.Write(MemoryMarshal.AsBytes(srcL.AsSpan(offset + pos, batch)));
                pos += batch;
            }

            return;
        }
#else
        if (typeof(T) == typeof(float))
        {
            WriteBlittableArrayLegacy(stream, (float[])(object)source, offset, count, sizeof(float));
            return;
        }

        if (typeof(T) == typeof(double))
        {
            WriteBlittableArrayLegacy(stream, (double[])(object)source, offset, count, sizeof(double));
            return;
        }

        if (typeof(T) == typeof(int))
        {
            WriteBlittableArrayLegacy(stream, (int[])(object)source, offset, count, sizeof(int));
            return;
        }

        if (typeof(T) == typeof(long))
        {
            WriteBlittableArrayLegacy(stream, (long[])(object)source, offset, count, sizeof(long));
            return;
        }
#endif

        WriteElementsBulkGeneric(stream, source, offset, count);
    }

#if AIDN_LEGACY_TFM
    private static void WriteBlittableArrayLegacy(Stream stream, Array source, int offset, int count, int elementSize)
    {
        int maxElems = LegacyStagingBufferBytes / elementSize;
        byte[] buf = ArrayPool<byte>.Shared.Rent(Math.Min(count, maxElems) * elementSize);
        try
        {
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                int bytes = batch * elementSize;
                Buffer.BlockCopy(source, (offset + pos) * elementSize, buf, 0, bytes);
                stream.Write(buf, 0, bytes);
                pos += batch;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buf);
        }
    }
#endif

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE TCP serialization fallback when T is not a primitive blittable type")]
    private void WriteElementsBulkGeneric(Stream stream, T[] source, int offset, int count)
    {
        double[] tmp = ArrayPool<double>.Shared.Rent(Math.Min(count, GenericDoubleBatchElements));
#if AIDN_LEGACY_TFM
        byte[] byteBuf = ArrayPool<byte>.Shared.Rent(Math.Min(count, GenericDoubleBatchElements) * sizeof(double));
#endif
        try
        {
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(GenericDoubleBatchElements, count - pos);
                for (int i = 0; i < batch; i++)
                {
                    tmp[i] = NumOps.ToDouble(source[offset + pos + i]);
                }

#if !AIDN_LEGACY_TFM
                stream.Write(MemoryMarshal.AsBytes(tmp.AsSpan(0, batch)));
#else
                int bytes = batch * sizeof(double);
                Buffer.BlockCopy(tmp, 0, byteBuf, 0, bytes);
                stream.Write(byteBuf, 0, bytes);
#endif
                pos += batch;
            }
        }
        finally
        {
#if AIDN_LEGACY_TFM
            ArrayPool<byte>.Shared.Return(byteBuf);
#endif
            ArrayPool<double>.Shared.Return(tmp);
        }
    }

    private void ReadElementsBulkInto(Stream stream, T[] destination, int dstOffset, int count)
    {
        if (typeof(T) == typeof(byte))
        {
            var dstB = (byte[])(object)destination;
#if !AIDN_LEGACY_TFM
            stream.ReadExactly(dstB.AsSpan(dstOffset, count));
#else
            ReadExactBytes(stream, dstB, dstOffset, count);
#endif
            return;
        }

#if !AIDN_LEGACY_TFM
        if (typeof(T) == typeof(float))
        {
            var dstF = (float[])(object)destination;
            const int maxElems = MaxDirectSpanWriteBytes / sizeof(float);
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                stream.ReadExactly(MemoryMarshal.AsBytes(dstF.AsSpan(dstOffset + pos, batch)));
                pos += batch;
            }

            return;
        }

        if (typeof(T) == typeof(double))
        {
            var dstD = (double[])(object)destination;
            const int maxElems = MaxDirectSpanWriteBytes / sizeof(double);
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                stream.ReadExactly(MemoryMarshal.AsBytes(dstD.AsSpan(dstOffset + pos, batch)));
                pos += batch;
            }

            return;
        }

        if (typeof(T) == typeof(int))
        {
            var dstI = (int[])(object)destination;
            const int maxElems = MaxDirectSpanWriteBytes / sizeof(int);
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                stream.ReadExactly(MemoryMarshal.AsBytes(dstI.AsSpan(dstOffset + pos, batch)));
                pos += batch;
            }

            return;
        }

        if (typeof(T) == typeof(long))
        {
            var dstL = (long[])(object)destination;
            const int maxElems = MaxDirectSpanWriteBytes / sizeof(long);
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                stream.ReadExactly(MemoryMarshal.AsBytes(dstL.AsSpan(dstOffset + pos, batch)));
                pos += batch;
            }

            return;
        }
#else
        if (typeof(T) == typeof(float))
        {
            ReadBlittableArrayLegacy(stream, (float[])(object)destination, dstOffset, count, sizeof(float));
            return;
        }

        if (typeof(T) == typeof(double))
        {
            ReadBlittableArrayLegacy(stream, (double[])(object)destination, dstOffset, count, sizeof(double));
            return;
        }

        if (typeof(T) == typeof(int))
        {
            ReadBlittableArrayLegacy(stream, (int[])(object)destination, dstOffset, count, sizeof(int));
            return;
        }

        if (typeof(T) == typeof(long))
        {
            ReadBlittableArrayLegacy(stream, (long[])(object)destination, dstOffset, count, sizeof(long));
            return;
        }
#endif

        ReadElementsBulkIntoGeneric(stream, destination, dstOffset, count);
    }

#if AIDN_LEGACY_TFM
    private static void ReadBlittableArrayLegacy(Stream stream, Array destination, int dstOffset, int count, int elementSize)
    {
        int maxElems = LegacyStagingBufferBytes / elementSize;
        byte[] buf = ArrayPool<byte>.Shared.Rent(Math.Min(count, maxElems) * elementSize);
        try
        {
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(maxElems, count - pos);
                int bytes = batch * elementSize;
                ReadExactBytes(stream, buf, 0, bytes);
                Buffer.BlockCopy(buf, 0, destination, (dstOffset + pos) * elementSize, bytes);
                pos += batch;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buf);
        }
    }
#endif

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE TCP deserialization fallback when T is not a primitive blittable type")]
    private void ReadElementsBulkIntoGeneric(Stream stream, T[] destination, int dstOffset, int count)
    {
        double[] tmp = ArrayPool<double>.Shared.Rent(Math.Min(count, GenericDoubleBatchElements));
#if AIDN_LEGACY_TFM
        byte[] byteBuf = ArrayPool<byte>.Shared.Rent(Math.Min(count, GenericDoubleBatchElements) * sizeof(double));
#endif
        try
        {
            int pos = 0;
            while (pos < count)
            {
                int batch = Math.Min(GenericDoubleBatchElements, count - pos);
#if !AIDN_LEGACY_TFM
                stream.ReadExactly(MemoryMarshal.AsBytes(tmp.AsSpan(0, batch)));
#else
                int bytes = batch * sizeof(double);
                ReadExactBytes(stream, byteBuf, 0, bytes);
                Buffer.BlockCopy(byteBuf, 0, tmp, 0, bytes);
#endif
                for (int i = 0; i < batch; i++)
                {
                    destination[dstOffset + pos + i] = NumOps.FromDouble(tmp[i]);
                }

                pos += batch;
            }
        }
        finally
        {
#if AIDN_LEGACY_TFM
            ArrayPool<byte>.Shared.Return(byteBuf);
#endif
            ArrayPool<double>.Shared.Return(tmp);
        }
    }
}
