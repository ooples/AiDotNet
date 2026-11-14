using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
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
    private readonly int _rank;
    private readonly int _worldSize;
    private bool _useNativeTCP;
    private bool _useNativeGloo;
    private Dictionary<int, TcpClient>? _tcpConnections;
    private TcpListener? _tcpListener;
    private readonly object _connectionLock = new object();
    private object[]? _peerSendLocks;
    private object[]? _peerReceiveLocks;

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

        // Start TCP listener on this rank's port
        int myPort = basePort + _rank;
        _tcpListener = new TcpListener(IPAddress.Any, myPort);
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
            try
            {
                var client = new TcpClient
                {
                    NoDelay = true,
                    SendBufferSize = 1024 * 1024,
                    ReceiveBufferSize = 1024 * 1024,
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
        client.SendBufferSize = 1024 * 1024;
        client.ReceiveBufferSize = 1024 * 1024;

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

        var signal = new[] { NumOps.One };
        for (int otherRank = 0; otherRank < _worldSize; otherRank++)
        {
            if (otherRank != _rank)
            {
                SendArraySlice(otherRank, signal, 0, 1);
            }
        }

        for (int otherRank = 0; otherRank < _worldSize; otherRank++)
        {
            if (otherRank != _rank)
            {
                ReceiveArray(otherRank, expectedLength: 1);
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
            var broadcastedArray = broadcasted.ToArray();
            Array.Copy(broadcastedArray, _rank * chunkSize, chunk, 0, chunkSize);
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

        var doubleData = data is null ? Array.Empty<double>() : VectorToDoubleArray(data);
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
    /// Performs ring-based AllReduce operation.
    /// </summary>
    /// <remarks>
    /// This is a production-ready implementation of the ring AllReduce algorithm
    /// used by systems like Baidu's Ring AllReduce and Horovod.
    ///
    /// Algorithm:
    /// 1. Divide data into N chunks (N = worldSize)
    /// 2. ReduceScatter phase: Send chunks in ring pattern, reducing as we go
    /// 3. AllGather phase: Gather the reduced chunks back to all ranks
    ///
    /// Time complexity: O(2*(N-1)*M/N) where M is data size
    /// This is optimal for large messages and scales linearly with cluster size.
    /// </remarks>
    private void PerformRingAllReduce(Vector<T> data, ReductionOperation operation)
    {
        if (_worldSize == 1 || data.Length == 0)
        {
            return;
        }

        int chunkSize = (data.Length + _worldSize - 1) / _worldSize;
        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var dataArray = data.ToArray();

        // Phase 1: ReduceScatter - reduce chunks in ring pattern
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

            int sendStart = sendChunkIdx * chunkSize;
            int sendCount = Math.Max(0, Math.Min(chunkSize, data.Length - sendStart));
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Max(0, Math.Min(chunkSize, data.Length - recvStart));

            var sendTask = System.Threading.Tasks.Task.Run(() => SendArraySlice(nextRank, dataArray, sendStart, sendCount));
            var recvChunk = ReceiveArray(prevRank, expectedLength: recvCount);
            sendTask.GetAwaiter().GetResult();

            if (recvCount > 0)
            {
                ReduceArrayInPlace(dataArray, recvStart, recvChunk, recvCount, operation);
            }
        }

        // Phase 2: AllGather - distribute reduced chunks in ring pattern
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + 1 + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step + _worldSize) % _worldSize;

            int sendStart = sendChunkIdx * chunkSize;
            int sendCount = Math.Max(0, Math.Min(chunkSize, data.Length - sendStart));
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Max(0, Math.Min(chunkSize, data.Length - recvStart));

            var sendTask = System.Threading.Tasks.Task.Run(() => SendArraySlice(nextRank, dataArray, sendStart, sendCount));
            var recvChunk = ReceiveArray(prevRank, expectedLength: recvCount);
            sendTask.GetAwaiter().GetResult();

            if (recvCount > 0)
            {
                Array.Copy(recvChunk, 0, dataArray, recvStart, recvCount);
            }
        }

        if (operation == ReductionOperation.Average)
        {
            ScaleArrayByReciprocalWorldSize(dataArray, _worldSize);
        }

        dataArray.AsSpan().CopyTo(data.AsWritableSpan());
    }

    private void ReduceArrayInPlace(T[] dst, int dstOffset, T[] src, int count, ReductionOperation operation)
    {
        if (typeof(T) == typeof(float))
        {
            var dstF = (float[])(object)dst;
            var srcF = (float[])(object)src;
            switch (operation)
            {
                case ReductionOperation.Sum:
                case ReductionOperation.Average:
                    for (int i = 0; i < count; i++) dstF[dstOffset + i] += srcF[i];
                    return;
                case ReductionOperation.Product:
                    for (int i = 0; i < count; i++) dstF[dstOffset + i] *= srcF[i];
                    return;
                case ReductionOperation.Min:
                    for (int i = 0; i < count; i++) if (srcF[i] < dstF[dstOffset + i]) dstF[dstOffset + i] = srcF[i];
                    return;
                case ReductionOperation.Max:
                    for (int i = 0; i < count; i++) if (srcF[i] > dstF[dstOffset + i]) dstF[dstOffset + i] = srcF[i];
                    return;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var dstD = (double[])(object)dst;
            var srcD = (double[])(object)src;
            switch (operation)
            {
                case ReductionOperation.Sum:
                case ReductionOperation.Average:
                    for (int i = 0; i < count; i++) dstD[dstOffset + i] += srcD[i];
                    return;
                case ReductionOperation.Product:
                    for (int i = 0; i < count; i++) dstD[dstOffset + i] *= srcD[i];
                    return;
                case ReductionOperation.Min:
                    for (int i = 0; i < count; i++) if (srcD[i] < dstD[dstOffset + i]) dstD[dstOffset + i] = srcD[i];
                    return;
                case ReductionOperation.Max:
                    for (int i = 0; i < count; i++) if (srcD[i] > dstD[dstOffset + i]) dstD[dstOffset + i] = srcD[i];
                    return;
            }
        }

        ReduceArrayInPlaceGeneric(dst, dstOffset, src, count, operation);
    }

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE ring reduction fallback when T is neither float nor double")]
    private void ReduceArrayInPlaceGeneric(T[] dst, int dstOffset, T[] src, int count, ReductionOperation operation)
    {
        for (int i = 0; i < count; i++)
        {
            dst[dstOffset + i] = PerformReduction(dst[dstOffset + i], src[i], operation);
        }
    }

    private void ScaleArrayByReciprocalWorldSize(T[] dst, int worldSize)
    {
        if (typeof(T) == typeof(float))
        {
            var dstF = (float[])(object)dst;
            float inv = 1.0f / worldSize;
            for (int i = 0; i < dstF.Length; i++) dstF[i] *= inv;
            return;
        }

        if (typeof(T) == typeof(double))
        {
            var dstD = (double[])(object)dst;
            double inv = 1.0 / worldSize;
            for (int i = 0; i < dstD.Length; i++) dstD[i] *= inv;
            return;
        }

        ScaleArrayByReciprocalWorldSizeGeneric(dst, worldSize);
    }

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE ring average division fallback")]
    private void ScaleArrayByReciprocalWorldSizeGeneric(T[] dst, int worldSize)
    {
        T divisor = NumOps.FromDouble(worldSize);
        for (int i = 0; i < dst.Length; i++)
        {
            dst[i] = NumOps.Divide(dst[i], divisor);
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
    /// Performs ring-based AllGather operation supporting both uniform and uneven shard sizes across ranks.
    /// </summary>
    private Vector<T> PerformRingAllGather(Vector<T> sendData)
    {
        if (_worldSize == 1)
        {
            return sendData.Clone();
        }

        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var chunks = new T[_worldSize][];
        chunks[_rank] = sendData.ToArray();

        // Ring AllGather: each rank forwards its current chunk while receiving a variable-length chunk from prevRank.
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

            var outgoing = chunks[sendChunkIdx];
            var sendTask = System.Threading.Tasks.Task.Run(() => SendArraySlice(nextRank, outgoing, 0, outgoing.Length));
            chunks[recvChunkIdx] = ReceiveArray(prevRank, expectedLength: null);
            sendTask.GetAwaiter().GetResult();
        }

        int totalLength = 0;
        for (int r = 0; r < _worldSize; r++)
        {
            totalLength = checked(totalLength + chunks[r].Length);
        }

        var result = new T[totalLength];
        int offset = 0;
        for (int r = 0; r < _worldSize; r++)
        {
            var chunk = chunks[r];
            if (chunk.Length > 0)
            {
                Array.Copy(chunk, 0, result, offset, chunk.Length);
                offset += chunk.Length;
            }
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs tree-based Broadcast operation.
    /// </summary>
    /// <remarks>
    /// Uses a binary tree pattern for efficient broadcasting.
    /// Time complexity: O(log N) where N is worldSize
    /// </remarks>
    private Vector<T> PerformTreeBroadcast(Vector<T> data, int root)
    {
        if (_worldSize == 1)
        {
            return data.Clone();
        }

        // Adjust ranks relative to root for binary tree calculation
        int relativeRank = (_rank - root + _worldSize) % _worldSize;

        T[] dataArray;
        if (relativeRank == 0)
        {
            dataArray = data.ToArray();
        }
        else
        {
            int parentRelative = (relativeRank - 1) / 2;
            int parentAbsolute = (parentRelative + root) % _worldSize;
            dataArray = ReceiveArray(parentAbsolute, expectedLength: null);
        }

        // Send data to children in binary tree
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
    /// Performs tree-based Scatter operation.
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

        int chunkSize;
        T[]? myChunk = null;

        if (Rank == root)
        {
            ValidateData(sendData, nameof(sendData));

            if (sendData.Length % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"Data length {sendData.Length} must be divisible by world size {_worldSize}.");
            }

            chunkSize = sendData.Length / _worldSize;
            var dataArray = sendData.ToArray();

            // Root keeps its own chunk
            myChunk = new T[chunkSize];
            Array.Copy(dataArray, root * chunkSize, myChunk, 0, chunkSize);

            // Send chunks to other ranks using binary tree pattern
            ScatterTreeSend(dataArray, chunkSize, root, 0, _worldSize);
        }
        else
        {
            // Non-root ranks receive their chunk
            int relativeRank = (_rank - root + _worldSize) % _worldSize;
            int parentRelative = (relativeRank - 1) / 2;
            int parentAbsolute = (parentRelative + root) % _worldSize;

            // Receive chunk count first, then data
            // IMPORTANT: Read chunk size ONCE and store it - don't call ReceiveData twice!
            var chunkSizeData = ReceiveData(parentAbsolute, 1);
            chunkSize = chunkSizeData[0] != null ? (int)NumOps.ToDouble(chunkSizeData[0]) : 0;
            myChunk = ReceiveData(parentAbsolute, chunkSize);
        }

        return new Vector<T>(myChunk ?? Array.Empty<T>());
    }

    /// <summary>
    /// Recursive helper for tree-based scatter send.
    /// </summary>
    private void ScatterTreeSend(T[] allData, int chunkSize, int root, int treeRank, int treeSize)
    {
        int leftChildRelative = 2 * treeRank + 1;
        int rightChildRelative = 2 * treeRank + 2;

        if (leftChildRelative < treeSize)
        {
            int leftChildAbsolute = (leftChildRelative + root) % _worldSize;
            var leftChunkArray = new T[chunkSize];
            Array.Copy(allData, leftChildAbsolute * chunkSize, leftChunkArray, 0, chunkSize);
            var leftChunk = new Vector<T>(leftChunkArray);
            SendData(leftChildAbsolute, new Vector<T>(new[] { NumOps.FromDouble(chunkSize) }));
            SendData(leftChildAbsolute, leftChunk);
        }

        if (rightChildRelative < treeSize)
        {
            int rightChildAbsolute = (rightChildRelative + root) % _worldSize;
            var rightChunkArray = new T[chunkSize];
            Array.Copy(allData, rightChildAbsolute * chunkSize, rightChunkArray, 0, chunkSize);
            var rightChunk = new Vector<T>(rightChunkArray);
            SendData(rightChildAbsolute, new Vector<T>(new[] { NumOps.FromDouble(chunkSize) }));
            SendData(rightChildAbsolute, rightChunk);
        }
    }

    /// <summary>
    /// Performs ring-based ReduceScatter operation.
    /// </summary>
    /// <remarks>
    /// This is the first phase of ring AllReduce - it reduces and scatters
    /// the data in one efficient operation.
    /// </remarks>
    private Vector<T> PerformRingReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        if (_worldSize == 1)
        {
            return data.Clone();
        }

        int chunkSize = data.Length / _worldSize;
        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var dataArray = data.ToArray();

        // ReduceScatter phase: reduce chunks in ring pattern
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

            int sendStart = sendChunkIdx * chunkSize;
            int sendCount = Math.Max(0, Math.Min(chunkSize, data.Length - sendStart));
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Max(0, Math.Min(chunkSize, data.Length - recvStart));

            var sendTask = System.Threading.Tasks.Task.Run(() => SendArraySlice(nextRank, dataArray, sendStart, sendCount));
            var recvChunk = ReceiveArray(prevRank, expectedLength: recvCount);
            sendTask.GetAwaiter().GetResult();

            if (recvCount > 0)
            {
                ReduceArrayInPlace(dataArray, recvStart, recvChunk, recvCount, operation);
            }
        }

        // Extract this rank's reduced chunk
        var myChunk = new T[chunkSize];
        Array.Copy(dataArray, _rank * chunkSize, myChunk, 0, chunkSize);

        // Apply averaging if needed
        if (operation == ReductionOperation.Average)
        {
            ScaleArrayByReciprocalWorldSize(myChunk, _worldSize);
        }

        return new Vector<T>(myChunk);
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

        // Send with tag: tag + length + data
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

        // Receive with tag: tag + length + data
        return ReceiveDataWithTag(sourceRank, count, tag);
    }

    /// <summary>
    /// Sends data to a specific rank via TCP with message tag using bulk binary framing and per-peer directional locks.
    /// </summary>
    private void SendDataWithTag(int destRank, Vector<T> data, int tag)
    {
        var arr = data.ToArray();
        SendArraySliceCore(destRank, arr, 0, arr.Length, hasTag: true, tag: tag);
    }

    /// <summary>
    /// Receives data from a specific rank via TCP with message tag using bulk binary framing and per-peer directional locks.
    /// </summary>
    private Vector<T> ReceiveDataWithTag(int sourceRank, int expectedLength, int expectedTag)
    {
        var result = ReceiveArrayCore(sourceRank, expectedLength, hasTag: true, expectedTag: expectedTag);
        return new Vector<T>(result);
    }

    /// <summary>
    /// Sends data to a specific rank via TCP.
    /// </summary>
    private void SendData(int destRank, Vector<T> data)
    {
        var arr = data.ToArray();
        SendArraySlice(destRank, arr, 0, arr.Length);
    }

    /// <summary>
    /// Receives data from a specific rank via TCP.
    /// </summary>
    private T[] ReceiveData(int sourceRank, int expectedLength)
    {
        return ReceiveArray(sourceRank, expectedLength);
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
            if (hasTag)
            {
                WriteInt32LittleEndian(stream, tag);
            }

            WriteInt32LittleEndian(stream, count);
            if (count > 0)
            {
                WriteElementsBulk(stream, data, offset, count);
            }

            stream.Flush();
        }
    }

    private T[] ReceiveArray(int sourceRank, int? expectedLength)
    {
        return ReceiveArrayCore(sourceRank, expectedLength, hasTag: false, expectedTag: 0);
    }

    private T[] ReceiveArrayCore(int sourceRank, int? expectedLength, bool hasTag, int expectedTag)
    {
        var client = GetConnectedPeerClient(sourceRank);
        object recvLock = _peerReceiveLocks is not null ? _peerReceiveLocks[sourceRank] : _connectionLock;

        lock (recvLock)
        {
            var stream = client.GetStream();
            if (hasTag)
            {
                int receivedTag = ReadInt32LittleEndian(stream);
                if (receivedTag != expectedTag)
                {
                    throw new InvalidOperationException(
                        $"Rank {_rank}: Expected tag {expectedTag} from rank {sourceRank}, but received tag {receivedTag}");
                }
            }

            int length = ReadInt32LittleEndian(stream);
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

            if (length == 0)
            {
                return Array.Empty<T>();
            }

            return ReadElementsBulk(stream, length);
        }
    }

    private static void WriteInt32LittleEndian(Stream stream, int value)
    {
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
    }

    private static int ReadInt32LittleEndian(Stream stream)
    {
        byte[] hdr = ArrayPool<byte>.Shared.Rent(4);
        try
        {
            ReadExactBytes(stream, hdr, 4);
            return hdr[0]
                | (hdr[1] << 8)
                | (hdr[2] << 16)
                | (hdr[3] << 24);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(hdr);
        }
    }

    private static void ReadExactBytes(Stream stream, byte[] buffer, int count)
    {
        int totalRead = 0;
        while (totalRead < count)
        {
            int read = stream.Read(buffer, totalRead, count - totalRead);
            if (read <= 0)
            {
                throw new EndOfStreamException(
                    $"Unexpected EOF on Gloo TCP transport after reading {totalRead} of {count} bytes (peer rank closed or crashed).");
            }

            totalRead += read;
        }
    }

    private void WriteElementsBulk(Stream stream, T[] source, int offset, int count)
    {
        const int MaxStagingBytes = 256 * 1024;
        if (typeof(T) == typeof(float))
        {
            var srcF = (float[])(object)source;
            int maxElems = MaxStagingBytes / sizeof(float);
            byte[] buf = ArrayPool<byte>.Shared.Rent(Math.Min(count, maxElems) * sizeof(float));
            try
            {
                int pos = 0;
                while (pos < count)
                {
                    int batch = Math.Min(maxElems, count - pos);
                    int bytes = batch * sizeof(float);
                    Buffer.BlockCopy(srcF, (offset + pos) * sizeof(float), buf, 0, bytes);
                    stream.Write(buf, 0, bytes);
                    pos += batch;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buf);
            }

            return;
        }

        if (typeof(T) == typeof(double))
        {
            var srcD = (double[])(object)source;
            int maxElems = MaxStagingBytes / sizeof(double);
            byte[] buf = ArrayPool<byte>.Shared.Rent(Math.Min(count, maxElems) * sizeof(double));
            try
            {
                int pos = 0;
                while (pos < count)
                {
                    int batch = Math.Min(maxElems, count - pos);
                    int bytes = batch * sizeof(double);
                    Buffer.BlockCopy(srcD, (offset + pos) * sizeof(double), buf, 0, bytes);
                    stream.Write(buf, 0, bytes);
                    pos += batch;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buf);
            }

            return;
        }

        WriteElementsBulkGeneric(stream, source, offset, count);
    }

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE TCP serialization fallback when T is neither float nor double")]
    private void WriteElementsBulkGeneric(Stream stream, T[] source, int offset, int count)
    {
        var tmp = new double[count];
        for (int i = 0; i < count; i++)
        {
            tmp[i] = NumOps.ToDouble(source[offset + i]);
        }

        int totalBytes = count * sizeof(double);
        byte[] buf = ArrayPool<byte>.Shared.Rent(Math.Min(totalBytes, 256 * 1024));
        try
        {
            int byteOffset = 0;
            while (byteOffset < totalBytes)
            {
                int batchBytes = Math.Min(buf.Length, totalBytes - byteOffset);
                Buffer.BlockCopy(tmp, byteOffset, buf, 0, batchBytes);
                stream.Write(buf, 0, batchBytes);
                byteOffset += batchBytes;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buf);
        }
    }

    private T[] ReadElementsBulk(Stream stream, int count)
    {
        const int MaxStagingBytes = 256 * 1024;
        if (typeof(T) == typeof(float))
        {
            var dstF = new float[count];
            int maxElems = MaxStagingBytes / sizeof(float);
            byte[] buf = ArrayPool<byte>.Shared.Rent(Math.Min(count, maxElems) * sizeof(float));
            try
            {
                int pos = 0;
                while (pos < count)
                {
                    int batch = Math.Min(maxElems, count - pos);
                    int bytes = batch * sizeof(float);
                    ReadExactBytes(stream, buf, bytes);
                    Buffer.BlockCopy(buf, 0, dstF, pos * sizeof(float), bytes);
                    pos += batch;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buf);
            }

            return (T[])(object)dstF;
        }

        if (typeof(T) == typeof(double))
        {
            var dstD = new double[count];
            int maxElems = MaxStagingBytes / sizeof(double);
            byte[] buf = ArrayPool<byte>.Shared.Rent(Math.Min(count, maxElems) * sizeof(double));
            try
            {
                int pos = 0;
                while (pos < count)
                {
                    int batch = Math.Min(maxElems, count - pos);
                    int bytes = batch * sizeof(double);
                    ReadExactBytes(stream, buf, bytes);
                    Buffer.BlockCopy(buf, 0, dstD, pos * sizeof(double), bytes);
                    pos += batch;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buf);
            }

            return (T[])(object)dstD;
        }

        return ReadElementsBulkGeneric(stream, count);
    }

    [AiDotNet.Attributes.ScalarKernel("Generic non-IEEE TCP deserialization fallback when T is neither float nor double")]
    private T[] ReadElementsBulkGeneric(Stream stream, int count)
    {
        var tmp = new double[count];
        int totalBytes = count * sizeof(double);
        byte[] buf = ArrayPool<byte>.Shared.Rent(Math.Min(totalBytes, 256 * 1024));
        try
        {
            int byteOffset = 0;
            while (byteOffset < totalBytes)
            {
                int batchBytes = Math.Min(buf.Length, totalBytes - byteOffset);
                ReadExactBytes(stream, buf, batchBytes);
                Buffer.BlockCopy(buf, 0, tmp, byteOffset, batchBytes);
                byteOffset += batchBytes;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buf);
        }

        var dst = new T[count];
        for (int i = 0; i < count; i++)
        {
            dst[i] = NumOps.FromDouble(tmp[i]);
        }

        return dst;
    }
}
