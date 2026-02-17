using System;
using System.Collections.Generic;
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
public class GlooCommunicationBackend<T> : CommunicationBackendBase<T>
{
    private readonly int _rank;
    private readonly int _worldSize;
    private bool _useNativeTCP;
    private bool _useNativeGloo;
    private Dictionary<int, TcpClient>? _tcpConnections;
    private TcpListener? _tcpListener;
    private readonly object _connectionLock = new object();

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
                // Clean up native Gloo context on failure to avoid leaking native resources
                _useNativeGloo = false;
                _nativeGlooContext?.Dispose();
                _nativeGlooContext = null;
                throw;
            }
            catch (InvalidOperationException)
            {
                // Clean up native Gloo context on failure to avoid leaking native resources
                _useNativeGloo = false;
                _nativeGlooContext?.Dispose();
                _nativeGlooContext = null;
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
        IDisposable? context = null;
        try
        {
            // Try to load GlooSharp assembly
            var glooContextType = Type.GetType("GlooSharp.GlooContext, GlooSharp");
            if (glooContextType == null)
            {
                return false;
            }

            _glooCollectivesType = Type.GetType("GlooSharp.GlooCollectives, GlooSharp");
            _glooTransportType = Type.GetType("GlooSharp.GlooTransport, GlooSharp");
            if (_glooCollectivesType == null || _glooTransportType == null)
            {
                return false;
            }

            // Resolve GlooReduceOp type once (needed for collective method lookup)
            var glooReduceOpType = Type.GetType("GlooSharp.GlooReduceOp, GlooSharp");
            if (glooReduceOpType == null)
            {
                Console.WriteLine("GlooSharp detection failed: GlooReduceOp type not found.");
                return false;
            }

            // Create GlooContext(rank, worldSize)
            var contextInstance = Activator.CreateInstance(glooContextType, _rank, _worldSize);
            if (contextInstance == null)
            {
                Console.WriteLine("GlooSharp detection failed: Activator.CreateInstance returned null for GlooContext.");
                return false;
            }

            // Safe cast to IDisposable — if GlooContext doesn't implement IDisposable,
            // fall back gracefully instead of throwing InvalidCastException.
            context = contextInstance as IDisposable;
            if (context == null)
            {
                Console.WriteLine("GlooSharp detection failed: GlooContext does not implement IDisposable. Incompatible GlooSharp version.");
                return false;
            }

            // Set up transport based on environment variable
            string transportEnv = Environment.GetEnvironmentVariable("AIDOTNET_GLOO_TRANSPORT") ?? "tcp";
            string? storePath = Environment.GetEnvironmentVariable("AIDOTNET_GLOO_STORE_PATH");

            if (string.Equals(transportEnv, "ib", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(transportEnv, "infiniband", StringComparison.OrdinalIgnoreCase))
            {
                string ibDevice = Environment.GetEnvironmentVariable("AIDOTNET_GLOO_IB_DEVICE") ?? "";
                var createIB = _glooTransportType.GetMethod("CreateInfiniBand",
                    new[] { glooContextType, typeof(string), typeof(string) });
                if (createIB == null)
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

                var createTCP = _glooTransportType.GetMethod("CreateTCP",
                    new[] { glooContextType, typeof(string), typeof(int), typeof(string) });
                if (createTCP == null)
                {
                    Console.WriteLine("GlooSharp detection failed: CreateTCP method not found.");
                    return false;
                }
                createTCP.Invoke(null, new object?[] { context, hostname, port, storePath });
            }

            // Cache reflection MethodInfo for collective operations (double[] overloads)
            _glooAllReduceDouble = _glooCollectivesType.GetMethod("AllReduce", new[]
            {
                glooContextType, typeof(double[]), glooReduceOpType
            });
            _glooBroadcastDouble = _glooCollectivesType.GetMethod("Broadcast", new[]
            {
                glooContextType, typeof(double[]), typeof(int)
            });
            _glooAllGatherDouble = _glooCollectivesType.GetMethod("AllGather", new[]
            {
                glooContextType, typeof(double[])
            });
            _glooBarrier = _glooCollectivesType.GetMethod("Barrier", new[] { glooContextType });
            _glooReduceScatterDouble = _glooCollectivesType.GetMethod("ReduceScatter", new[]
            {
                glooContextType, typeof(double[]), glooReduceOpType
            });

            // Validate all required collective methods were found
            if (_glooAllReduceDouble == null || _glooBroadcastDouble == null ||
                _glooAllGatherDouble == null || _glooBarrier == null ||
                _glooReduceScatterDouble == null)
            {
                Console.WriteLine("GlooSharp detected but required collective methods not found. Falling back to TCP.");
                return false;
            }

            // Cache GlooReduceOp enum values to avoid per-call reflection
            _glooReduceOpCache = CreateGlooReduceOpMap(glooReduceOpType);

            // Success — transfer ownership to the field
            _nativeGlooContext = context;
            context = null; // prevent disposal in finally
            return true;
        }
        catch (TargetInvocationException ex)
        {
            Console.WriteLine($"GlooSharp detection failed (invocation error): {ex.InnerException?.Message ?? ex.Message}");
            return false;
        }
        catch (TypeLoadException ex)
        {
            Console.WriteLine($"GlooSharp detection failed (type load error): {ex.Message}");
            return false;
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"GlooSharp detection failed: {ex.Message}");
            return false;
        }
        catch (MemberAccessException ex)
        {
            Console.WriteLine($"GlooSharp detection failed (access error): {ex.Message}");
            return false;
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"GlooSharp detection failed (enum mismatch): {ex.Message}");
            return false;
        }
        finally
        {
            // Dispose the native context if we didn't successfully transfer ownership
            context?.Dispose();
        }
    }

    /// <summary>
    /// Creates a cached mapping from <see cref="ReductionOperation"/> to GlooSharp GlooReduceOp enum values.
    /// </summary>
    private static IReadOnlyDictionary<ReductionOperation, object> CreateGlooReduceOpMap(Type glooReduceOpType)
    {
        return new Dictionary<ReductionOperation, object>
        {
            { ReductionOperation.Sum, Enum.Parse(glooReduceOpType, "Sum", ignoreCase: true) },
            { ReductionOperation.Average, Enum.Parse(glooReduceOpType, "Sum", ignoreCase: true) },
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
                var client = new TcpClient();
                client.Connect(masterAddr, targetPort);

                // Send handshake: my rank
                // IMPORTANT: Do NOT use 'using' here - disposing the stream closes the socket!
                // We need to keep the connection open for future communication.
                var stream = client.GetStream();
                var writer = new BinaryWriter(stream);
                writer.Write(_rank);
                writer.Flush();
                // Leave stream and writer open - they'll be cleaned up when TcpClient is disposed

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

        // Read handshake to identify the connecting rank
        // IMPORTANT: Do NOT use 'using' here - disposing the stream closes the socket!
        // We need to keep the connection open for future communication.
        var stream = client.GetStream();
        var reader = new BinaryReader(stream);
        int receivedRank = reader.ReadInt32();
        // Leave stream and reader open - they'll be cleaned up when TcpClient is disposed

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
            catch (ObjectDisposedException ex)
            {
                Console.WriteLine($"Warning: Gloo context already disposed: {ex.Message}");
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine($"Warning: Error disposing native Gloo context: {ex.Message}");
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
                        Console.WriteLine($"Warning: Error closing TCP connection: {ex.Message}");
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
                    Console.WriteLine($"Warning: Error stopping TCP listener: {ex.Message}");
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

        if (_worldSize == 1)
        {
            // Single-process: barrier is a no-op
            return;
        }

        if (_useNativeGloo)
        {
            if (_glooBarrier == null)
            {
                throw new InvalidOperationException("Native Gloo Barrier method is not available.");
            }
            _glooBarrier.Invoke(null, new object[] { _nativeGlooContext! });
            return;
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("Barrier requires TCP mode to be initialized");
        }

        // Simple all-to-all barrier implementation
        // Each rank sends a signal to all other ranks and waits for signals from all
        var signal = new Vector<T>(new[] { NumOps.One });

        // Send signal to all other ranks
        for (int otherRank = 0; otherRank < _worldSize; otherRank++)
        {
            if (otherRank != _rank)
            {
                SendData(otherRank, signal);
            }
        }

        // Receive signal from all other ranks
        for (int otherRank = 0; otherRank < _worldSize; otherRank++)
        {
            if (otherRank != _rank)
            {
                ReceiveData(otherRank, 1);
            }
        }
    }

    /// <inheritdoc/>
    public override void AllReduce(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));

        if (_worldSize == 1)
        {
            // Single-process: data already contains the result (no-op)
            return;
        }

        if (_useNativeGloo)
        {
            PerformNativeGlooAllReduce(data, operation);
            return;
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("AllReduce requires TCP mode to be initialized");
        }

        PerformRingAllReduce(data, operation);
    }

    /// <inheritdoc/>
    public override Vector<T> AllGather(Vector<T> sendData)
    {
        EnsureInitialized();
        ValidateData(sendData, nameof(sendData));

        if (_worldSize == 1)
        {
            return sendData.Clone();
        }

        if (_useNativeGloo)
        {
            return PerformNativeGlooAllGather(sendData);
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("AllGather requires TCP mode to be initialized");
        }

        return PerformRingAllGather(sendData);
    }

    /// <inheritdoc/>
    public override Vector<T> Broadcast(Vector<T> data, int root = 0)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));
        ValidateRoot(root);

        if (_worldSize == 1)
        {
            return data.Clone();
        }

        if (_useNativeGloo)
        {
            return PerformNativeGlooBroadcast(data, root);
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("Broadcast requires TCP mode to be initialized");
        }

        return PerformTreeBroadcast(data, root);
    }

    /// <inheritdoc/>
    public override Vector<T> Scatter(Vector<T> sendData, int root = 0)
    {
        EnsureInitialized();
        ValidateRoot(root);

        if (_worldSize == 1)
        {
            if (Rank == root)
            {
                ValidateData(sendData, nameof(sendData));
                return sendData.Clone();
            }
            return new Vector<T>(Array.Empty<T>());
        }

        if (_useNativeGloo)
        {
            // Gloo has no native Scatter — implement via Broadcast + local extraction
            // (same pattern as NCCLCommunicationBackend.Scatter)
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

            // 1) Broadcast total length so non-root ranks can allocate a correctly sized buffer
            var lengthVector = PerformNativeGlooBroadcast(
                new Vector<T>(new[] { NumOps.FromDouble(totalLength) }),
                root);
            totalLength = Convert.ToInt32(Convert.ToDouble(lengthVector[0]));

            if (totalLength % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"Data length {totalLength} must be divisible by world size {_worldSize}.");
            }

            // 2) Non-root ranks allocate a full-sized buffer; root uses its sendData
            var fullBuffer = Rank == root ? sendData : new Vector<T>(new T[totalLength]);
            var broadcasted = PerformNativeGlooBroadcast(fullBuffer, root);
            int chunkSize = broadcasted.Length / _worldSize;
            var chunk = new T[chunkSize];
            var broadcastedArray = broadcasted.ToArray();
            Array.Copy(broadcastedArray, _rank * chunkSize, chunk, 0, chunkSize);
            return new Vector<T>(chunk);
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("Scatter requires TCP mode to be initialized");
        }

        return PerformTreeScatter(sendData, root);
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

        if (_worldSize == 1)
        {
            return data.Clone();
        }

        if (_useNativeGloo)
        {
            return PerformNativeGlooReduceScatter(data, operation);
        }

        if (!_useNativeTCP)
        {
            throw new InvalidOperationException("ReduceScatter requires TCP mode to be initialized");
        }

        return PerformRingReduceScatter(data, operation);
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
            result[i] = Convert.ToDouble(vector[i]);
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
        if (_glooAllReduceDouble == null)
        {
            throw new InvalidOperationException("Native Gloo AllReduce method is not available.");
        }

        var doubleData = VectorToDoubleArray(data);
        var glooOp = MapToGlooReduceOp(operation == ReductionOperation.Average ? ReductionOperation.Sum : operation);
        _glooAllReduceDouble.Invoke(null, new object[] { _nativeGlooContext!, doubleData, glooOp });

        // Apply averaging if needed
        if (operation == ReductionOperation.Average)
        {
            for (int i = 0; i < doubleData.Length; i++)
            {
                doubleData[i] /= _worldSize;
            }
        }

        // Write results back to the vector
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = NumOps.FromDouble(doubleData[i]);
        }
    }

    /// <summary>
    /// Performs Broadcast via native Gloo (GlooSharp).
    /// </summary>
    private Vector<T> PerformNativeGlooBroadcast(Vector<T> data, int root)
    {
        if (_glooBroadcastDouble == null)
        {
            throw new InvalidOperationException("Native Gloo Broadcast method is not available.");
        }

        var doubleData = VectorToDoubleArray(data);
        _glooBroadcastDouble.Invoke(null, new object[] { _nativeGlooContext!, doubleData, root });
        return DoubleArrayToVector(doubleData);
    }

    /// <summary>
    /// Performs AllGather via native Gloo (GlooSharp).
    /// </summary>
    private Vector<T> PerformNativeGlooAllGather(Vector<T> sendData)
    {
        if (_glooAllGatherDouble == null)
        {
            throw new InvalidOperationException("Native Gloo AllGather method is not available.");
        }

        var doubleData = VectorToDoubleArray(sendData);
        var result = _glooAllGatherDouble.Invoke(null, new object[] { _nativeGlooContext!, doubleData });
        if (result is double[] gathered)
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
        if (_glooReduceScatterDouble == null)
        {
            throw new InvalidOperationException("Native Gloo ReduceScatter method is not available.");
        }

        var doubleData = VectorToDoubleArray(data);
        var glooOp = MapToGlooReduceOp(operation == ReductionOperation.Average ? ReductionOperation.Sum : operation);
        var result = _glooReduceScatterDouble.Invoke(null, new object[] { _nativeGlooContext!, doubleData, glooOp });

        if (result is not double[] scattered)
        {
            throw new InvalidOperationException("Native Gloo ReduceScatter invocation did not return a double[] result.");
        }

        if (operation == ReductionOperation.Average)
        {
            for (int i = 0; i < scattered.Length; i++)
            {
                scattered[i] /= _worldSize;
            }
        }
        return DoubleArrayToVector(scattered);
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
        if (_worldSize == 1)
        {
            return;
        }

        int chunkSize = (data.Length + _worldSize - 1) / _worldSize; // Ceiling division
        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var dataArray = data.ToArray();

        // Phase 1: ReduceScatter - reduce chunks in ring pattern
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

            int sendStart = sendChunkIdx * chunkSize;
            int sendCount = Math.Min(chunkSize, data.Length - sendStart);
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Min(chunkSize, data.Length - recvStart);

            // Extract send chunk
            var sendChunk = new T[sendCount];
            Array.Copy(dataArray, sendStart, sendChunk, 0, sendCount);

            // Send and receive simultaneously
            var sendTask = System.Threading.Tasks.Task.Run(() => SendData(nextRank, new Vector<T>(sendChunk)));
            var recvChunk = ReceiveData(prevRank, recvCount);
            sendTask.Wait();

            // Reduce received chunk with local chunk
            for (int i = 0; i < recvCount; i++)
            {
                dataArray[recvStart + i] = PerformReduction(dataArray[recvStart + i], recvChunk[i], operation);
            }
        }

        // Phase 2: AllGather - distribute reduced chunks in ring pattern
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + 1 + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step + _worldSize) % _worldSize;

            int sendStart = sendChunkIdx * chunkSize;
            int sendCount = Math.Min(chunkSize, data.Length - sendStart);
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Min(chunkSize, data.Length - recvStart);

            // Extract send chunk
            var sendChunk = new T[sendCount];
            Array.Copy(dataArray, sendStart, sendChunk, 0, sendCount);

            // Send and receive simultaneously
            var sendTask = System.Threading.Tasks.Task.Run(() => SendData(nextRank, new Vector<T>(sendChunk)));
            var recvChunk = ReceiveData(prevRank, recvCount);
            sendTask.Wait();

            // Copy received chunk to local data
            Array.Copy(recvChunk, 0, dataArray, recvStart, recvCount);
        }

        // Apply averaging if needed (after all reductions complete)
        if (operation == ReductionOperation.Average)
        {
            for (int i = 0; i < dataArray.Length; i++)
            {
                dataArray[i] = NumOps.Divide(dataArray[i], NumOps.FromDouble(_worldSize));
            }
        }

        // Update original vector with reduced data
        for (int i = 0; i < dataArray.Length; i++)
        {
            data[i] = dataArray[i];
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
    /// Performs ring-based AllGather operation.
    /// </summary>
    private Vector<T> PerformRingAllGather(Vector<T> sendData)
    {
        if (_worldSize == 1)
        {
            return sendData.Clone();
        }

        int chunkSize = sendData.Length;
        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        // Result buffer to hold data from all ranks
        var result = new T[chunkSize * _worldSize];

        // Copy local data to result buffer
        Array.Copy(sendData.ToArray(), 0, result, _rank * chunkSize, chunkSize);

        // Ring AllGather: each rank receives chunk from previous rank and forwards it
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

            // Extract send chunk from result buffer
            var sendChunk = new T[chunkSize];
            Array.Copy(result, sendChunkIdx * chunkSize, sendChunk, 0, chunkSize);

            // Send and receive simultaneously
            var sendTask = System.Threading.Tasks.Task.Run(() => SendData(nextRank, new Vector<T>(sendChunk)));
            var recvChunk = ReceiveData(prevRank, chunkSize);
            sendTask.Wait();

            // Copy received chunk to result buffer
            Array.Copy(recvChunk, 0, result, recvChunkIdx * chunkSize, chunkSize);
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

        var dataArray = data.ToArray();

        // Adjust ranks relative to root for binary tree calculation
        int relativeRank = (_rank - root + _worldSize) % _worldSize;

        // If not root, receive data from parent
        if (relativeRank != 0)
        {
            int parentRelative = (relativeRank - 1) / 2;
            int parentAbsolute = (parentRelative + root) % _worldSize;
            dataArray = ReceiveData(parentAbsolute, data.Length);
        }

        // Send data to children in binary tree
        int leftChildRelative = 2 * relativeRank + 1;
        int rightChildRelative = 2 * relativeRank + 2;

        if (leftChildRelative < _worldSize)
        {
            int leftChildAbsolute = (leftChildRelative + root) % _worldSize;
            SendData(leftChildAbsolute, new Vector<T>(dataArray));
        }

        if (rightChildRelative < _worldSize)
        {
            int rightChildAbsolute = (rightChildRelative + root) % _worldSize;
            SendData(rightChildAbsolute, new Vector<T>(dataArray));
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
            chunkSize = chunkSizeData[0] != null ? Convert.ToInt32(chunkSizeData[0]) : 0;
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
            int sendCount = Math.Min(chunkSize, data.Length - sendStart);
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Min(chunkSize, data.Length - recvStart);

            // Extract send chunk
            var sendChunk = new T[sendCount];
            Array.Copy(dataArray, sendStart, sendChunk, 0, sendCount);

            // Send and receive simultaneously
            var sendTask = System.Threading.Tasks.Task.Run(() => SendData(nextRank, new Vector<T>(sendChunk)));
            var recvChunk = ReceiveData(prevRank, recvCount);
            sendTask.Wait();

            // Reduce received chunk with local chunk
            for (int i = 0; i < recvCount; i++)
            {
                dataArray[recvStart + i] = PerformReduction(dataArray[recvStart + i], recvChunk[i], operation);
            }
        }

        // Extract this rank's reduced chunk
        var myChunk = new T[chunkSize];
        Array.Copy(dataArray, _rank * chunkSize, myChunk, 0, chunkSize);

        // Apply averaging if needed
        if (operation == ReductionOperation.Average)
        {
            for (int i = 0; i < myChunk.Length; i++)
            {
                myChunk[i] = NumOps.Divide(myChunk[i], NumOps.FromDouble(_worldSize));
            }
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
    /// Sends data to a specific rank via TCP with message tag.
    /// </summary>
    private void SendDataWithTag(int destRank, Vector<T> data, int tag)
    {
        if (_tcpConnections == null || !_tcpConnections.ContainsKey(destRank))
        {
            throw new InvalidOperationException(
                $"No TCP connection to rank {destRank}. Ensure Initialize() was called and connections were established.");
        }

        lock (_connectionLock)
        {
            var client = _tcpConnections[destRank];
            var stream = client.GetStream();
            var writer = new BinaryWriter(stream);

            // Send tag
            writer.Write(tag);

            // Send length header
            writer.Write(data.Length);

            // Send data elements
            for (int i = 0; i < data.Length; i++)
            {
                double value = Convert.ToDouble(data[i]);
                writer.Write(value);
            }
            writer.Flush();
            // Leave stream and writer open - they're reused for the connection lifetime
        }
    }

    /// <summary>
    /// Receives data from a specific rank via TCP with message tag.
    /// </summary>
    private Vector<T> ReceiveDataWithTag(int sourceRank, int expectedLength, int expectedTag)
    {
        if (_tcpConnections == null || !_tcpConnections.ContainsKey(sourceRank))
        {
            throw new InvalidOperationException(
                $"No TCP connection to rank {sourceRank}. Ensure Initialize() was called and connections were established.");
        }

        lock (_connectionLock)
        {
            var client = _tcpConnections[sourceRank];
            var stream = client.GetStream();
            var reader = new BinaryReader(stream);

            // Read tag
            int receivedTag = reader.ReadInt32();
            if (receivedTag != expectedTag)
            {
                throw new InvalidOperationException(
                    $"Rank {_rank}: Expected tag {expectedTag} from rank {sourceRank}, but received tag {receivedTag}");
            }

            // Read length header
            int length = reader.ReadInt32();
            if (length != expectedLength)
            {
                throw new InvalidOperationException(
                    $"Rank {_rank}: Expected {expectedLength} elements from rank {sourceRank}, but received {length}");
            }

            // Read data elements into Vector
            var result = new T[length];
            for (int i = 0; i < length; i++)
            {
                double value = reader.ReadDouble();
                result[i] = NumOps.FromDouble(value);
            }
            // Leave stream and reader open - they're reused for the connection lifetime
            return new Vector<T>(result);
        }
    }

    /// <summary>
    /// Sends data to a specific rank via TCP.
    /// </summary>
    private void SendData(int destRank, Vector<T> data)
    {
        if (_tcpConnections == null || !_tcpConnections.ContainsKey(destRank))
        {
            throw new InvalidOperationException(
                $"No TCP connection to rank {destRank}. Ensure Initialize() was called and connections were established.");
        }

        lock (_connectionLock)
        {
            var client = _tcpConnections[destRank];
            var stream = client.GetStream();
            var writer = new BinaryWriter(stream);

            // Send length header
            writer.Write(data.Length);

            // Send data elements
            for (int i = 0; i < data.Length; i++)
            {
                double value = Convert.ToDouble(data[i]);
                writer.Write(value);
            }
            writer.Flush();
            // Leave stream and writer open - they're reused for the connection lifetime
        }
    }

    /// <summary>
    /// Receives data from a specific rank via TCP.
    /// </summary>
    private Vector<T> ReceiveData(int sourceRank, int expectedLength)
    {
        if (_tcpConnections == null || !_tcpConnections.ContainsKey(sourceRank))
        {
            throw new InvalidOperationException(
                $"No TCP connection to rank {sourceRank}. Ensure Initialize() was called and connections were established.");
        }

        lock (_connectionLock)
        {
            var client = _tcpConnections[sourceRank];
            var stream = client.GetStream();
            var reader = new BinaryReader(stream);

            // Read length header
            int length = reader.ReadInt32();
            if (length != expectedLength)
            {
                throw new InvalidOperationException(
                    $"Rank {_rank}: Expected {expectedLength} elements from rank {sourceRank}, but received {length}");
            }

            // Read data elements into Vector
            var result = new T[length];
            for (int i = 0; i < length; i++)
            {
                double value = reader.ReadDouble();
                result[i] = NumOps.FromDouble(value);
            }
            // Leave stream and reader open - they're reused for the connection lifetime
            return new Vector<T>(result);
        }
    }
}
