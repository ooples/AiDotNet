using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

// NCCL data types (must be outside generic type for P/Invoke)
internal enum ncclDataType_t
{
    ncclInt8 = 0, ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2, ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6, ncclHalf = 6,
    ncclFloat32 = 7, ncclFloat = 7,
    ncclFloat64 = 8, ncclDouble = 8,
    ncclBfloat16 = 9,
    ncclNumTypes = 10
}

// NCCL reduction operations
internal enum ncclRedOp_t
{
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
    ncclNumOps = 5
}

// NCCL result codes
internal enum ncclResult_t
{
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8
}

/// <summary>
/// NVIDIA NCCL-based communication backend for GPU-to-GPU communication.
/// </summary>
/// <remarks>
/// <para><b>Overview:</b>
/// NCCL (NVIDIA Collective Communications Library) is optimized for multi-GPU communication
/// on NVIDIA GPUs. It provides highly optimized implementations of collective operations
/// that take advantage of NVLink, PCIe, and network topology for maximum throughput.
/// </para>
/// <para><b>Features:</b>
/// - Optimized for NVIDIA GPUs (NVLink, NVSwitch awareness)
/// - Near-optimal bandwidth utilization
/// - Supports multi-node multi-GPU configurations
/// - Ring and tree algorithms for different collective operations
/// - Essential for high-performance multi-GPU training
/// </para>
/// <para><b>Architecture:</b>
/// This backend supports two modes of operation:
///
/// 1. **Native NCCL Mode:**
///    Uses NCCL library with actual GPU memory for collective operations.
///    Requires CUDA toolkit and NCCL library. Provides near-optimal GPU bandwidth.
///
/// 2. **CPU Fallback Mode:**
///    When NCCL/CUDA not available, uses TCP-based ring algorithms similar to Gloo.
///    Allows development and testing on systems without NVIDIA GPUs.
///
/// The implementation features:
/// - Automatic NCCL detection and initialization
/// - TCP-based unique ID distribution for multi-node setup
/// - Environment-based rendezvous (AIDOTNET_MASTER_ADDR, AIDOTNET_MASTER_PORT)
/// - Proper CUDA stream synchronization
/// - Memory-efficient GPU operations
/// </para>
/// <para><b>Requirements for GPU Mode:</b>
/// - NVIDIA GPUs (compute capability 3.0+)
/// - CUDA toolkit 10.0+
/// - NCCL library 2.0+
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> NCCLCommunicationBackend provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class NCCLCommunicationBackend<T> : CommunicationBackendBase<T>
{
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly int _deviceId;
    private bool _ncclAvailable;
    private IntPtr _ncclComm;
    private IntPtr _cudaStream;

    // GPU memory buffers
    private IntPtr _gpuSendBuffer;
    private IntPtr _gpuRecvBuffer;
    private int _bufferSize;

    // TCP connections for fallback mode or unique ID distribution
    private Dictionary<int, TcpClient>? _tcpConnections;
    private TcpListener? _tcpListener;
    private readonly object _connectionLock = new();
    private bool _useTcpFallback;

    /// <summary>
    /// Creates a new NCCL communication backend.
    /// </summary>
    /// <param name="rank">This process's rank</param>
    /// <param name="worldSize">Total number of processes</param>
    /// <param name="deviceId">CUDA device ID for this process (default: use rank)</param>
    public NCCLCommunicationBackend(int rank = 0, int worldSize = 1, int deviceId = -1)
    {
        _rank = rank;
        _worldSize = worldSize;
        _deviceId = deviceId >= 0 ? deviceId : rank;
        _ncclAvailable = false;
        _ncclComm = IntPtr.Zero;
        _cudaStream = IntPtr.Zero;
        _gpuSendBuffer = IntPtr.Zero;
        _gpuRecvBuffer = IntPtr.Zero;
        _bufferSize = 0;
        _useTcpFallback = false;
    }

    /// <inheritdoc/>
    public override int Rank => _rank;

    /// <inheritdoc/>
    public override int WorldSize => _worldSize;

    /// <inheritdoc/>
    protected override void OnInitialize()
    {
        _tcpConnections = new Dictionary<int, TcpClient>();

        // Try to initialize NCCL
        try
        {
            InitializeNCCL();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"NCCL initialization failed: {ex.Message}");
            _ncclAvailable = false;
        }

        // If NCCL not available, use TCP fallback
        if (!_ncclAvailable)
        {
            if (_worldSize > 1)
            {
                Console.WriteLine("NCCL not available. Using TCP-based collective operations.");
                Console.WriteLine("For optimal GPU performance, install CUDA toolkit and NCCL library.");
                _useTcpFallback = true;
                InitializeTCPConnections();
            }
            else
            {
                Console.WriteLine("NCCLCommunicationBackend: Single-process mode (worldSize=1).");
            }
        }
    }

    /// <summary>
    /// Initializes NCCL communicator with proper multi-process setup.
    /// </summary>
    private void InitializeNCCL()
    {
        // Check if NCCL library is available
        ncclResult_t result = NcclNativeMethods.ncclGetVersion(out int version);
        if (result != ncclResult_t.ncclSuccess)
        {
            throw new InvalidOperationException("NCCL library not found or incompatible.");
        }

        Console.WriteLine($"NCCL library detected (version: {version / 1000}.{(version % 1000) / 100}.{version % 100})");

        // Set CUDA device
        CudaNativeMethods.cudaSetDevice(_deviceId);

        // Create CUDA stream
        result = CudaNativeMethods.cudaStreamCreate(out _cudaStream);
        if (result != ncclResult_t.ncclSuccess)
        {
            throw new InvalidOperationException($"Failed to create CUDA stream: {result}");
        }

        if (_worldSize == 1)
        {
            // Single-process NCCL initialization
            InitializeSingleProcessNCCL();
        }
        else
        {
            // Multi-process NCCL initialization
            InitializeMultiProcessNCCL();
        }

        _ncclAvailable = true;
        Console.WriteLine($"NCCL initialized successfully on GPU {_deviceId} (rank {_rank}/{_worldSize})");
    }

    /// <summary>
    /// Initializes NCCL for single-process mode.
    /// </summary>
    private void InitializeSingleProcessNCCL()
    {
        // Get unique ID
        var uniqueId = new NcclUniqueId();
        ncclResult_t result = NcclNativeMethods.ncclGetUniqueId(ref uniqueId);
        if (result != ncclResult_t.ncclSuccess)
        {
            throw new InvalidOperationException($"Failed to get NCCL unique ID: {result}");
        }

        // Initialize communicator
        result = NcclNativeMethods.ncclCommInitRank(out _ncclComm, 1, uniqueId, 0);
        if (result != ncclResult_t.ncclSuccess)
        {
            throw new InvalidOperationException($"Failed to initialize NCCL communicator: {result}");
        }
    }

    /// <summary>
    /// Initializes NCCL for multi-process mode with TCP-based unique ID distribution.
    /// </summary>
    private void InitializeMultiProcessNCCL()
    {
        // Ensure TCP is set up for unique ID distribution
        InitializeTCPConnections();

        NcclUniqueId uniqueId;

        if (_rank == 0)
        {
            // Rank 0 creates the unique ID
            uniqueId = new NcclUniqueId();
            ncclResult_t result = NcclNativeMethods.ncclGetUniqueId(ref uniqueId);
            if (result != ncclResult_t.ncclSuccess)
            {
                throw new InvalidOperationException($"Failed to get NCCL unique ID: {result}");
            }

            // Broadcast unique ID to all other ranks via TCP
            BroadcastUniqueIdTcp(uniqueId);
        }
        else
        {
            // Non-root ranks receive the unique ID
            uniqueId = ReceiveUniqueIdTcp();
        }

        // Initialize communicator with the shared unique ID
        ncclResult_t initResult = NcclNativeMethods.ncclCommInitRank(out _ncclComm, _worldSize, uniqueId, _rank);
        if (initResult != ncclResult_t.ncclSuccess)
        {
            throw new InvalidOperationException($"Failed to initialize NCCL communicator: {initResult}");
        }
    }

    /// <summary>
    /// Broadcasts NCCL unique ID from rank 0 to all other ranks.
    /// </summary>
    private void BroadcastUniqueIdTcp(NcclUniqueId uniqueId)
    {
        byte[] idBytes = uniqueId.ToBytes();

        for (int destRank = 1; destRank < _worldSize; destRank++)
        {
            if (_tcpConnections == null || !_tcpConnections.ContainsKey(destRank))
            {
                throw new InvalidOperationException($"No TCP connection to rank {destRank}");
            }

            lock (_connectionLock)
            {
                var client = _tcpConnections[destRank];
                var stream = client.GetStream();
                var writer = new BinaryWriter(stream);
                writer.Write(idBytes.Length);
                writer.Write(idBytes);
                writer.Flush();
            }
        }
    }

    /// <summary>
    /// Receives NCCL unique ID from rank 0.
    /// </summary>
    private NcclUniqueId ReceiveUniqueIdTcp()
    {
        if (_tcpConnections == null || !_tcpConnections.ContainsKey(0))
        {
            throw new InvalidOperationException("No TCP connection to rank 0");
        }

        lock (_connectionLock)
        {
            var client = _tcpConnections[0];
            var stream = client.GetStream();
            var reader = new BinaryReader(stream);
            int length = reader.ReadInt32();
            byte[] idBytes = reader.ReadBytes(length);
            return NcclUniqueId.FromBytes(idBytes);
        }
    }

    /// <summary>
    /// Initializes TCP connections for multi-process communication.
    /// </summary>
    private void InitializeTCPConnections()
    {
        if (_tcpConnections != null && _tcpConnections.Count > 0)
        {
            return; // Already initialized
        }

        _tcpConnections ??= new Dictionary<int, TcpClient>();

        if (_worldSize == 1)
        {
            return;
        }

        string? masterAddr = Environment.GetEnvironmentVariable("AIDOTNET_MASTER_ADDR");
        string? masterPortStr = Environment.GetEnvironmentVariable("AIDOTNET_MASTER_PORT");

        if (string.IsNullOrEmpty(masterAddr) || string.IsNullOrEmpty(masterPortStr))
        {
            throw new InvalidOperationException(
                "Multi-GPU NCCL requires environment variables:\n" +
                "- AIDOTNET_MASTER_ADDR: IP address of rank 0 (e.g., 192.168.1.10 or localhost)\n" +
                "- AIDOTNET_MASTER_PORT: Base port number (e.g., 29500)");
        }

        if (!int.TryParse(masterPortStr, out int basePort))
        {
            throw new InvalidOperationException($"Invalid AIDOTNET_MASTER_PORT: {masterPortStr}");
        }

        // Start TCP listener
        int myPort = basePort + _rank;
        _tcpListener = new TcpListener(IPAddress.Any, myPort);
        _tcpListener.Start();

        // Connect to ranks with lower rank number
        for (int otherRank = 0; otherRank < _rank; otherRank++)
        {
            ConnectToRank(otherRank, masterAddr, basePort);
        }

        // Accept connections from ranks with higher rank number
        int numExpectedConnections = _worldSize - _rank - 1;
        for (int i = 0; i < numExpectedConnections; i++)
        {
            AcceptConnectionFromAnyRank();
        }

        Console.WriteLine($"Rank {_rank}: TCP connections established for NCCL setup ({_tcpConnections.Count} peers)");
    }

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

                var stream = client.GetStream();
                var writer = new BinaryWriter(stream);
                writer.Write(_rank);
                writer.Flush();

                lock (_connectionLock)
                {
                    _tcpConnections![targetRank] = client;
                }
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
                        $"Failed to connect to rank {targetRank} at {masterAddr}:{targetPort}");
                }
            }
        }
    }

    private void AcceptConnectionFromAnyRank()
    {
        if (_tcpListener == null)
        {
            throw new InvalidOperationException("TCP listener not initialized");
        }

        var client = _tcpListener.AcceptTcpClient();
        var stream = client.GetStream();
        var reader = new BinaryReader(stream);
        int receivedRank = reader.ReadInt32();

        if (receivedRank <= _rank || receivedRank >= _worldSize)
        {
            client.Close();
            throw new InvalidOperationException($"Invalid connection from rank {receivedRank}");
        }

        lock (_connectionLock)
        {
            _tcpConnections![receivedRank] = client;
        }
    }

    /// <inheritdoc/>
    protected override void OnShutdown()
    {
        // Free GPU buffers
        if (_gpuSendBuffer != IntPtr.Zero)
        {
            CudaNativeMethods.cudaFree(_gpuSendBuffer);
            _gpuSendBuffer = IntPtr.Zero;
        }
        if (_gpuRecvBuffer != IntPtr.Zero)
        {
            CudaNativeMethods.cudaFree(_gpuRecvBuffer);
            _gpuRecvBuffer = IntPtr.Zero;
        }

        // Destroy NCCL communicator
        if (_ncclComm != IntPtr.Zero)
        {
            NcclNativeMethods.ncclCommDestroy(_ncclComm);
            _ncclComm = IntPtr.Zero;
        }

        // Destroy CUDA stream
        if (_cudaStream != IntPtr.Zero)
        {
            CudaNativeMethods.cudaStreamDestroy(_cudaStream);
            _cudaStream = IntPtr.Zero;
        }

        // Close TCP connections
        if (_tcpConnections != null)
        {
            lock (_connectionLock)
            {
                foreach (var connection in _tcpConnections.Values)
                {
                    try { connection.Close(); } catch { }
                }
                _tcpConnections.Clear();
            }
        }

        _tcpListener?.Stop();
        _tcpListener = null;
    }

    /// <inheritdoc/>
    public override void Barrier()
    {
        EnsureInitialized();

        if (_worldSize == 1)
        {
            return;
        }

        if (_ncclAvailable)
        {
            // NCCL barrier via dummy AllReduce
            var dummy = new Vector<T>(new T[1]);
            dummy[0] = NumOps.FromDouble(0);
            AllReduce(dummy, ReductionOperation.Sum);
        }
        else if (_useTcpFallback)
        {
            PerformTcpBarrier();
        }
    }

    /// <inheritdoc/>
    public override void AllReduce(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));

        if (_worldSize == 1)
        {
            // Single-process: apply average if needed, otherwise no-op
            if (operation == ReductionOperation.Average)
            {
                // Average of single value is the value itself
            }
            return;
        }

        if (_ncclAvailable)
        {
            PerformNcclAllReduce(data, operation);
        }
        else if (_useTcpFallback)
        {
            PerformTcpAllReduce(data, operation);
        }
    }

    /// <summary>
    /// Performs AllReduce using NCCL with GPU memory.
    /// </summary>
    private void PerformNcclAllReduce(Vector<T> data, ReductionOperation operation)
    {
        int count = data.Length;
        int byteSize = count * Marshal.SizeOf<T>();

        // Ensure GPU buffers are allocated
        EnsureGpuBuffers(byteSize);

        // Copy data to GPU
        var dataArray = data.ToArray();
        var handle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
        try
        {
            CudaNativeMethods.cudaMemcpyAsync(
                _gpuSendBuffer,
                handle.AddrOfPinnedObject(),
                (IntPtr)byteSize,
                CudaMemcpyKind.HostToDevice,
                _cudaStream);

            // Perform NCCL AllReduce
            ncclRedOp_t ncclOp = GetNcclOperation(operation);
            ncclDataType_t ncclType = GetNcclDataType();

            ncclResult_t result = NcclNativeMethods.ncclAllReduce(
                _gpuSendBuffer,
                _gpuRecvBuffer,
                (IntPtr)count,
                ncclType,
                ncclOp,
                _ncclComm,
                _cudaStream);

            if (result != ncclResult_t.ncclSuccess)
            {
                throw new InvalidOperationException($"NCCL AllReduce failed: {result}");
            }

            // Synchronize stream
            CudaNativeMethods.cudaStreamSynchronize(_cudaStream);

            // Copy result back to host
            CudaNativeMethods.cudaMemcpyAsync(
                handle.AddrOfPinnedObject(),
                _gpuRecvBuffer,
                (IntPtr)byteSize,
                CudaMemcpyKind.DeviceToHost,
                _cudaStream);

            CudaNativeMethods.cudaStreamSynchronize(_cudaStream);
        }
        finally
        {
            handle.Free();
        }

        // Copy result back to vector
        for (int i = 0; i < count; i++)
        {
            data[i] = dataArray[i];
        }

        // Apply averaging if needed (NCCL Sum was used)
        if (operation == ReductionOperation.Average)
        {
            for (int i = 0; i < count; i++)
            {
                data[i] = NumOps.Divide(data[i], NumOps.FromDouble(_worldSize));
            }
        }
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

        if (_ncclAvailable)
        {
            return PerformNcclAllGather(sendData);
        }
        else if (_useTcpFallback)
        {
            return PerformTcpAllGather(sendData);
        }

        return sendData.Clone();
    }

    /// <summary>
    /// Performs AllGather using NCCL with GPU memory.
    /// </summary>
    private Vector<T> PerformNcclAllGather(Vector<T> sendData)
    {
        int sendCount = sendData.Length;
        int recvCount = sendCount * _worldSize;
        int sendByteSize = sendCount * Marshal.SizeOf<T>();
        int recvByteSize = recvCount * Marshal.SizeOf<T>();

        // Allocate receive buffer
        IntPtr gpuRecvBuffer;
        CudaNativeMethods.cudaMalloc(out gpuRecvBuffer, (IntPtr)recvByteSize);

        // Ensure send buffer is allocated
        EnsureGpuBuffers(sendByteSize);

        var sendArray = sendData.ToArray();
        var recvArray = new T[recvCount];

        var sendHandle = GCHandle.Alloc(sendArray, GCHandleType.Pinned);
        var recvHandle = GCHandle.Alloc(recvArray, GCHandleType.Pinned);

        try
        {
            // Copy send data to GPU
            CudaNativeMethods.cudaMemcpyAsync(
                _gpuSendBuffer,
                sendHandle.AddrOfPinnedObject(),
                (IntPtr)sendByteSize,
                CudaMemcpyKind.HostToDevice,
                _cudaStream);

            // Perform NCCL AllGather
            ncclDataType_t ncclType = GetNcclDataType();
            ncclResult_t result = NcclNativeMethods.ncclAllGather(
                _gpuSendBuffer,
                gpuRecvBuffer,
                (IntPtr)sendCount,
                ncclType,
                _ncclComm,
                _cudaStream);

            if (result != ncclResult_t.ncclSuccess)
            {
                throw new InvalidOperationException($"NCCL AllGather failed: {result}");
            }

            // Synchronize
            CudaNativeMethods.cudaStreamSynchronize(_cudaStream);

            // Copy result back to host
            CudaNativeMethods.cudaMemcpyAsync(
                recvHandle.AddrOfPinnedObject(),
                gpuRecvBuffer,
                (IntPtr)recvByteSize,
                CudaMemcpyKind.DeviceToHost,
                _cudaStream);

            CudaNativeMethods.cudaStreamSynchronize(_cudaStream);
        }
        finally
        {
            sendHandle.Free();
            recvHandle.Free();
            CudaNativeMethods.cudaFree(gpuRecvBuffer);
        }

        return new Vector<T>(recvArray);
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

        if (_ncclAvailable)
        {
            return PerformNcclBroadcast(data, root);
        }
        else if (_useTcpFallback)
        {
            return PerformTcpBroadcast(data, root);
        }

        return data.Clone();
    }

    /// <summary>
    /// Performs Broadcast using NCCL with GPU memory.
    /// </summary>
    private Vector<T> PerformNcclBroadcast(Vector<T> data, int root)
    {
        int count = data.Length;
        int byteSize = count * Marshal.SizeOf<T>();

        EnsureGpuBuffers(byteSize);

        var dataArray = data.ToArray();
        var handle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);

        try
        {
            // Copy data to GPU (only root's data matters, but all copy for simplicity)
            CudaNativeMethods.cudaMemcpyAsync(
                _gpuSendBuffer,
                handle.AddrOfPinnedObject(),
                (IntPtr)byteSize,
                CudaMemcpyKind.HostToDevice,
                _cudaStream);

            // Perform NCCL Broadcast (in-place on send buffer)
            ncclDataType_t ncclType = GetNcclDataType();
            ncclResult_t result = NcclNativeMethods.ncclBroadcast(
                _gpuSendBuffer,
                _gpuSendBuffer,
                (IntPtr)count,
                ncclType,
                root,
                _ncclComm,
                _cudaStream);

            if (result != ncclResult_t.ncclSuccess)
            {
                throw new InvalidOperationException($"NCCL Broadcast failed: {result}");
            }

            // Synchronize
            CudaNativeMethods.cudaStreamSynchronize(_cudaStream);

            // Copy result back to host
            CudaNativeMethods.cudaMemcpyAsync(
                handle.AddrOfPinnedObject(),
                _gpuSendBuffer,
                (IntPtr)byteSize,
                CudaMemcpyKind.DeviceToHost,
                _cudaStream);

            CudaNativeMethods.cudaStreamSynchronize(_cudaStream);
        }
        finally
        {
            handle.Free();
        }

        return new Vector<T>(dataArray);
    }

    /// <inheritdoc/>
    public override Vector<T> Scatter(Vector<T> sendData, int root = 0)
    {
        EnsureInitialized();
        ValidateRoot(root);

        // NCCL doesn't have native scatter - implement via Broadcast + indexing
        if (_worldSize == 1)
        {
            if (Rank == root)
            {
                ValidateData(sendData, nameof(sendData));
                return sendData.Clone();
            }
            return new Vector<T>(Array.Empty<T>());
        }

        if (Rank == root)
        {
            ValidateData(sendData, nameof(sendData));
            if (sendData.Length % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"Data length {sendData.Length} must be divisible by world size {_worldSize}.");
            }
        }

        // Use Broadcast + local extraction
        Vector<T> broadcasted;
        if (_ncclAvailable)
        {
            broadcasted = PerformNcclBroadcast(sendData, root);
        }
        else if (_useTcpFallback)
        {
            broadcasted = PerformTcpBroadcast(sendData, root);
        }
        else
        {
            broadcasted = sendData.Clone();
        }

        int chunkSize = broadcasted.Length / _worldSize;
        var chunk = new T[chunkSize];
        var broadcastedArray = broadcasted.ToArray();
        Array.Copy(broadcastedArray, _rank * chunkSize, chunk, 0, chunkSize);

        return new Vector<T>(chunk);
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

        if (_ncclAvailable)
        {
            return PerformNcclReduceScatter(data, operation);
        }
        else if (_useTcpFallback)
        {
            return PerformTcpReduceScatter(data, operation);
        }

        // Fallback
        int chunkSize = data.Length / _worldSize;
        var chunk = new T[chunkSize];
        Array.Copy(data.ToArray(), _rank * chunkSize, chunk, 0, chunkSize);
        return new Vector<T>(chunk);
    }

    /// <summary>
    /// Performs ReduceScatter using NCCL with GPU memory.
    /// </summary>
    private Vector<T> PerformNcclReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        int sendCount = data.Length;
        int recvCount = sendCount / _worldSize;
        int sendByteSize = sendCount * Marshal.SizeOf<T>();
        int recvByteSize = recvCount * Marshal.SizeOf<T>();

        EnsureGpuBuffers(Math.Max(sendByteSize, recvByteSize));

        var sendArray = data.ToArray();
        var recvArray = new T[recvCount];

        var sendHandle = GCHandle.Alloc(sendArray, GCHandleType.Pinned);
        var recvHandle = GCHandle.Alloc(recvArray, GCHandleType.Pinned);

        try
        {
            // Copy send data to GPU
            CudaNativeMethods.cudaMemcpyAsync(
                _gpuSendBuffer,
                sendHandle.AddrOfPinnedObject(),
                (IntPtr)sendByteSize,
                CudaMemcpyKind.HostToDevice,
                _cudaStream);

            // Perform NCCL ReduceScatter
            ncclRedOp_t ncclOp = GetNcclOperation(operation);
            ncclDataType_t ncclType = GetNcclDataType();

            ncclResult_t result = NcclNativeMethods.ncclReduceScatter(
                _gpuSendBuffer,
                _gpuRecvBuffer,
                (IntPtr)recvCount,
                ncclType,
                ncclOp,
                _ncclComm,
                _cudaStream);

            if (result != ncclResult_t.ncclSuccess)
            {
                throw new InvalidOperationException($"NCCL ReduceScatter failed: {result}");
            }

            // Synchronize
            CudaNativeMethods.cudaStreamSynchronize(_cudaStream);

            // Copy result back to host
            CudaNativeMethods.cudaMemcpyAsync(
                recvHandle.AddrOfPinnedObject(),
                _gpuRecvBuffer,
                (IntPtr)recvByteSize,
                CudaMemcpyKind.DeviceToHost,
                _cudaStream);

            CudaNativeMethods.cudaStreamSynchronize(_cudaStream);
        }
        finally
        {
            sendHandle.Free();
            recvHandle.Free();
        }

        // Apply averaging if needed
        if (operation == ReductionOperation.Average)
        {
            for (int i = 0; i < recvCount; i++)
            {
                recvArray[i] = NumOps.Divide(recvArray[i], NumOps.FromDouble(_worldSize));
            }
        }

        return new Vector<T>(recvArray);
    }

    /// <inheritdoc/>
    public override void Send(Vector<T> data, int destinationRank, int tag = 0)
    {
        // NCCL does not support point-to-point operations
        throw new NotSupportedException(
            "NCCL does not support point-to-point Send/Receive operations. " +
            "Use GlooCommunicationBackend or MPICommunicationBackend for point-to-point communication.");
    }

    /// <inheritdoc/>
    public override Vector<T> Receive(int sourceRank, int count, int tag = 0)
    {
        // NCCL does not support point-to-point operations
        throw new NotSupportedException(
            "NCCL does not support point-to-point Send/Receive operations. " +
            "Use GlooCommunicationBackend or MPICommunicationBackend for point-to-point communication.");
    }

    #region GPU Buffer Management

    private void EnsureGpuBuffers(int requiredSize)
    {
        if (_bufferSize >= requiredSize)
        {
            return;
        }

        // Free old buffers
        if (_gpuSendBuffer != IntPtr.Zero)
        {
            CudaNativeMethods.cudaFree(_gpuSendBuffer);
        }
        if (_gpuRecvBuffer != IntPtr.Zero)
        {
            CudaNativeMethods.cudaFree(_gpuRecvBuffer);
        }

        // Allocate new buffers with some extra space
        int newSize = Math.Max(requiredSize, _bufferSize * 2);
        newSize = Math.Max(newSize, 1024 * 1024); // Minimum 1MB

        CudaNativeMethods.cudaMalloc(out _gpuSendBuffer, (IntPtr)newSize);
        CudaNativeMethods.cudaMalloc(out _gpuRecvBuffer, (IntPtr)newSize);
        _bufferSize = newSize;
    }

    #endregion

    #region TCP Fallback Methods

    private void PerformTcpBarrier()
    {
        var signal = new Vector<T>(new[] { NumOps.One });

        for (int otherRank = 0; otherRank < _worldSize; otherRank++)
        {
            if (otherRank != _rank)
            {
                SendDataTcp(otherRank, signal);
            }
        }

        for (int otherRank = 0; otherRank < _worldSize; otherRank++)
        {
            if (otherRank != _rank)
            {
                ReceiveDataTcp(otherRank, 1);
            }
        }
    }

    private void PerformTcpAllReduce(Vector<T> data, ReductionOperation operation)
    {
        // Ring AllReduce implementation
        int chunkSize = (data.Length + _worldSize - 1) / _worldSize;
        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var dataArray = data.ToArray();

        // ReduceScatter phase
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

            int sendStart = sendChunkIdx * chunkSize;
            int sendCount = Math.Min(chunkSize, data.Length - sendStart);
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Min(chunkSize, data.Length - recvStart);

            var sendChunk = new T[sendCount];
            Array.Copy(dataArray, sendStart, sendChunk, 0, sendCount);

            var sendTask = Task.Run(() => SendDataTcp(nextRank, new Vector<T>(sendChunk)));
            var recvChunk = ReceiveDataTcp(prevRank, recvCount);
            sendTask.Wait();

            for (int i = 0; i < recvCount; i++)
            {
                dataArray[recvStart + i] = PerformReduction(dataArray[recvStart + i], recvChunk[i], operation);
            }
        }

        // AllGather phase
        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + 1 + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step + _worldSize) % _worldSize;

            int sendStart = sendChunkIdx * chunkSize;
            int sendCount = Math.Min(chunkSize, data.Length - sendStart);
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Min(chunkSize, data.Length - recvStart);

            var sendChunk = new T[sendCount];
            Array.Copy(dataArray, sendStart, sendChunk, 0, sendCount);

            var sendTask = Task.Run(() => SendDataTcp(nextRank, new Vector<T>(sendChunk)));
            var recvChunk = ReceiveDataTcp(prevRank, recvCount);
            sendTask.Wait();

            Array.Copy(recvChunk, 0, dataArray, recvStart, recvCount);
        }

        if (operation == ReductionOperation.Average)
        {
            for (int i = 0; i < dataArray.Length; i++)
            {
                dataArray[i] = NumOps.Divide(dataArray[i], NumOps.FromDouble(_worldSize));
            }
        }

        for (int i = 0; i < dataArray.Length; i++)
        {
            data[i] = dataArray[i];
        }
    }

    private Vector<T> PerformTcpAllGather(Vector<T> sendData)
    {
        int chunkSize = sendData.Length;
        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var result = new T[chunkSize * _worldSize];
        Array.Copy(sendData.ToArray(), 0, result, _rank * chunkSize, chunkSize);

        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

            var sendChunk = new T[chunkSize];
            Array.Copy(result, sendChunkIdx * chunkSize, sendChunk, 0, chunkSize);

            var sendTask = Task.Run(() => SendDataTcp(nextRank, new Vector<T>(sendChunk)));
            var recvChunk = ReceiveDataTcp(prevRank, chunkSize);
            sendTask.Wait();

            Array.Copy(recvChunk, 0, result, recvChunkIdx * chunkSize, chunkSize);
        }

        return new Vector<T>(result);
    }

    private Vector<T> PerformTcpBroadcast(Vector<T> data, int root)
    {
        var dataArray = data.ToArray();
        int relativeRank = (_rank - root + _worldSize) % _worldSize;

        if (relativeRank != 0)
        {
            int parentRelative = (relativeRank - 1) / 2;
            int parentAbsolute = (parentRelative + root) % _worldSize;
            dataArray = ReceiveDataTcp(parentAbsolute, data.Length);
        }

        int leftChildRelative = 2 * relativeRank + 1;
        int rightChildRelative = 2 * relativeRank + 2;

        if (leftChildRelative < _worldSize)
        {
            int leftChildAbsolute = (leftChildRelative + root) % _worldSize;
            SendDataTcp(leftChildAbsolute, new Vector<T>(dataArray));
        }

        if (rightChildRelative < _worldSize)
        {
            int rightChildAbsolute = (rightChildRelative + root) % _worldSize;
            SendDataTcp(rightChildAbsolute, new Vector<T>(dataArray));
        }

        return new Vector<T>(dataArray);
    }

    private Vector<T> PerformTcpReduceScatter(Vector<T> data, ReductionOperation operation)
    {
        int chunkSize = data.Length / _worldSize;
        int nextRank = (_rank + 1) % _worldSize;
        int prevRank = (_rank - 1 + _worldSize) % _worldSize;

        var dataArray = data.ToArray();

        for (int step = 0; step < _worldSize - 1; step++)
        {
            int sendChunkIdx = (_rank - step + _worldSize) % _worldSize;
            int recvChunkIdx = (_rank - step - 1 + _worldSize) % _worldSize;

            int sendStart = sendChunkIdx * chunkSize;
            int sendCount = Math.Min(chunkSize, data.Length - sendStart);
            int recvStart = recvChunkIdx * chunkSize;
            int recvCount = Math.Min(chunkSize, data.Length - recvStart);

            var sendChunk = new T[sendCount];
            Array.Copy(dataArray, sendStart, sendChunk, 0, sendCount);

            var sendTask = Task.Run(() => SendDataTcp(nextRank, new Vector<T>(sendChunk)));
            var recvChunk = ReceiveDataTcp(prevRank, recvCount);
            sendTask.Wait();

            for (int i = 0; i < recvCount; i++)
            {
                dataArray[recvStart + i] = PerformReduction(dataArray[recvStart + i], recvChunk[i], operation);
            }
        }

        var myChunk = new T[chunkSize];
        Array.Copy(dataArray, _rank * chunkSize, myChunk, 0, chunkSize);

        if (operation == ReductionOperation.Average)
        {
            for (int i = 0; i < myChunk.Length; i++)
            {
                myChunk[i] = NumOps.Divide(myChunk[i], NumOps.FromDouble(_worldSize));
            }
        }

        return new Vector<T>(myChunk);
    }

    private T PerformReduction(T a, T b, ReductionOperation operation)
    {
        return operation switch
        {
            ReductionOperation.Sum or ReductionOperation.Average => NumOps.Add(a, b),
            ReductionOperation.Max => NumOps.GreaterThan(a, b) ? a : b,
            ReductionOperation.Min => NumOps.LessThan(a, b) ? a : b,
            ReductionOperation.Product => NumOps.Multiply(a, b),
            _ => throw new ArgumentException($"Unsupported operation: {operation}")
        };
    }

    private void SendDataTcp(int destRank, Vector<T> data)
    {
        if (_tcpConnections == null || !_tcpConnections.ContainsKey(destRank))
        {
            throw new InvalidOperationException($"No TCP connection to rank {destRank}");
        }

        lock (_connectionLock)
        {
            var client = _tcpConnections[destRank];
            var stream = client.GetStream();
            var writer = new BinaryWriter(stream);

            writer.Write(data.Length);
            for (int i = 0; i < data.Length; i++)
            {
                writer.Write(Convert.ToDouble(data[i]));
            }
            writer.Flush();
        }
    }

    private T[] ReceiveDataTcp(int sourceRank, int expectedLength)
    {
        if (_tcpConnections == null || !_tcpConnections.ContainsKey(sourceRank))
        {
            throw new InvalidOperationException($"No TCP connection to rank {sourceRank}");
        }

        lock (_connectionLock)
        {
            var client = _tcpConnections[sourceRank];
            var stream = client.GetStream();
            var reader = new BinaryReader(stream);

            int length = reader.ReadInt32();
            if (length != expectedLength)
            {
                throw new InvalidOperationException(
                    $"Expected {expectedLength} elements but received {length}");
            }

            var result = new T[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            return result;
        }
    }

    #endregion

    #region NCCL Helpers

    private ncclDataType_t GetNcclDataType()
    {
        var typeCode = Type.GetTypeCode(typeof(T));
        return typeCode switch
        {
            TypeCode.Single => ncclDataType_t.ncclFloat32,
            TypeCode.Double => ncclDataType_t.ncclFloat64,
            TypeCode.Int32 => ncclDataType_t.ncclInt32,
            TypeCode.Int64 => ncclDataType_t.ncclInt64,
            TypeCode.Byte => ncclDataType_t.ncclUint8,
            TypeCode.SByte => ncclDataType_t.ncclInt8,
            TypeCode.UInt32 => ncclDataType_t.ncclUint32,
            TypeCode.UInt64 => ncclDataType_t.ncclUint64,
            _ => throw new NotSupportedException($"Type {typeof(T)} is not supported by NCCL.")
        };
    }

    private ncclRedOp_t GetNcclOperation(ReductionOperation operation)
    {
        return operation switch
        {
            ReductionOperation.Sum => ncclRedOp_t.ncclSum,
            ReductionOperation.Product => ncclRedOp_t.ncclProd,
            ReductionOperation.Min => ncclRedOp_t.ncclMin,
            ReductionOperation.Max => ncclRedOp_t.ncclMax,
            ReductionOperation.Average => ncclRedOp_t.ncclSum, // We apply division after
            _ => throw new NotSupportedException($"Operation {operation} is not supported.")
        };
    }

    #endregion
}

/// <summary>
/// NCCL unique ID structure for communicator initialization.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NcclUniqueId
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public byte[] Internal;

    public NcclUniqueId()
    {
        Internal = new byte[128];
    }

    public byte[] ToBytes()
    {
        return Internal ?? new byte[128];
    }

    public static NcclUniqueId FromBytes(byte[] bytes)
    {
        var id = new NcclUniqueId();
        if (bytes.Length >= 128)
        {
            Array.Copy(bytes, id.Internal, 128);
        }
        else
        {
            Array.Copy(bytes, id.Internal, bytes.Length);
        }
        return id;
    }
}

/// <summary>
/// NCCL P/Invoke methods.
/// </summary>
internal static class NcclNativeMethods
{
    private const string NcclLibrary = "nccl";

    [DllImport(NcclLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t ncclGetVersion(out int version);

    [DllImport(NcclLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t ncclGetUniqueId(ref NcclUniqueId uniqueId);

    [DllImport(NcclLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t ncclCommInitRank(out IntPtr comm, int nranks, NcclUniqueId commId, int rank);

    [DllImport(NcclLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t ncclCommDestroy(IntPtr comm);

    [DllImport(NcclLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t ncclAllReduce(
        IntPtr sendbuff, IntPtr recvbuff, IntPtr count,
        ncclDataType_t datatype, ncclRedOp_t op,
        IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t ncclAllGather(
        IntPtr sendbuff, IntPtr recvbuff, IntPtr sendcount,
        ncclDataType_t datatype, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t ncclBroadcast(
        IntPtr sendbuff, IntPtr recvbuff, IntPtr count,
        ncclDataType_t datatype, int root, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t ncclReduceScatter(
        IntPtr sendbuff, IntPtr recvbuff, IntPtr recvcount,
        ncclDataType_t datatype, ncclRedOp_t op,
        IntPtr comm, IntPtr stream);
}

/// <summary>
/// CUDA memory copy direction.
/// </summary>
internal enum CudaMemcpyKind
{
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3
}

/// <summary>
/// CUDA P/Invoke methods for memory management and stream operations.
/// </summary>
internal static class CudaNativeMethods
{
    private const string CudaLibrary = "cudart64_12"; // CUDA 12.x runtime

    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t cudaSetDevice(int device);

    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t cudaMalloc(out IntPtr devPtr, IntPtr size);

    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t cudaFree(IntPtr devPtr);

    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t cudaStreamCreate(out IntPtr stream);

    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t cudaStreamDestroy(IntPtr stream);

    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t cudaStreamSynchronize(IntPtr stream);

    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ncclResult_t cudaMemcpyAsync(
        IntPtr dst, IntPtr src, IntPtr count,
        CudaMemcpyKind kind, IntPtr stream);
}
