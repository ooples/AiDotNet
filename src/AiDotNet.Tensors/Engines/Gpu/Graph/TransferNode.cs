using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Represents a memory transfer node in the execution graph.
/// Used for host-to-device, device-to-host, and device-to-device transfers.
/// </summary>
public sealed class TransferNode : ExecutionNode
{
    private readonly IGpuBuffer? _sourceBuffer;
    private readonly IGpuBuffer? _destinationBuffer;
    private readonly float[]? _sourceData;
    private float[]? _destinationData;
    private readonly int _size;

    /// <inheritdoc/>
    public override ExecutionNodeType NodeType => TransferType switch
    {
        TransferDirection.HostToDevice => ExecutionNodeType.TransferH2D,
        TransferDirection.DeviceToHost => ExecutionNodeType.TransferD2H,
        TransferDirection.DeviceToDevice => ExecutionNodeType.Copy,
        _ => ExecutionNodeType.TransferH2D
    };

    /// <summary>
    /// Gets the transfer direction.
    /// </summary>
    public TransferDirection TransferType { get; }

    /// <summary>
    /// Gets the size of the transfer in elements.
    /// </summary>
    public int Size => _size;

    /// <summary>
    /// Gets the destination buffer for H2D transfers.
    /// </summary>
    public IGpuBuffer? DestinationBuffer => _destinationBuffer;

    /// <summary>
    /// Gets the source buffer for D2H transfers.
    /// </summary>
    public IGpuBuffer? SourceBuffer => _sourceBuffer;

    /// <summary>
    /// Gets the downloaded data after D2H transfer.
    /// </summary>
    public float[]? DownloadedData => _destinationData;

    /// <inheritdoc/>
    public override int EstimatedCost => TransferType switch
    {
        TransferDirection.HostToDevice => (_size / 1024) + 5,  // Base cost + per-KB cost
        TransferDirection.DeviceToHost => (_size / 1024) + 10, // D2H is typically slower
        TransferDirection.DeviceToDevice => (_size / 2048) + 2, // D2D is fastest
        _ => 10
    };

    /// <inheritdoc/>
    public override string Name => $"{TransferType}_{_size}_{NodeId}";

    /// <summary>
    /// Creates a host-to-device transfer node.
    /// </summary>
    /// <param name="sourceData">Source data on host.</param>
    /// <param name="destinationBuffer">Destination buffer on device.</param>
    public static TransferNode CreateH2D(float[] sourceData, IGpuBuffer destinationBuffer)
    {
        return new TransferNode(TransferDirection.HostToDevice, sourceData, destinationBuffer);
    }

    /// <summary>
    /// Creates a device-to-host transfer node.
    /// </summary>
    /// <param name="sourceBuffer">Source buffer on device.</param>
    /// <param name="size">Number of elements to transfer.</param>
    public static TransferNode CreateD2H(IGpuBuffer sourceBuffer, int size)
    {
        return new TransferNode(TransferDirection.DeviceToHost, sourceBuffer, size);
    }

    /// <summary>
    /// Creates a device-to-device copy node.
    /// </summary>
    /// <param name="sourceBuffer">Source buffer.</param>
    /// <param name="destinationBuffer">Destination buffer.</param>
    /// <param name="size">Number of elements to copy.</param>
    public static TransferNode CreateD2D(IGpuBuffer sourceBuffer, IGpuBuffer destinationBuffer, int size)
    {
        return new TransferNode(TransferDirection.DeviceToDevice, sourceBuffer, destinationBuffer, size);
    }

    private TransferNode(TransferDirection direction, float[] sourceData, IGpuBuffer destinationBuffer)
    {
        TransferType = direction;
        _sourceData = sourceData ?? throw new ArgumentNullException(nameof(sourceData));
        _destinationBuffer = destinationBuffer ?? throw new ArgumentNullException(nameof(destinationBuffer));
        _size = sourceData.Length;
    }

    private TransferNode(TransferDirection direction, IGpuBuffer sourceBuffer, int size)
    {
        TransferType = direction;
        _sourceBuffer = sourceBuffer ?? throw new ArgumentNullException(nameof(sourceBuffer));
        _size = size;
        _destinationData = new float[size];
    }

    private TransferNode(TransferDirection direction, IGpuBuffer sourceBuffer, IGpuBuffer destinationBuffer, int size)
    {
        TransferType = direction;
        _sourceBuffer = sourceBuffer ?? throw new ArgumentNullException(nameof(sourceBuffer));
        _destinationBuffer = destinationBuffer ?? throw new ArgumentNullException(nameof(destinationBuffer));
        _size = size;
    }

    /// <inheritdoc/>
    public override void Execute(IDirectGpuBackend backend)
    {
        switch (TransferType)
        {
            case TransferDirection.HostToDevice:
                if (_sourceData != null && _destinationBuffer != null)
                {
                    // Use a temporary buffer to upload and copy
                    using var tempBuffer = backend.AllocateBuffer(_sourceData);
                    backend.Copy(tempBuffer, _destinationBuffer, _size);
                }
                break;

            case TransferDirection.DeviceToHost:
                if (_sourceBuffer != null)
                {
                    _destinationData = backend.DownloadBuffer(_sourceBuffer);
                }
                break;

            case TransferDirection.DeviceToDevice:
                if (_sourceBuffer != null && _destinationBuffer != null)
                {
                    backend.Copy(_sourceBuffer, _destinationBuffer, _size);
                }
                break;
        }
    }

    /// <inheritdoc/>
    public override void ExecuteAsync(IDirectGpuBackend backend, IGpuStream stream)
    {
        // Wait for dependencies on other streams
        foreach (var dep in Dependencies)
        {
            if (dep.CompletionSync != null && dep.AssignedStream != stream)
            {
                dep.CompletionSync.MakeStreamWait(stream);
            }
        }

        AssignedStream = stream;

        // Check if backend supports async operations
        if (backend is IAsyncGpuBackend asyncBackend)
        {
            ExecuteAsyncInternal(asyncBackend, stream);
        }
        else
        {
            // Fallback to synchronous execution
            Execute(backend);
        }

        // Record completion for dependents on other streams
        if (Dependents.Any(d => d.AssignedStream != stream))
        {
            var evt = stream.RecordEvent();
            CompletionSync = new TransferSyncPoint(evt, stream);
        }
    }

    private void ExecuteAsyncInternal(IAsyncGpuBackend backend, IGpuStream stream)
    {
        switch (TransferType)
        {
            case TransferDirection.HostToDevice:
                if (_sourceData != null && _destinationBuffer != null)
                {
                    backend.UploadBufferAsync(_sourceData, _destinationBuffer, stream);
                }
                break;

            case TransferDirection.DeviceToHost:
                if (_sourceBuffer != null && _destinationData != null)
                {
                    backend.DownloadBufferAsync(_sourceBuffer, _destinationData, stream);
                }
                break;

            case TransferDirection.DeviceToDevice:
                if (_sourceBuffer != null && _destinationBuffer != null)
                {
                    backend.CopyBufferAsync(_sourceBuffer, _destinationBuffer, _size, stream);
                }
                break;
        }
    }

    /// <inheritdoc/>
    public override ExecutionNode Clone()
    {
        return TransferType switch
        {
            TransferDirection.HostToDevice when _sourceData != null && _destinationBuffer != null
                => CreateH2D(_sourceData, _destinationBuffer),
            TransferDirection.DeviceToHost when _sourceBuffer != null
                => CreateD2H(_sourceBuffer, _size),
            TransferDirection.DeviceToDevice when _sourceBuffer != null && _destinationBuffer != null
                => CreateD2D(_sourceBuffer, _destinationBuffer, _size),
            _ => throw new InvalidOperationException("Cannot clone transfer node in invalid state.")
        };
    }

    private sealed class TransferSyncPoint : GpuSyncPoint
    {
        private readonly IGpuEvent _event;
        private readonly IGpuStream _stream;

        public TransferSyncPoint(IGpuEvent evt, IGpuStream stream)
        {
            _event = evt;
            _stream = stream;
        }

        public override bool IsComplete => _event.IsComplete;
        public override IGpuStream? Stream => _stream;
        public override IGpuEvent? Event => _event;

        public override void Wait() => _event.Synchronize();
        public override bool Poll() => _event.Query();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _event.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}

/// <summary>
/// Direction of memory transfer.
/// </summary>
public enum TransferDirection
{
    /// <summary>Host to device transfer.</summary>
    HostToDevice,
    /// <summary>Device to host transfer.</summary>
    DeviceToHost,
    /// <summary>Device to device transfer.</summary>
    DeviceToDevice
}
