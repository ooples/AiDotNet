using AiDotNet.Tensors.Engines.DirectGpu;

// Use the existing FusedActivationType from AiDotNet.Tensors.Engines
using FusedActivationType = AiDotNet.Tensors.Engines.FusedActivationType;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Extended GPU backend interface with async operations, streams, and events.
/// Backends that support async execution implement this interface for improved performance.
/// </summary>
/// <remarks>
/// <para><b>Design Philosophy:</b></para>
/// <para>
/// This interface extends <see cref="IDirectGpuBackend"/> with capabilities for:
/// - Multi-stream execution (compute/transfer overlap)
/// - GPU events for inter-stream synchronization
/// - Async memory transfers (non-blocking H2D/D2H)
/// - Sync points for deferred synchronization
/// </para>
/// <para><b>Performance Benefits:</b></para>
/// <list type="bullet">
/// <item>Overlap compute and memory transfer operations</item>
/// <item>Reduce CPU blocking with async APIs</item>
/// <item>Enable execution graph optimization</item>
/// <item>Support GPU-resident tensors with lazy sync</item>
/// </list>
/// </remarks>
public interface IAsyncGpuBackend : IDirectGpuBackend
{
    #region Capability Detection

    /// <summary>
    /// Gets whether this backend supports multiple concurrent streams.
    /// </summary>
    bool SupportsMultiStream { get; }

    /// <summary>
    /// Gets whether this backend supports GPU events for synchronization.
    /// </summary>
    bool SupportsEvents { get; }

    /// <summary>
    /// Gets whether this backend supports async memory transfers.
    /// </summary>
    bool SupportsAsyncTransfer { get; }

    /// <summary>
    /// Gets whether this backend supports execution graph capture.
    /// </summary>
    bool SupportsGraphCapture { get; }

    /// <summary>
    /// Gets the maximum number of concurrent streams supported.
    /// </summary>
    int MaxConcurrentStreams { get; }

    #endregion

    #region Stream Management

    /// <summary>
    /// Creates a new GPU stream with the specified type.
    /// </summary>
    /// <param name="streamType">The type of stream to create.</param>
    /// <returns>A new GPU stream.</returns>
    IGpuStream CreateStream(GpuStreamType streamType);

    /// <summary>
    /// Creates a new GPU stream with priority.
    /// </summary>
    /// <param name="streamType">The type of stream to create.</param>
    /// <param name="priority">Priority level (lower = higher priority).</param>
    /// <returns>A new GPU stream with the specified priority.</returns>
    IGpuStream CreateStream(GpuStreamType streamType, int priority);

    /// <summary>
    /// Gets the default stream for this backend.
    /// Operations on the default stream synchronize with all other streams.
    /// </summary>
    IGpuStream DefaultStream { get; }

    #endregion

    #region Event Management

    /// <summary>
    /// Creates a new GPU event for synchronization.
    /// </summary>
    /// <returns>A new GPU event.</returns>
    IGpuEvent CreateEvent();

    /// <summary>
    /// Creates an event with timing capability.
    /// </summary>
    /// <param name="enableTiming">If true, the event can be used for timing measurements.</param>
    /// <returns>A new GPU event.</returns>
    IGpuEvent CreateEvent(bool enableTiming);

    /// <summary>
    /// Records an event on the specified stream.
    /// </summary>
    /// <param name="gpuEvent">The event to record.</param>
    /// <param name="stream">The stream to record on.</param>
    void RecordEvent(IGpuEvent gpuEvent, IGpuStream stream);

    /// <summary>
    /// Makes a stream wait for an event.
    /// </summary>
    /// <param name="stream">The stream that should wait.</param>
    /// <param name="gpuEvent">The event to wait for.</param>
    void StreamWaitEvent(IGpuStream stream, IGpuEvent gpuEvent);

    #endregion

    #region Sync Point Management

    /// <summary>
    /// Creates a sync point at the current position in a stream.
    /// </summary>
    /// <param name="stream">The stream to create a sync point on.</param>
    /// <returns>A sync point that completes when all prior operations on the stream complete.</returns>
    GpuSyncPoint CreateSyncPoint(IGpuStream stream);

    /// <summary>
    /// Creates a sync point on the default stream.
    /// </summary>
    /// <returns>A sync point for the default stream.</returns>
    GpuSyncPoint CreateSyncPoint();

    #endregion

    #region Async Memory Operations

    /// <summary>
    /// Asynchronously uploads data from CPU to GPU.
    /// </summary>
    /// <param name="data">Source CPU data.</param>
    /// <param name="buffer">Destination GPU buffer.</param>
    /// <param name="stream">Stream to execute the transfer on.</param>
    void UploadBufferAsync(float[] data, IGpuBuffer buffer, IGpuStream stream);

    /// <summary>
    /// Asynchronously uploads data from a span to GPU.
    /// </summary>
    /// <param name="data">Source CPU data span.</param>
    /// <param name="buffer">Destination GPU buffer.</param>
    /// <param name="stream">Stream to execute the transfer on.</param>
    void UploadBufferAsync(ReadOnlySpan<float> data, IGpuBuffer buffer, IGpuStream stream);

    /// <summary>
    /// Asynchronously downloads data from GPU to CPU.
    /// </summary>
    /// <param name="buffer">Source GPU buffer.</param>
    /// <param name="destination">Destination CPU array.</param>
    /// <param name="stream">Stream to execute the transfer on.</param>
    void DownloadBufferAsync(IGpuBuffer buffer, float[] destination, IGpuStream stream);

    /// <summary>
    /// Asynchronously allocates a buffer and uploads data.
    /// </summary>
    /// <param name="data">Source CPU data.</param>
    /// <param name="stream">Stream to execute the transfer on.</param>
    /// <returns>A new GPU buffer with the data.</returns>
    IGpuBuffer AllocateBufferAsync(float[] data, IGpuStream stream);

    /// <summary>
    /// Asynchronously copies data between GPU buffers.
    /// </summary>
    /// <param name="source">Source GPU buffer.</param>
    /// <param name="destination">Destination GPU buffer.</param>
    /// <param name="size">Number of elements to copy.</param>
    /// <param name="stream">Stream to execute the copy on.</param>
    void CopyBufferAsync(IGpuBuffer source, IGpuBuffer destination, int size, IGpuStream stream);

    #endregion

    #region Stream-Aware Operations

    /// <summary>
    /// Executes GEMM on a specific stream.
    /// </summary>
    void GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IGpuStream stream);

    /// <summary>
    /// Executes fused GEMM+Bias+Activation on a specific stream.
    /// </summary>
    /// <param name="A">Input matrix.</param>
    /// <param name="B">Weight matrix.</param>
    /// <param name="bias">Bias vector.</param>
    /// <param name="output">Output buffer.</param>
    /// <param name="M">Batch size.</param>
    /// <param name="N">Output features.</param>
    /// <param name="K">Input features.</param>
    /// <param name="activation">Activation type to apply.</param>
    /// <param name="stream">Stream to execute on.</param>
    void FusedGemmBiasActivationAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output,
        int M, int N, int K, FusedActivationType activation, IGpuStream stream);

    /// <summary>
    /// Synchronizes a specific stream.
    /// </summary>
    /// <param name="stream">The stream to synchronize.</param>
    void SynchronizeStream(IGpuStream stream);

    #endregion

    #region Query Methods

    /// <summary>
    /// Queries whether all operations on a stream have completed without blocking.
    /// </summary>
    /// <param name="stream">The stream to query.</param>
    /// <returns>True if all operations are complete.</returns>
    bool QueryStreamComplete(IGpuStream stream);

    /// <summary>
    /// Queries whether an event has been reached without blocking.
    /// </summary>
    /// <param name="gpuEvent">The event to query.</param>
    /// <returns>True if the event has been reached.</returns>
    bool QueryEventComplete(IGpuEvent gpuEvent);

    /// <summary>
    /// Gets the elapsed time between two events in milliseconds.
    /// </summary>
    /// <param name="start">Start event.</param>
    /// <param name="end">End event.</param>
    /// <returns>Elapsed time in milliseconds.</returns>
    float GetEventElapsedTime(IGpuEvent start, IGpuEvent end);

    #endregion
}
