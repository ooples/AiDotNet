namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Represents a GPU stream (CUDA stream, OpenCL command queue, HIP stream).
/// Streams enable concurrent execution and compute/transfer overlap.
/// </summary>
public interface IGpuStream : IDisposable
{
    /// <summary>
    /// Gets the native handle of the stream.
    /// For CUDA: cudaStream_t, for OpenCL: cl_command_queue, for HIP: hipStream_t
    /// </summary>
    IntPtr Handle { get; }

    /// <summary>
    /// Gets the type of this stream.
    /// </summary>
    GpuStreamType StreamType { get; }

    /// <summary>
    /// Gets whether this is the default (null) stream.
    /// The default stream synchronizes with all other streams.
    /// </summary>
    bool IsDefault { get; }

    /// <summary>
    /// Gets the priority of this stream (lower = higher priority).
    /// Not all backends support stream priorities.
    /// </summary>
    int Priority { get; }

    /// <summary>
    /// Synchronizes the stream, blocking until all operations are complete.
    /// </summary>
    void Synchronize();

    /// <summary>
    /// Records an event on this stream.
    /// The event will be signaled when all prior operations on this stream complete.
    /// </summary>
    /// <returns>A new event recorded at the current point in the stream.</returns>
    IGpuEvent RecordEvent();

    /// <summary>
    /// Makes this stream wait for an event recorded on another stream.
    /// Operations enqueued after this call will not start until the event completes.
    /// </summary>
    /// <param name="gpuEvent">The event to wait for.</param>
    void WaitEvent(IGpuEvent gpuEvent);

    /// <summary>
    /// Queries whether all operations on this stream have completed.
    /// </summary>
    /// <returns>True if all operations are complete, false if operations are pending.</returns>
    bool Query();
}
