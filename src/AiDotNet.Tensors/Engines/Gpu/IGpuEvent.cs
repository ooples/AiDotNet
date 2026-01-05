namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Represents a GPU event for synchronization between streams.
/// Events are recorded on one stream and can be waited on by other streams.
/// </summary>
public interface IGpuEvent : IDisposable
{
    /// <summary>
    /// Gets the native handle of the event.
    /// For CUDA: cudaEvent_t, for OpenCL: cl_event, for HIP: hipEvent_t
    /// </summary>
    IntPtr Handle { get; }

    /// <summary>
    /// Gets whether the event has been recorded (signaled).
    /// </summary>
    bool IsRecorded { get; }

    /// <summary>
    /// Gets whether all operations before this event have completed.
    /// </summary>
    bool IsComplete { get; }

    /// <summary>
    /// Blocks the calling CPU thread until the event completes.
    /// </summary>
    void Synchronize();

    /// <summary>
    /// Queries whether the event has completed without blocking.
    /// </summary>
    /// <returns>True if the event has completed, false if still pending.</returns>
    bool Query();

    /// <summary>
    /// Gets the elapsed time in milliseconds between this event and a prior event.
    /// Both events must have been recorded and completed.
    /// </summary>
    /// <param name="startEvent">The event marking the start of the time interval.</param>
    /// <returns>Elapsed time in milliseconds.</returns>
    float GetElapsedTime(IGpuEvent startEvent);
}
