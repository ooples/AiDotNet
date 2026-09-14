namespace AiDotNet.ProgramSynthesis.Interfaces;

/// <summary>Optional, nonblocking and thread-safe runner-instance counters; individual reads are not a joint atomic snapshot.</summary>
public interface IProgramExecutionTelemetrySource
{
    /// <summary>Gets requests waiting for execution capacity on this instance.</summary>
    int QueuedExecutionCount { get; }
    /// <summary>Gets requests currently holding execution capacity on this instance.</summary>
    int ActiveExecutionCount { get; }
}
