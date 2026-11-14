using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Represents an in-flight asynchronous collective communication operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When training across multiple GPUs, waiting for gradients to finish copying
/// across devices before computing the remaining layers wastes time. An <see cref="IAsyncCollectiveHandle"/>
/// lets a collective operation (such as <c>AllReduce</c> on a finished gradient bucket) run in the background
/// while backward propagation continues on earlier layers, and then lets the trainer wait for completion right
/// before gradient unscaling and global-norm clipping.</para>
/// </remarks>
public interface IAsyncCollectiveHandle : IDisposable
{
    /// <summary>
    /// Gets whether the collective operation has completed.
    /// </summary>
    bool IsCompleted { get; }

    /// <summary>
    /// Gets the wall-clock duration spent executing the collective operation on the transport.
    /// </summary>
    TimeSpan Elapsed { get; }

    /// <summary>
    /// Blocks until the collective operation completes, rethrowing any transport error.
    /// </summary>
    void Wait();

    /// <summary>
    /// Asynchronously waits until the collective operation completes, rethrowing any transport error.
    /// </summary>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>A task that completes when the collective finishes.</returns>
    Task WaitAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents an in-flight asynchronous collective communication operation that produces a new vector result
/// (such as <c>ReduceScatter</c> or <c>AllGather</c>).
/// </summary>
/// <typeparam name="T">The numeric element type.</typeparam>
public interface IAsyncCollectiveResultHandle<T> : IAsyncCollectiveHandle
{
    /// <summary>
    /// Blocks until the collective operation completes and returns its output vector.
    /// </summary>
    /// <returns>The collective result vector.</returns>
    Vector<T> GetResult();
}

/// <summary>
/// Extends <see cref="ICommunicationBackend{T}"/> with non-blocking collective operations and raw transport
/// timing diagnostics so gradient bucketing can overlap communication with backward computation.
/// </summary>
/// <typeparam name="T">The numeric element type.</typeparam>
public interface IAsyncCommunicationBackend<T> : ICommunicationBackend<T>
{
    /// <summary>
    /// Gets the cumulative wall-clock milliseconds spent inside transport collective operations since initialization
    /// or the last <see cref="ResetDiagnostics"/> call.
    /// </summary>
    double TotalCollectiveMilliseconds { get; }

    /// <summary>
    /// Gets the total number of collective operations executed since initialization or the last
    /// <see cref="ResetDiagnostics"/> call.
    /// </summary>
    long CollectiveCount { get; }

    /// <summary>
    /// Resets <see cref="TotalCollectiveMilliseconds"/> and <see cref="CollectiveCount"/> to zero.
    /// </summary>
    void ResetDiagnostics();

    /// <summary>
    /// Enqueues an asynchronous in-place <see cref="ICommunicationBackend{T}.AllReduce"/> on <paramref name="data"/>
    /// in strict FIFO collective order.
    /// </summary>
    /// <param name="data">The vector to reduce in place.</param>
    /// <param name="operation">The reduction operation to apply across ranks.</param>
    /// <returns>A handle that completes once <paramref name="data"/> holds the reduced result.</returns>
    IAsyncCollectiveHandle AllReduceAsync(Vector<T> data, ReductionOperation operation);

    /// <summary>
    /// Enqueues an asynchronous <see cref="ICommunicationBackend{T}.ReduceScatter"/> on <paramref name="data"/>
    /// in strict FIFO collective order.
    /// </summary>
    /// <param name="data">The full vector (length divisible by <see cref="ICommunicationBackend{T}.WorldSize"/>) to reduce and scatter.</param>
    /// <param name="operation">The reduction operation to apply across ranks.</param>
    /// <returns>A result handle yielding this rank's reduced shard.</returns>
    IAsyncCollectiveResultHandle<T> ReduceScatterAsync(Vector<T> data, ReductionOperation operation);

    /// <summary>
    /// Enqueues an asynchronous <see cref="ICommunicationBackend{T}.AllGather"/> on <paramref name="sendData"/>
    /// in strict FIFO collective order.
    /// </summary>
    /// <param name="sendData">This rank's local shard to gather across all ranks.</param>
    /// <returns>A result handle yielding the concatenated vector from all ranks.</returns>
    IAsyncCollectiveResultHandle<T> AllGatherAsync(Vector<T> sendData);
}
