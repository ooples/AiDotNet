using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a component that can handle streaming data efficiently.
/// </summary>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public interface IStreamingDataHandler<TInput, TOutput>
{
    /// <summary>
    /// Processes a stream of data points asynchronously.
    /// </summary>
    /// <param name="dataStream">The enumerable stream of data.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    Task ProcessStreamAsync(IEnumerable<(TInput input, TOutput output)> dataStream, 
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets or sets the buffer size for batching streaming data.
    /// </summary>
    int StreamBufferSize { get; set; }
    
    /// <summary>
    /// Gets or sets the maximum time to wait before processing a partial batch.
    /// </summary>
    TimeSpan StreamBufferTimeout { get; set; }
}