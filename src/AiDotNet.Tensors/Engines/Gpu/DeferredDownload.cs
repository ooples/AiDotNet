// Copyright (c) AiDotNet. All rights reserved.
// Deferred download handle for GPU execution graphs.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu.Graph;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Represents a deferred GPU-to-CPU download that will be executed as part of an execution graph.
/// </summary>
/// <remarks>
/// <para>
/// When recording GPU operations, downloads can be deferred until graph execution to avoid
/// blocking the GPU pipeline. This class provides a handle to retrieve the data after execution.
/// </para>
/// <para><b>Usage:</b></para>
/// <code>
/// using var scope = context.BeginDeferred();
/// 
/// // Record operations
/// var output = scope.Execute(() => gpuEngine.FusedLinearGpu(input, weights, bias, activation));
/// 
/// // Request deferred download - doesn't block
/// var download = scope.DownloadDeferred(output.Buffer, output.ElementCount);
/// 
/// // Execute the graph
/// await scope.ExecuteAsync();
/// 
/// // Now data is available
/// float[] data = download.GetResult();
/// </code>
/// </remarks>
public sealed class DeferredDownload
{
    private readonly TransferNode _transferNode;
    private bool _isExecuted;

    /// <summary>
    /// Gets the size of the download in elements.
    /// </summary>
    public int Size => _transferNode.Size;

    /// <summary>
    /// Gets whether the download has been executed.
    /// </summary>
    public bool IsCompleted => _isExecuted;

    /// <summary>
    /// Gets the source buffer being downloaded.
    /// </summary>
    public IGpuBuffer? SourceBuffer => _transferNode.SourceBuffer;

    internal DeferredDownload(TransferNode transferNode)
    {
        _transferNode = transferNode ?? throw new ArgumentNullException(nameof(transferNode));
    }

    /// <summary>
    /// Gets the downloaded data after execution.
    /// </summary>
    /// <returns>The downloaded float array.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the download has not been executed yet.</exception>
    public float[] GetResult()
    {
        if (!_isExecuted)
        {
            throw new InvalidOperationException(
                "Deferred download has not been executed yet. Call Execute() or ExecuteAsync() on the DeferredScope first.");
        }

        return _transferNode.DownloadedData ?? throw new InvalidOperationException(
            "Download failed - no data available. The graph may not have executed correctly.");
    }

    /// <summary>
    /// Tries to get the downloaded data.
    /// </summary>
    /// <param name="result">The downloaded data if available.</param>
    /// <returns>True if data is available, false otherwise.</returns>
    public bool TryGetResult(out float[]? result)
    {
        if (_isExecuted && _transferNode.DownloadedData != null)
        {
            result = _transferNode.DownloadedData;
            return true;
        }

        result = null;
        return false;
    }

    /// <summary>
    /// Marks the download as executed. Called internally after graph execution.
    /// </summary>
    internal void MarkExecuted()
    {
        _isExecuted = true;
    }
}

/// <summary>
/// Generic version of DeferredDownload that converts to the target type.
/// </summary>
/// <typeparam name="T">The target element type.</typeparam>
public sealed class DeferredDownload<T>
{
    private readonly DeferredDownload _inner;

    /// <summary>
    /// Gets the size of the download in elements.
    /// </summary>
    public int Size => _inner.Size;

    /// <summary>
    /// Gets whether the download has been executed.
    /// </summary>
    public bool IsCompleted => _inner.IsCompleted;

    internal DeferredDownload(DeferredDownload inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    /// <summary>
    /// Gets the downloaded data after execution, converted to the target type.
    /// </summary>
    /// <returns>The downloaded data as a typed array.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the download has not been executed yet.</exception>
    public T[] GetResult()
    {
        float[] floatData = _inner.GetResult();
        return DirectGpuEngine.FromFloatArray<T>(floatData);
    }

    /// <summary>
    /// Gets the raw float data after execution.
    /// Use this when you need to construct higher-level types in the calling assembly.
    /// </summary>
    /// <returns>The downloaded float array.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the download has not been executed yet.</exception>
    public float[] GetResultAsFloat()
    {
        return _inner.GetResult();
    }

    /// <summary>
    /// Tries to get the downloaded data.
    /// </summary>
    /// <param name="result">The downloaded data if available.</param>
    /// <returns>True if data is available, false otherwise.</returns>
    public bool TryGetResult(out T[]? result)
    {
        if (_inner.TryGetResult(out float[]? floatResult) && floatResult != null)
        {
            result = DirectGpuEngine.FromFloatArray<T>(floatResult);
            return true;
        }

        result = null;
        return false;
    }
}
