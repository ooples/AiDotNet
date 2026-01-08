using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Builder for constructing execution graphs from recorded operations.
/// Tracks dependencies and provides fluent API for building graphs.
/// </summary>
public sealed class ExecutionGraphBuilder : IDisposable
{
    private readonly List<ExecutionNode> _nodes = new();
    private readonly Dictionary<IGpuBuffer, ExecutionNode> _lastWriteNode = new();
    private readonly Dictionary<IGpuBuffer, List<ExecutionNode>> _bufferDependents = new();
    private readonly List<string> _validationWarnings = new();
    private bool _isBuilt;
    private bool _disposed;

    /// <summary>
    /// Gets any validation warnings generated during the last Build() call.
    /// These indicate potential issues that don't prevent execution but may
    /// indicate unintended graph structure (e.g., orphaned nodes with no dependencies).
    /// </summary>
    public IReadOnlyList<string> ValidationWarnings => _validationWarnings;

    /// <summary>
    /// Gets the current number of nodes in the graph.
    /// </summary>
    public int NodeCount => _nodes.Count;

    /// <summary>
    /// Gets all nodes added to the builder.
    /// </summary>
    public IReadOnlyList<ExecutionNode> Nodes => _nodes;

    /// <summary>
    /// Gets whether the graph has been built.
    /// </summary>
    public bool IsBuilt => _isBuilt;

    /// <summary>
    /// Adds a node to the graph with automatic dependency tracking.
    /// </summary>
    /// <param name="node">The node to add.</param>
    /// <returns>The builder for chaining.</returns>
    public ExecutionGraphBuilder AddNode(ExecutionNode node)
    {
        ThrowIfBuilt();

        if (node == null)
        {
            throw new ArgumentNullException(nameof(node));
        }

        // Automatically add dependencies based on buffer usage
        TrackBufferDependencies(node);

        _nodes.Add(node);
        return this;
    }

    /// <summary>
    /// Adds a kernel node for GEMM operation.
    /// </summary>
    public ExecutionGraphBuilder AddGemm(
        IGpuTensor a, IGpuTensor b, IGpuTensor c,
        int m, int n, int k, float alpha, float beta,
        Action<IDirectGpuBackend, IGpuStream?> action)
    {
        var node = KernelNode.CreateGemm(a, b, c, m, n, k, alpha, beta, action);
        return AddNode(node);
    }

    /// <summary>
    /// Adds a kernel node for activation function.
    /// </summary>
    public ExecutionGraphBuilder AddActivation(
        IGpuTensor input, IGpuTensor output,
        FusedActivationType activation,
        Action<IDirectGpuBackend, IGpuStream?> action)
    {
        var node = KernelNode.CreateActivation(input, output, activation, action);
        return AddNode(node);
    }

    /// <summary>
    /// Adds an element-wise operation kernel node.
    /// </summary>
    public ExecutionGraphBuilder AddElementWise(
        IGpuTensor[] inputs, IGpuTensor output,
        ElementWiseOp operation,
        Action<IDirectGpuBackend, IGpuStream?> action)
    {
        var node = KernelNode.CreateElementWise(inputs, output, operation, action);
        return AddNode(node);
    }

    /// <summary>
    /// Adds a custom kernel node.
    /// </summary>
    public ExecutionGraphBuilder AddKernel(
        KernelType kernelType,
        IGpuTensor[] inputs,
        IGpuTensor[] outputs,
        Action<IDirectGpuBackend, IGpuStream?> action,
        Dictionary<string, object>? parameters = null)
    {
        var node = new KernelNode(kernelType, inputs, outputs, action, parameters);
        return AddNode(node);
    }

    /// <summary>
    /// Adds a host-to-device transfer.
    /// </summary>
    public ExecutionGraphBuilder AddUpload(float[] sourceData, IGpuBuffer destinationBuffer)
    {
        var node = TransferNode.CreateH2D(sourceData, destinationBuffer);
        return AddNode(node);
    }

    /// <summary>
    /// Adds a device-to-host transfer.
    /// </summary>
    public ExecutionGraphBuilder AddDownload(IGpuBuffer sourceBuffer, int size)
    {
        var node = TransferNode.CreateD2H(sourceBuffer, size);
        AddNode(node);
        return this;
    }

    /// <summary>
    /// Adds a device-to-host transfer and returns the transfer node for deferred access.
    /// </summary>
    /// <param name="sourceBuffer">The source GPU buffer to download.</param>
    /// <param name="size">The number of elements to download.</param>
    /// <returns>The transfer node that will contain the downloaded data after execution.</returns>
    public TransferNode AddDownloadWithHandle(IGpuBuffer sourceBuffer, int size)
    {
        var node = TransferNode.CreateD2H(sourceBuffer, size);
        AddNode(node);
        return node;
    }

    /// <summary>
    /// Adds a device-to-device copy.
    /// </summary>
    public ExecutionGraphBuilder AddCopy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        var node = TransferNode.CreateD2D(source, destination, size);
        return AddNode(node);
    }

    /// <summary>
    /// Adds a buffer allocation.
    /// </summary>
    public ExecutionGraphBuilder AddAllocate(int size, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        var node = AllocationNode.CreateAllocate(size, role);
        return AddNode(node);
    }

    /// <summary>
    /// Adds a buffer deallocation.
    /// </summary>
    public ExecutionGraphBuilder AddDeallocate(IGpuBuffer buffer)
    {
        var node = AllocationNode.CreateDeallocate(buffer);
        return AddNode(node);
    }

    /// <summary>
    /// Adds a barrier that synchronizes all preceding operations.
    /// </summary>
    public ExecutionGraphBuilder AddBarrier()
    {
        var node = new BarrierNode();

        // Add dependencies on all nodes without dependents
        foreach (var existingNode in _nodes)
        {
            if (existingNode.Dependents.Count == 0)
            {
                node.AddDependency(existingNode);
            }
        }

        return AddNode(node);
    }

    /// <summary>
    /// Adds a barrier for specific streams.
    /// </summary>
    public ExecutionGraphBuilder AddBarrier(IEnumerable<IGpuStream> streams)
    {
        var node = new BarrierNode(streams);
        return AddNode(node);
    }

    /// <summary>
    /// Adds an explicit dependency between two nodes.
    /// </summary>
    public ExecutionGraphBuilder AddDependency(ExecutionNode from, ExecutionNode to)
    {
        ThrowIfBuilt();
        to.AddDependency(from);
        return this;
    }

    /// <summary>
    /// Makes a node depend on the last write to a buffer.
    /// </summary>
    public ExecutionGraphBuilder DependOnBuffer(ExecutionNode node, IGpuBuffer buffer)
    {
        ThrowIfBuilt();

        if (_lastWriteNode.TryGetValue(buffer, out var lastWriter))
        {
            node.AddDependency(lastWriter);
        }

        return this;
    }

    /// <summary>
    /// Builds the execution graph.
    /// </summary>
    /// <returns>The built execution graph.</returns>
    public ExecutionGraph Build()
    {
        ThrowIfBuilt();

        // Validate the graph
        ValidateGraph();

        _isBuilt = true;
        return new ExecutionGraph(_nodes.ToList());
    }

    /// <summary>
    /// Builds and optimizes the execution graph with the specified options.
    /// </summary>
    /// <param name="options">Optimization options.</param>
    /// <param name="backend">Backend for optimization passes.</param>
    /// <returns>The optimized execution graph.</returns>
    public ExecutionGraph BuildOptimized(GpuExecutionOptions options, IAsyncGpuBackend backend)
    {
        var (graph, _) = BuildOptimizedWithStatistics(options, backend);
        return graph;
    }

    /// <summary>
    /// Builds and optimizes the execution graph, returning both the graph and optimization statistics.
    /// </summary>
    /// <param name="options">Optimization options.</param>
    /// <param name="backend">Backend for optimization passes.</param>
    /// <returns>A tuple containing the optimized execution graph and optimization statistics.</returns>
    public (ExecutionGraph Graph, Optimization.OptimizationStatistics Statistics) BuildOptimizedWithStatistics(
        GpuExecutionOptions options, IAsyncGpuBackend backend)
    {
        ThrowIfBuilt();
        ThrowIfDisposed();

        // Validate first
        ValidateGraph();

        var compiler = new GraphCompiler(options);
        var (optimizedGraph, statistics) = compiler.CompileWithStatistics(_nodes.ToList(), backend);

        _isBuilt = true;
        return (optimizedGraph, statistics);
    }

    /// <summary>
    /// Clears the builder for reuse.
    /// </summary>
    public void Clear()
    {
        _nodes.Clear();
        _lastWriteNode.Clear();
        _bufferDependents.Clear();
        _isBuilt = false;
    }

    private void TrackBufferDependencies(ExecutionNode node)
    {
        // Get input buffers and add dependencies on their last writers
        foreach (var input in node.InputTensors)
        {
            if (_lastWriteNode.TryGetValue(input.Buffer, out var lastWriter))
            {
                node.AddDependency(lastWriter);
            }

            // Track this node as a dependent of the buffer
            if (!_bufferDependents.TryGetValue(input.Buffer, out var dependents))
            {
                dependents = new List<ExecutionNode>();
                _bufferDependents[input.Buffer] = dependents;
            }
            dependents.Add(node);
        }

        // Track this node as the last writer to its output buffers
        foreach (var output in node.OutputTensors)
        {
            _lastWriteNode[output.Buffer] = node;
        }

        // Handle transfer nodes specially
        if (node is TransferNode transfer)
        {
            if (transfer.SourceBuffer != null)
            {
                if (_lastWriteNode.TryGetValue(transfer.SourceBuffer, out var srcWriter))
                {
                    node.AddDependency(srcWriter);
                }
            }

            if (transfer.DestinationBuffer != null)
            {
                _lastWriteNode[transfer.DestinationBuffer] = node;
            }
        }
    }

    private void ValidateGraph()
    {
        _validationWarnings.Clear();

        // Check for circular dependencies
        foreach (var node in _nodes)
        {
            if (!node.Validate())
            {
                throw new InvalidOperationException(
                    $"Circular dependency detected involving node {node.Name}");
            }
        }

        // Identify orphaned nodes (no dependencies, not first, not special types)
        // These may indicate unintended graph structure but are not errors
        var orphanedNodes = _nodes.Where(n =>
            n.Dependencies.Count == 0 &&
            _nodes.IndexOf(n) > 0 &&
            !(n is AllocationNode) &&
            !(n is BarrierNode)).ToList();

        foreach (var orphan in orphanedNodes)
        {
            _validationWarnings.Add(
                $"Node '{orphan.Name}' (type: {orphan.NodeType}) has no dependencies " +
                $"but is not at graph start. This may indicate missing dependency links.");
        }

        // Check for nodes with no dependents that aren't terminal operations
        var deadEndNodes = _nodes.Where(n =>
            n.Dependents.Count == 0 &&
            n.NodeType != ExecutionNodeType.TransferD2H &&
            n.NodeType != ExecutionNodeType.Deallocate &&
            n.NodeType != ExecutionNodeType.Barrier &&
            _nodes.IndexOf(n) < _nodes.Count - 1).ToList();

        foreach (var deadEnd in deadEndNodes)
        {
            _validationWarnings.Add(
                $"Node '{deadEnd.Name}' (type: {deadEnd.NodeType}) has no dependents. " +
                $"Its output may be unused (dead code).");
        }
    }

    private void ThrowIfBuilt()
    {
        // Also check disposed state since all public methods should verify both
        ThrowIfDisposed();

        if (_isBuilt)
        {
            throw new InvalidOperationException(
                "Graph has already been built. Call Clear() to reuse the builder.");
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ExecutionGraphBuilder));
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Clear();
    }
}

/// <summary>
/// Extension methods for building common graph patterns.
/// </summary>
public static class ExecutionGraphBuilderExtensions
{
    /// <summary>
    /// Adds a linear layer (GEMM + bias + optional activation).
    /// </summary>
    public static ExecutionGraphBuilder AddLinearLayer(
        this ExecutionGraphBuilder builder,
        IGpuTensor input, IGpuTensor weights, IGpuTensor bias, IGpuTensor output,
        int batchSize, int inputFeatures, int outputFeatures,
        FusedActivationType? activation,
        IDirectGpuBackend backend)
    {
        if (activation.HasValue && backend is IAsyncGpuBackend asyncBackend)
        {
            // Use fused kernel
            Action<IDirectGpuBackend, IGpuStream?> fusedAction = (b, stream) =>
            {
                if (b is IAsyncGpuBackend ab && stream != null)
                {
                    ab.FusedGemmBiasActivationAsync(
                        input.Buffer, weights.Buffer, bias.Buffer, output.Buffer,
                        batchSize, outputFeatures, inputFeatures,
                        activation.Value, stream);
                }
            };

            var fusedNode = FusedKernelNode.CreateGemmBiasActivation(
                input, weights, bias, output,
                batchSize, outputFeatures, inputFeatures,
                activation.Value, fusedAction);

            return builder.AddNode(fusedNode);
        }
        else
        {
            // Add separate nodes
            builder.AddGemm(input, weights, output, batchSize, outputFeatures, inputFeatures, 1f, 0f,
                (b, s) => b.Gemm(input.Buffer, weights.Buffer, output.Buffer,
                    batchSize, outputFeatures, inputFeatures, 1f, 0f));

            builder.AddElementWise(new[] { output, bias }, output, ElementWiseOp.Add,
                (b, s) => b.Add(output.Buffer, bias.Buffer, output.Buffer, output.Buffer.Size));

            if (activation.HasValue && activation.Value != FusedActivationType.None)
            {
                builder.AddActivation(output, output, activation.Value,
                    (b, s) => ApplyActivation(b, output.Buffer, output.Buffer, output.Buffer.Size, activation.Value));
            }

            return builder;
        }
    }

    /// <summary>
    /// Adds a batch of operations that can execute in parallel.
    /// </summary>
    public static ExecutionGraphBuilder AddParallelBatch(
        this ExecutionGraphBuilder builder,
        Action<ExecutionGraphBuilder>[] batchOperations)
    {
        if (batchOperations == null || batchOperations.Length == 0)
        {
            return builder;
        }

        // Record start position
        int startIdx = builder.NodeCount;

        // Add all operations
        foreach (var op in batchOperations)
        {
            op(builder);
        }

        // All nodes added in the batch don't depend on each other unless
        // they share buffers (which is handled automatically)

        return builder;
    }

    private static void ApplyActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size, FusedActivationType activation)
    {
        switch (activation)
        {
            case FusedActivationType.None:
                // Identity - copy input to output if they're different buffers
                if (input != output)
                {
                    backend.Copy(input, output, size);
                }
                break;
            case FusedActivationType.ReLU:
                backend.Relu(input, output, size);
                break;
            case FusedActivationType.Sigmoid:
                backend.Sigmoid(input, output, size);
                break;
            case FusedActivationType.Tanh:
                backend.Tanh(input, output, size);
                break;
            case FusedActivationType.GELU:
                backend.Gelu(input, output, size);
                break;
            case FusedActivationType.LeakyReLU:
                // Default alpha for LeakyReLU is 0.01
                backend.LeakyRelu(input, output, 0.01f, size);
                break;
            case FusedActivationType.Swish:
                backend.Swish(input, output, size);
                break;
            case FusedActivationType.Softmax:
                // Softmax requires batch and feature dimensions.
                // When only total size is available, treat as single batch.
                backend.Softmax(input, output, 1, size);
                break;
        }
    }
}
