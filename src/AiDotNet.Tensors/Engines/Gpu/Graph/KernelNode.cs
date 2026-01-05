using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Represents a kernel execution node in the execution graph.
/// Used for operations like GEMM, element-wise operations, reductions, etc.
/// </summary>
public sealed class KernelNode : ExecutionNode
{
    private readonly IGpuTensor[] _inputs;
    private readonly IGpuTensor[] _outputs;
    private readonly Action<IDirectGpuBackend, IGpuStream?> _kernelAction;

    /// <inheritdoc/>
    public override ExecutionNodeType NodeType => ExecutionNodeType.Kernel;

    /// <inheritdoc/>
    public override IReadOnlyList<IGpuTensor> InputTensors => _inputs;

    /// <inheritdoc/>
    public override IReadOnlyList<IGpuTensor> OutputTensors => _outputs;

    /// <inheritdoc/>
    public override bool CanFuse => KernelType.CanFuse();

    /// <summary>
    /// Gets the type of kernel operation.
    /// </summary>
    public KernelType KernelType { get; }

    /// <summary>
    /// Gets additional parameters for the kernel.
    /// </summary>
    public IReadOnlyDictionary<string, object> Parameters { get; }

    /// <inheritdoc/>
    public override int EstimatedCost => KernelType switch
    {
        KernelType.Gemm => 100,
        KernelType.BatchedGemm => 150,
        KernelType.Conv2D => 200,
        KernelType.Conv3D => 300,
        KernelType.Attention => 150,
        KernelType.Softmax => 20,
        KernelType.LayerNorm => 30,
        KernelType.BatchNorm => 30,
        KernelType.Activation => 10,
        KernelType.ElementWise => 5,
        KernelType.Reduction => 25,
        KernelType.Transpose => 15,
        KernelType.Pooling => 40,
        _ => 10
    };

    /// <inheritdoc/>
    public override string Name => $"{KernelType}_{NodeId}";

    /// <summary>
    /// Creates a new kernel node.
    /// </summary>
    /// <param name="kernelType">The type of kernel.</param>
    /// <param name="inputs">Input tensors.</param>
    /// <param name="outputs">Output tensors.</param>
    /// <param name="kernelAction">The action to execute the kernel.</param>
    /// <param name="parameters">Additional kernel parameters.</param>
    public KernelNode(
        KernelType kernelType,
        IGpuTensor[] inputs,
        IGpuTensor[] outputs,
        Action<IDirectGpuBackend, IGpuStream?> kernelAction,
        Dictionary<string, object>? parameters = null)
    {
        KernelType = kernelType;
        _inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        _outputs = outputs ?? throw new ArgumentNullException(nameof(outputs));
        _kernelAction = kernelAction ?? throw new ArgumentNullException(nameof(kernelAction));
        Parameters = parameters ?? new Dictionary<string, object>();
    }

    /// <inheritdoc/>
    public override void Execute(IDirectGpuBackend backend)
    {
        _kernelAction(backend, AssignedStream);
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
        _kernelAction(backend, stream);

        // Mark outputs as modified - handle common tensor types
        foreach (var output in _outputs)
        {
            // Record sync point for modified tensors
            // Note: Do NOT use 'using' here - KernelSyncPoint takes ownership of the event
            // and is responsible for disposing it when the sync point is no longer needed.
            // The tensor holds the sync point and disposes it when appropriate.
            var evt = stream.RecordEvent();
            var syncPoint = new KernelSyncPoint(evt, stream);

            // Handle common tensor types that support MarkModified
            switch (output)
            {
                case IGpuTensor<float> floatTensor:
                    floatTensor.MarkModified(syncPoint);
                    break;
                case IGpuTensor<double> doubleTensor:
                    doubleTensor.MarkModified(syncPoint);
                    break;
                case IGpuTensor<int> intTensor:
                    intTensor.MarkModified(syncPoint);
                    break;
                case IGpuTensor<long> longTensor:
                    longTensor.MarkModified(syncPoint);
                    break;
                // Note: Half/bfloat16 types can be added when supported
            }
        }

        // Record completion for dependents on other streams
        if (Dependents.Any(d => d.AssignedStream != stream))
        {
            var evt = stream.RecordEvent();
            CompletionSync = new KernelSyncPoint(evt, stream);
        }
    }

    /// <inheritdoc/>
    public override ExecutionNode Clone()
    {
        var clonedParams = new Dictionary<string, object>();
        foreach (var kvp in Parameters)
        {
            clonedParams[kvp.Key] = kvp.Value;
        }
        return new KernelNode(KernelType, _inputs, _outputs, _kernelAction, clonedParams);
    }

    /// <summary>
    /// Creates a GEMM kernel node.
    /// </summary>
    public static KernelNode CreateGemm(
        IGpuTensor a, IGpuTensor b, IGpuTensor c,
        int m, int n, int k, float alpha, float beta,
        Action<IDirectGpuBackend, IGpuStream?> action)
    {
        return new KernelNode(
            KernelType.Gemm,
            new[] { a, b },
            new[] { c },
            action,
            new Dictionary<string, object>
            {
                ["M"] = m,
                ["N"] = n,
                ["K"] = k,
                ["Alpha"] = alpha,
                ["Beta"] = beta
            });
    }

    /// <summary>
    /// Creates an activation kernel node.
    /// </summary>
    public static KernelNode CreateActivation(
        IGpuTensor input, IGpuTensor output,
        FusedActivationType activation,
        Action<IDirectGpuBackend, IGpuStream?> action)
    {
        return new KernelNode(
            KernelType.Activation,
            new[] { input },
            new[] { output },
            action,
            new Dictionary<string, object>
            {
                ["Activation"] = activation
            });
    }

    /// <summary>
    /// Creates an element-wise operation kernel node.
    /// </summary>
    public static KernelNode CreateElementWise(
        IGpuTensor[] inputs, IGpuTensor output,
        ElementWiseOp operation,
        Action<IDirectGpuBackend, IGpuStream?> action)
    {
        return new KernelNode(
            KernelType.ElementWise,
            inputs,
            new[] { output },
            action,
            new Dictionary<string, object>
            {
                ["Operation"] = operation
            });
    }

    private sealed class KernelSyncPoint : GpuSyncPoint
    {
        private readonly IGpuEvent _event;
        private readonly IGpuStream _stream;

        public KernelSyncPoint(IGpuEvent evt, IGpuStream stream)
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
/// Types of kernel operations.
/// </summary>
public enum KernelType
{
    /// <summary>General matrix multiplication.</summary>
    Gemm,
    /// <summary>Batched matrix multiplication.</summary>
    BatchedGemm,
    /// <summary>2D convolution.</summary>
    Conv2D,
    /// <summary>3D convolution.</summary>
    Conv3D,
    /// <summary>Attention mechanism.</summary>
    Attention,
    /// <summary>Softmax operation.</summary>
    Softmax,
    /// <summary>Layer normalization.</summary>
    LayerNorm,
    /// <summary>Batch normalization.</summary>
    BatchNorm,
    /// <summary>Activation function.</summary>
    Activation,
    /// <summary>Element-wise operation.</summary>
    ElementWise,
    /// <summary>Reduction operation (sum, mean, max, etc.).</summary>
    Reduction,
    /// <summary>Transpose or permutation.</summary>
    Transpose,
    /// <summary>Pooling operation.</summary>
    Pooling,
    /// <summary>Custom kernel.</summary>
    Custom
}

/// <summary>
/// Element-wise operation types.
/// </summary>
public enum ElementWiseOp
{
    /// <summary>Addition.</summary>
    Add,
    /// <summary>Subtraction.</summary>
    Subtract,
    /// <summary>Multiplication.</summary>
    Multiply,
    /// <summary>Division.</summary>
    Divide,
    /// <summary>Maximum.</summary>
    Maximum,
    /// <summary>Minimum.</summary>
    Minimum,
    /// <summary>Power.</summary>
    Power,
    /// <summary>Square root.</summary>
    Sqrt,
    /// <summary>Exponential.</summary>
    Exp,
    /// <summary>Natural logarithm.</summary>
    Log,
    /// <summary>Absolute value.</summary>
    Abs,
    /// <summary>Negation.</summary>
    Negate
}

/// <summary>
/// Extension methods for kernel types.
/// </summary>
public static class KernelTypeExtensions
{
    /// <summary>
    /// Gets whether a kernel type can be fused with other operations.
    /// </summary>
    public static bool CanFuse(this KernelType kernelType)
    {
        return kernelType switch
        {
            KernelType.Gemm => true,       // Can fuse with bias and activation
            KernelType.Conv2D => true,     // Can fuse with batch norm and activation
            KernelType.LayerNorm => true,  // Can fuse with activation
            KernelType.Activation => true, // Can be fused into preceding op
            KernelType.ElementWise => true, // Can be fused
            _ => false
        };
    }
}
