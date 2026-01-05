using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Represents a fused kernel execution node combining multiple operations.
/// Examples: GEMM+Bias+ReLU, Conv+BatchNorm+ReLU, etc.
/// </summary>
public sealed class FusedKernelNode : ExecutionNode
{
    private readonly IGpuTensor[] _inputs;
    private readonly IGpuTensor[] _outputs;
    private readonly Action<IDirectGpuBackend, IGpuStream?> _fusedKernelAction;
    private readonly List<ExecutionNode> _originalNodes;

    /// <inheritdoc/>
    public override ExecutionNodeType NodeType => ExecutionNodeType.FusedKernel;

    /// <inheritdoc/>
    public override IReadOnlyList<IGpuTensor> InputTensors => _inputs;

    /// <inheritdoc/>
    public override IReadOnlyList<IGpuTensor> OutputTensors => _outputs;

    /// <inheritdoc/>
    public override bool CanFuse => false; // Already fused

    /// <summary>
    /// Gets the type of fused operation.
    /// </summary>
    public FusedOperationType FusedType { get; }

    /// <summary>
    /// Gets the original nodes that were fused.
    /// </summary>
    public IReadOnlyList<ExecutionNode> OriginalNodes => _originalNodes;

    /// <summary>
    /// Gets the activation type if this fusion includes activation.
    /// </summary>
    public FusedActivationType? Activation { get; }

    /// <summary>
    /// Gets additional parameters for the fused kernel.
    /// </summary>
    public IReadOnlyDictionary<string, object> Parameters { get; }

    /// <inheritdoc/>
    public override int EstimatedCost => FusedType switch
    {
        FusedOperationType.GemmBias => 95,         // Slightly less than GEMM + Bias separate
        FusedOperationType.GemmBiasActivation => 90, // Even better fusion
        FusedOperationType.ConvBatchNorm => 180,
        FusedOperationType.ConvBatchNormActivation => 175,
        FusedOperationType.AttentionFused => 140,
        FusedOperationType.LayerNormActivation => 28,
        _ => 50
    };

    /// <inheritdoc/>
    public override string Name => $"{FusedType}_{NodeId}";

    /// <summary>
    /// Creates a new fused kernel node.
    /// </summary>
    /// <param name="fusedType">The type of fused operation.</param>
    /// <param name="inputs">Input tensors.</param>
    /// <param name="outputs">Output tensors.</param>
    /// <param name="fusedKernelAction">The action to execute the fused kernel.</param>
    /// <param name="originalNodes">The original nodes that were fused.</param>
    /// <param name="activation">Optional activation to apply.</param>
    /// <param name="parameters">Additional parameters.</param>
    public FusedKernelNode(
        FusedOperationType fusedType,
        IGpuTensor[] inputs,
        IGpuTensor[] outputs,
        Action<IDirectGpuBackend, IGpuStream?> fusedKernelAction,
        IEnumerable<ExecutionNode>? originalNodes = null,
        FusedActivationType? activation = null,
        Dictionary<string, object>? parameters = null)
    {
        FusedType = fusedType;
        _inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        _outputs = outputs ?? throw new ArgumentNullException(nameof(outputs));
        _fusedKernelAction = fusedKernelAction ?? throw new ArgumentNullException(nameof(fusedKernelAction));
        _originalNodes = originalNodes?.ToList() ?? new List<ExecutionNode>();
        Activation = activation;
        Parameters = parameters ?? new Dictionary<string, object>();
    }

    /// <inheritdoc/>
    public override void Execute(IDirectGpuBackend backend)
    {
        _fusedKernelAction(backend, AssignedStream);
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
        _fusedKernelAction(backend, stream);

        // Mark outputs as modified
        foreach (var output in _outputs)
        {
            if (output is IGpuTensor<float> typedTensor)
            {
                var evt = stream.RecordEvent();
                var syncPoint = new FusedSyncPoint(evt, stream);
                typedTensor.MarkModified(syncPoint);
            }
        }

        // Record completion for dependents on other streams
        if (Dependents.Any(d => d.AssignedStream != stream))
        {
            var evt = stream.RecordEvent();
            CompletionSync = new FusedSyncPoint(evt, stream);
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
        return new FusedKernelNode(
            FusedType,
            _inputs,
            _outputs,
            _fusedKernelAction,
            _originalNodes,
            Activation,
            clonedParams);
    }

    /// <summary>
    /// Creates a GEMM+Bias fused node.
    /// </summary>
    public static FusedKernelNode CreateGemmBias(
        IGpuTensor a, IGpuTensor b, IGpuTensor bias, IGpuTensor output,
        int m, int n, int k,
        Action<IDirectGpuBackend, IGpuStream?> action,
        IEnumerable<ExecutionNode>? originalNodes = null)
    {
        return new FusedKernelNode(
            FusedOperationType.GemmBias,
            new[] { a, b, bias },
            new[] { output },
            action,
            originalNodes,
            null,
            new Dictionary<string, object>
            {
                ["M"] = m,
                ["N"] = n,
                ["K"] = k
            });
    }

    /// <summary>
    /// Creates a GEMM+Bias+Activation fused node.
    /// </summary>
    public static FusedKernelNode CreateGemmBiasActivation(
        IGpuTensor a, IGpuTensor b, IGpuTensor bias, IGpuTensor output,
        int m, int n, int k,
        FusedActivationType activation,
        Action<IDirectGpuBackend, IGpuStream?> action,
        IEnumerable<ExecutionNode>? originalNodes = null)
    {
        return new FusedKernelNode(
            FusedOperationType.GemmBiasActivation,
            new[] { a, b, bias },
            new[] { output },
            action,
            originalNodes,
            activation,
            new Dictionary<string, object>
            {
                ["M"] = m,
                ["N"] = n,
                ["K"] = k
            });
    }

    /// <summary>
    /// Attempts to fuse multiple kernel nodes into a single fused node.
    /// </summary>
    /// <param name="nodes">Nodes to attempt to fuse.</param>
    /// <param name="backend">Backend to create fused kernel action.</param>
    /// <returns>A fused node if fusion is possible, null otherwise.</returns>
    public static FusedKernelNode? TryFuse(IReadOnlyList<KernelNode> nodes, IAsyncGpuBackend backend)
    {
        if (nodes.Count < 2)
        {
            return null;
        }

        // Check for GEMM + Bias + Activation pattern
        if (nodes.Count >= 2 && nodes[0].KernelType == KernelType.Gemm)
        {
            var gemmNode = nodes[0];

            // Look for bias add
            var biasNode = nodes.Skip(1).FirstOrDefault(n =>
                n.KernelType == KernelType.ElementWise &&
                n.Parameters.TryGetValue("Operation", out var op) &&
                (ElementWiseOp)op == ElementWiseOp.Add);

            if (biasNode != null)
            {
                // Look for activation
                var activationNode = nodes.Skip(1).FirstOrDefault(n => n.KernelType == KernelType.Activation);

                if (activationNode != null)
                {
                    // Can fuse GEMM + Bias + Activation
                    var activation = activationNode.Parameters.TryGetValue("Activation", out var act)
                        ? (FusedActivationType)act
                        : FusedActivationType.None;

                    var gemmInputs = gemmNode.InputTensors;
                    var biasInput = biasNode.InputTensors.FirstOrDefault(t => !gemmNode.OutputTensors.Contains(t));
                    var output = activationNode.OutputTensors[0];

                    if (biasInput != null && gemmInputs.Count >= 2)
                    {
                        var inputs = new[] { gemmInputs[0], gemmInputs[1], biasInput };

                        int m = gemmNode.Parameters.TryGetValue("M", out var mVal) ? (int)mVal : 0;
                        int n = gemmNode.Parameters.TryGetValue("N", out var nVal) ? (int)nVal : 0;
                        int k = gemmNode.Parameters.TryGetValue("K", out var kVal) ? (int)kVal : 0;

                        Action<IDirectGpuBackend, IGpuStream?> fusedAction = (b, stream) =>
                        {
                            if (b is IAsyncGpuBackend asyncBackend && stream != null)
                            {
                                asyncBackend.FusedGemmBiasActivationAsync(
                                    gemmInputs[0].Buffer, gemmInputs[1].Buffer, biasInput.Buffer, output.Buffer,
                                    m, n, k, activation, stream);
                            }
                        };

                        return CreateGemmBiasActivation(
                            inputs[0], inputs[1], inputs[2], output,
                            m, n, k, activation, fusedAction,
                            new[] { gemmNode, biasNode, activationNode });
                    }
                }
                else
                {
                    // Can fuse GEMM + Bias only
                    var gemmInputs = gemmNode.InputTensors;
                    var biasInput = biasNode.InputTensors.FirstOrDefault(t => !gemmNode.OutputTensors.Contains(t));
                    var output = biasNode.OutputTensors[0];

                    if (biasInput != null && gemmInputs.Count >= 2)
                    {
                        var inputs = new[] { gemmInputs[0], gemmInputs[1], biasInput };

                        int m = gemmNode.Parameters.TryGetValue("M", out var mVal) ? (int)mVal : 0;
                        int n = gemmNode.Parameters.TryGetValue("N", out var nVal) ? (int)nVal : 0;
                        int k = gemmNode.Parameters.TryGetValue("K", out var kVal) ? (int)kVal : 0;

                        Action<IDirectGpuBackend, IGpuStream?> fusedAction = (b, stream) =>
                        {
                            if (b is IAsyncGpuBackend asyncBackend && stream != null)
                            {
                                asyncBackend.FusedGemmBiasActivationAsync(
                                    gemmInputs[0].Buffer, gemmInputs[1].Buffer, biasInput.Buffer, output.Buffer,
                                    m, n, k, FusedActivationType.None, stream);
                            }
                        };

                        return CreateGemmBias(
                            inputs[0], inputs[1], inputs[2], output,
                            m, n, k, fusedAction,
                            new[] { gemmNode, biasNode });
                    }
                }
            }
        }

        return null;
    }

    private sealed class FusedSyncPoint : GpuSyncPoint
    {
        private readonly IGpuEvent _event;
        private readonly IGpuStream _stream;

        public FusedSyncPoint(IGpuEvent evt, IGpuStream stream)
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
/// Types of fused operations.
/// </summary>
public enum FusedOperationType
{
    /// <summary>GEMM with bias addition.</summary>
    GemmBias,
    /// <summary>GEMM with bias addition and activation.</summary>
    GemmBiasActivation,
    /// <summary>Convolution with batch normalization.</summary>
    ConvBatchNorm,
    /// <summary>Convolution with batch normalization and activation.</summary>
    ConvBatchNormActivation,
    /// <summary>Fused attention (Q*K^T/sqrt(d), softmax, *V).</summary>
    AttentionFused,
    /// <summary>Layer normalization with activation.</summary>
    LayerNormActivation,
    /// <summary>Custom fused operation.</summary>
    Custom
}
