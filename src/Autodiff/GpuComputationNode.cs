using AiDotNet.Gpu;
using AiDotNet.Helpers;

namespace AiDotNet.Autodiff;

/// <summary>
/// Represents a computation node that supports GPU acceleration for automatic differentiation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GpuComputationNode extends the automatic differentiation system to support GPU-accelerated
/// operations. It maintains both CPU and GPU representations of tensors, automatically managing
/// data transfers based on execution context policies.
/// </para>
/// <para><b>For Beginners:</b> This is like a regular ComputationNode but can use the GPU for speed!
///
/// Key features:
/// - Automatically decides when to use GPU vs CPU
/// - Manages GPU memory lifecycle
/// - Transparent to existing autodiff code
/// - Can mix CPU and GPU operations seamlessly
///
/// Example:
/// <code>
/// var context = new ExecutionContext(backend)
/// {
///     Strategy = PlacementStrategy.AutomaticPlacement
/// };
///
/// var node1 = GpuComputationNode.Create(tensor1, context, requiresGradient: true);
/// var node2 = GpuComputationNode.Create(tensor2, context, requiresGradient: true);
///
/// // Automatically uses GPU for large tensors
/// var result = GpuTensorOperations.Add(node1, node2, context);
/// result.Backward(); // Gradients computed on GPU where beneficial
/// </code>
/// </para>
/// </remarks>
public class GpuComputationNode<T> : ComputationNode<T>, IDisposable
    where T : unmanaged
{
    private bool _disposed;
    private GpuTensor<T>? _gpuValue;
    private GpuTensor<T>? _gpuGradient;

    /// <summary>
    /// Gets the execution context that controls CPU/GPU placement.
    /// </summary>
    public ExecutionContext? Context { get; }

    /// <summary>
    /// Gets or sets the GPU tensor value (null if data is on CPU).
    /// </summary>
    /// <remarks>
    /// When not null, this contains the same data as Value but on GPU.
    /// The execution context determines which version to use for operations.
    /// </remarks>
    public GpuTensor<T>? GpuValue
    {
        get => _gpuValue;
        set
        {
            if (_gpuValue != value)
            {
                _gpuValue?.Dispose();
                _gpuValue = value;
            }
        }
    }

    /// <summary>
    /// Gets or sets the GPU gradient tensor (null if gradient is on CPU or not computed).
    /// </summary>
    public GpuTensor<T>? GpuGradient
    {
        get => _gpuGradient;
        set
        {
            if (_gpuGradient != value)
            {
                _gpuGradient?.Dispose();
                _gpuGradient = value;
            }
        }
    }

    /// <summary>
    /// Gets a value indicating whether this node's value is currently on GPU.
    /// </summary>
    public bool IsOnGpu => GpuValue != null;

    /// <summary>
    /// Gets a value indicating whether this node's gradient is currently on GPU.
    /// </summary>
    public bool IsGradientOnGpu => GpuGradient != null;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuComputationNode{T}"/> class.
    /// </summary>
    /// <param name="value">The CPU tensor value.</param>
    /// <param name="context">The execution context for GPU placement decisions.</param>
    /// <param name="requiresGradient">Whether this node requires gradient computation.</param>
    /// <param name="parents">The parent nodes that were used to compute this value.</param>
    /// <param name="backwardFunction">The function to compute gradients during backpropagation.</param>
    /// <param name="name">Optional name for this node.</param>
    public GpuComputationNode(
        Tensor<T> value,
        ExecutionContext? context = null,
        bool requiresGradient = false,
        List<ComputationNode<T>>? parents = null,
        Action<Tensor<T>>? backwardFunction = null,
        string? name = null)
        : base(value, requiresGradient, parents, backwardFunction, name)
    {
        Context = context;
    }

    /// <summary>
    /// Creates a new GPU computation node with automatic placement.
    /// </summary>
    /// <param name="value">The tensor value.</param>
    /// <param name="context">The execution context.</param>
    /// <param name="requiresGradient">Whether gradients are needed.</param>
    /// <param name="name">Optional node name.</param>
    /// <returns>A new GPU computation node.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the recommended way to create GPU nodes.
    ///
    /// The method:
    /// 1. Creates the node with the CPU tensor
    /// 2. Checks if GPU should be used (based on context strategy)
    /// 3. Automatically transfers to GPU if beneficial
    /// 4. Returns a node ready to use
    ///
    /// The context handles all the complexity of deciding when to use GPU!
    /// </para>
    /// </remarks>
    public static GpuComputationNode<T> Create(
        Tensor<T> value,
        ExecutionContext? context,
        bool requiresGradient = false,
        string? name = null)
    {
        var node = new GpuComputationNode<T>(value, context, requiresGradient, name: name);

        // Automatically move to GPU if context suggests
        if (context != null && context.ShouldUseGpu(value))
        {
            node.MoveToGpu();
        }

        return node;
    }

    /// <summary>
    /// Moves the value to GPU memory.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This uploads the tensor data to GPU.
    ///
    /// When to call manually:
    /// - Usually you don't! Create() handles this automatically
    /// - Use when you know a sequence of GPU operations is coming
    /// - Useful for MinimizeTransfers strategy
    ///
    /// The CPU value remains available - both versions stay in sync.
    /// </para>
    /// </remarks>
    public void MoveToGpu()
    {
        if (IsOnGpu || Context?.GpuBackend == null)
        {
            return;
        }

        // Get the appropriate backend for type T
        var backend = Context.GpuBackend as IGpuBackend<T>;
        if (backend == null)
        {
            return;
        }

        GpuValue = backend.ToGpu(Value);
    }

    /// <summary>
    /// Moves the value back to CPU memory and disposes GPU memory.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This downloads data from GPU and frees GPU memory.
    ///
    /// When to call:
    /// - After completing all GPU operations
    /// - Before accessing individual elements
    /// - When GPU memory is running low
    ///
    /// The CPU value is updated with the latest GPU data before freeing.
    /// </para>
    /// </remarks>
    public void MoveToCpu()
    {
        if (!IsOnGpu || Context?.GpuBackend == null)
        {
            return;
        }

        var backend = Context.GpuBackend as IGpuBackend<T>;
        if (backend != null && GpuValue != null)
        {
            // Update CPU value with GPU data
            Value = backend.ToCpu(GpuValue);

            // Free GPU memory
            GpuValue?.Dispose();
            GpuValue = null;
        }
    }

    /// <summary>
    /// Ensures the value is available on GPU, transferring if necessary.
    /// </summary>
    /// <returns>The GPU tensor value.</returns>
    /// <exception cref="InvalidOperationException">If GPU backend is not available.</exception>
    public GpuTensor<T> EnsureOnGpu()
    {
        if (!IsOnGpu)
        {
            MoveToGpu();
        }

        if (GpuValue == null)
        {
            throw new InvalidOperationException("Failed to move tensor to GPU. GPU backend may not be available.");
        }

        return GpuValue;
    }

    /// <summary>
    /// Ensures the value is available on CPU, transferring if necessary.
    /// </summary>
    /// <returns>The CPU tensor value.</returns>
    public Tensor<T> EnsureOnCpu()
    {
        if (IsOnGpu && Context?.GpuBackend != null)
        {
            var backend = Context.GpuBackend as IGpuBackend<T>;
            if (backend != null && GpuValue != null)
            {
                // Update CPU value from GPU (but keep GPU copy)
                Value = backend.ToCpu(GpuValue);
            }
        }

        return Value;
    }

    /// <summary>
    /// Synchronizes CPU and GPU values, ensuring they match.
    /// </summary>
    /// <param name="preferGpu">If true, GPU value is treated as source of truth.</param>
    public void Synchronize(bool preferGpu = true)
    {
        if (!IsOnGpu || Context?.GpuBackend == null)
        {
            return;
        }

        var backend = Context.GpuBackend as IGpuBackend<T>;
        if (backend == null || GpuValue == null)
        {
            return;
        }

        if (preferGpu)
        {
            // GPU → CPU
            Value = backend.ToCpu(GpuValue);
        }
        else
        {
            // CPU → GPU
            GpuValue?.Dispose();
            GpuValue = backend.ToGpu(Value);
        }
    }

    /// <summary>
    /// Moves the gradient to GPU memory.
    /// </summary>
    /// <remarks>
    /// Used during backward pass when gradients are computed on GPU.
    /// </remarks>
    public void MoveGradientToGpu()
    {
        if (IsGradientOnGpu || Gradient == null || Context?.GpuBackend == null)
        {
            return;
        }

        var backend = Context.GpuBackend as IGpuBackend<T>;
        if (backend != null)
        {
            GpuGradient = backend.ToGpu(Gradient);
        }
    }

    /// <summary>
    /// Moves the gradient back to CPU memory.
    /// </summary>
    public void MoveGradientToCpu()
    {
        if (!IsGradientOnGpu || Context?.GpuBackend == null)
        {
            return;
        }

        var backend = Context.GpuBackend as IGpuBackend<T>;
        if (backend != null && GpuGradient != null)
        {
            Gradient = backend.ToCpu(GpuGradient);
            GpuGradient?.Dispose();
            GpuGradient = null;
        }
    }

    /// <summary>
    /// Disposes GPU resources held by this node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This frees GPU memory used by this node.
    ///
    /// IMPORTANT:
    /// - Always dispose GPU nodes when done
    /// - Use 'using' statements for automatic disposal
    /// - Not disposing causes GPU memory leaks
    /// - CPU data remains intact after disposal
    ///
    /// Example:
    /// <code>
    /// using (var node = GpuComputationNode.Create(tensor, context))
    /// {
    ///     // Use the node
    /// } // Automatically disposed here
    /// </code>
    /// </para>
    /// </remarks>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        GpuValue?.Dispose();
        GpuValue = null;

        GpuGradient?.Dispose();
        GpuGradient = null;

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer to ensure GPU memory is freed.
    /// </summary>
    ~GpuComputationNode()
    {
        Dispose();
    }

    /// <summary>
    /// Gets a string representation including GPU status.
    /// </summary>
    public override string ToString()
    {
        var location = IsOnGpu ? "GPU" : "CPU";
        var gradLocation = IsGradientOnGpu ? "GPU" : "CPU";
        var name = string.IsNullOrEmpty(Name) ? "Unnamed" : Name;
        return $"GpuComputationNode '{name}' [{string.Join("x", Value.Shape)}] " +
               $"Value@{location}, Gradient@{gradLocation}, RequiresGrad={RequiresGradient}";
    }
}
