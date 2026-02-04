using AiDotNet.LinearAlgebra;

namespace AiDotNet.MixedPrecision;

/// <summary>
/// Provides an ambient context for mixed-precision operations during forward and backward passes.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MixedPrecisionScope is like a "mode switch" that tells the neural network
/// to use lower precision (FP16) for faster computation while keeping certain operations in full precision (FP32)
/// for numerical stability.
///
/// The scope is automatically managed by the training loop - you don't need to create it yourself.
/// When inside a scope:
/// - Most operations use FP16 (faster, less memory)
/// - Certain layers (like LayerNorm, Softmax) stay in FP32 for stability
/// - The scope tracks both FP16 and FP32 versions of tensors so layers can access what they need
/// </para>
/// <para><b>Technical Details:</b> The scope uses a thread-static pattern to provide ambient context.
/// Layers can check <see cref="Current"/> to determine if mixed-precision is active and whether
/// they should use FP16 or FP32 based on the <see cref="LayerPrecisionPolicy"/>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Automatic usage (handled by MixedPrecisionTrainingLoop)
/// // Users typically don't need to create scopes manually
///
/// // For advanced manual usage:
/// using (var scope = new MixedPrecisionScope(context, policy))
/// {
///     // Forward pass happens here
///     var output = network.Forward(input);
///     // Layers check MixedPrecisionScope.Current to determine precision
/// }
/// </code>
/// </example>
public class MixedPrecisionScope : IDisposable
{
    /// <summary>
    /// Thread-static storage for the current scope, enabling ambient context access.
    /// </summary>
    [ThreadStatic]
    private static MixedPrecisionScope? _current;

    /// <summary>
    /// Gets the currently active mixed-precision scope, or null if not in a scope.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Layers check this property to see if mixed-precision is active.
    /// If it's null, the layer operates normally. If it's set, the layer may need to adjust its precision.
    /// </para>
    /// </remarks>
    public static MixedPrecisionScope? Current => _current;

    /// <summary>
    /// Sets the current scope (used by constructor and Dispose).
    /// This static method manages the thread-static ambient context.
    /// </summary>
    private static void SetCurrentScope(MixedPrecisionScope? scope) => _current = scope;

    /// <summary>
    /// The previous scope (for nested scope support).
    /// </summary>
    private readonly MixedPrecisionScope? _previous;

    /// <summary>
    /// The mixed-precision context providing loss scaling and weight management.
    /// </summary>
    private readonly MixedPrecisionContext _context;

    /// <summary>
    /// The policy determining which layers use which precision.
    /// </summary>
    private readonly LayerPrecisionPolicy _policy;

    /// <summary>
    /// Storage for FP32 versions of tensors (for layers that need full precision).
    /// </summary>
    private readonly Dictionary<string, Tensor<float>> _fp32Tensors;

    /// <summary>
    /// Storage for FP16 (Half) versions of tensors.
    /// </summary>
    private readonly Dictionary<string, Tensor<Half>> _fp16Tensors;

    /// <summary>
    /// Whether the scope has been disposed.
    /// </summary>
    private bool _disposed;

    /// <summary>
    /// Gets the mixed-precision context associated with this scope.
    /// </summary>
    public MixedPrecisionContext Context => _context;

    /// <summary>
    /// Gets the layer precision policy for this scope.
    /// </summary>
    public LayerPrecisionPolicy Policy => _policy;

    /// <summary>
    /// Gets whether this scope is currently active (is the current scope).
    /// </summary>
    public bool IsActive => _current == this;

    /// <summary>
    /// Creates a new mixed-precision scope.
    /// </summary>
    /// <param name="context">The mixed-precision context for weight and scale management.</param>
    /// <param name="policy">The policy determining layer-level precision (optional, uses default FP16 policy if null).</param>
    /// <exception cref="ArgumentNullException">Thrown when context is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a scope switches the neural network into mixed-precision mode.
    /// The scope should be disposed (via using block) when the forward pass is complete.
    /// </para>
    /// </remarks>
    public MixedPrecisionScope(MixedPrecisionContext context, LayerPrecisionPolicy? policy = null)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _policy = policy ?? LayerPrecisionPolicy.ForFP16();
        _fp32Tensors = new Dictionary<string, Tensor<float>>();
        _fp16Tensors = new Dictionary<string, Tensor<Half>>();
        _disposed = false;

        // Save previous scope and set this as current
        _previous = Current;
        SetCurrentScope(this);
    }

    /// <summary>
    /// Determines if a layer should use full precision (FP32) based on the policy.
    /// </summary>
    /// <param name="layerName">The name of the layer to check.</param>
    /// <returns>True if the layer should use FP32; false if it can use FP16.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some layers need higher precision to work correctly.
    /// This method checks the policy to see if a specific layer is one of them.
    ///
    /// Layers that typically need FP32:
    /// - LayerNorm, BatchNorm (compute mean/variance)
    /// - Softmax (involves exponentials)
    /// - Loss computation
    /// </para>
    /// </remarks>
    public bool ShouldUseFP32(string layerName)
    {
        return _policy.ShouldSkipMixedPrecision(layerName);
    }

    /// <summary>
    /// Determines if a layer should use higher precision than the default (FP32 or FP16 when default is FP8).
    /// </summary>
    /// <param name="layerName">The name of the layer to check.</param>
    /// <returns>True if the layer should use higher precision than the default.</returns>
    public bool ShouldUseHigherPrecision(string layerName)
    {
        return _policy.ShouldUseHigherPrecision(layerName);
    }

    /// <summary>
    /// Gets the precision type to use for a specific layer.
    /// </summary>
    /// <param name="layerName">The name of the layer.</param>
    /// <returns>The precision type the layer should use.</returns>
    public Enums.MixedPrecisionType GetLayerPrecision(string layerName)
    {
        return _policy.GetPrecision(layerName);
    }

    /// <summary>
    /// Registers an FP32 tensor and returns its FP16 equivalent.
    /// </summary>
    /// <param name="name">A unique name for the tensor (e.g., "input", "layer1_output").</param>
    /// <param name="fp32Tensor">The original FP32 tensor.</param>
    /// <returns>The FP16 version of the tensor for computation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method stores the original FP32 tensor and creates an FP16 copy.
    /// Layers that need FP32 can later retrieve the original using <see cref="GetFP32Tensor"/>.
    /// </para>
    /// </remarks>
    public Tensor<Half> RegisterAndCastToFP16(string name, Tensor<float> fp32Tensor)
    {
        if (fp32Tensor == null)
        {
            throw new ArgumentNullException(nameof(fp32Tensor));
        }

        // Store the FP32 version
        _fp32Tensors[name] = fp32Tensor;

        // Create and store the FP16 version
        var fp16Data = new Half[fp32Tensor.Length];
        for (int i = 0; i < fp32Tensor.Length; i++)
        {
            fp16Data[i] = (Half)fp32Tensor.GetFlatIndexValue(i);
        }

        var fp16Tensor = new Tensor<Half>(fp32Tensor.Shape, new Vector<Half>(fp16Data));
        _fp16Tensors[name] = fp16Tensor;

        return fp16Tensor;
    }

    /// <summary>
    /// Registers an FP32 tensor for tracking without creating an FP16 copy.
    /// </summary>
    /// <param name="name">A unique name to identify this tensor in the scope.</param>
    /// <param name="fp32Tensor">The FP32 tensor to register.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this method for layers that require FP32 precision
    /// (like BatchNorm, LayerNorm). It stores the tensor for tracking purposes without
    /// the overhead of creating an unnecessary FP16 copy.
    /// </para>
    /// </remarks>
    public void RegisterFP32Only(string name, Tensor<float> fp32Tensor)
    {
        if (fp32Tensor == null)
        {
            throw new ArgumentNullException(nameof(fp32Tensor));
        }

        // Store only the FP32 version - no FP16 copy needed for FP32-only layers
        _fp32Tensors[name] = fp32Tensor;
    }

    /// <summary>
    /// Retrieves the FP32 version of a previously registered tensor.
    /// </summary>
    /// <param name="name">The name used when registering the tensor.</param>
    /// <returns>The FP32 tensor, or null if not found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Layers that need full precision (like LayerNorm) use this
    /// to get the FP32 version of tensors, even when the network is operating in FP16 mode.
    /// </para>
    /// </remarks>
    public Tensor<float>? GetFP32Tensor(string name)
    {
        return _fp32Tensors.TryGetValue(name, out var tensor) ? tensor : null;
    }

    /// <summary>
    /// Retrieves the FP16 version of a previously registered tensor.
    /// </summary>
    /// <param name="name">The name used when registering the tensor.</param>
    /// <returns>The FP16 tensor, or null if not found.</returns>
    public Tensor<Half>? GetFP16Tensor(string name)
    {
        return _fp16Tensors.TryGetValue(name, out var tensor) ? tensor : null;
    }

    /// <summary>
    /// Checks if a tensor with the given name has been registered.
    /// </summary>
    /// <param name="name">The tensor name to check.</param>
    /// <returns>True if the tensor is registered; otherwise, false.</returns>
    public bool HasTensor(string name)
    {
        return _fp32Tensors.ContainsKey(name);
    }

    /// <summary>
    /// Converts an FP16 tensor back to FP32.
    /// </summary>
    /// <param name="fp16Tensor">The FP16 tensor to convert.</param>
    /// <returns>The FP32 version of the tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you need to convert intermediate results
    /// back to FP32, for example when passing data to a layer that requires full precision.
    /// </para>
    /// </remarks>
    public static Tensor<float> CastToFP32(Tensor<Half> fp16Tensor)
    {
        if (fp16Tensor == null)
        {
            throw new ArgumentNullException(nameof(fp16Tensor));
        }

        var fp32Data = new float[fp16Tensor.Length];
        for (int i = 0; i < fp16Tensor.Length; i++)
        {
            fp32Data[i] = (float)fp16Tensor.GetFlatIndexValue(i);
        }

        return new Tensor<float>(fp16Tensor.Shape, new Vector<float>(fp32Data));
    }

    /// <summary>
    /// Converts an FP32 tensor to FP16.
    /// </summary>
    /// <param name="fp32Tensor">The FP32 tensor to convert.</param>
    /// <returns>The FP16 version of the tensor.</returns>
    public static Tensor<Half> CastToFP16(Tensor<float> fp32Tensor)
    {
        if (fp32Tensor == null)
        {
            throw new ArgumentNullException(nameof(fp32Tensor));
        }

        var fp16Data = new Half[fp32Tensor.Length];
        for (int i = 0; i < fp32Tensor.Length; i++)
        {
            fp16Data[i] = (Half)fp32Tensor.GetFlatIndexValue(i);
        }

        return new Tensor<Half>(fp32Tensor.Shape, new Vector<Half>(fp16Data));
    }

    /// <summary>
    /// Clears all registered tensors to free memory.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this between training steps to release memory
    /// used by intermediate tensors. This is called automatically on Dispose.
    /// </para>
    /// </remarks>
    public void ClearTensors()
    {
        _fp32Tensors.Clear();
        _fp16Tensors.Clear();
    }

    /// <summary>
    /// Gets the number of registered tensors.
    /// </summary>
    public int RegisteredTensorCount => _fp32Tensors.Count;

    /// <summary>
    /// Disposes the scope and restores the previous scope (if any).
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // Clear tensor storage
        _fp32Tensors.Clear();
        _fp16Tensors.Clear();

        // Restore previous scope
        SetCurrentScope(_previous);

        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Gets a string representation of the scope's current state.
    /// </summary>
    public override string ToString()
    {
        return $"MixedPrecisionScope: Active={IsActive}, " +
               $"Policy={_policy.DefaultPrecision}, " +
               $"RegisteredTensors={RegisteredTensorCount}, " +
               $"LossScale={_context.LossScaler.Scale:F0}";
    }
}
