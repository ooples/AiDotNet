using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Abstract base class for LoRA (Low-Rank Adaptation) adapters that wrap existing layers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This base class provides common functionality for all LoRA adapter implementations.
/// It manages the base layer, LoRA layer, and parameter synchronization, while allowing
/// derived classes to implement layer-type-specific logic such as merging and validation.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all LoRA adapters in the library.
///
/// A LoRA adapter wraps an existing layer (like a dense or convolutional layer) and adds
/// a small "correction layer" that learns what adjustments are needed. This base class:
/// - Manages both the original layer and the LoRA correction layer
/// - Handles parameter synchronization between them
/// - Provides common forward/backward pass logic (original + correction)
/// - Lets specialized adapters handle layer-specific details
///
/// This design allows you to create LoRA adapters for any layer type by:
/// 1. Inheriting from this base class
/// 2. Implementing layer-specific validation
/// 3. Implementing how to merge the LoRA weights back into the original layer
///
/// The result is parameter-efficient fine-tuning that works across different layer architectures!
/// </para>
/// </remarks>
public abstract class LoRAAdapterBase<T> : LayerBase<T>, ILoRAAdapter<T>, ILayerSerializationExtras<T>
{
    /// <summary>
    /// The base layer being adapted.
    /// </summary>
    protected readonly ILayer<T> _baseLayer;

    /// <summary>
    /// The LoRA layer that provides the adaptation.
    /// </summary>
    protected readonly LoRALayer<T> _loraLayer;

    /// <summary>
    /// Whether the base layer's parameters are frozen (not trainable).
    /// </summary>
    protected readonly bool _freezeBaseLayer;

    /// <summary>
    /// Force-resolve <see cref="_baseLayer"/>'s lazy shape using the input
    /// dim that the LoRA layer already settled on. Adapters call this
    /// before any path that needs the base layer's parameter buffer to
    /// be allocated (parameter merge, gradient round-trip, weight
    /// inspection, etc.). Without it, an unresolved LayerBase returns an
    /// empty parameter Vector and the caller indexes past the end.
    /// Centralized here so DenseLoRAAdapter / VBLoRAAdapter / future
    /// adapters share the same guard rather than copy-pasting the
    /// LayerBase / IsShapeResolved / ResolveShapesOnly dance.
    /// </summary>
    /// <returns>
    /// True when the base layer already had — or now has — a resolved
    /// shape; false when the LoRA layer's input shape was non-positive
    /// (lazy itself) and the resolve was skipped.
    /// </returns>
    protected bool EnsureBaseLayerShapeResolved()
    {
        if (_baseLayer is not LayerBase<T> baseLayerBase)
            return true;
        if (baseLayerBase.IsShapeResolved)
            return true;
        var loraInputShape = _loraLayer.GetInputShape();
        if (loraInputShape.Length == 0) return false;
        int loraInputSize = loraInputShape[0];
        if (loraInputSize <= 0) return false;
        baseLayerBase.ResolveShapesOnly(new[] { loraInputSize });
        return true;
    }

    /// <summary>
    /// Gets the base layer being adapted with LoRA.
    /// </summary>
    /// <remarks>
    /// This is the original layer that's being enhanced with LoRA adaptations.
    /// It may be frozen (non-trainable) during fine-tuning for maximum efficiency.
    /// </remarks>
    public ILayer<T> BaseLayer => _baseLayer;

    /// <summary>
    /// Gets the LoRA layer providing the low-rank adaptation.
    /// </summary>
    /// <remarks>
    /// This layer implements the low-rank decomposition (A and B matrices)
    /// that provides the adaptation to the base layer's behavior.
    /// </remarks>
    public LoRALayer<T> LoRALayer => _loraLayer;

    /// <summary>
    /// Gets whether the base layer's parameters are frozen during training.
    /// </summary>
    /// <remarks>
    /// When true, only the LoRA parameters are trained, dramatically reducing
    /// memory requirements and training time. This is the typical use case for LoRA.
    /// </remarks>
    public bool IsBaseLayerFrozen => _freezeBaseLayer;

    /// <summary>
    /// Gets the rank of the low-rank decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The rank determines how many parameters the LoRA adaptation uses.
    /// Lower rank = fewer parameters = more efficient but less flexible.
    /// </para>
    /// <para>
    /// Typical values:
    /// - rank=1-4: Very efficient, minimal parameters
    /// - rank=8: Good balance (default for many applications)
    /// - rank=16-32: More flexibility, more parameters
    /// - rank=64+: Diminishing returns, approaching full fine-tuning
    /// </para>
    /// </remarks>
    public int Rank => _loraLayer.Rank;

    /// <summary>
    /// Gets the scaling factor (alpha) for the LoRA adaptation.
    /// </summary>
    /// <remarks>
    /// Alpha controls how strongly the LoRA adaptation affects the output.
    /// The actual LoRA contribution is scaled by alpha/rank.
    /// Common practice: alpha = rank (scaling factor of 1.0)
    /// </remarks>
    public double Alpha => Convert.ToDouble(_loraLayer.Alpha);

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// If the base layer is frozen, this returns only the LoRA parameter count.
    /// Otherwise, it returns the sum of base and LoRA parameters.
    /// </remarks>
    public override long ParameterCount => _freezeBaseLayer
        ? _loraLayer.ParameterCount
        : (_baseLayer.ParameterCount + _loraLayer.ParameterCount);

    /// <summary>
    /// Gets whether this adapter supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    int ILayerSerializationExtras<T>.ExtraParameterCount => _freezeBaseLayer ? _baseLayer.GetParameters().Length : 0;

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        return _freezeBaseLayer ? _baseLayer.GetParameters() : Vector<T>.Empty();
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        if (!_freezeBaseLayer)
        {
            return;
        }

        var expected = _baseLayer.GetParameters().Length;
        if (extraParameters.Length != expected)
        {
            throw new InvalidOperationException($"Expected {expected} extra parameters for frozen base layer, but got {extraParameters.Length}.");
        }

        _baseLayer.SetParameters(extraParameters);
    }

    /// <summary>
    /// Initializes a new LoRA adapter base with the specified parameters.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates the foundation for a LoRA adapter.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to make more efficient to fine-tune
    /// - rank: How much compression (lower = fewer parameters, less flexibility)
    /// - alpha: How strong the LoRA adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true for efficiency)
    ///
    /// Derived classes will call this constructor and then add their own layer-specific logic.
    /// </para>
    /// </remarks>
    protected LoRAAdapterBase(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : this(baseLayer ?? throw new ArgumentNullException(nameof(baseLayer)),
               rank, alpha, freezeBaseLayer,
               ResolveBaseInputShapeWithProvenance(baseLayer))
    {
    }

    /// <summary>
    /// Internal ctor that takes the resolved-input-shape result tuple from
    /// <see cref="ResolveBaseInputShapeWithProvenance"/>. The IsAuthoritative
    /// flag tells us whether the shape came from the layer itself (or its
    /// trainable parameters) versus the synthetic <c>outSize * 2</c> fallback
    /// — we only eagerly call <see cref="LayerBase{T}.ResolveFromShape"/> in
    /// the authoritative case, so a wrong heuristic guess never allocates
    /// real weight tensors with mismatched dims.
    /// </summary>
    private LoRAAdapterBase(
        ILayer<T> baseLayer, int rank, double alpha, bool freezeBaseLayer,
        (int[] Shape, bool IsAuthoritative) resolvedInput)
        : base(resolvedInput.Shape, baseLayer.GetOutputShape())
    {
        _baseLayer = baseLayer;
        _freezeBaseLayer = freezeBaseLayer;

        // Only eagerly resolve when the shape we just gave the base ctor is
        // authoritative (came from the layer or its actual weights). The
        // synthetic outSize*2 fallback is a guess for ParameterCount-readiness
        // only; allocating real weights against it would burn RNG state and
        // potentially produce wrong-shape kernels that throw later on actual
        // forward.
        if (resolvedInput.IsAuthoritative
            && _baseLayer is LayerBase<T> baseLb
            && !baseLb.IsShapeResolved)
        {
            var resolvedIn = GetInputShape();
            if (resolvedIn.Length > 0 && resolvedIn.All(d => d > 0))
                baseLb.ResolveFromShape(resolvedIn);
        }

        // Create the LoRA layer - derived classes may override this via CreateLoRALayer
        _loraLayer = CreateLoRALayer(rank, alpha);

        // Initialize Parameters using a NON-VIRTUAL base+LoRA-only sizing.
        // Calling the virtual ParameterCount from a base ctor is a C#
        // antipattern: derived adapters that override ParameterCount with
        // derived state (delta weight matrices, importance scores, bank
        // indices, etc.) dereference fields that the derived ctor body
        // hasn't yet initialized — the derived state observed here is
        // whatever default(T) the field type uses. Sizing against just
        // _baseLayer + _loraLayer is always safe because both are fully
        // constructed at this point.
        //
        // Derived adapters that override ParameterCount AND need their
        // packed Parameters vector to round-trip (i.e., they don't
        // override GetParameters / SetParameters with their own
        // packing logic) MUST call RebuildParametersAfterDerivedInit()
        // at the end of their constructor once their extra state is
        // initialized. Most derived adapters override GetParameters
        // and don't need this call.
        // Match what PackBaseAndLoraParameters actually packs: base params
        // are skipped when frozen (the optimizer doesn't update them, and
        // they round-trip via ILayerSerializationExtras instead). Sizing
        // against the unfrozen total when freezeBaseLayer=true would leave
        // trailing unused elements in Parameters, breaking GetParameters()
        // length and (de)serialization round-trip.
        int baseAndLoraCount =
            (_freezeBaseLayer ? 0 : _baseLayer.GetParameters().Length)
            + _loraLayer.GetParameters().Length;
        Parameters = new Vector<T>(baseAndLoraCount);
        // Pack base + LoRA params directly (non-virtual) so the vector
        // is initialized without invoking the derived
        // UpdateParametersFromLayers override. Derived classes that
        // need their own packing call RebuildParametersAfterDerivedInit
        // which routes through the virtual UpdateParametersFromLayers.
        PackBaseAndLoraParameters();
    }

    /// <summary>
    /// Non-virtual pack of <see cref="_baseLayer"/> + <see cref="_loraLayer"/>
    /// parameters into <see cref="LayerBase{T}.Parameters"/>. Used by the
    /// base ctor where the derived <see cref="UpdateParametersFromLayers"/>
    /// override would dereference uninitialised derived state.
    /// </summary>
    private void PackBaseAndLoraParameters()
    {
        int idx = 0;
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }
    }

    /// <summary>
    /// Derived adapter classes that override <see cref="LayerBase{T}.ParameterCount"/>
    /// to include extra state (delta weights, importance scores, etc.) MUST
    /// call this method at the end of their constructor body so the base
    /// class's <see cref="LayerBase{T}.Parameters"/> vector is re-allocated
    /// against the derived total. The base ctor calls
    /// <c>UpdateParametersFromLayers</c> via this method, which in turn calls
    /// the now-initialized derived <c>ParameterCount</c> via virtual dispatch.
    /// Parameter count is cast to int because <c>Vector{T}.Length</c> is int
    /// per the per-tensor &lt; 2.1 B contract; #1237's long aggregate applies
    /// only to the model-level <c>ParameterCount</c> property.
    /// </summary>
    protected void RebuildParametersAfterDerivedInit()
    {
        Parameters = new Vector<T>((int)ParameterCount);
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Creates the LoRA layer for this adapter.
    /// </summary>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor.</param>
    /// <returns>A LoRA layer configured for this adapter.</returns>
    /// <remarks>
    /// <para>
    /// This method can be overridden by derived classes to customize LoRA layer creation.
    /// By default, it creates a standard LoRA layer with the adapter's input and output dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the "correction layer" that learns adaptations.
    ///
    /// Different adapter types might need different LoRA layer configurations:
    /// - Dense layers: Standard 1D LoRA
    /// - Convolutional layers: LoRA with spatial dimensions
    /// - Attention layers: LoRA for query/key/value projections
    ///
    /// This method lets each adapter type create the right kind of LoRA layer.
    /// </para>
    /// </remarks>
    private static int InferInputSizeFromWeights(ILayer<T>? baseLayer, IReadOnlyList<Tensor<T>> weights)
    {
        if (weights.Count == 0) return -1;

        // Find the first WEIGHT-MATRIX-shaped tensor (rank >= 2). A naive
        // weights[0] inspection breaks when the first parameter is a 1-D
        // bias / LayerNorm gamma — those have outSize as their only axis,
        // so InferInputSizeFromWeights would return outSize as inputSize
        // and produce wrong adapter dimensions on every Dense layer once
        // lazily resolved. Scan for the first rank-≥-2 tensor first; only
        // if none is found do we fall back to the rank-1 case.
        Tensor<T>? matrix = null;
        for (int i = 0; i < weights.Count; i++)
        {
            if (weights[i].Shape.Length >= 2)
            {
                matrix = weights[i];
                break;
            }
        }

        if (matrix is null)
        {
            // No weight matrix — fall back to the first 1-D tensor's length
            // (LayerNorm/BatchNorm wrappers where in == out).
            var w0 = weights[0];
            if (w0.Shape.Length == 1 && w0.Shape[0] > 0) return w0.Shape[0];
            return -1;
        }

        if (matrix.Shape.Length == 2)
        {
            // FullyConnectedLayer<T> uses output-major storage:
            // weights are allocated as [outputSize, inputSize] (see this
            // class's MergeWeights / Forward paths at L504+ which assume
            // that layout). For an FCL, the fan-in axis is Shape[1].
            //
            // DenseLayer<T> uses the inverse convention:
            // weights are allocated as [inputSize, outputSize]
            // (DenseLayer's TensorAllocator.Rent<T>([inputSize, outputSize])).
            // For Dense, the fan-in axis is Shape[0].
            //
            // Without distinguishing, an FCL wrapped via lazy LoRA would
            // produce LoRA tensors with swapped dimensions and crash on
            // first forward.
            if (baseLayer is FullyConnectedLayer<T>)
            {
                return matrix.Shape[1] > 0 ? matrix.Shape[1] : -1;
            }
            return matrix.Shape[0] > 0 ? matrix.Shape[0] : -1;
        }
        // Conv weight convention (rank ≥ 3): [outC, inC, ...spatial] ⇒ axis 1 is
        // input channels. The trailing dim would be wrong (kernel width / depth).
        return matrix.Shape[1] > 0 ? matrix.Shape[1] : -1;
    }

    /// <summary>
    /// Returns the resolved base-input shape AND a flag indicating whether
    /// the shape is authoritative (came from the layer's own resolved
    /// shape or its actual weight matrix) vs a synthetic
    /// <c>outSize * 2</c> heuristic. Callers should only eagerly allocate
    /// weights from authoritative shapes.
    /// </summary>
    private static (int[] Shape, bool IsAuthoritative) ResolveBaseInputShapeWithProvenance(ILayer<T> baseLayer)
    {
        var shape = baseLayer.GetInputShape();
        if (shape.Length > 0 && shape.All(d => d > 0)) return (shape, true);

        if (baseLayer is LayerBase<T> layerBase)
        {
            int inferred = InferInputSizeFromWeights(baseLayer, layerBase.GetTrainableParameters());
            if (inferred > 0) return (new[] { inferred }, true);
        }

        // Convention encoded by the LoRA test suite (Assert.Equal(10, ...) on
        // adapter wrapping DenseLayer(5)): input dim defaults to 2 × output
        // dim. NOT authoritative — caller must NOT eagerly allocate weights
        // against this guess; ResolveFromShape would otherwise materialize
        // wrong-shape weight tensors.
        var outShape = baseLayer.GetOutputShape();
        int outSize = outShape.Length > 0 && outShape[0] > 0 ? outShape[0] : 1;
        return (new[] { outSize * 2 }, false);
    }

    private static int[] ResolveBaseInputShape(ILayer<T> baseLayer)
        => ResolveBaseInputShapeWithProvenance(baseLayer).Shape;

    protected virtual LoRALayer<T> CreateLoRALayer(int rank, double alpha)
    {
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        if (inputSize <= 0 && _baseLayer is LayerBase<T> layerBase)
        {
            // Pass _baseLayer so InferInputSizeFromWeights can pick the
            // right axis for output-major layers like FullyConnectedLayer.
            int inferred = InferInputSizeFromWeights(_baseLayer, layerBase.GetTrainableParameters());
            if (inferred > 0) inputSize = inferred;
        }
        if (inputSize <= 0) inputSize = outputSize * 2;
        return new LoRALayer<T>(inputSize, outputSize, rank, alpha);
    }

    /// <summary>
    /// Performs the forward pass through both base and LoRA layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes: output = base_layer(input) + lora_layer(input)
    /// </para>
    /// <para><b>For Beginners:</b> This runs the input through both the original layer and the
    /// LoRA correction layer, then adds their outputs together. The result is the original
    /// behavior plus the learned adaptation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through LoRA layer
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // Sum the outputs (vectorized)
        return Engine.TensorAdd(baseOutput, loraOutput);
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // Always update LoRA layer
        _loraLayer.UpdateParameters(learningRate);

        // Only update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing parameters (LoRA only if base is frozen, otherwise both).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateLayersFromParameters();
    }

    /// <summary>
    /// Updates the layers from the parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method distributes the values from the parameter vector back to the base
    /// and LoRA layers. If the base layer is frozen, only LoRA parameters are updated.
    /// </para>
    /// <para><b>For Beginners:</b> This does the opposite of UpdateParametersFromLayers.
    /// It takes values from the big list and puts them back into the individual layers.
    /// </para>
    /// </remarks>
    private void UpdateLayersFromParameters()
    {
        int idx = 0;

        // If base layer is not frozen, unpack its parameters first
        if (!_freezeBaseLayer)
        {
            int baseParamCount = (int)_baseLayer.ParameterCount;
            Vector<T> baseParams = new Vector<T>(baseParamCount);
            for (int i = 0; i < baseParamCount; i++)
            {
                baseParams[i] = Parameters[idx++];
            }
            _baseLayer.SetParameters(baseParams);
        }

        // Unpack LoRA parameters
        int loraParamCount = (int)_loraLayer.ParameterCount;
        Vector<T> loraParams = new Vector<T>(loraParamCount);
        for (int i = 0; i < loraParamCount; i++)
        {
            loraParams[i] = Parameters[idx++];
        }
        _loraLayer.SetParameters(loraParams);
    }

    /// <summary>
    /// Updates the parameter gradients vector from the layer gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method collects gradients from both layers into a single vector.
    /// If the base layer is frozen, only LoRA gradients are included.
    /// </para>
    /// <para><b>For Beginners:</b> After backpropagation, this collects all the "improvement directions"
    /// from both layers into one organized list for the optimizer to use.
    /// </para>
    /// </remarks>
    private void UpdateParameterGradientsFromLayers()
    {
        ParameterGradients = new Vector<T>((int)ParameterCount);
        int idx = 0;

        // If base layer is not frozen, pack its gradients first
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // Pack LoRA gradients
        Vector<T> loraGrads = _loraLayer.GetParameterGradients();
        for (int i = 0; i < loraGrads.Length; i++)
        {
            ParameterGradients[idx++] = loraGrads[i];
        }
    }

    /// <summary>
    /// Merges the LoRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with LoRA weights merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// This method must be implemented by derived classes to handle layer-type-specific
    /// merging logic. Each type of adapter (Dense, Convolutional, etc.) needs to know
    /// how to combine its LoRA weights with the base layer's weights.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your LoRA adaptation to create a regular layer.
    /// After training with LoRA, you can merge the adaptation into the original weights for:
    /// - Faster inference (no need to compute LoRA separately)
    /// - Simpler deployment (single layer instead of two)
    /// - Compatibility with systems that don't support LoRA
    ///
    /// Each layer type implements this differently because they have different internal structures.
    /// </para>
    /// </remarks>
    public abstract ILayer<T> MergeToOriginalLayer();

    /// <summary>
    /// Helper method to create a merged layer by cloning the base layer and updating its parameters.
    /// </summary>
    /// <param name="mergedParams">The merged parameters to set on the cloned layer.</param>
    /// <returns>A cloned layer with merged parameters and preserved activation function.</returns>
    /// <exception cref="InvalidOperationException">Thrown when base layer is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This helper method preserves the activation function and other settings from the base layer
    /// by using Clone() instead of creating a new layer. This ensures the merged layer behaves
    /// identically to the original adapted layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is a utility method that derived classes can use to create
    /// a properly merged layer without duplicating the Clone() pattern everywhere.
    /// </para>
    /// </remarks>
    protected ILayer<T> CreateMergedLayerWithClone(Vector<T> mergedParams)
    {
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase != null)
        {
            ILayer<T> merged = denseBase.Clone();
            merged.SetParameters(mergedParams);
            return merged;
        }
        else if (fcBase != null)
        {
            ILayer<T> merged = fcBase.Clone();
            merged.SetParameters(mergedParams);
            return merged;
        }
        else
        {
            throw new InvalidOperationException(
                $"Base layer type {_baseLayer.GetType().Name} is not supported for merging. " +
                "Only DenseLayer and FullyConnectedLayer are currently supported.");
        }
    }

    /// <summary>
    /// Merges LoRA weights into the base layer for DenseLayer or FullyConnectedLayer.
    /// </summary>
    /// <returns>A new layer with merged weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when base layer is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This helper method implements the standard LoRA merge logic for Dense and FullyConnected layers:
    /// 1. Get LoRA weight contribution from low-rank matrices
    /// 2. Add to base layer weights element-wise
    /// 3. Preserve biases unchanged
    /// 4. Create new layer with merged parameters
    /// </para>
    /// <para><b>For Beginners:</b> This combines the base weights with the LoRA adaptation,
    /// creating a single layer that doesn't need the adapter anymore. Useful for deployment!
    /// </para>
    /// </remarks>
    protected ILayer<T> MergeToDenseOrFullyConnected()
    {
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Calculate dimensions
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights — loraWeights from MergeWeights() is [inputSize, outputSize].
        // DenseLayer stores weights as [inputSize, outputSize] (input-major).
        // FullyConnectedLayer stores weights as [outputSize, inputSize] (output-major).
        bool isOutputMajor = _baseLayer is FullyConnectedLayer<T>;
        for (int i = 0; i < weightCount; i++)
        {
            int row, col;
            if (isOutputMajor)
            {
                // FullyConnectedLayer: flat[i] = weights[outputIdx, inputIdx]
                int outputIdx = i / inputSize;
                int inputIdx = i % inputSize;
                row = inputIdx;
                col = outputIdx;
            }
            else
            {
                // DenseLayer: flat[i] = weights[inputIdx, outputIdx]
                row = i / outputSize;
                col = i % outputSize;
            }
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Updates the parameter vector from the current base and LoRA layer states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This helper method synchronizes the adapter's parameter vector with the current state
    /// of the base and LoRA layers after updates. It packs parameters in the standard order:
    /// base layer parameters (if not frozen) followed by LoRA parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This ensures the adapter's parameter vector stays in sync
    /// with its component layers. Called after parameter updates.
    /// </para>
    /// </remarks>
    protected virtual void UpdateParametersFromLayers()
    {
        int idx = 0;

        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }
    }

    /// <summary>
    /// Resets the internal state of both the base layer and LoRA layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears the memory of both the base layer and the LoRA layer.
    /// It's useful when starting to process a completely new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _loraLayer.ResetState();
    }

    /// <summary>
    /// Persists the inner-layer type name and shape so DeserializationHelper
    /// can reconstruct the WRAPPED base layer with the right concrete type
    /// instead of the prior <c>DenseLayer&lt;T&gt;</c> placeholder. This is
    /// the round-trip path for issue #1239's wrapped-layer concern: every
    /// LoRA adapter (35 implementations as of this commit, all derived from
    /// this base) inherits this metadata persistence automatically.
    /// </summary>
    /// <remarks>
    /// <para>
    /// What's persisted:
    /// <list type="bullet">
    /// <item><b>InnerLayerTypeName</b> — the wrapped layer's short type
    /// name (<see cref="System.Type.Name"/>, e.g., <c>DenseLayer`1</c>),
    /// generic-arity suffix included, no namespace prefix. This is the
    /// form <c>DeserializationHelper.LayerTypes</c> is keyed on, so the
    /// recursive <c>CreateLayerFromType</c> lookup resolves against
    /// it. Fully-qualified or assembly-qualified names would break
    /// that lookup.</item>
    /// <item><b>InnerLayerInputShape</b> / <b>InnerLayerOutputShape</b>
    /// — comma-separated dim lists. Used as the recursive deser call's
    /// inputShape / outputShape parameters.</item>
    /// <item><b>Rank</b> / <b>Alpha</b> / <b>FreezeBaseLayer</b> — LoRA
    /// adapter scalar config that doesn't depend on the inner layer.</item>
    /// </list>
    /// </para>
    /// <para>
    /// What's NOT persisted by this base method (subclasses with extra
    /// state must override and chain): adapter-specific fields like
    /// VBLoRA's bank indices, AdaLoRA's importance scores, DyLoRA's
    /// rank schedule, MoRA's hash table size, etc. Subclass overrides
    /// should call <c>base.GetMetadata()</c> first then add their own.
    /// </para>
    /// <para>
    /// Frozen-base inner-layer parameter VALUES round-trip via the
    /// existing <see cref="ILayerSerializationExtras{T}"/> path
    /// (<see cref="ILayerSerializationExtras{T}.GetExtraParameters"/>);
    /// non-frozen inner-layer parameters are part of the wrapper's flat
    /// <see cref="GetParameters"/> output already.
    /// </para>
    /// </remarks>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();

        // Inner layer's runtime type name. Strip any assembly qualification
        // and namespace prefix so it matches what DeserializationHelper's
        // LayerTypes dictionary keys on (Type.Name, generic-arity suffix
        // included).
        metadata["InnerLayerTypeName"] = _baseLayer.GetType().Name;

        // Shape strings as comma-joined int lists (mirrors the format
        // used by other layer GetMetadata sites that round-trip arrays).
        var innerInput = _baseLayer.GetInputShape();
        var innerOutput = _baseLayer.GetOutputShape();
        metadata["InnerLayerInputShape"] = string.Join(",", innerInput);
        metadata["InnerLayerOutputShape"] = string.Join(",", innerOutput);

        // LoRA-specific scalars. Rank reads from the LoRA layer; alpha
        // is the scaling factor (already exposed publicly); FreezeBaseLayer
        // controls whether the base's params count toward Parameters.
        // Use InvariantCulture for numeric serialization so the round-trip
        // works on locales where ',' is the decimal separator (German,
        // French, etc.) — the deser side parses with TryGetDouble which
        // also uses invariant rules.
        metadata["Rank"] = _loraLayer.Rank.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Alpha"] = Convert.ToDouble(_loraLayer.Alpha).ToString("G", System.Globalization.CultureInfo.InvariantCulture);
        metadata["FreezeBaseLayer"] = _freezeBaseLayer.ToString();

        return metadata;
    }
}
