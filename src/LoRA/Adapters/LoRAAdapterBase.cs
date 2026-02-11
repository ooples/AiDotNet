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
    public override int ParameterCount => _freezeBaseLayer
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
        : base(
            (baseLayer ?? throw new ArgumentNullException(nameof(baseLayer))).GetInputShape(),
            baseLayer.GetOutputShape())
    {
        _baseLayer = baseLayer;
        _freezeBaseLayer = freezeBaseLayer;

        // Create the LoRA layer - derived classes may override this via CreateLoRALayer
        _loraLayer = CreateLoRALayer(rank, alpha);

        // Initialize parameters
        Parameters = new Vector<T>(ParameterCount);
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
    protected virtual LoRALayer<T> CreateLoRALayer(int rank, double alpha)
    {
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
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

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through both layers.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass propagates gradients through both the LoRA layer and (if not frozen)
    /// the base layer. The input gradients from both paths are summed.
    /// </para>
    /// <para><b>For Beginners:</b> During learning, this figures out how to improve both layers:
    /// - Always updates the LoRA layer (that's what we're training)
    /// - Only updates the base layer if it's not frozen
    /// - Combines the gradients from both paths to tell earlier layers how to improve
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward through LoRA layer
        Tensor<T> loraInputGrad = _loraLayer.Backward(outputGradient);

        // Backward through base layer
        // Note: Input gradients are always computed; base parameter updates are skipped in UpdateParameters if frozen
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(loraInputGrad[i], baseInputGrad[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromLayers();

        return inputGrad;
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
            int baseParamCount = _baseLayer.ParameterCount;
            Vector<T> baseParams = new Vector<T>(baseParamCount);
            for (int i = 0; i < baseParamCount; i++)
            {
                baseParams[i] = Parameters[idx++];
            }
            _baseLayer.SetParameters(baseParams);
        }

        // Unpack LoRA parameters
        int loraParamCount = _loraLayer.ParameterCount;
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
        ParameterGradients = new Vector<T>(ParameterCount);
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

        // Merge weights
        // baseParams is stored as [outputSize][inputSize] (row-major, output-major)
        // loraWeights from MergeWeights() is [inputSize, outputSize]
        // So we access loraWeights[inputIdx, outputIdx]
        for (int i = 0; i < weightCount; i++)
        {
            int outputIdx = i / inputSize;
            int inputIdx = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[inputIdx, outputIdx]);
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
    /// Gets whether this LoRA adapter supports JIT compilation.
    /// </summary>
    /// <value>True if both the base layer and LoRA layer support JIT compilation.</value>
    /// <remarks>
    /// <para>
    /// LoRA adapters support JIT compilation when both their component layers (the base layer
    /// and the LoRA layer) support JIT compilation. The computation graph combines both layers:
    /// output = base_layer(input) + lora_layer(input)
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation makes layers run faster by converting
    /// their math operations into optimized native code.
    ///
    /// A LoRA adapter can be JIT compiled when:
    /// - The base layer supports JIT compilation (has its weights initialized)
    /// - The LoRA layer supports JIT compilation (has its A and B matrices initialized)
    ///
    /// The JIT-compiled version computes both the base layer's output and the LoRA adaptation
    /// in parallel, then adds them together. This can provide significant speedup (5-10x).
    ///
    /// Alternatively, you can merge the LoRA weights into the base layer using MergeToOriginalLayer()
    /// for an even simpler and potentially faster deployment.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        _baseLayer.SupportsJitCompilation && _loraLayer.SupportsJitCompilation;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which input nodes will be added.</param>
    /// <returns>The output computation node representing the combined base + LoRA transformation.</returns>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when component layers are not initialized.</exception>
    /// <remarks>
    /// <para>
    /// The computation graph implements: output = base_layer(input) + lora_layer(input)
    ///
    /// This mirrors the Forward() method logic where:
    /// 1. The input is passed through the base layer
    /// 2. The same input is passed through the LoRA layer
    /// 3. The two outputs are added element-wise
    /// </para>
    /// <para><b>For Beginners:</b> This exports the LoRA adapter's computation as a graph of operations
    /// that can be optimized and compiled to fast native code.
    ///
    /// The graph represents:
    /// 1. Input → base layer computation → base output
    /// 2. Input → LoRA layer computation → LoRA output
    /// 3. base output + LoRA output → final output
    ///
    /// The JIT compiler can then fuse operations, apply SIMD vectorization, and perform
    /// other optimizations to make inference faster.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (!_baseLayer.SupportsJitCompilation)
            throw new InvalidOperationException(
                $"Base layer {_baseLayer.GetType().Name} does not support JIT compilation. " +
                "Ensure the base layer is properly initialized.");

        if (!_loraLayer.SupportsJitCompilation)
            throw new InvalidOperationException(
                "LoRA layer does not support JIT compilation. " +
                "Ensure the LoRA matrices are properly initialized.");

        // Export computation graphs from both component layers
        // The base layer and LoRA layer will each add their input nodes
        var baseInputNodes = new List<ComputationNode<T>>();
        var loraInputNodes = new List<ComputationNode<T>>();

        var baseOutputNode = _baseLayer.ExportComputationGraph(baseInputNodes);
        var loraOutputNode = _loraLayer.ExportComputationGraph(loraInputNodes);

        // Both layers should have created an input node as their first entry
        // We need to ensure they share the same input
        if (baseInputNodes.Count == 0 || loraInputNodes.Count == 0)
            throw new InvalidOperationException(
                "Component layers did not export input nodes correctly.");

        // Get the input node from the base layer (both layers expect the same input shape)
        var inputNode = baseInputNodes[0];

        // Add all input nodes to the caller's list
        // The input node is shared, so we add it once, then add parameters from both layers
        inputNodes.Add(inputNode);

        // Add base layer parameter nodes (skip the first which is the input)
        for (int i = 1; i < baseInputNodes.Count; i++)
        {
            inputNodes.Add(baseInputNodes[i]);
        }

        // Add LoRA layer parameter nodes (skip the first which is the input - same as base)
        for (int i = 1; i < loraInputNodes.Count; i++)
        {
            inputNodes.Add(loraInputNodes[i]);
        }

        // Combine the outputs: output = base_output + lora_output
        var combinedOutput = TensorOperations<T>.Add(baseOutputNode, loraOutputNode);

        return combinedOutput;
    }
}
