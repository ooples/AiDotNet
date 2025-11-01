namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Wraps an existing layer with LoRA functionality, allowing parameter-efficient fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The LoRAAdapter wraps an existing layer (called the base layer) and adds a LoRA layer in parallel.
/// During forward pass, both the base layer and LoRA layer process the input, and their outputs are
/// summed. The base layer's parameters can be frozen while only the LoRA parameters are trained.
/// </para>
/// <para><b>For Beginners:</b> This adapter lets you add LoRA to an existing layer without modifying it.
/// Think of it like adding a "correction layer" that learns what adjustments are needed:
///
/// - The base layer keeps its original weights (optionally frozen)
/// - The LoRA layer learns a small correction
/// - The final output is: original_output + lora_correction
///
/// This is incredibly useful for fine-tuning pre-trained models:
/// 1. Load a pre-trained model
/// 2. Wrap its layers with LoRAAdapter
/// 3. Freeze the base layers
/// 4. Train only the small LoRA corrections
/// 5. Achieve similar results with 100x fewer trainable parameters!
/// </para>
/// </remarks>
public class LoRAAdapter<T> : LayerBase<T>
{
    /// <summary>
    /// The base layer being adapted.
    /// </summary>
    private readonly ILayer<T> _baseLayer;

    /// <summary>
    /// The LoRA layer that provides the adaptation.
    /// </summary>
    private readonly LoRALayer<T> _loraLayer;

    /// <summary>
    /// Whether the base layer's parameters are frozen (not trainable).
    /// </summary>
    private readonly bool _freezeBaseLayer;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// If the base layer is frozen, this returns only the LoRA parameter count.
    /// Otherwise, it returns the sum of base and LoRA parameters.
    /// </remarks>
    public override int ParameterCount => _freezeBaseLayer ? _loraLayer.ParameterCount : (_baseLayer.ParameterCount + _loraLayer.ParameterCount);

    /// <summary>
    /// Gets whether this adapter supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new LoRA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the base layer doesn't have compatible dimensions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an adapter that adds LoRA to an existing layer.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to make more efficient to fine-tune
    /// - rank: How much compression (lower = fewer parameters, less flexibility)
    /// - alpha: How strong the LoRA adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true for efficiency)
    ///
    /// Example: If you have a dense layer with 1000x1000 weights, wrapping it with rank=8 LoRA
    /// (frozen) reduces trainable parameters from 1,000,000 to just 16,000!
    /// </para>
    /// </remarks>
    public LoRAAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(
            (baseLayer ?? throw new ArgumentNullException(nameof(baseLayer))).GetInputShape(),
            (baseLayer ?? throw new ArgumentNullException(nameof(baseLayer))).GetOutputShape())
    {
        _baseLayer = baseLayer;
        _freezeBaseLayer = freezeBaseLayer;

        // Validate base layer has single-dimensional input/output
        if (baseLayer.GetInputShape().Length != 1 || baseLayer.GetOutputShape().Length != 1)
        {
            throw new ArgumentException("LoRAAdapter currently only supports layers with 1D input/output shapes");
        }

        int inputSize = baseLayer.GetInputShape()[0];
        int outputSize = baseLayer.GetOutputShape()[0];

        // Create the LoRA layer
        _loraLayer = new LoRALayer<T>(inputSize, outputSize, rank, alpha);

        // Initialize parameters
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromLayers();
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
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        }

        Parameters = parameters.Clone();
        UpdateLayersFromParameters();
    }

    /// <summary>
    /// Updates the parameter vector from the current layer states.
    /// </summary>
    private void UpdateParametersFromLayers()
    {
        int idx = 0;

        // If base layer is not frozen, pack its parameters first
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // Pack LoRA parameters
        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }
    }

    /// <summary>
    /// Updates the layers from the parameter vector.
    /// </summary>
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
    /// <exception cref="InvalidOperationException">Thrown when the base layer type doesn't support merging.</exception>
    /// <remarks>
    /// <para>
    /// This is only supported for DenseLayer base layers currently. The LoRA weights are computed
    /// and added directly to the base layer's weight matrix.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your LoRA adaptation to create a regular layer.
    /// After training with LoRA, you can merge the adaptation into the original weights for:
    /// - Faster inference (no need to compute LoRA separately)
    /// - Simpler deployment (single layer instead of two)
    /// - Compatibility with systems that don't support LoRA
    ///
    /// Think of it like merging tracked changes in a document - you go from "original + changes"
    /// to a single updated version.
    /// </para>
    /// </remarks>
    public ILayer<T> MergeToSingleLayer()
    {
        if (_baseLayer is not DenseLayer<T> denseBase)
        {
            throw new InvalidOperationException("Merging is currently only supported for DenseLayer base layers");
        }

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Clone the base layer and get its current parameters
        Vector<T> baseParams = denseBase.GetParameters();

        // The DenseLayer stores parameters as [weights..., biases...]
        // We need to add the LoRA weights to the base weights
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Create a new dense layer with merged parameters
        DenseLayer<T> mergedLayer = new DenseLayer<T>(inputSize, outputSize, (IActivationFunction<T>?)null);
        mergedLayer.SetParameters(mergedParams);

        return mergedLayer;
    }

    /// <summary>
    /// Gets the underlying base layer.
    /// </summary>
    public ILayer<T> BaseLayer => _baseLayer;

    /// <summary>
    /// Gets the LoRA layer.
    /// </summary>
    public LoRALayer<T> LoRALayer => _loraLayer;

    /// <summary>
    /// Gets whether the base layer is frozen.
    /// </summary>
    public bool IsBaseLayerFrozen => _freezeBaseLayer;

    /// <summary>
    /// Gets the rank of the LoRA adaptation.
    /// </summary>
    public int Rank => _loraLayer.Rank;

    /// <summary>
    /// Gets the LoRA alpha scaling factor.
    /// </summary>
    public T Alpha => _loraLayer.Alpha;

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
}
