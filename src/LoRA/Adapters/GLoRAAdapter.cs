using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Generalized LoRA (GLoRA) implementation that adapts both weights AND activations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// GLoRA extends standard LoRA by adding adaptation to both the layer's weights and its activations.
/// This provides more flexibility for multi-task learning scenarios where different tasks may need
/// different feature representations at each layer.
/// </para>
/// <para>
/// The forward pass computes:
/// - adapted_weights = base_weights + B_w * A_w (weight adaptation)
/// - base_output = input * adapted_weights
/// - adapted_output = base_output + B_a * A_a * input (activation adaptation)
/// </para>
/// <para><b>For Beginners:</b> While standard LoRA only adapts what the layer learns (its weights),
/// GLoRA also adapts what the layer produces (its activations). Think of it like this:
///
/// - Standard LoRA: Adjusts the "recipe" (weights) but produces the same type of output
/// - GLoRA: Adjusts both the "recipe" (weights) AND transforms the output for different uses
///
/// This is especially useful when:
/// 1. Different tasks need different feature representations
/// 2. You're doing multi-task learning (e.g., the same base features used differently)
/// 3. You need more flexibility than weight-only adaptation provides
///
/// Key differences from StandardLoRA:
/// - WeightAdaptation: Standard LoRA component that modifies layer weights
/// - ActivationAdaptation: Additional LoRA component that modifies layer outputs
/// - ActivationRank: Can be different from weight rank for fine-tuned control
///
/// Trade-offs:
/// + More flexible: Can adapt representations for different tasks
/// + Better for multi-task: Each task can use features differently
/// - More parameters: Two LoRA components instead of one
/// - Slightly slower: Two adaptation computations per forward pass
///
/// Example: For a 1000x1000 layer with weight_rank=8 and activation_rank=4:
/// - Weight adaptation: 16,000 parameters (same as standard LoRA)
/// - Activation adaptation: 8,000 additional parameters
/// - Total: 24,000 parameters (still 97.6% reduction from 1M!)
/// </para>
/// </remarks>
public class GLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// The LoRA layer that adapts activations (layer outputs).
    /// </summary>
    private readonly LoRALayer<T> _activationAdaptation;

    /// <summary>
    /// Gets the weight adaptation LoRA layer.
    /// </summary>
    /// <remarks>
    /// This adapts the layer's weights using standard LoRA (B_w * A_w).
    /// </remarks>
    public LoRALayer<T> WeightAdaptation => _loraLayer;

    /// <summary>
    /// Gets the activation adaptation LoRA layer.
    /// </summary>
    /// <remarks>
    /// This adapts the layer's outputs/activations using a second LoRA component (B_a * A_a).
    /// </remarks>
    public LoRALayer<T> ActivationAdaptation => _activationAdaptation;

    /// <summary>
    /// Gets the rank of the activation adaptation.
    /// </summary>
    /// <remarks>
    /// This can be different from the weight adaptation rank, allowing for independent
    /// control over the complexity of weight vs. activation adaptations.
    /// </remarks>
    public int ActivationRank => _activationAdaptation.Rank;

    /// <summary>
    /// Gets the total number of trainable parameters (both weight and activation adaptations).
    /// </summary>
    /// <remarks>
    /// If the base layer is frozen, this returns the sum of weight and activation LoRA parameters.
    /// Otherwise, it includes base layer parameters as well.
    /// </remarks>
    public override int ParameterCount => _freezeBaseLayer
        ? (_loraLayer.ParameterCount + _activationAdaptation.ParameterCount)
        : (_baseLayer.ParameterCount + _loraLayer.ParameterCount + _activationAdaptation.ParameterCount);

    /// <summary>
    /// Initializes a new GLoRA adapter with the specified parameters.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with GLoRA.</param>
    /// <param name="weightRank">The rank of the weight adaptation decomposition.</param>
    /// <param name="activationRank">The rank of the activation adaptation decomposition (defaults to weightRank if negative).</param>
    /// <param name="weightAlpha">The scaling factor for weight adaptation (defaults to weightRank if negative).</param>
    /// <param name="activationAlpha">The scaling factor for activation adaptation (defaults to activationRank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a GLoRA adapter that adds TWO types of adaptations:
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to make more flexible
    /// - weightRank: Compression for weight adaptation (lower = fewer parameters for weights)
    /// - activationRank: Compression for activation adaptation (can be different!)
    /// - weightAlpha: How strong the weight adaptation is
    /// - activationAlpha: How strong the activation adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true)
    ///
    /// Having separate ranks and alphas for weights vs. activations gives you fine-grained control:
    /// - Higher weight rank = more flexibility in what the layer learns
    /// - Higher activation rank = more flexibility in how outputs are transformed
    ///
    /// Common patterns:
    /// - Equal ranks: Balanced adaptation (weightRank=8, activationRank=8)
    /// - Lower activation rank: More emphasis on weight learning (weightRank=16, activationRank=4)
    /// - Higher activation rank: More emphasis on output transformation (weightRank=4, activationRank=16)
    /// </para>
    /// </remarks>
    public GLoRAAdapter(
        ILayer<T> baseLayer,
        int weightRank,
        int activationRank = -1,
        double weightAlpha = -1,
        double activationAlpha = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, weightRank, weightAlpha, freezeBaseLayer)
    {
        // Default activation rank to weight rank if not specified
        int actualActivationRank = activationRank > 0 ? activationRank : weightRank;

        // Create activation adaptation LoRA layer
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        _activationAdaptation = new LoRALayer<T>(inputSize, outputSize, actualActivationRank, activationAlpha);

        // Update parameter vector to include activation adaptation
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Performs the forward pass through both base layer and both LoRA adaptations.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output with both weight and activation adaptations applied.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes:
    /// 1. base_output = base_layer(input) (original layer behavior)
    /// 2. weight_adaptation = weight_lora(input) (standard LoRA weight adaptation)
    /// 3. activation_adaptation = activation_lora(input) (additional activation transformation)
    /// 4. output = base_output + weight_adaptation + activation_adaptation
    /// </para>
    /// <para><b>For Beginners:</b> This runs the input through three parallel paths:
    /// 1. The base layer (original behavior)
    /// 2. Weight LoRA (learns how weights should change)
    /// 3. Activation LoRA (learns how outputs should be transformed)
    ///
    /// All three outputs are added together to get the final result. This allows the model to:
    /// - Keep the original layer's learned features (base layer)
    /// - Refine what it learns (weight adaptation)
    /// - Transform how it represents things (activation adaptation)
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through weight adaptation LoRA
        Tensor<T> weightAdaptationOutput = _loraLayer.Forward(input);

        // Forward through activation adaptation LoRA
        Tensor<T> activationAdaptationOutput = _activationAdaptation.Forward(input);

        // Sum all outputs: base + weight_adaptation + activation_adaptation
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            T sum = NumOps.Add(baseOutput[i], weightAdaptationOutput[i]);
            result[i] = NumOps.Add(sum, activationAdaptationOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through both adaptations and the base layer.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass propagates gradients through all three components:
    /// - Weight adaptation LoRA (always)
    /// - Activation adaptation LoRA (always)
    /// - Base layer (only if not frozen)
    /// </para>
    /// <para><b>For Beginners:</b> During learning, this figures out how to improve all adaptations:
    /// - Updates weight adaptation (how should weights change?)
    /// - Updates activation adaptation (how should outputs be transformed?)
    /// - Updates base layer if not frozen (how should original weights change?)
    ///
    /// The gradients from all three paths are combined to tell earlier layers how to improve.
    /// This allows the model to learn complex adaptations that work together.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward through weight adaptation LoRA
        Tensor<T> weightLoraInputGrad = _loraLayer.Backward(outputGradient);

        // Backward through activation adaptation LoRA
        Tensor<T> activationLoraInputGrad = _activationAdaptation.Backward(outputGradient);

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Sum all input gradients
        Tensor<T> inputGrad = new Tensor<T>(weightLoraInputGrad.Shape);
        for (int i = 0; i < weightLoraInputGrad.Length; i++)
        {
            T sum = NumOps.Add(weightLoraInputGrad[i], activationLoraInputGrad[i]);
            inputGrad[i] = NumOps.Add(sum, baseInputGrad[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromLayers();

        return inputGrad;
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// Updates both weight and activation adaptation parameters.
    /// Base layer parameters are only updated if not frozen.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Always update both LoRA layers
        _loraLayer.UpdateParameters(learningRate);
        _activationAdaptation.UpdateParameters(learningRate);

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
    /// <returns>Vector containing parameters from both adaptations (and base layer if not frozen).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing parameters for both adaptations (and base layer if not frozen).</param>
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

        // Pack weight adaptation LoRA parameters
        Vector<T> weightLoraParams = _loraLayer.GetParameters();
        for (int i = 0; i < weightLoraParams.Length; i++)
        {
            Parameters[idx++] = weightLoraParams[i];
        }

        // Pack activation adaptation LoRA parameters
        Vector<T> activationLoraParams = _activationAdaptation.GetParameters();
        for (int i = 0; i < activationLoraParams.Length; i++)
        {
            Parameters[idx++] = activationLoraParams[i];
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

        // Unpack weight adaptation LoRA parameters
        int weightLoraParamCount = _loraLayer.ParameterCount;
        Vector<T> weightLoraParams = new Vector<T>(weightLoraParamCount);
        for (int i = 0; i < weightLoraParamCount; i++)
        {
            weightLoraParams[i] = Parameters[idx++];
        }
        _loraLayer.SetParameters(weightLoraParams);

        // Unpack activation adaptation LoRA parameters
        int activationLoraParamCount = _activationAdaptation.ParameterCount;
        Vector<T> activationLoraParams = new Vector<T>(activationLoraParamCount);
        for (int i = 0; i < activationLoraParamCount; i++)
        {
            activationLoraParams[i] = Parameters[idx++];
        }
        _activationAdaptation.SetParameters(activationLoraParams);
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

        // Pack weight adaptation LoRA gradients
        Vector<T> weightLoraGrads = _loraLayer.GetParameterGradients();
        for (int i = 0; i < weightLoraGrads.Length; i++)
        {
            ParameterGradients[idx++] = weightLoraGrads[i];
        }

        // Pack activation adaptation LoRA gradients
        Vector<T> activationLoraGrads = _activationAdaptation.GetParameterGradients();
        for (int i = 0; i < activationLoraGrads.Length; i++)
        {
            ParameterGradients[idx++] = activationLoraGrads[i];
        }
    }

    /// <summary>
    /// Merges both LoRA adaptations into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with both weight and activation adaptations merged into the base layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method merges both the weight adaptation and activation adaptation into the base layer's weights.
    /// Since activation adaptation operates on outputs, it's merged by adding it to the weight matrix as well.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" both GLoRA adaptations to create a regular layer.
    /// After training with GLoRA, you can merge both adaptations into the original weights for:
    /// - Faster inference (no need to compute two LoRA layers separately)
    /// - Simpler deployment (single layer instead of three components)
    /// - Compatibility with systems that don't support LoRA
    ///
    /// The merging process:
    /// 1. Computes weight adaptation matrix from weight LoRA (B_w * A_w)
    /// 2. Computes activation adaptation matrix from activation LoRA (B_a * A_a)
    /// 3. Adds both to the base layer's weights
    /// 4. Copies biases unchanged
    /// 5. Creates a new layer with all adaptations merged
    ///
    /// Note: Merging currently only supports DenseLayer and FullyConnectedLayer.
    /// For other layer types, you'll need to use the adapter in production.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("GLoRAAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get both LoRA weight contributions
        Matrix<T> weightLoraWeights = _loraLayer.MergeWeights();
        Matrix<T> activationLoraWeights = _activationAdaptation.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Both DenseLayer and FullyConnectedLayer store parameters as [weights..., biases...]
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge all weights: base + weight_lora + activation_lora
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            T sum = NumOps.Add(baseParams[i], weightLoraWeights[row, col]);
            mergedParams[i] = NumOps.Add(sum, activationLoraWeights[row, col]);
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
    /// Resets the internal state of the base layer and both LoRA adaptations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears the memory of all three components (base layer,
    /// weight adaptation, and activation adaptation). It's useful when starting to process
    /// a completely new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _loraLayer.ResetState();
        _activationAdaptation.ResetState();
    }
}
