using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// LoRA-drop implementation: LoRA with dropout regularization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoRA-drop extends standard LoRA by adding dropout to the LoRA components during training.
/// During the forward pass in training mode, a random subset of LoRA components are "dropped out"
/// (set to zero), forcing the model to learn more robust adaptations that don't rely on any
/// single component.
/// </para>
/// <para>
/// Key differences from standard LoRA:
/// - Applies dropout to LoRA output during training
/// - Scales LoRA output by (1 - dropout_rate) during inference
/// - Improves generalization and reduces overfitting
/// - Particularly useful when adaptation data is limited
/// </para>
/// <para><b>For Beginners:</b> LoRA-drop adds dropout regularization to LoRA adapters.
///
/// Dropout is a technique where during training, we randomly "turn off" some neurons or components.
/// This prevents the model from becoming too dependent on specific components and forces it to
/// learn more general patterns.
///
/// Think of it like practicing a skill with random handicaps:
/// - Sometimes you practice with your left hand tied behind your back
/// - Sometimes you practice blindfolded
/// - This forces you to develop multiple strategies instead of relying on one approach
///
/// LoRA-drop applies this to LoRA adaptations:
/// - During training: Randomly drop some LoRA components (set them to zero)
/// - During inference: Use all components but scale them appropriately
/// - Result: More robust adaptations that generalize better to new data
///
/// Recommended dropout rates:
/// - 0.1 (10%): Light regularization, good starting point
/// - 0.2 (20%): Moderate regularization, common choice
/// - 0.3 (30%): Strong regularization, for small adaptation datasets
/// - Higher rates (&gt;0.5): Typically too aggressive, may harm performance
///
/// When to use LoRA-drop over standard LoRA:
/// - You have limited adaptation data (risk of overfitting)
/// - You need better generalization to unseen data
/// - You're fine-tuning on a very specific task but need to maintain general capabilities
/// - You've observed overfitting with standard LoRA
/// </para>
/// </remarks>
public class LoRADropAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Dropout rate (probability of dropping a component during training).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The dropout rate determines what fraction of LoRA output components are randomly
    /// set to zero during each training step. Common values are 0.1-0.3.
    /// </para>
    /// <para><b>For Beginners:</b> This is the probability that any given component gets "turned off"
    /// during training. For example, 0.2 means each component has a 20% chance of being dropped.
    /// </para>
    /// </remarks>
    private readonly double _dropoutRate;

    /// <summary>
    /// Mask indicating which components to drop in the current forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This boolean array has the same length as the LoRA output. True means keep the component,
    /// false means drop it (set to zero). The mask is regenerated randomly for each forward pass
    /// during training.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a binary on/off switch for each component.
    /// During training, we randomly set some to "off" (false) to apply dropout.
    /// </para>
    /// </remarks>
    private bool[]? _dropoutMask;

    /// <summary>
    /// Indicates whether the layer is in training mode (dropout active) or inference mode (dropout inactive).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, dropout is applied during forward passes. When false (inference mode),
    /// dropout is disabled and outputs are scaled by (1 - dropout_rate) for consistency.
    /// </para>
    /// <para><b>For Beginners:</b> This switch controls whether we're in "learning mode" or "using mode".
    /// During learning (training), we apply dropout. During use (inference), we turn it off.
    /// </para>
    /// </remarks>
    private bool _isTraining;

    /// <summary>
    /// Random number generator for dropout mask generation.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Gets the dropout rate used for regularization.
    /// </summary>
    public double DropoutRate => _dropoutRate;

    /// <summary>
    /// Gets or sets whether the layer is in training mode.
    /// </summary>
    /// <remarks>
    /// Set to true during training (dropout active), false during inference (dropout inactive).
    /// </remarks>
    public bool IsTraining
    {
        get => _isTraining;
        set => _isTraining = value;
    }

    /// <summary>
    /// Initializes a new LoRA-drop adapter with dropout regularization.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="dropoutRate">The dropout rate (probability of dropping a component). Common values: 0.1-0.3.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <param name="seed">Random seed for reproducible dropout masks (optional).</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when dropoutRate is not in [0, 1) range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a LoRA adapter with dropout regularization.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt
    /// - rank: How much compression to use (same as standard LoRA)
    /// - dropoutRate: What fraction to randomly drop during training (0.1 = 10%, 0.2 = 20%, etc.)
    /// - alpha: How strong the LoRA adaptation is
    /// - freezeBaseLayer: Whether to freeze the original layer (usually true)
    /// - seed: Optional random seed for reproducible results
    ///
    /// Example usage:
    /// ```csharp
    /// // Create a LoRA-drop adapter with 20% dropout
    /// var adapter = new LoRADropAdapter&lt;double&gt;(denseLayer, rank: 8, dropoutRate: 0.2);
    ///
    /// // Training mode (dropout active)
    /// adapter.SetTraining(true);
    /// var trainOutput = adapter.Forward(trainInput);
    ///
    /// // Inference mode (dropout inactive)
    /// adapter.SetTraining(false);
    /// var testOutput = adapter.Forward(testInput);
    /// ```
    /// </para>
    /// </remarks>
    public LoRADropAdapter(ILayer<T> baseLayer, int rank, double dropoutRate, double alpha = -1, bool freezeBaseLayer = true, int? seed = null)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (dropoutRate < 0.0 || dropoutRate >= 1.0)
        {
            throw new ArgumentException("Dropout rate must be in the range [0, 1)", nameof(dropoutRate));
        }

        _dropoutRate = dropoutRate;
        _isTraining = true; // Default to training mode
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        // Initialize dropout mask (will be regenerated on each forward pass during training)
        int outputSize = GetOutputShape()[0];
        _dropoutMask = new bool[outputSize];
    }

    /// <summary>
    /// Sets whether the layer is in training mode or inference mode.
    /// </summary>
    /// <param name="training">True for training mode (dropout active), false for inference mode (dropout inactive).</param>
    /// <remarks>
    /// <para>
    /// This method should be called to switch between training and inference modes.
    /// During training, dropout is applied. During inference, dropout is disabled and
    /// outputs are scaled appropriately.
    /// </para>
    /// <para><b>For Beginners:</b> Call this before you start training or testing:
    /// - Before training: `adapter.SetTraining(true)`
    /// - Before testing/inference: `adapter.SetTraining(false)`
    ///
    /// This ensures dropout is only used during training, not when making predictions.
    /// </para>
    /// </remarks>
    public void SetTraining(bool training)
    {
        _isTraining = training;
    }

    /// <summary>
    /// Generates a random dropout mask for the current forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For each component, generates a random value and compares it to the dropout rate.
    /// If the random value is greater than the dropout rate, the component is kept (true),
    /// otherwise it's dropped (false).
    /// </para>
    /// <para><b>For Beginners:</b> This randomly decides which components to keep and which to drop.
    /// Think of it like flipping a weighted coin for each component - if you get "heads"
    /// (random value &gt; dropout rate), you keep it; otherwise you drop it.
    /// </para>
    /// </remarks>
    private void GenerateDropoutMask()
    {
        if (_dropoutMask == null)
        {
            return;
        }

        for (int i = 0; i < _dropoutMask.Length; i++)
        {
            // Keep the component if random value is greater than dropout rate
            _dropoutMask[i] = _random.NextDouble() > _dropoutRate;
        }
    }

    /// <summary>
    /// Performs the forward pass with dropout applied to LoRA output.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and dropout-regularized LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// During training:
    /// 1. Generate new dropout mask
    /// 2. Compute LoRA output
    /// 3. Apply dropout mask (zero out dropped components)
    /// 4. Scale kept components by 1/(1-dropout_rate) to maintain expected value
    /// 5. Add to base layer output
    ///
    /// During inference:
    /// 1. Compute LoRA output
    /// 2. Scale by (1-dropout_rate) to match training expectation
    /// 3. Add to base layer output
    /// </para>
    /// <para><b>For Beginners:</b> This runs the input through the layer with dropout applied.
    ///
    /// Training mode:
    /// - Randomly drops some LoRA components
    /// - Scales up the remaining components to compensate
    /// - This forces the model to not rely on any single component
    ///
    /// Inference mode:
    /// - Uses all components
    /// - Scales them down to match what the model learned during training
    /// - This ensures consistent behavior between training and testing
    ///
    /// The scaling ensures that the expected output is the same whether or not dropout is active,
    /// which is important for stable training and accurate predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through LoRA layer
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // Apply dropout to LoRA output
        if (_isTraining)
        {
            // Training mode: apply dropout mask
            GenerateDropoutMask();

            // Scale factor to maintain expected value: 1 / (1 - dropout_rate)
            // This compensates for the components we're dropping
            T invKeepProb = NumOps.Divide(NumOps.One, NumOps.FromDouble(1.0 - _dropoutRate));

            for (int i = 0; i < loraOutput.Length; i++)
            {
                if (_dropoutMask != null && !_dropoutMask[i % _dropoutMask.Length])
                {
                    // Drop this component
                    loraOutput[i] = NumOps.Zero;
                }
                else
                {
                    // Keep this component and scale it
                    loraOutput[i] = NumOps.Multiply(loraOutput[i], invKeepProb);
                }
            }
        }
        else
        {
            // Inference mode: no dropout, but scale by (1 - dropout_rate)
            // This matches the expected value from training
            T scale = NumOps.FromDouble(1.0 - _dropoutRate);
            for (int i = 0; i < loraOutput.Length; i++)
            {
                loraOutput[i] = NumOps.Multiply(loraOutput[i], scale);
            }
        }

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass with dropout mask applied to gradients.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// During backpropagation, gradients are only propagated through components that were
    /// not dropped during the forward pass. This is achieved by applying the same dropout
    /// mask to the gradients and scaling appropriately.
    /// </para>
    /// <para><b>For Beginners:</b> This propagates gradients back through the layer.
    ///
    /// Key insight: Gradients only flow through the components that were active during
    /// the forward pass. If a component was dropped (set to zero), its gradient is also
    /// zero - we don't update it based on this training example.
    ///
    /// This ensures that:
    /// - Dropped components don't get updated (they were "turned off")
    /// - Kept components get normal gradient updates
    /// - The scaling from the forward pass is preserved in gradients
    ///
    /// The result is that the model learns to work with different subsets of components,
    /// making it more robust and less prone to overfitting.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Create a gradient for the LoRA layer
        Tensor<T> loraGradient = new Tensor<T>(outputGradient.Shape);

        if (_isTraining)
        {
            // Apply dropout mask and scaling to gradients
            T invKeepProb = NumOps.Divide(NumOps.One, NumOps.FromDouble(1.0 - _dropoutRate));

            for (int i = 0; i < outputGradient.Length; i++)
            {
                if (_dropoutMask != null && !_dropoutMask[i % _dropoutMask.Length])
                {
                    // This component was dropped - zero gradient
                    loraGradient[i] = NumOps.Zero;
                }
                else
                {
                    // This component was kept - propagate gradient with scaling
                    loraGradient[i] = NumOps.Multiply(outputGradient[i], invKeepProb);
                }
            }
        }
        else
        {
            // Inference mode: scale gradients by (1 - dropout_rate)
            T scale = NumOps.FromDouble(1.0 - _dropoutRate);
            for (int i = 0; i < outputGradient.Length; i++)
            {
                loraGradient[i] = NumOps.Multiply(outputGradient[i], scale);
            }
        }

        // Backward through LoRA layer with dropout-adjusted gradient
        Tensor<T> loraInputGrad = _loraLayer.Backward(loraGradient);

        // Backward through base layer with original gradient
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
    /// Updates parameter gradients from both layers (called by Backward).
    /// </summary>
    /// <remarks>
    /// This is a helper method that collects gradients from the base and LoRA layers
    /// into the unified parameter gradient vector. It respects the frozen state of the base layer.
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
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method merges the trained LoRA weights into the base layer to create a single
    /// layer that includes the adaptations. The dropout mechanism is not preserved in the
    /// merged layer - only the learned weights are incorporated.
    /// </para>
    /// <para><b>For Beginners:</b> After training with LoRA-drop, you can "bake in" the adaptations.
    ///
    /// This creates a regular layer that:
    /// - Contains the original weights plus the learned LoRA adaptations
    /// - Doesn't need the LoRA machinery anymore
    /// - Is faster for inference (no separate LoRA computation)
    /// - Doesn't include dropout (dropout is only for training)
    ///
    /// The merging process:
    /// 1. Computes the full LoRA weight contribution (A × B matrices)
    /// 2. Adds these weights to the base layer's weights
    /// 3. Creates a new DenseLayer with the combined weights
    ///
    /// Note: The merged layer is in "inference mode" - it represents what the model learned
    /// during training but doesn't include the dropout mechanism.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("LoRADropAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Both DenseLayer and FullyConnectedLayer store parameters as [weights..., biases...]
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
    /// Resets the internal state of both layers and clears the dropout mask.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears all cached data from both the base layer and LoRA layer,
    /// and resets the dropout mask. It's useful when starting to process a new batch or sequence.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _loraLayer.ResetState();

        // Reset dropout mask
        if (_dropoutMask != null)
        {
            for (int i = 0; i < _dropoutMask.Length; i++)
            {
                _dropoutMask[i] = false;
            }
        }
    }
}
