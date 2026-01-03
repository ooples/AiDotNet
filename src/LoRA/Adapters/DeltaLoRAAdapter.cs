using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Delta-LoRA adapter that focuses on parameter-efficient delta updates with momentum.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Delta-LoRA is a variant of LoRA that explicitly models the change (delta) in parameters
/// rather than the absolute values. This approach can achieve better convergence in certain
/// scenarios by focusing on the parameter update dynamics with momentum-based accumulation.
/// </para>
/// <para><b>For Beginners:</b> Think of Delta-LoRA as "change-focused" LoRA.
///
/// Regular LoRA learns: "What should the weights be?"
/// Delta-LoRA learns: "How should the weights change?"
///
/// This difference matters because:
/// 1. Changes (deltas) often have simpler patterns than absolute values
/// 2. Momentum helps smooth out noisy updates
/// 3. Can converge faster when the optimal adaptation is a smooth transformation
///
/// Key concepts:
/// - <b>Delta weights</b>: Accumulated changes to parameters (not the parameters themselves)
/// - <b>Delta scaling</b>: Controls how strongly deltas affect the output
/// - <b>Momentum</b>: Smooths updates by remembering previous changes
///
/// When Delta-LoRA works better than standard LoRA:
/// - Tasks requiring smooth, gradual adaptations
/// - Fine-tuning where the base model is already close to optimal
/// - Scenarios with noisy gradients that benefit from momentum
/// - Transfer learning where you want to preserve more of the original model's behavior
///
/// Example: If you're adapting a language model to a new domain, Delta-LoRA can
/// make smaller, more conservative changes that preserve the model's general knowledge
/// while adapting to domain-specific patterns.
/// </para>
/// </remarks>
public class DeltaLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Matrix storing the cumulative weight deltas (changes over time).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix accumulates the changes to the weights rather than storing absolute weight values.
    /// It has the same dimensions as the output of the LoRA layer (outputSize × inputSize).
    /// </para>
    /// <para><b>For Beginners:</b> This is like a running total of all the adjustments made during training.
    /// Instead of "what are the weights", it tracks "how much have they changed".
    /// </para>
    /// </remarks>
    private Matrix<T> _deltaWeights;

    /// <summary>
    /// Scaling factor applied to delta updates before adding to the output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the magnitude of the delta contribution. Lower values make smaller adjustments,
    /// higher values make larger adjustments. Typical range: 0.01 to 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a "sensitivity" knob. Higher values mean the
    /// accumulated changes have a stronger effect on the output.
    /// </para>
    /// </remarks>
    private readonly double _deltaScaling;

    /// <summary>
    /// Momentum factor for delta accumulation (0 to 1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls how much previous delta updates influence new updates.
    /// - 0.0 = No momentum (each update is independent)
    /// - 0.9 = High momentum (updates are heavily influenced by history)
    /// Typical value: 0.9
    /// </para>
    /// <para><b>For Beginners:</b> Momentum is like inertia in physics. It makes updates smoother
    /// by remembering the direction you were moving before. This helps avoid erratic changes and
    /// can speed up convergence.
    /// </para>
    /// </remarks>
    private readonly double _momentumFactor;

    /// <summary>
    /// Velocity matrix for momentum-based updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stores the moving average of gradients, used for momentum-based optimization.
    /// Has the same dimensions as _deltaWeights.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks the "speed and direction" of parameter changes.
    /// When gradients point in consistent directions, velocity builds up, making updates faster.
    /// When gradients change direction, velocity slows down, preventing oscillation.
    /// </para>
    /// </remarks>
    private Matrix<T> _velocity;

    /// <summary>
    /// Gradients for the delta weights computed during backpropagation.
    /// </summary>
    private Matrix<T>? _deltaGradients;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets the scaling factor for delta updates.
    /// </summary>
    public double DeltaScaling => _deltaScaling;

    /// <summary>
    /// Gets the momentum factor for delta accumulation.
    /// </summary>
    public double MomentumFactor => _momentumFactor;

    /// <summary>
    /// Gets the total number of trainable parameters including delta weights.
    /// </summary>
    /// <remarks>
    /// Includes base layer (if not frozen), LoRA layer, and delta weights matrix parameters.
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int baseCount = base.ParameterCount; // Base layer + LoRA layer
            int deltaCount = _deltaWeights.Rows * _deltaWeights.Columns;
            return baseCount + deltaCount;
        }
    }

    /// <summary>
    /// Initializes a new Delta-LoRA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with Delta-LoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="deltaScaling">Scaling factor for delta updates (default: 0.1).</param>
    /// <param name="momentumFactor">Momentum factor for delta accumulation (default: 0.9).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when deltaScaling or momentumFactor are out of valid range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a Delta-LoRA adapter with momentum-based updates.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt
    /// - rank: Compression level (lower = fewer parameters)
    /// - alpha: LoRA strength
    /// - deltaScaling: How strongly deltas affect output (0.01 to 1.0, default 0.1)
    /// - momentumFactor: How much to smooth updates (0.0 to 1.0, default 0.9)
    /// - freezeBaseLayer: Whether to lock the original layer (usually true)
    ///
    /// Recommended settings:
    /// - For stable tasks: deltaScaling=0.1, momentumFactor=0.9
    /// - For aggressive adaptation: deltaScaling=0.5, momentumFactor=0.5
    /// - For conservative adaptation: deltaScaling=0.01, momentumFactor=0.95
    /// </para>
    /// </remarks>
    public DeltaLoRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        double alpha = -1,
        double deltaScaling = 0.1,
        double momentumFactor = 0.9,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (deltaScaling <= 0.0)
        {
            throw new ArgumentException("Delta scaling must be positive", nameof(deltaScaling));
        }

        if (momentumFactor < 0.0 || momentumFactor >= 1.0)
        {
            throw new ArgumentException("Momentum factor must be in range [0.0, 1.0)", nameof(momentumFactor));
        }

        _deltaScaling = deltaScaling;
        _momentumFactor = momentumFactor;

        // Initialize delta weights and velocity matrices
        // Use [inputSize, outputSize] to match DenseLayer's industry standard convention
        int outputSize = GetOutputShape()[0];
        int inputSize = GetInputShape()[0];
        _deltaWeights = new Matrix<T>(inputSize, outputSize);
        _velocity = new Matrix<T>(inputSize, outputSize);

        // Initialize to zero
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                _deltaWeights[i, j] = NumOps.Zero;
                _velocity[i, j] = NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Performs the forward pass: output = base_layer(input) + LoRA(input) + delta_weights @ input * delta_scaling.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Combined output from base layer, LoRA layer, and delta weights.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes three components:
    /// 1. Base layer output (original layer behavior)
    /// 2. LoRA output (low-rank adaptation)
    /// 3. Delta output (accumulated parameter changes scaled by deltaScaling)
    /// </para>
    /// <para><b>For Beginners:</b> This combines three sources of information:
    /// - The original layer's predictions (base)
    /// - The LoRA adaptation (learned low-rank changes)
    /// - The accumulated deltas (momentum-smoothed changes)
    ///
    /// The delta component is what makes this different from standard LoRA - it explicitly
    /// applies the accumulated changes with scaling, allowing for more controlled adaptation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store input for backward pass
        _lastInput = input.Clone();

        // Get base layer output
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Get LoRA layer output
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // Compute delta contribution: delta_weights @ input * delta_scaling
        Tensor<T> deltaOutput = new Tensor<T>(baseOutput.Shape);

        // For each output dimension
        for (int i = 0; i < _deltaWeights.Rows; i++)
        {
            T sum = NumOps.Zero;
            // Dot product with input
            for (int j = 0; j < _deltaWeights.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_deltaWeights[i, j], input[j]));
            }
            // Apply delta scaling
            deltaOutput[i] = NumOps.Multiply(sum, NumOps.FromDouble(_deltaScaling));
        }

        // Combine all three outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(NumOps.Add(baseOutput[i], loraOutput[i]), deltaOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass, computing gradients for delta weights with momentum.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass:
    /// 1. Propagates gradients through base and LoRA layers (from base class)
    /// 2. Computes gradients for delta weights
    /// 3. Updates velocity using momentum
    /// 4. Accumulates all input gradients
    /// </para>
    /// <para><b>For Beginners:</b> This figures out how to improve all components:
    /// - The LoRA matrices (via the base class)
    /// - The delta weights (computed here)
    /// - Applies momentum to smooth out the delta updates
    ///
    /// Momentum helps by:
    /// - Accelerating convergence when gradients are consistent
    /// - Dampening oscillations when gradients are noisy
    /// - Creating smoother, more stable training dynamics
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        // Compute delta gradients: outputGradient ⊗ input (outer product)
        _deltaGradients = new Matrix<T>(_deltaWeights.Rows, _deltaWeights.Columns);

        for (int i = 0; i < _deltaWeights.Rows; i++)
        {
            for (int j = 0; j < _deltaWeights.Columns; j++)
            {
                // Gradient for delta[i,j] = outputGradient[i] * input[j] * delta_scaling
                T grad = NumOps.Multiply(
                    NumOps.Multiply(outputGradient[i], _lastInput[j]),
                    NumOps.FromDouble(_deltaScaling)
                );
                _deltaGradients[i, j] = grad;
            }
        }

        // Compute input gradient contribution from delta weights
        Tensor<T> deltaInputGrad = new Tensor<T>(_lastInput.Shape);
        for (int j = 0; j < _deltaWeights.Columns; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < _deltaWeights.Rows; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(
                    _deltaWeights[i, j],
                    NumOps.Multiply(outputGradient[i], NumOps.FromDouble(_deltaScaling))
                ));
            }
            deltaInputGrad[j] = sum;
        }

        // Backward through LoRA layer
        Tensor<T> loraInputGrad = _loraLayer.Backward(outputGradient);

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Combine all input gradients
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(
                NumOps.Add(loraInputGrad[i], baseInputGrad[i]),
                deltaInputGrad[i]
            );
        }

        return inputGrad;
    }

    /// <summary>
    /// Updates parameters using momentum-based delta updates.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// The update process:
    /// 1. Update base and LoRA parameters (via base class)
    /// 2. Update velocity with momentum: velocity = momentum * velocity + (1 - momentum) * gradient
    /// 3. Update delta weights: delta_weights -= learning_rate * velocity
    /// </para>
    /// <para><b>For Beginners:</b> This is where the momentum magic happens!
    ///
    /// Without momentum:
    /// - Updates can be jerky and unstable
    /// - Training might oscillate around the optimum
    ///
    /// With momentum:
    /// - Velocity builds up in consistent gradient directions (speeds up convergence)
    /// - Velocity dampens in inconsistent directions (reduces oscillation)
    /// - Results in smoother, faster convergence
    ///
    /// Think of it like pushing a shopping cart: if you keep pushing in the same direction,
    /// it picks up speed (momentum). If you change direction, it slows down first.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update base and LoRA parameters via base class
        base.UpdateParameters(learningRate);

        // Update delta weights with momentum
        if (_deltaGradients != null)
        {
            T momentumT = NumOps.FromDouble(_momentumFactor);
            T oneMinusMomentumT = NumOps.FromDouble(1.0 - _momentumFactor);

            for (int i = 0; i < _deltaWeights.Rows; i++)
            {
                for (int j = 0; j < _deltaWeights.Columns; j++)
                {
                    // Update velocity: v = momentum * v + (1 - momentum) * gradient
                    _velocity[i, j] = NumOps.Add(
                        NumOps.Multiply(momentumT, _velocity[i, j]),
                        NumOps.Multiply(oneMinusMomentumT, _deltaGradients[i, j])
                    );

                    // Update delta weights: delta -= learning_rate * velocity
                    _deltaWeights[i, j] = NumOps.Subtract(
                        _deltaWeights[i, j],
                        NumOps.Multiply(learningRate, _velocity[i, j])
                    );
                }
            }
        }
    }

    /// <summary>
    /// Gets the current delta weights matrix.
    /// </summary>
    /// <returns>A copy of the current delta weights.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows you the accumulated changes that Delta-LoRA has learned.
    /// You can use this to:
    /// - Visualize how the model is adapting
    /// - Compare different checkpoints during training
    /// - Understand which connections are changing the most
    /// </para>
    /// </remarks>
    public Matrix<T> GetCurrentDelta()
    {
        Matrix<T> copy = new Matrix<T>(_deltaWeights.Rows, _deltaWeights.Columns);
        for (int i = 0; i < _deltaWeights.Rows; i++)
        {
            for (int j = 0; j < _deltaWeights.Columns; j++)
            {
                copy[i, j] = _deltaWeights[i, j];
            }
        }
        return copy;
    }

    /// <summary>
    /// Gets the current parameters including base layer, LoRA layer, and delta weights.
    /// </summary>
    /// <returns>Vector containing all parameters (base + LoRA + delta weights flattened).</returns>
    /// <remarks>
    /// Parameters are packed in order: [base layer params (if not frozen)], [LoRA params], [delta weights].
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        Vector<T> baseParams = base.GetParameters(); // Base layer + LoRA layer
        Vector<T> allParams = new Vector<T>(ParameterCount);

        int idx = 0;

        // Copy base and LoRA parameters
        for (int i = 0; i < baseParams.Length; i++)
        {
            allParams[idx++] = baseParams[i];
        }

        // Pack delta weights
        for (int i = 0; i < _deltaWeights.Rows; i++)
        {
            for (int j = 0; j < _deltaWeights.Columns; j++)
            {
                allParams[idx++] = _deltaWeights[i, j];
            }
        }

        return allParams;
    }

    /// <summary>
    /// Sets the layer parameters including base layer, LoRA layer, and delta weights.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters.</param>
    /// <exception cref="ArgumentException">Thrown when parameter count doesn't match expected count.</exception>
    /// <remarks>
    /// Parameters must be packed in order: [base layer params (if not frozen)], [LoRA params], [delta weights].
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        int baseLoraCount = base.ParameterCount;

        // Extract base and LoRA parameters
        Vector<T> baseLoraParams = new Vector<T>(baseLoraCount);
        for (int i = 0; i < baseLoraCount; i++)
        {
            baseLoraParams[i] = parameters[i];
        }
        base.SetParameters(baseLoraParams);

        // Extract delta weights
        int idx = baseLoraCount;
        for (int i = 0; i < _deltaWeights.Rows; i++)
        {
            for (int j = 0; j < _deltaWeights.Columns; j++)
            {
                _deltaWeights[i, j] = parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Gets all parameter gradients including base layer, LoRA layer, and delta weight gradients.
    /// </summary>
    /// <returns>Vector containing all gradients.</returns>
    /// <remarks>
    /// <para>
    /// Gradient packing order matches GetParameters:
    /// [base layer gradients (if not frozen)], [LoRA gradients], [delta weight gradients].
    /// </para>
    /// <para><b>For Beginners:</b> This packs all the gradients computed during backpropagation
    /// so optimizers can update all parameters consistently. Without this override, optimizers
    /// would miss the delta weight gradients, causing them to never update correctly.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        Vector<T> baseGrads = base.GetParameterGradients(); // Base layer + LoRA layer gradients
        Vector<T> allGrads = new Vector<T>(ParameterCount);

        int idx = 0;

        // Copy base and LoRA gradients
        for (int i = 0; i < baseGrads.Length; i++)
        {
            allGrads[idx++] = baseGrads[i];
        }

        // Pack delta weight gradients
        if (_deltaGradients != null)
        {
            for (int i = 0; i < _deltaGradients.Rows; i++)
            {
                for (int j = 0; j < _deltaGradients.Columns; j++)
                {
                    allGrads[idx++] = _deltaGradients[i, j];
                }
            }
        }
        else
        {
            // If no gradients computed yet, fill with zeros
            int deltaCount = _deltaWeights.Rows * _deltaWeights.Columns;
            for (int i = 0; i < deltaCount; i++)
            {
                allGrads[idx++] = NumOps.Zero;
            }
        }

        return allGrads;
    }

    /// <summary>
    /// Merges the LoRA adaptation and delta weights into the base layer.
    /// </summary>
    /// <returns>A new layer with LoRA and delta weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method merges three components:
    /// 1. Base layer weights (original)
    /// 2. LoRA weights (low-rank adaptation)
    /// 3. Delta weights (momentum-accumulated changes, scaled by deltaScaling)
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" all the adaptations to create a single efficient layer.
    ///
    /// The final weights include:
    /// - Original pre-trained weights
    /// - + LoRA adaptations (B × A matrices)
    /// - + Delta weights (accumulated changes × scaling factor)
    ///
    /// After merging:
    /// - Faster inference (single layer instead of three components)
    /// - Simpler deployment (no need for special LoRA code)
    /// - Preserves all the learned adaptations
    ///
    /// This is typically done after training is complete and you want to deploy the model.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("DeltaLoRAAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        T deltaScalingT = NumOps.FromDouble(_deltaScaling);

        // Merge weights: base + LoRA + delta * scaling
        // DenseLayer uses [inputSize, outputSize] convention
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / outputSize;
            int col = i % outputSize;

            // Start with base weight
            T merged = baseParams[i];

            // Add LoRA contribution
            merged = NumOps.Add(merged, loraWeights[row, col]);

            // Add scaled delta contribution
            merged = NumOps.Add(merged, NumOps.Multiply(_deltaWeights[row, col], deltaScalingT));

            mergedParams[i] = merged;
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Resets the internal state including delta weights, velocity, and cached inputs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears all temporary state but preserves learned parameters.
    /// Use this when starting to process a completely new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        base.ResetState();
        _lastInput = null;
        _deltaGradients = null;
    }
}
