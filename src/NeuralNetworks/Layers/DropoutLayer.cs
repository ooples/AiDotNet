namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a dropout layer for neural networks to prevent overfitting.
/// </summary>
/// <remarks>
/// <para>
/// Dropout is a regularization technique that randomly deactivates a fraction of neurons during
/// training, which helps prevent neural networks from overfitting. Overfitting occurs when a model
/// learns patterns that are specific to the training data but don't generalize well to new data.
/// </para>
/// <para><b>For Beginners:</b> Dropout is like randomly turning off some brain cells during training to make the network more robust.
/// 
/// Imagine a team that always practices together:
/// - They might develop specific patterns that only work with familiar teammates
/// - If some players are absent, the team struggles
/// 
/// Dropout forces the network to work even when some neurons are missing:
/// - During training, random neurons are turned off (set to zero)
/// - This prevents any single neuron from becoming too important
/// - The network learns multiple ways to solve the same problem
/// - It's like practicing with different team combinations each time
/// 
/// During actual use (inference), all neurons are active, but their outputs are slightly reduced
/// to compensate for having more active neurons than during training.
/// 
/// This technique significantly reduces overfitting, which is when a network gets too specialized
/// to its training data and performs poorly on new data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
public class DropoutLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The probability of dropping out (deactivating) a neuron during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the probability (between 0 and 1) that any given neuron will be dropped out 
    /// (set to zero) during the forward pass in training mode. A value of 0.5 means each neuron has a 
    /// 50% chance of being dropped.
    /// </para>
    /// <para><b>For Beginners:</b> This is how likely each neuron is to be turned off during training.
    /// 
    /// Think of it like rolling a dice for each neuron:
    /// - A dropout rate of 0.2 means a 20% chance of turning off each neuron
    /// - A dropout rate of 0.5 means a 50% chance (like flipping a coin)
    /// 
    /// Common values:
    /// - Small networks: 0.2 to 0.3 (turning off 20-30% of neurons)
    /// - Larger networks: 0.4 to 0.5 (turning off 40-50% of neurons)
    /// 
    /// Higher dropout rates provide stronger regularization but might make training more difficult.
    /// Lower dropout rates provide milder regularization.
    /// </para>
    /// </remarks>
    private readonly T _dropoutRate;

    /// <summary>
    /// The scaling factor applied to active neurons during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This scaling factor is applied to the neurons that remain active during training. The value is
    /// 1/(1-dropoutRate) and compensates for the reduced sum of activations caused by dropout.
    /// </para>
    /// <para><b>For Beginners:</b> This is how much to amplify the remaining active neurons.
    /// 
    /// When some neurons are turned off:
    /// - The total signal would be weaker (reduced by the dropout percentage)
    /// - To compensate, we make the remaining neurons stronger
    /// - If we drop 50% of neurons, we make the remaining ones 2� stronger
    /// 
    /// The formula is simple: scale = 1 / (1 - dropout_rate)
    /// 
    /// Examples:
    /// - Dropout rate = 0.2 ? Scale = 1.25 (each remaining neuron is 25% stronger)
    /// - Dropout rate = 0.5 ? Scale = 2.0 (each remaining neuron is twice as strong)
    /// 
    /// This scaling ensures the expected sum of the activations remains the same during
    /// training and inference, which helps with stable learning.
    /// </para>
    /// </remarks>
    private readonly T _scale;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor received during the last forward pass. It is necessary
    /// for computing gradients during the backward pass (backpropagation).
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what input data was processed most recently.
    /// 
    /// During training:
    /// - The layer needs to remember what input it received
    /// - This helps when calculating how to improve the network
    /// - It's like keeping your work when solving a math problem
    /// 
    /// This value is automatically cleared between training batches to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The binary mask that indicates which neurons were kept active during the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the dropout mask generated during the forward pass. Each element is either 0
    /// (for dropped neurons) or the scale value (for active neurons). This mask is used during
    /// backpropagation to ensure gradients only flow through the neurons that were active during the forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This is a record of which neurons were on or off during the last calculation.
    /// 
    /// Think of it like a map:
    /// - It has the same shape as the input data
    /// - Each position has either:
    ///   - 0 (if that neuron was turned off)
    ///   - The scale value (if that neuron was kept on)
    /// 
    /// During backpropagation (learning):
    /// - Only the neurons that were active during the forward pass receive updates
    /// - This ensures consistency between forward and backward passes
    /// 
    /// This mask is randomly generated each time, creating different patterns of active/inactive
    /// neurons for each training example.
    /// </para>
    /// </remarks>
    private Tensor<T>? _dropoutMask;

    /// <summary>
    /// Gets a value indicating whether this layer supports training mode.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because dropout layers need to distinguish between training and inference modes.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the dropout layer behaves differently during training and inference.
    /// During training, neurons are randomly dropped, while during inference all neurons remain active.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the network that the layer behaves differently during training versus actual use.
    /// 
    /// A value of true means:
    /// - The layer needs to know whether it's in training mode or inference mode
    /// - It will apply dropout only during training
    /// - During actual use (inference), all neurons remain active
    /// 
    /// This is important because dropout is only applied during training to create robustness,
    /// but we want to use all available neurons when making actual predictions.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="DropoutLayer{T}"/> class.
    /// </summary>
    /// <param name="dropoutRate">
    /// The probability of dropping out a neuron during training, between 0 and 1.
    /// A value of 0.5 means 50% of neurons will be randomly dropped during training.
    /// Default value is 0.5.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the dropout rate is not between 0 and 1.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new dropout layer with the specified dropout rate. The dropout layer does not
    /// change the shape of the input data; it only modifies the values by setting some of them to zero during
    /// training. The constructor also calculates the scaling factor used to maintain the expected magnitude of
    /// activations.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the dropout layer with the specified dropout probability.
    /// 
    /// When creating a dropout layer, you only need to specify:
    /// - How likely each neuron is to be dropped (turned off) during training
    /// 
    /// For example:
    /// ```csharp
    /// // Create a layer that randomly turns off 30% of neurons during training
    /// var dropoutLayer = new DropoutLayer<float>(0.3);
    /// 
    /// // Create a layer with the default 50% dropout rate
    /// var defaultDropout = new DropoutLayer<float>();
    /// ```
    /// 
    /// The layer will automatically calculate the proper scaling factor to
    /// maintain the same average activation level before and after dropout.
    /// 
    /// The dropout rate must be between 0 and 1 (but not exactly 1), or the constructor
    /// will throw an exception.
    /// </para>
    /// </remarks>
    public DropoutLayer(double dropoutRate = 0.5)
        : base(Array.Empty<int>(), []) // Dropout layer doesn't change the shape of the input
    {
        if (dropoutRate < 0 || dropoutRate >= 1)
            throw new ArgumentException("Dropout rate must be between 0 and 1", nameof(dropoutRate));

        _dropoutRate = NumOps.FromDouble(dropoutRate);
        _scale = NumOps.FromDouble(1.0 / (1.0 - dropoutRate));
    }

    /// <summary>
    /// Performs the forward pass of the dropout layer.
    /// </summary>
    /// <param name="input">The input tensor from the previous layer.</param>
    /// <returns>
    /// During training: A tensor with randomly dropped neurons (set to zero) and the remaining
    /// neurons scaled up to maintain the expected output magnitude.
    /// During inference: The unchanged input tensor (no dropout is applied).
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the dropout layer. During training mode, it randomly
    /// deactivates neurons according to the dropout rate and scales up the remaining neurons by the scaling
    /// factor. During inference mode, the input is passed through unchanged. The method maintains
    /// a dropout mask that records which neurons were kept active for use during backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer processes input data by randomly turning off some neurons.
    /// 
    /// During training:
    /// 1. For each neuron in the input:
    ///    - Randomly decide if it should be active or dropped
    ///    - If dropped: Set its value to zero
    ///    - If kept: Scale its value up (multiply by the scale factor)
    /// 2. Remember which neurons were active in the dropout mask
    /// 
    /// During inference (when not training):
    /// - All neurons remain active
    /// - No scaling is applied
    /// - The input passes through unchanged
    /// 
    /// This random pattern of active/inactive neurons is different for each input,
    /// forcing the network to be resilient and not depend too much on any single neuron.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        if (!IsTrainingMode)
            return input;

        _dropoutMask = new Tensor<T>(input.Shape);
        var output = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Length; i++)
        {
            if (Random.NextDouble() > Convert.ToDouble(_dropoutRate))
            {
                _dropoutMask[i] = _scale;
                output[i] = NumOps.Multiply(input[i], _scale);
            }
            else
            {
                _dropoutMask[i] = NumOps.Zero;
                output[i] = NumOps.Zero;
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the dropout layer, propagating gradients to the previous layer.
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer.</param>
    /// <returns>
    /// The gradient tensor to be passed to the previous layer.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass (backpropagation) of the dropout layer. During training,
    /// it ensures that gradients only flow through the neurons that were active during the forward pass
    /// by applying the same dropout mask. During inference, the gradients pass through unchanged since
    /// no dropout was applied in the forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer passes error information back to previous layers.
    /// 
    /// During training:
    /// - Gradients represent how each neuron should change to improve
    /// - We only want to update neurons that were active during the forward pass
    /// - The dropout mask (which recorded which neurons were on/off) is applied to the gradients
    /// - Dropped neurons receive zero gradient (no update)
    /// - Active neurons receive the scaled gradient
    /// 
    /// During inference:
    /// - All gradients pass through unchanged
    /// - This matches the behavior of the forward pass where all neurons were active
    /// 
    /// This consistency between forward and backward passes is essential for proper training.
    /// </para>
    /// <exception cref="InvalidOperationException">
    /// Thrown when backward is called before a forward pass has been performed.
    /// </exception>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _dropoutMask == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (!IsTrainingMode)
            return outputGradient;

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int i = 0; i < outputGradient.Length; i++)
        {
            inputGradient[i] = NumOps.Multiply(outputGradient[i], _dropoutMask[i]);
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the layer based on the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is a required override from the base class, but the dropout layer has no
    /// trainable parameters to update, so it performs no operation.
    /// </para>
    /// <para><b>For Beginners:</b> This method does nothing for dropout layers because they have no adjustable weights.
    /// 
    /// Unlike most layers (like convolutional or dense layers):
    /// - Dropout layers don't have weights or biases to learn
    /// - They just apply a random on/off pattern and scaling
    /// - There's nothing to update during training
    /// 
    /// This method exists only to fulfill the requirements of the base layer class.
    /// The dropout layer participates in training by modifying activations and gradients,
    /// not by updating internal parameters.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Dropout layer has no parameters to update
    }

    /// <summary>
    /// Gets the trainable parameters of the layer.
    /// </summary>
    /// <returns>
    /// An empty vector since dropout layers have no trainable parameters.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method is a required override from the base class, but the dropout layer has no
    /// trainable parameters to retrieve, so it returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because dropout layers have no learnable values.
    /// 
    /// Unlike layers with weights and biases:
    /// - Dropout layers don't have any parameters that change during training
    /// - The dropout rate and scale are fixed when the layer is created
    /// - There are no values to save when storing a trained model
    /// 
    /// This method returns an empty vector (a vector of length zero),
    /// indicating there are no parameters to collect.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Dropout layer has no trainable parameters
        return new Vector<T>(0);
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method is not shown in the original code, but would typically be implemented to match
    /// the GetParameters method. For a dropout layer, it would accept an empty vector since there
    /// are no parameters to set.
    /// </para>
    /// <para><b>For Beginners:</b> This method would do nothing because dropout layers have no adjustable parameters.
    /// 
    /// Since dropout layers don't have learnable parameters:
    /// - There's nothing to set or update
    /// - The method would only verify that the input is an empty vector
    /// 
    /// This method would exist only to fulfill the contract of the base layer class.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        // Dropout layer has no parameters to set
        if (parameters.Length != 0)
        {
            throw new ArgumentException($"Expected 0 parameters, but got {parameters.Length}");
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached input and dropout mask
    /// from previous forward and backward passes. This is useful when starting to process a new batch of
    /// data or when switching between training and inference modes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input and dropout mask are cleared
    /// - The layer forgets previous calculations it performed
    /// - This frees up memory and prepares for new data
    /// 
    /// This is typically called:
    /// - Between training batches
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    /// 
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _dropoutMask = null;
    }
}