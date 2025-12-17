namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A neural network layer that adds random Gaussian noise to inputs during training.
/// </summary>
/// <remarks>
/// <para>
/// Gaussian noise layers help prevent overfitting by adding random noise to the input data.
/// This forces the network to learn more robust features that can withstand small variations.
/// The noise follows a Gaussian (normal) distribution with a specified mean and standard deviation.
/// During inference (testing/prediction), no noise is added to preserve predictable outputs.
/// </para>
/// <para><b>For Beginners:</b> This layer adds random "static" to your data during training to make the network more robust.
/// 
/// Think of it like training an athlete in challenging conditions:
/// - Training in rain and wind makes athletes perform better even in good weather
/// - Training with noise makes neural networks perform better on clean data
/// 
/// For example, in image recognition:
/// - During training: The layer slightly changes pixel values randomly
/// - This forces the network to focus on important patterns, not tiny details
/// - During testing/prediction: No noise is added, giving clean results
/// 
/// Gaussian noise is particularly useful because it follows the same distribution
/// as many natural variations in real-world data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
public class GaussianNoiseLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The mean (average value) of the Gaussian noise distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field defines the center of the Gaussian distribution used to generate noise.
    /// A mean of 0 ensures that the noise adds and subtracts values symmetrically,
    /// without shifting the overall data distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the center point of the random noise.
    /// 
    /// The mean value:
    /// - Is typically set to 0 for noise layers
    /// - With mean = 0, positive and negative noise values are equally likely
    /// - This ensures the noise doesn't systematically shift your data in one direction
    /// 
    /// Think of it like balancing noise around zero:
    /// - Some values get slightly increased
    /// - Some values get slightly decreased
    /// - On average, the data stays centered at the same place
    /// </para>
    /// </remarks>
    private readonly T _mean;

    /// <summary>
    /// The standard deviation of the Gaussian noise distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field defines the spread of the Gaussian distribution used to generate noise.
    /// A higher standard deviation results in larger noise values and stronger regularization,
    /// while a lower standard deviation produces smaller noise values and weaker regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strong the random noise is.
    /// 
    /// The standard deviation:
    /// - Determines how far from the mean the noise typically varies
    /// - Larger values create stronger noise (more regularization)
    /// - Smaller values create milder noise (less regularization)
    /// 
    /// For example:
    /// - Standard deviation = 0.1: Noise typically varies by about ±10% of the data range
    /// - Standard deviation = 0.5: Noise typically varies by about ±50% of the data range
    /// 
    /// Finding the right amount of noise is important:
    /// - Too much noise can prevent learning
    /// - Too little noise won't help prevent overfitting
    /// </para>
    /// </remarks>
    private readonly T _standardDeviation;

    /// <summary>
    /// The noise tensor from the last forward pass, saved for potential use in backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the noise tensor that was generated and used during the last forward pass.
    /// It is saved in case it's needed for the backward pass, though in simple additive noise,
    /// the gradient flows through unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what noise was added in the latest calculation.
    /// 
    /// During training:
    /// - The layer needs to keep track of exactly what noise was added
    /// - This helps ensure consistent behavior during backpropagation
    /// - It's like keeping a record of what "challenges" were introduced
    /// 
    /// This tensor has the same shape as the input data, with each position
    /// containing the specific noise value that was added at that position.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastNoise;
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>false</c> because the GaussianNoiseLayer doesn't have any trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the GaussianNoiseLayer doesn't have trainable parameters that need
    /// to be updated during backpropagation. The layer simply adds noise during the forward pass in
    /// training mode, but it doesn't learn or adapt based on the data.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer doesn't learn or change during training.
    /// 
    /// A value of false means:
    /// - The layer has no weights or biases to adjust
    /// - It performs a fixed operation (adding noise) rather than learning
    /// - It's a helper layer that assists the learning process of other layers
    /// 
    /// Unlike layers like convolutional or fully connected layers that learn patterns from data,
    /// the noise layer simply adds randomness with fixed statistical properties.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="GaussianNoiseLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input data (e.g., [height, width, channels]).</param>
    /// <param name="standardDeviation">The standard deviation of the Gaussian noise (default: 0.1).</param>
    /// <param name="mean">The mean of the Gaussian noise (default: 0).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Gaussian noise layer with the specified input shape,
    /// standard deviation, and mean. The output shape is the same as the input shape since
    /// the layer only adds noise element-wise without changing the dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the noise layer with your desired noise level.
    /// 
    /// When creating a Gaussian noise layer, you need to specify:
    /// - Input shape: The dimensions of your data
    /// - Standard deviation: How strong the noise should be (default: 0.1)
    /// - Mean: The center point of the noise distribution (default: 0)
    /// 
    /// For example:
    /// ```csharp
    /// // Add mild noise to 28×28 grayscale images
    /// var noiseLayer = new GaussianNoiseLayer<float>(new int[] { 28, 28, 1 }, 0.1);
    /// 
    /// // Add stronger noise to 32×32 color images
    /// var strongerNoise = new GaussianNoiseLayer<float>(new int[] { 32, 32, 3 }, 0.3);
    /// ```
    /// 
    /// The standard deviation is the most important parameter as it controls
    /// the strength of the regularization effect.
    /// </para>
    /// </remarks>
    public GaussianNoiseLayer(
        int[] inputShape,
        double standardDeviation = 0.1,
        double mean = 0)
        : base(inputShape, inputShape)
    {
        _mean = NumOps.FromDouble(mean);
        _standardDeviation = NumOps.FromDouble(standardDeviation);
    }

    /// <summary>
    /// Performs the forward pass by adding Gaussian noise to the input during training.
    /// </summary>
    /// <param name="input">The input tensor to the layer.</param>
    /// <returns>The input tensor with added noise during training, or unchanged during inference.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Gaussian noise layer. During training mode,
    /// it generates random Gaussian noise with the specified mean and standard deviation and adds it
    /// to the input tensor. During inference mode, it simply passes the input through unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer adds random noise during training.
    /// 
    /// The forward pass works differently depending on the mode:
    /// 
    /// During training mode:
    /// 1. Generate random noise following a Gaussian distribution
    /// 2. Add this noise to the input data
    /// 3. Save the noise for potential use in backward pass
    /// 4. Return the noisy data
    /// 
    /// During testing/prediction mode:
    /// 1. Simply pass the input through unchanged
    /// 2. No noise is added to ensure consistent results
    /// 
    /// This behavior is what makes noise layers useful for regularization:
    /// They make training more difficult but don't affect the final predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        if (IsTrainingMode)
        {
            _lastNoise = GenerateNoise(input.Shape);
            return Engine.TensorAdd(input, _lastNoise);
        }

        return input; // During inference, no noise is added
    }

    /// <summary>
    /// Performs the backward pass by passing the gradient unchanged.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass (backpropagation) of the Gaussian noise layer.
    /// Since adding noise is a simple element-wise operation, the gradient flows through unchanged.
    /// This means that during backpropagation, this layer simply passes the gradient as-is
    /// to the previous layer without modifying it.
    /// </para>
    /// <para><b>For Beginners:</b> This is where error information flows back through the layer during training.
    /// 
    /// During the backward pass:
    /// - The layer receives gradients (information about how to improve)
    /// - Since noise was just added element-wise, gradients flow through directly
    /// - No changes are made to the gradients
    /// 
    /// This is different from layers with parameters (like weights and biases):
    /// - Those layers would compute how to adjust their parameters
    /// - This layer has no parameters to adjust
    /// 
    /// The noise affected the forward pass, but during backpropagation,
    /// the gradients flow through unmodified.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        // The gradient flows through unchanged
        return outputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// Gaussian noise is added independently to the input during forward pass. During backward pass,
    /// the gradient simply flows through unchanged because the noise doesn't depend on the input.
    /// This is equivalent to an identity operation in the computation graph.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Create a simple computation graph where output = input + noise
        // Since noise is independent, gradient just flows through
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // In autodiff terms, this is just an identity operation
        // The forward pass adds noise, but noise doesn't depend on input, so gradient passes through
        var outputNode = inputNode; // Identity for gradient purposes

        // Set gradient and perform backward pass
        outputNode.Gradient = outputGradient;

        // For a simple identity operation, just return the gradient
        return outputGradient;
    }


    /// <summary>
    /// Generates a tensor of random Gaussian noise with the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the noise tensor to generate.</param>
    /// <returns>A tensor filled with random Gaussian noise.</returns>
    /// <remarks>
    /// <para>
    /// This helper method generates a tensor of random Gaussian noise using the Box-Muller transform.
    /// The Box-Muller transform converts uniform random numbers to Gaussian random numbers.
    /// Each element in the tensor is independently sampled from a Gaussian distribution with
    /// the mean and standard deviation specified in the layer's constructor.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a tensor filled with random noise values.
    /// 
    /// The method:
    /// 1. Creates an empty tensor with the same shape as the input
    /// 2. Fills it with random values that follow a bell curve (Gaussian) distribution
    /// 3. Uses the "Box-Muller transform" - a mathematical technique for generating Gaussian random numbers
    /// 
    /// Each value in the tensor:
    /// - Is centered around the specified mean (typically 0)
    /// - Has a spread determined by the standard deviation
    /// - Is generated independently from all other values
    /// 
    /// The result is a tensor of random noise that can be added to the input data.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateNoise(int[] shape)
    {
        // Use Engine to generate Gaussian noise in a vectorized manner
        var noiseVector = Engine.GenerateGaussianNoise(new Tensor<T>(shape).Length, _mean, _standardDeviation);
        return new Tensor<T>(shape, noiseVector);
    }

    /// <summary>
    /// Updates the parameters of the layer based on the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is a required override from the base class, but the Gaussian noise layer has no
    /// trainable parameters to update, so it performs no operation.
    /// </para>
    /// <para><b>For Beginners:</b> This method does nothing because noise layers have no adjustable weights.
    /// 
    /// Unlike most layers (like convolutional or fully connected layers):
    /// - Gaussian noise layers don't have weights or biases to learn
    /// - They just add random noise based on fixed settings
    /// - There's nothing to update during training
    /// 
    /// This method exists only to fulfill the requirements of the base layer class.
    /// The noise layer influences the network by making training more robust,
    /// not by adjusting internal parameters.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update for this layer
    }

    /// <summary>
    /// Gets the trainable parameters of the layer.
    /// </summary>
    /// <returns>
    /// An empty vector since this layer has no trainable parameters.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method is a required override from the base class, but the Gaussian noise layer has no
    /// trainable parameters to retrieve, so it returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because noise layers have no learnable values.
    /// 
    /// Unlike layers with weights and biases:
    /// - Gaussian noise layers don't have any parameters that change during training
    /// - They perform a fixed operation (adding noise) that doesn't involve learning
    /// - There are no values to save when storing a trained model
    /// 
    /// This method returns an empty vector, indicating there are no parameters to collect.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // GaussianNoiseLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached noise tensor from
    /// the previous forward pass. This is useful when starting to process a new batch of data
    /// or when switching between training and inference modes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    ///
    /// When resetting the state:
    /// - The saved noise tensor is cleared
    /// - This frees up memory
    /// - The layer will generate new random noise next time
    ///
    /// This is typically called:
    /// - Between training batches
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    ///
    /// It's like wiping a whiteboard clean before starting a new experiment.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastNoise = null;
        _lastInput = null;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because the JIT-compiled version uses inference mode (no noise added).
    /// </value>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the Gaussian noise layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node (same as input for inference mode).</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph for the Gaussian noise layer. During JIT compilation
    /// (which is typically for inference), no noise is added, so the layer simply passes through
    /// the input unchanged. This matches the behavior of Forward() when IsTrainingMode is false.
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create placeholder for input data
        var inputPlaceholder = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(inputPlaceholder, "input");

        inputNodes.Add(inputNode);

        // For JIT compilation (inference mode), Gaussian noise layer is identity: output = input
        // No noise is added during inference
        return inputNode;
    }
}
