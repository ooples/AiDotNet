namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A neural network layer that adds random Gaussian noise to inputs during training.
/// </summary>
/// <remarks>
/// Gaussian noise layers help prevent overfitting by adding random noise to the input data.
/// This forces the network to learn more robust features that can withstand small variations.
/// The noise follows a Gaussian (normal) distribution with a specified mean and standard deviation.
/// 
/// This layer is typically used:
/// - As a regularization technique to improve generalization
/// - To make the model more robust to input variations
/// - Early in the network, often after the input layer
/// 
/// During inference (testing/prediction), no noise is added to preserve predictable outputs.
/// </remarks>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
public class GaussianNoiseLayer<T> : LayerBase<T>
{
    private readonly T _mean;
    private readonly T _standardDeviation;
    private Tensor<T>? _lastNoise;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <remarks>
    /// This property returns false because the GaussianNoiseLayer doesn't have any
    /// trainable parameters that need to be updated during backpropagation.
    /// It simply adds noise during the forward pass in training mode.
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes a new instance of the GaussianNoiseLayer class.
    /// </summary>
    /// <param name="inputShape">The shape of the input data (e.g., [height, width, channels]).</param>
    /// <param name="standardDeviation">The standard deviation of the Gaussian noise (default: 0.1).</param>
    /// <param name="mean">The mean of the Gaussian noise (default: 0).</param>
    /// <remarks>
    /// The standard deviation controls how much noise is added:
    /// - Higher values create more noise and stronger regularization
    /// - Lower values create less noise and weaker regularization
    /// 
    /// The mean is typically set to 0 to ensure the noise doesn't shift the data distribution.
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
    /// During training mode:
    /// - Random Gaussian noise is generated with the specified mean and standard deviation
    /// - The noise is added to the input tensor
    /// - The noise is stored for potential use in the backward pass
    /// 
    /// During inference mode (testing/prediction):
    /// - No noise is added, and the input passes through unchanged
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (IsTrainingMode)
        {
            _lastNoise = GenerateNoise(input.Shape);
            return input.Add(_lastNoise);
        }

        return input; // During inference, no noise is added
    }

    /// <summary>
    /// Performs the backward pass by passing the gradient unchanged.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// Since adding noise is a simple element-wise operation, the gradient flows through unchanged.
    /// This means that during backpropagation, this layer simply passes the gradient as-is
    /// to the previous layer without modifying it.
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // The gradient flows through unchanged
        return outputGradient;
    }

    private Tensor<T> GenerateNoise(int[] shape)
    {
        var noise = new Tensor<T>(shape);
        for (int i = 0; i < noise.Length; i++)
        {
            T u1 = NumOps.FromDouble(Random.NextDouble());
            T u2 = NumOps.FromDouble(Random.NextDouble());
            T z = NumOps.Multiply(
                NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(-2.0), NumOps.Log(u1))),
                MathHelper.Cos(NumOps.Multiply(NumOps.FromDouble(2.0 * Math.PI), u2))
            );
            noise[i] = NumOps.Add(_mean, NumOps.Multiply(_standardDeviation, z));
        }

        return noise;
    }

    /// <summary>
    /// Updates the layer's parameters using the computed gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// This method is empty because the GaussianNoiseLayer has no trainable parameters to update.
    /// It's implemented to satisfy the base class interface requirements.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update for this layer
    }

    /// <summary>
    /// Gets all trainable parameters of the layer.
    /// </summary>
    /// <returns>An empty vector since this layer has no trainable parameters.</returns>
    /// <remarks>
    /// This method returns an empty vector because the GaussianNoiseLayer doesn't have
    /// any trainable parameters. It's implemented to satisfy the base class interface requirements.
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
    /// This method clears the cached noise tensor from the forward pass.
    /// It's typically called when starting a new training epoch or when switching
    /// between training and inference modes.
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastNoise = null;
    }
}