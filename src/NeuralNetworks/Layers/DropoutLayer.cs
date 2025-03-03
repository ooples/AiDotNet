namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a dropout layer for neural networks to prevent overfitting.
/// </summary>
/// <remarks>
/// Dropout is a regularization technique that randomly sets a fraction of input units to zero
/// during training, which helps prevent neural networks from overfitting. During testing,
/// no dropout is applied.
/// </remarks>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
public class DropoutLayer<T> : LayerBase<T>
{
    private readonly T _dropoutRate;
    private readonly T _scale;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _dropoutMask;

    /// <summary>
    /// Gets a value indicating whether this layer supports training mode.
    /// </summary>
    /// <remarks>
    /// Dropout layers behave differently during training versus inference,
    /// so this property always returns true.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="DropoutLayer{T}"/> class.
    /// </summary>
    /// <param name="dropoutRate">
    /// The probability of dropping out a neuron during training, between 0 and 1.
    /// A value of 0.5 means 50% of neurons will be randomly dropped during training.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the dropout rate is not between 0 and 1.
    /// </exception>
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
    /// During training, each neuron has a probability of <see cref="_dropoutRate"/> of being set to zero.
    /// The remaining neurons are scaled up by 1/(1-dropoutRate) to maintain the expected sum.
    /// During inference (when IsTrainingMode is false), no neurons are dropped and the input passes through unchanged.
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
    /// During training, gradients are only passed through for neurons that weren't dropped during the forward pass.
    /// During inference, gradients pass through unchanged.
    /// </remarks>
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
    /// This method has no effect for dropout layers as they have no trainable parameters.
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
    public override Vector<T> GetParameters()
    {
        // Dropout layer has no trainable parameters
        return new Vector<T>(0);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// Clears the cached input and dropout mask from previous forward and backward passes.
    /// This is typically called between training epochs or when switching between training and inference.
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _dropoutMask = null;
    }
}