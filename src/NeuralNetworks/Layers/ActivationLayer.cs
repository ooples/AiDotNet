namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A layer that applies an activation function to transform the input data.
/// <para>
/// Activation functions introduce non-linearity to neural networks. Non-linearity means the output isn't 
/// simply proportional to the input (like y = 2x). Instead, it can follow curves or more complex patterns.
/// Without non-linearity, a neural network—no matter how many layers—would behave just like a single layer,
/// severely limiting what it can learn.
/// </para>
/// <para>
/// Common activation functions include:
/// - ReLU: Returns 0 for negative inputs, or the input value for positive inputs
/// - Sigmoid: Squashes values between 0 and 1, useful for probabilities
/// - Tanh: Similar to sigmoid but outputs values between -1 and 1
/// </para>
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (like float, double, etc.)</typeparam>
public class ActivationLayer<T> : LayerBase<T>
{
    private Tensor<T>? _lastInput;
    private readonly bool _useVectorActivation;

    /// <summary>
    /// Indicates whether this layer has trainable parameters.
    /// <para>
    /// Always returns false because activation layers don't have parameters to train.
    /// Unlike layers such as Dense/Convolutional layers which have weights and biases
    /// that need updating during training, activation layers simply apply a fixed
    /// mathematical function to their inputs.
    /// </para>
    /// </summary>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Creates a new activation layer that applies a scalar activation function to each value individually.
    /// <para>
    /// A scalar activation function processes each number in your data independently.
    /// For example, if you have an input tensor with 100 values, the function will be applied
    /// 100 times, once to each value, without considering the other values.
    /// </para>
    /// <para>
    /// This is appropriate for most common activation functions like ReLU, Sigmoid, and Tanh.
    /// </para>
    /// </summary>
    /// <param name="inputShape">The shape (dimensions) of the input data, such as [batchSize, height, width, channels]</param>
    /// <param name="activationFunction">The activation function to apply (like ReLU, Sigmoid, etc.)</param>
    public ActivationLayer(int[] inputShape, IActivationFunction<T> activationFunction)
        : base(inputShape, inputShape, activationFunction)
    {
        _useVectorActivation = false;
    }

    /// <summary>
    /// Creates a new activation layer that applies a vector activation function to the entire tensor at once.
    /// <para>
    /// A vector activation function needs to consider multiple values together when processing.
    /// For example, the Softmax function needs to know all values in a vector to calculate
    /// the normalized probabilities across all elements.
    /// </para>
    /// <para>
    /// This type of activation is typically used for:
    /// - Softmax: Converts a vector of numbers into probabilities that sum to 1
    /// - Attention mechanisms: Where relationships between different positions in a sequence matter
    /// - Normalization functions: That need to consider statistics across multiple values
    /// </para>
    /// </summary>
    /// <param name="inputShape">The shape (dimensions) of the input data, such as [batchSize, height, width, channels]</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply</param>
    public ActivationLayer(int[] inputShape, IVectorActivationFunction<T> vectorActivationFunction)
        : base(inputShape, inputShape, vectorActivationFunction)
    {
        _useVectorActivation = true;
    }

    /// <summary>
    /// Processes the input data by applying the activation function.
    /// <para>
    /// This is called during the forward pass of the neural network, which is when
    /// data flows from the input layer through all hidden layers to the output layer.
    /// The forward pass is used both during training and when making predictions with a trained model.
    /// </para>
    /// <para>
    /// For example, if using ReLU activation, this method would replace all negative values in the input
    /// with zeros while keeping positive values unchanged.
    /// </para>
    /// </summary>
    /// <param name="input">The input data to process</param>
    /// <returns>The transformed data after applying the activation function</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        return _useVectorActivation ? ApplyVectorActivation(input) : ApplyScalarActivation(input);
    }

    /// <summary>
    /// Calculates how changes in the output affect the input during training.
    /// <para>
    /// This is called during the backward pass (backpropagation) when training the neural network.
    /// Backpropagation is the algorithm that determines how much each neuron contributed to the error
    /// in the network's prediction, allowing the network to adjust its parameters to reduce future errors.
    /// </para>
    /// <para>
    /// For activation layers, the backward pass calculates how the gradient (rate of change) of the error
    /// with respect to the layer's output should be modified to get the gradient with respect to the layer's input.
    /// This involves applying the derivative of the activation function.
    /// </para>
    /// <para>
    /// For example, with ReLU activation, the derivative is 1 for inputs that were positive, and 0 for inputs
    /// that were negative or zero. This means the gradient flows unchanged through positive activations
    /// but gets blocked (multiplied by zero) for negative activations.
    /// </para>
    /// </summary>
    /// <param name="outputGradient">How much the network's error changes with respect to this layer's output</param>
    /// <returns>How much the network's error changes with respect to this layer's input</returns>
    /// <exception cref="ForwardPassRequiredException">Thrown if called before Forward method</exception>
    /// <exception cref="TensorShapeMismatchException">Thrown if the gradient shape doesn't match the input shape</exception>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new ForwardPassRequiredException("ActivationLayer", GetType().Name);

        TensorValidator.ValidateShapesMatch(_lastInput, outputGradient, "Activation Layer", "Backward Pass");

        return _useVectorActivation 
            ? BackwardVectorActivation(outputGradient) 
            : BackwardScalarActivation(outputGradient);
    }

    private Tensor<T> ApplyScalarActivation(Tensor<T> input)
    {
        return input.Transform((x, _) => ScalarActivation!.Activate(x));
    }

    private Tensor<T> ApplyVectorActivation(Tensor<T> input)
    {
        return VectorActivation!.Activate(input);
    }

    private Tensor<T> BackwardScalarActivation(Tensor<T> outputGradient)
    {
        return _lastInput!.Transform((x, indices) => 
            NumOps.Multiply(ScalarActivation!.Derivative(x), outputGradient[indices]));
    }

    private Tensor<T> BackwardVectorActivation(Tensor<T> outputGradient)
    {
        return VectorActivation!.Derivative(_lastInput!) * outputGradient;
    }

    /// <summary>
    /// Updates the layer's internal parameters during training.
    /// <para>
    /// This method is part of the training process where layers adjust their parameters
    /// (weights and biases) based on the gradients calculated during backpropagation.
    /// </para>
    /// <para>
    /// For activation layers, this method does nothing because they have no trainable parameters.
    /// Unlike layers such as Dense layers which need to update their weights and biases,
    /// activation layers simply apply a fixed mathematical function.
    /// </para>
    /// </summary>
    /// <param name="learningRate">How quickly the network should learn from new data. Higher values mean bigger parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // Activation layer has no parameters to update
    }

    /// <summary>
    /// Gets all trainable parameters of this layer as a flat vector.
    /// <para>
    /// This method is useful for operations that need to work with all parameters at once,
    /// such as certain optimization algorithms, regularization techniques, or when saving a model.
    /// </para>
    /// <para>
    /// Returns an empty vector since activation layers have no trainable parameters.
    /// Other layer types like Dense layers would return their weights and biases.
    /// </para>
    /// </summary>
    /// <returns>An empty vector representing the layer's parameters</returns>
    public override Vector<T> GetParameters()
    {
        // Activation layers don't have parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Clears the layer's memory of previous inputs.
    /// <para>
    /// Neural networks maintain state between operations, especially during training.
    /// This method resets that state, which is useful in several scenarios:
    /// - When starting to process a new batch of data
    /// - Between training epochs
    /// - When switching from training to evaluation mode
    /// - When you want to ensure the layer behaves deterministically
    /// </para>
    /// <para>
    /// For activation layers, this means forgetting the last input that was processed,
    /// which was stored to help with the backward pass calculations.
    /// </para>
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
    }
}