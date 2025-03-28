namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a customizable layer that applies user-defined functions for both forward and backward passes.
/// </summary>
/// <remarks>
/// <para>
/// The Lambda Layer allows for custom transformations to be incorporated into a neural network by accepting
/// user-defined functions for both the forward and backward passes. This provides flexibility to implement
/// custom operations that aren't available as standard layers. The layer can optionally apply an activation
/// function after the custom transformation.
/// </para>
/// <para><b>For Beginners:</b> This layer lets you create your own custom operations in a neural network.
/// 
/// Think of the Lambda Layer as a "do-it-yourself" layer where:
/// - You provide your own custom function to process the data
/// - You can optionally provide a custom function for the learning process
/// - It gives you flexibility to implement operations not covered by standard layers
/// 
/// For example, if you wanted to apply a special mathematical transformation that isn't
/// available in standard layers, you could define that transformation and use it in a Lambda Layer.
/// 
/// This is an advanced feature that gives you complete control when standard layers
/// don't provide what you need.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LambdaLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The user-provided function that defines the forward pass transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This function takes an input tensor and returns a transformed output tensor. It defines the custom
    /// transformation that will be applied to the input data during the forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This is your custom function that processes the data.
    /// 
    /// The forward function:
    /// - Takes in your data
    /// - Applies your custom transformation
    /// - Returns the processed data
    /// 
    /// This is where you define exactly what you want this layer to do with the data.
    /// For example, you might create a function that scales certain features,
    /// combines features in a special way, or applies a custom mathematical formula.
    /// </para>
    /// </remarks>
    private readonly Func<Tensor<T>, Tensor<T>> _forwardFunction;

    /// <summary>
    /// The optional user-provided function that defines the backward pass transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This function takes the input tensor and the gradient tensor from the next layer, and returns
    /// the gradient with respect to the input of this layer. It defines how gradients should flow
    /// backward through this custom transformation during training.
    /// </para>
    /// <para><b>For Beginners:</b> This optional function handles the learning process for your custom layer.
    /// 
    /// The backward function:
    /// - Takes the original input and information about errors from later layers
    /// - Calculates how to adjust the input to reduce these errors
    /// - Is necessary if you want your network to learn through this custom layer
    /// 
    /// If you don't provide this function, the layer cannot participate in training,
    /// meaning that while it will transform data, the network cannot learn to optimize this transformation.
    /// 
    /// Writing this function correctly requires understanding of calculus and backpropagation.
    /// </para>
    /// </remarks>
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>>? _backwardFunction;

    /// <summary>
    /// Stores the input tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the output tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> if a backward function is provided; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The LambdaLayer supports training only if a backward function is provided.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its behavior during training
    /// - A backward function has been provided
    /// - It participates in the learning process
    /// 
    /// A value of false means:
    /// - No backward function was provided
    /// - The layer will always apply the same transformation
    /// - It doesn't participate in the learning process
    /// 
    /// This is determined by whether you provided a backward function when creating the layer.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _backwardFunction != null;

    /// <summary>
    /// Initializes a new instance of the <see cref="LambdaLayer{T}"/> class with the specified shapes, functions, and element-wise activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <param name="forwardFunction">The function to apply during the forward pass.</param>
    /// <param name="backwardFunction">The optional function to apply during the backward pass. If null, the layer will not support training.</param>
    /// <param name="activationFunction">The activation function to apply after the custom transformation. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Lambda Layer with the specified shapes, functions, and element-wise activation function.
    /// The input and output shapes must be specified as they may differ depending on the custom transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new custom layer with your functions.
    /// 
    /// When creating a Lambda Layer, you specify:
    /// - inputShape: The shape of the data that will come into your layer
    /// - outputShape: The shape of the data that will come out of your layer
    /// - forwardFunction: Your custom function that processes the data
    /// - backwardFunction (optional): Your custom function for learning
    /// - activationFunction (optional): A standard function to apply after your custom transformation
    /// 
    /// For example, if you have data with 10 features and want to transform it into 5 features,
    /// you would use inputShape=[10] and outputShape=[5], and provide a function that performs this transformation.
    /// </para>
    /// </remarks>
    public LambdaLayer(int[] inputShape, int[] outputShape, 
                       Func<Tensor<T>, Tensor<T>> forwardFunction, 
                       Func<Tensor<T>, Tensor<T>, Tensor<T>>? backwardFunction = null,
                       IActivationFunction<T>? activationFunction = null)
        : base(inputShape, outputShape, activationFunction ?? new ReLUActivation<T>())
    {
        _forwardFunction = forwardFunction;
        _backwardFunction = backwardFunction;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LambdaLayer{T}"/> class with the specified shapes, functions, and vector activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <param name="forwardFunction">The function to apply during the forward pass.</param>
    /// <param name="backwardFunction">The optional function to apply during the backward pass. If null, the layer will not support training.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after the custom transformation. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Lambda Layer with the specified shapes, functions, and vector activation function.
    /// Vector activation functions operate on entire vectors rather than individual elements, which can capture
    /// dependencies between different elements of the vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new custom layer with an advanced vector-based activation.
    /// 
    /// Vector activation functions:
    /// - Process entire groups of numbers together, not just one at a time
    /// - Can capture relationships between different features
    /// - May be more powerful for complex patterns
    /// 
    /// This constructor is useful when you need the layer to understand how different
    /// features interact with each other, rather than treating each feature independently.
    /// </para>
    /// </remarks>
    public LambdaLayer(int[] inputShape, int[] outputShape, 
                       Func<Tensor<T>, Tensor<T>> forwardFunction, 
                       Func<Tensor<T>, Tensor<T>, Tensor<T>>? backwardFunction = null,
                       IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShape, outputShape, vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _forwardFunction = forwardFunction;
        _backwardFunction = backwardFunction;
    }

    /// <summary>
    /// Performs the forward pass of the lambda layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after applying the custom transformation and activation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the lambda layer. It applies the user-defined forward function
    /// to the input tensor, followed by the activation function if one was specified. The input and output
    /// are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the custom layer.
    /// 
    /// During the forward pass:
    /// 1. Your custom function processes the input data
    /// 2. If specified, an activation function is applied to add non-linearity
    /// 3. The input and output are saved for use during training
    /// 
    /// This is where your custom transformation actually gets applied to the data
    /// as it flows through the network.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var output = _forwardFunction(input);
    
        if (ScalarActivation != null)
        {
            output = ApplyActivation(output);
        }
        else if (VectorActivation != null)
        {
            output = ApplyVectorActivation(output);
        }

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Performs the backward pass of the lambda layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward or when no backward function is provided.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the lambda layer, which is used during training to propagate
    /// error gradients back through the network. It applies the derivative of the activation function to the
    /// output gradient, then applies the user-defined backward function to compute the gradient with respect to
    /// the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives information about how its output contributed to errors
    /// 2. If an activation function was used, its effect is accounted for
    /// 3. Your custom backward function calculates how the input should change
    /// 
    /// This method will throw an error if:
    /// - The Forward method hasn't been called first
    /// - No backward function was provided when creating the layer
    /// 
    /// Writing a correct backward function requires understanding of calculus and
    /// how gradients flow through neural networks.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        Tensor<T> gradient = outputGradient;

        if (ScalarActivation != null)
        {
            gradient = ApplyActivationDerivative(gradient, _lastOutput);
        }
        else if (VectorActivation != null)
        {
            gradient = ApplyVectorActivationDerivative(gradient, _lastOutput);
        }

        if (_backwardFunction != null)
        {
            gradient = _backwardFunction(_lastInput, gradient);
        }
        else
        {
            throw new InvalidOperationException("Backward function not provided for this Lambda layer.");
        }

        return gradient;
    }

    /// <summary>
    /// Update parameters is a no-op for the lambda layer since it typically doesn't have trainable parameters.
    /// </summary>
    /// <param name="learningRate">The learning rate (unused in this layer).</param>
    /// <remarks>
    /// <para>
    /// This method is implemented as required by the LayerBase interface but typically does nothing for the LambdaLayer
    /// since most custom transformations don't have trainable parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method exists but typically does nothing for Lambda layers.
    /// 
    /// Since Lambda layers:
    /// - Usually don't have their own weights or biases
    /// - Rely on the custom functions you provide
    /// 
    /// This method is included only because all layers must have this method,
    /// but it doesn't usually do anything for Lambda layers. If your custom functions
    /// have parameters that need updating, you would need to handle that separately.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Lambda layers typically don't have trainable parameters
    }

    /// <summary>
    /// Applies the element-wise activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>The activated tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the element-wise activation function to the input tensor. It processes each element
    /// of the tensor individually.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the standard activation function to each value.
    /// 
    /// The method:
    /// - Creates a new tensor to hold the results
    /// - Applies the activation function to each value one at a time
    /// - Returns the new tensor with all values transformed
    /// 
    /// This adds non-linearity to your layer, which helps the network learn complex patterns.
    /// </para>
    /// </remarks>
    private new Tensor<T> ApplyActivation(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = ScalarActivation!.Activate(input[i]);
        }

        return output;
    }

    /// <summary>
    /// Applies the vector activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>The activated tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the vector activation function to the input tensor. Vector activation functions
    /// operate on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the advanced vector activation function.
    /// 
    /// The method:
    /// - Processes the entire tensor at once
    /// - Uses the vector activation function that was provided
    /// - Can capture relationships between different values
    /// 
    /// This is a more sophisticated way to add non-linearity, allowing the activation
    /// to consider how different features relate to each other.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyVectorActivation(Tensor<T> input)
    {
        return VectorActivation!.Activate(input);
    }

    /// <summary>
    /// Applies the derivative of the element-wise activation function to the gradient tensor.
    /// </summary>
    /// <param name="gradient">The gradient tensor to modify.</param>
    /// <param name="output">The output from the last forward pass.</param>
    /// <returns>The gradient tensor with the activation derivative applied.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of the element-wise activation function to the gradient tensor.
    /// This is used during the backward pass to calculate gradients for backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive the activation function is to changes.
    /// 
    /// During backpropagation (learning):
    /// - We need to know how much a small change in input affects the output
    /// - This method calculates that sensitivity for each value individually
    /// - It multiplies the incoming gradient by the derivative of the activation function
    /// 
    /// This is a key part of the math that allows neural networks to learn through backpropagation.
    /// </para>
    /// </remarks>
    private new Tensor<T> ApplyActivationDerivative(Tensor<T> gradient, Tensor<T> output)
    {
        var result = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            result[i] = NumOps.Multiply(gradient[i], ScalarActivation!.Derivative(output[i]));
        }

        return result;
    }

    /// <summary>
    /// Applies the derivative of the vector activation function to the gradient tensor.
    /// </summary>
    /// <param name="gradient">The gradient tensor to modify.</param>
    /// <param name="output">The output from the last forward pass.</param>
    /// <returns>The gradient tensor with the activation derivative applied.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of the vector activation function to the gradient tensor.
    /// This is used during the backward pass to calculate gradients for backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive the vector activation function is to changes.
    /// 
    /// During backpropagation:
    /// - This method uses the vector derivative of the activation function
    /// - It applies this to the entire tensor at once
    /// - It captures how changes to each value affect other values
    /// 
    /// This is the vector-based version of the activation derivative, which allows for more
    /// complex relationships between different features during learning.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyVectorActivationDerivative(Tensor<T> gradient, Tensor<T> output)
    {
        return gradient.ElementwiseMultiply(VectorActivation!.Derivative(output));
    }

    /// <summary>
    /// Returns an empty vector since the lambda layer typically has no trainable parameters.
    /// </summary>
    /// <returns>An empty vector.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector since the LambdaLayer typically has no trainable parameters.
    /// It is implemented as required by the LayerBase interface.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because there are typically no parameters.
    /// 
    /// Since Lambda layers:
    /// - Usually don't have their own weights or biases
    /// - Rely on the custom functions you provide
    /// 
    /// This method returns an empty vector to indicate there are no parameters.
    /// If your custom functions have parameters, you would need to handle saving
    /// and loading them separately.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // LambdaLayer typically has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from the forward pass.
    /// This includes the last input and output tensors.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input and output from previous data are cleared
    /// - The layer is ready for new data without being influenced by previous data
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _lastOutput = null;
    }
}