namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a readout layer that performs the final mapping from features to output in a neural network.
/// </summary>
/// <remarks>
/// <para>
/// The ReadoutLayer is typically used as the final layer in a neural network to transform features 
/// extracted by previous layers into the desired output format. It applies a linear transformation 
/// (weights and bias) followed by an activation function. This layer is similar to a dense or fully 
/// connected layer but is specifically designed for outputting the final results.
/// </para>
/// <para><b>For Beginners:</b> This layer serves as the final "decision maker" in a neural network.
/// 
/// Think of the ReadoutLayer as a panel of judges in a competition:
/// - Each judge (output neuron) receives information from all contestants (input features)
/// - Each judge has their own preferences (weights) for different skills
/// - Judges combine all this information to make their final scores (outputs)
/// - An activation function then shapes these scores into the desired format
/// 
/// For example, in an image classification network:
/// - Previous layers extract features like edges, shapes, and patterns
/// - The ReadoutLayer takes all these features and combines them into class scores
/// - If there are 10 possible classes, the ReadoutLayer might have 10 outputs
/// - Each output represents the network's confidence that the image belongs to that class
/// 
/// This layer learns which features are most important for each output category during training.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ReadoutLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Matrix<double> storing the weight parameters for connections between inputs and outputs.
    /// </summary>
    /// <remarks>
    /// This matrix has dimensions [outputSize, inputSize], where each row represents the weights
    /// for one output neuron. These weights determine how strongly each input feature influences
    /// each output neuron and are the primary trainable parameters of the layer.
    /// </remarks>
    private Matrix<T> _weights = default!;
    
    /// <summary>
    /// Vector<double> storing the bias parameters for each output neuron.
    /// </summary>
    /// <remarks>
    /// This vector has length outputSize, where each element is a constant value added to the
    /// weighted sum for the corresponding output neuron. Biases allow the network to shift the
    /// activation function, giving it more flexibility to fit the data.
    /// </remarks>
    private Vector<T> _bias = default!;
    
    /// <summary>
    /// Matrix<double> storing the gradients of the loss with respect to the weight parameters.
    /// </summary>
    /// <remarks>
    /// This matrix has the same dimensions as the _weights matrix and stores the accumulated
    /// gradients for all weight parameters during the backward pass. These gradients are used
    /// to update the weights during the parameter update step.
    /// </remarks>
    private Matrix<T> _weightGradients = default!;
    
    /// <summary>
    /// Vector<double> storing the gradients of the loss with respect to the bias parameters.
    /// </summary>
    /// <remarks>
    /// This vector has the same length as the _bias vector and stores the accumulated
    /// gradients for all bias parameters during the backward pass. These gradients are used
    /// to update the biases during the parameter update step.
    /// </remarks>
    private Vector<T> _biasGradients = default!;
    
    /// <summary>
    /// Stores the input vector from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached input is needed during the backward pass to compute gradients. It holds the
    /// input vector that was processed in the most recent forward pass. The vector is null
    /// before the first forward pass or after a reset.
    /// </remarks>
    private Vector<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for ReadoutLayer, indicating that the layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the ReadoutLayer has trainable parameters (weights and biases) that
    /// can be optimized during the training process using backpropagation. The gradients of these parameters
    /// are calculated during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has values (weights and biases) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process of the neural network
    /// 
    /// When you train a neural network containing this layer, the weights and biases will 
    /// automatically adjust to better recognize patterns specific to your data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReadoutLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer.</param>
    /// <param name="scalarActivation">The activation function to apply to individual elements of the output.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReadoutLayer with the specified dimensions and a scalar activation function.
    /// The weights are initialized with small random values, and the biases are initialized to zero. A scalar
    /// activation function is applied element-wise to each output neuron independently.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new readout layer for your neural network using a simple activation function.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many features come into the layer
    /// - outputSize: How many outputs you want from the layer
    /// - scalarActivation: How to transform each output (e.g., sigmoid, ReLU, tanh)
    /// 
    /// A scalar activation means each output is calculated independently. For example, in a 10-class
    /// classification problem, you might use inputSize=512 (512 features from previous layers),
    /// outputSize=10 (one for each class), and a softmax activation to get class probabilities.
    /// 
    /// The layer starts with small random weights and zero biases that will be refined during training.
    /// </para>
    /// </remarks>
    public ReadoutLayer(int inputSize, int outputSize, IActivationFunction<T> scalarActivation) 
        : base([inputSize], [outputSize], scalarActivation)
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _bias = new Vector<T>(outputSize);
        _weightGradients = new Matrix<T>(outputSize, inputSize);
        _biasGradients = new Vector<T>(outputSize);

        InitializeParameters(inputSize, outputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ReadoutLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer.</param>
    /// <param name="vectorActivation">The activation function to apply to the entire output vector.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReadoutLayer with the specified dimensions and a vector activation function.
    /// The weights are initialized with small random values, and the biases are initialized to zero. A vector
    /// activation function is applied to the entire output vector at once, which allows for interactions between
    /// different output neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new readout layer for your neural network using an advanced activation function.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many features come into the layer
    /// - outputSize: How many outputs you want from the layer
    /// - vectorActivation: How to transform the entire output as a group
    /// 
    /// A vector activation means all outputs are calculated together, which can capture relationships between outputs.
    /// For example, softmax is a vector activation that ensures all outputs sum to 1, making them behave like probabilities.
    /// 
    /// This is particularly useful for:
    /// - Multi-class classification (using softmax activation)
    /// - Problems where outputs should be interdependent
    /// - Cases where you need to enforce specific constraints across all outputs
    /// 
    /// The layer starts with small random weights and zero biases that will be refined during training.
    /// </para>
    /// </remarks>
    public ReadoutLayer(int inputSize, int outputSize, IVectorActivationFunction<T> vectorActivation) 
        : base([inputSize], [outputSize], vectorActivation)
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _bias = new Vector<T>(outputSize);
        _weightGradients = new Matrix<T>(outputSize, inputSize);
        _biasGradients = new Vector<T>(outputSize);

        InitializeParameters(inputSize, outputSize);
    }

    /// <summary>
    /// Performs the forward pass of the readout layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after readout processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the readout layer. It converts the input tensor to a vector,
    /// applies a linear transformation (weights and bias), and then applies the activation function. The input
    /// is cached for use during the backward pass. The method handles both scalar and vector activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the readout layer.
    /// 
    /// During the forward pass:
    /// 1. Your input data is flattened into a simple list of numbers
    /// 2. Each output neuron calculates a weighted sum of all inputs plus its bias
    /// 3. The activation function transforms these sums into the final outputs
    /// 
    /// The formula for each output is basically:
    /// output = activation(weights × inputs + bias)
    /// 
    /// This is similar to how a teacher might grade an exam:
    /// - Different questions have different weights (more important questions get more points)
    /// - There might be a curve applied to the final scores (activation function)
    /// 
    /// The layer saves the input for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.ToVector();
        var output = _weights * _lastInput + _bias;

        if (UsingVectorActivation)
        {
            return Tensor<T>.FromVector(VectorActivation!.Activate(output));
        }
        else
        {
            return Tensor<T>.FromVector(output.Transform(x => ScalarActivation!.Activate(x)));
        }
    }

    /// <summary>
    /// Performs the backward pass of the readout layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the readout layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradients of the loss
    /// with respect to the weights and biases (to update the layer's parameters) and with respect to
    /// the input (to propagate back to previous layers). The method handles both scalar and vector
    /// activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The error gradient from the loss function or next layer is received
    /// 2. This gradient is adjusted based on the activation function used
    /// 3. The layer calculates how each weight and bias should change to reduce the error
    /// 4. The layer calculates how the previous layer's output should change
    /// 
    /// This is like giving feedback to improve performance:
    /// - "This feature was too important in your decision-making" (weight too high)
    /// - "You're not paying enough attention to this feature" (weight too low)
    /// - "You're consistently scoring too high/low" (bias adjustment needed)
    /// 
    /// These calculations are at the heart of how neural networks learn from their mistakes.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        var gradient = outputGradient.ToVector();

        if (UsingVectorActivation)
        {
            var activationDerivative = VectorActivation!.Derivative(_weights * _lastInput + _bias);
            var diagonalDerivative = activationDerivative.Diagonal();
            gradient = gradient.PointwiseMultiply(diagonalDerivative);
        }
        else
        {
            gradient = gradient.PointwiseMultiply((_weights * _lastInput + _bias).Transform(x => ScalarActivation!.Derivative(x)));
        }

        _weightGradients = Matrix<T>.OuterProduct(gradient, _lastInput);
        _biasGradients = gradient;

        return Tensor<T>.FromVector(_weights.Transpose() * gradient);
    }

    /// <summary>
    /// Updates the parameters of the readout layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the readout layer based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter
    /// updates. This method should be called after the backward pass to apply the calculated updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. The weight values are adjusted based on their gradients
    /// 2. The bias values are adjusted based on their gradients
    /// 3. The learning rate controls how big each update step is
    /// 
    /// This is like making small adjustments based on feedback:
    /// - Weights that contributed to errors are reduced
    /// - Weights that would have helped are increased
    /// - The learning rate determines how quickly the model adapts
    /// 
    /// Smaller learning rates mean slower but more stable learning, while larger learning rates
    /// mean faster but potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = NumOps.Subtract(_weights[i, j], NumOps.Multiply(learningRate, _weightGradients[i, j]));
            }

            _bias[i] = NumOps.Subtract(_bias[i], NumOps.Multiply(learningRate, _biasGradients[i]));
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the readout layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (weights and biases).</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) of the readout layer as a
    /// single vector. The weights are stored first, followed by the biases. This is useful for optimization
    /// algorithms that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the readout layer.
    /// 
    /// The parameters:
    /// - Are the weights and biases that the readout layer learns during training
    /// - Control how the layer processes information
    /// - Are returned as a single list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The weights are stored first in the vector, followed by all the bias values.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weights.Rows * _weights.Columns + _bias.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy weights
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                parameters[index++] = _weights[i, j];
            }
        }
    
        // Copy bias
        for (int i = 0; i < _bias.Length; i++)
        {
            parameters[index++] = _bias[i];
        }
    
        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the readout layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (weights and biases) to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters (weights and biases) of the readout layer from a single vector.
    /// The vector should contain the weight values first, followed by the bias values. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the weights and biases in the readout layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct total length
    /// - The first part of the vector is used for the weights
    /// - The second part of the vector is used for the biases
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _weights.Rows * _weights.Columns + _bias.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weights
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = parameters[index++];
            }
        }
    
        // Set bias
        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the readout layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the readout layer, including the cached input from the
    /// forward pass and the gradients from the backward pass. This is useful when starting to process
    /// a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored input from previous calculations is cleared
    /// - Calculated gradients are reset to zero
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// The weights and biases (the learned parameters) are not reset,
    /// only the temporary state information.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    
        // Reset gradients
        _weightGradients = new Matrix<T>(_weights.Rows, _weights.Columns);
        _biasGradients = new Vector<T>(_bias.Length);
    }

    /// <summary>
    /// Initializes the weights and biases of the readout layer with small random values and zeros.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer.</param>
    /// <remarks>
    /// This private method initializes the weights with small random values centered around zero
    /// and initializes the biases to zero. This provides a good starting point for training
    /// the readout layer, avoiding issues like vanishing or exploding gradients.
    /// </remarks>
    private void InitializeParameters(int inputSize, int outputSize)
    {
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = NumOps.FromDouble((Random.NextDouble() - 0.5) * 0.1);
            }

            _bias[i] = NumOps.Zero;
        }
    }
}