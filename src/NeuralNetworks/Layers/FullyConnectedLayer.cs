namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected layer in a neural network where every input neuron connects to every output neuron.
/// </summary>
/// <remarks>
/// <para>
/// A fully connected layer, also known as a dense layer, is a fundamental building block in neural networks.
/// It connects every input neuron to every output neuron with learnable weights. Each output neuron also has
/// a learnable bias term. The layer applies a linear transformation followed by an activation function to
/// produce its output. Fully connected layers are particularly useful for learning complex patterns and 
/// for classification tasks.
/// </para>
/// <para><b>For Beginners:</b> A fully connected layer connects every input to every output, like a complete web of connections.
/// 
/// Imagine you have inputs representing different features:
/// - Each feature (input) connects to every possible output
/// - Each connection has a strength (weight) that can be adjusted
/// - Each output also has a starting value (bias)
/// 
/// For example, in an image classification task:
/// - Inputs might be flattened features from convolutional layers
/// - Each output might represent a score for a different category
/// - The connections (weights) learn which features are important for each category
/// 
/// Fully connected layers are excellent at combining features to make final decisions.
/// They're often used toward the end of a neural network to interpret the features
/// extracted by earlier layers.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class FullyConnectedLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight matrix connecting input neurons to output neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the learnable weights for the connections between input and output neurons.
    /// The shape is [outputSize, inputSize], where each element represents the strength of the connection
    /// between an input neuron and an output neuron.
    /// </para>
    /// <para><b>For Beginners:</b> These weights determine how strongly each input affects each output.
    /// 
    /// Think of weights like importance factors:
    /// - Positive weights mean "this input increases this output"
    /// - Negative weights mean "this input decreases this output"
    /// - Larger values (positive or negative) mean stronger influence
    /// - Values near zero mean weak influence
    /// 
    /// During training:
    /// - The network adjusts these weights to find the best relationships
    /// - Important connections get stronger weights
    /// - Unimportant connections get weights closer to zero
    /// 
    /// The matrix has one row per output neuron and one column per input neuron,
    /// so every input-output pair has exactly one weight value.
    /// </para>
    /// </remarks>
    private Matrix<T> _weights = default!;

    /// <summary>
    /// The bias values for each output neuron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the learnable bias terms for each output neuron. The biases are added
    /// to the weighted sum of inputs before applying the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> Biases are like default or starting values for each output.
    /// 
    /// Biases serve several important purposes:
    /// - They allow outputs to be activated even when all inputs are zero
    /// - They act like an adjustable threshold for each neuron
    /// - They help the network learn more effectively
    /// 
    /// For example:
    /// - A neuron with a large negative bias is "reluctant" to activate
    /// - A neuron with a large positive bias "wants" to activate
    /// - During training, biases adjust to find the optimal activation threshold
    /// 
    /// Each output neuron has its own bias value that can be learned independently.
    /// </para>
    /// </remarks>
    private Vector<T> _biases = default!;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the input received during the last forward pass. It is necessary for computing
    /// gradients during the backward pass (backpropagation).
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what input data was processed most recently.
    /// 
    /// During training:
    /// - The layer needs to remember what input values it processed
    /// - This helps when calculating how to improve the weights and biases
    /// - It's like keeping your work when solving a math problem
    /// 
    /// This value is automatically cleared between training batches to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The output tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the output produced during the last forward pass. It is used during
    /// backpropagation to compute certain gradients, particularly for activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This stores what the layer output after its most recent calculation.
    /// 
    /// During training:
    /// - The network needs to remember what predictions it made
    /// - This helps calculate how to improve the weights and biases
    /// - The output values are used when computing how to adjust parameters
    /// 
    /// This is also cleared after each training batch to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// The gradients for the weights, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the gradients of the loss with respect to each weight. These gradients are
    /// used to update the weights during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each weight value.
    /// 
    /// During training:
    /// - The network calculates how each weight contributed to errors
    /// - Gradients show both direction and amount to change each weight
    /// - Larger gradients mean bigger adjustments are needed
    /// 
    /// For example:
    /// - A positive gradient means "decrease this weight to reduce error"
    /// - A negative gradient means "increase this weight to reduce error"
    /// - The magnitude indicates how strongly the weight should change
    /// 
    /// These gradients are used in the UpdateParameters method to actually
    /// modify the weights.
    /// </para>
    /// </remarks>
    private Matrix<T>? _weightsGradient;

    /// <summary>
    /// The gradients for the biases, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the gradients of the loss with respect to each bias. These gradients are
    /// used to update the biases during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each bias value.
    /// 
    /// During training:
    /// - The network calculates how each bias contributed to errors
    /// - These gradients show how to adjust the "threshold" of each neuron
    /// - They work just like weight gradients, but for bias values
    /// 
    /// For example:
    /// - If a neuron activates too easily, its bias gradient will be positive
    ///   (suggesting to decrease the bias)
    /// - If a neuron doesn't activate enough, its bias gradient will be negative
    ///   (suggesting to increase the bias)
    /// 
    /// Each output neuron has its own bias gradient that guides its adjustment.
    /// </para>
    /// </remarks>
    private Vector<T>? _biasesGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because fully connected layers have trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the fully connected layer supports training through backpropagation.
    /// The layer has trainable parameters (weights and biases) that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its weights and biases during training
    /// - It will improve its performance as it sees more data
    /// - It has parameters that are updated to make better predictions
    /// 
    /// Fully connected layers are primary learning components in neural networks,
    /// as they contain trainable parameters that adapt to recognize patterns in the data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="FullyConnectedLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputSize">The number of input neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="activationFunction">The activation function to apply after the linear transformation. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new fully connected layer with the specified input size, output size, and
    /// activation function. The weights are initialized with small random values using Xavier/Glorot initialization,
    /// and the biases are initialized to zero. The activation function operates on individual scalar values
    /// in the output tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the fully connected layer with the specific number of inputs and outputs you need.
    /// 
    /// When creating a fully connected layer, you need to specify:
    /// - Input size: How many values are coming into the layer
    /// - Output size: How many values you want the layer to produce
    /// - Activation function: How to introduce non-linearity (like ReLU or Sigmoid)
    /// 
    /// For example:
    /// ```csharp
    /// // Create a hidden layer with 784 inputs (e.g., from a 28×28 image), 
    /// // 128 outputs, and ReLU activation
    /// var hiddenLayer = new FullyConnectedLayer<float>(784, 128);
    /// 
    /// // Create an output layer with 128 inputs (from previous layer),
    /// // 10 outputs (e.g., for 10 classes), and Sigmoid activation
    /// var outputLayer = new FullyConnectedLayer<float>(128, 10, new SigmoidActivation<float>());
    /// ```
    /// 
    /// The constructor automatically initializes weights with appropriate small random
    /// values that help training converge effectively.
    /// </para>
    /// </remarks>
    public FullyConnectedLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="FullyConnectedLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputSize">The number of input neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after the linear transformation. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new fully connected layer with the specified input size, output size, and
    /// vector activation function. The weights are initialized with small random values using Xavier/Glorot initialization,
    /// and the biases are initialized to zero. Unlike the other constructor, this one accepts a vector activation
    /// function that operates on entire vectors rather than individual scalar values.
    /// </para>
    /// <para><b>For Beginners:</b> This is an alternative setup that uses a different kind of activation function.
    /// 
    /// This constructor is almost identical to the first one, but with one key difference:
    /// - Regular activation: processes each output value separately
    /// - Vector<double> activation: processes the entire output vector together
    /// 
    /// Vector<double> activation functions like Softmax are useful for:
    /// - Classification problems (choosing between multiple categories)
    /// - Problems where outputs need to sum to 1 (like probabilities)
    /// - Cases where output values should influence each other
    /// 
    /// For example:
    /// ```csharp
    /// // Create an output layer with Softmax activation for multi-class classification
    /// var outputLayer = new FullyConnectedLayer<float>(256, 10, new SoftmaxActivation<float>());
    /// ```
    /// 
    /// Softmax makes sure that increasing one output decreases all others,
    /// which is perfect for classification tasks where outputs represent class probabilities.
    /// </para>
    /// </remarks>
    public FullyConnectedLayer(int inputSize, int outputSize, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([inputSize], [outputSize], vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases with appropriate values for effective training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using Xavier/Glorot initialization, which helps with training
    /// convergence by setting initial values to a scale appropriate for the layer's dimensions.
    /// The biases are initialized to zero. This initialization strategy helps to prevent vanishing or
    /// exploding gradients at the start of training.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial values for the weights and biases before training.
    /// 
    /// For good training:
    /// - Weights need to start with small random values
    /// - These values are carefully scaled based on layer size
    /// - Too large or too small values can make training difficult
    /// 
    /// The method uses "Xavier initialization," which is a popular way to set
    /// initial weights that helps the network learn effectively:
    /// - It considers both input and output sizes
    /// - It scales the random values appropriately
    /// - It helps signals flow well through the network from the beginning
    /// 
    /// Biases are simply initialized to zero, as they'll adjust during training.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Initialize weights and biases (e.g., Xavier/Glorot initialization)
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_weights.Rows + _weights.Columns)));
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble()), scale);
            }

            _biases[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Performs the forward pass of the fully connected layer.
    /// </summary>
    /// <param name="input">The input tensor to process. Shape: [batchSize, inputSize].</param>
    /// <returns>The output tensor after the linear transformation and activation. Shape: [batchSize, outputSize].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the fully connected layer. For each example in the batch,
    /// it performs a matrix multiplication between the input vector and the weight matrix, adds the bias vector,
    /// and applies the activation function to produce the final output. The input and output are cached for
    /// use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer processes input data to produce outputs.
    /// 
    /// The forward pass works in these steps for each example in the batch:
    /// 1. Extract the input vector for this example
    /// 2. Multiply the input vector by the weight matrix
    ///    - Each output neuron computes a weighted sum of all inputs
    /// 3. Add the bias vector to the result
    ///    - Each output gets its own bias value added
    /// 4. Apply the activation function
    ///    - This introduces non-linearity, helping the network learn complex patterns
    /// 5. Store the result in the output tensor
    /// 
    /// This process transforms the input data through the layer's learned parameters,
    /// producing output values that will either be passed to the next layer or
    /// used as the final network output.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputSize = input.Shape[1];
        int outputSize = _weights.Rows;

        var output = new Tensor<T>([batchSize, outputSize]);

        for (int i = 0; i < batchSize; i++)
        {
            var inputVector = new Vector<T>(inputSize);
            for (int j = 0; j < inputSize; j++)
            {
                inputVector[j] = input[i, j];
            }

            var outputVector = _weights.Multiply(inputVector).Add(_biases);
            outputVector = ApplyActivation(outputVector);

            for (int j = 0; j < outputSize; j++)
            {
                output[i, j] = outputVector[j];
            }
        }

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Performs the backward pass of the fully connected layer to compute gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [batchSize, outputSize].</param>
    /// <returns>The gradient tensor to be passed to the previous layer. Shape: [batchSize, inputSize].</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass (backpropagation) of the fully connected layer. It computes
    /// the gradients of the loss with respect to the layer's weights, biases, and inputs. These gradients
    /// are used to update the parameters during training and to propagate the error signal back to the previous layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer learns from its mistakes during training.
    /// 
    /// The backward pass works in these steps for each example in the batch:
    /// 1. Extract the gradient and necessary vectors for this example
    /// 2. Apply the activation function derivative to the gradient
    ///    - This accounts for how the activation function affected the output
    /// 3. Calculate weight gradients using outer product
    ///    - Shows how each weight contributed to the error
    /// 4. Accumulate bias gradients
    ///    - Shows how each bias affected the output
    /// 5. Calculate input gradients to pass back to previous layer
    ///    - Helps earlier layers learn as well
    /// 
    /// The gradients tell us:
    /// - How to adjust each weight and bias to reduce errors
    /// - How the error signal should flow back to previous layers
    /// 
    /// All gradients are accumulated across the batch before being used
    /// to update parameters.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        var weightsGradient = new Matrix<T>(_weights.Rows, _weights.Columns);
        var biasesGradient = new Vector<T>(_biases.Length);

        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape[1];
        int outputSize = _weights.Rows;

        for (int i = 0; i < batchSize; i++)
        {
            var outputGradientVector = new Vector<T>(outputSize);
            var lastOutputVector = new Vector<T>(outputSize);
            var inputVector = new Vector<T>(inputSize);

            for (int j = 0; j < outputSize; j++)
            {
                outputGradientVector[j] = outputGradient[i, j];
                lastOutputVector[j] = _lastOutput[i, j];
            }

            for (int j = 0; j < inputSize; j++)
            {
                inputVector[j] = _lastInput[i, j];
            }

            var delta = ApplyActivationDerivative(lastOutputVector, outputGradientVector);
            weightsGradient = weightsGradient.Add(Matrix<T>.OuterProduct(delta, inputVector));
            biasesGradient = biasesGradient.Add(delta);

            var inputGradientVector = _weights.Transpose().Multiply(delta);
            for (int j = 0; j < inputSize; j++)
            {
                inputGradient[i, j] = inputGradientVector[j];
            }
        }

        _weightsGradient = weightsGradient;
        _biasesGradient = biasesGradient;

        return inputGradient;
    }

    /// <summary>
    /// Updates the weights and biases using the calculated gradients and the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases based on the gradients calculated during the backward pass.
    /// The learning rate determines the size of the parameter updates. Smaller learning rates lead to more
    /// stable but slower training, while larger learning rates can lead to faster but potentially unstable training.
    /// </para>
    /// <para><b>For Beginners:</b> This method actually changes the weights and biases to improve future predictions.
    /// 
    /// After figuring out how each parameter should change:
    /// - Each weight and bias is adjusted in the direction that reduces errors
    /// - The learning rate controls how big these adjustments are
    /// 
    /// Think of it like adjusting a recipe after tasting:
    /// - Too salty? Reduce salt next time (adjust weights/biases)
    /// - But make small adjustments (learning rate), not drastic ones
    /// 
    /// For example, with a learning rate of 0.01:
    /// - A gradient of 0.5 would change the parameter by -0.005
    /// - A gradient of -2.0 would change the parameter by +0.02
    /// 
    /// The minus sign in the code is because we want to go in the opposite
    /// direction of the gradient to minimize error.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
        _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) of the layer as a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving
    /// and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the layer's learnable values into a single list.
    /// 
    /// The parameters include:
    /// - All the weight values (the majority of the parameters)
    /// - All the bias values (one per output neuron)
    /// 
    /// This combined list is useful for:
    /// - Saving a trained model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need all parameters together
    /// 
    /// For example, a layer with 100 inputs and 10 outputs would have:
    /// - 1,000 weight parameters (100 × 10)
    /// - 10 bias parameters (one per output)
    /// - Totaling 1,010 parameters in the returned vector
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weights.Rows * _weights.Columns + _biases.Length;
        var parameters = new Vector<T>(totalParams);

        int index = 0;

        // Copy weights parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                parameters[index++] = _weights[i, j];
            }
        }

        // Copy biases parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            parameters[index++] = _biases[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters (weights and biases) of the layer from a single vector.
    /// This is useful for loading saved model weights or for implementing optimization algorithms
    /// that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's learnable values from a provided list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the exact right length
    /// - The values are distributed back to the weights and biases
    /// - This allows loading previously trained weights
    /// 
    /// Use cases include:
    /// - Restoring a saved model
    /// - Using pre-trained weights
    /// - Testing specific weight configurations
    /// 
    /// The method throws an error if the provided vector doesn't contain exactly the right number of values.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Rows * _weights.Columns + _biases.Length)
        {
            throw new ArgumentException($"Expected {_weights.Rows * _weights.Columns + _biases.Length} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set weights parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = parameters[index++];
            }
        }

        // Set biases parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing all cached values from forward
    /// and backward passes. This is useful when starting to process a new batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input and output are cleared
    /// - The calculated gradients are cleared
    /// - The layer forgets previous calculations it performed
    /// 
    /// This is typically called:
    /// - Between training batches to free up memory
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    /// 
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// Note that this doesn't affect the learned weights and biases, just the
    /// temporary working data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }
}