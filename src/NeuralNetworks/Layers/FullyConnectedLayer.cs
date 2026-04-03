using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class FullyConnectedLayer<T> : LayerBase<T>
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
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _weights;

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
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _biases;

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
    /// Tracks whether the last forward pass input was rank-1, so backward can preserve rank.
    /// </summary>
    private bool _inputWas1D;

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
    private Tensor<T>? _lastPreActivation;

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
    private Tensor<T>? _weightsGradient;

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
    private Tensor<T>? _biasesGradient;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuInput;
    private Tensor<T>? _gpuPreActivation;
    private int[] _gpuInputShape = [];

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
        _weights = new Tensor<T>([outputSize, inputSize]);
        _biases = new Tensor<T>([outputSize]);

        InitializeParameters();

        // Register tensors for GPU memory persistence
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
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
    /// - Vector activation: processes the entire output vector together
    /// 
    /// Vector activation functions like Softmax are useful for:
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
        _weights = new Tensor<T>([outputSize, inputSize]);
        _biases = new Tensor<T>([outputSize]);

        InitializeParameters();

        // Register tensors for GPU memory persistence
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Gets the total number of trainable parameters (weights + biases).
    /// </summary>
    public override int ParameterCount => _weights.Shape[0] * _weights.Shape[1] + _biases.Shape[0];

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
        InitializeLayerWeights(_weights, _weights.Shape[0], _weights.Shape[1]);
        InitializeLayerBiases(_biases);
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
        // Auto-reshape 1D input to [1, N] for matmul compatibility
        _inputWas1D = input.Shape.Length == 1;
        if (_inputWas1D)
        {
            input = input.Reshape(1, input.Length);
        }

        _lastInput = input;

        // Compute output = input * weights^T + biases using Engine operations
        // input: [batchSize, inputSize]
        // weights: [outputSize, inputSize]

        // Transpose weights to [inputSize, outputSize]
        var weightsT = Engine.TensorTranspose(_weights);

        // Matrix multiply: [batch, input] * [input, output] -> [batch, output]
        var linearOutput = Engine.TensorMatMul(input, weightsT);

        // Add biases (broadcast)
        var biasBroadcast = _biases.Reshape(1, _biases.Shape[0]);
        var biasedOutput = Engine.TensorBroadcastAdd(linearOutput, biasBroadcast);

        var result = ApplyActivation(biasedOutput);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastPreActivation = biasedOutput;
            _lastOutput = result;
        }

        // Preserve original rank: if input was 1D, output should be 1D
        if (_inputWas1D)
        {
            result = result.Reshape(result.Length);
        }

        return result;
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

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_biases);
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
        // Flatten weight tensor and concatenate with biases
        int weightCount = _weights.Shape[0] * _weights.Shape[1];
        int biasCount = _biases.Shape[0];
        var parameters = new Vector<T>(weightCount + biasCount);

        int index = 0;
        for (int i = 0; i < _weights.Shape[0]; i++)
        {
            for (int j = 0; j < _weights.Shape[1]; j++)
            {
                parameters[index++] = _weights[i, j];
            }
        }
        for (int i = 0; i < biasCount; i++)
        {
            parameters[index++] = _biases[i];
        }

        return parameters;
    }

    /// <summary>
    /// Gets the gradients of all trainable parameters in this layer.
    /// </summary>
    public override Vector<T> GetParameterGradients()
    {
        if (_weightsGradient == null || _biasesGradient == null)
        {
            return new Vector<T>(ParameterCount);
        }

        int weightCount = _weightsGradient.Shape[0] * _weightsGradient.Shape[1];
        int biasCount = _biasesGradient.Shape[0];
        var gradients = new Vector<T>(weightCount + biasCount);

        int index = 0;
        for (int i = 0; i < _weightsGradient.Shape[0]; i++)
        {
            for (int j = 0; j < _weightsGradient.Shape[1]; j++)
            {
                gradients[index++] = _weightsGradient[i, j];
            }
        }
        for (int i = 0; i < biasCount; i++)
        {
            gradients[index++] = _biasesGradient[i];
        }

        return gradients;
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
        int weightCount = _weights.Shape[0] * _weights.Shape[1];
        int biasCount = _biases.Shape[0];

        if (parameters.Length != weightCount + biasCount)
        {
            throw new ArgumentException($"Expected {weightCount + biasCount} parameters, but got {parameters.Length}", nameof(parameters));
        }

        // Extract weights from flat vector
        int index = 0;
        for (int i = 0; i < _weights.Shape[0]; i++)
        {
            for (int j = 0; j < _weights.Shape[1]; j++)
            {
                _weights[i, j] = parameters[index++];
            }
        }

        // Extract biases
        for (int i = 0; i < biasCount; i++)
        {
            _biases[i] = parameters[index++];
        }

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Clears stored gradients for weights and biases.
    /// </summary>
    public override void ClearGradients()
    {
        if (_weightsGradient != null)
        {
            _weightsGradient.Fill(NumOps.Zero);
        }

        if (_biasesGradient != null)
        {
            _biasesGradient.Fill(NumOps.Zero);
        }

        base.ClearGradients();
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
        _lastPreActivation = null;
        _weightsGradient = null;
        _biasesGradient = null;

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuPreActivation = null;
        _gpuInputShape = [];
    }

    /// <inheritdoc />
    public override Tensor<T>? GetWeights() => _weights;

    /// <inheritdoc />
    public override Tensor<T>? GetBiases() => _biases;

    /// <summary>
    /// Gets whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs a GPU-resident forward pass, keeping tensors on GPU.
    /// Use this for chained layer execution to avoid CPU round-trips.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensors (uses first input).</param>
    /// <returns>GPU-resident output tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown if GPU execution is not available.</exception>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var input = inputs[0];
        int[] inputShape = input.Shape.ToArray();

        // FullyConnectedLayer stores weights as [outputSize, inputSize]
        // We need to transpose for FusedLinearGpu which expects [inputSize, outputSize]
        int outputSize = _weights.Shape[0];
        int inputSize = _weights.Shape[1];

        // Determine batch size and input features
        int batchSize = inputShape.Length >= 2 ? inputShape[0] : 1;
        int inputFeatures = inputShape.Length >= 2 ? inputShape[1] : inputShape[0];

        // Validate dimensions
        if (inputFeatures != inputSize)
            throw new ArgumentException($"Input feature dimension {inputFeatures} does not match weights input dimension {inputSize}");

        // Transpose weights from [outputSize, inputSize] to [inputSize, outputSize]
        // This is needed because FusedLinearGpu expects weights in [inputSize, outputSize] format
        var weightsT = Engine.TensorTranspose(_weights);

        // Get the fused activation type using the base class method
        var fusedActivation = GetFusedActivationType();

        // Handle input shape conversion for FusedLinearGpu
        Tensor<T> input2D = input;
        bool needsReshape = inputShape.Length != 2;

        if (inputShape.Length == 1)
        {
            // 1D input [features] -> [1, features]
            input2D = input.Reshape([1, inputFeatures]);
        }
        else if (inputShape.Length > 2)
        {
            // ND input -> flatten to [batchDim, features]
            input2D = input.Reshape([batchSize, inputFeatures]);
        }

        // Use GPU-resident FusedLinear - NO CPU round-trip
        // Result is [batchDim, outputSize]
        var result = gpuEngine.FusedLinearGpu(input2D, weightsT, _biases, fusedActivation);

        // Cache state for backward pass only during training
        if (IsTrainingMode)
        {
            // Cache GPU tensors for GPU-resident backward pass
            _gpuInput = input2D;
            _gpuInputShape = inputShape;

            // For fused activations, we need pre-activation for gradient computation
            if (fusedActivation != FusedActivationType.None)
            {
                var preActivation = gpuEngine.FusedLinearGpu(input2D, weightsT, _biases, FusedActivationType.None);
                _gpuPreActivation = preActivation;
                _lastOutput = preActivation;
            }
            else
            {
                _gpuPreActivation = result;
                _lastOutput = result;
            }

            // Also cache CPU tensors for fallback backward pass
            _lastInput = input;
        }

        // Reshape output back to original batch dimensions if needed
        if (inputShape.Length == 1)
        {
            // 1D input -> 1D output [outputSize]
            result = result.Reshape([outputSize]);
        }
        else if (inputShape.Length > 2)
        {
            // ND input -> ND output with same leading dimensions
            int[] outputShape = new int[inputShape.Length];
            for (int i = 0; i < inputShape.Length - 1; i++)
            {
                outputShape[i] = inputShape[i];
            }
            outputShape[^1] = outputSize;
            result = result.Reshape(outputShape);
        }
        // 2D input: result is already [batch, outputSize]

        return result;
    }
}
