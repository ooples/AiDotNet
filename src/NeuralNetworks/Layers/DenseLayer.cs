namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected (dense) layer in a neural network.
/// </summary>
/// <remarks>
/// <para>
/// A dense layer connects every input neuron to every output neuron, with each connection having
/// a learnable weight. This is the most basic and widely used type of neural network layer.
/// Dense layers are capable of learning complex patterns by adjusting these weights during training.
/// </para>
/// <para><b>For Beginners:</b> A dense layer is like a voting system where every input gets to vote on every output.
/// 
/// Think of it like this:
/// - Each input sends information to every output
/// - Each connection has a different "importance" (weight)
/// - The layer learns which connections should be strong and which should be weak
/// 
/// For example, in an image recognition task:
/// - One input might detect a curved edge
/// - Another might detect a straight line
/// - The dense layer combines these features to recognize higher-level patterns
/// 
/// Dense layers are the building blocks of many neural networks because they can learn
/// almost any relationship between inputs and outputs, given enough neurons and training data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DenseLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weights tensor that transforms input features to output features.
    /// </summary>
    private Tensor<T> _weights = default!;

    /// <summary>
    /// The bias tensor added to the weighted sum of inputs.
    /// </summary>
    private Tensor<T> _biases = default!;

    /// <summary>
    /// Gradient of the weights tensor during backpropagation.
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Gradient of the biases tensor during backpropagation.
    /// </summary>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Cached input tensor from the forward pass, used during backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    public override int ParameterCount => _weights.Length + _biases.Length;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the DenseLayer class with scalar activation.
    /// </summary>
    /// <param name="inputSize">Number of input features.</param>
    /// <param name="outputSize">Number of output features.</param>
    /// <param name="activationFunction">Activation function to apply to outputs.</param>
    public DenseLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        // Store weights and biases directly as tensors
        _weights = new Tensor<T>([outputSize, inputSize]);
        _biases = new Tensor<T>([outputSize]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the DenseLayer class with vector activation.
    /// </summary>
    /// <param name="inputSize">Number of input features.</param>
    /// <param name="outputSize">Number of output features.</param>
    /// <param name="vectorActivation">Vector<double> activation function to apply to outputs.</param>
    public DenseLayer(int inputSize, int outputSize, IVectorActivationFunction<T>? vectorActivation = null)
        : base([inputSize], [outputSize], vectorActivation ?? new ReLUActivation<T>())
    {
        _weights = new Tensor<T>([outputSize, inputSize]);
        _biases = new Tensor<T>([outputSize]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes weights using Xavier/Glorot initialization and zeros biases.
    /// </summary>
    private void InitializeParameters()
    {
        double scale = Math.Sqrt(2.0 / (InputShape[0] + OutputShape[0]));

        // Fallback to element-wise initialization
        for (int i = 0; i < _weights.Shape[0]; i++)
        {
            for (int j = 0; j < _weights.Shape[1]; j++)
            {
                _weights[i, j] = NumOps.FromDouble(Random.NextDouble() * 2 * scale - scale);
            }
        }

        // Initialize biases to zero using tensor operations
        _biases.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Performs forward pass computation.
    /// </summary>
    /// <param name="input">Input tensor with shape [batchSize, inputSize].</param>
    /// <returns>Output tensor with shape [batchSize, outputSize].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // Cache input for backward pass
        _lastInput = input;

        // Validate input dimensions
        if (input.Rank < 2)
            throw new ArgumentException($"Input tensor must have at least 2 dimensions, got {input.Rank}");

        // Ensure input has correct feature dimension
        if (input.Shape[input.Shape.Length - 1] != InputShape[0])
            throw new ArgumentException($"Expected input features dimension {InputShape[0]}, got {input.Shape[input.Shape.Length - 1]}");

        // Perform dense layer computation: output = input � weights^T + biases
        // Optimization: Use efficient tensor operations without creating intermediate tensors
        var output = input.MatrixMultiply(_weights.Transpose());
        output = output.Add(_biases);

        // Apply activation function
        if (UsingVectorActivation)
        {
            return VectorActivation!.Activate(output);
        }
        else
        {
            // Apply scalar activation efficiently using tensor operations
            var result = new Tensor<T>(output.Shape);

            if (output.Rank == 2)
            {
                // Fast path for common case of 2D tensors
                for (int i = 0; i < output.Shape[0]; i++)
                {
                    for (int j = 0; j < output.Shape[1]; j++)
                    {
                        result[i, j] = ScalarActivation!.Activate(output[i, j]);
                    }
                }
            }
            else
            {
                // Apply to arbitrary rank tensors
                int[] indices = new int[output.Rank];
                FillActivationTensor(output, result, indices, 0);
            }

            return result;
        }
    }

    /// <summary>
    /// Recursively applies activation function to tensor elements.
    /// </summary>
    private void FillActivationTensor(Tensor<T> input, Tensor<T> output, int[] indices, int dimension)
    {
        if (dimension == input.Rank)
        {
            output[indices] = ScalarActivation!.Activate(input[indices]);
            return;
        }

        for (int i = 0; i < input.Shape[dimension]; i++)
        {
            indices[dimension] = i;
            FillActivationTensor(input, output, indices, dimension + 1);
        }
    }

    /// <summary>
    /// Performs backward pass computation to calculate gradients.
    /// </summary>
    /// <param name="outputGradient">Gradient tensor flowing backward from the next layer.</param>
    /// <returns>Gradient tensor with respect to this layer's input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass");

        // 1. Apply activation derivative to output gradient
        Tensor<T> activationGradient;

        if (UsingVectorActivation)
        {
            activationGradient = VectorActivation!.Derivative(outputGradient);
        }
        else
        {
            // Apply scalar activation derivative efficiently
            activationGradient = new Tensor<T>(outputGradient.Shape);

            if (outputGradient.Rank == 2)
            {
                // Optimize for common 2D case
                for (int i = 0; i < outputGradient.Shape[0]; i++)
                {
                    for (int j = 0; j < outputGradient.Shape[1]; j++)
                    {
                        activationGradient[i, j] = ScalarActivation!.Derivative(outputGradient[i, j]);
                    }
                }
            }
            else
            {
                // Handle arbitrary dimensions
                int[] indices = new int[outputGradient.Rank];
                for (int flatIndex = 0; flatIndex < outputGradient.Length; flatIndex++)
                {
                    outputGradient.GetIndicesFromFlatIndex(flatIndex, indices);
                    activationGradient[indices] = ScalarActivation!.Derivative(outputGradient[indices]);
                }
            }
        }

        // 2. Calculate weight gradients: weightsGradient = activationGradient^T � input
        // Ensure input has correct shape [batchSize, inputFeatures]
        int batchSize = _lastInput.Shape[0];
        var inputReshaped = _lastInput;

        if (_lastInput.Rank > 2)
        {
            // Reshape only if necessary
            inputReshaped = _lastInput.Reshape(batchSize, _lastInput.Length / batchSize);
        }

        // Compute weight gradients directly with tensor operations
        _weightsGradient = activationGradient.Transpose().MatrixMultiply(inputReshaped);

        // 3. Calculate bias gradients by summing over batch dimension
        _biasesGradient = activationGradient.Sum([0]);

        // 4. Calculate input gradients: inputGradient = activationGradient � weights
        var inputGradient = activationGradient.MatrixMultiply(_weights);

        // Ensure output gradient has same shape as original input
        if (_lastInput.Rank > 2)
        {
            inputGradient = inputGradient.Reshape(_lastInput.Shape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates layer parameters using calculated gradients.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters");

        // Update weights and biases using tensor operations
        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
        _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all layer parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        int totalParams = _weights.Length + _biases.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy weights
        for (int i = 0; i < _weights.Shape[0]; i++)
        {
            for (int j = 0; j < _weights.Shape[1]; j++)
            {
                parameters[index++] = _weights[i, j];
            }
        }

        // Copy biases
        for (int i = 0; i < _biases.Length; i++)
        {
            parameters[index++] = _biases[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets all layer parameters from a single vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Length + _biases.Length)
            throw new ArgumentException($"Expected {_weights.Length + _biases.Length} parameters, got {parameters.Length}");

        int index = 0;

        // Set weights
        for (int i = 0; i < _weights.Shape[0]; i++)
        {
            for (int j = 0; j < _weights.Shape[1]; j++)
            {
                _weights[i, j] = parameters[index++];
            }
        }

        // Set biases
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets layer state between sequences or batches.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Creates a deep copy of this layer.
    /// </summary>
    public override LayerBase<T> Clone()
    {
        DenseLayer<T> copy;

        if (UsingVectorActivation)
        {
            copy = new DenseLayer<T>(InputShape[0], OutputShape[0], VectorActivation);
        }
        else
        {
            copy = new DenseLayer<T>(InputShape[0], OutputShape[0], ScalarActivation);
        }

        // Copy all parameters
        copy._weights = _weights.Clone();
        copy._biases = _biases.Clone();

        return copy;
    }

    /// <summary>
    /// Gets the weights tensor of this layer.
    /// </summary>
    /// <returns>The weights tensor.</returns>
    public Tensor<T> GetWeights()
    {
        return _weights.Clone();
    }

    /// <summary>
    /// Sets the weights tensor of this layer.
    /// </summary>
    /// <param name="weights">The new weights tensor.</param>
    public void SetWeights(Tensor<T> weights)
    {
        if (weights.Shape[0] != _weights.Shape[0] || weights.Shape[1] != _weights.Shape[1])
        {
            throw new ArgumentException($"Weight shape mismatch. Expected {_weights.Shape[0]}x{_weights.Shape[1]}, got {weights.Shape[0]}x{weights.Shape[1]}");
        }
        _weights = weights.Clone();
    }

    /// <summary>
    /// Gets the biases tensor of this layer.
    /// </summary>
    /// <returns>The biases tensor.</returns>
    public Tensor<T> GetBiases()
    {
        return _biases.Clone();
    }

    /// <summary>
    /// Sets the biases tensor of this layer.
    /// </summary>
    /// <param name="biases">The new biases tensor.</param>
    public void SetBiases(Tensor<T> biases)
    {
        if (biases.Length != _biases.Length)
        {
            throw new ArgumentException($"Bias length mismatch. Expected {_biases.Length}, got {biases.Length}");
        }
        _biases = biases.Clone();
    }
}