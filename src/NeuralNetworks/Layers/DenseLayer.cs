namespace AiDotNet.NeuralNetworks.Layers;

public class DenseLayer<T> : LayerBase<T>
{
    private Matrix<T> _weights;
    private Vector<T> _biases;
    private Matrix<T>? _weightsGradient;
    private Vector<T>? _biasesGradient;
    private Tensor<T>? _lastInput;

    public override int ParameterCount => (_weights.Rows * _weights.Columns) + _biases.Length;

    public override bool SupportsTraining => true;

    public DenseLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);

        InitializeParameters();
    }

    public DenseLayer(int inputSize, int outputSize, IVectorActivationFunction<T>? vectorActivation = null)
        : base([inputSize], [outputSize], vectorActivation ?? new ReLUActivation<T>())
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Initialize weights and biases (e.g., using Xavier/Glorot initialization)
        var random = new Random();
        var scale = Math.Sqrt(2.0 / (InputShape[0] + OutputShape[0]));

        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = NumOps.FromDouble(Random.NextDouble() * scale - scale / 2);
            }

            _biases[i] = NumOps.Zero; // Initialize biases to zero
        }
    }

    public void SetWeights(Matrix<T> weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        // Validate dimensions
        if (weights.Rows != OutputShape[0] || weights.Columns != InputShape[0])
        {
            throw new ArgumentException($"Weight matrix dimensions must be {OutputShape[0]}x{InputShape[0]}, but got {weights.Rows}x{weights.Columns}");
        }

        // Set the weights directly
        _weights = weights;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        var flattenedInput = input.Reshape(batchSize, input.Shape[1]);
        var output = flattenedInput.Multiply(_weights.Transpose()).Add(_biases);

        if (UsingVectorActivation)
        {
            return VectorActivation!.Activate(output);
        }
        else
        {
            return ApplyActivation(output);
        }
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        Tensor<T> activationGradient;
        if (UsingVectorActivation)
        {
            activationGradient = VectorActivation!.Derivative(outputGradient);
        }
        else
        {
            // Apply scalar activation derivative element-wise
            activationGradient = new Tensor<T>(outputGradient.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
            {
                activationGradient[i] = ScalarActivation!.Derivative(outputGradient[i]);
            }
        }

        var flattenedInput = _lastInput.Reshape(batchSize, _lastInput.Shape[1]);

        _weightsGradient = activationGradient.Transpose([1, 0]).ToMatrix().Multiply(flattenedInput.ToMatrix());
        _biasesGradient = activationGradient.Sum([0]).ToMatrix().ToColumnVector();

        var inputGradient = activationGradient.Multiply(_weights);

        return inputGradient.Reshape(_lastInput.Shape);
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
        _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weights.Rows * _weights.Columns + _biases.Length;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy weight parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                parameters[index++] = _weights[i, j];
            }
        }
    
        // Copy bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            parameters[index++] = _biases[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Rows * _weights.Columns + _biases.Length)
        {
            throw new ArgumentException($"Expected {_weights.Rows * _weights.Columns + _biases.Length} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weight parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = parameters[index++];
            }
        }
    
        // Set bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }
}