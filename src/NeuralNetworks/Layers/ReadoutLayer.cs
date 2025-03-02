namespace AiDotNet.NeuralNetworks.Layers;

public class ReadoutLayer<T> : LayerBase<T>
{
    private Matrix<T> _weights;
    private Vector<T> _bias;
    private Matrix<T> _weightGradients;
    private Vector<T> _biasGradients;
    private Vector<T>? _lastInput;

    public override bool SupportsTraining => true;

    public ReadoutLayer(int inputSize, int outputSize, IActivationFunction<T> scalarActivation) 
        : base([inputSize], [outputSize], scalarActivation)
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _bias = new Vector<T>(outputSize);
        _weightGradients = new Matrix<T>(outputSize, inputSize);
        _biasGradients = new Vector<T>(outputSize);

        InitializeParameters(inputSize, outputSize);
    }

    public ReadoutLayer(int inputSize, int outputSize, IVectorActivationFunction<T> vectorActivation) 
        : base([inputSize], [outputSize], vectorActivation)
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _bias = new Vector<T>(outputSize);
        _weightGradients = new Matrix<T>(outputSize, inputSize);
        _biasGradients = new Vector<T>(outputSize);

        InitializeParameters(inputSize, outputSize);
    }

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

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    
        // Reset gradients
        _weightGradients = new Matrix<T>(_weights.Rows, _weights.Columns);
        _biasGradients = new Vector<T>(_bias.Length);
    }
}