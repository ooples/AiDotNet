namespace AiDotNet.NeuralNetworks.Layers;

public class FullyConnectedLayer<T> : LayerBase<T>
{
    private Matrix<T> _weights;
    private Vector<T> _biases;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Matrix<T>? _weightsGradient;
    private Vector<T>? _biasesGradient;

    public FullyConnectedLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);

        InitializeParameters();
    }

    public FullyConnectedLayer(int inputSize, int outputSize, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([inputSize], [outputSize], vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);

        InitializeParameters();
    }

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
            outputVector = ApplyActivationToVector(outputVector);

            for (int j = 0; j < outputSize; j++)
            {
                output[i, j] = outputVector[j];
            }
        }

        _lastOutput = output;
        return output;
    }

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

    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
        _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
    }
}