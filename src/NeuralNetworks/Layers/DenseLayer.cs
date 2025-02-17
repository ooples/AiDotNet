namespace AiDotNet.NeuralNetworks.Layers;

public class DenseLayer<T> : LayerBase<T>
{
    private Matrix<T> _weights;
    private Vector<T> _biases;
    private Matrix<T>? _weightsGradient;
    private Vector<T>? _biasesGradient;
    private Tensor<T>? _lastInput;

    public DenseLayer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [outputSize], activationFunction ?? new ReLUActivation<T>())
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);
        InitializeParameters();
    }

    public DenseLayer(int inputSize, int outputSize, IVectorActivationFunction<T> vectorActivation)
        : base([inputSize], [outputSize], vectorActivation)
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
                _weights[i, j] = (T)Convert.ChangeType(random.NextDouble() * scale - scale / 2, typeof(T));
            }
            _biases[i] = NumOps.Zero; // Initialize biases to zero
        }
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
}