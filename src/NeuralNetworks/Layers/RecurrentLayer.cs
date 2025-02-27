namespace AiDotNet.NeuralNetworks.Layers;

public class RecurrentLayer<T> : LayerBase<T>
{
    private Matrix<T> _inputWeights;
    private Matrix<T> _hiddenWeights;
    private Vector<T> _biases;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastHiddenState;
    private Tensor<T>? _lastOutput;
    private Matrix<T>? _inputWeightsGradient;
    private Matrix<T>? _hiddenWeightsGradient;
    private Vector<T>? _biasesGradient;

    public RecurrentLayer(int inputSize, int hiddenSize, IActivationFunction<T>? activationFunction = null)
        : base([inputSize], [hiddenSize], activationFunction ?? new TanhActivation<T>())
    {
        _inputWeights = new Matrix<T>(hiddenSize, inputSize);
        _hiddenWeights = new Matrix<T>(hiddenSize, hiddenSize);
        _biases = new Vector<T>(hiddenSize);

        InitializeParameters();
    }

    public RecurrentLayer(int inputSize, int hiddenSize, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([inputSize], [hiddenSize], vectorActivationFunction ?? new TanhActivation<T>())
    {
        _inputWeights = new Matrix<T>(hiddenSize, inputSize);
        _hiddenWeights = new Matrix<T>(hiddenSize, hiddenSize);
        _biases = new Vector<T>(hiddenSize);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Initialize weights and biases (e.g., Xavier/Glorot initialization)
        T inputScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputWeights.Rows + _inputWeights.Columns)));
        T hiddenScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_hiddenWeights.Rows + _hiddenWeights.Columns)));

        for (int i = 0; i < _inputWeights.Rows; i++)
        {
            for (int j = 0; j < _inputWeights.Columns; j++)
            {
                _inputWeights[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), inputScale);
            }

            for (int j = 0; j < _hiddenWeights.Columns; j++)
            {
                _hiddenWeights[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), hiddenScale);
            }

            _biases[i] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int sequenceLength = input.Shape[0];
        int batchSize = input.Shape[1];
        int inputSize = input.Shape[2];
        int hiddenSize = _inputWeights.Rows;

        var output = new Tensor<T>([sequenceLength, batchSize, hiddenSize]);
        var hiddenState = new Tensor<T>([sequenceLength + 1, batchSize, hiddenSize]);

        // Initialize the first hidden state with zeros
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < hiddenSize; h++)
            {
                hiddenState[0, b, h] = NumOps.Zero;
            }
        }

        for (int t = 0; t < sequenceLength; t++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                var inputVector = new Vector<T>(inputSize);
                var prevHiddenVector = new Vector<T>(hiddenSize);

                for (int i = 0; i < inputSize; i++)
                {
                    inputVector[i] = input[t, b, i];
                }
                for (int h = 0; h < hiddenSize; h++)
                {
                    prevHiddenVector[h] = hiddenState[t, b, h];
                }

                var newHiddenVector = _inputWeights.Multiply(inputVector)
                    .Add(_hiddenWeights.Multiply(prevHiddenVector))
                    .Add(_biases);

                newHiddenVector = ApplyActivationToVector(newHiddenVector);

                for (int h = 0; h < hiddenSize; h++)
                {
                    output[t, b, h] = newHiddenVector[h];
                    hiddenState[t + 1, b, h] = newHiddenVector[h];
                }
            }
        }

        _lastHiddenState = hiddenState;
        _lastOutput = output;

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastHiddenState == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int sequenceLength = _lastInput.Shape[0];
        int batchSize = _lastInput.Shape[1];
        int inputSize = _lastInput.Shape[2];
        int hiddenSize = _inputWeights.Rows;

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        var inputWeightsGradient = new Matrix<T>(_inputWeights.Rows, _inputWeights.Columns);
        var hiddenWeightsGradient = new Matrix<T>(_hiddenWeights.Rows, _hiddenWeights.Columns);
        var biasesGradient = new Vector<T>(_biases.Length);

        var nextHiddenGradient = new Tensor<T>([batchSize, hiddenSize]);

        for (int t = sequenceLength - 1; t >= 0; t--)
        {
            var currentGradient = new Tensor<T>([batchSize, hiddenSize]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    currentGradient[b, h] = NumOps.Add(outputGradient[t, b, h], nextHiddenGradient[b, h]);
                }

                var inputVector = new Vector<T>(inputSize);
                var prevHiddenVector = new Vector<T>(hiddenSize);
                var gradientVector = new Vector<T>(hiddenSize);

                for (int i = 0; i < inputSize; i++)
                {
                    inputVector[i] = _lastInput[t, b, i];
                }
                for (int h = 0; h < hiddenSize; h++)
                {
                    prevHiddenVector[h] = _lastHiddenState[t, b, h];
                    gradientVector[h] = currentGradient[b, h];
                }

                var activationDerivative = ApplyActivationDerivative(prevHiddenVector, gradientVector);

                inputWeightsGradient = inputWeightsGradient.Add(Matrix<T>.OuterProduct(activationDerivative, inputVector));
                hiddenWeightsGradient = hiddenWeightsGradient.Add(Matrix<T>.OuterProduct(activationDerivative, prevHiddenVector));
                biasesGradient = biasesGradient.Add(activationDerivative);

                var inputGradientVector = _inputWeights.Transpose().Multiply(activationDerivative);
                var hiddenGradientVector = _hiddenWeights.Transpose().Multiply(activationDerivative);

                for (int i = 0; i < inputSize; i++)
                {
                    inputGradient[t, b, i] = inputGradientVector[i];
                }
                for (int h = 0; h < hiddenSize; h++)
                {
                    nextHiddenGradient[b, h] = hiddenGradientVector[h];
                }
            }
        }

        _inputWeightsGradient = inputWeightsGradient;
        _hiddenWeightsGradient = hiddenWeightsGradient;
        _biasesGradient = biasesGradient;

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_inputWeightsGradient == null || _hiddenWeightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _inputWeights = _inputWeights.Subtract(_inputWeightsGradient.Multiply(learningRate));
        _hiddenWeights = _hiddenWeights.Subtract(_hiddenWeightsGradient.Multiply(learningRate));
        _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
    }
}