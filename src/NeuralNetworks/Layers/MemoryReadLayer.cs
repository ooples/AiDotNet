namespace AiDotNet.NeuralNetworks.Layers;

public class MemoryReadLayer<T> : LayerBase<T>
{
    private Matrix<T> _keyWeights;
    private Matrix<T> _valueWeights;
    private Matrix<T> _outputWeights;
    private Vector<T> _outputBias;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastMemory;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastAttentionScores;

    private Matrix<T>? _keyWeightsGradient;
    private Matrix<T>? _valueWeightsGradient;
    private Matrix<T>? _outputWeightsGradient;
    private Vector<T>? _outputBiasGradient;

    public override bool SupportsTraining => true;

    public MemoryReadLayer(int inputDimension, int memoryDimension, int outputDimension, IActivationFunction<T>? activationFunction = null)
        : base([inputDimension], [outputDimension], activationFunction ?? new LinearActivation<T>())
    {
        _keyWeights = new Matrix<T>(inputDimension, memoryDimension);
        _valueWeights = new Matrix<T>(memoryDimension, outputDimension);
        _outputWeights = new Matrix<T>(outputDimension, outputDimension);
        _outputBias = new Vector<T>(outputDimension);

        InitializeParameters();
    }

    public MemoryReadLayer(int inputDimension, int memoryDimension, int outputDimension, IVectorActivationFunction<T>? activationFunction = null)
        : base([inputDimension], [outputDimension], activationFunction ?? new LinearActivation<T>())
    {
        _keyWeights = new Matrix<T>(inputDimension, memoryDimension);
        _valueWeights = new Matrix<T>(memoryDimension, outputDimension);
        _outputWeights = new Matrix<T>(outputDimension, outputDimension);
        _outputBias = new Vector<T>(outputDimension);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_keyWeights.Rows + _keyWeights.Columns)));
        InitializeMatrix(_keyWeights, scale);
        InitializeMatrix(_valueWeights, scale);
        InitializeMatrix(_outputWeights, scale);

        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = NumOps.Zero;
        }
    }

    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    public Tensor<T> Forward(Tensor<T> input, Tensor<T> memory)
    {
        _lastInput = input;
        _lastMemory = memory;

        var keys = input.Multiply(_keyWeights);
        var attentionScores = keys.Multiply(memory.Transpose([1, 0]));
        
        var softmaxActivation = new SoftmaxActivation<T>();
        var attentionWeights = softmaxActivation.Activate(attentionScores);
        _lastAttentionScores = attentionWeights;

        var readValues = attentionWeights.Multiply(memory);
        var output = readValues.Multiply(_valueWeights).Multiply(_outputWeights).Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastMemory == null || _lastOutput == null || _lastAttentionScores == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        _outputWeightsGradient = activationGradient.Transpose([1, 0]).Multiply(_lastOutput).ToMatrix();
        _outputBiasGradient = activationGradient.Sum([0]).ToVector();

        var valueGradient = activationGradient.Multiply(_outputWeights.Transpose()).Multiply(_valueWeights.Transpose());

        var softmaxActivation = new SoftmaxActivation<T>();
        var softmaxDerivative = softmaxActivation.Derivative(_lastAttentionScores);
        var attentionWeightsGradient = softmaxDerivative.ElementwiseMultiply(valueGradient.Multiply(_lastMemory.Transpose([1, 0])));

        _keyWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(attentionWeightsGradient.Multiply(_lastMemory)).ToMatrix();
        _valueWeightsGradient = _lastMemory.Transpose([1, 0]).Multiply(_lastAttentionScores.Transpose([1, 0]).Multiply(activationGradient)).ToMatrix();

        var inputGradient = attentionWeightsGradient.Multiply(_keyWeights.Transpose());
        var memoryGradient = attentionWeightsGradient.Transpose([1, 0]).Multiply(_lastInput.Multiply(_keyWeights));

        // Combine inputGradient and memoryGradient into a single Tensor
        return CombineGradients(inputGradient, memoryGradient);
    }

    private static Tensor<T> CombineGradients(Tensor<T> inputGradient, Tensor<T> memoryGradient)
    {
        // Assuming we want to concatenate the gradients along the first dimension
        return Tensor<T>.Concatenate([inputGradient, memoryGradient], 0);
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_keyWeightsGradient == null || _valueWeightsGradient == null || _outputWeightsGradient == null || _outputBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _keyWeights = _keyWeights.Subtract(_keyWeightsGradient.Multiply(learningRate));
        _valueWeights = _valueWeights.Subtract(_valueWeightsGradient.Multiply(learningRate));
        _outputWeights = _outputWeights.Subtract(_outputWeightsGradient.Multiply(learningRate));
        _outputBias = _outputBias.Subtract(_outputBiasGradient.Multiply(learningRate));
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new InvalidOperationException("MemoryReadLayer requires both input and memory tensors. Use the Forward(Tensor<T> input, Tensor<T> memory) method instead.");
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
                          _outputWeights.Rows * _outputWeights.Columns +
                          _outputBias.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy key weights
        for (int i = 0; i < _keyWeights.Rows; i++)
        {
            for (int j = 0; j < _keyWeights.Columns; j++)
            {
                parameters[index++] = _keyWeights[i, j];
            }
        }
    
        // Copy value weights
        for (int i = 0; i < _valueWeights.Rows; i++)
        {
            for (int j = 0; j < _valueWeights.Columns; j++)
            {
                parameters[index++] = _valueWeights[i, j];
            }
        }
    
        // Copy output weights
        for (int i = 0; i < _outputWeights.Rows; i++)
        {
            for (int j = 0; j < _outputWeights.Columns; j++)
            {
                parameters[index++] = _outputWeights[i, j];
            }
        }
    
        // Copy output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            parameters[index++] = _outputBias[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
                          _outputWeights.Rows * _outputWeights.Columns +
                          _outputBias.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set key weights
        for (int i = 0; i < _keyWeights.Rows; i++)
        {
            for (int j = 0; j < _keyWeights.Columns; j++)
            {
                _keyWeights[i, j] = parameters[index++];
            }
        }
    
        // Set value weights
        for (int i = 0; i < _valueWeights.Rows; i++)
        {
            for (int j = 0; j < _valueWeights.Columns; j++)
            {
                _valueWeights[i, j] = parameters[index++];
            }
        }
    
        // Set output weights
        for (int i = 0; i < _outputWeights.Rows; i++)
        {
            for (int j = 0; j < _outputWeights.Columns; j++)
            {
                _outputWeights[i, j] = parameters[index++];
            }
        }
    
        // Set output bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastMemory = null;
        _lastOutput = null;
        _lastAttentionScores = null;
    
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }
}