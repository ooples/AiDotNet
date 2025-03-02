namespace AiDotNet.NeuralNetworks.Layers;

public class MemoryWriteLayer<T> : LayerBase<T>
{
    private Matrix<T> _queryWeights;
    private Matrix<T> _keyWeights;
    private Matrix<T> _valueWeights;
    private Matrix<T> _outputWeights;
    private Vector<T> _outputBias;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastMemory;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastAttentionScores;

    private Matrix<T>? _queryWeightsGradient;
    private Matrix<T>? _keyWeightsGradient;
    private Matrix<T>? _valueWeightsGradient;
    private Matrix<T>? _outputWeightsGradient;
    private Vector<T>? _outputBiasGradient;

    public override bool SupportsTraining => true;

    public MemoryWriteLayer(int inputDimension, int memoryDimension, IActivationFunction<T>? activationFunction = null)
        : base([inputDimension], [memoryDimension], activationFunction ?? new LinearActivation<T>())
    {
        _queryWeights = new Matrix<T>(inputDimension, memoryDimension);
        _keyWeights = new Matrix<T>(inputDimension, memoryDimension);
        _valueWeights = new Matrix<T>(inputDimension, memoryDimension);
        _outputWeights = new Matrix<T>(memoryDimension, memoryDimension);
        _outputBias = new Vector<T>(memoryDimension);

        InitializeParameters();
    }

    public MemoryWriteLayer(int inputDimension, int memoryDimension, IVectorActivationFunction<T>? activationFunction = null)
        : base([inputDimension], [memoryDimension], activationFunction ?? new LinearActivation<T>())
    {
        _queryWeights = new Matrix<T>(inputDimension, memoryDimension);
        _keyWeights = new Matrix<T>(inputDimension, memoryDimension);
        _valueWeights = new Matrix<T>(inputDimension, memoryDimension);
        _outputWeights = new Matrix<T>(memoryDimension, memoryDimension);
        _outputBias = new Vector<T>(memoryDimension);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_queryWeights.Rows + _queryWeights.Columns)));
        InitializeMatrix(_queryWeights, scale);
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

        var queries = input.Multiply(_queryWeights);
        var keys = input.Multiply(_keyWeights);
        var values = input.Multiply(_valueWeights);

        var attentionScores = queries.Multiply(memory.Transpose([1, 0]));
        attentionScores = attentionScores.Multiply(NumOps.FromDouble(1.0 / Math.Sqrt(keys.Shape[1])));

        var softmaxActivation = new SoftmaxActivation<T>();
        var attentionWeights = softmaxActivation.Activate(attentionScores);
        _lastAttentionScores = attentionWeights;

        var writeValues = values.Multiply(attentionWeights);
        var output = writeValues.Multiply(_outputWeights).Add(_outputBias);
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

        var writeValuesGradient = activationGradient.Multiply(_outputWeights.Transpose());

        var softmaxActivation = new SoftmaxActivation<T>();
        var softmaxDerivative = softmaxActivation.Derivative(_lastAttentionScores);
        var attentionWeightsGradient = softmaxDerivative.ElementwiseMultiply(
            writeValuesGradient.Multiply(_lastInput.Multiply(_valueWeights).Transpose([1, 0])));

        var queriesGradient = attentionWeightsGradient.Multiply(_lastMemory);
        var keysGradient = attentionWeightsGradient.Transpose([1, 0]).Multiply(_lastInput);
        var valuesGradient = _lastAttentionScores.Transpose([1, 0]).Multiply(writeValuesGradient);

        _queryWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(queriesGradient).ToMatrix();
        _keyWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(keysGradient).ToMatrix();
        _valueWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(valuesGradient).ToMatrix();

        var inputGradient = queriesGradient.Multiply(_queryWeights.Transpose())
                            .Add(keysGradient.Multiply(_keyWeights.Transpose()))
                            .Add(valuesGradient.Multiply(_valueWeights.Transpose()));

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null || _outputWeightsGradient == null || _outputBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _queryWeights = _queryWeights.Subtract(_queryWeightsGradient.Multiply(learningRate));
        _keyWeights = _keyWeights.Subtract(_keyWeightsGradient.Multiply(learningRate));
        _valueWeights = _valueWeights.Subtract(_valueWeightsGradient.Multiply(learningRate));
        _outputWeights = _outputWeights.Subtract(_outputWeightsGradient.Multiply(learningRate));
        _outputBias = _outputBias.Subtract(_outputBiasGradient.Multiply(learningRate));
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // For a memory write layer, we need both input and memory
        // When only input is provided, we can create an empty memory tensor
        // or use a zero-initialized memory tensor of appropriate size
        int batchSize = input.Shape[0];
        int memoryDimension = _queryWeights.Columns;
    
        // Create an empty memory tensor with the same batch size as input
        // and the memory dimension of the layer
        var emptyMemory = new Tensor<T>([batchSize, memoryDimension]);
    
        // Initialize with zeros
        for (int i = 0; i < emptyMemory.Length; i++)
        {
            emptyMemory[i] = NumOps.Zero;
        }
    
        // Call the overloaded Forward method with the empty memory
        return Forward(input, emptyMemory);
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _queryWeights.Rows * _queryWeights.Columns +
                          _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
                          _outputWeights.Rows * _outputWeights.Columns +
                          _outputBias.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy query weights
        for (int i = 0; i < _queryWeights.Rows; i++)
        {
            for (int j = 0; j < _queryWeights.Columns; j++)
            {
                parameters[index++] = _queryWeights[i, j];
            }
        }
    
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
        int totalParams = _queryWeights.Rows * _queryWeights.Columns +
                          _keyWeights.Rows * _keyWeights.Columns +
                          _valueWeights.Rows * _valueWeights.Columns +
                          _outputWeights.Rows * _outputWeights.Columns +
                          _outputBias.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set query weights
        for (int i = 0; i < _queryWeights.Rows; i++)
        {
            for (int j = 0; j < _queryWeights.Columns; j++)
            {
                _queryWeights[i, j] = parameters[index++];
            }
        }
    
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
    
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }
}