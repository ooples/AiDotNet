namespace AiDotNet.NeuralNetworks.Layers;

public class MultiHeadAttentionLayer<T> : LayerBase<T>
{
    private Matrix<T> _queryWeights;
    private Matrix<T> _keyWeights;
    private Matrix<T> _valueWeights;
    private Matrix<T> _outputWeights;
    private Vector<T> _outputBias;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastAttentionScores;

    private Matrix<T>? _queryWeightsGradient;
    private Matrix<T>? _keyWeightsGradient;
    private Matrix<T>? _valueWeightsGradient;
    private Matrix<T>? _outputWeightsGradient;
    private Vector<T>? _outputBiasGradient;

    private readonly int _headCount;
    private readonly int _headDimension;

    public override bool SupportsTraining => true;

    public MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IActivationFunction<T>? activationFunction = null)
        : base([sequenceLength, embeddingDimension], [sequenceLength, embeddingDimension], activationFunction ?? new IdentityActivation<T>())
    {
        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeParameters();
    }

    public MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([sequenceLength, embeddingDimension], [sequenceLength, embeddingDimension], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

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

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape[1];
        int embeddingDimension = input.Shape[2];

        var queries = input.Multiply(_queryWeights);
        var keys = input.Multiply(_keyWeights);
        var values = input.Multiply(_valueWeights);

        queries = queries.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        keys = keys.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        values = values.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);

        var attentionScores = queries.Multiply(keys.Transpose([0, 1, 3, 2]));
        attentionScores = attentionScores.Multiply(NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension)));

        var softmaxActivation = new SoftmaxActivation<T>();
        var attentionWeights = softmaxActivation.Activate(attentionScores);
        _lastAttentionScores = attentionWeights;

        var attentionOutput = attentionWeights.Multiply(values);
        attentionOutput = attentionOutput.Transpose([0, 2, 1, 3]).Reshape(batchSize, sequenceLength, embeddingDimension);

        var output = attentionOutput.Multiply(_outputWeights).Add(_outputBias);
        _lastOutput = ApplyActivation(output);

        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastAttentionScores == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];
        int embeddingDimension = _lastInput.Shape[2];

        var attentionOutputGradient = activationGradient.Multiply(_outputWeights.Transpose());
        _outputWeightsGradient = activationGradient.Transpose([0, 2, 1]).Multiply(_lastOutput).Sum([0]).ToMatrix();
        _outputBiasGradient = activationGradient.Sum([0, 1]).ToVector();

        attentionOutputGradient = attentionOutputGradient.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 1, 3]);

        var softmaxActivation = new SoftmaxActivation<T>();
        var softmaxDerivative = softmaxActivation.Derivative(_lastAttentionScores);
        var attentionWeightsGradient = softmaxDerivative.ElementwiseMultiply(attentionOutputGradient.Multiply(_lastInput.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 3, 1])));

        var queriesGradient = attentionWeightsGradient.Multiply(_lastInput.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 3, 1]));
        var keysGradient = attentionWeightsGradient.Transpose([0, 1, 3, 2]).Multiply(_lastInput.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 1, 3]));
        var valuesGradient = _lastAttentionScores.Transpose([0, 1, 3, 2]).Multiply(attentionOutputGradient);

        queriesGradient = queriesGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);
        keysGradient = keysGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);
        valuesGradient = valuesGradient.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);

        _queryWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(queriesGradient).Sum([0]).ToMatrix();
        _keyWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(keysGradient).Sum([0]).ToMatrix();
        _valueWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(valuesGradient).Sum([0]).ToMatrix();

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
        _lastOutput = null;
        _lastAttentionScores = null;
    
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }
}