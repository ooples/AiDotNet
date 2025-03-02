namespace AiDotNet.NeuralNetworks.Layers;

public class EmbeddingLayer<T> : LayerBase<T>
{
    private Matrix<T> _embeddingMatrix;
    private Matrix<T>? _embeddingGradient;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => true;

    public EmbeddingLayer(int vocabularySize, int embeddingDimension)
        : base([1], [embeddingDimension])
    {
        _embeddingMatrix = new Matrix<T>(vocabularySize, embeddingDimension);
        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Initialize embedding matrix with small random values
        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _embeddingMatrix.Columns));

        for (int i = 0; i < _embeddingMatrix.Rows; i++)
        {
            for (int j = 0; j < _embeddingMatrix.Columns; j++)
            {
                _embeddingMatrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int sequenceLength = input.Shape[0];
        int batchSize = input.Shape[1];

        var output = new Tensor<T>([sequenceLength, batchSize, _embeddingMatrix.Columns]);

        for (int t = 0; t < sequenceLength; t++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                int index = Convert.ToInt32(input[t, b, 0]);
                for (int d = 0; d < _embeddingMatrix.Columns; d++)
                {
                    output[t, b, d] = _embeddingMatrix[index, d];
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int sequenceLength = _lastInput.Shape[0];
        int batchSize = _lastInput.Shape[1];

        _embeddingGradient = new Matrix<T>(_embeddingMatrix.Rows, _embeddingMatrix.Columns);

        for (int t = 0; t < sequenceLength; t++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                int index = Convert.ToInt32(_lastInput[t, b, 0]);
                for (int d = 0; d < _embeddingMatrix.Columns; d++)
                {
                    _embeddingGradient[index, d] = NumOps.Add(_embeddingGradient[index, d], outputGradient[t, b, d]);
                }
            }
        }

        // We don't compute input gradients for embedding layer
        return new Tensor<T>(_lastInput.Shape);
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_embeddingGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _embeddingMatrix = _embeddingMatrix.Subtract(_embeddingGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _embeddingMatrix.Rows * _embeddingMatrix.Columns;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy embedding matrix parameters
        for (int i = 0; i < _embeddingMatrix.Rows; i++)
        {
            for (int j = 0; j < _embeddingMatrix.Columns; j++)
            {
                parameters[index++] = _embeddingMatrix[i, j];
            }
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _embeddingMatrix.Rows * _embeddingMatrix.Columns)
        {
            throw new ArgumentException($"Expected {_embeddingMatrix.Rows * _embeddingMatrix.Columns} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set embedding matrix parameters
        for (int i = 0; i < _embeddingMatrix.Rows; i++)
        {
            for (int j = 0; j < _embeddingMatrix.Columns; j++)
            {
                _embeddingMatrix[i, j] = parameters[index++];
            }
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _embeddingGradient = null;
    }
}