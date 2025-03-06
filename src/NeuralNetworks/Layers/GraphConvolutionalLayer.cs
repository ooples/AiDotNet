namespace AiDotNet.NeuralNetworks.Layers;

public class GraphConvolutionalLayer<T> : LayerBase<T>
{
    private Matrix<T> _weights;
    private Vector<T> _bias;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _adjacencyMatrix;

    private Matrix<T>? _weightsGradient;
    private Vector<T>? _biasGradient;

    public override bool SupportsTraining => true;

    public GraphConvolutionalLayer(int inputFeatures, int outputFeatures, IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _weights = new Matrix<T>(inputFeatures, outputFeatures);
        _bias = new Vector<T>(outputFeatures);

        InitializeParameters();
    }

    public GraphConvolutionalLayer(int inputFeatures, int outputFeatures, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([inputFeatures], [outputFeatures], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _weights = new Matrix<T>(inputFeatures, outputFeatures);
        _bias = new Vector<T>(outputFeatures);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_weights.Rows + _weights.Columns)));
        InitializeMatrix(_weights, scale);

        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = NumOps.Zero;
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

    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _adjacencyMatrix = adjacencyMatrix;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException("Adjacency matrix must be set using the SetAdjacencyMatrix method before calling Forward.");
        }

        _lastInput = input;

        int batchSize = input.Shape[0];
        int numNodes = input.Shape[1];
        int inputFeatures = input.Shape[2];
        int outputFeatures = _weights.Columns;

        // Perform graph convolution: A * X * W
        var xw = input.Multiply(_weights);
        var output = _adjacencyMatrix.Multiply(xw.ToMatrix());

        // Add bias
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < outputFeatures; f++)
                {
                    output[b, n, f] = NumOps.Add(output[b, n, f], _bias[f]);
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];
        int inputFeatures = _lastInput.Shape[2];
        int outputFeatures = _weights.Columns;

        // Calculate gradients for weights and bias
        _weightsGradient = new Matrix<T>(inputFeatures, outputFeatures);
        _biasGradient = new Vector<T>(outputFeatures);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    for (int f_in = 0; f_in < inputFeatures; f_in++)
                    {
                        for (int f_out = 0; f_out < outputFeatures; f_out++)
                        {
                            T gradValue = NumOps.Multiply(_adjacencyMatrix[b, i, j], 
                                NumOps.Multiply(_lastInput[b, j, f_in], activationGradient[b, i, f_out]));
                            _weightsGradient[f_in, f_out] = NumOps.Add(_weightsGradient[f_in, f_out], gradValue);
                        }
                    }
                }
            }
        }

        // Calculate bias gradient
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < outputFeatures; f++)
                {
                    _biasGradient[f] = NumOps.Add(_biasGradient[f], activationGradient[b, n, f]);
                }
            }
        }

        // Calculate input gradient
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    for (int f_in = 0; f_in < inputFeatures; f_in++)
                    {
                        for (int f_out = 0; f_out < outputFeatures; f_out++)
                        {
                            T gradValue = NumOps.Multiply(_adjacencyMatrix[b, j, i],
                                NumOps.Multiply(activationGradient[b, j, f_out], _weights[f_in, f_out]));
                            inputGradient[b, i, f_in] = NumOps.Add(inputGradient[b, i, f_in], gradValue);
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
        _bias = _bias.Subtract(_biasGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weights.Rows * _weights.Columns + _bias.Length;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy weights parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                parameters[index++] = _weights[i, j];
            }
        }
    
        // Copy bias parameters
        for (int i = 0; i < _bias.Length; i++)
        {
            parameters[index++] = _bias[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Rows * _weights.Columns + _bias.Length)
        {
            throw new ArgumentException($"Expected {_weights.Rows * _weights.Columns + _bias.Length} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weights parameters
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = parameters[index++];
            }
        }
    
        // Set bias parameters
        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasGradient = null;
    }
}