namespace AiDotNet.NeuralNetworks.Layers;

public class DigitCapsuleLayer<T> : LayerBase<T>
{
    private Tensor<T> _weights;
    private Tensor<T>? _weightsGradient;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastCouplings;

    private readonly int _inputCapsules;
    private readonly int _inputCapsuleDimension;
    private readonly int _numClasses;
    private readonly int _outputCapsuleDimension;
    private readonly int _routingIterations;

    public override bool SupportsTraining => true;

    public DigitCapsuleLayer(int inputCapsules, int inputCapsuleDimension, int numClasses, int outputCapsuleDimension, int routingIterations)
        : base([inputCapsules, inputCapsuleDimension], [numClasses, outputCapsuleDimension], new SquashActivation<T>() as IActivationFunction<T>)
    {
        _inputCapsules = inputCapsules;
        _inputCapsuleDimension = inputCapsuleDimension;
        _numClasses = numClasses;
        _outputCapsuleDimension = outputCapsuleDimension;
        _routingIterations = routingIterations;
        _weights = new Tensor<T>([inputCapsules, numClasses, inputCapsuleDimension, outputCapsuleDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputCapsules * _inputCapsuleDimension)));
        for (int i = 0; i < _inputCapsules; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                for (int k = 0; k < _inputCapsuleDimension; k++)
                {
                    for (int l = 0; l < _outputCapsuleDimension; l++)
                    {
                        _weights[i, j, k, l] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                    }
                }
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        var predictions = new Tensor<T>(new[] { batchSize, _inputCapsules, _numClasses, _outputCapsuleDimension });
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _inputCapsules; i++)
            {
                for (int j = 0; j < _numClasses; j++)
                {
                    var inputCapsule = input.SubTensor(b, i);
                    var weightMatrix = _weights.SubTensor(i, j);
                    var result = inputCapsule.MatrixMultiply(weightMatrix);
                    predictions.SetSubTensor(new[] { b, i, j }, result);
                }
            }
        }

        var couplings = new Tensor<T>(new[] { batchSize, _inputCapsules, _numClasses });
        couplings.Fill(NumOps.Zero);

        var output = new Tensor<T>(new[] { batchSize, _numClasses, _outputCapsuleDimension });

        for (int iteration = 0; iteration < _routingIterations; iteration++)
        {
            var softmaxActivation = new SoftmaxActivation<T>();
            var routingWeights = softmaxActivation.Activate(couplings);

            for (int b = 0; b < batchSize; b++)
            {
                for (int j = 0; j < _numClasses; j++)
                {
                    var weightedSum = new Tensor<T>(new[] { _outputCapsuleDimension });
                    for (int i = 0; i < _inputCapsules; i++)
                    {
                        var predictionVector = predictions.SubTensor(b, i, j);
                        var scaledPrediction = predictionVector.Multiply(routingWeights[b, i, j]);
                        weightedSum = weightedSum.Add(scaledPrediction);
                    }
                    var activatedOutput = ApplyActivation(weightedSum);
                    output.SetSubTensor(new[] { b, j }, activatedOutput);
                }
            }

            if (iteration < _routingIterations - 1)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < _inputCapsules; i++)
                    {
                        for (int j = 0; j < _numClasses; j++)
                        {
                            var predictionVector = predictions.SubTensor(b, i, j);
                            var outputVector = output.SubTensor(b, j);
                            var dotProduct = predictionVector.DotProduct(outputVector);
                            couplings[b, i, j] = NumOps.Add(couplings[b, i, j], dotProduct);
                        }
                    }
                }
            }
        }

        _lastOutput = output;
        _lastCouplings = couplings;

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastCouplings == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        _weightsGradient = new Tensor<T>(_weights.Shape);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        var softmaxActivation = new SoftmaxActivation<T>();
        var routingWeights = softmaxActivation.Activate(_lastCouplings);
        var routingWeightsGradient = softmaxActivation.Derivative(_lastCouplings);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _inputCapsules; i++)
            {
                for (int j = 0; j < _numClasses; j++)
                {
                    var inputCapsule = _lastInput.SubTensor(b, i);
                    var outputCapsule = _lastOutput.SubTensor(b, j);
                    var predictionGradient = activationGradient.SubTensor(b, j).Multiply(routingWeights[b, i, j]);

                    for (int k = 0; k < _inputCapsuleDimension; k++)
                    {
                        for (int l = 0; l < _outputCapsuleDimension; l++)
                        {
                            _weightsGradient[i, j, k, l] = NumOps.Add(_weightsGradient[i, j, k, l],
                                NumOps.Multiply(inputCapsule[k], predictionGradient[l]));
                        }
                    }

                    var gradientUpdate = _weights.SubTensor(i, j).MatrixMultiply(predictionGradient);
                    for (int k = 0; k < _inputCapsuleDimension; k++)
                    {
                        inputGradient[b, i, k] = NumOps.Add(inputGradient[b, i, k], gradientUpdate[k]);
                    }

                    var couplingGradient = NumOps.Multiply(outputCapsule.ToVector().DotProduct(activationGradient.SubTensor(b, j).ToVector()),
                        routingWeightsGradient[b, i, j]);

                    var couplingGradientUpdate = _weights.SubTensor(i, j).MatrixMultiply(new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { couplingGradient })));
                    for (int k = 0; k < _inputCapsuleDimension; k++)
                    {
                        inputGradient[b, i, k] = NumOps.Add(inputGradient[b, i, k], couplingGradientUpdate[0, k]);
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weights.Length;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy weight parameters
        for (int i = 0; i < _inputCapsules; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                for (int k = 0; k < _inputCapsuleDimension; k++)
                {
                    for (int l = 0; l < _outputCapsuleDimension; l++)
                    {
                        parameters[index++] = _weights[i, j, k, l];
                    }
                }
            }
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Length)
        {
            throw new ArgumentException($"Expected {_weights.Length} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weight parameters
        for (int i = 0; i < _inputCapsules; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                for (int k = 0; k < _inputCapsuleDimension; k++)
                {
                    for (int l = 0; l < _outputCapsuleDimension; l++)
                    {
                        _weights[i, j, k, l] = parameters[index++];
                    }
                }
            }
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastCouplings = null;
        _weightsGradient = null;
    }
}