namespace AiDotNet.NeuralNetworks.Layers;

public class CapsuleLayer<T> : LayerBase<T>
{
    private readonly int _numCapsules;
    private readonly int _capsuleDimension;
    private readonly int _numRoutingIterations;
    private Tensor<T> _transformationMatrix;
    private Vector<T> _bias;
    private Tensor<T>? _transformationMatrixGradient;
    private Vector<T>? _biasGradient;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastCouplingCoefficients;

    public CapsuleLayer(int inputCapsules, int inputDimension, int numCapsules, int capsuleDimension, int numRoutingIterations, IActivationFunction<T>? activationFunction = null)
        : base([inputCapsules, inputDimension], [numCapsules, capsuleDimension], activationFunction ?? new SquashActivation<T>())
    {
        _numCapsules = numCapsules;
        _capsuleDimension = capsuleDimension;
        _numRoutingIterations = numRoutingIterations;

        _transformationMatrix = new Tensor<T>([inputCapsules, inputDimension, numCapsules, capsuleDimension]);
        _bias = new Vector<T>(numCapsules * capsuleDimension);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        int totalElements = _transformationMatrix.Shape.Aggregate(1, (acc, dim) => acc * dim);
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / totalElements));
        InitializeTensor(_transformationMatrix, scale);

        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = NumOps.Zero;
        }
    }

    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        for (int i = 0; i < tensor.Shape.Aggregate(1, (acc, dim) => acc * dim); i++)
        {
            tensor.SetFlatIndex(i, NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale));
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputCapsules = input.Shape[1];
        int inputDimension = input.Shape[2];

        // Reshape input for matrix multiplication
        var reshapedInput = input.Reshape(batchSize * inputCapsules, inputDimension);

        // Perform transformation
        var transformedInput = reshapedInput.Multiply(_transformationMatrix);
        transformedInput = transformedInput.Reshape(batchSize, inputCapsules, _numCapsules, _capsuleDimension);

        // Initialize coupling coefficients
        var couplingCoefficients = new Tensor<T>([batchSize, inputCapsules, _numCapsules]);
        couplingCoefficients.Fill(NumOps.FromDouble(1.0 / _numCapsules));

        // Declare output tensor outside the loop
        Tensor<T> output = null!;

        // Perform dynamic routing
        for (int i = 0; i < _numRoutingIterations; i++)
        {
            var weightedSum = new Tensor<T>([batchSize, _numCapsules, _capsuleDimension]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int j = 0; j < inputCapsules; j++)
                {
                    for (int k = 0; k < _numCapsules; k++)
                    {
                        for (int d = 0; d < _capsuleDimension; d++)
                        {
                            weightedSum[b, k, d] = NumOps.Add(weightedSum[b, k, d], 
                                NumOps.Multiply(couplingCoefficients[b, j, k], transformedInput[b, j, k, d]));
                        }
                    }
                }
            }

            // Apply bias after the weighted sum
            for (int b = 0; b < batchSize; b++)
            {
                for (int k = 0; k < _numCapsules; k++)
                {
                    for (int d = 0; d < _capsuleDimension; d++)
                    {
                        weightedSum[b, k, d] = NumOps.Add(weightedSum[b, k, d], _bias[k * _capsuleDimension + d]);
                    }
                }
            }

            // Apply squash activation
            output = ApplyActivation(weightedSum);

            // Update coupling coefficients
            if (i < _numRoutingIterations - 1)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    for (int j = 0; j < inputCapsules; j++)
                    {
                        for (int k = 0; k < _numCapsules; k++)
                        {
                            T agreement = NumOps.Zero;
                            for (int d = 0; d < _capsuleDimension; d++)
                            {
                                agreement = NumOps.Add(agreement, 
                                    NumOps.Multiply(transformedInput[b, j, k, d], output[b, k, d]));
                            }
                            couplingCoefficients[b, j, k] = NumOps.Add(couplingCoefficients[b, j, k], agreement);
                        }
                    }
                }

                couplingCoefficients = ApplySoftmax(couplingCoefficients);
            }
        }

        _lastOutput = output;
        _lastCouplingCoefficients = couplingCoefficients;

        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastCouplingCoefficients == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputCapsules = _lastInput.Shape[1];
        int inputDimension = _lastInput.Shape[2];

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        _transformationMatrixGradient = new Tensor<T>([inputCapsules, inputDimension, _numCapsules, _capsuleDimension]);
        _biasGradient = new Vector<T>(_numCapsules * _capsuleDimension);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputCapsules; i++)
            {
                for (int j = 0; j < _numCapsules; j++)
                {
                    for (int d = 0; d < _capsuleDimension; d++)
                    {
                        T grad = outputGradient[b, j, d];
                        T coeff = _lastCouplingCoefficients[b, i, j];

                        // Update bias gradient
                        _biasGradient[j * _capsuleDimension + d] = NumOps.Add(
                            _biasGradient[j * _capsuleDimension + d],
                            grad
                        );

                        for (int k = 0; k < inputDimension; k++)
                        {
                            T input = _lastInput[b, i, k];
                            _transformationMatrixGradient[i, k, j, d] = NumOps.Add(
                                _transformationMatrixGradient[i, k, j, d], 
                                NumOps.Multiply(NumOps.Multiply(grad, coeff), input)
                            );
                            inputGradient[b, i, k] = NumOps.Add(
                                inputGradient[b, i, k], 
                                NumOps.Multiply(NumOps.Multiply(grad, coeff), _transformationMatrix[i, k, j, d])
                            );
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_transformationMatrixGradient == null || _biasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _transformationMatrix = _transformationMatrix.Subtract(_transformationMatrixGradient.Multiply(learningRate));
        _bias = _bias.Subtract(_biasGradient.Multiply(learningRate));
    }

    private Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        var softmax = new SoftmaxActivation<T>();
        return softmax.Activate(input);
    }
}