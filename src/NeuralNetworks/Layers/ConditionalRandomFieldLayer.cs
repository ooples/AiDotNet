namespace AiDotNet.NeuralNetworks.Layers;

public class ConditionalRandomFieldLayer<T> : LayerBase<T>
{
    private Matrix<T> _transitionMatrix;
    private Vector<T> _startScores;
    private Vector<T> _endScores;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    private Matrix<T>? _transitionMatrixGradient;
    private Vector<T>? _startScoresGradient;
    private Vector<T>? _endScoresGradient;

    private readonly int _numClasses;
    private readonly int _sequenceLength;

    public override bool SupportsTraining => true;

    public ConditionalRandomFieldLayer(int numClasses, int sequenceLength, IActivationFunction<T>? scalarActivation = null)
        : base([sequenceLength, numClasses], [sequenceLength, numClasses], scalarActivation ?? new IdentityActivation<T>())
    {
        _numClasses = numClasses;
        _sequenceLength = sequenceLength;
        _transitionMatrix = new Matrix<T>(_numClasses, _numClasses);
        _startScores = new Vector<T>(_numClasses);
        _endScores = new Vector<T>(_numClasses);

        InitializeParameters();
    }

    public ConditionalRandomFieldLayer(int numClasses, int sequenceLength, IVectorActivationFunction<T>? vectorActivation = null)
        : base([sequenceLength, numClasses], [sequenceLength, numClasses], vectorActivation ?? new IdentityActivation<T>())
    {
        _numClasses = numClasses;
        _sequenceLength = sequenceLength;
        _transitionMatrix = new Matrix<T>(_numClasses, _numClasses);
        _startScores = new Vector<T>(_numClasses);
        _endScores = new Vector<T>(_numClasses);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_numClasses + _numClasses)));
        InitializeMatrix(_transitionMatrix, scale);
        InitializeVector(_startScores, scale);
        InitializeVector(_endScores, scale);
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

    private void InitializeVector(Vector<T> vector, T scale)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        var output = new Tensor<T>([batchSize, _sequenceLength, _numClasses]);

        for (int b = 0; b < batchSize; b++)
        {
            var sequenceScores = new Matrix<T>(_sequenceLength, _numClasses);

            // Apply optional feature transformation
            for (int t = 0; t < _sequenceLength; t++)
            {
                var featureVector = new Vector<T>(_numClasses);
                for (int c = 0; c < _numClasses; c++)
                {
                    featureVector[c] = input[b, t, c];
                }

                if (UsingVectorActivation)
                {
                    featureVector = VectorActivation!.Activate(featureVector);
                }
                else if (ScalarActivation != null)
                {
                    for (int c = 0; c < _numClasses; c++)
                    {
                        featureVector[c] = ScalarActivation.Activate(featureVector[c]);
                    }
                }

                for (int c = 0; c < _numClasses; c++)
                {
                    sequenceScores[t, c] = featureVector[c];
                }
            }

            // Viterbi algorithm
            var viterbi = new Matrix<T>(_sequenceLength, _numClasses);
            var backpointers = new Matrix<int>(_sequenceLength, _numClasses);

            // Initialize first timestep
            for (int c = 0; c < _numClasses; c++)
            {
                viterbi[0, c] = NumOps.Add(_startScores[c], sequenceScores[0, c]);
            }

            // Recursion
            for (int t = 1; t < _sequenceLength; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    T maxScore = NumOps.MinValue;
                    int maxPrevClass = -1;

                    for (int prevC = 0; prevC < _numClasses; prevC++)
                    {
                        T score = NumOps.Add(
                            NumOps.Add(
                                viterbi[t - 1, prevC],
                                _transitionMatrix[prevC, c]
                            ),
                            sequenceScores[t, c]
                        );

                        if (NumOps.GreaterThan(score, maxScore))
                        {
                            maxScore = score;
                            maxPrevClass = prevC;
                        }
                    }

                    viterbi[t, c] = maxScore;
                    backpointers[t, c] = maxPrevClass;
                }
            }

            // Termination
            T maxFinalScore = NumOps.MinValue;
            int maxFinalClass = -1;
            for (int c = 0; c < _numClasses; c++)
            {
                T finalScore = NumOps.Add(viterbi[_sequenceLength - 1, c], _endScores[c]);
                if (NumOps.GreaterThan(finalScore, maxFinalScore))
                {
                    maxFinalScore = finalScore;
                    maxFinalClass = c;
                }
            }

            // Backtracking
            var bestPath = new int[_sequenceLength];
            bestPath[_sequenceLength - 1] = maxFinalClass;
            for (int t = _sequenceLength - 2; t >= 0; t--)
            {
                bestPath[t] = backpointers[t + 1, bestPath[t + 1]];
            }

            // Set output
            for (int t = 0; t < _sequenceLength; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    output[b, t, c] = c == bestPath[t] ? NumOps.One : NumOps.Zero;
                }
            }
        }

        _lastOutput = output;
        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _transitionMatrixGradient = new Matrix<T>(_numClasses, _numClasses);
        _startScoresGradient = new Vector<T>(_numClasses);
        _endScoresGradient = new Vector<T>(_numClasses);

        for (int b = 0; b < batchSize; b++)
        {
            // Compute gradients for transition matrix, start scores, and end scores
            for (int t = 0; t < _sequenceLength; t++)
            {
                for (int c = 0; c < _numClasses; c++)
                {
                    T grad = outputGradient[b, t, c];

                    if (t == 0)
                    {
                        _startScoresGradient[c] = NumOps.Add(_startScoresGradient[c], grad);
                    }
                    else if (t == _sequenceLength - 1)
                    {
                        _endScoresGradient[c] = NumOps.Add(_endScoresGradient[c], grad);
                    }

                    if (t > 0)
                    {
                        for (int prevC = 0; c < _numClasses; prevC++)
                        {
                            _transitionMatrixGradient[prevC, c] = NumOps.Add(_transitionMatrixGradient[prevC, c], grad);
                        }
                    }

                    // Compute input gradient
                    inputGradient[b, t, c] = grad;
                }
            }
        }

        // Apply activation function gradient if applicable
        if (UsingVectorActivation)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < _sequenceLength; t++)
                {
                    var input = new Vector<T>(_numClasses);
                    var grad = new Vector<T>(_numClasses);
                    for (int c = 0; c < _numClasses; c++)
                    {
                        input[c] = _lastInput[b, t, c];
                        grad[c] = inputGradient[b, t, c];
                    }

                    var derivativeMatrix = VectorActivation!.Derivative(input);
                    var result = derivativeMatrix.Multiply(grad);

                    for (int c = 0; c < _numClasses; c++)
                    {
                        inputGradient[b, t, c] = result[c];
                    }
                }
            }
        }
        else if (ScalarActivation != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < _sequenceLength; t++)
                {
                    for (int c = 0; c < _numClasses; c++)
                    {
                        T derivative = ScalarActivation.Derivative(_lastInput[b, t, c]);
                        inputGradient[b, t, c] = NumOps.Multiply(derivative, inputGradient[b, t, c]);
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_transitionMatrixGradient == null || _startScoresGradient == null || _endScoresGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        for (int i = 0; i < _numClasses; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                _transitionMatrix[i, j] = NumOps.Subtract(_transitionMatrix[i, j], 
                    NumOps.Multiply(learningRate, _transitionMatrixGradient[i, j]));
            }

            _startScores[i] = NumOps.Subtract(_startScores[i], 
                NumOps.Multiply(learningRate, _startScoresGradient[i]));
            _endScores[i] = NumOps.Subtract(_endScores[i], 
                NumOps.Multiply(learningRate, _endScoresGradient[i]));
        }
    }

    public override Vector<T> GetParameters()
    {
        // Flatten all parameters into a single vector
        int totalParams = _numClasses * _numClasses + _numClasses * 2;
        var parameters = new Vector<T>(totalParams);
        
        int index = 0;
        
        // Copy transition matrix parameters
        for (int i = 0; i < _numClasses; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                parameters[index++] = _transitionMatrix[i, j];
            }
        }
        
        // Copy start scores
        for (int i = 0; i < _numClasses; i++)
        {
            parameters[index++] = _startScores[i];
        }
        
        // Copy end scores
        for (int i = 0; i < _numClasses; i++)
        {
            parameters[index++] = _endScores[i];
        }
        
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _numClasses * _numClasses + _numClasses * 2;
        
        if (parameters.Length != totalParams)
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        
        int index = 0;
        
        // Set transition matrix parameters
        for (int i = 0; i < _numClasses; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                _transitionMatrix[i, j] = parameters[index++];
            }
        }
        
        // Set start scores
        for (int i = 0; i < _numClasses; i++)
        {
            _startScores[i] = parameters[index++];
        }
        
        // Set end scores
        for (int i = 0; i < _numClasses; i++)
        {
            _endScores[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _transitionMatrixGradient = null;
        _startScoresGradient = null;
        _endScoresGradient = null;
    }
}