namespace AiDotNet.NeuralNetworks.Layers;

public class ReservoirLayer<T> : LayerBase<T>
{
    private readonly int _inputSize;
    private readonly int _reservoirSize;
    private readonly double _connectionProbability;
    private readonly double _spectralRadius;
    private readonly double _inputScaling;
    private readonly double _leakingRate;

    private Matrix<T> _reservoirWeights;
    private Vector<T> _reservoirState;

    public override bool SupportsTraining => false;

    public ReservoirLayer(
        int inputSize, 
        int reservoirSize, 
        double connectionProbability = 0.1, 
        double spectralRadius = 0.9, 
        double inputScaling = 1.0, 
        double leakingRate = 1.0)
        : base([inputSize], [reservoirSize])
    {
        _inputSize = inputSize;
        _reservoirSize = reservoirSize;
        _connectionProbability = connectionProbability;
        _spectralRadius = spectralRadius;
        _inputScaling = inputScaling;
        _leakingRate = leakingRate;

        _reservoirWeights = new Matrix<T>(_reservoirSize, _reservoirSize);
        _reservoirState = new Vector<T>(_reservoirSize);

        InitializeReservoir();
    }

    private void InitializeReservoir()
    {
        // Initialize reservoir weights
        for (int i = 0; i < _reservoirWeights.Rows; i++)
        {
            for (int j = 0; j < _reservoirWeights.Columns; j++)
            {
                if (Random.NextDouble() < _connectionProbability)
                {
                    _reservoirWeights[i, j] = NumOps.FromDouble(Random.NextDouble() - 0.5);
                }
                else
                {
                    _reservoirWeights[i, j] = NumOps.Zero;
                }
            }
        }

        // Scale the reservoir weights to achieve the desired spectral radius
        T maxEigenvalue = ComputeMaxEigenvalue(_reservoirWeights);
        T scaleFactor = NumOps.FromDouble(_spectralRadius / Convert.ToDouble(maxEigenvalue));
        _reservoirWeights = _reservoirWeights.Multiply(scaleFactor);

        // Initialize reservoir state to zeros
        for (int i = 0; i < _reservoirState.Length; i++)
        {
            _reservoirState[i] = NumOps.Zero;
        }
    }

    private T ComputeMaxEigenvalue(Matrix<T> matrix)
    {
        // Power iteration method with improvements for better convergence
        int maxIterations = 1000;
        double tolerance = 1e-10;
    
        // Start with a random vector instead of all ones
        Vector<T> v = Vector<T>.CreateRandom(matrix.Rows);
        for (int i = 0; i < v.Length; i++)
        {
            v[i] = NumOps.FromDouble(Random.NextDouble() - 0.5);
        }
    
        // Normalize the initial vector
        T initialNorm = v.Norm();
        if (!NumOps.Equals(initialNorm, NumOps.Zero))
        {
            v = v.Divide(initialNorm);
        }
    
        T prevEigenvalue = NumOps.Zero;
        T currentEigenvalue;
    
        for (int i = 0; i < maxIterations; i++)
        {
            // Apply matrix to vector
            Vector<T> Av = matrix.Multiply(v);
        
            // Calculate Rayleigh quotient for better eigenvalue approximation
            T rayleighQuotient = v.DotProduct(Av);
        
            // Normalize the vector
            T norm = Av.Norm();
            if (NumOps.Equals(norm, NumOps.Zero))
            {
                // If we get a zero vector, the matrix might be nilpotent
                return NumOps.Zero;
            }
        
            v = Av.Divide(norm);
            currentEigenvalue = rayleighQuotient;
        
            // Check for convergence
            T diff = NumOps.Abs(NumOps.Subtract(currentEigenvalue, prevEigenvalue));
            if (Convert.ToDouble(diff) < tolerance && i > 5)
            {
                return NumOps.Abs(currentEigenvalue);
            }
        
            prevEigenvalue = currentEigenvalue;
        }
    
        // Return absolute value to ensure positive spectral radius
        return NumOps.Abs(prevEigenvalue);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Shape.Length != 2 || input.Shape[0] != 1)
            throw new ArgumentException("Input must be a 2D tensor with shape [1, inputSize]");

        Vector<T> inputVector = input.ToVector();
        Vector<T> scaledInput = inputVector.Multiply(NumOps.FromDouble(_inputScaling));

        Vector<T> reservoirInput = _reservoirWeights.Multiply(_reservoirState).Add(scaledInput);
        Vector<T> newState = ApplyActivation(reservoirInput);

        _reservoirState = _reservoirState.Multiply(NumOps.FromDouble(1 - _leakingRate))
            .Add(newState.Multiply(NumOps.FromDouble(_leakingRate)));

        return Tensor<T>.FromVector(_reservoirState);
    }

    private Vector<T> ApplyActivation(Vector<T> input)
    {
        return input.Transform(x => ScalarActivation!.Activate(x));
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // In ESN, we don't backpropagate through the reservoir
        throw new NotImplementedException("Backward pass is not implemented for ReservoirLayer in Echo State Networks.");
    }

    public override void UpdateParameters(T learningRate)
    {
        // In ESN, we don't update the reservoir weights
        throw new NotImplementedException("Parameter update is not implemented for ReservoirLayer in Echo State Networks.");
    }

    public Vector<T> GetState()
    {
        return _reservoirState;
    }

    public override void ResetState()
    {
        for (int i = 0; i < _reservoirState.Length; i++)
        {
            _reservoirState[i] = NumOps.Zero;
        }
    }

    public override Vector<T> GetParameters()
    {
        // In Echo State Networks, the reservoir weights are typically not trained
        // But we still provide access to them for inspection or manual modification
        int totalParams = _reservoirWeights.Rows * _reservoirWeights.Columns;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
        for (int i = 0; i < _reservoirWeights.Rows; i++)
        {
            for (int j = 0; j < _reservoirWeights.Columns; j++)
            {
                parameters[index++] = _reservoirWeights[i, j];
            }
        }
    
        return parameters;
    }
}