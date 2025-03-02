namespace AiDotNet.NeuralNetworks.Layers;

public class GatedLinearUnitLayer<T> : LayerBase<T>
{
    private Matrix<T> _linearWeights;
    private Matrix<T> _gateWeights;
    private Vector<T> _linearBias;
    private Vector<T> _gateBias;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastLinearOutput;
    private Tensor<T>? _lastGateOutput;

    private Matrix<T>? _linearWeightsGradient;
    private Matrix<T>? _gateWeightsGradient;
    private Vector<T>? _linearBiasGradient;
    private Vector<T>? _gateBiasGradient;

    public override bool SupportsTraining => true;

    public GatedLinearUnitLayer(int inputDimension, int outputDimension, IActivationFunction<T>? gateActivation = null)
        : base([inputDimension], [outputDimension], gateActivation ?? new SigmoidActivation<T>())
    {
        _linearWeights = new Matrix<T>(outputDimension, inputDimension);
        _gateWeights = new Matrix<T>(outputDimension, inputDimension);
        _linearBias = new Vector<T>(outputDimension);
        _gateBias = new Vector<T>(outputDimension);

        InitializeParameters();
    }

    public GatedLinearUnitLayer(int inputDimension, int outputDimension, IVectorActivationFunction<T>? gateActivation = null)
        : base([inputDimension], [outputDimension], gateActivation ?? new SigmoidActivation<T>())
    {
        _linearWeights = new Matrix<T>(outputDimension, inputDimension);
        _gateWeights = new Matrix<T>(outputDimension, inputDimension);
        _linearBias = new Vector<T>(outputDimension);
        _gateBias = new Vector<T>(outputDimension);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_linearWeights.Rows + _linearWeights.Columns)));
        InitializeMatrix(_linearWeights, scale);
        InitializeMatrix(_gateWeights, scale);

        for (int i = 0; i < _linearBias.Length; i++)
        {
            _linearBias[i] = NumOps.Zero;
            _gateBias[i] = NumOps.Zero;
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
        int inputDimension = input.Shape[1];

        var linearOutput = input.Multiply(_linearWeights).Add(_linearBias);
        var gateOutput = input.Multiply(_gateWeights).Add(_gateBias);

        _lastLinearOutput = linearOutput;
        _lastGateOutput = ApplyActivation(gateOutput);

        var output = _lastLinearOutput.ElementwiseMultiply(_lastGateOutput);

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastLinearOutput == null || _lastGateOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var linearGradient = outputGradient.ElementwiseMultiply(_lastGateOutput);
        var gateGradient = outputGradient.ElementwiseMultiply(_lastLinearOutput);

        gateGradient = ApplyActivationDerivative(_lastGateOutput, gateGradient);

        _linearWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(linearGradient).ToMatrix();
        _gateWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(gateGradient).ToMatrix();

        _linearBiasGradient = linearGradient.Sum([0]).ToVector();
        _gateBiasGradient = gateGradient.Sum([0]).ToVector();

        var inputGradient = linearGradient.Multiply(_linearWeights.Transpose())
                            .Add(gateGradient.Multiply(_gateWeights.Transpose()));

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_linearWeightsGradient == null || _gateWeightsGradient == null || 
            _linearBiasGradient == null || _gateBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _linearWeights = _linearWeights.Subtract(_linearWeightsGradient.Multiply(learningRate));
        _gateWeights = _gateWeights.Subtract(_gateWeightsGradient.Multiply(learningRate));
        _linearBias = _linearBias.Subtract(_linearBiasGradient.Multiply(learningRate));
        _gateBias = _gateBias.Subtract(_gateBiasGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _linearWeights.Rows * _linearWeights.Columns + 
                          _gateWeights.Rows * _gateWeights.Columns + 
                          _linearBias.Length + _gateBias.Length;
    
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy linear weights parameters
        for (int i = 0; i < _linearWeights.Rows; i++)
        {
            for (int j = 0; j < _linearWeights.Columns; j++)
            {
                parameters[index++] = _linearWeights[i, j];
            }
        }
    
        // Copy gate weights parameters
        for (int i = 0; i < _gateWeights.Rows; i++)
        {
            for (int j = 0; j < _gateWeights.Columns; j++)
            {
                parameters[index++] = _gateWeights[i, j];
            }
        }
    
        // Copy linear bias parameters
        for (int i = 0; i < _linearBias.Length; i++)
        {
            parameters[index++] = _linearBias[i];
        }
    
        // Copy gate bias parameters
        for (int i = 0; i < _gateBias.Length; i++)
        {
            parameters[index++] = _gateBias[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int expectedLength = _linearWeights.Rows * _linearWeights.Columns + 
                             _gateWeights.Rows * _gateWeights.Columns + 
                             _linearBias.Length + _gateBias.Length;
    
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set linear weights parameters
        for (int i = 0; i < _linearWeights.Rows; i++)
        {
            for (int j = 0; j < _linearWeights.Columns; j++)
            {
                _linearWeights[i, j] = parameters[index++];
            }
        }
    
        // Set gate weights parameters
        for (int i = 0; i < _gateWeights.Rows; i++)
        {
            for (int j = 0; j < _gateWeights.Columns; j++)
            {
                _gateWeights[i, j] = parameters[index++];
            }
        }
    
        // Set linear bias parameters
        for (int i = 0; i < _linearBias.Length; i++)
        {
            _linearBias[i] = parameters[index++];
        }
    
        // Set gate bias parameters
        for (int i = 0; i < _gateBias.Length; i++)
        {
            _gateBias[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastLinearOutput = null;
        _lastGateOutput = null;
        _linearWeightsGradient = null;
        _gateWeightsGradient = null;
        _linearBiasGradient = null;
        _gateBiasGradient = null;
    }
}