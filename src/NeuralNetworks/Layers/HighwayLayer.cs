namespace AiDotNet.NeuralNetworks.Layers;

public class HighwayLayer<T> : LayerBase<T>
{
    private Matrix<T> _transformWeights;
    private Vector<T> _transformBias;
    private Matrix<T> _gateWeights;
    private Vector<T> _gateBias;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastTransformOutput;
    private Tensor<T>? _lastGateOutput;

    private Matrix<T>? _transformWeightsGradient;
    private Vector<T>? _transformBiasGradient;
    private Matrix<T>? _gateWeightsGradient;
    private Vector<T>? _gateBiasGradient;

    private readonly IActivationFunction<T>? _transformActivation;
    private readonly IActivationFunction<T>? _gateActivation;
    private readonly IVectorActivationFunction<T>? _vectorTransformActivation;
    private readonly IVectorActivationFunction<T>? _vectorGateActivation;

    public HighwayLayer(int inputDimension, IActivationFunction<T>? transformActivation = null, IActivationFunction<T>? gateActivation = null)
        : base([inputDimension], [inputDimension], transformActivation ?? new TanhActivation<T>())
    {
        _transformWeights = new Matrix<T>(inputDimension, inputDimension);
        _transformBias = new Vector<T>(inputDimension);
        _gateWeights = new Matrix<T>(inputDimension, inputDimension);
        _gateBias = new Vector<T>(inputDimension);

        _transformActivation = transformActivation ?? new TanhActivation<T>();
        _gateActivation = gateActivation ?? new SigmoidActivation<T>();

        InitializeParameters();
    }

    public HighwayLayer(int inputDimension, IVectorActivationFunction<T>? transformActivation = null, IVectorActivationFunction<T>? gateActivation = null)
        : base([inputDimension], [inputDimension], transformActivation ?? new TanhActivation<T>())
    {
        _transformWeights = new Matrix<T>(inputDimension, inputDimension);
        _transformBias = new Vector<T>(inputDimension);
        _gateWeights = new Matrix<T>(inputDimension, inputDimension);
        _gateBias = new Vector<T>(inputDimension);

        _vectorTransformActivation = transformActivation ?? new TanhActivation<T>();
        _vectorGateActivation = gateActivation ?? new SigmoidActivation<T>();

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_transformWeights.Rows + _transformWeights.Columns)));
        InitializeMatrix(_transformWeights, scale);
        InitializeMatrix(_gateWeights, scale);

        for (int i = 0; i < _transformBias.Length; i++)
        {
            _transformBias[i] = NumOps.Zero;
            _gateBias[i] = NumOps.FromDouble(-1.0); // Initialize gate bias to negative values to allow more information flow initially
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

        var transformOutput = input.Multiply(_transformWeights).Add(_transformBias);
        transformOutput = ApplyActivation(transformOutput, _transformActivation, _vectorTransformActivation);
        _lastTransformOutput = transformOutput;

        var gateOutput = input.Multiply(_gateWeights).Add(_gateBias);
        gateOutput = ApplyActivation(gateOutput, _gateActivation, _vectorGateActivation);
        _lastGateOutput = gateOutput;

        var output = gateOutput.ElementwiseMultiply(transformOutput)
            .Add(input.ElementwiseMultiply(gateOutput.ElementwiseSubtract(Tensor<T>.CreateDefault(gateOutput.Shape, NumOps.One))));

        _lastOutput = output;
        return output;
    }

    private Tensor<T> ApplyActivation(Tensor<T> input, IActivationFunction<T>? scalarActivation, IVectorActivationFunction<T>? vectorActivation)
    {
        if (vectorActivation != null)
        {
            return vectorActivation.Activate(input);
        }
        else if (scalarActivation != null)
        {
            return input.Transform((x, _) => scalarActivation.Activate(x));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastTransformOutput == null || _lastGateOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputDimension = _lastInput.Shape[1];

        var gateGradient = outputGradient.ElementwiseMultiply(_lastTransformOutput.ElementwiseSubtract(_lastInput));
        gateGradient = ApplyActivationDerivative(gateGradient, _lastGateOutput, _gateActivation, _vectorGateActivation);

        var transformGradient = outputGradient.ElementwiseMultiply(_lastGateOutput);
        transformGradient = ApplyActivationDerivative(transformGradient, _lastTransformOutput, _transformActivation, _vectorTransformActivation);

        _gateWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(gateGradient).ToMatrix();
        _gateBiasGradient = gateGradient.Sum([0]).ToVector();

        _transformWeightsGradient = _lastInput.Transpose([1, 0]).Multiply(transformGradient).ToMatrix();
        _transformBiasGradient = transformGradient.Sum([0]).ToVector();

        var inputGradient = gateGradient.Multiply(_gateWeights.Transpose())
            .Add(transformGradient.Multiply(_transformWeights.Transpose()))
            .Add(outputGradient.ElementwiseMultiply(_lastGateOutput.ElementwiseSubtract(Tensor<T>.CreateDefault(_lastGateOutput.Shape, NumOps.One))));

        return inputGradient;
    }

    private Tensor<T> ApplyActivationDerivative(Tensor<T> gradient, Tensor<T> lastOutput, IActivationFunction<T>? scalarActivation, IVectorActivationFunction<T>? vectorActivation)
    {
        if (vectorActivation != null)
        {
            return gradient.ElementwiseMultiply(vectorActivation.Derivative(lastOutput));
        }
        else if (scalarActivation != null)
        {
            return gradient.ElementwiseMultiply(lastOutput.Transform((x, _) => scalarActivation.Derivative(x)));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_transformWeightsGradient == null || _transformBiasGradient == null || 
            _gateWeightsGradient == null || _gateBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _transformWeights = _transformWeights.Subtract(_transformWeightsGradient.Multiply(learningRate));
        _transformBias = _transformBias.Subtract(_transformBiasGradient.Multiply(learningRate));
        _gateWeights = _gateWeights.Subtract(_gateWeightsGradient.Multiply(learningRate));
        _gateBias = _gateBias.Subtract(_gateBiasGradient.Multiply(learningRate));
    }
}