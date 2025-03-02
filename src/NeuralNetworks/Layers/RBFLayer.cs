namespace AiDotNet.NeuralNetworks.Layers;

public class RBFLayer<T> : LayerBase<T>
{
    private Matrix<T> _centers;
    private Vector<T> _widths;
    private IRadialBasisFunction<T> _rbf;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    private Matrix<T>? _centersGradient;
    private Vector<T>? _widthsGradient;

    public override bool SupportsTraining => true;

    public RBFLayer(int inputSize, int outputSize, IRadialBasisFunction<T> rbf)
        : base([inputSize], [outputSize])
    {
        _centers = new Matrix<T>(outputSize, inputSize);
        _widths = new Vector<T>(outputSize);
        _rbf = rbf;

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_centers.Rows + _centers.Columns)));
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                _centers[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }

            _widths[i] = NumOps.FromDouble(Random.NextDouble());
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        var output = new Tensor<T>([batchSize, _centers.Rows]);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < _centers.Rows; j++)
            {
                T distance = CalculateDistance(input.GetVector(i), _centers.GetRow(j));
                output[i, j] = _rbf.Compute(distance);
            }
        }

        _lastOutput = output;
        return output;
    }

    private T CalculateDistance(Vector<T> x, Vector<T> center)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T diff = NumOps.Subtract(x[i], center[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape[1];
        int outputSize = _centers.Rows;

        _centersGradient = new Matrix<T>(outputSize, inputSize);
        _widthsGradient = new Vector<T>(outputSize);

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                T distance = CalculateDistance(_lastInput.GetVector(i), _centers.GetRow(j));
                T rbfDerivative = _rbf.ComputeDerivative(distance);

                for (int k = 0; k < inputSize; k++)
                {
                    T inputDiff = NumOps.Subtract(_lastInput[i, k], _centers[j, k]);
                    T gradient = NumOps.Multiply(outputGradient[i, j], rbfDerivative);
                    T centerGradient = NumOps.Multiply(gradient, inputDiff);

                    _centersGradient[j, k] = NumOps.Add(_centersGradient[j, k], centerGradient);
                    inputGradient[i, k] = NumOps.Add(inputGradient[i, k], NumOps.Negate(centerGradient));
                }

                _widthsGradient[j] = NumOps.Add(_widthsGradient[j], 
                    NumOps.Multiply(outputGradient[i, j], _rbf.ComputeWidthDerivative(distance)));
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_centersGradient == null || _widthsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                _centers[i, j] = NumOps.Subtract(_centers[i, j], 
                    NumOps.Multiply(learningRate, _centersGradient[i, j]));
            }
            _widths[i] = NumOps.Subtract(_widths[i], 
                NumOps.Multiply(learningRate, _widthsGradient[i]));
        }
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _centers.Rows * _centers.Columns + _widths.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy centers
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                parameters[index++] = _centers[i, j];
            }
        }
    
        // Copy widths
        for (int i = 0; i < _widths.Length; i++)
        {
            parameters[index++] = _widths[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _centers.Rows * _centers.Columns + _widths.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set centers
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                _centers[i, j] = parameters[index++];
            }
        }
    
        // Set widths
        for (int i = 0; i < _widths.Length; i++)
        {
            _widths[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _centersGradient = null;
        _widthsGradient = null;
    }
}