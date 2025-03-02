namespace AiDotNet.NeuralNetworks.Layers;

public class LayerNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private Vector<T> _gamma;
    private Vector<T> _beta;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastNormalized;
    private Vector<T>? _lastMean;
    private Vector<T>? _lastStd;
    private Vector<T>? _gammaGradient;
    private Vector<T>? _betaGradient;

    public override bool SupportsTraining => true;

    public LayerNormalizationLayer(int featureSize, double epsilon = 1e-5)
        : base([featureSize], [featureSize])
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _gamma = Vector<T>.CreateDefault(featureSize, NumOps.One);
        _beta = new Vector<T>(featureSize);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];

        var output = new Tensor<T>(input.Shape);
        _lastMean = new Vector<T>(batchSize);
        _lastStd = new Vector<T>(batchSize);
        _lastNormalized = new Tensor<T>(input.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            var sample = new Vector<T>(featureSize);
            for (int j = 0; j < featureSize; j++)
            {
                sample[j] = input[i, j];
            }

            _lastMean[i] = sample.Mean();
            _lastStd[i] = NumOps.Sqrt(NumOps.Add(sample.Variance(), _epsilon));

            for (int j = 0; j < featureSize; j++)
            {
                _lastNormalized[i, j] = NumOps.Divide(NumOps.Subtract(input[i, j], _lastMean[i]), _lastStd[i]);
                output[i, j] = NumOps.Add(NumOps.Multiply(_lastNormalized[i, j], _gamma[j]), _beta[j]);
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormalized == null || _lastMean == null || _lastStd == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int featureSize = _lastInput.Shape[1];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _gammaGradient = new Vector<T>(featureSize);
        _betaGradient = new Vector<T>(featureSize);

        for (int i = 0; i < batchSize; i++)
        {
            var dxhat = new Vector<T>(featureSize);
            T dvariance = NumOps.Zero;
            T dmean = NumOps.Zero;

            for (int j = 0; j < featureSize; j++)
            {
                T dy = outputGradient[i, j];
                dxhat[j] = NumOps.Multiply(dy, _gamma[j]);
                _gammaGradient[j] = NumOps.Add(_gammaGradient[j], NumOps.Multiply(dy, _lastNormalized[i, j]));
                _betaGradient[j] = NumOps.Add(_betaGradient[j], dy);
            }

            for (int j = 0; j < featureSize; j++)
            {
                T xhat = _lastNormalized[i, j];
                dvariance = NumOps.Add(dvariance, NumOps.Multiply(dxhat[j], NumOps.Multiply(NumOps.Subtract(_lastInput[i, j], _lastMean[i]), NumOps.FromDouble(-0.5 / Math.Pow(Convert.ToDouble(_lastStd[i]), 3)))));
                dmean = NumOps.Add(dmean, NumOps.Multiply(dxhat[j], NumOps.FromDouble(-1.0 / Convert.ToDouble(_lastStd[i]))));
            }

            T sumDiff = NumOps.Zero;
            for (int j = 0; j < featureSize; j++)
            {
                sumDiff = NumOps.Add(sumDiff, NumOps.Subtract(_lastInput[i, j], _lastMean[i]));
            }

            dmean = NumOps.Add(dmean, NumOps.Multiply(NumOps.Multiply(dvariance, NumOps.FromDouble(-2.0 / featureSize)), sumDiff));

            for (int j = 0; j < featureSize; j++)
            {
                T dx = NumOps.Add(
                    NumOps.Divide(dxhat[j], _lastStd[i]),
                    NumOps.Add(
                        NumOps.Multiply(dvariance, NumOps.Divide(NumOps.FromDouble(2), NumOps.Multiply(NumOps.FromDouble(featureSize), NumOps.Subtract(_lastInput[i, j], _lastMean[i])))),
                        NumOps.Divide(NumOps.FromDouble(1.0), NumOps.Multiply(NumOps.FromDouble(featureSize), dmean))
                    )
                );
                inputGradient[i, j] = dx;
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = _gamma.Subtract(_gammaGradient.Multiply(learningRate));
        _beta = _beta.Subtract(_betaGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _gamma.Length + _beta.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy gamma parameters
        for (int i = 0; i < _gamma.Length; i++)
        {
            parameters[index++] = _gamma[i];
        }
    
        // Copy beta parameters
        for (int i = 0; i < _beta.Length; i++)
        {
            parameters[index++] = _beta[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _gamma.Length + _beta.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set gamma parameters
        for (int i = 0; i < _gamma.Length; i++)
        {
            _gamma[i] = parameters[index++];
        }
    
        // Set beta parameters
        for (int i = 0; i < _beta.Length; i++)
        {
            _beta[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastNormalized = null;
        _lastMean = null;
        _lastStd = null;
        _gammaGradient = null;
        _betaGradient = null;
    }
}