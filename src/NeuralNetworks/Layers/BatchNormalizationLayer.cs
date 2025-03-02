namespace AiDotNet.NeuralNetworks.Layers;

public class BatchNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private readonly T _momentum;
    private Vector<T> _gamma;
    private Vector<T> _beta;
    private Vector<T> _runningMean;
    private Vector<T> _runningVariance;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastNormalized;
    private Vector<T>? _lastMean;
    private Vector<T>? _lastVariance;
    private Vector<T>? _gammaGradient;
    private Vector<T>? _betaGradient;
    private bool _isTraining;

    public override bool SupportsTraining => true;

    public BatchNormalizationLayer(int featureSize, double epsilon = 1e-5, double momentum = 0.9)
        : base([featureSize], [featureSize])
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _momentum = NumOps.FromDouble(momentum);
        _gamma = Vector<T>.CreateDefault(featureSize, NumOps.One);
        _beta = new Vector<T>(featureSize);
        _runningMean = new Vector<T>(featureSize);
        _runningVariance = Vector<T>.CreateDefault(featureSize, NumOps.One);
        _isTraining = true;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];

        var output = new Tensor<T>(input.Shape);

        if (IsTrainingMode)
        {
            _lastMean = ComputeMean(input);
            _lastVariance = ComputeVariance(input, _lastMean);

            // Update running statistics
            _runningMean = UpdateRunningStatistic(_runningMean, _lastMean);
            _runningVariance = UpdateRunningStatistic(_runningVariance, _lastVariance);

            _lastNormalized = Normalize(input, _lastMean, _lastVariance);
        }
        else
        {
            _lastNormalized = Normalize(input, _runningMean, _runningVariance);
        }

        // Scale and shift
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                output[i, j] = NumOps.Add(NumOps.Multiply(_lastNormalized[i, j], _gamma[j]), _beta[j]);
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormalized == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int featureSize = _lastInput.Shape[1];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _gammaGradient = new Vector<T>(featureSize);
        _betaGradient = new Vector<T>(featureSize);

        var varianceEpsilon = _lastVariance.Add(_epsilon);
        var invStd = varianceEpsilon.Transform(NumOps.Sqrt).Transform(x => MathHelper.Reciprocal(x));

        for (int j = 0; j < featureSize; j++)
        {
            T sumDy = NumOps.Zero;
            T sumDyXmu = NumOps.Zero;

            for (int i = 0; i < batchSize; i++)
            {
                T dy = outputGradient[i, j];
                T xmu = _lastNormalized[i, j];

                sumDy = NumOps.Add(sumDy, dy);
                sumDyXmu = NumOps.Add(sumDyXmu, NumOps.Multiply(dy, xmu));

                _gammaGradient[j] = NumOps.Add(_gammaGradient[j], NumOps.Multiply(dy, xmu));
                _betaGradient[j] = NumOps.Add(_betaGradient[j], dy);
            }

            T invN = NumOps.FromDouble(1.0 / batchSize);
            T invVar = invStd[j];

            for (int i = 0; i < batchSize; i++)
            {
                T xmu = _lastNormalized[i, j];
                T dy = outputGradient[i, j];

                inputGradient[i, j] = NumOps.Multiply(
                    _gamma[j],
                    NumOps.Multiply(
                        invN,
                        NumOps.Multiply(
                            invVar,
                            NumOps.Subtract(
                                NumOps.Multiply(NumOps.FromDouble(batchSize), dy),
                                NumOps.Add(sumDy, NumOps.Multiply(xmu, sumDyXmu))
                            )
                        )
                    )
                );
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

    private Vector<T> ComputeMean(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];
        var mean = new Vector<T>(featureSize);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                mean[j] = NumOps.Add(mean[j], input[i, j]);
            }
        }

        return mean.Divide(NumOps.FromDouble(batchSize));
    }

    private Vector<T> ComputeVariance(Tensor<T> input, Vector<T> mean)
    {
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];
        var variance = new Vector<T>(featureSize);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                T diff = NumOps.Subtract(input[i, j], mean[j]);
                variance[j] = NumOps.Add(variance[j], NumOps.Multiply(diff, diff));
            }
        }

        return variance.Divide(NumOps.FromDouble(batchSize));
    }

    private Tensor<T> Normalize(Tensor<T> input, Vector<T> mean, Vector<T> variance)
    {
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];
        var normalized = new Tensor<T>(input.Shape);

        var varianceEpsilon = variance.Add(_epsilon);
        var invStd = varianceEpsilon.Transform(NumOps.Sqrt).Transform(x => MathHelper.Reciprocal(x));

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                normalized[i, j] = NumOps.Multiply(NumOps.Subtract(input[i, j], mean[j]), invStd[j]);
            }
        }

        return normalized;
    }

    private Vector<T> UpdateRunningStatistic(Vector<T> runningStatistic, Vector<T> batchStatistic)
    {
        return runningStatistic.Multiply(_momentum).Add(batchStatistic.Multiply(NumOps.Subtract(NumOps.One, _momentum)));
    }

    public override Vector<T> GetParameters()
    {
        // Concatenate gamma and beta parameters
        int featureSize = InputShape[0];
        var parameters = new Vector<T>(featureSize * 2);
        
        for (int i = 0; i < featureSize; i++)
        {
            parameters[i] = _gamma[i];
            parameters[i + featureSize] = _beta[i];
        }
        
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int featureSize = InputShape[0];
        if (parameters.Length != featureSize * 2)
            throw new ArgumentException($"Expected {featureSize * 2} parameters, but got {parameters.Length}");
        
        for (int i = 0; i < featureSize; i++)
        {
            _gamma[i] = parameters[i];
            _beta[i] = parameters[i + featureSize];
        }
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastNormalized = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;
    }
}