using AiDotNet.UncertaintyQuantification.Interfaces;

namespace AiDotNet.UncertaintyQuantification.Layers;

/// <summary>
/// Implements a Bayesian dense (fully-connected) layer using variational inference.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Bayesian Dense Layer is similar to a regular dense layer, but instead
/// of having fixed weights, it learns probability distributions over weights.
///
/// This is based on the "Bayes by Backprop" algorithm which uses variational inference to
/// approximate the true posterior distribution of weights.
///
/// The layer maintains two sets of parameters for each weight:
/// - Mean (μ): The average value of the weight
/// - Standard deviation (σ): How much the weight varies
///
/// During forward passes, weights are sampled from these distributions, allowing the network
/// to express uncertainty in its predictions.
/// </para>
/// </remarks>
public class BayesianDenseLayer<T> : LayerBase<T>, IBayesianLayer<T>
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly T _priorSigma;

    // Weight parameters (mean and log variance)
    private Matrix<T> _weightMean;
    private Matrix<T> _weightLogVar;
    private Vector<T> _biasMean;
    private Vector<T> _biasLogVar;

    // Sampled weights for current forward pass
    private Matrix<T>? _sampledWeights;
    private Vector<T>? _sampledBias;

    // Gradients
    private Matrix<T> _weightMeanGradient;
    private Matrix<T> _weightLogVarGradient;
    private Vector<T> _biasMeanGradient;
    private Vector<T> _biasLogVarGradient;

    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training mode.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the BayesianDenseLayer class.
    /// </summary>
    /// <param name="inputSize">The number of input features.</param>
    /// <param name="outputSize">The number of output features.</param>
    /// <param name="priorSigma">The standard deviation of the prior distribution (default: 1.0).</param>
    /// <remarks>
    /// <b>For Beginners:</b> The prior sigma controls how spread out the initial weight distributions are.
    /// A larger value means more initial uncertainty, a smaller value means the model starts more confident.
    /// </remarks>
    public BayesianDenseLayer(int inputSize, int outputSize, double priorSigma = 1.0)
        : base([inputSize], [outputSize])
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _priorSigma = NumOps.FromDouble(priorSigma);

        // Initialize weight and bias parameters
        InitializeParameters();
    }

    private void InitializeParameters()
    {
        var random = new Random();

        // Initialize weight means with Xavier initialization
        _weightMean = new Matrix<T>(_outputSize, _inputSize);
        var scale = Math.Sqrt(2.0 / (_inputSize + _outputSize));
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _weightMean[i, j] = NumOps.FromDouble(random.NextGaussian(0, scale));
            }
        }

        // Initialize weight log variances to small values (start relatively confident)
        _weightLogVar = new Matrix<T>(_outputSize, _inputSize);
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _weightLogVar[i, j] = NumOps.FromDouble(-5.0); // exp(-5) ≈ 0.0067
            }
        }

        // Initialize bias parameters
        _biasMean = new Vector<T>(_outputSize);
        _biasLogVar = new Vector<T>(_outputSize);
        for (int i = 0; i < _outputSize; i++)
        {
            _biasMean[i] = NumOps.Zero;
            _biasLogVar[i] = NumOps.FromDouble(-5.0);
        }

        // Initialize gradients
        _weightMeanGradient = new Matrix<T>(_outputSize, _inputSize);
        _weightLogVarGradient = new Matrix<T>(_outputSize, _inputSize);
        _biasMeanGradient = new Vector<T>(_outputSize);
        _biasLogVarGradient = new Vector<T>(_outputSize);
    }

    /// <summary>
    /// Samples weights from the learned distributions.
    /// </summary>
    public void SampleWeights()
    {
        var random = new Random();

        _sampledWeights = new Matrix<T>(_outputSize, _inputSize);
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                var mean = _weightMean[i, j];
                var std = NumOps.Sqrt(NumOps.Exp(_weightLogVar[i, j]));
                var epsilon = NumOps.FromDouble(random.NextGaussian());
                _sampledWeights[i, j] = NumOps.Add(mean, NumOps.Multiply(std, epsilon));
            }
        }

        _sampledBias = new Vector<T>(_outputSize);
        for (int i = 0; i < _outputSize; i++)
        {
            var mean = _biasMean[i];
            var std = NumOps.Sqrt(NumOps.Exp(_biasLogVar[i]));
            var epsilon = NumOps.FromDouble(random.NextGaussian());
            _sampledBias[i] = NumOps.Add(mean, NumOps.Multiply(std, epsilon));
        }
    }

    /// <summary>
    /// Computes the KL divergence between the weight distribution and the prior.
    /// </summary>
    /// <returns>The KL divergence value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This measures how different the learned weight distributions are
    /// from a simple Gaussian prior. This is added to the loss during training to regularize
    /// the network and prevent overfitting.
    /// </remarks>
    public T GetKLDivergence()
    {
        var kl = NumOps.Zero;
        var priorVar = NumOps.Multiply(_priorSigma, _priorSigma);

        // KL divergence for weights
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                var variance = NumOps.Exp(_weightLogVar[i, j]);
                var meanSquared = NumOps.Multiply(_weightMean[i, j], _weightMean[i, j]);

                // KL(q||p) = 0.5 * (variance/prior_var + mean²/prior_var - 1 - log(variance/prior_var))
                var term1 = NumOps.Divide(variance, priorVar);
                var term2 = NumOps.Divide(meanSquared, priorVar);
                var term3 = _weightLogVar[i, j];
                var term4 = NumOps.Log(priorVar);

                var klTerm = NumOps.Multiply(
                    NumOps.FromDouble(0.5),
                    NumOps.Subtract(
                        NumOps.Add(NumOps.Add(term1, term2), NumOps.Negate(NumOps.One)),
                        NumOps.Subtract(term3, term4)
                    )
                );
                kl = NumOps.Add(kl, klTerm);
            }
        }

        // KL divergence for biases
        for (int i = 0; i < _outputSize; i++)
        {
            var variance = NumOps.Exp(_biasLogVar[i]);
            var meanSquared = NumOps.Multiply(_biasMean[i], _biasMean[i]);

            var term1 = NumOps.Divide(variance, priorVar);
            var term2 = NumOps.Divide(meanSquared, priorVar);
            var term3 = _biasLogVar[i];
            var term4 = NumOps.Log(priorVar);

            var klTerm = NumOps.Multiply(
                NumOps.FromDouble(0.5),
                NumOps.Subtract(
                    NumOps.Add(NumOps.Add(term1, term2), NumOps.Negate(NumOps.One)),
                    NumOps.Subtract(term3, term4)
                )
            );
            kl = NumOps.Add(kl, klTerm);
        }

        return kl;
    }

    /// <summary>
    /// Performs the forward pass using sampled weights.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Sample weights if not already sampled
        if (_sampledWeights == null || _sampledBias == null)
        {
            SampleWeights();
        }

        // Flatten input if needed
        var flatInput = input.Reshape([input.Length]);

        // Compute output = weights * input + bias
        var output = new Vector<T>(_outputSize);
        for (int i = 0; i < _outputSize; i++)
        {
            var sum = _sampledBias![i];
            for (int j = 0; j < _inputSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_sampledWeights![i, j], flatInput[j]));
            }
            output[i] = sum;
        }

        return new Tensor<T>([_outputSize], output.ToArray());
    }

    /// <summary>
    /// Performs the backward pass and accumulates gradients.
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _sampledWeights == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var flatInput = _lastInput.Reshape([_lastInput.Length]);
        var flatGradient = outputGradient.Reshape([outputGradient.Length]);

        // Accumulate gradients for weight means and log variances
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                // Gradient w.r.t. weight mean
                var gradMean = NumOps.Multiply(flatGradient[i], flatInput[j]);
                _weightMeanGradient[i, j] = NumOps.Add(_weightMeanGradient[i, j], gradMean);

                // Gradient w.r.t. weight log variance (from reparameterization trick)
                var epsilon = NumOps.Divide(
                    NumOps.Subtract(_sampledWeights[i, j], _weightMean[i, j]),
                    NumOps.Add(NumOps.Sqrt(NumOps.Exp(_weightLogVar[i, j])), NumOps.FromDouble(1e-8))
                );
                var gradLogVar = NumOps.Multiply(
                    NumOps.Multiply(gradMean, epsilon),
                    NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Exp(_weightLogVar[i, j]))
                );
                _weightLogVarGradient[i, j] = NumOps.Add(_weightLogVarGradient[i, j], gradLogVar);
            }

            // Gradient w.r.t. bias
            _biasMeanGradient[i] = NumOps.Add(_biasMeanGradient[i], flatGradient[i]);

            var biasEpsilon = NumOps.Divide(
                NumOps.Subtract(_sampledBias![i], _biasMean[i]),
                NumOps.Add(NumOps.Sqrt(NumOps.Exp(_biasLogVar[i])), NumOps.FromDouble(1e-8))
            );
            var biasGradLogVar = NumOps.Multiply(
                NumOps.Multiply(flatGradient[i], biasEpsilon),
                NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Exp(_biasLogVar[i]))
            );
            _biasLogVarGradient[i] = NumOps.Add(_biasLogVarGradient[i], biasGradLogVar);
        }

        // Compute input gradient
        var inputGradient = new Vector<T>(_inputSize);
        for (int j = 0; j < _inputSize; j++)
        {
            var sum = NumOps.Zero;
            for (int i = 0; i < _outputSize; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_sampledWeights[i, j], flatGradient[i]));
            }
            inputGradient[j] = sum;
        }

        return new Tensor<T>(_lastInput.Shape, inputGradient.ToArray());
    }

    /// <summary>
    /// Updates parameters using the accumulated gradients.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        // Update weight means
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _weightMean[i, j] = NumOps.Subtract(
                    _weightMean[i, j],
                    NumOps.Multiply(learningRate, _weightMeanGradient[i, j])
                );
                _weightLogVar[i, j] = NumOps.Subtract(
                    _weightLogVar[i, j],
                    NumOps.Multiply(learningRate, _weightLogVarGradient[i, j])
                );
            }
        }

        // Update biases
        for (int i = 0; i < _outputSize; i++)
        {
            _biasMean[i] = NumOps.Subtract(
                _biasMean[i],
                NumOps.Multiply(learningRate, _biasMeanGradient[i])
            );
            _biasLogVar[i] = NumOps.Subtract(
                _biasLogVar[i],
                NumOps.Multiply(learningRate, _biasLogVarGradient[i])
            );
        }

        // Clear gradients after update
        ClearGradients();
    }

    /// <summary>
    /// Gets all trainable parameters.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var paramCount = _outputSize * _inputSize * 2 + _outputSize * 2;
        var parameters = new Vector<T>(paramCount);
        int idx = 0;

        // Pack weight means
        for (int i = 0; i < _outputSize; i++)
            for (int j = 0; j < _inputSize; j++)
                parameters[idx++] = _weightMean[i, j];

        // Pack weight log variances
        for (int i = 0; i < _outputSize; i++)
            for (int j = 0; j < _inputSize; j++)
                parameters[idx++] = _weightLogVar[i, j];

        // Pack bias means
        for (int i = 0; i < _outputSize; i++)
            parameters[idx++] = _biasMean[i];

        // Pack bias log variances
        for (int i = 0; i < _outputSize; i++)
            parameters[idx++] = _biasLogVar[i];

        return parameters;
    }

    /// <summary>
    /// Sets all trainable parameters.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        var expectedCount = _outputSize * _inputSize * 2 + _outputSize * 2;
        if (parameters.Length != expectedCount)
            throw new ArgumentException($"Expected {expectedCount} parameters, got {parameters.Length}");

        int idx = 0;

        // Unpack weight means
        for (int i = 0; i < _outputSize; i++)
            for (int j = 0; j < _inputSize; j++)
                _weightMean[i, j] = parameters[idx++];

        // Unpack weight log variances
        for (int i = 0; i < _outputSize; i++)
            for (int j = 0; j < _inputSize; j++)
                _weightLogVar[i, j] = parameters[idx++];

        // Unpack bias means
        for (int i = 0; i < _outputSize; i++)
            _biasMean[i] = parameters[idx++];

        // Unpack bias log variances
        for (int i = 0; i < _outputSize; i++)
            _biasLogVar[i] = parameters[idx++];
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _sampledWeights = null;
        _sampledBias = null;
        ClearGradients();
    }
}

/// <summary>
/// Extension methods for generating Gaussian random numbers.
/// </summary>
internal static class RandomExtensions
{
    /// <summary>
    /// Generates a random number from a Gaussian distribution using Box-Muller transform.
    /// </summary>
    public static double NextGaussian(this Random random, double mean = 0.0, double stdDev = 1.0)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}
