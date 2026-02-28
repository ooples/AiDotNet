using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
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
    private readonly Random _rng;
    private readonly object _rngLock = new();
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly T _priorSigma;

    // Weight parameters (mean and log variance)
    private Matrix<T> _weightMean = null!;
    private Matrix<T> _weightLogVar = null!;
    private Vector<T> _biasMean = null!;
    private Vector<T> _biasLogVar = null!;

    // Sampled weights for current forward pass
    private Matrix<T>? _sampledWeights;
    private Vector<T>? _sampledBias;

    // Gradients
    private Matrix<T> _weightMeanGradient = null!;
    private Matrix<T> _weightLogVarGradient = null!;
    private Vector<T> _biasMeanGradient = null!;
    private Vector<T> _biasLogVarGradient = null!;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastPreActivation;

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
    /// <param name="randomSeed">Optional random seed for reproducible sampling.</param>
    /// <remarks>
    /// <b>For Beginners:</b> The prior sigma controls how spread out the initial weight distributions are.
    /// A larger value means more initial uncertainty, a smaller value means the model starts more confident.
    /// </remarks>
    public BayesianDenseLayer(int inputSize, int outputSize, double priorSigma = 1.0, int? randomSeed = null)
        : this(inputSize, outputSize, scalarActivation: new ReLUActivation<T>(), priorSigma: priorSigma, randomSeed: randomSeed)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="BayesianDenseLayer{T}"/> class with a custom activation.
    /// </summary>
    /// <param name="inputSize">The number of input features.</param>
    /// <param name="outputSize">The number of output features.</param>
    /// <param name="scalarActivation">The activation function to apply.</param>
    /// <param name="priorSigma">The standard deviation of the prior distribution (default: 1.0).</param>
    /// <param name="randomSeed">Optional random seed for reproducible sampling.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This overload lets you choose what activation the layer uses, while still keeping
    /// Bayesian uncertainty for the weights.
    /// </para>
    /// </remarks>
    public BayesianDenseLayer(int inputSize, int outputSize, IActivationFunction<T>? scalarActivation, double priorSigma = 1.0, int? randomSeed = null)
        : base([inputSize], [outputSize], scalarActivation ?? new ReLUActivation<T>())
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _priorSigma = NumOps.FromDouble(priorSigma);
        _rng = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize weight and bias parameters
        InitializeParameters();
    }

    /// <inheritdoc/>
    public override int ParameterCount => _outputSize * _inputSize * 2 + _outputSize * 2;

    private void InitializeParameters()
    {
        // Initialize weight means with Xavier initialization
        _weightMean = new Matrix<T>(_outputSize, _inputSize);
        var scale = Math.Sqrt(2.0 / (_inputSize + _outputSize));
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _weightMean[i, j] = NumOps.FromDouble(NextGaussian(0, scale));
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
        _sampledWeights = new Matrix<T>(_outputSize, _inputSize);
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                var mean = _weightMean[i, j];
                var std = NumOps.Sqrt(NumOps.Exp(_weightLogVar[i, j]));
                var epsilon = NumOps.FromDouble(NextGaussian());
                _sampledWeights[i, j] = NumOps.Add(mean, NumOps.Multiply(std, epsilon));
            }
        }

        _sampledBias = new Vector<T>(_outputSize);
        for (int i = 0; i < _outputSize; i++)
        {
            var mean = _biasMean[i];
            var std = NumOps.Sqrt(NumOps.Exp(_biasLogVar[i]));
            var epsilon = NumOps.FromDouble(NextGaussian());
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
        _lastPreActivation = null;

        // Always resample weights during training for proper Bayesian inference.
        // During inference, reuse previously sampled weights for consistency.
        if (_sampledWeights == null || _sampledBias == null || IsTrainingMode)
        {
            SampleWeights();
        }

        int batch;
        if (input.Rank == 1)
        {
            batch = 1;
        }
        else
        {
            batch = input.Shape[0];
            if (batch <= 0)
            {
                throw new ArgumentException("Expected input tensor to have a positive batch dimension (Shape[0]).", nameof(input));
            }
        }

        var expectedLength = batch * _inputSize;
        if (input.Length != expectedLength)
        {
            throw new ArgumentException($"Expected input tensor length {expectedLength} for batch {batch} and inputSize {_inputSize}, but got {input.Length}.", nameof(input));
        }

        var inputFlat = input.ToVector();
        var outputVector = new Vector<T>(batch * _outputSize);

        for (int b = 0; b < batch; b++)
        {
            var inputOffset = b * _inputSize;
            var outputOffset = b * _outputSize;

            for (int i = 0; i < _outputSize; i++)
            {
                var sum = _sampledBias![i];
                for (int j = 0; j < _inputSize; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_sampledWeights![i, j], inputFlat[inputOffset + j]));
                }
                outputVector[outputOffset + i] = sum;
            }
        }

        var outputShape = input.Rank == 1 ? new[] { _outputSize } : new[] { batch, _outputSize };
        var preActivation = new Tensor<T>(outputShape, outputVector);
        _lastPreActivation = preActivation;

        return ApplyActivation(preActivation);
    }

    /// <summary>
    /// Performs the backward pass and accumulates gradients.
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _sampledWeights == null || _lastPreActivation == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batch;
        if (_lastInput.Rank == 1)
        {
            batch = 1;
        }
        else
        {
            batch = _lastInput.Shape[0];
            if (batch <= 0)
            {
                throw new ArgumentException("Expected last input tensor to have a positive batch dimension (Shape[0]).", nameof(outputGradient));
            }
        }

        var expectedGradientLength = batch * _outputSize;
        if (outputGradient.Length != expectedGradientLength)
        {
            throw new ArgumentException($"Expected output gradient length {expectedGradientLength} for batch {batch} and outputSize {_outputSize}, but got {outputGradient.Length}.", nameof(outputGradient));
        }

        var activationGradient = ApplyActivationDerivative(_lastPreActivation, outputGradient);
        var flatInput = _lastInput.ToVector();
        var flatGradient = activationGradient.ToVector();

        // Accumulate gradients for weight means and log variances
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                // Gradient w.r.t. weight mean
                var gradMean = NumOps.Zero;
                for (int b = 0; b < batch; b++)
                {
                    var inputValue = flatInput[b * _inputSize + j];
                    var gradValue = flatGradient[b * _outputSize + i];
                    gradMean = NumOps.Add(gradMean, NumOps.Multiply(gradValue, inputValue));
                }
                _weightMeanGradient[i, j] = NumOps.Add(_weightMeanGradient[i, j], gradMean);

                // Gradient w.r.t. weight log variance (from reparameterization trick)
                var weightStd = NumOps.Sqrt(NumOps.Exp(_weightLogVar[i, j]));
                var epsilon = NumOps.Divide(
                    NumOps.Subtract(_sampledWeights[i, j], _weightMean[i, j]),
                    NumOps.Add(weightStd, NumOps.FromDouble(1e-8))
                );
                var gradLogVar = NumOps.Multiply(
                    NumOps.Multiply(gradMean, epsilon),
                    NumOps.Multiply(NumOps.FromDouble(0.5), weightStd)
                );
                _weightLogVarGradient[i, j] = NumOps.Add(_weightLogVarGradient[i, j], gradLogVar);
            }

            // Gradient w.r.t. bias
            var biasGradMean = NumOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                biasGradMean = NumOps.Add(biasGradMean, flatGradient[b * _outputSize + i]);
            }
            _biasMeanGradient[i] = NumOps.Add(_biasMeanGradient[i], biasGradMean);

            var biasStd = NumOps.Sqrt(NumOps.Exp(_biasLogVar[i]));
            var biasEpsilon = NumOps.Divide(
                NumOps.Subtract(_sampledBias![i], _biasMean[i]),
                NumOps.Add(biasStd, NumOps.FromDouble(1e-8))
            );
            var biasGradLogVar = NumOps.Multiply(
                NumOps.Multiply(biasGradMean, biasEpsilon),
                NumOps.Multiply(NumOps.FromDouble(0.5), biasStd)
            );
            _biasLogVarGradient[i] = NumOps.Add(_biasLogVarGradient[i], biasGradLogVar);
        }

        // Compute input gradient
        var inputGradient = new Vector<T>(batch * _inputSize);
        for (int b = 0; b < batch; b++)
        {
            var inputOffset = b * _inputSize;
            var gradOffset = b * _outputSize;

            for (int j = 0; j < _inputSize; j++)
            {
                var sum = NumOps.Zero;
                for (int i = 0; i < _outputSize; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_sampledWeights[i, j], flatGradient[gradOffset + i]));
                }
                inputGradient[inputOffset + j] = sum;
            }
        }

        return new Tensor<T>(_lastInput.Shape, inputGradient);
    }

    /// <inheritdoc/>
    public void AddKLDivergenceGradients(T klScale)
    {
        var priorVar = NumOps.Multiply(_priorSigma, _priorSigma);
        var half = NumOps.FromDouble(0.5);

        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                var meanGrad = NumOps.Divide(_weightMean[i, j], priorVar);
                _weightMeanGradient[i, j] = NumOps.Add(_weightMeanGradient[i, j], NumOps.Multiply(klScale, meanGrad));

                var variance = NumOps.Exp(_weightLogVar[i, j]);
                var logVarGrad = NumOps.Multiply(half, NumOps.Subtract(NumOps.Divide(variance, priorVar), NumOps.One));
                _weightLogVarGradient[i, j] = NumOps.Add(_weightLogVarGradient[i, j], NumOps.Multiply(klScale, logVarGrad));
            }

            var biasMeanGrad = NumOps.Divide(_biasMean[i], priorVar);
            _biasMeanGradient[i] = NumOps.Add(_biasMeanGradient[i], NumOps.Multiply(klScale, biasMeanGrad));

            var biasVar = NumOps.Exp(_biasLogVar[i]);
            var biasLogVarGrad = NumOps.Multiply(half, NumOps.Subtract(NumOps.Divide(biasVar, priorVar), NumOps.One));
            _biasLogVarGradient[i] = NumOps.Add(_biasLogVarGradient[i], NumOps.Multiply(klScale, biasLogVarGrad));
        }
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

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        SetParameters(parameters);
        _sampledWeights = null;
        _sampledBias = null;
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
        _lastPreActivation = null;
        _sampledWeights = null;
        _sampledBias = null;
        ClearGradients();
    }

    private double NextGaussian(double mean = 0.0, double stdDev = 1.0)
    {
        lock (_rngLock)
        {
            return _rng.NextGaussian(mean, stdDev);
        }
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();

        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _weightMeanGradient[i, j] = NumOps.Zero;
                _weightLogVarGradient[i, j] = NumOps.Zero;
            }
        }

        _biasMeanGradient.Fill(NumOps.Zero);
        _biasLogVarGradient.Fill(NumOps.Zero);
    }

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
        => throw new NotSupportedException($"{GetType().Name} does not currently support JIT compilation.");

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;
}
