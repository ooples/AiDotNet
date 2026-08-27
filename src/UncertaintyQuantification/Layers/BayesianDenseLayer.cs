// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
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
// A fully-connected layer that happens to sample its weights: the shape rule is DenseLayer's, and it is
// read off ForwardTraced rather than assumed. The guard checks only the LAST dimension - "expects
// last-dimension feature size {_inputSize}" - and the tail reconstructs the output as the input's shape
// with `outputShape[^1] = _outputSize`. Every leading axis is carried through untouched, at any rank.
//
// Rank 1 and rank 4 are declared because this layer ACCEPTS them, not because they are typical. The
// rank-4 axes are named [Batch, Channels, Height, Features] rather than with placeholders because that
// is exactly the convolution hand-off, and naming it makes the consequence legible: fed a feature map,
// this layer maps WIDTH as its feature axis - almost never what an author meant, and worth being able
// to see rather than discovering from a size mismatch two layers later.
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Per-position projection: the leading axes are carried through untouched.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class BayesianDenseLayer<T> : LayerBase<T>, IBayesianLayer<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written rather than generated, for the same reason <c>DenseLayer</c> writes its own: the last
    /// axis takes a size the CONFIGURATION fixes, and an attribute cannot carry a constructor argument.
    /// <c>ForwardTraced</c> ends by copying every leading dimension and overwriting only the last -
    /// <c>outputShape[^1] = _outputSize</c> - with the rank-1 and rank-2 special cases reshaping to
    /// <c>[_outputSize]</c> and returning <c>[batch, _outputSize]</c> respectively. Same rule, three
    /// spellings.
    /// </para>
    /// <para>
    /// Sampling the posterior does not touch the shape: the sampled weights are added to
    /// <c>_weightMean</c>, which is <c>[_outputSize, _inputSize]</c> whether or not a sample is drawn.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_outputSize <= 0 || inputRank < 1) return null;

        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputSize));
        OutputAxisContract Pass(TensorAxis axis) => new(axis, AxisRelation.Same(axis));

        // Enumerated rather than generated, because every leading axis needs a DISTINCT role: a relation
        // refers to its input BY ROLE, so two anonymous placeholders in one layout could not be told
        // apart and the whole naming would be refused. Ranks beyond four decline honestly.
        return inputRank switch
        {
            1 => new[] { features },
            2 => new[] { Pass(TensorAxis.Batch), features },
            3 => new[] { Pass(TensorAxis.Batch), Pass(TensorAxis.Time), features },
            4 => new[]
            {
                Pass(TensorAxis.Batch), Pass(TensorAxis.Channels), Pass(TensorAxis.Height), features,
            },
            _ => null,
        };
    }

    private readonly Random _rng;
    private readonly object _rngLock = new();
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly T _priorSigma;

    // Posterior parameters are tensors, not detached Matrix/Vector copies. The
    // tape keys gradients by tensor reference identity, so Forward must use the
    // same instances exposed by GetTrainableParameters. Fields are mutable
    // because the contiguous ParameterBuffer rebinds them to buffer-backed
    // views through SetTrainableParameters.
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _weightMean = Tensor<T>.Empty();
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _weightLogVar = Tensor<T>.Empty();
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _biasMean = Tensor<T>.Empty();
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _biasLogVar = Tensor<T>.Empty();

    // SampleWeights stores epsilon, rather than a detached sampled parameter.
    // Forward applies the reparameterization with Engine operations so both μ
    // and log σ² remain connected to the active gradient tape.
    [Scratch]
    private Tensor<T>? _sampledWeightEpsilon;
    [Scratch]
    private Tensor<T>? _sampledBiasEpsilon;
    [Scratch]
    private bool _samplePending;

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
        : base([inputSize], [outputSize], scalarActivation ?? new IdentityActivation<T>())
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));
        if (double.IsNaN(priorSigma) || double.IsInfinity(priorSigma) || priorSigma <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(priorSigma), "priorSigma must be finite and positive.");

        _inputSize = inputSize;
        _outputSize = outputSize;
        _priorSigma = NumOps.FromDouble(priorSigma);
        _rng = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize weight and bias parameters
        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Initialize weight means with Xavier initialization
        _weightMean = new Tensor<T>([_outputSize, _inputSize]);
        var scale = Math.Sqrt(2.0 / (_inputSize + _outputSize));
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _weightMean[i, j] = NumOps.FromDouble(NextGaussian(0, scale));
            }
        }

        // Initialize weight log variances to small values (start relatively confident)
        _weightLogVar = new Tensor<T>([_outputSize, _inputSize]);
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _weightLogVar[i, j] = NumOps.FromDouble(-5.0); // exp(-5) ≈ 0.0067
            }
        }

        // Initialize bias parameters
        _biasMean = new Tensor<T>([_outputSize]);
        _biasLogVar = new Tensor<T>([_outputSize]);
        for (int i = 0; i < _outputSize; i++)
        {
            _biasMean[i] = NumOps.Zero;
            _biasLogVar[i] = NumOps.FromDouble(-5.0);
        }
    }

    /// <inheritdoc/>

    /// <inheritdoc/>

    private static void ValidateShapeMatch(Tensor<T> incoming, Tensor<T> existing, string parameterName)
    {
        if (incoming.Rank != existing.Rank || incoming.Length != existing.Length)
            throw new ArgumentException(
                $"Shape mismatch for {parameterName}: incoming [{string.Join(",", incoming.Shape)}], " +
                $"expected [{string.Join(",", existing.Shape)}].");

        for (int dimension = 0; dimension < incoming.Rank; dimension++)
        {
            if (incoming.Shape[dimension] != existing.Shape[dimension])
                throw new ArgumentException(
                    $"Shape mismatch for {parameterName}: incoming [{string.Join(",", incoming.Shape)}], " +
                    $"expected [{string.Join(",", existing.Shape)}].");
        }
    }

    /// <summary>
    /// Samples weights from the learned distributions.
    /// </summary>
    public void SampleWeights()
    {
        FillSampleEpsilon();
        _samplePending = true;
    }

    private void FillSampleEpsilon()
    {
        _sampledWeightEpsilon ??= new Tensor<T>([_outputSize, _inputSize]);
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                _sampledWeightEpsilon[i, j] = NumOps.FromDouble(NextGaussian());
            }
        }

        _sampledBiasEpsilon ??= new Tensor<T>([_outputSize]);
        for (int i = 0; i < _outputSize; i++)
            _sampledBiasEpsilon[i] = NumOps.FromDouble(NextGaussian());

        // A compiled tape retains these epsilon tensors by reference. Notify
        // persistent-device backends after in-place refresh so replay observes
        // the new posterior sample without rebuilding the fused optimizer plan.
        Engine.InvalidatePersistentTensor(_sampledWeightEpsilon);
        Engine.InvalidatePersistentTensor(_sampledBiasEpsilon);
    }

    /// <summary>
    /// Refreshes the posterior sample captured by a compiled training tape.
    /// </summary>
    internal void RefreshCompiledTrainingSample()
    {
        if (!IsTrainingMode)
            return;

        FillSampleEpsilon();
        _samplePending = false;
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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank == 0)
            throw new ArgumentException("BayesianDenseLayer expects an input with at least one dimension.", nameof(input));

        int featureSize = input.Rank == 1 ? input.Length : input.Shape[input.Rank - 1];
        if (featureSize != _inputSize)
            throw new ArgumentException(
                $"BayesianDenseLayer expects last-dimension feature size {_inputSize}, got {featureSize} " +
                $"for input shape [{string.Join(",", input.Shape)}].",
                nameof(input));

        // Ordinary inference uses posterior means. Explicit SampleWeights calls
        // (PredictWithUncertainty) are consumed once. Training draws a fresh
        // reparameterized sample for every eager forward; compiled replay
        // refreshes the same captured tensor references between steps.
        bool useSample = IsTrainingMode || _samplePending;
        if (IsTrainingMode && !_samplePending)
            FillSampleEpsilon();
        if (useSample && (_sampledWeightEpsilon is null || _sampledBiasEpsilon is null))
            throw new InvalidOperationException("Bayesian posterior sample was not initialized.");

        Tensor<T> effectiveWeights = _weightMean;
        Tensor<T> effectiveBias = _biasMean;
        if (useSample)
        {
            var weightStd = Engine.TensorSqrt(Engine.TensorExp(_weightLogVar));
            effectiveWeights = Engine.TensorAdd(
                _weightMean,
                Engine.TensorMultiply(weightStd, _sampledWeightEpsilon!));

            var biasStd = Engine.TensorSqrt(Engine.TensorExp(_biasLogVar));
            effectiveBias = Engine.TensorAdd(
                _biasMean,
                Engine.TensorMultiply(biasStd, _sampledBiasEpsilon!));
        }

        _samplePending = false;

        int batchSize;
        Tensor<T> flatInput;
        if (input.Rank == 1)
        {
            batchSize = 1;
            flatInput = Engine.Reshape(input, [1, _inputSize]);
        }
        else if (input.Rank == 2)
        {
            batchSize = input.Shape[0];
            flatInput = input;
        }
        else
        {
            batchSize = 1;
            for (int dimension = 0; dimension < input.Rank - 1; dimension++)
                batchSize *= input.Shape[dimension];
            flatInput = Engine.Reshape(input, [batchSize, _inputSize]);
        }

        // Retain the historical [output,input] parameter layout used by
        // GetParameters/SetParameters, transposing through the Engine so the
        // operation remains connected to the tape.
        var weightTranspose = Engine.TensorTranspose(effectiveWeights);
        var preActivation = Engine.TensorMatMul(flatInput, weightTranspose);
        preActivation = Engine.TensorAdd(
            preActivation,
            Engine.Reshape(effectiveBias, [1, _outputSize]));
        var activated = ApplyActivation(preActivation);

        if (input.Rank == 1) return Engine.Reshape(activated, [_outputSize]);
        if (input.Rank == 2) return activated;

        var outputShape = new int[input.Rank];
        for (int dimension = 0; dimension < input.Rank - 1; dimension++)
            outputShape[dimension] = input.Shape[dimension];
        outputShape[^1] = _outputSize;
        return Engine.Reshape(activated, outputShape);
    }

    /// <inheritdoc/>
    public void AddKLDivergenceGradients(T klScale)
    {
        var priorVar = NumOps.Multiply(_priorSigma, _priorSigma);
        var half = NumOps.FromDouble(0.5);
        var gradients = GetParameterGradients();
        int gradientIndex = 0;

        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                var meanGrad = NumOps.Divide(_weightMean[i, j], priorVar);
                gradients[gradientIndex] = NumOps.Add(
                    gradients[gradientIndex], NumOps.Multiply(klScale, meanGrad));
                gradientIndex++;
            }
        }

        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                var variance = NumOps.Exp(_weightLogVar[i, j]);
                var logVarGrad = NumOps.Multiply(
                    half,
                    NumOps.Subtract(NumOps.Divide(variance, priorVar), NumOps.One));
                gradients[gradientIndex] = NumOps.Add(
                    gradients[gradientIndex], NumOps.Multiply(klScale, logVarGrad));
                gradientIndex++;
            }
        }

        for (int i = 0; i < _outputSize; i++)
        {
            var meanGradient = NumOps.Divide(_biasMean[i], priorVar);
            gradients[gradientIndex] = NumOps.Add(
                gradients[gradientIndex], NumOps.Multiply(klScale, meanGradient));
            gradientIndex++;
        }

        for (int i = 0; i < _outputSize; i++)
        {
            var variance = NumOps.Exp(_biasLogVar[i]);
            var logVarGradient = NumOps.Multiply(
                half,
                NumOps.Subtract(NumOps.Divide(variance, priorVar), NumOps.One));
            gradients[gradientIndex] = NumOps.Add(
                gradients[gradientIndex], NumOps.Multiply(klScale, logVarGradient));
            gradientIndex++;
        }
    }

    /// <summary>
    /// Updates parameters using the accumulated gradients.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        if (ParameterGradients is null) return;
        if (ParameterGradients.Length != ParameterCount)
            throw new InvalidOperationException(
                $"BayesianDenseLayer gradient buffer length {ParameterGradients.Length} " +
                $"does not match ParameterCount {ParameterCount}.");

        int gradientIndex = 0;
        foreach (var parameter in GetTrainableParameters())
        {
            for (int i = 0; i < parameter.Length; i++)
            {
                parameter[i] = NumOps.Subtract(
                    parameter[i],
                    NumOps.Multiply(learningRate, ParameterGradients[gradientIndex++]));
            }
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        SetParameters(parameters);
        _sampledWeightEpsilon = null;
        _sampledBiasEpsilon = null;
        _samplePending = false;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _sampledWeightEpsilon = null;
        _sampledBiasEpsilon = null;
        _samplePending = false;
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
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputSize"] = _inputSize.ToString(invariant);
        metadata["OutputSize"] = _outputSize.ToString(invariant);
        metadata["PriorSigma"] = NumOps.ToDouble(_priorSigma).ToString("R", invariant);
        return metadata;
    }
}
