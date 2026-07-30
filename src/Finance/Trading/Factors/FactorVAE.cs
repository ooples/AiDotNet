using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Variational autoencoder for learning disentangled financial factors.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// FactorVAE combines a variational autoencoder with a disentanglement penalty
/// so each latent dimension captures a distinct factor.
/// </para>
/// <para>
/// <b>For Beginners:</b> The model compresses market data into a small set of hidden
/// variables (factors). The disentanglement penalty encourages each factor to capture
/// a different driver of returns rather than mixing everything together.
/// </para>
/// <para>
/// Reference: Kim &amp; Mnih (2019). "Disentangling by Factorising"
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for disentangled factor learning via VAE (50 assets, 10 features, 5 latent factors)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 60, inputWidth: 10, inputDepth: 1, outputSize: 5);
///
/// // Training mode: VAE learns independent latent factors with disentanglement penalty
/// var model = new FactorVAE&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained FactorVAE model
/// var onnxModel = new FactorVAE&lt;double&gt;(architecture, "factor_vae.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation corrected. arXiv 2005.02634 is "Dependency Aware Filter Pruning", unrelated. FactorVAE is
// AAAI 2022 (vol. 36 no. 4, pp. 4468-4476) and is not on arXiv, so the canonical DOI is used; the year
// was also wrong (2020 -> 2022).
//
// IMPLEMENTATION NOTE: the VAE and dynamic-factor machinery are present, but the paper's named
// contribution — a PRIOR-POSTERIOR learning method, where a prior factor distribution conditioned on
// observable features is aligned against a posterior informed by future returns — has no counterpart
// here (no prior/posterior split at all). Verifying and closing that gap is tracked separately.
[ResearchPaper("FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns",
    "https://doi.org/10.1609/aaai.v36i4.20369",
    Year = 2022,
    Authors = "Yitong Duan, Lei Wang, Qizhong Zhang, Jian Li")]
public class FactorVAE<T> : FinancialModelBase<T>, IFactorModel<T>
{
    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    
    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly FactorVAEOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _numFactors;
    private int _numAssets;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _latentDimension;
    private int _sequenceLength;
    private int _predictionHorizon;
    private double _beta;
    private double _gamma;
    private double _dropoutRate;
    private readonly Random _random;

    #endregion

    #region Interface Properties

    /// <summary>
    /// Gets whether the model is using native layers (true) or ONNX inference (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode supports training, ONNX mode is for fast predictions.
    /// </para>
    /// </remarks>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported in the current mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training is only available in native mode.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of latent factors learned by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many hidden drivers of returns the model discovers.
    /// </para>
    /// </remarks>
    public int NumFactors => _numFactors;

    /// <summary>
    /// Gets the number of assets covered by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the asset universe being modeled.
    /// </para>
    /// </remarks>
    public int NumAssets => _numAssets;

    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each asset has this many features (prices, indicators, etc.).
    /// </para>
    /// </remarks>
    public override int NumFeatures => _numFeatures;

    /// <summary>
    /// Gets the input sequence length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many time steps of history the model sees at once.
    /// </para>
    /// </remarks>
    public override int SequenceLength => _sequenceLength;

    /// <summary>
    /// Gets the prediction horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How far ahead the model is trained to forecast.
    /// </para>
    /// </remarks>
    public override int PredictionHorizon => _predictionHorizon;

    /// <summary>
    /// Gets the dimension of the latent space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the compressed representation the VAE learns.
    /// </para>
    /// </remarks>
    public int LatentDimension => _latentDimension;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new FactorVAE in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a trained VAE stored as
    /// an ONNX file and want fast, read-only inference.
    /// </para>
    /// </remarks>
    public FactorVAE(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FactorVAEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _options = options ?? new FactorVAEOptions<T>();
        Options = _options;
        _options.Validate();

        _numFactors = _options.NumFactors;
        _numAssets = _options.NumAssets;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _latentDimension = _options.LatentDimension;
        _sequenceLength = _options.SequenceLength;
        _predictionHorizon = _options.PredictionHorizon;
        _beta = _options.Beta;
        _gamma = _options.Gamma;
        _dropoutRate = _options.DropoutRate;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
        InstallFactorVAEObjective();
    }

    /// <summary>
    /// Initializes a new FactorVAE in native mode for training and inference.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to train a FactorVAE from scratch
    /// and learn disentangled market factors from your own data.
    /// </para>
    /// </remarks>
    public FactorVAE(
        NeuralNetworkArchitecture<T> architecture,
        FactorVAEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        OnnxModelPath = null;
        OnnxSession = null;

        _options = options ?? new FactorVAEOptions<T>();
        Options = _options;
        _options.Validate();

        _numFactors = _options.NumFactors;
        _numAssets = _options.NumAssets;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _latentDimension = _options.LatentDimension;
        _sequenceLength = _options.SequenceLength;
        _predictionHorizon = _options.PredictionHorizon;
        _beta = _options.Beta;
        _gamma = _options.Gamma;
        _dropoutRate = _options.DropoutRate;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
        InstallFactorVAEObjective();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for FactorVAE.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The default architecture has three main parts:
    /// </para>
    /// <para>
    /// 1. Encoder: Compresses inputs into a latent representation
    /// 2. Disentangler: Encourages factors to be independent
    /// 3. Decoder: Reconstructs inputs from the latent factors
    /// </para>
    /// <para>
    /// If you provide custom layers, the model uses them instead.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultFactorVAELayers(
                Architecture,
                _numFeatures,
                _hiddenDimension,
                _latentDimension,
                _numFactors,
                _dropoutRate,
                _numAssets));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Runs a forward pass to reconstruct inputs or generate factor outputs.
    /// </summary>
    /// <param name="input">Input tensor of market features.</param>
    /// <returns>Model output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This passes data through the VAE to produce outputs
    /// that reflect the learned latent factors.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!_useNativeMode) return PredictOnnx(input);
        if (!HasFactorSpans) return PredictNative(input);

        // Prediction uses the PRIOR branch: the future returns the posterior needs are, by definition,
        // unavailable here. The prior's mean is used rather than a sample, so predictions are
        // deterministic — sampling at inference would make the same input give different answers.
        var features = RunSpan(input, FeatureSpanStart, FeatureSpanEnd);
        var prior = RunSpan(features, PriorSpanStart, PriorSpanEnd);
        var (priorMean, _) = SplitMeanAndLogVariance(prior);
        return DecodeReturns(priorMean, features);
    }

    #endregion

    #region Prior-Posterior Learning (Duan et al., AAAI 2022)

    /// <summary>Layer-span boundaries into <c>Layers</c> for the four sub-networks.</summary>
    private int FeatureSpanStart => 0;

    private int FeatureSpanEnd => LayerHelper<T>.FactorVAEFeatureExtractorLayerCount;

    private int PriorSpanStart => FeatureSpanEnd;

    private int PriorSpanEnd => PriorSpanStart + LayerHelper<T>.FactorVAEPriorLayerCount;

    private int PosteriorSpanStart => PriorSpanEnd;

    private int PosteriorSpanEnd => PosteriorSpanStart + LayerHelper<T>.FactorVAEPosteriorLayerCount;

    private int AlphaSpanStart => PosteriorSpanEnd;

    private int AlphaSpanEnd => AlphaSpanStart + LayerHelper<T>.FactorVAEAlphaLayerCount;

    private int BetaSpanStart => AlphaSpanEnd;

    private int BetaSpanEnd => BetaSpanStart + LayerHelper<T>.FactorVAEBetaLayerCount;

    private int DecoderSpanEnd => BetaSpanEnd;

    /// <summary>
    /// True when <c>Layers</c> matches the four-span layout this model drives explicitly. A caller that
    /// supplied a custom architecture keeps the plain sequential path instead.
    /// </summary>
    private bool HasFactorSpans => Layers.Count == DecoderSpanEnd;

    /// <summary>
    /// The KL divergence between posterior and prior from the most recent training forward pass, kept
    /// tape-connected so <see cref="FactorVAEObjective{T}"/> can add it to the reconstruction term and
    /// have the gradient reach BOTH heads.
    /// </summary>
    private Tensor<T>? _lastKlDivergence;

    /// <summary>
    /// Gets the KL divergence recorded by the last training forward pass, or null if none ran.
    /// </summary>
    internal Tensor<T>? LastKlDivergence => _lastKlDivergence;

    /// <inheritdoc/>
    /// <remarks>
    /// Training runs the POSTERIOR branch, which is allowed to see the realized returns
    /// (<paramref name="input"/> carries features; the targets reach us through
    /// <see cref="SetPosteriorReturns"/>). Factors are sampled from the posterior via the
    /// reparameterization trick so the sampling stays differentiable, and the KL against the prior is
    /// recorded for the objective. Without the KL the prior is never trained, and prediction — which can
    /// only use the prior — would be untrained no matter how well training loss fell.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!HasFactorSpans) return base.ForwardForTraining(input);

        var features = RunSpan(input, FeatureSpanStart, FeatureSpanEnd);

        var prior = RunSpan(features, PriorSpanStart, PriorSpanEnd);
        var (priorMean, priorLogVar) = SplitMeanAndLogVariance(prior);

        // The posterior consumes features together with the realized returns. When no returns have been
        // supplied (a caller invoking the training forward directly), fall back to the prior so the pass
        // still produces a prediction rather than throwing.
        var returns = _posteriorReturns;
        if (returns is null)
        {
            _lastKlDivergence = null;
            return DecodeReturns(priorMean, features);
        }

        var posteriorInput = Engine.Concat([features, AlignReturns(returns, features)], features.Shape.Length - 1);
        var posterior = RunSpan(posteriorInput, PosteriorSpanStart, PosteriorSpanEnd);
        var (postMean, postLogVar) = SplitMeanAndLogVariance(posterior);

        var factors = ReparameterizedSample(postMean, postLogVar);
        _lastKlDivergence = GaussianKlDivergence(postMean, postLogVar, priorMean, priorLogVar);

        return DecodeReturns(factors, features);
    }

    /// <summary>
    /// Replaces the plain reconstruction loss with the paper's objective, so the KL between posterior
    /// and prior is actually optimized.
    /// </summary>
    /// <remarks>
    /// Installed after construction rather than passed to <c>base(...)</c> because the KL weight comes
    /// from the options, which are not yet assigned when the base constructor runs. Only installed when
    /// the four-span layout is present — a caller-supplied custom architecture has no prior/posterior
    /// heads for the KL to relate.
    /// </remarks>
    private void InstallFactorVAEObjective()
    {
        if (!_useNativeMode || !HasFactorSpans) return;

        // The tape-based objective needs the concrete base type (ComputeTapeLoss is declared there, not
        // on ILossFunction). A caller-supplied loss that is not tape-capable falls back to MSE on the
        // returns, which is the paper's reconstruction term anyway.
        var reconstruction = _lossFunction as LossFunctionBase<T> ?? new MeanSquaredErrorLoss<T>();

        LossFunction = new FactorVAEObjective<T>(
            reconstruction,
            () => _lastKlDivergence,
            _options.KlWeight);
    }

    /// <summary>Realized returns for the current training step, consumed by the posterior branch.</summary>
    private Tensor<T>? _posteriorReturns;

    /// <summary>
    /// Supplies the realized returns the posterior branch conditions on. Cleared after each step so a
    /// later pass cannot silently reuse a previous batch's future — which would leak information across
    /// steps and inflate training performance.
    /// </summary>
    internal void SetPosteriorReturns(Tensor<T>? returns) => _posteriorReturns = returns;

    /// <summary>Runs layers in <c>[start, end)</c> sequentially.</summary>
    private Tensor<T> RunSpan(Tensor<T> x, int start, int end)
    {
        var current = x;
        for (int i = start; i < end; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Splits a <c>2 * numFactors</c>-wide head into its mean and log-variance halves along the last
    /// axis, using a recorded narrow so both halves stay on the tape.
    /// </summary>
    private (Tensor<T> Mean, Tensor<T> LogVariance) SplitMeanAndLogVariance(Tensor<T> head)
    {
        int axis = head.Shape.Length - 1;
        int width = head.Shape[axis] / 2;
        var mean = Engine.TensorNarrow(head, axis, 0, width);
        var logVar = Engine.TensorNarrow(head, axis, width, width);
        return (mean, logVar);
    }

    /// <summary>
    /// Samples <c>z = mean + exp(logVar / 2) * epsilon</c> — the reparameterization trick.
    /// </summary>
    /// <remarks>
    /// The randomness is isolated in <c>epsilon</c>, which is a constant with respect to the parameters,
    /// so the gradient flows through mean and log-variance. Sampling the Gaussian directly would put the
    /// randomness inside the graph and leave both heads without a gradient.
    /// </remarks>
    private Tensor<T> ReparameterizedSample(Tensor<T> mean, Tensor<T> logVariance)
    {
        var epsilon = new Tensor<T>(mean.Shape.ToArray());
        for (int i = 0; i < epsilon.Length; i++)
        {
            // Box-Muller from the seeded RNG keeps sampling reproducible for a given seed.
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            epsilon[i] = NumOps.FromDouble(
                Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        var stdDev = Engine.TensorExp(
            Engine.TensorMultiplyScalar(logVariance, NumOps.FromDouble(0.5)));
        return Engine.TensorAdd(mean, Engine.TensorMultiply(stdDev, epsilon));
    }

    /// <summary>
    /// KL divergence between two diagonal Gaussians, summed over factors and averaged over the batch:
    /// <c>sum[ (logVar_p - logVar_q) / 2 + (var_q + (m_q - m_p)^2) / (2 var_p) - 1/2 ]</c>.
    /// </summary>
    private Tensor<T> GaussianKlDivergence(
        Tensor<T> postMean, Tensor<T> postLogVar, Tensor<T> priorMean, Tensor<T> priorLogVar)
    {
        var postVar = Engine.TensorExp(postLogVar);
        var priorVar = Engine.TensorExp(priorLogVar);

        var logRatio = Engine.TensorMultiplyScalar(
            Engine.TensorSubtract(priorLogVar, postLogVar), NumOps.FromDouble(0.5));

        var meanDiff = Engine.TensorSubtract(postMean, priorMean);
        var numerator = Engine.TensorAdd(postVar, Engine.TensorMultiply(meanDiff, meanDiff));
        var ratio = Engine.TensorMultiplyScalar(
            Engine.TensorDivide(numerator, priorVar), NumOps.FromDouble(0.5));

        var perFactor = Engine.TensorAddScalar(
            Engine.TensorAdd(logRatio, ratio), NumOps.FromDouble(-0.5));

        var axes = Enumerable.Range(0, perFactor.Shape.Length).ToArray();
        return Engine.ReduceMean(perFactor, axes, keepDims: false);
    }

    /// <summary>
    /// Decodes factors into predicted cross-sectional returns using the paper's linear factor
    /// structure, <c>y = alpha + beta * z</c> (equations 18-19).
    /// </summary>
    /// <remarks>
    /// <c>alpha</c> is each stock's idiosyncratic expected return and <c>beta</c> its exposures to the
    /// factors, both read off the stock latent features; the factors enter ONLY through the linear
    /// product. This is the dynamic-factor-model half of the paper, and it is also why an imperfect
    /// prior degrades gracefully: alpha still carries the baseline return.
    /// </remarks>
    private Tensor<T> DecodeReturns(Tensor<T> factors, Tensor<T> features)
    {
        var alpha = RunSpan(features, AlphaSpanStart, AlphaSpanEnd);   // [.., numAssets]
        var betaFlat = RunSpan(features, BetaSpanStart, BetaSpanEnd);  // [.., numAssets * numFactors]

        int numFactors = factors.Shape[factors.Shape.Length - 1];
        int numAssets = alpha.Shape[alpha.Shape.Length - 1];

        if (alpha.Shape.Length == 1)
        {
            // Unbatched: beta is [numAssets, numFactors], z is [numFactors].
            var beta = Engine.Reshape(betaFlat, [numAssets, numFactors]);
            var z = Engine.Reshape(factors, [numFactors, 1]);
            var contribution = Engine.Reshape(Engine.TensorMatMul(beta, z), [numAssets]);
            return Engine.TensorAdd(alpha, contribution);
        }

        int batch = alpha.Shape[0];
        var betaBatched = Engine.Reshape(betaFlat, [batch, numAssets, numFactors]);
        var zBatched = Engine.Reshape(factors, [batch, numFactors, 1]);
        var product = Engine.BatchMatMul(betaBatched, zBatched);       // [batch, numAssets, 1]
        return Engine.TensorAdd(alpha, Engine.Reshape(product, [batch, numAssets]));
    }

    /// <summary>
    /// Reshapes realized returns so they can be concatenated onto the features along the last axis.
    /// </summary>
    private Tensor<T> AlignReturns(Tensor<T> returns, Tensor<T> features)
    {
        int rank = features.Shape.Length;
        if (returns.Shape.Length == rank) return returns;

        // Flatten everything after the leading (batch) axis onto the feature axis.
        int lead = rank > 1 ? features.Shape[0] : 1;
        int rest = returns.Length / Math.Max(1, lead);
        return rank > 1
            ? Engine.Reshape(returns, [lead, rest])
            : Engine.Reshape(returns, [returns.Length]);
    }

    /// <summary>
    /// Trains the model on a batch of inputs and targets.
    /// </summary>
    /// <param name="input">Input tensor of market features.</param>
    /// <param name="target">Target tensor for reconstruction.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the VAE to reconstruct inputs while
    /// keeping factors disentangled. The beta and gamma settings control this balance.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        // Issue #1166: the old body computed a loss + gradient and then
        // called _optimizer.UpdateParameters(Layers) without a backward
        // pass, so every layer's UpdateParameters threw "Backward pass
        // must be called before updating parameters." Delegate to
        // FinancialModelBase.Train — it routes through the tape-based
        // NeuralNetworkBase.TrainWithTape flow (GradientTape forward +
        // tape.ComputeGradients + optimizer.Step) that every other
        // NeuralNetworkBase subclass uses.
        //
        // The realized returns are handed to the posterior branch for the duration of this step only.
        // Clearing them afterwards matters: a later forward that reused the previous batch's returns
        // would be conditioning on a future it should not see, which inflates apparent accuracy.
        SetPosteriorReturns(target);
        try
        {
            base.Train(input, target);
        }
        finally
        {
            SetPosteriorReturns(null);
        }
    }

    /// <summary>
    /// Updates model parameters from a flat vector.
    /// </summary>
    /// <param name="parameters">Flat parameter vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you load saved weights into the model,
    /// which is useful for serialization and fine-tuning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = checked((int)layer.ParameterCount);
            if (count <= 0)
                continue;

            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    /// <summary>
    /// Gets metadata describing this model instance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata summarizes the model setup for diagnostics.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NumFactors"] = _numFactors,
                ["LatentDimension"] = _latentDimension,
                ["Beta"] = _beta,
                ["Gamma"] = _gamma,
                ["UseNativeMode"] = _useNativeMode
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Used by the framework to clone models with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new FactorVAEOptions<T>
        {
            NumFactors = _numFactors,
            NumAssets = _numAssets,
            NumFeatures = _numFeatures,
            HiddenDimension = _hiddenDimension,
            LatentDimension = _latentDimension,
            SequenceLength = _sequenceLength,
            PredictionHorizon = _predictionHorizon,
            Beta = _beta,
            Gamma = _gamma,
            DropoutRate = _dropoutRate
        };

        return new FactorVAE<T>(Architecture, optionsCopy);
    }

    /// <summary>
    /// Serializes model-specific data.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves the model configuration so it can be restored later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFactors);
        writer.Write(_numAssets);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_latentDimension);
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_beta);
        writer.Write(_gamma);
        writer.Write(_dropoutRate);
    }

    /// <summary>
    /// Deserializes model-specific data.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Restores the saved configuration when loading a model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numFactors = reader.ReadInt32();
        _numAssets = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _latentDimension = reader.ReadInt32();
        _sequenceLength = reader.ReadInt32();
        _predictionHorizon = reader.ReadInt32();
        _beta = reader.ReadDouble();
        _gamma = reader.ReadDouble();
        _dropoutRate = reader.ReadDouble();
    }

    #endregion

    #region IFactorModel Implementation

    /// <summary>
    /// Extracts latent factors from asset returns.
    /// </summary>
    /// <param name="returns">Asset returns tensor.</param>
    /// <returns>Factor representation tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This passes the data through the encoder to obtain
    /// the compact factor representation.
    /// </para>
    /// </remarks>
    public Tensor<T> ExtractFactors(Tensor<T> returns)
    {
        var current = returns;
        int encoderEnd = Math.Min(5, Layers.Count - 3);
        for (int i = 0; i < encoderEnd; i++)
        {
            current = Layers[i].Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Computes factor loadings for each asset.
    /// </summary>
    /// <param name="returns">Asset returns tensor.</param>
    /// <returns>Factor loadings matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Factor loadings show how much each asset depends on each factor.
    /// </para>
    /// </remarks>
    public Tensor<T> GetFactorLoadings(Tensor<T> returns)
    {
        return new Tensor<T>(new[] { _numAssets, _numFactors });
    }

    /// <summary>
    /// Predicts expected returns from factor exposures.
    /// </summary>
    /// <param name="factorExposures">Tensor of factor exposures.</param>
    /// <returns>Predicted returns tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converts factor values into expected asset returns.
    /// </para>
    /// </remarks>
    public Tensor<T> PredictReturns(Tensor<T> factorExposures)
    {
        return Predict(factorExposures);
    }

    /// <summary>
    /// Computes the factor covariance matrix.
    /// </summary>
    /// <param name="returns">Asset returns tensor.</param>
    /// <returns>Factor covariance matrix tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how factors move together, which matters for risk.
    /// </para>
    /// </remarks>
    public Tensor<T> GetFactorCovariance(Tensor<T> returns)
    {
        return new Tensor<T>(new[] { _numFactors, _numFactors });
    }

    /// <summary>
    /// Computes alpha (excess return) for each asset.
    /// </summary>
    /// <param name="returns">Asset returns tensor.</param>
    /// <param name="factorReturns">Factor returns tensor.</param>
    /// <returns>Alpha tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Alpha is the portion of returns not explained by factors.
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeAlpha(Tensor<T> returns, Tensor<T> factorReturns)
    {
        return new Tensor<T>(new[] { _numAssets });
    }

    /// <summary>
    /// Gets factor model metrics.
    /// </summary>
    /// <returns>Dictionary of factor metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a small summary of the model configuration.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFactorMetrics()
    {
        return new Dictionary<string, T>
        {
            ["NumFactors"] = NumOps.FromDouble(_numFactors),
            ["LatentDimension"] = NumOps.FromDouble(_latentDimension),
            ["Beta"] = NumOps.FromDouble(_beta),
            ["Gamma"] = NumOps.FromDouble(_gamma)
        };
    }

    #endregion

    #region IFinancialModel Implementation

    /// <summary>
    /// Generates a forecast using the model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Optional quantiles (unused for factor prediction).</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forecasting here means using the learned factors to
    /// predict the next returns or reconstructed outputs.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null)
    {
        return Predict(input);
    }

    /// <summary>
    /// Training-mode forward pass. Runs the native layer stack directly on the live gradient tape.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default <see cref="FinancialModelBase{T}.ForwardNativeForTraining"/> delegates to
    /// <see cref="Forecast"/>, which here calls <see cref="NeuralNetworks.NeuralNetworkBase{T}.Predict"/>
    /// — the INFERENCE path. Predict runs inside a <c>TensorArena</c> inference scope that detaches its
    /// output from the gradient tape, so during <c>TrainWithTape</c> the backward pass reaches no
    /// parameters and every training step is a silent no-op (the GradientFlow_ShouldBeNonZeroAndFinite,
    /// Training_ShouldChangeParameters and LossStrictlyDecreasesOnMemorizationTask failures). Route
    /// training through <see cref="PredictNative"/>, which walks the raw layer stack so the forward is
    /// recorded on the tape and gradients flow to every layer's parameters. PredictNative also runs the
    /// encoder BatchNorm in inference mode (running stats) — required so a batch-of-one training step
    /// does not normalize every feature to its own mean and collapse the output. Mirrors the NTM #1670 /
    /// FactorTransformer tape-severance fix.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        return PredictNative(input);
    }

    /// <summary>
    /// Gets financial metrics for the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Provides factor-focused metrics from this model.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return GetFactorMetrics();
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Runs a forward pass using native layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This passes data through the C# layers to get outputs.
    /// </para>
    /// </remarks>
    private Tensor<T> PredictNative(Tensor<T> input)
    {
        // Inference mode is REQUIRED here: the encoder stack contains
        // BatchNormalizationLayers, which in training mode normalize across the
        // batch axis. A single-instance prediction (batch = 1) then has each
        // feature's batch-mean equal to its own value, so the normalized output
        // is ~0 regardless of the input and every constant input collapses to the
        // same prediction. Inference mode uses the running statistics instead, so
        // BatchNorm stays affine and the input level propagates.
        SetTrainingMode(false);

        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Runs a forward pass using the ONNX runtime.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor from ONNX inference.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This uses a pretrained ONNX file for fast predictions.
    /// </para>
    /// </remarks>
    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = OnnxSession.InputMetadata.Keys.First();

        using var results = OnnxSession.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        });

        var outputTensor = results.First().AsTensor<float>();
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources used by the model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Always dispose models when finished to free memory,
    /// especially if an ONNX session was loaded.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxSession?.Dispose();
            foreach (var layer in Layers)
            {
                if (layer is IDisposable disposable)
                    disposable.Dispose();
            }
        }
        base.Dispose(disposing);
    }

    #endregion
}

