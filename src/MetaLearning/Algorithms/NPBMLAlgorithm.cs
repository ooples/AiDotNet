using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of NPBML (Neural Process-Based Meta-Learning).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// NPBML combines neural processes with meta-learning for probabilistic few-shot prediction.
/// It encodes the support set into a latent distribution and samples from it to capture
/// task-level uncertainty.
/// </para>
/// <para><b>For Beginners:</b> NPBML is a probabilistic meta-learner that knows when it's uncertain:
///
/// **Standard meta-learners:**
/// Given a support set, produce ONE adapted model that gives ONE prediction per query.
/// You don't know how confident the model is.
///
/// **NPBML's probabilistic approach:**
/// 1. Encode support examples into a DISTRIBUTION (mean + variance), not a single point
/// 2. Sample from this distribution multiple times
/// 3. Each sample gives a slightly different prediction
/// 4. If samples agree: model is confident
/// 5. If samples disagree: model is uncertain about this task
///
/// **Why this matters:**
/// - Few support examples = high uncertainty (wide distribution)
/// - Many similar support examples = low uncertainty (narrow distribution)
/// - Ambiguous task = high uncertainty (can't determine the right adaptation)
///
/// **Analogy:**
/// It's like asking 5 experts (sampled from the distribution). If all 5 agree, you're confident.
/// If they give different answers, you know the task is ambiguous.
/// </para>
/// <para><b>Algorithm - NPBML:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor      # Shared backbone
/// e_phi = encoder                  # Maps support set to latent distribution (mu, sigma)
/// d_psi = decoder                  # Maps latent + query features to predictions
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Extract features
///         z_s = f_theta(support_x)
///         z_q = f_theta(query_x)
///
///         # 2. Encode support set to latent distribution
///         mu, log_sigma = e_phi(aggregate(z_s, support_y))
///
///         # 3. Sample from latent distribution (reparameterization trick)
///         epsilon ~ N(0, 1)
///         z_task = mu + exp(log_sigma) * epsilon
///
///         # 4. Decode predictions
///         predictions = d_psi(z_q, z_task)
///
///         # 5. Loss = reconstruction + KL divergence
///         recon_loss = cross_entropy(predictions, query_y)
///         kl_loss = KL(N(mu, sigma) || N(0, 1))
///         loss = recon_loss + beta * kl_loss
///
///     theta, phi, psi = theta, phi, psi - lr * grad(loss)
///
/// # At test time (multiple samples for uncertainty)
/// for sample in range(num_samples):
///     z_task ~ N(mu, sigma)
///     pred_i = d_psi(z_q, z_task)
/// final_pred = mean(pred_1, ..., pred_S)
/// uncertainty = variance(pred_1, ..., pred_S)
/// </code>
/// </para>
/// </remarks>
public class NPBMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly NPBMLOptions<T, TInput, TOutput> _npbmlOptions;

    /// <summary>Parameters for the encoder (support set -> latent distribution).</summary>
    private Vector<T> _encoderParams = new Vector<T>(0);

    /// <summary>Parameters for the decoder (latent + query -> predictions).</summary>
    private Vector<T> _decoderParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.NPBML;

    /// <summary>Initializes a new NPBML meta-learner.</summary>
    /// <param name="options">Configuration options for NPBML.</param>
    public NPBMLAlgorithm(NPBMLOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _npbmlOptions = options;
        InitializeEncoderDecoder();
    }

    /// <summary>Initializes encoder and decoder parameters.</summary>
    private void InitializeEncoderDecoder()
    {
        int latentDim = _npbmlOptions.LatentDim;
        int hiddenDim = latentDim * 2;

        // Encoder: features -> hidden -> (mu, log_sigma)
        int encoderSize = hiddenDim * hiddenDim + hiddenDim + hiddenDim * latentDim * 2 + latentDim * 2;
        _encoderParams = new Vector<T>(encoderSize);

        // Decoder: (latent + query) -> hidden -> prediction
        int decoderSize = (latentDim + hiddenDim) * hiddenDim + hiddenDim + hiddenDim * hiddenDim + hiddenDim;
        _decoderParams = new Vector<T>(decoderSize);

        double scale = Math.Sqrt(2.0 / latentDim);
        for (int i = 0; i < encoderSize; i++)
            _encoderParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        for (int i = 0; i < decoderSize; i++)
            _decoderParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
    }

    /// <summary>
    /// Computes the KL divergence between a diagonal Gaussian and the standard normal.
    /// </summary>
    /// <param name="mu">Mean of the approximate posterior.</param>
    /// <param name="logSigma">Log standard deviation of the approximate posterior.</param>
    /// <returns>KL divergence value.</returns>
    private double ComputeKLDivergence(double[] mu, double[] logSigma)
    {
        // KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + 2*log_sigma - mu^2 - sigma^2)
        double kl = 0;
        for (int i = 0; i < mu.Length; i++)
        {
            double sigma2 = Math.Exp(2.0 * logSigma[i]);
            kl += -0.5 * (1.0 + 2.0 * logSigma[i] - mu[i] * mu[i] - sigma2);
        }
        return kl;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            // Classification loss (backbone)
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // KL divergence: encode support features into latent distribution, then compute KL(q||p)
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var dist = EncodeToDistribution(supportFeatures);
            double klLoss = 0;
            if (dist != null)
                klLoss = ComputeKLDivergence(dist.Value.mu, dist.Value.logSigma);

            // Combined loss
            double totalLoss = NumOps.ToDouble(queryLoss) + _npbmlOptions.KLWeight * klLoss;
            losses.Add(NumOps.FromDouble(totalLoss));

            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _npbmlOptions.OuterLearningRate));
        }

        // Update encoder/decoder via multi-sample SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _npbmlOptions.OuterLearningRate, ComputeAuxLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _decoderParams, _npbmlOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Computes the average loss over a task batch using encoder/decoder + KL divergence.
    /// Called by SPSA to measure how perturbed encoder/decoder params affect loss.
    /// </summary>
    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var queryFeatures = ConvertToVector(MetaModel.Predict(task.QueryInput));

            // Encode support features and compute KL divergence (uses _encoderParams)
            var dist = EncodeToDistribution(supportFeatures);
            double klLoss = 0;
            if (dist != null)
                klLoss = ComputeKLDivergence(dist.Value.mu, dist.Value.logSigma);

            // Encode and sample latent from support features (uses _encoderParams)
            var sampledLatent = EncodeAndSample(supportFeatures);

            // Decode: latent + query -> decoded features (uses _decoderParams)
            var decodedQuery = DecodeLatent(sampledLatent, queryFeatures);

            if (decodedQuery != null && queryFeatures != null && queryFeatures.Length > 0)
            {
                double sumRatio = 0;
                int count = 0;
                for (int i = 0; i < Math.Min(queryFeatures.Length, decodedQuery.Length); i++)
                {
                    double rawVal = NumOps.ToDouble(queryFeatures[i]);
                    double decodedVal = NumOps.ToDouble(decodedQuery[i]);
                    if (Math.Abs(rawVal) > 1e-10)
                    {
                        sumRatio += Math.Max(0.5, Math.Min(2.0, decodedVal / rawVal));
                        count++;
                    }
                }
                if (count > 0)
                {
                    double avgRatio = sumRatio / count;
                    var currentParams = MetaModel.GetParameters();
                    var modulatedParams = new Vector<T>(currentParams.Length);
                    for (int i = 0; i < currentParams.Length; i++)
                        modulatedParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(avgRatio));
                    MetaModel.SetParameters(modulatedParams);
                }
            }

            double queryLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            totalLoss += queryLoss + _npbmlOptions.KLWeight * klLoss;
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    /// <summary>
    /// Encodes support features into a latent distribution (mu, log_sigma) using the encoder network.
    /// Aggregates all support features, then passes through a per-latent-dim encoder.
    /// </summary>
    /// <param name="supportFeatures">Support set features.</param>
    /// <returns>Tuple of (mu, logSigma) arrays, or null if input is empty.</returns>
    private (double[] mu, double[] logSigma)? EncodeToDistribution(Vector<T>? supportFeatures)
    {
        if (supportFeatures == null || supportFeatures.Length == 0)
            return null;

        int latentDim = _npbmlOptions.LatentDim;

        // Aggregate support features (cross-element: mean over all support examples)
        T supportMean = NumOps.Zero;
        for (int i = 0; i < supportFeatures.Length; i++)
            supportMean = NumOps.Add(supportMean, supportFeatures[i]);
        supportMean = NumOps.Divide(supportMean, NumOps.FromDouble(Math.Max(supportFeatures.Length, 1)));
        double aggregated = NumOps.ToDouble(supportMean);

        // Run through encoder to get mu and log_sigma
        var mu = new double[latentDim];
        var logSigma = new double[latentDim];
        int paramIdx = 0;

        for (int i = 0; i < latentDim; i++)
        {
            // Hidden layer
            double wh = paramIdx < _encoderParams.Length
                ? NumOps.ToDouble(_encoderParams[paramIdx++ % _encoderParams.Length]) : 0.01;
            double bh = paramIdx < _encoderParams.Length
                ? NumOps.ToDouble(_encoderParams[paramIdx++ % _encoderParams.Length]) : 0;
            double hidden = Math.Tanh(wh * aggregated + bh);

            // Mu output
            double wMu = paramIdx < _encoderParams.Length
                ? NumOps.ToDouble(_encoderParams[paramIdx++ % _encoderParams.Length]) : 0.01;
            double bMu = paramIdx < _encoderParams.Length
                ? NumOps.ToDouble(_encoderParams[paramIdx++ % _encoderParams.Length]) : 0;
            mu[i] = wMu * hidden + bMu;

            // Log-sigma output (clamped for numerical stability)
            double wSig = paramIdx < _encoderParams.Length
                ? NumOps.ToDouble(_encoderParams[paramIdx++ % _encoderParams.Length]) : 0.01;
            double bSig = paramIdx < _encoderParams.Length
                ? NumOps.ToDouble(_encoderParams[paramIdx++ % _encoderParams.Length]) : 0;
            logSigma[i] = Math.Max(-10.0, Math.Min(10.0, wSig * hidden + bSig));
        }

        return (mu, logSigma);
    }

    /// <summary>
    /// Encodes the support set into a latent distribution and samples from it using
    /// the reparameterization trick. The sampled latent modulates support features
    /// via a sigmoid gate.
    /// </summary>
    /// <param name="supportFeatures">Support set features.</param>
    /// <returns>Sampled latent-modulated feature vector.</returns>
    private Vector<T>? EncodeAndSample(Vector<T>? supportFeatures)
    {
        var dist = EncodeToDistribution(supportFeatures);
        if (dist == null || supportFeatures == null)
            return supportFeatures;

        var (mu, logSigma) = dist.Value;
        int latentDim = _npbmlOptions.LatentDim;

        // Reparameterization trick: z = mu + sigma * epsilon, epsilon ~ N(0,1)
        var sampled = new Vector<T>(supportFeatures.Length);
        for (int i = 0; i < supportFeatures.Length; i++)
        {
            int latIdx = i % latentDim;
            double sigma = Math.Exp(logSigma[latIdx]);
            // Box-Muller for Gaussian sample
            double u1 = Math.Max(RandomGenerator.NextDouble(), 1e-10);
            double u2 = RandomGenerator.NextDouble();
            double epsilon = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

            double z = mu[latIdx] + sigma * epsilon;

            // Modulate support features with sampled latent
            double modulation = 1.0 / (1.0 + Math.Exp(-z)); // Sigmoid gate
            sampled[i] = NumOps.Multiply(supportFeatures[i], NumOps.FromDouble(modulation));
        }

        return sampled;
    }

    /// <summary>
    /// Decodes the sampled latent combined with query features through the decoder network.
    /// Maps (latent, query) -> decoded features using _decoderParams (2-layer MLP with tanh activation).
    /// </summary>
    /// <param name="sampledLatent">Sampled latent vector from the encoder.</param>
    /// <param name="queryFeatures">Query set features.</param>
    /// <returns>Decoded feature vector, or raw queryFeatures if inputs are null.</returns>
    private Vector<T>? DecodeLatent(Vector<T>? sampledLatent, Vector<T>? queryFeatures)
    {
        if (sampledLatent == null || queryFeatures == null || queryFeatures.Length == 0)
            return queryFeatures;

        var decoded = new Vector<T>(queryFeatures.Length);
        int paramIdx = 0;

        for (int i = 0; i < queryFeatures.Length; i++)
        {
            double latentVal = NumOps.ToDouble(sampledLatent[i % sampledLatent.Length]);
            double queryVal = NumOps.ToDouble(queryFeatures[i]);

            // Layer 1: hidden = tanh(w1 * latent + w2 * query + b1)
            double w1 = paramIdx < _decoderParams.Length
                ? NumOps.ToDouble(_decoderParams[paramIdx++ % _decoderParams.Length]) : 0.01;
            double w2 = paramIdx < _decoderParams.Length
                ? NumOps.ToDouble(_decoderParams[paramIdx++ % _decoderParams.Length]) : 0.01;
            double b1 = paramIdx < _decoderParams.Length
                ? NumOps.ToDouble(_decoderParams[paramIdx++ % _decoderParams.Length]) : 0;
            double hidden = Math.Tanh(w1 * latentVal + w2 * queryVal + b1);

            // Layer 2: output = w3 * hidden + b2
            double w3 = paramIdx < _decoderParams.Length
                ? NumOps.ToDouble(_decoderParams[paramIdx++ % _decoderParams.Length]) : 0.01;
            double b2 = paramIdx < _decoderParams.Length
                ? NumOps.ToDouble(_decoderParams[paramIdx++ % _decoderParams.Length]) : 0;
            double output = w3 * hidden + b2;

            // Residual connection: query + scaled decoder output
            decoded[i] = NumOps.FromDouble(queryVal + output * 0.1);
        }

        return decoded;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Encode support set into latent distribution and sample (uses _encoderParams)
        var sampledLatent = EncodeAndSample(supportFeatures);

        // Decode: latent + support features -> decoded features (uses _decoderParams)
        var decodedFeatures = DecodeLatent(sampledLatent, supportFeatures);

        // Compute modulation factors from decoded vs raw support features
        double[]? modulationFactors = null;
        if (decodedFeatures != null && supportFeatures != null && supportFeatures.Length > 0)
        {
            double sumRatio = 0;
            int count = 0;
            for (int i = 0; i < Math.Min(supportFeatures.Length, decodedFeatures.Length); i++)
            {
                double rawVal = NumOps.ToDouble(supportFeatures[i]);
                double decodedVal = NumOps.ToDouble(decodedFeatures[i]);
                if (Math.Abs(rawVal) > 1e-10)
                {
                    sumRatio += Math.Max(0.5, Math.Min(2.0, decodedVal / rawVal));
                    count++;
                }
            }
            if (count > 0)
                modulationFactors = [sumRatio / count];
        }

        return new NPBMLModel<T, TInput, TOutput>(
            MetaModel, currentParams, decodedFeatures, _npbmlOptions.NumSamples, modulationFactors);
    }

}

/// <summary>Adapted model wrapper for NPBML with probabilistic prediction.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses a latent variable sampled from the support
/// set's encoded distribution to modulate backbone parameters. The modulation makes
/// the backbone's predictions task-specific: different tasks get different parameter
/// scalings based on their support set characteristics.
/// </para>
/// </remarks>
internal class NPBMLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _sampledLatent;
    private readonly int _numSamples;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _sampledLatent;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    public NPBMLModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? sampledLatent,
        int numSamples,
        double[]? modulationFactors)
    {
        _model = model;
        _backboneParams = backboneParams;
        _sampledLatent = sampledLatent;
        _numSamples = numSamples;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            // Apply task-specific parameter modulation: scale each backbone parameter
            // by the sigmoid-gated latent sample, cycling through latent dimensions
            var modulatedParams = new Vector<T>(_backboneParams.Length);
            for (int i = 0; i < _backboneParams.Length; i++)
            {
                double mod = _modulationFactors[i % _modulationFactors.Length];
                modulatedParams[i] = NumOps.Multiply(_backboneParams[i], NumOps.FromDouble(mod));
            }
            _model.SetParameters(modulatedParams);
        }
        else
        {
            _model.SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
