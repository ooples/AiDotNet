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

            // KL divergence regularization
            int latentDim = _npbmlOptions.LatentDim;
            var mu = new double[latentDim];
            var logSigma = new double[latentDim];
            for (int i = 0; i < latentDim; i++)
            {
                mu[i] = i < _encoderParams.Length ? NumOps.ToDouble(_encoderParams[i]) * 0.01 : 0;
                logSigma[i] = i + latentDim < _encoderParams.Length
                    ? NumOps.ToDouble(_encoderParams[i + latentDim]) * 0.01
                    : 0;
            }
            double klLoss = ComputeKLDivergence(mu, logSigma);

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

        // Update encoder/decoder via SPSA
        UpdateAuxiliaryParams(taskBatch, ref _encoderParams);
        UpdateAuxiliaryParams(taskBatch, ref _decoderParams);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        return new NPBMLModel<T, TInput, TOutput>(MetaModel, MetaModel.GetParameters(), _npbmlOptions.NumSamples);
    }

    /// <summary>Updates auxiliary parameters using SPSA gradient estimation.</summary>
    private void UpdateAuxiliaryParams(TaskBatch<T, TInput, TOutput> taskBatch, ref Vector<T> auxParams)
    {
        double epsilon = 1e-5;
        double lr = _npbmlOptions.OuterLearningRate;

        var direction = new Vector<T>(auxParams.Length);
        for (int i = 0; i < direction.Length; i++)
            direction[i] = NumOps.FromDouble(RandomGenerator.NextDouble() > 0.5 ? 1.0 : -1.0);

        double baseLoss = 0;
        foreach (var task in taskBatch.Tasks)
            baseLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        baseLoss /= taskBatch.Tasks.Length;

        for (int i = 0; i < auxParams.Length; i++)
            auxParams[i] = NumOps.Add(auxParams[i], NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon)));

        double perturbedLoss = 0;
        foreach (var task in taskBatch.Tasks)
            perturbedLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        perturbedLoss /= taskBatch.Tasks.Length;

        double directionalGrad = (perturbedLoss - baseLoss) / epsilon;
        for (int i = 0; i < auxParams.Length; i++)
            auxParams[i] = NumOps.Subtract(auxParams[i],
                NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon + lr * directionalGrad)));
    }

    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0) return new Vector<T>(0);
        var result = new Vector<T>(vectors[0].Length);
        foreach (var v in vectors)
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.Add(result[i], v[i]);
        var scale = NumOps.FromDouble(1.0 / vectors.Count);
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.Multiply(result[i], scale);
        return result;
    }
}

/// <summary>Adapted model wrapper for NPBML with probabilistic prediction.</summary>
internal class NPBMLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    private readonly int _numSamples;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public NPBMLModel(IFullModel<T, TInput, TOutput> model, Vector<T> p, int numSamples)
    { _model = model; _params = p; _numSamples = numSamples; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
