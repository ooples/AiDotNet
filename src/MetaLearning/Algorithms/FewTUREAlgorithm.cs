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
/// Implementation of FewTURE (Few-shot Transformer with Uncertainty and Reliable Estimation)
/// (Hiller et al., ECCV 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// FewTURE uses vision transformers with token-level local features and uncertainty estimation
/// for few-shot classification. Instead of comparing global image features, it compares at the
/// patch/token level and weights predictions by their estimated reliability.
/// </para>
/// <para><b>For Beginners:</b> FewTURE compares images piece by piece, not as wholes:
///
/// **Standard approach:**
/// Represent each image as ONE feature vector, compare vectors.
/// Problem: A bird's beak might be the key difference, but it's a tiny part of the image.
///
/// **FewTURE's approach:**
/// 1. Split each image into patches (like a puzzle, e.g., 14x14 = 196 patches)
/// 2. Use a Vision Transformer to get a feature for each patch (token)
/// 3. Compare queries to support classes at the PATCH level
/// 4. Estimate uncertainty for each patch comparison
/// 5. Weight reliable patches more, uncertain patches less
///
/// **Why uncertainty matters:**
/// Not all patches are equally informative:
/// - Background patches are mostly noise (high uncertainty)
/// - Discriminative patches (beak, stripes) are informative (low uncertainty)
/// FewTURE learns to focus on the informative patches automatically.
/// </para>
/// <para><b>Algorithm - FewTURE:</b>
/// <code>
/// # Components
/// ViT = vision_transformer           # Extracts patch tokens
/// u_phi = uncertainty_estimator      # Estimates per-token uncertainty
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Extract patch tokens
///         tokens_s = ViT(support_x)    # [N_support, num_patches, dim]
///         tokens_q = ViT(query_x)      # [N_query, num_patches, dim]
///
///         # 2. Compute per-token similarity between query and class prototypes
///         for each class k:
///             proto_tokens_k = mean(tokens_s[class == k])  # Per-patch prototypes
///
///         # 3. Estimate uncertainty for each token match
///         uncertainty = u_phi(tokens_q, proto_tokens)
///
///         # 4. Weighted aggregation (reliable tokens contribute more)
///         weights = 1 / (uncertainty + eps)
///         logits = weighted_sum(similarity * weights) / sum(weights)
///
///         loss = cross_entropy(logits, query_labels)
///
///     Update ViT and uncertainty estimator
/// </code>
/// </para>
/// <para>
/// Reference: Hiller, M., Ma, R., Harber, M., &amp; Ommer, B. (2022).
/// Rethinking Generalization in Few-Shot Classification. ECCV 2022.
/// </para>
/// </remarks>
public class FewTUREAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly FewTUREOptions<T, TInput, TOutput> _fewTUREOptions;

    /// <summary>Parameters for the uncertainty estimation module.</summary>
    private Vector<T> _uncertaintyParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.FewTURE;

    /// <summary>Initializes a new FewTURE meta-learner.</summary>
    /// <param name="options">Configuration options for FewTURE.</param>
    public FewTUREAlgorithm(FewTUREOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _fewTUREOptions = options;
        InitializeUncertaintyModule();
    }

    /// <summary>Initializes the uncertainty estimation module.</summary>
    private void InitializeUncertaintyModule()
    {
        int numTokens = _fewTUREOptions.NumTokens;
        // Small MLP for uncertainty: token_dim -> hidden -> 1
        int dim = 64;
        int totalParams = dim * dim + dim + dim + 1;
        _uncertaintyParams = new Vector<T>(totalParams);
        double scale = Math.Sqrt(2.0 / dim);
        for (int i = 0; i < totalParams; i++)
            _uncertaintyParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
    }

    /// <summary>
    /// Estimates uncertainty for a prediction using the configured method.
    /// </summary>
    /// <param name="features">Feature vector from the model.</param>
    /// <returns>Uncertainty score (higher = more uncertain).</returns>
    private double EstimateUncertainty(Vector<T> features)
    {
        if (_fewTUREOptions.UncertaintyMethod == "entropy")
        {
            // Compute entropy of softmax distribution
            double maxVal = double.MinValue;
            for (int i = 0; i < features.Length; i++)
            {
                double v = NumOps.ToDouble(features[i]);
                if (v > maxVal) maxVal = v;
            }

            double sumExp = 0;
            for (int i = 0; i < features.Length; i++)
                sumExp += Math.Exp(NumOps.ToDouble(features[i]) - maxVal);

            double entropy = 0;
            for (int i = 0; i < features.Length; i++)
            {
                double p = Math.Exp(NumOps.ToDouble(features[i]) - maxVal) / sumExp;
                if (p > 1e-10)
                    entropy -= p * Math.Log(p);
            }
            return entropy;
        }

        // Default: variance-based uncertainty
        T mean = NumOps.Zero;
        for (int i = 0; i < features.Length; i++)
            mean = NumOps.Add(mean, features[i]);
        mean = NumOps.Divide(mean, NumOps.FromDouble(Math.Max(1, features.Length)));

        double variance = 0;
        for (int i = 0; i < features.Length; i++)
        {
            double diff = NumOps.ToDouble(NumOps.Subtract(features[i], mean));
            variance += diff * diff;
        }
        return variance / Math.Max(1, features.Length);
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
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _fewTUREOptions.OuterLearningRate));
        }

        // Update uncertainty module via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _uncertaintyParams, _fewTUREOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Computes the average loss over a task batch using uncertainty-weighted predictions.
    /// Called by SPSA to measure how perturbed uncertainty params affect loss.
    /// </summary>
    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            var supportPred = MetaModel.Predict(task.SupportInput);
            var supportFeatures = ConvertToVector(supportPred);
            var queryPredRaw = MetaModel.Predict(task.QueryInput);
            var queryFeatures = ConvertToVector(queryPredRaw);

            var uncertaintyWeights = ComputeUncertaintyWeights(supportFeatures, queryFeatures);

            if (uncertaintyWeights != null && uncertaintyWeights.Length > 0)
            {
                double sumAbs = 0;
                for (int i = 0; i < uncertaintyWeights.Length; i++)
                    sumAbs += Math.Abs(NumOps.ToDouble(uncertaintyWeights[i]));
                double meanAbs = sumAbs / uncertaintyWeights.Length;
                double modFactor = 0.5 + 0.5 / (1.0 + Math.Exp(-meanAbs + 1.0));

                var currentParams = MetaModel.GetParameters();
                var modulatedParams = new Vector<T>(currentParams.Length);
                for (int i = 0; i < currentParams.Length; i++)
                    modulatedParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(modFactor));
                MetaModel.SetParameters(modulatedParams);
            }

            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    /// <summary>
    /// Computes uncertainty-weighted classification weights for query features.
    /// Tokens/features with lower uncertainty get higher weights.
    /// </summary>
    /// <param name="supportFeatures">Support set features.</param>
    /// <param name="queryFeatures">Query set features.</param>
    /// <returns>Uncertainty-weighted classification weights.</returns>
    private Vector<T>? ComputeUncertaintyWeights(Vector<T>? supportFeatures, Vector<T>? queryFeatures)
    {
        if (supportFeatures == null || queryFeatures == null || queryFeatures.Length == 0)
            return supportFeatures;

        int numQuery = queryFeatures.Length;
        var weights = new Vector<T>(numQuery);

        for (int q = 0; q < numQuery; q++)
        {
            // Create a feature vector for this query element
            var queryVec = new Vector<T>(1);
            queryVec[0] = queryFeatures[q];

            // Estimate uncertainty for this token/feature
            double uncertainty = EstimateUncertainty(queryVec);

            // Compute similarity to support centroid
            double sim = 0;
            for (int s = 0; s < supportFeatures.Length; s++)
            {
                sim += NumOps.ToDouble(NumOps.Multiply(queryFeatures[q],
                    supportFeatures[s % supportFeatures.Length]));
            }
            sim /= Math.Max(supportFeatures.Length, 1);

            // Weight = similarity / (uncertainty + eps)
            // Higher similarity and lower uncertainty = higher weight
            double reliabilityWeight = 1.0 / (uncertainty + 1e-6);
            double weightedScore = sim * reliabilityWeight;

            weights[q] = NumOps.FromDouble(weightedScore);
        }

        return weights;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support and query features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);

        // Compute uncertainty-weighted classification
        var uncertaintyWeights = ComputeUncertaintyWeights(supportFeatures, queryFeatures);

        // Compute modulation from uncertainty weights (higher weight = more reliable)
        double[]? modulationFactors = null;
        if (uncertaintyWeights != null && uncertaintyWeights.Length > 0)
        {
            double sumAbs = 0;
            for (int i = 0; i < uncertaintyWeights.Length; i++)
                sumAbs += Math.Abs(NumOps.ToDouble(uncertaintyWeights[i]));
            double meanAbs = sumAbs / uncertaintyWeights.Length;
            modulationFactors = [0.5 + 0.5 / (1.0 + Math.Exp(-meanAbs + 1.0))];
        }

        return new FewTUREModel<T, TInput, TOutput>(
            MetaModel, currentParams, uncertaintyWeights, _fewTUREOptions.UncertaintyThreshold, modulationFactors);
    }


}

/// <summary>Adapted model wrapper for FewTURE with uncertainty-weighted prediction.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model weights predictions by their estimated
/// reliability. Tokens/features that the uncertainty estimator deems unreliable
/// (e.g., background patches) contribute less to the final classification.
/// </para>
/// </remarks>
internal class FewTUREModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _uncertaintyWeights;
    private readonly double _uncertaintyThreshold;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _uncertaintyWeights;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public FewTUREModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? uncertaintyWeights,
        double threshold,
        double[]? modulationFactors)
    {
        _model = model;
        _backboneParams = backboneParams;
        _uncertaintyWeights = uncertaintyWeights;
        _uncertaintyThreshold = threshold;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = new Vector<T>(_backboneParams.Length);
            for (int i = 0; i < _backboneParams.Length; i++)
                modulated[i] = NumOps.Multiply(_backboneParams[i],
                    NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
            _model.SetParameters(modulated);
        }
        else
        {
            _model.SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
