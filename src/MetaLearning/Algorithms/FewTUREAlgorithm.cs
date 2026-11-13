using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Rethinking Generalization in Few-Shot Classification",
    "https://arxiv.org/abs/2206.07267",
    Year = 2022,
    Authors = "Markus Hiller, Rongkai Ma, Mehrtash Harber, Bjorn Ommer")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class FewTUREAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly FewTUREOptions<T, TInput, TOutput> _fewTUREOptions;

    /// <summary>Parameters for the uncertainty estimation module.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _uncertaintyParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.FewTURE;

    /// <summary>Initializes a new FewTURE meta-learner.</summary>
    /// <param name="options">Configuration options for FewTURE.</param>
    public FewTUREAlgorithm(FewTUREOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
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
    /// The uncertainty MLP's per-element parameters as an [n, 4] table: element i reads w1, b1, w2, b2 in order
    /// from the learned parameters, and once they run out every read takes its default (0.01, 0, 0.01, 0).
    /// </summary>
    private Tensor<T> UncertaintyParameterTable(int n)
    {
        T small = NumOps.FromDouble(0.01);
        return ParameterModulation.DefaultedTable(
            Engine, _uncertaintyParams, n, new Vector<T>(new[] { small, NumOps.Zero, small, NumOps.Zero }));
    }

    /// <summary>
    /// Per-element uncertainty scores sigmoid(w2 * ReLU(w1 * x + b1) + b2), with element i using row i of the
    /// [n, 4] parameter table.
    /// </summary>
    private Vector<T> UncertaintyScores(Vector<T> features, Tensor<T> table)
    {
        var hidden = Engine.TensorMax(Tensor<T>.FromVector(Engine.Add(
            Engine.Multiply(ParameterModulation.Column(Engine, table, 0), features),
            ParameterModulation.Column(Engine, table, 1))), NumOps.Zero).ToVector();
        return Engine.Sigmoid(Engine.Add(
            Engine.Multiply(ParameterModulation.Column(Engine, table, 2), hidden),
            ParameterModulation.Column(Engine, table, 3)));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        ApplyOuterUpdate(initParams, metaGradients, _fewTUREOptions.OuterLearningRate);

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
        var initParams = ParamModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);

            var supportPred = MetaModel.Predict(task.SupportInput);
            var supportFeatures = ConvertToVector(supportPred);
            var queryPredRaw = MetaModel.Predict(task.QueryInput);
            var queryFeatures = ConvertToVector(queryPredRaw);

            var uncertaintyWeights = ComputeUncertaintyWeights(supportFeatures, queryFeatures);

            if (uncertaintyWeights != null && uncertaintyWeights.Length > 0)
            {
                // 0.5 + 0.5 * sigmoid(mean|w| - 1) scales the backbone.
                ParamModel.SetParameters(ParameterModulation.Apply(
                    Engine, ParamModel.GetParameters(), ParameterModulation.MeanAbsGate(Engine, uncertaintyWeights)));
            }

            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        ParamModel.SetParameters(initParams);
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

        // Each query element is scored on its own, so every one uses the first parameter row.
        var firstRow = UncertaintyParameterTable(1);
        var uncertainty = UncertaintyScores(
            queryFeatures, Engine.TensorBroadcastTo(firstRow, new[] { numQuery, 4 }));

        // Similarity to the support centroid: q * mean(support), the mean accumulated in double as the scalar loop
        // did (a Half running sum over the support features stops growing past 2048)
        T supportMean = NumOps.FromDouble(WideReduction.Mean(Engine, supportFeatures));
        var sim = Engine.Multiply(queryFeatures, supportMean);

        // Weight = similarity / (uncertainty + eps)
        // Higher similarity and lower uncertainty = higher weight
        return Engine.Divide(sim, Engine.Add(uncertainty, Engine.Fill(numQuery, NumOps.FromDouble(1e-6))));
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = ParamModel.GetParameters();

        // Extract support and query features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);

        // Compute uncertainty-weighted classification
        var uncertaintyWeights = ComputeUncertaintyWeights(supportFeatures, queryFeatures);

        // Compute modulation from uncertainty weights (higher weight = more reliable)
        Vector<T>? modulationFactors = null;
        if (uncertaintyWeights != null && uncertaintyWeights.Length > 0)
        {
            modulationFactors = ParameterModulation.MeanAbsGate(Engine, uncertaintyWeights);
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
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _uncertaintyWeights;
    private readonly double _uncertaintyThreshold;
    // Not readonly: a readonly Vector<T> field is not persisted by the generated model state, and these
    // factors belong in the adapted model's checkpoint as the double[] they replace did.
    private Vector<T>? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _uncertaintyWeights;

    /// <inheritdoc/>
    public Vector<T>? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public FewTUREModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? uncertaintyWeights,
        double threshold,
        Vector<T>? modulationFactors)
    {
        // Predict pushes the adapted parameters into this model, so it is a copy: holding the meta-learner's
        // MetaModel would let every prediction overwrite the meta-model.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).Clone();
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
            var modulated = ParameterModulation.Apply(AiDotNetEngine.Current, _backboneParams, _modulationFactors);
            InterfaceGuard.Parameterizable(_model).SetParameters(modulated);
        }
        else
        {
            InterfaceGuard.Parameterizable(_model).SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
