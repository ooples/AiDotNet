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
/// Implementation of ConstellationNet (structured part-based few-shot learning) (Xu et al., ICLR 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// ConstellationNet detects discriminative parts in examples and models their spatial
/// relationships as "constellations." Classification is performed by matching the
/// constellation structure between queries and support examples.
/// </para>
/// <para><b>For Beginners:</b> ConstellationNet recognizes objects by their PARTS and ARRANGEMENT:
///
/// **The insight:**
/// A face isn't just "eyes + nose + mouth" - it's the specific ARRANGEMENT of these parts.
/// Similarly, a bird isn't just "beak + wing + tail" - it's how they're positioned.
/// ConstellationNet captures both the parts AND their spatial arrangement.
///
/// **How it works:**
/// 1. **Part detection:** For each example, detect K discriminative parts
///    - Each part has a feature vector (what it looks like)
///    - Each part has a position (where it is)
/// 2. **Constellation formation:** Model relationships between parts
///    - Pairwise spatial relationships between all parts
///    - Creates a "constellation" = graph of parts + spatial edges
/// 3. **Constellation matching:** Compare query and support constellations
///    - Match parts between query and support (feature similarity)
///    - Compare spatial arrangements (structural similarity)
///    - Combined score determines classification
///
/// **Why constellations help:**
/// Two classes of birds might share similar colors but differ in proportions.
/// A constellation captures "beak-to-eye distance" and "wing-to-tail ratio" naturally.
/// </para>
/// <para><b>Algorithm - ConstellationNet:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor         # Backbone
/// d_phi = part_detector               # Detects K parts per example
/// r_psi = relation_network            # Models spatial relationships
///
/// # Meta-training
/// for each task T_i:
///     # 1. Extract features
///     features = f_theta(all_examples)
///
///     # 2. Detect parts
///     for each example:
///         parts = d_phi(features)             # K parts, each with feature + position
///
///     # 3. Build constellations (spatial relationships)
///     for each example:
///         constellation = r_psi(parts)        # Pairwise part relationships
///
///     # 4. Compare query constellations to support constellations
///     for each query:
///         for each class k:
///             # Part matching (features)
///             part_score = match_parts(query_parts, class_k_parts)
///             # Structural matching (spatial arrangement)
///             struct_score = match_structure(query_constellation, class_k_constellation)
///             score_k = combine(part_score, struct_score)
///
///     loss = cross_entropy(scores, query_labels)
///
///     theta, phi, psi = theta, phi, psi - lr * grad(loss)
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation replaced entirely — it named the WRONG PAPER, not just a wrong URL. The recorded title
// "Attentive Weights Generation for Few Shot Learning via Information Maximization" (AWGIM) is a real
// but different paper, and the recorded arXiv id 2003.03313 is "Categories of orthogonality spaces",
// unrelated to either. This class implements ConstellationNet — cell feature clustering with a dense
// part representation, whose relationships are then modeled by attention — which is what its name says
// and what the code does. ConstellationNet has no arXiv preprint, so the ICLR 2021 OpenReview record is
// the canonical reference. AWGIM itself is AWGIMAlgorithm (#1929).
[ResearchPaper("Attentional Constellation Nets for Few-Shot Learning",
    "https://openreview.net/forum?id=vujTf_I8Kmc",
    Year = 2021,
    Authors = "Weijian Xu, Yifan Xu, Huaijin Wang, Zhuowen Tu")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ConstellationNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly ConstellationNetOptions<T, TInput, TOutput> _constellationOptions;

    /// <summary>Parameters for the part detection module.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _partDetectorParams = new Vector<T>(0);

    /// <summary>Parameters for the spatial relation module.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _relationParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ConstellationNet;

    /// <summary>Initializes a new ConstellationNet meta-learner.</summary>
    /// <param name="options">Configuration options for ConstellationNet.</param>
    public ConstellationNetAlgorithm(ConstellationNetOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _constellationOptions = options;
        InitializePartDetector();
    }

    /// <summary>Initializes part detector and relation module parameters.</summary>
    private void InitializePartDetector()
    {
        int numParts = _constellationOptions.NumParts;
        int partDim = _constellationOptions.PartFeatureDim;

        // Part detector: feature_dim -> numParts attention scores + part features
        int detectorSize = partDim * numParts + numParts;
        _partDetectorParams = new Vector<T>(detectorSize);

        // Relation module: pairwise spatial relationships
        int relationSize = _constellationOptions.UseSpatialRelations
            ? numParts * numParts * partDim + partDim
            : 0;
        _relationParams = new Vector<T>(Math.Max(1, relationSize));

        double scale = Math.Sqrt(2.0 / partDim);
        for (int i = 0; i < _partDetectorParams.Length; i++)
            _partDetectorParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        for (int i = 0; i < _relationParams.Length; i++)
            _relationParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
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
        ApplyOuterUpdate(initParams, metaGradients, _constellationOptions.OuterLearningRate);

        // Update part detector and relation module via multi-sample SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _partDetectorParams, _constellationOptions.OuterLearningRate, ComputeAuxLoss);
        if (_constellationOptions.UseSpatialRelations)
            UpdateAuxiliaryParamsSPSA(taskBatch, ref _relationParams, _constellationOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Computes the average loss over a task batch using part detection and constellation scoring.
    /// Called by SPSA to measure how perturbed part/relation params affect loss.
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
            var detectedParts = DetectParts(supportFeatures);
            var adaptedFeatures = _constellationOptions.UseSpatialRelations
                ? ComputeConstellationScores(detectedParts) ?? detectedParts
                : detectedParts;

            if (adaptedFeatures != null && supportFeatures != null && supportFeatures.Length > 0)
            {
                ParameterModulation.ScaleByMeanClampedRatio(Engine, ParamModel, supportFeatures, adaptedFeatures, 0.5, 2.0, 1e-10);
            }

            var queryPred = MetaModel.Predict(task.QueryInput);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(queryPred, task.QueryOutput));
        }

        ParamModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    /// <summary>
    /// Detects K discriminative parts from features using attention-based selection.
    /// Each part attends over all feature positions and produces a weighted aggregation.
    /// </summary>
    /// <param name="features">Raw features from the backbone.</param>
    /// <returns>Vector of K part features, one per detected part.</returns>
    private Vector<T>? DetectParts(Vector<T>? features)
    {
        if (features == null || features.Length == 0)
            return features;

        int numParts = _constellationOptions.NumParts;
        int n = features.Length;
        if (numParts <= 0) return new Vector<T>(0);

        // Each part reads n attention weights then one bias from the detector parameters, cyclically:
        // row k of the [numParts, n + 1] table is part k's weights followed by its bias.
        var table = ParameterModulation.Table(Engine, _partDetectorParams, numParts, n + 1);
        var weights = ParameterModulation.Columns(Engine, table, 0, n);
        var bias = ParameterModulation.Column(Engine, table, n);

        // Attention scores s[k, i] = f_i * w[k, i], then a max-shifted softmax over positions.
        var f = Engine.Reshape(Tensor<T>.FromVector(features), new[] { 1, n });
        var scores = Engine.TensorMultiply(weights, Engine.TensorBroadcastTo(f, new[] { numParts, n }));
        var maxScore = Engine.TensorBroadcastTo(Engine.ReduceMax(scores, new[] { 1 }, true), new[] { numParts, n });
        var exps = Engine.TensorExp(Engine.TensorMin(Engine.TensorSubtract(scores, maxScore), NumOps.FromDouble(10.0)));
        var sumExp = Engine.TensorMax(Engine.ReduceSum(exps, new[] { 1 }, false), NumOps.FromDouble(1e-10));

        // Part feature = attention-weighted aggregate of the features plus the part bias.
        var weighted = Engine.Reshape(
            Engine.TensorMatMul(exps, Engine.Reshape(f, new[] { n, 1 })), new[] { numParts });
        return Engine.Add(Engine.TensorDivide(weighted, sumExp).ToVector(), bias);
    }

    /// <summary>
    /// Computes constellation scores: pairwise spatial relationships between detected parts.
    /// Each part's representation is enriched with learned relational context from all other parts.
    /// </summary>
    /// <param name="parts">Detected part features.</param>
    /// <returns>Parts enriched with pairwise relational context.</returns>
    private Vector<T>? ComputeConstellationScores(Vector<T>? parts)
    {
        if (parts == null || parts.Length < 2 || !_constellationOptions.UseSpatialRelations)
            return parts;

        int K = parts.Length;

        // Each part i reads K - 1 relation weights (for j != i, in order) then one bias, cyclically: row i of the
        // [K, K] table. Spreading row i's weights over columns j != i gives the relation matrix W, zero on the
        // diagonal.
        var table = ParameterModulation.Table(Engine, _relationParams, K, K);
        var spread = new int[K * K];
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                spread[i * K + j] = i * K + (j < i ? j : Math.Max(j - 1, 0));
            }
        }

        var diagonal = Engine.Reshape(Engine.TensorEye<T>(K), new[] { K * K });
        var relation = Engine.TensorMultiply(
            Engine.TensorIndexSelect(Engine.Reshape(table, new[] { K * K }), new Tensor<int>(spread, new[] { K * K }), 0),
            Engine.TensorSubtract(Engine.TensorOnesLike(diagonal), diagonal));
        var bias = ParameterModulation.Column(Engine, table, K - 1);

        // relSum_i = sum_j W[i, j] * (p_i - p_j)^2 (a learned distance-like relation), then a tanh gate.
        var p = Engine.Reshape(Tensor<T>.FromVector(parts), new[] { K, 1 });
        var diff = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(p, new[] { K, K }),
            Engine.TensorBroadcastTo(Engine.Reshape(p, new[] { 1, K }), new[] { K, K }));
        var relSum = Engine.ReduceSum(
            Engine.TensorMultiply(Engine.Reshape(relation, new[] { K, K }), Engine.TensorSquare(diff)), new[] { 1 }, false);
        return Engine.Add(parts, Engine.Tanh(Engine.Add(relSum.ToVector(), bias)));
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = ParamModel.GetParameters();

        // Extract support features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Detect discriminative parts from support features
        var detectedParts = DetectParts(supportFeatures);

        // Compute constellation (pairwise spatial relationships between parts)
        Vector<T>? constellation = null;
        if (_constellationOptions.UseSpatialRelations)
            constellation = ComputeConstellationScores(detectedParts);

        // Compute feature-delta modulation from part detection + constellation
        Vector<T>? modulationFactors = null;
        var adaptedFeatures = constellation ?? detectedParts;
        if (adaptedFeatures != null && supportFeatures != null && supportFeatures.Length > 0)
        {
            modulationFactors = ParameterModulation.MeanClampedRatio(Engine, supportFeatures, adaptedFeatures, 0.5, 2.0, 1e-10)
                ?? new Vector<T>(new[] { NumOps.One });
        }

        return new ConstellationNetModel<T, TInput, TOutput>(
            MetaModel, currentParams, detectedParts, constellation, modulationFactors);
    }

}

/// <summary>Adapted model wrapper for ConstellationNet with part-based constellation matching.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model stores detected parts and their constellation
/// (spatial relationship structure). During classification, query examples can be compared
/// against support classes not just by overall similarity, but by matching specific parts
/// and checking whether their spatial arrangement is consistent.
/// </para>
/// <para><b>Thread Safety:</b> This class is NOT thread-safe. Predict mutates the shared
/// model's parameters via SetParameters. Callers must ensure single-threaded access.</para>
/// </remarks>
internal class ConstellationNetModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _detectedParts;
    private readonly Vector<T>? _constellation;
    private readonly Vector<T>? _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _constellation ?? _detectedParts;

    /// <inheritdoc/>
    public Vector<T>? ParameterModulationFactors => _modulationFactors;

    public ConstellationNetModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? detectedParts,
        Vector<T>? constellation,
        Vector<T>? modulationFactors)
    {
        // Predict pushes the adapted parameters into this model, so it is a copy: holding the meta-learner's
        // MetaModel would let every prediction overwrite the meta-model.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).Clone();
        _backboneParams = backboneParams;
        _detectedParts = detectedParts;
        _constellation = constellation;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulatedParams = ParameterModulation.Apply(AiDotNetEngine.Current, _backboneParams, _modulationFactors);
            InterfaceGuard.Parameterizable(_model).SetParameters(modulatedParams);
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
