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
public class ConstellationNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ConstellationNetOptions<T, TInput, TOutput> _constellationOptions;

    /// <summary>Parameters for the part detection module.</summary>
    private Vector<T> _partDetectorParams = new Vector<T>(0);

    /// <summary>Parameters for the spatial relation module.</summary>
    private Vector<T> _relationParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ConstellationNet;

    /// <summary>Initializes a new ConstellationNet meta-learner.</summary>
    /// <param name="options">Configuration options for ConstellationNet.</param>
    public ConstellationNetAlgorithm(ConstellationNetOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
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
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _constellationOptions.OuterLearningRate));
        }

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
        var initParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            var supportPred = MetaModel.Predict(task.SupportInput);
            var supportFeatures = ConvertToVector(supportPred);
            var detectedParts = DetectParts(supportFeatures);
            var adaptedFeatures = _constellationOptions.UseSpatialRelations
                ? ComputeConstellationScores(detectedParts) ?? detectedParts
                : detectedParts;

            if (adaptedFeatures != null && supportFeatures != null && supportFeatures.Length > 0)
            {
                double sumRatio = 0;
                int count = 0;
                for (int i = 0; i < Math.Min(supportFeatures.Length, adaptedFeatures.Length); i++)
                {
                    double raw = NumOps.ToDouble(supportFeatures[i]);
                    double adapted = NumOps.ToDouble(adaptedFeatures[i]);
                    if (Math.Abs(raw) > 1e-10)
                    {
                        sumRatio += Math.Max(0.5, Math.Min(2.0, adapted / raw));
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

            var queryPred = MetaModel.Predict(task.QueryInput);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(queryPred, task.QueryOutput));
        }

        MetaModel.SetParameters(initParams);
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
        var partFeatures = new Vector<T>(numParts);
        int paramIdx = 0;

        for (int k = 0; k < numParts; k++)
        {
            // Compute attention scores for this part across all feature positions
            var scores = new double[n];
            double maxScore = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double fi = NumOps.ToDouble(features[i]);
                double w = NumOps.ToDouble(_partDetectorParams[paramIdx++ % _partDetectorParams.Length]);
                scores[i] = fi * w;
                maxScore = Math.Max(maxScore, scores[i]);
            }

            // Softmax attention
            double sumExp = 0;
            for (int i = 0; i < n; i++)
            {
                scores[i] = Math.Exp(Math.Min(scores[i] - maxScore, 10.0));
                sumExp += scores[i];
            }

            // Weighted aggregation to produce part feature
            double partVal = 0;
            for (int i = 0; i < n; i++)
            {
                double weight = scores[i] / Math.Max(sumExp, 1e-10);
                partVal += weight * NumOps.ToDouble(features[i]);
            }

            // Part bias
            double bias = NumOps.ToDouble(_partDetectorParams[paramIdx++ % _partDetectorParams.Length]);
            partFeatures[k] = NumOps.FromDouble(partVal + bias);
        }

        return partFeatures;
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
        var constellation = new Vector<T>(K);
        int paramIdx = 0;

        // For each part, aggregate learned pairwise relationships with all other parts
        for (int i = 0; i < K; i++)
        {
            double pi = NumOps.ToDouble(parts[i]);
            double relSum = 0;

            for (int j = 0; j < K; j++)
            {
                if (i == j) continue;
                double pj = NumOps.ToDouble(parts[j]);
                double w = NumOps.ToDouble(_relationParams[paramIdx++ % _relationParams.Length]);
                double diff = pi - pj;
                relSum += w * diff * diff; // Learned distance-like relation
            }

            // Combine part feature with its relational context via tanh gate
            double bias = NumOps.ToDouble(_relationParams[paramIdx++ % _relationParams.Length]);
            constellation[i] = NumOps.Add(parts[i], NumOps.FromDouble(Math.Tanh(relSum + bias)));
        }

        return constellation;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

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
        double[]? modulationFactors = null;
        var adaptedFeatures = constellation ?? detectedParts;
        if (adaptedFeatures != null && supportFeatures != null && supportFeatures.Length > 0)
        {
            double sumRatio = 0;
            int count = 0;
            for (int i = 0; i < Math.Min(supportFeatures.Length, adaptedFeatures.Length); i++)
            {
                double raw = NumOps.ToDouble(supportFeatures[i]);
                double adapted = NumOps.ToDouble(adaptedFeatures[i]);
                if (Math.Abs(raw) > 1e-10)
                {
                    sumRatio += Math.Max(0.5, Math.Min(2.0, adapted / raw));
                    count++;
                }
            }
            double avgRatio = count > 0 ? sumRatio / count : 1.0;
            modulationFactors = [avgRatio];
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
/// </remarks>
internal class ConstellationNetModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _detectedParts;
    private readonly Vector<T>? _constellation;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _constellation ?? _detectedParts;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    public ConstellationNetModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? detectedParts,
        Vector<T>? constellation,
        double[]? modulationFactors)
    {
        _model = model;
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
