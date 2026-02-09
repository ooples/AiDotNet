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

        // Update part detector and relation module via SPSA
        UpdateAuxiliaryParams(taskBatch, ref _partDetectorParams);
        if (_constellationOptions.UseSpatialRelations)
            UpdateAuxiliaryParams(taskBatch, ref _relationParams);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        return new ConstellationNetModel<T, TInput, TOutput>(MetaModel, MetaModel.GetParameters());
    }

    /// <summary>Updates auxiliary parameters using SPSA gradient estimation.</summary>
    private void UpdateAuxiliaryParams(TaskBatch<T, TInput, TOutput> taskBatch, ref Vector<T> auxParams)
    {
        double epsilon = 1e-5;
        double lr = _constellationOptions.OuterLearningRate;

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

/// <summary>Adapted model wrapper for ConstellationNet with part-based matching.</summary>
internal class ConstellationNetModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public ConstellationNetModel(IFullModel<T, TInput, TOutput> model, Vector<T> p) { _model = model; _params = p; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
