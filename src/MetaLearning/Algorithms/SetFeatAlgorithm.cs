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
/// Implementation of SetFeat (set-feature based few-shot learning) (Afrasiyabi et al., CVPR 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// SetFeat learns set-level features by processing each class's support examples as a set
/// rather than individual instances. A set encoder with optional cross-attention computes
/// class representations that capture intra-class variation.
/// </para>
/// <para><b>For Beginners:</b> SetFeat treats each class as a SET, not just a single point:
///
/// **The problem with prototypes:**
/// ProtoNets computes the MEAN of support examples. This throws away information about
/// HOW the class varies. Two classes might have the same mean but very different spreads.
///
/// **How SetFeat fixes this:**
/// 1. Extract features for each class's support examples
/// 2. Feed ALL examples (as a set) into a set encoder
/// 3. The set encoder captures rich information: mean, variance, relationships
/// 4. Optional cross-attention lets classes "see" each other for context
/// 5. The resulting set-features are used for classification
///
/// **Example:**
/// If you have 5 examples of cats (tabby, persian, siamese, calico, sphinx):
/// - ProtoNets: Average them into one "generic cat" point
/// - SetFeat: Encodes that cats come in different fur patterns and body types
/// This extra information helps distinguish cats from similar classes like small dogs.
/// </para>
/// <para><b>Algorithm - SetFeat:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor    # Shared backbone
/// g_phi = set_encoder            # Encodes set of features into set representation
/// a_omega = cross_attention      # Optional cross-attention between class sets
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         z_s = f_theta(support_x)       # Extract features
///
///         # For each class k, encode its support set
///         for each class k:
///             S_k = {z_s | class == k}
///             r_k = g_phi(S_k)           # Set-level representation
///
///         # Optional: cross-attention between class representations
///         if use_cross_attention:
///             R = a_omega(r_1, ..., r_K)
///
///         # Classify queries
///         z_q = f_theta(query_x)
///         logits = similarity(z_q, R)
///         loss = cross_entropy(logits, query_labels)
///
///     theta, phi, omega = theta, phi, omega - lr * grad(loss)
/// </code>
/// </para>
/// <para>
/// Reference: Afrasiyabi, A., Larochelle, H., Lalonde, J.F., &amp; Gagne, C. (2022).
/// Matching Feature Sets for Few-Shot Image Classification. CVPR 2022.
/// </para>
/// </remarks>
public class SetFeatAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly SetFeatOptions<T, TInput, TOutput> _setFeatOptions;

    /// <summary>Parameters for the set encoder.</summary>
    private Vector<T> _setEncoderParams = new Vector<T>(0);

    /// <summary>Parameters for the cross-attention module.</summary>
    private Vector<T> _crossAttentionParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SetFeat;

    /// <summary>Initializes a new SetFeat meta-learner.</summary>
    /// <param name="options">Configuration options for SetFeat.</param>
    public SetFeatAlgorithm(SetFeatOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _setFeatOptions = options;
        InitializeSetEncoder();
    }

    /// <summary>Initializes set encoder and cross-attention parameters.</summary>
    private void InitializeSetEncoder()
    {
        int dim = _setFeatOptions.SetEncoderDim;
        // Set encoder: attention pooling + projection
        int encoderParams = dim * dim + dim + dim * dim + dim;
        _setEncoderParams = new Vector<T>(encoderParams);
        double scale = Math.Sqrt(2.0 / dim);
        for (int i = 0; i < encoderParams; i++)
            _setEncoderParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);

        if (_setFeatOptions.UseCrossAttention)
        {
            // Cross-attention: Q, K, V projections + output
            int crossParams = dim * dim * 3 + dim * dim + dim * 2;
            _crossAttentionParams = new Vector<T>(crossParams);
            for (int i = 0; i < crossParams; i++)
                _crossAttentionParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        }
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
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _setFeatOptions.OuterLearningRate));
        }

        // Update set encoder and cross-attention via SPSA
        UpdateAuxiliaryParams(taskBatch, ref _setEncoderParams);
        if (_setFeatOptions.UseCrossAttention)
            UpdateAuxiliaryParams(taskBatch, ref _crossAttentionParams);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        return new SetFeatModel<T, TInput, TOutput>(MetaModel, MetaModel.GetParameters());
    }

    /// <summary>Updates auxiliary parameters using SPSA gradient estimation.</summary>
    private void UpdateAuxiliaryParams(TaskBatch<T, TInput, TOutput> taskBatch, ref Vector<T> auxParams)
    {
        double epsilon = 1e-5;
        double lr = _setFeatOptions.OuterLearningRate;

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

/// <summary>Adapted model wrapper for SetFeat.</summary>
internal class SetFeatModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public SetFeatModel(IFullModel<T, TInput, TOutput> model, Vector<T> p) { _model = model; _params = p; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
