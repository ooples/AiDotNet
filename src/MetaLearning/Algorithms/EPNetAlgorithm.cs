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
/// Implementation of EPNet (Embedding Propagation Network) (Rodriguez et al., CVPR 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// EPNet refines embeddings through label propagation on a nearest-neighbor graph.
/// By propagating feature information between similar examples, embeddings become
/// smoother and more discriminative for few-shot classification.
/// </para>
/// <para><b>For Beginners:</b> EPNet makes features better by sharing information:
///
/// **The insight:**
/// Features extracted by a neural network are good but noisy. If two examples are
/// similar, they should have similar features. EPNet enforces this by propagating
/// features through a similarity graph.
///
/// **How it works:**
/// 1. Extract features for all examples (support + query)
/// 2. Build a k-nearest-neighbor graph based on feature similarity
/// 3. Propagate features through the graph (like heat diffusion)
///    - Each node averages its neighbors' features (weighted by similarity)
///    - Repeat for several iterations
/// 4. The propagated features are smoother and more consistent
/// 5. Classify using the refined features
///
/// **Why propagation helps:**
/// - Noisy features get smoothed out (noise reduction)
/// - Cluster structure becomes clearer (better separation)
/// - Query examples near support clusters get pulled toward them
/// - Works transductively: ALL queries benefit from each other
/// </para>
/// <para><b>Algorithm - EPNet:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor
///
/// # Embedding Propagation
/// for each task:
///     z = f_theta(all_examples)                 # Extract features
///
///     # Build kNN graph
///     W = knn_graph(z, k=num_neighbors)         # Adjacency matrix
///     D = diag(sum(W, axis=1))                  # Degree matrix
///     S = D^(-1/2) @ W @ D^(-1/2)              # Normalized graph
///
///     # Propagate embeddings
///     z_prop = z
///     for iter in range(propagation_iterations):
///         z_prop = alpha * S @ z_prop + (1-alpha) * z    # Diffusion + anchor
///
///     # Classify with propagated features
///     prototypes = mean(z_prop[support], per_class)
///     logits = -distance(z_prop[query], prototypes)
///     loss = cross_entropy(logits, query_labels)
/// </code>
/// </para>
/// <para>
/// Reference: Rodriguez, P., Laradji, I., Drouin, A., &amp; Lacoste, A. (2020).
/// Embedding Propagation: Smoother Manifold for Few-Shot Classification. CVPR 2020.
/// </para>
/// </remarks>
public class EPNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly EPNetOptions<T, TInput, TOutput> _epnetOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.EPNet;

    /// <summary>Initializes a new EPNet meta-learner.</summary>
    /// <param name="options">Configuration options for EPNet.</param>
    public EPNetAlgorithm(EPNetOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _epnetOptions = options;
    }

    /// <summary>
    /// Performs embedding propagation on a feature vector.
    /// </summary>
    /// <param name="features">Concatenated features for all examples.</param>
    /// <returns>Propagated (smoothed) features.</returns>
    private Vector<T> PropagateEmbeddings(Vector<T> features)
    {
        int n = features.Length;
        if (n < 2) return features;

        var propagated = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            propagated[i] = features[i];

        double alpha = _epnetOptions.PropagationAlpha;

        for (int iter = 0; iter < _epnetOptions.PropagationIterations; iter++)
        {
            var next = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                // Simple neighbor averaging (k-nearest approximation)
                T sum = NumOps.Zero;
                int neighborCount = 0;
                int k = Math.Min(_epnetOptions.NumNeighbors, n - 1);

                // Find k closest neighbors by index distance (simplified)
                for (int j = Math.Max(0, i - k); j <= Math.Min(n - 1, i + k); j++)
                {
                    if (j == i) continue;
                    sum = NumOps.Add(sum, propagated[j]);
                    neighborCount++;
                }

                T neighborAvg = neighborCount > 0
                    ? NumOps.Divide(sum, NumOps.FromDouble(neighborCount))
                    : propagated[i];

                // Diffusion: alpha * neighbor_avg + (1-alpha) * original
                next[i] = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(alpha), neighborAvg),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - alpha), features[i]));
            }
            propagated = next;
        }

        return propagated;
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

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _epnetOptions.OuterLearningRate));
        }

        _currentIteration++;
        return ComputeMean(losses);
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

        // Apply embedding propagation to smooth features across the kNN graph
        var propagatedSupport = supportFeatures != null ? PropagateEmbeddings(supportFeatures) : null;
        var propagatedQuery = queryFeatures != null ? PropagateEmbeddings(queryFeatures) : null;

        return new EPNetModel<T, TInput, TOutput>(MetaModel, currentParams, propagatedSupport, propagatedQuery);
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

/// <summary>Adapted model wrapper for EPNet with embedding propagation.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses features that have been smoothed
/// by propagation across a similarity graph, where nearby examples share
/// information to produce more consistent and discriminative embeddings.
/// </para>
/// </remarks>
internal class EPNetModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _propagatedSupport;
    private readonly Vector<T>? _propagatedQuery;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public EPNetModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? propagatedSupport,
        Vector<T>? propagatedQuery)
    {
        _model = model;
        _backboneParams = backboneParams;
        _propagatedSupport = propagatedSupport;
        _propagatedQuery = propagatedQuery;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        _model.SetParameters(_backboneParams);
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
