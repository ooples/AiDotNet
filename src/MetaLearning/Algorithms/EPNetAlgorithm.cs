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
    /// Performs embedding propagation on a feature vector using a proper kNN similarity graph.
    /// Builds pairwise distance matrix, selects k-nearest neighbors by feature similarity,
    /// constructs a normalized adjacency matrix, and iteratively diffuses features.
    /// </summary>
    /// <param name="features">Concatenated features for all examples.</param>
    /// <returns>Propagated (smoothed) features.</returns>
    private Vector<T> PropagateEmbeddings(Vector<T> features)
    {
        int n = features.Length;
        if (n < 2) return features;

        int k = Math.Min(_epnetOptions.NumNeighbors, n - 1);
        double alpha = _epnetOptions.PropagationAlpha;

        // Step 1: Compute pairwise squared distances between all feature elements
        var distances = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            double fi = NumOps.ToDouble(features[i]);
            for (int j = i + 1; j < n; j++)
            {
                double fj = NumOps.ToDouble(features[j]);
                double diff = fi - fj;
                double dist = diff * diff;
                distances[i * n + j] = dist;
                distances[j * n + i] = dist;
            }
        }

        // Step 2: Build kNN adjacency matrix W using feature similarity
        // For each node, find k nearest neighbors by distance and compute RBF weights
        var W = new double[n * n];

        // Compute median distance for adaptive RBF bandwidth
        var allDists = new List<double>(n * (n - 1) / 2);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                allDists.Add(distances[i * n + j]);
        allDists.Sort();
        double medianDist = allDists.Count > 0 ? allDists[allDists.Count / 2] : 1.0;
        double sigma2 = Math.Max(medianDist, 1e-10); // RBF bandwidth = median distance

        for (int i = 0; i < n; i++)
        {
            // Collect distances to all other nodes
            var neighborDists = new (int idx, double dist)[n - 1];
            int count = 0;
            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                neighborDists[count++] = (j, distances[i * n + j]);
            }

            // Partial sort: find k smallest distances
            Array.Sort(neighborDists, 0, count, Comparer<(int idx, double dist)>.Create(
                (a, b) => a.dist.CompareTo(b.dist)));

            // Set RBF weights for k nearest neighbors (symmetric)
            int numNeighbors = Math.Min(k, count);
            for (int ni = 0; ni < numNeighbors; ni++)
            {
                int j = neighborDists[ni].idx;
                double rbfWeight = Math.Exp(-neighborDists[ni].dist / (2.0 * sigma2));
                W[i * n + j] = rbfWeight;
                W[j * n + i] = rbfWeight; // Symmetrize
            }
        }

        // Step 3: Compute degree matrix D and normalized adjacency S = D^(-1/2) W D^(-1/2)
        var degree = new double[n];
        for (int i = 0; i < n; i++)
        {
            double d = 0;
            for (int j = 0; j < n; j++)
                d += W[i * n + j];
            degree[i] = d;
        }

        var S = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            double di = degree[i] > 1e-10 ? 1.0 / Math.Sqrt(degree[i]) : 0;
            for (int j = 0; j < n; j++)
            {
                double dj = degree[j] > 1e-10 ? 1.0 / Math.Sqrt(degree[j]) : 0;
                S[i * n + j] = di * W[i * n + j] * dj;
            }
        }

        // Step 4: Iterative feature propagation: z_prop = alpha * S @ z_prop + (1-alpha) * z_original
        var propagated = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            propagated[i] = features[i];

        for (int iter = 0; iter < _epnetOptions.PropagationIterations; iter++)
        {
            var next = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                // Weighted neighbor aggregation: sum_j S[i,j] * z_prop[j]
                double aggregated = 0;
                for (int j = 0; j < n; j++)
                    aggregated += S[i * n + j] * NumOps.ToDouble(propagated[j]);

                // Diffusion with anchor to original features
                next[i] = NumOps.FromDouble(
                    alpha * aggregated + (1.0 - alpha) * NumOps.ToDouble(features[i]));
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

        // Transductive propagation: concatenate support + query onto a single graph,
        // propagate features jointly, then split back. This is the core EPNet insight:
        // query features benefit from support features and vice versa.
        Vector<T>? propagatedSupport = null;
        Vector<T>? propagatedQuery = null;

        int supportLen = supportFeatures?.Length ?? 0;
        int queryLen = queryFeatures?.Length ?? 0;

        if (supportLen > 0 || queryLen > 0)
        {
            // Concatenate support + query into a single feature vector
            var combined = new Vector<T>(supportLen + queryLen);
            for (int i = 0; i < supportLen; i++)
                combined[i] = supportFeatures is not null ? supportFeatures[i] : NumOps.Zero;
            for (int i = 0; i < queryLen; i++)
                combined[supportLen + i] = queryFeatures is not null ? queryFeatures[i] : NumOps.Zero;

            // Propagate on the joint graph
            var propagated = PropagateEmbeddings(combined);

            // Split back into support and query portions
            if (supportLen > 0)
            {
                propagatedSupport = new Vector<T>(supportLen);
                for (int i = 0; i < supportLen; i++)
                    propagatedSupport[i] = propagated[i];
            }
            if (queryLen > 0)
            {
                propagatedQuery = new Vector<T>(queryLen);
                for (int i = 0; i < queryLen; i++)
                    propagatedQuery[i] = propagated[supportLen + i];
            }
        }

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
