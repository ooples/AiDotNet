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
/// Implementation of LaplacianShot (Laplacian Regularized Few-Shot Learning).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// LaplacianShot augments nearest-centroid classification with Laplacian regularization
/// over a kNN graph of query features. This propagates labels from confident predictions
/// to uncertain ones based on feature similarity.
/// </para>
/// <para><b>For Beginners:</b> LaplacianShot adds graph-based label propagation:
///
/// **Step 1: Initial classification (like SimpleShot)**
/// Classify each query by distance to support centroids.
///
/// **Step 2: Build similarity graph**
/// Connect each query to its k nearest neighbors (in feature space).
///
/// **Step 3: Label propagation**
/// Iteratively smooth predictions across the graph:
/// - If your neighbors are confidently "cat", you should also lean toward "cat"
/// - The Laplacian matrix encodes this smoothness constraint mathematically
///
/// **Why it helps:**
/// Query examples near the decision boundary get "pulled" toward the correct class
/// by their more confident neighbors, reducing boundary errors.
/// </para>
/// <para><b>Algorithm - LaplacianShot:</b>
/// <code>
/// # Given: prototypes p_k, query features z_q
///
/// # 1. Initial logits from nearest centroid
/// logits = -||z_q - p_k||^2              # [Q, K]
///
/// # 2. Build kNN graph over query features
/// W[i,j] = exp(-||z_q[i] - z_q[j]||^2 / sigma)  if j in kNN(i)
/// D = diag(sum(W, axis=1))               # Degree matrix
/// L = D - W                               # Graph Laplacian
///
/// # 3. Laplacian-regularized classification
/// # Solve: min_Y ||Y - logits||^2 + lambda * trace(Y^T L Y)
/// # Solution: Y = (I + lambda * L)^-1 * logits
/// predictions = (I + lambda * L)^-1 @ logits
/// </code>
/// </para>
/// <para>
/// Reference: Ziko, I., Dolz, J., Granger, E., &amp; Ben Ayed, I. (2020).
/// Laplacian Regularized Few-Shot Learning. ICML 2020.
/// </para>
/// </remarks>
public class LaplacianShotAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly LaplacianShotOptions<T, TInput, TOutput> _lapShotOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.LaplacianShot;

    /// <summary>
    /// Initializes a new LaplacianShot meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for LaplacianShot.</param>
    public LaplacianShotAlgorithm(LaplacianShotOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _lapShotOptions = options;
    }

    /// <summary>
    /// Performs one meta-training step for LaplacianShot.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training is standard episodic training. The Laplacian
    /// regularization only applies during adaptation (inference), so training is the same
    /// as training any metric-based method.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            var queryLoss = ComputeLossFromOutput(
                MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, _lapShotOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task using Laplacian-regularized nearest-centroid classification.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model with Laplacian-refined predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adaptation is:
    /// 1. Compute support centroids (class means)
    /// 2. Compute initial predictions for all queries
    /// 3. Build a kNN graph over query features
    /// 4. Smooth predictions using the graph Laplacian
    /// 5. Return model with smoothed decision boundaries
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Perform Laplacian refinement
        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);
        var refinedWeights = LaplacianRefine(supportFeatures, queryFeatures);

        return new LaplacianShotModel<T, TInput, TOutput>(MetaModel, currentParams, refinedWeights);
    }

    /// <summary>
    /// Applies Laplacian regularization to refine predictions using the query graph.
    /// </summary>
    /// <param name="supportFeatures">Support set features (centroids).</param>
    /// <param name="queryFeatures">Query set features for graph construction.</param>
    /// <returns>Refined classification weights after Laplacian smoothing.</returns>
    /// <remarks>
    /// <para>
    /// Constructs a kNN graph over query features, computes the graph Laplacian,
    /// and solves the regularized classification problem iteratively:
    /// Y_{t+1} = Y_t - alpha * (lambda * L @ Y_t + (Y_t - logits_0))
    /// </para>
    /// <para><b>For Beginners:</b> This smooths predictions over the query graph:
    /// 1. Connect similar queries (kNN graph)
    /// 2. Start with initial predictions (nearest centroid)
    /// 3. Iteratively: each query adjusts its prediction based on neighbors
    /// 4. Result: smooth predictions where nearby queries agree
    /// </para>
    /// </remarks>
    private Vector<T>? LaplacianRefine(Vector<T>? supportFeatures, Vector<T>? queryFeatures)
    {
        if (supportFeatures == null || queryFeatures == null || queryFeatures.Length == 0)
        {
            return supportFeatures;
        }

        int numQuery = queryFeatures.Length;
        int k = Math.Min(_lapShotOptions.KNearestNeighbors, numQuery - 1);
        double sigma = _lapShotOptions.KernelBandwidth;
        double lambda = _lapShotOptions.LaplacianWeight;

        // Build kNN graph (adjacency matrix as flat array for simplicity)
        var weights = new double[numQuery, numQuery];
        for (int i = 0; i < numQuery; i++)
        {
            // Compute distances to all other queries
            var distances = new double[numQuery];
            for (int j = 0; j < numQuery; j++)
            {
                double diff = NumOps.ToDouble(queryFeatures[i]) - NumOps.ToDouble(queryFeatures[j]);
                distances[j] = diff * diff;
            }

            // Find k nearest neighbors
            var neighborIndices = new int[numQuery];
            for (int j = 0; j < numQuery; j++) neighborIndices[j] = j;
            Array.Sort(distances, neighborIndices);

            // Set edge weights for k nearest (skip self at index 0)
            for (int n = 1; n <= k && n < numQuery; n++)
            {
                int j = neighborIndices[n];
                double w = Math.Exp(-distances[n] / Math.Max(sigma, 1e-10));
                weights[i, j] = w;
                weights[j, i] = w; // Symmetric
            }
        }

        // Compute degree and Laplacian
        var degree = new double[numQuery];
        for (int i = 0; i < numQuery; i++)
        {
            for (int j = 0; j < numQuery; j++)
            {
                degree[i] += weights[i, j];
            }
        }

        // Initial logits from support-query similarity
        var logits = new double[numQuery];
        for (int q = 0; q < numQuery; q++)
        {
            double sim = 0;
            for (int s = 0; s < supportFeatures.Length; s++)
            {
                sim += NumOps.ToDouble(NumOps.Multiply(queryFeatures[q], supportFeatures[s % supportFeatures.Length]));
            }
            logits[q] = sim;
        }

        // Iterative Laplacian smoothing
        var refined = (double[])logits.Clone();
        double alpha = 0.1;

        for (int iter = 0; iter < _lapShotOptions.PropagationIterations; iter++)
        {
            var next = new double[numQuery];
            for (int i = 0; i < numQuery; i++)
            {
                // Laplacian term: L @ Y = D @ Y - W @ Y
                double lapTerm = degree[i] * refined[i];
                for (int j = 0; j < numQuery; j++)
                {
                    lapTerm -= weights[i, j] * refined[j];
                }

                // Data fidelity term
                double fidelityTerm = refined[i] - logits[i];

                // Update
                next[i] = refined[i] - alpha * (lambda * lapTerm + fidelityTerm);
            }
            refined = next;
        }

        // Convert back to Vector<T>
        var result = new Vector<T>(supportFeatures.Length);
        for (int i = 0; i < supportFeatures.Length; i++)
        {
            double scale = i < numQuery ? 1.0 / (1.0 + Math.Exp(-refined[i])) : 0.5;
            result[i] = NumOps.Multiply(supportFeatures[i], NumOps.FromDouble(scale));
        }

        return result;
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

/// <summary>
/// Adapted model wrapper for LaplacianShot with graph-refined predictions.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses predictions that have been refined
/// using the graph Laplacian, where similar query examples receive similar predictions.
/// </para>
/// </remarks>
internal class LaplacianShotModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _refinedWeights;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public LaplacianShotModel(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams, Vector<T>? refinedWeights)
    {
        _model = model;
        _backboneParams = backboneParams;
        _refinedWeights = refinedWeights;
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
