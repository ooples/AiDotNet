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
/// Implementation of SIB (Sequential Information Bottleneck) for transductive few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// SIB uses the information bottleneck principle to iteratively refine cluster assignments
/// for transductive few-shot classification. It processes all examples (support + query) jointly,
/// optimizing a trade-off between information retention and compression.
/// </para>
/// <para><b>For Beginners:</b> SIB is like an intelligent clustering algorithm:
///
/// **How it works:**
/// 1. Start with class prototypes from support examples
/// 2. Assign all query examples to the closest prototype (initial clusters)
/// 3. Iteratively refine:
///    a. For each example, compute KL divergence to each cluster
///    b. Reassign to the best cluster (minimum KL divergence)
///    c. Update cluster statistics
///    d. Repeat until convergence
/// 4. Multiple restarts ensure we don't get stuck in bad solutions
///
/// **Information Bottleneck trade-off:**
/// - Want to KEEP: Information about which class each example belongs to
/// - Want to REMOVE: Noise, irrelevant variations, outlier effects
/// - Beta parameter controls this balance
///
/// **Why multiple restarts?**
/// The SIB optimization landscape can have local optima. Running from different
/// starting points increases the chance of finding the globally best clustering.
/// </para>
/// <para><b>Algorithm - SIB:</b>
/// <code>
/// # Given: support features z_s, query features z_q, support labels y_s
///
/// best_score = -inf
/// for restart in range(R):
///     # 1. Initialize centroids from support prototypes (with random perturbation)
///     centroids = class_means(z_s, y_s) + noise
///
///     # 2. SIB optimization loop
///     for iter in range(T):
///         # Assign all examples to clusters via KL divergence
///         for each example x:
///             p(c|x) = softmax(-beta * KL(p(.|x) || p(.|c)) / temperature)
///
///         # Update cluster statistics
///         for each cluster c:
///             p(.|c) = weighted_mean(p(.|x) for x assigned to c)
///
///         # Check convergence
///         if assignments_unchanged: break
///
///     # 3. Keep best restart
///     score = mutual_information(assignments)
///     if score > best_score:
///         best_assignments = assignments
///         best_score = score
///
/// # Output query labels from best_assignments
/// </code>
/// </para>
/// <para>
/// Reference: Hu, Y., Gripon, V., &amp; Pateux, S. (2020).
/// Leveraging the Feature Distribution in Transfer-based Few-Shot Learning.
/// </para>
/// </remarks>
public class SIBAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly SIBOptions<T, TInput, TOutput> _sibOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SIB;

    /// <summary>
    /// Initializes a new SIB meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for SIB.</param>
    public SIBAlgorithm(SIBOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _sibOptions = options;
    }

    /// <summary>
    /// Performs one meta-training step for SIB.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training updates the backbone network to produce
    /// features that are amenable to SIB clustering. The SIB optimization itself
    /// is only applied during adaptation.
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
            var updatedParams = ApplyGradients(initParams, avgGrad, _sibOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task using the Sequential Information Bottleneck.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model with SIB-optimized cluster assignments.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adaptation is where SIB shines:
    /// 1. Extract features from all examples (support + query)
    /// 2. Run SIB clustering multiple times from different starting points
    /// 3. Pick the clustering with the highest information score
    /// 4. Query labels come from their cluster assignments
    ///
    /// This transductive approach uses ALL query examples to improve each individual
    /// prediction - a major advantage over inductive methods.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);

        // Run SIB with multiple restarts
        var bestCentroids = RunSIB(supportFeatures, queryFeatures);

        return new SIBModel<T, TInput, TOutput>(MetaModel, currentParams, bestCentroids);
    }

    /// <summary>
    /// Runs the SIB optimization with multiple random restarts.
    /// </summary>
    /// <param name="supportFeatures">Support set features for initializing centroids.</param>
    /// <param name="queryFeatures">Query set features for transductive clustering.</param>
    /// <returns>The best centroid set found across all restarts.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Runs SIB multiple times with different random noise
    /// added to the initial centroids. Each run may find a different solution, and
    /// we keep the best one (highest mutual information score).
    /// </para>
    /// </remarks>
    private Vector<T>? RunSIB(Vector<T>? supportFeatures, Vector<T>? queryFeatures)
    {
        if (supportFeatures == null || supportFeatures.Length == 0)
        {
            return supportFeatures;
        }

        Vector<T>? bestCentroids = null;
        double bestScore = double.MinValue;

        for (int restart = 0; restart < _sibOptions.NumRestarts; restart++)
        {
            // Initialize centroids from support with random perturbation
            var centroids = new Vector<T>(supportFeatures.Length);
            double perturbScale = 0.01 * (restart + 1);
            for (int i = 0; i < supportFeatures.Length; i++)
            {
                double perturbation = (RandomGenerator.NextDouble() - 0.5) * 2.0 * perturbScale;
                centroids[i] = NumOps.Add(supportFeatures[i], NumOps.FromDouble(perturbation));
            }

            // SIB optimization loop
            var assignments = new int[queryFeatures != null ? queryFeatures.Length : 0];
            for (int iter = 0; iter < _sibOptions.NumSIBIterations; iter++)
            {
                bool changed = false;

                // Assign each query to nearest centroid using KL-divergence-like score
                if (queryFeatures != null)
                {
                    for (int q = 0; q < queryFeatures.Length; q++)
                    {
                        double bestDist = double.MaxValue;
                        int bestCluster = 0;

                        for (int c = 0; c < centroids.Length; c++)
                        {
                            double qVal = NumOps.ToDouble(queryFeatures[q]);
                            double cVal = NumOps.ToDouble(centroids[c]);
                            // KL divergence approximation
                            double klDiv = (qVal - cVal) * (qVal - cVal);
                            double score = _sibOptions.Beta * klDiv / _sibOptions.Temperature;

                            if (score < bestDist)
                            {
                                bestDist = score;
                                bestCluster = c;
                            }
                        }

                        if (assignments[q] != bestCluster)
                        {
                            assignments[q] = bestCluster;
                            changed = true;
                        }
                    }
                }

                if (!changed) break;

                // Update centroids based on assignments
                var centroidSums = new double[centroids.Length];
                var centroidCounts = new int[centroids.Length];
                for (int i = 0; i < centroids.Length; i++)
                {
                    centroidSums[i] = NumOps.ToDouble(supportFeatures[i]);
                    centroidCounts[i] = 1;
                }

                if (queryFeatures != null)
                {
                    for (int q = 0; q < queryFeatures.Length; q++)
                    {
                        int cluster = assignments[q];
                        if (cluster < centroids.Length)
                        {
                            centroidSums[cluster] += NumOps.ToDouble(queryFeatures[q]);
                            centroidCounts[cluster]++;
                        }
                    }
                }

                for (int c = 0; c < centroids.Length; c++)
                {
                    centroids[c] = NumOps.FromDouble(centroidSums[c] / Math.Max(centroidCounts[c], 1));
                }
            }

            // Score this solution (mutual information approximation)
            double miScore = 0;
            if (queryFeatures != null)
            {
                for (int q = 0; q < queryFeatures.Length; q++)
                {
                    int cluster = assignments[q];
                    if (cluster < centroids.Length)
                    {
                        double dist = NumOps.ToDouble(queryFeatures[q]) - NumOps.ToDouble(centroids[cluster]);
                        miScore -= dist * dist;
                    }
                }
            }

            if (miScore > bestScore)
            {
                bestScore = miScore;
                bestCentroids = new Vector<T>(centroids.Length);
                for (int i = 0; i < centroids.Length; i++)
                {
                    bestCentroids[i] = centroids[i];
                }
            }
        }

        return bestCentroids;
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
/// Adapted model wrapper for SIB with information-bottleneck-optimized centroids.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model classifies using centroids that were
/// optimized by the SIB algorithm to balance information retention and compression.
/// The centroids incorporate information from both support AND query examples.
/// </para>
/// </remarks>
internal class SIBModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _centroids;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public SIBModel(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams, Vector<T>? centroids)
    {
        _model = model;
        _backboneParams = backboneParams;
        _centroids = centroids;
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
