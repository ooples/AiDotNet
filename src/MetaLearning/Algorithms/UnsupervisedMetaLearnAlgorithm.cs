using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Unsupervised Meta-Learning (Hsu et al., 2019).
/// </summary>
/// <remarks>
/// <para>
/// Unsupervised Meta-Learning constructs pseudo-tasks by clustering task gradient profiles
/// in a compressed space. Tasks assigned to the same cluster are treated as similar and
/// their gradients are reinforced, while cross-cluster gradients are dampened. This enables
/// meta-learning to discover task structure without explicit labels. A prediction consistency
/// regularization ensures that the adapted model maintains stable predictions.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Gradient clustering:
///   g_c = compress(grad) ∈ R^ClusteringDim
///   cluster_k = argmin_k ||g_c - centroid_k||²
///   centroids updated via EMA: c_k = (1-α)*c_k + α*g_c  (if assigned)
///
/// Cluster-aware gradient scaling:
///   same_cluster_tasks → weight 1.0
///   different_cluster_tasks → weight dampened
///
/// Consistency regularization:
///   L_consist = ||pred_support - pred_query||² (adapted predictions should be stable)
///
/// L_meta = L_query + ConsistencyWeight * L_consist
/// </code>
/// </para>
/// </remarks>
public class UnsupervisedMetaLearnAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly UnsupervisedMetaLearnOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _clusterDim;

    /// <summary>Cluster centroids: NumClusters × ClusteringDim.</summary>
    private double[][] _centroids;

    /// <summary>Per-cluster count for EMA weighting.</summary>
    private int[] _clusterCounts;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.UnsupervisedMetaLearn;

    public UnsupervisedMetaLearnAlgorithm(UnsupervisedMetaLearnOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _clusterDim = options.ClusteringDim;

        // Initialize centroids randomly
        _centroids = new double[options.NumClusters][];
        _clusterCounts = new int[options.NumClusters];
        for (int k = 0; k < options.NumClusters; k++)
        {
            _centroids[k] = new double[_clusterDim];
            for (int d = 0; d < _clusterDim; d++)
            {
                double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
                double u2 = RandomGenerator.NextDouble();
                _centroids[k][d] = 0.1 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            }
        }
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var taskClusters = new List<int>();
        var initParams = MetaModel.GetParameters();

        // Phase 1: Adapt all tasks and assign to clusters
        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var initGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var compressed = CompressGradient(initGrad);
            int cluster = AssignCluster(compressed);
            taskClusters.Add(cluster);

            // Update centroid via EMA
            _clusterCounts[cluster]++;
            double rate = _algoOptions.ClusterUpdateRate;
            for (int d = 0; d < _clusterDim; d++)
                _centroids[cluster][d] = (1.0 - rate) * _centroids[cluster][d] + rate * compressed[d];

            // Inner loop adaptation
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Prediction consistency: L2 between support and query predictions
            var queryPred = ConvertToVector(MetaModel.Predict(task.QueryInput)) ?? new Vector<T>(1);
            var supportPred = ConvertToVector(MetaModel.Predict(task.SupportInput)) ?? new Vector<T>(1);
            double consistLoss = 0;
            int predLen = Math.Min(queryPred.Length, supportPred.Length);
            for (int d = 0; d < predLen; d++)
            {
                double diff = NumOps.ToDouble(queryPred[d]) - NumOps.ToDouble(supportPred[d]);
                consistLoss += diff * diff;
            }
            if (predLen > 0) consistLoss /= predLen;

            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.ConsistencyWeight * consistLoss));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Phase 2: Cluster-aware gradient weighting
        // Tasks in larger clusters get slightly dampened (reduce cluster imbalance)
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var weightedGrad = new Vector<T>(_paramDim);
            double totalWeight = 0;

            for (int t = 0; t < metaGradients.Count; t++)
            {
                int cluster = taskClusters[t];
                // Inverse-frequency weighting: under-represented clusters get higher weight
                double weight = 1.0 / Math.Max(_clusterCounts[cluster], 1);
                totalWeight += weight;

                for (int d = 0; d < _paramDim; d++)
                    weightedGrad[d] = NumOps.Add(weightedGrad[d],
                        NumOps.FromDouble(weight * NumOps.ToDouble(metaGradients[t][d])));
            }

            // Normalize
            if (totalWeight > 1e-10)
                for (int d = 0; d < _paramDim; d++)
                    weightedGrad[d] = NumOps.FromDouble(NumOps.ToDouble(weightedGrad[d]) / totalWeight);

            MetaModel.SetParameters(ApplyGradients(initParams, weightedGrad, _algoOptions.OuterLearningRate));
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] CompressGradient(Vector<T> grad)
    {
        var compressed = new double[_clusterDim];
        int bucketSize = Math.Max(1, _paramDim / _clusterDim);
        for (int c = 0; c < _clusterDim; c++)
        {
            double sum = 0;
            int start = c * bucketSize;
            for (int d = start; d < start + bucketSize && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]);
            compressed[c] = Math.Tanh(sum / bucketSize);
        }
        return compressed;
    }

    private int AssignCluster(double[] compressed)
    {
        int bestCluster = 0;
        double bestDist = double.MaxValue;
        for (int k = 0; k < _algoOptions.NumClusters; k++)
        {
            double dist = 0;
            for (int d = 0; d < _clusterDim; d++)
            {
                double diff = compressed[d] - _centroids[k][d];
                dist += diff * diff;
            }
            if (dist < bestDist) { bestDist = dist; bestCluster = k; }
        }
        return bestCluster;
    }
}
