using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of HyperCLIP: Contrastive Learning for Hypernetwork-based Meta-Learning.
/// </summary>
/// <remarks>
/// <para>
/// HyperCLIP uses contrastive alignment between task embeddings (from support gradients)
/// and parameter embeddings (from adapted parameter deltas). An InfoNCE contrastive loss
/// aligns each task's gradient signature with its adapted parameter fingerprint, learning
/// a shared projection space. This cross-modal alignment improves adaptation quality.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// For each task in batch:
///   Task embedding: t_i = project_task(compress(grad_support_i))
///   Adapt: θ_i via MAML
///   Param embedding: p_i = project_param(compress(θ_i - θ_init))
///
/// Contrastive loss (InfoNCE across batch):
///   sim(i,j) = cosine(t_i, p_j) / τ
///   L_NCE = -Σ_i log(exp(sim(i,i)) / Σ_j exp(sim(i,j)))
///
/// L_meta = Σ L_query_i + ContrastiveWeight * L_NCE
/// Outer: update θ_init, update projections via SPSA
/// </code>
/// </para>
/// </remarks>
public class HyperCLIPAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly HyperCLIPOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Projection weights: task projection (embDim × projDim) + param projection (embDim × projDim).</summary>
    private Vector<T> _projectionWeights;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.HyperCLIP;

    public HyperCLIPAlgorithm(HyperCLIPOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;

        int embDim = options.EmbeddingDim;
        int projDim = options.ProjectionDim;
        // Task projection + param projection
        int totalWeights = 2 * embDim * projDim;
        _projectionWeights = new Vector<T>(totalWeights);
        double scale = 1.0 / Math.Sqrt(projDim);
        for (int i = 0; i < totalWeights; i++)
            _projectionWeights[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();
        int embDim = _algoOptions.EmbeddingDim;
        int projDim = _algoOptions.ProjectionDim;
        int numTasks = taskBatch.Tasks.Length;

        // Collect task and param embeddings for contrastive loss
        var taskEmbeddings = new double[numTasks][];
        var paramEmbeddings = new double[numTasks][];

        for (int i = 0; i < numTasks; i++)
        {
            var task = taskBatch.Tasks[i];

            // Task embedding from support gradient
            MetaModel.SetParameters(initParams);
            var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var gradFeatures = CompressGradient(supportGrad, embDim);
            taskEmbeddings[i] = ProjectTask(gradFeatures);

            // Standard MAML adaptation
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            // Param embedding from parameter delta
            var paramDelta = CompressParamDelta(initParams, adaptedParams, embDim);
            paramEmbeddings[i] = ProjectParam(paramDelta);

            // Query loss
            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // InfoNCE contrastive loss
        double contrastiveLoss = ComputeInfoNCE(taskEmbeddings, paramEmbeddings, numTasks, projDim);

        // Add contrastive loss to all task losses
        if (numTasks > 1)
        {
            var contrastiveT = NumOps.FromDouble(_algoOptions.ContrastiveWeight * contrastiveLoss);
            for (int i = 0; i < losses.Count; i++)
                losses[i] = NumOps.Add(losses[i], contrastiveT);
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _projectionWeights, _algoOptions.OuterLearningRate * 0.1, ComputeCLIPLoss);

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

    private double[] CompressGradient(Vector<T> grad, int targetDim)
    {
        var result = new double[targetDim];
        int bucketSize = Math.Max(1, _paramDim / targetDim);
        for (int e = 0; e < targetDim; e++)
        {
            double sum = 0;
            int start = e * bucketSize;
            for (int d = start; d < start + bucketSize && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]);
            result[e] = Math.Tanh(sum / bucketSize);
        }
        return result;
    }

    private double[] CompressParamDelta(Vector<T> initParams, Vector<T> adaptedParams, int targetDim)
    {
        var result = new double[targetDim];
        int bucketSize = Math.Max(1, _paramDim / targetDim);
        for (int e = 0; e < targetDim; e++)
        {
            double sum = 0;
            int start = e * bucketSize;
            for (int d = start; d < start + bucketSize && d < _paramDim; d++)
                sum += NumOps.ToDouble(adaptedParams[d]) - NumOps.ToDouble(initParams[d]);
            result[e] = Math.Tanh(sum / bucketSize);
        }
        return result;
    }

    private double[] ProjectTask(double[] features)
    {
        int embDim = _algoOptions.EmbeddingDim;
        int projDim = _algoOptions.ProjectionDim;
        var projected = new double[projDim];
        // Task projection: first embDim * projDim weights
        for (int p = 0; p < projDim; p++)
        {
            double sum = 0;
            for (int e = 0; e < embDim; e++)
                sum += NumOps.ToDouble(_projectionWeights[p * embDim + e]) * features[e];
            projected[p] = sum;
        }
        // L2 normalize
        L2Normalize(projected);
        return projected;
    }

    private double[] ProjectParam(double[] features)
    {
        int embDim = _algoOptions.EmbeddingDim;
        int projDim = _algoOptions.ProjectionDim;
        int offset = embDim * projDim; // After task projection weights
        var projected = new double[projDim];
        for (int p = 0; p < projDim; p++)
        {
            double sum = 0;
            for (int e = 0; e < embDim; e++)
                sum += NumOps.ToDouble(_projectionWeights[offset + p * embDim + e]) * features[e];
            projected[p] = sum;
        }
        L2Normalize(projected);
        return projected;
    }

    private static void L2Normalize(double[] vec)
    {
        double norm = 0;
        for (int i = 0; i < vec.Length; i++) norm += vec[i] * vec[i];
        norm = Math.Sqrt(norm) + 1e-10;
        for (int i = 0; i < vec.Length; i++) vec[i] /= norm;
    }

    private double ComputeInfoNCE(double[][] taskEmb, double[][] paramEmb, int n, int dim)
    {
        if (n <= 1) return 0;
        double tau = _algoOptions.ContrastiveTemperature;
        double loss = 0;

        for (int i = 0; i < n; i++)
        {
            // Compute similarities: sim(t_i, p_j) for all j
            double maxSim = double.NegativeInfinity;
            var sims = new double[n];
            for (int j = 0; j < n; j++)
            {
                double dot = 0;
                for (int d = 0; d < dim; d++) dot += taskEmb[i][d] * paramEmb[j][d];
                sims[j] = dot / tau;
                if (sims[j] > maxSim) maxSim = sims[j];
            }

            // Log-sum-exp for numerical stability
            double sumExp = 0;
            for (int j = 0; j < n; j++) sumExp += Math.Exp(sims[j] - maxSim);
            loss += -(sims[i] - maxSim - Math.Log(sumExp + 1e-10));
        }

        return loss / n;
    }

    private double ComputeCLIPLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                ap = ApplyGradients(ap, g, _algoOptions.InnerLearningRate);
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
