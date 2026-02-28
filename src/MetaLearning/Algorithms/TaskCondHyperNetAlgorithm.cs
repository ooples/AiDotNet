using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Task-Conditioned HyperNetwork for meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// A hypernetwork generates task-specific parameter deltas conditioned on a task embedding
/// derived from support-set gradient statistics. The chunked hypernetwork architecture
/// processes the task embedding through a shared hidden layer, then uses per-chunk output
/// heads to produce parameter deltas for each chunk of the target network.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Task embedding: e = compress(grad_support)  →  [EmbeddingDim]
///
/// Hypernetwork (chunked architecture):
///   hidden = tanh(W_h · e + b_h)               →  [HyperHiddenDim]
///   For each chunk c:
///     Δθ_c = W_c · hidden                       →  [ChunkSize]
///
/// Adapted params: θ_adapted = θ_init + Δθ
/// Fine-tune: K standard MAML steps on top
///
/// L_meta = L_query(θ_adapted) + EmbeddingRegWeight * ||e||²
/// Outer: update θ_init via meta-gradient, update W via SPSA
/// </code>
/// </para>
/// </remarks>
public class TaskCondHyperNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly TaskCondHyperNetOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numChunks;

    /// <summary>Hypernetwork weights: W_h (embDim × hiddenDim) + b_h (hiddenDim) + per-chunk W_c (hiddenDim × chunkSize).</summary>
    private Vector<T> _hyperNetWeights;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.TaskCondHyperNet;

    public TaskCondHyperNetAlgorithm(TaskCondHyperNetOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _numChunks = (_paramDim + options.ChunkSize - 1) / options.ChunkSize;

        int embDim = options.EmbeddingDim;
        int hidDim = options.HyperHiddenDim;
        int chunkSize = options.ChunkSize;

        // W_h: embDim * hidDim + b_h: hidDim + per-chunk W_c: numChunks * hidDim * chunkSize
        int totalWeights = embDim * hidDim + hidDim + _numChunks * hidDim * chunkSize;
        _hyperNetWeights = new Vector<T>(totalWeights);
        double scale = 1.0 / Math.Sqrt(hidDim);
        for (int i = 0; i < totalWeights; i++)
            _hyperNetWeights[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();
        int embDim = _algoOptions.EmbeddingDim;

        foreach (var task in taskBatch.Tasks)
        {
            // Compute task embedding from support gradient
            MetaModel.SetParameters(initParams);
            var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var embedding = CompressGradient(supportGrad, embDim);

            // Generate parameter delta via hypernetwork
            var delta = RunHyperNetwork(embedding);

            // Apply hypernetwork-generated delta to init params
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                adaptedParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(delta[d]));

            // Fine-tune with standard MAML steps
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Embedding regularization
            double embNorm = 0;
            for (int e = 0; e < embDim; e++) embNorm += embedding[e] * embedding[e];
            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.EmbeddingRegWeight * embNorm / embDim));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _hyperNetWeights, _algoOptions.OuterLearningRate * 0.1, ComputeHyperLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        int embDim = _algoOptions.EmbeddingDim;

        MetaModel.SetParameters(initParams);
        var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
        var embedding = CompressGradient(supportGrad, embDim);
        var delta = RunHyperNetwork(embedding);

        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            adaptedParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(delta[d]));

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

    private double[] RunHyperNetwork(double[] embedding)
    {
        int embDim = _algoOptions.EmbeddingDim;
        int hidDim = _algoOptions.HyperHiddenDim;
        int chunkSize = _algoOptions.ChunkSize;

        // Layer 1: hidden = tanh(W_h * embedding + b_h)
        var hidden = new double[hidDim];
        int wIdx = 0;
        for (int h = 0; h < hidDim; h++)
        {
            double sum = 0;
            for (int e = 0; e < embDim; e++)
                sum += NumOps.ToDouble(_hyperNetWeights[wIdx++]) * embedding[e];
            hidden[h] = sum;
        }
        // Add bias
        for (int h = 0; h < hidDim; h++)
            hidden[h] = Math.Tanh(hidden[h] + NumOps.ToDouble(_hyperNetWeights[wIdx++]));

        // Layer 2: per-chunk output heads → parameter deltas
        var delta = new double[_paramDim];
        for (int c = 0; c < _numChunks; c++)
        {
            int startParam = c * chunkSize;
            int endParam = Math.Min(startParam + chunkSize, _paramDim);
            for (int p = startParam; p < endParam; p++)
            {
                double sum = 0;
                for (int h = 0; h < hidDim; h++)
                    sum += NumOps.ToDouble(_hyperNetWeights[wIdx + (p - startParam) * hidDim + h]) * hidden[h];
                delta[p] = sum * 0.01; // Small initial scale to avoid disrupting init
            }
            wIdx += chunkSize * hidDim;
        }

        return delta;
    }

    private double ComputeHyperLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        int embDim = _algoOptions.EmbeddingDim;
        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var sg = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var emb = CompressGradient(sg, embDim);
            var delta = RunHyperNetwork(emb);
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                ap[d] = NumOps.Add(initParams[d], NumOps.FromDouble(delta[d]));
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
