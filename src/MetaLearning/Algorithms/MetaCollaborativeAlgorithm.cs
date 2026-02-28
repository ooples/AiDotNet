using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-Collaborative Learning for cross-domain few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-Collaborative Learning maintains a set of domain-specific gradient momentum buffers
/// and uses gradient alignment (cosine similarity) between tasks to modulate cross-task
/// knowledge transfer. Tasks with well-aligned gradients reinforce each other's updates;
/// conflicting gradients are dampened via a PCGrad-inspired projection.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Domain buffers: {m_1, ..., m_K} (one per domain slot)
///
/// For each meta-batch:
///   1. Compute per-task gradients g_τ
///   2. Assign each task to nearest domain slot by gradient cosine similarity
///   3. Update domain buffers: m_k ← β*m_k + (1-β)*mean(g_τ ∈ domain k)
///   4. Collaborative gradient for task τ:
///      g'_τ = g_τ + w_align * Σ_k sim(g_τ, m_k) * m_k  (positive alignment)
///      If sim(g_τ, m_k) &lt; 0: project out conflicting component (PCGrad)
///   5. Inner loop uses g'_τ, outer loop aggregates normally
/// </code>
/// </para>
/// </remarks>
public class MetaCollaborativeAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaCollaborativeOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Domain-specific gradient momentum buffers.</summary>
    private readonly double[][] _domainBuffers;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaCollaborative;

    public MetaCollaborativeAlgorithm(MetaCollaborativeOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;

        _domainBuffers = new double[options.NumDomainSlots][];
        for (int k = 0; k < options.NumDomainSlots; k++)
            _domainBuffers[k] = new double[_paramDim];
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();
        var tasks = taskBatch.Tasks;

        // Compute initial gradients for all tasks
        var taskGrads = new List<Vector<T>>();
        foreach (var task in tasks)
        {
            MetaModel.SetParameters(initParams);
            taskGrads.Add(ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput)));
        }

        // Assign tasks to domain slots and update buffers
        var assignments = AssignToDomains(taskGrads);
        UpdateDomainBuffers(taskGrads, assignments);

        // Inner loop with collaborative gradients
        for (int tIdx = 0; tIdx < tasks.Length; tIdx++)
        {
            var task = tasks[tIdx];
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Apply collaborative modulation
                var collabGrad = ComputeCollaborativeGradient(grad, tIdx);
                adaptedParams = ApplyGradients(adaptedParams, collabGrad, _algoOptions.InnerLearningRate);
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
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

            // Use domain buffers for collaborative adaptation
            var collabGrad = ComputeCollaborativeGradientFromBuffers(grad);
            adaptedParams = ApplyGradients(adaptedParams, collabGrad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Assigns each task to the domain slot with highest gradient cosine similarity.
    /// </summary>
    private int[] AssignToDomains(List<Vector<T>> taskGrads)
    {
        var assignments = new int[taskGrads.Count];
        bool buffersInitialized = false;
        for (int k = 0; k < _algoOptions.NumDomainSlots; k++)
        {
            double norm = 0;
            for (int d = 0; d < _paramDim; d++) norm += _domainBuffers[k][d] * _domainBuffers[k][d];
            if (norm > 1e-10) { buffersInitialized = true; break; }
        }

        for (int t = 0; t < taskGrads.Count; t++)
        {
            if (!buffersInitialized)
            {
                assignments[t] = t % _algoOptions.NumDomainSlots;
                continue;
            }

            double bestSim = double.NegativeInfinity;
            int bestSlot = 0;
            for (int k = 0; k < _algoOptions.NumDomainSlots; k++)
            {
                double sim = CosineSimilarity(taskGrads[t], _domainBuffers[k]);
                if (sim > bestSim) { bestSim = sim; bestSlot = k; }
            }
            assignments[t] = bestSlot;
        }
        return assignments;
    }

    /// <summary>
    /// Updates domain buffers with EMA of assigned task gradients.
    /// </summary>
    private void UpdateDomainBuffers(List<Vector<T>> taskGrads, int[] assignments)
    {
        double beta = _algoOptions.GradientMomentum;
        var counts = new int[_algoOptions.NumDomainSlots];
        var sums = new double[_algoOptions.NumDomainSlots][];
        for (int k = 0; k < _algoOptions.NumDomainSlots; k++)
            sums[k] = new double[_paramDim];

        for (int t = 0; t < taskGrads.Count; t++)
        {
            int k = assignments[t];
            counts[k]++;
            for (int d = 0; d < _paramDim; d++)
                sums[k][d] += NumOps.ToDouble(taskGrads[t][d]);
        }

        for (int k = 0; k < _algoOptions.NumDomainSlots; k++)
        {
            if (counts[k] > 0)
            {
                for (int d = 0; d < _paramDim; d++)
                    _domainBuffers[k][d] = beta * _domainBuffers[k][d] + (1 - beta) * sums[k][d] / counts[k];
            }
        }
    }

    /// <summary>
    /// Computes collaborative gradient: adds aligned domain signals, projects out conflicting ones.
    /// </summary>
    private Vector<T> ComputeCollaborativeGradient(Vector<T> taskGrad, int taskIdx)
    {
        var result = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) result[d] = taskGrad[d];

        for (int k = 0; k < _algoOptions.NumDomainSlots; k++)
        {
            double sim = CosineSimilarity(taskGrad, _domainBuffers[k]);
            if (sim > 0)
            {
                // Positive alignment: add domain buffer signal
                for (int d = 0; d < _paramDim; d++)
                    result[d] = NumOps.Add(result[d], NumOps.FromDouble(_algoOptions.AlignmentWeight * sim * _domainBuffers[k][d]));
            }
            else if (sim < -0.1)
            {
                // PCGrad: project out conflicting component
                double dot = 0, bufNorm = 0;
                for (int d = 0; d < _paramDim; d++)
                {
                    dot += NumOps.ToDouble(result[d]) * _domainBuffers[k][d];
                    bufNorm += _domainBuffers[k][d] * _domainBuffers[k][d];
                }
                if (bufNorm > 1e-10)
                {
                    double proj = dot / bufNorm;
                    for (int d = 0; d < _paramDim; d++)
                        result[d] = NumOps.Subtract(result[d], NumOps.FromDouble(proj * _domainBuffers[k][d]));
                }
            }
        }

        return result;
    }

    private Vector<T> ComputeCollaborativeGradientFromBuffers(Vector<T> grad)
    {
        return ComputeCollaborativeGradient(grad, -1);
    }

    private double CosineSimilarity(Vector<T> a, double[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int d = 0; d < _paramDim; d++)
        {
            double aVal = NumOps.ToDouble(a[d]);
            dot += aVal * b[d];
            normA += aVal * aVal;
            normB += b[d] * b[d];
        }
        return dot / (Math.Sqrt(normA * normB) + 1e-10);
    }
}
