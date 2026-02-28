using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of MePo: Memory Prototypes for continual few-shot meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// MePo maintains a memory bank of gradient-space prototypes from previously encountered tasks.
/// When adapting to a new task, it compresses the initial gradient into the prototype space,
/// retrieves the K nearest prototypes, and uses their weighted average to regularize
/// the adaptation trajectory. This prevents catastrophic forgetting by anchoring adaptation
/// to known good trajectories, while still allowing task-specific fine-tuning.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Prototype computation: p = compress(∇L_support) ∈ R^PrototypeDim
///
/// Memory retrieval: top-K nearest prototypes by cosine similarity
///   sim(p, m_i) = p·m_i / (||p|| * ||m_i||)
///   weights: w_i = softmax(sim_i)
///   retrieved: r = Σ w_i * m_i
///
/// Inner loop with prototype regularization:
///   θ ← θ - η * (grad + PrototypeRegWeight * (compress(θ - θ_init) - r))
///
/// Memory update: store new prototype, evict oldest if full
/// Outer loop: standard meta-gradient update
/// </code>
/// </para>
/// </remarks>
public class MePoAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MePoOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _protoDim;

    /// <summary>Memory bank: list of prototype vectors.</summary>
    private readonly List<double[]> _memoryBank;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MePo;

    public MePoAlgorithm(MePoOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _protoDim = options.PrototypeDim;
        _memoryBank = new List<double[]>();
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Compute task prototype from initial support gradient
            MetaModel.SetParameters(initParams);
            var initGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var prototype = CompressToPrototype(initGrad);

            // Retrieve nearest prototypes from memory
            var retrieved = RetrievePrototypes(prototype);

            // Inner loop with prototype regularization
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Compute compressed parameter delta
                var paramDelta = CompressParamDelta(adaptedParams, initParams);

                for (int d = 0; d < _paramDim; d++)
                {
                    double gradVal = NumOps.ToDouble(grad[d]);

                    // Prototype regularization: pull compressed delta toward retrieved prototype
                    double regGrad = 0;
                    if (retrieved != null)
                    {
                        int cd = d % _protoDim;
                        regGrad = _algoOptions.PrototypeRegWeight * (paramDelta[cd] - retrieved[cd]);
                    }

                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * (gradVal + regGrad)));
                }
            }

            // Store prototype in memory
            StorePrototype(prototype);

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

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
        MetaModel.SetParameters(initParams);
        var initGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
        var prototype = CompressToPrototype(initGrad);
        var retrieved = RetrievePrototypes(prototype);

        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var paramDelta = CompressParamDelta(adaptedParams, initParams);

            for (int d = 0; d < _paramDim; d++)
            {
                double gradVal = NumOps.ToDouble(grad[d]);
                double regGrad = 0;
                if (retrieved != null)
                {
                    int cd = d % _protoDim;
                    regGrad = _algoOptions.PrototypeRegWeight * (paramDelta[cd] - retrieved[cd]);
                }
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * (gradVal + regGrad)));
            }
        }

        StorePrototype(prototype);
        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] CompressToPrototype(Vector<T> grad)
    {
        var proto = new double[_protoDim];
        // Average pooling: group gradient dims into _protoDim buckets
        int bucketSize = Math.Max(1, _paramDim / _protoDim);
        for (int p = 0; p < _protoDim; p++)
        {
            double sum = 0;
            int count = 0;
            int start = p * bucketSize;
            for (int d = start; d < start + bucketSize && d < grad.Length; d++)
            {
                sum += NumOps.ToDouble(grad[d]);
                count++;
            }
            proto[p] = count > 0 ? Math.Tanh(sum / count) : 0;
        }
        return proto;
    }

    private double[] CompressParamDelta(Vector<T> current, Vector<T> initial)
    {
        var delta = new double[_protoDim];
        int bucketSize = Math.Max(1, _paramDim / _protoDim);
        for (int p = 0; p < _protoDim; p++)
        {
            double sum = 0;
            int count = 0;
            int start = p * bucketSize;
            for (int d = start; d < start + bucketSize && d < _paramDim; d++)
            {
                sum += NumOps.ToDouble(current[d]) - NumOps.ToDouble(initial[d]);
                count++;
            }
            delta[p] = count > 0 ? sum / count : 0;
        }
        return delta;
    }

    private double[]? RetrievePrototypes(double[] query)
    {
        if (_memoryBank.Count == 0) return null;

        // Compute cosine similarity to all stored prototypes
        double queryNorm = 0;
        for (int p = 0; p < _protoDim; p++) queryNorm += query[p] * query[p];
        queryNorm = Math.Sqrt(queryNorm) + 1e-10;

        var similarities = new List<(int idx, double sim)>();
        for (int i = 0; i < _memoryBank.Count; i++)
        {
            double dot = 0, memNorm = 0;
            for (int p = 0; p < _protoDim; p++)
            {
                dot += query[p] * _memoryBank[i][p];
                memNorm += _memoryBank[i][p] * _memoryBank[i][p];
            }
            memNorm = Math.Sqrt(memNorm) + 1e-10;
            similarities.Add((i, dot / (queryNorm * memNorm)));
        }

        // Sort by similarity descending, take top-K
        similarities.Sort((a, b) => b.sim.CompareTo(a.sim));
        int topK = Math.Min(_algoOptions.RetrievalTopK, similarities.Count);

        // Softmax-weighted average of top-K
        double maxSim = similarities[0].sim;
        double sumExp = 0;
        var weights = new double[topK];
        for (int i = 0; i < topK; i++)
        {
            weights[i] = Math.Exp(similarities[i].sim - maxSim);
            sumExp += weights[i];
        }
        for (int i = 0; i < topK; i++) weights[i] /= (sumExp + 1e-10);

        var result = new double[_protoDim];
        for (int i = 0; i < topK; i++)
        {
            int idx = similarities[i].idx;
            for (int p = 0; p < _protoDim; p++)
                result[p] += weights[i] * _memoryBank[idx][p];
        }
        return result;
    }

    private void StorePrototype(double[] prototype)
    {
        if (_memoryBank.Count >= _algoOptions.MemorySize)
            _memoryBank.RemoveAt(0);
        _memoryBank.Add(prototype);
    }
}
