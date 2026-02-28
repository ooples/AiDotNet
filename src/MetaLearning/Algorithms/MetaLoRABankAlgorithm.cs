using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-LoRA Bank (2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-LoRA Bank maintains a bank of K diverse LoRA modules, each representing a
/// learned adaptation pattern. For a new task, a gating network computes task-conditioned
/// scores and selects the top-K modules via sparse gating. The selected modules are
/// combined with learned gating weights to produce the task-specific adaptation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: base params θ, bank of LoRA modules {M_1,...,M_K}, gating network G
///
/// For each task:
///   1. Compute task embedding from support features: e = avg(model(x_support))
///   2. Compute gating scores: s_k = G(e) for each module k
///   3. Select top-K modules by score
///   4. Compute gating weights: w = softmax(top-K scores / temperature)
///   5. Combine modules: Δθ = Σ_{k ∈ top-K} w_k * M_k
///   6. Adapted params: θ' = θ + Δθ
///   7. Fine-tune coefficients on support set (inner loop on gating weights only)
///
/// Outer loop: update base params, all LoRA modules, and gating network
/// </code>
/// </para>
/// <para><b>Advantages:</b> Each module specializes in a different type of task adaptation.
/// The gating mechanism enables compositional generalization — novel tasks can be handled
/// by new combinations of existing modules.
/// </para>
/// </remarks>
public class MetaLoRABankAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaLoRABankOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// Bank of LoRA modules. Each module is a low-rank basis vector in parameter space
    /// (length = rank * paramDim). Module k's basis vectors start at k * rank * paramDim.
    /// </summary>
    private Vector<T> _moduleBank;

    /// <summary>
    /// Gating network parameters: maps task embeddings to module scores.
    /// Linear: embeddingDim → bankSize (weights + bias).
    /// </summary>
    private Vector<T> _gatingParams;

    private readonly int _paramDim;
    private readonly int _bankSize;
    private readonly int _rank;
    private readonly int _topK;
    private readonly int _embeddingDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaLoRABank;

    public MetaLoRABankAlgorithm(MetaLoRABankOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _bankSize = Math.Max(1, options.BankSize);
        _rank = Math.Max(1, options.Rank);
        _topK = Math.Max(1, Math.Min(options.TopK, _bankSize));
        _embeddingDim = Math.Min(_paramDim, 64);

        // Initialize module bank: each module has rank basis vectors of length paramDim
        int moduleSize = _rank * _paramDim;
        _moduleBank = new Vector<T>(_bankSize * moduleSize);
        double initScale = 0.01;
        for (int i = 0; i < _moduleBank.Length; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            _moduleBank[i] = NumOps.FromDouble(z * initScale);
        }

        // Initialize gating network: linear(embeddingDim → bankSize)
        int gatingSize = _embeddingDim * _bankSize + _bankSize;
        _gatingParams = new Vector<T>(gatingSize);
        double gateScale = 1.0 / Math.Sqrt(_embeddingDim);
        for (int i = 0; i < gatingSize; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            _gatingParams[i] = NumOps.FromDouble(z * gateScale);
        }
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = MetaModel.GetParameters();
        var moduleUsageCounts = new double[_bankSize];

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(baseParams);

            // Compute task embedding from support features
            var taskEmbed = ComputeTaskEmbedding(task.SupportInput);

            // Compute gating scores and select top-K modules
            var scores = ComputeGatingScores(taskEmbed);
            var (selectedIndices, gatingWeights) = SelectTopK(scores);

            // Track module usage for load balancing
            foreach (int idx in selectedIndices)
                moduleUsageCounts[idx] += 1.0;

            // Combine selected modules into task-specific parameter delta
            var paramDelta = CombineModules(selectedIndices, gatingWeights);

            // Inner loop: fine-tune gating weights on support set
            var currentWeights = new double[gatingWeights.Length];
            Array.Copy(gatingWeights, currentWeights, gatingWeights.Length);

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                var delta = CombineModulesWithWeights(selectedIndices, currentWeights);
                var adaptedParams = AddVectors(baseParams, delta);
                MetaModel.SetParameters(adaptedParams);

                var fullGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Compute gradient w.r.t. each gating weight
                for (int k = 0; k < selectedIndices.Length; k++)
                {
                    double gradW = ComputeModuleGradient(fullGrad, selectedIndices[k]);
                    currentWeights[k] -= _algoOptions.InnerLearningRate * gradW;
                }

                // Re-normalize weights to sum to 1
                NormalizeWeights(currentWeights);
            }

            // Evaluate on query set with final adapted params
            var finalDelta = CombineModulesWithWeights(selectedIndices, currentWeights);
            var finalParams = AddVectors(baseParams, finalDelta);
            MetaModel.SetParameters(finalParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Add load balancing regularization
            double loadBalanceLoss = ComputeLoadBalanceLoss(moduleUsageCounts, taskBatch.Tasks.Length);
            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.LoadBalanceRegularization * loadBalanceLoss));
            losses.Add(totalLoss);

            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update base params
        MetaModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update module bank and gating network via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _moduleBank, _algoOptions.OuterLearningRate * 0.1, ComputeBankLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _gatingParams, _algoOptions.OuterLearningRate, ComputeBankLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();
        MetaModel.SetParameters(baseParams);

        var taskEmbed = ComputeTaskEmbedding(task.SupportInput);
        var scores = ComputeGatingScores(taskEmbed);
        var (selectedIndices, gatingWeights) = SelectTopK(scores);

        // Fine-tune gating weights on support set
        var currentWeights = new double[gatingWeights.Length];
        Array.Copy(gatingWeights, currentWeights, gatingWeights.Length);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var delta = CombineModulesWithWeights(selectedIndices, currentWeights);
            var adaptedParams = AddVectors(baseParams, delta);
            MetaModel.SetParameters(adaptedParams);

            var fullGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            for (int k = 0; k < selectedIndices.Length; k++)
            {
                double gradW = ComputeModuleGradient(fullGrad, selectedIndices[k]);
                currentWeights[k] -= _algoOptions.InnerLearningRate * gradW;
            }
            NormalizeWeights(currentWeights);
        }

        var finalDelta = CombineModulesWithWeights(selectedIndices, currentWeights);
        var finalParams = AddVectors(baseParams, finalDelta);
        MetaModel.SetParameters(baseParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, finalParams);
    }

    private double[] ComputeTaskEmbedding(TInput supportInput)
    {
        var features = ConvertToVector(MetaModel.Predict(supportInput));
        var embed = new double[_embeddingDim];
        if (features == null) return embed;

        for (int i = 0; i < _embeddingDim && i < features.Length; i++)
            embed[i] = NumOps.ToDouble(features[i]);
        return embed;
    }

    private double[] ComputeGatingScores(double[] taskEmbed)
    {
        var scores = new double[_bankSize];
        int biasOffset = _embeddingDim * _bankSize;
        for (int k = 0; k < _bankSize; k++)
        {
            double sum = 0;
            for (int i = 0; i < _embeddingDim; i++)
                sum += taskEmbed[i] * NumOps.ToDouble(_gatingParams[k * _embeddingDim + i]);
            if (biasOffset + k < _gatingParams.Length)
                sum += NumOps.ToDouble(_gatingParams[biasOffset + k]);
            scores[k] = sum;
        }
        return scores;
    }

    private (int[] indices, double[] weights) SelectTopK(double[] scores)
    {
        // Find top-K indices
        var indexScorePairs = new (int index, double score)[_bankSize];
        for (int k = 0; k < _bankSize; k++)
            indexScorePairs[k] = (k, scores[k]);
        Array.Sort(indexScorePairs, (a, b) => b.score.CompareTo(a.score));

        var indices = new int[_topK];
        var topScores = new double[_topK];
        for (int k = 0; k < _topK; k++)
        {
            indices[k] = indexScorePairs[k].index;
            topScores[k] = indexScorePairs[k].score;
        }

        // Softmax over top-K scores
        double maxScore = topScores[0];
        double sumExp = 0;
        var weights = new double[_topK];
        for (int k = 0; k < _topK; k++)
        {
            weights[k] = Math.Exp((topScores[k] - maxScore) / _algoOptions.GatingTemperature);
            sumExp += weights[k];
        }
        for (int k = 0; k < _topK; k++)
            weights[k] /= sumExp;

        return (indices, weights);
    }

    private Vector<T> CombineModules(int[] selectedIndices, double[] weights)
    {
        return CombineModulesWithWeights(selectedIndices, weights);
    }

    private Vector<T> CombineModulesWithWeights(int[] selectedIndices, double[] weights)
    {
        var delta = new Vector<T>(_paramDim);
        int moduleSize = _rank * _paramDim;

        for (int k = 0; k < selectedIndices.Length; k++)
        {
            if (Math.Abs(weights[k]) < 1e-10) continue;
            int moduleOffset = selectedIndices[k] * moduleSize;

            // Sum all rank basis vectors within the module, weighted by gating weight
            for (int r = 0; r < _rank; r++)
            {
                int basisOffset = moduleOffset + r * _paramDim;
                double w = weights[k] / _rank; // Average over rank vectors
                T wT = NumOps.FromDouble(w);
                for (int d = 0; d < _paramDim; d++)
                {
                    if (basisOffset + d < _moduleBank.Length)
                        delta[d] = NumOps.Add(delta[d], NumOps.Multiply(_moduleBank[basisOffset + d], wT));
                }
            }
        }
        return delta;
    }

    private double ComputeModuleGradient(Vector<T> fullGrad, int moduleIndex)
    {
        // Dot product of full gradient with module's mean basis vector
        int moduleOffset = moduleIndex * _rank * _paramDim;
        double dot = 0;
        for (int r = 0; r < _rank; r++)
        {
            int basisOffset = moduleOffset + r * _paramDim;
            for (int d = 0; d < _paramDim && basisOffset + d < _moduleBank.Length; d++)
                dot += NumOps.ToDouble(fullGrad[d]) * NumOps.ToDouble(_moduleBank[basisOffset + d]);
        }
        return dot / _rank;
    }

    private Vector<T> AddVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            result[d] = NumOps.Add(a[d], b[d]);
        return result;
    }

    private static void NormalizeWeights(double[] weights)
    {
        double sum = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = Math.Max(0, weights[i]); // Ensure non-negative
            sum += weights[i];
        }
        if (sum > 1e-10)
            for (int i = 0; i < weights.Length; i++) weights[i] /= sum;
        else
            for (int i = 0; i < weights.Length; i++) weights[i] = 1.0 / weights.Length;
    }

    private static double ComputeLoadBalanceLoss(double[] usageCounts, int numTasks)
    {
        if (numTasks == 0) return 0;
        double mean = 0;
        for (int i = 0; i < usageCounts.Length; i++) mean += usageCounts[i];
        mean /= usageCounts.Length;

        double variance = 0;
        for (int i = 0; i < usageCounts.Length; i++)
        {
            double diff = usageCounts[i] / numTasks - mean / numTasks;
            variance += diff * diff;
        }
        return variance / usageCounts.Length;
    }

    private double ComputeBankLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(baseParams);
            var taskEmbed = ComputeTaskEmbedding(task.SupportInput);
            var scores = ComputeGatingScores(taskEmbed);
            var (selectedIndices, gatingWeights) = SelectTopK(scores);
            var delta = CombineModulesWithWeights(selectedIndices, gatingWeights);
            var adaptedParams = AddVectors(baseParams, delta);
            MetaModel.SetParameters(adaptedParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
