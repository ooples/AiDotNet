using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation
/// Based on Meta Learning (Zhang et al., NAACL 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// AutoLoRA addresses LoRA's limitation of uniform rank assignment across all layers by
/// using a bi-level optimization to automatically discover the optimal rank for each
/// parameter group.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// State: base params θ, rank-1 component bank {Δ_g,j}, selection logits {β_g,j}
///
/// Inner loop (weight update on support/train data):
///   For each group g, compute selection weights: α_g,j = softmax(β_g / temperature)
///   Δθ_g = Σ_j α_g,j * Δ_g,j  (weighted sum of rank-1 components)
///   θ' = θ + Δθ
///   Update {Δ_g,j} via gradient descent on support loss L(θ')
///
/// Outer loop (rank selection on query/val data):
///   Update {β_g,j} via gradient descent on query loss L(θ')
///   Add rank regularization: L_reg = Σ_g Σ_j α_g,j  (encourages sparsity)
///
/// Rank determination (after meta-training):
///   For group g: effective_rank = |{j : α_g,j ≥ threshold}|
/// </code>
/// </para>
/// </remarks>
public class AutoLoRAAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly AutoLoRAOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// Rank-1 components for all groups. Stored as flat vector:
    /// [group_0_comp_0, group_0_comp_1, ..., group_G_comp_R]
    /// Each component has length paramsPerGroup.
    /// </summary>
    private Vector<T> _rankComponents;

    /// <summary>
    /// Selection logits β for each (group, component) pair.
    /// Length = numGroups * maxRank.
    /// </summary>
    private Vector<T> _selectionLogits;

    private readonly int _paramDim;
    private readonly int _numGroups;
    private readonly int _maxRank;
    private readonly int _paramsPerGroup;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.AutoLoRA;

    public AutoLoRAAlgorithm(AutoLoRAOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _numGroups = Math.Max(1, options.NumRankGroups);
        _maxRank = Math.Max(1, options.MaxRank);
        _paramsPerGroup = (_paramDim + _numGroups - 1) / _numGroups;

        // Initialize rank-1 components (small random values)
        int totalComponents = _numGroups * _maxRank * _paramsPerGroup;
        _rankComponents = new Vector<T>(totalComponents);
        double initScale = 0.01;
        for (int i = 0; i < totalComponents; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            _rankComponents[i] = NumOps.FromDouble(z * initScale);
        }

        // Initialize selection logits to uniform (all components equally likely)
        _selectionLogits = new Vector<T>(_numGroups * _maxRank);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = MetaModel.GetParameters();

        // Compute current selection weights via softmax
        var selectionWeights = ComputeSelectionWeights();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: update rank components on support data
            // Compose adapted params using current selection weights
            var adaptedParams = ComposeAdaptedParams(baseParams, selectionWeights);
            MetaModel.SetParameters(adaptedParams);

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Update rank components (inner loop) by projecting gradient onto each component
                UpdateRankComponents(grad, baseParams, selectionWeights);

                // Recompose with updated components
                adaptedParams = ComposeAdaptedParams(baseParams, selectionWeights);
                MetaModel.SetParameters(adaptedParams);
            }

            // Evaluate on query set
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Add rank regularization: penalize active rank (encourages sparsity)
            double rankReg = ComputeRankRegularization(selectionWeights);
            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.RankRegularization * rankReg));
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

        // Update selection logits and rank components via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _selectionLogits, _algoOptions.OuterLearningRate, ComputeAutoLoRALoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _rankComponents, _algoOptions.OuterLearningRate * 0.1, ComputeAutoLoRALoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();
        var selectionWeights = ComputeSelectionWeights();

        // Apply thresholding: only use components with α ≥ threshold (rank determination)
        var thresholdedWeights = ApplyThreshold(selectionWeights);

        var adaptedParams = ComposeAdaptedParams(baseParams, thresholdedWeights);
        MetaModel.SetParameters(adaptedParams);

        // Fine-tune rank components on support set
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            UpdateRankComponents(grad, baseParams, thresholdedWeights);
            adaptedParams = ComposeAdaptedParams(baseParams, thresholdedWeights);
            MetaModel.SetParameters(adaptedParams);
        }

        MetaModel.SetParameters(baseParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Computes softmax selection weights for each (group, rank) pair.
    /// Returns array[group][rank].
    /// </summary>
    private double[][] ComputeSelectionWeights()
    {
        var weights = new double[_numGroups][];
        for (int g = 0; g < _numGroups; g++)
        {
            weights[g] = new double[_maxRank];
            double maxLogit = double.NegativeInfinity;
            for (int j = 0; j < _maxRank; j++)
            {
                double logit = NumOps.ToDouble(_selectionLogits[g * _maxRank + j]);
                if (logit > maxLogit) maxLogit = logit;
                weights[g][j] = logit;
            }

            double sumExp = 0;
            for (int j = 0; j < _maxRank; j++)
            {
                weights[g][j] = Math.Exp(weights[g][j] - maxLogit);
                sumExp += weights[g][j];
            }
            for (int j = 0; j < _maxRank; j++)
                weights[g][j] /= sumExp;
        }
        return weights;
    }

    /// <summary>
    /// Composes adapted parameters by adding weighted rank-1 components to base params.
    /// </summary>
    private Vector<T> ComposeAdaptedParams(Vector<T> baseParams, double[][] selectionWeights)
    {
        var adapted = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            adapted[d] = baseParams[d];

        for (int g = 0; g < _numGroups; g++)
        {
            int groupStart = g * _paramsPerGroup;
            int groupEnd = Math.Min(groupStart + _paramsPerGroup, _paramDim);

            for (int j = 0; j < _maxRank; j++)
            {
                double w = selectionWeights[g][j];
                if (w < 1e-10) continue;

                int compOffset = (g * _maxRank + j) * _paramsPerGroup;
                T wT = NumOps.FromDouble(w);

                for (int d = groupStart; d < groupEnd; d++)
                {
                    int compIdx = compOffset + (d - groupStart);
                    if (compIdx < _rankComponents.Length)
                        adapted[d] = NumOps.Add(adapted[d], NumOps.Multiply(_rankComponents[compIdx], wT));
                }
            }
        }

        return adapted;
    }

    /// <summary>
    /// Updates rank-1 components using projected gradients from the full parameter gradient.
    /// </summary>
    private void UpdateRankComponents(Vector<T> fullGrad, Vector<T> baseParams, double[][] selectionWeights)
    {
        for (int g = 0; g < _numGroups; g++)
        {
            int groupStart = g * _paramsPerGroup;
            int groupEnd = Math.Min(groupStart + _paramsPerGroup, _paramDim);

            for (int j = 0; j < _maxRank; j++)
            {
                double w = selectionWeights[g][j];
                if (w < 1e-10) continue;

                int compOffset = (g * _maxRank + j) * _paramsPerGroup;
                double lr = _algoOptions.InnerLearningRate * w;

                for (int d = groupStart; d < groupEnd; d++)
                {
                    int compIdx = compOffset + (d - groupStart);
                    if (compIdx < _rankComponents.Length)
                    {
                        double gradVal = NumOps.ToDouble(fullGrad[d]);
                        _rankComponents[compIdx] = NumOps.Subtract(_rankComponents[compIdx],
                            NumOps.FromDouble(lr * gradVal));
                    }
                }
            }
        }
    }

    /// <summary>
    /// Applies threshold to selection weights, zeroing out components below threshold.
    /// </summary>
    private double[][] ApplyThreshold(double[][] selectionWeights)
    {
        var thresholded = new double[_numGroups][];
        for (int g = 0; g < _numGroups; g++)
        {
            thresholded[g] = new double[_maxRank];
            double sum = 0;
            for (int j = 0; j < _maxRank; j++)
            {
                thresholded[g][j] = selectionWeights[g][j] >= _algoOptions.RankThreshold ? selectionWeights[g][j] : 0;
                sum += thresholded[g][j];
            }
            // Re-normalize
            if (sum > 1e-10)
                for (int j = 0; j < _maxRank; j++) thresholded[g][j] /= sum;
            else
                for (int j = 0; j < _maxRank; j++) thresholded[g][j] = 1.0 / _maxRank;
        }
        return thresholded;
    }

    private double ComputeRankRegularization(double[][] selectionWeights)
    {
        double total = 0;
        for (int g = 0; g < _numGroups; g++)
            for (int j = 0; j < _maxRank; j++)
                total += selectionWeights[g][j]; // L1 on selection weights
        return total;
    }

    private double ComputeAutoLoRALoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = MetaModel.GetParameters();
        var selectionWeights = ComputeSelectionWeights();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = ComposeAdaptedParams(baseParams, selectionWeights);
            MetaModel.SetParameters(adaptedParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        totalLoss += _algoOptions.RankRegularization * ComputeRankRegularization(selectionWeights);
        MetaModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
