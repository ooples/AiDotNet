using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Dynamic Task Sampling: difficulty-aware gradient reweighting for meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// Dynamic Task Sampling maintains running difficulty estimates for tasks in the meta-batch
/// and uses difficulty-proportional weighting on meta-gradients. Tasks with higher post-adaptation
/// query loss (= harder tasks) receive higher gradient weights, focusing meta-learning
/// on areas where the model struggles most. An exploration bonus (UCB-style) ensures that
/// tasks seen less frequently still receive gradient signal.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Per-task difficulty: d_i = EMA(d_i, L_query_i)
/// Task visit count: n_i (for UCB exploration)
///
/// Gradient weight: w_i = softmax(d_i / τ + ExplorationCoeff * sqrt(log(N) / n_i))
///   where N = total meta-iterations
///
/// Weighted meta-gradient: g = Σ_i w_i * grad_i
///
/// Outer loop: θ ← θ - η * g
/// </code>
/// </para>
/// </remarks>
public class DynamicTaskSamplingAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly DynamicTaskSamplingOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Running difficulty estimate per task slot in meta-batch.</summary>
    private double[] _taskDifficulty;

    /// <summary>Visit count per task slot (for UCB exploration).</summary>
    private int[] _taskVisits;

    /// <summary>Total meta-iteration count.</summary>
    private int _totalIterations;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DynamicTaskSampling;

    public DynamicTaskSamplingAlgorithm(DynamicTaskSamplingOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _taskDifficulty = new double[options.MetaBatchSize];
        _taskVisits = new int[options.MetaBatchSize];
        _totalIterations = 0;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var taskLossValues = new List<double>();
        var initParams = MetaModel.GetParameters();
        _totalIterations++;

        // Ensure arrays are large enough
        if (_taskDifficulty.Length < taskBatch.Tasks.Length)
        {
            _taskDifficulty = new double[taskBatch.Tasks.Length];
            _taskVisits = new int[taskBatch.Tasks.Length];
        }

        for (int t = 0; t < taskBatch.Tasks.Length; t++)
        {
            var task = taskBatch.Tasks[t];
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
            double lossVal = NumOps.ToDouble(queryLoss);

            // Update difficulty estimate
            _taskVisits[t]++;
            _taskDifficulty[t] = _algoOptions.DifficultyDecay * _taskDifficulty[t]
                               + (1.0 - _algoOptions.DifficultyDecay) * lossVal;

            losses.Add(queryLoss);
            taskLossValues.Add(lossVal);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Compute difficulty-proportional weights with UCB exploration
        int numTasks = taskBatch.Tasks.Length;
        var weights = new double[numTasks];
        double maxWeight = double.NegativeInfinity;
        double logN = Math.Log(_totalIterations + 1);

        for (int t = 0; t < numTasks; t++)
        {
            double ucbBonus = _algoOptions.ExplorationCoeff
                            * Math.Sqrt(logN / Math.Max(_taskVisits[t], 1));
            weights[t] = _taskDifficulty[t] / _algoOptions.TaskTemperature + ucbBonus;
            if (weights[t] > maxWeight) maxWeight = weights[t];
        }

        // Softmax normalization
        double sumExp = 0;
        for (int t = 0; t < numTasks; t++) { weights[t] = Math.Exp(weights[t] - maxWeight); sumExp += weights[t]; }
        for (int t = 0; t < numTasks; t++) weights[t] /= (sumExp + 1e-10);

        // Weighted meta-gradient
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var weightedGrad = new Vector<T>(_paramDim);
            for (int t = 0; t < numTasks; t++)
                for (int d = 0; d < _paramDim; d++)
                    weightedGrad[d] = NumOps.Add(weightedGrad[d],
                        NumOps.FromDouble(weights[t] * NumOps.ToDouble(metaGradients[t][d])));

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
}
