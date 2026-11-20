using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning with Differentiable Convex Optimization",
    "https://arxiv.org/abs/1904.03758",
    Year = 2019,
    Authors = "Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, Stefano Soatto")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class DynamicTaskSamplingAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly DynamicTaskSamplingOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Running difficulty estimate per task slot in meta-batch.</summary>
    [Scratch]
    private Vector<T> _taskDifficulty;

    /// <summary>Visit count per task slot (for UCB exploration).</summary>
    private int[] _taskVisits;

    /// <summary>Total meta-iteration count.</summary>
    private int _totalIterations;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DynamicTaskSampling;

    public DynamicTaskSamplingAlgorithm(DynamicTaskSamplingOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.");
        if (options.TaskTemperature <= 0)
            throw new ArgumentException("TaskTemperature must be positive.", nameof(options));
        if (options.DifficultyDecay <= 0 || options.DifficultyDecay >= 1)
            throw new ArgumentException("DifficultyDecay must be in (0, 1).", nameof(options));

        _taskDifficulty = new Vector<T>(options.MetaBatchSize);
        _taskVisits = new int[options.MetaBatchSize];
        _totalIterations = 0;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0)
            return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        _totalIterations++;

        // Ensure arrays are large enough
        if (_taskDifficulty.Length < taskBatch.Tasks.Length)
        {
            _taskDifficulty = new Vector<T>(taskBatch.Tasks.Length);
            _taskVisits = new int[taskBatch.Tasks.Length];
        }

        try
        {
            for (int t = 0; t < taskBatch.Tasks.Length; t++)
            {
                var task = taskBatch.Tasks[t];
                var adaptedParams = initParams.Clone();

                for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
                {
                    ParamModel.SetParameters(adaptedParams);
                    var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                    adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
                }

                ParamModel.SetParameters(adaptedParams);
                var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

                // The difficulty estimate of slot t is updated from the collected losses in UpdateDifficulty.
                _taskVisits[t]++;

                losses.Add(queryLoss);
                metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
            }
        }
        finally
        {
            ParamModel.SetParameters(initParams);
            UpdateDifficulty(losses);
        }

        // Difficulty-proportional weights with UCB exploration:
        // w_t = difficulty_t / temperature + c * sqrt(log(N + 1) / max(visits_t, 1)), then a softmax.
        int numTasks = taskBatch.Tasks.Length;
        var visits = new int[numTasks];
        Array.Copy(_taskVisits, visits, numTasks);
        var visitCounts = Engine.Max(new Tensor<int>(visits, new[] { numTasks }).Cast<T>().ToVector(), Engine.Fill(numTasks, NumOps.One));
        var ucbBonus = Engine.Multiply(Engine.ReciprocalSqrt(visitCounts),
            NumOps.FromDouble(_algoOptions.ExplorationCoeff * Math.Sqrt(Math.Log(_totalIterations + 1))));
        var logits = Engine.Add(
            Engine.Divide(_taskDifficulty.GetSubVector(0, numTasks), NumOps.FromDouble(_algoOptions.TaskTemperature)), ucbBonus);

        // Softmax normalization (max-shifted, with the epsilon in the denominator)
        const double SoftmaxEpsilon = 1e-10;
        var logitsTensor = Tensor<T>.FromVector(logits);
        var exps = Engine.TensorExp(Engine.TensorAddScalar(logitsTensor, NumOps.Negate(Engine.TensorMaxValue(logitsTensor))));
        var weights = Engine.TensorDivideScalar(exps, NumOps.Add(Engine.TensorSum(exps), NumOps.FromDouble(SoftmaxEpsilon)));

        // Weighted meta-gradient: sum_t w_t g_t
        if (metaGradients.Count > 0)
        {
            var gradRows = new Tensor<T>[numTasks];
            for (int t = 0; t < numTasks; t++)
                gradRows[t] = Tensor<T>.FromVector(metaGradients[t]);
            var weightedGrad = Engine.TensorMatMul(
                Engine.Reshape(weights, new[] { 1, numTasks }), Engine.TensorStack(gradRows, 0)).ToVector();

            ParamModel.SetParameters(ApplyOuterGradients(initParams, weightedGrad, _algoOptions.OuterLearningRate));
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var adaptedParams = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// EMA difficulty update for the first <c>losses.Count</c> task slots:
    /// difficulty_t = decay * difficulty_t + (1 - decay) * loss_t.
    /// </summary>
    private void UpdateDifficulty(List<T> losses)
    {
        int n = losses.Count;
        if (n == 0)
            return;

        var updated = Engine.Add(
            Engine.Multiply(_taskDifficulty.GetSubVector(0, n), NumOps.FromDouble(_algoOptions.DifficultyDecay)),
            Engine.Multiply(new Vector<T>(losses), NumOps.FromDouble(1.0 - _algoOptions.DifficultyDecay)));
        _taskDifficulty = n == _taskDifficulty.Length
            ? updated
            : Vector<T>.Concatenate(updated, _taskDifficulty.GetSubVector(n, _taskDifficulty.Length - n));
    }
}
