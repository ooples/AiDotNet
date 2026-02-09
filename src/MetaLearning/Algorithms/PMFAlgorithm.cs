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
/// Implementation of PMF (P>M>F: Pre-training, Meta-training, Fine-tuning).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// PMF implements a three-stage training pipeline:
/// 1. Pre-training: Standard supervised training on base classes
/// 2. Meta-training: Episodic training for few-shot adaptation
/// 3. Fine-tuning: Optional per-task fine-tuning during adaptation
/// </para>
/// <para><b>For Beginners:</b> PMF combines the best of three worlds:
///
/// **Pre-training:** Learn general features from lots of data (foundation)
/// **Meta-training:** Learn to adapt quickly from few examples (specialization)
/// **Fine-tuning:** Squeeze out extra accuracy on each test task (polish)
///
/// This simple pipeline achieves remarkable results because each stage
/// addresses a different aspect of few-shot learning.
/// </para>
/// <para>
/// Reference: Hu, S.X., Li, D., Stuhmer, J., Kim, M., &amp; Hospedales, T.M. (2022).
/// Pushing the Limits of Simple Pipelines for Few-Shot Learning. ICLR 2022.
/// </para>
/// </remarks>
public class PMFAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly PMFOptions<T, TInput, TOutput> _pmfOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.PMF;

    /// <summary>
    /// Initializes a new PMF meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for PMF.</param>
    public PMFAlgorithm(PMFOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _pmfOptions = options;
    }

    /// <summary>
    /// Performs one meta-training step (Stage 2: M).
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is Stage 2 of PMF where the pretrained model
    /// learns to adapt quickly. It uses MAML-style inner/outer loop training.
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

            // Inner loop adaptation
            var taskParams = new Vector<T>(initParams.Length);
            for (int i = 0; i < initParams.Length; i++)
                taskParams[i] = initParams[i];

            for (int step = 0; step < _pmfOptions.AdaptationSteps; step++)
            {
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                taskParams = ApplyGradients(taskParams, grad, _pmfOptions.InnerLearningRate);
                MetaModel.SetParameters(taskParams);
            }

            // Outer loop loss on query set
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            MetaModel.SetParameters(initParams);
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _pmfOptions.OuterLearningRate));
        }

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task with optional fine-tuning (Stage 3: F).
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model after inner-loop adaptation and optional fine-tuning.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each new task:
    /// 1. Start from the meta-learned initialization
    /// 2. Run gradient descent on the support set (inner loop)
    /// 3. Optionally fine-tune for extra steps (Stage 3)
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var adaptedParams = MetaModel.GetParameters();

        // Inner-loop adaptation
        for (int step = 0; step < _pmfOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            adaptedParams = ApplyGradients(adaptedParams, grad, _pmfOptions.InnerLearningRate);
        }

        // Optional fine-tuning (Stage 3)
        if (_pmfOptions.EnableFineTuning)
        {
            for (int step = 0; step < _pmfOptions.FineTuningSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                adaptedParams = ApplyGradients(adaptedParams, grad, _pmfOptions.FineTuningLearningRate);
            }
        }

        return new PMFModel<T, TInput, TOutput>(MetaModel, adaptedParams);
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

/// <summary>Adapted model wrapper for PMF.</summary>
internal class PMFModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public PMFModel(IFullModel<T, TInput, TOutput> model, Vector<T> adaptedParams)
    { _model = model; _params = adaptedParams; }

    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
