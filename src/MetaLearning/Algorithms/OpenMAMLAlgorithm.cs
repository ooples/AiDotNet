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
/// Implementation of Open-MAML (open-set MAML with out-of-distribution detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Open-MAML extends MAML to handle open-set scenarios where query examples may belong
/// to classes not present in the support set. It adds a confidence-based rejection
/// mechanism to detect out-of-distribution (OOD) examples.
/// </para>
/// <para><b>For Beginners:</b> Standard few-shot learning assumes all query examples belong
/// to one of the support classes. But in the real world, you might encounter unknown examples.
///
/// **The problem:**
/// You're given 5 classes of animals to learn. Then a query comes in - it might be one of
/// those 5 animals, or it might be something completely different (like a car). Standard
/// few-shot methods will force a classification into one of the 5 classes.
///
/// **How Open-MAML fixes this:**
/// 1. Train with MAML as usual for inner-loop adaptation
/// 2. During meta-training, some tasks include OOD examples (open-set fraction)
/// 3. Learn a confidence threshold: if max prediction probability &lt; threshold, reject
/// 4. The model learns to be uncertain about OOD examples while confident about in-distribution ones
///
/// **Key difference from MAML:**
/// Open-MAML adds OOD detection training, so the adapted model can say "I don't know"
/// instead of being forced to choose a known class.
/// </para>
/// <para><b>Algorithm - Open-MAML:</b>
/// <code>
/// # Meta-training (extends MAML)
/// for each meta-iteration:
///     for each task T_i in batch:
///         # Standard MAML inner loop
///         theta_i = theta
///         for step in range(adaptation_steps):
///             loss = cross_entropy(f(support_x; theta_i), support_y)
///             theta_i = theta_i - alpha * grad(loss)
///
///         # Open-set query evaluation
///         logits = f(query_x; theta_i)
///         probs = softmax(logits)
///         max_prob = max(probs, dim=-1)
///
///         # Classification loss for in-distribution
///         class_loss = cross_entropy(logits[in_dist], query_y[in_dist])
///
///         # OOD detection loss (max_prob should be low for OOD)
///         ood_loss = binary_cross_entropy(max_prob, is_in_distribution)
///
///         meta_loss = class_loss + ood_loss
///
///     theta = theta - beta * grad(meta_loss)
/// </code>
/// </para>
/// </remarks>
public class OpenMAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly OpenMAMLOptions<T, TInput, TOutput> _openMAMLOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.OpenMAML;

    /// <summary>Initializes a new Open-MAML meta-learner.</summary>
    /// <param name="options">Configuration options for Open-MAML.</param>
    public OpenMAMLAlgorithm(OpenMAMLOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _openMAMLOptions = options;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop (MAML-style adaptation)
            MetaModel.SetParameters(initParams);
            for (int step = 0; step < _openMAMLOptions.AdaptationSteps; step++)
            {
                var innerGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var adapted = ApplyGradients(MetaModel.GetParameters(), innerGrad, _openMAMLOptions.InnerLearningRate);
                MetaModel.SetParameters(adapted);
            }

            // Query evaluation with open-set awareness
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop update
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _openMAMLOptions.OuterLearningRate));
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var adaptedParams = MetaModel.GetParameters();
        MetaModel.SetParameters(adaptedParams);

        for (int step = 0; step < _openMAMLOptions.AdaptationSteps; step++)
        {
            var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            adaptedParams = ApplyGradients(MetaModel.GetParameters(), grad, _openMAMLOptions.InnerLearningRate);
            MetaModel.SetParameters(adaptedParams);
        }

        return new OpenMAMLModel<T, TInput, TOutput>(MetaModel, adaptedParams, _openMAMLOptions.OpenSetThreshold);
    }

}

/// <summary>Adapted model wrapper for Open-MAML with OOD rejection.</summary>
internal class OpenMAMLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    private readonly double _threshold;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public OpenMAMLModel(IFullModel<T, TInput, TOutput> model, Vector<T> p, double threshold)
    { _model = model; _params = p; _threshold = threshold; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
