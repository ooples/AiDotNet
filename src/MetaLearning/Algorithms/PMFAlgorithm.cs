using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

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
/// 1. Pre-training: the feature extractor arrives pretrained (the model passed in as <c>MetaModel</c>).
/// 2. Meta-training: episodic training of that extractor as a ProtoNet with a cosine classifier. Each class's
///    prototype is the mean of its support features, and the logits of a query are its cosine similarity to each
///    prototype divided by a learnable temperature (<see cref="PMFOptions{T, TInput, TOutput}.Temperature"/>).
/// 3. Fine-tuning: optional per-task fine-tuning of the extractor on the support set with the same ProtoNet loss,
///    before the task's prototypes are taken.
/// </para>
/// <para>
/// Meta-training used to be first-order MAML: inner gradient steps on the configured loss of the extractor's raw
/// output, which never formed a prototype, and <see cref="PMFOptions{T, TInput, TOutput}.DistanceMetric"/> was a
/// string nothing read.
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference",
    "https://arxiv.org/abs/2204.07305",
    Year = 2022,
    Authors = "Shell Xu Hu, Da Li, Jan Stuhmer, Minyoung Kim, Timothy M. Hospedales")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class PMFAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly PMFOptions<T, TInput, TOutput> _pmfOptions;

    /// <summary>
    /// The cosine classifier's scale <c>1 / temperature</c>, one value, learned during meta-training when
    /// <see cref="PMFOptions{T, TInput, TOutput}.LearnTemperature"/> is set (the reference ProtoNet's <c>scale_cls</c>).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _cosineScale = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.PMF;

    /// <summary>
    /// Initializes a new PMF meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for PMF.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the temperature is not a positive finite number.</exception>
    public PMFAlgorithm(PMFOptions<T, TInput, TOutput> options)
        : base(
            (options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
            options.LossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _pmfOptions = options;
        if (double.IsNaN(options.Temperature) || options.Temperature <= 0 || double.IsInfinity(options.Temperature))
            throw new ArgumentException("Temperature must be a positive finite number.", nameof(options));

        _cosineScale = new Vector<T>(new[] { NumOps.FromDouble(1.0 / options.Temperature) });
        RequirePlainSgdInnerSteps(options.EnableFineTuning, "a separate fine-tuning learning rate");
    }

    /// <summary>Whether meta-training updates the cosine scale.</summary>
    private bool LearnsScale => _pmfOptions.LearnTemperature && _pmfOptions.DistanceMetric == PMFDistanceMetric.Cosine;

    /// <summary>
    /// Performs one meta-training step (Stage 2: M): the ProtoNet loss of each episode's query examples,
    /// differentiated through the prototypes into the feature extractor and into the cosine scale.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average query loss across the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is Stage 2 of PMF. For each episode the support examples of a class are
    /// averaged into a prototype, each query is scored by its cosine similarity to every prototype, and the
    /// extractor is updated so queries score highest against their own class.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var body = ParamModel.GetParameters();
        Vector<T>? bodyGradient = null, scaleGradient = null;
        var losses = new List<T>(taskBatch.Tasks.Length);
        foreach (var task in taskBatch.Tasks)
        {
            var (loss, taskBody, taskScale) = EpisodeGradient(task);
            losses.Add(loss);
            bodyGradient = bodyGradient is null ? taskBody.Clone() : Engine.Add(bodyGradient, taskBody);
            scaleGradient = scaleGradient is null ? taskScale.Clone() : Engine.Add(scaleGradient, taskScale);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        bodyGradient = ClipGradients(Engine.Divide(bodyGradient ?? new Vector<T>(body.Length), batchSize));
        scaleGradient = ClipGradients(Engine.Divide(scaleGradient ?? new Vector<T>(_cosineScale.Length), batchSize));

        if (LearnsScale)
        {
            var updated = ApplyOuterGradients(
                _pmfOptions.OuterLearningRate, (body, bodyGradient), (_cosineScale, scaleGradient));
            ParamModel.SetParameters(updated[0]);
            _cosineScale = updated[1];
        }
        else
        {
            ParamModel.SetParameters(ApplyOuterGradients(body, bodyGradient, _pmfOptions.OuterLearningRate));
        }

        return NumOps.Divide(Engine.Sum(new Vector<T>(losses)), batchSize);
    }

    /// <summary>
    /// Adapts to a new task (Stage 3: F): optionally fine-tunes the extractor on the support set with the ProtoNet
    /// loss, then classifies by the support set's prototypes.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>A prototype classifier over the task's classes.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each new task:
    /// 1. Start from the meta-trained feature extractor
    /// 2. Optionally fine-tune it for a few steps on the support set, scoring the support examples against
    ///    their own class prototypes (Stage 3)
    /// 3. Average each class's support features into its prototype and classify by cosine similarity
    /// </para>
    /// <para>
    /// The paper fine-tunes against augmented copies of the support set; with no augmentation available here, the
    /// support examples themselves are the pseudo-queries.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        var initParams = ParamModel.GetParameters();
        var adaptedParams = initParams;

        if (_pmfOptions.EnableFineTuning && _pmfOptions.FineTuningSteps > 0)
        {
            var supportLabels = PMFClassifier<T>.ReadLabels(task.SupportOutput);
            var episode = PrototypeEpisode<T>.Build(supportLabels, supportLabels);
            var stacked = ClassifierOutputs<T>.StackRows(task.SupportInput, task.SupportInput);
            var scale = Tensor<T>.FromVector(_cosineScale);
            adaptedParams = initParams.Clone();
            try
            {
                for (int step = 0; step < _pmfOptions.FineTuningSteps; step++)
                {
                    ParamModel.SetParameters(adaptedParams);
                    var grad = BodyGradient(episode, stacked, scale);
                    adaptedParams = ApplyInnerGradients(adaptedParams, grad, _pmfOptions.FineTuningLearningRate, step == 0);
                }
            }
            finally
            {
                ParamModel.SetParameters(initParams); // Restore base parameters
            }
        }

        return new PMFModel<T, TInput, TOutput>(
            MetaModel, adaptedParams, task.SupportInput, task.SupportOutput, _pmfOptions.DistanceMetric,
            _cosineScale.Clone());
    }

    /// <inheritdoc/>
    /// <remarks>
    /// For score outputs (the probabilities a Tensor or Matrix output carries) this is the configured loss on their
    /// logarithm against the class indices - cross-entropy by default. For a Vector output, which carries the
    /// predicted class of each example rather than scores, it is the classification error rate.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    #region Episode

    /// <summary>One episode's query loss and its exact gradient with respect to the extractor and the scale.</summary>
    private (T Loss, Vector<T> Body, Vector<T> Scale) EpisodeGradient(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = PrototypeEpisode<T>.Build(
            PMFClassifier<T>.ReadLabels(task.SupportOutput), PMFClassifier<T>.ReadLabels(task.QueryOutput));
        var stacked = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var scale = Tensor<T>.FromVector(_cosineScale);

        var bodyGradient = BodyGradient(episode, stacked, scale);

        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = ClassifierOutputs<T>.AsRows(MetaModel.Predict(stacked));
        }

        var scaleGradient = new Vector<T>(_cosineScale.Length);
        T loss;
        if (!LearnsScale)
        {
            using var noGrad = new NoGradScope<T>();
            loss = LossFunction.ComputeTapeLoss(EpisodeLogits(embeddings, episode, scale), episode.QueryTarget)[0];
        }
        else
        {
            loss = AiDotNet.Training.TapeGradients<T>.Of(
                () => LossFunction.ComputeTapeLoss(EpisodeLogits(embeddings, episode, scale), episode.QueryTarget),
                new List<Tensor<T>> { scale },
                gradients =>
                {
                    if (gradients.TryGetValue(scale, out var gradient))
                        scaleGradient = gradient.ToVector();
                });
        }

        return (loss, bodyGradient, scaleGradient);
    }

    /// <summary>
    /// The extractor's gradient: the whole episode - prototypes and queries - rebuilt from the embeddings on the
    /// tape and scored against the query labels.
    /// </summary>
    private Vector<T> BodyGradient(PrototypeEpisode<T> episode, TInput stackedInput, Tensor<T> scale)
    {
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));
        var composed = new EmbeddingClassificationLoss<T>(
            embeddings => EpisodeLogits(embeddings, episode, scale),
            LossFunction,
            episode.QueryTarget);
        return ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);
    }

    /// <summary>Logits of an episode's query rows from the stacked support-then-query embeddings.</summary>
    private Tensor<T> EpisodeLogits(Tensor<T> embeddings, PrototypeEpisode<T> episode, Tensor<T> scale)
    {
        var engine = AiDotNetEngine.Current;
        var support = engine.TensorMatMul(episode.SupportSelector, embeddings);
        var query = engine.TensorMatMul(episode.QuerySelector, embeddings);
        var prototypes = PrototypeMetric<T>.Prototypes(support, episode.Membership, null);
        return PMFClassifier<T>.Logits(query, prototypes, _pmfOptions.DistanceMetric, scale, episode.ClassSlots);
    }

    #endregion

    #region Test hooks

    /// <summary>One episode's query logits, <c>[query, classes]</c>, from the current state.</summary>
    internal Tensor<T> EpisodeLogitsForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = PrototypeEpisode<T>.Build(
            PMFClassifier<T>.ReadLabels(task.SupportOutput), PMFClassifier<T>.ReadLabels(task.QueryOutput));
        using var noGrad = new NoGradScope<T>();
        var embeddings = ClassifierOutputs<T>.AsRows(
            MetaModel.Predict(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput)));
        return EpisodeLogits(embeddings, episode, Tensor<T>.FromVector(_cosineScale));
    }

    /// <summary>One episode's query loss and exact gradient from the current state.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> Scale) EpisodeGradientForTesting(IMetaLearningTask<T, TInput, TOutput> task)
        => EpisodeGradient(task);

    /// <summary>Gets or sets a copy of the cosine classifier's scale, <c>1 / temperature</c> (for tests).</summary>
    internal Vector<T> CosineScaleForTesting { get => _cosineScale.Clone(); set => _cosineScale = value.Clone(); }

    #endregion

}

/// <summary>
/// PMF's ProtoNet classifier, written with engine tensor ops so the tape records it when one is live.
/// </summary>
internal static class PMFClassifier<T>
{
    /// <summary>
    /// Logits <c>[query, classes]</c>: <c>scale * cos(q, c_k)</c> for the cosine classifier (<c>scale = 1 / tau</c>),
    /// or <c>-||q - c_k||^2</c> for squared Euclidean.
    /// </summary>
    /// <param name="query">Query embeddings, <c>[query, width]</c>.</param>
    /// <param name="prototypes">Class prototypes (support means), <c>[classes, width]</c>.</param>
    /// <param name="metric">The metric.</param>
    /// <param name="scale">The cosine scale, one value; unused by squared Euclidean.</param>
    /// <param name="classSlots">The class label behind each prototype row.</param>
    public static Tensor<T> Logits(
        Tensor<T> query, Tensor<T> prototypes, PMFDistanceMetric metric, Tensor<T> scale, int[] classSlots)
    {
        if (metric == PMFDistanceMetric.SquaredEuclidean)
        {
            return PrototypeMetric<T>.Logits(
                query, prototypes, ProtoNetsDistanceFunction.Euclidean, 1.0, null, null, classSlots, 1.0);
        }

        var engine = AiDotNetEngine.Current;
        var q = PrototypeMetric<T>.Normalized(query, normalize: true);
        var p = PrototypeMetric<T>.Normalized(prototypes, normalize: true);
        var cosine = engine.TensorMatMul(q, engine.TensorTranspose(p));
        var broadcastScale = engine.TensorBroadcastTo(
            engine.Reshape(scale, new[] { 1, 1 }), new[] { cosine.Shape[0], cosine.Shape[1] });
        return engine.TensorMultiply(cosine, broadcastScale);
    }

    /// <summary>The class index of each label, rounding half to even as Math.Round does.</summary>
    public static int[] ReadLabels(object? labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, int.MaxValue);
        return AiDotNetEngine.Current.TensorRound(tensor).Cast<int>().ToArray();
    }
}

/// <summary>
/// Adapted model for PMF: its own copy of the (optionally fine-tuned) extractor, the support set's class
/// prototypes, and the cosine scale. <see cref="Predict"/> returns class probabilities, <c>[rows, classes]</c>, for
/// Tensor and Matrix outputs, and the most probable class of each example for a Vector output.
/// </summary>
internal class PMFModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Tensor<T> _prototypes;
    private readonly int[] _classSlots;
    private readonly PMFDistanceMetric _metric;
    private readonly Tensor<T> _scale;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>Builds the prototypes of the support set under the adapted parameters.</summary>
    public PMFModel(IFullModel<T, TInput, TOutput> model, Vector<T> adaptedParams, TInput supportInput,
        TOutput supportOutput, PMFDistanceMetric metric, Vector<T> scale)
    {
        // A copy: the meta-learner's MetaModel keeps its own parameters.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).DeepCopy();
        InterfaceGuard.Parameterizable(_model).SetParameters(adaptedParams);
        _metric = metric;
        _scale = Tensor<T>.FromVector(scale);

        var supportLabels = PMFClassifier<T>.ReadLabels(supportOutput);
        if (supportLabels.Length == 0)
            throw new ArgumentException("The support set is empty, so there are no prototypes.", nameof(supportOutput));

        var episode = PrototypeEpisode<T>.Build(supportLabels, Array.Empty<int>());
        _classSlots = episode.ClassSlots;
        using var noGrad = new NoGradScope<T>();
        var support = ClassifierOutputs<T>.AsRows(_model.Predict(supportInput));
        _prototypes = PrototypeMetric<T>.Prototypes(support, episode.Membership, null);
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var query = ClassifierOutputs<T>.AsRows(_model.Predict(input));
        var probabilities = engine.Softmax(
            PMFClassifier<T>.Logits(query, _prototypes, _metric, _scale, _classSlots), axis: 1);

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            // The most probable column of each row (the first on a tie), read out as its class label.
            if (probabilities.Shape[0] == 0) return (TOutput)(object)new Vector<T>(0);
            var best = engine.TensorArgMax(probabilities, 1);
            var slots = PrototypeEpisode<T>.IndexTensor(_classSlots).Cast<T>();
            return (TOutput)(object)engine.TensorIndexSelect(slots, best, 0).ToVector();
        }

        return ClassifierOutputs<T>.ToOutput<TOutput>(probabilities);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
