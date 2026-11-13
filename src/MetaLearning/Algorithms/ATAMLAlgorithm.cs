using System;
using System.Collections.Generic;
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
using AiDotNet.Validation;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of ATAML: Attentive Task-Agnostic Meta-Learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Jiang et al. 2018 split a MAML-style learner in two. The shared parameters <c>theta_E</c> - the encoder -
/// learn a task-agnostic representation and are never adapted per task. The task-specific parameters
/// <c>theta_T = {theta_ATT, theta_W}</c> - a content-based attention vector and a softmax classifier - are the
/// only ones the inner loop touches. The meta-update then moves both, through the loss measured AFTER
/// adaptation (Algorithm 1, lines 5-8).
/// </para>
/// <para><b>The attentive base learner (eq. 4-6).</b> The encoder maps an input to states <c>s_t</c>. Each
/// state is scored by an inner product with the attention vector, <c>alpha_t = theta_ATT . s_t</c>, which
/// rescales it, and the rescaled states are averaged into one context vector:
/// <c>c = mean_t(alpha_t s_t)</c>. The context is classified by <c>softmax(c; theta_W)</c>. Attention here
/// weights the ENCODER'S REPRESENTATION - in the paper, the words of a document.
/// </para>
/// <para>
/// <b>What this replaced.</b> The previous implementation bucketed the GRADIENT into fixed-size chunks, took a
/// softmax over those buckets, and used the result as per-parameter learning-rate multipliers while adapting
/// EVERY parameter. That is a learned step size - essentially Meta-SGD - not attention over a representation,
/// and it had neither the shared/task-specific split nor a classifier head. The attention projection was also
/// trained by simultaneous perturbation. None of that is in the paper.
/// </para>
/// <para>
/// <b>For Beginners:</b> ATAML learns one shared "reader" that stays fixed for every task, plus a small
/// per-task part that decides which bits of what was read matter and turns them into a class. Only the small
/// part changes when it meets a new task.
/// </para>
/// <para>
/// Reference: Jiang, X., Havaei, M., Chartrand, G., et al. (2018). On the Importance of Attention in
/// Meta-Learning for Few-Shot Text Classification.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("On the Importance of Attention in Meta-Learning for Few-Shot Text Classification",
    "https://arxiv.org/abs/1806.00852",
    Year = 2018,
    Authors = "Xiang Jiang, Mohammad Havaei, Gabriel Chartrand, Hassan Chouaib, Thomas Vincent, Andrew Jesson, Nicolas Chapados, Stan Matwin")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ATAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ATAMLOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// theta_T, flat: the attention vector <c>[width]</c>, then the classifier <c>[classes, width]</c> and its
    /// bias <c>[classes]</c>. Empty until the first episode shows the encoder's state width.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _taskParameters = new Vector<T>(0);

    /// <summary>The encoder's per-example state width; zero before the first episode.</summary>
    private int _stateWidth;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ATAML;

    /// <summary>Initializes a new ATAML meta-learner.</summary>
    /// <param name="options">ATAML options.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the configuration is invalid.</exception>
    public ATAMLAlgorithm(ATAMLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               // Jiang et al. eq. 6 classifies with a softmax, so the objective is cross-entropy. The inner
               // model's DefaultLossFunction is squared error for every embedding model here, which is a
               // regression objective. The caller's own LossFunction still wins when supplied.
               options.LossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>(),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _algoOptions = options;
        if (!options.IsValid())
        {
            throw new ArgumentException("ATAML configuration is invalid. Check all parameters.", nameof(options));
        }
    }

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _algoOptions;

    /// <summary>
    /// Algorithm 1: adapt theta_T on each task's support set, measure the query loss after adaptation, and move
    /// the shared encoder and the task parameters' initialisation by that loss.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on.</param>
    /// <returns>The average post-adaptation query loss across the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        EnsureTaskParameterShape(taskBatch.Tasks);

        var shared = ParamModel.GetParameters();
        Vector<T>? sharedGradient = null, taskGradient = null;
        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Line 5: theta_T,i = theta_T - eta * grad_{theta_T} L(support). Only theta_T moves.
            var adapted = AdaptTaskParameters(task);

            // Lines 6-8: the meta-loss is the QUERY loss at {theta_T,i, theta_E}. The encoder's gradient comes
            // through the head, so the loss handed to the model is the head-composed one.
            var queryLabels = ReadLabels(task.QueryOutput);
            var queryTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { queryLabels.Length, 1 }));
            var composed = new EmbeddingObjectiveLoss<T>(
                states => HeadLoss(states, queryLabels, Leaves(adapted)));

            sharedGradient = Accumulate(sharedGradient, ComputeGradients(MetaModel, task.QueryInput, queryTarget, composed));

            Tensor<T> queryStates;
            using (new NoGradScope<T>())
            {
                queryStates = ClassifierOutputs<T>.AsRows(MetaModel.Predict(task.QueryInput));
            }

            var leaves = Leaves(adapted);
            Vector<T>? queryGradient = null;
            T queryLoss = AiDotNet.Training.TapeGradients<T>.Of(
                () => HeadLoss(queryStates, queryLabels, leaves),
                leaves,
                gradients => queryGradient = Flatten(gradients, leaves),
                persistent: true);
            totalLoss = NumOps.Add(totalLoss, queryLoss);
            taskGradient = Accumulate(taskGradient, queryGradient ?? new Vector<T>(_taskParameters.Length));
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        sharedGradient = Scale(sharedGradient ?? new Vector<T>(shared.Length), batchSize);
        taskGradient = Scale(taskGradient ?? new Vector<T>(_taskParameters.Length), batchSize);

        if (_algoOptions.GradientClipThreshold.HasValue && _algoOptions.GradientClipThreshold.Value > 0)
        {
            double threshold = _algoOptions.GradientClipThreshold.Value;
            sharedGradient = ClipGradients(sharedGradient, threshold);
            if (taskGradient.Length > 0) taskGradient = ClipGradients(taskGradient, threshold);
        }

        double beta = _algoOptions.OuterLearningRate;
        var updated = ApplyOuterGradients(beta, (shared, sharedGradient), (_taskParameters, taskGradient));
        ParamModel.SetParameters(updated[0]);
        _taskParameters = updated[1];

        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Line 5 only: the shared encoder is left exactly as meta-training produced it, and the returned model
    /// carries this task's adapted attention and classifier.
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        EnsureTaskParameterShape(new[] { task });
        return new ATAMLAdaptedModel<T, TInput, TOutput>(
            MetaModel, AdaptTaskParameters(task), _stateWidth, _algoOptions.NumClasses);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns one probability per class for each example, so this is the configured loss of
    /// their logarithm against the class indices - cross-entropy by default.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    #region Attentive base learner

    /// <summary>
    /// Eq. 4-6: attention-weighted states, averaged into a context, classified by the softmax head. Returns
    /// logits, because the configured loss applies the softmax itself.
    /// </summary>
    /// <remarks>
    /// The encoder here emits ONE state per example, so the average over positions in eq. 5 is over a single
    /// term. A sequence encoder - the paper's setting, where the states are a document's words - gives the
    /// mean its full effect without any change to this code.
    /// </remarks>
    private Tensor<T> Logits(Tensor<T> states, IReadOnlyList<Tensor<T>> head)
    {
        var engine = AiDotNetEngine.Current;

        // alpha_t = theta_ATT . s_t, one scalar per state.
        var alpha = engine.ReduceSum(
            engine.TensorMultiply(states, engine.Reshape(head[0], new[] { 1, head[0].Length })),
            new[] { 1 }, keepDims: true);

        // s'_t = alpha_t s_t, and c is their mean - a single state per example here.
        var context = engine.TensorMultiply(states, alpha);

        return engine.TensorAdd(
            engine.TensorMatMul(context, engine.TensorTranspose(head[1])),
            engine.Reshape(head[2], new[] { 1, head[2].Length }));
    }

    /// <summary>The cross-entropy of the head's logits against the class indices, as a one-element tensor.</summary>
    private Tensor<T> HeadLoss(Tensor<T> states, int[] labels, IReadOnlyList<Tensor<T>> head)
    {
        var target = new Tensor<int>(labels, new[] { labels.Length }).Cast<T>();
        return AiDotNetEngine.Current.Reshape(LossFunction.ComputeTapeLoss(Logits(states, head), target), new[] { 1 });
    }

    /// <summary>
    /// Line 5: theta_T,i, the task parameters after AdaptationSteps gradient steps on the support set. The
    /// encoder is never touched - that is the whole task-agnostic half of the method.
    /// </summary>
    private Vector<T> AdaptTaskParameters(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var labels = ReadLabels(task.SupportOutput);
        Tensor<T> states;
        using (new NoGradScope<T>())
        {
            states = ClassifierOutputs<T>.AsRows(MetaModel.Predict(task.SupportInput));
        }

        var adapted = CloneVector(_taskParameters);
        T innerLearningRate = NumOps.FromDouble(_algoOptions.InnerLearningRate);
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var leaves = Leaves(adapted);
            var gradient = new Vector<T>(adapted.Length);
            AiDotNet.Training.TapeGradients<T>.Of(
                () => HeadLoss(states, labels, leaves),
                leaves,
                gradients => gradient = Flatten(gradients, leaves),
                persistent: true);

            adapted = Engine.Subtract(adapted, Engine.Multiply(gradient, innerLearningRate));
        }

        return adapted;
    }

    /// <summary>theta_ATT, theta_W and its bias as tape leaves, in the flat layout's order.</summary>
    private List<Tensor<T>> Leaves(Vector<T> flat)
    {
        var leaves = new List<Tensor<T>>();
        if (flat.Length == 0 || _stateWidth == 0) return leaves;

        int width = _stateWidth, classes = _algoOptions.NumClasses, position = 0;
        void Take(params int[] shape)
        {
            int size = 1;
            foreach (int dimension in shape) size *= dimension;
            leaves.Add(new Tensor<T>(shape, flat.GetSubVector(position, size)));
            position += size;
        }

        Take(width);
        Take(classes, width);
        Take(classes);
        return leaves;
    }

    private Vector<T> Flatten(Dictionary<Tensor<T>, Tensor<T>> gradients, IReadOnlyList<Tensor<T>> leaves)
    {
        // Each leaf's gradient (zero when the tape recorded none) in layout order, cut or zero-padded to theta_T.
        var parts = new List<Vector<T>>(leaves.Count);
        foreach (var leaf in leaves)
        {
            parts.Add(gradients.TryGetValue(leaf, out var gradient) ? gradient.ToVector() : new Vector<T>(leaf.Length));
        }

        var flat = parts.Count > 0 ? Vector<T>.Concatenate(parts) : new Vector<T>(0);
        int length = _taskParameters.Length;
        if (flat.Length > length) return flat.GetSubVector(0, length);
        return flat.Length < length ? Vector<T>.Concatenate(flat, new Vector<T>(length - flat.Length)) : flat;
    }

    /// <summary>Sizes theta_T once the encoder's state width is known.</summary>
    private void EnsureTaskParameterShape(IEnumerable<IMetaLearningTask<T, TInput, TOutput>> tasks)
    {
        if (_taskParameters.Length > 0) return;
        var first = tasks.FirstOrDefault();
        if (first is null) return;

        using (new NoGradScope<T>())
        {
            _stateWidth = ClassifierOutputs<T>.AsRows(MetaModel.Predict(first.SupportInput)).Shape[1];
        }

        int width = _stateWidth, classes = _algoOptions.NumClasses;
        _taskParameters = new Vector<T>(width + classes * width + classes);

        // theta_ATT starts near one so the first pass is close to an unweighted mean of the states, and the
        // classifier takes the usual fan-in uniform initialisation.
        int position = 0;
        for (int i = 0; i < width; i++)
        {
            _taskParameters[position++] = NumOps.FromDouble(1.0 + 0.1 * (RandomGenerator.NextDouble() - 0.5));
        }

        double bound = 1.0 / Math.Sqrt(Math.Max(width, 1));
        for (int i = 0; i < classes * width; i++)
        {
            _taskParameters[position++] = NumOps.FromDouble((2.0 * RandomGenerator.NextDouble() - 1.0) * bound);
        }
    }

    private int[] ReadLabels(TOutput labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, int.MaxValue);
        return Engine.TensorRound(tensor).Cast<int>().ToArray();
    }

    #endregion

    #region Test hooks

    /// <summary>The task parameters after adapting to one task, for tests.</summary>
    internal Vector<T> AdaptedTaskParametersForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureTaskParameterShape(new[] { task });
        return AdaptTaskParameters(task);
    }

    /// <summary>Gets or sets a copy of theta_T (for tests).</summary>
    internal Vector<T> TaskParametersForTesting
    {
        get => CloneVector(_taskParameters);
        set => _taskParameters = CloneVector(value);
    }

    #endregion

    #region Helpers

    private Vector<T> CloneVector(Vector<T> source)
    {
        return source.Clone();
    }

    private Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
    {
        if (sum is null) return CloneVector(values);

        // Adds the overlapping prefix; any tail of sum beyond values is kept as is.
        if (values.Length >= sum.Length)
        {
            return Engine.Add(sum, values.Length == sum.Length ? values : values.GetSubVector(0, sum.Length));
        }

        return Vector<T>.Concatenate(
            Engine.Add(sum.GetSubVector(0, values.Length), values),
            sum.GetSubVector(values.Length, sum.Length - values.Length));
    }

    private Vector<T> Scale(Vector<T> values, T divisor) => Engine.Divide(values, divisor);

    #endregion
}

/// <summary>The attention and classifier ATAML adapted to one task, over the shared encoder.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The shared "reader" is unchanged; this model holds only the small per-task part
/// that decides which parts of what was read matter, and turns them into class probabilities.
/// </para>
/// </remarks>
internal class ATAMLAdaptedModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _encoder;
    private readonly Vector<T> _taskParameters;
    private readonly int _stateWidth;
    private readonly int _classes;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public ATAMLAdaptedModel(
        IFullModel<T, TInput, TOutput> encoder,
        Vector<T> taskParameters,
        int stateWidth,
        int classes)
    {
        Guard.NotNull(encoder);
        Guard.NotNull(taskParameters);
        _encoder = encoder.DeepCopy();
        _taskParameters = taskParameters;
        _stateWidth = stateWidth;
        _classes = classes;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var states = ClassifierOutputs<T>.AsRows(_encoder.Predict(input));
        int rows = states.Shape[0], width = states.Shape[1];

        // theta_T = [theta_ATT (stateWidth), W (classes x stateWidth), b (classes)]; entries missing from a short
        // vector contribute nothing, exactly as zeros would.
        int expected = _stateWidth + _classes * _stateWidth + _classes;
        var parameters = _taskParameters.Length >= expected
            ? _taskParameters
            : Vector<T>.Concatenate(_taskParameters, new Vector<T>(expected - _taskParameters.Length));
        var flat = Tensor<T>.FromVector(parameters);
        int m = Math.Min(width, _stateWidth);

        var attention = engine.Reshape(engine.TensorSlice(flat, new[] { 0 }, new[] { m }), new[] { m, 1 });
        var weights = engine.TensorSlice(
            engine.Reshape(engine.TensorSlice(flat, new[] { _stateWidth }, new[] { _classes * _stateWidth }),
                new[] { _classes, _stateWidth }),
            new[] { 0, 0 }, new[] { _classes, m });
        var bias = engine.Reshape(
            engine.TensorSlice(flat, new[] { _stateWidth + _classes * _stateWidth }, new[] { _classes }),
            new[] { 1, _classes });
        var used = engine.TensorSlice(states, new[] { 0, 0 }, new[] { rows, m });

        // alpha = theta_ATT . s, then c = alpha * s, logits = c W^T + b, scores = softmax(logits).
        var alpha = engine.TensorMatMul(used, attention);
        var context = engine.TensorMultiply(used, engine.TensorBroadcastTo(alpha, used.Shape.ToArray()));
        var logits = engine.TensorAdd(engine.TensorMatMul(context, engine.TensorTranspose(weights)),
            engine.TensorBroadcastTo(bias, new[] { rows, _classes }));
        var scores = engine.Softmax(logits, 1);

        return ClassifierOutputs<T>.ToOutput<TOutput>(scores);
    }

    /// <summary>Training is not supported on an adapted model.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException(
            "Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
