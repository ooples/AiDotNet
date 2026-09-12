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
/// Implementation of Prototypical Networks (ProtoNets) algorithm for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Prototypical Networks (Snell et al. 2017) learn an embedding in which each class is represented by the mean of
/// its embedded support examples, its prototype, and a query is classified by a softmax over negative distances to
/// the prototypes: <c>p(y = k | x) = softmax(-d(f(x), c_k))</c>, with d the squared Euclidean distance. Learning
/// minimises <c>-log p(y = k | x)</c> of the true class by SGD on the embedding.
/// </para>
/// <para>
/// <b>The gradient reaches the embedding through the prototypes.</b> Support and query examples go through the
/// embedding together, and the prototypes, distances and softmax are built from those embeddings on the tape, so the
/// loss is differentiated through both the query embeddings and the prototypes the support embeddings form.
/// </para>
/// <para>
/// <b>Extensions beyond the paper, all off by default and learned on the same objective.</b>
/// <see cref="ProtoNetsOptions{T,TInput,TOutput}.UseAttentionMechanism"/> weights each support example in its
/// prototype by a softmax of a learned attention score (uniform - the paper's mean - at initialisation).
/// <see cref="ProtoNetsDistanceFunction.Mahalanobis"/> learns a diagonal metric, a Bregman divergence as Snell et al.
/// discuss. <see cref="ProtoNetsOptions{T,TInput,TOutput}.UseAdaptiveClassScaling"/> learns a positive scale per class
/// slot, in the spirit of TADAM's learned metric scaling (Oreshkin et al. 2018).
/// </para>
/// <para><b>For Beginners:</b> ProtoNets learns to recognise new classes from a few examples: it averages each
/// class's examples into a "prototype" and labels a new example by the closest prototype. Training shapes the
/// feature space so that examples of one class cluster around their prototype.
/// </para>
/// <para>
/// Reference: Snell, J., Swersky, K., &amp; Zemel, R. (2017).
/// Prototypical Networks for Few-shot Learning.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Prototypical Networks for Few-shot Learning",
    "https://arxiv.org/abs/1703.05175",
    Year = 2017,
    Authors = "Snell, J., Swersky, K., & Zemel, R.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ProtoNetsAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ProtoNetsOptions<T, TInput, TOutput> _protoNetsOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _protoNetsOptions;

    /// <summary>
    /// The learned attention query <c>w</c>, one weight per embedding dimension; empty until the first episode shows
    /// the embedding width, and while attention is off. Zero is uniform attention - the paper's mean.
    /// </summary>
    /// <remarks>
    /// It replaces a 0x0 "attention weights" matrix that nothing ever trained or read: the option did nothing.
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _attentionQuery = new Vector<T>(0);

    /// <summary>
    /// The learned log-diagonal <c>rho</c> of the Mahalanobis metric, <c>m = MahalanobisScaling * exp(rho)</c>; empty
    /// until the first episode shows the embedding width, and for the other distances.
    /// </summary>
    /// <remarks>
    /// It replaces a "simplified" Mahalanobis distance that was squared Euclidean times a fixed constant.
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _mahalanobisLogScale = new Vector<T>(0);

    /// <summary>
    /// The learned log scale <c>kappa</c> per class slot, <c>scale = exp(kappa)</c>; grown to the largest class index
    /// seen, and empty while adaptive class scaling is off.
    /// </summary>
    /// <remarks>
    /// It replaces a per-class scaling dictionary that was created empty and never written: the option did nothing.
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _classLogScale = new Vector<T>(0);

    /// <summary>
    /// Initializes a new instance of the ProtoNetsAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for ProtoNets.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    public ProtoNetsAlgorithm(ProtoNetsOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            null) // ProtoNets doesn't use inner optimizer (non-parametric adaptation)
    {
        _protoNetsOptions = options;

        if (!_protoNetsOptions.IsValid())
        {
            throw new ArgumentException("ProtoNets configuration is invalid. Check all parameters.", nameof(options));
        }

        if (_protoNetsOptions.DistanceFunction == ProtoNetsDistanceFunction.Mahalanobis && _protoNetsOptions.MahalanobisScaling <= 0)
        {
            throw new ArgumentException("MahalanobisScaling must be positive: it is the metric's initial scale.", nameof(options));
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.ProtoNets"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ProtoNets;

    /// <summary>
    /// Performs one meta-training step: for each episode, the prototype loss of its query examples, differentiated
    /// through the prototypes into the embedding and into any learned metric.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on.</param>
    /// <returns>The average query loss across the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// This used to report the prototype cross-entropy but differentiate something else: the embedding's gradient was
    /// the configured loss of its raw output against the class indices, which trains the embedding to BE the label
    /// and never involves a prototype.
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        EnsureMetricShapes(taskBatch.Tasks);

        var body = ParamModel.GetParameters();
        Vector<T>? bodyGradient = null, attentionGradient = null, mahalanobisGradient = null, classGradient = null;
        T totalLoss = NumOps.Zero;
        foreach (var task in taskBatch.Tasks)
        {
            var (loss, taskBody, attention, mahalanobis, classes) = EpisodeGradient(task);
            totalLoss = NumOps.Add(totalLoss, loss);
            bodyGradient = Accumulate(bodyGradient, taskBody);
            attentionGradient = Accumulate(attentionGradient, attention);
            mahalanobisGradient = Accumulate(mahalanobisGradient, mahalanobis);
            classGradient = Accumulate(classGradient, classes);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        bodyGradient = Scale(bodyGradient ?? new Vector<T>(body.Length), batchSize);
        attentionGradient = Scale(attentionGradient ?? new Vector<T>(_attentionQuery.Length), batchSize);
        mahalanobisGradient = Scale(mahalanobisGradient ?? new Vector<T>(_mahalanobisLogScale.Length), batchSize);
        classGradient = Scale(classGradient ?? new Vector<T>(_classLogScale.Length), batchSize);

        if (_protoNetsOptions.GradientClipThreshold.HasValue && _protoNetsOptions.GradientClipThreshold.Value > 0)
        {
            double threshold = _protoNetsOptions.GradientClipThreshold.Value;
            bodyGradient = ClipGradients(bodyGradient, threshold);
            if (attentionGradient.Length > 0) attentionGradient = ClipGradients(attentionGradient, threshold);
            if (mahalanobisGradient.Length > 0) mahalanobisGradient = ClipGradients(mahalanobisGradient, threshold);
            if (classGradient.Length > 0) classGradient = ClipGradients(classGradient, threshold);
        }

        double beta = _protoNetsOptions.OuterLearningRate;
        ParamModel.SetParameters(ApplyGradients(body, bodyGradient, beta));
        if (_attentionQuery.Length > 0) _attentionQuery = ApplyGradients(_attentionQuery, attentionGradient, beta);
        if (_mahalanobisLogScale.Length > 0) _mahalanobisLogScale = ApplyGradients(_mahalanobisLogScale, mahalanobisGradient, beta);
        if (_classLogScale.Length > 0) _classLogScale = ApplyGradients(_classLogScale, classGradient, beta);

        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <summary>
    /// Adapts to a new task by computing class prototypes from the support set.
    /// </summary>
    /// <param name="task">The new task containing support set examples.</param>
    /// <returns>A PrototypicalModel that classifies by distance to the prototypes, with the learned metric.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        EnsureMetricShapes(new[] { task });
        return new PrototypicalModel<T, TInput, TOutput>(
            MetaModel, task.SupportInput, task.SupportOutput, _protoNetsOptions, NumOps,
            CloneVector(_attentionQuery), CloneVector(_mahalanobisLogScale), CloneVector(_classLogScale));
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

    /// <summary>
    /// One episode's query loss and the exact gradient of that loss with respect to the embedding and every learned
    /// metric parameter.
    /// </summary>
    private (T Loss, Vector<T> Body, Vector<T> Attention, Vector<T> Mahalanobis, Vector<T> Classes) EpisodeGradient(
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = PrototypeEpisode<T>.Build(
            ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));
        var stackedInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));

        // The embedding's gradient: the whole episode - prototypes and queries - rebuilt from the embeddings on the
        // tape, scored against the query labels.
        var attention = ParameterTensor(_attentionQuery);
        var mahalanobis = ParameterTensor(_mahalanobisLogScale);
        var classScales = ParameterTensor(_classLogScale);
        var composed = new EmbeddingClassificationLoss<T>(
            embeddings => EpisodeLogits(embeddings, episode, attention, mahalanobis, classScales),
            LossFunction,
            episode.QueryTarget);
        var bodyGradient = ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);

        // The metric's gradient: the same logits, differentiated with respect to the learned parameters.
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = ClassifierOutputs<T>.AsRows(MetaModel.Predict(stackedInput));
        }

        var sources = new List<Tensor<T>>();
        if (attention is not null) sources.Add(attention);
        if (mahalanobis is not null) sources.Add(mahalanobis);
        if (classScales is not null) sources.Add(classScales);

        T loss;
        var attentionGradient = new Vector<T>(_attentionQuery.Length);
        var mahalanobisGradient = new Vector<T>(_mahalanobisLogScale.Length);
        var classGradient = new Vector<T>(_classLogScale.Length);
        if (sources.Count == 0)
        {
            using var noGrad = new NoGradScope<T>();
            loss = LossFunction.ComputeTapeLoss(
                EpisodeLogits(embeddings, episode, null, null, null), episode.QueryTarget)[0];
        }
        else
        {
            using var tape = new GradientTape<T>();
            var episodeLoss = LossFunction.ComputeTapeLoss(
                EpisodeLogits(embeddings, episode, attention, mahalanobis, classScales), episode.QueryTarget);
            loss = episodeLoss[0];
            var gradients = tape.ComputeGradients(episodeLoss, sources);
            attentionGradient = GradientOf(gradients, attention, attentionGradient.Length);
            mahalanobisGradient = GradientOf(gradients, mahalanobis, mahalanobisGradient.Length);
            classGradient = GradientOf(gradients, classScales, classGradient.Length);
        }

        return (loss, bodyGradient, attentionGradient, mahalanobisGradient, classGradient);
    }

    /// <summary>Logits of an episode's query rows from the stacked support-then-query embeddings.</summary>
    private Tensor<T> EpisodeLogits(
        Tensor<T> embeddings, PrototypeEpisode<T> episode,
        Tensor<T>? attention, Tensor<T>? mahalanobis, Tensor<T>? classScales)
    {
        var engine = AiDotNetEngine.Current;
        var rows = PrototypeMetric<T>.Normalized(embeddings, _protoNetsOptions.NormalizeFeatures);
        var support = engine.TensorMatMul(episode.SupportSelector, rows);
        var query = engine.TensorMatMul(episode.QuerySelector, rows);
        var prototypes = PrototypeMetric<T>.Prototypes(support, episode.Membership, attention);
        return PrototypeMetric<T>.Logits(
            query, prototypes, _protoNetsOptions.DistanceFunction, _protoNetsOptions.MahalanobisScaling,
            mahalanobis, classScales, episode.ClassSlots, _protoNetsOptions.Temperature);
    }

    /// <summary>
    /// Sizes the learned metric to the embedding width and the largest class index of the given tasks. Attention and
    /// the Mahalanobis diagonal start at zero - uniform attention and <c>m = MahalanobisScaling</c> - and class scales
    /// start at zero log scale, so an untrained extension is exactly the paper's metric.
    /// </summary>
    private void EnsureMetricShapes(IEnumerable<IMetaLearningTask<T, TInput, TOutput>> tasks)
    {
        var list = tasks.ToList();
        if (list.Count == 0) return;

        bool needWidth = (_protoNetsOptions.UseAttentionMechanism && _attentionQuery.Length == 0)
            || (_protoNetsOptions.DistanceFunction == ProtoNetsDistanceFunction.Mahalanobis && _mahalanobisLogScale.Length == 0);
        if (needWidth)
        {
            int width;
            using (new NoGradScope<T>())
            {
                width = ClassifierOutputs<T>.AsRows(MetaModel.Predict(list[0].SupportInput)).Shape[1];
            }

            if (_protoNetsOptions.UseAttentionMechanism && _attentionQuery.Length == 0)
                _attentionQuery = new Vector<T>(width);
            if (_protoNetsOptions.DistanceFunction == ProtoNetsDistanceFunction.Mahalanobis && _mahalanobisLogScale.Length == 0)
                _mahalanobisLogScale = new Vector<T>(width);
        }

        if (_protoNetsOptions.UseAdaptiveClassScaling)
        {
            int slots = list.Max(t => Math.Max(ReadLabels(t.SupportOutput).DefaultIfEmpty(-1).Max(),
                ReadLabels(t.QueryOutput).DefaultIfEmpty(-1).Max())) + 1;
            if (slots > _classLogScale.Length)
            {
                var grown = new Vector<T>(slots);
                for (int i = 0; i < _classLogScale.Length; i++) grown[i] = _classLogScale[i];
                _classLogScale = grown;
            }
        }
    }

    private static int[] ReadLabels(TOutput labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, int.MaxValue);
        var indices = new int[tensor.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
        return indices;
    }

    private static Tensor<T>? ParameterTensor(Vector<T> values) => values.Length > 0 ? Tensor<T>.FromVector(values) : null;

    private static Vector<T> GradientOf(Dictionary<Tensor<T>, Tensor<T>> gradients, Tensor<T>? source, int length)
    {
        var result = new Vector<T>(length);
        if (source is null || !gradients.TryGetValue(source, out var gradient)) return result;
        for (int i = 0; i < length; i++) result[i] = gradient[i];
        return result;
    }

    #endregion

    #region Test hooks

    /// <summary>One episode's query loss and exact gradient from the current state, for gradient checks.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> Attention, Vector<T> Mahalanobis, Vector<T> Classes) EpisodeGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureMetricShapes(new[] { task });
        return EpisodeGradient(task);
    }

    /// <summary>One episode's query loss from the current state.</summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureMetricShapes(new[] { task });
        var episode = PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));
        using var noGrad = new NoGradScope<T>();
        var embeddings = ClassifierOutputs<T>.AsRows(MetaModel.Predict(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput)));
        return LossFunction.ComputeTapeLoss(
            EpisodeLogits(embeddings, episode, ParameterTensor(_attentionQuery), ParameterTensor(_mahalanobisLogScale),
                ParameterTensor(_classLogScale)),
            episode.QueryTarget)[0];
    }

    /// <summary>Gets or sets a copy of the learned attention query (for tests).</summary>
    internal Vector<T> AttentionQueryForTesting { get => CloneVector(_attentionQuery); set => _attentionQuery = CloneVector(value); }

    /// <summary>Gets or sets a copy of the learned Mahalanobis log-diagonal (for tests).</summary>
    internal Vector<T> MahalanobisLogScaleForTesting { get => CloneVector(_mahalanobisLogScale); set => _mahalanobisLogScale = CloneVector(value); }

    /// <summary>Gets or sets a copy of the learned class log scales (for tests).</summary>
    internal Vector<T> ClassLogScaleForTesting { get => CloneVector(_classLogScale); set => _classLogScale = CloneVector(value); }

    #endregion

    #region Helpers

    /// <summary>Converts a tensor to a matrix, one row per leading index.</summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor) => PrototypeMetric<T>.ToMatrix(tensor);

    private static Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
    {
        if (sum is null) return CloneVector(values);
        for (int i = 0; i < sum.Length; i++) sum[i] = NumOps.Add(sum[i], values[i]);
        return sum;
    }

    private static Vector<T> Scale(Vector<T> values, T divisor)
    {
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.Divide(values[i], divisor);
        return values;
    }

    private static Vector<T> CloneVector(Vector<T> source)
    {
        var clone = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    #endregion
}

/// <summary>
/// The fixed bookkeeping of one episode: which stacked rows are support and which are query, which support rows
/// belong to which class, and each query row's class column.
/// </summary>
internal sealed class PrototypeEpisode<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private PrototypeEpisode(int rows, Tensor<T> supportSelector, Tensor<T> querySelector, Tensor<T> membership,
        Tensor<T> queryTarget, int[] classSlots)
    {
        Rows = rows;
        SupportSelector = supportSelector;
        QuerySelector = querySelector;
        Membership = membership;
        QueryTarget = queryTarget;
        ClassSlots = classSlots;
    }

    /// <summary>Support plus query rows.</summary>
    public int Rows { get; }

    /// <summary><c>[support, rows]</c>: picks the support rows out of the stacked embeddings.</summary>
    public Tensor<T> SupportSelector { get; }

    /// <summary><c>[query, rows]</c>: picks the query rows out of the stacked embeddings.</summary>
    public Tensor<T> QuerySelector { get; }

    /// <summary><c>[classes, support]</c>: 1 where a support row belongs to a class.</summary>
    public Tensor<T> Membership { get; }

    /// <summary>Each query row's class column, <c>[query]</c>.</summary>
    public Tensor<T> QueryTarget { get; }

    /// <summary>The class label behind each column, in ascending order.</summary>
    public int[] ClassSlots { get; }

    /// <summary>Builds the episode from its support and query labels.</summary>
    /// <exception cref="ArgumentException">A query label has no support example, so it has no prototype.</exception>
    public static PrototypeEpisode<T> Build(int[] supportLabels, int[] queryLabels)
    {
        var classes = supportLabels.Distinct().OrderBy(label => label).ToArray();
        var column = new Dictionary<int, int>();
        for (int c = 0; c < classes.Length; c++) column[classes[c]] = c;

        int support = supportLabels.Length;
        int query = queryLabels.Length;
        int rows = support + query;

        var supportSelector = new Tensor<T>(new[] { support, rows });
        for (int s = 0; s < support; s++) supportSelector[s * rows + s] = Ops.One;
        var querySelector = new Tensor<T>(new[] { query, rows });
        for (int q = 0; q < query; q++) querySelector[q * rows + support + q] = Ops.One;

        var membership = new Tensor<T>(new[] { classes.Length, support });
        for (int s = 0; s < support; s++) membership[column[supportLabels[s]] * support + s] = Ops.One;

        var queryTarget = new Tensor<T>(new[] { query });
        for (int q = 0; q < query; q++)
        {
            if (!column.TryGetValue(queryLabels[q], out int c))
            {
                throw new ArgumentException(
                    $"Query label {queryLabels[q]} has no support example, so it has no prototype.", nameof(queryLabels));
            }

            queryTarget[q] = Ops.FromDouble(c);
        }

        return new PrototypeEpisode<T>(rows, supportSelector, querySelector, membership, queryTarget, classes);
    }
}

/// <summary>
/// The prototype classifier's arithmetic, written with engine tensor ops so the tape records it when one is live:
/// prototypes from support embeddings, and logits from query embeddings.
/// </summary>
internal static class PrototypeMetric<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Rows scaled to unit length when asked; otherwise the rows as they are.</summary>
    public static Tensor<T> Normalized(Tensor<T> rows, bool normalize)
    {
        if (!normalize) return rows;
        var engine = AiDotNetEngine.Current;
        var squaredNorm = engine.ReduceSum(engine.TensorMultiply(rows, rows), new[] { 1 }, keepDims: true);
        var norm = engine.TensorSqrt(engine.TensorAddScalar(squaredNorm, Ops.FromDouble(1e-12)));
        return engine.TensorDivide(rows, norm);
    }

    /// <summary>
    /// Class prototypes, <c>[classes, width]</c>: the mean of each class's support embeddings (Snell et al. 2017), or,
    /// with a learned attention query <c>w</c>, their softmax(<c>h . w</c>)-weighted mean within the class.
    /// </summary>
    public static Tensor<T> Prototypes(Tensor<T> support, Tensor<T> membership, Tensor<T>? attention)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> weights;
        if (attention is null)
        {
            weights = membership;
        }
        else
        {
            int width = support.Shape[1];
            var scores = engine.TensorTranspose(engine.TensorMatMul(support, engine.Reshape(attention, new[] { width, 1 })));
            var reducedMax = engine.ReduceMax(scores, new[] { 1 }, keepDims: true, out _);
            var shifted = engine.TensorAdd(scores, engine.TensorNegate(engine.StopGradient(reducedMax)));
            weights = engine.TensorMultiply(membership, engine.TensorExp(shifted));
        }

        var total = engine.ReduceSum(weights, new[] { 1 }, keepDims: true);
        var normalized = engine.TensorDivide(weights, engine.TensorClampMin(total, Ops.FromDouble(1e-12)));
        return engine.TensorMatMul(normalized, support);
    }

    /// <summary>
    /// Logits <c>-scale_c * d(q, c_k) / temperature</c>, <c>[query, classes]</c>, for squared Euclidean, cosine or a
    /// learned diagonal Mahalanobis distance.
    /// </summary>
    public static Tensor<T> Logits(
        Tensor<T> query, Tensor<T> prototypes, ProtoNetsDistanceFunction distance, double mahalanobisScaling,
        Tensor<T>? mahalanobisLogScale, Tensor<T>? classLogScale, int[] classSlots, double temperature)
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> distances;
        if (distance == ProtoNetsDistanceFunction.Cosine)
        {
            var q = Normalized(query, normalize: true);
            var p = Normalized(prototypes, normalize: true);
            var similarity = engine.TensorMatMul(q, engine.TensorTranspose(p));
            distances = engine.TensorAddScalar(engine.TensorNegate(similarity), Ops.One);
        }
        else
        {
            // Squared Euclidean (Snell et al. 2017) - the diagonal metric m weights each dimension for Mahalanobis:
            // d(q, p) = q.m.q + p.m.p - 2 q.m.p, all in two-dimensional ops.
            Tensor<T>? metric = null;
            if (distance == ProtoNetsDistanceFunction.Mahalanobis)
            {
                int width = query.Shape[1];
                var logScale = mahalanobisLogScale ?? new Tensor<T>(new[] { width });
                metric = engine.TensorMultiplyScalar(
                    engine.Reshape(engine.TensorExp(logScale), new[] { 1, width }), Ops.FromDouble(mahalanobisScaling));
            }

            var weightedQuery = metric is null ? query : engine.TensorMultiply(query, metric);
            var weightedPrototypes = metric is null ? prototypes : engine.TensorMultiply(prototypes, metric);
            var queryNorm = engine.ReduceSum(engine.TensorMultiply(weightedQuery, query), new[] { 1 }, keepDims: true);
            var prototypeNorm = engine.ReduceSum(
                engine.TensorMultiply(weightedPrototypes, prototypes), new[] { 1 }, keepDims: true);
            var cross = engine.TensorMatMul(weightedQuery, engine.TensorTranspose(prototypes));
            distances = engine.TensorAdd(
                engine.TensorAdd(queryNorm, engine.TensorTranspose(prototypeNorm)),
                engine.TensorMultiplyScalar(cross, Ops.FromDouble(-2.0)));
        }

        if (classLogScale is not null)
        {
            var selector = new Tensor<T>(new[] { classLogScale.Length, classSlots.Length });
            for (int c = 0; c < classSlots.Length; c++)
            {
                if (classSlots[c] < classLogScale.Length) selector[classSlots[c] * classSlots.Length + c] = Ops.One;
            }

            var logScales = engine.TensorMatMul(engine.Reshape(classLogScale, new[] { 1, classLogScale.Length }), selector);
            distances = engine.TensorMultiply(distances, engine.TensorExp(logScales));
        }

        return engine.TensorMultiplyScalar(distances, Ops.FromDouble(-1.0 / temperature));
    }

    /// <summary>A tensor as a matrix, one row per leading index.</summary>
    public static Matrix<T> ToMatrix(Tensor<T> tensor)
    {
        var rows = ClassifierOutputs<T>.AsRows(tensor);
        int r = rows.Shape[0], c = rows.Shape[1];
        var matrix = new Matrix<T>(r, c);
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++) matrix[i, j] = rows[i * c + j];
        return matrix;
    }
}

/// <summary>
/// Prototypical model for few-shot classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// The adapted state of ProtoNets for one task: its own copy of the embedding, the class prototypes of the support
/// set, and the learned metric. <see cref="Predict"/> returns class probabilities, <c>[rows, classes]</c>, for Tensor
/// and Matrix outputs, and the most probable class of each example for a Vector output.
/// </para>
/// <para><b>For Beginners:</b> After adapting ProtoNets to a new task, you get this model. It classifies new
/// examples instantly by finding the nearest class prototype.
/// </para>
/// </remarks>
public class PrototypicalModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly ProtoNetsOptions<T, TInput, TOutput> _options;
    private readonly INumericOperations<T> _numOps;
    private readonly Tensor<T> _prototypes;
    private readonly int[] _classSlots;
    private readonly Tensor<T>? _mahalanobisLogScale;
    private readonly Tensor<T>? _classLogScale;

    /// <summary>
    /// Initializes a new instance of the PrototypicalModel with the paper's metric.
    /// </summary>
    /// <param name="featureEncoder">The trained feature encoder; the model keeps its own copy.</param>
    /// <param name="supportInputs">Support set inputs for computing prototypes.</param>
    /// <param name="supportOutputs">Support set outputs (class indices).</param>
    /// <param name="options">ProtoNets configuration options.</param>
    /// <param name="numOps">Numeric operations for type T.</param>
    public PrototypicalModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        TInput supportInputs,
        TOutput supportOutputs,
        ProtoNetsOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps)
        : this(featureEncoder, supportInputs, supportOutputs, options, numOps,
            new Vector<T>(0), new Vector<T>(0), new Vector<T>(0))
    {
    }

    /// <summary>Initializes the model with the metric ProtoNets learned.</summary>
    internal PrototypicalModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        TInput supportInputs,
        TOutput supportOutputs,
        ProtoNetsOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps,
        Vector<T> attentionQuery,
        Vector<T> mahalanobisLogScale,
        Vector<T> classLogScale)
    {
        Guard.NotNull(featureEncoder);
        Guard.NotNull(options);
        Guard.NotNull(numOps);
        _featureEncoder = featureEncoder.DeepCopy();
        _options = options;
        _numOps = numOps;
        _mahalanobisLogScale = mahalanobisLogScale.Length > 0 ? Tensor<T>.FromVector(mahalanobisLogScale) : null;
        _classLogScale = classLogScale.Length > 0 ? Tensor<T>.FromVector(classLogScale) : null;

        var labels = ClassifierOutputs<T>.Labels(supportOutputs, int.MaxValue);
        var supportLabels = new int[labels.Length];
        for (int i = 0; i < supportLabels.Length; i++) supportLabels[i] = (int)Math.Round(numOps.ToDouble(labels[i]));
        if (supportLabels.Length == 0)
            throw new ArgumentException("The support set is empty, so there are no prototypes.", nameof(supportOutputs));

        var episode = PrototypeEpisode<T>.Build(supportLabels, Array.Empty<int>());
        _classSlots = episode.ClassSlots;
        using var noGrad = new NoGradScope<T>();
        var support = PrototypeMetric<T>.Normalized(
            ClassifierOutputs<T>.AsRows(_featureEncoder.Predict(supportInputs)), options.NormalizeFeatures);
        _prototypes = PrototypeMetric<T>.Prototypes(
            support, episode.Membership, attentionQuery.Length > 0 ? Tensor<T>.FromVector(attentionQuery) : null);
    }

    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Makes predictions using prototype-based classification.
    /// </summary>
    /// <param name="input">The input to classify.</param>
    /// <returns>Class probabilities per example, or the predicted class per example for a Vector output.</returns>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var query = PrototypeMetric<T>.Normalized(
            ClassifierOutputs<T>.AsRows(_featureEncoder.Predict(input)), _options.NormalizeFeatures);
        var logits = PrototypeMetric<T>.Logits(
            query, _prototypes, _options.DistanceFunction, _options.MahalanobisScaling,
            _mahalanobisLogScale, _classLogScale, _classSlots, _options.Temperature);
        var probabilities = engine.Softmax(logits, axis: 1);

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            int rows = probabilities.Shape[0], classes = probabilities.Shape[1];
            var predicted = new Vector<T>(rows);
            for (int r = 0; r < rows; r++)
            {
                int best = 0;
                for (int c = 1; c < classes; c++)
                {
                    if (_numOps.GreaterThan(probabilities[r * classes + c], probabilities[r * classes + best])) best = c;
                }

                predicted[r] = _numOps.FromDouble(_classSlots[best]);
            }

            return (TOutput)(object)predicted;
        }

        return ClassifierOutputs<T>.ToOutput<TOutput>(probabilities);
    }

    /// <summary>
    /// Trains the model (not applicable for prototype-based models).
    /// </summary>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Prototype models don't support training. Compute new prototypes instead.");
    }

    /// <summary>
    /// Updates model parameters (not applicable for prototype-based models).
    /// </summary>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Prototype models don't have trainable parameters.");
    }

    /// <summary>
    /// Gets model parameters (not applicable for prototype-based models).
    /// </summary>
    public Vector<T> GetParameters()
    {
        throw new NotSupportedException("Prototype models don't have trainable parameters.");
    }

    /// <summary>Converts a tensor to a matrix, one row per leading index.</summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor) => PrototypeMetric<T>.ToMatrix(tensor);

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>Model metadata including information about prototypes.</returns>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}
