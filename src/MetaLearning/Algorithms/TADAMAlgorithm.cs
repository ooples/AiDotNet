using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Task-Dependent Adaptive Metric (TADAM) algorithm for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// TADAM (Oreshkin, Rodriguez and Lacoste, NeurIPS 2018) extends Prototypical Networks with:
/// 1. Metric scaling (Sec. 2.1, Eq. 3): <c>p(y = k | x) = softmax(-alpha d(f(x), c_k))</c> with ONE learned scalar
///    <c>alpha</c> - not a per-dimension weight vector and not a separate temperature, which <c>alpha</c> already is.
/// 2. Task conditioning (Sec. 2.2): a task-embedding network (TEN) maps the task representation <c>c_bar</c>, the
///    mean of the class prototypes, to FiLM parameters, with a scalar post-multiplier that starts at zero and is
///    L2-penalized, so an untrained TEN is the identity.
/// 3. Auxiliary co-training (Sec. 2.3), here an extra classification term on the support set.
/// </para>
/// <para>
/// The feature extractor is a black-box model, so FiLM is applied to its OUTPUT features rather than inside its
/// layers: <c>h' = (1 + gamma) h</c>. A shift <c>beta</c> would cancel exactly in every query-prototype difference
/// at that level, so only the scale is generated. The extractor, <c>alpha</c> and the TEN all train on the exact
/// gradient of the episode's query cross-entropy: the logits are built with engine tensor ops, so the extractor's
/// gradient flows through the prototypes and the metric (it was trained against the raw labels before).
/// </para>
/// <para><b>For Beginners:</b> TADAM improves on ProtoNets by making the feature
/// extractor "aware" of the current task:
///
/// **How it works:**
/// 1. Extract features from support set examples
/// 2. Summarize the task as the average of its class prototypes
/// 3. Use that summary to rescale each feature dimension for this task
/// 4. Compute prototypes from the rescaled features
/// 5. Classify queries by their scaled distance to each prototype
/// </para>
/// <para><b>Algorithm - TADAM:</b>
/// <code>
/// for each episode:
///     h = f_theta(x)                                   # support and query features
///     c_bar = mean_k(mean(h[support, class == k]))     # task representation
///     gamma = gamma0 * TEN(c_bar)                      # TEN(c) = W2 relu(W1 c + b1) + b2
///     h' = (1 + gamma) * h
///     c_k = mean(h'[support, class == k])
///     p(y = k | q) = softmax_k(-alpha * ||h'_q - c_k||^2)
///     loss = cross_entropy(p, y_query) + lambda * gamma0^2  [+ aux_weight * support cross-entropy]
///     update(theta, alpha, TEN, gamma0) by gradient descent
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning",
    "https://arxiv.org/abs/1805.10123",
    Year = 2018,
    Authors = "Oreshkin, B. N., Rodriguez, P., & Lacoste, A.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class TADAMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly TADAMOptions<T, TInput, TOutput> _tadamOptions;

    // The learned metric scale alpha (Eq. 3), a single scalar held as a length-1 vector.
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _metricScale;

    // The task-embedding network, flattened: W1 [width, hidden], b1 [hidden], W2 [hidden, width], b2 [width], then
    // the scalar post-multiplier gamma0. Sized from the first episode's feature width.
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _taskEncoder;

    private int _taskEncoderWidth;

    /// <summary>
    /// Initializes a new instance of the TADAMAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for TADAM.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    public TADAMAlgorithm(TADAMOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            // Oreshkin et al. 2018 minimise -log p(y = k|x) with p = softmax(-alpha d(f(x), c_k)): cross-entropy over
            // class indices. The caller's own LossFunction still wins when supplied.
            options.LossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _tadamOptions = options;

        // alpha starts at 1 / InitialTemperature: logits -alpha d are the -d / tau of a temperature-scaled softmax.
        // Without metric scaling alpha stays at that value, which is Prototypical Networks (the paper's ablation).
        _metricScale = Engine.Fill(1, NumOps.FromDouble(1.0 / _tadamOptions.InitialTemperature));
        _taskEncoder = new Vector<T>(0);

        // Validate configuration
        if (!_tadamOptions.IsValid())
        {
            throw new ArgumentException("TADAM configuration is invalid. Check all parameters.", nameof(options));
        }
    }

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.TADAM;

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var losses = new List<T>(taskBatch.BatchSize);
        foreach (var task in taskBatch.Tasks)
        {
            losses.Add(TrainEpisode(task));
        }

        // Return average loss
        return NumOps.Divide(Engine.Sum(new Vector<T>(losses.ToArray())), NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns one probability per class for each example, so this is the configured loss of
    /// their logarithm against the class indices - cross-entropy by default. A Vector output carries the predicted
    /// class of each example instead, and its loss is the classification error rate.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    /// <inheritdoc />
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        using var noGrad = new NoGradScope<T>();
        var labels = ReadLabels(task.SupportOutput);
        var episode = PrototypeEpisode<T>.Build(labels, Array.Empty<int>());
        var support = PrototypeMetric<T>.Normalized(Embed(task.SupportInput), _tadamOptions.NormalizeEmbeddings);
        int width = support.Shape[1];

        // Prototypes are stored unscaled; the model multiplies each query-prototype difference by the task's FiLM
        // scale, which equals comparing the conditioned query with the conditioned prototype.
        var scale = FeatureScale(support, episode.Membership, TaskEncoderLeaves());
        var prototypeRows = PrototypeMetric<T>.Prototypes(support, episode.Membership, null);

        var prototypes = new Dictionary<int, Tensor<T>>();
        for (int c = 0; c < episode.ClassSlots.Length; c++)
        {
            prototypes[episode.ClassSlots[c]] = Engine.TensorSlice(prototypeRows, new[] { c, 0 }, new[] { 1, width })
                .Reshape(new[] { width });
        }

        return new TADAMModel<T, TInput, TOutput>(
            MetaModel,
            prototypes,
            scale is null ? new Vector<T>(0) : scale.ToVector(),
            NumOps.Divide(NumOps.One, _metricScale[0]),
            _tadamOptions);
    }

    /// <summary>
    /// Trains on a single episode: one exact gradient step on the extractor, alpha and the task encoder.
    /// </summary>
    private T TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var supportLabels = ReadLabels(task.SupportOutput);
        var episode = PrototypeEpisode<T>.Build(supportLabels, ReadLabels(task.QueryOutput));
        var stackedInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);

        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = Embed(stackedInput);
        }

        var alpha = _tadamOptions.UseMetricScaling ? Tensor<T>.FromVector(_metricScale) : null;
        var encoder = TaskEncoderLeaves();

        // The extractor's gradient: the whole episode - TEN, prototypes, metric - rebuilt from the embeddings on the tape.
        var bodyGradient = EmbeddingGradient(stackedInput, episode, alpha, encoder);
        T auxiliaryLoss = NumOps.Zero;
        if (_tadamOptions.UseAuxiliaryCoTraining)
        {
            // Auxiliary co-training: every support example classified against the support prototypes.
            var auxiliaryEpisode = PrototypeEpisode<T>.Build(supportLabels, supportLabels);
            var auxiliaryInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.SupportInput);
            T weight = NumOps.FromDouble(_tadamOptions.AuxiliaryLossWeight);
            bodyGradient = Engine.Add(
                bodyGradient, Engine.Multiply(EmbeddingGradient(auxiliaryInput, auxiliaryEpisode, alpha, encoder), weight));
            using (new NoGradScope<T>())
            {
                auxiliaryLoss = NumOps.Multiply(weight, LossFunction.ComputeTapeLoss(
                    EpisodeLogits(Embed(auxiliaryInput), auxiliaryEpisode, alpha, encoder), auxiliaryEpisode.QueryTarget).GetFlat(0));
            }
        }

        // The metric's and the task encoder's gradients: the same objective, differentiated with respect to them.
        var leaves = new List<Tensor<T>>();
        if (alpha is not null) leaves.Add(alpha);
        if (encoder is not null) leaves.AddRange(encoder.All);

        var alphaGradient = new Vector<T>(_metricScale.Length);
        var encoderGradient = new Vector<T>(_taskEncoder.Length);
        T loss;
        if (leaves.Count == 0)
        {
            using var noGrad = new NoGradScope<T>();
            loss = EpisodeObjective(embeddings, episode, alpha, encoder).GetFlat(0);
        }
        else
        {
            loss = AiDotNet.Training.TapeGradients<T>.Of(
                () => EpisodeObjective(embeddings, episode, alpha, encoder),
                leaves,
                gradients =>
                {
                    alphaGradient = GradientOf(gradients, alpha, alphaGradient.Length);
                    if (encoder is not null)
                    {
                        encoderGradient = Vector<T>.Concatenate(
                            encoder.All.Select(leaf => GradientOf(gradients, leaf, leaf.Length)).ToList());
                    }
                });
        }

        // One outer step on everything the episode trains.
        if (_options.GradientClipThreshold.HasValue && _options.GradientClipThreshold.Value > 0)
        {
            bodyGradient = ClipGradients(bodyGradient, _options.GradientClipThreshold.Value);
        }

        ParamModel.SetParameters(ApplyOuterGradients(ParamModel.GetParameters(), bodyGradient, _options.OuterLearningRate));
        T rate = NumOps.FromDouble(_options.OuterLearningRate);
        if (alpha is not null)
            _metricScale = Engine.Subtract(_metricScale, Engine.Multiply(alphaGradient, rate));
        if (encoder is not null)
            _taskEncoder = Engine.Subtract(_taskEncoder, Engine.Multiply(encoderGradient, rate));

        return NumOps.Add(loss, auxiliaryLoss);
    }

    /// <summary>The extractor's gradient of an episode's query cross-entropy, through prototypes, TEN and metric.</summary>
    private Vector<T> EmbeddingGradient(TInput stackedInput, PrototypeEpisode<T> episode, Tensor<T>? alpha, TaskEncoder? encoder)
    {
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));
        var composed = new EmbeddingClassificationLoss<T>(
            embeddings => EpisodeLogits(embeddings, episode, alpha, encoder), LossFunction, episode.QueryTarget);
        return ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);
    }

    /// <summary>The query cross-entropy plus the L2 penalty <c>lambda * gamma0^2</c> on the TEN post-multiplier.</summary>
    private Tensor<T> EpisodeObjective(Tensor<T> embeddings, PrototypeEpisode<T> episode, Tensor<T>? alpha, TaskEncoder? encoder)
    {
        var loss = LossFunction.ComputeTapeLoss(EpisodeLogits(embeddings, episode, alpha, encoder), episode.QueryTarget);
        if (encoder is null || _tadamOptions.L2Regularization <= 0) return loss;
        var penalty = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(encoder.PostMultiplier, encoder.PostMultiplier), NumOps.FromDouble(_tadamOptions.L2Regularization));
        return Engine.TensorAdd(Engine.Reshape(loss, new[] { 1 }), penalty);
    }

    /// <summary>
    /// Logits <c>-alpha ||h'_q - c_k||^2</c> of an episode's query rows from the stacked support-then-query embeddings.
    /// </summary>
    private Tensor<T> EpisodeLogits(Tensor<T> embeddings, PrototypeEpisode<T> episode, Tensor<T>? alpha, TaskEncoder? encoder)
    {
        var rows = PrototypeMetric<T>.Normalized(embeddings, _tadamOptions.NormalizeEmbeddings);
        var support = Engine.TensorMatMul(episode.SupportSelector, rows);
        var query = Engine.TensorMatMul(episode.QuerySelector, rows);

        var scale = FeatureScale(support, episode.Membership, encoder);
        if (scale is not null)
        {
            support = Engine.TensorMultiply(support, scale);
            query = Engine.TensorMultiply(query, scale);
        }

        var prototypes = PrototypeMetric<T>.Prototypes(support, episode.Membership, null);
        var logits = PrototypeMetric<T>.Logits(
            query, prototypes, ProtoNetsDistanceFunction.Euclidean, 1.0, null, null, episode.ClassSlots, 1.0);
        return alpha is null ? logits : Engine.TensorMultiply(logits, Engine.Reshape(alpha, new[] { 1, 1 }));
    }

    /// <summary>
    /// The FiLM scale <c>1 + gamma0 * TEN(c_bar)</c>, <c>[1, width]</c>, where <c>c_bar</c> is the mean of the
    /// unconditioned class prototypes (Sec. 2.2); null when task conditioning is off.
    /// </summary>
    private Tensor<T>? FeatureScale(Tensor<T> support, Tensor<T> membership, TaskEncoder? encoder)
    {
        if (encoder is null) return null;
        var taskRepresentation = Engine.ReduceMean(
            PrototypeMetric<T>.Prototypes(support, membership, null), new[] { 0 }, keepDims: true);
        var hidden = Engine.ReLU(Engine.TensorAdd(Engine.TensorMatMul(taskRepresentation, encoder.W1), encoder.B1));
        var gamma = Engine.TensorAdd(Engine.TensorMatMul(hidden, encoder.W2), encoder.B2);
        return Engine.TensorAddScalar(
            Engine.TensorMultiply(gamma, Engine.Reshape(encoder.PostMultiplier, new[] { 1, 1 })), NumOps.One);
    }

    /// <summary>Feature rows <c>[rows, width]</c>; the first call sizes the task encoder to the width.</summary>
    private Tensor<T> Embed(TInput input)
    {
        var rows = ClassifierOutputs<T>.AsRows(MetaModel.Predict(input));
        int width = rows.Shape[1];
        if (width != _tadamOptions.EmbeddingDimension)
        {
            throw new InvalidOperationException(
                $"TADAM's feature extractor produced {width}-wide features, but EmbeddingDimension is "
                + $"{_tadamOptions.EmbeddingDimension}. Set EmbeddingDimension to the extractor's output width.");
        }

        EnsureTaskEncoder(width);
        return rows;
    }

    /// <summary>
    /// Sizes the TEN for <paramref name="width"/>-wide features: W1 and W2 uniform in <c>+/- 1 / sqrt(fan_in)</c>,
    /// biases and the post-multiplier zero, so the untrained TEN conditions nothing (<c>gamma = 0</c>) yet the
    /// post-multiplier still receives a gradient.
    /// </summary>
    private void EnsureTaskEncoder(int width)
    {
        if (!_tadamOptions.UseTaskConditioning || _taskEncoderWidth == width) return;

        int hidden = _tadamOptions.TaskEmbeddingDimension;
        var random = RandomGenerator;
        var raw = new double[2 * width * hidden];
        for (int i = 0; i < raw.Length; i++) raw[i] = (2.0 * random.NextDouble()) - 1.0;
        var uniform = new Tensor<double>(raw, new[] { raw.Length }).Cast<T>().ToVector();

        _taskEncoder = Vector<T>.Concatenate(new[]
        {
            Engine.Multiply(uniform.GetSubVector(0, width * hidden), NumOps.FromDouble(1.0 / Math.Sqrt(width))),
            new Vector<T>(hidden),
            Engine.Multiply(uniform.GetSubVector(width * hidden, width * hidden), NumOps.FromDouble(1.0 / Math.Sqrt(hidden))),
            new Vector<T>(width + 1),
        });
        _taskEncoderWidth = width;
    }

    /// <summary>The task encoder's weights as tensors (fresh leaves for one tape), or null without conditioning.</summary>
    private TaskEncoder? TaskEncoderLeaves()
    {
        if (!_tadamOptions.UseTaskConditioning || _taskEncoderWidth == 0) return null;
        int width = _taskEncoderWidth, hidden = _tadamOptions.TaskEmbeddingDimension;
        int offset = 0;
        Tensor<T> Next(int count, int[] shape)
        {
            var leaf = Tensor<T>.FromVector(_taskEncoder.GetSubVector(offset, count), shape);
            offset += count;
            return leaf;
        }

        var w1 = Next(width * hidden, new[] { width, hidden });
        var b1 = Next(hidden, new[] { 1, hidden });
        var w2 = Next(hidden * width, new[] { hidden, width });
        var b2 = Next(width, new[] { 1, width });
        var postMultiplier = Next(1, new[] { 1 });
        return new TaskEncoder(w1, b1, w2, b2, postMultiplier);
    }

    /// <summary>The TEN's weights, in their flattened order.</summary>
    private sealed class TaskEncoder
    {
        public TaskEncoder(Tensor<T> w1, Tensor<T> b1, Tensor<T> w2, Tensor<T> b2, Tensor<T> postMultiplier)
        {
            W1 = w1;
            B1 = b1;
            W2 = w2;
            B2 = b2;
            PostMultiplier = postMultiplier;
            All = new[] { w1, b1, w2, b2, postMultiplier };
        }

        public Tensor<T> W1 { get; }
        public Tensor<T> B1 { get; }
        public Tensor<T> W2 { get; }
        public Tensor<T> B2 { get; }
        public Tensor<T> PostMultiplier { get; }
        public IReadOnlyList<Tensor<T>> All { get; }
    }

    private static int[] ReadLabels(TOutput labels)
        => ClassifierOutputs<T>.Labels(labels, int.MaxValue).Cast<int>().ToArray();

    private static Vector<T> GradientOf(Dictionary<Tensor<T>, Tensor<T>> gradients, Tensor<T>? source, int length)
    {
        if (source is null || !gradients.TryGetValue(source, out var gradient)) return new Vector<T>(length);
        return new Vector<T>(gradient.ToArray());
    }

    #region Test hooks

    /// <summary>One episode's objective at the current state, for gradient checks.</summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        using var noGrad = new NoGradScope<T>();
        var episode = PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));
        var embeddings = Embed(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput));
        var alpha = _tadamOptions.UseMetricScaling ? Tensor<T>.FromVector(_metricScale) : null;
        return EpisodeObjective(embeddings, episode, alpha, TaskEncoderLeaves()).GetFlat(0);
    }

    /// <summary>The exact gradient of <see cref="EpisodeLossForTesting"/> with respect to alpha and the TEN.</summary>
    internal (Vector<T> Alpha, Vector<T> TaskEncoder) MetricGradientForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = Embed(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput));
        }

        var alpha = Tensor<T>.FromVector(_metricScale);
        var encoder = TaskEncoderLeaves();
        var leaves = new List<Tensor<T>> { alpha };
        if (encoder is not null) leaves.AddRange(encoder.All);
        var alphaGradient = new Vector<T>(1);
        var encoderGradient = new Vector<T>(_taskEncoder.Length);
        AiDotNet.Training.TapeGradients<T>.Of(
            () => EpisodeObjective(embeddings, episode, alpha, encoder),
            leaves,
            gradients =>
            {
                alphaGradient = GradientOf(gradients, alpha, 1);
                if (encoder is not null)
                {
                    encoderGradient = Vector<T>.Concatenate(
                        encoder.All.Select(leaf => GradientOf(gradients, leaf, leaf.Length)).ToList());
                }
            });
        return (alphaGradient, encoderGradient);
    }

    /// <summary>Gets or sets a copy of alpha, as a length-1 vector (for tests).</summary>
    internal Vector<T> MetricScaleForTesting { get => _metricScale.Clone(); set => _metricScale = value.Clone(); }

    /// <summary>Gets or sets a copy of the flattened TEN weights (for tests).</summary>
    internal Vector<T> TaskEncoderForTesting { get => _taskEncoder.Clone(); set => _taskEncoder = value.Clone(); }

    #endregion
}
