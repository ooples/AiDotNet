using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
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
/// Implementation of MCL: episodic meta-learning with a supervised contrastive auxiliary loss.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Two objectives train one embedding. The episodic term is the prototypical loss (Snell et al. 2017): each class's
/// support embeddings average into a prototype and each query is scored by a softmax over negative squared
/// distances. The auxiliary term is the supervised contrastive loss (Khosla et al. 2020, eq. 2): over the
/// unit-normalised outputs of a projection head, each example is pulled toward every other example of its class and
/// pushed away from the rest,
/// <c>L_i = -mean over p in P(i) of log( exp(z_i . z_p / tau) / sum over a != i of exp(z_i . z_a / tau) )</c>,
/// where <c>P(i)</c> is the set of other examples sharing example <c>i</c>'s label. The objective is
/// <c>L_proto + ContrastiveWeight * L_supcon</c>, and both terms reach the embedding through exact tape gradients.
/// </para>
/// <para>
/// <b>What this replaced.</b> The contrastive term compared single scalars of a flattened feature vector - the
/// "cosine similarity" of two scalars is just the sign of their product - and inferred each "example's" class from
/// its position in that vector rather than from the labels. The projection head indexed one weight matrix through a
/// wrapping modulus and was trained by simultaneous perturbation rather than by its gradient. No prototypes were
/// computed at all, despite the documented algorithm, and the embedding's own update differentiated its raw output
/// against the class indices, so neither the episodic structure nor the contrastive term was in the meta-gradient.
/// Adaptation multiplied the shared backbone's parameters by a scalar derived from the projections, which mutated
/// the meta-model every time a task was adapted.
/// </para>
/// <para><b>For Beginners:</b> MCL teaches the features two things at once: be good at the few-shot task (the
/// episodic loss), and be well organised, with same-class examples close together and different-class examples far
/// apart (the contrastive loss).
/// </para>
/// <para>
/// References: Snell, J., Swersky, K., &amp; Zemel, R. (2017). Prototypical Networks for Few-shot Learning.
/// Khosla, P., Teterwak, P., Wang, C., et al. (2020). Supervised Contrastive Learning.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Supervised Contrastive Learning",
    "https://arxiv.org/abs/2004.11362",
    Year = 2020,
    Authors = "Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, et al.")]
[ResearchPaper("Prototypical Networks for Few-shot Learning",
    "https://arxiv.org/abs/1703.05175",
    Year = 2017,
    Authors = "Snell, J., Swersky, K., & Zemel, R.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MCLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MCLOptions<T, TInput, TOutput> _mclOptions;

    /// <summary>
    /// The projection head, flattened as <c>W1 [ProjectionDim, width]</c>, <c>b1</c>,
    /// <c>W2 [ProjectionDim, ProjectionDim]</c>, <c>b2</c>. Empty until the first episode shows the embedding width.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _projectionWeights = new Vector<T>(0);

    /// <summary>The embedding width the projection head was sized for; zero before the first episode.</summary>
    private int _embeddingWidth;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MCL;

    /// <summary>Initializes a new MCL meta-learner.</summary>
    /// <param name="options">Configuration options for MCL.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    public MCLAlgorithm(MCLOptions<T, TInput, TOutput> options)
        : base(
            (options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _mclOptions = options;
    }

    /// <summary>
    /// Performs one meta-training step: each task's prototypical loss plus its supervised contrastive term,
    /// differentiated exactly into the embedding network and the projection head.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on.</param>
    /// <returns>The average combined loss across the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        EnsureProjectionShape(taskBatch.Tasks);

        var body = ParamModel.GetParameters();
        Vector<T>? bodyGradient = null, projectionGradient = null;
        T totalLoss = NumOps.Zero;
        foreach (var task in taskBatch.Tasks)
        {
            var (loss, taskBody, taskProjection) = EpisodeGradient(task);
            totalLoss = NumOps.Add(totalLoss, loss);
            bodyGradient = Accumulate(bodyGradient, taskBody);
            projectionGradient = Accumulate(projectionGradient, taskProjection);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        bodyGradient = Scale(bodyGradient ?? new Vector<T>(body.Length), batchSize);
        projectionGradient = Scale(projectionGradient ?? new Vector<T>(_projectionWeights.Length), batchSize);

        if (_mclOptions.GradientClipThreshold.HasValue && _mclOptions.GradientClipThreshold.Value > 0)
        {
            double threshold = _mclOptions.GradientClipThreshold.Value;
            bodyGradient = ClipGradients(bodyGradient, threshold);
            if (projectionGradient.Length > 0) projectionGradient = ClipGradients(projectionGradient, threshold);
        }

        double beta = _mclOptions.OuterLearningRate;
        ParamModel.SetParameters(ApplyGradients(body, bodyGradient, beta));
        if (_projectionWeights.Length > 0)
        {
            _projectionWeights = ApplyGradients(_projectionWeights, projectionGradient, beta);
        }

        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Adaptation is the prototypical one: the class prototypes of the support set, under the embedding as
    /// meta-training left it. It used to rescale the shared backbone's parameters by a factor derived from the
    /// projections, which mutated the meta-model itself.
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        EnsureProjectionShape(new[] { task });
        var episode = PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), Array.Empty<int>());

        using var noGrad = new NoGradScope<T>();
        var support = ClassifierOutputs<T>.AsRows(MetaModel.Predict(task.SupportInput));
        var prototypes = PrototypeMetric<T>.Prototypes(support, episode.Membership, attention: null);
        var projected = _projectionWeights.Length > 0 ? Project(support).ToVector() : null;
        return new MCLModel<T, TInput, TOutput>(MetaModel, prototypes, episode.ClassSlots, NumOps, projected);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns class probabilities, so this is the configured loss of their logarithm against the
    /// class indices - cross-entropy by default. A Vector output carries one predicted class per example instead,
    /// and its loss is the classification error rate.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    #region Episode

    /// <summary>
    /// One episode's combined loss and its exact gradient with respect to the embedding network and the projection
    /// head.
    /// </summary>
    private (T Loss, Vector<T> Body, Vector<T> Projection) EpisodeGradient(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var supportLabels = ReadLabels(task.SupportOutput);
        var queryLabels = ReadLabels(task.QueryOutput);
        var episode = PrototypeEpisode<T>.Build(supportLabels, queryLabels);
        var allLabels = supportLabels.Concat(queryLabels).ToArray();
        var stackedInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));

        // The embedding's gradient: the whole objective - prototypes, query scores and the contrastive term -
        // rebuilt from the embeddings on the tape.
        var head = ProjectionHead();
        var composed = new EmbeddingObjectiveLoss<T>(embeddings => Objective(embeddings, episode, allLabels, head));
        var bodyGradient = ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);

        // The head's gradient: the same objective, differentiated with respect to the projection weights with the
        // embeddings held fixed.
        Tensor<T> embeddingRows;
        using (new NoGradScope<T>())
        {
            embeddingRows = ClassifierOutputs<T>.AsRows(MetaModel.Predict(stackedInput));
        }

        T loss;
        var projectionGradient = new Vector<T>(_projectionWeights.Length);
        if (head.Count == 0)
        {
            using var noGrad = new NoGradScope<T>();
            loss = Objective(embeddingRows, episode, allLabels, head)[0];
        }
        else
        {
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = true });
            var objective = Objective(embeddingRows, episode, allLabels, head);
            loss = objective[0];
            var gradients = tape.ComputeGradients(objective, head);
            int offset = 0;
            foreach (var leaf in head)
            {
                if (gradients.TryGetValue(leaf, out var gradient))
                {
                    for (int i = 0; i < leaf.Length; i++) projectionGradient[offset + i] = gradient[i];
                }

                offset += leaf.Length;
            }
        }

        return (loss, bodyGradient, projectionGradient);
    }

    /// <summary>The prototypical loss plus the weighted supervised contrastive loss, from the stacked embeddings.</summary>
    private Tensor<T> Objective(
        Tensor<T> embeddings, PrototypeEpisode<T> episode, int[] allLabels, IReadOnlyList<Tensor<T>> head)
    {
        var engine = AiDotNetEngine.Current;
        var support = engine.TensorMatMul(episode.SupportSelector, embeddings);
        var query = engine.TensorMatMul(episode.QuerySelector, embeddings);

        var prototypes = PrototypeMetric<T>.Prototypes(support, episode.Membership, attention: null);
        var logits = PrototypeMetric<T>.Logits(
            query, prototypes, ProtoNetsDistanceFunction.Euclidean, mahalanobisScaling: 1.0,
            mahalanobisLogScale: null, classLogScale: null, episode.ClassSlots, temperature: 1.0);
        var objective = Scalar(LossFunction.ComputeTapeLoss(logits, episode.QueryTarget));

        if (head.Count == 0 || _mclOptions.ContrastiveWeight <= 0) return objective;

        var contrastive = SupervisedContrastiveLoss(Project(embeddings, head), allLabels);
        return engine.TensorAdd(objective,
            engine.TensorMultiplyScalar(contrastive, NumOps.FromDouble(_mclOptions.ContrastiveWeight)));
    }

    /// <summary>
    /// Khosla et al. 2020, eq. 2: over unit-normalised projections, each anchor is pulled toward the mean log
    /// probability of its positives - the other examples carrying its label - against every other example as a
    /// negative.
    /// </summary>
    private Tensor<T> SupervisedContrastiveLoss(Tensor<T> projections, int[] labels)
    {
        var engine = AiDotNetEngine.Current;
        int rows = projections.Shape[0];
        var unit = PrototypeMetric<T>.Normalized(projections, normalize: true);
        var similarity = engine.TensorMultiplyScalar(
            engine.TensorMatMul(unit, engine.TensorTranspose(unit)),
            NumOps.FromDouble(1.0 / Math.Max(_mclOptions.ContrastiveTemperature, 1e-10)));

        // The denominator runs over a != i, so the diagonal is pushed out of the log-sum-exp by a large negative
        // constant rather than dropped, which keeps every step a whole-tensor operation.
        var selfMask = new Tensor<T>(new[] { rows, rows });
        var weights = new Tensor<T>(new[] { rows, rows });
        var counts = new int[rows];
        for (int i = 0; i < rows; i++)
        {
            selfMask[i * rows + i] = NumOps.FromDouble(-1e9);
            for (int j = 0; j < rows; j++)
            {
                if (i != j && labels[i] == labels[j]) counts[i]++;
            }
        }

        // An anchor with no positive has no term in the paper's sum, so it is left out of both the weights and the
        // average.
        int anchors = 0;
        for (int i = 0; i < rows; i++)
        {
            if (counts[i] == 0) continue;
            anchors++;
            for (int j = 0; j < rows; j++)
            {
                if (i != j && labels[i] == labels[j]) weights[i * rows + j] = NumOps.FromDouble(1.0 / counts[i]);
            }
        }

        var masked = engine.TensorAdd(similarity, selfMask);
        var max = engine.ReduceMax(masked, new[] { 1 }, true, out _);
        var stable = engine.StopGradient(max);
        var logSumExp = engine.TensorAdd(
            engine.TensorLog(engine.ReduceSum(
                engine.TensorExp(engine.TensorAdd(masked, engine.TensorNegate(stable))), new[] { 1 }, keepDims: true)),
            stable);

        if (anchors == 0)
        {
            return Scalar(engine.TensorMultiplyScalar(engine.ReduceSum(similarity, null), NumOps.Zero));
        }

        var logProbabilities = engine.TensorAdd(similarity, engine.TensorNegate(logSumExp));
        var weighted = engine.ReduceSum(engine.TensorMultiply(logProbabilities, weights), null);
        return Scalar(engine.TensorMultiplyScalar(weighted, NumOps.FromDouble(-1.0 / anchors)));
    }

    /// <summary>The projection head's two layers, <c>W2 relu(W1 h + b1) + b2</c>, one row per example.</summary>
    private Tensor<T> Project(Tensor<T> embeddings, IReadOnlyList<Tensor<T>>? head = null)
    {
        var engine = AiDotNetEngine.Current;
        var leaves = head ?? ProjectionHead();
        if (leaves.Count == 0) return embeddings;
        var hidden = engine.ReLU(engine.TensorAdd(
            engine.TensorMatMul(embeddings, engine.TensorTranspose(leaves[0])),
            engine.Reshape(leaves[1], new[] { 1, leaves[1].Length })));
        return engine.TensorAdd(
            engine.TensorMatMul(hidden, engine.TensorTranspose(leaves[2])),
            engine.Reshape(leaves[3], new[] { 1, leaves[3].Length }));
    }

    /// <summary>The projection head's weights as tape leaves, in the order the flat layout holds them.</summary>
    private List<Tensor<T>> ProjectionHead()
    {
        var leaves = new List<Tensor<T>>();
        if (_projectionWeights.Length == 0 || _embeddingWidth == 0) return leaves;

        int projection = _mclOptions.ProjectionDim, position = 0;
        void Take(params int[] shape)
        {
            var leaf = new Tensor<T>(shape);
            for (int i = 0; i < leaf.Length; i++) leaf[i] = _projectionWeights[position + i];
            position += leaf.Length;
            leaves.Add(leaf);
        }

        Take(projection, _embeddingWidth);
        Take(projection);
        Take(projection, projection);
        Take(projection);
        return leaves;
    }

    /// <summary>Sizes the projection head to the embedding width, with the usual fan-in uniform initialisation.</summary>
    private void EnsureProjectionShape(IEnumerable<IMetaLearningTask<T, TInput, TOutput>> tasks)
    {
        if (_projectionWeights.Length > 0) return;
        var first = tasks.FirstOrDefault();
        if (first is null) return;

        int width;
        using (new NoGradScope<T>())
        {
            width = ClassifierOutputs<T>.AsRows(MetaModel.Predict(first.SupportInput)).Shape[1];
        }

        int projection = _mclOptions.ProjectionDim;
        _embeddingWidth = width;
        _projectionWeights = new Vector<T>(projection * width + projection + projection * projection + projection);
        int position = 0;
        void Fill(int count, int fanIn)
        {
            double bound = 1.0 / Math.Sqrt(Math.Max(fanIn, 1));
            for (int i = 0; i < count; i++)
            {
                _projectionWeights[position + i] = NumOps.FromDouble((2.0 * RandomGenerator.NextDouble() - 1.0) * bound);
            }

            position += count;
        }

        Fill(projection * width, width);
        Fill(projection, width);
        Fill(projection * projection, projection);
        Fill(projection, projection);
    }

    private static int[] ReadLabels(TOutput labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, int.MaxValue);
        var indices = new int[tensor.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
        return indices;
    }

    #endregion

    #region Test hooks

    /// <summary>One episode's combined loss and exact gradient from the current state, for gradient checks.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> Projection) EpisodeGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureProjectionShape(new[] { task });
        return EpisodeGradient(task);
    }

    /// <summary>One episode's combined loss from the current state.</summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureProjectionShape(new[] { task });
        var supportLabels = ReadLabels(task.SupportOutput);
        var queryLabels = ReadLabels(task.QueryOutput);
        var episode = PrototypeEpisode<T>.Build(supportLabels, queryLabels);
        using var noGrad = new NoGradScope<T>();
        var embeddings = ClassifierOutputs<T>.AsRows(
            MetaModel.Predict(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput)));
        return Objective(embeddings, episode, supportLabels.Concat(queryLabels).ToArray(), ProjectionHead())[0];
    }

    /// <summary>One episode's supervised contrastive term on its own.</summary>
    internal T ContrastiveLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureProjectionShape(new[] { task });
        var labels = ReadLabels(task.SupportOutput).Concat(ReadLabels(task.QueryOutput)).ToArray();
        using var noGrad = new NoGradScope<T>();
        var embeddings = ClassifierOutputs<T>.AsRows(
            MetaModel.Predict(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput)));
        return SupervisedContrastiveLoss(Project(embeddings), labels)[0];
    }

    /// <summary>Gets or sets a copy of the projection head's weights (for tests).</summary>
    internal Vector<T> ProjectionWeightsForTesting
    {
        get => CloneVector(_projectionWeights);
        set => _projectionWeights = CloneVector(value);
    }

    #endregion

    #region Helpers

    private static Tensor<T> Scalar(Tensor<T> value) => AiDotNetEngine.Current.Reshape(value, new[] { 1 });

    private Vector<T> CloneVector(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) copy[i] = source[i];
        return copy;
    }

    private Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
    {
        if (sum is null) return CloneVector(values);
        for (int i = 0; i < sum.Length; i++) sum[i] = NumOps.Add(sum[i], values[i]);
        return sum;
    }

    private Vector<T> Scale(Vector<T> values, T divisor)
    {
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.Divide(values[i], divisor);
        return values;
    }

    #endregion
}

/// <summary>The prototypes MCL adapted to one task, with the projections that shaped its embedding.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// The adapted state of MCL for one task: its own copy of the embedding and the class prototypes of the support set.
/// <see cref="Predict"/> returns class probabilities, <c>[rows, classes]</c>, for Tensor and Matrix outputs, and the
/// most probable class of each example for a Vector output.
/// </para>
/// <para><b>For Beginners:</b> After meta-training, adapting to a task just averages each class's support examples
/// into a prototype. A new example is labelled by the nearest prototype.
/// </para>
/// </remarks>
internal class MCLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly INumericOperations<T> _numOps;
    private readonly Tensor<T> _prototypes;
    private readonly int[] _classSlots;
    private readonly Vector<T>? _projectedSupport;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    /// <remarks>The support set as the contrastive projection head sees it.</remarks>
    public Vector<T>? AdaptedSupportFeatures => _projectedSupport;

    /// <inheritdoc/>
    /// <remarks>
    /// None: adaptation is prototypical, so it never rescales the embedding network's parameters. It used to.
    /// </remarks>
    public double[]? ParameterModulationFactors => null;

    /// <summary>Initializes the adapted model with the task's prototypes.</summary>
    /// <param name="featureEncoder">The trained feature encoder; the model keeps its own copy.</param>
    /// <param name="prototypes">The support set's class prototypes, <c>[classes, width]</c>.</param>
    /// <param name="classSlots">The class label behind each prototype, in ascending order.</param>
    /// <param name="numOps">Numeric operations for type T.</param>
    /// <param name="projectedSupport">The projected support features, or null before the head is sized.</param>
    public MCLModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Tensor<T> prototypes,
        int[] classSlots,
        INumericOperations<T> numOps,
        Vector<T>? projectedSupport)
    {
        Guard.NotNull(featureEncoder);
        Guard.NotNull(numOps);
        _featureEncoder = featureEncoder.DeepCopy();
        _prototypes = prototypes;
        _classSlots = classSlots;
        _numOps = numOps;
        _projectedSupport = projectedSupport;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var query = ClassifierOutputs<T>.AsRows(_featureEncoder.Predict(input));
        var logits = PrototypeMetric<T>.Logits(
            query, _prototypes, ProtoNetsDistanceFunction.Euclidean, mahalanobisScaling: 1.0,
            mahalanobisLogScale: null, classLogScale: null, _classSlots, temperature: 1.0);
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

    /// <summary>Training is not supported on an adapted model.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException(
            "Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
