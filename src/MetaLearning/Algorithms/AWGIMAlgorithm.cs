using AiDotNet.Attributes;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// AWGIM: Attentive Weights Generation for few-shot learning via Information Maximization
/// (Guo &amp; Cheung, CVPR 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// AWGIM generates a linear classifier for every query example in one forward pass, with no inner
/// optimization loop. LEO, which it builds on, does run one. A contextual path (self-attention over the
/// support set) and an attentive path (the query attending to the support set) are decoded into a weight
/// distribution per (query, support row) pair. Averaging over each class's rows gives the query its own
/// classifier. See <see cref="AwgimNetwork{T}"/> for the generator.
/// </para>
/// <para>
/// <b>Objective.</b> L = CE(query) + α1·CE(support, classified by every query's weights)
/// + α2·‖r_c(w) − sg(contextual code)‖² + α3·‖r_q(w) − sg(attentive code)‖².
/// The two reconstruction terms are the variational surrogates of the mutual information between the
/// generated weights and the support set and the query. They keep the weights from discarding what the
/// two paths computed.
/// </para>
/// <para>
/// <b>Meta-update.</b> AdamW, as the authors train it: decoupled weight decay, a staircase learning-rate
/// decay, and gradients clipped first by value, then by each tensor's norm. Should the loss or a gradient be
/// non-finite, the step instead follows the gradient of the kernels' L2 penalty, the reference's recovery
/// rule. The feature encoder receives the same objective's gradient through its embeddings and is updated
/// by the same rule.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most meta-learners adapt a model to a new task by taking a few training steps
/// on the examples. AWGIM instead writes the classifier directly: it reads the task's examples and the
/// example to classify, and outputs the classifier's weights. Because it writes a separate classifier for
/// each example it classifies, it can focus on whichever support examples matter for that query.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Attentive Weights Generation for Few Shot Learning via Information Maximization",
    "https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Attentive_Weights_Generation_for_Few_Shot_Learning_via_Information_Maximization_CVPR_2020_paper.pdf",
    Year = 2020,
    Authors = "Yiluan Guo, Ngai-Man Cheung")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, ReferenceBatchSize = 64,
    Phase = TrainingPhase.PreTraining,
    Source = "Guo & Cheung 2020, official code (main.py): AdamW at 2e-4 with weight decay 1e-6, decayed by 0.2 every "
           + "15,000 steps, 64 tasks per batch, gradients clipped at 0.1 by value and by norm.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class AWGIMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private const double AdamBeta1 = 0.9;
    private const double AdamBeta2 = 0.999;
    private const double AdamEpsilon = 1e-8;

    private readonly AWGIMOptions<T, TInput, TOutput> _awgimOptions;
    private readonly AwgimNetwork<T> _network;
    private readonly bool[] _kernelMask;

    [TrainableParameter]
    private Vector<T> _weights;

    // AdamW state for the generator weights and the feature encoder.
    private readonly Vector<T> _weightMoment1;
    private readonly Vector<T> _weightMoment2;
    private Vector<T>? _bodyMoment1;
    private Vector<T>? _bodyMoment2;
    private int _updates;

    /// <summary>Creates AWGIM over the feature encoder in <paramref name="options"/>.</summary>
    /// <param name="options">The configuration; <see cref="AWGIMOptions{T, TInput, TOutput}.MetaModel"/> is required.</param>
    public AWGIMAlgorithm(AWGIMOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.None)
    {
        if (!options.IsValid())
            throw new ArgumentException("AWGIM configuration is invalid. Check all parameters.", nameof(options));

        _awgimOptions = options;
        _network = new AwgimNetwork<T>(options.EmbeddingDimension, options.LatentDimension, options.NumHeads, options.DecoderLayers);
        _kernelMask = _network.KernelMask();
        _weights = _network.Initialize(RandomGenerator);
        _weightMoment1 = new Vector<T>(_weights.Length);
        _weightMoment2 = new Vector<T>(_weights.Length);
    }

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _awgimOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.AWGIM;

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));

        var body = ParamModel.GetParameters();
        Vector<T>? bodySum = null, weightSum = null;
        T total = NumOps.Zero;
        bool finite = true;
        foreach (var task in taskBatch.Tasks)
        {
            var (loss, bodyGradient, weightGradient) = EpisodeGradient(task, RandomGenerator);
            total = NumOps.Add(total, loss);
            finite &= IsFinite(loss) && IsFinite(bodyGradient) && IsFinite(weightGradient);
            bodySum = Accumulate(bodySum, bodyGradient);
            weightSum = Accumulate(weightSum, weightGradient);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        var bodyGrad = Divide(bodySum ?? new Vector<T>(body.Length), batchSize);
        var weightGrad = Divide(weightSum ?? new Vector<T>(_weights.Length), batchSize);

        if (!finite)
        {
            // The reference's recovery: follow the gradient of the kernels' L2 penalty (0.5·Σw², gradient w)
            // instead of a non-finite one, so one bad batch cannot poison the weights.
            var weights = Tensor<T>.FromVector(_weights);
            weightGrad = Engine.TensorWhere(_kernelMask, weights, Engine.TensorZerosLike(weights)).ToVector();
            bodyGrad = new Vector<T>(body.Length);
        }

        weightGrad = Clip(weightGrad, _network.CreateLeaves(_weights).Ranges);
        bodyGrad = Clip(bodyGrad, new[] { (0, bodyGrad.Length) });

        _updates++;
        double rate = LearningRate();
        _bodyMoment1 ??= new Vector<T>(body.Length);
        _bodyMoment2 ??= new Vector<T>(body.Length);
        ParamModel.SetParameters(AdamW(body, bodyGrad, _bodyMoment1, _bodyMoment2, rate));
        _weights = AdamW(_weights, weightGrad, _weightMoment1, _weightMoment2, rate);

        return NumOps.Divide(total, batchSize);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));

        // Adaptation is the support set itself: the model generates each query's classifier when it predicts.
        var episode = Episode(task);
        var (support, _) = Embed(task, episode);
        return new AWGIMModel<T, TInput, TOutput>(MetaModel, _network, CloneVector(_weights), support,
            episode.Membership, ClassAverager(episode, support.Shape[0]), episode.ClassSlots, _awgimOptions);
    }

    /// <inheritdoc/>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    /// <summary>The objective's gradient for one episode: the feature encoder's and the generator's.</summary>
    private (T Loss, Vector<T> Body, Vector<T> Weights) EpisodeGradient(IMetaLearningTask<T, TInput, TOutput> task, Random random)
    {
        var episode = Episode(task);
        int supportRows = episode.SupportSelector.Shape[0];
        var averager = ClassAverager(episode, supportRows);
        var supportTarget = SupportTarget(episode, supportRows);
        int seed = random.Next();

        // The feature encoder's gradient: the objective rebuilt from its embeddings on the tape, the generator
        // weights held constant. The same seed replays the same dropout and weight noise in both passes.
        var frozen = _network.CreateLeaves(_weights);
        var composed = new EmbeddingObjectiveLoss<T>(embeddings =>
        {
            var engine = AiDotNetEngine.Current;
            var s = CheckWidth(engine.TensorMatMul(episode.SupportSelector, embeddings));
            var q = engine.TensorMatMul(episode.QuerySelector, embeddings);
            return Objective(frozen, s, q, episode, averager, supportTarget, Training(seed));
        });
        var stackedInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));
        var bodyGradient = ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);

        // The generator's gradient from fixed embeddings.
        var (support, query) = Embed(task, episode);
        var leaves = _network.CreateLeaves(_weights);
        T loss;
        Vector<T> weightGradient;
        weightGradient = new Vector<T>(_weights.Length);
        loss = AiDotNet.Training.TapeGradients<T>.Of(
            () => Objective(leaves, support, query, episode, averager, supportTarget, Training(seed)),
            leaves.All,
            gradients => weightGradient = leaves.FlattenGradients(gradients, _weights.Length),
            persistent: true);

        return (loss, bodyGradient, weightGradient);
    }

    private AwgimNetwork<T>.Noise Training(int seed)
        => AwgimNetwork<T>.Noise.Training(RandomHelper.CreateSeededRandom(seed), _awgimOptions.DropoutRate);

    private Tensor<T> Objective(AwgimNetwork<T>.Leaves leaves, Tensor<T> support, Tensor<T> query,
        PrototypeEpisode<T> episode, Tensor<T> averager, Tensor<T> supportTarget, AwgimNetwork<T>.Noise noise)
    {
        var engine = AiDotNetEngine.Current;
        var generated = _network.Generate(leaves, support, query, episode.Membership, averager, noise);

        var queryLogits = AwgimNetwork<T>.QueryLogits(engine, noise.Drop(query), generated.ClassWeights);
        var objective = Scalar(LossFunction.ComputeTapeLoss(queryLogits, episode.QueryTarget));

        if (_awgimOptions.SupportClassificationWeight > 0)
        {
            int queryRows = query.Shape[0];
            var supportLogits = AwgimNetwork<T>.SupportLogits(engine, noise.Drop(support), generated.ClassWeights);
            var tiledTarget = engine.Reshape(engine.TensorBroadcastTo(
                engine.Reshape(supportTarget, new[] { 1, supportTarget.Length }), new[] { queryRows, supportTarget.Length }),
                new[] { queryRows * supportTarget.Length });
            objective = AddWeighted(objective, Scalar(LossFunction.ComputeTapeLoss(supportLogits, tiledTarget)),
                _awgimOptions.SupportClassificationWeight);
        }

        if (_awgimOptions.ContextReconstructionWeight > 0)
        {
            objective = AddWeighted(objective,
                _network.Reconstruction(leaves, "reconstruct.context", generated.Sampled, generated.ContextCode, noise),
                _awgimOptions.ContextReconstructionWeight);
        }

        if (_awgimOptions.QueryReconstructionWeight > 0)
        {
            objective = AddWeighted(objective,
                _network.Reconstruction(leaves, "reconstruct.query", generated.Sampled, generated.QueryCode, noise),
                _awgimOptions.QueryReconstructionWeight);
        }

        return objective;
    }

    #region Optimizer

    private double LearningRate()
    {
        int decays = _awgimOptions.LearningRateDecaySteps > 0 ? (_updates - 1) / _awgimOptions.LearningRateDecaySteps : 0;
        return _awgimOptions.OuterLearningRate * Math.Pow(_awgimOptions.LearningRateDecayRate, decays);
    }

    /// <summary>AdamW: decoupled weight decay (applied as w·decay, as TensorFlow's AdamW does), then Adam.</summary>
    private Vector<T> AdamW(Vector<T> parameters, Vector<T> gradient, Vector<T> moment1, Vector<T> moment2, double rate)
    {
        double correction1 = 1 - Math.Pow(AdamBeta1, _updates);
        double correction2 = 1 - Math.Pow(AdamBeta2, _updates);
        int n = parameters.Length;

        // m = b1 m + (1 - b1) g, v = b2 v + (1 - b2) g^2, both kept in the caller's moment buffers.
        var m = Engine.Add(Engine.Multiply(moment1, NumOps.FromDouble(AdamBeta1)),
            Engine.Multiply(gradient, NumOps.FromDouble(1 - AdamBeta1)));
        var v = Engine.Add(Engine.Multiply(moment2, NumOps.FromDouble(AdamBeta2)),
            Engine.Multiply(Engine.Multiply(gradient, gradient), NumOps.FromDouble(1 - AdamBeta2)));
        m.AsSpan().CopyTo(moment1.AsWritableSpan());
        v.AsSpan().CopyTo(moment2.AsWritableSpan());

        // w <- w - decay w, then w <- w - rate (m / c1) / (sqrt(v / c2) + eps)
        var decayed = Engine.Multiply(parameters, NumOps.FromDouble(1 - _awgimOptions.WeightDecay));
        var denominator = Engine.Add(
            Engine.Sqrt(Engine.Divide(v, NumOps.FromDouble(correction2))), Engine.Fill(n, NumOps.FromDouble(AdamEpsilon)));
        var step = Engine.Divide(Engine.Multiply(m, NumOps.FromDouble(rate / correction1)), denominator);
        return Engine.Subtract(decayed, step);
    }

    /// <summary>Clip by value, then each tensor's gradient by its own norm, as the reference does.</summary>
    private Vector<T> Clip(Vector<T> gradient, IEnumerable<(int Offset, int Length)> tensors)
    {
        var clipped = CloneVector(gradient);
        if (_awgimOptions.GradientClipThreshold is double bound)
        {
            clipped = Engine.Clamp(clipped, NumOps.FromDouble(-bound), NumOps.FromDouble(bound));
        }

        if (_awgimOptions.GradientNormClipThreshold is double normBound)
        {
            // One scale per tensor range (1 when its norm is within the bound), laid out over the whole vector.
            T limit = NumOps.FromDouble(normBound);
            var scales = new List<Vector<T>>();
            int position = 0;
            foreach (var (offset, length) in tensors.OrderBy(range => range.Offset))
            {
                if (offset > position) scales.Add(Engine.Fill(offset - position, NumOps.One));
                T norm = Engine.Norm(clipped.GetSubVector(offset, length));
                T factor = NumOps.LessThanOrEquals(norm, limit) ? NumOps.One : NumOps.Divide(limit, norm);
                scales.Add(Engine.Fill(length, factor));
                position = offset + length;
            }

            if (position < clipped.Length) scales.Add(Engine.Fill(clipped.Length - position, NumOps.One));
            if (scales.Count > 0) clipped = Engine.Multiply(clipped, Vector<T>.Concatenate(scales));
        }

        return clipped;
    }

    #endregion

    #region Episode

    private (Tensor<T> Support, Tensor<T> Query) Embed(IMetaLearningTask<T, TInput, TOutput> task, PrototypeEpisode<T> episode)
    {
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = ClassifierOutputs<T>.AsRows(
                MetaModel.Predict(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput)));
        }

        var engine = AiDotNetEngine.Current;
        var support = CheckWidth(engine.TensorMatMul(episode.SupportSelector, embeddings));
        return (support, engine.TensorMatMul(episode.QuerySelector, embeddings));
    }

    private Tensor<T> CheckWidth(Tensor<T> embeddings)
    {
        if (embeddings.Shape[1] != _awgimOptions.EmbeddingDimension)
        {
            throw new InvalidOperationException(
                $"The feature encoder emits {embeddings.Shape[1]}-wide embeddings per example but EmbeddingDimension is "
                + $"{_awgimOptions.EmbeddingDimension}. EmbeddingDimension is the width of the representation AWGIM "
                + "classifies - the feature encoder's per-example output width.");
        }

        return embeddings;
    }

    private PrototypeEpisode<T> Episode(IMetaLearningTask<T, TInput, TOutput> task)
        => PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));

    private int[] ReadLabels(TOutput labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, _awgimOptions.NumClasses);
        return Engine.TensorRound(tensor).Cast<int>().ToArray();
    }

    /// <summary><c>[classes, support]</c>: each class's row-average over its support rows.</summary>
    internal static Tensor<T> ClassAverager(PrototypeEpisode<T> episode, int rows)
    {
        var engine = AiDotNetEngine.Current;
        int classes = episode.ClassSlots.Length;
        var shape = new[] { classes, rows };
        var membership = engine.Reshape(episode.Membership, shape);
        var members = engine.TensorBroadcastTo(engine.ReduceSum(membership, new[] { 1 }, true), shape);
        return engine.TensorDivide(membership, members);
    }

    private static Tensor<T> SupportTarget(PrototypeEpisode<T> episode, int rows)
    {
        // The last class whose membership exceeds 0.5 (0 when none does): the largest c * [membership > 0.5].
        var engine = AiDotNetEngine.Current;
        int classes = episode.ClassSlots.Length;
        if (classes == 0) return new Tensor<T>(new[] { rows });

        var shape = new[] { classes, rows };
        var member = engine.TensorGreaterThan(engine.Reshape(episode.Membership, shape), NumOps.FromDouble(0.5));
        var classIndex = new Tensor<int>(Enumerable.Range(0, classes).ToArray(), new[] { classes, 1 }).Cast<T>();
        var labelled = engine.TensorMultiply(member, engine.TensorBroadcastTo(classIndex, shape));
        return engine.ReduceMax(labelled, new[] { 0 }, false);
    }

    #endregion

    #region Test hooks

    /// <summary>The flat range of the generator weights whose names start with <paramref name="prefix"/>.</summary>
    internal (int Offset, int Length) WeightRangeForTesting(string prefix) => _network.RangeOf(prefix);

    /// <summary>The generator weights, for tests that check gradients against finite differences.</summary>
    internal Vector<T> WeightsForTesting { get => CloneVector(_weights); set => _weights = CloneVector(value); }

    /// <summary>One episode's objective at a fixed noise seed: evaluation mode when <paramref name="seed"/> is null.</summary>
    internal T ObjectiveForTesting(IMetaLearningTask<T, TInput, TOutput> task, int? seed)
    {
        var episode = Episode(task);
        var (support, query) = Embed(task, episode);
        int rows = episode.SupportSelector.Shape[0];
        var noise = seed.HasValue ? Training(seed.Value) : AwgimNetwork<T>.Noise.None;
        return Objective(_network.CreateLeaves(_weights), support, query, episode, ClassAverager(episode, rows),
            SupportTarget(episode, rows), noise)[0];
    }

    /// <summary>The analytic gradient of <see cref="ObjectiveForTesting"/> with respect to the generator weights.</summary>
    internal Vector<T> WeightGradientForTesting(IMetaLearningTask<T, TInput, TOutput> task, int? seed)
    {
        var episode = Episode(task);
        var (support, query) = Embed(task, episode);
        int rows = episode.SupportSelector.Shape[0];
        var noise = seed.HasValue ? Training(seed.Value) : AwgimNetwork<T>.Noise.None;
        var leaves = _network.CreateLeaves(_weights);
        var weightGradient = new Vector<T>(_weights.Length);
        AiDotNet.Training.TapeGradients<T>.Of(
            () => Objective(leaves, support, query, episode, ClassAverager(episode, rows), SupportTarget(episode, rows), noise),
            leaves.All,
            gradients => weightGradient = leaves.FlattenGradients(gradients, _weights.Length),
            persistent: true);
        return weightGradient;
    }

    #endregion

    #region Helpers

    private bool IsFinite(T value)
    {
        double v = NumOps.ToDouble(value);
        return !double.IsNaN(v) && !double.IsInfinity(v);
    }

    // x * 0 is NaN exactly where x is NaN or infinite, so the sum of the zeroed vector is finite iff every entry is.
    private bool IsFinite(Vector<T> values)
        => IsFinite(Engine.Sum(Engine.Multiply(values, NumOps.Zero)));

    private static Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
    {
        if (sum is null) return CloneVector(values);
        return AiDotNetEngine.Current.Add(sum, values);
    }

    private static Vector<T> Divide(Vector<T> values, T by) => AiDotNetEngine.Current.Divide(values, by);

    private static Vector<T> CloneVector(Vector<T> values) => values.Clone();

    private static Tensor<T> Scalar(Tensor<T> value) => AiDotNetEngine.Current.Reshape(value, new[] { 1 });

    private static Tensor<T> AddWeighted(Tensor<T> total, Tensor<T> term, double weight)
        => AiDotNetEngine.Current.TensorAdd(total, AiDotNetEngine.Current.TensorMultiplyScalar(term, NumOps.FromDouble(weight)));

    #endregion
}
