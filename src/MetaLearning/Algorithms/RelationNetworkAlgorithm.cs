using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Modules;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Relation Networks algorithm for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Relation Networks (Sung et al. 2018) learn the comparison itself. An embedding module <c>f</c> embeds the sample
/// (support) and query examples; for K-shot tasks each class's embeddings are summed element-wise into one class
/// feature; a relation module <c>g</c> maps the concatenation <c>C(f(x_i), f(x_j))</c> to a relation score
/// <c>r_(i,j)</c> in (0, 1) (eq. 1). Both modules are trained end to end to regress the scores onto the match
/// indicator with mean squared error, <c>sum (r_(i,j) - 1(y_i == y_j))^2</c> (eq. 2).
/// </para>
/// <para>
/// <b>What used to happen instead.</b> The embedding module was never called: tensor inputs went to the relation
/// module as raw features and matrix inputs became an empty tensor. The relation module was one dot product and a
/// sigmoid. The loss was cross-entropy over a softmax of mean per-sample scores, and the embedding network's update
/// differentiated its own raw output against the labels, never the relation scores.
/// </para>
/// <para>
/// <b>Extensions</b>, all learned on the same objective and off by default except where the paper is the default:
/// the relation module's architecture (<see cref="RelationModuleType"/>), how a class's support examples meet a query
/// (<see cref="RelationAggregationMethod"/>), several relation heads averaged, a learned linear map of the
/// embeddings before pairing, and dropout inside the relation module while meta-training.
/// </para>
/// <para><b>For Beginners:</b> Relation Networks learn how to compare examples: a small network looks at a class's
/// examples and a new example side by side and says how related they are. The class with the highest relation
/// score is the prediction.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Learning to Compare: Relation Network for Few-Shot Learning",
    "https://arxiv.org/abs/1711.06025",
    Year = 2018,
    Authors = "Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H. S., & Hospedales, T. M.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class RelationNetworkAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly RelationNetworkOptions<T, TInput, TOutput> _relationOptions;

    /// <summary>The relation heads' weights, head after head; empty until the first episode shows the width.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _relationWeights = new Vector<T>(0);

    /// <summary>The learned linear map of the embeddings, <c>[width, width]</c>, identity at the start.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _featureTransform = new Vector<T>(0);

    /// <summary>The bilinear attention of <see cref="RelationAggregationMethod.Attention"/> pooling, zero at the start.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _poolingWeights = new Vector<T>(0);

    /// <summary>The per-shot log weights of <see cref="RelationAggregationMethod.LearnedWeighting"/>, zero at the start.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _shotWeights = new Vector<T>(0);

    /// <summary>The embedding width the learned parts were sized for; zero before the first episode.</summary>
    private int _embeddingWidth;

    /// <summary>
    /// Initializes a new instance of the RelationNetworkAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for Relation Networks.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    public RelationNetworkAlgorithm(RelationNetworkOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? new MeanSquaredErrorLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _relationOptions = options;

        if (!_relationOptions.IsValid())
        {
            throw new ArgumentException("Relation Network configuration is invalid. Check all parameters.", nameof(options));
        }
    }

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.RelationNetwork;

    /// <inheritdoc/>
    /// <remarks>
    /// One step over the batch: each episode's relation loss, differentiated exactly into the embedding module and
    /// every learned part of the comparison, averaged over the batch. The L2 strengths are weight decay on the
    /// gradients (they used to be added to the reported loss only, and changed nothing).
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        EnsureRelationShapes(taskBatch.Tasks);

        var body = ParamModel.GetParameters();
        Vector<T>? bodyGradient = null, relationGradient = null, transformGradient = null, poolingGradient = null, shotGradient = null;
        T totalLoss = NumOps.Zero;
        foreach (var task in taskBatch.Tasks)
        {
            var (loss, taskBody, relation, transform, pooling, shots) = EpisodeGradient(task, training: true);
            totalLoss = NumOps.Add(totalLoss, loss);
            bodyGradient = Accumulate(bodyGradient, taskBody);
            relationGradient = Accumulate(relationGradient, relation);
            transformGradient = Accumulate(transformGradient, transform);
            poolingGradient = Accumulate(poolingGradient, pooling);
            shotGradient = Accumulate(shotGradient, shots);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        bodyGradient = Scale(bodyGradient ?? new Vector<T>(body.Length), batchSize);
        relationGradient = Scale(relationGradient ?? new Vector<T>(_relationWeights.Length), batchSize);
        transformGradient = Scale(transformGradient ?? new Vector<T>(_featureTransform.Length), batchSize);
        poolingGradient = Scale(poolingGradient ?? new Vector<T>(_poolingWeights.Length), batchSize);
        shotGradient = Scale(shotGradient ?? new Vector<T>(_shotWeights.Length), batchSize);

        AddDecay(bodyGradient, body, _relationOptions.FeatureEncoderL2Reg);
        AddDecay(relationGradient, _relationWeights, _relationOptions.RelationModuleL2Reg);

        if (_relationOptions.GradientClipThreshold.HasValue && _relationOptions.GradientClipThreshold.Value > 0)
        {
            double threshold = _relationOptions.GradientClipThreshold.Value;
            bodyGradient = ClipGradients(bodyGradient, threshold);
            if (relationGradient.Length > 0) relationGradient = ClipGradients(relationGradient, threshold);
            if (transformGradient.Length > 0) transformGradient = ClipGradients(transformGradient, threshold);
            if (poolingGradient.Length > 0) poolingGradient = ClipGradients(poolingGradient, threshold);
            if (shotGradient.Length > 0) shotGradient = ClipGradients(shotGradient, threshold);
        }

        double beta = _relationOptions.OuterLearningRate;
        ParamModel.SetParameters(ApplyGradients(body, bodyGradient, beta));
        if (_relationWeights.Length > 0) _relationWeights = ApplyGradients(_relationWeights, relationGradient, beta);
        if (_featureTransform.Length > 0) _featureTransform = ApplyGradients(_featureTransform, transformGradient, beta);
        if (_poolingWeights.Length > 0) _poolingWeights = ApplyGradients(_poolingWeights, poolingGradient, beta);
        if (_shotWeights.Length > 0) _shotWeights = ApplyGradients(_shotWeights, shotGradient, beta);

        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        EnsureRelationShapes(new[] { task });
        return new RelationNetworkModel<T, TInput, TOutput>(
            MetaModel, task.SupportInput, task.SupportOutput, _relationOptions, HeadCount,
            CloneVector(_relationWeights), CloneVector(_featureTransform), CloneVector(_poolingWeights), CloneVector(_shotWeights));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns relation scores per class, so this is the configured loss - mean squared error by
    /// default, eq. 2 - of the scores against the match indicators. A Vector output carries one predicted class per
    /// example instead, and its loss is the classification error rate.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => RelationScorer<T>.OutputLoss(LossFunction, predictions, expectedOutput, _relationOptions.NumClasses);

    #region Episode

    private int HeadCount => _relationOptions.UseMultiHeadRelation ? _relationOptions.NumHeads : 1;

    /// <summary>
    /// One episode's relation loss and its exact gradient with respect to the embedding module and every learned part
    /// of the comparison. While <paramref name="training"/>, dropout masks are drawn once and shared by both passes.
    /// </summary>
    private (T Loss, Vector<T> Body, Vector<T> Relation, Vector<T> Transform, Vector<T> Pooling, Vector<T> Shots) EpisodeGradient(
        IMetaLearningTask<T, TInput, TOutput> task, bool training)
    {
        var supportLabels = ReadLabels(task.SupportOutput);
        var episode = PrototypeEpisode<T>.Build(supportLabels, ReadLabels(task.QueryOutput));
        var slots = RelationScorer<T>.ShotSlots(supportLabels);
        var target = RelationScorer<T>.OneHot(episode.QueryTarget, episode.ClassSlots.Length);
        var stackedInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));
        var scorer = CreateScorer(training);

        // The embedding module's gradient: every relation score rebuilt from the embeddings on the tape. The
        // comparison's weights are constants here.
        var composed = new EmbeddingClassificationLoss<T>(
            embeddings => EpisodeScores(embeddings, episode, slots, scorer), LossFunction, target);
        var bodyGradient = ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);

        // The comparison's gradient: the same scores from fixed embeddings, against its own weights.
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = ClassifierOutputs<T>.AsRows(MetaModel.Predict(stackedInput));
        }

        var relationGradient = new Vector<T>(_relationWeights.Length);
        var transformGradient = new Vector<T>(_featureTransform.Length);
        var poolingGradient = new Vector<T>(_poolingWeights.Length);
        var shotGradient = new Vector<T>(_shotWeights.Length);
        T loss;
        if (scorer.Leaves.Count == 0)
        {
            using var noGrad = new NoGradScope<T>();
            loss = LossFunction.ComputeTapeLoss(EpisodeScores(embeddings, episode, slots, scorer), target)[0];
        }
        else
        {
            using var tape = new GradientTape<T>();
            var episodeLoss = LossFunction.ComputeTapeLoss(EpisodeScores(embeddings, episode, slots, scorer), target);
            loss = episodeLoss[0];
            var gradients = tape.ComputeGradients(episodeLoss, scorer.Leaves.ToList());
            scorer.CopyGradients(gradients, relationGradient, transformGradient, poolingGradient, shotGradient);
        }

        return (loss, bodyGradient, relationGradient, transformGradient, poolingGradient, shotGradient);
    }

    /// <summary>Relation scores <c>[queries, classes]</c> from the stacked support-then-query embeddings.</summary>
    private static Tensor<T> EpisodeScores(Tensor<T> embeddings, PrototypeEpisode<T> episode, int[] slots, RelationScorer<T> scorer)
    {
        var engine = AiDotNetEngine.Current;
        var support = engine.TensorMatMul(episode.SupportSelector, embeddings);
        var query = engine.TensorMatMul(episode.QuerySelector, embeddings);
        return scorer.Scores(support, query, episode.Membership, slots);
    }

    private RelationScorer<T> CreateScorer(bool training)
        => new RelationScorer<T>(
            _relationOptions.RelationType, _relationOptions.RelationHiddenDimension, _relationOptions.AggregationMethod,
            HeadCount, _relationWeights, _featureTransform, _poolingWeights, _shotWeights, _embeddingWidth,
            training ? RandomGenerator : null, _relationOptions.RelationDropout);

    /// <summary>
    /// Sizes the learned parts to the embedding width: the relation heads at PyTorch's default linear
    /// initialisation, the feature map at the identity, the pooling attention and shot weights at zero - uniform
    /// pooling, the untrained value of both.
    /// </summary>
    private void EnsureRelationShapes(IEnumerable<IMetaLearningTask<T, TInput, TOutput>> tasks)
    {
        var list = tasks.ToList();
        if (list.Count == 0) return;

        if (_relationWeights.Length == 0)
        {
            int width;
            using (new NoGradScope<T>())
            {
                width = ClassifierOutputs<T>.AsRows(MetaModel.Predict(list[0].SupportInput)).Shape[1];
            }

            _embeddingWidth = width;
            int perHead = RelationFunction<T>.ParameterCount(_relationOptions.RelationType, width, _relationOptions.RelationHiddenDimension);
            _relationWeights = new Vector<T>(HeadCount * perHead);
            for (int head = 0; head < HeadCount; head++)
            {
                RelationFunction<T>.Initialize(_relationOptions.RelationType, _relationWeights, head * perHead, width,
                    _relationOptions.RelationHiddenDimension, RandomGenerator);
            }
        }

        int d = _embeddingWidth;
        if (_relationOptions.ApplyFeatureTransform && _featureTransform.Length == 0)
        {
            _featureTransform = new Vector<T>(d * d);
            for (int i = 0; i < d; i++) _featureTransform[i * d + i] = NumOps.One;
        }

        if (_relationOptions.AggregationMethod == RelationAggregationMethod.Attention && _poolingWeights.Length == 0)
        {
            _poolingWeights = new Vector<T>(d * d);
        }

        if (_relationOptions.AggregationMethod == RelationAggregationMethod.LearnedWeighting)
        {
            int shots = list.Max(t => RelationScorer<T>.ShotSlots(ReadLabels(t.SupportOutput)).DefaultIfEmpty(-1).Max()) + 1;
            if (shots > _shotWeights.Length)
            {
                var grown = new Vector<T>(shots);
                for (int i = 0; i < _shotWeights.Length; i++) grown[i] = _shotWeights[i];
                _shotWeights = grown;
            }
        }
    }

    private int[] ReadLabels(TOutput labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, _relationOptions.NumClasses);
        var indices = new int[tensor.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
        return indices;
    }

    #endregion

    #region Test hooks

    /// <summary>One episode's relation loss and exact gradient from the current state, without dropout.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> Relation, Vector<T> Transform, Vector<T> Pooling, Vector<T> Shots) EpisodeGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureRelationShapes(new[] { task });
        return EpisodeGradient(task, training: false);
    }

    /// <summary>One episode's relation loss from the current state, without dropout.</summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        EnsureRelationShapes(new[] { task });
        var supportLabels = ReadLabels(task.SupportOutput);
        var episode = PrototypeEpisode<T>.Build(supportLabels, ReadLabels(task.QueryOutput));
        using var noGrad = new NoGradScope<T>();
        var embeddings = ClassifierOutputs<T>.AsRows(
            MetaModel.Predict(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput)));
        var scores = EpisodeScores(embeddings, episode, RelationScorer<T>.ShotSlots(supportLabels), CreateScorer(training: false));
        return LossFunction.ComputeTapeLoss(scores, RelationScorer<T>.OneHot(episode.QueryTarget, episode.ClassSlots.Length))[0];
    }

    /// <summary>Gets or sets a copy of the relation heads' weights (for tests).</summary>
    internal Vector<T> RelationWeightsForTesting { get => CloneVector(_relationWeights); set => _relationWeights = CloneVector(value); }

    /// <summary>Gets or sets a copy of the feature map (for tests).</summary>
    internal Vector<T> FeatureTransformForTesting { get => CloneVector(_featureTransform); set => _featureTransform = CloneVector(value); }

    /// <summary>Gets or sets a copy of the pooling attention (for tests).</summary>
    internal Vector<T> PoolingWeightsForTesting { get => CloneVector(_poolingWeights); set => _poolingWeights = CloneVector(value); }

    /// <summary>Gets or sets a copy of the per-shot weights (for tests).</summary>
    internal Vector<T> ShotWeightsForTesting { get => CloneVector(_shotWeights); set => _shotWeights = CloneVector(value); }

    #endregion

    #region Helpers

    private static void AddDecay(Vector<T> gradient, Vector<T> parameters, double strength)
    {
        if (strength <= 0) return;
        T s = NumOps.FromDouble(strength);
        for (int i = 0; i < gradient.Length && i < parameters.Length; i++)
        {
            gradient[i] = NumOps.Add(gradient[i], NumOps.Multiply(s, parameters[i]));
        }
    }

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
/// The comparison of a Relation Network in engine tensor ops: the feature map, how each class meets each query, the
/// relation heads and their dropout - relation scores <c>[queries, classes]</c> a live tape differentiates.
/// </summary>
internal sealed class RelationScorer<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly RelationAggregationMethod _pooling;
    private readonly RelationFunction<T>[] _heads;
    private readonly int _perHead;
    private readonly int _width;
    private readonly Tensor<T>? _transform;
    private readonly Tensor<T>? _attention;
    private readonly Tensor<T>? _shots;
    private readonly Random? _dropoutRandom;
    private readonly double _dropout;
    private readonly Dictionary<(int Head, int Rows), Tensor<T>> _masks = new Dictionary<(int Head, int Rows), Tensor<T>>();
    private readonly List<Tensor<T>> _leaves = new List<Tensor<T>>();

    /// <summary>Unpacks the learned parts; an empty vector leaves that part at its untrained value.</summary>
    internal RelationScorer(
        RelationModuleType type, int hidden, RelationAggregationMethod pooling, int heads,
        Vector<T> relation, Vector<T> transform, Vector<T> poolingWeights, Vector<T> shots, int width,
        Random? dropoutRandom, double dropout)
    {
        _pooling = pooling;
        _width = width;
        _dropoutRandom = dropoutRandom;
        _dropout = dropout;
        _perHead = RelationFunction<T>.ParameterCount(type, width, hidden);
        _heads = new RelationFunction<T>[heads];
        for (int h = 0; h < heads; h++)
        {
            _heads[h] = new RelationFunction<T>(type, relation, h * _perHead, width, hidden);
            _leaves.AddRange(_heads[h].Leaves);
        }

        if (transform.Length > 0)
        {
            _transform = Tensor<T>.FromVector(transform);
            _leaves.Add(_transform);
        }

        if (pooling == RelationAggregationMethod.Attention && poolingWeights.Length > 0)
        {
            _attention = Tensor<T>.FromVector(poolingWeights);
            _leaves.Add(_attention);
        }

        if (pooling == RelationAggregationMethod.LearnedWeighting && shots.Length > 0)
        {
            _shots = Tensor<T>.FromVector(shots);
            _leaves.Add(_shots);
        }
    }

    /// <summary>Every learned tensor.</summary>
    internal IReadOnlyList<Tensor<T>> Leaves => _leaves;

    /// <summary>
    /// Relation scores <c>[queries, classes]</c> for support embeddings <c>[support, width]</c> and query embeddings
    /// <c>[queries, width]</c>, with <paramref name="membership"/> <c>[classes, support]</c> and each support row's
    /// index within its class.
    /// </summary>
    internal Tensor<T> Scores(Tensor<T> support, Tensor<T> query, Tensor<T> membership, int[] shotSlots)
    {
        var engine = AiDotNetEngine.Current;
        support = Transform(support);
        query = Transform(query);
        int queries = query.Shape[0], supportRows = support.Shape[0], classes = membership.Shape[0];

        if (_pooling == RelationAggregationMethod.EmbeddingSum)
        {
            // Sung et al. 2018: "we element-wise sum over the embedding module outputs of all samples from each
            // training class to form this class' feature map", then one relation per class.
            return engine.Reshape(Relate(engine.TensorMatMul(membership, support), query), new[] { queries, classes });
        }

        var perShot = engine.Reshape(Relate(support, query), new[] { queries, supportRows });
        switch (_pooling)
        {
            case RelationAggregationMethod.Max:
            {
                Tensor<T>? pooled = null;
                for (int c = 0; c < classes; c++)
                {
                    var members = Enumerable.Range(0, supportRows).Where(s => Ops.ToDouble(membership[c * supportRows + s]) > 0.5).ToArray();
                    var select = new Tensor<T>(new[] { supportRows, members.Length });
                    for (int m = 0; m < members.Length; m++) select[members[m] * members.Length + m] = Ops.One;
                    var best = engine.ReduceMax(engine.TensorMatMul(perShot, select), new[] { 1 }, keepDims: true, out _);
                    var placed = engine.TensorMatMul(best, Unit(classes, c));
                    pooled = pooled is null ? placed : engine.TensorAdd(pooled, placed);
                }

                return pooled ?? new Tensor<T>(new[] { queries, classes });
            }
            case RelationAggregationMethod.Attention:
            {
                // Weights softmax_s(x_s' U q) within each class; U = 0 is the mean.
                Tensor<T> weights;
                if (_attention is null)
                {
                    weights = Ones(queries, supportRows);
                }
                else
                {
                    var u = engine.Reshape(_attention, new[] { _width, _width });
                    var logits = engine.TensorMatMul(engine.TensorMatMul(query, engine.TensorTranspose(u)), engine.TensorTranspose(support));
                    var max = engine.ReduceMax(logits, new[] { 1 }, keepDims: true, out _);
                    weights = engine.TensorExp(engine.TensorAdd(logits, engine.TensorNegate(engine.StopGradient(max))));
                }

                return WeightedPool(perShot, weights, membership);
            }
            case RelationAggregationMethod.LearnedWeighting:
            {
                // Weight exp(w_k) for the k-th shot of every class; w = 0 is the mean.
                Tensor<T> weights;
                if (_shots is null)
                {
                    weights = Ones(1, supportRows);
                }
                else
                {
                    var slotSelect = new Tensor<T>(new[] { _shots.Length, supportRows });
                    for (int s = 0; s < supportRows; s++)
                    {
                        if (shotSlots[s] < _shots.Length) slotSelect[shotSlots[s] * supportRows + s] = Ops.One;
                    }

                    weights = engine.TensorExp(engine.TensorMatMul(engine.Reshape(_shots, new[] { 1, _shots.Length }), slotSelect));
                }

                return WeightedPool(perShot, weights, membership);
            }
            default:
            {
                var mean = new Tensor<T>(new[] { classes, supportRows });
                for (int c = 0; c < classes; c++)
                {
                    double count = Enumerable.Range(0, supportRows).Sum(s => Ops.ToDouble(membership[c * supportRows + s]));
                    for (int s = 0; s < supportRows; s++)
                    {
                        mean[c * supportRows + s] = Ops.FromDouble(Ops.ToDouble(membership[c * supportRows + s]) / Math.Max(1.0, count));
                    }
                }

                return engine.TensorMatMul(perShot, engine.TensorTranspose(mean));
            }
        }
    }

    /// <summary>Scatters a tape's gradients back into the four flat vectors.</summary>
    internal void CopyGradients(
        Dictionary<Tensor<T>, Tensor<T>> gradients, Vector<T> relation, Vector<T> transform, Vector<T> pooling, Vector<T> shots)
    {
        for (int h = 0; h < _heads.Length; h++) _heads[h].CopyGradients(gradients, relation, h * _perHead);
        Copy(gradients, _transform, transform);
        Copy(gradients, _attention, pooling);
        Copy(gradients, _shots, shots);
    }

    /// <summary>Each support row's index among the rows of its class, in order of appearance.</summary>
    internal static int[] ShotSlots(int[] supportLabels)
    {
        var seen = new Dictionary<int, int>();
        var slots = new int[supportLabels.Length];
        for (int i = 0; i < supportLabels.Length; i++)
        {
            seen.TryGetValue(supportLabels[i], out int count);
            slots[i] = count;
            seen[supportLabels[i]] = count + 1;
        }

        return slots;
    }

    /// <summary>The match indicators <c>1(y == c)</c>, <c>[rows, classes]</c>, for class columns.</summary>
    internal static Tensor<T> OneHot(Tensor<T> columns, int classes)
    {
        var target = new Tensor<T>(new[] { columns.Length, classes });
        for (int i = 0; i < columns.Length; i++)
        {
            int c = (int)Math.Round(Ops.ToDouble(columns[i]));
            if (c >= 0 && c < classes) target[i * classes + c] = Ops.One;
        }

        return target;
    }

    /// <summary>
    /// The loss of adapted predictions: the configured loss of relation scores <c>[rows, classes]</c> against the
    /// match indicators, or the error rate of a Vector of predicted classes.
    /// </summary>
    internal static T OutputLoss(ILossFunction<T> loss, object? predictions, object? expected, int numClasses)
    {
        var labels = ClassifierOutputs<T>.Labels(expected, numClasses);
        if (predictions is Vector<T> predictedClasses)
        {
            int wrong = 0;
            for (int i = 0; i < labels.Length; i++)
            {
                if (i >= predictedClasses.Length || Math.Abs(Ops.ToDouble(predictedClasses[i]) - Ops.ToDouble(labels[i])) > 0.5)
                    wrong++;
            }

            return Ops.FromDouble(labels.Length == 0 ? 0 : (double)wrong / labels.Length);
        }

        var scores = ClassifierOutputs<T>.ScoreRows(predictions, labels.Length);
        using var noGrad = new NoGradScope<T>();
        return loss.ComputeTapeLoss(scores, OneHot(labels, scores.Shape[1]))[0];
    }

    /// <summary>Relation scores <c>[queries * rows, 1]</c> of every (sample row, query) pair; row <c>q * rows + i</c>.</summary>
    private Tensor<T> Relate(Tensor<T> samples, Tensor<T> queries)
    {
        var engine = AiDotNetEngine.Current;
        int rows = samples.Shape[0], q = queries.Shape[0], pairs = rows * q;
        var pickSample = new Tensor<T>(new[] { pairs, rows });
        var pickQuery = new Tensor<T>(new[] { pairs, q });
        for (int qi = 0; qi < q; qi++)
        {
            for (int i = 0; i < rows; i++)
            {
                pickSample[(qi * rows + i) * rows + i] = Ops.One;
                pickQuery[(qi * rows + i) * q + qi] = Ops.One;
            }
        }

        var pairedSamples = engine.TensorMatMul(pickSample, samples);
        var pairedQueries = engine.TensorMatMul(pickQuery, queries);
        Tensor<T>? sum = null;
        for (int h = 0; h < _heads.Length; h++)
        {
            var scores = _heads[h].Scores(pairedSamples, pairedQueries, Mask(h, pairs));
            sum = sum is null ? scores : engine.TensorAdd(sum, scores);
        }

        var total = sum ?? new Tensor<T>(new[] { pairs, 1 });
        return _heads.Length > 1 ? engine.TensorMultiplyScalar(total, Ops.FromDouble(1.0 / _heads.Length)) : total;
    }

    /// <summary>A head's inverted-dropout mask for this many pairs, drawn once and reused for both passes.</summary>
    private Tensor<T>? Mask(int head, int rows)
    {
        if (_dropoutRandom is null || _dropout <= 0) return null;
        if (_masks.TryGetValue((head, rows), out var cached)) return cached;
        int hidden = _heads[head].Hidden;
        var mask = new Tensor<T>(new[] { rows, hidden });
        T keep = Ops.FromDouble(1.0 / (1.0 - _dropout));
        for (int i = 0; i < mask.Length; i++) mask[i] = _dropoutRandom.NextDouble() >= _dropout ? keep : Ops.Zero;
        _masks[(head, rows)] = mask;
        return mask;
    }

    private Tensor<T> Transform(Tensor<T> rows)
    {
        if (_transform is null) return rows;
        var engine = AiDotNetEngine.Current;
        return engine.TensorMatMul(rows, engine.TensorTranspose(engine.Reshape(_transform, new[] { _width, _width })));
    }

    /// <summary>Per class, the weighted mean of its shots' scores: <c>(w * r) M' / (w M')</c>.</summary>
    private static Tensor<T> WeightedPool(Tensor<T> perShot, Tensor<T> weights, Tensor<T> membership)
    {
        var engine = AiDotNetEngine.Current;
        var classesOf = engine.TensorTranspose(membership);
        return engine.TensorDivide(
            engine.TensorMatMul(engine.TensorMultiply(perShot, weights), classesOf),
            engine.TensorMatMul(weights, classesOf));
    }

    private static void Copy(Dictionary<Tensor<T>, Tensor<T>> gradients, Tensor<T>? leaf, Vector<T> into)
    {
        if (leaf is null || !gradients.TryGetValue(leaf, out var gradient)) return;
        for (int i = 0; i < into.Length; i++) into[i] = gradient[i];
    }

    private static Tensor<T> Ones(int rows, int columns)
    {
        var ones = new Tensor<T>(new[] { rows, columns });
        for (int i = 0; i < ones.Length; i++) ones[i] = Ops.One;
        return ones;
    }

    private static Tensor<T> Unit(int columns, int column)
    {
        var unit = new Tensor<T>(new[] { 1, columns });
        unit[column] = Ops.One;
        return unit;
    }
}
