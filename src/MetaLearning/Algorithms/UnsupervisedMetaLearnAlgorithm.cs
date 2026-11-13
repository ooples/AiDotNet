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
/// Implementation of CACTUs-MAML, "Unsupervised Learning via Meta-Learning" (Hsu, Levine and Finn, ICLR 2019).
/// </summary>
/// <remarks>
/// <para>
/// CACTUs (Clustering to Automatically Construct Tasks for Unsupervised meta-learning) never reads the labels
/// of the meta-training data. It clusters embeddings of the unlabeled examples with k-means, treats clusters as
/// classes, and builds N-way K-shot pseudo-tasks from them; MAML then meta-trains on the pseudo-tasks. Labels
/// are only used at meta-test time, by <see cref="Adapt"/>.
/// </para>
/// <para><b>Algorithm (paper Algorithm 1, per meta-training batch):</b>
/// <code>
/// pool  = all inputs of the batch's tasks, labels discarded;  Z = embed(pool)
/// for each task slot:
///   partition = k-means(Z * a, k), a_d ~ U(0, 1) per dimension    (a randomly scaled partition P_p, Eq. 1)
///   pick N clusters with at least K + Q members; draw K + Q members of each
///   pseudo-task: support = K per cluster, query = Q per cluster, label = cluster rank 0..N-1
///   theta' = MAML inner loop on the pseudo support; meta-gradient from the pseudo query
/// theta &lt;- theta - beta * mean meta-gradient
/// </code>
/// </para>
/// <para>
/// N, K and Q are each slot task's NumWays, NumShots and NumQueryPerClass, so pseudo-tasks have the episode
/// shape the data loader produces. The embedding is the flattened input (the paper also uses pre-trained
/// unsupervised embeddings; pass embedded inputs to use one). Inputs must be <see cref="Tensor{T}"/> or
/// <see cref="Matrix{T}"/> with examples on the first axis; targets may be class indices
/// (<see cref="Vector{T}"/>, rank-1 tensor) or one-hot rows (<see cref="Matrix{T}"/>, rank-2 tensor).
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unsupervised Learning via Meta-Learning",
    "https://arxiv.org/abs/1810.02334",
    Year = 2019,
    Authors = "Hsu, K., Levine, S., & Finn, C.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class UnsupervisedMetaLearnAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly UnsupervisedMetaLearnOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.UnsupervisedMetaLearn;

    public UnsupervisedMetaLearnAlgorithm(UnsupervisedMetaLearnOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.NumClusters < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumClusters must be non-negative (0 = automatic).");
        if (options.KMeansIterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "KMeansIterations must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        // The unlabeled pool: every input of the batch, labels discarded.
        var parts = new List<TInput>();
        foreach (var task in taskBatch.Tasks)
        {
            parts.Add(task.SupportInput);
            parts.Add(task.QueryInput);
        }
        var pool = ConcatInputs(parts);
        var embeddings = Embed(pool);

        foreach (var task in taskBatch.Tasks)
        {
            var pseudo = BuildPseudoTask(pool, embeddings, task);
            if (pseudo is null) continue;

            var adaptedParams = AdaptFrom(initParams, pseudo.Value.SupportX, pseudo.Value.SupportY);
            ParamModel.SetParameters(adaptedParams);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(pseudo.Value.QueryX), pseudo.Value.QueryY));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, pseudo.Value.QueryX, pseudo.Value.QueryY)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);
        return ComputeMean(losses);
    }
    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var adaptedParams = AdaptFrom(initParams, task.SupportInput, task.SupportOutput);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>The MAML inner loop on (<paramref name="x"/>, <paramref name="y"/>).</summary>
    private Vector<T> AdaptFrom(Vector<T> initParams, TInput x, TOutput y)
    {
        var adaptedParams = initParams.Clone();
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, x, y));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        return adaptedParams;
    }

    /// <summary>
    /// One CACTUs pseudo-task shaped like <paramref name="slot"/>: a fresh k-means partition of the pool, N clusters
    /// with at least K + Q members, K support and Q query examples per cluster. Null when fewer than two
    /// clusters are large enough.
    /// </summary>
    internal (TInput SupportX, TOutput SupportY, TInput QueryX, TOutput QueryY)? BuildPseudoTask(
        TInput pool, Tensor<T> embeddings, IMetaLearningTask<T, TInput, TOutput> slot)
    {
        int n = embeddings.Shape[0];
        int shots = Math.Max(1, slot.NumShots);
        int queries = Math.Max(1, slot.NumQueryPerClass);
        int perCluster = shots + queries;
        int k = _algoOptions.NumClusters > 0 ? _algoOptions.NumClusters : Math.Max(2, n / perCluster);
        k = Math.Min(k, n);

        var assignment = KMeans(embeddings, k, DrawDimensionScale(embeddings.Shape[1]));
        var members = new List<int>[k];
        for (int c = 0; c < k; c++) members[c] = new List<int>();
        for (int i = 0; i < n; i++) members[assignment[i]].Add(i);

        var eligible = new List<int>();
        for (int c = 0; c < k; c++) if (members[c].Count >= perCluster) eligible.Add(c);
        int ways = Math.Min(Math.Max(2, slot.NumWays), eligible.Count);
        if (ways < 2) return null;

        Shuffle(eligible);
        var supportIdx = new int[ways * shots];
        var queryIdx = new int[ways * queries];
        var supportLabels = new int[ways * shots];
        var queryLabels = new int[ways * queries];
        for (int w = 0; w < ways; w++)
        {
            var chosen = members[eligible[w]];
            Shuffle(chosen);
            for (int s = 0; s < shots; s++)
            {
                supportIdx[w * shots + s] = chosen[s];
                supportLabels[w * shots + s] = w;
            }
            for (int q = 0; q < queries; q++)
            {
                queryIdx[w * queries + q] = chosen[shots + q];
                queryLabels[w * queries + q] = w;
            }
        }

        return (InputHelper<T, TInput>.GetBatch(pool, supportIdx), Labels(supportLabels, slot.SupportOutput, ways),
                InputHelper<T, TInput>.GetBatch(pool, queryIdx), Labels(queryLabels, slot.QueryOutput, ways));
    }

    /// <summary>
    /// The random scaling of one CACTUs partition (Sec. 2.3, Eq. 1): every embedding dimension is multiplied by its
    /// own a_d ~ U(0, 1), so each k-means run sees a different diagonal metric and the partitions differ (the
    /// authors' code: np.multiply(encodings, np.random.uniform(0, 1, dim))). One NextDouble per dimension, in order.
    /// </summary>
    internal Tensor<T> DrawDimensionScale(int dim)
    {
        var weights = new double[dim];
        for (int d = 0; d < dim; d++) weights[d] = RandomGenerator.NextDouble();
        return new Tensor<double>(weights, new[] { 1, dim }).Cast<T>();
    }

    /// <summary>
    /// Lloyd's k-means from k distinct random examples under the plain Euclidean metric; returns each example's
    /// cluster.
    /// </summary>
    internal int[] KMeans(Tensor<T> embeddings, int k) => KMeans(embeddings, k, null);

    /// <summary>
    /// Lloyd's k-means from k distinct random examples on the embeddings with every dimension multiplied by
    /// <paramref name="dimensionScale"/> ([1, dim]) when one is given; returns each example's cluster. Distances drop
    /// the per-example ||x||^2, which does not change the arg-min.
    /// </summary>
    /// <remarks>
    /// The embeddings are first scaled by 1 / (max|x| sqrt(dim)), so every ||x|| &lt;= 1: a positive scale changes
    /// neither the arg-min nor the clustering, and it keeps ||c||^2 and x . c inside a narrow T (300^2 already
    /// overflows Half). Cluster sizes are counted as ints and each mean is the one-hot rows weighted by 1 / size,
    /// so neither a Half count (which stalls at 2048) nor an unnormalized column sum can saturate.
    /// </remarks>
    internal int[] KMeans(Tensor<T> embeddings, int k, Tensor<T>? dimensionScale)
    {
        int n = embeddings.Shape[0];
        int dim = embeddings.Shape[1];
        if (dimensionScale is not null)
            embeddings = Engine.TensorMultiply(embeddings, Engine.TensorBroadcastTo(dimensionScale, new[] { n, dim }));
        T peak = Engine.TensorMaxValue(Engine.TensorAbs(embeddings));
        if (NumOps.GreaterThan(peak, NumOps.Zero))
            embeddings = Engine.TensorMultiplyScalar(
                Engine.TensorDivideScalar(embeddings, peak), NumOps.FromDouble(1.0 / Math.Sqrt(dim)));

        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int i = 0; i < k; i++)
        {
            int j = i + RandomGenerator.Next(n - i);
            (order[i], order[j]) = (order[j], order[i]);
        }

        var initial = new int[k];
        Array.Copy(order, initial, k);
        var centroids = Engine.TensorIndexSelect(embeddings, new Tensor<int>(initial, new[] { k }), 0);
        var assignment = Assign(embeddings, centroids);
        for (int iter = 0; iter < _algoOptions.KMeansIterations; iter++)
        {
            var counts = Engine.TensorBinCount(assignment, k).Cast<double>().Reshape(new[] { 1, k });
            var inverseCounts = Engine.TensorReciprocal(Engine.TensorMax(counts, 1.0)).Cast<T>();
            var weights = Engine.TensorMultiply(
                Engine.TensorOneHot<T>(assignment, k), Engine.TensorBroadcastTo(inverseCounts, new[] { n, k }));
            var means = Engine.TensorMatMul(Engine.TensorTranspose(weights), embeddings);
            var occupied = Engine.TensorBroadcastTo(
                Engine.TensorGreaterThan(counts, 0.0).Cast<T>().Reshape(new[] { k, 1 }), new[] { k, dim });
            centroids = Engine.TensorWhere(occupied, means, centroids);
            assignment = Assign(embeddings, centroids);
        }

        return assignment.ToArray();
    }

    /// <summary>arg-min over clusters of ||c||^2 - 2 x . c.</summary>
    private Tensor<int> Assign(Tensor<T> embeddings, Tensor<T> centroids)
    {
        int n = embeddings.Shape[0];
        int k = centroids.Shape[0];
        var centroidNorms = Engine.ReduceSum(Engine.TensorSquare(centroids), new[] { 1 }, keepDims: false)
            .Reshape(new[] { 1, k });
        var cross = Engine.TensorMatMul(embeddings, Engine.TensorTranspose(centroids));
        var distances = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(centroidNorms, new[] { n, k }),
            Engine.TensorMultiplyScalar(cross, NumOps.FromDouble(2.0)));
        return Engine.TensorArgMin(distances, 1);
    }

    /// <summary>In-place Fisher-Yates shuffle with the learner's generator.</summary>
    private void Shuffle(List<int> items)
    {
        for (int i = items.Count - 1; i > 0; i--)
        {
            int j = RandomGenerator.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }

    /// <summary>The flattened inputs as an [examples, features] tensor.</summary>
    private static Tensor<T> Embed(TInput pool)
    {
        var tensor = (object?)pool switch
        {
            Tensor<T> t => t,
            Matrix<T> m => Tensor<T>.FromMatrix(m),
            _ => throw new NotSupportedException(
                $"CACTUs clusters Tensor<T> or Matrix<T> inputs; got {typeof(TInput).Name}."),
        };
        int n = tensor.Shape[0];
        return tensor.Reshape(new[] { n, n == 0 ? 0 : tensor.Length / n });
    }

    /// <summary>Concatenates inputs along the example axis.</summary>
    private TInput ConcatInputs(List<TInput> parts)
    {
        var tensors = new Tensor<T>[parts.Count];
        bool matrices = false;
        for (int i = 0; i < parts.Count; i++)
        {
            switch ((object?)parts[i])
            {
                case Tensor<T> t: tensors[i] = t; break;
                case Matrix<T> m: tensors[i] = Tensor<T>.FromMatrix(m); matrices = true; break;
                default:
                    throw new NotSupportedException(
                        $"CACTUs clusters Tensor<T> or Matrix<T> inputs; got {typeof(TInput).Name}.");
            }
        }

        var joined = Engine.TensorConcatenate(tensors, 0);
        return matrices ? (TInput)(object)joined.ToMatrix() : (TInput)(object)joined;
    }

    /// <summary>Pseudo-labels in the format of <paramref name="template"/>: class indices or one-hot rows.</summary>
    private TOutput Labels(int[] ids, TOutput template, int ways)
    {
        var indices = new Tensor<int>(ids, new[] { ids.Length });
        object labels = (object?)template switch
        {
            Vector<T> => indices.Cast<T>().ToVector(),
            Tensor<T> t when t.Rank == 1 => indices.Cast<T>(),
            Tensor<T> t when t.Rank == 2 && t.Shape[1] >= ways => Engine.TensorOneHot<T>(indices, t.Shape[1]),
            Matrix<T> m when m.Columns >= ways => Engine.TensorOneHot<T>(indices, m.Columns).ToMatrix(),
            _ => throw new NotSupportedException(
                $"CACTUs writes class-index or one-hot targets with at least {ways} columns; got {typeof(TOutput).Name}."),
        };
        return (TOutput)labels;
    }
}
