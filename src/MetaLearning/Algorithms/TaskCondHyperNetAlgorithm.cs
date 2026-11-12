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
/// Implementation of Task-Conditioned HyperNetwork for meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// A hypernetwork generates task-specific parameter deltas conditioned on a task embedding
/// derived from support-set gradient statistics. The chunked hypernetwork architecture
/// processes the task embedding through a shared hidden layer, then uses per-chunk output
/// heads to produce parameter deltas for each chunk of the target network.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Task embedding: e = compress(grad_support)  →  [EmbeddingDim]
///
/// Hypernetwork (chunked architecture):
///   hidden = tanh(W_h · e + b_h)               →  [HyperHiddenDim]
///   For each chunk c:
///     Δθ_c = W_c · hidden                       →  [ChunkSize]
///
/// Adapted params: θ_adapted = θ_init + Δθ
/// Fine-tune: K standard MAML steps on top
///
/// L_meta = L_query(θ_adapted)
/// Outer: θ_init and the hypernetwork weights W both descend the first-order meta-gradient
///        g = ∇L_query(θ_adapted); W gets it by backpropagation through Δθ = 0.01 · W_c · hidden.
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("HyperNetworks",
    "https://arxiv.org/abs/1609.09106",
    Year = 2016,
    Authors = "Ha, D., Dai, A., & Le, Q. V.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class TaskCondHyperNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly TaskCondHyperNetOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numChunks;

    /// <summary>Hypernetwork weights: W_h (embDim × hiddenDim) + b_h (hiddenDim) + per-chunk W_c (hiddenDim × chunkSize).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _hyperNetWeights;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.TaskCondHyperNet;

    public TaskCondHyperNetAlgorithm(TaskCondHyperNetOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.EmbeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "EmbeddingDim must be positive.");
        if (options.HyperHiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "HyperHiddenDim must be positive.");
        if (options.ChunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ChunkSize must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _numChunks = (_paramDim + options.ChunkSize - 1) / options.ChunkSize;

        int embDim = options.EmbeddingDim;
        int hidDim = options.HyperHiddenDim;
        int chunkSize = options.ChunkSize;

        // W_h: embDim * hidDim + b_h: hidDim + per-chunk W_c: numChunks * hidDim * chunkSize
        int totalWeights = embDim * hidDim + hidDim + _numChunks * hidDim * chunkSize;
        // U(-0.5, 0.5) / sqrt(hidDim). The draws stay one NextDouble per weight, in order, so seeded runs
        // reproduce; the loop only stores them and the arithmetic is one engine op.
        var draws = new Vector<T>(totalWeights);
        for (int i = 0; i < totalWeights; i++)
            draws[i] = NumOps.FromDouble(RandomGenerator.NextDouble());
        _hyperNetWeights = Engine.Multiply(
            Engine.Subtract(draws, Engine.Fill(totalWeights, NumOps.FromDouble(0.5))),
            NumOps.FromDouble(1.0 / Math.Sqrt(hidDim)));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var hyperGradient = new Vector<T>(_hyperNetWeights.Length);
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var (adaptedParams, embedding, hidden) = AdaptFrom(initParams, task);

            ParamModel.SetParameters(adaptedParams);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            var queryGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));
            metaGradients.Add(queryGrad);
            hyperGradient = Engine.Add(hyperGradient, HyperNetworkGradient(embedding, hidden, queryGrad));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        // The hypernetwork descends the same first-order meta-gradient, backpropagated through delta.
        _hyperNetWeights = Engine.Subtract(_hyperNetWeights, Engine.Multiply(hyperGradient,
            NumOps.FromDouble(_algoOptions.OuterLearningRate * 0.1 / taskBatch.Tasks.Length)));

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var (adaptedParams, _, _) = AdaptFrom(initParams, task);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>Embeds the task, applies the generated delta, then fine-tunes with K MAML steps.</summary>
    private (Vector<T> Adapted, Vector<T> Embedding, Tensor<T> Hidden) AdaptFrom(
        Vector<T> initParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        ParamModel.SetParameters(initParams);
        var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
        var embedding = CompressVector(supportGrad, _algoOptions.EmbeddingDim);
        var (delta, hidden) = RunHyperNetwork(embedding);

        var adaptedParams = Engine.Add(initParams, delta);
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        return (adaptedParams, embedding, hidden);
    }

    /// <summary>Output scale of the chunk heads: delta = DeltaScale * W_c * hidden.</summary>
    private const double DeltaScale = 0.01;

    /// <summary>
    /// The hypernetwork: hidden = tanh(W_h e + b_h), then every chunk head W_c maps hidden to its slice of delta.
    /// </summary>
    /// <remarks>
    /// Weight layout: W_h is [HyperHiddenDim, EmbeddingDim] row-major, then b_h, then the chunk heads stacked
    /// as one [numChunks * ChunkSize, HyperHiddenDim] row-major block; rows past the parameter count are unused.
    /// </remarks>
    private (Vector<T> Delta, Tensor<T> Hidden) RunHyperNetwork(Vector<T> embedding)
    {
        var (wh, bh, wc) = SplitWeights();
        var pre = Engine.TensorAdd(
            Engine.TensorMatMul(wh, Tensor<T>.FromVector(embedding, new[] { _algoOptions.EmbeddingDim, 1 })), bh);
        var hidden = Engine.TensorTanh(pre);
        var full = Engine.TensorMultiplyScalar(Engine.TensorMatMul(wc, hidden), NumOps.FromDouble(DeltaScale));
        return (full.ToVector().GetSubVector(0, _paramDim), hidden);
    }

    /// <summary>
    /// d(g · delta)/dW for the hypernetwork, i.e. the chain rule of <see cref="RunHyperNetwork"/> applied to
    /// <paramref name="deltaGradient"/> = dL/d delta, laid out like the weight vector.
    /// </summary>
    private Vector<T> HyperNetworkGradient(Vector<T> embedding, Tensor<T> hidden, Vector<T> deltaGradient)
    {
        int hidDim = _algoOptions.HyperHiddenDim;
        int rows = _numChunks * _algoOptions.ChunkSize;
        var (_, _, wc) = SplitWeights();

        var padded = rows == _paramDim
            ? deltaGradient
            : Vector<T>.Concatenate(new[] { deltaGradient, new Vector<T>(rows - _paramDim) });
        var gs = Engine.TensorMultiplyScalar(Tensor<T>.FromVector(padded, new[] { rows, 1 }), NumOps.FromDouble(DeltaScale));

        var dWc = Engine.TensorMatMul(gs, Engine.TensorTranspose(hidden));
        var dHidden = Engine.TensorMatMul(Engine.TensorTranspose(wc), gs);
        var tanhSlope = Engine.TensorSubtract(
            Tensor<T>.CreateDefault(new[] { hidDim, 1 }, NumOps.One), Engine.TensorMultiply(hidden, hidden));
        var dPre = Engine.TensorMultiply(dHidden, tanhSlope);
        var dWh = Engine.TensorMatMul(dPre, Tensor<T>.FromVector(embedding, new[] { 1, _algoOptions.EmbeddingDim }));

        return Vector<T>.Concatenate(new[] { dWh.ToVector(), dPre.ToVector(), dWc.ToVector() });
    }

    /// <summary>Views the flat weight vector as W_h [hid, emb], b_h [hid, 1] and the stacked heads W_c [rows, hid].</summary>
    private (Tensor<T> Wh, Tensor<T> Bh, Tensor<T> Wc) SplitWeights()
    {
        int embDim = _algoOptions.EmbeddingDim;
        int hidDim = _algoOptions.HyperHiddenDim;
        int rows = _numChunks * _algoOptions.ChunkSize;
        int whCount = hidDim * embDim;
        return (
            Tensor<T>.FromVector(_hyperNetWeights.GetSubVector(0, whCount), new[] { hidDim, embDim }),
            Tensor<T>.FromVector(_hyperNetWeights.GetSubVector(whCount, hidDim), new[] { hidDim, 1 }),
            Tensor<T>.FromVector(_hyperNetWeights.GetSubVector(whCount + hidDim, rows * hidDim), new[] { rows, hidDim }));
    }

    /// <summary>Test hook: the hypernetwork's delta and its weight gradient for a given dL/d delta.</summary>
    internal (Vector<T> Delta, Vector<T> WeightGradient) HyperNetworkProbe(Vector<T> embedding, Vector<T> deltaGradient)
    {
        var (delta, hidden) = RunHyperNetwork(embedding);
        return (delta, HyperNetworkGradient(embedding, hidden, deltaGradient));
    }

    /// <summary>Test hook: the flat hypernetwork weights.</summary>
    internal Vector<T> HyperNetWeights
    {
        get => _hyperNetWeights;
        set => _hyperNetWeights = value;
    }
}
