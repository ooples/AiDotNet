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
/// Implementation of MetaMix (Chen et al., ICPR 2020): MAML whose outer loss is computed on virtual
/// examples mixed from pairs of query examples.
/// </summary>
/// <remarks>
/// <para><b>Algorithm (paper Algorithm 1, Eq. 2-7):</b>
/// <code>
/// For each task T_i:
///   θ'_i = θ - α ∇_θ L_S(f_θ)                       (inner loop on the support set, unchanged)
///   pick Z random pairs (m, n) from the query set Q_i; λ ~ Beta(α̌, α̌)
///   x̂_z = λ x_m + (1-λ) x_n,  ŷ_z = λ y_m + (1-λ) y_n
/// θ ← θ - β ∇_θ Σ_i L_Q̂(f_θ'_i)                     (the mixed set replaces the query set)
/// </code>
/// </para>
/// <para>
/// <see cref="MetaTaskOptions{T, TInput, TOutput}.MixSource"/> can instead draw the first example of each
/// pair from the support set. Inputs and targets must be <see cref="Tensor{T}"/>, <see cref="Matrix{T}"/>
/// or <see cref="Vector{T}"/> with examples along the first axis.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation corrected. arXiv 2003.04274 is "Describe NMR relaxation by effective diffusion equation",
// an unrelated physics paper; the correct id is 2009.13735. The subtitle was also wrong — the paper is
// "Interpolation-based Consistency Regularization", not "Task Interpolation" (that phrasing belongs to
// a different task-interpolation line of work) — and it is ICPR 2020, not 2021. Authors were right.
[ResearchPaper("MetaMix: Improved Meta-Learning with Interpolation-based Consistency Regularization",
    "https://arxiv.org/abs/2009.13735",
    Year = 2020,
    Authors = "Yangbin Chen, Yun Ma, Tom Ko, Jianping Wang, Qing Li")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaTaskAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly MetaTaskOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaTask;

    public MetaTaskAlgorithm(MetaTaskOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var mixedLosses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        try
        {
            foreach (var task in taskBatch.Tasks)
            {
                var adaptedParams = Adapted(initParams, task);

                // Eq. 4-5: Z virtual examples mixed from random pairs, one λ ~ Beta(α̌, α̌) per episode.
                var (first, second) = DrawPairs(task);
                double lambda = SampleBeta(_algoOptions.InterpolationAlpha, _algoOptions.InterpolationAlpha);
                var (mixedX, mixedY) = BuildMixedSet(task, first, second, lambda);

                // Eq. 6-7: the mixed set replaces the query set in the outer objective.
                ParamModel.SetParameters(adaptedParams);
                mixedLosses.Add(ComputeLossFromOutput(MetaModel.Predict(mixedX), mixedY));
                metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, mixedX, mixedY)));
            }
        }
        finally
        {
            // A failed episode (DrawPairs on an empty set, a throwing loss) must not leave task-adapted weights in;
            // on success ApplyOuterUpdate starts from initParams too.
            ParamModel.SetParameters(initParams);
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);
        return ComputeMean(mixedLosses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = Adapted(initParams, task);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>Eq. 3: the MAML inner loop on the support set, starting from a copy of <paramref name="initParams"/>.</summary>
    private Vector<T> Adapted(Vector<T> initParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var adaptedParams = initParams.Clone();
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        return adaptedParams;
    }

    /// <summary>
    /// The number of examples in <paramref name="input"/>: a vector holds one scalar example per element (the
    /// batch-size helper reads a vector as one example), a matrix or tensor one per row.
    /// </summary>
    private static int ExampleCount(TInput input)
        => input is Vector<T> vector ? vector.Length : InputHelper<T, TInput>.GetBatchSize(input);

    /// <summary>
    /// Algorithm 1, line 9: Z random example pairs. The first index is into the query set (or the support
    /// set under <see cref="MetaMixSource.SupportAndQuery"/>), the second into the query set.
    /// </summary>
    private (int[] First, int[] Second) DrawPairs(IMetaLearningTask<T, TInput, TOutput> task)
    {
        int queryCount = ExampleCount(task.QueryInput);
        int firstCount = _algoOptions.MixSource == MetaMixSource.SupportAndQuery
            ? ExampleCount(task.SupportInput)
            : queryCount;
        if (queryCount <= 0 || firstCount <= 0)
        {
            throw new ArgumentException(
                "MetaMix needs a non-empty query set (and support set when mixing both).", nameof(task));
        }

        int z = _algoOptions.NumMixedPairs > 0 ? _algoOptions.NumMixedPairs : queryCount;
        var first = new int[z];
        var second = new int[z];
        for (int k = 0; k < z; k++)
        {
            first[k] = RandomGenerator.Next(firstCount);
            second[k] = RandomGenerator.Next(queryCount);
        }

        return (first, second);
    }

    /// <summary>
    /// Eq. 4-5: x̂ = λ x_first + (1-λ) x_second and ŷ = λ y_first + (1-λ) y_second for each pair.
    /// </summary>
    internal (TInput X, TOutput Y) BuildMixedSet(
        IMetaLearningTask<T, TInput, TOutput> task, int[] first, int[] second, double lambda)
    {
        T weight = NumOps.FromDouble(lambda);
        bool fromSupport = _algoOptions.MixSource == MetaMixSource.SupportAndQuery;
        var firstX = InputHelper<T, TInput>.GetBatch(fromSupport ? task.SupportInput : task.QueryInput, first);
        var firstY = InputHelper<T, TOutput>.GetBatch(fromSupport ? task.SupportOutput : task.QueryOutput, first);
        var secondX = InputHelper<T, TInput>.GetBatch(task.QueryInput, second);
        var secondY = InputHelper<T, TOutput>.GetBatch(task.QueryOutput, second);
        return (Mix(firstX, secondX, weight), Mix(firstY, secondY, weight));
    }

    /// <summary>λ a + (1-λ) b on whole containers, i.e. a lerp from b towards a by λ.</summary>
    private TData Mix<TData>(TData a, TData b, T lambda)
    {
        object mixed = (object?)a switch
        {
            Tensor<T> ta when b is Tensor<T> tb => Engine.TensorLerp(tb, ta, lambda),
            Matrix<T> ma when b is Matrix<T> mb =>
                Engine.TensorLerp(Tensor<T>.FromMatrix(mb), Tensor<T>.FromMatrix(ma), lambda).ToMatrix(),
            Vector<T> va when b is Vector<T> vb => Engine.Lerp(vb, va, lambda),
            _ => throw new NotSupportedException(
                $"MetaMix mixes Tensor<T>, Matrix<T> or Vector<T> data; got {typeof(TData).Name}."),
        };
        return (TData)mixed;
    }

}
