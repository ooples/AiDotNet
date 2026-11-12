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
/// Implementation of OML: Online-aware Meta-Learning (Javed &amp; White, NeurIPS 2019).
/// </summary>
/// <remarks>
/// <para>
/// OML partitions model parameters into a Representation Learning Network (RLN, parameters theta) and a
/// Prediction Learning Network (PLN, parameters W). The inner loop is ONLINE: it updates only W, one
/// support example at a time, as a continual learner would see a trajectory. The meta-objective then
/// asks the adapted network to fit both that trajectory and random samples from the task (the query set),
/// so that sequential updates do not interfere with what was learned before. Sparse representations
/// emerge from this objective; the paper adds no sparsity penalty.
/// </para>
/// <para><b>Algorithm (paper Algorithm 1):</b>
/// <code>
/// W_0 = W
/// for j = 1..k (each support example x_j in turn):  W_j = W_{j-1} - beta * grad_W L(f(x_j; theta, W_{j-1}))
/// meta-loss: L(f(S_traj U X_rand; theta, W_k)),  update theta and W with it (first order)
/// </code>
/// </para>
/// <para>
/// PLN indices are the last <c>PlnFraction</c> of the flat parameter vector. <c>SparsityPenalty</c> adds an
/// optional L1 term on the RLN parameters (off by default, as in the paper).
/// </para>
/// </remarks>[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning Representations for Continual Learning",
    "https://arxiv.org/abs/1905.12588",
    Year = 2019,
    Authors = "Khurram Javed, Martha White")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class OMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly OMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _plnStart;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.OML;

    public OMLAlgorithm(OMLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.InnerAndOuter)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters. OML requires a model with at least one parameter.");
        _plnStart = Math.Min(Math.Max((int)(_paramDim * (1.0 - options.PlnFraction)), 0), _paramDim);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        T penalty = NumOps.FromDouble(_algoOptions.SparsityPenalty);

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = AdaptPredictionNetwork(task, initParams);
            ParamModel.SetParameters(adaptedParams);

            // Meta-loss on S_traj U X_rand: the per-example mean over both sets.
            int nSupport = InputHelper<T, TInput>.GetBatchSize(task.SupportInput);
            int nQuery = InputHelper<T, TInput>.GetBatchSize(task.QueryInput);
            T wSupport = NumOps.FromDouble((double)nSupport / (nSupport + nQuery));
            T wQuery = NumOps.FromDouble((double)nQuery / (nSupport + nQuery));
            T trajectoryLoss = ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput);
            T randomLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            var gradient = Engine.Add(
                Engine.Multiply(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput), wSupport),
                Engine.Multiply(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput), wQuery));
            T loss = NumOps.Add(NumOps.Multiply(trajectoryLoss, wSupport), NumOps.Multiply(randomLoss, wQuery));

            // Optional L1 on the RLN parameters: penalty * ||theta||_1, gradient penalty * sign(theta).
            if (_plnStart > 0 && _algoOptions.SparsityPenalty > 0)
            {
                var rln = adaptedParams.GetSubVector(0, _plnStart);
                loss = NumOps.Add(loss, NumOps.Multiply(penalty, Engine.Sum(Engine.Abs(rln))));
                gradient = Engine.Add(gradient, Vector<T>.Concatenate(new[]
                {
                    Engine.Multiply(Engine.Sign(rln), penalty),
                    new Vector<T>(_paramDim - _plnStart),
                }));
            }

            losses.Add(loss);
            metaGradients.Add(ClipGradients(gradient));
        }

        // Outer loop: update ALL parameters (both RLN and PLN).
        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = AdaptPredictionNetwork(task, initParams);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// The inner loop: <c>AdaptationSteps</c> passes over the support set, updating only the PLN. With
    /// <c>OnlineInnerLoop</c> (the paper) each pass takes one step per support example, in order.
    /// </summary>
    private Vector<T> AdaptPredictionNetwork(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var adaptedParams = initParams.Clone();
        int examples = _algoOptions.OnlineInnerLoop ? InputHelper<T, TInput>.GetBatchSize(task.SupportInput) : 1;
        bool first = true;
        for (int pass = 0; pass < _algoOptions.AdaptationSteps; pass++)
        {
            for (int j = 0; j < examples; j++)
            {
                var input = task.SupportInput;
                var target = task.SupportOutput;
                if (_algoOptions.OnlineInnerLoop)
                {
                    var index = new[] { j };
                    input = InputHelper<T, TInput>.GetBatch(task.SupportInput, index);
                    target = InputHelper<T, TOutput>.GetBatch(task.SupportOutput, index);
                }

                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, input, target));
                adaptedParams = StepPredictionNetwork(adaptedParams, grad, first);
                first = false;
            }
        }

        return adaptedParams;
    }

    /// <summary>
    /// One inner step on the prediction network's parameters only (the representation network stays frozen).
    /// </summary>
    private Vector<T> StepPredictionNetwork(Vector<T> adaptedParams, Vector<T> grad, bool firstStep)
    {
        int count = _paramDim - _plnStart;
        var updated = ApplyInnerGradients(
            adaptedParams.GetSubVector(_plnStart, count), grad.GetSubVector(_plnStart, count),
            _algoOptions.InnerLearningRate, firstStep);
        return _plnStart == 0
            ? updated
            : Vector<T>.Concatenate(new[] { adaptedParams.GetSubVector(0, _plnStart), updated });
    }
}
