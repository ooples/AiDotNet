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
/// Implementation of iTAML: incremental Task-Agnostic Meta-Learning (Rajasegaran et al., CVPR 2020).
/// </summary>
/// <remarks>
/// <para>
/// iTAML keeps a single set of meta-parameters balanced across all tasks seen so far. Each task in the
/// batch takes r inner steps from the shared parameters; the meta-update then moves the shared parameters
/// towards the average of the task-adapted ones, by a weight eta that shrinks as training progresses, so
/// later tasks cannot pull the parameters away from earlier ones as fast as the first did.
/// </para>
/// <para><b>Algorithm (paper Algorithm 1):</b>
/// <code>
/// Phi_base = Phi
/// for each task i in the batch:  Phi_i = Phi;  repeat r times: Phi_i = Optimizer(Phi_i, L_i)
/// eta = exp(-beta * t / T)
/// Phi = eta * (1/n) sum_i Phi_i + (1 - eta) * Phi_base
/// </code>
/// </para>
/// <para>
/// t / T is the fraction of <c>NumMetaIterations</c> completed. Inference-time adaptation (Algorithm 3)
/// is the same r-step update on the task's support set.
/// </para>
/// </remarks>[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("iTAML: An Incremental Task-Agnostic Meta-learning Approach",
    "https://arxiv.org/abs/2003.11652",
    Year = 2020,
    Authors = "Jathushan Rajasegaran, Salman Khan, Munawar Hayat, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class iTAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly iTAMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Meta-iterations completed, the t in eta = exp(-beta * t / T).</summary>
    private int _metaIteration;
    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.iTAML;

    public iTAMLAlgorithm(iTAMLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.EtaDecay < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "EtaDecay must be non-negative.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var baseParams = ParamModel.GetParameters();
        var adaptedSum = new Vector<T>(_paramDim);

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = AdaptFrom(baseParams, task);
            adaptedSum = Engine.Add(adaptedSum, adaptedParams);

            ParamModel.SetParameters(adaptedParams);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        // Phi = eta * mean_i Phi_i + (1 - eta) * Phi_base, i.e. a step of size eta along Phi_base - mean_i Phi_i.
        var meanAdapted = Engine.Divide(adaptedSum, NumOps.FromDouble(taskBatch.Tasks.Length));
        double eta = CurrentEta();
        ParamModel.SetParameters(baseParams);
        ParamModel.SetParameters(ApplyOuterGradients(baseParams, Engine.Subtract(baseParams, meanAdapted), eta));
        _metaIteration++;

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = AdaptFrom(initParams, task);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>eta = exp(-beta * t / T) for the current meta-iteration.</summary>
    internal double CurrentEta()
    {
        double progress = _algoOptions.NumMetaIterations > 0
            ? Math.Min(1.0, (double)_metaIteration / _algoOptions.NumMetaIterations)
            : 0.0;
        return Math.Exp(-_algoOptions.EtaDecay * progress);
    }

    /// <summary>r inner steps on the task's support set from <paramref name="start"/>.</summary>
    private Vector<T> AdaptFrom(Vector<T> start, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var adaptedParams = start.Clone();
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        return adaptedParams;
    }
}
