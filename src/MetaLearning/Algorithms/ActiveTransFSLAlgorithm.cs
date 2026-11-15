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
/// Implementation of ActiveTransFSL: Active Transductive Few-Shot Learning.
/// </summary>
/// <remarks>
/// <para>
/// ActiveTransFSL performs standard inductive adaptation on support data, then applies
/// transductive refinement using query gradients. The refinement is focused on the most
/// uncertain parameter dimensions (measured by gradient magnitude), implementing an
/// active learning strategy in parameter space for transductive inference.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Phase 1 (Inductive): Standard MAML adaptation on support
///   θ_inductive = θ - η * Σ ∇L_support
///
/// Phase 2 (Transductive): Selective refinement using query gradients
///   For each refinement step:
///     grad_query = ∇L_query(θ_current)
///     uncertainty_d = |grad_query_d|
///     Select top-f fraction by uncertainty
///     θ_d ← θ_d - TransductiveLR * grad_query_d  (only for selected d)
///
/// L_meta = (1-TransductiveWeight) * L_inductive + TransductiveWeight * L_transductive
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Transductive Information Maximization for Few-Shot Learning",
    "https://arxiv.org/abs/2008.11297",
    Year = 2020,
    Authors = "Malik Boudiaf, Ziko Imtiaz Masud, Jerome Rony, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ActiveTransFSLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly ActiveTransFSLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ActiveTransFSL;

    public ActiveTransFSLAlgorithm(ActiveTransFSLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));
        if (options.TransductiveLR < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "TransductiveLR must be non-negative.");
        if (options.SelectionFraction <= 0 || options.SelectionFraction > 1)
            throw new ArgumentOutOfRangeException(nameof(options), "SelectionFraction must be in (0, 1].");
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        T transductiveLR = NumOps.FromDouble(_algoOptions.TransductiveLR);
        double w = _algoOptions.TransductiveWeight;
        T inductiveShare = NumOps.FromDouble(1.0 - w);
        T transductiveShare = NumOps.FromDouble(w);
        int threshIdx = (int)((1.0 - _algoOptions.SelectionFraction) * _paramDim);
        if (threshIdx >= _paramDim) threshIdx = _paramDim - 1;

        foreach (var task in taskBatch.Tasks)
        {
            // Phase 1: Inductive adaptation on support
            var adaptedParams = initParams.Clone();

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
            }

            // Inductive loss
            ParamModel.SetParameters(adaptedParams);
            var inductiveLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Phase 2: Active transductive refinement
            var transParams = adaptedParams.Clone();

            for (int rStep = 0; rStep < _algoOptions.TransductiveRefinementSteps; rStep++)
            {
                ParamModel.SetParameters(transParams);
                var queryGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));

                // Uncertainty per parameter is |grad|; the threshold is its threshIdx-th smallest value, so the
                // top (1 - threshIdx / n) fraction of parameters is updated.
                var uncertainty = Engine.TensorAbs(Tensor<T>.FromVector(queryGrad));
                var sorted = Engine.TensorSort(uncertainty, 0, false).Values;
                T threshold = sorted[threshIdx];

                // Selectively update only uncertain parameters: mask = 1 where uncertainty >= threshold
                var selected = Engine.TensorSubtract(
                    Engine.TensorOnesLike(uncertainty), Engine.TensorLessThan(uncertainty, threshold)).ToVector();
                transParams = Engine.Subtract(transParams, Engine.Multiply(Engine.Multiply(queryGrad, selected), transductiveLR));
            }

            ParamModel.SetParameters(transParams);
            var transductiveLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Combine inductive and transductive losses using TransductiveWeight
            // L_meta = (1 - w) * L_inductive + w * L_transductive
            losses.Add(NumOps.Add(
                NumOps.Multiply(inductiveShare, inductiveLoss), NumOps.Multiply(transductiveShare, transductiveLoss)));

            // Compute combined gradient consistent with the combined loss:
            // grad_meta = (1-w) * grad_inductive + w * grad_transductive
            ParamModel.SetParameters(adaptedParams);
            var inductiveGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            ParamModel.SetParameters(transParams);
            var transductiveGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);

            var combinedGrad = Engine.Add(
                Engine.Multiply(inductiveGrad, inductiveShare), Engine.Multiply(transductiveGrad, transductiveShare));
            metaGradients.Add(ClipGradients(combinedGrad));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var adaptedParams = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }
}
