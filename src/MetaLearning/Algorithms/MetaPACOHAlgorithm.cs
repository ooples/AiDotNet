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
/// Implementation of Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning with per-group
/// prior variances.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-PACOH extends PACOH by partitioning the parameter space into G groups and
/// learning independent prior log-variances {log(σ²_g)} for each group. This allows
/// different parts of the network (e.g., early vs late layers) to have different levels
/// of flexibility during task adaptation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: prior mean μ_P, per-group log-variances {log(σ²_g)}
///
/// Inner loop (MAP with group-specific regularization):
///   For each task τ:
///     θ_τ = μ_P
///     For each step:
///       grad_task = ∂L/∂θ
///       grad_prior_d = (θ_d - μ_P_d) / σ²_{g(d)}  (g(d) = group of dim d)
///       θ_τ ← θ_τ - η * (grad_task + λ * grad_prior)
///
/// Outer loop (hierarchical PAC-Bayes bound):
///   L = (1/T)Σ Loss_τ + (λ/T) * (Σ_g KL_g(Q_g || P_g) + KL(P || P_0) + ln(2√T / δ))
///   Update μ_P, {log(σ²_g)} via gradient descent
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees",
    "https://arxiv.org/abs/2002.05551",
    Year = 2021,
    Authors = "Jonas Rothfuss, Vincent Fortuin, Martin Josifoski, Andreas Krause")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaPACOHAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly MetaPACOHOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>Prior mean (meta-learned).</summary>
    private Vector<T> _priorMean;

    /// <summary>Per-group prior log-variances. Length = NumPriorGroups.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _groupLogVars;

    /// <summary>Group assignments: _groupOf[d] = group index for parameter d.</summary>
    private readonly int[] _groupOf;

    /// <summary>Number of parameters per group.</summary>
    private readonly int[] _groupSize;

    private readonly int _paramDim;
    private readonly int _numGroups;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaPACOH;

    public MetaPACOHAlgorithm(MetaPACOHOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.InnerAndOuter)
    {
        if (double.IsNaN(options.Delta) || options.Delta <= 0 || options.Delta >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Delta must lie in (0, 1).");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters. MetaPACOH requires a model with at least one parameter.");
        _numGroups = Math.Min(Math.Max(options.NumPriorGroups, 1), _paramDim);

        _priorMean = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters();

        // Initialize per-group log-variances
        _groupLogVars = new Vector<T>(_numGroups);
        _groupLogVars.Fill(NumOps.FromDouble(options.InitialLogVariance));

        // Assign parameters to groups (even partition)
        _groupOf = new int[_paramDim];
        _groupSize = new int[_numGroups];
        for (int d = 0; d < _paramDim; d++)
        {
            int g = (int)((long)d * _numGroups / _paramDim);
            _groupOf[d] = g;
            _groupSize[g]++;
        }
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var adaptedParamsList = new List<Vector<T>>();
        var metaGradients = new List<Vector<T>>();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: MAP estimation with group-specific prior regularization, from the prior mean
            var theta = AdaptToTask(task, clipTaskGradients: true);
            adaptedParamsList.Add(theta);
            ParamModel.SetParameters(theta);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Compute hierarchical PAC-Bayesian bound: per-group KL + hyper-prior KL
        var pacBayesLoss = BoundPenalty(
            NumOps.Add(ComputeHierarchicalKL(adaptedParamsList), ComputeHyperPriorKL()), taskBatch.Tasks.Length);

        // Update prior mean via meta-gradients
        ParamModel.SetParameters(_priorMean);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            _priorMean = ApplyOuterGradients(_priorMean, avgGrad, _algoOptions.OuterLearningRate);
            ParamModel.SetParameters(_priorMean);
        }

        // Update per-group log-variances via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _groupLogVars, _algoOptions.OuterLearningRate * 0.1, ComputeMetaPACOHLoss);

        return NumOps.Add(ComputeMean(losses), pacBayesLoss);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var theta = AdaptToTask(task, clipTaskGradients: true);
        ParamModel.SetParameters(_priorMean);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, theta);
    }

    /// <summary>Each parameter's prior variance σ²_{g(d)} + 1e-10, read from its group's log-variance.</summary>
    private Vector<T> ParameterPriorVariance()
    {
        using var groupIndex = new Tensor<int>(_groupOf, new[] { _paramDim });
        var perParameterLogVar = Engine.TensorIndexSelect(Tensor<T>.FromVector(_groupLogVars), groupIndex, 0).ToVector();
        return Engine.Exp(perParameterLogVar) + NumOps.FromDouble(1e-10);
    }

    /// <summary>
    /// MAP adaptation from the prior mean: each step descends the support loss plus
    /// KLCoefficient * (θ_d - μ_P_d) / σ²_{g(d)}.
    /// </summary>
    private Vector<T> AdaptToTask(IMetaLearningTask<T, TInput, TOutput> task, bool clipTaskGradients)
    {
        var theta = _priorMean.Clone();
        var priorVariance = ParameterPriorVariance();
        var klCoefficient = NumOps.FromDouble(_algoOptions.KLCoefficient);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(theta);
            var taskGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            if (clipTaskGradients) taskGrad = ClipGradients(taskGrad);

            // Prior gradient with per-group variances, then combined gradient
            var priorGrad = Engine.Divide(Engine.Subtract(theta, _priorMean), priorVariance);
            var stepGradient = Engine.Add(taskGrad, Engine.Multiply(priorGrad, klCoefficient));
            theta = ApplyInnerGradients(theta, stepGradient, _algoOptions.InnerLearningRate, step == 0);
        }

        return theta;
    }

    /// <summary>
    /// The PAC-Bayesian complexity term KLCoefficient * (KL + ln(2 sqrt(n) / δ)) / n over n tasks, with the
    /// confidence term of the PAC-Bayes bound (Rothfuss et al. 2021, Thm. 1).
    /// </summary>
    private T BoundPenalty(T kl, int taskCount)
    {
        int n = Math.Max(taskCount, 1);
        double confidence = Math.Log(2.0 * Math.Sqrt(n) / _algoOptions.Delta);
        return NumOps.Multiply(
            NumOps.FromDouble(_algoOptions.KLCoefficient / n), NumOps.Add(kl, NumOps.FromDouble(confidence)));
    }

    private T ComputeHierarchicalKL(List<Vector<T>> adaptedParams)
    {
        if (adaptedParams.Count == 0) return NumOps.Zero;

        // Empirical posterior mean and variance across tasks
        var (posteriorMean, posteriorVar) = PACOHAlgorithm<T, TInput, TOutput>.EmpiricalMoments(adaptedParams, _paramDim);
        posteriorVar = posteriorVar + NumOps.FromDouble(1e-10);

        // Sum KL per dimension using group-specific prior variance
        var priorVar = ParameterPriorVariance();
        var diff = Engine.Subtract(posteriorMean, _priorMean);
        var terms = Engine.Add(
            Engine.Divide(Engine.Add(Engine.Multiply(diff, diff), posteriorVar), priorVar),
            VectorLog(Engine.Divide(priorVar, posteriorVar)));
        return NumOps.Multiply(
            NumOps.FromDouble(0.5), NumOps.Subtract(Engine.Sum(terms), NumOps.FromDouble(_paramDim)));
    }

    /// <summary>KL of the group log-variances to the hyper-prior: Σ_g (log σ²_g - h)² / (2 (e^h + 1e-10)).</summary>
    private T ComputeHyperPriorKL()
    {
        double hyperVar = Math.Exp(_algoOptions.HyperPriorLogVar) + 1e-10;
        var diff = _groupLogVars - NumOps.FromDouble(_algoOptions.HyperPriorLogVar);
        return NumOps.Divide(Engine.Sum(Engine.Multiply(diff, diff)), NumOps.FromDouble(2.0 * hyperVar));
    }

    private double ComputeMetaPACOHLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var adapted = new List<Vector<T>>();

        foreach (var task in taskBatch.Tasks)
        {
            var theta = AdaptToTask(task, clipTaskGradients: false);
            adapted.Add(theta);
            ParamModel.SetParameters(theta);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        var bound = losses.Count == 0
            ? NumOps.Zero
            : NumOps.Add(
                ComputeMean(losses),
                BoundPenalty(NumOps.Add(ComputeHierarchicalKL(adapted), ComputeHyperPriorKL()), taskBatch.Tasks.Length));
        ParamModel.SetParameters(_priorMean);
        return NumOps.ToDouble(bound);
    }
}
