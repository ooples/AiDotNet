using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of SDCL: Self-Distillation Collaborative Learning for meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// SDCL applies knowledge distillation within the meta-learning loop. A teacher model
/// (EMA of student parameters) provides soft targets that regularize the adapted student.
/// The distillation loss (symmetric KL divergence between teacher and student output
/// distributions) stabilizes adaptation and prevents overfitting to the few support examples.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Teacher params: θ_T (EMA of student: θ_T ← α*θ_T + (1-α)*θ_S)
///
/// Inner loop:
///   For each task τ:
///     θ_S ← θ_0
///     For each step:
///       L_task = CrossEntropy(f(x;θ_S), y)
///       L_distill = KL(softmax(f(x;θ_T)/τ) || softmax(f(x;θ_S)/τ))
///       L_total = L_task + w_distill * τ² * L_distill
///       θ_S ← θ_S - η * ∇L_total
///
/// Outer loop:
///   Update θ_0 via average query gradients
///   Update θ_T via EMA of θ_0
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Born Again Neural Networks",
    "https://arxiv.org/abs/1805.04770",
    Year = 2018,
    Authors = "Tommaso Furlanello, Zachary C. Lipton, Michael Tschannen, Laurent Itti, Anima Anandkumar")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class SDCLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly SDCLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Teacher model parameters (EMA of student).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _teacherParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SDCL;

    public SDCLAlgorithm(SDCLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.InnerAndOuter)
    {
        if (options.DistillationTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "DistillationTemperature must be positive.");
        if (options.DistillationWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "DistillationWeight must be non-negative.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;

        // Initialize teacher as copy of student
        _teacherParams = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Clone();
        _distillationLoss = new TemperatureDistillationLoss(options.DistillationTemperature);
    }

    /// <summary>tau^2 KL(softmax(teacher / tau) || softmax(student / tau)), the inner loop's distillation loss.</summary>
    private readonly TemperatureDistillationLoss _distillationLoss;

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var paramModel = ParamModel;
        var initParams = paramModel.GetParameters();

        try
        {
        foreach (var task in taskBatch.Tasks)
        {
            // Get teacher's soft targets on support and query sets
            paramModel.SetParameters(_teacherParams);
            var teacherSupport = MetaModel.Predict(task.SupportInput);
            var teacherQueryPred = ConvertToVector(MetaModel.Predict(task.QueryInput))
                ?? throw new InvalidOperationException("Teacher model returned non-vectorizable prediction for query input.");

            // Inner loop: adapt student with task loss + distillation loss
            var studentParams = AdaptStudent(initParams, task, teacherSupport);

            // Evaluate on query set, plus the distillation loss there
            paramModel.SetParameters(studentParams);
            var queryOutput = MetaModel.Predict(task.QueryInput);
            losses.Add(WithDistillation(ComputeLossFromOutput(queryOutput, task.QueryOutput), teacherQueryPred, queryOutput));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        }
        catch
        {
            // Restore model to original state on exception
            paramModel.SetParameters(initParams);
            throw;
        }
        // Outer loop: update student
        paramModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var newParams = ApplyOuterGradients(initParams, avgGrad, _algoOptions.OuterLearningRate);
            paramModel.SetParameters(newParams);

            // Update teacher via EMA: theta_T <- alpha theta_T + (1 - alpha) theta_S.
            _teacherParams = Engine.Lerp(newParams, _teacherParams, NumOps.FromDouble(_algoOptions.TeacherEmaDecay));
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var paramModel = ParamModel;
        var initParams = paramModel.GetParameters();

        // Get teacher predictions
        paramModel.SetParameters(_teacherParams);
        var teacherSupport = MetaModel.Predict(task.SupportInput);
        var studentParams = AdaptStudent(initParams, task, teacherSupport);
        paramModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, studentParams);
    }

    /// <summary>
    /// The inner loop: theta_S &lt;- theta_S - eta * (grad L_task + w * grad L_distill), both gradients by backpropagation.
    /// </summary>
    private Vector<T> AdaptStudent(Vector<T> initParams, IMetaLearningTask<T, TInput, TOutput> task, TOutput teacherSupport)
    {
        var studentParams = initParams.Clone();
        var weight = NumOps.FromDouble(_algoOptions.DistillationWeight);
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(studentParams);
            var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var distillGrad = ComputeDistillationGradient(task.SupportInput, teacherSupport);
            var stepGradient = Engine.Add(taskGrad, Engine.Multiply(distillGrad, weight));
            studentParams = ApplyInnerGradients(studentParams, stepGradient, _algoOptions.InnerLearningRate, step == 0);
        }

        return studentParams;
    }

    /// <summary>
    /// d/d theta of tau^2 KL(softmax(teacher / tau) || softmax(f(x; theta) / tau)) at the current parameters, by
    /// backpropagating the temperature distillation loss through the model (it was a one-sample SPSA estimate).
    /// </summary>
    private Vector<T> ComputeDistillationGradient(TInput supportInput, TOutput teacherSupport)
        => ComputeGradients(MetaModel, supportInput, teacherSupport, _distillationLoss);

    /// <summary>queryLoss + w * tau^2 KL(teacher || student) on the query predictions.</summary>
    private T WithDistillation(T queryLoss, Vector<T> teacherQueryPred, TOutput studentQueryOutput)
    {
        var studentQueryPred = ConvertToVector(studentQueryOutput) ?? new Vector<T>(1);
        return NumOps.Add(queryLoss, NumOps.Multiply(
            NumOps.FromDouble(_algoOptions.DistillationWeight), ComputeKLDivergence(teacherQueryPred, studentQueryPred)));
    }

    /// <summary>
    /// tau^2 KL(softmax(teacher / tau) || softmax(student / tau)) over the first min(lengths) outputs.
    /// </summary>
    internal T ComputeKLDivergence(Vector<T> teacherPred, Vector<T> studentPred)
    {
        int dim = Math.Min(teacherPred.Length, studentPred.Length);
        if (dim == 0) return NumOps.Zero;
        return _distillationLoss.CalculateLoss(studentPred.GetSubVector(0, dim), teacherPred.GetSubVector(0, dim));
    }

    /// <summary>
    /// Hinton-style distillation loss tau^2 KL(p_T || p_S), p = softmax(logits / tau); "actual" is the teacher's
    /// logits, "predicted" the student's. Terms with p_T &lt;= 1e-10 contribute nothing, and p_S is offset by 1e-10.
    /// </summary>
    private sealed class TemperatureDistillationLoss : LossFunctionBase<T>
    {
        private readonly double _temperature;

        public TemperatureDistillationLoss(double temperature) => _temperature = temperature;

        public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
            => ComputeTapeLoss(Tensor<T>.FromVector(predicted), Tensor<T>.FromVector(actual)).GetFlat(0);

        public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
        {
            int axis = predicted.Rank - 1;
            T inverseTemperature = NumOps.FromDouble(1.0 / _temperature);
            var pTeacher = Engine.TensorSoftmax(Engine.TensorMultiplyScalar(target, inverseTemperature), axis);
            var pStudent = Engine.TensorSoftmax(Engine.TensorMultiplyScalar(predicted, inverseTemperature), axis);
            var mask = Engine.TensorGreaterThan(pTeacher, NumOps.FromDouble(1e-10));
            var logRatio = Engine.TensorSubtract(
                Engine.TensorLog(Engine.TensorMax(pTeacher, NumOps.FromDouble(1e-10))),
                Engine.TensorLog(Engine.TensorAddScalar(pStudent, NumOps.FromDouble(1e-10))));
            var terms = Engine.TensorMultiply(mask, Engine.TensorMultiply(pTeacher, logRatio));
            var allAxes = Enumerable.Range(0, terms.Rank).ToArray();
            return Engine.TensorMultiplyScalar(
                Engine.ReduceSum(terms, allAxes, keepDims: false), NumOps.FromDouble(_temperature * _temperature));
        }
    }
}
