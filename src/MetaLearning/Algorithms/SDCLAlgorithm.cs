using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

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
public class SDCLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly SDCLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Teacher model parameters (EMA of student).</summary>
    private Vector<T> _teacherParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SDCL;

    public SDCLAlgorithm(SDCLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.DistillationTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "DistillationTemperature must be positive.");
        if (options.DistillationWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "DistillationWeight must be non-negative.");

        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;

        // Initialize teacher as copy of student
        var initParams = options.MetaModel.GetParameters();
        _teacherParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) _teacherParams[d] = initParams[d];
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Get teacher's soft targets on support set
            MetaModel.SetParameters(_teacherParams);
            var teacherSupportPred = ConvertToVector(MetaModel.Predict(task.SupportInput)) ?? new Vector<T>(1);
            var teacherQueryPred = ConvertToVector(MetaModel.Predict(task.QueryInput)) ?? new Vector<T>(1);

            // Inner loop: adapt student with task loss + distillation loss
            var studentParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) studentParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(studentParams);
                var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Compute distillation gradient: encourage student to match teacher's soft outputs
                var distillGrad = ComputeDistillationGradient(studentParams, teacherSupportPred, task.SupportInput);

                // Combined gradient
                for (int d = 0; d < _paramDim; d++)
                {
                    double combined = NumOps.ToDouble(taskGrad[d]) + _algoOptions.DistillationWeight * NumOps.ToDouble(distillGrad[d]);
                    studentParams[d] = NumOps.Subtract(studentParams[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
                }
            }

            // Evaluate on query set
            MetaModel.SetParameters(studentParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Add distillation loss on query set
            var studentQueryPred = ConvertToVector(MetaModel.Predict(task.QueryInput)) ?? new Vector<T>(1);
            double distillLoss = ComputeKLDivergence(teacherQueryPred, studentQueryPred);
            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.DistillationWeight * distillLoss));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update student
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var newParams = ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate);
            MetaModel.SetParameters(newParams);

            // Update teacher via EMA
            double alpha = _algoOptions.TeacherEmaDecay;
            for (int d = 0; d < _paramDim; d++)
            {
                double teacher = alpha * NumOps.ToDouble(_teacherParams[d]) + (1 - alpha) * NumOps.ToDouble(newParams[d]);
                _teacherParams[d] = NumOps.FromDouble(teacher);
            }
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();

        // Get teacher predictions
        MetaModel.SetParameters(_teacherParams);
        var teacherPred = ConvertToVector(MetaModel.Predict(task.SupportInput)) ?? new Vector<T>(1);

        var studentParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) studentParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(studentParams);
            var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            var distillGrad = ComputeDistillationGradient(studentParams, teacherPred, task.SupportInput);

            for (int d = 0; d < _paramDim; d++)
            {
                double combined = NumOps.ToDouble(taskGrad[d]) + _algoOptions.DistillationWeight * NumOps.ToDouble(distillGrad[d]);
                studentParams[d] = NumOps.Subtract(studentParams[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * combined));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, studentParams);
    }

    private const double SpsaPerturbScale = 0.01;

    /// <summary>
    /// Computes the distillation gradient: ∂KL(teacher || student) / ∂θ_student.
    /// Uses SPSA (Simultaneous Perturbation Stochastic Approximation) for finite-difference
    /// gradient estimation when analytical gradients are not available.
    /// </summary>
    private Vector<T> ComputeDistillationGradient(Vector<T> studentParams, Vector<T> teacherPred, TInput supportInput)
    {
        var grad = new Vector<T>(_paramDim);

        // SPSA: perturb all parameters simultaneously with Rademacher random variables
        var delta = new double[_paramDim];
        for (int d = 0; d < _paramDim; d++)
            delta[d] = RandomGenerator.NextDouble() < 0.5 ? -1.0 : 1.0;

        // Compute loss at studentParams + perturbation
        var paramsPlus = new Vector<T>(_paramDim);
        var paramsMinus = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            double p = NumOps.ToDouble(studentParams[d]);
            paramsPlus[d] = NumOps.FromDouble(p + SpsaPerturbScale * delta[d]);
            paramsMinus[d] = NumOps.FromDouble(p - SpsaPerturbScale * delta[d]);
        }

        double lossPlus = ComputeKLDivergence(teacherPred, GetPredictionWithParams(paramsPlus, supportInput));
        double lossMinus = ComputeKLDivergence(teacherPred, GetPredictionWithParams(paramsMinus, supportInput));

        // SPSA gradient estimate: g_d = (L+ - L-) / (2 * c * delta_d)
        double gradScale = (lossPlus - lossMinus) / (2.0 * SpsaPerturbScale);
        for (int d = 0; d < _paramDim; d++)
        {
            grad[d] = NumOps.FromDouble(gradScale / delta[d]);
        }

        return grad;
    }

    /// <summary>
    /// Gets a prediction by temporarily setting model parameters and running forward pass.
    /// </summary>
    private Vector<T> GetPredictionWithParams(Vector<T> parameters, TInput input)
    {
        var savedParams = MetaModel.GetParameters();
        MetaModel.SetParameters(parameters);
        var prediction = ConvertToVector(MetaModel.Predict(input)) ?? new Vector<T>(1);
        MetaModel.SetParameters(savedParams);
        return prediction;
    }

    /// <summary>
    /// Computes symmetric KL divergence between two output vectors treated as soft distributions.
    /// KL(p || q) ≈ Σ_i (p_i - q_i)² / (2 * max(q_i, ε)) for numerical stability.
    /// </summary>
    private double ComputeKLDivergence(Vector<T> teacherPred, Vector<T> studentPred)
    {
        double temp = _algoOptions.DistillationTemperature;
        int dim = Math.Min(teacherPred.Length, studentPred.Length);
        if (dim == 0) return 0;

        // Compute softened distributions
        double maxT = double.NegativeInfinity, maxS = double.NegativeInfinity;
        for (int i = 0; i < dim; i++)
        {
            double t = NumOps.ToDouble(teacherPred[i]) / temp;
            double s = NumOps.ToDouble(studentPred[i]) / temp;
            if (t > maxT) maxT = t;
            if (s > maxS) maxS = s;
        }

        double sumExpT = 0, sumExpS = 0;
        for (int i = 0; i < dim; i++)
        {
            sumExpT += Math.Exp(NumOps.ToDouble(teacherPred[i]) / temp - maxT);
            sumExpS += Math.Exp(NumOps.ToDouble(studentPred[i]) / temp - maxS);
        }

        // KL(teacher || student)
        double kl = 0;
        for (int i = 0; i < dim; i++)
        {
            double pT = Math.Exp(NumOps.ToDouble(teacherPred[i]) / temp - maxT) / (sumExpT + 1e-10);
            double pS = Math.Exp(NumOps.ToDouble(studentPred[i]) / temp - maxS) / (sumExpS + 1e-10);
            if (pT > 1e-10)
                kl += pT * Math.Log(pT / (pS + 1e-10));
        }

        return temp * temp * kl;
    }
}
