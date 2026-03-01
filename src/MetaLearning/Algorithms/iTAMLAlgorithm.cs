using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of iTAML: incremental Task-Agnostic Meta-Learning (Rajasegaran et al., 2020).
/// </summary>
/// <remarks>
/// <para>
/// iTAML prevents catastrophic forgetting by maintaining an exponential moving average (EMA)
/// teacher model. The meta-objective combines task-specific loss with a knowledge distillation
/// loss that preserves the teacher's predictions. Task-balanced gradient weighting normalizes
/// gradient magnitudes across tasks to prevent any single task from dominating.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Teacher model: θ_teacher = EMA(θ_student) with decay α
///
/// Inner loop: standard MAML-style adaptation
///   θ_adapted = θ - η * ∇L_support(θ)
///
/// Distillation loss: KL between softened student/teacher predictions
///   L_distill = Σ_d (p_teacher_d - p_student_d)² / temperature²
///   (simplified L2 distillation for generic T)
///
/// Task-balanced gradient: normalize each task gradient by its L2 norm
///   g_normalized = g / ||g|| if TaskBalancingEnabled
///
/// L_meta = L_query + DistillationWeight * L_distill
/// Outer: update student, then teacher = α * teacher + (1-α) * student
/// </code>
/// </para>
/// </remarks>
public class iTAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly iTAMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Teacher model parameters (EMA of student).</summary>
    private Vector<T> _teacherParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.iTAML;

    public iTAMLAlgorithm(iTAMLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
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
            // Inner loop: standard MAML adaptation
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            // Query loss with adapted params
            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Knowledge distillation loss: L2 distance between student and teacher predictions
            // Student prediction (already computed) vs teacher prediction
            var studentPred = ConvertToVector(MetaModel.Predict(task.QueryInput)) ?? new Vector<T>(1);
            MetaModel.SetParameters(_teacherParams);
            var teacherPred = ConvertToVector(MetaModel.Predict(task.QueryInput)) ?? new Vector<T>(1);

            double distillLoss = 0;
            double tempSq = _algoOptions.DistillationTemperature * _algoOptions.DistillationTemperature;
            int predLen = Math.Min(studentPred.Length, teacherPred.Length);
            for (int d = 0; d < predLen; d++)
            {
                double diff = NumOps.ToDouble(studentPred[d]) - NumOps.ToDouble(teacherPred[d]);
                distillLoss += diff * diff / tempSq;
            }
            if (predLen > 0) distillLoss /= predLen;

            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.DistillationWeight * distillLoss));

            losses.Add(totalLoss);

            // Compute meta-gradient from adapted student
            MetaModel.SetParameters(adaptedParams);
            var metaGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));

            // Task-balanced gradient normalization
            if (_algoOptions.TaskBalancingEnabled)
            {
                double gradNorm = 0;
                for (int d = 0; d < _paramDim; d++)
                    gradNorm += NumOps.ToDouble(metaGrad[d]) * NumOps.ToDouble(metaGrad[d]);
                gradNorm = Math.Sqrt(gradNorm) + 1e-10;
                for (int d = 0; d < _paramDim; d++)
                    metaGrad[d] = NumOps.FromDouble(NumOps.ToDouble(metaGrad[d]) / gradNorm);
            }

            metaGradients.Add(metaGrad);
        }

        // Outer loop: update student params
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var newParams = ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate);
            MetaModel.SetParameters(newParams);

            // Update teacher via EMA
            double decay = _algoOptions.TeacherEmaDecay;
            for (int d = 0; d < _paramDim; d++)
                _teacherParams[d] = NumOps.FromDouble(
                    decay * NumOps.ToDouble(_teacherParams[d])
                  + (1.0 - decay) * NumOps.ToDouble(newParams[d]));
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }
}
