using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of MetaTask: Meta-learned Task Augmentation via gradient interpolation.
/// </summary>
/// <remarks>
/// <para>
/// MetaTask augments the meta-learning task distribution by generating synthetic tasks
/// from convex combinations of real task gradients. For each synthetic task, two real tasks
/// are randomly selected and their gradients are interpolated using a Beta-distributed
/// coefficient. The meta-objective combines losses from both real and synthetic tasks.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// For each real task t_i:
///   Standard MAML inner loop → loss_real_i
///
/// For each synthetic task (s = 1..NumSyntheticTasks):
///   Sample pair (t_i, t_j) randomly
///   λ ~ Beta(α, α)
///   grad_s = λ * grad_i + (1-λ) * grad_j  (gradient interpolation)
///   Adapt using interpolated gradients → loss_synthetic_s
///
/// L_meta = mean(loss_real) + SyntheticWeight * mean(loss_synthetic)
/// </code>
/// </para>
/// </remarks>
public class MetaTaskAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaTaskOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaTask;

    public MetaTaskAlgorithm(MetaTaskOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var realLosses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        // Collect per-task gradients for interpolation
        var taskGradients = new List<Vector<T>>();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            Vector<T>? lastGrad = null;
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                lastGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, lastGrad, _algoOptions.InnerLearningRate);
            }

            if (lastGrad != null)
                taskGradients.Add(lastGrad);

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            realLosses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Generate synthetic tasks via gradient interpolation
        var syntheticLosses = new List<T>();
        if (taskGradients.Count >= 2)
        {
            for (int s = 0; s < _algoOptions.NumSyntheticTasks; s++)
            {
                // Sample random pair
                int i = RandomGenerator.Next(taskGradients.Count);
                int j = RandomGenerator.Next(taskGradients.Count - 1);
                if (j >= i) j++;

                // Sample Beta(α,α) interpolation coefficient
                double lambda = SampleBeta(_algoOptions.InterpolationAlpha, _algoOptions.InterpolationAlpha);

                // Interpolate gradients and adapt
                var adaptedParams = new Vector<T>(_paramDim);
                for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

                for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
                {
                    // Mix gradients from the two selected tasks
                    var interpGrad = new Vector<T>(_paramDim);
                    for (int d = 0; d < _paramDim; d++)
                        interpGrad[d] = NumOps.FromDouble(
                            lambda * NumOps.ToDouble(taskGradients[i][d])
                          + (1.0 - lambda) * NumOps.ToDouble(taskGradients[j][d]));

                    adaptedParams = ApplyGradients(adaptedParams, interpGrad, _algoOptions.InnerLearningRate);
                }

                // Evaluate synthetic task using a randomly selected real task's query set
                int evalIdx = RandomGenerator.Next(taskBatch.Tasks.Length);
                MetaModel.SetParameters(adaptedParams);
                var synthLoss = ComputeLossFromOutput(
                    MetaModel.Predict(taskBatch.Tasks[evalIdx].QueryInput),
                    taskBatch.Tasks[evalIdx].QueryOutput);
                syntheticLosses.Add(synthLoss);
            }
        }

        // Combine real and synthetic losses
        T totalLoss = ComputeMean(realLosses);
        if (syntheticLosses.Count > 0)
            totalLoss = NumOps.Add(totalLoss,
                NumOps.Multiply(NumOps.FromDouble(_algoOptions.SyntheticWeight), ComputeMean(syntheticLosses)));

        // Outer update
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        return totalLoss;
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

    private double SampleBeta(double alpha, double beta)
    {
        double x = SampleGamma(alpha);
        double y = SampleGamma(beta);
        return x / (x + y + 1e-10);
    }

    private double SampleGamma(double shape)
    {
        if (shape < 1.0)
        {
            double u = RandomGenerator.NextDouble();
            return SampleGamma(shape + 1.0) * Math.Pow(u, 1.0 / shape);
        }
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            double z = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            double v = Math.Pow(1.0 + c * z, 3);
            if (v > 0 && Math.Log(RandomGenerator.NextDouble()) < 0.5 * z * z + d - d * v + d * Math.Log(v))
                return d * v;
        }
    }
}
