using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-FDMixup: Feature-Distribution Mixup for cross-domain
/// few-shot learning (Xu et al., CVPR 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-FDMixup performs gradient-level mixup between tasks in a meta-batch to improve
/// cross-domain robustness. For each task, with probability p, its inner-loop gradient
/// is mixed with a randomly selected other task's gradient using a Beta-distributed
/// coefficient λ. The outer loop also applies feature distribution alignment by
/// penalizing the variance of per-task gradient directions.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// For each meta-batch:
///   1. Compute per-task gradients g_τ on support sets
///   2. With probability p, mix gradients:
///      g'_τ = λ * g_τ + (1-λ) * g_{τ'}, where λ ~ Beta(α, α)
///   3. Inner loop: θ_τ ← θ_0 - η * g'_τ (for each step)
///   4. Outer loop: θ_0 ← θ_0 - η_outer * [mean(∇L_query) + w_align * Var(g_τ)]
/// </code>
/// </para>
/// </remarks>
public class MetaFDMixupAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaFDMixupOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaFDMixup;

    public MetaFDMixupAlgorithm(MetaFDMixupOptions<T, TInput, TOutput> options)
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
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();
        var tasks = taskBatch.Tasks;

        // Step 1: Compute initial task gradients for mixup pool
        var taskGradPool = new List<Vector<T>>();
        foreach (var task in tasks)
        {
            MetaModel.SetParameters(initParams);
            taskGradPool.Add(ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput)));
        }

        // Step 2: Inner loop with gradient mixup
        for (int tIdx = 0; tIdx < tasks.Length; tIdx++)
        {
            var task = tasks[tIdx];
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var taskGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Apply gradient mixup with probability p
                if (RandomGenerator.NextDouble() < _algoOptions.MixupProbability && tasks.Length > 1)
                {
                    // Sample mixup partner (different task)
                    int partner = tIdx;
                    while (partner == tIdx)
                        partner = RandomGenerator.Next(tasks.Length);

                    // Sample λ ~ Beta(α, α) using Gamma trick
                    double lambda = SampleBeta(_algoOptions.MixupAlpha, _algoOptions.MixupAlpha);

                    // Mix gradients: g' = λ*g_task + (1-λ)*g_partner
                    var partnerGrad = taskGradPool[partner];
                    for (int d = 0; d < _paramDim; d++)
                    {
                        double mixed = lambda * NumOps.ToDouble(taskGrad[d]) + (1.0 - lambda) * NumOps.ToDouble(partnerGrad[d]);
                        taskGrad[d] = NumOps.FromDouble(mixed);
                    }
                }

                adaptedParams = ApplyGradients(adaptedParams, taskGrad, _algoOptions.InnerLearningRate);
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Compute feature distribution alignment penalty (gradient direction variance)
        double alignmentPenalty = ComputeGradientAlignmentPenalty(taskGradPool);

        // Outer loop
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            // Add alignment gradient: penalize high variance in gradient directions
            for (int d = 0; d < _paramDim; d++)
            {
                double alignGrad = _algoOptions.AlignmentWeight * ComputeAlignmentGradient(taskGradPool, d);
                avgGrad[d] = NumOps.Add(avgGrad[d], NumOps.FromDouble(alignGrad));
            }
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        var totalLoss = NumOps.Add(ComputeMean(losses), NumOps.FromDouble(alignmentPenalty));
        return totalLoss;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = currentParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(currentParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Samples from Beta(α, β) using the Gamma distribution trick.
    /// Beta(a,b) = Gamma(a)/(Gamma(a)+Gamma(b)).
    /// Gamma sampling via Marsaglia and Tsang's method simplified for our case.
    /// </summary>
    private double SampleBeta(double alpha, double beta)
    {
        double x = SampleGamma(alpha);
        double y = SampleGamma(beta);
        return x / (x + y + 1e-10);
    }

    private double SampleGamma(double shape)
    {
        if (shape >= 1.0)
        {
            // Marsaglia-Tsang
            double d = shape - 1.0 / 3.0;
            double c = 1.0 / Math.Sqrt(9.0 * d);
            while (true)
            {
                double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
                double u2 = RandomGenerator.NextDouble();
                double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                double v = 1.0 + c * z;
                if (v > 0)
                {
                    v = v * v * v;
                    if (Math.Log(u1) < 0.5 * z * z + d - d * v + d * Math.Log(v))
                        return d * v;
                }
            }
        }
        else
        {
            // For shape < 1: Gamma(a) = Gamma(a+1) * U^(1/a)
            double g = SampleGamma(shape + 1.0);
            return g * Math.Pow(Math.Max(1e-10, RandomGenerator.NextDouble()), 1.0 / shape);
        }
    }

    /// <summary>
    /// Computes variance of gradient directions as alignment penalty.
    /// </summary>
    private double ComputeGradientAlignmentPenalty(List<Vector<T>> gradients)
    {
        if (gradients.Count <= 1) return 0;

        // Normalize gradients to unit vectors and compute variance of directions
        var normalized = new List<double[]>();
        foreach (var g in gradients)
        {
            double norm = 0;
            for (int d = 0; d < _paramDim; d++)
                norm += NumOps.ToDouble(g[d]) * NumOps.ToDouble(g[d]);
            norm = Math.Sqrt(norm) + 1e-10;

            var n = new double[_paramDim];
            for (int d = 0; d < _paramDim; d++)
                n[d] = NumOps.ToDouble(g[d]) / norm;
            normalized.Add(n);
        }

        // Compute per-dimension variance of normalized gradients
        double totalVar = 0;
        for (int d = 0; d < _paramDim; d++)
        {
            double mean = 0;
            foreach (var n in normalized) mean += n[d];
            mean /= normalized.Count;
            double var_d = 0;
            foreach (var n in normalized) var_d += (n[d] - mean) * (n[d] - mean);
            totalVar += var_d / normalized.Count;
        }

        return _algoOptions.AlignmentWeight * totalVar;
    }

    private double ComputeAlignmentGradient(List<Vector<T>> gradients, int dim)
    {
        if (gradients.Count <= 1) return 0;
        double mean = 0;
        foreach (var g in gradients) mean += NumOps.ToDouble(g[dim]);
        mean /= gradients.Count;
        double gradVar = 0;
        foreach (var g in gradients) gradVar += 2.0 * (NumOps.ToDouble(g[dim]) - mean) / gradients.Count;
        return gradVar;
    }
}
