using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of BMAML: Bayesian Model-Agnostic Meta-Learning
/// (Yoon et al., NeurIPS 2018).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// BMAML uses Stein Variational Gradient Descent (SVGD) to maintain a particle ensemble
/// {θ¹, ..., θᴹ} that approximates the posterior distribution over task-adapted parameters.
/// Each particle is updated using both the task loss gradient and a repulsive kernel term
/// that encourages diversity among particles.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Particles: {θ¹_τ, ..., θᴹ_τ} initialized near θ_0 (meta params)
///
/// Inner loop (SVGD adaptation for each task τ):
///   For each step:
///     For each particle i:
///       Attractive: a_i = (1/M) Σ_j k(θ_j, θ_i) * ∇_{θ_j} log p(D_s|θ_j)
///       Repulsive:  r_i = (1/M) Σ_j ∇_{θ_j} k(θ_j, θ_i)
///       θ_i ← θ_i + ε * (a_i + α * r_i)
///
///   Loss_τ = (1/M) Σ_i L(θ_i; D_q)
///
/// Outer loop:
///   θ_0 ← θ_0 - η * (1/T) Σ_τ ∇_{θ_0} Loss_τ
///
/// RBF kernel: k(θ_i, θ_j) = exp(-||θ_i - θ_j||²/(2h²))
/// Bandwidth h: median heuristic h² = median(pairwise distances²) / log(M)
/// </code>
/// </para>
/// </remarks>
public class BMAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly BMAMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.BMAML;

    public BMAMLAlgorithm(BMAMLOptions<T, TInput, TOutput> options)
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

        foreach (var task in taskBatch.Tasks)
        {
            // Initialize M particles near meta-parameters with Gaussian perturbation
            var particles = InitializeParticles(initParams);

            // Inner loop: SVGD adaptation
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                var newParticles = new Vector<T>[_algoOptions.NumParticles];

                // Compute task gradients for all particles
                var taskGrads = new Vector<T>[_algoOptions.NumParticles];
                for (int i = 0; i < _algoOptions.NumParticles; i++)
                {
                    MetaModel.SetParameters(particles[i]);
                    taskGrads[i] = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                }

                // Compute kernel bandwidth via median heuristic
                double bandwidth = ComputeMedianBandwidth(particles);

                // SVGD update for each particle
                for (int i = 0; i < _algoOptions.NumParticles; i++)
                {
                    var svgdUpdate = ComputeSVGDUpdate(particles, taskGrads, i, bandwidth);
                    newParticles[i] = new Vector<T>(_paramDim);
                    for (int d = 0; d < _paramDim; d++)
                    {
                        newParticles[i][d] = NumOps.Add(particles[i][d],
                            NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(svgdUpdate[d])));
                    }
                }

                particles = newParticles;
            }

            // Compute ensemble query loss: (1/M) Σ_i L(θ_i; D_q)
            var taskLosses = new List<T>();
            var taskMetaGrads = new List<Vector<T>>();
            for (int i = 0; i < _algoOptions.NumParticles; i++)
            {
                MetaModel.SetParameters(particles[i]);
                var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
                taskLosses.Add(queryLoss);
                taskMetaGrads.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
            }

            losses.Add(ComputeMean(taskLosses));
            metaGradients.Add(AverageVectors(taskMetaGrads));
        }

        // Outer loop: update meta-parameters
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var newParams = ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate);
            MetaModel.SetParameters(newParams);
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var particles = InitializeParticles(initParams);

        // SVGD inner loop
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var newParticles = new Vector<T>[_algoOptions.NumParticles];

            var taskGrads = new Vector<T>[_algoOptions.NumParticles];
            for (int i = 0; i < _algoOptions.NumParticles; i++)
            {
                MetaModel.SetParameters(particles[i]);
                taskGrads[i] = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            }

            double bandwidth = ComputeMedianBandwidth(particles);

            for (int i = 0; i < _algoOptions.NumParticles; i++)
            {
                var svgdUpdate = ComputeSVGDUpdate(particles, taskGrads, i, bandwidth);
                newParticles[i] = new Vector<T>(_paramDim);
                for (int d = 0; d < _paramDim; d++)
                {
                    newParticles[i][d] = NumOps.Add(particles[i][d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(svgdUpdate[d])));
                }
            }

            particles = newParticles;
        }

        // Return ensemble mean as adapted parameters
        var ensembleMean = AverageVectors(particles.ToList());
        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, ensembleMean);
    }

    /// <summary>
    /// Initializes M particles near the meta-parameters with Gaussian perturbation.
    /// </summary>
    private Vector<T>[] InitializeParticles(Vector<T> metaParams)
    {
        var particles = new Vector<T>[_algoOptions.NumParticles];
        for (int i = 0; i < _algoOptions.NumParticles; i++)
        {
            particles[i] = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
            {
                // Box-Muller for Gaussian noise
                double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
                double u2 = RandomGenerator.NextDouble();
                double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                particles[i][d] = NumOps.Add(metaParams[d],
                    NumOps.FromDouble(_algoOptions.ParticleInitScale * noise));
            }
        }
        return particles;
    }

    /// <summary>
    /// Computes the RBF kernel bandwidth using the median heuristic:
    /// h² = median(pairwise squared distances) / log(M).
    /// </summary>
    private double ComputeMedianBandwidth(Vector<T>[] particles)
    {
        if (_algoOptions.KernelBandwidth.HasValue)
            return _algoOptions.KernelBandwidth.Value;

        int m = particles.Length;
        if (m <= 1) return 1.0;

        var pairwiseDists = new List<double>();
        for (int i = 0; i < m; i++)
            for (int j = i + 1; j < m; j++)
            {
                double dist2 = 0;
                for (int d = 0; d < _paramDim; d++)
                {
                    double diff = NumOps.ToDouble(particles[i][d]) - NumOps.ToDouble(particles[j][d]);
                    dist2 += diff * diff;
                }
                pairwiseDists.Add(dist2);
            }

        if (pairwiseDists.Count == 0) return 1.0;
        pairwiseDists.Sort();
        double median = pairwiseDists[pairwiseDists.Count / 2];
        double h2 = median / Math.Max(Math.Log(m), 1.0);
        return Math.Max(h2, 1e-10);
    }

    /// <summary>
    /// Computes the SVGD update direction for particle i:
    /// φ(θ_i) = (1/M) Σ_j [k(θ_j, θ_i) * ∇ log p(D|θ_j) + α * ∇_{θ_j} k(θ_j, θ_i)]
    /// </summary>
    private Vector<T> ComputeSVGDUpdate(Vector<T>[] particles, Vector<T>[] taskGrads, int i, double bandwidth)
    {
        int m = particles.Length;
        var update = new Vector<T>(_paramDim);

        for (int j = 0; j < m; j++)
        {
            // Compute RBF kernel k(θ_j, θ_i) = exp(-||θ_j - θ_i||² / (2h²))
            double dist2 = 0;
            for (int d = 0; d < _paramDim; d++)
            {
                double diff = NumOps.ToDouble(particles[j][d]) - NumOps.ToDouble(particles[i][d]);
                dist2 += diff * diff;
            }
            double kernel = Math.Exp(-dist2 / (2.0 * bandwidth));

            for (int d = 0; d < _paramDim; d++)
            {
                // Attractive term: k(θ_j, θ_i) * (-∇L(θ_j)) [negative gradient = ∇ log p]
                double attractive = kernel * (-NumOps.ToDouble(taskGrads[j][d]));

                // Repulsive term: ∇_{θ_j} k(θ_j, θ_i) = k * (θ_i - θ_j) / h²
                double diffJI = NumOps.ToDouble(particles[i][d]) - NumOps.ToDouble(particles[j][d]);
                double repulsive = kernel * diffJI / bandwidth;

                update[d] = NumOps.Add(update[d],
                    NumOps.FromDouble((attractive + _algoOptions.SVGDRepulsiveWeight * repulsive) / m));
            }
        }

        return update;
    }
}
