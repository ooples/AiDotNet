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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Bayesian Model-Agnostic Meta-Learning",
    "https://arxiv.org/abs/1806.03836",
    Year = 2018,
    Authors = "Jaesik Yoon, Taesup Kim, Ousmane Dia, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class BMAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly BMAMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.BMAML;

    public BMAMLAlgorithm(BMAMLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters. BMAML requires a model with at least one parameter.");
        if (options.NumParticles <= 0)
            throw new ArgumentException("NumParticles must be positive.", nameof(options));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        T innerLearningRate = NumOps.FromDouble(_algoOptions.InnerLearningRate);

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
                    ParamModel.SetParameters(particles[i]);
                    taskGrads[i] = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                }

                // Compute kernel bandwidth via median heuristic
                double bandwidth = ComputeMedianBandwidth(particles);

                // SVGD update for each particle
                for (int i = 0; i < _algoOptions.NumParticles; i++)
                {
                    var svgdUpdate = ComputeSVGDUpdate(particles, taskGrads, i, bandwidth);
                    newParticles[i] = Engine.Add(particles[i], Engine.Multiply(svgdUpdate, innerLearningRate));
                }

                particles = newParticles;
            }

            // Compute ensemble query loss: (1/M) Σ_i L(θ_i; D_q)
            var taskLosses = new List<T>();
            var taskMetaGrads = new List<Vector<T>>();
            for (int i = 0; i < _algoOptions.NumParticles; i++)
            {
                ParamModel.SetParameters(particles[i]);
                var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
                taskLosses.Add(queryLoss);
                taskMetaGrads.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
            }

            losses.Add(ComputeMean(taskLosses));
            metaGradients.Add(AverageVectors(taskMetaGrads));
        }

        // Outer loop: update meta-parameters
        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var particles = InitializeParticles(initParams);
        T innerLearningRate = NumOps.FromDouble(_algoOptions.InnerLearningRate);

        // SVGD inner loop
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var newParticles = new Vector<T>[_algoOptions.NumParticles];

            var taskGrads = new Vector<T>[_algoOptions.NumParticles];
            for (int i = 0; i < _algoOptions.NumParticles; i++)
            {
                ParamModel.SetParameters(particles[i]);
                taskGrads[i] = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            }

            double bandwidth = ComputeMedianBandwidth(particles);

            for (int i = 0; i < _algoOptions.NumParticles; i++)
            {
                var svgdUpdate = ComputeSVGDUpdate(particles, taskGrads, i, bandwidth);
                newParticles[i] = Engine.Add(particles[i], Engine.Multiply(svgdUpdate, innerLearningRate));
            }

            particles = newParticles;
        }

        // Return ensemble mean as adapted parameters
        var ensembleMean = AverageVectors(particles.ToList());
        ParamModel.SetParameters(initParams);
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
                double noise = SampleNormal();
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

        // All pairwise squared distances [m, m] from the broadcast difference of the stacked particles.
        var stacked = StackRows(particles);
        var shape = new[] { m, m, _paramDim };
        var diff = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(Engine.Reshape(stacked, new[] { 1, m, _paramDim }), shape),
            Engine.TensorBroadcastTo(Engine.Reshape(stacked, new[] { m, 1, _paramDim }), shape));
        var distances = Engine.Reshape(Engine.ReduceSum(Engine.TensorSquare(diff), new[] { 2 }, false), new[] { m * m });

        // The strict upper triangle (i < j), in row order, then the median as the sorted list's middle entry.
        var upper = new List<int>(m * (m - 1) / 2);
        for (int i = 0; i < m; i++)
            for (int j = i + 1; j < m; j++)
                upper.Add(i * m + j);

        var pairwise = Engine.TensorIndexSelect(distances, new Tensor<int>(upper.ToArray(), new[] { upper.Count }), 0);
        var sorted = Engine.TensorSort(pairwise, 0, false).Values;

        // The bandwidth is a scalar kernel setting, so the median leaves T once here.
        double median = NumOps.ToDouble(sorted[upper.Count / 2]);
        double h2 = median / Math.Max(Math.Log(m), 1.0);
        return Math.Max(h2, 1e-10);
    }

    /// <summary>Stacks equal-length vectors as the rows of an [n, length] tensor.</summary>
    private Tensor<T> StackRows(Vector<T>[] rows)
    {
        return Engine.TensorStack(rows.Select(row => Tensor<T>.FromVector(row)).ToArray(), 0);
    }

    /// <summary>
    /// Computes the SVGD update direction for particle i:
    /// φ(θ_i) = (1/M) Σ_j [k(θ_j, θ_i) * ∇ log p(D|θ_j) + α * ∇_{θ_j} k(θ_j, θ_i)]
    /// </summary>
    private Vector<T> ComputeSVGDUpdate(Vector<T>[] particles, Vector<T>[] taskGrads, int i, double bandwidth)
    {
        int m = particles.Length;
        var shape = new[] { m, _paramDim };

        // diff_j = theta_j - theta_i; RBF kernel k_j = exp(-||diff_j||^2 / (2h^2))
        var stacked = StackRows(particles);
        var diff = Engine.TensorSubtract(stacked,
            Engine.TensorBroadcastTo(Engine.Reshape(Tensor<T>.FromVector(particles[i]), new[] { 1, _paramDim }), shape));
        var dist2 = Engine.ReduceSum(Engine.TensorSquare(diff), new[] { 1 }, true);
        var kernel = Engine.TensorExp(Engine.TensorMultiplyScalar(dist2, NumOps.FromDouble(-1.0 / (2.0 * bandwidth))));

        // phi = (1/m) sum_j k_j [-grad_j + w (theta_i - theta_j) / h^2] = -(1/m) k^T (G + (w / h^2) diff):
        // the attractive term follows -grad L (= grad log p), the repulsive term is grad_{theta_j} k(theta_j, theta_i).
        var terms = Engine.TensorAdd(StackRows(taskGrads),
            Engine.TensorMultiplyScalar(diff, NumOps.FromDouble(_algoOptions.SVGDRepulsiveWeight / bandwidth)));
        var weighted = Engine.TensorMatMul(Engine.TensorTranspose(kernel), terms);
        return Engine.TensorMultiplyScalar(weighted, NumOps.FromDouble(-1.0 / m)).ToVector();
    }
}
