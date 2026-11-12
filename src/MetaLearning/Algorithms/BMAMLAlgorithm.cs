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

        // Retain only the strict upper triangle; parameter differences use bounded blocks.
        var pairwise = new double[checked((int)((long)m * (m - 1) / 2))];
        int pair = 0;
        for (int i = 0; i < m; i++)
            for (int j = i + 1; j < m; j++)
                pairwise[pair++] = SquaredParticleDistance(particles[i], particles[j]);
        Array.Sort(pairwise);
        double h2 = pairwise[pairwise.Length / 2] / Math.Max(Math.Log(m), 1.0);
        return Math.Max(h2, 1e-10);
    }

    private const int ParameterBlockSize = 16384;

    private Tensor<double> ParticleBlock(Vector<T> particle, int start, int count)
        => Tensor<T>.FromVector(particle.GetSubVector(start, count)).Cast<double>();

    private double SquaredParticleDistance(Vector<T> left, Vector<T> right)
    {
        double distance = 0;
        for (int start = 0; start < _paramDim;)
        {
            int count = Math.Min(ParameterBlockSize, _paramDim - start);
            var difference = Engine.TensorSubtract(ParticleBlock(left, start, count), ParticleBlock(right, start, count));
            distance += Engine.TensorSum(Engine.TensorSquare(difference));
            start += count;
        }
        return distance;
    }

    /// <summary>Computes attractive and repulsive SVGD terms without stacking particle parameters.</summary>
    private Vector<T> ComputeSVGDUpdate(Vector<T>[] particles, Vector<T>[] taskGrads, int i, double bandwidth)
    {
        int m = particles.Length;
        var kernels = new double[m];
        for (int j = 0; j < m; j++)
            kernels[j] = Math.Exp(-SquaredParticleDistance(particles[j], particles[i]) / (2.0 * bandwidth));

        var update = new Vector<T>(_paramDim);
        for (int start = 0; start < _paramDim;)
        {
            int count = Math.Min(ParameterBlockSize, _paramDim - start);
            var target = ParticleBlock(particles[i], start, count);
            var sum = new Tensor<double>([count]);
            for (int j = 0; j < m; j++)
            {
                var difference = Engine.TensorSubtract(ParticleBlock(particles[j], start, count), target);
                var terms = Engine.TensorAdd(ParticleBlock(taskGrads[j], start, count),
                    Engine.TensorMultiplyScalar(difference, _algoOptions.SVGDRepulsiveWeight / bandwidth));
                sum = Engine.TensorAdd(sum, Engine.TensorMultiplyScalar(terms, kernels[j]));
            }
            Engine.TensorMultiplyScalar(sum, -1.0 / m).Cast<T>()
                .CopyTo(update.AsWritableSpan().Slice(start, count));
            start += count;
        }
        return update;
    }
}
