using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of MOCA: Meta-learning with Online Complementary Augmentation.
/// </summary>
/// <remarks>
/// <para>
/// MOCA augments the meta-learning task distribution by generating complementary tasks
/// in gradient space. For each real task, it creates augmented versions by perturbing the
/// gradient with a direction orthogonal to the original gradient, scaled by historical
/// gradient statistics. This explores complementary adaptation directions that improve
/// generalization and robustness of the meta-learned initialization.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Gradient history: g_mean = EMA(g_mean, grad), g_var = EMA(g_var, (grad - g_mean)²)
///
/// Complementary perturbation:
///   noise ~ N(0, I)
///   proj = noise - (noise · grad / ||grad||²) * grad   (Gram-Schmidt: orthogonal component)
///   perturbation = AugmentationStrength * sqrt(g_var) * normalize(proj)
///
/// Augmented gradient: grad_aug = grad + perturbation
///
/// Inner loop (original): θ' = θ - η * grad
/// Inner loop (augmented): θ'' = θ - η * grad_aug
///
/// L_meta = L_query(θ') + ComplementaryWeight * L_query(θ'')
/// Outer loop: update θ with combined meta-gradient
/// </code>
/// </para>
/// </remarks>
public class MOCAAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MOCAOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Running mean of gradients across tasks (EMA).</summary>
    private double[] _gradMean;

    /// <summary>Running variance of gradients across tasks (EMA).</summary>
    private double[] _gradVar;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MOCA;

    public MOCAAlgorithm(MOCAOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _gradMean = new double[_paramDim];
        _gradVar = new double[_paramDim];
        for (int d = 0; d < _paramDim; d++) _gradVar[d] = 1.0;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // === Original task adaptation ===
            var adaptedOriginal = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedOriginal[d] = initParams[d];

            Vector<T>? lastGrad = null;
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedOriginal);
                lastGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedOriginal = ApplyGradients(adaptedOriginal, lastGrad, _algoOptions.InnerLearningRate);
            }

            // Update gradient history
            if (lastGrad != null)
            {
                double momentum = _algoOptions.HistoryMomentum;
                for (int d = 0; d < _paramDim; d++)
                {
                    double gVal = NumOps.ToDouble(lastGrad[d]);
                    double diff = gVal - _gradMean[d];
                    _gradMean[d] = momentum * _gradMean[d] + (1.0 - momentum) * gVal;
                    _gradVar[d] = momentum * _gradVar[d] + (1.0 - momentum) * diff * diff;
                }
            }

            MetaModel.SetParameters(adaptedOriginal);
            var originalLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // === Augmented task adaptation(s) ===
            var augLosses = new List<T>();
            for (int aug = 0; aug < _algoOptions.NumAugmentedTasks; aug++)
            {
                var adaptedAug = new Vector<T>(_paramDim);
                for (int d = 0; d < _paramDim; d++) adaptedAug[d] = initParams[d];

                for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
                {
                    MetaModel.SetParameters(adaptedAug);
                    var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                    // Create complementary perturbation orthogonal to gradient
                    var augGrad = CreateAugmentedGradient(grad);
                    adaptedAug = ApplyGradients(adaptedAug, augGrad, _algoOptions.InnerLearningRate);
                }

                MetaModel.SetParameters(adaptedAug);
                augLosses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            }

            // Combine original and augmented losses
            T totalLoss = originalLoss;
            if (augLosses.Count > 0)
            {
                T augMean = ComputeMean(augLosses);
                totalLoss = NumOps.Add(originalLoss,
                    NumOps.Multiply(NumOps.FromDouble(_algoOptions.ComplementaryWeight), augMean));
            }

            losses.Add(totalLoss);
            MetaModel.SetParameters(adaptedOriginal);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
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

    /// <summary>
    /// Creates an augmented gradient by adding a complementary perturbation orthogonal to
    /// the original gradient direction, scaled by historical variance.
    /// </summary>
    private Vector<T> CreateAugmentedGradient(Vector<T> grad)
    {
        var augGrad = new Vector<T>(_paramDim);

        // Generate random noise
        var noise = new double[_paramDim];
        for (int d = 0; d < _paramDim; d++)
        {
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            noise[d] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }

        // Gram-Schmidt: project out the gradient direction to get orthogonal component
        double gradNormSq = 0;
        double dotProduct = 0;
        for (int d = 0; d < _paramDim; d++)
        {
            double gVal = NumOps.ToDouble(grad[d]);
            gradNormSq += gVal * gVal;
            dotProduct += noise[d] * gVal;
        }

        if (gradNormSq > 1e-10)
        {
            double scale = dotProduct / gradNormSq;
            for (int d = 0; d < _paramDim; d++)
                noise[d] -= scale * NumOps.ToDouble(grad[d]);
        }

        // Normalize the orthogonal component
        double noiseNorm = 0;
        for (int d = 0; d < _paramDim; d++) noiseNorm += noise[d] * noise[d];
        noiseNorm = Math.Sqrt(noiseNorm) + 1e-10;

        // Scale by augmentation strength and historical variance
        for (int d = 0; d < _paramDim; d++)
        {
            double perturbation = _algoOptions.AugmentationStrength
                                * Math.Sqrt(_gradVar[d] + 1e-10)
                                * noise[d] / noiseNorm;
            augGrad[d] = NumOps.Add(grad[d], NumOps.FromDouble(perturbation));
        }

        return augGrad;
    }
}
