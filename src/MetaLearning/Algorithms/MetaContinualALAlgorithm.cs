using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of MetaContinualAL: Meta-Continual Active Learning with uncertainty-guided
/// parameter-selective adaptation.
/// </summary>
/// <remarks>
/// <para>
/// MetaContinualAL uses gradient-norm-based uncertainty estimation to identify the most
/// informative parameter dimensions and focuses adaptation effort there. A running EMA
/// calibration tracks mean/variance of per-parameter gradient magnitudes. Parameters with
/// above-average uncertainty receive amplified learning rates, while low-uncertainty
/// parameters are dampened — similar to active learning's acquisition function but applied
/// in parameter space.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Per-parameter uncertainty: u_d = |∂L/∂θ_d|
/// Running stats: μ_d = EMA(μ_d, u_d), σ²_d = EMA(σ²_d, (u_d - μ_d)²)
///
/// Acquisition mask (top-f fraction by z-score):
///   z_d = (u_d - μ_d) / (σ_d + ε)
///   mask_d = 1 if z_d in top-f, else dampening_factor
///
/// Inner loop:
///   θ_d ← θ_d - η * mask_d * (grad_d + ExplorationBonus * σ_d * noise_d)
///
/// Outer loop: standard meta-gradient update
/// </code>
/// </para>
/// </remarks>
public class MetaContinualALAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaContinualALOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Running mean of per-parameter gradient magnitudes.</summary>
    private double[] _uncertaintyMean;

    /// <summary>Running variance of per-parameter gradient magnitudes.</summary>
    private double[] _uncertaintyVar;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaContinualAL;

    public MetaContinualALAlgorithm(MetaContinualALOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _uncertaintyMean = new double[_paramDim];
        _uncertaintyVar = new double[_paramDim];
        for (int d = 0; d < _paramDim; d++) _uncertaintyVar[d] = 1.0;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Compute per-parameter uncertainty (gradient magnitude)
                var uncertainty = new double[_paramDim];
                for (int d = 0; d < _paramDim; d++)
                    uncertainty[d] = Math.Abs(NumOps.ToDouble(grad[d]));

                // Update running statistics
                double decay = _algoOptions.UncertaintyDecay;
                for (int d = 0; d < _paramDim; d++)
                {
                    double diff = uncertainty[d] - _uncertaintyMean[d];
                    _uncertaintyMean[d] = decay * _uncertaintyMean[d] + (1.0 - decay) * uncertainty[d];
                    _uncertaintyVar[d] = decay * _uncertaintyVar[d] + (1.0 - decay) * diff * diff;
                }

                // Compute z-scores and find threshold for acquisition mask
                var zScores = new double[_paramDim];
                for (int d = 0; d < _paramDim; d++)
                    zScores[d] = (uncertainty[d] - _uncertaintyMean[d]) / (Math.Sqrt(_uncertaintyVar[d]) + 1e-10);

                // Find threshold: top-f fraction gets full learning rate
                var sorted = new double[_paramDim];
                Array.Copy(zScores, sorted, _paramDim);
                Array.Sort(sorted);
                int threshIdx = (int)((1.0 - _algoOptions.AcquisitionFraction) * _paramDim);
                if (threshIdx >= _paramDim) threshIdx = _paramDim - 1;
                double threshold = sorted[threshIdx];

                // Apply acquisition-weighted gradient
                for (int d = 0; d < _paramDim; d++)
                {
                    double mask = zScores[d] >= threshold ? 1.0 : 0.1;
                    double gradVal = NumOps.ToDouble(grad[d]);

                    // Exploration bonus: noise proportional to uncertainty
                    double noise = 0;
                    if (_algoOptions.ExplorationBonus > 0)
                    {
                        double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
                        double u2 = RandomGenerator.NextDouble();
                        noise = _algoOptions.ExplorationBonus * Math.Sqrt(_uncertaintyVar[d])
                              * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
                    }

                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * mask * (gradVal + noise)));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
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

            var uncertainty = new double[_paramDim];
            for (int d = 0; d < _paramDim; d++)
                uncertainty[d] = Math.Abs(NumOps.ToDouble(grad[d]));

            var zScores = new double[_paramDim];
            for (int d = 0; d < _paramDim; d++)
                zScores[d] = (uncertainty[d] - _uncertaintyMean[d]) / (Math.Sqrt(_uncertaintyVar[d]) + 1e-10);

            var sorted = new double[_paramDim];
            Array.Copy(zScores, sorted, _paramDim);
            Array.Sort(sorted);
            int threshIdx = (int)((1.0 - _algoOptions.AcquisitionFraction) * _paramDim);
            if (threshIdx >= _paramDim) threshIdx = _paramDim - 1;
            double threshold = sorted[threshIdx];

            for (int d = 0; d < _paramDim; d++)
            {
                double mask = zScores[d] >= threshold ? 1.0 : 0.1;
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * mask * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }
}
