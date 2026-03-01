using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of DREAM: Directed REward Augmented Meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// DREAM meta-learns a reward shaping function that transforms the raw task loss into a more
/// informative signal for the inner loop. The reward shaper is a small MLP that takes
/// (loss, gradient_norm, step/K) as input and outputs a scalar multiplier for the gradient.
/// This enables curriculum-like adaptation where early steps may be scaled differently
/// than later steps.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Reward shaper: R(loss, ||grad||, step/K) → scaling factor
///   R = MLP with hidden_dim, tanh activation, sigmoid output
///
/// Inner loop:
///   For step k:
///     grad = ∂L/∂θ
///     scale = R(L, ||grad||, k/K)
///     θ ← θ - η * scale * grad + discount^k * shaped_bonus
///
/// Outer loop:
///   L_meta = L_query + w_shape * Σ_k discount^k * |1 - scale_k|
///   Update θ_0 and R jointly
/// </code>
/// </para>
/// </remarks>
public class DREAMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly DREAMOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Reward shaper parameters: 3-input → hidden → 1-output MLP.</summary>
    private Vector<T> _shaperParams;

    private readonly int _hiddenDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DREAM;

    public DREAMAlgorithm(DREAMOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _hiddenDim = options.RewardShaperHiddenDim;

        // Shaper MLP: input(3) → hidden → output(1)
        int shaperSize = 3 * _hiddenDim + _hiddenDim; // weights + output weights
        _shaperParams = new Vector<T>(shaperSize);
        for (int i = 0; i < shaperSize; i++)
        {
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            _shaperParams[i] = NumOps.FromDouble(0.1 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
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

            double shapingPenalty = 0;
            int K = _algoOptions.AdaptationSteps;

            for (int step = 0; step < K; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                double loss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput));

                // Compute gradient norm
                double gradNorm = 0;
                for (int d = 0; d < _paramDim; d++)
                    gradNorm += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(grad[d]);
                gradNorm = Math.Sqrt(gradNorm);

                // Reward shaper: scale factor
                double progress = (double)step / Math.Max(K - 1, 1);
                double scale = ComputeRewardScale(loss, gradNorm, progress);

                // Apply scaled gradient
                for (int d = 0; d < _paramDim; d++)
                {
                    double scaledGrad = scale * NumOps.ToDouble(grad[d]);
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * scaledGrad));
                }

                // Shaping penalty: deviation from unit scale
                double discountFactor = Math.Pow(_algoOptions.ShapingDiscount, step);
                shapingPenalty += discountFactor * Math.Abs(1.0 - scale);
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.RewardShapingWeight * shapingPenalty));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _shaperParams, _algoOptions.OuterLearningRate * 0.1, ComputeDREAMLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        int K = _algoOptions.AdaptationSteps;
        for (int step = 0; step < K; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            double loss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput));

            double gradNorm = 0;
            for (int d = 0; d < _paramDim; d++)
                gradNorm += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(grad[d]);
            gradNorm = Math.Sqrt(gradNorm);

            double progress = (double)step / Math.Max(K - 1, 1);
            double scale = ComputeRewardScale(loss, gradNorm, progress);

            for (int d = 0; d < _paramDim; d++)
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * scale * NumOps.ToDouble(grad[d])));
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Computes the reward scaling factor using the shaper MLP:
    /// input = (loss, grad_norm, progress) → hidden(tanh) → sigmoid(output)
    /// </summary>
    private double ComputeRewardScale(double loss, double gradNorm, double progress)
    {
        var input = new double[] { loss, gradNorm, progress };
        var hidden = new double[_hiddenDim];
        int offset = 0;

        // Layer 1: input(3) → hidden (tanh)
        for (int h = 0; h < _hiddenDim; h++)
        {
            double sum = 0;
            for (int i = 0; i < 3; i++)
                sum += input[i] * NumOps.ToDouble(_shaperParams[offset + h * 3 + i]);
            hidden[h] = Math.Tanh(sum);
        }
        offset += 3 * _hiddenDim;

        // Layer 2: hidden → scalar (sigmoid for [0, 2] range)
        double output = 0;
        for (int h = 0; h < _hiddenDim; h++)
            output += hidden[h] * NumOps.ToDouble(_shaperParams[offset + h]);

        // Sigmoid scaled to [0, 2] — allows both dampening and amplification
        return 2.0 / (1.0 + Math.Exp(-output));
    }

    private double ComputeDREAMLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adapted = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adapted[d] = initParams[d];

            int K = _algoOptions.AdaptationSteps;
            for (int step = 0; step < K; step++)
            {
                MetaModel.SetParameters(adapted);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                double loss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput));
                double gn = 0;
                for (int d = 0; d < _paramDim; d++) gn += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(grad[d]);
                double scale = ComputeRewardScale(loss, Math.Sqrt(gn), (double)step / Math.Max(K - 1, 1));
                for (int d = 0; d < _paramDim; d++)
                    adapted[d] = NumOps.Subtract(adapted[d], NumOps.FromDouble(_algoOptions.InnerLearningRate * scale * NumOps.ToDouble(grad[d])));
            }

            MetaModel.SetParameters(adapted);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
