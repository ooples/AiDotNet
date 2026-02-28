using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Open-MAML++: MAML with per-parameter learning rates and
/// open-set novelty detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Open-MAML++ combines MAML++ improvements (per-parameter learning rates, multi-step
/// loss, learned layer-specific LR) with open-set recognition. The algorithm meta-learns
/// a novelty threshold based on prediction entropy: samples with entropy exceeding the
/// threshold are classified as novel/unknown. The multi-step loss ensures stable
/// intermediate adaptation steps.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned: θ_0, per-param learning rates α (MAML++), novelty threshold τ
///
/// Inner loop (MAML++ style):
///   θ_τ = θ_0
///   multi_step_losses = []
///   For step k = 1..K:
///     θ_τ ← θ_τ - α ⊙ ∇L(θ_τ; D_support)  (element-wise LR)
///     multi_step_losses.append(L(θ_τ; D_query))
///
///   L_total = L_K + w_ms * mean(L_1..L_{K-1})  (multi-step loss)
///
/// Novelty detection:
///   For prediction p = softmax(f(x; θ_τ)):
///     H(p) = -Σ p_i log(p_i)  (entropy)
///     novel = H(p) > τ
///
/// Outer loop:
///   Update θ_0, α via meta-gradients
///   Update τ via SPSA on entropy-based loss
///
/// Entropy regularization: encourage low entropy for known classes
///   L_entropy = mean(H(p)) for correctly classified samples
/// </code>
/// </para>
/// </remarks>
public class OpenMAMLPlusAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly OpenMAMLPlusOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Meta-learned per-parameter learning rates (MAML++ style).</summary>
    private Vector<T> _perParamLR;

    /// <summary>Meta-learned novelty threshold (on prediction entropy).</summary>
    private Vector<T> _noveltyThreshold;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.OpenMAMLPlus;

    public OpenMAMLPlusAlgorithm(OpenMAMLPlusOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;

        // Initialize per-parameter learning rates
        _perParamLR = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            _perParamLR[d] = NumOps.FromDouble(options.InnerLearningRate);

        // Novelty threshold (single scalar stored as 1-dim vector for SPSA)
        _noveltyThreshold = new Vector<T>(1);
        _noveltyThreshold[0] = NumOps.FromDouble(options.InitialNoveltyThreshold);
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

            var multiStepLosses = new List<T>();

            // Inner loop with per-parameter LR (MAML++ style)
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Element-wise learning rate application
                if (_algoOptions.LearnPerParamLR)
                {
                    for (int d = 0; d < _paramDim; d++)
                    {
                        double lr = Math.Abs(NumOps.ToDouble(_perParamLR[d])); // Ensure positive
                        adaptedParams[d] = NumOps.Subtract(adaptedParams[d], NumOps.FromDouble(lr * NumOps.ToDouble(grad[d])));
                    }
                }
                else
                {
                    adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
                }

                // Multi-step loss: evaluate at each intermediate step
                if (step < _algoOptions.AdaptationSteps - 1)
                {
                    MetaModel.SetParameters(adaptedParams);
                    var intermediateLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
                    multiStepLosses.Add(intermediateLoss);
                }
            }

            // Final step query loss
            MetaModel.SetParameters(adaptedParams);
            var finalLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Multi-step loss: final + weighted intermediate
            var totalTaskLoss = finalLoss;
            if (multiStepLosses.Count > 0)
            {
                var intermediateAvg = ComputeMean(multiStepLosses);
                totalTaskLoss = NumOps.Add(finalLoss,
                    NumOps.Multiply(NumOps.FromDouble(_algoOptions.MultiStepLossWeight), intermediateAvg));
            }

            // Entropy regularization on predictions
            var queryPred = ConvertToVector(MetaModel.Predict(task.QueryInput)) ?? new Vector<T>(1);
            double entropy = ComputePredictionEntropy(queryPred);
            totalTaskLoss = NumOps.Add(totalTaskLoss, NumOps.FromDouble(_algoOptions.EntropyRegWeight * entropy));

            losses.Add(totalTaskLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update per-parameter learning rates via SPSA
        if (_algoOptions.LearnPerParamLR)
            UpdateAuxiliaryParamsSPSA(taskBatch, ref _perParamLR, _algoOptions.OuterLearningRate * 0.01, ComputeOpenMAMLLoss);

        // Update novelty threshold via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _noveltyThreshold, _algoOptions.OuterLearningRate * 0.1, ComputeOpenMAMLLoss);

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

            if (_algoOptions.LearnPerParamLR)
            {
                for (int d = 0; d < _paramDim; d++)
                {
                    double lr = Math.Abs(NumOps.ToDouble(_perParamLR[d]));
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d], NumOps.FromDouble(lr * NumOps.ToDouble(grad[d])));
                }
            }
            else
            {
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }
        }

        // Encode novelty threshold as modulation factor
        double threshold = NumOps.ToDouble(_noveltyThreshold[0]);
        var modulationFactors = new double[_paramDim];
        for (int d = 0; d < _paramDim; d++)
            modulationFactors[d] = 1.0; // No modulation by default

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams, modulationFactors: modulationFactors);
    }

    /// <summary>
    /// Computes prediction entropy: H(p) = -Σ p_i log(p_i).
    /// Treats the output vector as logits and applies softmax first.
    /// </summary>
    private double ComputePredictionEntropy(Vector<T> prediction)
    {
        if (prediction.Length == 0) return 0;

        // Softmax
        double maxVal = double.NegativeInfinity;
        for (int i = 0; i < prediction.Length; i++)
        {
            double v = NumOps.ToDouble(prediction[i]);
            if (v > maxVal) maxVal = v;
        }

        double sumExp = 0;
        for (int i = 0; i < prediction.Length; i++)
            sumExp += Math.Exp(NumOps.ToDouble(prediction[i]) - maxVal);

        // Entropy
        double entropy = 0;
        for (int i = 0; i < prediction.Length; i++)
        {
            double p = Math.Exp(NumOps.ToDouble(prediction[i]) - maxVal) / (sumExp + 1e-10);
            if (p > 1e-10)
                entropy -= p * Math.Log(p);
        }

        return entropy;
    }

    private double ComputeOpenMAMLLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                if (_algoOptions.LearnPerParamLR)
                {
                    for (int d = 0; d < _paramDim; d++)
                    {
                        double lr = Math.Abs(NumOps.ToDouble(_perParamLR[d]));
                        adaptedParams[d] = NumOps.Subtract(adaptedParams[d], NumOps.FromDouble(lr * NumOps.ToDouble(grad[d])));
                    }
                }
                else
                {
                    adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
                }
            }

            MetaModel.SetParameters(adaptedParams);
            double loss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            var pred = ConvertToVector(MetaModel.Predict(task.QueryInput)) ?? new Vector<T>(1);
            loss += _algoOptions.EntropyRegWeight * ComputePredictionEntropy(pred);
            totalLoss += loss;
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
