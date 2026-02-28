using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of FreqPrompt: Frequency-domain prompt tuning for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// FreqPrompt meta-learns a set of frequency basis vectors and their coefficients that
/// act as additive "prompts" in the parameter space. During adaptation, only the prompt
/// coefficients are updated (the backbone is frozen), which enables efficient adaptation
/// with very few parameters. Low-frequency prompts capture coarse domain shifts while
/// high-frequency prompts handle fine-grained adjustments.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned: backbone θ_base, prompt basis B (K × D), initial coefficients c_0 (K × 1)
///
/// Prompt reconstruction: prompt(c) = Σ_k c_k * B_k (linear combination of basis)
/// Effective parameters: θ = θ_base + prompt(c)
///
/// Inner loop (only adapts coefficients c):
///   c_τ = c_0
///   For each step:
///     grad_θ = ∂L(θ_base + prompt(c_τ))/∂θ
///     grad_c_k = ⟨grad_θ, B_k⟩  (project gradient onto basis)
///     reg_k = penalty * freq_weight(k) * c_k  (penalize high-freq more)
///     c_τ_k ← c_τ_k - η * (grad_c_k + reg_k)
///
/// Outer loop: update θ_base, B, c_0 via meta-gradients
/// </code>
/// </para>
/// </remarks>
public class FreqPromptAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly FreqPromptOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numComponents;

    /// <summary>Prompt basis vectors: flat array of K * paramDim values.</summary>
    private Vector<T> _promptBasis;

    /// <summary>Meta-learned initial prompt coefficients: length K.</summary>
    private Vector<T> _promptCoeffsInit;

    /// <summary>Per-frequency regularization weights (higher for high-freq).</summary>
    private readonly double[] _freqWeights;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.FreqPrompt;

    public FreqPromptAlgorithm(FreqPromptOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _numComponents = options.NumFreqComponents;

        // Initialize basis vectors using DCT-like cosine basis
        _promptBasis = new Vector<T>(_numComponents * _paramDim);
        for (int k = 0; k < _numComponents; k++)
            for (int d = 0; d < _paramDim; d++)
            {
                double basis = Math.Cos(Math.PI * (k + 1) * (2 * d + 1) / (2.0 * _paramDim));
                _promptBasis[k * _paramDim + d] = NumOps.FromDouble(options.PromptInitScale * basis);
            }

        // Initialize prompt coefficients to small values
        _promptCoeffsInit = new Vector<T>(_numComponents);
        for (int k = 0; k < _numComponents; k++)
            _promptCoeffsInit[k] = NumOps.FromDouble(options.PromptInitScale * 0.1);

        // Frequency-dependent regularization: low freq (k=0) gets least penalty
        _freqWeights = new double[_numComponents];
        for (int k = 0; k < _numComponents; k++)
            _freqWeights[k] = 1.0 + k * options.HighFreqPenalty;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Initialize task-specific coefficients from meta-learned init
            var coeffs = new double[_numComponents];
            for (int k = 0; k < _numComponents; k++)
                coeffs[k] = NumOps.ToDouble(_promptCoeffsInit[k]);

            // Inner loop: only adapt coefficients (backbone frozen)
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                var effectiveParams = ComputeEffectiveParams(baseParams, coeffs);
                MetaModel.SetParameters(effectiveParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Project gradient onto basis vectors to get coefficient gradients
                for (int k = 0; k < _numComponents; k++)
                {
                    double gradC = 0;
                    for (int d = 0; d < _paramDim; d++)
                        gradC += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(_promptBasis[k * _paramDim + d]);

                    // Add frequency-dependent regularization
                    double reg = _freqWeights[k] * _algoOptions.HighFreqPenalty * coeffs[k];
                    coeffs[k] -= _algoOptions.InnerLearningRate * (gradC + reg);
                }
            }

            // Evaluate with adapted prompt
            var finalParams = ComputeEffectiveParams(baseParams, coeffs);
            MetaModel.SetParameters(finalParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update backbone
        MetaModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var newBaseParams = ApplyGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate);
            MetaModel.SetParameters(newBaseParams);
        }

        // Update prompt basis and initial coefficients via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _promptBasis, _algoOptions.OuterLearningRate * 0.1, ComputeFreqPromptLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _promptCoeffsInit, _algoOptions.OuterLearningRate * 0.5, ComputeFreqPromptLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();

        var coeffs = new double[_numComponents];
        for (int k = 0; k < _numComponents; k++)
            coeffs[k] = NumOps.ToDouble(_promptCoeffsInit[k]);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var effectiveParams = ComputeEffectiveParams(baseParams, coeffs);
            MetaModel.SetParameters(effectiveParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            for (int k = 0; k < _numComponents; k++)
            {
                double gradC = 0;
                for (int d = 0; d < _paramDim; d++)
                    gradC += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(_promptBasis[k * _paramDim + d]);
                double reg = _freqWeights[k] * _algoOptions.HighFreqPenalty * coeffs[k];
                coeffs[k] -= _algoOptions.InnerLearningRate * (gradC + reg);
            }
        }

        var adaptedParams = ComputeEffectiveParams(baseParams, coeffs);
        MetaModel.SetParameters(baseParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Computes effective parameters: θ = θ_base + Σ_k c_k * B_k.
    /// </summary>
    private Vector<T> ComputeEffectiveParams(Vector<T> baseParams, double[] coeffs)
    {
        var result = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) result[d] = baseParams[d];

        for (int k = 0; k < _numComponents; k++)
        {
            for (int d = 0; d < _paramDim; d++)
            {
                double basisVal = NumOps.ToDouble(_promptBasis[k * _paramDim + d]);
                result[d] = NumOps.Add(result[d], NumOps.FromDouble(coeffs[k] * basisVal));
            }
        }

        return result;
    }

    private double ComputeFreqPromptLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var baseParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var coeffs = new double[_numComponents];
            for (int k = 0; k < _numComponents; k++)
                coeffs[k] = NumOps.ToDouble(_promptCoeffsInit[k]);

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                var effectiveParams = ComputeEffectiveParams(baseParams, coeffs);
                MetaModel.SetParameters(effectiveParams);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);

                for (int k = 0; k < _numComponents; k++)
                {
                    double gradC = 0;
                    for (int d = 0; d < _paramDim; d++)
                        gradC += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(_promptBasis[k * _paramDim + d]);
                    coeffs[k] -= _algoOptions.InnerLearningRate * gradC;
                }
            }

            var finalParams = ComputeEffectiveParams(baseParams, coeffs);
            MetaModel.SetParameters(finalParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
