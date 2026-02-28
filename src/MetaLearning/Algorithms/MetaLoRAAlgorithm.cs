using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-LoRA: Low-Rank Adaptation for Meta-Learning (2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-LoRA applies Low-Rank Adaptation (LoRA) to the meta-learning inner loop.
/// Instead of adapting all d model parameters per task (as in MAML), it meta-learns
/// r low-rank basis vectors {v_1, ..., v_r} in parameter space and only adapts
/// r scalar coefficients {c_1, ..., c_r} during the inner loop.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: base parameters θ, basis vectors {v_i}, initial coefficients {c_i^0}
///
/// Inner loop (per task):
///   1. Initialize c = c^0
///   2. For each adaptation step:
///      a. Compute adapted params: θ' = θ + (α/r) * Σ c_i * v_i
///      b. Evaluate loss on support set with θ'
///      c. Compute gradient ∂L/∂c_j = (α/r) * ⟨∂L/∂θ', v_j⟩  (project full gradient onto basis)
///      d. Update c_j ← c_j - η_inner * ∂L/∂c_j
///   3. Evaluate query loss with final adapted params
///
/// Outer loop:
///   - Update θ using meta-gradients from query losses
///   - Update basis vectors {v_i} and initial coefficients {c_i^0} via SPSA
/// </code>
/// </para>
/// <para><b>Advantages over MAML:</b>
/// Inner loop only adapts r parameters instead of d, making adaptation much cheaper
/// for large models. The low-rank constraint also acts as an implicit regularizer,
/// preventing overfitting on few-shot support sets.
/// </para>
/// </remarks>
public class MetaLoRAAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaLoRAOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// Low-rank basis vectors stored as a flat vector of length rank * paramDim.
    /// Basis vector i occupies indices [i*paramDim, (i+1)*paramDim).
    /// </summary>
    private Vector<T> _loraBasis;

    /// <summary>
    /// Meta-learned initial coefficients for the low-rank basis (length = rank).
    /// </summary>
    private Vector<T> _loraCoeffInit;

    private readonly int _paramDim;
    private readonly int _rank;
    private readonly double _scalingFactor;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaLoRA;

    public MetaLoRAAlgorithm(MetaLoRAOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _rank = Math.Max(1, options.Rank);
        _scalingFactor = options.ScalingAlpha / _rank;

        // Initialize basis vectors with small random values (Kaiming-like initialization)
        _loraBasis = new Vector<T>(_rank * _paramDim);
        double initStd = options.BasisInitStdDev;
        for (int i = 0; i < _loraBasis.Length; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            _loraBasis[i] = NumOps.FromDouble(z * initStd);
        }

        // Initialize coefficients to zero (no initial offset)
        _loraCoeffInit = new Vector<T>(_rank);
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
            var coeffs = new double[_rank];
            for (int i = 0; i < _rank; i++)
                coeffs[i] = NumOps.ToDouble(_loraCoeffInit[i]);

            // Inner loop: adapt coefficients on support set
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                // Compute adapted params: θ' = θ + scaling * Σ c_i * v_i
                var adaptedParams = ComputeAdaptedParams(baseParams, coeffs);
                MetaModel.SetParameters(adaptedParams);

                // Compute full gradient ∂L/∂θ' on support set
                var fullGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Project gradient onto each basis vector: ∂L/∂c_j = scaling * ⟨∂L/∂θ', v_j⟩
                for (int j = 0; j < _rank; j++)
                {
                    double gradCoeff = _scalingFactor * DotProductWithBasis(fullGrad, j);
                    coeffs[j] -= _algoOptions.InnerLearningRate * gradCoeff;
                }
            }

            // Evaluate on query set with final adapted params
            var finalAdapted = ComputeAdaptedParams(baseParams, coeffs);
            MetaModel.SetParameters(finalAdapted);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            // Compute meta-gradient for base parameters
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update base parameters
        MetaModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update basis vectors and initial coefficients via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _loraBasis, _algoOptions.OuterLearningRate, ComputeLoRALoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _loraCoeffInit, _algoOptions.OuterLearningRate, ComputeLoRALoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();

        // Initialize task-specific coefficients from meta-learned init
        var coeffs = new double[_rank];
        for (int i = 0; i < _rank; i++)
            coeffs[i] = NumOps.ToDouble(_loraCoeffInit[i]);

        // Inner loop: adapt coefficients on support set
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var adaptedParams = ComputeAdaptedParams(baseParams, coeffs);
            MetaModel.SetParameters(adaptedParams);

            var fullGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            for (int j = 0; j < _rank; j++)
            {
                double gradCoeff = _scalingFactor * DotProductWithBasis(fullGrad, j);
                coeffs[j] -= _algoOptions.InnerLearningRate * gradCoeff;
            }
        }

        var finalParams = ComputeAdaptedParams(baseParams, coeffs);
        MetaModel.SetParameters(baseParams); // Restore base params
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, finalParams);
    }

    /// <summary>
    /// Computes adapted parameters: θ' = θ_base + (α/r) * Σ c_i * v_i.
    /// </summary>
    private Vector<T> ComputeAdaptedParams(Vector<T> baseParams, double[] coeffs)
    {
        var adapted = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            adapted[d] = baseParams[d];

        for (int i = 0; i < _rank; i++)
        {
            double scaledCoeff = _scalingFactor * coeffs[i];
            if (Math.Abs(scaledCoeff) < 1e-15) continue;

            int basisOffset = i * _paramDim;
            T scale = NumOps.FromDouble(scaledCoeff);
            for (int d = 0; d < _paramDim; d++)
                adapted[d] = NumOps.Add(adapted[d], NumOps.Multiply(_loraBasis[basisOffset + d], scale));
        }

        return adapted;
    }

    /// <summary>
    /// Computes dot product of a gradient vector with the i-th basis vector.
    /// </summary>
    private double DotProductWithBasis(Vector<T> gradient, int basisIndex)
    {
        double dot = 0;
        int offset = basisIndex * _paramDim;
        int len = Math.Min(_paramDim, gradient.Length);
        for (int d = 0; d < len; d++)
            dot += NumOps.ToDouble(gradient[d]) * NumOps.ToDouble(_loraBasis[offset + d]);
        return dot;
    }

    /// <summary>
    /// Loss function for SPSA-based updates of basis vectors and initial coefficients.
    /// </summary>
    private double ComputeLoRALoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            var coeffs = new double[_rank];
            for (int i = 0; i < _rank; i++)
                coeffs[i] = NumOps.ToDouble(_loraCoeffInit[i]);

            // Quick adaptation
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                var adaptedParams = ComputeAdaptedParams(baseParams, coeffs);
                MetaModel.SetParameters(adaptedParams);
                var fullGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                for (int j = 0; j < _rank; j++)
                {
                    double gradCoeff = _scalingFactor * DotProductWithBasis(fullGrad, j);
                    coeffs[j] -= _algoOptions.InnerLearningRate * gradCoeff;
                }
            }

            var finalAdapted = ComputeAdaptedParams(baseParams, coeffs);
            MetaModel.SetParameters(finalAdapted);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
