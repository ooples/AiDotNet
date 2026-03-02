using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of DiscoRL: Discovery-based meta-RL with reusable skill discovery.
/// </summary>
/// <remarks>
/// <para>
/// DiscoRL discovers reusable "skills" as low-rank directions in parameter space. Each
/// skill is a rank-R basis spanning a subspace of the full parameter space. A gating
/// network selects which skills to activate based on the initial task gradient signal.
/// During adaptation, only the skill coefficients are updated, enabling efficient reuse.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Skills: S_1..S_K, each S_k ∈ R^{R × D} (rank-R subspace)
/// Gating: g = softmax(W_g * compress(grad_0) / temperature)
///
/// Skill composition: Δθ = Σ_k g_k * S_k^T * α_k
///   where α_k ∈ R^R are per-skill coefficients
///
/// Inner loop: adapt α_k (skill coefficients)
///   α_k ← α_k - η * S_k * ∇L  (project gradient onto skill subspace)
///
/// Outer loop: update base params, skills S_k, gating W_g
/// Entropy bonus: H(g) to encourage diverse skill usage
/// </code>
/// </para>
/// </remarks>
public class DiscoRLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly DiscoRLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numSkills;
    private readonly int _skillRank;
    private readonly int _compressedDim;

    private const double SpsaLearningRateMultiplier = 0.1;

    /// <summary>Skill basis vectors: numSkills * skillRank * compressedDim.</summary>
    private Vector<T> _skillBasis;

    /// <summary>Gating network parameters: compressedDim * numSkills.</summary>
    private Vector<T> _gatingParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DiscoRL;

    public DiscoRLAlgorithm(DiscoRLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.NumSkills <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumSkills must be positive.");
        if (options.SkillRank <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "SkillRank must be positive.");
        if (options.SelectionTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "SelectionTemperature must be positive.");

        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _numSkills = options.NumSkills;
        _skillRank = options.SkillRank;
        _compressedDim = Math.Min(_paramDim, 64);

        _skillBasis = new Vector<T>(_numSkills * _skillRank * _compressedDim);
        for (int i = 0; i < _skillBasis.Length; i++)
            _skillBasis[i] = NumOps.FromDouble(0.01 * SampleNormal());

        _gatingParams = new Vector<T>(_compressedDim * _numSkills);
        for (int i = 0; i < _gatingParams.Length; i++)
            _gatingParams[i] = NumOps.FromDouble(0.01 * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0)
            return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Compute initial gradient for skill selection
            MetaModel.SetParameters(initParams);
            var initGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var compressed = CompressGradient(initGrad);

            // Gating: select skills
            var gating = ComputeGating(compressed);

            // Initialize per-skill coefficients
            var coeffs = new double[_numSkills * _skillRank];

            // Inner loop: adapt skill coefficients
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                // Compose parameter delta from skills
                var delta = ComposeSkillDelta(coeffs, gating);
                var effectiveParams = new Vector<T>(_paramDim);
                for (int d = 0; d < _paramDim; d++)
                    effectiveParams[d] = NumOps.Add(adaptedParams[d], delta[d % _compressedDim]);

                MetaModel.SetParameters(effectiveParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                var compGrad = CompressGradient(grad);

                // Project gradient onto skill subspaces to get coefficient gradients
                for (int k = 0; k < _numSkills; k++)
                {
                    double gk = NumOps.ToDouble(gating[k]);
                    if (gk < 1e-6) continue;
                    for (int r = 0; r < _skillRank; r++)
                    {
                        double projGrad = 0;
                        for (int c = 0; c < _compressedDim; c++)
                        {
                            int basisIdx = (k * _skillRank + r) * _compressedDim + c;
                            projGrad += NumOps.ToDouble(compGrad[c]) * NumOps.ToDouble(_skillBasis[basisIdx]);
                        }
                        coeffs[k * _skillRank + r] -= _algoOptions.InnerLearningRate * gk * projGrad;
                    }
                }
            }

            // Evaluate with final adapted parameters
            var finalDelta = ComposeSkillDelta(coeffs, gating);
            var finalParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                finalParams[d] = NumOps.Add(adaptedParams[d], finalDelta[d % _compressedDim]);

            MetaModel.SetParameters(finalParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Entropy bonus for diverse skill usage
            double entropy = 0;
            for (int k = 0; k < _numSkills; k++)
            {
                double gk = NumOps.ToDouble(gating[k]);
                if (gk > 1e-10) entropy -= gk * Math.Log(gk);
            }
            var totalLoss = NumOps.Subtract(queryLoss, NumOps.FromDouble(_algoOptions.SkillEntropyBonus * entropy));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _skillBasis, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeDiscoRLLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _gatingParams, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeDiscoRLLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = MetaModel.GetParameters();
        MetaModel.SetParameters(initParams);
        var initGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
        var gating = ComputeGating(CompressGradient(initGrad));
        var coeffs = new double[_numSkills * _skillRank];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var delta = ComposeSkillDelta(coeffs, gating);
            var effectiveParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                effectiveParams[d] = NumOps.Add(initParams[d], delta[d % _compressedDim]);

            MetaModel.SetParameters(effectiveParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var compGrad = CompressGradient(grad);

            for (int k = 0; k < _numSkills; k++)
            {
                double gk = NumOps.ToDouble(gating[k]);
                if (gk < 1e-6) continue;
                for (int r = 0; r < _skillRank; r++)
                {
                    double projGrad = 0;
                    for (int c = 0; c < _compressedDim; c++)
                        projGrad += NumOps.ToDouble(compGrad[c]) * NumOps.ToDouble(_skillBasis[(k * _skillRank + r) * _compressedDim + c]);
                    coeffs[k * _skillRank + r] -= _algoOptions.InnerLearningRate * gk * projGrad;
                }
            }
        }

        var finalDelta = ComposeSkillDelta(coeffs, gating);
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            adaptedParams[d] = NumOps.Add(initParams[d], finalDelta[d % _compressedDim]);

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private Vector<T> CompressGradient(Vector<T> grad)
    {
        var compressed = new Vector<T>(_compressedDim);
        for (int c = 0; c < _compressedDim && c < grad.Length; c++)
            compressed[c] = grad[c];
        return compressed;
    }

    private Vector<T> ComputeGating(Vector<T> compressed)
    {
        var logits = new double[_numSkills];
        for (int k = 0; k < _numSkills; k++)
        {
            double sum = 0;
            for (int c = 0; c < _compressedDim; c++)
                sum += NumOps.ToDouble(compressed[c]) * NumOps.ToDouble(_gatingParams[k * _compressedDim + c]);
            logits[k] = sum / _algoOptions.SelectionTemperature;
        }

        // Softmax
        double maxLogit = logits[0];
        for (int k = 1; k < _numSkills; k++) if (logits[k] > maxLogit) maxLogit = logits[k];
        double sumExp = 0;
        for (int k = 0; k < _numSkills; k++) { logits[k] = Math.Exp(logits[k] - maxLogit); sumExp += logits[k]; }
        for (int k = 0; k < _numSkills; k++) logits[k] /= (sumExp + 1e-10);

        var result = new Vector<T>(_numSkills);
        for (int k = 0; k < _numSkills; k++)
            result[k] = NumOps.FromDouble(logits[k]);
        return result;
    }

    private Vector<T> ComposeSkillDelta(double[] coeffs, Vector<T> gating)
    {
        var delta = new Vector<T>(_compressedDim);
        for (int k = 0; k < _numSkills; k++)
        {
            double gk = NumOps.ToDouble(gating[k]);
            if (gk < 1e-6) continue;
            for (int r = 0; r < _skillRank; r++)
            {
                double coeff = coeffs[k * _skillRank + r];
                for (int c = 0; c < _compressedDim; c++)
                    delta[c] = NumOps.Add(delta[c],
                        NumOps.FromDouble(gk * coeff * NumOps.ToDouble(_skillBasis[(k * _skillRank + r) * _compressedDim + c])));
            }
        }
        return delta;
    }

    private double ComputeDiscoRLLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var initGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var gating = ComputeGating(CompressGradient(initGrad));
            var coeffs = new double[_numSkills * _skillRank];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                var delta = ComposeSkillDelta(coeffs, gating);
                var ep = new Vector<T>(_paramDim);
                for (int d = 0; d < _paramDim; d++) ep[d] = NumOps.Add(initParams[d], delta[d % _compressedDim]);
                MetaModel.SetParameters(ep);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                var cg = CompressGradient(grad);
                for (int k = 0; k < _numSkills; k++)
                {
                    double gk = NumOps.ToDouble(gating[k]);
                    for (int r = 0; r < _skillRank; r++)
                    {
                        double pg = 0;
                        for (int c = 0; c < _compressedDim; c++) pg += NumOps.ToDouble(cg[c]) * NumOps.ToDouble(_skillBasis[(k * _skillRank + r) * _compressedDim + c]);
                        coeffs[k * _skillRank + r] -= _algoOptions.InnerLearningRate * gk * pg;
                    }
                }
            }
            var fd = ComposeSkillDelta(coeffs, gating);
            var fp = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) fp[d] = NumOps.Add(initParams[d], fd[d % _compressedDim]);
            MetaModel.SetParameters(fp);
            double queryLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));

            // Include entropy bonus to match MetaTrain objective
            double entropy = 0;
            for (int k = 0; k < _numSkills; k++)
            {
                double gk = NumOps.ToDouble(gating[k]);
                if (gk > 1e-10) entropy -= gk * Math.Log(gk);
            }
            totalLoss += queryLoss - _algoOptions.SkillEntropyBonus * entropy;
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
