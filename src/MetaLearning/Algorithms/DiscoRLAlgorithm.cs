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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Discovery of Options via Meta-Learned Subgoals",
    "https://arxiv.org/abs/2102.06741",
    Year = 2021,
    Authors = "Vivek Veeriah, Tom Zahavy, Matteo Hessel, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class DiscoRLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly DiscoRLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numSkills;
    private readonly int _skillRank;
    private readonly int _compressedDim;

    private const double SpsaLearningRateMultiplier = 0.1;

    /// <summary>Skill basis vectors: numSkills * skillRank * compressedDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _skillBasis;

    /// <summary>Gating network parameters: compressedDim * numSkills.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _gatingParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.DiscoRL;

    public DiscoRLAlgorithm(DiscoRLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        if (options.NumSkills <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumSkills must be positive.");
        if (options.SkillRank <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "SkillRank must be positive.");
        if (options.SelectionTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "SelectionTemperature must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
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
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var finalParams = AdaptSkills(task, initParams, out var gating);

            ParamModel.SetParameters(finalParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Entropy bonus for diverse skill usage
            var totalLoss = NumOps.Subtract(queryLoss,
                NumOps.Multiply(NumOps.FromDouble(_algoOptions.SkillEntropyBonus), GatingEntropy(gating)));

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
        var initParams = ParamModel.GetParameters();
        var adaptedParams = AdaptSkills(task, initParams, out _);

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// The skill-coefficient inner loop: gating comes from the clipped support gradient at theta_0, then each
    /// step evaluates theta_0 + delta(coeffs) cycled over the parameters and moves every active skill's
    /// coefficients by -lr * g_k * (B_kr . g_c). Returns theta_0 + delta(final coeffs).
    /// </summary>
    private Vector<T> AdaptSkills(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams, out Vector<T> gating)
    {
        // Compute initial gradient for skill selection
        ParamModel.SetParameters(initParams);
        var initGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
        gating = ComputeGating(CompressGradient(initGrad));

        // Per-(skill, rank) gate, zero for inactive skills (g_k < 1e-6)
        var rankGates = PerRank(ActiveGating(gating));
        var basis = Tensor<T>.FromVector(_skillBasis, new[] { _numSkills * _skillRank, _compressedDim });
        T learningRate = NumOps.FromDouble(_algoOptions.InnerLearningRate);

        // Inner loop: adapt skill coefficients
        var coeffs = new Vector<T>(_numSkills * _skillRank);
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            // Compose parameter delta from skills
            ParamModel.SetParameters(WithDelta(initParams, ComposeSkillDelta(coeffs, gating)));
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var compGrad = Tensor<T>.FromVector(CompressGradient(grad), new[] { _compressedDim, 1 });

            // Project gradient onto skill subspaces to get coefficient gradients
            var projGrad = Engine.TensorMatMul(basis, compGrad).ToVector();
            coeffs = Engine.Subtract(coeffs, Engine.Multiply(Engine.Multiply(rankGates, projGrad), learningRate));
        }

        // Evaluate with final adapted parameters
        return WithDelta(initParams, ComposeSkillDelta(coeffs, gating));
    }

    /// <summary>theta + delta[d mod compressedDim].</summary>
    private Vector<T> WithDelta(Vector<T> parameters, Vector<T> delta)
        => Engine.Add(parameters.GetSubVector(0, _paramDim), ParameterModulation.Cycle(Engine, delta, _paramDim));

    /// <summary>The gating with skills below 1e-6 zeroed.</summary>
    private Vector<T> ActiveGating(Vector<T> gating)
    {
        var inactive = Engine.TensorLessThan(Tensor<T>.FromVector(gating), NumOps.FromDouble(1e-6)).ToVector();
        return Engine.Multiply(gating, Engine.Subtract(Engine.Fill(_numSkills, NumOps.One), inactive));
    }

    /// <summary>Each skill's value repeated for its skillRank coefficients.</summary>
    private Vector<T> PerRank(Vector<T> perSkill)
    {
        var repeated = Engine.TensorBroadcastTo(
            Tensor<T>.FromVector(perSkill, new[] { _numSkills, 1 }), new[] { _numSkills, _skillRank });
        return Engine.Reshape(repeated, new[] { _numSkills * _skillRank }).ToVector();
    }

    /// <summary>-sum_k g_k log g_k over skills with g_k &gt; 1e-10.</summary>
    private T GatingEntropy(Vector<T> gating)
    {
        var g = Tensor<T>.FromVector(gating);
        var included = Engine.TensorGreaterThan(g, NumOps.FromDouble(1e-10));
        var terms = Engine.TensorMultiply(Engine.TensorMultiply(g, Engine.TensorLog(Engine.TensorMax(g, NumOps.FromDouble(1e-10)))), included);
        return NumOps.Negate(Engine.TensorSum(terms));
    }

    /// <summary>The first compressedDim gradient entries, zero-padded.</summary>
    private Vector<T> CompressGradient(Vector<T> grad)
    {
        if (grad.Length >= _compressedDim)
            return grad.GetSubVector(0, _compressedDim);
        return Vector<T>.Concatenate(grad, new Vector<T>(_compressedDim - grad.Length));
    }

    /// <summary>softmax(G c / temperature), normalized by (sum + 1e-10), with G stored [numSkills, compressedDim].</summary>
    private Vector<T> ComputeGating(Vector<T> compressed)
    {
        var weights = Tensor<T>.FromVector(_gatingParams, new[] { _numSkills, _compressedDim });
        var logits = Engine.TensorDivideScalar(
            Engine.TensorMatMul(weights, Tensor<T>.FromVector(compressed, new[] { _compressedDim, 1 })),
            NumOps.FromDouble(_algoOptions.SelectionTemperature));

        // Softmax
        var exp = Engine.TensorExp(Engine.TensorSubtractScalar(logits, Engine.TensorMaxValue(logits)));
        return Engine.TensorDivideScalar(exp, NumOps.Add(Engine.TensorSum(exp), NumOps.FromDouble(1e-10))).ToVector();
    }

    /// <summary>sum over active skills k and ranks r of g_k * coeff_kr * B_kr.</summary>
    private Vector<T> ComposeSkillDelta(Vector<T> coeffs, Vector<T> gating)
    {
        var weighted = Engine.Multiply(PerRank(ActiveGating(gating)), coeffs);
        var basis = Tensor<T>.FromVector(_skillBasis, new[] { _numSkills * _skillRank, _compressedDim });
        return Engine.TensorMatMul(Tensor<T>.FromVector(weighted, new[] { 1, _numSkills * _skillRank }), basis).ToVector();
    }

    private double ComputeDiscoRLLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();
        T entropyBonus = NumOps.FromDouble(_algoOptions.SkillEntropyBonus);
        foreach (var task in taskBatch.Tasks)
        {
            var fp = AdaptSkills(task, initParams, out var gating);
            ParamModel.SetParameters(fp);
            T queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Include entropy bonus to match MetaTrain objective
            losses.Add(NumOps.Subtract(queryLoss, NumOps.Multiply(entropyBonus, GatingEntropy(gating))));
        }
        ParamModel.SetParameters(initParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
