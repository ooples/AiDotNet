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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Visual Prompt Tuning",
    "https://arxiv.org/abs/2203.12119",
    Year = 2022,
    Authors = "Menglin Jia, Luming Tang, Bor-Chun Chen, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class FreqPromptAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly FreqPromptOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _numComponents;

    /// <summary>Prompt basis vectors: flat array of K * paramDim values.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _promptBasis;

    /// <summary>Meta-learned initial prompt coefficients: length K.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _promptCoeffsInit;

    /// <summary>Per-frequency regularization weights (higher for high-freq).</summary>
    private readonly Vector<T> _freqWeights;

    private const double SpsaBasisLearningRateMultiplier = 0.1;
    private const double SpsaCoeffsLearningRateMultiplier = 0.5;
    private const double CoeffInitScaleFactor = 0.1;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.FreqPrompt;

    public FreqPromptAlgorithm(FreqPromptOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.InnerAndOuter)
    {
        if (options.NumFreqComponents <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumFreqComponents must be positive.");
        if (options.PromptInitScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "PromptInitScale must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _numComponents = options.NumFreqComponents;

        // Initialize basis vectors using DCT-like cosine basis
        // B[k, d] = scale * cos(pi * (k + 1) * (2d + 1) / (2D))
        var engine = AiDotNetEngine.Current;
        var frequencies = engine.TensorArange<T>(1, _numComponents + 1, 1);
        var positions = engine.TensorArange<T>(1, 2.0 * _paramDim + 1, 2);
        var phase = engine.TensorMultiplyScalar(engine.TensorOuter(frequencies, positions),
            NumOps.FromDouble(Math.PI / (2.0 * _paramDim)));
        _promptBasis = engine.TensorMultiplyScalar(engine.TensorCos(phase), NumOps.FromDouble(options.PromptInitScale)).ToVector();

        // Initialize prompt coefficients to small values
        _promptCoeffsInit = engine.Fill(_numComponents, NumOps.FromDouble(options.PromptInitScale * CoeffInitScaleFactor));

        // Frequency-dependent regularization: low freq (k=0) gets least penalty, w_k = 1 + k * penalty
        _freqWeights = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<T>(0, _numComponents, 1), NumOps.FromDouble(options.HighFreqPenalty)),
            NumOps.One).ToVector();
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: only adapt coefficients (backbone frozen)
            var coeffs = AdaptCoefficients(task, baseParams);

            // Evaluate with adapted prompt
            var finalParams = ComputeEffectiveParams(baseParams, coeffs);
            ParamModel.SetParameters(finalParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update backbone
        ParamModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var newBaseParams = ApplyOuterGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate);
            ParamModel.SetParameters(newBaseParams);
        }

        // Update prompt basis and initial coefficients via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _promptBasis, _algoOptions.OuterLearningRate * SpsaBasisLearningRateMultiplier, ComputeFreqPromptLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _promptCoeffsInit, _algoOptions.OuterLearningRate * SpsaCoeffsLearningRateMultiplier, ComputeFreqPromptLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = ParamModel.GetParameters();

        var coeffs = AdaptCoefficients(task, baseParams);

        var adaptedParams = ComputeEffectiveParams(baseParams, coeffs);
        ParamModel.SetParameters(baseParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Inner loop on the support set: the coefficients start at their meta-learned init and step along
    /// B g + w * c, the gradient projected onto the prompt basis plus the frequency-dependent regularization.
    /// </summary>
    private Vector<T> AdaptCoefficients(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> baseParams)
    {
        var coeffs = _promptCoeffsInit.Clone();
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var effectiveParams = ComputeEffectiveParams(baseParams, coeffs);
            ParamModel.SetParameters(effectiveParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            // Project gradient onto basis vectors to get coefficient gradients, then add the
            // frequency-dependent regularization (_freqWeights already encodes HighFreqPenalty)
            var projected = Engine.TensorMatMul(BasisMatrix(),
                Tensor<T>.FromVector(grad.Length == _paramDim ? grad : grad.GetSubVector(0, _paramDim), new[] { _paramDim, 1 })).ToVector();
            var stepGradient = Engine.Add(projected, Engine.Multiply(_freqWeights, coeffs));
            coeffs = ApplyInnerGradients(coeffs, stepGradient, _algoOptions.InnerLearningRate, step == 0);
        }

        return coeffs;
    }

    /// <summary>The prompt basis as a [K, paramDim] matrix, one basis vector per row.</summary>
    private Tensor<T> BasisMatrix() => Engine.Reshape(Tensor<T>.FromVector(_promptBasis), new[] { _numComponents, _paramDim });

    /// <summary>
    /// Computes effective parameters: theta = theta_base + sum_k c_k * B_k.
    /// </summary>
    private Vector<T> ComputeEffectiveParams(Vector<T> baseParams, Vector<T> coeffs)
    {
        var offset = Engine.TensorMatMul(Tensor<T>.FromVector(coeffs, new[] { 1, _numComponents }), BasisMatrix()).ToVector();
        return Engine.Add(baseParams.Length == _paramDim ? baseParams : baseParams.GetSubVector(0, _paramDim), offset);
    }

    private double ComputeFreqPromptLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var baseParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Match MetaTrain: include frequency-dependent regularization
            var coeffs = AdaptCoefficients(task, baseParams);

            var finalParams = ComputeEffectiveParams(baseParams, coeffs);
            ParamModel.SetParameters(finalParams);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        ParamModel.SetParameters(baseParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
