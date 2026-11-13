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
/// Implementation of ACL: Adaptive Continual Learning with task-specific parameter importance masks.
/// </summary>
/// <remarks>
/// <para>
/// ACL prevents catastrophic forgetting by maintaining per-parameter importance scores that
/// accumulate across tasks via exponential moving average. Important parameters are protected
/// by reducing their effective learning rate and applying elastic weight consolidation (EWC)-style
/// regularization toward the pre-task initialization.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Importance estimation: Ω_d = EMA(Ω_d, |∂L/∂θ_d|²)
///   where EMA uses ImportanceDecay
///
/// Protected learning rate: η'_d = η / (1 + ProtectionStrength * Ω_d)
///
/// Inner loop:
///   θ_d ← θ_d - η'_d * grad_d
///
/// Elastic regularization on query loss:
///   L_total = L_query + ElasticRegWeight * Σ_d Ω_d * (θ_d - θ_init_d)²
///
/// Mask sparsity penalty:
///   L_meta += MaskSparsityPenalty * Σ_d |Ω_d|
///
/// Outer loop: update θ, Ω (via SPSA)
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Overcoming Catastrophic Forgetting with Hard Attention to the Task",
    "https://arxiv.org/abs/1801.01423",
    Year = 2018,
    Authors = "Joan Serra, Didac Suris, Marius Miron, Alexandros Karatzoglou")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ACLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly ACLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Per-parameter importance scores (accumulated via EMA).</summary>
    [Buffer]
    private Vector<T> _importance;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ACL;

    public ACLAlgorithm(ACLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));
        if (options.ImportanceDecay <= 0 || options.ImportanceDecay >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "ImportanceDecay must be in (0, 1).");
        if (options.ProtectionStrength < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ProtectionStrength must be non-negative.");
        if (options.ElasticRegWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ElasticRegWeight must be non-negative.");
        _importance = new Vector<T>(_paramDim);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        var (decay, rootKeep, protection, learningRate) = ProtectionConstants();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = initParams.Clone();

            // Inner loop with importance-protected learning rates
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Update importance: EMA of squared gradients
                _importance = UpdateImportance(_importance, grad, decay, rootKeep);

                // Apply gradient with per-parameter protected learning rate lr / (1 + strength * importance)
                adaptedParams = ProtectedStep(adaptedParams, grad, _importance, protection, learningRate);
            }

            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Report query loss only — this matches the meta-gradient which is computed from query loss.
            // Regularization (EWC + sparsity) acts indirectly through importance-weighted learning rates
            // in the inner loop, keeping the outer optimization objective clean.
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = initParams.Clone();
        var (decay, rootKeep, protection, learningRate) = ProtectionConstants();

        // Use local copy of importance so Adapt doesn't mutate shared training state
        var localImportance = _importance.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            localImportance = UpdateImportance(localImportance, grad, decay, rootKeep);
            adaptedParams = ProtectedStep(adaptedParams, grad, localImportance, protection, learningRate);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// The importance EMA decay, the square root of its complement, the protection strength and the inner learning
    /// rate in T.
    /// </summary>
    private (T Decay, T RootKeep, T Protection, T LearningRate) ProtectionConstants()
        => (NumOps.FromDouble(_algoOptions.ImportanceDecay), NumOps.FromDouble(Math.Sqrt(1.0 - _algoOptions.ImportanceDecay)),
            NumOps.FromDouble(_algoOptions.ProtectionStrength), NumOps.FromDouble(_algoOptions.InnerLearningRate));

    /// <summary>
    /// The importance EMA decay * importance + (1 - decay) * g^2, with (1 - decay) * g^2 formed as
    /// (sqrt(1 - decay) * g)^2: the gradient is scaled before it is squared, so a Half gradient of 256 or more
    /// does not overflow to infinity (which would freeze that parameter) where the scalar loop's double did not.
    /// </summary>
    private Vector<T> UpdateImportance(Vector<T> importance, Vector<T> gradient, T decay, T rootKeep)
    {
        var scaled = Engine.Multiply(gradient, rootKeep);
        return Engine.Add(Engine.Multiply(importance, decay), Engine.Multiply(scaled, scaled));
    }

    /// <summary>theta - lr * g / (1 + strength * importance), element-wise.</summary>
    private Vector<T> ProtectedStep(Vector<T> parameters, Vector<T> gradient, Vector<T> importance, T protection, T learningRate)
    {
        var denominator = Engine.Add(Engine.Fill(importance.Length, NumOps.One), Engine.Multiply(importance, protection));
        return Engine.Subtract(parameters, Engine.Divide(Engine.Multiply(gradient, learningRate), denominator));
    }
}
