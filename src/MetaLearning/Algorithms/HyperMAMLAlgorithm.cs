using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of HyperMAML (hypernetwork-based MAML initialization).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// HyperMAML uses a hypernetwork to generate task-specific initial parameters for MAML,
/// rather than using a single shared initialization. The hypernetwork conditions on
/// support set statistics to produce a better starting point for each task.
/// </para>
/// <para><b>For Beginners:</b> HyperMAML improves MAML's starting point for each task:
///
/// **The problem with standard MAML:**
/// MAML learns ONE initialization that's supposed to be good for ALL tasks.
/// But some tasks are very different from each other, so one starting point
/// can't be optimal for everything.
///
/// **How HyperMAML fixes this:**
/// 1. Look at the support set for the current task
/// 2. Compute statistics about this specific task (means, variances, etc.)
/// 3. Feed these statistics into a hypernetwork
/// 4. The hypernetwork generates a CUSTOM initialization for this task
/// 5. Then proceed with standard MAML inner-loop adaptation from this better start
///
/// **The benefit:**
/// - Task A (dog breeds): Gets an initialization tuned for fine-grained visual features
/// - Task B (vehicles vs animals): Gets an initialization tuned for coarse category differences
/// - Fewer inner-loop steps needed because the starting point is already close
/// </para>
/// <para><b>Algorithm - HyperMAML:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor          # Shared backbone
/// h_psi = hypernetwork                 # Generates task-specific initialization
/// theta_shared = shared_initialization # Fallback/base initialization
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Compute support set statistics
///         z_s = f_theta(support_x)
///         stats = [mean(z_s), var(z_s)]
///
///         # 2. Generate task-specific initialization
///         theta_task = h_psi(stats)
///
///         # 3. Blend with shared initialization
///         theta_init = alpha * theta_task + (1-alpha) * theta_shared
///
///         # 4. Standard MAML inner loop from blended init
///         theta_i = theta_init
///         for step in range(adaptation_steps):
///             loss = cross_entropy(f(support_x; theta_i), support_y)
///             theta_i = theta_i - lr * grad(loss)
///
///         meta_loss = cross_entropy(f(query_x; theta_i), query_y)
///
///     theta, psi = theta, psi - lr * grad(meta_loss)
/// </code>
/// </para>
/// </remarks>
public class HyperMAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly HyperMAMLOptions<T, TInput, TOutput> _hyperMAMLOptions;

    /// <summary>Parameters for the initialization hypernetwork.</summary>
    private Vector<T> _hypernetParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.HyperMAML;

    /// <summary>Initializes a new HyperMAML meta-learner.</summary>
    /// <param name="options">Configuration options for HyperMAML.</param>
    public HyperMAMLAlgorithm(HyperMAMLOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _hyperMAMLOptions = options;
        InitializeHypernetwork();
    }

    /// <summary>Initializes the hypernetwork parameters.</summary>
    private void InitializeHypernetwork()
    {
        int hiddenDim = _hyperMAMLOptions.HypernetHiddenDim;
        int numLayers = _hyperMAMLOptions.HypernetNumLayers;
        // Multi-layer hypernetwork: input -> hidden layers -> output
        int totalParams = hiddenDim * hiddenDim * numLayers + hiddenDim * numLayers;
        _hypernetParams = new Vector<T>(totalParams);
        double scale = Math.Sqrt(2.0 / hiddenDim);
        for (int i = 0; i < totalParams; i++)
        {
            _hypernetParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        }
    }

    /// <summary>
    /// Generates a task-specific initialization by running support features through the hypernetwork,
    /// then blends it with the shared initialization.
    /// </summary>
    /// <param name="sharedInit">The shared meta-learned initialization.</param>
    /// <param name="supportFeatures">Support set features for conditioning.</param>
    /// <returns>Blended task-specific initialization.</returns>
    private Vector<T> GenerateTaskInit(Vector<T> sharedInit, Vector<T>? supportFeatures)
    {
        if (supportFeatures == null || supportFeatures.Length == 0)
            return sharedInit;

        // Compute support set statistics
        T mean = NumOps.Zero;
        T variance = NumOps.Zero;
        for (int i = 0; i < supportFeatures.Length; i++)
            mean = NumOps.Add(mean, supportFeatures[i]);
        mean = NumOps.Divide(mean, NumOps.FromDouble(Math.Max(supportFeatures.Length, 1)));

        for (int i = 0; i < supportFeatures.Length; i++)
        {
            T diff = NumOps.Subtract(supportFeatures[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(Math.Max(supportFeatures.Length, 1)));

        double meanVal = NumOps.ToDouble(mean);
        double varVal = NumOps.ToDouble(variance);

        // Generate per-parameter modulations via hypernetwork
        double blendAlpha = _hyperMAMLOptions.BlendFactor;
        var blended = new Vector<T>(sharedInit.Length);
        int paramIdx = 0;

        for (int i = 0; i < sharedInit.Length; i++)
        {
            // Hypernetwork: MLP(stats) -> per-parameter offset
            double w1 = paramIdx < _hypernetParams.Length
                ? NumOps.ToDouble(_hypernetParams[paramIdx++ % _hypernetParams.Length]) : 0.01;
            double w2 = paramIdx < _hypernetParams.Length
                ? NumOps.ToDouble(_hypernetParams[paramIdx++ % _hypernetParams.Length]) : 0.01;
            double b = paramIdx < _hypernetParams.Length
                ? NumOps.ToDouble(_hypernetParams[paramIdx++ % _hypernetParams.Length]) : 0;
            double offset = Math.Tanh(w1 * meanVal + w2 * varVal + b) * 0.1;

            // Blend: alpha * (shared + offset) + (1-alpha) * shared
            double sharedVal = NumOps.ToDouble(sharedInit[i]);
            double taskVal = sharedVal + offset;
            blended[i] = NumOps.FromDouble(blendAlpha * taskVal + (1.0 - blendAlpha) * sharedVal);
        }

        return blended;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Generate task-specific initialization from support features
            MetaModel.SetParameters(initParams);
            var supportPred = MetaModel.Predict(task.SupportInput);
            var supportFeatures = ConvertToVector(supportPred);
            var taskInit = GenerateTaskInit(initParams, supportFeatures);
            MetaModel.SetParameters(taskInit);

            // Inner loop adaptation from task-specific init
            for (int step = 0; step < _hyperMAMLOptions.AdaptationSteps; step++)
            {
                var innerGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var adapted = ApplyGradients(MetaModel.GetParameters(), innerGrad, _hyperMAMLOptions.InnerLearningRate);
                MetaModel.SetParameters(adapted);
            }

            // Query evaluation
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop update
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _hyperMAMLOptions.OuterLearningRate));
        }

        // Update hypernetwork via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _hypernetParams, _hyperMAMLOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var sharedInit = MetaModel.GetParameters();

        // Generate task-specific initialization from support features
        MetaModel.SetParameters(sharedInit);
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var taskInit = GenerateTaskInit(sharedInit, supportFeatures);
        MetaModel.SetParameters(taskInit);

        // Inner loop adaptation from task-specific init
        var adaptedParams = taskInit;
        for (int step = 0; step < _hyperMAMLOptions.AdaptationSteps; step++)
        {
            var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            adaptedParams = ApplyGradients(MetaModel.GetParameters(), grad, _hyperMAMLOptions.InnerLearningRate);
            MetaModel.SetParameters(adaptedParams);
        }

        return new HyperMAMLModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

}

/// <summary>Adapted model wrapper for HyperMAML.</summary>
internal class HyperMAMLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public HyperMAMLModel(IFullModel<T, TInput, TOutput> model, Vector<T> p) { _model = model; _params = p; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
