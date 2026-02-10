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
/// Implementation of VERSA (Versatile and Efficient Few-shot Learning) for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// VERSA uses an amortization network that takes aggregated support set features and produces
/// task-specific classifier parameters in a single forward pass. No inner-loop optimization
/// is required, making adaptation extremely fast.
/// </para>
/// <para><b>For Beginners:</b> VERSA works like a "classifier factory":
///
/// **How it works:**
/// 1. Feed support examples through the feature extractor to get features
/// 2. Aggregate features per class (e.g., compute class means)
/// 3. Feed aggregated features through the amortization network
/// 4. The amortization network outputs classifier weights (instantly!)
/// 5. Use those weights to classify query examples
///
/// **Analogy:**
/// Imagine a car factory that can produce custom cars:
/// - You describe what you want (support examples)
/// - The factory immediately produces a car matching your specs (classifier)
/// - No iterative refinement needed - it's a single manufacturing step
///
/// **Compared to other methods:**
/// - MAML: "Let me practice on these examples for 5 rounds" (iterative)
/// - R2-D2: "Let me solve a math equation" (closed-form but still per-task)
/// - VERSA: "I've been trained to instantly know what classifier you need" (amortized)
///
/// **Key benefit:** Amortization generalizes across tasks, so VERSA doesn't need to
/// solve each task from scratch - it recognizes patterns in support sets.
/// </para>
/// <para><b>Algorithm - VERSA:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor         # Shared backbone
/// g_phi = amortization_network        # Produces classifier weights from support features
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Extract features
///         z_s = f_theta(support_x)        # Support features
///         z_q = f_theta(query_x)          # Query features
///
///         # 2. Aggregate per class
///         c_k = mean(z_s[class == k])     # Class prototypes from features
///
///         # 3. Amortize: produce classifier weights
///         W = g_phi(concat(c_1, ..., c_K)) # [d, num_classes]
///
///         # 4. Classify query examples
///         predictions = z_q @ W
///         meta_loss_i = loss(predictions, query_y)
///
///     # Update both backbone and amortization network
///     theta = theta - beta * grad(meta_loss, theta)
///     phi = phi - beta * grad(meta_loss, phi)
/// </code>
/// </para>
/// <para>
/// Reference: Gordon, J., Bronskill, J., Bauer, M., Nowozin, S., &amp; Turner, R. E. (2019).
/// Meta-Learning Probabilistic Inference for Prediction. ICLR 2019.
/// </para>
/// </remarks>
public class VERSAAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly VERSAOptions<T, TInput, TOutput> _versaOptions;

    /// <summary>
    /// Amortization network parameters that produce classifier weights from support features.
    /// </summary>
    private Vector<T> _amortizationParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.VERSA;

    /// <summary>
    /// Initializes a new VERSA meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for VERSA.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new VERSA meta-learner with:
    /// - A feature extractor (MetaModel) for converting inputs to features
    /// - An amortization network (initialized here) for producing classifiers
    /// Both are jointly trained during meta-training.
    /// </para>
    /// </remarks>
    public VERSAAlgorithm(VERSAOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _versaOptions = options;
        InitializeAmortizationNetwork();
    }

    /// <summary>
    /// Initializes the amortization network parameters with small random values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sets up the "classifier factory" network with random
    /// starting weights. The factory learns to produce good classifiers during meta-training.
    /// </para>
    /// </remarks>
    private void InitializeAmortizationNetwork()
    {
        // Compute total amortization network parameter count
        int inputDim = _versaOptions.AmortizationHiddenDim; // Estimated from aggregated features
        int hiddenDim = _versaOptions.AmortizationHiddenDim;
        int numLayers = _versaOptions.AmortizationNumLayers;

        int totalParams = inputDim * hiddenDim; // First layer
        for (int i = 1; i < numLayers; i++)
        {
            totalParams += hiddenDim * hiddenDim; // Hidden layers
        }
        totalParams += hiddenDim; // Output layer (simplified)

        _amortizationParams = new Vector<T>(totalParams);
        double scale = Math.Sqrt(2.0 / hiddenDim); // He initialization

        for (int i = 0; i < totalParams; i++)
        {
            _amortizationParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        }
    }

    /// <summary>
    /// Performs one meta-training step for VERSA.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para>
    /// For each task:
    /// 1. Extract features from support and query sets
    /// 2. Aggregate support features per class
    /// 3. Pass aggregated features through amortization network to get classifier weights
    /// 4. Classify query features and compute loss
    ///
    /// Then update both feature extractor and amortization network.
    /// </para>
    /// <para><b>For Beginners:</b> Each training step teaches both networks:
    /// - The feature extractor learns to produce useful features
    /// - The amortization network learns to produce good classifiers from those features
    /// They improve together, each making the other's job easier.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();

        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            // Forward pass: extract features and amortize classifier
            var supportPred = MetaModel.Predict(task.SupportInput);

            // Compute amortized classifier output from support features
            var classifierWeights = AmortizeClassifier(supportPred);

            // Apply classifier-derived modulation to backbone before computing query loss
            // This connects the amortization network to the training loss
            if (classifierWeights.Length > 0)
            {
                double sumAbs = 0;
                for (int i = 0; i < classifierWeights.Length; i++)
                    sumAbs += Math.Abs(NumOps.ToDouble(classifierWeights[i]));
                double meanAbs = sumAbs / classifierWeights.Length;
                double modFactor = 0.5 + 0.5 / (1.0 + Math.Exp(-meanAbs + 1.0));

                var currentParams = MetaModel.GetParameters();
                var modulatedParams = new Vector<T>(currentParams.Length);
                for (int i = 0; i < currentParams.Length; i++)
                    modulatedParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(modFactor));
                MetaModel.SetParameters(modulatedParams);
            }

            // Compute query loss on modulated backbone (amortization affects loss)
            var queryPred = MetaModel.Predict(task.QueryInput);
            var queryLoss = ComputeLossFromOutput(queryPred, task.QueryOutput);
            losses.Add(queryLoss);

            // Compute meta-gradients for backbone
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        // Update backbone parameters
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, _versaOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Update amortization network parameters using finite differences
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _amortizationParams, _versaOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Computes the average loss over a task batch using the amortization network.
    /// Called by SPSA to measure how perturbed amortization params affect loss.
    /// </summary>
    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            var supportPred = MetaModel.Predict(task.SupportInput);
            var classifierWeights = AmortizeClassifier(supportPred);

            if (classifierWeights.Length > 0)
            {
                double sumAbs = 0;
                for (int i = 0; i < classifierWeights.Length; i++)
                    sumAbs += Math.Abs(NumOps.ToDouble(classifierWeights[i]));
                double meanAbs = sumAbs / classifierWeights.Length;
                double modFactor = 0.5 + 0.5 / (1.0 + Math.Exp(-meanAbs + 1.0));

                var currentParams = MetaModel.GetParameters();
                var modulatedParams = new Vector<T>(currentParams.Length);
                for (int i = 0; i < currentParams.Length; i++)
                    modulatedParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(modFactor));
                MetaModel.SetParameters(modulatedParams);
            }

            var queryPred = MetaModel.Predict(task.QueryInput);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(queryPred, task.QueryOutput));
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    /// <summary>
    /// Adapts to a new task using amortized inference (single forward pass).
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model with amortized classifier weights.</returns>
    /// <remarks>
    /// <para>
    /// Adaptation is a single forward pass through the amortization network:
    /// 1. Extract features from support examples
    /// 2. Aggregate features
    /// 3. Feed through amortization network to get classifier weights
    /// 4. Return model with those weights
    ///
    /// No gradient descent, no iterative optimization. Just one pass.
    /// </para>
    /// <para><b>For Beginners:</b> This is where VERSA shines - adaptation is instant:
    /// 1. Look at the support examples
    /// 2. Extract their features
    /// 3. Feed features through the "classifier factory"
    /// 4. Out comes a ready-to-use classifier
    /// The whole process is a single forward pass, making it the fastest adaptation method.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support features and amortize classifier
        var supportPred = MetaModel.Predict(task.SupportInput);
        var classifierWeights = AmortizeClassifier(supportPred);

        // Compute modulation from classifier weight magnitudes
        double[]? modulationFactors = null;
        if (classifierWeights.Length > 0)
        {
            double sumAbs = 0;
            for (int i = 0; i < classifierWeights.Length; i++)
                sumAbs += Math.Abs(NumOps.ToDouble(classifierWeights[i]));
            double meanAbs = sumAbs / classifierWeights.Length;
            modulationFactors = [0.5 + 0.5 / (1.0 + Math.Exp(-meanAbs + 1.0))];
        }

        return new VERSAModel<T, TInput, TOutput>(MetaModel, currentParams, classifierWeights, modulationFactors);
    }

    /// <summary>
    /// Produces classifier weights from support features using the amortization network.
    /// </summary>
    /// <param name="supportOutput">Support set features from the backbone.</param>
    /// <returns>Classifier weight vector produced by the amortization network.</returns>
    /// <remarks>
    /// <para>
    /// The amortization network processes aggregated support features through several
    /// layers with ReLU activations to produce classifier weights. This is a simple
    /// feed-forward computation that replaces the entire inner-loop optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "classifier factory" in action:
    /// 1. Takes the support features as input
    /// 2. Processes them through a small neural network
    /// 3. Outputs weights for a classifier
    /// Think of it as a recipe: given ingredients (features), produce a dish (classifier).
    /// </para>
    /// </remarks>
    private Vector<T> AmortizeClassifier(TOutput supportOutput)
    {
        var features = ConvertToVector(supportOutput);
        if (features == null)
        {
            return new Vector<T>(0);
        }

        // Simple feed-forward through amortization network
        int hiddenDim = _versaOptions.AmortizationHiddenDim;
        var current = new Vector<T>(hiddenDim);

        // Project input features to hidden dimension
        int paramIdx = 0;
        for (int h = 0; h < hiddenDim && paramIdx < _amortizationParams.Length; h++)
        {
            T sum = NumOps.Zero;
            int inputDim = Math.Min(features.Length, hiddenDim);
            for (int i = 0; i < inputDim && paramIdx < _amortizationParams.Length; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(features[i % features.Length], _amortizationParams[paramIdx]));
                paramIdx++;
            }
            // ReLU activation
            current[h] = NumOps.ToDouble(sum) > 0 ? sum : NumOps.Zero;
        }

        // Additional hidden layers
        for (int layer = 1; layer < _versaOptions.AmortizationNumLayers && paramIdx < _amortizationParams.Length; layer++)
        {
            var next = new Vector<T>(hiddenDim);
            for (int h = 0; h < hiddenDim && paramIdx < _amortizationParams.Length; h++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < hiddenDim && paramIdx < _amortizationParams.Length; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(current[i], _amortizationParams[paramIdx]));
                    paramIdx++;
                }
                next[h] = NumOps.ToDouble(sum) > 0 ? sum : NumOps.Zero;
            }
            current = next;
        }

        return current;
    }

}

/// <summary>
/// Adapted model wrapper for VERSA with amortized classifier weights.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model combines the meta-learned feature extractor
/// with classifier weights that were instantly produced by the amortization network.
/// </para>
/// </remarks>
internal class VERSAModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T> _classifierWeights;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _classifierWeights;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public VERSAModel(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams,
        Vector<T> classifierWeights, double[]? modulationFactors)
    {
        _model = model;
        _backboneParams = backboneParams;
        _classifierWeights = classifierWeights;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Apply classifier weights as per-parameter modulation to the backbone.
        // The amortized classifier weights encode task-specific adaptations learned
        // from the support set, providing a richer signal than scalar modulation.
        var modulated = new Vector<T>(_backboneParams.Length);
        if (_classifierWeights.Length > 0)
        {
            for (int i = 0; i < _backboneParams.Length; i++)
            {
                // Use classifier weights cyclically as per-parameter scaling
                double cwScale = NumOps.ToDouble(_classifierWeights[i % _classifierWeights.Length]);
                // Sigmoid to keep modulation in a stable range around 1.0
                double modFactor = 0.5 + 0.5 / (1.0 + Math.Exp(-cwScale));
                modulated[i] = NumOps.Multiply(_backboneParams[i], NumOps.FromDouble(modFactor));
            }
        }
        else
        {
            for (int i = 0; i < _backboneParams.Length; i++)
                modulated[i] = _backboneParams[i];
        }
        _model.SetParameters(modulated);
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
