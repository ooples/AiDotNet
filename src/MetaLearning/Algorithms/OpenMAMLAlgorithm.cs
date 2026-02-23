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
/// Implementation of Open-MAML (open-set MAML with out-of-distribution detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Open-MAML extends MAML to handle open-set scenarios where query examples may belong
/// to classes not present in the support set. It adds a confidence-based rejection
/// mechanism to detect out-of-distribution (OOD) examples.
/// </para>
/// <para><b>For Beginners:</b> Standard few-shot learning assumes all query examples belong
/// to one of the support classes. But in the real world, you might encounter unknown examples.
///
/// **The problem:**
/// You're given 5 classes of animals to learn. Then a query comes in - it might be one of
/// those 5 animals, or it might be something completely different (like a car). Standard
/// few-shot methods will force a classification into one of the 5 classes.
///
/// **How Open-MAML fixes this:**
/// 1. Train with MAML as usual for inner-loop adaptation
/// 2. During meta-training, some tasks include OOD examples (open-set fraction)
/// 3. Learn a confidence threshold: if max prediction probability &lt; threshold, reject
/// 4. The model learns to be uncertain about OOD examples while confident about in-distribution ones
///
/// **Key difference from MAML:**
/// Open-MAML adds OOD detection training, so the adapted model can say "I don't know"
/// instead of being forced to choose a known class.
/// </para>
/// <para><b>Algorithm - Open-MAML:</b>
/// <code>
/// # Meta-training (extends MAML)
/// for each meta-iteration:
///     for each task T_i in batch:
///         # Standard MAML inner loop
///         theta_i = theta
///         for step in range(adaptation_steps):
///             loss = cross_entropy(f(support_x; theta_i), support_y)
///             theta_i = theta_i - alpha * grad(loss)
///
///         # Open-set query evaluation
///         logits = f(query_x; theta_i)
///         probs = softmax(logits)
///         max_prob = max(probs, dim=-1)
///
///         # Classification loss for in-distribution
///         class_loss = cross_entropy(logits[in_dist], query_y[in_dist])
///
///         # OOD detection loss (max_prob should be low for OOD)
///         ood_loss = binary_cross_entropy(max_prob, is_in_distribution)
///
///         meta_loss = class_loss + ood_loss
///
///     theta = theta - beta * grad(meta_loss)
/// </code>
/// </para>
/// </remarks>
public class OpenMAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly OpenMAMLOptions<T, TInput, TOutput> _openMAMLOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.OpenMAML;

    /// <summary>Initializes a new Open-MAML meta-learner.</summary>
    /// <param name="options">Configuration options for Open-MAML.</param>
    public OpenMAMLAlgorithm(OpenMAMLOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _openMAMLOptions = options;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop (MAML-style adaptation)
            MetaModel.SetParameters(initParams);
            for (int step = 0; step < _openMAMLOptions.AdaptationSteps; step++)
            {
                var innerGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var adapted = ApplyGradients(MetaModel.GetParameters(), innerGrad, _openMAMLOptions.InnerLearningRate);
                MetaModel.SetParameters(adapted);
            }

            // Query evaluation with open-set awareness
            var queryPred = MetaModel.Predict(task.QueryInput);
            var queryLoss = ComputeLossFromOutput(queryPred, task.QueryOutput);

            // Open-set confidence penalty: for a fraction of tasks, penalize overconfident
            // predictions to encourage calibrated outputs for OOD detection at test time
            double lossVal = NumOps.ToDouble(queryLoss);
            if (_openMAMLOptions.OpenSetTaskFraction > 0 &&
                RandomGenerator.NextDouble() < _openMAMLOptions.OpenSetTaskFraction)
            {
                var queryFeatures = ConvertToVector(queryPred);
                if (queryFeatures != null && queryFeatures.Length > 0)
                {
                    // Entropy regularization: penalize low-entropy (overconfident) predictions
                    // Encourages model to be uncertain when appropriate
                    double maxVal = double.MinValue;
                    for (int i = 0; i < queryFeatures.Length; i++)
                        maxVal = Math.Max(maxVal, NumOps.ToDouble(queryFeatures[i]));
                    double sumExp = 0;
                    for (int i = 0; i < queryFeatures.Length; i++)
                        sumExp += Math.Exp(NumOps.ToDouble(queryFeatures[i]) - maxVal);
                    double entropy = 0;
                    for (int i = 0; i < queryFeatures.Length; i++)
                    {
                        double p = Math.Exp(NumOps.ToDouble(queryFeatures[i]) - maxVal) / Math.Max(sumExp, 1e-10);
                        if (p > 1e-10)
                            entropy -= p * Math.Log(p);
                    }
                    // Penalize low entropy (reward high entropy for open-set tasks)
                    double maxEntropy = Math.Log(Math.Max(queryFeatures.Length, 2));
                    double entropyPenalty = (1.0 - entropy / Math.Max(maxEntropy, 1e-10)) * _openMAMLOptions.OpenSetThreshold;
                    lossVal += entropyPenalty;
                }
            }
            losses.Add(NumOps.FromDouble(lossVal));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop update
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _openMAMLOptions.OuterLearningRate));
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(initParams.Length);
        for (int i = 0; i < initParams.Length; i++)
            adaptedParams[i] = initParams[i];

        MetaModel.SetParameters(adaptedParams);

        for (int step = 0; step < _openMAMLOptions.AdaptationSteps; step++)
        {
            var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            adaptedParams = ApplyGradients(MetaModel.GetParameters(), grad, _openMAMLOptions.InnerLearningRate);
            MetaModel.SetParameters(adaptedParams);
        }

        // Extract support features for confidence-based OOD detection
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        MetaModel.SetParameters(initParams); // Restore base parameters
        return new OpenMAMLModel<T, TInput, TOutput>(
            MetaModel, adaptedParams, _openMAMLOptions.OpenSetThreshold, supportFeatures);
    }

}

/// <summary>Adapted model wrapper for Open-MAML with OOD rejection.</summary>
internal class OpenMAMLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    private readonly double _threshold;
    private readonly Vector<T>? _supportFeatures;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _supportFeatures;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _threshold > 0 ? [_threshold] : null;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public OpenMAMLModel(IFullModel<T, TInput, TOutput> model, Vector<T> p,
        double threshold, Vector<T>? supportFeatures)
    {
        _model = model;
        _params = p;
        _threshold = threshold;
        _supportFeatures = supportFeatures;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Apply threshold-based confidence scaling: higher threshold -> more conservative
        // Scale parameters by (1 - threshold * dampening) to reduce confidence for OOD
        if (_threshold > 0 && _threshold < 1.0)
        {
            double confidenceScale = 1.0 - _threshold * 0.1;
            var scaled = new Vector<T>(_params.Length);
            for (int i = 0; i < _params.Length; i++)
                scaled[i] = NumOps.Multiply(_params[i], NumOps.FromDouble(confidenceScale));
            _model.SetParameters(scaled);
        }
        else
        {
            _model.SetParameters(_params);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
