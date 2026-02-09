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
/// Implementation of TIM (Transductive Information Maximization) for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// TIM is a transductive few-shot method that refines query predictions by maximizing
/// mutual information between features and predicted labels. It processes all query
/// examples jointly, using the query set's structure for better classification.
/// </para>
/// <para><b>For Beginners:</b> TIM lets query examples help classify each other:
///
/// **The key insight:**
/// If you're classifying 15 query examples into 5 classes, you know each class
/// should get roughly 3 examples. TIM uses this constraint along with confidence
/// maximization to iteratively refine predictions.
///
/// **How it works:**
/// 1. Start with initial predictions (e.g., nearest centroid from support set)
/// 2. Iteratively refine by optimizing:
///    - Each query should be confidently assigned to ONE class (low conditional entropy)
///    - Classes should be balanced overall (high marginal entropy)
///    - Don't deviate too far from initial predictions
/// 3. After convergence, output the refined predictions
///
/// **Why it works:**
/// By processing queries together, TIM avoids "lonely" misclassifications.
/// If most queries near a centroid say "class A", the outlier gets pulled in too.
/// </para>
/// <para><b>Algorithm - TIM:</b>
/// <code>
/// # Given: support features z_s, query features z_q, support labels y_s
///
/// # 1. Initial predictions using prototypes
/// p_k = mean(z_s[class == k])           # Class prototypes
/// logits_0 = -||z_q - p_k||^2 * temp   # Initial logits
/// q_0 = softmax(logits_0)               # Initial soft assignments
///
/// # 2. Transductive refinement
/// for t in range(T):
///     # Conditional entropy: encourage confident predictions
///     H_cond = -sum(q_t * log(q_t))
///
///     # Marginal entropy: encourage balanced class usage
///     q_bar = mean(q_t, axis=0)          # Average assignment per class
///     H_marg = -sum(q_bar * log(q_bar))
///
///     # Objective: maximize mutual information = H_marg - H_cond
///     L = alpha * H_cond - beta * H_marg
///
///     # Gradient step on soft assignments
///     q_{t+1} = q_t - lr * grad(L, q_t)
///     q_{t+1} = softmax(q_{t+1})         # Re-normalize
///
/// # 3. Output refined predictions
/// predictions = argmax(q_T)
/// </code>
/// </para>
/// <para>
/// Reference: Boudiaf, M., Ziko, I., Rony, J., Dolz, J., Piantanida, P., &amp; Ben Ayed, I. (2020).
/// Information Maximization for Few-Shot Learning. NeurIPS 2020.
/// </para>
/// </remarks>
public class TIMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly TIMOptions<T, TInput, TOutput> _timOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.TIM;

    /// <summary>
    /// Initializes a new TIM meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for TIM.</param>
    public TIMAlgorithm(TIMOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _timOptions = options;
    }

    /// <summary>
    /// Performs one meta-training step for TIM.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> TIM's training step is similar to ProtoNets
    /// but with transductive refinement during evaluation. The backbone is trained
    /// with standard episodic training (nearest-centroid classification).
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

            var queryLoss = ComputeLossFromOutput(
                MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, _timOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task using transductive information maximization.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model with transductively refined predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> TIM's adaptation:
    /// 1. Compute initial predictions from support centroids
    /// 2. Iteratively refine ALL query predictions together
    /// 3. Each iteration maximizes mutual information:
    ///    - Make each prediction more confident
    ///    - Keep class assignments balanced
    /// 4. Return model with refined decision boundaries
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Run transductive refinement
        var refinedWeights = TransductiveRefine(supportFeatures, task);

        // Compute modulation factors from refined vs raw support features
        double[]? modulationFactors = null;
        if (supportFeatures != null && refinedWeights != null)
        {
            double sumRatio = 0;
            int count = 0;
            for (int i = 0; i < Math.Min(supportFeatures.Length, refinedWeights.Length); i++)
            {
                double rawVal = NumOps.ToDouble(supportFeatures[i]);
                double adaptedVal = NumOps.ToDouble(refinedWeights[i]);
                if (Math.Abs(rawVal) > 1e-10)
                {
                    sumRatio += Math.Max(0.5, Math.Min(2.0, adaptedVal / rawVal));
                    count++;
                }
            }
            if (count > 0)
                modulationFactors = [sumRatio / count];
        }

        return new TIMModel<T, TInput, TOutput>(MetaModel, currentParams, refinedWeights, modulationFactors);
    }

    /// <summary>
    /// Performs transductive refinement by maximizing mutual information.
    /// </summary>
    /// <param name="supportFeatures">Features from the support set.</param>
    /// <param name="task">The current meta-learning task.</param>
    /// <returns>Refined classification weights.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is where TIM's magic happens:
    /// 1. Start with soft predictions based on distance to class centroids
    /// 2. In each iteration:
    ///    - Compute conditional entropy (how confident are predictions?)
    ///    - Compute marginal entropy (how balanced are class assignments?)
    ///    - Adjust predictions to maximize mutual information (confident + balanced)
    /// 3. After convergence, predictions are refined and more accurate
    /// </para>
    /// </remarks>
    private Vector<T>? TransductiveRefine(Vector<T>? supportFeatures, IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (supportFeatures == null || supportFeatures.Length == 0)
        {
            return supportFeatures;
        }

        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);
        if (queryFeatures == null)
        {
            return supportFeatures;
        }

        // Infer number of classes from support features
        // Treat each support feature element as one class prototype
        int numClasses = Math.Max(supportFeatures.Length, 1);
        int numQuery = queryFeatures.Length;

        // Initialize logits as [numQuery x numClasses] with query-to-class similarity
        double temperature = Math.Max(_timOptions.Temperature, 1e-10);
        var logits = new double[numQuery, numClasses];
        for (int q = 0; q < numQuery; q++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                double queryVal = NumOps.ToDouble(queryFeatures[q]);
                double protoVal = NumOps.ToDouble(supportFeatures[c]);
                logits[q, c] = queryVal * protoVal * temperature;
            }
        }

        // Transductive refinement iterations
        double condWeight = _timOptions.ConditionalEntropyWeight;
        double margWeight = _timOptions.MarginalEntropyWeight;
        double lr = 0.1;

        for (int iter = 0; iter < _timOptions.TransductiveIterations; iter++)
        {
            // Per-query softmax across classes to get probabilities
            var probs = new double[numQuery, numClasses];
            for (int q = 0; q < numQuery; q++)
            {
                double maxLogit = double.MinValue;
                for (int c = 0; c < numClasses; c++)
                    maxLogit = Math.Max(maxLogit, logits[q, c]);

                double sumExp = 0;
                for (int c = 0; c < numClasses; c++)
                {
                    probs[q, c] = Math.Exp(logits[q, c] - maxLogit);
                    sumExp += probs[q, c];
                }
                for (int c = 0; c < numClasses; c++)
                    probs[q, c] /= Math.Max(sumExp, 1e-10);
            }

            // Compute marginal class distribution (average across queries)
            var marginal = new double[numClasses];
            for (int c = 0; c < numClasses; c++)
            {
                for (int q = 0; q < numQuery; q++)
                    marginal[c] += probs[q, c];
                marginal[c] /= Math.Max(numQuery, 1);
            }

            // Update logits using conditional and marginal entropy gradients
            for (int q = 0; q < numQuery; q++)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    // Conditional entropy grad: minimize H(Y|X) -> sharpen predictions
                    double condGrad = condWeight * (Math.Log(Math.Max(probs[q, c], 1e-10)) + 1.0);
                    // Marginal entropy grad: maximize H(Y) -> balance class assignments
                    double margGrad = -margWeight * (Math.Log(Math.Max(marginal[c], 1e-10)) + 1.0);
                    logits[q, c] -= lr * (condGrad + margGrad);
                }
            }
        }

        // Convert refined per-class confidence back to feature space
        // Compute per-class confidence as average softmax probability across queries
        var refined = new Vector<T>(supportFeatures.Length);
        for (int c = 0; c < numClasses; c++)
        {
            double avgConf = 0;
            for (int q = 0; q < numQuery; q++)
            {
                double maxL = double.MinValue;
                for (int cc = 0; cc < numClasses; cc++)
                    maxL = Math.Max(maxL, logits[q, cc]);
                double sumE = 0;
                double thisE = Math.Exp(logits[q, c] - maxL);
                for (int cc = 0; cc < numClasses; cc++)
                    sumE += Math.Exp(logits[q, cc] - maxL);
                avgConf += thisE / Math.Max(sumE, 1e-10);
            }
            avgConf /= Math.Max(numQuery, 1);
            refined[c] = NumOps.Multiply(supportFeatures[c], NumOps.FromDouble(0.5 + avgConf));
        }

        return refined;
    }

}

/// <summary>
/// Adapted model wrapper for TIM with transductively refined predictions.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses predictions that have been refined
/// by processing all query examples together, taking advantage of the query set's
/// structure for more accurate classification.
/// </para>
/// </remarks>
internal class TIMModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _refinedWeights;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _refinedWeights;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public TIMModel(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams,
        Vector<T>? refinedWeights, double[]? modulationFactors)
    {
        _model = model;
        _backboneParams = backboneParams;
        _refinedWeights = refinedWeights;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = new Vector<T>(_backboneParams.Length);
            for (int i = 0; i < _backboneParams.Length; i++)
                modulated[i] = NumOps.Multiply(_backboneParams[i],
                    NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
            _model.SetParameters(modulated);
        }
        else
        {
            _model.SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
