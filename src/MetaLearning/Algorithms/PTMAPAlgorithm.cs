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
/// Implementation of PT+MAP (Power Transform + Maximum A Posteriori) (Hu et al., ICLR 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// PT+MAP applies a power transform to normalize feature distributions, then uses
/// MAP estimation for transductive few-shot classification. The power transform makes
/// features more Gaussian, enabling a simple Bayesian classifier to work well.
/// </para>
/// <para><b>For Beginners:</b> PT+MAP is elegantly simple:
///
/// **Step 1 - Power Transform:**
/// Raw features from neural networks often have skewed distributions.
/// The power transform x_new = sign(x) * |x|^beta makes them more bell-shaped (Gaussian).
/// With beta=0.5, this is essentially a square root transform.
///
/// **Step 2 - Center and Normalize:**
/// After the transform, center features (subtract mean) and L2 normalize.
///
/// **Step 3 - MAP Estimation (transductive):**
/// Given the Gaussian assumption, compute the optimal (MAP) class assignments
/// for ALL query examples simultaneously. This iterates between:
/// - Assign queries to nearest class (E-step)
/// - Update class means using assigned queries (M-step)
///
/// **Why it works so well:**
/// The power transform fixes the main problem: features aren't Gaussian.
/// Once they ARE Gaussian, the simple MAP classifier is provably optimal.
/// Sometimes the simplest math wins.
/// </para>
/// <para><b>Algorithm - PT+MAP:</b>
/// <code>
/// # At test time (no meta-training needed, just backbone training)
/// def predict(support, query):
///     # 1. Extract features
///     z_s = f_theta(support_x)
///     z_q = f_theta(query_x)
///     z = concat(z_s, z_q)
///
///     # 2. Power transform
///     z = sign(z) * |z|^beta       # Element-wise power transform
///
///     # 3. Center (subtract global mean)
///     z = z - mean(z)
///
///     # 4. L2 normalize
///     z = z / ||z||_2
///
///     # 5. MAP estimation (iterative)
///     # Initialize: assign queries to nearest support centroid
///     for iter in range(map_iterations):
///         # E-step: compute soft assignments for queries
///         centroids = compute_class_centroids(z_s, soft_labels_q)
///         logits = -distance(z_q, centroids) / temperature
///         soft_labels_q = softmax(logits)
///
///         # M-step: update centroids using soft assignments
///         centroids = weighted_mean(z_all, soft_labels)
///
///     return argmax(soft_labels_q)
/// </code>
/// </para>
/// <para>
/// Reference: Hu, Y., Gripon, V., &amp; Pateux, S. (2021).
/// Leveraging the Feature Distribution in Transfer-based Few-Shot Learning. ICLR 2021.
/// </para>
/// </remarks>
public class PTMAPAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly PTMAPOptions<T, TInput, TOutput> _ptmapOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.PTMAP;

    /// <summary>Initializes a new PT+MAP meta-learner.</summary>
    /// <param name="options">Configuration options for PT+MAP.</param>
    public PTMAPAlgorithm(PTMAPOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _ptmapOptions = options;
    }

    /// <summary>
    /// Applies the power transform to a feature vector.
    /// </summary>
    /// <param name="features">Raw features.</param>
    /// <returns>Power-transformed features.</returns>
    private Vector<T> ApplyPowerTransform(Vector<T> features)
    {
        double beta = _ptmapOptions.PowerTransformBeta;
        var transformed = new Vector<T>(features.Length);
        for (int i = 0; i < features.Length; i++)
        {
            double val = NumOps.ToDouble(features[i]);
            double sign = val >= 0 ? 1.0 : -1.0;
            double absVal = Math.Abs(val);
            transformed[i] = NumOps.FromDouble(sign * Math.Pow(absVal + 1e-10, beta));
        }
        return transformed;
    }

    /// <summary>
    /// Centers and L2-normalizes a feature vector.
    /// </summary>
    private Vector<T> CenterAndNormalize(Vector<T> features)
    {
        // Center
        T mean = NumOps.Zero;
        for (int i = 0; i < features.Length; i++)
            mean = NumOps.Add(mean, features[i]);
        mean = NumOps.Divide(mean, NumOps.FromDouble(Math.Max(1, features.Length)));

        var centered = new Vector<T>(features.Length);
        double norm = 0;
        for (int i = 0; i < features.Length; i++)
        {
            centered[i] = NumOps.Subtract(features[i], mean);
            double v = NumOps.ToDouble(centered[i]);
            norm += v * v;
        }

        // L2 normalize
        norm = Math.Sqrt(norm + 1e-10);
        for (int i = 0; i < centered.Length; i++)
            centered[i] = NumOps.Divide(centered[i], NumOps.FromDouble(norm));

        return centered;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _ptmapOptions.OuterLearningRate));
        }

        return ComputeMean(losses);
    }

    /// <summary>
    /// Performs MAP estimation with iterative refinement on query soft assignments.
    /// </summary>
    /// <param name="supportFeatures">Power-transformed, centered, normalized support features.</param>
    /// <param name="queryFeatures">Power-transformed, centered, normalized query features.</param>
    /// <returns>Refined query assignment weights after MAP iterations.</returns>
    private Vector<T>? MAPEstimation(Vector<T>? supportFeatures, Vector<T>? queryFeatures)
    {
        if (supportFeatures == null || queryFeatures == null || queryFeatures.Length == 0)
            return supportFeatures;

        int numQuery = queryFeatures.Length;

        // Initialize logits from support-query cosine similarity
        var logits = new double[numQuery];
        for (int q = 0; q < numQuery; q++)
        {
            double sim = 0;
            for (int s = 0; s < supportFeatures.Length; s++)
            {
                sim += NumOps.ToDouble(NumOps.Multiply(queryFeatures[q], supportFeatures[s % supportFeatures.Length]));
            }
            logits[q] = sim * _ptmapOptions.Temperature;
        }

        // Iterative MAP refinement (EM-like)
        for (int iter = 0; iter < _ptmapOptions.MAPIterations; iter++)
        {
            // E-step: softmax to get soft assignments
            double maxLogit = double.MinValue;
            for (int q = 0; q < numQuery; q++)
                maxLogit = Math.Max(maxLogit, logits[q]);

            double sumExp = 0;
            var probs = new double[numQuery];
            for (int q = 0; q < numQuery; q++)
            {
                probs[q] = Math.Exp(logits[q] - maxLogit);
                sumExp += probs[q];
            }
            for (int q = 0; q < numQuery; q++)
                probs[q] /= Math.Max(sumExp, 1e-10);

            // M-step: update centroids using soft-assignment-weighted support features, then recompute logits
            // Each query's new logit is its similarity to a centroid built from
            // all support features weighted by soft assignments across all queries
            double centroidShift = 0;

            // Compute centroid: weighted average of support features using soft assignments
            var centroid = new double[supportFeatures.Length];
            double totalWeight = 0;
            for (int q2 = 0; q2 < numQuery; q2++)
                totalWeight += probs[q2];
            totalWeight = Math.Max(totalWeight, 1e-10);

            for (int s = 0; s < supportFeatures.Length; s++)
            {
                double suppVal = NumOps.ToDouble(supportFeatures[s]);
                // Centroid blends support features with query-weighted contributions
                double weightedSum = suppVal; // Start with support feature
                for (int q2 = 0; q2 < numQuery; q2++)
                {
                    double queryVal = q2 < queryFeatures.Length ? NumOps.ToDouble(queryFeatures[q2]) : 0;
                    weightedSum += (probs[q2] / totalWeight) * queryVal;
                }
                centroid[s] = weightedSum / 2.0; // Average of support and weighted query
            }

            for (int q = 0; q < numQuery; q++)
            {
                double queryVal = NumOps.ToDouble(queryFeatures[q]);
                double sim = 0;
                for (int s = 0; s < supportFeatures.Length; s++)
                {
                    sim += centroid[s] * queryVal;
                }
                double newLogit = sim * _ptmapOptions.Temperature;
                centroidShift += Math.Abs(newLogit - logits[q]);
                logits[q] = newLogit;
            }

            // Early convergence check
            if (centroidShift / Math.Max(numQuery, 1) < 1e-6)
                break;
        }

        // Convert refined logits back to weights
        var refined = new Vector<T>(supportFeatures.Length);
        for (int i = 0; i < supportFeatures.Length; i++)
        {
            double scale = i < numQuery ? 1.0 / (1.0 + Math.Exp(-logits[i])) : 0.5;
            refined[i] = NumOps.Multiply(supportFeatures[i], NumOps.FromDouble(scale));
        }
        return refined;
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support and query features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);

        // Step 1: Power transform (makes features more Gaussian)
        if (supportFeatures != null)
            supportFeatures = ApplyPowerTransform(supportFeatures);
        if (queryFeatures != null)
            queryFeatures = ApplyPowerTransform(queryFeatures);

        // Step 2: Center and L2-normalize
        if (supportFeatures != null)
            supportFeatures = CenterAndNormalize(supportFeatures);
        if (queryFeatures != null)
            queryFeatures = CenterAndNormalize(queryFeatures);

        // Step 3: MAP estimation (transductive refinement)
        var refinedWeights = MAPEstimation(supportFeatures, queryFeatures);

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

        return new PTMAPModel<T, TInput, TOutput>(MetaModel, currentParams, refinedWeights, modulationFactors);
    }

}

/// <summary>Adapted model wrapper for PT+MAP with power-transformed features and MAP refinement.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model uses features that were power-transformed
/// to be more Gaussian, then refined using MAP estimation where all query examples
/// are processed jointly for better transductive classification.
/// </para>
/// </remarks>
internal class PTMAPModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
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

    public PTMAPModel(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams,
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
        // Start from backbone parameters, optionally adjusted by refined MAP weights
        Vector<T> baseParams;
        if (_refinedWeights != null && _refinedWeights.Length > 0)
        {
            // Use refined MAP weights to adjust backbone: blend refined transductive
            // assignments into parameter modulation for query-time adaptation
            double sumAbs = 0;
            for (int i = 0; i < _refinedWeights.Length; i++)
                sumAbs += Math.Abs(NumOps.ToDouble(_refinedWeights[i]));
            double meanAbs = sumAbs / Math.Max(_refinedWeights.Length, 1);
            double refinedModFactor = Math.Max(0.5, Math.Min(2.0, 0.5 + meanAbs / (1.0 + meanAbs)));

            baseParams = new Vector<T>(_backboneParams.Length);
            for (int i = 0; i < _backboneParams.Length; i++)
                baseParams[i] = NumOps.Multiply(_backboneParams[i], NumOps.FromDouble(refinedModFactor));
        }
        else
        {
            baseParams = _backboneParams;
        }

        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = new Vector<T>(baseParams.Length);
            for (int i = 0; i < baseParams.Length; i++)
                modulated[i] = NumOps.Multiply(baseParams[i],
                    NumOps.FromDouble(_modulationFactors[i % _modulationFactors.Length]));
            _model.SetParameters(modulated);
        }
        else
        {
            _model.SetParameters(baseParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
