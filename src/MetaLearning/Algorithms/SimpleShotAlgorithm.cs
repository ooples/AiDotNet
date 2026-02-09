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
/// Implementation of SimpleShot for few-shot learning via nearest-centroid classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// SimpleShot shows that a well-trained feature extractor combined with simple feature
/// normalization and nearest-centroid classification can match or exceed many complex
/// meta-learning algorithms. No inner-loop adaptation is needed.
/// </para>
/// <para><b>For Beginners:</b> SimpleShot is the "surprisingly effective baseline":
///
/// **How it works:**
/// 1. Train a feature extractor normally (no episodic training needed)
/// 2. For each new task:
///    a. Extract features from all support examples
///    b. Normalize features (L2 or centered L2)
///    c. Compute class centroids (average feature per class)
///    d. Classify queries by nearest centroid
///
/// **Why it matters:**
/// SimpleShot demonstrates that much of few-shot learning performance comes from
/// having a good feature extractor, not from complex adaptation mechanisms.
/// Many SOTA methods' improvements come from better features, not better meta-learning.
///
/// **When to use:**
/// - As a strong baseline before trying complex methods
/// - When you need fast inference with no adaptation cost
/// - When you have a good pretrained feature extractor
/// </para>
/// <para><b>Algorithm - SimpleShot:</b>
/// <code>
/// # Training (standard, no episodes needed)
/// Train f_theta on base classes with standard cross-entropy
///
/// # For each few-shot task:
///     # 1. Extract features
///     z_s = f_theta(support_x)     # Support features [N*K, d]
///     z_q = f_theta(query_x)       # Query features [N*Q, d]
///
///     # 2. Normalize features
///     if CL2N:
///         mean = mean(z_s)          # Compute centroid of all support features
///         z_s = (z_s - mean) / ||z_s - mean||   # Center then L2 normalize
///         z_q = (z_q - mean) / ||z_q - mean||
///     elif L2:
///         z_s = z_s / ||z_s||
///         z_q = z_q / ||z_q||
///
///     # 3. Compute class centroids
///     c_k = mean(z_s[class == k])   # Centroid for class k
///
///     # 4. Classify by nearest centroid
///     predictions = argmin_k distance(z_q, c_k)
/// </code>
/// </para>
/// <para>
/// Reference: Wang, Y., Chao, W.L., Weinberger, K.Q., &amp; van der Maaten, L. (2019).
/// SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning.
/// </para>
/// </remarks>
public class SimpleShotAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly SimpleShotOptions<T, TInput, TOutput> _simpleShotOptions;

    /// <summary>
    /// Cached feature mean for CL2N normalization, computed from training features.
    /// </summary>
    private Vector<T>? _featureMean;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SimpleShot;

    /// <summary>
    /// Initializes a new SimpleShot meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for SimpleShot.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a SimpleShot instance with a feature extractor.
    /// SimpleShot trains the feature extractor normally (no episodic training), then
    /// uses nearest-centroid classification with normalized features for few-shot tasks.
    /// </para>
    /// </remarks>
    public SimpleShotAlgorithm(SimpleShotOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _simpleShotOptions = options;
    }

    /// <summary>
    /// Performs one meta-training step for SimpleShot.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para>
    /// SimpleShot trains the backbone with standard classification loss.
    /// No episodic training is needed, but we use the meta-learning framework
    /// for consistency. The backbone learns to produce good features that enable
    /// nearest-centroid classification.
    /// </para>
    /// <para><b>For Beginners:</b> Training is straightforward:
    /// 1. For each task, predict on support and query examples
    /// 2. Compute classification loss
    /// 3. Update the feature extractor with gradient descent
    /// No inner-loop adaptation is needed - just standard training.
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

            // Standard forward pass and loss computation
            var queryLoss = ComputeLossFromOutput(
                MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            // Compute gradients for the backbone
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, _simpleShotOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Update feature mean for CL2N normalization
        UpdateFeatureMean(taskBatch);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task using nearest-centroid classification with normalized features.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model that classifies by nearest centroid.</returns>
    /// <remarks>
    /// <para>
    /// Adaptation in SimpleShot is simply:
    /// 1. Extract and normalize support features
    /// 2. Compute class centroids
    /// 3. Store centroids for query-time classification
    ///
    /// No gradient descent or optimization is needed.
    /// </para>
    /// <para><b>For Beginners:</b> Adaptation is instant and simple:
    /// 1. Run support examples through the feature extractor
    /// 2. Normalize the features (center and scale)
    /// 3. Average features per class to get centroids
    /// 4. Done - queries are classified by finding the nearest centroid
    ///
    /// This is why SimpleShot is so fast - adaptation is just feature extraction
    /// and averaging, no gradient computation needed.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Normalize features
        var normalizedFeatures = NormalizeFeatures(supportFeatures);

        // Compute modulation from normalized vs raw support features
        double[]? modulationFactors = null;
        if (supportFeatures != null && normalizedFeatures != null)
        {
            double sumRatio = 0;
            int count = 0;
            for (int i = 0; i < Math.Min(supportFeatures.Length, normalizedFeatures.Length); i++)
            {
                double rawVal = NumOps.ToDouble(supportFeatures[i]);
                double adaptedVal = NumOps.ToDouble(normalizedFeatures[i]);
                if (Math.Abs(rawVal) > 1e-10)
                {
                    sumRatio += Math.Max(0.5, Math.Min(2.0, adaptedVal / rawVal));
                    count++;
                }
            }
            if (count > 0)
                modulationFactors = [sumRatio / count];
        }

        return new SimpleShotModel<T, TInput, TOutput>(
            MetaModel, currentParams, normalizedFeatures, _featureMean,
            _simpleShotOptions.NormalizationType, _simpleShotOptions.DistanceMetric, modulationFactors);
    }

    /// <summary>
    /// Normalizes features using the configured normalization method.
    /// </summary>
    /// <param name="features">Raw feature vector to normalize.</param>
    /// <returns>Normalized feature vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Normalization makes features comparable:
    /// - L2: Scales features to unit length (direction matters, not magnitude)
    /// - CL2N: Centers features first (removes bias), then scales to unit length
    /// Without normalization, some features could dominate the distance computation
    /// simply because they have larger magnitudes.
    /// </para>
    /// </remarks>
    private Vector<T>? NormalizeFeatures(Vector<T>? features)
    {
        if (features == null || features.Length == 0)
        {
            return features;
        }

        string normType = _simpleShotOptions.NormalizationType.ToUpperInvariant();

        if (normType == "CL2N" && _featureMean != null)
        {
            // Center: subtract mean
            var centered = new Vector<T>(features.Length);
            for (int i = 0; i < features.Length; i++)
            {
                int meanIdx = i % _featureMean.Length;
                centered[i] = NumOps.Subtract(features[i], _featureMean[meanIdx]);
            }
            features = centered;
        }

        if (normType == "L2" || normType == "CL2N")
        {
            // L2 normalize
            T normSq = NumOps.Zero;
            for (int i = 0; i < features.Length; i++)
            {
                normSq = NumOps.Add(normSq, NumOps.Multiply(features[i], features[i]));
            }
            double norm = Math.Sqrt(NumOps.ToDouble(normSq));
            if (norm > 1e-10)
            {
                var normalized = new Vector<T>(features.Length);
                for (int i = 0; i < features.Length; i++)
                {
                    normalized[i] = NumOps.Divide(features[i], NumOps.FromDouble(norm));
                }
                return normalized;
            }
        }

        return features;
    }

    /// <summary>
    /// Updates the cached feature mean from the current task batch.
    /// </summary>
    /// <param name="taskBatch">The current task batch.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> For CL2N normalization, we need to know the "average"
    /// feature vector so we can center all features around zero. This method computes
    /// that average from the training data and stores it for use during adaptation.
    /// </para>
    /// </remarks>
    private void UpdateFeatureMean(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (_simpleShotOptions.NormalizationType.ToUpperInvariant() != "CL2N")
        {
            return;
        }

        var allFeatures = new List<Vector<T>>();
        foreach (var task in taskBatch.Tasks)
        {
            var features = ConvertToVector(MetaModel.Predict(task.SupportInput));
            if (features != null)
            {
                allFeatures.Add(features);
            }
        }

        if (allFeatures.Count > 0)
        {
            int dim = allFeatures[0].Length;
            _featureMean = new Vector<T>(dim);
            foreach (var feat in allFeatures)
            {
                for (int i = 0; i < dim && i < feat.Length; i++)
                {
                    _featureMean[i] = NumOps.Add(_featureMean[i], feat[i]);
                }
            }
            var scale = NumOps.FromDouble(1.0 / allFeatures.Count);
            for (int i = 0; i < dim; i++)
            {
                _featureMean[i] = NumOps.Multiply(_featureMean[i], scale);
            }
        }
    }

}

/// <summary>
/// Adapted model wrapper for SimpleShot using nearest-centroid classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model classifies new examples by:
/// 1. Extracting features using the trained backbone
/// 2. Normalizing the features
/// 3. Finding the nearest class centroid (stored from support set)
/// It's fast and simple - just one forward pass plus a distance computation.
/// </para>
/// </remarks>
internal class SimpleShotModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _supportFeatures;
    private readonly Vector<T>? _featureMean;
    private readonly string _normalizationType;
    private readonly string _distanceMetric;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _supportFeatures;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public SimpleShotModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? supportFeatures,
        Vector<T>? featureMean,
        string normalizationType,
        string distanceMetric,
        double[]? modulationFactors)
    {
        _model = model;
        _backboneParams = backboneParams;
        _supportFeatures = supportFeatures;
        _featureMean = featureMean;
        _normalizationType = normalizationType;
        _distanceMetric = distanceMetric;
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
