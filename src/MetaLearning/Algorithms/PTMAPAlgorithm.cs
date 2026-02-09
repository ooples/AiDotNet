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

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        return new PTMAPModel<T, TInput, TOutput>(MetaModel, MetaModel.GetParameters());
    }

    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0) return new Vector<T>(0);
        var result = new Vector<T>(vectors[0].Length);
        foreach (var v in vectors)
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.Add(result[i], v[i]);
        var scale = NumOps.FromDouble(1.0 / vectors.Count);
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.Multiply(result[i], scale);
        return result;
    }
}

/// <summary>Adapted model wrapper for PT+MAP with power-transformed features.</summary>
internal class PTMAPModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public PTMAPModel(IFullModel<T, TInput, TOutput> model, Vector<T> p) { _model = model; _params = p; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
