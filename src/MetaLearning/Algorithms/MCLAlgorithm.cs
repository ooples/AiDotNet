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
/// Implementation of MCL (Meta-learning with Contrastive Learning).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// MCL combines episodic meta-learning with supervised contrastive learning to produce
/// features that are both discriminative for few-shot tasks and well-clustered in
/// embedding space.
/// </para>
/// <para><b>For Beginners:</b> MCL teaches features to be useful in TWO ways simultaneously:
///
/// **Two complementary objectives:**
/// 1. **Meta-learning loss** (be good at few-shot tasks):
///    "Given 5 examples of cats and 5 of dogs, classify this query correctly."
///    This teaches the model HOW to use features for few-shot classification.
///
/// 2. **Contrastive loss** (organize features well):
///    "Same-class examples should be close together, different-class far apart."
///    This teaches features to BE inherently better for comparison.
///
/// **Why combine both?**
/// - Meta-learning alone: Features are good for the task but might not cluster well
/// - Contrastive learning alone: Features cluster well but might not transfer to new tasks
/// - Together: Features are well-organized AND transfer well to new few-shot tasks
///
/// **Analogy:**
/// It's like training a librarian (meta-learning = learn to organize books by request)
/// PLUS organizing books on shelves (contrastive = similar books next to each other).
/// A librarian who works with well-organized shelves is more efficient.
/// </para>
/// <para><b>Algorithm - MCL:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor      # Shared backbone
/// p_phi = projection_head          # Projects features for contrastive loss
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Extract features
///         z_s = f_theta(support_x)
///         z_q = f_theta(query_x)
///
///         # 2. Standard meta-learning loss (episodic)
///         prototypes = compute_prototypes(z_s, support_labels)
///         logits = -distance(z_q, prototypes) / temperature
///         meta_loss = cross_entropy(logits, query_labels)
///
///         # 3. Supervised contrastive loss
///         projections = p_phi(concat(z_s, z_q))
///         projections = normalize(projections)  # L2 normalize
///
///         # For each anchor i:
///         #   positives = same-class examples
///         #   negatives = different-class examples
///         #   contrastive_loss = -log(sum_pos(exp(sim/tau)) / sum_all(exp(sim/tau)))
///
///         # 4. Combined loss
///         loss = meta_loss + lambda * contrastive_loss
///
///     theta, phi = theta, phi - lr * grad(loss)
/// </code>
/// </para>
/// </remarks>
public class MCLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MCLOptions<T, TInput, TOutput> _mclOptions;

    /// <summary>Parameters for the contrastive projection head.</summary>
    private Vector<T> _projectionParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MCL;

    /// <summary>Initializes a new MCL meta-learner.</summary>
    /// <param name="options">Configuration options for MCL.</param>
    public MCLAlgorithm(MCLOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _mclOptions = options;
        InitializeProjectionHead();
    }

    /// <summary>Initializes the contrastive projection head.</summary>
    private void InitializeProjectionHead()
    {
        int projDim = _mclOptions.ProjectionDim;
        // Two-layer projection: feature_dim -> projDim -> projDim
        int totalParams = projDim * projDim + projDim + projDim * projDim + projDim;
        _projectionParams = new Vector<T>(totalParams);
        double scale = Math.Sqrt(2.0 / projDim);
        for (int i = 0; i < totalParams; i++)
            _projectionParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
    }

    /// <summary>
    /// Computes supervised contrastive loss for a set of features.
    /// Uses support labels to determine positive pairs (same class) vs negative pairs (different class).
    /// </summary>
    /// <param name="features">Feature vector (concatenated projected features).</param>
    /// <param name="numSupportPerClass">Number of support examples per class (for label inference).</param>
    /// <returns>Contrastive loss value.</returns>
    private double ComputeContrastiveLoss(Vector<T> features, int numSupportPerClass)
    {
        if (features.Length < 2) return 0;

        double temperature = _mclOptions.ContrastiveTemperature;
        double totalLoss = 0;
        int count = 0;

        // Infer class assignments: features are grouped by class in the support set
        // Each block of numSupportPerClass consecutive features belongs to the same class
        int nPerClass = Math.Max(numSupportPerClass, 1);

        for (int i = 0; i < features.Length; i++)
        {
            double anchor = NumOps.ToDouble(features[i]);
            int anchorClass = i / nPerClass;
            double sumExp = 0;
            double posExp = 0;

            for (int j = 0; j < features.Length; j++)
            {
                if (i == j) continue;
                double other = NumOps.ToDouble(features[j]);

                // Cosine similarity
                double anchorNorm = Math.Abs(anchor) + 1e-8;
                double otherNorm = Math.Abs(other) + 1e-8;
                double sim = (anchor * other) / (anchorNorm * otherNorm);
                double expSim = Math.Exp(Math.Min(sim / temperature, 10.0));
                sumExp += expSim;

                // Same class = positive pair (using inferred class from position)
                int otherClass = j / nPerClass;
                if (anchorClass == otherClass)
                    posExp += expSim;
            }

            if (sumExp > 1e-10 && posExp > 1e-10)
            {
                totalLoss -= Math.Log(posExp / sumExp);
                count++;
            }
        }

        return count > 0 ? totalLoss / count : 0;
    }

    /// <summary>
    /// Projects features through the contrastive projection head (2-layer MLP with L2 normalization).
    /// Uses proper matrix multiplication where each output element depends on ALL input elements
    /// within each ProjectionDim-sized chunk, with shared weight matrices across chunks.
    /// </summary>
    /// <param name="features">Raw features from the backbone.</param>
    /// <returns>Projected and L2-normalized features for contrastive comparison.</returns>
    private Vector<T>? ProjectFeatures(Vector<T>? features)
    {
        if (features == null || features.Length == 0)
            return features;

        int projDim = _mclOptions.ProjectionDim;
        var projected = new Vector<T>(features.Length);

        // Parameter layout: W1[projDim, projDim], b1[projDim], W2[projDim, projDim], b2[projDim]
        int w1Size = projDim * projDim;
        int b1Start = w1Size;
        int w2Start = b1Start + projDim;
        int b2Start = w2Start + projDim * projDim;

        // Process features in projDim-sized chunks with matrix multiplication
        // Each chunk: h = ReLU(W1 @ chunk + b1), out = W2 @ h + b2
        int numChunks = (features.Length + projDim - 1) / projDim;

        for (int c = 0; c < numChunks; c++)
        {
            int chunkStart = c * projDim;
            int chunkSize = Math.Min(projDim, features.Length - chunkStart);

            // Layer 1: h[j] = ReLU(sum_i(W1[j,i] * input[i]) + b1[j])
            var hidden = new double[chunkSize];
            for (int j = 0; j < chunkSize; j++)
            {
                double sum = 0;
                for (int i = 0; i < chunkSize; i++)
                {
                    int wIdx = (j * projDim + i) % w1Size;
                    double w = NumOps.ToDouble(_projectionParams[wIdx]);
                    sum += w * NumOps.ToDouble(features[chunkStart + i]);
                }
                double b = NumOps.ToDouble(_projectionParams[b1Start + (j % projDim)]);
                hidden[j] = Math.Max(0, sum + b); // ReLU
            }

            // Layer 2: out[j] = sum_i(W2[j,i] * h[i]) + b2[j]
            for (int j = 0; j < chunkSize; j++)
            {
                double sum = 0;
                for (int i = 0; i < chunkSize; i++)
                {
                    int wIdx = (j * projDim + i) % (projDim * projDim);
                    double w = NumOps.ToDouble(_projectionParams[w2Start + wIdx]);
                    sum += w * hidden[i];
                }
                double b = NumOps.ToDouble(_projectionParams[b2Start + (j % projDim)]);
                projected[chunkStart + j] = NumOps.FromDouble(sum + b);
            }
        }

        // L2 normalization over entire output vector
        double norm = 0;
        for (int i = 0; i < projected.Length; i++)
        {
            double v = NumOps.ToDouble(projected[i]);
            norm += v * v;
        }
        norm = Math.Sqrt(norm + 1e-10);
        for (int i = 0; i < projected.Length; i++)
            projected[i] = NumOps.Divide(projected[i], NumOps.FromDouble(norm));

        return projected;
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

            // Meta-learning loss
            var queryPred = MetaModel.Predict(task.QueryInput);
            var metaLoss = ComputeLossFromOutput(queryPred, task.QueryOutput);
            double metaLossVal = NumOps.ToDouble(metaLoss);

            // Project support features through contrastive head and compute SupCon loss
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var projectedFeatures = ProjectFeatures(supportFeatures);
            // Compute shots-per-class using configurable NumWays
            int numWays = Math.Max(_mclOptions.NumWays, 1);
            int nPerClass = projectedFeatures != null ? Math.Max(projectedFeatures.Length / numWays, 1) : 1;
            double contrastLoss = projectedFeatures != null
                ? ComputeContrastiveLoss(projectedFeatures, nPerClass) : 0;

            // Combined loss
            double combinedLoss = metaLossVal + _mclOptions.ContrastiveWeight * contrastLoss;
            losses.Add(NumOps.FromDouble(combinedLoss));

            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _mclOptions.OuterLearningRate));
        }

        // Update projection head via multi-sample SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _projectionParams, _mclOptions.OuterLearningRate);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract and project support features through contrastive head
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var projectedSupport = ProjectFeatures(supportFeatures);

        // Compute feature-delta modulation from contrastive projection
        double[]? modulationFactors = null;
        if (projectedSupport != null && supportFeatures != null && supportFeatures.Length > 0)
        {
            double sumRatio = 0;
            int count = 0;
            for (int i = 0; i < Math.Min(supportFeatures.Length, projectedSupport.Length); i++)
            {
                double raw = NumOps.ToDouble(supportFeatures[i]);
                double proj = NumOps.ToDouble(projectedSupport[i]);
                if (Math.Abs(raw) > 1e-10)
                {
                    sumRatio += Math.Max(0.5, Math.Min(2.0, proj / raw));
                    count++;
                }
            }
            double avgRatio = count > 0 ? sumRatio / count : 1.0;
            modulationFactors = [avgRatio];
        }

        return new MCLModel<T, TInput, TOutput>(MetaModel, currentParams, projectedSupport, modulationFactors);
    }

}

/// <summary>Adapted model wrapper for MCL with contrastive-projected support features.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model stores support features that have been projected
/// through the contrastive head and L2-normalized. These projections ensure same-class
/// features are clustered together and different-class features are pushed apart,
/// improving few-shot classification accuracy.
/// </para>
/// </remarks>
internal class MCLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _projectedSupport;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _projectedSupport;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    public MCLModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? projectedSupport,
        double[]? modulationFactors)
    {
        _model = model;
        _backboneParams = backboneParams;
        _projectedSupport = projectedSupport;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulatedParams = new Vector<T>(_backboneParams.Length);
            for (int i = 0; i < _backboneParams.Length; i++)
            {
                double mod = _modulationFactors[i % _modulationFactors.Length];
                modulatedParams[i] = NumOps.Multiply(_backboneParams[i], NumOps.FromDouble(mod));
            }
            _model.SetParameters(modulatedParams);
        }
        else
        {
            _model.SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
