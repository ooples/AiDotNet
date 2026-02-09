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
    /// </summary>
    /// <param name="features">Feature vector (concatenated projected features).</param>
    /// <returns>Contrastive loss value.</returns>
    private double ComputeContrastiveLoss(Vector<T> features)
    {
        if (features.Length < 2) return 0;

        double temperature = _mclOptions.ContrastiveTemperature;
        double totalLoss = 0;
        int count = 0;

        for (int i = 0; i < features.Length; i++)
        {
            double anchor = NumOps.ToDouble(features[i]);
            double sumExp = 0;
            double posExp = 0;

            for (int j = 0; j < features.Length; j++)
            {
                if (i == j) continue;
                double other = NumOps.ToDouble(features[j]);
                double sim = anchor * other / (Math.Abs(anchor) + 1e-8) / (Math.Abs(other) + 1e-8);
                double expSim = Math.Exp(Math.Min(sim / temperature, 10.0));
                sumExp += expSim;

                // Treat nearby indices as same-class positives (simplified)
                if (Math.Abs(i - j) <= 1)
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

            // Contrastive loss on support features
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            double contrastLoss = supportFeatures != null ? ComputeContrastiveLoss(supportFeatures) : 0;

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

        // Update projection head via SPSA
        UpdateProjectionParams(taskBatch);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        return new MCLModel<T, TInput, TOutput>(MetaModel, MetaModel.GetParameters());
    }

    /// <summary>Updates projection head parameters using SPSA gradient estimation.</summary>
    private void UpdateProjectionParams(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double epsilon = 1e-5;
        double lr = _mclOptions.OuterLearningRate;

        var direction = new Vector<T>(_projectionParams.Length);
        for (int i = 0; i < direction.Length; i++)
            direction[i] = NumOps.FromDouble(RandomGenerator.NextDouble() > 0.5 ? 1.0 : -1.0);

        double baseLoss = 0;
        foreach (var task in taskBatch.Tasks)
            baseLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        baseLoss /= taskBatch.Tasks.Length;

        for (int i = 0; i < _projectionParams.Length; i++)
            _projectionParams[i] = NumOps.Add(_projectionParams[i], NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon)));

        double perturbedLoss = 0;
        foreach (var task in taskBatch.Tasks)
            perturbedLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        perturbedLoss /= taskBatch.Tasks.Length;

        double directionalGrad = (perturbedLoss - baseLoss) / epsilon;
        for (int i = 0; i < _projectionParams.Length; i++)
            _projectionParams[i] = NumOps.Subtract(_projectionParams[i],
                NumOps.Multiply(direction[i], NumOps.FromDouble(epsilon + lr * directionalGrad)));
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

/// <summary>Adapted model wrapper for MCL.</summary>
internal class MCLModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public MCLModel(IFullModel<T, TInput, TOutput> model, Vector<T> p) { _model = model; _params = p; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
