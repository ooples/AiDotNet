using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of FRN (Few-shot Classification via Feature Map Reconstruction) (Wertheimer et al., CVPR 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// FRN classifies queries by attempting to reconstruct each query's feature map
/// from the feature maps of each class's support examples. The class whose support
/// features best reconstruct the query is chosen as the predicted class.
/// </para>
/// <para><b>For Beginners:</b> FRN asks "which class can best explain this query?":
///
/// **The core idea:**
/// Instead of comparing features directly (distance), try to RECONSTRUCT the query's
/// features using each class's support features. The class that can best rebuild
/// the query's features is the most likely match.
///
/// **How reconstruction works:**
/// For class k with support features S_k:
/// 1. Find optimal weights w* = argmin ||query - S_k @ w||^2 + lambda * ||w||^2
/// 2. This is ridge regression: w* = (S_k^T @ S_k + lambda*I)^(-1) @ S_k^T @ query
/// 3. Reconstruction: query_hat = S_k @ w*
/// 4. Reconstruction error: ||query - query_hat||^2
///
/// **Why reconstruction is better than distance:**
/// - Distance: "How far is the query from the class center?"
/// - Reconstruction: "Can the class's patterns explain the query's patterns?"
/// - Reconstruction is more expressive because it uses MULTIPLE support examples
///   as building blocks, not just their mean.
///
/// **Example:**
/// Cat class has: tabby, persian, siamese support examples.
/// A new calico query is a MIX of tabby and siamese patterns.
/// Distance to mean might be far, but reconstruction from tabby + siamese is great.
/// FRN correctly classifies it as a cat.
/// </para>
/// <para><b>Algorithm - FRN:</b>
/// <code>
/// # For each query q and each class k:
/// # 1. Collect class k support features: S_k = [s_1, s_2, ..., s_n]
/// # 2. Solve: w* = (S_k^T S_k + lambda I)^-1 S_k^T q
/// # 3. Reconstruct: q_hat_k = S_k w*
/// # 4. Error: e_k = ||q - q_hat_k||^2
/// # 5. Predict: class = argmin_k(e_k)
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         z_s = f_theta(support_x)          # Support features
///         z_q = f_theta(query_x)            # Query features
///
///         for each query q:
///             for each class k:
///                 # Ridge regression reconstruction
///                 S_k = z_s[class == k]
///                 w = (S_k^T S_k + lambda I)^-1 S_k^T q
///                 error_k = ||q - S_k w||^2
///
///             logits = -[error_1, ..., error_K]
///
///         loss = cross_entropy(logits, query_labels)
///
///     theta = theta - lr * grad(loss)
/// </code>
/// </para>
/// <para>
/// Reference: Wertheimer, D., Tang, L., &amp; Hariharan, B. (2021).
/// Few-Shot Classification With Feature Map Reconstruction Networks. CVPR 2021.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Few-Shot Classification With Feature Map Reconstruction Networks",
    "https://arxiv.org/abs/2012.01506",
    Year = 2021,
    Authors = "Davis Wertheimer, Luming Tang, Bharath Hariharan")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class FRNAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly FRNOptions<T, TInput, TOutput> _frnOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.FRN;

    /// <summary>Initializes a new FRN meta-learner.</summary>
    /// <param name="options">Configuration options for FRN.</param>
    public FRNAlgorithm(FRNOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _frnOptions = options;
    }

    /// <summary>Solves a linear system Ax = b using Gaussian elimination.</summary>
    internal static double[] SolveLinearSystemStatic(double[,] A, double[] b, int n)
    {
        // Augmented matrix
        var aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            for (int j = 0; j <= n; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            if (Math.Abs(aug[col, col]) < 1e-12) continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            if (Math.Abs(aug[i, i]) < 1e-12)
            {
                x[i] = 0;
                continue;
            }
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= aug[i, i];
        }
        return x;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _frnOptions.OuterLearningRate);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = ParamModel.GetParameters();

        // Extract support features for reconstruction
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);

        // Extract query features
        var queryPred = MetaModel.Predict(task.QueryInput);
        var queryFeatures = ConvertToVector(queryPred);

        // Compute reconstruction-based classification weights
        // For each query, find the reconstruction error from support features
        Vector<T>? reconstructionWeights = null;
        if (supportFeatures != null && queryFeatures != null)
        {
            reconstructionWeights = ComputeReconstructionWeights(supportFeatures, queryFeatures);
        }

        // Compute modulation from reconstruction weight magnitudes
        Vector<T>? modulationFactors = null;
        if (reconstructionWeights != null && reconstructionWeights.Length > 0)
        {
            modulationFactors = ParameterModulation.MeanAbsGate(Engine, reconstructionWeights);
        }

        return new FRNModel<T, TInput, TOutput>(
            MetaModel, currentParams, supportFeatures, reconstructionWeights,
            _frnOptions.ReconstructionLambda, modulationFactors);
    }

    /// <summary>
    /// Computes reconstruction-based classification weights for query features.
    /// Splits the flat feature vectors into per-example multi-dimensional feature vectors
    /// (using NumComponents to determine the number of support examples), then computes
    /// ridge regression reconstruction error for each query example.
    /// </summary>
    /// <param name="supportFeatures">Support set features (flattened: NumComponents * featureDim).</param>
    /// <param name="queryFeatures">Query set features (flattened).</param>
    /// <returns>Reconstruction weights (sigmoid of negative errors) for classification.</returns>
    private Vector<T> ComputeReconstructionWeights(Vector<T> supportFeatures, Vector<T> queryFeatures)
    {
        int numComponents = _frnOptions.NumComponents;

        // Split support features into per-example multi-dimensional vectors
        // Each support example has featureDim = supportFeatures.Length / numComponents dimensions
        int featureDim = Math.Max(supportFeatures.Length / Math.Max(numComponents, 1), 1);
        int actualComponents = Math.Min(numComponents, supportFeatures.Length / Math.Max(featureDim, 1));
        actualComponents = Math.Max(actualComponents, 1);

        // S [n, featureDim]: a shorter last example is zero-padded, which leaves every dot product unchanged
        var (support, _, n) = ExampleRows.Split(Engine, supportFeatures, actualComponents, featureDim);
        if (n == 0)
        {
            return Engine.Fill(queryFeatures.Length, NumOps.FromDouble(0.5));
        }

        // Split query features into per-example vectors using the same feature dimension
        int numQueries = Math.Max(queryFeatures.Length / Math.Max(featureDim, 1), 1);
        var (queries, queryMask, queryCount) = ExampleRows.Split(Engine, queryFeatures, numQueries, featureDim);
        if (queryCount == 0)
        {
            return new Vector<T>(numQueries);
        }

        // Ridge regression for every query at once: (S S^T + lambda I) W = S Q^T, error_q = ||q - W_q^T S||^2 over
        // the query's own entries (lower = better match)
        var gram = Engine.TensorAdd(
            Engine.TensorMatMul(support, Engine.TensorTranspose(support)),
            Engine.TensorMultiplyScalar(Engine.TensorEye<T>(n), NumOps.FromDouble(_frnOptions.ReconstructionLambda)));
        var coefficients = ExampleRows.SolveSymmetricPositiveDefinite(
            gram, Engine.TensorMatMul(support, Engine.TensorTranspose(queries)));
        var reconstructed = Engine.TensorMatMul(Engine.TensorTranspose(coefficients), support);
        var residual = Engine.TensorMultiply(Engine.TensorSubtract(queries, reconstructed), queryMask);
        var errors = Engine.ReduceSum(Engine.TensorSquare(residual), new[] { 1 }, false);

        // Convert error to weight via sigmoid of negative error (scaled by feature dim)
        var weights = Engine.TensorSigmoid(
            Engine.TensorMultiplyScalar(errors, NumOps.FromDouble(-1.0 / Math.Max(featureDim, 1)))).ToVector();
        return queryCount == numQueries
            ? weights
            : Vector<T>.Concatenate(weights, new Vector<T>(numQueries - queryCount));
    }

}

/// <summary>Adapted model wrapper for FRN with reconstruction-based classification.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model classifies queries by asking "which class's
/// support features can best reconstruct this query?" Lower reconstruction error
/// means a better match. The support features and reconstruction weights from
/// adaptation are stored for reference.
/// </para>
/// </remarks>
internal class FRNModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _supportFeatures;
    private readonly Vector<T>? _reconstructionWeights;
    private readonly double _lambda;
    // Not readonly: a readonly Vector<T> field is not persisted by the generated model state, and these
    // factors belong in the adapted model's checkpoint as the double[] they replace did.
    private Vector<T>? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _supportFeatures;

    /// <inheritdoc/>
    public Vector<T>? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public FRNModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? supportFeatures,
        Vector<T>? reconstructionWeights,
        double lambda,
        Vector<T>? modulationFactors)
    {
        // Predict pushes the adapted parameters into this model, so it is a copy: holding the meta-learner's
        // MetaModel would let every prediction overwrite the meta-model.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).Clone();
        _backboneParams = backboneParams;
        _supportFeatures = supportFeatures;
        _reconstructionWeights = reconstructionWeights;
        _lambda = lambda;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = ParameterModulation.Apply(AiDotNetEngine.Current, _backboneParams, _modulationFactors);
            InterfaceGuard.Parameterizable(_model).SetParameters(modulated);
        }
        else
        {
            InterfaceGuard.Parameterizable(_model).SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
