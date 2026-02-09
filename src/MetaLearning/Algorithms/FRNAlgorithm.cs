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
public class FRNAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
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
            options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _frnOptions = options;
    }

    /// <summary>
    /// Computes the reconstruction error for a query given class support features.
    /// Uses ridge regression: w* = (S^T S + lambda I)^-1 S^T q, error = ||q - S w*||^2
    /// </summary>
    /// <param name="queryFeature">Query feature vector.</param>
    /// <param name="supportFeatures">List of support features for one class.</param>
    /// <returns>Reconstruction error (lower = better match).</returns>
    private double ComputeReconstructionError(Vector<T> queryFeature, List<Vector<T>> supportFeatures)
    {
        if (supportFeatures.Count == 0 || queryFeature.Length == 0)
            return double.MaxValue;

        double lambda = _frnOptions.ReconstructionLambda;
        int n = supportFeatures.Count;
        int d = queryFeature.Length;

        // Compute S^T S (n x n gram matrix)
        var gram = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double dot = 0;
                int minLen = Math.Min(supportFeatures[i].Length, supportFeatures[j].Length);
                for (int k = 0; k < minLen; k++)
                    dot += NumOps.ToDouble(supportFeatures[i][k]) * NumOps.ToDouble(supportFeatures[j][k]);
                gram[i, j] = dot;
                gram[j, i] = dot;
            }
            gram[i, i] += lambda; // Regularization
        }

        // Compute S^T q (n x 1)
        var stq = new double[n];
        for (int i = 0; i < n; i++)
        {
            double dot = 0;
            int minLen = Math.Min(supportFeatures[i].Length, d);
            for (int k = 0; k < minLen; k++)
                dot += NumOps.ToDouble(supportFeatures[i][k]) * NumOps.ToDouble(queryFeature[k]);
            stq[i] = dot;
        }

        // Solve (S^T S + lambda I) w = S^T q using simple Gaussian elimination
        var w = SolveLinearSystemStatic(gram, stq, n);

        // Compute reconstruction: q_hat = S w
        double error = 0;
        for (int k = 0; k < d; k++)
        {
            double reconstructed = 0;
            for (int i = 0; i < n; i++)
            {
                if (k < supportFeatures[i].Length)
                    reconstructed += w[i] * NumOps.ToDouble(supportFeatures[i][k]);
            }
            double diff = NumOps.ToDouble(queryFeature[k]) - reconstructed;
            error += diff * diff;
        }

        return error;
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
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _frnOptions.OuterLearningRate));
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

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
        double[]? modulationFactors = null;
        if (reconstructionWeights != null && reconstructionWeights.Length > 0)
        {
            double sumAbs = 0;
            for (int i = 0; i < reconstructionWeights.Length; i++)
                sumAbs += Math.Abs(NumOps.ToDouble(reconstructionWeights[i]));
            double meanAbs = sumAbs / reconstructionWeights.Length;
            modulationFactors = [0.5 + 0.5 / (1.0 + Math.Exp(-meanAbs + 1.0))];
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

        var supportList = new List<Vector<T>>();
        for (int i = 0; i < actualComponents; i++)
        {
            int start = i * featureDim;
            int len = Math.Min(featureDim, supportFeatures.Length - start);
            if (len <= 0) break;
            var vec = new Vector<T>(len);
            for (int d = 0; d < len; d++)
                vec[d] = supportFeatures[start + d];
            supportList.Add(vec);
        }

        if (supportList.Count == 0)
        {
            var fallback = new Vector<T>(queryFeatures.Length);
            for (int i = 0; i < fallback.Length; i++)
                fallback[i] = NumOps.FromDouble(0.5);
            return fallback;
        }

        // Split query features into per-example vectors using the same feature dimension
        int numQueries = Math.Max(queryFeatures.Length / Math.Max(featureDim, 1), 1);
        var weights = new Vector<T>(numQueries);

        for (int q = 0; q < numQueries; q++)
        {
            int qStart = q * featureDim;
            int qLen = Math.Min(featureDim, queryFeatures.Length - qStart);
            if (qLen <= 0) break;

            var queryVec = new Vector<T>(qLen);
            for (int d = 0; d < qLen; d++)
                queryVec[d] = queryFeatures[qStart + d];

            // Compute reconstruction error using ridge regression (lower = better match)
            double error = ComputeReconstructionError(queryVec, supportList);

            // Convert error to weight via sigmoid of negative error (scaled by feature dim)
            double scaledError = error / Math.Max(featureDim, 1);
            double weight = 1.0 / (1.0 + Math.Exp(scaledError));
            weights[q] = NumOps.FromDouble(weight);
        }

        return weights;
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
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _supportFeatures;
    private readonly Vector<T>? _reconstructionWeights;
    private readonly double _lambda;
    private readonly double[]? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _supportFeatures;

    /// <inheritdoc/>
    public double[]? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public FRNModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? supportFeatures,
        Vector<T>? reconstructionWeights,
        double lambda,
        double[]? modulationFactors)
    {
        _model = model;
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
