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
        var w = SolveLinearSystem(gram, stq, n);

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
    private static double[] SolveLinearSystem(double[,] A, double[] b, int n)
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

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        return new FRNModel<T, TInput, TOutput>(MetaModel, MetaModel.GetParameters());
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

/// <summary>Adapted model wrapper for FRN with reconstruction-based classification.</summary>
internal class FRNModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _params;
    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();
    public FRNModel(IFullModel<T, TInput, TOutput> model, Vector<T> p) { _model = model; _params = p; }
    /// <inheritdoc/>
    public TOutput Predict(TInput input) { _model.SetParameters(_params); return _model.Predict(input); }
    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }
    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
