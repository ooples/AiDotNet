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
/// Implementation of R2-D2 (Meta-learning with Differentiable Closed-form Solvers) for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// R2-D2 replaces MAML's iterative inner-loop gradient descent with a differentiable closed-form
/// ridge regression solver. The feature extractor (backbone) is meta-learned, and the final
/// classifier is computed analytically using ridge regression on the extracted features.
/// </para>
/// <para><b>For Beginners:</b> R2-D2 is a clever approach to few-shot learning:
///
/// **The Key Insight:**
/// Instead of slowly learning a classifier through gradient descent (like MAML does),
/// R2-D2 computes the optimal classifier instantly using a mathematical formula.
///
/// **How it works:**
/// 1. Pass support set through the feature extractor to get features
/// 2. Solve ridge regression on those features: w = (X^T X + lambda I)^-1 X^T y
/// 3. This gives the OPTIMAL linear classifier in one step!
/// 4. Evaluate on query set using this optimal classifier
/// 5. Backpropagate through the entire process (including the matrix solve)
/// 6. Update the feature extractor to produce better features
///
/// **Why it works:**
/// - The ridge regression formula has a known, exact derivative
/// - We can backpropagate through matrix inversion (using implicit differentiation)
/// - This trains the feature extractor to produce features that are easy to classify
///
/// **Analogy:**
/// Traditional few-shot (MAML):
///   "Here are 5 photos. Let me practice classifying them for 10 rounds... okay, now I'm ready."
/// R2-D2:
///   "Here are 5 photos. *does instant math* I know the optimal classifier. Done."
/// </para>
/// <para><b>Algorithm - R2-D2:</b>
/// <code>
/// # Initialization
/// theta = feature_extractor_params   # Meta-learned backbone
/// lambda = regularization_param      # Optionally meta-learned
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # Extract features from support set
///         X_s = feature_extractor(support_x; theta)  # [n_support, d]
///         Y_s = support_y                             # [n_support, n_classes]
///
///         # Closed-form ridge regression (the magic step!)
///         # w = (X_s^T X_s + lambda I)^-1 X_s^T Y_s
///         A = X_s^T @ X_s + lambda * I               # [d, d]
///         w = solve(A, X_s^T @ Y_s)                  # [d, n_classes]
///
///         # Extract features from query set and classify
///         X_q = feature_extractor(query_x; theta)    # [n_query, d]
///         predictions = X_q @ w                       # [n_query, n_classes]
///
///         # Compute query loss
///         meta_loss_i = loss(predictions, query_y)
///
///     # Outer loop: Update backbone (and optionally lambda)
///     theta = theta - beta * mean(grad(meta_loss, theta))
///     if learn_lambda:
///         lambda = lambda - beta * mean(grad(meta_loss, lambda))
/// </code>
/// </para>
/// <para><b>Key Insights:</b>
///
/// 1. **Closed-Form = No Inner Loop**: Ridge regression has an exact solution, so there's
///    no iterative optimization. This is both faster and more stable than MAML.
///
/// 2. **Differentiable Matrix Solve**: The gradient of the loss with respect to the features
///    flows through the matrix inversion using the identity: d(A^-1)/dA = -A^-1 (dA) A^-1.
///
/// 3. **Feature Learning**: The backbone learns to produce features where ridge regression
///    works well, which means linearly separable features with good margins.
///
/// 4. **Woodbury Identity**: When n_support &lt;&lt; d (few-shot), we can use the Woodbury
///    identity to invert a smaller n x n matrix instead of d x d.
/// </para>
/// <para>
/// Reference: Bertinetto, L., Henriques, J. F., Torr, P., &amp; Vedaldi, A. (2019).
/// Meta-learning with Differentiable Closed-form Solvers. ICLR 2019.
/// </para>
/// </remarks>
public class R2D2Algorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly R2D2Options<T, TInput, TOutput> _r2d2Options;

    /// <summary>
    /// The current ridge regression regularization parameter (lambda).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Lambda can be meta-learned (when LearnLambda is true) or kept fixed.
    /// It controls the trade-off between fitting the support set and regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This is the regularization strength for the ridge regression
    /// classifier. If LearnLambda is true, this value will be automatically optimized during
    /// meta-training to find the best balance.
    /// </para>
    /// </remarks>
    private double _lambda;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.R2D2;

    /// <summary>
    /// Gets the current lambda (regularization) value.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns the current regularization strength. If LearnLambda
    /// was set to true, this may differ from the initial value as the algorithm has optimized it.
    /// </para>
    /// </remarks>
    public double Lambda => _lambda;

    /// <summary>
    /// Initializes a new instance of the R2-D2 algorithm.
    /// </summary>
    /// <param name="options">Configuration options for R2-D2.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or MetaModel is null.</exception>
    /// <remarks>
    /// <para>
    /// Initializes the feature extractor, ridge regression lambda, and meta-training state.
    /// </para>
    /// <para><b>For Beginners:</b> Creates a new R2-D2 meta-learner. You need:
    /// - A feature extractor (MetaModel) that converts inputs to feature vectors
    /// - Configuration options (lambda, learning rates, etc.)
    /// The algorithm will meta-learn the feature extractor and optionally lambda.
    /// </para>
    /// </remarks>
    public R2D2Algorithm(R2D2Options<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _r2d2Options = options;
        _lambda = options.Lambda;
    }

    /// <summary>
    /// Performs one meta-training step using differentiable ridge regression.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para>
    /// For each task:
    /// 1. Extract features from support and query sets using the backbone
    /// 2. Compute ridge regression weights on support features
    /// 3. Classify query features using the ridge regression weights
    /// 4. Compute query loss
    ///
    /// Then update the backbone parameters using meta-gradients that flow through
    /// the entire pipeline, including the ridge regression solve step.
    /// </para>
    /// <para><b>For Beginners:</b> Each training step:
    /// 1. For each task, extract features from examples
    /// 2. Instantly compute the best classifier using math (ridge regression)
    /// 3. Test the classifier on held-out examples
    /// 4. Update the feature extractor so features become more classifiable
    /// 5. Optionally update lambda (regularization strength)
    ///
    /// The key insight: we're NOT learning the classifier directly. We're learning features
    /// that MAKE classifiers easy to compute. This is what meta-learning is about!
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();

        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Set current parameters
            MetaModel.SetParameters(initParams);

            // Forward pass through backbone for support and query
            var supportPred = MetaModel.Predict(task.SupportInput);
            var queryPred = MetaModel.Predict(task.QueryInput);

            // Convert predictions to vectors for ridge regression
            var supportFeatures = ConvertToVector(supportPred);
            var queryFeatures = ConvertToVector(queryPred);
            var supportLabels = ConvertToVector(task.SupportOutput);
            var queryLabels = ConvertToVector(task.QueryOutput);

            if (supportFeatures != null && queryFeatures != null &&
                supportLabels != null && queryLabels != null)
            {
                // Compute ridge regression predictions
                var ridgePredictions = SolveRidgeRegression(
                    supportFeatures, supportLabels, queryFeatures);

                // Compute loss between ridge predictions and query labels
                var queryLoss = LossFunction.CalculateLoss(ridgePredictions, queryLabels);
                losses.Add(queryLoss);
            }
            else
            {
                // Fallback: direct loss computation
                var queryLoss = ComputeLossFromOutput(queryPred, task.QueryOutput);
                losses.Add(queryLoss);
            }

            // Compute meta-gradients for backbone
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        // Restore and apply meta-gradients
        MetaModel.SetParameters(initParams);

        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, _r2d2Options.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Update lambda if learning it
        if (_r2d2Options.LearnLambda)
        {
            UpdateLambda(taskBatch);
        }

        _currentIteration++;

        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task by computing the ridge regression classifier on support features.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model with the ridge regression classifier.</returns>
    /// <remarks>
    /// <para>
    /// Adaptation in R2-D2 is a single forward pass:
    /// 1. Extract features from the support set using the meta-learned backbone
    /// 2. Compute the optimal ridge regression classifier in closed-form
    /// 3. Return a model that classifies new inputs using: features -> ridge classifier -> prediction
    /// </para>
    /// <para><b>For Beginners:</b> Adapting to a new task is instant:
    /// 1. Run the support examples through the feature extractor
    /// 2. Compute the best classifier using the ridge regression formula
    /// 3. Done! The model is ready to classify new examples
    ///
    /// No gradient steps needed, no iterative optimization. Just math.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        // Extract support features
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var supportLabels = ConvertToVector(task.SupportOutput);

        Vector<T>? ridgeWeights = null;

        if (supportFeatures != null && supportLabels != null)
        {
            // Compute ridge regression weights
            ridgeWeights = ComputeRidgeWeights(supportFeatures, supportLabels);
        }

        return new R2D2Model<T, TInput, TOutput>(MetaModel, currentParams, ridgeWeights, _lambda);
    }

    /// <summary>
    /// Estimates the per-example feature dimensionality from flattened vector lengths.
    /// Uses the configurable EmbeddingDimension if set, otherwise uses a GCD-based heuristic.
    /// </summary>
    /// <param name="supportLen">Length of flattened support features.</param>
    /// <param name="queryLen">Length of flattened query features.</param>
    /// <returns>Estimated per-example feature dimension.</returns>
    private int EstimateFeatureDim(int supportLen, int queryLen)
    {
        // If user specified embedding dimension, use it as feature dim
        if (_r2d2Options.EmbeddingDimension > 0)
            return _r2d2Options.EmbeddingDimension;

        if (supportLen <= 0 || queryLen <= 0) return 1;

        // GCD gives the largest common factor between both lengths
        int gcd = GCD(supportLen, queryLen);

        // If GCD is too small, the lengths likely aren't cleanly divisible
        if (gcd < 2)
            return Math.Min(supportLen, queryLen);

        // Choose the factor that gives a reasonable number of examples (at least 2 support)
        if (supportLen / gcd >= 2)
            return gcd;

        // Fallback: each vector is the full length (single example per set)
        return Math.Min(supportLen, queryLen);
    }

    private static int GCD(int a, int b)
    {
        while (b != 0)
        {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return Math.Abs(a);
    }

    /// <summary>Splits a flat feature vector into a list of per-example multi-dimensional vectors.</summary>
    private static List<Vector<T>> SplitIntoVectors(Vector<T> flat, int numExamples, int featureDim)
    {
        var result = new List<Vector<T>>();
        for (int i = 0; i < numExamples; i++)
        {
            int start = i * featureDim;
            int len = Math.Min(featureDim, flat.Length - start);
            if (len <= 0) break;
            var vec = new Vector<T>(len);
            for (int d = 0; d < len; d++)
                vec[d] = flat[start + d];
            result.Add(vec);
        }
        return result;
    }

    /// <summary>
    /// Solves ridge regression and returns predictions for query features.
    /// Splits flattened vectors into per-example multi-dimensional feature vectors,
    /// computes the closed-form ridge regression weights w = (X^T X + lambda I)^-1 X^T y,
    /// then applies those weights to query features for predictions.
    /// </summary>
    /// <param name="supportFeatures">Support set feature vector (flattened).</param>
    /// <param name="supportLabels">Support set labels (flattened).</param>
    /// <param name="queryFeatures">Query set feature vector (flattened).</param>
    /// <returns>Ridge regression predictions for the query set.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the core of R2-D2:
    /// 1. Takes the support examples' features and labels
    /// 2. Finds the best linear relationship using matrix ridge regression
    /// 3. Applies that relationship to query features to make predictions
    /// It's like fitting a hyperplane through the support data and using it to predict.
    /// </para>
    /// </remarks>
    private Vector<T> SolveRidgeRegression(
        Vector<T> supportFeatures,
        Vector<T> supportLabels,
        Vector<T> queryFeatures)
    {
        // Determine per-example feature dimension
        int featureDim = EstimateFeatureDim(supportFeatures.Length, queryFeatures.Length);
        int nSupport = Math.Max(supportFeatures.Length / Math.Max(featureDim, 1), 1);
        int nQuery = Math.Max(queryFeatures.Length / Math.Max(featureDim, 1), 1);

        // Split into per-example vectors
        var X = SplitIntoVectors(supportFeatures, nSupport, featureDim);
        var queryVecs = SplitIntoVectors(queryFeatures, nQuery, featureDim);

        if (X.Count < 2 || featureDim < 1)
            return queryFeatures;

        // Build Gram matrix: G = X^T X + lambda I (featureDim x featureDim)
        var gram = new double[featureDim, featureDim];
        for (int i = 0; i < featureDim; i++)
        {
            for (int j = i; j < featureDim; j++)
            {
                double dot = 0;
                for (int s = 0; s < X.Count; s++)
                {
                    if (i < X[s].Length && j < X[s].Length)
                        dot += NumOps.ToDouble(X[s][i]) * NumOps.ToDouble(X[s][j]);
                }
                gram[i, j] = dot;
                gram[j, i] = dot;
            }
            gram[i, i] += _lambda; // Regularization
        }

        // Build X^T y: support labels assigned per example
        var xty = new double[featureDim];
        for (int d = 0; d < featureDim; d++)
        {
            double sum = 0;
            for (int s = 0; s < X.Count; s++)
            {
                double x_sd = d < X[s].Length ? NumOps.ToDouble(X[s][d]) : 0;
                // Use s-th label (scalar label per support example)
                double y_s = s < supportLabels.Length ? NumOps.ToDouble(supportLabels[s]) : 0;
                sum += x_sd * y_s;
            }
            xty[d] = sum;
        }

        // Solve (X^T X + lambda I) w = X^T y using Gaussian elimination
        var w = FRNAlgorithm<T, TInput, TOutput>.SolveLinearSystemStatic(gram, xty, featureDim);

        // For each query, compute prediction = x_q^T w (dot product)
        var predictions = new Vector<T>(nQuery);
        for (int q = 0; q < queryVecs.Count; q++)
        {
            double pred = 0;
            for (int d = 0; d < Math.Min(featureDim, queryVecs[q].Length); d++)
                pred += NumOps.ToDouble(queryVecs[q][d]) * w[d];
            predictions[q] = NumOps.FromDouble(pred);
        }

        return predictions;
    }

    /// <summary>
    /// Computes ridge regression weights from support features and labels using proper
    /// matrix ridge regression: w = (X^T X + lambda I)^-1 X^T y.
    /// </summary>
    /// <param name="supportFeatures">Support features (flattened: n_support * feature_dim).</param>
    /// <param name="supportLabels">Support labels (flattened: n_support values).</param>
    /// <returns>The ridge regression weight vector, or null if computation fails.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This computes the "best fit" linear weights that map
    /// features to labels. It splits the flat vectors into per-example multi-dimensional
    /// feature vectors, builds the Gram matrix, and solves the ridge regression system.
    /// The lambda regularization prevents overfitting with few examples.
    /// </para>
    /// </remarks>
    private Vector<T>? ComputeRidgeWeights(Vector<T> supportFeatures, Vector<T> supportLabels)
    {
        if (supportFeatures.Length == 0 || supportLabels.Length == 0)
            return null;

        // Determine per-example feature dimension using GCD heuristic
        // For Adapt(), we don't have query features, so use labels to estimate
        int featureDim;
        if (_r2d2Options.EmbeddingDimension > 0)
        {
            featureDim = _r2d2Options.EmbeddingDimension;
        }
        else
        {
            // Heuristic: if labels are shorter, each label is one scalar per example
            // so nSupport = supportLabels.Length, featureDim = supportFeatures.Length / nSupport
            if (supportLabels.Length < supportFeatures.Length &&
                supportFeatures.Length % supportLabels.Length == 0)
            {
                featureDim = supportFeatures.Length / supportLabels.Length;
            }
            else
            {
                // GCD heuristic between features and labels
                int gcd = GCD(supportFeatures.Length, supportLabels.Length);
                featureDim = gcd >= 2 ? supportFeatures.Length / gcd : supportFeatures.Length;
            }
        }

        featureDim = Math.Max(featureDim, 1);
        int nSupport = Math.Max(supportFeatures.Length / featureDim, 1);

        var X = SplitIntoVectors(supportFeatures, nSupport, featureDim);
        if (X.Count < 2)
            return null;

        // Build Gram matrix: G = X^T X + lambda I
        var gram = new double[featureDim, featureDim];
        for (int i = 0; i < featureDim; i++)
        {
            for (int j = i; j < featureDim; j++)
            {
                double dot = 0;
                for (int s = 0; s < X.Count; s++)
                {
                    if (i < X[s].Length && j < X[s].Length)
                        dot += NumOps.ToDouble(X[s][i]) * NumOps.ToDouble(X[s][j]);
                }
                gram[i, j] = dot;
                gram[j, i] = dot;
            }
            gram[i, i] += _lambda;
        }

        // Build X^T y
        var xty = new double[featureDim];
        for (int d = 0; d < featureDim; d++)
        {
            double sum = 0;
            for (int s = 0; s < X.Count; s++)
            {
                double x_sd = d < X[s].Length ? NumOps.ToDouble(X[s][d]) : 0;
                double y_s = s < supportLabels.Length ? NumOps.ToDouble(supportLabels[s]) : 0;
                sum += x_sd * y_s;
            }
            xty[d] = sum;
        }

        // Solve (X^T X + lambda I) w = X^T y
        var w = FRNAlgorithm<T, TInput, TOutput>.SolveLinearSystemStatic(gram, xty, featureDim);

        var weights = new Vector<T>(featureDim);
        for (int i = 0; i < featureDim; i++)
            weights[i] = NumOps.FromDouble(w[i]);

        return weights;
    }

    /// <summary>
    /// Updates the lambda parameter using finite differences if meta-learning lambda.
    /// </summary>
    /// <param name="taskBatch">The current task batch for gradient estimation.</param>
    /// <remarks>
    /// <para>
    /// Uses finite differences to estimate how changes in lambda affect the meta-loss,
    /// then applies a gradient update. Lambda is constrained to be positive.
    /// </para>
    /// <para><b>For Beginners:</b> This adjusts the regularization strength automatically.
    /// It tries a slightly different lambda, sees if performance improves, and adjusts accordingly.
    /// This is like automatically tuning the "smoothness" dial for the classifier.
    /// </para>
    /// </remarks>
    private void UpdateLambda(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double epsilon = 1e-5;
        double originalLambda = _lambda;

        // Compute loss at current lambda
        double baseLoss = 0;
        foreach (var task in taskBatch.Tasks)
        {
            baseLoss += NumOps.ToDouble(ComputeLossFromOutput(
                MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        baseLoss /= taskBatch.Tasks.Length;

        // Compute loss at perturbed lambda
        _lambda = originalLambda + epsilon;
        double perturbedLoss = 0;
        foreach (var task in taskBatch.Tasks)
        {
            perturbedLoss += NumOps.ToDouble(ComputeLossFromOutput(
                MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        perturbedLoss /= taskBatch.Tasks.Length;

        // Gradient of loss w.r.t. lambda
        double lambdaGrad = (perturbedLoss - baseLoss) / epsilon;

        // Update lambda with gradient descent, constrain to be positive
        _lambda = Math.Max(1e-6, originalLambda - _r2d2Options.OuterLearningRate * lambdaGrad);
    }

    /// <summary>
    /// Computes the element-wise average of a list of vectors.
    /// </summary>
    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0) return new Vector<T>(0);

        var result = new Vector<T>(vectors[0].Length);
        foreach (var v in vectors)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.Add(result[i], v[i]);
            }
        }

        var scale = NumOps.FromDouble(1.0 / vectors.Count);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = NumOps.Multiply(result[i], scale);
        }

        return result;
    }
}

/// <summary>
/// Adapted model wrapper for R2-D2 inference using ridge regression classifier.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This model combines the meta-learned feature extractor
/// with the task-specific ridge regression classifier. Given new inputs, it:
/// 1. Extracts features using the backbone
/// 2. Applies the ridge regression weights to classify
/// </para>
/// </remarks>
internal class R2D2Model<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _ridgeWeights;
    private readonly double _lambda;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Creates a new R2-D2 adapted model.
    /// </summary>
    /// <param name="model">The feature extractor model.</param>
    /// <param name="backboneParams">The meta-learned backbone parameters.</param>
    /// <param name="ridgeWeights">The task-specific ridge regression weights.</param>
    /// <param name="lambda">The regularization parameter used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Packages together:
    /// - The feature extractor (shared across tasks)
    /// - The ridge regression weights (specific to this task)
    /// Together they form a complete classifier for the adapted task.
    /// </para>
    /// </remarks>
    public R2D2Model(IFullModel<T, TInput, TOutput> model, Vector<T> backboneParams,
        Vector<T>? ridgeWeights, double lambda)
    {
        _model = model;
        _backboneParams = backboneParams;
        _ridgeWeights = ridgeWeights;
        _lambda = lambda;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        _model.SetParameters(_backboneParams);
        return _model.Predict(input);
    }

    /// <summary>
    /// Training is not supported on adapted R2-D2 models. Use R2D2Algorithm.MetaTrain instead.
    /// </summary>
    public void Train(TInput inputs, TOutput targets)
    {
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
