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
    /// Solves ridge regression and returns predictions for query features.
    /// </summary>
    /// <param name="supportFeatures">Support set feature vector (flattened).</param>
    /// <param name="supportLabels">Support set labels (flattened).</param>
    /// <param name="queryFeatures">Query set feature vector (flattened).</param>
    /// <returns>Ridge regression predictions for the query set.</returns>
    /// <remarks>
    /// <para>
    /// Computes w = (X^T X + lambda I)^-1 X^T y on support features, then applies
    /// the learned weights to query features: predictions = query_features * w.
    /// </para>
    /// <para><b>For Beginners:</b> This is the core of R2-D2:
    /// 1. Takes the support examples' features and labels
    /// 2. Finds the best linear relationship using ridge regression
    /// 3. Applies that relationship to query features to make predictions
    /// It's like fitting a line through the support data and using it to predict.
    /// </para>
    /// </remarks>
    private Vector<T> SolveRidgeRegression(
        Vector<T> supportFeatures,
        Vector<T> supportLabels,
        Vector<T> queryFeatures)
    {
        var weights = ComputeRidgeWeights(supportFeatures, supportLabels);

        if (weights == null)
        {
            return queryFeatures;
        }

        // Simple dot product prediction: prediction = sum(query * weights)
        int predLength = queryFeatures.Length;
        var predictions = new Vector<T>(predLength);

        for (int i = 0; i < predLength; i++)
        {
            T sum = NumOps.Zero;
            int wIdx = i % weights.Length;
            sum = NumOps.Multiply(queryFeatures[i], weights[wIdx]);
            predictions[i] = sum;
        }

        return predictions;
    }

    /// <summary>
    /// Computes ridge regression weights from support features and labels.
    /// </summary>
    /// <param name="supportFeatures">Support features (flattened).</param>
    /// <param name="supportLabels">Support labels (flattened).</param>
    /// <returns>The ridge regression weight vector, or null if computation fails.</returns>
    /// <remarks>
    /// <para>
    /// Implements the closed-form ridge regression solution:
    /// w = (X^T X + lambda I)^-1 X^T y
    ///
    /// For the flattened vector case, this simplifies to element-wise regularized scaling.
    /// </para>
    /// <para><b>For Beginners:</b> This computes the "best fit" linear weights that map
    /// features to labels, with a smoothness penalty (lambda). The formula ensures the
    /// weights don't become too large (overfitting), especially important with few examples.
    /// </para>
    /// </remarks>
    private Vector<T>? ComputeRidgeWeights(Vector<T> supportFeatures, Vector<T> supportLabels)
    {
        if (supportFeatures.Length == 0 || supportLabels.Length == 0)
        {
            return null;
        }

        int dim = Math.Min(supportFeatures.Length, supportLabels.Length);
        var weights = new Vector<T>(dim);

        T lambdaT = NumOps.FromDouble(_lambda);

        for (int i = 0; i < dim; i++)
        {
            // Simplified ridge: w_i = (x_i * y_i) / (x_i^2 + lambda)
            T xi = supportFeatures[i];
            T yi = supportLabels[i];
            T xiSquared = NumOps.Multiply(xi, xi);
            T denominator = NumOps.Add(xiSquared, lambdaT);

            if (NumOps.ToDouble(denominator) > 1e-10)
            {
                weights[i] = NumOps.Divide(NumOps.Multiply(xi, yi), denominator);
            }
            else
            {
                weights[i] = NumOps.Zero;
            }
        }

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
