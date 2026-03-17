using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Super Learner (Stacking) ensemble for optimal model combination.
/// </summary>
/// <remarks>
/// <para>
/// Super Learner combines multiple base models using cross-validated predictions to train
/// an optimal meta-learner. It's proven to perform at least as well as the best single
/// base learner (oracle inequality).
/// </para>
/// <para>
/// <b>For Beginners:</b> Super Learner is an ensemble technique that:
///
/// 1. Takes multiple different models (your "library" of algorithms)
/// 2. Uses cross-validation to see how well each model predicts
/// 3. Learns the best way to combine their predictions
/// 4. Creates a final model that's at least as good as the best individual model
///
/// <b>Key advantage:</b> You don't have to choose which model is best - Super Learner
/// figures that out automatically and combines them optimally.
///
/// <b>Example usage:</b>
/// - Add a linear model (handles linear relationships)
/// - Add a random forest (handles interactions)
/// - Add a neural network (handles complex patterns)
/// - Super Learner learns to use each when appropriate
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Super Learner ensemble combining multiple base models
/// var baseModels = new IFullModel&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;[]
/// {
///     new RidgeRegression&lt;double&gt;(),
///     new MultipleRegression&lt;double&gt;()
/// };
/// var model = new SuperLearner&lt;double&gt;(baseModels);
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 2, new double[] {
///     1, 2,  3, 4,  5, 6,  7, 8,  9, 10,  11, 12 });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with cross-validated optimal meta-learner
/// model.Train(features, targets);
///
/// // Predict using the optimally weighted combination
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 13, 14 });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Super Learner", "https://doi.org/10.2202/1544-6115.1309", Year = 2007, Authors = "Mark J. van der Laan, Eric C. Polley, Alan E. Hubbard")]
public class SuperLearner<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Initializes a new instance with a default base model.
    /// </summary>
    public SuperLearner()
        : this(new IFullModel<T, Matrix<T>, Vector<T>>[] { new AiDotNet.Regression.RidgeRegression<T>() })
    {
    }

    /// <summary>
    /// Base models in the library.
    /// </summary>
    private List<IFullModel<T, Matrix<T>, Vector<T>>> _baseModels;

    /// <summary>
    /// Meta-learner weights or coefficients.
    /// </summary>
    private Vector<T>? _metaWeights;

    /// <summary>
    /// Meta-learner intercept.
    /// </summary>
    private T _metaIntercept;

    /// <summary>
    /// Cross-validation performance of each base model.
    /// </summary>
    private Vector<T>? _cvPerformance;

    /// <summary>
    /// Means of base model predictions (for normalization).
    /// </summary>
    private Vector<T>? _predMeans;

    /// <summary>
    /// Standard deviations of base model predictions (for normalization).
    /// </summary>
    private Vector<T>? _predStds;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly SuperLearnerOptions _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of Super Learner.
    /// </summary>
    /// <param name="baseModels">Collection of base models to combine.</param>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public SuperLearner(
        IEnumerable<IFullModel<T, Matrix<T>, Vector<T>>> baseModels,
        SuperLearnerOptions? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _baseModels = baseModels.ToList();
        _options = options ?? new SuperLearnerOptions();
        _metaIntercept = NumOps.Zero;
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();

        if (_baseModels.Count == 0)
        {
            throw new ArgumentException("At least one base model is required.", nameof(baseModels));
        }
    }

    /// <summary>
    /// Adds a base model to the library.
    /// </summary>
    /// <param name="model">Model to add.</param>
    public void AddBaseModel(IFullModel<T, Matrix<T>, Vector<T>> model)
    {
        _baseModels.Add(model);
    }

    /// <inheritdoc/>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;
        int numModels = _baseModels.Count;

        // Generate cross-validation fold indices
        var foldIndices = GenerateFoldIndices(n);

        // Collect out-of-fold predictions for meta-training
        var metaFeatures = new Matrix<T>(n, numModels);
        _cvPerformance = new Vector<T>(numModels);

        // Train each fold and collect out-of-fold predictions
        for (int fold = 0; fold < _options.NumFolds; fold++)
        {
            var (trainIdx, valIdx) = GetFoldSplit(foldIndices, fold);

            var xTrain = ExtractRows(x, trainIdx);
            var yTrain = ExtractValues(y, trainIdx);
            var xVal = ExtractRows(x, valIdx);

            // Train each base model on this fold
            for (int m = 0; m < numModels; m++)
            {
                // Clone model for this fold
                var model = CloneModel(_baseModels[m]);
                model.Train(xTrain, yTrain);

                // Get out-of-fold predictions
                var predictions = model.Predict(xVal);

                for (int i = 0; i < valIdx.Length; i++)
                {
                    metaFeatures[valIdx[i], m] = predictions[i];
                }

                // Accumulate CV error for performance weighting
                T foldMse = NumOps.Zero;
                for (int i = 0; i < valIdx.Length; i++)
                {
                    T diff = NumOps.Subtract(y[valIdx[i]], predictions[i]);
                    foldMse = NumOps.Add(foldMse, NumOps.Multiply(diff, diff));
                }
                _cvPerformance[m] = NumOps.Add(_cvPerformance[m], foldMse);
            }
        }

        // Finalize CV performance (lower is better)
        T nT = NumOps.FromDouble(n);
        for (int m = 0; m < numModels; m++)
        {
            _cvPerformance[m] = NumOps.Divide(_cvPerformance[m], nT);
        }

        // Normalize base predictions if requested
        if (_options.NormalizeBasePredictions)
        {
            NormalizeMetaFeatures(metaFeatures);
        }

        // Train meta-learner
        TrainMetaLearner(metaFeatures, y);

        // Retrain base models on full data if requested
        if (_options.RetrainOnFullData)
        {
            for (int m = 0; m < numModels; m++)
            {
                _baseModels[m].Train(x, y);
            }
        }
        else
        {
            // Train models on full data (they've only seen folds so far)
            for (int m = 0; m < numModels; m++)
            {
                _baseModels[m].Train(x, y);
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_metaWeights == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        int n = input.Rows;
        int numModels = _baseModels.Count;

        // Get predictions from all base models
        var basePredictions = new Matrix<T>(n, numModels);
        for (int m = 0; m < numModels; m++)
        {
            var preds = _baseModels[m].Predict(input);
            for (int i = 0; i < n; i++)
            {
                basePredictions[i, m] = preds[i];
            }
        }

        // Normalize if we normalized during training
        if (_options.NormalizeBasePredictions && _predMeans != null && _predStds != null)
        {
            NormalizeMetaFeaturesForPrediction(basePredictions);
        }

        // Combine using meta-weights
        var result = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            T combined = _metaIntercept;
            for (int m = 0; m < numModels; m++)
            {
                combined = NumOps.Add(combined, NumOps.Multiply(basePredictions[i, m], _metaWeights[m]));
            }
            result[i] = combined;
        }

        return result;
    }

    /// <summary>
    /// Gets the meta-learner weights for each base model.
    /// </summary>
    /// <returns>Array of weights (higher = more important).</returns>
    public Vector<T> GetMetaWeights()
    {
        return _metaWeights ?? new Vector<T>(0);
    }

    /// <summary>
    /// Gets the cross-validation performance (MSE) of each base model.
    /// </summary>
    /// <returns>Array of MSE values (lower is better).</returns>
    public Vector<T> GetCVPerformance()
    {
        return _cvPerformance ?? new Vector<T>(0);
    }

    /// <summary>
    /// Gets the contribution of each base model based on weights.
    /// </summary>
    /// <returns>Array of contribution percentages.</returns>
    public Vector<T> GetModelContributions()
    {
        if (_metaWeights == null)
        {
            return new Vector<T>(0);
        }

        var absWeights = new Vector<T>(_metaWeights.Length);
        T sum = NumOps.Zero;
        for (int i = 0; i < _metaWeights.Length; i++)
        {
            absWeights[i] = NumOps.Abs(_metaWeights[i]);
            sum = NumOps.Add(sum, absWeights[i]);
        }

        var result = new Vector<T>(_metaWeights.Length);
        T epsilon = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(sum, epsilon))
        {
            T equalWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(_metaWeights.Length));
            for (int i = 0; i < _metaWeights.Length; i++)
            {
                result[i] = equalWeight;
            }
        }
        else
        {
            for (int i = 0; i < _metaWeights.Length; i++)
            {
                result[i] = NumOps.Divide(absWeights[i], sum);
            }
        }

        return result;
    }

    /// <summary>
    /// Generates fold indices for cross-validation.
    /// </summary>
    private int[] GenerateFoldIndices(int n)
    {
        var indices = Enumerable.Range(0, n).ToArray();

        // Shuffle
        for (int i = n - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Assign fold indices
        var foldIndices = new int[n];
        for (int i = 0; i < n; i++)
        {
            foldIndices[indices[i]] = i % _options.NumFolds;
        }

        return foldIndices;
    }

    /// <summary>
    /// Gets the train/validation split for a fold.
    /// </summary>
    private (int[] train, int[] val) GetFoldSplit(int[] foldIndices, int fold)
    {
        var train = new List<int>();
        var val = new List<int>();

        for (int i = 0; i < foldIndices.Length; i++)
        {
            if (foldIndices[i] == fold)
            {
                val.Add(i);
            }
            else
            {
                train.Add(i);
            }
        }

        return ([.. train], [.. val]);
    }

    /// <summary>
    /// Extracts rows from a matrix.
    /// </summary>
    private Matrix<T> ExtractRows(Matrix<T> x, int[] indices)
    {
        var result = new Matrix<T>(indices.Length, x.Columns);
        for (int i = 0; i < indices.Length; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                result[i, j] = x[indices[i], j];
            }
        }
        return result;
    }

    /// <summary>
    /// Extracts values from a vector.
    /// </summary>
    private Vector<T> ExtractValues(Vector<T> y, int[] indices)
    {
        var result = new Vector<T>(indices.Length);
        for (int i = 0; i < indices.Length; i++)
        {
            result[i] = y[indices[i]];
        }
        return result;
    }

    /// <summary>
    /// Clones a model by creating a new instance.
    /// </summary>
    private IFullModel<T, Matrix<T>, Vector<T>> CloneModel(IFullModel<T, Matrix<T>, Vector<T>> model)
    {
        return model.Clone();
    }

    /// <summary>
    /// Normalizes meta-features (base model predictions).
    /// </summary>
    private void NormalizeMetaFeatures(Matrix<T> features)
    {
        int n = features.Rows;
        int m = features.Columns;
        T nT = NumOps.FromDouble(n);
        T epsilon = NumOps.FromDouble(1e-10);

        _predMeans = new Vector<T>(m);
        _predStds = new Vector<T>(m);

        for (int j = 0; j < m; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, features[i, j]);
            }
            _predMeans[j] = NumOps.Divide(sum, nT);

            T sumSq = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(features[i, j], _predMeans[j]);
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
            }
            _predStds[j] = NumOps.Sqrt(NumOps.Divide(sumSq, nT));
            if (NumOps.LessThan(_predStds[j], epsilon))
            {
                _predStds[j] = NumOps.One;
            }

            for (int i = 0; i < n; i++)
            {
                features[i, j] = NumOps.Divide(NumOps.Subtract(features[i, j], _predMeans[j]), _predStds[j]);
            }
        }
    }

    /// <summary>
    /// Normalizes meta-features using stored means/stds.
    /// </summary>
    private void NormalizeMetaFeaturesForPrediction(Matrix<T> features)
    {
        if (_predMeans is null || _predStds is null)
        {
            throw new InvalidOperationException("Normalization parameters not computed. Model must be trained first.");
        }

        int n = features.Rows;
        int m = features.Columns;

        for (int j = 0; j < m; j++)
        {
            for (int i = 0; i < n; i++)
            {
                features[i, j] = NumOps.Divide(NumOps.Subtract(features[i, j], _predMeans[j]), _predStds[j]);
            }
        }
    }

    /// <summary>
    /// Trains the meta-learner.
    /// </summary>
    private void TrainMetaLearner(Matrix<T> metaFeatures, Vector<T> y)
    {
        int n = metaFeatures.Rows;
        int m = metaFeatures.Columns;

        switch (_options.MetaLearnerType)
        {
            case SuperLearnerMetaLearner.SimpleAverage:
                TrainSimpleAverage(m);
                break;

            case SuperLearnerMetaLearner.PerformanceWeighted:
                TrainPerformanceWeighted(m);
                break;

            case SuperLearnerMetaLearner.NonNegativeLeastSquares:
                TrainNNLS(metaFeatures, y);
                break;

            case SuperLearnerMetaLearner.Ridge:
                TrainRidge(metaFeatures, y);
                break;

            case SuperLearnerMetaLearner.LinearRegression:
                TrainLinearRegression(metaFeatures, y);
                break;

            case SuperLearnerMetaLearner.Lasso:
                TrainLasso(metaFeatures, y);
                break;

            default:
                TrainNNLS(metaFeatures, y);
                break;
        }
    }

    /// <summary>
    /// Simple averaging (equal weights).
    /// </summary>
    private void TrainSimpleAverage(int numModels)
    {
        _metaWeights = new Vector<T>(numModels);
        T weight = NumOps.Divide(NumOps.One, NumOps.FromDouble(numModels));
        for (int m = 0; m < numModels; m++)
        {
            _metaWeights[m] = weight;
        }
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Performance-weighted averaging.
    /// </summary>
    private void TrainPerformanceWeighted(int numModels)
    {
        if (_cvPerformance is null)
        {
            throw new InvalidOperationException("CV performance not computed. Train the model first.");
        }

        _metaWeights = new Vector<T>(numModels);
        T epsilon = NumOps.FromDouble(1e-10);

        // Convert MSE to weights (inverse of performance)
        var invMse = new Vector<T>(numModels);
        T sum = NumOps.Zero;
        for (int i = 0; i < numModels; i++)
        {
            invMse[i] = NumOps.Divide(NumOps.One, NumOps.Add(_cvPerformance[i], epsilon));
            sum = NumOps.Add(sum, invMse[i]);
        }

        for (int m = 0; m < numModels; m++)
        {
            _metaWeights[m] = NumOps.Divide(invMse[m], sum);
        }
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Non-negative least squares.
    /// </summary>
    private void TrainNNLS(Matrix<T> X, Vector<T> y)
    {
        int n = X.Rows;
        int m = X.Columns;
        T zero = NumOps.Zero;
        T two = NumOps.FromDouble(2.0);
        T learningRate = NumOps.FromDouble(0.01);
        T nT = NumOps.FromDouble(n);
        T epsilon = NumOps.FromDouble(1e-10);
        T tolerance = NumOps.FromDouble(_options.MetaLearnerTolerance);

        // Initialize weights
        var weights = new Vector<T>(m);
        T initWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(m));
        for (int j = 0; j < m; j++)
        {
            weights[j] = initWeight;
        }

        // Active set method for NNLS
        for (int iter = 0; iter < _options.MetaLearnerMaxIterations; iter++)
        {
            // Compute gradient
            var grad = new Vector<T>(m);
            for (int j = 0; j < m; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    T pred = zero;
                    for (int k = 0; k < m; k++)
                    {
                        pred = NumOps.Add(pred, NumOps.Multiply(X[i, k], weights[k]));
                    }
                    T residual = NumOps.Subtract(pred, y[i]);
                    grad[j] = NumOps.Add(grad[j], NumOps.Divide(NumOps.Multiply(NumOps.Multiply(two, X[i, j]), residual), nT));
                }
            }

            // Projected gradient descent
            T maxChange = zero;
            for (int j = 0; j < m; j++)
            {
                T newWeight = NumOps.Subtract(weights[j], NumOps.Multiply(learningRate, grad[j]));
                // Project to non-negative
                if (NumOps.LessThan(newWeight, zero))
                {
                    newWeight = zero;
                }
                T change = NumOps.Abs(NumOps.Subtract(newWeight, weights[j]));
                if (NumOps.GreaterThan(change, maxChange))
                {
                    maxChange = change;
                }
                weights[j] = newWeight;
            }

            // Normalize weights to sum to 1 (optional but often helpful)
            T sumW = zero;
            for (int j = 0; j < m; j++)
            {
                sumW = NumOps.Add(sumW, weights[j]);
            }
            if (NumOps.GreaterThan(sumW, epsilon))
            {
                for (int j = 0; j < m; j++)
                {
                    weights[j] = NumOps.Divide(weights[j], sumW);
                }
            }

            if (NumOps.LessThan(maxChange, tolerance))
            {
                break;
            }
        }

        _metaWeights = new Vector<T>(weights);
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Ridge regression meta-learner.
    /// </summary>
    private void TrainRidge(Matrix<T> X, Vector<T> y)
    {
        int n = X.Rows;
        int m = X.Columns;
        T lambda = NumOps.FromDouble(_options.MetaLearnerRegularization);
        T nT = NumOps.FromDouble(n);

        // X'X + lambda*I
        var XtX = new Matrix<T>(m, m);
        var Xty = new Vector<T>(m);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                Xty[j] = NumOps.Add(Xty[j], NumOps.Multiply(X[i, j], y[i]));
                for (int k = 0; k < m; k++)
                {
                    XtX[j, k] = NumOps.Add(XtX[j, k], NumOps.Multiply(X[i, j], X[i, k]));
                }
            }
        }

        // Add regularization
        for (int j = 0; j < m; j++)
        {
            XtX[j, j] = NumOps.Add(XtX[j, j], NumOps.Multiply(lambda, nT));
        }

        // Solve (X'X + lambda*I)^(-1) * X'y
        var XtXinv = InvertMatrix(XtX);
        var weights = new Vector<T>(m);

        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < m; k++)
            {
                weights[j] = NumOps.Add(weights[j], NumOps.Multiply(XtXinv[j, k], Xty[k]));
            }
        }

        _metaWeights = new Vector<T>(weights);
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Linear regression meta-learner.
    /// </summary>
    private void TrainLinearRegression(Matrix<T> X, Vector<T> y)
    {
        int n = X.Rows;
        int m = X.Columns;

        // X'X
        var XtX = new Matrix<T>(m, m);
        var Xty = new Vector<T>(m);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                Xty[j] = NumOps.Add(Xty[j], NumOps.Multiply(X[i, j], y[i]));
                for (int k = 0; k < m; k++)
                {
                    XtX[j, k] = NumOps.Add(XtX[j, k], NumOps.Multiply(X[i, j], X[i, k]));
                }
            }
        }

        // Solve (X'X)^(-1) * X'y with small regularization for stability
        T stabilityEpsilon = NumOps.FromDouble(1e-6);
        for (int j = 0; j < m; j++)
        {
            XtX[j, j] = NumOps.Add(XtX[j, j], stabilityEpsilon);
        }

        var XtXinv = InvertMatrix(XtX);
        var weights = new Vector<T>(m);

        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < m; k++)
            {
                weights[j] = NumOps.Add(weights[j], NumOps.Multiply(XtXinv[j, k], Xty[k]));
            }
        }

        _metaWeights = new Vector<T>(weights);
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Lasso regression meta-learner (coordinate descent).
    /// </summary>
    private void TrainLasso(Matrix<T> X, Vector<T> y)
    {
        int n = X.Rows;
        int m = X.Columns;
        T lambda = NumOps.FromDouble(_options.MetaLearnerRegularization);
        T nT = NumOps.FromDouble(n);
        T zero = NumOps.Zero;
        T tolerance = NumOps.FromDouble(_options.MetaLearnerTolerance);
        T lambdaN = NumOps.Multiply(lambda, nT);
        T negLambdaN = NumOps.Negate(lambdaN);

        var weights = new Vector<T>(m);
        T initWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(m));
        for (int j = 0; j < m; j++)
        {
            weights[j] = initWeight;
        }

        // Coordinate descent
        for (int iter = 0; iter < _options.MetaLearnerMaxIterations; iter++)
        {
            T maxChange = zero;

            for (int j = 0; j < m; j++)
            {
                // Compute partial residual
                var residual = new Vector<T>(n);
                for (int i = 0; i < n; i++)
                {
                    residual[i] = y[i];
                    for (int k = 0; k < m; k++)
                    {
                        if (k != j)
                        {
                            residual[i] = NumOps.Subtract(residual[i], NumOps.Multiply(X[i, k], weights[k]));
                        }
                    }
                }

                // Soft thresholding
                T rho = zero;
                T sumXjSq = zero;
                for (int i = 0; i < n; i++)
                {
                    rho = NumOps.Add(rho, NumOps.Multiply(X[i, j], residual[i]));
                    sumXjSq = NumOps.Add(sumXjSq, NumOps.Multiply(X[i, j], X[i, j]));
                }

                T newWeight;
                if (NumOps.LessThan(rho, negLambdaN))
                {
                    newWeight = NumOps.Divide(NumOps.Add(rho, lambdaN), sumXjSq);
                }
                else if (NumOps.GreaterThan(rho, lambdaN))
                {
                    newWeight = NumOps.Divide(NumOps.Subtract(rho, lambdaN), sumXjSq);
                }
                else
                {
                    newWeight = zero;
                }

                T change = NumOps.Abs(NumOps.Subtract(newWeight, weights[j]));
                if (NumOps.GreaterThan(change, maxChange))
                {
                    maxChange = change;
                }
                weights[j] = newWeight;
            }

            if (NumOps.LessThan(maxChange, tolerance))
            {
                break;
            }
        }

        _metaWeights = new Vector<T>(weights);
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Simple matrix inversion using Gaussian elimination.
    /// </summary>
    private Matrix<T> InvertMatrix(Matrix<T> A)
    {
        int n = A.Rows;
        var augmented = new Matrix<T>(n, 2 * n);
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n + i] = NumOps.One;
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(augmented[row, col]), NumOps.Abs(augmented[maxRow, col])))
                {
                    maxRow = row;
                }
            }

            for (int j = 0; j < 2 * n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            T pivot = augmented[col, col];
            if (NumOps.LessThan(NumOps.Abs(pivot), epsilon))
            {
                pivot = epsilon;
            }

            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] = NumOps.Divide(augmented[col, j], pivot);
            }

            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    T factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] = NumOps.Subtract(augmented[row, j], NumOps.Multiply(factor, augmented[col, j]));
                    }
                }
            }
        }

        var inverse = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, n + j];
            }
        }

        return inverse;
    }

    /// <inheritdoc/>
    protected override T PredictSingle(Vector<T> input)
    {
        var matrix = new Matrix<T>(1, input.Length);
        for (int j = 0; j < input.Length; j++)
        {
            matrix[0, j] = input[j];
        }

        var result = Predict(matrix);
        return result[0];
    }

    /// <inheritdoc/>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        Train(x, y);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumBaseModels", _baseModels.Count },
                { "MetaLearnerType", _options.MetaLearnerType.ToString() },
                { "NumFolds", _options.NumFolds },
                { "NumFeatures", _numFeatures }
            }
        };
    }

    /// <summary>
    /// SuperLearner is an ensemble that doesn't support optimizer parameter injection.
    /// </summary>
    public override int ParameterCount => 0;

    /// <summary>
    /// Returns all features since the ensemble uses sub-models on all features.
    /// </summary>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        return Enumerable.Range(0, _numFeatures > 0 ? _numFeatures : 0);
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Options
        writer.Write(_options.NumFolds);
        writer.Write((int)_options.MetaLearnerType);
        writer.Write(_numFeatures);

        // Meta weights
        writer.Write(_metaWeights?.Length ?? 0);
        if (_metaWeights != null)
        {
            foreach (var w in _metaWeights)
            {
                writer.Write(NumOps.ToDouble(w));
            }
        }
        writer.Write(NumOps.ToDouble(_metaIntercept));

        // Normalization params
        writer.Write(_predMeans?.Length ?? 0);
        if (_predMeans != null && _predStds != null)
        {
            foreach (var mean in _predMeans)
            {
                writer.Write(NumOps.ToDouble(mean));
            }
            foreach (var std in _predStds)
            {
                writer.Write(NumOps.ToDouble(std));
            }
        }

        // Note: Base models need to be serialized separately in a real implementation

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        base.Deserialize(reader.ReadBytes(baseLen));

        _options.NumFolds = reader.ReadInt32();
        _options.MetaLearnerType = (SuperLearnerMetaLearner)reader.ReadInt32();
        _numFeatures = reader.ReadInt32();

        int numWeights = reader.ReadInt32();
        if (numWeights > 0)
        {
            _metaWeights = new Vector<T>(numWeights);
            for (int i = 0; i < numWeights; i++)
            {
                _metaWeights[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
        _metaIntercept = NumOps.FromDouble(reader.ReadDouble());

        int numMeans = reader.ReadInt32();
        if (numMeans > 0)
        {
            _predMeans = new Vector<T>(numMeans);
            _predStds = new Vector<T>(numMeans);
            for (int i = 0; i < numMeans; i++)
            {
                _predMeans[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            for (int i = 0; i < numMeans; i++)
            {
                _predStds[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new SuperLearner<T>(_baseModels, _options, Regularization);
    }
}
