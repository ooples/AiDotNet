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
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with cross-validated optimal meta-learner
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SuperLearner&lt;double&gt;(baseModels))
///     .Build(features, targets);
///
/// // Predict using the optimally weighted combination
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Super Learner", "https://doi.org/10.2202/1544-6115.1309", Year = 2007, Authors = "Mark J. van der Laan, Eric C. Polley, Alan E. Hubbard")]
public partial class SuperLearner<T> : NonLinearRegressionBase<T>
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
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _metaWeights;

    /// <summary>
    /// Meta-learner intercept.
    /// </summary>
    private T _metaIntercept;

    /// <summary>
    /// Cross-validation performance of each base model.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
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

        // Centering base predictions removes their location. Restore it with the least-squares
        // intercept for the fitted weights so normalization changes conditioning, not predictions.
        if (_options.NormalizeBasePredictions && _metaWeights is not null)
        {
            _metaIntercept = FitMetaIntercept(metaFeatures, y, _metaWeights);
        }

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
            var predRow = new Vector<T>(numModels);
            for (int m = 0; m < numModels; m++) predRow[m] = basePredictions[i, m];
            result[i] = NumOps.Add(_metaIntercept, Engine.DotProduct(predRow, _metaWeights));
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

        // The super learner combines its base models with a CONVEX combination — weights that are
        // non-negative and sum to one (van der Laan, Polley and Hubbard, 2007). Written out, that
        // is exactly a convex quadratic program:
        //
        //     minimize  ½·wᵀ(XᵀX)w − (Xᵀy)ᵀw     subject to  w ≥ 0,  Σw = 1
        //
        // which is ½‖Xw − y‖² expanded, dropping the constant ½‖y‖².
        //
        // This replaces a hand-rolled loop that described itself as an "active set method" but was
        // projected gradient descent with a hardcoded learning rate of 0.01, re-normalizing the
        // weights to sum to one after every step. That normalization silently changed the problem
        // being solved — the fixed point of project-then-rescale is not the constrained optimum —
        // and the gradient was computed with a triple-nested loop that recomputed the full
        // prediction for every (row, column) pair, costing O(iterations · n · m²) where O(n · m)
        // per iteration suffices.
        var quadratic = new Matrix<T>(m, m);
        var linear = new Vector<T>(m);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                linear[j] = NumOps.Subtract(linear[j], NumOps.Multiply(X[i, j], y[i]));
                for (int k = 0; k < m; k++)
                {
                    quadratic[j, k] = NumOps.Add(quadratic[j, k], NumOps.Multiply(X[i, j], X[i, k]));
                }
            }
        }

        var simplexRow = new Matrix<T>(1, m);
        for (int j = 0; j < m; j++) simplexRow[0, j] = NumOps.One;

        var simplexTotal = new Vector<T>(1);
        simplexTotal[0] = NumOps.One;

        var program = new AiDotNet.Solvers.QuadraticProgramming.QuadraticProgram<T>(
            quadratic,
            linear,
            equalityMatrix: simplexRow,
            equalityBounds: simplexTotal,
            lowerBounds: new Vector<T>(m));       // all zeros: w >= 0

        var solver = new AiDotNet.Solvers.QuadraticProgramming.ActiveSetQuadraticProgramSolver<T>(
            new ActiveSetQuadraticProgramSolverOptions
            {
                MaxIterations = _options.MetaLearnerMaxIterations,
                Tolerance = _options.MetaLearnerTolerance,
            });

        var solution = solver.Solve(program);

        if (solution.Status == AiDotNet.Solvers.LinearProgramming.LinearProgramStatus.Optimal &&
            solution.Solution is not null)
        {
            _metaWeights = solution.Solution;
        }
        else
        {
            // The simplex constraint always admits feasible points, so this is unreachable for a
            // well-formed meta-feature matrix; fall back to equal weights rather than leaving the
            // ensemble with no combination rule at all.
            var equalWeights = new Vector<T>(m);
            T uniform = NumOps.Divide(NumOps.One, NumOps.FromDouble(m));
            for (int j = 0; j < m; j++) equalWeights[j] = uniform;
            _metaWeights = equalWeights;
        }

        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Fits the unpenalized intercept for fixed meta-learner weights.
    /// </summary>
    private T FitMetaIntercept(Matrix<T> x, Vector<T> y, Vector<T> weights)
    {
        T residualSum = NumOps.Zero;
        for (int i = 0; i < x.Rows; i++)
        {
            T prediction = NumOps.Zero;
            for (int j = 0; j < x.Columns; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(x[i, j], weights[j]));
            }

            residualSum = NumOps.Add(residualSum, NumOps.Subtract(y[i], prediction));
        }

        return NumOps.Divide(residualSum, NumOps.FromDouble(x.Rows));
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
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector and anything pairing the two by
    /// length saw parameters the model claimed not to have.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Returns all features since the ensemble uses sub-models on all features.
    /// </summary>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        return Enumerable.Range(0, _numFeatures > 0 ? _numFeatures : 0);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new SuperLearner<T>(_baseModels, _options, Regularization);
    }
}
