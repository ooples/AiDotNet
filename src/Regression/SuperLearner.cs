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
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SuperLearner<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Base models in the library.
    /// </summary>
    private List<IFullModel<T, Matrix<T>, Vector<T>>> _baseModels;

    /// <summary>
    /// Meta-learner weights or coefficients.
    /// </summary>
    private T[]? _metaWeights;

    /// <summary>
    /// Meta-learner intercept.
    /// </summary>
    private T _metaIntercept;

    /// <summary>
    /// Cross-validation performance of each base model.
    /// </summary>
    private double[]? _cvPerformance;

    /// <summary>
    /// Means of base model predictions (for normalization).
    /// </summary>
    private double[]? _predMeans;

    /// <summary>
    /// Standard deviations of base model predictions (for normalization).
    /// </summary>
    private double[]? _predStds;

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
        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();

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
        var metaFeatures = new double[n, numModels];
        var yData = y.Select(yi => NumOps.ToDouble(yi)).ToArray();
        _cvPerformance = new double[numModels];

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
                    metaFeatures[valIdx[i], m] = NumOps.ToDouble(predictions[i]);
                }

                // Accumulate CV error for performance weighting
                double foldMse = 0;
                for (int i = 0; i < valIdx.Length; i++)
                {
                    double diff = yData[valIdx[i]] - NumOps.ToDouble(predictions[i]);
                    foldMse += diff * diff;
                }
                _cvPerformance[m] += foldMse;
            }
        }

        // Finalize CV performance (lower is better)
        for (int m = 0; m < numModels; m++)
        {
            _cvPerformance[m] /= n;
        }

        // Normalize base predictions if requested
        if (_options.NormalizeBasePredictions)
        {
            NormalizeMetaFeatures(metaFeatures);
        }

        // Train meta-learner
        TrainMetaLearner(metaFeatures, yData);

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
        var basePredictions = new double[n, numModels];
        for (int m = 0; m < numModels; m++)
        {
            var preds = _baseModels[m].Predict(input);
            for (int i = 0; i < n; i++)
            {
                basePredictions[i, m] = NumOps.ToDouble(preds[i]);
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
            double combined = NumOps.ToDouble(_metaIntercept);
            for (int m = 0; m < numModels; m++)
            {
                combined += basePredictions[i, m] * NumOps.ToDouble(_metaWeights[m]);
            }
            result[i] = NumOps.FromDouble(combined);
        }

        return result;
    }

    /// <summary>
    /// Gets the meta-learner weights for each base model.
    /// </summary>
    /// <returns>Array of weights (higher = more important).</returns>
    public T[] GetMetaWeights()
    {
        return _metaWeights ?? Array.Empty<T>();
    }

    /// <summary>
    /// Gets the cross-validation performance (MSE) of each base model.
    /// </summary>
    /// <returns>Array of MSE values (lower is better).</returns>
    public double[] GetCVPerformance()
    {
        return _cvPerformance ?? Array.Empty<double>();
    }

    /// <summary>
    /// Gets the contribution of each base model based on weights.
    /// </summary>
    /// <returns>Array of contribution percentages.</returns>
    public double[] GetModelContributions()
    {
        if (_metaWeights == null)
        {
            return Array.Empty<double>();
        }

        var absWeights = _metaWeights.Select(w => Math.Abs(NumOps.ToDouble(w))).ToArray();
        double sum = absWeights.Sum();

        if (sum < 1e-10)
        {
            return absWeights.Select(_ => 1.0 / _metaWeights.Length).ToArray();
        }

        return absWeights.Select(w => w / sum).ToArray();
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
        // Serialize and deserialize to clone
        byte[] data = model.Serialize();
        var clone = (IFullModel<T, Matrix<T>, Vector<T>>)Activator.CreateInstance(model.GetType())!;
        clone.Deserialize(data);
        return clone;
    }

    /// <summary>
    /// Normalizes meta-features (base model predictions).
    /// </summary>
    private void NormalizeMetaFeatures(double[,] features)
    {
        int n = features.GetLength(0);
        int m = features.GetLength(1);

        _predMeans = new double[m];
        _predStds = new double[m];

        for (int j = 0; j < m; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += features[i, j];
            }
            _predMeans[j] = sum / n;

            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = features[i, j] - _predMeans[j];
                sumSq += diff * diff;
            }
            _predStds[j] = Math.Sqrt(sumSq / n);
            if (_predStds[j] < 1e-10) _predStds[j] = 1;

            for (int i = 0; i < n; i++)
            {
                features[i, j] = (features[i, j] - _predMeans[j]) / _predStds[j];
            }
        }
    }

    /// <summary>
    /// Normalizes meta-features using stored means/stds.
    /// </summary>
    private void NormalizeMetaFeaturesForPrediction(double[,] features)
    {
        int n = features.GetLength(0);
        int m = features.GetLength(1);

        for (int j = 0; j < m; j++)
        {
            for (int i = 0; i < n; i++)
            {
                features[i, j] = (features[i, j] - _predMeans![j]) / _predStds![j];
            }
        }
    }

    /// <summary>
    /// Trains the meta-learner.
    /// </summary>
    private void TrainMetaLearner(double[,] metaFeatures, double[] y)
    {
        int n = metaFeatures.GetLength(0);
        int m = metaFeatures.GetLength(1);

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
        _metaWeights = new T[numModels];
        double weight = 1.0 / numModels;
        for (int m = 0; m < numModels; m++)
        {
            _metaWeights[m] = NumOps.FromDouble(weight);
        }
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Performance-weighted averaging.
    /// </summary>
    private void TrainPerformanceWeighted(int numModels)
    {
        _metaWeights = new T[numModels];

        // Convert MSE to weights (inverse of performance)
        var invMse = _cvPerformance!.Select(mse => 1.0 / (mse + 1e-10)).ToArray();
        double sum = invMse.Sum();

        for (int m = 0; m < numModels; m++)
        {
            _metaWeights[m] = NumOps.FromDouble(invMse[m] / sum);
        }
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Non-negative least squares.
    /// </summary>
    private void TrainNNLS(double[,] X, double[] y)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);

        // Initialize weights
        var weights = new double[m];
        for (int j = 0; j < m; j++)
        {
            weights[j] = 1.0 / m;
        }

        // Active set method for NNLS
        for (int iter = 0; iter < _options.MetaLearnerMaxIterations; iter++)
        {
            // Compute gradient
            var grad = new double[m];
            for (int j = 0; j < m; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    double pred = 0;
                    for (int k = 0; k < m; k++)
                    {
                        pred += X[i, k] * weights[k];
                    }
                    grad[j] += 2 * X[i, j] * (pred - y[i]) / n;
                }
            }

            // Projected gradient descent
            double maxChange = 0;
            for (int j = 0; j < m; j++)
            {
                double newWeight = weights[j] - 0.01 * grad[j];
                newWeight = Math.Max(0, newWeight);  // Project to non-negative
                maxChange = Math.Max(maxChange, Math.Abs(newWeight - weights[j]));
                weights[j] = newWeight;
            }

            // Normalize weights to sum to 1 (optional but often helpful)
            double sumW = weights.Sum();
            if (sumW > 1e-10)
            {
                for (int j = 0; j < m; j++)
                {
                    weights[j] /= sumW;
                }
            }

            if (maxChange < _options.MetaLearnerTolerance)
            {
                break;
            }
        }

        _metaWeights = weights.Select(w => NumOps.FromDouble(w)).ToArray();
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Ridge regression meta-learner.
    /// </summary>
    private void TrainRidge(double[,] X, double[] y)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);
        double lambda = _options.MetaLearnerRegularization;

        // X'X + lambda*I
        var XtX = new double[m, m];
        var Xty = new double[m];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                Xty[j] += X[i, j] * y[i];
                for (int k = 0; k < m; k++)
                {
                    XtX[j, k] += X[i, j] * X[i, k];
                }
            }
        }

        // Add regularization
        for (int j = 0; j < m; j++)
        {
            XtX[j, j] += lambda * n;
        }

        // Solve (X'X + lambda*I)^(-1) * X'y
        var XtXinv = InvertMatrix(XtX);
        var weights = new double[m];

        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < m; k++)
            {
                weights[j] += XtXinv[j, k] * Xty[k];
            }
        }

        _metaWeights = weights.Select(w => NumOps.FromDouble(w)).ToArray();
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Linear regression meta-learner.
    /// </summary>
    private void TrainLinearRegression(double[,] X, double[] y)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);

        // X'X
        var XtX = new double[m, m];
        var Xty = new double[m];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                Xty[j] += X[i, j] * y[i];
                for (int k = 0; k < m; k++)
                {
                    XtX[j, k] += X[i, j] * X[i, k];
                }
            }
        }

        // Solve (X'X)^(-1) * X'y with small regularization for stability
        for (int j = 0; j < m; j++)
        {
            XtX[j, j] += 1e-6;
        }

        var XtXinv = InvertMatrix(XtX);
        var weights = new double[m];

        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < m; k++)
            {
                weights[j] += XtXinv[j, k] * Xty[k];
            }
        }

        _metaWeights = weights.Select(w => NumOps.FromDouble(w)).ToArray();
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Lasso regression meta-learner (coordinate descent).
    /// </summary>
    private void TrainLasso(double[,] X, double[] y)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);
        double lambda = _options.MetaLearnerRegularization;

        var weights = new double[m];
        for (int j = 0; j < m; j++)
        {
            weights[j] = 1.0 / m;
        }

        // Coordinate descent
        for (int iter = 0; iter < _options.MetaLearnerMaxIterations; iter++)
        {
            double maxChange = 0;

            for (int j = 0; j < m; j++)
            {
                // Compute partial residual
                var residual = new double[n];
                for (int i = 0; i < n; i++)
                {
                    residual[i] = y[i];
                    for (int k = 0; k < m; k++)
                    {
                        if (k != j)
                        {
                            residual[i] -= X[i, k] * weights[k];
                        }
                    }
                }

                // Soft thresholding
                double rho = 0;
                double sumXjSq = 0;
                for (int i = 0; i < n; i++)
                {
                    rho += X[i, j] * residual[i];
                    sumXjSq += X[i, j] * X[i, j];
                }

                double newWeight;
                if (rho < -lambda * n)
                {
                    newWeight = (rho + lambda * n) / sumXjSq;
                }
                else if (rho > lambda * n)
                {
                    newWeight = (rho - lambda * n) / sumXjSq;
                }
                else
                {
                    newWeight = 0;
                }

                maxChange = Math.Max(maxChange, Math.Abs(newWeight - weights[j]));
                weights[j] = newWeight;
            }

            if (maxChange < _options.MetaLearnerTolerance)
            {
                break;
            }
        }

        _metaWeights = weights.Select(w => NumOps.FromDouble(w)).ToArray();
        _metaIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Simple matrix inversion using Gaussian elimination.
    /// </summary>
    private double[,] InvertMatrix(double[,] A)
    {
        int n = A.GetLength(0);
        var augmented = new double[n, 2 * n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n + i] = 1;
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            for (int j = 0; j < 2 * n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            double pivot = augmented[col, col];
            if (Math.Abs(pivot) < 1e-10) pivot = 1e-10;

            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] /= pivot;
            }

            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    double factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] -= factor * augmented[col, j];
                    }
                }
            }
        }

        var inverse = new double[n, n];
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
    protected override ModelType GetModelType()
    {
        return ModelType.SuperLearner;
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
            ModelType = ModelType.SuperLearner,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumBaseModels", _baseModels.Count },
                { "MetaLearnerType", _options.MetaLearnerType.ToString() },
                { "NumFolds", _options.NumFolds },
                { "NumFeatures", _numFeatures }
            }
        };
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
        if (_predMeans != null)
        {
            foreach (var mean in _predMeans)
            {
                writer.Write(mean);
            }
            foreach (var std in _predStds!)
            {
                writer.Write(std);
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
            _metaWeights = new T[numWeights];
            for (int i = 0; i < numWeights; i++)
            {
                _metaWeights[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
        _metaIntercept = NumOps.FromDouble(reader.ReadDouble());

        int numMeans = reader.ReadInt32();
        if (numMeans > 0)
        {
            _predMeans = new double[numMeans];
            _predStds = new double[numMeans];
            for (int i = 0; i < numMeans; i++)
            {
                _predMeans[i] = reader.ReadDouble();
            }
            for (int i = 0; i < numMeans; i++)
            {
                _predStds[i] = reader.ReadDouble();
            }
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new SuperLearner<T>(_baseModels, _options, Regularization);
    }
}
