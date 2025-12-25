using AiDotNet.Models.Options;

namespace AiDotNet.Classification.DiscriminantAnalysis;

/// <summary>
/// Quadratic Discriminant Analysis classifier.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// QDA is similar to LDA but allows each class to have its own covariance matrix,
/// which creates quadratic (curved) decision boundaries.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Quadratic Discriminant Analysis (QDA) is a classification technique that:
///
/// 1. Models each class with its own Gaussian distribution
/// 2. Each class has its own covariance matrix (unlike LDA)
/// 3. Decision boundaries are quadratic (curved) instead of linear
///
/// When to use QDA over LDA:
/// - When classes have different covariance structures
/// - When you have enough samples per class to estimate covariance reliably
/// - When decision boundaries are naturally curved
///
/// Trade-offs:
/// - More flexible than LDA (can capture curved boundaries)
/// - Needs more parameters (separate covariance per class)
/// - More prone to overfitting with small datasets
/// - Computationally more expensive than LDA
/// </para>
/// </remarks>
public class QuadraticDiscriminantAnalysis<T> : ProbabilisticClassifierBase<T>
{
    /// <summary>
    /// Gets the QDA-specific options.
    /// </summary>
    protected new DiscriminantAnalysisOptions<T> Options => (DiscriminantAnalysisOptions<T>)base.Options;

    /// <summary>
    /// Class means for each class.
    /// </summary>
    private Matrix<T>? _classMeans;

    /// <summary>
    /// Covariance matrix for each class.
    /// </summary>
    private Matrix<T>[]? _classCovariances;

    /// <summary>
    /// Inverse of covariance matrix for each class.
    /// </summary>
    private Matrix<T>[]? _classCovarianceInverses;

    /// <summary>
    /// Log determinant of covariance matrix for each class.
    /// </summary>
    private Vector<T>? _classLogDets;

    /// <summary>
    /// Class priors (prior probabilities).
    /// </summary>
    private Vector<T>? _classPriors;

    /// <summary>
    /// Initializes a new instance of the QuadraticDiscriminantAnalysis class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public QuadraticDiscriminantAnalysis(DiscriminantAnalysisOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new DiscriminantAnalysisOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.QuadraticDiscriminantAnalysis;

    /// <summary>
    /// Trains the QDA classifier on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Compute class means
        _classMeans = ComputeClassMeans(x, y);

        // Compute class priors
        _classPriors = ComputeClassPriors(y);

        // Compute per-class covariance matrices
        _classCovariances = new Matrix<T>[NumClasses];
        _classCovarianceInverses = new Matrix<T>[NumClasses];
        _classLogDets = new Vector<T>(NumClasses);

        for (int c = 0; c < NumClasses; c++)
        {
            _classCovariances[c] = ComputeClassCovariance(x, y, c);

            // Add regularization if specified
            if (Options.RegularizationParam > 0)
            {
                AddRegularization(_classCovariances[c]);
            }

            // Compute inverse and log determinant
            _classCovarianceInverses[c] = ComputeInverse(_classCovariances[c]);
            _classLogDets[c] = ComputeLogDeterminant(_classCovariances[c]);
        }
    }

    /// <summary>
    /// Computes the mean vector for each class.
    /// </summary>
    private Matrix<T> ComputeClassMeans(Matrix<T> x, Vector<T> y)
    {
        var means = new Matrix<T>(NumClasses, NumFeatures);

        for (int c = 0; c < NumClasses; c++)
        {
            T classLabel = ClassLabels![c];
            int count = 0;

            for (int i = 0; i < x.Rows; i++)
            {
                if (NumOps.Compare(y[i], classLabel) == 0)
                {
                    count++;
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        means[c, j] = NumOps.Add(means[c, j], x[i, j]);
                    }
                }
            }

            if (count > 0)
            {
                T countT = NumOps.FromDouble(count);
                for (int j = 0; j < NumFeatures; j++)
                {
                    means[c, j] = NumOps.Divide(means[c, j], countT);
                }
            }
        }

        return means;
    }

    /// <summary>
    /// Computes class prior probabilities.
    /// </summary>
    private Vector<T> ComputeClassPriors(Vector<T> y)
    {
        var priors = new Vector<T>(NumClasses);
        int n = y.Length;

        for (int c = 0; c < NumClasses; c++)
        {
            T classLabel = ClassLabels![c];
            int count = 0;

            for (int i = 0; i < n; i++)
            {
                if (NumOps.Compare(y[i], classLabel) == 0)
                {
                    count++;
                }
            }

            priors[c] = NumOps.Divide(NumOps.FromDouble(count), NumOps.FromDouble(n));
        }

        return priors;
    }

    /// <summary>
    /// Computes the covariance matrix for a specific class.
    /// </summary>
    private Matrix<T> ComputeClassCovariance(Matrix<T> x, Vector<T> y, int classIndex)
    {
        var covariance = new Matrix<T>(NumFeatures, NumFeatures);
        T classLabel = ClassLabels![classIndex];
        int count = 0;

        // Get class mean
        var mean = new Vector<T>(NumFeatures);
        for (int j = 0; j < NumFeatures; j++)
        {
            mean[j] = _classMeans![classIndex, j];
        }

        // Compute covariance for this class
        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.Compare(y[i], classLabel) == 0)
            {
                count++;

                // Compute centered sample
                var centered = new Vector<T>(NumFeatures);
                for (int j = 0; j < NumFeatures; j++)
                {
                    centered[j] = NumOps.Subtract(x[i, j], mean[j]);
                }

                // Outer product: centered * centered^T
                for (int j = 0; j < NumFeatures; j++)
                {
                    for (int k = 0; k < NumFeatures; k++)
                    {
                        T product = NumOps.Multiply(centered[j], centered[k]);
                        covariance[j, k] = NumOps.Add(covariance[j, k], product);
                    }
                }
            }
        }

        // Divide by (n - 1) for unbiased estimate
        T denominator = NumOps.FromDouble(Math.Max(1, count - 1));
        for (int j = 0; j < NumFeatures; j++)
        {
            for (int k = 0; k < NumFeatures; k++)
            {
                covariance[j, k] = NumOps.Divide(covariance[j, k], denominator);
            }
        }

        return covariance;
    }

    /// <summary>
    /// Adds regularization to the covariance matrix.
    /// </summary>
    private void AddRegularization(Matrix<T> covariance)
    {
        T regParam = NumOps.FromDouble(Options.RegularizationParam);

        for (int i = 0; i < NumFeatures; i++)
        {
            covariance[i, i] = NumOps.Add(covariance[i, i], regParam);
        }
    }

    /// <summary>
    /// Computes the inverse of a matrix using Gaussian elimination.
    /// </summary>
    private Matrix<T> ComputeInverse(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var augmented = new Matrix<T>(n, 2 * n);

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = matrix[i, j];
            }
            augmented[i, n + i] = NumOps.One;
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            T maxVal = NumOps.Abs(augmented[col, col]);
            for (int row = col + 1; row < n; row++)
            {
                T val = NumOps.Abs(augmented[row, col]);
                if (NumOps.Compare(val, maxVal) > 0)
                {
                    maxVal = val;
                    maxRow = row;
                }
            }

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    T temp = augmented[col, j];
                    augmented[col, j] = augmented[maxRow, j];
                    augmented[maxRow, j] = temp;
                }
            }

            // Scale pivot row
            T pivot = augmented[col, col];
            T minPivot = NumOps.FromDouble(1e-12);
            if (NumOps.Compare(NumOps.Abs(pivot), minPivot) < 0)
            {
                pivot = minPivot;
            }

            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] = NumOps.Divide(augmented[col, j], pivot);
            }

            // Eliminate other rows
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

        // Extract inverse
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

    /// <summary>
    /// Computes the log determinant of a matrix using LU decomposition.
    /// </summary>
    private T ComputeLogDeterminant(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var lu = new Matrix<T>(n, n);

        // Copy matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                lu[i, j] = matrix[i, j];
            }
        }

        T logDet = NumOps.Zero;
        int signChanges = 0;

        // LU decomposition with partial pivoting
        for (int k = 0; k < n; k++)
        {
            // Find pivot
            int maxRow = k;
            T maxVal = NumOps.Abs(lu[k, k]);
            for (int i = k + 1; i < n; i++)
            {
                T val = NumOps.Abs(lu[i, k]);
                if (NumOps.Compare(val, maxVal) > 0)
                {
                    maxVal = val;
                    maxRow = i;
                }
            }

            if (maxRow != k)
            {
                signChanges++;
                for (int j = 0; j < n; j++)
                {
                    T temp = lu[k, j];
                    lu[k, j] = lu[maxRow, j];
                    lu[maxRow, j] = temp;
                }
            }

            T pivot = lu[k, k];
            T minPivot = NumOps.FromDouble(1e-12);
            if (NumOps.Compare(NumOps.Abs(pivot), minPivot) < 0)
            {
                pivot = minPivot;
            }

            logDet = NumOps.Add(logDet, NumOps.Log(NumOps.Abs(pivot)));

            for (int i = k + 1; i < n; i++)
            {
                lu[i, k] = NumOps.Divide(lu[i, k], pivot);
                for (int j = k + 1; j < n; j++)
                {
                    lu[i, j] = NumOps.Subtract(lu[i, j], NumOps.Multiply(lu[i, k], lu[k, j]));
                }
            }
        }

        return logDet;
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            int bestClass = 0;
            T bestProb = probs[i, 0];

            for (int c = 1; c < NumClasses; c++)
            {
                if (NumOps.Compare(probs[i, c], bestProb) > 0)
                {
                    bestProb = probs[i, c];
                    bestClass = c;
                }
            }

            predictions[i] = ClassLabels![bestClass];
        }

        return predictions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_classMeans is null || _classCovarianceInverses is null ||
            _classPriors is null || _classLogDets is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var logProbs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            T maxLogProb = NumOps.FromDouble(double.NegativeInfinity);

            for (int c = 0; c < NumClasses; c++)
            {
                // Get sample
                var sample = new Vector<T>(NumFeatures);
                for (int j = 0; j < NumFeatures; j++)
                {
                    sample[j] = input[i, j];
                }

                // Compute (x - mu_c)
                var diff = new Vector<T>(NumFeatures);
                for (int j = 0; j < NumFeatures; j++)
                {
                    diff[j] = NumOps.Subtract(sample[j], _classMeans[c, j]);
                }

                // Compute (x - mu_c)^T * Sigma_c^(-1) * (x - mu_c)
                T mahalanobis = NumOps.Zero;
                for (int j = 0; j < NumFeatures; j++)
                {
                    T temp = NumOps.Zero;
                    for (int k = 0; k < NumFeatures; k++)
                    {
                        temp = NumOps.Add(temp, NumOps.Multiply(_classCovarianceInverses[c][j, k], diff[k]));
                    }
                    mahalanobis = NumOps.Add(mahalanobis, NumOps.Multiply(diff[j], temp));
                }

                // Log posterior = log(prior) - 0.5 * log|Sigma_c| - 0.5 * mahalanobis
                T logPrior = NumOps.Log(_classPriors[c]);
                T logDetTerm = NumOps.Multiply(NumOps.FromDouble(-0.5), _classLogDets[c]);
                T mahalanobisTerm = NumOps.Multiply(NumOps.FromDouble(-0.5), mahalanobis);

                logProbs[i, c] = NumOps.Add(NumOps.Add(logPrior, logDetTerm), mahalanobisTerm);

                if (NumOps.Compare(logProbs[i, c], maxLogProb) > 0)
                {
                    maxLogProb = logProbs[i, c];
                }
            }

            // Convert to probabilities using softmax
            T sumExp = NumOps.Zero;
            for (int c = 0; c < NumClasses; c++)
            {
                T expVal = NumOps.Exp(NumOps.Subtract(logProbs[i, c], maxLogProb));
                sumExp = NumOps.Add(sumExp, expVal);
            }

            T logSumExp = NumOps.Add(maxLogProb, NumOps.Log(sumExp));
            for (int c = 0; c < NumClasses; c++)
            {
                logProbs[i, c] = NumOps.Exp(NumOps.Subtract(logProbs[i, c], logSumExp));
            }
        }

        return logProbs;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var logProbs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                T p = probs[i, c];
                T minP = NumOps.FromDouble(1e-15);
                if (NumOps.Compare(p, minP) < 0)
                {
                    p = minP;
                }
                logProbs[i, c] = NumOps.Log(p);
            }
        }

        return logProbs;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new QuadraticDiscriminantAnalysis<T>(new DiscriminantAnalysisOptions<T>
        {
            RegularizationParam = Options.RegularizationParam
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (QuadraticDiscriminantAnalysis<T>)CreateNewInstance();

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (_classMeans is not null)
        {
            clone._classMeans = new Matrix<T>(_classMeans.Rows, _classMeans.Columns);
            for (int i = 0; i < _classMeans.Rows; i++)
            {
                for (int j = 0; j < _classMeans.Columns; j++)
                {
                    clone._classMeans[i, j] = _classMeans[i, j];
                }
            }
        }

        if (_classPriors is not null)
        {
            clone._classPriors = new Vector<T>(_classPriors.Length);
            for (int i = 0; i < _classPriors.Length; i++)
            {
                clone._classPriors[i] = _classPriors[i];
            }
        }

        if (_classLogDets is not null)
        {
            clone._classLogDets = new Vector<T>(_classLogDets.Length);
            for (int i = 0; i < _classLogDets.Length; i++)
            {
                clone._classLogDets[i] = _classLogDets[i];
            }
        }

        if (_classCovariances is not null)
        {
            clone._classCovariances = new Matrix<T>[NumClasses];
            for (int c = 0; c < NumClasses; c++)
            {
                clone._classCovariances[c] = new Matrix<T>(NumFeatures, NumFeatures);
                for (int i = 0; i < NumFeatures; i++)
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        clone._classCovariances[c][i, j] = _classCovariances[c][i, j];
                    }
                }
            }
        }

        if (_classCovarianceInverses is not null)
        {
            clone._classCovarianceInverses = new Matrix<T>[NumClasses];
            for (int c = 0; c < NumClasses; c++)
            {
                clone._classCovarianceInverses[c] = new Matrix<T>(NumFeatures, NumFeatures);
                for (int i = 0; i < NumFeatures; i++)
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        clone._classCovarianceInverses[c][i, j] = _classCovarianceInverses[c][i, j];
                    }
                }
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["RegularizationParam"] = Options.RegularizationParam;
        return metadata;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // QDA doesn't have a simple parameter vector like linear classifiers
        // Return flattened class means for serialization
        if (_classMeans is null)
        {
            return new Vector<T>(0);
        }

        int length = NumClasses * NumFeatures;
        var parameters = new Vector<T>(length);
        int idx = 0;
        for (int c = 0; c < NumClasses; c++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                parameters[idx++] = _classMeans[c, j];
            }
        }
        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // QDA doesn't support setting parameters directly
        // The model must be trained to compute proper covariance matrices
        throw new NotSupportedException("QDA does not support setting parameters directly. Use Train() instead.");
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("QDA does not support setting parameters directly. Use Train() instead.");
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // QDA is a closed-form solution, not gradient-based
        // Return zero gradients
        return new Vector<T>(NumClasses * NumFeatures);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // QDA is a closed-form solution, not gradient-based
        // This is a no-op
    }
}
