using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Linear;

/// <summary>
/// Ridge Classifier - converts regression to classification using regularized least squares.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Ridge Classifier uses ridge regression (L2 regularized least squares) and then
/// converts the continuous predictions to class labels.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Ridge Classifier treats classification as a regression problem:
///
/// How it works:
/// 1. Convert class labels to numbers (-1 and +1 for binary)
/// 2. Fit a ridge regression to these numbers
/// 3. For prediction, output whichever class the regression is closest to
///
/// Why use Ridge Classifier:
/// - Very fast training (closed-form solution)
/// - Works well when number of features is large
/// - Stable due to regularization
/// - Good baseline classifier
///
/// Trade-offs:
/// - Doesn't optimize classification accuracy directly
/// - May not work as well as logistic regression for probability estimates
/// - Assumes linear relationship between features and class labels
/// </para>
/// </remarks>
public class RidgeClassifier<T> : LinearClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance of the RidgeClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public RidgeClassifier(LinearClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.RidgeClassifier;

    /// <summary>
    /// Trains the Ridge Classifier using closed-form solution.
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

        // Convert labels to +1/-1 for binary classification
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        var yRegression = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            yRegression[i] = NumOps.Compare(y[i], positiveClass) == 0
                ? NumOps.One
                : NumOps.Negate(NumOps.One);
        }

        // Center the data if fitting intercept
        var xCentered = x;
        T yMean = NumOps.Zero;
        Vector<T>? xMean = null;

        if (Options.FitIntercept)
        {
            xMean = ComputeMean(x);
            yMean = ComputeMean(yRegression);

            // Center X and y
            xCentered = CenterMatrix(x, xMean);
            yRegression = CenterVector(yRegression, yMean);
        }

        // Compute ridge regression solution: w = (X'X + alpha*I)^(-1) X'y
        T alpha = NumOps.FromDouble(Options.Alpha);

        // Compute X'X
        var xtx = new Matrix<T>(NumFeatures, NumFeatures);
        for (int i = 0; i < NumFeatures; i++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < x.Rows; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(xCentered[k, i], xCentered[k, j]));
                }
                xtx[i, j] = sum;

                // Add regularization to diagonal
                if (i == j)
                {
                    xtx[i, j] = NumOps.Add(xtx[i, j], alpha);
                }
            }
        }

        // Compute X'y
        var xty = new Vector<T>(NumFeatures);
        for (int j = 0; j < NumFeatures; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < x.Rows; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(xCentered[i, j], yRegression[i]));
            }
            xty[j] = sum;
        }

        // Solve the linear system using Cholesky or simple Gaussian elimination
        Weights = SolveLinearSystem(xtx, xty);

        // Compute intercept
        if (Options.FitIntercept && xMean is not null)
        {
            // intercept = y_mean - w'x_mean
            Intercept = yMean;
            for (int j = 0; j < NumFeatures; j++)
            {
                Intercept = NumOps.Subtract(Intercept, NumOps.Multiply(Weights[j], xMean[j]));
            }
        }
        else
        {
            Intercept = NumOps.Zero;
        }
    }

    /// <summary>
    /// Computes the mean of each column in a matrix.
    /// </summary>
    private Vector<T> ComputeMean(Matrix<T> x)
    {
        var mean = new Vector<T>(x.Columns);
        for (int j = 0; j < x.Columns; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < x.Rows; i++)
            {
                sum = NumOps.Add(sum, x[i, j]);
            }
            mean[j] = NumOps.Divide(sum, NumOps.FromDouble(x.Rows));
        }
        return mean;
    }

    /// <summary>
    /// Computes the mean of a vector.
    /// </summary>
    private T ComputeMean(Vector<T> v)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v.Length; i++)
        {
            sum = NumOps.Add(sum, v[i]);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(v.Length));
    }

    /// <summary>
    /// Centers a matrix by subtracting column means.
    /// </summary>
    private Matrix<T> CenterMatrix(Matrix<T> x, Vector<T> mean)
    {
        var centered = new Matrix<T>(x.Rows, x.Columns);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                centered[i, j] = NumOps.Subtract(x[i, j], mean[j]);
            }
        }
        return centered;
    }

    /// <summary>
    /// Centers a vector by subtracting the mean.
    /// </summary>
    private Vector<T> CenterVector(Vector<T> v, T mean)
    {
        var centered = new Vector<T>(v.Length);
        for (int i = 0; i < v.Length; i++)
        {
            centered[i] = NumOps.Subtract(v[i], mean);
        }
        return centered;
    }

    /// <summary>
    /// Solves Ax = b using Gaussian elimination with partial pivoting.
    /// </summary>
    private Vector<T> SolveLinearSystem(Matrix<T> a, Vector<T> b)
    {
        int n = a.Rows;

        // Create augmented matrix [A | b]
        var aug = new Matrix<T>(n, n + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                aug[i, j] = a[i, j];
            }
            aug[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            T maxVal = NumOps.Abs(aug[col, col]);
            for (int row = col + 1; row < n; row++)
            {
                T val = NumOps.Abs(aug[row, col]);
                if (NumOps.Compare(val, maxVal) > 0)
                {
                    maxVal = val;
                    maxRow = row;
                }
            }

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j <= n; j++)
                {
                    T temp = aug[col, j];
                    aug[col, j] = aug[maxRow, j];
                    aug[maxRow, j] = temp;
                }
            }

            // Eliminate below
            T pivot = aug[col, col];
            if (NumOps.Compare(NumOps.Abs(pivot), NumOps.FromDouble(1e-12)) < 0)
            {
                // Near-zero pivot, add small value for stability
                pivot = NumOps.FromDouble(1e-12);
            }

            for (int row = col + 1; row < n; row++)
            {
                T factor = NumOps.Divide(aug[row, col], pivot);
                for (int j = col; j <= n; j++)
                {
                    aug[row, j] = NumOps.Subtract(aug[row, j], NumOps.Multiply(factor, aug[col, j]));
                }
            }
        }

        // Back substitution
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            T sum = aug[i, n];
            for (int j = i + 1; j < n; j++)
            {
                sum = NumOps.Subtract(sum, NumOps.Multiply(aug[i, j], x[j]));
            }

            T diag = aug[i, i];
            if (NumOps.Compare(NumOps.Abs(diag), NumOps.FromDouble(1e-12)) < 0)
            {
                diag = NumOps.FromDouble(1e-12);
            }
            x[i] = NumOps.Divide(sum, diag);
        }

        return x;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new RidgeClassifier<T>(new LinearClassifierOptions<T>
        {
            Alpha = Options.Alpha,
            FitIntercept = Options.FitIntercept
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (RidgeClassifier<T>)CreateNewInstance();

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone.Intercept = Intercept;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (Weights is not null)
        {
            clone.Weights = new Vector<T>(Weights.Length);
            for (int i = 0; i < Weights.Length; i++)
            {
                clone.Weights[i] = Weights[i];
            }
        }

        return clone;
    }
}
