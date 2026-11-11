using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// // Create ridge classifier with L2 regularized least squares
/// var options = new LinearClassifierOptions&lt;double&gt;();
///
/// // Prepare training data
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train using closed-form ridge regression solution
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new RidgeClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class based on closest regression target
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Ridge Regression: Biased Estimation for Nonorthogonal Problems", "https://doi.org/10.1080/00401706.1970.10488634")]
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

        // Validate binary classification
        if (NumClasses != 2)
        {
            throw new NotSupportedException(
                "RidgeClassifier currently supports binary classification. " +
                "Use OneVsRestClassifier for multi-class problems.");
        }

        // Convert labels to +1/-1 for binary classification
        var yRegression = SignedBinaryTargets(y);

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

        // Vectorized: X'X = X^T @ X using Engine.TensorMatMul
        var xTensor = Tensor<T>.FromMatrix(xCentered);
        var xTranspose = xTensor.Transpose([1, 0]);
        var xtxTensor = Engine.TensorMatMul(xTranspose, xTensor);

        // Convert to Matrix and add regularization to diagonal
        var xtx = Engine.TensorAdd(xtxTensor,
            Engine.TensorMultiplyScalar(Engine.TensorEye<T>(NumFeatures), alpha)).ToMatrix();

        // Vectorized: X'y = X^T @ y
        var yTensor = Tensor<T>.FromVector(yRegression).Reshape(x.Rows, 1);
        var xtyTensor = Engine.TensorMatMul(xTranspose, yTensor);
        var xty = xtyTensor.Reshape(NumFeatures).ToVector();

        // Solve the linear system using Cholesky or simple Gaussian elimination
        Weights = SolveLinearSystem(xtx, xty);

        // Compute intercept
        if (Options.FitIntercept && xMean is not null)
        {
            // intercept = y_mean - w'x_mean
            Intercept = NumOps.Subtract(yMean, Engine.DotProduct(Weights, xMean));
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
        if (x.Rows == 0)
        {
            return new Vector<T>(x.Columns);
        }

        return Engine.ReduceMean(Tensor<T>.FromMatrix(x), new[] { 0 }, false).ToVector();
    }

    /// <summary>
    /// Computes the mean of a vector.
    /// </summary>
    private T ComputeMean(Vector<T> v)
    {
        return v.Length == 0 ? NumOps.Divide(NumOps.Zero, NumOps.Zero) : Engine.Mean(v);
    }

    /// <summary>
    /// Centers a matrix by subtracting column means.
    /// </summary>
    private Matrix<T> CenterMatrix(Matrix<T> x, Vector<T> mean)
    {
        if (x.Rows == 0)
        {
            return new Matrix<T>(0, x.Columns);
        }

        return Engine.TensorSubtract(Tensor<T>.FromMatrix(x), mean.AsTensor().Reshape(1, mean.Length)).ToMatrix();
    }

    /// <summary>
    /// Centers a vector by subtracting the mean.
    /// </summary>
    private Vector<T> CenterVector(Vector<T> v, T mean)
    {
        return v.Length == 0 ? new Vector<T>(0) : Engine.TensorSubtractScalar(v.AsTensor(), mean).ToVector();
    }

    /// <summary>
    /// Solves Ax = b using Gaussian elimination with partial pivoting.
    /// </summary>
    private Vector<T> SolveLinearSystem(Matrix<T> a, Vector<T> b)
    {
        int n = a.Rows;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // LU solve with partial pivoting; a singular system falls back to the pseudo-inverse solution
        var aTensor = Tensor<T>.FromMatrix(a);
        var bColumn = b.AsTensor().Reshape(n, 1);
        var (solution, info) = Linalg.SolveEx(aTensor, bColumn);
        bool singular = info.Length > 0 && info.ToArray()[0] != 0;
        var result = singular ? Engine.TensorMatMul(Linalg.Pinv(aTensor), bColumn) : solution;
        return result.Reshape(n).ToVector();
    }
}
