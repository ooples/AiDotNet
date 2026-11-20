using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Ridge Regression (L2 regularized linear regression), which extends ordinary least squares
/// by adding a penalty term proportional to the squared magnitude of the coefficients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Ridge Regression solves the following optimization problem:
/// minimize ||y - Xw||^2 + alpha * ||w||^2
///
/// This has a closed-form solution: w = (X^T X + alpha * I)^(-1) X^T y
///
/// The L2 penalty shrinks coefficients toward zero but never sets them exactly to zero,
/// making Ridge Regression suitable for problems where all features are expected to contribute.
/// </para>
/// <para><b>For Beginners:</b> Ridge Regression is a safer version of linear regression.
///
/// Regular linear regression can become unstable when:
/// - You have many features relative to samples
/// - Some features are highly correlated with each other
/// - The data contains noise
///
/// Ridge Regression fixes these issues by adding a "penalty" for large coefficients:
/// - It prevents any single feature from dominating the prediction
/// - It makes the model more stable and reliable
/// - It typically improves predictions on new, unseen data
///
/// Think of it like putting rubber bands on a flexible ruler - the bands (regularization)
/// keep the ruler from bending too wildly (overfitting), while still allowing it to
/// follow the general trend of the data.
///
/// Example usage:
/// ```csharp
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.1, 7.2, 11.0, 15.1 });
///
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new RidgeRegression&lt;double&gt;(new RidgeRegressionOptions&lt;double&gt; { Alpha = 1.0 }))
///     .Build(features, targets);
///
/// var newFeatures = new Matrix&lt;double&gt;(new double[,] { { 9.0, 10.0 } });
/// var predictions = result.Predict(newFeatures);
/// ```
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create ridge regression with L2 regularization
/// var options = new RidgeRegressionOptions&lt;double&gt;();
///
/// // Prepare training data
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 2.1, 3.9, 6.2, 7.8, 10.1 });
///
/// // Train with L2 penalty to prevent overfitting
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new RidgeRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for new input
/// var newFeatures = new Matrix&lt;double&gt;(new double[,] { { 11, 12 } });
/// var prediction = result.Predict(newFeatures);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Ridge Regression: Biased Estimation for Nonorthogonal Problems", "https://doi.org/10.1080/00401706.1970.10488634")]
public class RidgeRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Gets the configuration options specific to Ridge Regression.
    /// </summary>
    private new RidgeRegressionOptions<T> Options => (RidgeRegressionOptions<T>)base.Options;

    /// <summary>
    /// Initializes a new instance of the <see cref="RidgeRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for Ridge Regression. If null, default options are used.</param>
    /// <param name="regularization">Optional additional regularization strategy.</param>
    /// <remarks>
    /// <para>
    /// Creates a new Ridge Regression model with the specified options. The primary regularization
    /// is controlled by the Alpha parameter in the options. An additional regularization strategy
    /// can be provided for more complex regularization schemes.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Ridge Regression model.
    ///
    /// You can customize the model by providing options:
    /// ```csharp
    /// // Create with default options (alpha = 1.0)
    /// var ridge = new RidgeRegression&lt;double&gt;();
    ///
    /// // Create with custom alpha
    /// var options = new RidgeRegressionOptions&lt;double&gt; { Alpha = 0.5 };
    /// var ridge = new RidgeRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public RidgeRegression(RidgeRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new RidgeRegressionOptions<T>(), regularization)
    {
    }

    /// <summary>
    /// Trains the Ridge Regression model using the provided training data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each sample.</param>
    /// <remarks>
    /// <para>
    /// Training computes the closed-form solution:
    /// w = (X^T X + alpha * I)^(-1) X^T y
    ///
    /// This is numerically stable due to the regularization term, which ensures the matrix
    /// X^T X + alpha * I is always invertible (positive definite).
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to make predictions from your data.
    ///
    /// The training process:
    /// 1. Adds an intercept column (if UseIntercept = true)
    /// 2. Computes the optimal coefficients using a direct mathematical formula
    /// 3. Stores the coefficients for making predictions later
    ///
    /// Unlike some other methods, Ridge Regression has a direct solution - no iterative
    /// optimization is needed, making training fast and deterministic.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
            throw new ArgumentException("The number of rows in X must match the length of y.");

        int numFeatures = x.Columns + (Options.UseIntercept ? 1 : 0);

        // X^T X and X^T y over [1, X] when fitting an intercept (the intercept is column 0).
        var (xTx, xTy) = typeof(T) == typeof(double) || typeof(T) == typeof(float)
            ? NormalEquationsInDouble(x, y, Options.UseIntercept)
            : NormalEquations(x, y, Options.UseIntercept);

        // Add alpha * I to the diagonal (regularization term)
        // Note: If using intercept, we typically don't regularize the intercept term
        T alpha = NumOps.FromDouble(Options.Alpha);
        int startIdx = Options.UseIntercept ? 1 : 0;
        if (numFeatures > startIdx)
        {
            var ridgeDiagonal = Vector<T>.Concatenate(
                Engine.FillZero<T>(startIdx), Engine.Fill(numFeatures - startIdx, alpha));
            xTx = RegressionEngineOps2.AddDiagonal(xTx, ridgeDiagonal);
        }

        // Apply additional regularization if provided
        var regularizedXTx = xTx.Add(Regularization.Regularize(xTx));

        // Solve (X^T X + alpha * I) * w = X^T y
        var solution = SolveSystemWithDecomposition(regularizedXTx, xTy);

        // Extract coefficients and intercept
        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = RegressionEngineOps2.Slice(solution, 1, numFeatures - 1);
        }
        else
        {
            Intercept = NumOps.Zero;
            Coefficients = solution;
        }
    }

    /// <summary>Rows per block of <see cref="NormalEquationsInDouble"/>'s parallel pass.</summary>
    private const int GramBlockRows = 1024;

    /// <summary>
    /// <c>X^T X</c> and <c>X^T y</c>, over <c>[1, X]</c> when <paramref name="withIntercept"/> is set, for any numeric type.
    /// </summary>
    private (Matrix<T> XtX, Vector<T> XtY) NormalEquations(Matrix<T> x, Vector<T> y, bool withIntercept)
    {
        var design = withIntercept ? x.AddConstantColumn(NumOps.One) : x;
        var transposed = design.Transpose();
        return (transposed.Multiply(design), transposed.Multiply(y));
    }

    /// <summary>
    /// <see cref="NormalEquations"/> for <c>double</c> and <c>float</c>, whose values convert to double exactly: one pass
    /// over the rows accumulating the upper triangle of <c>X^T X</c> and <c>X^T y</c>, without copying the design matrix.
    /// </summary>
    /// <remarks>
    /// The intercept column was appended by copying the whole matrix, then transposed twice for two general multiplies;
    /// on 20,000 x 20 that took about twice scikit-learn's whole fit. Fixed blocks of rows run in parallel and their sums are
    /// added in block order, so the result does not depend on the thread count.
    /// </remarks>
    private (Matrix<T> XtX, Vector<T> XtY) NormalEquationsInDouble(Matrix<T> x, Vector<T> y, bool withIntercept)
    {
        int rows = x.Rows;
        int columns = x.Columns;
        int shift = withIntercept ? 1 : 0;
        int width = columns + shift;
        int blocks = (rows + GramBlockRows - 1) / GramBlockRows;

        // Each block keeps the upper triangle of the Gram matrix, row-major [width, width], then X^T y.
        var partials = new double[blocks][];
        System.Threading.Tasks.Parallel.For(0, blocks, block =>
        {
            var sums = new double[width * width + width];
            var values = new double[width];
            if (withIntercept) values[0] = 1.0;
            int end = Math.Min(rows, (block + 1) * GramBlockRows);
            for (int i = block * GramBlockRows; i < end; i++)
            {
                var row = x.GetRowReadOnlySpan(i);
                for (int j = 0; j < columns; j++) values[shift + j] = NumOps.ToDouble(row[j]);
                double target = NumOps.ToDouble(y[i]);
                for (int a = 0; a < width; a++)
                {
                    double va = values[a];
                    int at = a * width;
                    for (int b = a; b < width; b++) sums[at + b] += va * values[b];
                    sums[width * width + a] += va * target;
                }
            }

            partials[block] = sums;
        });

        var total = new double[width * width + width];
        foreach (var sums in partials)
            for (int m = 0; m < total.Length; m++) total[m] += sums[m];

        var xTx = new Matrix<T>(width, width);
        var xTy = new Vector<T>(width);
        for (int a = 0; a < width; a++)
        {
            for (int b = a; b < width; b++)
            {
                T value = NumOps.FromDouble(total[a * width + b]);
                xTx[a, b] = value;
                xTx[b, a] = value;
            }

            xTy[a] = NumOps.FromDouble(total[width * width + a]);
        }

        return (xTx, xTy);
    }

    /// <summary>
    /// Solves a linear system using the configured decomposition method.
    /// </summary>
    private Vector<T> SolveSystemWithDecomposition(Matrix<T> a, Vector<T> b)
    {
        return Options.DecompositionType switch
        {
            MatrixDecompositionType.Cholesky => new CholeskyDecomposition<T>(a).Solve(b),
            MatrixDecompositionType.Svd => new SvdDecomposition<T>(a).Solve(b),
            MatrixDecompositionType.Qr => new QrDecomposition<T>(a).Solve(b),
            MatrixDecompositionType.Lu => new LuDecomposition<T>(a).Solve(b),
            _ => SolveSystem(a, b)
        };
    }

    /// <summary>
    /// Gets metadata about the Ridge Regression model.
    /// </summary>
    /// <returns>A ModelMetadata object containing model information.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Alpha"] = Options.Alpha;
        metadata.AdditionalInfo["DecompositionType"] = Options.DecompositionType.ToString();

        return metadata;
    }

}
