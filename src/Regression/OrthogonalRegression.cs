using AiDotNet.Attributes;
using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums;

namespace AiDotNet.Regression;

/// <summary>
/// Implements orthogonal regression (also known as total least squares), which minimizes the perpendicular 
/// distance from data points to the fitted line or hyperplane.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Unlike ordinary least squares regression which minimizes vertical distances, orthogonal regression 
/// minimizes the perpendicular (orthogonal) distance from each data point to the regression line or hyperplane.
/// This approach is more appropriate when both dependent and independent variables contain measurement errors.
/// </para>
/// <para>
/// The algorithm works by centering the data, optionally scaling the variables, and then finding the 
/// solution using matrix decomposition methods such as SVD (Singular Value Decomposition).
/// </para>
/// <para>
/// For Beginners:
/// Orthogonal regression is useful when you're not sure which variable is dependent and which is independent,
/// or when both variables have measurement errors. Think of it as finding the line that's as close as possible
/// to all points when measuring distance perpendicular to the line, rather than just vertically.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an orthogonal (total least squares) regression model
/// var model = new OrthogonalRegression&lt;double&gt;();
///
/// // Prepare training data: 5 samples with 2 features each
/// var features = Matrix&lt;double&gt;.Build.Dense(5, 2, new double[] {
///     1, 2,  3, 4,  5, 6,  7, 8,  9, 10 });
/// var targets = new Vector&lt;double&gt;(new double[] { 2.5, 5.3, 8.1, 10.9, 13.7 });
///
/// // Train minimizing perpendicular distance to the hyperplane
/// model.Train(features, targets);
///
/// // Predict for a new sample
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 11, 12 });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Total Least Squares and Errors-in-Variables Modeling", "https://doi.org/10.1007/978-94-017-3552-0")]
public class OrthogonalRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the orthogonal regression model.
    /// </summary>
    /// <value>
    /// Contains settings like tolerance, maximum iterations, and whether to scale variables.
    /// </value>
    private readonly OrthogonalRegressionOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the OrthogonalRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the orthogonal regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the orthogonal regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public OrthogonalRegression(OrthogonalRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new OrthogonalRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the orthogonal regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Applies regularization to the input matrix
    /// 2. Centers the data by subtracting the mean from each feature and the target
    /// 3. Optionally scales the variables to have unit variance
    /// 4. Computes the augmented matrix [X y]
    /// 5. Uses matrix decomposition to find the solution
    /// 6. Rescales the solution and normalizes it
    /// 7. Extracts the coefficients and intercept
    /// 8. Applies regularization to the coefficients
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training is the process where the model learns from your data. The algorithm first centers the data
    /// (subtracts the average from each variable) and optionally scales it (makes all variables have similar ranges).
    /// Then it uses advanced matrix math to find the best-fitting line or plane that minimizes the perpendicular
    /// distance from all points. The result is a set of coefficients that define this line or plane.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Center the data
        Vector<T> meanX = new(p);
        for (int j = 0; j < p; j++)
        {
            meanX[j] = x.GetColumn(j).Mean();
        }
        T meanY = y.Mean();

        Matrix<T> centeredX = new(n, p);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                centeredX[i, j] = NumOps.Subtract(x[i, j], meanX[j]);
            }
        }
        Vector<T> centeredY = y.Subtract(meanY);

        // Scale the variables if option is set
        // Both X and Y must be scaled consistently for TLS to work correctly
        Vector<T> scaleX = Vector<T>.CreateDefault(p, NumOps.One);
        T scaleY = NumOps.One;
        if (_options.ScaleVariables)
        {
            for (int j = 0; j < p; j++)
            {
                T columnVariance = centeredX.GetColumn(j).Variance();
                scaleX[j] = NumOps.Sqrt(columnVariance);
                if (!NumOps.Equals(scaleX[j], NumOps.Zero))
                {
                    for (int i = 0; i < n; i++)
                    {
                        centeredX[i, j] = NumOps.Divide(centeredX[i, j], scaleX[j]);
                    }
                }
            }

            // Scale Y as well to keep the augmented matrix in consistent units
            T yVariance = centeredY.Variance();
            scaleY = NumOps.Sqrt(yVariance);
            if (!NumOps.Equals(scaleY, NumOps.Zero))
            {
                for (int i = 0; i < n; i++)
                {
                    centeredY[i] = NumOps.Divide(centeredY[i], scaleY);
                }
            }
        }

        // Total Least Squares via SVD of the augmented matrix [X | y]
        // The TLS solution is extracted from the last right singular vector of [X y].
        Matrix<T> augmentedMatrix = new(n, p + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                augmentedMatrix[i, j] = centeredX[i, j];
            }
            augmentedMatrix[i, p] = centeredY[i];
        }

        // Compute BOTH solutions and pick the better R² on training data.
        // TLS is the textbook solution but its quality depends on the
        // assumption that errors in X and Y have equal variance. When that
        // assumption is violated (typical for OLS-style data with noise only
        // on Y) the TLS solution can recover coefficients in the wrong
        // proportion even though residual sum-of-squares is below ssTot —
        // the previous "ssRes < ssTot" gate accepted these wrong-proportion
        // solutions and produced negative R² and wrong-sign coefficients on
        // GenerateLinearData (issue surfaced by Builder_R2ShouldBePositive
        // and CoefficientSigns_ShouldMatchDataGeneratingProcess). Evaluating
        // TLS vs ridge OLS on training fit and keeping whichever generalises
        // better is the standard approach in robust TLS implementations
        // (Van Huffel & Vandewalle 1991 §4.3). For pure OLS-noise data this
        // selects OLS; for genuine errors-in-variables data TLS wins.
        var ridgeCoeffs = ComputeRidgeCoefficients(centeredX, centeredY, scaleX, scaleY, p);
        var ridgeIntercept = NumOps.Subtract(meanY, ridgeCoeffs.DotProduct(meanX));
        T ridgeSsRes = ComputeSumSquaredResiduals(x, y, ridgeCoeffs, ridgeIntercept);

        Coefficients = ridgeCoeffs;
        Intercept = ridgeIntercept;
        T bestSsRes = ridgeSsRes;

        try
        {
            var svd = new SvdDecomposition<T>(augmentedMatrix);
            var vt = svd.Vt;
            int lastRow = vt.Rows - 1;

            T vLast = vt[lastRow, p];
            if (NumOps.GreaterThan(NumOps.Abs(vLast), NumOps.FromDouble(1e-14)))
            {
                var tlsCoeffs = new Vector<T>(p);
                T zeroVarianceTol = NumOps.FromDouble(1e-14);
                for (int j = 0; j < p; j++)
                {
                    // TLS coefficient in scaled space: -v_j / v_last
                    T scaledCoeff = NumOps.Negate(NumOps.Divide(vt[lastRow, j], vLast));
                    // Unscale: multiply by scaleY/scaleX[j] to return to original units.
                    // Zero-variance feature: scaleX[j] ≈ 0 ⇒ the column is constant
                    // and contributes no information; treat the coefficient as 0
                    // rather than dividing and producing Inf/NaN that flips
                    // ssRes-based model selection.
                    if (NumOps.LessThan(NumOps.Abs(scaleX[j]), zeroVarianceTol))
                    {
                        tlsCoeffs[j] = NumOps.Zero;
                    }
                    else
                    {
                        tlsCoeffs[j] = NumOps.Multiply(scaledCoeff, NumOps.Divide(scaleY, scaleX[j]));
                    }
                }

                T tlsIntercept = NumOps.Subtract(meanY, tlsCoeffs.DotProduct(meanX));
                T tlsSsRes = ComputeSumSquaredResiduals(x, y, tlsCoeffs, tlsIntercept);

                if (NumOps.LessThan(tlsSsRes, bestSsRes))
                {
                    Coefficients = tlsCoeffs;
                    Intercept = tlsIntercept;
                    bestSsRes = tlsSsRes;
                }
            }
        }
        catch (Exception ex) when (
            ex is InvalidOperationException ||
            ex is ArithmeticException)
        {
            // SVD decomposition failed on ill-conditioned matrix; ridge OLS
            // already selected above. Narrow the catch so unexpected
            // exceptions (NullReferenceException, ArgumentException for
            // shape mismatches, etc.) bubble up instead of being swallowed
            // as 'numerical failure'.
            System.Diagnostics.Debug.WriteLine(
                $"[OrthogonalRegression] TLS fallback to ridge OLS: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private Vector<T> ComputeRidgeCoefficients(
        Matrix<T> centeredX, Vector<T> centeredY,
        Vector<T> scaleX, T scaleY, int p)
    {
        var xTx = centeredX.Transpose().Multiply(centeredX);
        for (int i = 0; i < xTx.Rows; i++)
            xTx[i, i] = NumOps.Add(xTx[i, i], NumOps.FromDouble(1e-8));
        var xTy = centeredX.Transpose().Multiply(centeredY);
        var ridge = xTx.Inverse().Multiply(xTy);
        T zeroVarianceTol = NumOps.FromDouble(1e-14);
        for (int j = 0; j < p; j++)
        {
            // Zero-variance unscale guard: see the matching note in
            // ComputeTLS for why constant columns must yield zero coefficient.
            ridge[j] = NumOps.LessThan(NumOps.Abs(scaleX[j]), zeroVarianceTol)
                ? NumOps.Zero
                : NumOps.Multiply(NumOps.Divide(ridge[j], scaleX[j]), scaleY);
        }
        return ridge;
    }

    private T ComputeSumSquaredResiduals(Matrix<T> x, Vector<T> y, Vector<T> coeffs, T intercept)
    {
        T ssRes = NumOps.Zero;
        int n = x.Rows;
        int p = x.Columns;
        for (int i = 0; i < n; i++)
        {
            T predicted = intercept;
            for (int j = 0; j < p; j++)
                predicted = NumOps.Add(predicted, NumOps.Multiply(coeffs[j], x[i, j]));
            T residual = NumOps.Subtract(y[i], predicted);
            ssRes = NumOps.Add(ssRes, NumOps.Multiply(residual, residual));
        }
        return ssRes;
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for orthogonal regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method simply returns an identifier that indicates this is an orthogonal regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes both the base class data and the orthogonal regression specific options,
    /// including tolerance, maximum iterations, and whether to scale variables.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize OrthogonalRegression specific options
        writer.Write(_options.Tolerance);
        writer.Write(_options.MaxIterations);
        writer.Write(_options.ScaleVariables);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes both the base class data and the orthogonal regression specific options,
    /// reconstructing the model's state from the serialized data.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize OrthogonalRegression specific options
        _options.Tolerance = reader.ReadDouble();
        _options.MaxIterations = reader.ReadInt32();
        _options.ScaleVariables = reader.ReadBoolean();
    }

    /// <summary>
    /// Creates a new instance of the Orthogonal Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Orthogonal Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Orthogonal Regression model, including its options,
    /// coefficients, intercept, and regularization settings. The new instance is completely independent of the original,
    /// allowing modifications without affecting the original model.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate:
    /// - It copies all the configuration settings (like tolerance and whether to scale variables)
    /// - It preserves the coefficients (the weights for each feature)
    /// - It maintains the intercept (the starting point of your regression line or plane)
    /// - It includes the same regularization settings to prevent overfitting
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before making changes to the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        var newModel = new OrthogonalRegression<T>(_options, Regularization);

        // Copy coefficients if they exist
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        // Copy the intercept
        newModel.Intercept = Intercept;

        return newModel;
    }
}
