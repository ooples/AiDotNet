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
/// var options = new RidgeRegressionOptions&lt;double&gt; { Alpha = 1.0 };
/// var ridge = new RidgeRegression&lt;double&gt;(options);
/// ridge.Train(features, targets);
/// var predictions = ridge.Predict(newFeatures);
/// ```
/// </para>
/// </remarks>
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
        Matrix<T> xProcessed = x;

        // Add intercept column if needed
        if (Options.UseIntercept)
        {
            xProcessed = xProcessed.AddConstantColumn(NumOps.One);
        }

        int numFeatures = xProcessed.Columns;

        // Compute X^T X
        var xTx = xProcessed.Transpose().Multiply(xProcessed);

        // Add alpha * I to the diagonal (regularization term)
        // Note: If using intercept, we typically don't regularize the intercept term
        T alpha = NumOps.FromDouble(Options.Alpha);
        int startIdx = Options.UseIntercept ? 1 : 0;
        for (int i = startIdx; i < numFeatures; i++)
        {
            xTx[i, i] = NumOps.Add(xTx[i, i], alpha);
        }

        // Apply additional regularization if provided
        var regularizedXTx = xTx.Add(Regularization.Regularize(xTx));

        // Compute X^T y
        var xTy = xProcessed.Transpose().Multiply(y);

        // Solve (X^T X + alpha * I) * w = X^T y
        var solution = SolveSystemWithDecomposition(regularizedXTx, xTy);

        // Extract coefficients and intercept
        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = new Vector<T>(numFeatures - 1);
            for (int i = 1; i < numFeatures; i++)
            {
                Coefficients[i - 1] = solution[i];
            }
        }
        else
        {
            Intercept = NumOps.Zero;
            Coefficients = solution;
        }
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

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    /// <returns>The ModelType enumeration value for Ridge Regression.</returns>
    protected override ModelType GetModelType()
    {
        return ModelType.RidgeRegression;
    }

    /// <summary>
    /// Creates a new instance of Ridge Regression with the same configuration.
    /// </summary>
    /// <returns>A new instance with the same options.</returns>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new RidgeRegression<T>(Options, Regularization);
    }

    /// <summary>
    /// Serializes the Ridge Regression model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize Ridge-specific data
        writer.Write(Options.Alpha);
        writer.Write((int)Options.DecompositionType);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a Ridge Regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize Ridge-specific data
        Options.Alpha = reader.ReadDouble();
        Options.DecompositionType = (MatrixDecompositionType)reader.ReadInt32();
    }
}
