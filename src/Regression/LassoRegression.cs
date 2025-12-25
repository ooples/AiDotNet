namespace AiDotNet.Regression;

/// <summary>
/// Implements Lasso Regression (L1 regularized linear regression), which extends ordinary least squares
/// by adding a penalty term proportional to the absolute magnitude of the coefficients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Lasso Regression solves the following optimization problem:
/// minimize (1/2n) * ||y - Xw||^2 + alpha * ||w||_1
///
/// Unlike Ridge Regression, Lasso uses coordinate descent optimization because
/// the L1 penalty is not differentiable at zero.
///
/// The L1 penalty can shrink coefficients exactly to zero, making Lasso
/// useful for automatic feature selection.
/// </para>
/// <para><b>For Beginners:</b> Lasso Regression automatically selects important features.
///
/// While Ridge Regression keeps all features but shrinks their coefficients,
/// Lasso can completely eliminate unimportant features by setting their
/// coefficients to zero. This is useful when:
/// - You have many features and want to identify the most important ones
/// - You want a simpler, more interpretable model
/// - You suspect only a few features actually matter
///
/// Example usage:
/// ```csharp
/// var options = new LassoRegressionOptions&lt;double&gt; { Alpha = 1.0 };
/// var lasso = new LassoRegression&lt;double&gt;(options);
/// lasso.Train(features, targets);
/// var predictions = lasso.Predict(newFeatures);
///
/// // Check which features were selected (non-zero coefficients)
/// var selectedFeatures = lasso.GetActiveFeatureIndices();
/// ```
/// </para>
/// </remarks>
public class LassoRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Gets the configuration options specific to Lasso Regression.
    /// </summary>
    private new LassoRegressionOptions<T> Options => (LassoRegressionOptions<T>)base.Options;

    /// <summary>
    /// Stores the number of iterations used in the last training.
    /// </summary>
    private int _iterationsUsed;

    /// <summary>
    /// Initializes a new instance of the <see cref="LassoRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for Lasso Regression. If null, default options are used.</param>
    /// <param name="regularization">Optional additional regularization strategy.</param>
    /// <remarks>
    /// <para>
    /// Creates a new Lasso Regression model with the specified options. The primary regularization
    /// is controlled by the Alpha parameter in the options.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Lasso Regression model.
    ///
    /// You can customize the model by providing options:
    /// ```csharp
    /// // Create with default options (alpha = 1.0)
    /// var lasso = new LassoRegression&lt;double&gt;();
    ///
    /// // Create with custom alpha (more aggressive feature selection)
    /// var options = new LassoRegressionOptions&lt;double&gt; { Alpha = 2.0 };
    /// var lasso = new LassoRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public LassoRegression(LassoRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new LassoRegressionOptions<T>(), regularization)
    {
    }

    /// <summary>
    /// Trains the Lasso Regression model using coordinate descent optimization.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each sample.</param>
    /// <remarks>
    /// <para>
    /// Training uses coordinate descent optimization, which updates one coefficient at a time
    /// while holding others fixed. For each coefficient, the update uses the soft-thresholding
    /// operator to apply the L1 penalty.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to make predictions from your data.
    ///
    /// The training process uses an iterative algorithm called coordinate descent:
    /// 1. Start with all coefficients at zero (or warm start from previous solution)
    /// 2. For each feature, compute the optimal coefficient value
    /// 3. Apply "soft thresholding" which can set small values to exactly zero
    /// 4. Repeat until coefficients stop changing significantly
    ///
    /// This approach is slower than Ridge Regression's direct solution, but it's
    /// the only way to get exact zero coefficients for feature selection.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;

        Matrix<T> xProcessed = x;

        // Add intercept column if needed
        if (Options.UseIntercept)
        {
            xProcessed = xProcessed.AddConstantColumn(NumOps.One);
            p = xProcessed.Columns;
        }

        // Initialize coefficients (warm start or zeros)
        // Warm start only if coefficients were previously set (Length > 0) and match expected size
        Vector<T> w;
        if (Options.WarmStart && Coefficients.Length > 0 && Coefficients.Length == (Options.UseIntercept ? p - 1 : p))
        {
            // Use existing coefficients as warm start
            w = new Vector<T>(p);
            if (Options.UseIntercept)
            {
                w[0] = Intercept;
                for (int j = 0; j < Coefficients.Length; j++)
                {
                    w[j + 1] = Coefficients[j];
                }
            }
            else
            {
                for (int j = 0; j < Coefficients.Length; j++)
                {
                    w[j] = Coefficients[j];
                }
            }
        }
        else
        {
            w = new Vector<T>(p);
        }

        // Precompute X^T X diagonal and X^T for efficiency
        var xSquaredSum = new Vector<T>(p);
        for (int j = 0; j < p; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(xProcessed[i, j], xProcessed[i, j]));
            }
            xSquaredSum[j] = sum;
        }

        // Compute initial residuals
        var residuals = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            T prediction = NumOps.Zero;
            for (int j = 0; j < p; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(xProcessed[i, j], w[j]));
            }
            residuals[i] = NumOps.Subtract(y[i], prediction);
        }

        T alpha = NumOps.FromDouble(Options.Alpha);
        T tolerance = NumOps.FromDouble(Options.Tolerance);
        int startIdx = Options.UseIntercept ? 1 : 0; // Don't regularize intercept

        // Coordinate descent
        for (_iterationsUsed = 0; _iterationsUsed < Options.MaxIterations; _iterationsUsed++)
        {
            T maxChange = NumOps.Zero;

            for (int j = 0; j < p; j++)
            {
                T oldW = w[j];

                // Compute partial residual (add back contribution of current coefficient)
                for (int i = 0; i < n; i++)
                {
                    residuals[i] = NumOps.Add(residuals[i], NumOps.Multiply(xProcessed[i, j], oldW));
                }

                // Compute correlation with residuals
                T rho = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    rho = NumOps.Add(rho, NumOps.Multiply(xProcessed[i, j], residuals[i]));
                }

                // Apply soft-thresholding (don't regularize intercept)
                // Intercept or zero column gets no regularization; otherwise apply L1 soft-thresholding
                T newW = (j < startIdx || NumOps.Equals(xSquaredSum[j], NumOps.Zero))
                    ? (NumOps.Equals(xSquaredSum[j], NumOps.Zero) ? NumOps.Zero : NumOps.Divide(rho, xSquaredSum[j]))
                    : SoftThreshold(rho, alpha, xSquaredSum[j]);

                w[j] = newW;

                // Update residuals
                for (int i = 0; i < n; i++)
                {
                    residuals[i] = NumOps.Subtract(residuals[i], NumOps.Multiply(xProcessed[i, j], newW));
                }

                // Track maximum change
                T change = NumOps.Abs(NumOps.Subtract(newW, oldW));
                if (NumOps.Compare(change, maxChange) > 0)
                {
                    maxChange = change;
                }
            }

            // Check convergence
            if (NumOps.Compare(maxChange, tolerance) < 0)
            {
                break;
            }
        }

        // Extract coefficients and intercept
        if (Options.UseIntercept)
        {
            Intercept = w[0];
            Coefficients = new Vector<T>(p - 1);
            for (int j = 1; j < p; j++)
            {
                Coefficients[j - 1] = w[j];
            }
        }
        else
        {
            Intercept = NumOps.Zero;
            Coefficients = w;
        }
    }

    /// <summary>
    /// Applies the soft-thresholding operator for L1 regularization.
    /// </summary>
    /// <param name="rho">The correlation value.</param>
    /// <param name="alpha">The regularization parameter.</param>
    /// <param name="xSquaredSum">The sum of squared feature values.</param>
    /// <returns>The soft-thresholded coefficient.</returns>
    private T SoftThreshold(T rho, T alpha, T xSquaredSum)
    {
        // Soft-thresholding: sign(rho) * max(0, |rho| - alpha) / xSquaredSum
        T absRho = NumOps.Abs(rho);

        if (NumOps.Compare(absRho, alpha) <= 0)
        {
            return NumOps.Zero;
        }

        T sign = NumOps.Compare(rho, NumOps.Zero) >= 0 ? NumOps.One : NumOps.Negate(NumOps.One);
        T magnitude = NumOps.Divide(NumOps.Subtract(absRho, alpha), xSquaredSum);

        return NumOps.Multiply(sign, magnitude);
    }

    /// <summary>
    /// Gets the number of iterations used in the last training.
    /// </summary>
    public int IterationsUsed => _iterationsUsed;

    /// <summary>
    /// Gets the number of non-zero coefficients (selected features).
    /// </summary>
    public int NumberOfSelectedFeatures => GetActiveFeatureIndices().Count();

    /// <summary>
    /// Gets metadata about the Lasso Regression model.
    /// </summary>
    /// <returns>A ModelMetadata object containing model information.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Alpha"] = Options.Alpha;
        metadata.AdditionalInfo["MaxIterations"] = Options.MaxIterations;
        metadata.AdditionalInfo["Tolerance"] = Options.Tolerance;
        metadata.AdditionalInfo["IterationsUsed"] = _iterationsUsed;
        metadata.AdditionalInfo["NumberOfSelectedFeatures"] = NumberOfSelectedFeatures;

        return metadata;
    }

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    /// <returns>The ModelType enumeration value for Lasso Regression.</returns>
    protected override ModelType GetModelType()
    {
        return ModelType.LassoRegression;
    }

    /// <summary>
    /// Creates a new instance of Lasso Regression with the same configuration.
    /// </summary>
    /// <returns>A new instance with the same options.</returns>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new LassoRegression<T>(Options, Regularization);
    }

    /// <summary>
    /// Serializes the Lasso Regression model to a byte array.
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

        // Serialize Lasso-specific data
        writer.Write(Options.Alpha);
        writer.Write(Options.MaxIterations);
        writer.Write(Options.Tolerance);
        writer.Write(Options.WarmStart);
        writer.Write(_iterationsUsed);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a Lasso Regression model from a byte array.
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

        // Deserialize Lasso-specific data
        Options.Alpha = reader.ReadDouble();
        Options.MaxIterations = reader.ReadInt32();
        Options.Tolerance = reader.ReadDouble();
        Options.WarmStart = reader.ReadBoolean();
        _iterationsUsed = reader.ReadInt32();
    }
}
