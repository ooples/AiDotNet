namespace AiDotNet.Regression;

/// <summary>
/// Implements Elastic Net Regression (combined L1 and L2 regularized linear regression),
/// which extends ordinary least squares by adding both L1 (Lasso) and L2 (Ridge) penalty terms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Elastic Net Regression solves the following optimization problem:
/// minimize (1/2n) * ||y - Xw||^2 + alpha * l1_ratio * ||w||_1 + alpha * (1 - l1_ratio) * ||w||^2 / 2
///
/// Like Lasso, Elastic Net uses coordinate descent optimization because
/// the L1 penalty is not differentiable at zero.
///
/// Elastic Net combines the benefits of both Ridge and Lasso:
/// - Feature selection from L1 (can set coefficients to exactly zero)
/// - Stability with correlated features from L2 (groups correlated features together)
/// </para>
/// <para><b>For Beginners:</b> Elastic Net gives you the best of both worlds.
///
/// Lasso is great at selecting important features, but when features are correlated,
/// it tends to arbitrarily pick one and zero out the others. Ridge handles correlated
/// features well but doesn't do feature selection.
///
/// Elastic Net solves both problems:
/// - It can still set coefficients to zero (like Lasso) for feature selection
/// - It groups correlated features together (like Ridge) instead of picking arbitrarily
///
/// Example usage:
/// ```csharp
/// var options = new ElasticNetRegressionOptions&lt;double&gt; { Alpha = 1.0, L1Ratio = 0.5 };
/// var elasticNet = new ElasticNetRegression&lt;double&gt;(options);
/// elasticNet.Train(features, targets);
/// var predictions = elasticNet.Predict(newFeatures);
///
/// // Check which features were selected (non-zero coefficients)
/// var selectedFeatures = elasticNet.GetActiveFeatureIndices();
/// ```
/// </para>
/// </remarks>
public class ElasticNetRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Gets the configuration options specific to Elastic Net Regression.
    /// </summary>
    private new ElasticNetRegressionOptions<T> Options => (ElasticNetRegressionOptions<T>)base.Options;

    /// <summary>
    /// Stores the number of iterations used in the last training.
    /// </summary>
    private int _iterationsUsed;

    /// <summary>
    /// Initializes a new instance of the <see cref="ElasticNetRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for Elastic Net Regression. If null, default options are used.</param>
    /// <param name="regularization">Optional additional regularization strategy.</param>
    /// <remarks>
    /// <para>
    /// Creates a new Elastic Net Regression model with the specified options. The primary regularization
    /// is controlled by the Alpha and L1Ratio parameters in the options.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Elastic Net Regression model.
    ///
    /// You can customize the model by providing options:
    /// ```csharp
    /// // Create with default options (alpha = 1.0, l1_ratio = 0.5)
    /// var elasticNet = new ElasticNetRegression&lt;double&gt;();
    ///
    /// // Create with custom parameters (more Lasso-like)
    /// var options = new ElasticNetRegressionOptions&lt;double&gt; { Alpha = 1.0, L1Ratio = 0.7 };
    /// var elasticNet = new ElasticNetRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public ElasticNetRegression(ElasticNetRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new ElasticNetRegressionOptions<T>(), regularization)
    {
    }

    /// <summary>
    /// Trains the Elastic Net Regression model using coordinate descent optimization.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each sample.</param>
    /// <remarks>
    /// <para>
    /// Training uses coordinate descent optimization, which updates one coefficient at a time
    /// while holding others fixed. For each coefficient, the update uses a modified soft-thresholding
    /// operator that accounts for both L1 and L2 penalties.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to make predictions from your data.
    ///
    /// The training process uses coordinate descent similar to Lasso:
    /// 1. Start with all coefficients at zero (or warm start from previous solution)
    /// 2. For each feature, compute the optimal coefficient value
    /// 3. Apply soft thresholding with L1 penalty, scaled by L2 penalty
    /// 4. Repeat until coefficients stop changing significantly
    ///
    /// The L2 penalty makes the solution more stable and helps when features are correlated.
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
        T l1Ratio = NumOps.FromDouble(Options.L1Ratio);
        T l2Ratio = NumOps.Subtract(NumOps.One, l1Ratio);
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

                // Apply elastic net soft-thresholding (don't regularize intercept)
                // Intercept or zero column gets no regularization; otherwise apply elastic net
                T newW = (j < startIdx || NumOps.Equals(xSquaredSum[j], NumOps.Zero))
                    ? (NumOps.Equals(xSquaredSum[j], NumOps.Zero) ? NumOps.Zero : NumOps.Divide(rho, xSquaredSum[j]))
                    : ElasticNetSoftThreshold(rho, alpha, l1Ratio, l2Ratio, xSquaredSum[j]);

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
    /// Applies the elastic net soft-thresholding operator for combined L1 and L2 regularization.
    /// </summary>
    /// <param name="rho">The correlation value.</param>
    /// <param name="alpha">The overall regularization parameter.</param>
    /// <param name="l1Ratio">The ratio of L1 penalty (0 to 1).</param>
    /// <param name="l2Ratio">The ratio of L2 penalty (1 - l1Ratio).</param>
    /// <param name="xSquaredSum">The sum of squared feature values.</param>
    /// <returns>The elastic net soft-thresholded coefficient.</returns>
    /// <remarks>
    /// The elastic net update is:
    /// w_j = sign(rho) * max(0, |rho| - alpha * l1_ratio) / (x_squared_sum + alpha * l2_ratio)
    /// </remarks>
    private T ElasticNetSoftThreshold(T rho, T alpha, T l1Ratio, T l2Ratio, T xSquaredSum)
    {
        // Compute L1 threshold: alpha * l1_ratio
        T l1Threshold = NumOps.Multiply(alpha, l1Ratio);

        // Compute L2 denominator adjustment: alpha * l2_ratio
        T l2Adjustment = NumOps.Multiply(alpha, l2Ratio);

        T absRho = NumOps.Abs(rho);

        // Soft-thresholding with L1 penalty
        if (NumOps.Compare(absRho, l1Threshold) <= 0)
        {
            return NumOps.Zero;
        }

        T sign = NumOps.Compare(rho, NumOps.Zero) >= 0 ? NumOps.One : NumOps.Negate(NumOps.One);
        T numerator = NumOps.Subtract(absRho, l1Threshold);
        T denominator = NumOps.Add(xSquaredSum, l2Adjustment);

        return NumOps.Multiply(sign, NumOps.Divide(numerator, denominator));
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
    /// Gets metadata about the Elastic Net Regression model.
    /// </summary>
    /// <returns>A ModelMetadata object containing model information.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Alpha"] = Options.Alpha;
        metadata.AdditionalInfo["L1Ratio"] = Options.L1Ratio;
        metadata.AdditionalInfo["MaxIterations"] = Options.MaxIterations;
        metadata.AdditionalInfo["Tolerance"] = Options.Tolerance;
        metadata.AdditionalInfo["IterationsUsed"] = _iterationsUsed;
        metadata.AdditionalInfo["NumberOfSelectedFeatures"] = NumberOfSelectedFeatures;

        return metadata;
    }

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    /// <returns>The ModelType enumeration value for Elastic Net Regression.</returns>
    protected override ModelType GetModelType()
    {
        return ModelType.ElasticNetRegression;
    }

    /// <summary>
    /// Creates a new instance of Elastic Net Regression with the same configuration.
    /// </summary>
    /// <returns>A new instance with the same options.</returns>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new ElasticNetRegression<T>(Options, Regularization);
    }

    /// <summary>
    /// Serializes the Elastic Net Regression model to a byte array.
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

        // Serialize Elastic Net-specific data
        writer.Write(Options.Alpha);
        writer.Write(Options.L1Ratio);
        writer.Write(Options.MaxIterations);
        writer.Write(Options.Tolerance);
        writer.Write(Options.WarmStart);
        writer.Write(_iterationsUsed);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes an Elastic Net Regression model from a byte array.
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

        // Deserialize Elastic Net-specific data
        Options.Alpha = reader.ReadDouble();
        Options.L1Ratio = reader.ReadDouble();
        Options.MaxIterations = reader.ReadInt32();
        Options.Tolerance = reader.ReadDouble();
        Options.WarmStart = reader.ReadBoolean();
        _iterationsUsed = reader.ReadInt32();
    }
}
