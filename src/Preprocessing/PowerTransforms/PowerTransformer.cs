using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.PowerTransforms;

/// <summary>
/// Specifies the power transformation method.
/// </summary>
public enum PowerTransformMethod
{
    /// <summary>
    /// Box-Cox transformation. Requires strictly positive data.
    /// </summary>
    BoxCox,

    /// <summary>
    /// Yeo-Johnson transformation. Works with positive, negative, and zero values.
    /// </summary>
    YeoJohnson
}

/// <summary>
/// Applies power transformations (Box-Cox or Yeo-Johnson) to make data more Gaussian-like.
/// </summary>
/// <remarks>
/// <para>
/// PowerTransformer applies a power transformation to each feature to make it more Gaussian-like.
/// Box-Cox requires strictly positive data, while Yeo-Johnson works with any values.
/// The transformation can help stabilize variance and improve the fit of models that assume normality.
/// </para>
/// <para><b>For Beginners:</b> This transformer makes your data more "bell-curve shaped":
/// - Box-Cox: For strictly positive data (prices, counts)
/// - Yeo-Johnson: Works with any data including negatives and zeros
///
/// After transformation, features will have more normal (Gaussian) distributions,
/// which helps many machine learning algorithms perform better.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PowerTransformer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly PowerTransformMethod _method;
    private readonly bool _standardize;

    // Fitted parameters
    private Vector<T>? _lambdas;
    private Vector<T>? _mean;
    private Vector<T>? _stdDev;

    /// <summary>
    /// Gets the power transformation method used.
    /// </summary>
    public PowerTransformMethod Method => _method;

    /// <summary>
    /// Gets whether standardization is applied after transformation.
    /// </summary>
    public bool Standardize => _standardize;

    /// <summary>
    /// Gets the optimal lambda parameters for each feature.
    /// </summary>
    public Vector<T>? Lambdas => _lambdas;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="PowerTransformer{T}"/>.
    /// </summary>
    /// <param name="method">The transformation method. Defaults to YeoJohnson.</param>
    /// <param name="standardize">Whether to standardize after transformation. Defaults to true.</param>
    /// <param name="columnIndices">The column indices to transform, or null for all columns.</param>
    public PowerTransformer(
        PowerTransformMethod method = PowerTransformMethod.YeoJohnson,
        bool standardize = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _method = method;
        _standardize = standardize;
    }

    /// <summary>
    /// Learns the optimal lambda parameters for each feature.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        var lambdas = new T[numColumns];
        var means = new T[numColumns];
        var stdDevs = new T[numColumns];

        // Default values for columns not processed
        for (int i = 0; i < numColumns; i++)
        {
            lambdas[i] = NumOps.One;
            means[i] = NumOps.Zero;
            stdDevs[i] = NumOps.One;
        }

        foreach (var col in columnsToProcess)
        {
            var column = data.GetColumn(col);

            // Check for Box-Cox constraint
            if (_method == PowerTransformMethod.BoxCox)
            {
                for (int i = 0; i < column.Length; i++)
                {
                    if (NumOps.Compare(column[i], NumOps.Zero) <= 0)
                    {
                        throw new ArgumentException(
                            $"Box-Cox transformation requires strictly positive data. " +
                            $"Found non-positive value in column {col}. Use YeoJohnson instead.");
                    }
                }
            }

            // Find optimal lambda using simple grid search
            double bestLambda = FindOptimalLambda(column);
            lambdas[col] = NumOps.FromDouble(bestLambda);

            // Transform data and compute mean/stddev for standardization
            if (_standardize)
            {
                var transformed = TransformColumn(column, NumOps.FromDouble(bestLambda));

                T sum = NumOps.Zero;
                for (int i = 0; i < transformed.Length; i++)
                {
                    sum = NumOps.Add(sum, transformed[i]);
                }
                T mean = NumOps.Divide(sum, NumOps.FromDouble(transformed.Length));

                T varianceSum = NumOps.Zero;
                for (int i = 0; i < transformed.Length; i++)
                {
                    T diff = NumOps.Subtract(transformed[i], mean);
                    varianceSum = NumOps.Add(varianceSum, NumOps.Multiply(diff, diff));
                }
                T variance = NumOps.Divide(varianceSum, NumOps.FromDouble(transformed.Length));
                T stdDev = NumOps.Sqrt(variance);
                if (NumOps.Compare(stdDev, NumOps.Zero) == 0)
                {
                    stdDev = NumOps.One;
                }

                means[col] = mean;
                stdDevs[col] = stdDev;
            }
        }

        _lambdas = new Vector<T>(lambdas);
        _mean = new Vector<T>(means);
        _stdDev = new Vector<T>(stdDevs);
    }

    private double FindOptimalLambda(Vector<T> column)
    {
        // Simple grid search for lambda that minimizes log-likelihood
        double bestLambda = 1.0;
        double bestScore = double.MinValue;

        // Search range and step
        double minLambda = -2.0;
        double maxLambda = 2.0;
        double step = 0.1;

        for (double lambda = minLambda; lambda <= maxLambda; lambda += step)
        {
            double score = ComputeLogLikelihood(column, lambda);
            if (score > bestScore)
            {
                bestScore = score;
                bestLambda = lambda;
            }
        }

        return bestLambda;
    }

    private double ComputeLogLikelihood(Vector<T> column, double lambda)
    {
        var transformed = TransformColumn(column, NumOps.FromDouble(lambda));

        // Compute mean and variance
        double sum = 0;
        for (int i = 0; i < transformed.Length; i++)
        {
            sum += NumOps.ToDouble(transformed[i]);
        }
        double mean = sum / transformed.Length;

        double variance = 0;
        for (int i = 0; i < transformed.Length; i++)
        {
            double diff = NumOps.ToDouble(transformed[i]) - mean;
            variance += diff * diff;
        }
        variance /= transformed.Length;

        if (variance <= 0) return double.MinValue;

        // Log-likelihood for normal distribution
        double logLikelihood = -0.5 * transformed.Length * Math.Log(2 * Math.PI * variance);
        logLikelihood -= 0.5 * transformed.Length; // Normalized sum of squared deviations

        // Add Jacobian term
        for (int i = 0; i < column.Length; i++)
        {
            double x = NumOps.ToDouble(column[i]);
            if (_method == PowerTransformMethod.BoxCox)
            {
                logLikelihood += (lambda - 1) * Math.Log(x);
            }
            else // Yeo-Johnson
            {
                if (x >= 0)
                {
                    logLikelihood += (lambda - 1) * Math.Log(x + 1);
                }
                else
                {
                    logLikelihood += (1 - lambda) * Math.Log(1 - x);
                }
            }
        }

        return logLikelihood;
    }

    private T[] TransformColumn(Vector<T> column, T lambda)
    {
        var result = new T[column.Length];
        double lambdaDouble = NumOps.ToDouble(lambda);

        for (int i = 0; i < column.Length; i++)
        {
            double x = NumOps.ToDouble(column[i]);
            double transformed;

            if (_method == PowerTransformMethod.BoxCox)
            {
                transformed = BoxCoxTransform(x, lambdaDouble);
            }
            else
            {
                transformed = YeoJohnsonTransform(x, lambdaDouble);
            }

            result[i] = NumOps.FromDouble(transformed);
        }

        return result;
    }

    private double BoxCoxTransform(double x, double lambda)
    {
        if (Math.Abs(lambda) < 1e-10)
        {
            return Math.Log(x);
        }
        return (Math.Pow(x, lambda) - 1) / lambda;
    }

    private double YeoJohnsonTransform(double x, double lambda)
    {
        if (x >= 0)
        {
            if (Math.Abs(lambda) < 1e-10)
            {
                return Math.Log(x + 1);
            }
            return (Math.Pow(x + 1, lambda) - 1) / lambda;
        }
        else
        {
            if (Math.Abs(lambda - 2) < 1e-10)
            {
                return -Math.Log(1 - x);
            }
            return -(Math.Pow(1 - x, 2 - lambda) - 1) / (2 - lambda);
        }
    }

    private double InverseBoxCoxTransform(double y, double lambda)
    {
        if (Math.Abs(lambda) < 1e-10)
        {
            return Math.Exp(y);
        }
        return Math.Pow(lambda * y + 1, 1 / lambda);
    }

    private double InverseYeoJohnsonTransform(double y, double lambda)
    {
        if (y >= 0)
        {
            if (Math.Abs(lambda) < 1e-10)
            {
                return Math.Exp(y) - 1;
            }
            return Math.Pow(lambda * y + 1, 1 / lambda) - 1;
        }
        else
        {
            if (Math.Abs(lambda - 2) < 1e-10)
            {
                return 1 - Math.Exp(-y);
            }
            return 1 - Math.Pow(-(2 - lambda) * y + 1, 1 / (2 - lambda));
        }
    }

    /// <summary>
    /// Transforms the data by applying the power transformation.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lambdas is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    double x = NumOps.ToDouble(value);
                    double lambda = NumOps.ToDouble(_lambdas[j]);
                    double transformed;

                    if (_method == PowerTransformMethod.BoxCox)
                    {
                        transformed = BoxCoxTransform(x, lambda);
                    }
                    else
                    {
                        transformed = YeoJohnsonTransform(x, lambda);
                    }

                    if (_standardize && _mean is not null && _stdDev is not null)
                    {
                        transformed = (transformed - NumOps.ToDouble(_mean[j])) / NumOps.ToDouble(_stdDev[j]);
                    }

                    value = NumOps.FromDouble(transformed);
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the power transformation.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_lambdas is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                T value = data[i, j];

                if (processSet.Contains(j))
                {
                    double y = NumOps.ToDouble(value);
                    double lambda = NumOps.ToDouble(_lambdas[j]);

                    // Undo standardization first
                    if (_standardize && _mean is not null && _stdDev is not null)
                    {
                        y = y * NumOps.ToDouble(_stdDev[j]) + NumOps.ToDouble(_mean[j]);
                    }

                    double original;
                    if (_method == PowerTransformMethod.BoxCox)
                    {
                        original = InverseBoxCoxTransform(y, lambda);
                    }
                    else
                    {
                        original = InverseYeoJohnsonTransform(y, lambda);
                    }

                    value = NumOps.FromDouble(original);
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (PowerTransformer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
