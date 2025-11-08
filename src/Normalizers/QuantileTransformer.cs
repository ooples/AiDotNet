namespace AiDotNet.Normalizers;

/// <summary>
/// Transforms features to follow a uniform or normal distribution using quantile information.
/// </summary>
/// <remarks>
/// <para>
/// The QuantileTransformer applies a non-linear transformation that maps the input distribution to
/// either a uniform or normal (Gaussian) distribution. This transformation is based on computing
/// quantiles from the training data and using them to map values. It is particularly effective at
/// reducing the impact of outliers and can make non-normal distributions more suitable for algorithms
/// that assume normality.
/// </para>
/// <para>
/// The transformation works by:
/// 1. Computing n quantiles from the training data (default: 1000)
/// 2. For each value, finding its rank in the quantile distribution
/// 3. Mapping this rank to the target distribution (uniform or normal)
/// </para>
/// <para>
/// This method is ideal when:
/// - Your data has a highly skewed or non-normal distribution
/// - You need robust handling of outliers
/// - Your algorithm performs better with normally-distributed features
/// - You want to reduce the impact of extreme values
/// </para>
/// <para><b>For Beginners:</b> QuantileTransformer is like redistributing data to match a desired pattern.
///
/// Think of it as reorganizing exam scores:
/// - Original scores might be clustered (many 70s and 80s, few 90s and 100s)
/// - Uniform distribution spreads them evenly (equal numbers at each level)
/// - Normal distribution creates a bell curve (most in the middle, fewer at extremes)
///
/// How it works:
/// 1. During training, it learns the "shape" of your data by computing many quantiles
/// 2. For each new value, it finds where it ranks (percentile)
/// 3. It maps this ranking to the target distribution
///
/// For example, with uniform output:
/// - Original: [1, 1, 2, 3, 5, 8, 13, 100, 1000]
/// - After transform: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
/// - Notice how extreme values (100, 1000) are no longer dominating
///
/// Benefits:
/// - Makes outliers less problematic
/// - Works with any input distribution
/// - Can make data more suitable for many ML algorithms
/// - Preserves the rank ordering of values
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
public class QuantileTransformer<T, TInput, TOutput> : NormalizerBase<T, TInput, TOutput>
{
    private readonly string _outputDistribution;
    private readonly int _nQuantiles;

    /// <summary>
    /// Initializes a new instance of the <see cref="QuantileTransformer{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="outputDistribution">The target distribution: "uniform" or "normal". Default is "uniform".</param>
    /// <param name="nQuantiles">The number of quantiles to compute. Default is 1000.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up your quantile transformation system.
    ///
    /// Parameters to choose:
    /// - outputDistribution: "uniform" spreads values evenly, "normal" creates a bell curve
    /// - nQuantiles: More quantiles (1000) = more accurate but slower, fewer (100) = faster but less precise
    ///
    /// Most users should stick with the defaults (uniform, 1000 quantiles).
    /// </para>
    /// </remarks>
    public QuantileTransformer(string outputDistribution = "uniform", int nQuantiles = 1000) : base()
    {
        _outputDistribution = outputDistribution.ToLowerInvariant();
        _nQuantiles = nQuantiles;

        if (_outputDistribution != "uniform" && _outputDistribution != "normal")
        {
            throw new ArgumentException("outputDistribution must be either 'uniform' or 'normal'", nameof(outputDistribution));
        }

        if (_nQuantiles < 10)
        {
            throw new ArgumentException("nQuantiles must be at least 10", nameof(nQuantiles));
        }
    }

    /// <summary>
    /// Normalizes output data by transforming to the target distribution.
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    /// <returns>A tuple containing the normalized data and the normalization parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method transforms your data to follow the target distribution
    /// (uniform or normal). It learns the quantiles from your data and uses them to perform the mapping.
    /// </para>
    /// </remarks>
    public override (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data)
    {
        if (data is Vector<T> vector)
        {
            // Compute quantiles from the data
            var sortedData = vector.ToArray();
            Array.Sort(sortedData, (a, b) => NumOps.Compare(a, b));

            var quantiles = new List<T>();
            for (int i = 0; i < _nQuantiles; i++)
            {
                T quantileValue = NumOps.FromDouble((double)i / (_nQuantiles - 1));
                T value = StatisticsHelper<T>.CalculateQuantile(sortedData, quantileValue);
                quantiles.Add(value);
            }

            // Transform the data
            var transformed = new T[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                transformed[i] = TransformValue(vector[i], quantiles);
            }

            var parameters = new NormalizationParameters<T>
            {
                Method = NormalizationMethod.QuantileTransformer,
                Quantiles = quantiles,
                OutputDistribution = _outputDistribution
            };

            return ((TOutput)(object)Vector<T>.FromArray(transformed), parameters);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor to apply quantile transformation
            var flattenedTensor = tensor.ToVector();

            // Compute quantiles from the data
            var sortedData = flattenedTensor.ToArray();
            Array.Sort(sortedData, (a, b) => NumOps.Compare(a, b));

            var quantiles = new List<T>();
            for (int i = 0; i < _nQuantiles; i++)
            {
                T quantileValue = NumOps.FromDouble((double)i / (_nQuantiles - 1));
                T value = StatisticsHelper<T>.CalculateQuantile(sortedData, quantileValue);
                quantiles.Add(value);
            }

            // Transform the data
            var transformed = new T[flattenedTensor.Length];
            for (int i = 0; i < flattenedTensor.Length; i++)
            {
                transformed[i] = TransformValue(flattenedTensor[i], quantiles);
            }

            // Convert back to tensor with the same shape
            var transformedVector = Vector<T>.FromArray(transformed);
            var transformedTensor = Tensor<T>.FromVector(transformedVector);
            if (tensor.Shape.Length > 1)
            {
                transformedTensor = transformedTensor.Reshape(tensor.Shape);
            }

            var parameters = new NormalizationParameters<T>
            {
                Method = NormalizationMethod.QuantileTransformer,
                Quantiles = quantiles,
                OutputDistribution = _outputDistribution
            };

            return ((TOutput)(object)transformedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Normalizes input data by transforming each feature independently to the target distribution.
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    /// <returns>A tuple containing the normalized data and a list of normalization parameters for each feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method transforms multiple features at once.
    /// Each feature (column) is transformed separately to the target distribution based on
    /// its own quantiles.
    /// </para>
    /// </remarks>
    public override (TInput, List<NormalizationParameters<T>>) NormalizeInput(TInput data)
    {
        if (data is Matrix<T> matrix)
        {
            var normalizedColumns = new List<Vector<T>>();
            var parameters = new List<NormalizationParameters<T>>();

            for (int i = 0; i < matrix.Columns; i++)
            {
                var column = matrix.GetColumn(i);

                // Compute quantiles from the column
                var sortedData = column.ToArray();
                Array.Sort(sortedData, (a, b) => NumOps.Compare(a, b));

                var quantiles = new List<T>();
                for (int j = 0; j < _nQuantiles; j++)
                {
                    T quantileValue = NumOps.FromDouble((double)j / (_nQuantiles - 1));
                    T value = StatisticsHelper<T>.CalculateQuantile(sortedData, quantileValue);
                    quantiles.Add(value);
                }

                // Transform the column
                var transformed = new T[column.Length];
                for (int j = 0; j < column.Length; j++)
                {
                    transformed[j] = TransformValue(column[j], quantiles);
                }

                normalizedColumns.Add(Vector<T>.FromArray(transformed));
                parameters.Add(new NormalizationParameters<T>
                {
                    Method = NormalizationMethod.QuantileTransformer,
                    Quantiles = quantiles,
                    OutputDistribution = _outputDistribution
                });
            }

            var normalizedMatrix = Matrix<T>.FromColumnVectors(normalizedColumns);
            return ((TInput)(object)normalizedMatrix, parameters);
        }
        else if (data is Tensor<T> tensor && tensor.Shape.Length == 2)
        {
            // Convert 2D tensor to matrix for column-wise normalization
            var rows = tensor.Shape[0];
            var cols = tensor.Shape[1];
            var newMatrix = new Matrix<T>(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    newMatrix[i, j] = tensor[i, j];
                }
            }

            // Normalize each column separately
            var normalizedColumns = new List<Vector<T>>();
            var parameters = new List<NormalizationParameters<T>>();

            for (int i = 0; i < cols; i++)
            {
                var column = newMatrix.GetColumn(i);

                // Compute quantiles from the column
                var sortedData = column.ToArray();
                Array.Sort(sortedData, (a, b) => NumOps.Compare(a, b));

                var quantiles = new List<T>();
                for (int j = 0; j < _nQuantiles; j++)
                {
                    T quantileValue = NumOps.FromDouble((double)j / (_nQuantiles - 1));
                    T value = StatisticsHelper<T>.CalculateQuantile(sortedData, quantileValue);
                    quantiles.Add(value);
                }

                // Transform the column
                var transformed = new T[column.Length];
                for (int j = 0; j < column.Length; j++)
                {
                    transformed[j] = TransformValue(column[j], quantiles);
                }

                normalizedColumns.Add(Vector<T>.FromArray(transformed));
                parameters.Add(new NormalizationParameters<T>
                {
                    Method = NormalizationMethod.QuantileTransformer,
                    Quantiles = quantiles,
                    OutputDistribution = _outputDistribution
                });
            }

            // Convert back to tensor
            var normalizedMatrix = Matrix<T>.FromColumnVectors(normalizedColumns);
            var normalizedTensor = new Tensor<T>(new[] { normalizedMatrix.Rows, normalizedMatrix.Columns }, normalizedMatrix);

            return ((TInput)(object)normalizedTensor, parameters);
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TInput).Name}. " +
            $"Supported types are Matrix<{typeof(T).Name}> and 2D Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Reverses the quantile transformation.
    /// </summary>
    /// <param name="data">The normalized data to denormalize.</param>
    /// <param name="parameters">The normalization parameters used during normalization.</param>
    /// <returns>The denormalized data in its original scale.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts your transformed values back to their
    /// original distribution by reversing the quantile mapping.
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters)
    {
        if (data is Vector<T> vector)
        {
            var denormalized = new T[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                denormalized[i] = InverseTransformValue(vector[i], parameters.Quantiles, parameters.OutputDistribution);
            }

            return (TOutput)(object)Vector<T>.FromArray(denormalized);
        }
        else if (data is Tensor<T> tensor)
        {
            // Flatten tensor for denormalization
            var flattenedTensor = tensor.ToVector();
            var denormalized = new T[flattenedTensor.Length];
            for (int i = 0; i < flattenedTensor.Length; i++)
            {
                denormalized[i] = InverseTransformValue(flattenedTensor[i], parameters.Quantiles, parameters.OutputDistribution);
            }

            // Convert back to tensor with the same shape
            var denormalizedVector = Vector<T>.FromArray(denormalized);
            var denormalizedTensor = Tensor<T>.FromVector(denormalizedVector);
            if (tensor.Shape.Length > 1)
            {
                denormalizedTensor = denormalizedTensor.Reshape(tensor.Shape);
            }

            return (TOutput)(object)denormalizedTensor;
        }

        throw new InvalidOperationException(
            $"Unsupported data type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Denormalizes model coefficients (not supported for quantile transformation).
    /// </summary>
    /// <param name="coefficients">The model coefficients from a model trained on normalized data.</param>
    /// <param name="xParams">The normalization parameters used for the input features.</param>
    /// <param name="yParams">The normalization parameters used for the target variable.</param>
    /// <returns>Throws NotSupportedException as coefficient denormalization is not meaningful for quantile transformation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Quantile transformation is non-linear, so there's no simple way to
    /// adjust model coefficients. If you need to use the model with original data, you should either:
    /// 1. Keep the transformation and apply it to new data before prediction, or
    /// 2. Use a linear normalization method instead (like MinMax or ZScore).
    /// </para>
    /// </remarks>
    public override TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        throw new NotSupportedException(
            "Coefficient denormalization is not supported for QuantileTransformer because it is a non-linear transformation. " +
            "Apply the inverse transformation to predictions instead.");
    }

    /// <summary>
    /// Denormalizes Y-intercept (not supported for quantile transformation).
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original target vector.</param>
    /// <param name="coefficients">The model coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the target variable.</param>
    /// <returns>Throws NotSupportedException as intercept denormalization is not meaningful for quantile transformation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like coefficients, the Y-intercept can't be simply adjusted for non-linear
    /// transformations. Use the transformation on all your data consistently instead.
    /// </para>
    /// </remarks>
    public override T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        throw new NotSupportedException(
            "Intercept denormalization is not supported for QuantileTransformer because it is a non-linear transformation. " +
            "Apply the inverse transformation to predictions instead.");
    }

    /// <summary>
    /// Transforms a single value using the learned quantiles.
    /// </summary>
    private T TransformValue(T value, List<T> quantiles)
    {
        // Find the position of the value in the quantile distribution
        int lowerIndex = 0;
        int upperIndex = quantiles.Count - 1;

        // Handle values outside the range
        if (NumOps.LessThanOrEqual(value, quantiles[0]))
        {
            return _outputDistribution == "uniform" ? NumOps.Zero : NumOps.FromDouble(-8.0); // Approximate -infinity for normal
        }
        if (NumOps.GreaterThanOrEqual(value, quantiles[quantiles.Count - 1]))
        {
            return _outputDistribution == "uniform" ? NumOps.One : NumOps.FromDouble(8.0); // Approximate infinity for normal
        }

        // Binary search for the position
        int left = 0;
        int right = quantiles.Count - 2; // We want quantiles[i] < value <= quantiles[i+1]
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (NumOps.LessThanOrEqual(value, quantiles[mid + 1]))
            {
                if (NumOps.GreaterThan(value, quantiles[mid]))
                {
                    lowerIndex = mid;
                    upperIndex = mid + 1;
                    break;
                }
                else
                {
                    right = mid - 1;
                }
            }
            else
            {
                left = mid + 1;
            }
        }

        // Linear interpolation with division-by-zero protection
        T lowerValue = quantiles[lowerIndex];
        T upperValue = quantiles[upperIndex];
        T lowerPercentile = NumOps.FromDouble((double)lowerIndex / (quantiles.Count - 1));
        T upperPercentile = NumOps.FromDouble((double)upperIndex / (quantiles.Count - 1));

        T percentile;
        if (NumOps.Equals(upperValue, lowerValue))
        {
            // If quantile values are equal (duplicate values in data), return the midpoint percentile
            percentile = NumOps.Divide(NumOps.Add(lowerPercentile, upperPercentile), NumOps.FromDouble(2.0));
        }
        else
        {
            T fraction = NumOps.Divide(
                NumOps.Subtract(value, lowerValue),
                NumOps.Subtract(upperValue, lowerValue)
            );
            percentile = NumOps.Add(lowerPercentile, NumOps.Multiply(fraction, NumOps.Subtract(upperPercentile, lowerPercentile)));
        }

        // Map to output distribution
        if (_outputDistribution == "uniform")
        {
            return percentile;
        }
        else // normal
        {
            // Convert percentile to normal distribution using inverse CDF approximation
            return InverseNormalCDF(percentile);
        }
    }

    /// <summary>
    /// Inverse transforms a value back to the original distribution.
    /// </summary>
    private T InverseTransformValue(T transformedValue, List<T> quantiles, string outputDistribution)
    {
        T percentile;

        if (outputDistribution == "uniform")
        {
            percentile = transformedValue;
        }
        else // normal
        {
            // Convert from normal distribution back to percentile
            percentile = NormalCDF(transformedValue);
        }

        // Clamp percentile to [0, 1]
        if (NumOps.LessThan(percentile, NumOps.Zero))
            percentile = NumOps.Zero;
        if (NumOps.GreaterThan(percentile, NumOps.One))
            percentile = NumOps.One;

        // Map percentile back to original value using quantiles
        double percentileDouble = NumOps.ToDouble(percentile);
        double position = percentileDouble * (quantiles.Count - 1);
        int lowerIndex = (int)Math.Floor(position);
        int upperIndex = Math.Min(lowerIndex + 1, quantiles.Count - 1);

        if (lowerIndex == upperIndex)
        {
            return quantiles[lowerIndex];
        }

        T fraction = NumOps.FromDouble(position - lowerIndex);
        return NumOps.Add(
            NumOps.Multiply(quantiles[lowerIndex], NumOps.Subtract(NumOps.One, fraction)),
            NumOps.Multiply(quantiles[upperIndex], fraction)
        );
    }

    /// <summary>
    /// Approximates the inverse normal CDF (quantile function) using the Beasley-Springer-Moro algorithm.
    /// </summary>
    private T InverseNormalCDF(T p)
    {
        double pDouble = NumOps.ToDouble(p);

        // Clamp to valid range
        if (pDouble <= 0.0) return NumOps.FromDouble(-8.0);
        if (pDouble >= 1.0) return NumOps.FromDouble(8.0);

        // Beasley-Springer-Moro approximation
        double q = pDouble - 0.5;

        if (Math.Abs(q) <= 0.425)
        {
            double r = 0.180625 - q * q;
            double num = ((((((2.5090809287301226727e3 * r +
                              3.3430575583588128105e4) * r +
                             6.7265770927008700853e4) * r +
                            4.5921953931549871457e4) * r +
                           1.3731693765509461125e4) * r +
                          1.9715909503065514427e3) * r +
                         1.3314166789178437745e2) * r +
                        3.3871328727963666080e0;

            double den = (((((((5.2264952788528545610e3 * r +
                               2.8729085735721942674e4) * r +
                              3.9307895800092710610e4) * r +
                             2.1213794301586595867e4) * r +
                            5.3941960214247511077e3) * r +
                           6.8718700749205790830e2) * r +
                          4.2313330701600911252e1) * r +
                         1.0);

            return NumOps.FromDouble(q * num / den);
        }
        else
        {
            double r = q < 0 ? pDouble : 1.0 - pDouble;
            r = Math.Sqrt(-Math.Log(r));

            double num, den;
            if (r <= 5.0)
            {
                r = r - 1.6;
                num = ((((((7.74545014278341407640e-4 * r +
                           2.27238449892691845833e-2) * r +
                          2.41780725177450611770e-1) * r +
                         1.27045825245236838258e0) * r +
                        3.64784832476320460504e0) * r +
                       5.76949722146069140550e0) * r +
                      4.63033784615654529590e0) * r +
                     1.42343711074968357734e0;

                den = ((((((1.05075007164441684324e-9 * r +
                           5.47593808499534494600e-4) * r +
                          1.51986665636164571966e-2) * r +
                         1.48103976427480074590e-1) * r +
                        6.89767334985100004550e-1) * r +
                       1.67638483018380384940e0) * r +
                      2.05319162663775882187e0) * r +
                     1.0;
            }
            else
            {
                r = r - 5.0;
                num = ((((((2.01033439929228813265e-7 * r +
                           2.71155556874348757815e-5) * r +
                          1.24266094738807843860e-3) * r +
                         2.65321895265761230930e-2) * r +
                        2.96560571828504891230e-1) * r +
                       1.78482653991729133580e0) * r +
                      5.46378491116411436990e0) * r +
                     6.65790464350110377720e0;

                den = ((((((2.04426310338993978564e-15 * r +
                           1.42151175831644588870e-7) * r +
                          1.84631831751005468180e-5) * r +
                         7.86869131145613259100e-4) * r +
                        1.48753612908506148525e-2) * r +
                       1.36929880922735805310e-1) * r +
                      5.99832206555887937690e-1) * r +
                     1.0;
            }

            double result = num / den;
            if (q < 0) result = -result;
            return NumOps.FromDouble(result);
        }
    }

    /// <summary>
    /// Approximates the normal CDF using the error function.
    /// </summary>
    private T NormalCDF(T x)
    {
        double xDouble = NumOps.ToDouble(x);
        // CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
        double result = 0.5 * (1.0 + Erf(xDouble / Math.Sqrt(2.0)));
        return NumOps.FromDouble(result);
    }

    /// <summary>
    /// Approximates the error function using the Abramowitz and Stegun approximation.
    /// </summary>
    private double Erf(double x)
    {
        // Save the sign of x
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        // Constants
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        // Abramowitz and Stegun formula 7.1.26
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }
}
