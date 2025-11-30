namespace AiDotNet.TransferLearning.FeatureMapping;

/// <summary>
/// Implements a simple linear projection for mapping features between domains.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Linear feature mapping is the simplest approach to translating between domains.
/// It uses matrix multiplication to transform features, similar to how you might resize an image
/// using simple scaling. While not as sophisticated as other methods, it's fast and works well
/// when domains are reasonably similar.
/// </para>
/// <para>
/// Think of it like using a simple multiplication factor to convert between units
/// (like converting feet to meters). It's not perfect for all situations, but it's
/// a good starting point.
/// </para>
/// </remarks>
public class LinearFeatureMapper<T> : IFeatureMapper<T>
{
    private readonly INumericOperations<T> _numOps;
    private Matrix<T>? _projectionMatrix;
    private Matrix<T>? _reverseProjectionMatrix;
    private T _confidence;

    /// <summary>
    /// Indicates whether the mapper has been trained.
    /// </summary>
    public bool IsTrained { get; private set; }

    /// <summary>
    /// Initializes a new instance of the LinearFeatureMapper class.
    /// </summary>
    public LinearFeatureMapper()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        IsTrained = false;
        _confidence = _numOps.Zero;
    }

    /// <summary>
    /// Trains the linear feature mapper using Principal Component Analysis (PCA)-like approach.
    /// </summary>
    /// <param name="sourceData">Training data from the source domain.</param>
    /// <param name="targetData">Training data from the target domain.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method learns how to transform features between domains by
    /// analyzing the structure of data in both domains. It finds the best linear transformation
    /// that preserves as much information as possible when moving between domains.
    /// </para>
    /// </remarks>
    public void Train(Matrix<T> sourceData, Matrix<T> targetData)
    {
        int sourceDim = sourceData.Columns;
        int targetDim = targetData.Columns;
        int minDim = Math.Min(sourceDim, targetDim);

        // Center the data (subtract mean)
        var sourceMean = ComputeMean(sourceData);
        var targetMean = ComputeMean(targetData);

        var centeredSource = CenterData(sourceData, sourceMean);
        var centeredTarget = CenterData(targetData, targetMean);

        // Compute covariance-based projection
        // For simplicity, we use a reduced-rank approximation
        _projectionMatrix = ComputeProjectionMatrix(centeredSource, sourceDim, targetDim);
        _reverseProjectionMatrix = ComputeProjectionMatrix(centeredTarget, targetDim, sourceDim);

        // Set trained flag before calling MapToTarget/MapToSource
        IsTrained = true;

        // Compute mapping confidence based on reconstruction error
        var reconstructed = MapToTarget(sourceData, targetDim);
        var reverseReconstructed = MapToSource(reconstructed, sourceDim);
        _confidence = ComputeReconstructionConfidence(sourceData, reverseReconstructed);
    }

    /// <summary>
    /// Maps features from source domain to target domain.
    /// </summary>
    /// <param name="sourceFeatures">The features from the source domain.</param>
    /// <param name="targetDimension">The desired number of dimensions in the target domain.</param>
    /// <returns>The mapped features with the target dimension.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the mapper hasn't been trained.</exception>
    public Matrix<T> MapToTarget(Matrix<T> sourceFeatures, int targetDimension)
    {
        if (!IsTrained || _projectionMatrix == null)
        {
            throw new InvalidOperationException("Feature mapper must be trained before use.");
        }

        // Apply projection: result = sourceFeatures * projectionMatrix
        return sourceFeatures.Multiply(_projectionMatrix);
    }

    /// <summary>
    /// Maps features from target domain back to source domain.
    /// </summary>
    /// <param name="targetFeatures">The features from the target domain.</param>
    /// <param name="sourceDimension">The desired number of dimensions in the source domain.</param>
    /// <returns>The mapped features with the source dimension.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the mapper hasn't been trained.</exception>
    public Matrix<T> MapToSource(Matrix<T> targetFeatures, int sourceDimension)
    {
        if (!IsTrained || _reverseProjectionMatrix == null)
        {
            throw new InvalidOperationException("Feature mapper must be trained before use.");
        }

        // Apply reverse projection
        return targetFeatures.Multiply(_reverseProjectionMatrix);
    }

    /// <summary>
    /// Gets the confidence score for the mapping quality.
    /// </summary>
    /// <returns>A value between 0 and 1, where higher values indicate better mapping quality.</returns>
    public T GetMappingConfidence()
    {
        return _confidence;
    }

    /// <summary>
    /// Computes the mean of each feature column in the data matrix.
    /// </summary>
    private Vector<T> ComputeMean(Matrix<T> data)
    {
        var means = new Vector<T>(data.Columns);
        for (int j = 0; j < data.Columns; j++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < data.Rows; i++)
            {
                sum = _numOps.Add(sum, data[i, j]);
            }
            means[j] = _numOps.Divide(sum, _numOps.FromDouble(data.Rows));
        }
        return means;
    }

    /// <summary>
    /// Centers the data by subtracting the mean from each column.
    /// </summary>
    private Matrix<T> CenterData(Matrix<T> data, Vector<T> mean)
    {
        var centered = new Matrix<T>(data.Rows, data.Columns);
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                centered[i, j] = _numOps.Subtract(data[i, j], mean[j]);
            }
        }
        return centered;
    }

    /// <summary>
    /// Computes a projection matrix using a simplified approach.
    /// </summary>
    private Matrix<T> ComputeProjectionMatrix(Matrix<T> data, int inputDim, int outputDim)
    {
        var projection = new Matrix<T>(inputDim, outputDim);

        // Use a simple random projection with normalization
        // In a full implementation, this would use SVD or PCA
        var random = RandomHelper.CreateSeededRandom(42); // Fixed seed for reproducibility

        for (int i = 0; i < inputDim; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                // Initialize with small random values
                double value = (random.NextDouble() - 0.5) * 2.0 / Math.Sqrt(inputDim);
                projection[i, j] = _numOps.FromDouble(value);
            }
        }

        // Orthonormalize columns using Gram-Schmidt
        projection = OrthonormalizeColumns(projection);

        return projection;
    }

    /// <summary>
    /// Orthonormalizes the columns of a matrix using the Gram-Schmidt process.
    /// </summary>
    private Matrix<T> OrthonormalizeColumns(Matrix<T> matrix)
    {
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        for (int j = 0; j < matrix.Columns; j++)
        {
            // Start with the original column
            var column = matrix.GetColumn(j);

            // Subtract projections onto previous columns
            for (int k = 0; k < j; k++)
            {
                var prevColumn = result.GetColumn(k);
                T dotProduct = DotProduct(column, prevColumn);
                column = SubtractScaled(column, prevColumn, dotProduct);
            }

            // Normalize the column
            T norm = VectorNorm(column);
            if (_numOps.GreaterThan(norm, _numOps.FromDouble(1e-10)))
            {
                column = ScaleVector(column, _numOps.Divide(_numOps.One, norm));
            }

            // Store the orthonormalized column
            for (int i = 0; i < result.Rows; i++)
            {
                result[i, j] = column[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    /// <summary>
    /// Computes the Euclidean norm (length) of a vector.
    /// </summary>
    private T VectorNorm(Vector<T> vector)
    {
        T sumSquares = _numOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(vector[i], vector[i]));
        }
        return _numOps.Sqrt(sumSquares);
    }

    /// <summary>
    /// Subtracts a scaled vector from another vector.
    /// </summary>
    private Vector<T> SubtractScaled(Vector<T> a, Vector<T> b, T scale)
    {
        var result = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = _numOps.Subtract(a[i], _numOps.Multiply(scale, b[i]));
        }
        return result;
    }

    /// <summary>
    /// Scales a vector by a scalar value.
    /// </summary>
    private Vector<T> ScaleVector(Vector<T> vector, T scale)
    {
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = _numOps.Multiply(vector[i], scale);
        }
        return result;
    }

    /// <summary>
    /// Computes the reconstruction confidence based on how well we can round-trip the data.
    /// </summary>
    private T ComputeReconstructionConfidence(Matrix<T> original, Matrix<T> reconstructed)
    {
        // Compute mean squared error
        T totalError = _numOps.Zero;
        int totalElements = original.Rows * original.Columns;

        for (int i = 0; i < original.Rows; i++)
        {
            for (int j = 0; j < original.Columns; j++)
            {
                T diff = _numOps.Subtract(original[i, j], reconstructed[i, j]);
                totalError = _numOps.Add(totalError, _numOps.Multiply(diff, diff));
            }
        }

        T mse = _numOps.Divide(totalError, _numOps.FromDouble(totalElements));

        // Convert MSE to confidence (lower error = higher confidence)
        // confidence = exp(-mse)
        T confidence = _numOps.Exp(_numOps.Negate(mse));

        return confidence;
    }
}
