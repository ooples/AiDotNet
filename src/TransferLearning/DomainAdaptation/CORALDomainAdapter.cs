using AiDotNet.Helpers;

namespace AiDotNet.TransferLearning.DomainAdaptation;

/// <summary>
/// Implements domain adaptation using CORrelation ALignment (CORAL).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> CORAL (CORrelation ALignment) aligns the second-order statistics
/// (covariances) of source and target domains. Think of it as making sure the "spread" and
/// "correlation patterns" of features are similar in both domains.
/// </para>
/// <para>
/// Imagine you have two datasets: one where features vary a lot, and another where they
/// vary less. CORAL adjusts the data so both have similar variability patterns, making
/// transfer learning more effective.
/// </para>
/// </remarks>
public class CORALDomainAdapter<T> : IDomainAdapter<T>
{
    private readonly INumericOperations<T> _numOps;
    private Matrix<T>? _transformationMatrix;

    /// <summary>
    /// Gets the name of the adaptation method.
    /// </summary>
    public string AdaptationMethod => "CORAL (CORrelation ALignment)";

    /// <summary>
    /// Gets whether this adapter requires training.
    /// </summary>
    public bool RequiresTraining => true;

    /// <summary>
    /// Initializes a new instance of the CORALDomainAdapter class.
    /// </summary>
    public CORALDomainAdapter()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Trains the CORAL adapter by computing the transformation matrix.
    /// </summary>
    /// <param name="sourceData">Training data from the source domain.</param>
    /// <param name="targetData">Training data from the target domain.</param>
    public void Train(Matrix<T> sourceData, Matrix<T> targetData)
    {
        // Compute covariance matrices
        var sourceCov = ComputeCovariance(sourceData);
        var targetCov = ComputeCovariance(targetData);

        // Compute the CORAL transformation: C_s^{-1/2} * C_t^{1/2}
        _transformationMatrix = ComputeCORALTransformation(sourceCov, targetCov);
    }

    /// <summary>
    /// Adapts source data to match target distribution using CORAL.
    /// </summary>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <param name="targetData">Data from the target domain.</param>
    /// <returns>Adapted source data.</returns>
    public Matrix<T> AdaptSource(Matrix<T> sourceData, Matrix<T> targetData)
    {
        if (_transformationMatrix == null)
        {
            Train(sourceData, targetData);
        }

        // Center the source data
        var sourceMean = ComputeMean(sourceData);
        var centeredSource = CenterData(sourceData, sourceMean);

        // Apply CORAL transformation
        var adapted = centeredSource.Multiply(_transformationMatrix!);

        // Add back the target mean
        var targetMean = ComputeMean(targetData);
        return DecenterData(adapted, targetMean);
    }

    /// <summary>
    /// Adapts target data to match source distribution.
    /// </summary>
    /// <param name="targetData">Data from the target domain.</param>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <returns>Adapted target data.</returns>
    public Matrix<T> AdaptTarget(Matrix<T> targetData, Matrix<T> sourceData)
    {
        // Compute inverse transformation
        var targetCov = ComputeCovariance(targetData);
        var sourceCov = ComputeCovariance(sourceData);
        var inverseTransform = ComputeCORALTransformation(targetCov, sourceCov);

        // Center the target data
        var targetMean = ComputeMean(targetData);
        var centeredTarget = CenterData(targetData, targetMean);

        // Apply inverse transformation
        var adapted = centeredTarget.Multiply(inverseTransform);

        // Add back the source mean
        var sourceMean = ComputeMean(sourceData);
        return DecenterData(adapted, sourceMean);
    }

    /// <summary>
    /// Computes the domain discrepancy using Frobenius norm of covariance difference.
    /// </summary>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <param name="targetData">Data from the target domain.</param>
    /// <returns>The discrepancy measure.</returns>
    public T ComputeDomainDiscrepancy(Matrix<T> sourceData, Matrix<T> targetData)
    {
        var sourceCov = ComputeCovariance(sourceData);
        var targetCov = ComputeCovariance(targetData);

        // Compute Frobenius norm of the difference
        return ComputeFrobeniusNorm(MatrixSubtract(sourceCov, targetCov));
    }

    /// <summary>
    /// Computes the mean of each feature column.
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
    /// Centers the data by subtracting the mean.
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
    /// Decenters the data by adding the mean.
    /// </summary>
    private Matrix<T> DecenterData(Matrix<T> data, Vector<T> mean)
    {
        var decentered = new Matrix<T>(data.Rows, data.Columns);
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                decentered[i, j] = _numOps.Add(data[i, j], mean[j]);
            }
        }
        return decentered;
    }

    /// <summary>
    /// Computes the covariance matrix of the data.
    /// </summary>
    private Matrix<T> ComputeCovariance(Matrix<T> data)
    {
        // Center the data
        var mean = ComputeMean(data);
        var centered = CenterData(data, mean);

        // Compute covariance: (1/n) * X^T * X
        int n = centered.Rows;
        int d = centered.Columns;
        var cov = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T sum = _numOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(centered[k, i], centered[k, j]));
                }
                cov[i, j] = _numOps.Divide(sum, _numOps.FromDouble(n));
            }
        }

        // Add regularization to ensure invertibility
        var regularization = _numOps.FromDouble(1e-5);
        for (int i = 0; i < d; i++)
        {
            cov[i, i] = _numOps.Add(cov[i, i], regularization);
        }

        return cov;
    }

    /// <summary>
    /// Computes the CORAL transformation matrix.
    /// </summary>
    private Matrix<T> ComputeCORALTransformation(Matrix<T> sourceCov, Matrix<T> targetCov)
    {
        // Compute C_s^{-1/2} and C_t^{1/2}
        // For simplicity, we use a diagonal approximation
        var sourceInvSqrt = ComputeMatrixInverseSqrt(sourceCov);
        var targetSqrt = ComputeMatrixSqrt(targetCov);

        // Multiply to get the transformation
        return sourceInvSqrt.Multiply(targetSqrt);
    }

    /// <summary>
    /// Computes the matrix square root using eigendecomposition approximation.
    /// </summary>
    private Matrix<T> ComputeMatrixSqrt(Matrix<T> matrix)
    {
        // Simplified implementation: use diagonal approximation
        int d = matrix.Rows;
        var result = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            // Approximate sqrt by taking square root of diagonal elements
            T maxValue = MathHelper.Max(matrix[i, i], _numOps.FromDouble(1e-10));
            result[i, i] = _numOps.Sqrt(maxValue);
        }

        return result;
    }

    /// <summary>
    /// Computes the inverse square root of a matrix.
    /// </summary>
    private Matrix<T> ComputeMatrixInverseSqrt(Matrix<T> matrix)
    {
        // Simplified implementation: use diagonal approximation
        int d = matrix.Rows;
        var result = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            // Compute 1/sqrt(diagonal element)
            T maxValue = MathHelper.Max(matrix[i, i], _numOps.FromDouble(1e-10));
            T sqrtDiag = _numOps.Sqrt(maxValue);
            result[i, i] = _numOps.Divide(_numOps.One, sqrtDiag);
        }

        return result;
    }

    /// <summary>
    /// Subtracts two matrices.
    /// </summary>
    private Matrix<T> MatrixSubtract(Matrix<T> a, Matrix<T> b)
    {
        var result = new Matrix<T>(a.Rows, a.Columns);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                result[i, j] = _numOps.Subtract(a[i, j], b[i, j]);
            }
        }
        return result;
    }

    /// <summary>
    /// Computes the Frobenius norm of a matrix.
    /// </summary>
    private T ComputeFrobeniusNorm(Matrix<T> matrix)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                T val = matrix[i, j];
                sum = _numOps.Add(sum, _numOps.Multiply(val, val));
            }
        }
        return _numOps.Sqrt(sum);
    }
}
