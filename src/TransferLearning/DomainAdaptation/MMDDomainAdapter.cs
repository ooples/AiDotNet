using AiDotNet.Interfaces;
using AiDotNet.Kernels;

namespace AiDotNet.TransferLearning.DomainAdaptation;

/// <summary>
/// Implements domain adaptation using Maximum Mean Discrepancy (MMD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Maximum Mean Discrepancy (MMD) is a way to measure how different two
/// distributions are. Think of it like comparing the "average characteristics" of two groups.
/// This adapter minimizes the difference between the average properties of source and target data.
/// </para>
/// <para>
/// Imagine you have photos from two different cameras. MMD would measure how different the
/// "average photo" from each camera is, and then adjust them to have similar average properties.
/// </para>
/// </remarks>
public class MMDDomainAdapter<T> : IDomainAdapter<T>
{
    private readonly INumericOperations<T> _numOps;
    private IKernelFunction<T> _kernel;
    private double _sigma; // Bandwidth parameter for Gaussian kernel

    /// <summary>
    /// Gets the name of the adaptation method.
    /// </summary>
    public string AdaptationMethod => "Maximum Mean Discrepancy (MMD)";

    /// <summary>
    /// Gets whether this adapter requires training.
    /// </summary>
    public bool RequiresTraining => false; // MMD is non-parametric

    /// <summary>
    /// Initializes a new instance of the MMDDomainAdapter class.
    /// </summary>
    /// <param name="sigma">Bandwidth parameter for the Gaussian kernel. If not specified, uses median heuristic.</param>
    public MMDDomainAdapter(double sigma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma;

        // Use Gaussian kernel for MMD
        _kernel = new GaussianKernel<T>(_sigma);
    }

    /// <summary>
    /// Trains the adapter (no-op for MMD as it's non-parametric).
    /// </summary>
    public void Train(Matrix<T> sourceData, Matrix<T> targetData)
    {
        // MMD is non-parametric and doesn't require training
        // Optionally, we could compute the optimal sigma using median heuristic here
        _sigma = ComputeMedianHeuristic(sourceData, targetData);
        _kernel = new GaussianKernel<T>(_sigma);
    }

    /// <summary>
    /// Adapts source data to match target distribution using MMD.
    /// </summary>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <param name="targetData">Data from the target domain.</param>
    /// <returns>Adapted source data.</returns>
    public Matrix<T> AdaptSource(Matrix<T> sourceData, Matrix<T> targetData)
    {
        // Compute the mean embeddings in kernel space
        var sourceMeanEmbedding = ComputeMeanEmbedding(sourceData);
        var targetMeanEmbedding = ComputeMeanEmbedding(targetData);

        // Compute the shift needed in input space (simplified approach)
        var shift = ComputeDistributionShift(sourceData, targetData);

        // Apply the shift to source data
        return ApplyShift(sourceData, shift);
    }

    /// <summary>
    /// Adapts target data to match source distribution.
    /// </summary>
    /// <param name="targetData">Data from the target domain.</param>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <returns>Adapted target data.</returns>
    public Matrix<T> AdaptTarget(Matrix<T> targetData, Matrix<T> sourceData)
    {
        // Compute the shift in the opposite direction
        var shift = ComputeDistributionShift(targetData, sourceData);
        return ApplyShift(targetData, shift);
    }

    /// <summary>
    /// Computes the Maximum Mean Discrepancy between two domains.
    /// </summary>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <param name="targetData">Data from the target domain.</param>
    /// <returns>The MMD value (non-negative, lower is better).</returns>
    public T ComputeDomainDiscrepancy(Matrix<T> sourceData, Matrix<T> targetData)
    {
        int m = sourceData.Rows;
        int n = targetData.Rows;

        // Compute kernel matrices
        T sourceSourceSum = ComputeKernelSum(sourceData, sourceData);
        T targetTargetSum = ComputeKernelSum(targetData, targetData);
        T sourceTargetSum = ComputeKernelSum(sourceData, targetData);

        // MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        T mmd2 = _numOps.Divide(sourceSourceSum, _numOps.FromDouble(m * m));
        mmd2 = _numOps.Add(mmd2, _numOps.Divide(targetTargetSum, _numOps.FromDouble(n * n)));
        mmd2 = _numOps.Subtract(mmd2, _numOps.Multiply(_numOps.FromDouble(2.0),
            _numOps.Divide(sourceTargetSum, _numOps.FromDouble(m * n))));

        // Return sqrt(MMD^2)
        T maxValue = MathHelper.Max(mmd2, _numOps.Zero);
        return _numOps.Sqrt(maxValue); // Ensure non-negative due to numerical errors
    }

    /// <summary>
    /// Computes the mean embedding of data in kernel space.
    /// </summary>
    private Vector<T> ComputeMeanEmbedding(Matrix<T> data)
    {
        // For simplicity, compute mean in input space
        // In full implementation, this would be in kernel space
        var mean = new Vector<T>(data.Columns);
        for (int j = 0; j < data.Columns; j++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < data.Rows; i++)
            {
                sum = _numOps.Add(sum, data[i, j]);
            }
            mean[j] = _numOps.Divide(sum, _numOps.FromDouble(data.Rows));
        }
        return mean;
    }

    /// <summary>
    /// Computes the distribution shift between two domains.
    /// </summary>
    private Vector<T> ComputeDistributionShift(Matrix<T> fromData, Matrix<T> toData)
    {
        var fromMean = ComputeMeanEmbedding(fromData);
        var toMean = ComputeMeanEmbedding(toData);

        // Compute the difference in means
        var shift = new Vector<T>(fromMean.Length);
        for (int i = 0; i < shift.Length; i++)
        {
            shift[i] = _numOps.Subtract(toMean[i], fromMean[i]);
        }

        return shift;
    }

    /// <summary>
    /// Applies a shift to data.
    /// </summary>
    private Matrix<T> ApplyShift(Matrix<T> data, Vector<T> shift)
    {
        var result = new Matrix<T>(data.Rows, data.Columns);
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                result[i, j] = _numOps.Add(data[i, j], shift[j]);
            }
        }
        return result;
    }

    /// <summary>
    /// Computes the sum of kernel evaluations between two datasets.
    /// </summary>
    private T ComputeKernelSum(Matrix<T> data1, Matrix<T> data2)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < data1.Rows; i++)
        {
            for (int j = 0; j < data2.Rows; j++)
            {
                var x1 = data1.GetRow(i);
                var x2 = data2.GetRow(j);
                sum = _numOps.Add(sum, _kernel.Calculate(x1, x2));
            }
        }
        return sum;
    }

    /// <summary>
    /// Computes the median heuristic for kernel bandwidth selection.
    /// </summary>
    private double ComputeMedianHeuristic(Matrix<T> data1, Matrix<T> data2)
    {
        // Sample a subset of pairwise distances
        int sampleSize = Math.Min(100, Math.Min(data1.Rows, data2.Rows));
        var distances = new List<T>();
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < sampleSize; i++)
        {
            int idx1 = random.Next(data1.Rows);
            int idx2 = random.Next(data2.Rows);

            var x1 = data1.GetRow(idx1);
            var x2 = data2.GetRow(idx2);

            T distance = ComputeEuclideanDistance(x1, x2);
            distances.Add(distance);
        }

        // Sort and find median
        distances.Sort((a, b) => Convert.ToDouble(a).CompareTo(Convert.ToDouble(b)));
        int medianIndex = distances.Count / 2;
        double median = Convert.ToDouble(distances[medianIndex]);

        // Sigma = median / sqrt(2)
        return median / Math.Sqrt(2.0);
    }

    /// <summary>
    /// Computes the Euclidean distance between two vectors.
    /// </summary>
    private T ComputeEuclideanDistance(Vector<T> x1, Vector<T> x2)
    {
        T sumSquares = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = _numOps.Subtract(x1[i], x2[i]);
            sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(diff, diff));
        }
        return _numOps.Sqrt(sumSquares);
    }
}
