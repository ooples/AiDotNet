using AiDotNet.Helpers;
using AiDotNet.Tensors;

namespace AiDotNet.Metrics;

/// <summary>
/// Chamfer Distance metric for 3D point cloud comparison.
/// </summary>
/// <remarks>
/// <para>
/// Chamfer Distance measures the average squared distance from each point in one set
/// to its nearest neighbor in the other set, computed bidirectionally.
/// CD(X,Y) = (1/|X|)Σ_x min_y ||x-y||² + (1/|Y|)Σ_y min_x ||y-x||²
/// </para>
/// <para>
/// Lower Chamfer Distance indicates better point cloud similarity.
/// </para>
/// <para><b>Usage in 3D AI:</b>
/// - Point cloud completion evaluation
/// - 3D reconstruction quality
/// - Shape generation evaluation
/// - NeRF/Gaussian Splatting geometry quality
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ChamferDistance<T> where T : struct
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Whether to use squared distances (faster) or Euclidean distances.
    /// </summary>
    private readonly bool _squared;

    /// <summary>
    /// Initializes a new instance of the Chamfer Distance metric.
    /// </summary>
    /// <param name="squared">If true, returns squared distances (default). If false, returns Euclidean.</param>
    public ChamferDistance(bool squared = true)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _squared = squared;
    }

    /// <summary>
    /// Computes Chamfer Distance between two point clouds.
    /// </summary>
    /// <param name="pointsA">First point cloud [N, 3].</param>
    /// <param name="pointsB">Second point cloud [M, 3].</param>
    /// <returns>Chamfer Distance (sum of both directions). Lower is better.</returns>
    public T Compute(Tensor<T> pointsA, Tensor<T> pointsB)
    {
        if (pointsA == null) throw new ArgumentNullException(nameof(pointsA));
        if (pointsB == null) throw new ArgumentNullException(nameof(pointsB));

        if (pointsA.Rank != 2 || pointsB.Rank != 2)
        {
            throw new ArgumentException("Point clouds must be 2D tensors [N, D]");
        }

        if (pointsA.Shape[1] != pointsB.Shape[1])
        {
            throw new ArgumentException("Point clouds must have the same dimension");
        }

        T distAtoB = ComputeOneWay(pointsA, pointsB);
        T distBtoA = ComputeOneWay(pointsB, pointsA);

        return _numOps.Add(distAtoB, distBtoA);
    }

    /// <summary>
    /// Computes one-way Chamfer Distance from source to target.
    /// </summary>
    /// <param name="source">Source point cloud [N, D].</param>
    /// <param name="target">Target point cloud [M, D].</param>
    /// <returns>Mean distance from source points to nearest target points.</returns>
    /// <exception cref="ArgumentException">Thrown when target is empty but source is not.</exception>
    public T ComputeOneWay(Tensor<T> source, Tensor<T> target)
    {
        int numSource = source.Shape[0];
        int numTarget = target.Shape[0];
        int dim = source.Shape[1];

        // Empty source: no points to match, so distance is 0 (vacuously true)
        if (numSource == 0)
        {
            return _numOps.Zero;
        }

        // Empty target with non-empty source: undefined/infinite distance (no points to match to)
        if (numTarget == 0)
        {
            throw new ArgumentException("Cannot compute Chamfer Distance: target point cloud is empty but source has points", nameof(target));
        }

        T totalDist = _numOps.Zero;

        for (int i = 0; i < numSource; i++)
        {
            T minDist = _numOps.FromDouble(double.MaxValue);

            for (int j = 0; j < numTarget; j++)
            {
                T dist = ComputePointDistance(source, target, i, j, dim);
                if (_numOps.Compare(dist, minDist) < 0)
                {
                    minDist = dist;
                }
            }

            totalDist = _numOps.Add(totalDist, minDist);
        }

        return _numOps.Divide(totalDist, _numOps.FromDouble(numSource));
    }

    /// <summary>
    /// Computes distance between two specific points.
    /// </summary>
    private T ComputePointDistance(Tensor<T> a, Tensor<T> b, int idxA, int idxB, int dim)
    {
        T sum = _numOps.Zero;

        for (int d = 0; d < dim; d++)
        {
            T diff = _numOps.Subtract(a[idxA * dim + d], b[idxB * dim + d]);
            sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
        }

        if (_squared)
        {
            return sum;
        }
        else
        {
            return _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(sum)));
        }
    }

    /// <summary>
    /// Computes Chamfer Distance for batched point clouds.
    /// </summary>
    /// <param name="batchA">First batch [B, N, D].</param>
    /// <param name="batchB">Second batch [B, M, D].</param>
    /// <returns>Array of Chamfer Distances, one per batch item.</returns>
    public T[] ComputeBatch(Tensor<T> batchA, Tensor<T> batchB)
    {
        if (batchA == null) throw new ArgumentNullException(nameof(batchA));
        if (batchB == null) throw new ArgumentNullException(nameof(batchB));

        if (batchA.Rank != 3 || batchB.Rank != 3)
        {
            throw new ArgumentException("Batch computation requires 3D tensors [B, N, D]");
        }

        if (batchA.Shape[0] != batchB.Shape[0])
        {
            throw new ArgumentException($"Batch sizes must match: {batchA.Shape[0]} vs {batchB.Shape[0]}");
        }

        if (batchA.Shape[2] != batchB.Shape[2])
        {
            throw new ArgumentException($"Point dimensions must match: {batchA.Shape[2]} vs {batchB.Shape[2]}");
        }

        int batchSize = batchA.Shape[0];
        int numA = batchA.Shape[1];
        int numB = batchB.Shape[1];
        int dim = batchA.Shape[2];

        var results = new T[batchSize];

        for (int batch = 0; batch < batchSize; batch++)
        {
            // Extract single point clouds from batch
            var pointsA = ExtractPointCloud(batchA, batch, numA, dim);
            var pointsB = ExtractPointCloud(batchB, batch, numB, dim);

            results[batch] = Compute(pointsA, pointsB);
        }

        return results;
    }

    /// <summary>
    /// Extracts a single point cloud from a batch.
    /// </summary>
    private Tensor<T> ExtractPointCloud(Tensor<T> batch, int batchIdx, int numPoints, int dim)
    {
        var data = new T[numPoints * dim];
        int offset = batchIdx * numPoints * dim;

        for (int i = 0; i < numPoints * dim; i++)
        {
            data[i] = batch[offset + i];
        }

        return new Tensor<T>(new[] { numPoints, dim }, new Vector<T>(data));
    }
}

/// <summary>
/// Earth Mover's Distance (EMD) / Wasserstein Distance for point cloud comparison.
/// </summary>
/// <remarks>
/// <para>
/// EMD measures the minimum cost to transform one distribution into another,
/// where cost is the sum of distances moved weighted by the amount moved.
/// Uses an approximation based on optimal assignment for efficiency.
/// </para>
/// <para>
/// Lower EMD indicates better point cloud similarity.
/// </para>
/// <para><b>Usage in 3D AI:</b>
/// - Point cloud generation evaluation
/// - 3D shape comparison
/// - More robust than Chamfer Distance for some applications
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EarthMoversDistance<T> where T : struct
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Number of iterations for the Sinkhorn algorithm approximation.
    /// </summary>
    private readonly int _iterations;

    /// <summary>
    /// Regularization parameter for Sinkhorn algorithm.
    /// </summary>
    private readonly double _epsilon;

    /// <summary>
    /// Initializes a new instance of the EMD metric.
    /// </summary>
    /// <param name="iterations">Sinkhorn iterations. Default is 100.</param>
    /// <param name="epsilon">Regularization parameter. Default is 0.01.</param>
    public EarthMoversDistance(int iterations = 100, double epsilon = 0.01)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _iterations = iterations;
        _epsilon = epsilon;
    }

    /// <summary>
    /// Computes approximate EMD between two point clouds using Sinkhorn algorithm.
    /// </summary>
    /// <param name="pointsA">First point cloud [N, D].</param>
    /// <param name="pointsB">Second point cloud [M, D] (should have same N for exact EMD).</param>
    /// <returns>Approximate EMD. Lower is better.</returns>
    public T Compute(Tensor<T> pointsA, Tensor<T> pointsB)
    {
        if (pointsA == null) throw new ArgumentNullException(nameof(pointsA));
        if (pointsB == null) throw new ArgumentNullException(nameof(pointsB));

        if (pointsA.Rank != 2 || pointsB.Rank != 2)
        {
            throw new ArgumentException("Point clouds must be 2D tensors [N, D]");
        }

        if (pointsA.Shape[1] != pointsB.Shape[1])
        {
            throw new ArgumentException($"Point dimensions must match: {pointsA.Shape[1]} vs {pointsB.Shape[1]}");
        }

        int n = pointsA.Shape[0];
        int m = pointsB.Shape[0];
        int dim = pointsA.Shape[1];

        // Handle empty point clouds
        if (n == 0 || m == 0)
        {
            return _numOps.Zero;
        }

        // Compute cost matrix (pairwise distances)
        var costMatrix = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                double dist = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = _numOps.ToDouble(pointsA[i * dim + d]) - _numOps.ToDouble(pointsB[j * dim + d]);
                    dist += diff * diff;
                }
                costMatrix[i, j] = Math.Sqrt(dist);
            }
        }

        // Sinkhorn algorithm for approximate optimal transport
        var K = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                K[i, j] = Math.Exp(-costMatrix[i, j] / _epsilon);
            }
        }

        // Uniform marginals
        var a = new double[n];
        var b = new double[m];
        for (int i = 0; i < n; i++) a[i] = 1.0 / n;
        for (int j = 0; j < m; j++) b[j] = 1.0 / m;

        var u = new double[n];
        var v = new double[m];
        for (int i = 0; i < n; i++) u[i] = 1.0;
        for (int j = 0; j < m; j++) v[j] = 1.0;

        // Sinkhorn iterations
        for (int iter = 0; iter < _iterations; iter++)
        {
            // Update u
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < m; j++)
                {
                    sum += K[i, j] * v[j];
                }
                u[i] = a[i] / Math.Max(sum, 1e-10);
            }

            // Update v
            for (int j = 0; j < m; j++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += K[i, j] * u[i];
                }
                v[j] = b[j] / Math.Max(sum, 1e-10);
            }
        }

        // Compute transport plan and EMD
        double emd = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                double transportAmount = u[i] * K[i, j] * v[j];
                emd += transportAmount * costMatrix[i, j];
            }
        }

        return _numOps.FromDouble(emd);
    }
}

/// <summary>
/// F-Score metric for 3D reconstruction evaluation.
/// </summary>
/// <remarks>
/// <para>
/// F-Score combines precision and recall at a given distance threshold.
/// Precision = fraction of predicted points within threshold of a ground truth point.
/// Recall = fraction of ground truth points within threshold of a predicted point.
/// F-Score = 2 * (Precision * Recall) / (Precision + Recall)
/// </para>
/// <para><b>Usage in 3D AI:</b>
/// - 3D reconstruction quality evaluation
/// - Mesh surface accuracy assessment
/// - Point cloud completion evaluation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FScore<T> where T : struct
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Distance threshold for considering a point as correctly reconstructed.
    /// </summary>
    private readonly double _threshold;

    /// <summary>
    /// Initializes a new instance of the F-Score metric.
    /// </summary>
    /// <param name="threshold">Distance threshold. Default is 0.01 (1% of bounding box).</param>
    public FScore(double threshold = 0.01)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = threshold;
    }

    /// <summary>
    /// Computes F-Score between predicted and ground truth point clouds.
    /// </summary>
    /// <param name="predicted">Predicted point cloud [N, 3].</param>
    /// <param name="groundTruth">Ground truth point cloud [M, 3].</param>
    /// <returns>F-Score value between 0 and 1. Higher is better.</returns>
    public T Compute(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (groundTruth == null) throw new ArgumentNullException(nameof(groundTruth));

        var (precision, recall) = ComputePrecisionRecall(predicted, groundTruth);

        double p = _numOps.ToDouble(precision);
        double r = _numOps.ToDouble(recall);

        if (p + r < 1e-10)
        {
            return _numOps.Zero;
        }

        double fScore = 2.0 * p * r / (p + r);
        return _numOps.FromDouble(fScore);
    }

    /// <summary>
    /// Computes precision and recall separately.
    /// </summary>
    /// <param name="predicted">Predicted point cloud.</param>
    /// <param name="groundTruth">Ground truth point cloud.</param>
    /// <returns>Tuple of (precision, recall).</returns>
    public (T precision, T recall) ComputePrecisionRecall(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        int numPred = predicted.Shape[0];
        int numGT = groundTruth.Shape[0];
        int dim = predicted.Shape[1];

        // Handle empty point clouds
        if (numPred == 0 && numGT == 0)
        {
            return (_numOps.FromDouble(1.0), _numOps.FromDouble(1.0));
        }
        if (numPred == 0)
        {
            return (_numOps.Zero, _numOps.Zero);
        }
        if (numGT == 0)
        {
            return (_numOps.Zero, _numOps.Zero);
        }

        double thresholdSquared = _threshold * _threshold;

        // Precision: fraction of predicted points close to any GT point
        int precisionCount = 0;
        for (int i = 0; i < numPred; i++)
        {
            for (int j = 0; j < numGT; j++)
            {
                double distSquared = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = _numOps.ToDouble(predicted[i * dim + d]) - _numOps.ToDouble(groundTruth[j * dim + d]);
                    distSquared += diff * diff;
                }

                if (distSquared <= thresholdSquared)
                {
                    precisionCount++;
                    break;
                }
            }
        }

        // Recall: fraction of GT points close to any predicted point
        int recallCount = 0;
        for (int j = 0; j < numGT; j++)
        {
            for (int i = 0; i < numPred; i++)
            {
                double distSquared = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = _numOps.ToDouble(groundTruth[j * dim + d]) - _numOps.ToDouble(predicted[i * dim + d]);
                    distSquared += diff * diff;
                }

                if (distSquared <= thresholdSquared)
                {
                    recallCount++;
                    break;
                }
            }
        }

        T precision = _numOps.FromDouble((double)precisionCount / numPred);
        T recall = _numOps.FromDouble((double)recallCount / numGT);

        return (precision, recall);
    }
}

/// <summary>
/// 3D Intersection over Union (3D IoU) for voxel and bounding box evaluation.
/// </summary>
/// <remarks>
/// <para>
/// 3D IoU measures the overlap between two 3D volumes.
/// IoU = Volume(Intersection) / Volume(Union)
/// </para>
/// <para><b>Usage in 3D AI:</b>
/// - Voxel-based 3D detection
/// - 3D bounding box evaluation
/// - Occupancy grid comparison
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class IoU3D<T> where T : struct
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the 3D IoU metric.
    /// </summary>
    public IoU3D()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes 3D IoU between two voxel grids (binary occupancy).
    /// </summary>
    /// <param name="voxelsA">First voxel grid [D, H, W] with binary values.</param>
    /// <param name="voxelsB">Second voxel grid [D, H, W] with binary values.</param>
    /// <returns>IoU value between 0 and 1. Higher is better.</returns>
    public T ComputeVoxelIoU(Tensor<T> voxelsA, Tensor<T> voxelsB)
    {
        if (voxelsA == null) throw new ArgumentNullException(nameof(voxelsA));
        if (voxelsB == null) throw new ArgumentNullException(nameof(voxelsB));

        if (voxelsA.Length != voxelsB.Length)
        {
            throw new ArgumentException($"Voxel grids must have the same size: {voxelsA.Length} vs {voxelsB.Length}");
        }

        long intersection = 0;
        long unionCount = 0;

        for (int i = 0; i < voxelsA.Length; i++)
        {
            bool a = _numOps.ToDouble(voxelsA[i]) > 0.5;
            bool b = _numOps.ToDouble(voxelsB[i]) > 0.5;

            if (a && b) intersection++;
            if (a || b) unionCount++;
        }

        if (unionCount == 0) return _numOps.Zero;

        return _numOps.FromDouble((double)intersection / unionCount);
    }

    /// <summary>
    /// Computes 3D IoU between two axis-aligned bounding boxes.
    /// </summary>
    /// <param name="boxA">First box [x_min, y_min, z_min, x_max, y_max, z_max].</param>
    /// <param name="boxB">Second box [x_min, y_min, z_min, x_max, y_max, z_max].</param>
    /// <returns>IoU value between 0 and 1. Higher is better.</returns>
    public T ComputeBoxIoU(T[] boxA, T[] boxB)
    {
        if (boxA == null) throw new ArgumentNullException(nameof(boxA));
        if (boxB == null) throw new ArgumentNullException(nameof(boxB));

        if (boxA.Length != 6 || boxB.Length != 6)
        {
            throw new ArgumentException("Boxes must have 6 values [x_min, y_min, z_min, x_max, y_max, z_max]");
        }

        // Validate box coordinates (min should be <= max for each dimension)
        double aXMin = _numOps.ToDouble(boxA[0]), aYMin = _numOps.ToDouble(boxA[1]), aZMin = _numOps.ToDouble(boxA[2]);
        double aXMax = _numOps.ToDouble(boxA[3]), aYMax = _numOps.ToDouble(boxA[4]), aZMax = _numOps.ToDouble(boxA[5]);
        double bXMin = _numOps.ToDouble(boxB[0]), bYMin = _numOps.ToDouble(boxB[1]), bZMin = _numOps.ToDouble(boxB[2]);
        double bXMax = _numOps.ToDouble(boxB[3]), bYMax = _numOps.ToDouble(boxB[4]), bZMax = _numOps.ToDouble(boxB[5]);

        if (aXMin > aXMax || aYMin > aYMax || aZMin > aZMax)
        {
            throw new ArgumentException("Box A has invalid coordinates: min values must be <= max values", nameof(boxA));
        }

        if (bXMin > bXMax || bYMin > bYMax || bZMin > bZMax)
        {
            throw new ArgumentException("Box B has invalid coordinates: min values must be <= max values", nameof(boxB));
        }

        // Compute intersection
        double xMin = Math.Max(aXMin, bXMin);
        double yMin = Math.Max(aYMin, bYMin);
        double zMin = Math.Max(aZMin, bZMin);
        double xMax = Math.Min(aXMax, bXMax);
        double yMax = Math.Min(aYMax, bYMax);
        double zMax = Math.Min(aZMax, bZMax);

        double interX = Math.Max(0, xMax - xMin);
        double interY = Math.Max(0, yMax - yMin);
        double interZ = Math.Max(0, zMax - zMin);
        double intersection = interX * interY * interZ;

        // Compute union
        double volA = (aXMax - aXMin) * (aYMax - aYMin) * (aZMax - aZMin);
        double volB = (bXMax - bXMin) * (bYMax - bYMin) * (bZMax - bZMin);

        double union = volA + volB - intersection;

        if (union <= 0) return _numOps.Zero;

        return _numOps.FromDouble(intersection / union);
    }
}
