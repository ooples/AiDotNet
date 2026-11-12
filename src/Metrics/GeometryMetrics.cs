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

        // Mean over source points of the distance to the nearest target point.
        var engine = AiDotNetEngine.Current;
        var nearest = PointCloudDistances.NearestSquared(source, target);
        return engine.TensorMean(_squared ? nearest : engine.TensorSqrt(nearest));
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

        if (batchSize == 0 || (numA == 0 && numB == 0))
        {
            return new T[batchSize];
        }

        if (numA == 0 || numB == 0)
        {
            // One side empty: the other direction has no target, exactly as Compute reports it.
            throw new ArgumentException("Cannot compute Chamfer Distance: target point cloud is empty but source has points", "target");
        }

        // Process batch items independently; each direction reduces bounded point-pair blocks.
        var results = new T[batchSize];
        var engine = AiDotNetEngine.Current;
        int dimensions = batchA.Shape[2];
        for (int item = 0; item < batchSize; item++)
        {
            var a = engine.TensorSlice(batchA, new[] { item, 0, 0 }, new[] { 1, numA, dimensions })
                .Reshape(new[] { numA, dimensions });
            var b = engine.TensorSlice(batchB, new[] { item, 0, 0 }, new[] { 1, numB, dimensions })
                .Reshape(new[] { numB, dimensions });
            results[item] = Compute(a, b);
        }
        return results;
    }
}

/// <summary>
/// Pairwise squared Euclidean distances between point clouds.
/// </summary>
internal static class PointCloudDistances
{
    // Bound coordinate intermediates independently of the number of points and batch items.
    private const int CoordinateBlockElements = 1 << 18;

    /// <summary>Nearest squared distance for each source point, using bounded point-pair blocks.</summary>
    internal static Tensor<T> NearestSquared<T>(Tensor<T> source, Tensor<T> target) where T : struct
        => ReduceRows(source, target, (T?)null);

    /// <summary>Whether each source point has a neighbour within the squared threshold, using bounded blocks.</summary>
    internal static Tensor<T> WithinThreshold<T>(Tensor<T> source, Tensor<T> target, T threshold) where T : struct
        => ReduceRows(source, target, (T?)threshold);

    private static Tensor<T> ReduceRows<T>(Tensor<T> source, Tensor<T> target, T? threshold) where T : struct
    {
        var engine = AiDotNetEngine.Current;
        int count = source.Shape[0], targets = target.Shape[0], dimensions = source.Shape[1];
        int coordinateWidth = Math.Max(1, dimensions);
        int rowBlock = Math.Min(256, Math.Max(1, CoordinateBlockElements / coordinateWidth));
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < count; start += rowBlock)
        {
            int rows = Math.Min(rowBlock, count - start);
            var a = engine.TensorSlice(source, new[] { start, 0 }, new[] { rows, dimensions });
            int targetBlock = Math.Max(1, CoordinateBlockElements / (rows * coordinateWidth));
            Tensor<T>? reduced = null;
            for (int offset = 0; offset < targets; offset += targetBlock)
            {
                int columns = Math.Min(targetBlock, targets - offset);
                var b = engine.TensorSlice(target, new[] { offset, 0 }, new[] { columns, dimensions });
                var squared = dimensions == 0 ? new Tensor<T>(new[] { rows, columns }) : Squared(a, b);
                Tensor<T> partial;
                if (threshold.HasValue)
                {
                    // NaN distances never match. Complementing a greater-than comparison would match them.
                    var matches = engine.TensorLessEqual(squared, engine.TensorFullLike(squared, threshold.Value));
                    partial = engine.ReduceMax(matches, new[] { 1 }, false);
                    reduced = reduced is null ? partial : engine.TensorMax(reduced, partial);
                }
                else
                {
                    partial = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(squared), new[] { 1 }, false));
                    reduced = reduced is null ? partial : engine.TensorMin(reduced, partial);
                }
            }
            blocks.Add(reduced ?? throw new ArgumentException("Target point cloud must not be empty.", nameof(target)));
        }
        return blocks.Count == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(blocks.ToArray(), 0);
    }


    /// <summary>
    /// Squared distances between every pair of points: [N, D] x [M, D] -> [N, M], or batched
    /// [B, N, D] x [B, M, D] -> [B, N, M]. Each entry is sum_d (a_d - b_d)^2 of the coordinate differences.
    /// </summary>
    internal static Tensor<T> Squared<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        if (a.Rank == 2)
        {
            int n = a.Shape[0], m = b.Shape[0], d = a.Shape[1];
            var pairShape = new[] { n, m, d };
            var diff = engine.TensorSubtract(
                engine.TensorBroadcastTo(a.Reshape(new[] { n, 1, d }), pairShape),
                engine.TensorBroadcastTo(b.Reshape(new[] { 1, m, d }), pairShape));
            return engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 2 }, false);
        }

        int batch = a.Shape[0], na = a.Shape[1], nb = b.Shape[1], dim = a.Shape[2];
        var batchPairShape = new[] { batch, na, nb, dim };
        var batchDiff = engine.TensorSubtract(
            engine.TensorBroadcastTo(a.Reshape(new[] { batch, na, 1, dim }), batchPairShape),
            engine.TensorBroadcastTo(b.Reshape(new[] { batch, 1, nb, dim }), batchPairShape));
        return engine.ReduceSum(engine.TensorMultiply(batchDiff, batchDiff), new[] { 3 }, false);
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
    /// <remarks>
    /// Entropic optimal transport (Cuturi 2013) with uniform marginals and Euclidean ground cost, iterated in the
    /// log domain (POT's sinkhorn_log): log u = log a - LSE_j(log K_ij + log v_j), log v = log b -
    /// LSE_i(log K_ij + log u_i), with log K = -C / epsilon. This is the same fixed point as the scaling form,
    /// without exp(-C / epsilon) underflowing to zero when distances are large relative to epsilon.
    /// </remarks>
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

        // Handle empty point clouds
        if (n == 0 || m == 0)
        {
            return _numOps.Zero;
        }

        var engine = AiDotNetEngine.Current;

        // Cost matrix (pairwise Euclidean distances) and log of the Gibbs kernel.
        var cost = engine.TensorSqrt(PointCloudDistances.Squared(pointsA, pointsB));
        var logKernel = engine.TensorMultiplyScalar(cost, _numOps.FromDouble(-1.0 / _epsilon));

        // Uniform marginals; u = v = 1 to start.
        T logA = _numOps.FromDouble(-Math.Log(n));
        T logB = _numOps.FromDouble(-Math.Log(m));
        var logU = new Tensor<T>(new[] { n, 1 });
        var logV = new Tensor<T>(new[] { 1, m });
        var planShape = new[] { n, m };

        for (int iter = 0; iter < _iterations; iter++)
        {
            var rowLse = engine.TensorLogSumExp(
                engine.TensorAdd(logKernel, engine.TensorBroadcastTo(logV, planShape)), 1, true);
            logU = engine.TensorAddScalar(engine.TensorNegate(rowLse), logA);

            var colLse = engine.TensorLogSumExp(
                engine.TensorAdd(logKernel, engine.TensorBroadcastTo(logU, planShape)), 0, true);
            logV = engine.TensorAddScalar(engine.TensorNegate(colLse), logB);
        }

        // Transport plan P_ij = u_i K_ij v_j and EMD = sum_ij P_ij C_ij.
        var logPlan = engine.TensorAdd(
            engine.TensorAdd(logKernel, engine.TensorBroadcastTo(logU, planShape)),
            engine.TensorBroadcastTo(logV, planShape));
        return engine.TensorSum(engine.TensorMultiply(engine.TensorExp(logPlan), cost));
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

        T sum = _numOps.Add(precision, recall);
        if (_numOps.LessThan(sum, _numOps.FromDouble(1e-10)))
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(precision, recall)), sum);
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

        var engine = AiDotNetEngine.Current;

        // Count points with any matching neighbour; integer counts avoid low-precision reduction overflow.
        T thresholdSquared = _numOps.FromDouble(_threshold * _threshold);
        var predictedMatches = PointCloudDistances.WithinThreshold(predicted, groundTruth, thresholdSquared);
        var targetMatches = PointCloudDistances.WithinThreshold(groundTruth, predicted, thresholdSquared);
        T precision = _numOps.FromDouble((double)engine.TensorCountNonzero(predictedMatches) / numPred);
        T recall = _numOps.FromDouble((double)engine.TensorCountNonzero(targetMatches) / numGT);

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

        // A voxel is occupied when its value exceeds 0.5.
        var engine = AiDotNetEngine.Current;
        int count = voxelsA.Length;
        T half = _numOps.FromDouble(0.5);
        var a = engine.TensorGreaterThan(voxelsA.Reshape(new[] { count }), half);
        var b = engine.TensorGreaterThan(voxelsB.Reshape(new[] { count }), half);

        // Count as integers: summing the 0/1 masks in T saturates Half at 65504 and drops increments of 1 above
        // 2048 (Half) or 256 (BFloat16).
        long intersection = engine.TensorCountNonzero(engine.TensorMultiply(a, b));
        long unionCount = engine.TensorCountNonzero(engine.TensorMax(a, b));

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
        var engine = AiDotNetEngine.Current;
        var minA = new Tensor<T>(new[] { boxA[0], boxA[1], boxA[2] }, new[] { 3 });
        var maxA = new Tensor<T>(new[] { boxA[3], boxA[4], boxA[5] }, new[] { 3 });
        var minB = new Tensor<T>(new[] { boxB[0], boxB[1], boxB[2] }, new[] { 3 });
        var maxB = new Tensor<T>(new[] { boxB[3], boxB[4], boxB[5] }, new[] { 3 });

        if (_numOps.GreaterThan(engine.TensorSum(engine.TensorGreaterThan(minA, maxA)), _numOps.Zero))
        {
            throw new ArgumentException("Box A has invalid coordinates: min values must be <= max values", nameof(boxA));
        }

        if (_numOps.GreaterThan(engine.TensorSum(engine.TensorGreaterThan(minB, maxB)), _numOps.Zero))
        {
            throw new ArgumentException("Box B has invalid coordinates: min values must be <= max values", nameof(boxB));
        }

        // Intersection extent per axis, clamped at zero, and the box volumes.
        var overlap = engine.TensorMax(
            engine.TensorSubtract(engine.TensorMin(maxA, maxB), engine.TensorMax(minA, minB)), _numOps.Zero);
        T intersection = engine.Product(overlap.ToVector());
        T volA = engine.Product(engine.TensorSubtract(maxA, minA).ToVector());
        T volB = engine.Product(engine.TensorSubtract(maxB, minB).ToVector());

        T union = _numOps.Subtract(_numOps.Add(volA, volB), intersection);

        if (!_numOps.GreaterThan(union, _numOps.Zero)) return _numOps.Zero;

        return _numOps.Divide(intersection, union);
    }
}
