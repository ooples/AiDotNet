using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides sampling utilities for point cloud data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class PointCloudSampling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Performs uniform random sampling on a point cloud.
    /// </summary>
    public static PointCloudData<T> UniformSample(PointCloudData<T> pointCloud, int numSamples, int? seed = null)
    {
        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        int numPoints = pointCloud.NumPoints;

        if (numSamples >= numPoints)
        {
            return pointCloud;
        }

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        var indices = new int[numPoints];
        for (int i = 0; i < numPoints; i++)
        {
            indices[i] = i;
        }

        for (int i = 0; i < numSamples; i++)
        {
            int j = random.Next(i, numPoints);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var chosen = new int[numSamples];
        Array.Copy(indices, chosen, numSamples);
        return GatherPoints(pointCloud, chosen);
    }

    /// <summary>
    /// Performs farthest point sampling (FPS) on a point cloud.
    /// </summary>
    /// <param name="pointCloud">The input point cloud to sample from.</param>
    /// <param name="numSamples">The number of points to sample.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A new point cloud with the sampled points.</returns>
    /// <exception cref="ArgumentException">Thrown when point cloud has fewer than 3 features (xyz coordinates).</exception>
    public static PointCloudData<T> FarthestPointSample(PointCloudData<T> pointCloud, int numSamples, int? seed = null)
    {
        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        int numPoints = pointCloud.NumPoints;
        int numFeatures = pointCloud.NumFeatures;

        if (numSamples >= numPoints)
        {
            return pointCloud;
        }

        if (numFeatures < 3)
        {
            throw new ArgumentException(
                $"FarthestPointSample requires at least 3 features (xyz coordinates). Got {numFeatures} features.",
                nameof(pointCloud));
        }

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        var engine = AiDotNetEngine.Current;
        var positions = GeometryEngineOps<T>.Positions(pointCloud.Points);
        var selectedIndices = new int[numSamples];
        var minDistances = engine.TensorFullLike(new Tensor<T>(new[] { numPoints }), NumOps.MaxValue);
        var minusOne = NumOps.FromDouble(-1);

        int firstIdx = random.Next(numPoints);
        selectedIndices[0] = firstIdx;
        minDistances = UpdateDistances(positions, firstIdx, minDistances);

        for (int i = 1; i < numSamples; i++)
        {
            // First index of the largest running distance, as the strict "> best" scan from -1 finds it;
            // NaN distances never win, so they are scored -1 (an all-NaN scan keeps index 0).
            var scores = engine.TensorWhere(
                engine.TensorIsNan(minDistances), engine.TensorFullLike(minDistances, minusOne), minDistances);
            var best = engine.TensorMaxValue(scores);
            int farthestIdx = engine.TensorNonzero(engine.TensorEquals(scores, best)).ToArray()[0];

            selectedIndices[i] = farthestIdx;
            minDistances = UpdateDistances(positions, farthestIdx, minDistances);
        }

        return GatherPoints(pointCloud, selectedIndices);
    }

    /// <summary>
    /// Performs Poisson disk sampling on a point cloud.
    /// </summary>
    /// <param name="pointCloud">The input point cloud to sample from.</param>
    /// <param name="minDistance">The minimum distance between sampled points.</param>
    /// <param name="maxSamples">Maximum number of samples to return. 0 for unlimited.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A new point cloud with the sampled points.</returns>
    /// <exception cref="ArgumentException">Thrown when point cloud has fewer than 3 features (xyz coordinates).</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when minDistance is not positive.</exception>
    public static PointCloudData<T> PoissonDiskSample(
        PointCloudData<T> pointCloud,
        T minDistance,
        int maxSamples = 0,
        int? seed = null)
    {
        if (NumOps.LessThanOrEquals(minDistance, NumOps.Zero))
        {
            throw new ArgumentOutOfRangeException(nameof(minDistance), "Minimum distance must be positive.");
        }

        int numPoints = pointCloud.NumPoints;
        int numFeatures = pointCloud.NumFeatures;

        if (numPoints == 0)
        {
            return pointCloud;
        }

        if (numFeatures < 3)
        {
            throw new ArgumentException(
                $"PoissonDiskSample requires at least 3 features (xyz coordinates). Got {numFeatures} features.",
                nameof(pointCloud));
        }

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var minDistSq = NumOps.Multiply(minDistance, minDistance);

        var indices = new int[numPoints];
        for (int i = 0; i < numPoints; i++)
        {
            indices[i] = i;
        }

        for (int i = numPoints - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var engine = AiDotNetEngine.Current;
        var positions = GeometryEngineOps<T>.Positions(pointCloud.Points);
        var selectedIndices = new List<int>();

        foreach (int idx in indices)
        {
            if (maxSamples > 0 && selectedIndices.Count >= maxSamples)
            {
                break;
            }

            // The candidate is rejected when an already selected point lies strictly closer than the minimum
            // distance. Only the selected points are measured (the first candidate has none to check), so each
            // candidate costs work proportional to the sample, not to the whole cloud.
            bool tooClose = false;
            if (selectedIndices.Count > 0)
            {
                var selectedIds = selectedIndices.ToArray();
                var selected = engine.TensorIndexSelect(
                    positions, new Tensor<int>(new[] { selectedIds.Length }, new Vector<int>(selectedIds)), 0);
                var candidate = engine.TensorSlice(positions, new[] { idx, 0 }, new[] { 1, 3 });
                var offsets = engine.TensorSubtract(selected, engine.TensorBroadcastTo(candidate, selected.Shape.ToArray()));
                var distancesSq = GeometryEngineOps<T>.RowSquaredNorms(offsets);
                tooClose = engine.TensorCountNonzero(engine.TensorLessThan(distancesSq, minDistSq)) > 0;
            }

            if (!tooClose)
            {
                selectedIndices.Add(idx);
            }
        }

        return GatherPoints(pointCloud, selectedIndices.ToArray());
    }

    /// <summary>
    /// Samples points by voxel grid downsampling.
    /// </summary>
    /// <param name="pointCloud">The input point cloud to sample from.</param>
    /// <param name="voxelSize">The size of each voxel in the grid.</param>
    /// <returns>A new downsampled point cloud with one point per occupied voxel.</returns>
    /// <exception cref="ArgumentException">Thrown when point cloud has fewer than 3 features (xyz coordinates).</exception>
    public static PointCloudData<T> VoxelGridSample(PointCloudData<T> pointCloud, T voxelSize)
    {
        int numPoints = pointCloud.NumPoints;
        int numFeatures = pointCloud.NumFeatures;

        if (numPoints == 0 || NumOps.LessThanOrEquals(voxelSize, NumOps.Zero))
        {
            return pointCloud;
        }

        if (numFeatures < 3)
        {
            throw new ArgumentException(
                $"VoxelGridSample requires at least 3 features (xyz coordinates). Got {numFeatures} features.",
                nameof(pointCloud));
        }

        if (typeof(T) == typeof(Half))
        {
            // Half voxel sums overflow above 65504 and its counts stop being exact above 2048: average in float.
            var widened = PointCloudSampling<float>.VoxelGridSample(
                new PointCloudData<float>(
                    pointCloud.Points.Cast<float>(),
                    pointCloud.Labels is null ? null : Tensor<T>.FromVector(pointCloud.Labels).Cast<float>().ToVector()),
                NumOps.ToFloat(voxelSize));
            return new PointCloudData<T>(
                widened.Points.Cast<T>(),
                widened.Labels is null ? null : Tensor<float>.FromVector(widened.Labels).Cast<T>().ToVector());
        }

        // Compute bounding box minimum for origin-relative voxel indexing
        // This is the standard approach (used by Open3D, PCL, etc.) and ensures
        // consistent voxelization regardless of point cloud position
        var engine = AiDotNetEngine.Current;
        var positions = GeometryEngineOps<T>.Positions(pointCloud.Points);
        var (min, _) = GeometryEngineOps<T>.RunningColumnBounds(positions);
        var relative = engine.TensorSubtract(positions, engine.TensorBroadcastTo(min.Reshape(1, 3), positions.Shape.ToArray()));
        var cells = engine.TensorFloor(engine.TensorDivideScalar(relative, voxelSize));

        // Equal cell coordinates share an id per axis, so the id triple identifies the voxel.
        var cellIds = new int[3][];
        for (int axis = 0; axis < 3; axis++)
        {
            var (_, inverse, _) = engine.TensorUniqueWithInfo(GeometryEngineOps<T>.Column(cells, axis), false, true, false);
            cellIds[axis] = inverse?.ToArray()
                ?? throw new InvalidOperationException("TensorUniqueWithInfo returned no inverse indices.");
        }

        // Voxels are numbered in order of first appearance, the order the voxel dictionary enumerated them.
        var voxelOf = new Dictionary<(int, int, int), int>();
        var members = new List<List<int>>();
        var groupOf = new int[numPoints];
        for (int i = 0; i < numPoints; i++)
        {
            var key = (cellIds[0][i], cellIds[1][i], cellIds[2][i]);
            if (!voxelOf.TryGetValue(key, out int group))
            {
                group = members.Count;
                voxelOf[key] = group;
                members.Add([]);
            }

            groupOf[i] = group;
            members[group].Add(i);
        }

        int numSamples = members.Count;
        var groupIndex = new Tensor<int>(groupOf, new[] { numPoints });
        var sums = engine.TensorIndexAdd(new Tensor<T>(new[] { numSamples, numFeatures }), 0, groupIndex, pointCloud.Points);
        // Member counts are taken in int (summing ones in T saturates), then converted once.
        var countValues = new int[numSamples];
        for (int g = 0; g < numSamples; g++)
        {
            countValues[g] = members[g].Count;
        }

        var counts = new Tensor<int>(countValues, new[] { numSamples, 1 }).Cast<T>();
        var tensor = engine.TensorDivide(sums, engine.TensorBroadcastTo(counts, sums.Shape.ToArray()));

        Vector<T>? sampledLabels = null;
        if (pointCloud.Labels != null)
        {
            var (labelValues, labelInverse, _) = engine.TensorUniqueWithInfo(
                Tensor<T>.FromVector(pointCloud.Labels), false, true, false);
            var labelIds = labelInverse?.ToArray()
                ?? throw new InvalidOperationException("TensorUniqueWithInfo returned no inverse indices.");
            var majority = new int[numSamples];
            for (int g = 0; g < numSamples; g++)
            {
                // Most frequent label; ties go to the label seen first, as the stable descending sort picked.
                var labelCounts = new Dictionary<int, int>();
                var firstSeen = new List<int>();
                foreach (int idx in members[g])
                {
                    int label = labelIds[idx];
                    if (!labelCounts.TryGetValue(label, out int count))
                    {
                        firstSeen.Add(label);
                    }

                    labelCounts[label] = count + 1;
                }

                int best = firstSeen[0];
                foreach (int label in firstSeen)
                {
                    if (labelCounts[label] > labelCounts[best])
                    {
                        best = label;
                    }
                }

                majority[g] = best;
            }

            var majorityIndex = new Tensor<int>(majority, new[] { numSamples });
            sampledLabels = engine.TensorIndexSelect(labelValues, majorityIndex, 0).ToVector();
        }

        return new PointCloudData<T>(tensor, sampledLabels);
    }

    private static Tensor<T> UpdateDistances(Tensor<T> positions, int newPointIdx, Tensor<T> minDistances)
    {
        var engine = AiDotNetEngine.Current;
        int numPoints = positions.Shape[0];
        var newPoint = engine.TensorSlice(positions, new[] { newPointIdx, 0 }, new[] { 1, 3 });
        var offsets = engine.TensorSubtract(positions, engine.TensorBroadcastTo(newPoint, positions.Shape.ToArray()));
        var distances = GeometryEngineOps<T>.RowSquaredNorms(offsets).Reshape(numPoints);

        // Only a strictly smaller distance replaces the running minimum (NaN never does).
        var updated = engine.TensorWhere(engine.TensorLessThan(distances, minDistances), distances, minDistances);
        updated[newPointIdx] = NumOps.Zero;
        return updated;
    }

    private static PointCloudData<T> GatherPoints(PointCloudData<T> pointCloud, int[] indices)
    {
        if (indices.Length == 0)
        {
            var emptyLabels = pointCloud.Labels != null ? new Vector<T>(new T[0]) : null;
            return new PointCloudData<T>(new Tensor<T>(new[] { 0, pointCloud.NumFeatures }), emptyLabels);
        }

        var engine = AiDotNetEngine.Current;
        var index = new Tensor<int>(indices, new[] { indices.Length });
        var points = engine.TensorIndexSelect(pointCloud.Points, index, 0);
        Vector<T>? labels = null;
        if (pointCloud.Labels != null)
        {
            labels = engine.TensorIndexSelect(Tensor<T>.FromVector(pointCloud.Labels), index, 0).ToVector();
        }

        return new PointCloudData<T>(points, labels);
    }
}
