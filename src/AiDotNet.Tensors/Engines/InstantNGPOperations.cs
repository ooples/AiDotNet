using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Provides vectorized implementations of Instant-NGP operations.
/// These operations are shared between CPU and GPU engines.
/// </summary>
public static class InstantNGPOperations
{
    #region Multiresolution Hash Encoding

    /// <summary>
    /// Performs multiresolution hash encoding for Instant-NGP.
    /// </summary>
    public static Tensor<T> MultiresolutionHashEncoding<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel)
    {
        if (positions == null) throw new ArgumentNullException(nameof(positions));
        if (hashTables == null) throw new ArgumentNullException(nameof(hashTables));
        if (resolutions == null) throw new ArgumentNullException(nameof(resolutions));
        if (positions.Shape.Length != 2 || positions.Shape[1] != 3)
            throw new ArgumentException("Positions must be [N, 3].", nameof(positions));
        if (hashTables.Length != resolutions.Length)
            throw new ArgumentException("Hash tables and resolutions count mismatch.");
        if (featuresPerLevel <= 0)
            throw new ArgumentOutOfRangeException(nameof(featuresPerLevel));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numPoints = positions.Shape[0];
        int numLevels = hashTables.Length;
        int totalFeatures = numLevels * featuresPerLevel;

        var features = new T[numPoints * totalFeatures];

        Parallel.For(0, numPoints, n =>
        {
            double px = numOps.ToDouble(positions.GetFlat(n * 3));
            double py = numOps.ToDouble(positions.GetFlat(n * 3 + 1));
            double pz = numOps.ToDouble(positions.GetFlat(n * 3 + 2));

            // Clamp to [0, 1]
            px = Clamp01(px);
            py = Clamp01(py);
            pz = Clamp01(pz);

            for (int level = 0; level < numLevels; level++)
            {
                int resolution = resolutions[level];
                var table = hashTables[level];
                int tableSize = table.Shape[0];

                // Scale position to grid
                double gx = px * resolution;
                double gy = py * resolution;
                double gz = pz * resolution;

                // Get grid cell coordinates
                int x0 = (int)Math.Floor(gx);
                int y0 = (int)Math.Floor(gy);
                int z0 = (int)Math.Floor(gz);

                // Interpolation weights
                double fx = gx - x0;
                double fy = gy - y0;
                double fz = gz - z0;

                int x1 = x0 + 1;
                int y1 = y0 + 1;
                int z1 = z0 + 1;

                // Trilinear interpolation weights
                double w000 = (1 - fx) * (1 - fy) * (1 - fz);
                double w001 = (1 - fx) * (1 - fy) * fz;
                double w010 = (1 - fx) * fy * (1 - fz);
                double w011 = (1 - fx) * fy * fz;
                double w100 = fx * (1 - fy) * (1 - fz);
                double w101 = fx * (1 - fy) * fz;
                double w110 = fx * fy * (1 - fz);
                double w111 = fx * fy * fz;

                // Hash indices for 8 corners
                int h000 = SpatialHash(x0, y0, z0, tableSize);
                int h001 = SpatialHash(x0, y0, z1, tableSize);
                int h010 = SpatialHash(x0, y1, z0, tableSize);
                int h011 = SpatialHash(x0, y1, z1, tableSize);
                int h100 = SpatialHash(x1, y0, z0, tableSize);
                int h101 = SpatialHash(x1, y0, z1, tableSize);
                int h110 = SpatialHash(x1, y1, z0, tableSize);
                int h111 = SpatialHash(x1, y1, z1, tableSize);

                int featureBase = n * totalFeatures + level * featuresPerLevel;

                // Interpolate features
                for (int f = 0; f < featuresPerLevel; f++)
                {
                    double value =
                        w000 * numOps.ToDouble(table.GetFlat(h000 * featuresPerLevel + f)) +
                        w001 * numOps.ToDouble(table.GetFlat(h001 * featuresPerLevel + f)) +
                        w010 * numOps.ToDouble(table.GetFlat(h010 * featuresPerLevel + f)) +
                        w011 * numOps.ToDouble(table.GetFlat(h011 * featuresPerLevel + f)) +
                        w100 * numOps.ToDouble(table.GetFlat(h100 * featuresPerLevel + f)) +
                        w101 * numOps.ToDouble(table.GetFlat(h101 * featuresPerLevel + f)) +
                        w110 * numOps.ToDouble(table.GetFlat(h110 * featuresPerLevel + f)) +
                        w111 * numOps.ToDouble(table.GetFlat(h111 * featuresPerLevel + f));

                    features[featureBase + f] = numOps.FromDouble(value);
                }
            }
        });

        return new Tensor<T>(features, [numPoints, totalFeatures]);
    }

    /// <summary>
    /// Computes the backward pass for multiresolution hash encoding.
    /// </summary>
    public static Tensor<T>[] MultiresolutionHashEncodingBackward<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel,
        Tensor<T> outputGradient)
    {
        if (positions == null) throw new ArgumentNullException(nameof(positions));
        if (hashTables == null) throw new ArgumentNullException(nameof(hashTables));
        if (outputGradient == null) throw new ArgumentNullException(nameof(outputGradient));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numPoints = positions.Shape[0];
        int numLevels = hashTables.Length;
        int totalFeatures = numLevels * featuresPerLevel;

        // Create gradient tensors for each hash table
        var hashGradients = new Tensor<T>[numLevels];
        var lockObjects = new object[numLevels][];

        for (int level = 0; level < numLevels; level++)
        {
            int tableSize = hashTables[level].Shape[0];
            hashGradients[level] = new Tensor<T>(new T[tableSize * featuresPerLevel], [tableSize, featuresPerLevel]);
            lockObjects[level] = new object[tableSize];
            for (int i = 0; i < tableSize; i++)
            {
                lockObjects[level][i] = new object();
            }
        }

        Parallel.For(0, numPoints, n =>
        {
            double px = Clamp01(numOps.ToDouble(positions.GetFlat(n * 3)));
            double py = Clamp01(numOps.ToDouble(positions.GetFlat(n * 3 + 1)));
            double pz = Clamp01(numOps.ToDouble(positions.GetFlat(n * 3 + 2)));

            for (int level = 0; level < numLevels; level++)
            {
                int resolution = resolutions[level];
                int tableSize = hashTables[level].Shape[0];
                var gradTable = hashGradients[level];

                double gx = px * resolution;
                double gy = py * resolution;
                double gz = pz * resolution;

                int x0 = (int)Math.Floor(gx);
                int y0 = (int)Math.Floor(gy);
                int z0 = (int)Math.Floor(gz);

                double fx = gx - x0;
                double fy = gy - y0;
                double fz = gz - z0;

                int x1 = x0 + 1;
                int y1 = y0 + 1;
                int z1 = z0 + 1;

                double w000 = (1 - fx) * (1 - fy) * (1 - fz);
                double w001 = (1 - fx) * (1 - fy) * fz;
                double w010 = (1 - fx) * fy * (1 - fz);
                double w011 = (1 - fx) * fy * fz;
                double w100 = fx * (1 - fy) * (1 - fz);
                double w101 = fx * (1 - fy) * fz;
                double w110 = fx * fy * (1 - fz);
                double w111 = fx * fy * fz;

                int h000 = SpatialHash(x0, y0, z0, tableSize);
                int h001 = SpatialHash(x0, y0, z1, tableSize);
                int h010 = SpatialHash(x0, y1, z0, tableSize);
                int h011 = SpatialHash(x0, y1, z1, tableSize);
                int h100 = SpatialHash(x1, y0, z0, tableSize);
                int h101 = SpatialHash(x1, y0, z1, tableSize);
                int h110 = SpatialHash(x1, y1, z0, tableSize);
                int h111 = SpatialHash(x1, y1, z1, tableSize);

                int gradBase = n * totalFeatures + level * featuresPerLevel;

                for (int f = 0; f < featuresPerLevel; f++)
                {
                    double grad = numOps.ToDouble(outputGradient.GetFlat(gradBase + f));
                    if (Math.Abs(grad) < 1e-10) continue;

                    // Accumulate gradients with locking for thread safety
                    AccumulateGradient(gradTable, lockObjects[level], h000, f, featuresPerLevel, grad * w000, numOps);
                    AccumulateGradient(gradTable, lockObjects[level], h001, f, featuresPerLevel, grad * w001, numOps);
                    AccumulateGradient(gradTable, lockObjects[level], h010, f, featuresPerLevel, grad * w010, numOps);
                    AccumulateGradient(gradTable, lockObjects[level], h011, f, featuresPerLevel, grad * w011, numOps);
                    AccumulateGradient(gradTable, lockObjects[level], h100, f, featuresPerLevel, grad * w100, numOps);
                    AccumulateGradient(gradTable, lockObjects[level], h101, f, featuresPerLevel, grad * w101, numOps);
                    AccumulateGradient(gradTable, lockObjects[level], h110, f, featuresPerLevel, grad * w110, numOps);
                    AccumulateGradient(gradTable, lockObjects[level], h111, f, featuresPerLevel, grad * w111, numOps);
                }
            }
        });

        return hashGradients;
    }

    private static void AccumulateGradient<T>(
        Tensor<T> gradTable,
        object[] locks,
        int hashIdx,
        int featureIdx,
        int featuresPerLevel,
        double grad,
        INumericOperations<T> numOps)
    {
        int idx = hashIdx * featuresPerLevel + featureIdx;
        lock (locks[hashIdx])
        {
            double current = numOps.ToDouble(gradTable.GetFlat(idx));
            gradTable.SetFlat(idx, numOps.FromDouble(current + grad));
        }
    }

    #endregion

    #region Occupancy Grid

    /// <summary>
    /// Updates occupancy grid for efficient ray sampling.
    /// </summary>
    public static Tensor<T> UpdateOccupancyGrid<T>(
        Tensor<T> occupancyGrid,
        Tensor<T> densities,
        Tensor<T> positions,
        int gridSize,
        T threshold,
        T decayFactor)
    {
        if (occupancyGrid == null) throw new ArgumentNullException(nameof(occupancyGrid));
        if (densities == null) throw new ArgumentNullException(nameof(densities));
        if (positions == null) throw new ArgumentNullException(nameof(positions));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numSamples = positions.Shape[0];
        double thresholdVal = numOps.ToDouble(threshold);
        double decayVal = numOps.ToDouble(decayFactor);

        // Create new grid as copy
        var newGrid = new T[gridSize * gridSize * gridSize];
        Array.Copy(occupancyGrid.Data, newGrid, newGrid.Length);

        // Apply decay
        for (int i = 0; i < newGrid.Length; i++)
        {
            newGrid[i] = numOps.FromDouble(numOps.ToDouble(newGrid[i]) * decayVal);
        }

        // Use striped locking to reduce contention while avoiding memory overhead of per-cell locks
        // Number of stripes is a power of 2 for efficient modulo operation via bitwise AND
        const int NumStripes = 256;
        var stripeLocks = new object[NumStripes];
        for (int i = 0; i < NumStripes; i++)
        {
            stripeLocks[i] = new object();
        }

        Parallel.For(0, numSamples, i =>
        {
            double px = numOps.ToDouble(positions.GetFlat(i * 3));
            double py = numOps.ToDouble(positions.GetFlat(i * 3 + 1));
            double pz = numOps.ToDouble(positions.GetFlat(i * 3 + 2));
            double density = numOps.ToDouble(densities.GetFlat(i));

            // Clamp to [0, 1]
            px = Clamp01(px);
            py = Clamp01(py);
            pz = Clamp01(pz);

            // Map to grid cell
            int gx = Math.Min((int)(px * gridSize), gridSize - 1);
            int gy = Math.Min((int)(py * gridSize), gridSize - 1);
            int gz = Math.Min((int)(pz * gridSize), gridSize - 1);

            int gridIdx = (gx * gridSize + gy) * gridSize + gz;

            // Use striped lock based on grid index to reduce contention
            int stripeIdx = gridIdx & (NumStripes - 1);
            lock (stripeLocks[stripeIdx])
            {
                double current = numOps.ToDouble(newGrid[gridIdx]);
                double alpha = 1.0 - Math.Exp(-density);
                newGrid[gridIdx] = numOps.FromDouble(Math.Max(current, alpha));
            }
        });

        return new Tensor<T>(newGrid, [gridSize, gridSize, gridSize]);
    }

    /// <summary>
    /// Samples rays while skipping empty space using occupancy grid.
    /// </summary>
    public static (Tensor<T> positions, Tensor<T> directions, Tensor<bool> validMask, Tensor<T> tValues) 
        SampleRaysWithOccupancy<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        uint[] occupancyBitfield,
        int gridSize,
        Vector<T> sceneBoundsMin,
        Vector<T> sceneBoundsMax,
        T nearBound,
        T farBound,
        int maxSamples)
    {
        if (rayOrigins == null) throw new ArgumentNullException(nameof(rayOrigins));
        if (rayDirections == null) throw new ArgumentNullException(nameof(rayDirections));
        if (occupancyBitfield == null) throw new ArgumentNullException(nameof(occupancyBitfield));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numRays = rayOrigins.Shape[0];

        double near = numOps.ToDouble(nearBound);
        double far = numOps.ToDouble(farBound);

        double minX = numOps.ToDouble(sceneBoundsMin[0]);
        double minY = numOps.ToDouble(sceneBoundsMin[1]);
        double minZ = numOps.ToDouble(sceneBoundsMin[2]);
        double maxX = numOps.ToDouble(sceneBoundsMax[0]);
        double maxY = numOps.ToDouble(sceneBoundsMax[1]);
        double maxZ = numOps.ToDouble(sceneBoundsMax[2]);

        double invSizeX = 1.0 / Math.Max(1e-10, maxX - minX);
        double invSizeY = 1.0 / Math.Max(1e-10, maxY - minY);
        double invSizeZ = 1.0 / Math.Max(1e-10, maxZ - minZ);

        int totalSamples = numRays * maxSamples;
        var positions = new T[totalSamples * 3];
        var directions = new T[totalSamples * 3];
        var validMask = new bool[totalSamples];
        var tValues = new T[numRays * maxSamples];

        int totalCells = gridSize * gridSize * gridSize;

        Parallel.For(0, numRays, r =>
        {
            double ox = numOps.ToDouble(rayOrigins.GetFlat(r * 3));
            double oy = numOps.ToDouble(rayOrigins.GetFlat(r * 3 + 1));
            double oz = numOps.ToDouble(rayOrigins.GetFlat(r * 3 + 2));
            double dx = numOps.ToDouble(rayDirections.GetFlat(r * 3));
            double dy = numOps.ToDouble(rayDirections.GetFlat(r * 3 + 1));
            double dz = numOps.ToDouble(rayDirections.GetFlat(r * 3 + 2));

            // Compute ray-scene intersection
            double tMin = near;
            double tMax = far;

            if (!ComputeRayBounds(ox, oy, oz, dx, dy, dz, minX, maxX, minY, maxY, minZ, maxZ, ref tMin, ref tMax))
            {
                // Ray doesn't intersect scene
                return;
            }

            double step = (tMax - tMin) / maxSamples;
            int baseSampleIdx = r * maxSamples;

            for (int s = 0; s < maxSamples; s++)
            {
                double t = tMin + step * (s + 0.5);
                double px = ox + t * dx;
                double py = oy + t * dy;
                double pz = oz + t * dz;

                // Normalize to [0, 1]
                double nx = (px - minX) * invSizeX;
                double ny = (py - minY) * invSizeY;
                double nz = (pz - minZ) * invSizeZ;

                nx = Clamp01(nx);
                ny = Clamp01(ny);
                nz = Clamp01(nz);

                // Map to grid cell
                int gx = Math.Min((int)(nx * gridSize), gridSize - 1);
                int gy = Math.Min((int)(ny * gridSize), gridSize - 1);
                int gz = Math.Min((int)(nz * gridSize), gridSize - 1);

                int gridIdx = (gx * gridSize + gy) * gridSize + gz;

                // Check occupancy bitfield
                bool isOccupied = gridIdx < totalCells && IsBitSet(occupancyBitfield, gridIdx);

                int sampleIdx = baseSampleIdx + s;
                int posIdx = sampleIdx * 3;

                positions[posIdx] = numOps.FromDouble(px);
                positions[posIdx + 1] = numOps.FromDouble(py);
                positions[posIdx + 2] = numOps.FromDouble(pz);
                directions[posIdx] = rayDirections.GetFlat(r * 3);
                directions[posIdx + 1] = rayDirections.GetFlat(r * 3 + 1);
                directions[posIdx + 2] = rayDirections.GetFlat(r * 3 + 2);
                validMask[sampleIdx] = isOccupied;
                tValues[sampleIdx] = numOps.FromDouble(t);
            }
        });

        return (
            new Tensor<T>(positions, [totalSamples, 3]),
            new Tensor<T>(directions, [totalSamples, 3]),
            new Tensor<bool>(validMask, [totalSamples]),
            new Tensor<T>(tValues, [numRays, maxSamples])
        );
    }

    #endregion

    #region Helper Methods

    private static int SpatialHash(int x, int y, int z, int tableSize)
    {
        unchecked
        {
            const uint p1 = 73856093;
            const uint p2 = 19349663;
            const uint p3 = 83492791;

            uint hx = (uint)x * p1;
            uint hy = (uint)y * p2;
            uint hz = (uint)z * p3;

            return (int)((hx ^ hy ^ hz) % (uint)tableSize);
        }
    }

    private static bool IsBitSet(uint[] bitfield, int index)
    {
        int word = index >> 5;  // index / 32
        int bit = index & 31;   // index % 32
        if (word >= bitfield.Length) return false;
        return (bitfield[word] & (1u << bit)) != 0;
    }

    private static bool ComputeRayBounds(
        double ox, double oy, double oz,
        double dx, double dy, double dz,
        double minX, double maxX,
        double minY, double maxY,
        double minZ, double maxZ,
        ref double tMin, ref double tMax)
    {
        if (!IntersectAxis(ox, dx, minX, maxX, ref tMin, ref tMax)) return false;
        if (!IntersectAxis(oy, dy, minY, maxY, ref tMin, ref tMax)) return false;
        if (!IntersectAxis(oz, dz, minZ, maxZ, ref tMin, ref tMax)) return false;
        return tMax >= tMin;
    }

    private static bool IntersectAxis(
        double o, double d, double minBound, double maxBound,
        ref double tMin, ref double tMax)
    {
        const double eps = 1e-8;
        if (Math.Abs(d) < eps)
        {
            return o >= minBound && o <= maxBound;
        }

        double inv = 1.0 / d;
        double t1 = (minBound - o) * inv;
        double t2 = (maxBound - o) * inv;

        if (t1 > t2)
        {
            (t1, t2) = (t2, t1);
        }

        tMin = Math.Max(tMin, t1);
        tMax = Math.Min(tMax, t2);
        return tMax >= tMin;
    }

    private static double Clamp01(double value)
    {
        if (value <= 0.0) return 0.0;
        if (value >= 1.0 - 1e-6) return 1.0 - 1e-6;
        return value;
    }

    #endregion
}
