using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Optimization
{
    /// <summary>
    /// Provides loop optimization techniques including tiling and vectorization hints.
    /// These utilities help maximize performance for tensor operations.
    /// </summary>
    public static class LoopOptimizer
    {
        /// <summary>
        /// 2D loop tiling for matrix operations
        /// </summary>
        public static void Tile2D(
            int rows, int cols,
            int tileSize,
            Action<int, int, int, int> tileAction)
        {
            for (int i = 0; i < rows; i += tileSize)
            {
                int iEnd = Math.Min(i + tileSize, rows);

                for (int j = 0; j < cols; j += tileSize)
                {
                    int jEnd = Math.Min(j + tileSize, cols);

                    tileAction(i, iEnd, j, jEnd);
                }
            }
        }

        /// <summary>
        /// 3D loop tiling for tensor operations
        /// </summary>
        public static void Tile3D(
            int dim1, int dim2, int dim3,
            int tileSize1, int tileSize2, int tileSize3,
            Action<int, int, int, int, int, int> tileAction)
        {
            for (int i = 0; i < dim1; i += tileSize1)
            {
                int iEnd = Math.Min(i + tileSize1, dim1);

                for (int j = 0; j < dim2; j += tileSize2)
                {
                    int jEnd = Math.Min(j + tileSize2, dim2);

                    for (int k = 0; k < dim3; k += tileSize3)
                    {
                        int kEnd = Math.Min(k + tileSize3, dim3);

                        tileAction(i, iEnd, j, jEnd, k, kEnd);
                    }
                }
            }
        }

        /// <summary>
        /// Loop unrolling hint - processes elements in groups
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void UnrollBy4(int length, Action<int> action)
        {
            int i = 0;
            int unrolledLength = length & ~3; // Round down to multiple of 4

            // Unrolled loop
            for (; i < unrolledLength; i += 4)
            {
                action(i);
                action(i + 1);
                action(i + 2);
                action(i + 3);
            }

            // Remainder
            for (; i < length; i++)
            {
                action(i);
            }
        }

        /// <summary>
        /// Loop unrolling by 8 for better SIMD utilization
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void UnrollBy8(int length, Action<int> action)
        {
            int i = 0;
            int unrolledLength = length & ~7;

            for (; i < unrolledLength; i += 8)
            {
                action(i);
                action(i + 1);
                action(i + 2);
                action(i + 3);
                action(i + 4);
                action(i + 5);
                action(i + 6);
                action(i + 7);
            }

            for (; i < length; i++)
            {
                action(i);
            }
        }

        /// <summary>
        /// Strip mining - breaks loop into chunks for better cache utilization
        /// </summary>
        public static void StripMine(int totalSize, int stripSize, Action<int, int> stripAction)
        {
            for (int start = 0; start < totalSize; start += stripSize)
            {
                int end = Math.Min(start + stripSize, totalSize);
                stripAction(start, end);
            }
        }

        /// <summary>
        /// Loop fusion helper - executes multiple operations in a single pass
        /// </summary>
        public static void Fuse(int length, params Action<int>[] actions)
        {
            for (int i = 0; i < length; i++)
            {
                foreach (var action in actions)
                {
                    action(i);
                }
            }
        }

        /// <summary>
        /// Loop interchange optimization for better cache locality
        /// Automatically chooses better loop order based on access pattern
        /// </summary>
        public static void OptimalOrder2D(
            int rows, int cols,
            bool rowMajorAccess,
            Action<int, int> action)
        {
            if (rowMajorAccess)
            {
                // Standard order for row-major access
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        action(i, j);
                    }
                }
            }
            else
            {
                // Interchanged order for column-major access
                for (int j = 0; j < cols; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        action(i, j);
                    }
                }
            }
        }

        /// <summary>
        /// Parallel loop tiling with work stealing
        /// </summary>
        public static void ParallelTile2D(
            int rows, int cols,
            int tileSize,
            Action<int, int, int, int> tileAction)
        {
            int numTilesI = (rows + tileSize - 1) / tileSize;
            int numTilesJ = (cols + tileSize - 1) / tileSize;
            int totalTiles = numTilesI * numTilesJ;

            System.Threading.Tasks.Parallel.For(0, totalTiles, tileIdx =>
            {
                int ti = tileIdx / numTilesJ;
                int tj = tileIdx % numTilesJ;

                int iStart = ti * tileSize;
                int iEnd = Math.Min(iStart + tileSize, rows);

                int jStart = tj * tileSize;
                int jEnd = Math.Min(jStart + tileSize, cols);

                tileAction(iStart, iEnd, jStart, jEnd);
            });
        }

        /// <summary>
        /// Automatically determines optimal tile size based on data dimensions and cache size
        /// </summary>
        public static int DetermineOptimalTileSize(int dimension, int elementSize = 4)
        {
            var caps = PlatformDetector.Capabilities;
            int l1Size = caps.L1CacheSize;

            // Aim to fit two tiles in L1 cache (one read, one write)
            int maxElements = l1Size / (2 * elementSize);

            // Find power of 2 that fits
            int tileSize = 16; // Minimum tile size
            while (tileSize * tileSize * 2 < maxElements && tileSize < dimension)
            {
                tileSize *= 2;
            }

            return Math.Min(tileSize, dimension);
        }
    }
}
