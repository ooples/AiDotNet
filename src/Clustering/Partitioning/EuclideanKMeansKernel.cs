using SimdVector = System.Numerics.Vector<double>;
using System.Runtime.InteropServices;

namespace AiDotNet.Clustering.Partitioning;

/// <summary>
/// The Euclidean Lloyd step on a dense row-major <see cref="double"/> copy of the data: nearest-center assignment by
/// <c>||x||^2 - 2 x.c + ||c||^2</c>, center sums, and the inertia, over fixed row blocks run in parallel.
/// </summary>
/// <remarks>
/// <para>
/// KMeans used to compute every point-to-center distance through the generic distance metric, one generic numeric
/// operation per element and a fresh vector per row: about 50x slower than scikit-learn on 20,000 x 20. This is the
/// formulation scikit-learn uses, with the per-row norms computed once.
/// </para>
/// <para>
/// Blocks are a fixed number of rows, not a function of the thread count, and their partial sums are merged in block
/// order, so a fit is bit-identical however many threads run it.
/// </para>
/// </remarks>
internal sealed class EuclideanKMeansKernel
{
    private const int BlockRows = 1024;

    private readonly double[] _x;
    private readonly double[] _rowNorms;

    public EuclideanKMeansKernel(double[] x, int rows, int columns)
    {
        _x = x;
        Rows = rows;
        Columns = columns;
        _rowNorms = new double[rows];
        for (int i = 0; i < rows; i++)
            _rowNorms[i] = Dot(x, i * columns, x, i * columns, columns);
    }

    public int Rows { get; }

    public int Columns { get; }

    public double[] Data => _x;

    private int BlockCount => (Rows + BlockRows - 1) / BlockRows;

    /// <summary>
    /// Assigns every row to its nearest center (the lowest index on a tie) and, when <paramref name="sums"/> and
    /// <paramref name="counts"/> are given, accumulates each cluster's coordinate sums and counts.
    /// </summary>
    public void Assign(double[] centers, int clusters, int[] labels, double[]? sums, int[]? counts)
    {
        int columns = Columns;

        // Centers transposed to [column, cluster], the clusters padded to whole SIMD vectors: a row's dot products
        // with every center then accumulate a vector of clusters per column. A padding cluster has an infinite norm,
        // so it is never nearest.
        int width = SimdVector.Count;
        int vectors = (clusters + width - 1) / width;
        int padded = vectors * width;
        var centersByColumn = new double[columns * padded];
        var centerNorms = new double[padded];
        for (int k = 0; k < padded; k++)
        {
            if (k >= clusters)
            {
                centerNorms[k] = double.PositiveInfinity;
                continue;
            }

            centerNorms[k] = Dot(centers, k * columns, centers, k * columns, columns);
            for (int j = 0; j < columns; j++) centersByColumn[j * padded + k] = centers[k * columns + j];
        }

        bool accumulate = sums is not null && counts is not null;
        int blocks = BlockCount;
        var blockSums = accumulate ? new double[blocks][] : null;
        var blockCounts = accumulate ? new int[blocks][] : null;

        System.Threading.Tasks.Parallel.For(0, blocks, block =>
        {
            int start = block * BlockRows;
            int end = Math.Min(start + BlockRows, Rows);
            var localSums = accumulate ? new double[clusters * columns] : null;
            var localCounts = accumulate ? new int[clusters] : null;
            var dots = new double[padded];
            var dotVectors = MemoryMarshal.Cast<double, SimdVector>(new Span<double>(dots));
            var centerVectors = MemoryMarshal.Cast<double, SimdVector>(new ReadOnlySpan<double>(centersByColumn));

            for (int i = start; i < end; i++)
            {
                int offset = i * columns;
                RowDots(_x, offset, columns, centerVectors, vectors, dotVectors);

                int best = 0;
                double bestDistance = double.MaxValue;
                double rowNorm = _rowNorms[i];
                for (int k = 0; k < clusters; k++)
                {
                    double distance = rowNorm + centerNorms[k] - 2.0 * dots[k];
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        best = k;
                    }
                }

                labels[i] = best;
                if (localSums is not null && localCounts is not null)
                {
                    localCounts[best]++;
                    int target = best * columns;
                    for (int j = 0; j < columns; j++)
                        localSums[target + j] += _x[offset + j];
                }
            }

            if (blockSums is not null && localSums is not null) blockSums[block] = localSums;
            if (blockCounts is not null && localCounts is not null) blockCounts[block] = localCounts;
        });

        if (sums is not null && counts is not null && blockSums is not null && blockCounts is not null)
        {
            Array.Clear(sums, 0, sums.Length);
            Array.Clear(counts, 0, counts.Length);
            for (int block = 0; block < blocks; block++)
            {
                var localSums = blockSums[block];
                var localCounts = blockCounts[block];
                for (int m = 0; m < sums.Length; m++) sums[m] += localSums[m];
                for (int k = 0; k < clusters; k++) counts[k] += localCounts[k];
            }
        }
    }

    /// <summary>
    /// One row's dot products with every center: <paramref name="centerVectors"/> holds the centers by column, each
    /// column <paramref name="vectors"/> SIMD vectors of clusters. Up to two vectors of clusters accumulate in
    /// registers rather than through memory.
    /// </summary>
    private static void RowDots(double[] x, int offset, int columns, ReadOnlySpan<SimdVector> centerVectors,
        int vectors, Span<SimdVector> dots)
    {
        if (vectors == 1)
        {
            var a0 = SimdVector.Zero;
            for (int j = 0; j < columns; j++) a0 += new SimdVector(x[offset + j]) * centerVectors[j];
            dots[0] = a0;
        }
        else if (vectors == 2)
        {
            var a0 = SimdVector.Zero;
            var a1 = SimdVector.Zero;
            for (int j = 0; j < columns; j++)
            {
                var value = new SimdVector(x[offset + j]);
                a0 += value * centerVectors[2 * j];
                a1 += value * centerVectors[2 * j + 1];
            }

            dots[0] = a0;
            dots[1] = a1;
        }
        else
        {
            dots.Clear();
            for (int j = 0; j < columns; j++)
            {
                var value = new SimdVector(x[offset + j]);
                int at = j * vectors;
                for (int v = 0; v < vectors; v++) dots[v] += value * centerVectors[at + v];
            }
        }
    }

    /// <summary>The sum of squared distances from each row to the center its label names, computed directly.</summary>
    public double Inertia(double[] centers, int[] labels)
    {
        int columns = Columns;
        int blocks = BlockCount;
        var partial = new double[blocks];
        System.Threading.Tasks.Parallel.For(0, blocks, block =>
        {
            int start = block * BlockRows;
            int end = Math.Min(start + BlockRows, Rows);
            double total = 0;
            for (int i = start; i < end; i++)
                total += SquaredDistance(_x, i * columns, centers, labels[i] * columns, columns);
            partial[block] = total;
        });

        double inertia = 0;
        for (int block = 0; block < blocks; block++) inertia += partial[block];
        return inertia;
    }

    /// <summary>Lowers each row's entry of <paramref name="minDistances"/> to its squared distance to one center.</summary>
    public void UpdateMinSquaredDistances(double[] centers, int center, double[] minDistances)
    {
        int columns = Columns;
        int offset = center * columns;
        System.Threading.Tasks.Parallel.For(0, BlockCount, block =>
        {
            int start = block * BlockRows;
            int end = Math.Min(start + BlockRows, Rows);
            for (int i = start; i < end; i++)
            {
                double distance = SquaredDistance(_x, i * columns, centers, offset, columns);
                if (distance < minDistances[i]) minDistances[i] = distance;
            }
        });
    }

    /// <summary>Each row's Euclidean distance to every center, row-major <c>[rows, clusters]</c>.</summary>
    public double[] Distances(double[] centers, int clusters)
    {
        int columns = Columns;
        var result = new double[Rows * clusters];
        System.Threading.Tasks.Parallel.For(0, BlockCount, block =>
        {
            int start = block * BlockRows;
            int end = Math.Min(start + BlockRows, Rows);
            for (int i = start; i < end; i++)
                for (int k = 0; k < clusters; k++)
                    result[i * clusters + k] = Math.Sqrt(SquaredDistance(_x, i * columns, centers, k * columns, columns));
        });
        return result;
    }

    /// <summary>The squared Euclidean distance between two rows of length <paramref name="length"/>.</summary>
    internal static double SquaredDistance(double[] a, int aOffset, double[] b, int bOffset, int length)
    {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int j = 0;
        for (; j + 4 <= length; j += 4)
        {
            double d0 = a[aOffset + j] - b[bOffset + j];
            double d1 = a[aOffset + j + 1] - b[bOffset + j + 1];
            double d2 = a[aOffset + j + 2] - b[bOffset + j + 2];
            double d3 = a[aOffset + j + 3] - b[bOffset + j + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
        }

        for (; j < length; j++)
        {
            double d = a[aOffset + j] - b[bOffset + j];
            s0 += d * d;
        }

        return (s0 + s1) + (s2 + s3);
    }

    private static double Dot(double[] a, int aOffset, double[] b, int bOffset, int length)
    {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int j = 0;
        for (; j + 4 <= length; j += 4)
        {
            s0 += a[aOffset + j] * b[bOffset + j];
            s1 += a[aOffset + j + 1] * b[bOffset + j + 1];
            s2 += a[aOffset + j + 2] * b[bOffset + j + 2];
            s3 += a[aOffset + j + 3] * b[bOffset + j + 3];
        }

        for (; j < length; j++)
            s0 += a[aOffset + j] * b[bOffset + j];

        return (s0 + s1) + (s2 + s3);
    }
}
