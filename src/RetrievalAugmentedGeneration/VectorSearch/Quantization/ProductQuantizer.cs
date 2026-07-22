using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization
{
    /// <summary>
    /// Product Quantizer (PQ) that splits each vector into <c>M</c> contiguous subspaces,
    /// learns a small k-means codebook (default 256 centroids) per subspace, and encodes a
    /// vector as one byte per subspace. Supports Asymmetric Distance Computation (ADC) for
    /// fast, memory-light nearest-neighbor ranking.
    /// </summary>
    /// <remarks>
    /// <para>
    /// PQ is the headline compression technique behind FAISS <c>IndexPQ</c>/<c>IndexIVFPQ</c>.
    /// A <c>D</c>-dimensional float vector (4*D bytes) is reduced to just <c>M</c> bytes.
    /// With the common choice of 256 centroids per subspace, each subspace index fits in a
    /// single byte, so an embedding of dimension 768 encoded with M=96 needs only 96 bytes
    /// versus 3072 bytes raw (32x reduction), or 768 bytes as SQ8.
    /// </para>
    /// <para>
    /// <b>Asymmetric Distance Computation (ADC):</b> at query time we do NOT quantize the
    /// query. Instead, for each subspace we precompute the squared L2 distance from the raw
    /// query subvector to every centroid of that subspace, building an
    /// <c>M x Ksub</c> lookup table. The (approximate) squared distance between the query and
    /// any stored code is then just the sum of <c>M</c> table look-ups. This equals the exact
    /// squared L2 distance between the raw query and the reconstructed database vector, which
    /// is why ADC is both fast and accurate.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> We chop each vector into M pieces. For each piece position we
    /// learn 256 "typical pieces" (centroids). A vector is then described by which typical
    /// piece is closest in each position - just M small numbers. To compare a query we
    /// measure the query against those 256 typical pieces once, then adding up the right
    /// numbers gives the distance to any stored vector almost for free.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class ProductQuantizer<T> : IVectorQuantizer<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly int _subspaceCount;          // M
        private readonly int _centroidsPerSubspace;   // Ksub (<= 256)
        private readonly int _maxIterations;
        private readonly int _seed;

        private int _dimension;
        private int _subDimension;                    // Dsub = D / M
        // _codebooks[m][c] is the centroid c of subspace m, length _subDimension.
        private double[][][]? _codebooks;

        /// <summary>
        /// Initializes a new instance of the <see cref="ProductQuantizer{T}"/> class.
        /// </summary>
        /// <param name="subspaceCount">Number of subspaces (M). The vector dimension must be divisible by this. Default: 8.</param>
        /// <param name="centroidsPerSubspace">Number of k-means centroids per subspace (Ksub, 1-256). Default: 256.</param>
        /// <param name="maxIterations">Maximum Lloyd's k-means iterations per subspace. Default: 25.</param>
        /// <param name="seed">Random seed for deterministic k-means initialization. Default: 42.</param>
        public ProductQuantizer(int subspaceCount = 8, int centroidsPerSubspace = 256, int maxIterations = 25, int seed = 42)
        {
            if (subspaceCount <= 0)
                throw new ArgumentException("Subspace count (M) must be positive.", nameof(subspaceCount));
            if (centroidsPerSubspace <= 0 || centroidsPerSubspace > 256)
                throw new ArgumentException("Centroids per subspace (Ksub) must be between 1 and 256.", nameof(centroidsPerSubspace));
            if (maxIterations <= 0)
                throw new ArgumentException("Max iterations must be positive.", nameof(maxIterations));

            _numOps = MathHelper.GetNumericOperations<T>();
            _subspaceCount = subspaceCount;
            _centroidsPerSubspace = centroidsPerSubspace;
            _maxIterations = maxIterations;
            _seed = seed;
        }

        /// <inheritdoc/>
        public bool IsTrained => _codebooks != null;

        /// <inheritdoc/>
        public int Dimension => _dimension;

        /// <inheritdoc/>
        public int CodeLength => IsTrained ? _subspaceCount : 0;

        /// <summary>
        /// Gets the number of subspaces (M).
        /// </summary>
        public int SubspaceCount => _subspaceCount;

        /// <summary>
        /// Gets the number of centroids learned per subspace (Ksub).
        /// </summary>
        public int CentroidsPerSubspace => _centroidsPerSubspace;

        /// <summary>
        /// Gets the dimensionality of each subspace (Dsub = D / M), or 0 if untrained.
        /// </summary>
        public int SubDimension => _subDimension;

        /// <summary>
        /// Gets the learned codebooks with shape [M][Ksub][Dsub], or null if untrained.
        /// </summary>
        public double[][][]? Codebooks => _codebooks;

        /// <inheritdoc/>
        public void Train(IEnumerable<Vector<T>> vectors)
        {
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));

            // Materialize training data as double arrays.
            var data = new List<double[]>();
            int dim = -1;
            foreach (var vector in vectors)
            {
                if (vector == null)
                    throw new ArgumentException("Training set contains a null vector.", nameof(vectors));

                var arr = vector.ToArray();
                if (dim < 0)
                    dim = arr.Length;
                else if (arr.Length != dim)
                    throw new ArgumentException("All training vectors must have the same dimensionality.", nameof(vectors));

                var row = new double[arr.Length];
                for (int i = 0; i < arr.Length; i++)
                    row[i] = Convert.ToDouble(arr[i]);
                data.Add(row);
            }

            if (data.Count == 0)
                throw new ArgumentException("Training set must contain at least one vector.", nameof(vectors));
            if (dim % _subspaceCount != 0)
                throw new ArgumentException(
                    "Vector dimension (" + dim + ") must be divisible by the subspace count (" + _subspaceCount + ").",
                    nameof(vectors));

            _dimension = dim;
            _subDimension = dim / _subspaceCount;

            var codebooks = new double[_subspaceCount][][];
            for (int m = 0; m < _subspaceCount; m++)
            {
                int offset = m * _subDimension;
                var subvectors = new double[data.Count][];
                for (int n = 0; n < data.Count; n++)
                {
                    var sub = new double[_subDimension];
                    Array.Copy(data[n], offset, sub, 0, _subDimension);
                    subvectors[n] = sub;
                }

                // Deterministic per-subspace seed so results are reproducible and subspaces differ.
                codebooks[m] = RunKMeans(subvectors, _centroidsPerSubspace, _subDimension, _maxIterations, _seed + m);
            }

            _codebooks = codebooks;
        }

        /// <inheritdoc/>
        public byte[] Encode(Vector<T> vector)
        {
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));
            if (!IsTrained)
                throw new InvalidOperationException("ProductQuantizer must be trained before encoding.");

            var arr = vector.ToArray();
            if (arr.Length != _dimension)
                throw new ArgumentException("Vector dimensionality does not match the trained dimensionality.", nameof(vector));

            var full = new double[_dimension];
            for (int i = 0; i < _dimension; i++)
                full[i] = Convert.ToDouble(arr[i]);

            var code = new byte[_subspaceCount];
            var sub = new double[_subDimension];
            for (int m = 0; m < _subspaceCount; m++)
            {
                int offset = m * _subDimension;
                Array.Copy(full, offset, sub, 0, _subDimension);

                var centroids = _codebooks![m];
                int best = 0;
                double bestDist = double.MaxValue;
                for (int c = 0; c < centroids.Length; c++)
                {
                    double d = SquaredDistance(sub, centroids[c]);
                    if (d < bestDist)
                    {
                        bestDist = d;
                        best = c;
                    }
                }

                code[m] = (byte)best;
            }

            return code;
        }

        /// <inheritdoc/>
        public Vector<T> Decode(byte[] code)
        {
            if (code == null)
                throw new ArgumentNullException(nameof(code));
            if (!IsTrained)
                throw new InvalidOperationException("ProductQuantizer must be trained before decoding.");
            if (code.Length != _subspaceCount)
                throw new ArgumentException("Code length does not match the subspace count.", nameof(code));

            var values = new T[_dimension];
            for (int m = 0; m < _subspaceCount; m++)
            {
                int offset = m * _subDimension;
                var centroid = _codebooks![m][code[m]];
                for (int i = 0; i < _subDimension; i++)
                    values[offset + i] = _numOps.FromDouble(centroid[i]);
            }

            return new Vector<T>(values);
        }

        /// <summary>
        /// Builds the Asymmetric Distance Computation (ADC) lookup table for a query. Entry
        /// [m][c] holds the squared L2 distance between the query's subvector m and centroid c
        /// of subspace m.
        /// </summary>
        /// <param name="query">The raw (unquantized) query vector.</param>
        /// <returns>A table with shape [M][Ksub].</returns>
        public double[][] BuildDistanceTable(Vector<T> query)
        {
            if (query == null)
                throw new ArgumentNullException(nameof(query));
            if (!IsTrained)
                throw new InvalidOperationException("ProductQuantizer must be trained before building a distance table.");

            var arr = query.ToArray();
            if (arr.Length != _dimension)
                throw new ArgumentException("Query dimensionality does not match the trained dimensionality.", nameof(query));

            var full = new double[_dimension];
            for (int i = 0; i < _dimension; i++)
                full[i] = Convert.ToDouble(arr[i]);

            var table = new double[_subspaceCount][];
            var sub = new double[_subDimension];
            for (int m = 0; m < _subspaceCount; m++)
            {
                int offset = m * _subDimension;
                Array.Copy(full, offset, sub, 0, _subDimension);

                var centroids = _codebooks![m];
                var row = new double[centroids.Length];
                for (int c = 0; c < centroids.Length; c++)
                    row[c] = SquaredDistance(sub, centroids[c]);
                table[m] = row;
            }

            return table;
        }

        /// <summary>
        /// Computes the approximate squared L2 distance between a query (via its ADC table)
        /// and an encoded vector, as the sum of one table look-up per subspace.
        /// </summary>
        /// <param name="distanceTable">A table produced by <see cref="BuildDistanceTable"/>.</param>
        /// <param name="code">A code produced by <see cref="Encode"/>.</param>
        /// <returns>The approximate squared L2 distance.</returns>
        public double ComputeAsymmetricDistance(double[][] distanceTable, byte[] code)
        {
            if (distanceTable == null)
                throw new ArgumentNullException(nameof(distanceTable));
            if (code == null)
                throw new ArgumentNullException(nameof(code));
            if (code.Length != _subspaceCount)
                throw new ArgumentException("Code length does not match the subspace count.", nameof(code));

            double distance = 0.0;
            for (int m = 0; m < _subspaceCount; m++)
                distance += distanceTable[m][code[m]];

            return distance;
        }

        /// <summary>
        /// Runs Lloyd's k-means on the supplied subvectors and returns a codebook of exactly
        /// <paramref name="k"/> centroids (shape is always [k][dim] so callers can rely on it).
        /// </summary>
        private static double[][] RunKMeans(double[][] points, int k, int dim, int maxIterations, int seed)
        {
            int n = points.Length;
            var centroids = new double[k][];

            // Deterministic initialization: sample distinct points where possible.
            var random = RandomHelper.CreateSeededRandom(seed);
            var chosen = new HashSet<int>();
            int effectiveK = Math.Min(k, n);
            for (int c = 0; c < effectiveK; c++)
            {
                int idx;
                int guard = 0;
                do
                {
                    idx = random.Next(n);
                    guard++;
                }
                while (chosen.Contains(idx) && guard < n * 4);

                chosen.Add(idx);
                centroids[c] = (double[])points[idx].Clone();
            }

            // If we have fewer points than k, pad remaining centroids by duplicating existing
            // ones so the codebook always has the expected shape.
            for (int c = effectiveK; c < k; c++)
                centroids[c] = (double[])centroids[c % Math.Max(1, effectiveK)].Clone();

            if (n == 0)
            {
                for (int c = 0; c < k; c++)
                    centroids[c] = new double[dim];
                return centroids;
            }

            var assignments = new int[n];
            for (int i = 0; i < n; i++)
                assignments[i] = -1;

            for (int iter = 0; iter < maxIterations; iter++)
            {
                bool changed = false;

                // Assignment step.
                for (int i = 0; i < n; i++)
                {
                    int best = 0;
                    double bestDist = double.MaxValue;
                    for (int c = 0; c < k; c++)
                    {
                        double d = SquaredDistance(points[i], centroids[c]);
                        if (d < bestDist)
                        {
                            bestDist = d;
                            best = c;
                        }
                    }

                    if (assignments[i] != best)
                    {
                        assignments[i] = best;
                        changed = true;
                    }
                }

                // Update step.
                var sums = new double[k][];
                var counts = new int[k];
                for (int c = 0; c < k; c++)
                    sums[c] = new double[dim];

                for (int i = 0; i < n; i++)
                {
                    int c = assignments[i];
                    var p = points[i];
                    var s = sums[c];
                    for (int d = 0; d < dim; d++)
                        s[d] += p[d];
                    counts[c]++;
                }

                for (int c = 0; c < k; c++)
                {
                    if (counts[c] == 0)
                        continue; // Keep the previous centroid for empty clusters.

                    var s = sums[c];
                    var newCentroid = new double[dim];
                    for (int d = 0; d < dim; d++)
                        newCentroid[d] = s[d] / counts[c];
                    centroids[c] = newCentroid;
                }

                if (!changed && iter > 0)
                    break; // Converged.
            }

            return centroids;
        }

        private static double SquaredDistance(double[] a, double[] b)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Length; i++)
            {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }

            return sum;
        }
    }
}
