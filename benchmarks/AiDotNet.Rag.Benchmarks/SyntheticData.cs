using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Rag.Benchmarks
{
    /// <summary>
    /// Deterministic, seeded generator for clustered dense embeddings.
    /// </summary>
    /// <remarks>
    /// Produces a corpus of <c>N</c> vectors drawn from a small set of Gaussian
    /// clusters plus a disjoint set of query vectors drawn from the same clusters.
    /// Everything is seeded, so a given (seed, dim, N) tuple always yields identical
    /// data across runs and machines.
    /// </remarks>
    internal static class SyntheticData
    {
        /// <summary>A single generated dataset: the indexed corpus and the query set.</summary>
        internal sealed class Dataset
        {
            public required (string Id, Vector<double> Vector)[] Corpus { get; init; }
            public required Vector<double>[] Queries { get; init; }
            public int Dimension { get; init; }
            public int ClusterCount { get; init; }
        }

        /// <summary>
        /// Generates a clustered dataset.
        /// </summary>
        /// <param name="n">Number of corpus vectors.</param>
        /// <param name="dim">Vector dimension.</param>
        /// <param name="clusterCount">Number of Gaussian cluster centers.</param>
        /// <param name="queryCount">Number of query vectors (disjoint samples from the same clusters).</param>
        /// <param name="noise">Per-component standard deviation of the cluster noise.</param>
        /// <param name="seed">Master seed.</param>
        internal static Dataset Generate(
            int n,
            int dim,
            int clusterCount,
            int queryCount,
            double noise = 0.35,
            int seed = 20240719)
        {
            var rng = new Random(seed);

            // Cluster centers: each component ~ N(0, 1), spread across the space.
            var centers = new double[clusterCount][];
            for (int c = 0; c < clusterCount; c++)
            {
                var center = new double[dim];
                for (int d = 0; d < dim; d++)
                    center[d] = Gaussian(rng);
                centers[c] = center;
            }

            var corpus = new (string, Vector<double>)[n];
            for (int i = 0; i < n; i++)
            {
                int c = i % clusterCount; // deterministic round-robin cluster assignment
                corpus[i] = ($"doc-{i}", Sample(rng, centers[c], noise, dim));
            }

            var queries = new Vector<double>[queryCount];
            for (int q = 0; q < queryCount; q++)
            {
                int c = rng.Next(clusterCount);
                queries[q] = Sample(rng, centers[c], noise, dim);
            }

            return new Dataset
            {
                Corpus = corpus,
                Queries = queries,
                Dimension = dim,
                ClusterCount = clusterCount,
            };
        }

        private static Vector<double> Sample(Random rng, double[] center, double noise, int dim)
        {
            var data = new double[dim];
            for (int d = 0; d < dim; d++)
                data[d] = center[d] + noise * Gaussian(rng);
            return new Vector<double>(data);
        }

        /// <summary>Standard-normal sample via Box-Muller (seeded through <paramref name="rng"/>).</summary>
        private static double Gaussian(Random rng)
        {
            // u1 must be > 0 for Log; Random.NextDouble() is in [0,1).
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        /// <summary>Builds an id -&gt; vector batch dictionary for <c>AddBatch</c>.</summary>
        internal static Dictionary<string, Vector<double>> ToBatch((string Id, Vector<double> Vector)[] corpus)
        {
            var batch = new Dictionary<string, Vector<double>>(corpus.Length);
            foreach (var (id, vector) in corpus)
                batch[id] = vector;
            return batch;
        }
    }
}
