using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;

using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;

namespace AiDotNet.Rag.Benchmarks
{
    /// <summary>
    /// Deterministic, Stopwatch-based comparative benchmark for AiDotNet's
    /// RAG / ANN vector-search stack. CPU-only; no GPU and no external vector DB.
    /// </summary>
    internal static class Program
    {
        private const int Dim = 128;
        private const int K = 10;
        private const int QueryCount = 200;
        private const int LatencyRepeats = 3;

        private static readonly int[] CorpusSizes = { 1_000, 10_000 };

        private static void Main()
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

            Console.WriteLine("# AiDotNet RAG / ANN Benchmark");
            Console.WriteLine();
            PrintEnvironment();

            var annResults = new List<AnnResult>();
            foreach (int n in CorpusSizes)
                annResults.AddRange(RunAnnSuite(n));

            PrintAnnTables(annResults);

            var retrieval = RunRetrievalSuite();
            PrintRetrievalTable(retrieval);

            Console.WriteLine();
            Console.WriteLine("Benchmark complete.");
        }

        // ------------------------------------------------------------------
        // ANN recall / latency / build-time suite
        // ------------------------------------------------------------------

        private sealed record AnnResult(
            int N,
            string Index,
            double Recall,
            double BuildMs,
            double P50Micros,
            double P95Micros,
            double QueriesPerSec);

        private static List<AnnResult> RunAnnSuite(int n)
        {
            Console.Error.WriteLine($"[info] generating dataset N={n} ...");
            int clusterCount = Math.Max(8, (int)Math.Sqrt(n));
            var data = SyntheticData.Generate(n, Dim, clusterCount, QueryCount);
            var batch = SyntheticData.ToBatch(data.Corpus);

            // Ground truth: exact top-K neighbors per query, from a brute-force Flat index.
            Console.Error.WriteLine($"[info] computing brute-force ground truth N={n} ...");
            var groundTruthFlat = new FlatIndex<double>(new CosineSimilarityMetric<double>());
            groundTruthFlat.AddBatch(batch);
            var groundTruth = new List<HashSet<string>>(data.Queries.Length);
            foreach (var q in data.Queries)
                groundTruth.Add(groundTruthFlat.Search(q, K).Select(r => r.Id).ToHashSet());

            var results = new List<AnnResult>();

            // Index configurations. IVF/LSH scale their partitioning with N.
            int ivfClusters = Math.Max(4, (int)Math.Sqrt(n));
            int ivfProbes = Math.Max(2, ivfClusters / 8);

            var configs = new (string Name, Func<IVectorIndex<double>> Factory)[]
            {
                ("Flat (exact)", () => new FlatIndex<double>(new CosineSimilarityMetric<double>())),
                ($"IVF (nlist={ivfClusters}, nprobe={ivfProbes})",
                    () => new IVFIndex<double>(new CosineSimilarityMetric<double>(), ivfClusters, ivfProbes)),
                ("HNSW (M=16, efC=200, efS=50)",
                    () => new HNSWIndex<double>(new CosineSimilarityMetric<double>(), 16, 200, 50)),
                ("LSH (tables=10, bits=12)",
                    () => new LSHIndex<double>(new CosineSimilarityMetric<double>(), 10, 12)),
            };

            foreach (var (name, factory) in configs)
            {
                Console.Error.WriteLine($"[info] N={n} index={name} ...");
                results.Add(Measure(n, name, factory, batch, data.Queries, groundTruth));
            }

            return results;
        }

        private static AnnResult Measure(
            int n,
            string name,
            Func<IVectorIndex<double>> factory,
            Dictionary<string, Vector<double>> batch,
            Vector<double>[] queries,
            List<HashSet<string>> groundTruth)
        {
            var index = factory();

            // Build/train time = bulk insert + one warmup query (forces any lazy
            // structure, e.g. IVF's k-means clustering, to be built and timed).
            var sw = Stopwatch.StartNew();
            index.AddBatch(batch);
            _ = index.Search(queries[0], K);
            sw.Stop();
            double buildMs = sw.Elapsed.TotalMilliseconds;

            // Recall@K vs brute-force ground truth.
            double recallSum = 0.0;
            for (int i = 0; i < queries.Length; i++)
            {
                var approx = index.Search(queries[i], K).Select(r => r.Id).ToList();
                recallSum += IrMetrics.NeighborRecall(approx, groundTruth[i], K);
            }
            double recall = recallSum / queries.Length;

            // Latency: per-query wall time across repeated passes.
            var micros = new List<double>(queries.Length * LatencyRepeats);
            double totalSeconds = 0.0;
            for (int rep = 0; rep < LatencyRepeats; rep++)
            {
                foreach (var q in queries)
                {
                    long start = Stopwatch.GetTimestamp();
                    _ = index.Search(q, K);
                    long end = Stopwatch.GetTimestamp();
                    double sec = (double)(end - start) / Stopwatch.Frequency;
                    totalSeconds += sec;
                    micros.Add(sec * 1_000_000.0);
                }
            }

            double p50 = Stats.Percentile(micros, 50);
            double p95 = Stats.Percentile(micros, 95);
            double qps = micros.Count / totalSeconds;

            return new AnnResult(n, name, recall, buildMs, p50, p95, qps);
        }

        // ------------------------------------------------------------------
        // End-to-end retrieval suite (VectorDocumentStore + stub embedder)
        // ------------------------------------------------------------------

        private sealed record RetrievalResult(
            double RecallAt5,
            double RecallAt10,
            double Mrr,
            double Ndcg10,
            double P50Micros,
            double P95Micros,
            int DocCount,
            int QueryCount);

        private static RetrievalResult RunRetrievalSuite()
        {
            Console.Error.WriteLine("[info] end-to-end retrieval suite ...");
            var embedder = new StubEmbedder(Dim);
            var store = new InMemoryDocumentStore<double>(embedder.Dimension);

            var docs = LabeledCorpus.Documents();
            var vectorDocs = docs.Select(d => new VectorDocument<double>(
                new Document<double>(d.Id, d.Content),
                embedder.Embed(d.Content))).ToList();
            store.AddBatch(vectorDocs);

            var queries = LabeledCorpus.Queries();

            double recall5 = 0, recall10 = 0, mrr = 0, ndcg = 0;
            var micros = new List<double>(queries.Length * LatencyRepeats);

            foreach (var query in queries)
            {
                var qv = embedder.Embed(query.Text);
                var ranked = store.GetSimilar(qv, 10).Select(d => d.Id).ToList();

                recall5 += IrMetrics.RecallAtK(ranked, query.RelevantIds, 5);
                recall10 += IrMetrics.RecallAtK(ranked, query.RelevantIds, 10);
                mrr += IrMetrics.ReciprocalRank(ranked, query.RelevantIds);
                ndcg += IrMetrics.NdcgAtK(ranked, query.RelevantIds, 10);
            }

            for (int rep = 0; rep < LatencyRepeats; rep++)
            {
                foreach (var query in queries)
                {
                    var qv = embedder.Embed(query.Text);
                    long start = Stopwatch.GetTimestamp();
                    _ = store.GetSimilar(qv, 10).ToList();
                    long end = Stopwatch.GetTimestamp();
                    micros.Add((double)(end - start) / Stopwatch.Frequency * 1_000_000.0);
                }
            }

            int q = queries.Length;
            return new RetrievalResult(
                recall5 / q, recall10 / q, mrr / q, ndcg / q,
                Stats.Percentile(micros, 50), Stats.Percentile(micros, 95),
                docs.Length, q);
        }

        // ------------------------------------------------------------------
        // Reporting
        // ------------------------------------------------------------------

        private static void PrintEnvironment()
        {
            Console.WriteLine("## Environment");
            Console.WriteLine();
            Console.WriteLine($"- OS: {RuntimeInformation.OSDescription} ({RuntimeInformation.OSArchitecture})");
            Console.WriteLine($"- CPU: {Environment.ProcessorCount} logical cores, arch {RuntimeInformation.ProcessArchitecture}");
            Console.WriteLine($"- Runtime: {RuntimeInformation.FrameworkDescription}");
            Console.WriteLine($"- Server GC: {System.Runtime.GCSettings.IsServerGC}");
            Console.WriteLine($"- Config: dim={Dim}, k={K}, queries={QueryCount}, latency repeats={LatencyRepeats}");
            Console.WriteLine("- Note: CPU-only. No GPU and no external vector DB are exercised.");
            Console.WriteLine();
        }

        private static void PrintAnnTables(List<AnnResult> results)
        {
            Console.WriteLine("## ANN index results");
            Console.WriteLine();
            foreach (int n in CorpusSizes)
            {
                Console.WriteLine($"### N = {n:N0} vectors, dim = {Dim}");
                Console.WriteLine();
                Console.WriteLine("| Index | Recall@10 | Build+train (ms) | p50 (us) | p95 (us) | Throughput (q/s) |");
                Console.WriteLine("|-------|-----------|------------------|----------|----------|------------------|");
                foreach (var r in results.Where(x => x.N == n))
                {
                    Console.WriteLine(
                        $"| {r.Index} | {r.Recall:F3} | {r.BuildMs:F1} | {r.P50Micros:F1} | {r.P95Micros:F1} | {r.QueriesPerSec:N0} |");
                }
                Console.WriteLine();
            }
        }

        private static void PrintRetrievalTable(RetrievalResult r)
        {
            Console.WriteLine("## End-to-end retrieval (InMemoryDocumentStore, HNSW-backed, stub embedder)");
            Console.WriteLine();
            Console.WriteLine($"Corpus: {r.DocCount} labeled docs, {r.QueryCount} labeled queries (5 topics).");
            Console.WriteLine();
            Console.WriteLine("| Recall@5 | Recall@10 | MRR | nDCG@10 | p50 (us) | p95 (us) |");
            Console.WriteLine("|----------|-----------|-----|---------|----------|----------|");
            Console.WriteLine(
                $"| {r.RecallAt5:F3} | {r.RecallAt10:F3} | {r.Mrr:F3} | {r.Ndcg10:F3} | {r.P50Micros:F1} | {r.P95Micros:F1} |");
            Console.WriteLine();
        }
    }
}
