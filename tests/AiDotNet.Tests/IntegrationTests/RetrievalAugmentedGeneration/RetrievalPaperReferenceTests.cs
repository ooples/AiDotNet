using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Rerankers and retriever score aggregation checked against the papers they implement.
/// </summary>
public class RetrievalPaperReferenceTests
{
    private const double Tolerance = 1e-12;

    /// <summary>
    /// Carbonell and Goldstein 1998: MMR = argmax[λ rel_i - (1 - λ) max_j sim(i, j)] uses the true maximum
    /// similarity to the selected set. With λ = 0.5 and A selected first, B (rel 0.5, cos -1) scores
    /// 0.25 + 0.5 = 0.75 and C (rel 0.6, cos -0.2) scores 0.3 + 0.1 = 0.4, so B is next. Master clamped the
    /// maximum at 0, scored B 0.25 and C 0.3, and picked C.
    /// </summary>
    [Fact]
    public void Mmr_UsesTrueMaximumSimilarity_ForNegativeCosines()
    {
        var embeddings = new Dictionary<string, Vector<double>>
        {
            ["A"] = new Vector<double>(new[] { 1.0, 0.0 }),
            ["B"] = new Vector<double>(new[] { -1.0, 0.0 }),
            ["C"] = new Vector<double>(new[] { -0.2, Math.Sqrt(1.0 - 0.04) }),
        };
        var docs = new[] { Scored("A", 1.0), Scored("B", 0.5), Scored("C", 0.6) };
        var reranker = new MaximalMarginalRelevanceReranker<double>(d => embeddings[d.Id], lambda: 0.5);

        var order = reranker.Rerank("q", docs).Select(d => d.Id).ToArray();

        Assert.Equal(new[] { "A", "B", "C" }, order);
    }

    /// <summary>
    /// Cormack, Clarke and Buettcher 2009: RRF(d) = Σ 1 / (k + r(d)). With k = 60, "a" at ranks 1 and 2
    /// scores 1/61 + 1/62 and "b" at ranks 2 and 1 the same; "c" at rank 3 once scores 1/63.
    /// Fused documents carry their score (master left HasRelevanceScore false).
    /// </summary>
    [Fact]
    public void Rrf_FusedScores_MatchCormack2009()
    {
        var rrf = new ReciprocalRankFusion<double>();
        var fused = rrf.FuseRankings(new List<List<Document<double>>>
        {
            new() { new Document<double>("a", "a"), new Document<double>("b", "b"), new Document<double>("c", "c") },
            new() { new Document<double>("b", "b"), new Document<double>("a", "a") },
        }, topK: 3);

        Assert.Equal(new[] { "a", "b", "c" }, fused.Select(d => d.Id).ToArray());
        Assert.Equal(1.0 / 61 + 1.0 / 62, fused[0].RelevanceScore, Tolerance);
        Assert.Equal(1.0 / 61 + 1.0 / 62, fused[1].RelevanceScore, Tolerance);
        Assert.Equal(1.0 / 63, fused[2].RelevanceScore, Tolerance);
        Assert.All(fused, d => Assert.True(d.HasRelevanceScore));
    }

    /// <summary>RRF is generic in T: master sorted through Convert.ToDouble, which throws for Half.</summary>
    [Fact]
    public void Rrf_Rerank_WorksForHalf()
    {
        var rrf = new ReciprocalRankFusion<Half>();
        var docs = new[] { new Document<Half>("a", "a"), new Document<Half>("b", "b") };

        var reranked = rrf.Rerank("q", docs).ToList();

        Assert.Equal(new[] { "a", "b" }, reranked.Select(d => d.Id).ToArray());
        Assert.Equal((Half)(1.0 / 61), reranked[0].RelevanceScore);
    }

    /// <summary>
    /// Weighted multi-vector aggregation weighs vector N of "docId_vector_N" by 1 / (N + 1). The store returns
    /// vectors in similarity order (vector 1 at 0.8 before vector 0 at 0.2), so the score is
    /// (0.2 * 1 + 0.8 * 0.5) / 1.5 = 0.4. Master weighted by arrival order and returned 0.6.
    /// </summary>
    [Fact]
    public void MultiVector_WeightedAggregation_WeighsByVectorPosition()
    {
        var scores = new List<(int position, double score)> { (1, 0.8), (0, 0.2) };

        Assert.Equal(0.4, MultiVectorRetriever<double>.AggregateScores(scores, MultiVectorAggregation.Weighted), Tolerance);
        Assert.Equal(0.8, MultiVectorRetriever<double>.AggregateScores(scores, MultiVectorAggregation.Max), Tolerance);
        Assert.Equal(0.5, MultiVectorRetriever<double>.AggregateScores(scores, MultiVectorAggregation.Mean), Tolerance);
    }

    /// <summary>
    /// Okapi BM25 (Robertson and Zaragoza 2009, Eq. 3.15) with k1 = 1.5, b = 0.75 and the Lucene idf
    /// ln(1 + (N - df + 0.5) / (df + 0.5)) on "a b", "a a c", "c" for query "c": N = 3, df = 2, avgdl = 5/3,
    /// idf = ln(1 + 1.5 / 2.5). The idf without the 1 + is ln(0.6) &lt; 0 here, which scored both documents
    /// containing "c" below "a b" (score 0); they must rank first.
    /// </summary>
    [Fact]
    public void Bm25_Scores_MatchRobertsonZaragozaWithLuceneIdf()
    {
        var retriever = new BM25Retriever<double>(Store<double>("a b", "a a c", "c"), k1: 1.5, b: 0.75);

        var ranked = retriever.Retrieve("c", topK: 3).ToList();
        var results = ranked.ToDictionary(d => d.Id, d => d.RelevanceScore);

        Assert.Equal(new[] { "d2", "d1", "d0" }, ranked.Select(d => d.Id).ToArray());
        double idf = Math.Log(1.0 + 1.5 / 2.5);
        double avgdl = 5.0 / 3.0;
        double Term(double tf, double dl) => idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * dl / avgdl));
        Assert.Equal(Term(1, 3), results["d1"], Tolerance);
        Assert.Equal(Term(1, 1), results["d2"], Tolerance);
        Assert.Equal(0.0, results["d0"], Tolerance);
    }

    /// <summary>k1 must be non-negative and b in [0, 1] (Robertson and Zaragoza 2009, Sec. 3.4).</summary>
    [Fact]
    public void Bm25_RejectsOutOfRangeParameters()
    {
        var store = Store<double>("a");

        Assert.Throws<ArgumentOutOfRangeException>(() => new BM25Retriever<double>(store, k1: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BM25Retriever<double>(store, b: 1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BM25Retriever<double>(store, b: -0.5));
    }

    /// <summary>BM25 and TF-IDF are generic in T: master computed the idf / ranking through Convert.ToDouble,
    /// which throws for Half.</summary>
    [Fact]
    public void SparseRetrievers_WorkForHalf()
    {
        var store = Store<Half>("a b", "a a c", "c");

        var bm25 = new BM25Retriever<Half>(store).Retrieve("b", topK: 3).ToList();
        var tfidf = new TFIDFRetriever<Half>(store).Retrieve("b", topK: 3).ToList();

        Assert.Equal("d0", bm25[0].Id);
        Assert.Equal("d0", tfidf[0].Id);

        // tf = 1 / max count 1, idf = ln(3 / 1)
        Assert.Equal((Half)Math.Log(3.0), tfidf[0].RelevanceScore);
    }

    /// <summary>
    /// The TF-IDF statistics are cached per corpus. Replacing a document without changing the count must
    /// rebuild them: the cache used to be keyed on the count alone and kept scoring the old text, so
    /// "cherry" matched nothing after d1 became "cherry".
    /// </summary>
    [Fact]
    public void Tfidf_ReplacedDocument_InvalidatesCachedStatistics()
    {
        var store = Store<double>("apple", "banana");
        var retriever = new TFIDFRetriever<double>(store);
        Assert.Equal(0.0, retriever.Retrieve("cherry", topK: 2).Max(d => d.RelevanceScore), Tolerance);

        store.Clear();
        store.Add(new VectorDocument<double>(new Document<double>("d0", "apple"), new Vector<double>(1)));
        store.Add(new VectorDocument<double>(new Document<double>("d1", "cherry"), new Vector<double>(1)));

        var results = retriever.Retrieve("cherry", topK: 2).ToList();

        // tf = 1 / max count 1, idf = ln(2 / 1)
        Assert.Equal("d1", results[0].Id);
        Assert.Equal(Math.Log(2.0), results[0].RelevanceScore, Tolerance);
    }

    /// <summary>
    /// ChromaDB hits score 1 / (1 + distance). A hit with a null or missing distance used to score as distance
    /// 0, i.e. a perfect match with relevance 1; it must be rejected as a malformed response.
    /// </summary>
    [Fact]
    public void ChromaDb_QueryResponse_RejectsMissingDistances()
    {
        var valid = JObject.Parse(
            "{\"ids\":[[\"a\",\"b\"]],\"documents\":[[\"x\",\"y\"]],\"distances\":[[0.25,3]]}");
        var hits = ChromaDBDocumentStore<double>.ParseQueryResponse(valid);
        Assert.Equal(1.0 / 1.25, hits[0].RelevanceScore, Tolerance);
        Assert.Equal(1.0 / 4.0, hits[1].RelevanceScore, Tolerance);

        var nullDistance = JObject.Parse(
            "{\"ids\":[[\"a\",\"b\"]],\"documents\":[[\"x\",\"y\"]],\"distances\":[[0.25,null]]}");
        Assert.Throws<InvalidOperationException>(() => ChromaDBDocumentStore<double>.ParseQueryResponse(nullDistance));

        var noDistances = JObject.Parse("{\"ids\":[[\"a\"]],\"documents\":[[\"x\"]]}");
        Assert.Throws<InvalidOperationException>(() => ChromaDBDocumentStore<double>.ParseQueryResponse(noDistances));

        var shortDistances = JObject.Parse(
            "{\"ids\":[[\"a\",\"b\"]],\"documents\":[[\"x\",\"y\"]],\"distances\":[[0.25]]}");
        Assert.Throws<InvalidOperationException>(() => ChromaDBDocumentStore<double>.ParseQueryResponse(shortDistances));
    }

    /// <summary>
    /// Temperature sampling uses softmax(z / τ). With τ = 0.001 the Half logits 70 and 69 scale to 70000 and
    /// 69000, beyond the Half maximum 65504; dividing in Half made both +inf and the softmax NaN. In double the
    /// distribution is [1, e^-1000] = [1, 0].
    /// </summary>
    [Fact]
    public void NeuralGenerator_TemperatureSoftmax_DoesNotOverflowForHalf()
    {
        var probabilities = NeuralGenerator<Half>.ApplyTemperatureAndSoftmax(new[] { (Half)70, (Half)69 }, 0.001);

        Assert.Equal(1.0, probabilities[0], Tolerance);
        Assert.Equal(0.0, probabilities[1], Tolerance);

        // A moderate temperature still matches softmax([1, 2, 3] / 0.5) computed directly.
        var moderate = NeuralGenerator<Half>.ApplyTemperatureAndSoftmax(new[] { (Half)1, (Half)2, (Half)3 }, 0.5);
        double z = Math.Exp(2) + Math.Exp(4) + Math.Exp(6);
        Assert.Equal(Math.Exp(2) / z, moderate[0], 1e-9);
        Assert.Equal(Math.Exp(6) / z, moderate[2], 1e-9);
    }

    /// <summary>
    /// RRF over two disjoint 50,000-document rankings. The fused scores are a per-document sum, which must stay
    /// linear in the hits: the one-hot [occurrences x documents] membership used for it here had 10^10 entries
    /// and could not be allocated.
    /// </summary>
    [Fact]
    public void Rrf_LargeDisjointRankings_FuseWithLinearMemory()
    {
        const int perList = 50000;
        var lists = new List<List<Document<double>>>();
        for (int l = 0; l < 2; l++)
        {
            var list = new List<Document<double>>(perList);
            for (int r = 0; r < perList; r++)
                list.Add(new Document<double>($"l{l}-{r}", "x"));
            lists.Add(list);
        }

        var fused = new ReciprocalRankFusion<double>().FuseRankings(lists, topK: 2 * perList);

        Assert.Equal(2 * perList, fused.Count);
        Assert.Equal(1.0 / 61, fused[0].RelevanceScore, Tolerance);
        Assert.Equal(1.0 / 61, fused[1].RelevanceScore, Tolerance);
        Assert.Equal(1.0 / (60 + perList), fused[fused.Count - 1].RelevanceScore, Tolerance);
    }

    /// <summary>Returns one shared hit (score 0.25) plus a disjoint block of hits (score 0.5) per call.</summary>
    private sealed class DisjointBlockRetriever : IRetriever<double>
    {
        private readonly int _perCall;

        public DisjointBlockRetriever(int perCall) => _perCall = perCall;

        public int Calls { get; private set; }
        public int DefaultTopK => _perCall + 1;
        public IEnumerable<Document<double>> Retrieve(string query) => Retrieve(query, DefaultTopK);
        public IEnumerable<Document<double>> Retrieve(string query, int topK) => Retrieve(query, topK, new Dictionary<string, object>());

        public IEnumerable<Document<double>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            int call = Calls++;
            var hits = new List<Document<double>>(_perCall + 1)
            {
                new Document<double>("shared", "x") { RelevanceScore = 0.25, HasRelevanceScore = true }
            };
            for (int i = 0; i < _perCall; i++)
                hits.Add(new Document<double>($"q{call}-{i}", "x") { RelevanceScore = 0.5, HasRelevanceScore = true });
            return hits;
        }

        public System.Threading.Tasks.Task<IEnumerable<Document<double>>> RetrieveAsync(
            string query, int topK, Dictionary<string, object>? metadataFilters = null,
            System.Threading.CancellationToken cancellationToken = default)
            => System.Threading.Tasks.Task.FromResult(Retrieve(query, topK, metadataFilters ?? new Dictionary<string, object>()));
    }

    /// <summary>
    /// Multi-query CombSUM over disjoint 50,000-hit result lists: each document's score is the sum of its hit
    /// scores, which must stay linear in the hits. The one-hot [hits x documents] membership used for the sum
    /// had at least 10^10 entries here and could not be allocated.
    /// </summary>
    [Fact]
    public void MultiQuery_LargeDisjointResults_SumScoresWithLinearMemory()
    {
        const int perCall = 50000;
        var baseRetriever = new DisjointBlockRetriever(perCall);
        var retriever = new MultiQueryRetriever<double>(baseRetriever, numQueries: 3);

        var results = retriever.Retrieve("query", topK: perCall + 1).ToList();

        Assert.True(baseRetriever.Calls >= 2);
        Assert.Equal(perCall + 1, results.Count);
        Assert.Equal("shared", results[0].Id);
        Assert.Equal(0.25 * baseRetriever.Calls, results[0].RelevanceScore, Tolerance);
        Assert.Equal(0.5, results[1].RelevanceScore, Tolerance);
    }

    /// <summary>
    /// BM25 term saturation tf (k1 + 1) / (tf + k1 (1 - b + b dl / avgdl)) is 0 for an absent term. With k1 = 0,
    /// or b = 1 on an empty document, the denominator is 0 too, and the 0 / 0 made the whole score NaN.
    /// k1 = 0 reduces BM25 to Σ idf over the present terms (Robertson and Zaragoza 2009, Sec. 3.4).
    /// </summary>
    [Fact]
    public void Bm25_AbsentTerms_DoNotProduceNaN()
    {
        var store = Store<double>("a");
        double idf = Math.Log(1.0 + 1.5 / 1.5); // N = 2, df = 1

        var binary = new BM25Retriever<double>(store, k1: 0, b: 0.75).ScoreDocuments(
            new List<string> { "a", "c" }, new List<List<string>> { new() { "a", "b" }, new() { "c" } });
        Assert.Equal(idf, binary[0], Tolerance);
        Assert.Equal(idf, binary[1], Tolerance);

        var emptyDocument = new BM25Retriever<double>(store, k1: 1.5, b: 1).ScoreDocuments(
            new List<string> { "a" }, new List<List<string>> { new() { "a" }, new() });
        double avgdl = 0.5;
        Assert.Equal(idf * 2.5 / (1 + 1.5 * (1.0 / avgdl)), emptyDocument[0], Tolerance);
        Assert.Equal(0.0, emptyDocument[1], Tolerance);
    }

    private static InMemoryDocumentStore<TNum> Store<TNum>(params string[] contents)
    {
        var store = new InMemoryDocumentStore<TNum>(1);
        for (int i = 0; i < contents.Length; i++)
        {
            store.Add(new VectorDocument<TNum>(new Document<TNum>("d" + i, contents[i]), new Vector<TNum>(1)));
        }

        return store;
    }

    private static Document<double> Scored(string id, double score)
        => new Document<double>(id, id) { RelevanceScore = score, HasRelevanceScore = true };
}
