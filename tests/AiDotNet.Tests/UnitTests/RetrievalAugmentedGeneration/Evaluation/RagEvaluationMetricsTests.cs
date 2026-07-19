using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Evaluation;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Verifies the upgraded RAG evaluation metrics: pure-math IR metrics with exact expected values,
/// embedding-similarity metrics driven by a controlled fake embedding model, and LLM-judge metrics
/// driven by a scripted fake text generator. All tests are offline (no network / real model).
/// </summary>
public class RagEvaluationMetricsTests
{
    // ---- Test doubles -------------------------------------------------------

    /// <summary>Records prompts and returns a scripted response.</summary>
    private sealed class FakeTextGen : ITextGenerator
    {
        private readonly Func<string, string> _responder;
        public List<string> Prompts { get; } = new();
        public FakeTextGen(Func<string, string> responder) => _responder = responder;
        public string Generate(string prompt)
        {
            Prompts.Add(prompt);
            return _responder(prompt);
        }
    }

    /// <summary>Returns controlled vectors from an exact string-to-vector map (dimension 2).</summary>
    private sealed class FakeEmbeddingModel : IEmbeddingModel<double>
    {
        private readonly Dictionary<string, double[]> _map;
        public FakeEmbeddingModel(Dictionary<string, double[]> map) => _map = map;
        public int EmbeddingDimension => 2;
        public int MaxTokens => 512;
        public Vector<double> Embed(string text)
        {
            if (text != null && _map.TryGetValue(text, out var v))
                return new Vector<double>(v);
            // Deterministic non-zero default so cosine never divides by zero for unmapped text.
            return new Vector<double>(new[] { 1.0, 0.0 });
        }
        public Task<Vector<double>> EmbedAsync(string text) => Task.FromResult(Embed(text));
        public Matrix<double> EmbedBatch(IEnumerable<string> texts) => throw new NotSupportedException();
        public Task<Matrix<double>> EmbedBatchAsync(IEnumerable<string> texts) => throw new NotSupportedException();
    }

    private static GroundedAnswer<double> Answer(string query, string answer, params Document<double>[] docs)
    {
        return new GroundedAnswer<double>
        {
            Query = query,
            Answer = answer,
            SourceDocuments = docs.ToList().AsReadOnly()
        };
    }

    private static IReadOnlyList<string> Ranked(params string[] ids) => ids;
    private static ISet<string> RelevantSet(params string[] ids) => new HashSet<string>(ids);

    // ---- IR metrics (exact numeric assertions) ------------------------------

    [Fact]
    public void PrecisionAtK_KnownRanking()
    {
        var ranked = Ranked("d1", "d2", "d3", "d4"); // relevant at ranks 1 and 3
        var rel = RelevantSet("d1", "d3");

        Assert.Equal(1.0, RetrievalMetrics.PrecisionAtK(ranked, rel, 1), 6);
        Assert.Equal(0.5, RetrievalMetrics.PrecisionAtK(ranked, rel, 2), 6);
        Assert.Equal(0.5, RetrievalMetrics.PrecisionAtK(ranked, rel, 4), 6);
    }

    [Fact]
    public void RecallAtK_KnownRanking()
    {
        var ranked = Ranked("d1", "d2", "d3", "d4");
        var rel = RelevantSet("d1", "d3");

        Assert.Equal(0.5, RetrievalMetrics.RecallAtK(ranked, rel, 2), 6);
        Assert.Equal(1.0, RetrievalMetrics.RecallAtK(ranked, rel, 4), 6);
        Assert.Equal(0.0, RetrievalMetrics.RecallAtK(ranked, RelevantSet(), 4), 6);
    }

    [Fact]
    public void HitRateAtK_KnownRanking()
    {
        var ranked = Ranked("d1", "d2", "d3");
        Assert.Equal(1.0, RetrievalMetrics.HitRateAtK(ranked, RelevantSet("d3"), 3), 6);
        Assert.Equal(0.0, RetrievalMetrics.HitRateAtK(ranked, RelevantSet("d3"), 2), 6);
        Assert.Equal(0.0, RetrievalMetrics.HitRateAtK(ranked, RelevantSet("zzz"), 3), 6);
    }

    [Fact]
    public void ReciprocalRank_And_MRR()
    {
        Assert.Equal(1.0, RetrievalMetrics.ReciprocalRank(Ranked("d1", "d2"), RelevantSet("d1")), 6);
        Assert.Equal(0.5, RetrievalMetrics.ReciprocalRank(Ranked("d2", "d1"), RelevantSet("d1")), 6);
        Assert.Equal(0.0, RetrievalMetrics.ReciprocalRank(Ranked("d2", "d3"), RelevantSet("d1")), 6);

        var rankings = new[] { Ranked("d1", "d2"), Ranked("d2", "d1") };
        var relevants = new[] { RelevantSet("d1"), RelevantSet("d1") };
        Assert.Equal(0.75, RetrievalMetrics.MeanReciprocalRank(rankings, relevants), 6); // (1.0 + 0.5)/2
    }

    [Fact]
    public void AveragePrecision_And_MAP()
    {
        // ranked d1(rel) d2 d3(rel) d4, R=2: AP = (P@1 + P@3)/2 = (1 + 2/3)/2 = 0.833333
        var ranked = Ranked("d1", "d2", "d3", "d4");
        var rel = RelevantSet("d1", "d3");
        Assert.Equal(0.8333333, RetrievalMetrics.AveragePrecision(ranked, rel), 6);

        var rankings = new[] { ranked, Ranked("d1") };
        var relevants = new[] { rel, RelevantSet("d1") };
        // MAP = (0.833333 + 1.0)/2 = 0.916667
        Assert.Equal(0.9166667, RetrievalMetrics.MeanAveragePrecision(rankings, relevants), 6);
    }

    [Fact]
    public void NdcgAtK_BinaryRelevance_KnownRanking()
    {
        // ranked d1(rel) d2 d3(rel): DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        // IDCG (2 relevant) = 1/log2(2) + 1/log2(3) = 1 + 0.6309298 = 1.6309298
        // nDCG = 1.5 / 1.6309298 = 0.9197208
        var ranked = Ranked("d1", "d2", "d3");
        var rel = RelevantSet("d1", "d3");
        Assert.Equal(0.9197208, RetrievalMetrics.NdcgAtK(ranked, rel, 3), 6);

        // Perfect ranking -> 1.0
        Assert.Equal(1.0, RetrievalMetrics.NdcgAtK(Ranked("d1", "d3", "d2"), rel, 3), 6);

        // No relevant -> 0
        Assert.Equal(0.0, RetrievalMetrics.NdcgAtK(ranked, RelevantSet(), 3), 6);
    }

    [Fact]
    public void NdcgAtK_GradedRelevance_KnownRanking()
    {
        // ranked b,a,c ; gains a=3,b=2,c=1
        // DCG = 2/log2(2) + 3/log2(3) + 1/log2(4) = 2 + 1.8927893 + 0.5 = 4.3927893
        // IDCG(order a,b,c) = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2618595 + 0.5 = 4.7618595
        // nDCG = 4.3927893 / 4.7618595 = 0.92249
        var ranked = Ranked("b", "a", "c");
        var gains = new Dictionary<string, double> { ["a"] = 3, ["b"] = 2, ["c"] = 1 };
        Assert.Equal(0.92249, RetrievalMetrics.NdcgAtK(ranked, gains, 3), 5);
    }

    // ---- Embedding-similarity metrics ---------------------------------------

    [Fact]
    public void AnswerSimilarity_WithEmbeddingModel_UsesCosine()
    {
        var emb = new FakeEmbeddingModel(new Dictionary<string, double[]>
        {
            ["cat"] = new[] { 1.0, 0.0 },
            ["kitten"] = new[] { 1.0, 1.0 },
        });
        var metric = new AnswerSimilarityMetric<double>(emb);

        var score = metric.Evaluate(Answer("q", "cat"), "kitten");
        Assert.Equal(0.7071068, score, 6); // cos([1,0],[1,1])
    }

    [Fact]
    public void AnswerSimilarity_WithoutModel_FallsBackToJaccard()
    {
        var metric = new AnswerSimilarityMetric<double>();
        // words {a,b,c} vs {b,c,d}: intersection 2, union 4 -> 0.5
        var score = metric.Evaluate(Answer("q", "a b c"), "b c d");
        Assert.Equal(0.5, score, 6);
    }

    [Fact]
    public void ContextRelevance_WithEmbeddingModel_AveragesCosine()
    {
        var emb = new FakeEmbeddingModel(new Dictionary<string, double[]>
        {
            ["the query"] = new[] { 1.0, 0.0 },
            ["doc one"] = new[] { 1.0, 0.0 }, // cos 1
            ["doc two"] = new[] { 0.0, 1.0 }, // cos 0
        });
        var metric = new ContextRelevanceMetric<double>(emb);

        var ans = Answer("the query", "answer",
            new Document<double>("1", "doc one"),
            new Document<double>("2", "doc two"));

        Assert.Equal(0.5, metric.Evaluate(ans), 6);
    }

    [Fact]
    public void AnswerRelevance_WithModels_MeanQuestionQueryCosine()
    {
        var gen = new FakeTextGen(_ => "question one\nquestion two");
        var emb = new FakeEmbeddingModel(new Dictionary<string, double[]>
        {
            ["the query"] = new[] { 1.0, 0.0 },
            ["question one"] = new[] { 1.0, 0.0 }, // cos 1
            ["question two"] = new[] { 0.0, 1.0 }, // cos 0
        });
        var metric = new AnswerRelevanceMetric<double>(gen, emb, numQuestions: 2);

        var score = metric.Evaluate(Answer("the query", "some answer"));
        Assert.Equal(0.5, score, 6);
        Assert.Single(gen.Prompts);
    }

    [Fact]
    public void AnswerCorrectness_WithJudgeAndEmbedding_AveragesSignals()
    {
        var gen = new FakeTextGen(_ => "8"); // 8/10 = 0.8
        var emb = new FakeEmbeddingModel(new Dictionary<string, double[]>
        {
            ["ans"] = new[] { 1.0, 1.0 },
            ["ref"] = new[] { 1.0, 0.0 }, // cos = 0.7071068
        });
        var metric = new AnswerCorrectnessMetric<double>(gen, emb);

        var score = metric.Evaluate(Answer("q", "ans"), "ref");
        Assert.Equal((0.8 + 0.7071068) / 2.0, score, 6);
    }

    // ---- LLM-judge metrics --------------------------------------------------

    [Fact]
    public void Faithfulness_WithJudge_ScoresSupportedClaims()
    {
        // Answer has three claims; the judge supports the first two, rejects the third.
        var gen = new FakeTextGen(p =>
            p.Contains("Statement: The sky is blue") ? "yes" :
            p.Contains("Statement: Water is wet") ? "yes" :
            "no");
        var metric = new FaithfulnessMetric<double>(gen);

        var ans = Answer("q", "The sky is blue. Water is wet. Cats can fly.",
            new Document<double>("1", "context about sky, water, and cats"));

        var score = metric.Evaluate(ans);
        Assert.Equal(2.0 / 3.0, score, 6);
        Assert.Equal(3, gen.Prompts.Count);
    }

    [Fact]
    public void Faithfulness_WithoutGenerator_FallsBackToWordOverlap()
    {
        var metric = new FaithfulnessMetric<double>();
        // answer words {alpha,beta}; source has alpha only -> 1/2
        var ans = Answer("q", "alpha beta", new Document<double>("1", "alpha only here"));
        Assert.Equal(0.5, metric.Evaluate(ans), 6);
    }

    [Fact]
    public void ContextPrecision_WithJudge_AveragePrecisionOverRanks()
    {
        // docs: relevant, not, relevant -> weighted (P@1=1 + P@3=2/3)/2 relevant = 0.833333
        var gen = new FakeTextGen(p => p.Contains("relevant") ? "yes" : "no");
        var metric = new ContextPrecisionMetric<double>(gen);

        var ans = Answer("q", "answer",
            new Document<double>("1", "alpha relevant"),
            new Document<double>("2", "beta noise"),
            new Document<double>("3", "gamma relevant"));

        Assert.Equal(0.8333333, metric.Evaluate(ans, "ground truth"), 6);
    }

    [Fact]
    public void ContextRecall_WithJudge_FractionOfAttributableClaims()
    {
        // ground truth has two claims; judge attributes the first, not the second -> 0.5
        var gen = new FakeTextGen(p => p.Contains("Statement: Fact one") ? "yes" : "no");
        var metric = new ContextRecallMetric<double>(gen);

        var ans = Answer("q", "answer", new Document<double>("1", "some retrieved context"));

        var score = metric.Evaluate(ans, "Fact one. Fact two.");
        Assert.Equal(0.5, score, 6);
        Assert.Equal(2, gen.Prompts.Count);
    }

    [Fact]
    public void ContextRecall_WithoutModels_FallsBackToWordCoverage()
    {
        var metric = new ContextRecallMetric<double>();
        // GT claim "alpha beta gamma": context covers alpha,beta (2/3 >= 0.5) -> attributable
        var ans = Answer("q", "answer", new Document<double>("1", "alpha beta present"));
        var score = metric.Evaluate(ans, "alpha beta gamma");
        Assert.Equal(1.0, score, 6);
    }

    // ---- Evaluator wiring ---------------------------------------------------

    [Fact]
    public void RAGEvaluator_EnumeratesInjectedMetrics_IncludingNewOnes()
    {
        var emb = new FakeEmbeddingModel(new Dictionary<string, double[]>
        {
            ["the query"] = new[] { 1.0, 0.0 },
            ["ctx"] = new[] { 1.0, 0.0 },
        });
        var metrics = new IRAGMetric<double>[]
        {
            new ContextRelevanceMetric<double>(emb),
            new FaithfulnessMetric<double>(),
        };
        var evaluator = new RAGEvaluator<double>(metrics);

        var ans = Answer("the query", "ctx", new Document<double>("1", "ctx"));
        var result = evaluator.Evaluate(ans);

        Assert.True(result.MetricScores.ContainsKey("Context Relevance"));
        Assert.True(result.MetricScores.ContainsKey("Faithfulness"));
    }
}
