#nullable disable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace AiDotNetTests.UnitTests.RAG.Rerankers
{
    /// <summary>
    /// Tests for <see cref="OnnxCrossEncoderReranker{T}"/>.
    ///
    /// The scoring step (real ONNX inference) is isolated behind the virtual <c>ScorePairs</c> seam so
    /// the ranking logic can be exercised with a fake scorer, keeping CI free of any model/network
    /// dependency. A gated integration test loads a real model from an env var and skips when unset.
    /// </summary>
    public class OnnxCrossEncoderRerankerTests
    {
        /// <summary>
        /// Test double: overrides the ONNX inference seam with a caller-supplied scoring function so the
        /// ranking logic can be verified without a real model file.
        /// </summary>
        private sealed class FakeOnnxCrossEncoderReranker : OnnxCrossEncoderReranker<double>
        {
            private readonly Func<string, string, double> _scorer;

            public int ScorePairsCallCount { get; private set; }

            public FakeOnnxCrossEncoderReranker(Func<string, string, double> scorer)
                : base("fake-model.onnx", "fake-tokenizer", maxLength: 128, maxPairsToScore: 100)
            {
                _scorer = scorer;
            }

            protected override double[] ScorePairs(string query, IList<string> documentContents)
            {
                ScorePairsCallCount++;
                var scores = new double[documentContents.Count];
                for (int i = 0; i < documentContents.Count; i++)
                {
                    scores[i] = _scorer(query, documentContents[i]);
                }
                return scores;
            }
        }

        private static List<Document<double>> MakeDocs(params (string Id, string Content)[] items)
        {
            var docs = new List<Document<double>>();
            foreach (var (id, content) in items)
            {
                docs.Add(new Document<double>(id, content));
            }
            return docs;
        }

        // ---------- Constructor validation ----------

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            using var reranker = new OnnxCrossEncoderReranker<double>("model.onnx", "tokenizer-dir");
            Assert.NotNull(reranker);
            Assert.True(reranker.ModifiesScores);
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        public void Constructor_WithInvalidModelPath_Throws(string modelPath)
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                new OnnxCrossEncoderReranker<double>(modelPath, "tokenizer-dir"));
            Assert.Contains("Model path cannot be empty", ex.Message);
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        public void Constructor_WithInvalidTokenizerPath_Throws(string tokenizerPath)
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                new OnnxCrossEncoderReranker<double>("model.onnx", tokenizerPath));
            Assert.Contains("Tokenizer path cannot be empty", ex.Message);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        public void Constructor_WithInvalidMaxLength_Throws(int maxLength)
        {
            Assert.Throws<ArgumentException>(() =>
                new OnnxCrossEncoderReranker<double>("model.onnx", "tokenizer-dir", maxLength: maxLength));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-5)]
        public void Constructor_WithInvalidMaxPairsToScore_Throws(int maxPairs)
        {
            Assert.Throws<ArgumentException>(() =>
                new OnnxCrossEncoderReranker<double>("model.onnx", "tokenizer-dir", maxPairsToScore: maxPairs));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-2)]
        public void Constructor_WithInvalidBatchSize_Throws(int batchSize)
        {
            Assert.Throws<ArgumentException>(() =>
                new OnnxCrossEncoderReranker<double>("model.onnx", "tokenizer-dir", batchSize: batchSize));
        }

        // ---------- Missing-model behavior (no silent fallback) ----------

        [Fact]
        public void Rerank_WithMissingModelFile_ThrowsFileNotFound()
        {
            using var reranker = new OnnxCrossEncoderReranker<double>("nonexistent-model.onnx", "nonexistent-tokenizer");
            var docs = MakeDocs(("d1", "some content"), ("d2", "other content"));

            // Rerank is lazy via IEnumerable; force enumeration.
            Assert.Throws<FileNotFoundException>(() => reranker.Rerank("query", docs).ToList());
        }

        [Fact]
        public void ScorePair_WithMissingModelFile_ThrowsFileNotFound()
        {
            using var reranker = new OnnxCrossEncoderReranker<double>("nonexistent-model.onnx", "nonexistent-tokenizer");
            Assert.Throws<FileNotFoundException>(() => reranker.ScorePair("query", "document"));
        }

        // ---------- Ranking logic via injected fake scorer ----------

        [Fact]
        public void Rerank_ReordersDocumentsByDescendingScore()
        {
            // Score by how many query words appear in the document content.
            var reranker = new FakeOnnxCrossEncoderReranker((query, doc) =>
            {
                var qWords = query.Split(' ');
                return qWords.Count(w => doc.Contains(w));
            });

            var docs = MakeDocs(
                ("low", "unrelated text about cooking"),
                ("high", "machine learning models learn from data"),
                ("mid", "learning is fun"));

            var result = reranker.Rerank("machine learning", docs).ToList();

            Assert.Equal(3, result.Count);
            Assert.Equal("high", result[0].Id); // matches both "machine" and "learning"
            Assert.Equal("mid", result[1].Id);  // matches "learning"
            Assert.Equal("low", result[2].Id);  // matches neither
            Assert.Equal(1, reranker.ScorePairsCallCount);
        }

        [Fact]
        public void Rerank_SetsRelevanceScoreAndFlag()
        {
            var scoreMap = new Dictionary<string, double>
            {
                { "a", 0.1 },
                { "b", 0.9 },
                { "c", 0.5 },
            };
            var reranker = new FakeOnnxCrossEncoderReranker((query, doc) => scoreMap[doc]);

            var docs = MakeDocs(("a", "a"), ("b", "b"), ("c", "c"));

            var result = reranker.Rerank("q", docs).ToList();

            // Ordered b (0.9) > c (0.5) > a (0.1)
            Assert.Equal("b", result[0].Id);
            Assert.Equal("c", result[1].Id);
            Assert.Equal("a", result[2].Id);

            foreach (var doc in result)
            {
                Assert.True(doc.HasRelevanceScore);
            }
            Assert.Equal(0.9, result[0].RelevanceScore, 6);
            Assert.Equal(0.5, result[1].RelevanceScore, 6);
            Assert.Equal(0.1, result[2].RelevanceScore, 6);
        }

        [Fact]
        public void Rerank_TopK_LimitsResults()
        {
            var reranker = new FakeOnnxCrossEncoderReranker((query, doc) => doc.Length);
            var docs = MakeDocs(
                ("d1", "x"),
                ("d2", "xx"),
                ("d3", "xxx"),
                ("d4", "xxxx"));

            var result = reranker.Rerank("q", docs, topK: 2).ToList();

            Assert.Equal(2, result.Count);
            Assert.Equal("d4", result[0].Id);
            Assert.Equal("d3", result[1].Id);
        }

        [Fact]
        public void Rerank_EmptyDocuments_ReturnsEmpty()
        {
            var reranker = new FakeOnnxCrossEncoderReranker((query, doc) => 1.0);
            var result = reranker.Rerank("q", new List<Document<double>>()).ToList();
            Assert.Empty(result);
        }

        [Fact]
        public void Rerank_PreservesDocumentCount()
        {
            var reranker = new FakeOnnxCrossEncoderReranker((query, doc) => doc.GetHashCode());
            var docs = MakeDocs(
                ("d1", "one"), ("d2", "two"), ("d3", "three"),
                ("d4", "four"), ("d5", "five"), ("d6", "six"), ("d7", "seven"));

            var result = reranker.Rerank("q", docs).ToList();
            Assert.Equal(7, result.Count);
        }

        // ---------- Convenience ctor on CrossEncoderReranker ----------

        [Fact]
        public void CrossEncoderReranker_BackedByOnnxReranker_Reranks()
        {
            var scoreMap = new Dictionary<string, double>
            {
                { "low", 0.2 },
                { "high", 0.95 },
            };
            var onnx = new FakeOnnxCrossEncoderReranker((query, doc) => scoreMap[doc]);
            var reranker = new CrossEncoderReranker<double>(onnx);

            var docs = MakeDocs(("low", "low"), ("high", "high"));
            var result = reranker.Rerank("q", docs).ToList();

            Assert.Equal("high", result[0].Id);
            Assert.Equal("low", result[1].Id);
        }

        [Fact]
        public void CrossEncoderReranker_WithNullOnnxReranker_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new CrossEncoderReranker<double>((OnnxCrossEncoderReranker<double>)null));
        }

        // ---------- ExtractScoresFromLogits (pure inference helper) ----------

        [Fact]
        public void ExtractScoresFromLogits_SingleLabel_ReturnsThatValue()
        {
            var flat = new float[] { 0.5f, -1.2f, 3.3f };
            var scores = OnnxCrossEncoderReranker<double>.ExtractScoresFromLogits(flat, batch: 3, numLabels: 1, labelIndex: -1);

            Assert.Equal(3, scores.Length);
            Assert.Equal(0.5, scores[0], 5);
            Assert.Equal(-1.2, scores[1], 5);
            Assert.Equal(3.3, scores[2], 5);
        }

        [Fact]
        public void ExtractScoresFromLogits_TwoLabels_DefaultReadsLastLabel()
        {
            // batch 2, numLabels 2: [ [neg, pos], [neg, pos] ]
            var flat = new float[] { 0.1f, 0.9f, 0.8f, 0.2f };
            var scores = OnnxCrossEncoderReranker<double>.ExtractScoresFromLogits(flat, batch: 2, numLabels: 2, labelIndex: -1);

            Assert.Equal(0.9, scores[0], 5); // last label of item 0
            Assert.Equal(0.2, scores[1], 5); // last label of item 1
        }

        [Fact]
        public void ExtractScoresFromLogits_ExplicitLabelIndex_ReadsThatColumn()
        {
            var flat = new float[] { 0.1f, 0.9f, 0.8f, 0.2f };
            var scores = OnnxCrossEncoderReranker<double>.ExtractScoresFromLogits(flat, batch: 2, numLabels: 2, labelIndex: 0);

            Assert.Equal(0.1, scores[0], 5);
            Assert.Equal(0.8, scores[1], 5);
        }

        [Fact]
        public void ExtractScoresFromLogits_OutOfRangeIndex_FallsBackToLastLabel()
        {
            var flat = new float[] { 0.1f, 0.9f };
            var scores = OnnxCrossEncoderReranker<double>.ExtractScoresFromLogits(flat, batch: 1, numLabels: 2, labelIndex: 99);
            Assert.Equal(0.9, scores[0], 5);
        }

        // ---------- BuildPairInputs (ONNX input plumbing) ----------

        [Fact]
        public void BuildPairInputs_AlwaysIncludesInputIds()
        {
            var declared = new HashSet<string> { "input_ids" };
            var ids = new long[] { 101, 5, 102 };
            var mask = new long[] { 1, 1, 1 };
            var types = new long[] { 0, 0, 1 };
            var shape = new[] { 1, 3 };

            var inputs = OnnxCrossEncoderReranker<double>.BuildPairInputs(declared, ids, mask, types, shape);

            Assert.Contains(inputs, v => v.Name == "input_ids");
            Assert.DoesNotContain(inputs, v => v.Name == "attention_mask");
            Assert.DoesNotContain(inputs, v => v.Name == "token_type_ids");

            var idsTensor = inputs.Single(v => v.Name == "input_ids").AsTensor<long>();
            Assert.Equal(shape, idsTensor.Dimensions.ToArray());
        }

        [Fact]
        public void BuildPairInputs_IncludesDeclaredOptionalInputs()
        {
            var declared = new HashSet<string> { "input_ids", "attention_mask", "token_type_ids" };
            var ids = new long[] { 101, 5, 6, 102 };
            var mask = new long[] { 1, 1, 1, 1 };
            var types = new long[] { 0, 0, 1, 1 };
            var shape = new[] { 1, 4 };

            var inputs = OnnxCrossEncoderReranker<double>.BuildPairInputs(declared, ids, mask, types, shape);

            Assert.Contains(inputs, v => v.Name == "input_ids");
            Assert.Contains(inputs, v => v.Name == "attention_mask");
            Assert.Contains(inputs, v => v.Name == "token_type_ids");

            var typeTensor = inputs.Single(v => v.Name == "token_type_ids").AsTensor<long>();
            Assert.Equal(new long[] { 0, 0, 1, 1 }, typeTensor.ToArray());
        }

        // ---------- TruncateLongestFirst ----------

        [Fact]
        public void TruncateLongestFirst_TrimsLongerListFirst()
        {
            var query = new List<string> { "q1", "q2" };
            var doc = new List<string> { "d1", "d2", "d3", "d4", "d5", "d6" };

            OnnxCrossEncoderReranker<double>.TruncateLongestFirst(query, doc, available: 4);

            Assert.Equal(4, query.Count + doc.Count);
            // Query (2) is shorter, so document is trimmed down to 2.
            Assert.Equal(2, query.Count);
            Assert.Equal(2, doc.Count);
        }

        [Fact]
        public void TruncateLongestFirst_NoTruncationWhenWithinBudget()
        {
            var query = new List<string> { "q1" };
            var doc = new List<string> { "d1", "d2" };

            OnnxCrossEncoderReranker<double>.TruncateLongestFirst(query, doc, available: 10);

            Assert.Equal(1, query.Count);
            Assert.Equal(2, doc.Count);
        }

        [Fact]
        public void TruncateLongestFirst_ZeroBudget_EmptiesBoth()
        {
            var query = new List<string> { "q1", "q2" };
            var doc = new List<string> { "d1" };

            OnnxCrossEncoderReranker<double>.TruncateLongestFirst(query, doc, available: 0);

            Assert.Empty(query);
            Assert.Empty(doc);
        }

        // ---------- Gated integration test (real model) ----------

        /// <summary>
        /// End-to-end test against a real cross-encoder ONNX model. Set the env var
        /// <c>AIDOTNET_CROSSENCODER_ONNX</c> to the model file path and
        /// <c>AIDOTNET_CROSSENCODER_TOKENIZER</c> to the tokenizer directory to run it.
        /// Skips (does not fail) when those are unset, keeping CI model/network-free.
        /// </summary>
        [SkippableFact]
        [Trait("Category", "Integration")]
        public void Integration_RealModel_RanksRelevantDocumentFirst()
        {
            var modelPath = Environment.GetEnvironmentVariable("AIDOTNET_CROSSENCODER_ONNX");
            var tokenizerPath = Environment.GetEnvironmentVariable("AIDOTNET_CROSSENCODER_TOKENIZER");

            Skip.If(string.IsNullOrWhiteSpace(modelPath) || string.IsNullOrWhiteSpace(tokenizerPath),
                "Set AIDOTNET_CROSSENCODER_ONNX and AIDOTNET_CROSSENCODER_TOKENIZER to run the real-model integration test.");

            Skip.IfNot(File.Exists(modelPath), $"Model file not found: {modelPath}");

            using var reranker = new OnnxCrossEncoderReranker<double>(modelPath, tokenizerPath);

            var docs = MakeDocs(
                ("relevant", "The Eiffel Tower is located in Paris, France."),
                ("irrelevant", "Bananas are a good source of potassium."));

            var result = reranker.Rerank("Where is the Eiffel Tower?", docs).ToList();

            Assert.Equal(2, result.Count);
            Assert.Equal("relevant", result[0].Id);
            Assert.True(result[0].HasRelevanceScore);
            Assert.True(result[0].RelevanceScore >= result[1].RelevanceScore);
        }
    }
}
