using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.ContextCompression
{
    /// <summary>
    /// Base class for context compressor tests with shared test utilities.
    /// </summary>
    public abstract class ContextCompressorTestBase
    {
        protected static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();

        /// <summary>
        /// Creates sample documents for testing.
        /// </summary>
        protected List<Document<double>> CreateSampleDocuments()
        {
            return new List<Document<double>>
            {
                new Document<double>("doc1", "Machine learning is a subset of artificial intelligence. It enables computers to learn from data without explicit programming. Neural networks are a key component of deep learning.")
                {
                    Metadata = new Dictionary<string, object> { { "source", "ml_intro.txt" } },
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                },
                new Document<double>("doc2", "Python is a popular programming language for machine learning. Libraries like TensorFlow and PyTorch make it easy to build neural networks. Data scientists prefer Python for its simplicity.")
                {
                    Metadata = new Dictionary<string, object> { { "source", "python_ml.txt" } },
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                },
                new Document<double>("doc3", "The weather today is sunny and warm. Many people enjoy outdoor activities during such pleasant conditions. Parks are crowded on sunny days.")
                {
                    Metadata = new Dictionary<string, object> { { "source", "weather.txt" } },
                    RelevanceScore = 0.1,
                    HasRelevanceScore = true
                }
            };
        }

        /// <summary>
        /// Creates a document with specified length for testing compression.
        /// </summary>
        protected Document<double> CreateDocumentWithLength(string id, int sentenceCount)
        {
            var sentences = new List<string>();
            for (int i = 0; i < sentenceCount; i++)
            {
                sentences.Add($"This is sentence number {i + 1} containing some test content.");
            }
            return new Document<double>(id, string.Join(" ", sentences));
        }

        /// <summary>
        /// Creates a very large document for testing edge cases (100KB+).
        /// </summary>
        protected Document<double> CreateLargeDocument(string id)
        {
            var content = string.Join(" ", Enumerable.Repeat("This is a very long document with lots of content that needs to be compressed effectively.", 2000));
            return new Document<double>(id, content);
        }

        /// <summary>
        /// Creates a document with Unicode content.
        /// </summary>
        protected Document<double> CreateUnicodeDocument(string id)
        {
            return new Document<double>(id, "机器学习是人工智能的子集。El aprendizaje automático es poderoso. Машинное обучение важно. 🤖 Emojis are Unicode too!");
        }

        /// <summary>
        /// Creates a document with special characters.
        /// </summary>
        protected Document<double> CreateSpecialCharDocument(string id)
        {
            return new Document<double>(id, "Special chars: @#$%^&*()! Testing with symbols <>=+- and punctuation... Multiple?? Questions!! End.");
        }

        /// <summary>
        /// Asserts that documents have been compressed.
        /// </summary>
        protected void AssertCompressed(List<Document<double>> original, List<Document<double>> compressed, bool allowEqual = false)
        {
            Assert.NotNull(compressed);

            var originalTotalLength = original.Sum(d => d.Content.Length);
            var compressedTotalLength = compressed.Sum(d => d.Content.Length);

            if (allowEqual)
            {
                Assert.True(compressedTotalLength <= originalTotalLength,
                    $"Compressed length ({compressedTotalLength}) should be less than or equal to original ({originalTotalLength})");
            }
            else
            {
                Assert.True(compressedTotalLength < originalTotalLength,
                    $"Compressed length ({compressedTotalLength}) should be less than original ({originalTotalLength})");
            }
        }

        /// <summary>
        /// Calculates the compression ratio.
        /// </summary>
        protected double CalculateCompressionRatio(List<Document<double>> original, List<Document<double>> compressed)
        {
            var originalLength = original.Sum(d => d.Content.Length);
            var compressedLength = compressed.Sum(d => d.Content.Length);
            return originalLength > 0 ? (double)compressedLength / originalLength : 0;
        }

        /// <summary>
        /// Asserts that metadata is preserved.
        /// </summary>
        protected void AssertMetadataPreserved(List<Document<double>> original, List<Document<double>> compressed)
        {
            for (int i = 0; i < Math.Min(original.Count, compressed.Count); i++)
            {
                if (original[i].Metadata != null && compressed[i].Metadata != null)
                {
                    foreach (var kvp in original[i].Metadata)
                    {
                        Assert.True(compressed[i].Metadata.ContainsKey(kvp.Key),
                            $"Metadata key '{kvp.Key}' should be preserved");
                    }
                }
            }
        }

        /// <summary>
        /// Asserts that relevance scores are preserved.
        /// </summary>
        protected void AssertRelevanceScoresPreserved(List<Document<double>> original, List<Document<double>> compressed)
        {
            for (int i = 0; i < Math.Min(original.Count, compressed.Count); i++)
            {
                if (original[i].HasRelevanceScore)
                {
                    Assert.True(compressed[i].HasRelevanceScore, "HasRelevanceScore should be preserved");
                    Assert.Equal(original[i].RelevanceScore, compressed[i].RelevanceScore);
                }
            }
        }
    }
}
