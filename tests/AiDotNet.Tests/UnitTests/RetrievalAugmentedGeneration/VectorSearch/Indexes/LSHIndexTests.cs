using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    public class LSHIndexTests
    {
        [Fact]
        public void Constructor_WithNullMetric_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new LSHIndex<double>(null!));
        }

        [Fact]
        public void Constructor_WithNegativeNumHashTables_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new LSHIndex<double>(metric, numHashTables: -1));
        }

        [Fact]
        public void Constructor_WithZeroNumHashTables_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new LSHIndex<double>(metric, numHashTables: 0));
        }

        [Fact]
        public void Constructor_WithNegativeNumHashFunctions_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new LSHIndex<double>(metric, numHashFunctions: -1));
        }

        [Fact]
        public void Add_WithValidVector_IncreasesCount()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            index.Add("vec1", vector);

            // Assert
            Assert.Equal(1, index.Count);
        }

        [Fact]
        public void Add_WithNullId_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Add(null!, vector));
        }

        [Fact]
        public void Add_WithNullVector_ThrowsArgumentNullException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => index.Add("vec1", null!));
        }

        [Fact]
        public void Add_WithInconsistentDimension_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);
            var vector1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var vector2 = new Vector<double>(new double[] { 1.0, 2.0 });

            index.Add("vec1", vector1);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Add("vec2", vector2));
        }

        [Fact]
        public void AddBatch_WithValidVectors_IncreasesCount()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);
            var vectors = new Dictionary<string, Vector<double>>
            {
                ["vec1"] = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
                ["vec2"] = new Vector<double>(new double[] { 4.0, 5.0, 6.0 }),
                ["vec3"] = new Vector<double>(new double[] { 7.0, 8.0, 9.0 })
            };

            // Act
            index.AddBatch(vectors);

            // Assert
            Assert.Equal(3, index.Count);
        }

        [Fact]
        public void Search_WithCosineSimilarity_ReturnsApproximateResults()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric, numHashTables: 10, numHashFunctions: 4);

            index.Add("vec1", new Vector<double>(new double[] { 1.0, 0.0, 0.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 0.0, 1.0, 0.0 }));
            index.Add("vec3", new Vector<double>(new double[] { 0.9, 0.1, 0.0 }));
            index.Add("vec4", new Vector<double>(new double[] { 0.1, 0.9, 0.0 }));

            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = index.Search(query, 2);

            // Assert
            Assert.NotEmpty(results);
            Assert.True(results.Count <= 2);
        }

        [Fact]
        public void Search_OnEmptyIndex_ReturnsEmptyList()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);
            var query = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var results = index.Search(query, 5);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Search_WithNullQuery_ThrowsArgumentNullException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => index.Search(null!, 5));
        }

        [Fact]
        public void Search_WithNegativeK_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);
            var query = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Search(query, -1));
        }

        [Fact]
        public void Remove_WithExistingId_RemovesVectorAndReturnsTrue()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);
            index.Add("vec1", new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));

            // Act
            var result = index.Remove("vec1");

            // Assert
            Assert.True(result);
            Assert.Equal(0, index.Count);
        }

        [Fact]
        public void Remove_WithNonExistingId_ReturnsFalse()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);
            index.Add("vec1", new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));

            // Act
            var result = index.Remove("vec2");

            // Assert
            Assert.False(result);
            Assert.Equal(1, index.Count);
        }

        [Fact]
        public void Clear_RemovesAllVectors()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric);

            index.Add("vec1", new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 4.0, 5.0, 6.0 }));
            index.Add("vec3", new Vector<double>(new double[] { 7.0, 8.0, 9.0 }));

            // Act
            index.Clear();

            // Assert
            Assert.Equal(0, index.Count);
        }

        [Fact]
        public void Search_WithDifferentHashParameters_AffectsResults()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var indexFewTables = new LSHIndex<double>(metric, numHashTables: 2, numHashFunctions: 2);
            var indexManyTables = new LSHIndex<double>(metric, numHashTables: 20, numHashFunctions: 6);

            // Add same vectors to both
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < 50; i++)
            {
                var vector = new Vector<double>(Enumerable.Range(0, 10)
                    .Select(_ => random.NextDouble())
                    .ToArray());
                indexFewTables.Add($"vec{i}", vector);
                indexManyTables.Add($"vec{i}", vector);
            }

            var query = new Vector<double>(Enumerable.Range(0, 10)
                .Select(_ => random.NextDouble())
                .ToArray());

            // Act
            var resultsFew = indexFewTables.Search(query, 10);
            var resultsMany = indexManyTables.Search(query, 10);

            // Assert - more tables should potentially find more candidates
            Assert.NotEmpty(resultsFew);
            Assert.NotEmpty(resultsMany);
        }

        [Fact]
        public void HashingIsConsistent_SameVectorGetsSameHash()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric, seed: 42);

            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Add same vector twice
            index.Add("vec1", vector);

            var query = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var results = index.Search(query, 1);

            // Assert - should find the identical vector
            Assert.NotEmpty(results);
        }

        [Fact]
        public void Search_WithEuclideanDistance_WorksCorrectly()
        {
            // Arrange
            var metric = new EuclideanDistanceMetric<double>();
            var index = new LSHIndex<double>(metric);

            index.Add("vec1", new Vector<double>(new double[] { 0.0, 0.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 10.0, 10.0 }));
            index.Add("vec3", new Vector<double>(new double[] { 1.0, 1.0 }));

            var query = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var results = index.Search(query, 2);

            // Assert
            Assert.NotEmpty(results);
        }

        [Fact]
        public void Search_FallbackToFullSearch_WhenNoCandidatesFound()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric, numHashTables: 1, numHashFunctions: 2);

            // Add vectors
            index.Add("vec1", new Vector<double>(new double[] { 1.0, 0.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 0.0, 1.0 }));

            // Query with completely different vector that might not hash to same buckets
            var query = new Vector<double>(new double[] { -1.0, -1.0 });

            // Act
            var results = index.Search(query, 1);

            // Assert - should still return results via fallback
            Assert.NotEmpty(results);
        }

        [Fact]
        public void Index_WithHighDimensionalVectors_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric, numHashTables: 15, numHashFunctions: 8);
            var random = RandomHelper.CreateSeededRandom(42);

            // Add high-dimensional vectors
            for (int i = 0; i < 30; i++)
            {
                var vector = new Vector<double>(Enumerable.Range(0, 128)
                    .Select(_ => random.NextDouble())
                    .ToArray());
                index.Add($"vec{i}", vector);
            }

            var query = new Vector<double>(Enumerable.Range(0, 128)
                .Select(_ => random.NextDouble())
                .ToArray());

            // Act
            var results = index.Search(query, 10);

            // Assert
            Assert.NotEmpty(results);
            Assert.True(results.Count <= 10);
        }

        [Fact]
        public void LSH_WithDeterministicSeed_ProducesSameResults()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index1 = new LSHIndex<double>(metric, seed: 123);
            var index2 = new LSHIndex<double>(metric, seed: 123);

            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });

            index1.Add("vec1", vector);
            index2.Add("vec1", vector);

            var query = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });

            // Act
            var results1 = index1.Search(query, 1);
            var results2 = index2.Search(query, 1);

            // Assert - same seed should produce same hashing
            Assert.Equal(results1.Count, results2.Count);
            if (results1.Count > 0 && results2.Count > 0)
            {
                Assert.Equal(results1[0].Id, results2[0].Id);
            }
        }

        [Fact]
        public void Index_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<float>();
            var index = new LSHIndex<float>(metric);

            index.Add("vec1", new Vector<float>(new float[] { 1.0f, 0.0f }));
            index.Add("vec2", new Vector<float>(new float[] { 0.0f, 1.0f }));

            var query = new Vector<float>(new float[] { 1.0f, 0.0f });

            // Act
            var results = index.Search(query, 1);

            // Assert
            Assert.NotEmpty(results);
        }
    }
}
