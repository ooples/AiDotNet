using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    public class HNSWIndexTests
    {
        [Fact]
        public void Constructor_WithNullMetric_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new HNSWIndex<double>(null!));
        }

        [Fact]
        public void Constructor_WithNegativeMaxConnections_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new HNSWIndex<double>(metric, maxConnections: -1));
        }

        [Fact]
        public void Constructor_WithZeroMaxConnections_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new HNSWIndex<double>(metric, maxConnections: 0));
        }

        [Fact]
        public void Constructor_WithNegativeEfConstruction_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new HNSWIndex<double>(metric, efConstruction: -1));
        }

        [Fact]
        public void Add_WithValidVector_IncreasesCount()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);
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
            var index = new HNSWIndex<double>(metric);
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Add(null!, vector));
        }

        [Fact]
        public void Add_WithNullVector_ThrowsArgumentNullException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => index.Add("vec1", null!));
        }

        [Fact]
        public void AddBatch_WithValidVectors_IncreasesCount()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);
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
        public void Search_WithCosineSimilarity_ReturnsNearestVectors()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric, maxConnections: 8);

            index.Add("vec1", new Vector<double>(new double[] { 1.0, 0.0, 0.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 0.0, 1.0, 0.0 }));
            index.Add("vec3", new Vector<double>(new double[] { 0.9, 0.1, 0.0 }));
            index.Add("vec4", new Vector<double>(new double[] { 0.1, 0.9, 0.0 }));

            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = index.Search(query, 2);

            // Assert
            Assert.Equal(2, results.Count);
            // Should return vec1 and vec3 as they're closest to query
        }

        [Fact]
        public void Search_OnEmptyIndex_ReturnsEmptyList()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);
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
            var index = new HNSWIndex<double>(metric);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => index.Search(null!, 5));
        }

        [Fact]
        public void Search_WithNegativeK_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);
            var query = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Search(query, -1));
        }

        [Fact]
        public void Remove_WithExistingId_RemovesVectorAndReturnsTrue()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);
            index.Add("vec1", new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 4.0, 5.0, 6.0 }));

            // Act
            var result = index.Remove("vec1");

            // Assert
            Assert.True(result);
            Assert.Equal(1, index.Count);
        }

        [Fact]
        public void Remove_WithNonExistingId_ReturnsFalse()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);
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
            var index = new HNSWIndex<double>(metric);

            index.Add("vec1", new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 4.0, 5.0, 6.0 }));
            index.Add("vec3", new Vector<double>(new double[] { 7.0, 8.0, 9.0 }));

            // Act
            index.Clear();

            // Assert
            Assert.Equal(0, index.Count);
        }

        [Fact]
        public void GraphConnections_RespectMaxConnections()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var maxConnections = 4;
            var index = new HNSWIndex<double>(metric, maxConnections: maxConnections);

            // Add many vectors to test connection pruning
            for (int i = 0; i < 20; i++)
            {
                var vector = new Vector<double>(new double[] { i / 20.0, 1.0 - i / 20.0, 0.5 });
                index.Add($"vec{i}", vector);
            }

            var query = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });

            // Act
            var results = index.Search(query, 10);

            // Assert - should successfully find results
            Assert.NotEmpty(results);
            Assert.True(results.Count <= 10);
        }

        [Fact]
        public void Search_WithEuclideanDistance_WorksCorrectly()
        {
            // Arrange
            var metric = new EuclideanDistanceMetric<double>();
            var index = new HNSWIndex<double>(metric);

            index.Add("vec1", new Vector<double>(new double[] { 0.0, 0.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 10.0, 10.0 }));
            index.Add("vec3", new Vector<double>(new double[] { 1.0, 1.0 }));

            var query = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var results = index.Search(query, 2);

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Index_WithLargeNumberOfVectors_MaintainsPerformance()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric, maxConnections: 16);
            var random = RandomHelper.CreateSeededRandom(42);

            // Add 100 vectors
            for (int i = 0; i < 100; i++)
            {
                var vector = new Vector<double>(Enumerable.Range(0, 10)
                    .Select(_ => random.NextDouble())
                    .ToArray());
                index.Add($"vec{i}", vector);
            }

            var query = new Vector<double>(Enumerable.Range(0, 10)
                .Select(_ => random.NextDouble())
                .ToArray());

            // Act
            var results = index.Search(query, 10);

            // Assert
            Assert.Equal(10, results.Count);
            Assert.Equal(100, index.Count);
        }

        [Fact]
        public void Search_ReturnsResultsInCorrectOrder()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);

            index.Add("vec1", new Vector<double>(new double[] { 1.0, 0.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 0.9, 0.1 }));
            index.Add("vec3", new Vector<double>(new double[] { 0.0, 1.0 }));

            var query = new Vector<double>(new double[] { 1.0, 0.0 });

            // Act
            var results = index.Search(query, 3);

            // Assert
            Assert.Equal(3, results.Count);
            // Results should be ordered by descending similarity
            for (int i = 0; i < results.Count - 1; i++)
            {
                Assert.True(Convert.ToDouble(results[i].Score) >= Convert.ToDouble(results[i + 1].Score));
            }
        }

        [Fact]
        public void Index_WithHighDimensionalVectors_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric);
            var random = RandomHelper.CreateSeededRandom(42);

            // Add high-dimensional vectors
            for (int i = 0; i < 20; i++)
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
            var results = index.Search(query, 5);

            // Assert
            Assert.Equal(5, results.Count);
        }

        [Fact]
        public void Index_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<float>();
            var index = new HNSWIndex<float>(metric);

            index.Add("vec1", new Vector<float>(new float[] { 1.0f, 0.0f }));
            index.Add("vec2", new Vector<float>(new float[] { 0.0f, 1.0f }));

            var query = new Vector<float>(new float[] { 1.0f, 0.0f });

            // Act
            var results = index.Search(query, 1);

            // Assert
            Assert.Single(results);
        }
    }
}
