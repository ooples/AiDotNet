using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Indexes
{
    public class IVFIndexTests
    {
        [Fact]
        public void Constructor_WithNullMetric_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new IVFIndex<double>(null!));
        }

        [Fact]
        public void Constructor_WithNegativeNumClusters_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new IVFIndex<double>(metric, numClusters: -1));
        }

        [Fact]
        public void Constructor_WithZeroNumClusters_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new IVFIndex<double>(metric, numClusters: 0));
        }

        [Fact]
        public void Constructor_WithNegativeNumProbes_ThrowsArgumentException()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => new IVFIndex<double>(metric, numClusters: 10, numProbes: -1));
        }

        [Fact]
        public void Add_WithValidVector_IncreasesCount()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 5);
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            index.Add("vec1", vector);

            // Assert
            Assert.Equal(1, index.Count);
        }

        [Fact]
        public void AddBatch_WithValidVectors_IncreasesCount()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 5);
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
            var index = new IVFIndex<double>(metric, numClusters: 3, numProbes: 1);

            // Add vectors in clusters
            index.Add("vec1", new Vector<double>(new double[] { 1.0, 0.0, 0.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 0.9, 0.1, 0.0 }));
            index.Add("vec3", new Vector<double>(new double[] { 0.0, 1.0, 0.0 }));
            index.Add("vec4", new Vector<double>(new double[] { 0.0, 0.9, 0.1 }));

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
            var index = new IVFIndex<double>(metric, numClusters: 5);
            var query = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var results = index.Search(query, 5);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Search_WithMultipleProbes_ImproveRecall()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var indexSingleProbe = new IVFIndex<double>(metric, numClusters: 5, numProbes: 1);
            var indexMultiProbe = new IVFIndex<double>(metric, numClusters: 5, numProbes: 3);

            // Add 20 vectors
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < 20; i++)
            {
                var vector = new Vector<double>(Enumerable.Range(0, 10)
                    .Select(_ => random.NextDouble())
                    .ToArray());
                indexSingleProbe.Add($"vec{i}", vector);
                indexMultiProbe.Add($"vec{i}", vector);
            }

            var query = new Vector<double>(Enumerable.Range(0, 10)
                .Select(_ => random.NextDouble())
                .ToArray());

            // Act
            var resultsSingle = indexSingleProbe.Search(query, 10);
            var resultsMulti = indexMultiProbe.Search(query, 10);

            // Assert - multi-probe should find same or more results
            Assert.True(resultsMulti.Count >= resultsSingle.Count);
        }

        [Fact]
        public void Remove_WithExistingId_RemovesVectorAndReturnsTrue()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 3);
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
            var index = new IVFIndex<double>(metric, numClusters: 3);
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
            var index = new IVFIndex<double>(metric, numClusters: 3);

            index.Add("vec1", new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 4.0, 5.0, 6.0 }));

            // Act
            index.Clear();

            // Assert
            Assert.Equal(0, index.Count);
        }

        [Fact]
        public void Search_WithEuclideanDistance_WorksCorrectly()
        {
            // Arrange
            var metric = new EuclideanDistanceMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 3, numProbes: 2);

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
        public void IndexRebuilds_AfterAddingVectors()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 2);

            // Add first batch
            index.Add("vec1", new Vector<double>(new double[] { 1.0, 0.0 }));
            index.Add("vec2", new Vector<double>(new double[] { 0.0, 1.0 }));

            var query = new Vector<double>(new double[] { 1.0, 0.0 });
            var results1 = index.Search(query, 1);

            // Add more vectors
            index.Add("vec3", new Vector<double>(new double[] { 0.95, 0.05 }));

            // Act - should trigger rebuild
            var results2 = index.Search(query, 2);

            // Assert
            Assert.NotEmpty(results1);
            Assert.NotEmpty(results2);
        }

        [Fact]
        public void Search_WithFewerClustersThanVectors_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 2);

            // Add 10 vectors
            for (int i = 0; i < 10; i++)
            {
                var vector = new Vector<double>(new double[] { i / 10.0, 1.0 - i / 10.0 });
                index.Add($"vec{i}", vector);
            }

            var query = new Vector<double>(new double[] { 0.5, 0.5 });

            // Act
            var results = index.Search(query, 5);

            // Assert
            Assert.NotEmpty(results);
            Assert.True(results.Count <= 5);
        }

        [Fact]
        public void Search_WithMoreClustersThanVectors_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 100);

            // Add only 5 vectors
            for (int i = 0; i < 5; i++)
            {
                var vector = new Vector<double>(new double[] { i / 5.0, 1.0 - i / 5.0 });
                index.Add($"vec{i}", vector);
            }

            var query = new Vector<double>(new double[] { 0.5, 0.5 });

            // Act
            var results = index.Search(query, 3);

            // Assert
            Assert.NotEmpty(results);
        }

        [Fact]
        public void Index_WithHighDimensionalVectors_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 5);
            var random = RandomHelper.CreateSeededRandom(42);

            // Add 50 high-dimensional vectors
            for (int i = 0; i < 50; i++)
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
    }
}
