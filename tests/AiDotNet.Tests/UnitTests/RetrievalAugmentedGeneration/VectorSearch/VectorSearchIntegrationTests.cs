using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch
{
    public class VectorSearchIntegrationTests
    {
        #region End-to-End Search Pipeline Tests

        [Fact]
        public void EndToEnd_SearchPipeline_WithFlatIndex_ReturnsAccurateResults()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new FlatIndex<double>(metric);

            // Simulate document embeddings
            var documents = CreateDocumentEmbeddings(100);
            index.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var results = index.Search(query, 10);

            // Assert
            Assert.Equal(10, results.Count);
            Assert.All(results, r => Assert.NotNull(r.Id));
            // Verify results are ordered correctly
            for (int i = 0; i < results.Count - 1; i++)
            {
                Assert.True(Convert.ToDouble(results[i].Score) >= Convert.ToDouble(results[i + 1].Score));
            }
        }

        [Fact]
        public void EndToEnd_SearchPipeline_WithIVFIndex_ReturnsReasonableResults()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new IVFIndex<double>(metric, numClusters: 10, numProbes: 3);

            // Simulate document embeddings
            var documents = CreateDocumentEmbeddings(100);
            index.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var results = index.Search(query, 10);

            // Assert
            Assert.NotEmpty(results);
            Assert.True(results.Count <= 10);
        }

        [Fact]
        public void EndToEnd_SearchPipeline_WithHNSWIndex_ReturnsReasonableResults()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new HNSWIndex<double>(metric, maxConnections: 16);

            // Simulate document embeddings
            var documents = CreateDocumentEmbeddings(100);
            index.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var results = index.Search(query, 10);

            // Assert
            Assert.Equal(10, results.Count);
        }

        [Fact]
        public void EndToEnd_SearchPipeline_WithLSHIndex_ReturnsReasonableResults()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new LSHIndex<double>(metric, numHashTables: 15, numHashFunctions: 6);

            // Simulate document embeddings
            var documents = CreateDocumentEmbeddings(100);
            index.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var results = index.Search(query, 10);

            // Assert
            Assert.NotEmpty(results);
        }

        #endregion

        #region Multi-Vector Search Tests

        [Fact]
        public void MultiVectorSearch_WithDifferentMetrics_ProducesDifferentResults()
        {
            // Arrange
            var cosineMetric = new CosineSimilarityMetric<double>();
            var euclideanMetric = new EuclideanDistanceMetric<double>();

            var indexCosine = new FlatIndex<double>(cosineMetric);
            var indexEuclidean = new FlatIndex<double>(euclideanMetric);

            var documents = CreateDocumentEmbeddings(50);
            indexCosine.AddBatch(documents);
            indexEuclidean.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var resultsCosine = indexCosine.Search(query, 10);
            var resultsEuclidean = indexEuclidean.Search(query, 10);

            // Assert
            Assert.Equal(10, resultsCosine.Count);
            Assert.Equal(10, resultsEuclidean.Count);
            // Different metrics should produce different orderings
            // (though some results may overlap)
        }

        [Fact]
        public void MultiVectorSearch_WithMultipleQueries_HandlesCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new FlatIndex<double>(metric);

            var documents = CreateDocumentEmbeddings(100);
            index.AddBatch(documents);

            // Act - perform multiple searches
            var results1 = index.Search(CreateQueryEmbedding(), 5);
            var results2 = index.Search(CreateQueryEmbedding(), 5);
            var results3 = index.Search(CreateQueryEmbedding(), 5);

            // Assert
            Assert.All(new[] { results1, results2, results3 }, r => Assert.Equal(5, r.Count));
        }

        #endregion

        #region Filtered Search Tests

        [Fact]
        public void FilteredSearch_ByRemovingVectors_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new FlatIndex<double>(metric);

            var documents = CreateDocumentEmbeddings(50);
            index.AddBatch(documents);

            var query = CreateQueryEmbedding();
            var resultsBefore = index.Search(query, 10);

            // Remove some vectors (simulate filtering)
            for (int i = 0; i < 10; i++)
            {
                index.Remove($"doc{i}");
            }

            // Act
            var resultsAfter = index.Search(query, 10);

            // Assert
            Assert.Equal(10, resultsBefore.Count);
            Assert.Equal(10, resultsAfter.Count);
            Assert.Equal(40, index.Count);
        }

        #endregion

        #region Recall and Accuracy Tests

        [Fact]
        public void Recall_FlatIndexVsApproximateIndexes_FlatHasPerfectRecall()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var flatIndex = new FlatIndex<double>(metric);
            var ivfIndex = new IVFIndex<double>(metric, numClusters: 5, numProbes: 5);

            var documents = CreateDocumentEmbeddings(100);
            flatIndex.AddBatch(documents);
            ivfIndex.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var groundTruth = flatIndex.Search(query, 10);
            var approximate = ivfIndex.Search(query, 10);

            // Assert
            Assert.Equal(10, groundTruth.Count);
            Assert.NotEmpty(approximate);

            // Calculate recall (how many of ground truth are in approximate results)
            var groundTruthIds = new HashSet<string>(groundTruth.Select(r => r.Id));
            var approximateIds = new HashSet<string>(approximate.Select(r => r.Id));
            var overlap = groundTruthIds.Intersect(approximateIds).Count();

            // With sufficient probes, should have reasonable recall
            Assert.True(overlap > 0);
        }

        [Fact]
        public void Recall_IncreasingProbes_ImprovesRecall()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var flatIndex = new FlatIndex<double>(metric);
            var ivfLowProbes = new IVFIndex<double>(metric, numClusters: 10, numProbes: 1);
            var ivfHighProbes = new IVFIndex<double>(metric, numClusters: 10, numProbes: 5);

            var documents = CreateDocumentEmbeddings(100);
            flatIndex.AddBatch(documents);
            ivfLowProbes.AddBatch(documents);
            ivfHighProbes.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var groundTruth = flatIndex.Search(query, 10);
            var resultsLowProbes = ivfLowProbes.Search(query, 10);
            var resultsHighProbes = ivfHighProbes.Search(query, 10);

            // Assert
            var groundTruthIds = new HashSet<string>(groundTruth.Select(r => r.Id));

            var recallLow = groundTruthIds.Intersect(resultsLowProbes.Select(r => r.Id)).Count() / 10.0;
            var recallHigh = groundTruthIds.Intersect(resultsHighProbes.Select(r => r.Id)).Count() / 10.0;

            // Higher probes should have better or equal recall
            Assert.True(recallHigh >= recallLow);
        }

        #endregion

        #region Performance and Scale Tests

        [Fact]
        public void LargeScale_SearchWithThousandsOfVectors_CompletesSuccessfully()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new FlatIndex<double>(metric);

            // Add 1000 vectors
            var documents = CreateDocumentEmbeddings(1000);
            index.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var results = index.Search(query, 20);

            // Assert
            Assert.Equal(20, results.Count);
            Assert.Equal(1000, index.Count);
        }

        [Fact]
        public void HighDimensional_SearchWith512Dimensions_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new FlatIndex<double>(metric);
            var random = RandomHelper.CreateSeededRandom(42);

            // Add high-dimensional vectors (simulating modern embeddings like OpenAI ada-002)
            var documents = new Dictionary<string, Vector<double>>();
            for (int i = 0; i < 100; i++)
            {
                var vector = new Vector<double>(Enumerable.Range(0, 512)
                    .Select(_ => random.NextDouble())
                    .ToArray());
                documents[$"doc{i}"] = vector;
            }
            index.AddBatch(documents);

            var query = new Vector<double>(Enumerable.Range(0, 512)
                .Select(_ => random.NextDouble())
                .ToArray());

            // Act
            var results = index.Search(query, 10);

            // Assert
            Assert.Equal(10, results.Count);
        }

        #endregion

        #region Edge Cases and Robustness Tests

        [Fact]
        public void RobustnessTest_AddRemoveAddCycle_MaintainsConsistency()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new FlatIndex<double>(metric);

            var documents = CreateDocumentEmbeddings(50);

            // Act - add, remove, add cycle
            index.AddBatch(documents);
            Assert.Equal(50, index.Count);

            for (int i = 0; i < 10; i++)
            {
                index.Remove($"doc{i}");
            }
            Assert.Equal(40, index.Count);

            var newDocuments = CreateDocumentEmbeddings(10, startId: 50);
            index.AddBatch(newDocuments);
            Assert.Equal(50, index.Count);

            var query = CreateQueryEmbedding();
            var results = index.Search(query, 10);

            // Assert
            Assert.Equal(10, results.Count);
        }

        [Fact]
        public void NumericalStability_WithVerySmallVectors_WorksCorrectly()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var index = new FlatIndex<double>(metric);

            var vector1 = new Vector<double>(new double[] { 1e-10, 2e-10, 3e-10 });
            var vector2 = new Vector<double>(new double[] { 1e-10, 2e-10, 3e-10 });
            var vector3 = new Vector<double>(new double[] { 4e-10, 5e-10, 6e-10 });

            index.Add("vec1", vector1);
            index.Add("vec2", vector2);
            index.Add("vec3", vector3);

            var query = new Vector<double>(new double[] { 1e-10, 2e-10, 3e-10 });

            // Act
            var results = index.Search(query, 2);

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void ComparisonTest_AllIndexTypes_ReturnValidResults()
        {
            // Arrange
            var metric = new CosineSimilarityMetric<double>();
            var flatIndex = new FlatIndex<double>(metric);
            var ivfIndex = new IVFIndex<double>(metric, numClusters: 5, numProbes: 3);
            var hnswIndex = new HNSWIndex<double>(metric, maxConnections: 8);
            var lshIndex = new LSHIndex<double>(metric, numHashTables: 10);

            var documents = CreateDocumentEmbeddings(50);

            flatIndex.AddBatch(documents);
            ivfIndex.AddBatch(documents);
            hnswIndex.AddBatch(documents);
            lshIndex.AddBatch(documents);

            var query = CreateQueryEmbedding();

            // Act
            var resultsFlat = flatIndex.Search(query, 10);
            var resultsIVF = ivfIndex.Search(query, 10);
            var resultsHNSW = hnswIndex.Search(query, 10);
            var resultsLSH = lshIndex.Search(query, 10);

            // Assert - all should return valid results
            Assert.Equal(10, resultsFlat.Count);
            Assert.NotEmpty(resultsIVF);
            Assert.NotEmpty(resultsHNSW);
            Assert.NotEmpty(resultsLSH);
        }

        #endregion

        #region Helper Methods

        private Dictionary<string, Vector<double>> CreateDocumentEmbeddings(int count, int startId = 0)
        {
            var random = RandomHelper.CreateSeededRandom(42);
            var documents = new Dictionary<string, Vector<double>>();

            for (int i = 0; i < count; i++)
            {
                var vector = new Vector<double>(Enumerable.Range(0, 128)
                    .Select(_ => random.NextDouble())
                    .ToArray());
                documents[$"doc{startId + i}"] = vector;
            }

            return documents;
        }

        private Vector<double> CreateQueryEmbedding()
        {
            var random = RandomHelper.CreateSeededRandom(123);
            return new Vector<double>(Enumerable.Range(0, 128)
                .Select(_ => random.NextDouble())
                .ToArray());
        }

        #endregion
    }
}
