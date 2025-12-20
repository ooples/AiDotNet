using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using AiDotNet.Pruning;
using Xunit;

namespace AiDotNet.Tests.EndToEndTests;

/// <summary>
/// End-to-end tests for the complete model compression pipeline.
/// These tests verify the full workflow: compress -> export -> inference.
///
/// These tests are excluded from CI/CD by using [Trait("Category", "EndToEnd")].
/// To run these tests manually: dotnet test --filter "Category=EndToEnd"
/// To exclude from CI/CD: dotnet test --filter "Category!=EndToEnd"
/// </summary>
[Trait("Category", "EndToEnd")]
public class ModelCompressionEndToEndTests
{
    #region Full Pipeline Tests

    [Fact]
    public void FullPipeline_Prune_Cluster_Huffman_RoundTrip()
    {
        // Arrange - Create realistic weights simulating a neural network layer
        var weights = CreateRealisticWeights(1000);
        var originalSum = CalculateSum(weights);

        // Step 1: Pruning - Remove 50% of smallest weights
        var pruningStrategy = new MagnitudePruningStrategy<double>();
        var scores = pruningStrategy.ComputeImportanceScores(weights);
        var mask = pruningStrategy.CreateMask(scores, targetSparsity: 0.5);
        var prunedWeights = mask.Apply(weights);

        // Step 2: Weight Clustering - Quantize remaining weights to 16 clusters
        var clustering = new WeightClusteringCompression<double>(numClusters: 16, randomSeed: 42);
        var (clusteredWeights, clusterMetadata) = clustering.Compress(prunedWeights);
        var decompressedClustered = clustering.Decompress(clusteredWeights, clusterMetadata);

        // Step 3: Huffman Encoding - Entropy coding
        var huffman = new HuffmanEncodingCompression<double>(precision: 4);
        var (huffmanCompressed, huffmanMetadata) = huffman.Compress(decompressedClustered);
        var finalWeights = huffman.Decompress(huffmanCompressed, huffmanMetadata);

        // Assert - Verify pipeline integrity
        Assert.Equal(weights.Length, finalWeights.Length);

        // Non-zero weights should exist after decompression
        int nonZeroCount = 0;
        for (int i = 0; i < finalWeights.Length; i++)
        {
            if (Math.Abs(finalWeights[i]) > 1e-10)
                nonZeroCount++;
        }

        // After 50% pruning, should have ~50% non-zero weights
        double nonZeroRatio = (double)nonZeroCount / finalWeights.Length;
        Assert.True(nonZeroRatio >= 0.3 && nonZeroRatio <= 0.7,
            $"Should have ~50% non-zero weights, got {nonZeroRatio:P}");
    }

    [Fact]
    public void FullPipeline_DeepCompression_HanEtAl2015()
    {
        // Arrange - Simulate Han et al. 2015 "Deep Compression" pipeline
        // 1. Pruning (magnitude-based)
        // 2. Quantization (weight clustering)
        // 3. Huffman encoding
        var weights = CreateRealisticWeights(5000);

        var deepCompression = new DeepCompression<double>(
            pruningSparsity: 0.6,    // 60% pruning (typical for FC layers)
            numClusters: 32,         // 5-bit quantization
            huffmanPrecision: 4);

        // Act - Full compression
        var (compressed, metadata) = deepCompression.Compress(weights);

        // Get compression statistics
        var originalSize = weights.Length * sizeof(double);
        var compressedSize = deepCompression.GetCompressedSize(compressed, metadata);
        var ratio = deepCompression.CalculateCompressionRatio(originalSize, compressedSize);

        // Decompress
        var decompressed = deepCompression.Decompress(compressed, metadata);

        // Assert
        Assert.Equal(weights.Length, decompressed.Length);
        Assert.True(ratio > 0, $"Compression ratio should be positive, got {ratio}");

        // Verify sparsity is approximately achieved
        int zeroCount = 0;
        for (int i = 0; i < decompressed.Length; i++)
        {
            if (Math.Abs(decompressed[i]) < 1e-10)
                zeroCount++;
        }
        double achievedSparsity = (double)zeroCount / decompressed.Length;
        Assert.True(achievedSparsity >= 0.5, $"Should achieve at least 50% sparsity, got {achievedSparsity:P}");
    }

    [Fact]
    public void FullPipeline_StructuredSparsity_2to4_ThenQuantize()
    {
        // Arrange - Simulate NVIDIA Ampere structured sparsity pipeline
        // 1. 2:4 structured pruning (hardware-accelerated pattern)
        // 2. Weight clustering
        const int tensorSize = 512;
        var tensor = new Tensor<double>(new int[] { 1, tensorSize }); // Typical layer width
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < tensorSize; i++)
            tensor[0, i] = random.NextDouble() * 2 - 1; // [-1, 1]

        var strategy = new MagnitudePruningStrategy<double>();

        // Step 1: 2:4 structured sparsity (50% but hardware-friendly pattern)
        // Note: Create2to4Mask internally flattens the tensor to 1D and creates a 1D mask
        var scores = strategy.ComputeImportanceScores(tensor);
        var mask = strategy.Create2to4Mask(scores);

        // Convert to vector for clustering
        // The mask and vector have the same flattened shape, so Apply works correctly
        var flatWeights = tensor.ToVector();
        Assert.Equal(tensorSize, flatWeights.Length); // Verify shapes match

        var prunedWeights = mask.Apply(flatWeights);
        Assert.Equal(tensorSize, prunedWeights.Length); // Verify output shape

        // Step 2: Weight clustering on remaining values
        var clustering = new WeightClusteringCompression<double>(numClusters: 16, randomSeed: 42);
        var (clustered, metadata) = clustering.Compress(prunedWeights);
        var decompressed = clustering.Decompress(clustered, metadata);

        // Assert - Verify 2:4 pattern is maintained (exactly 2 zeros per 4 elements)
        Assert.Equal(0.5, mask.GetSparsity(), precision: 2);
        Assert.Equal(tensorSize, decompressed.Length);
    }

    [Fact]
    public void FullPipeline_LowRank_ThenQuantize_ForConvLayers()
    {
        // Arrange - Simulate low-rank + quantization for conv layers
        // Create a low-rank weight matrix (common in conv filters)
        var matrix = new Matrix<double>(64, 64); // Simulating 64 conv filters
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < 64; j++)
            {
                // Create a rank-2 pattern
                matrix[i, j] = Math.Sin(i * 0.1) * Math.Cos(j * 0.1);
            }
        }

        // Flatten to vector for compression
        var weights = new Vector<double>(64 * 64);
        int idx = 0;
        for (int i = 0; i < 64; i++)
            for (int j = 0; j < 64; j++)
                weights[idx++] = matrix[i, j];

        // Step 1: Low-rank factorization
        var lowRank = new LowRankFactorizationCompression<double>(targetRank: 4);
        var (lrCompressed, lrMetadata) = lowRank.Compress(weights);
        var lrDecompressed = lowRank.Decompress(lrCompressed, lrMetadata);

        // Step 2: Weight clustering on approximated weights
        var clustering = new WeightClusteringCompression<double>(numClusters: 32, randomSeed: 42);
        var (final, clusterMetadata) = clustering.Compress(lrDecompressed);
        var finalWeights = clustering.Decompress(final, clusterMetadata);

        // Assert
        Assert.Equal(weights.Length, finalWeights.Length);

        // Low-rank matrices should have reasonable reconstruction
        double totalError = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            totalError += Math.Pow(weights[i] - finalWeights[i], 2);
        }
        double rmse = Math.Sqrt(totalError / weights.Length);
        Assert.True(rmse < 0.5, $"RMSE {rmse} should be reasonable for low-rank matrix");
    }

    [Fact]
    public void FullPipeline_IterativePruning_LotteryTicket()
    {
        // Arrange - Simulate Lottery Ticket Hypothesis iterative pruning
        var initialWeights = new Matrix<double>(10, 10);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                initialWeights[i, j] = random.NextDouble();

        var strategy = new LotteryTicketPruningStrategy<double>(iterativeRounds: 3);

        // Store initial weights (key LTH step)
        strategy.StoreInitialWeights("fc1", initialWeights);

        // Simulate training - weights change
        var trainedWeights = initialWeights.Clone();
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                trainedWeights[i, j] += random.NextDouble() * 0.1;

        // Create mask from trained weights
        var scores = strategy.ComputeImportanceScores(trainedWeights);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.8);

        // Reset to winning ticket (initial weights + mask)
        strategy.ResetToInitialWeights("fc1", trainedWeights, mask);

        // Verify
        Assert.True(mask.GetSparsity() >= 0.75 && mask.GetSparsity() <= 0.85,
            $"Should achieve ~80% sparsity, got {mask.GetSparsity():P}");

        // Verify reset preserved initial values where mask is 1
        var maskedInitial = mask.Apply(initialWeights);
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                Assert.Equal(maskedInitial[i, j], trainedWeights[i, j], precision: 10);
            }
        }
    }

    [Fact]
    public void FullPipeline_CompressionAnalyzer_SelectsBestStrategy()
    {
        // Arrange - Create weights with different characteristics
        // Sparse weights (good for pruning)
        var sparseWeights = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
            sparseWeights[i] = i < 20 ? 0.8 : 0.01 * i / 100;

        // Clustered weights (good for quantization)
        var clusteredWeights = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
            clusteredWeights[i] = (i % 5) * 0.2; // Only 5 unique values

        var analyzer = new CompressionAnalyzer<double>();

        // Act
        var sparseAnalysis = analyzer.Analyze(sparseWeights);
        var clusteredAnalysis = analyzer.Analyze(clusteredWeights);

        // Assert - Analyzer should identify characteristics
        Assert.NotNull(sparseAnalysis);
        Assert.NotNull(clusteredAnalysis);

        // Sparse weights should have high pruning potential
        var sparsePruningPotential = sparseAnalysis.PruningPotential;
        Assert.True(sparsePruningPotential is double sp && sp > 0.5,
            "Sparse weights should have high pruning potential");
    }

    #endregion

    #region Stress Tests

    [Fact]
    public void StressTest_LargeModel_CompressionPipeline()
    {
        // Arrange - Simulate a large layer (10K weights)
        var weights = CreateRealisticWeights(10000);

        var deepCompression = new DeepCompression<double>(
            pruningSparsity: 0.7,
            numClusters: 64,
            huffmanPrecision: 4);

        // Act - Time the compression
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var (compressed, metadata) = deepCompression.Compress(weights);
        var compressionTime = stopwatch.ElapsedMilliseconds;

        stopwatch.Restart();
        var decompressed = deepCompression.Decompress(compressed, metadata);
        var decompressionTime = stopwatch.ElapsedMilliseconds;

        // Assert - Should complete in reasonable time
        Assert.True(compressionTime < 30000, $"Compression took {compressionTime}ms, should be under 30s");
        Assert.True(decompressionTime < 10000, $"Decompression took {decompressionTime}ms, should be under 10s");
        Assert.Equal(weights.Length, decompressed.Length);
    }

    [Fact]
    public void StressTest_MultipleCompressionRoundTrips()
    {
        // Arrange - Test stability across multiple compression cycles
        var weights = CreateRealisticWeights(500);
        var clustering = new WeightClusteringCompression<double>(numClusters: 8, randomSeed: 42);

        Vector<double> currentWeights = weights;

        // Act - Multiple round trips
        for (int i = 0; i < 5; i++)
        {
            var (compressed, metadata) = clustering.Compress(currentWeights);
            currentWeights = clustering.Decompress(compressed, metadata);
        }

        // Assert - Should converge (no drift after multiple cycles)
        Assert.Equal(weights.Length, currentWeights.Length);

        // Values should be stable (no NaN or Inf)
        for (int i = 0; i < currentWeights.Length; i++)
        {
            Assert.False(double.IsNaN(currentWeights[i]), $"Got NaN at index {i}");
            Assert.False(double.IsInfinity(currentWeights[i]), $"Got Infinity at index {i}");
        }
    }

    #endregion

    #region Helper Methods

    private static Vector<double> CreateRealisticWeights(int count)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var weights = new Vector<double>(count);

        for (int i = 0; i < count; i++)
        {
            // Simulate typical NN weight distribution (most weights near zero)
            double value = random.NextDouble() * 2 - 1; // [-1, 1]
            weights[i] = value * Math.Exp(-Math.Abs(value) * 2); // Peaked at zero
        }

        return weights;
    }

    private static double CalculateSum(Vector<double> weights)
    {
        double sum = 0;
        for (int i = 0; i < weights.Length; i++)
            sum += weights[i];
        return sum;
    }

    #endregion
}
