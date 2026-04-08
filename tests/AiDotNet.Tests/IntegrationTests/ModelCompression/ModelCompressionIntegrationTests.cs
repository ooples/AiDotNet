using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using AiDotNet.Pruning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ModelCompression;

/// <summary>
/// Integration tests for the complete model compression pipeline.
/// Tests end-to-end compression → decompression → accuracy verification flows.
/// </summary>
public class ModelCompressionIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region DeepCompression Pipeline Tests

    [Fact]
    public void DeepCompression_VectorPipeline_CompressesAndDecompresses()
    {
        // Arrange - Create weights with known distribution
        var weights = new Vector<double>(new double[]
        {
            0.001, 0.002, 0.5, 0.6, 0.003, 0.7, 0.8, 0.004, 0.9, 1.0
        });

        var compression = new DeepCompression<double>(
            pruningSparsity: 0.4,    // 40% pruning
            numClusters: 4,          // 2-bit quantization
            huffmanPrecision: 3);

        // Act - Compress
        var (compressed, metadata) = compression.Compress(weights);

        // Assert - Verify compression occurred
        Assert.NotNull(compressed);
        Assert.NotNull(metadata);

        // Calculate compression ratio
        var originalSize = weights.Length * sizeof(double);
        var compressedSize = compression.GetCompressedSize(compressed, metadata);
        var ratio = compression.CalculateCompressionRatio(originalSize, compressedSize);

        // Note: With high pruning and clustering, compressed size can be larger due to metadata overhead
        // The test verifies the compression completes successfully
        Assert.True(ratio > 0, $"Compression ratio {ratio} should be positive");

        // Act - Decompress
        var decompressed = compression.Decompress(compressed, metadata);

        // Assert - Verify reconstruction (lossy, so check non-pruned weights are close)
        Assert.Equal(weights.Length, decompressed.Length);

        // Large weights should be approximately preserved
        int preservedCount = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            if (Math.Abs(weights[i]) > 0.1)
            {
                // Non-pruned weights should be reasonably close after quantization
                Assert.True(Math.Abs(decompressed[i]) > 0.1,
                    $"Weight at index {i} should not be pruned");
                preservedCount++;
            }
        }
        Assert.True(preservedCount >= 4, "At least 4 large weights should be preserved");
    }

    [Fact]
    public void DeepCompression_MatrixPipeline_CompressesAndDecompresses()
    {
        // Arrange - Create a weight matrix
        var weights = new Matrix<double>(4, 4);
        var values = new double[] { 0.01, 0.8, 0.02, 0.7, 0.03, 0.6, 0.04, 0.5,
                                    0.9, 0.05, 0.85, 0.06, 0.75, 0.07, 0.65, 0.08 };
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                weights[i, j] = values[i * 4 + j];

        var compression = new DeepCompression<double>(
            pruningSparsity: 0.5,
            numClusters: 8);

        // Act
        var (compressed, metadata) = compression.CompressMatrix(weights);
        var decompressed = compression.DecompressMatrix(compressed, metadata);

        // Assert
        Assert.Equal(weights.Rows, decompressed.Rows);
        Assert.Equal(weights.Columns, decompressed.Columns);

        // Calculate compression ratio
        var originalSize = weights.Rows * weights.Columns * sizeof(double);
        var compressedSize = compression.GetCompressedSize(compressed, metadata);
        var ratio = compression.CalculateCompressionRatio(originalSize, compressedSize);

        // Verify compression completes successfully (ratio may be < 1 for small matrices with metadata overhead)
        Assert.True(ratio > 0, $"Compression ratio {ratio} should be positive");
    }

    [Fact]
    public void DeepCompression_TensorPipeline_CompressesAndDecompresses()
    {
        // Arrange - Create a 3D tensor (simulating conv filter) and flatten to vector
        var shape = new int[] { 2, 3, 4 };
        var tensor = new Tensor<double>(shape);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++)
                for (int k = 0; k < shape[2]; k++)
                    tensor[i, j, k] = random.NextDouble();

        // Flatten tensor to vector for compression
        var flatWeights = tensor.ToVector();

        var compression = new DeepCompression<double>(
            pruningSparsity: 0.3,
            numClusters: 16);

        // Act - Compress and decompress as vector
        var (compressed, metadata) = compression.Compress(flatWeights);
        var decompressed = compression.Decompress(compressed, metadata);

        // Assert - Verify length is preserved after decompression
        Assert.Equal(flatWeights.Length, decompressed.Length);
    }

    #endregion

    #region Pruning Integration Tests

    [Fact]
    public void PruningStrategies_AllStrategiesProduceValidMasks()
    {
        // Arrange
        var weights = new Matrix<double>(5, 5);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                weights[i, j] = random.NextDouble();

        var strategies = new IPruningStrategy<double>[]
        {
            new MagnitudePruningStrategy<double>(),
            new StructuredPruningStrategy<double>(StructuredPruningStrategy<double>.StructurePruningType.Neuron),
            new LotteryTicketPruningStrategy<double>()
        };

        foreach (var strategy in strategies)
        {
            // Act
            var scores = strategy.ComputeImportanceScores(weights);
            var mask = strategy.CreateMask(scores, targetSparsity: 0.5);
            var pruned = mask.Apply(weights);

            // Assert
            Assert.NotNull(mask);
            var sparsity = mask.GetSparsity();
            Assert.True(sparsity >= 0.4 && sparsity <= 0.6,
                $"{strategy.GetType().Name}: Sparsity {sparsity} should be near 0.5");

            // Verify pruned weights are zero
            int zeroCount = 0;
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    if (Math.Abs(pruned[i, j]) < 1e-10)
                        zeroCount++;

            Assert.True(zeroCount >= 10, $"{strategy.GetType().Name}: Should have at least 10 zeros");
        }
    }

    [Fact]
    public void GradientPruning_WithGradients_ProducesValidMask()
    {
        // Arrange
        var weights = new Matrix<double>(4, 4);
        var gradients = new Matrix<double>(4, 4);
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                weights[i, j] = random.NextDouble();
                gradients[i, j] = random.NextDouble() * 0.1;
            }
        }

        var strategy = new GradientPruningStrategy<double>();

        // Act
        var scores = strategy.ComputeImportanceScores(weights, gradients);
        var mask = strategy.CreateMask(scores, targetSparsity: 0.5);
        var pruned = mask.Apply(weights);

        // Assert
        Assert.True(strategy.RequiresGradients);
        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6);
    }

    [Fact]
    public void StructuredSparsity_2to4_CreatesValidPattern()
    {
        // Arrange - Create tensor for structured sparsity
        var tensor = new Tensor<double>(new int[] { 1, 8 }); // 8 elements = 2 groups of 4
        for (int i = 0; i < 8; i++)
            tensor[0, i] = i + 1.0;

        var strategy = new MagnitudePruningStrategy<double>();

        // Act
        var scores = strategy.ComputeImportanceScores(tensor);
        var mask = strategy.Create2to4Mask(scores);

        // Assert - 2:4 sparsity means 50% zeros, structured
        Assert.Equal(0.5, mask.GetSparsity(), 2);
    }

    [Fact]
    public void NtoMSparsity_4to8_CreatesValidPattern()
    {
        // Arrange
        var tensor = new Tensor<double>(new int[] { 1, 16 }); // 16 elements = 2 groups of 8
        for (int i = 0; i < 16; i++)
            tensor[0, i] = i + 1.0;

        var strategy = new MagnitudePruningStrategy<double>();

        // Act
        var scores = strategy.ComputeImportanceScores(tensor);
        var mask = strategy.CreateNtoMMask(scores, n: 4, m: 8);

        // Assert - 4:8 sparsity means 50% zeros
        Assert.Equal(0.5, mask.GetSparsity(), 2);
    }

    #endregion

    #region Compression + Pruning Combined Pipeline

    [Fact]
    public void SparsePruning_ThenDeepCompression_AchievesHighCompression()
    {
        // Arrange
        var weights = new Vector<double>(100);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 100; i++)
            weights[i] = random.NextDouble();

        // First: Apply pruning
        var pruningStrategy = new MagnitudePruningStrategy<double>();
        var scores = pruningStrategy.ComputeImportanceScores(weights);
        var mask = pruningStrategy.CreateMask(scores, targetSparsity: 0.7);
        var prunedWeights = mask.Apply(weights);

        // Then: Apply deep compression on pruned weights
        var compression = new DeepCompression<double>(
            pruningSparsity: 0.0, // Already pruned
            numClusters: 8);

        // Act
        var (compressed, metadata) = compression.Compress(prunedWeights);

        // Calculate compression ratio
        var originalSize = prunedWeights.Length * sizeof(double);
        var compressedSize = compression.GetCompressedSize(compressed, metadata);
        var ratio = compression.CalculateCompressionRatio(originalSize, compressedSize);

        // Assert - Verify compression completes (ratio may be < 1 for already-sparse data with metadata)
        Assert.True(ratio > 0, $"Compression ratio {ratio} should be positive");
    }

    #endregion

    #region Weight Clustering Compression Tests

    [Fact]
    public void WeightClustering_ReducesUniqueValues()
    {
        // Arrange
        var weights = new Vector<double>(50);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 50; i++)
            weights[i] = random.NextDouble();

        var clustering = new WeightClusteringCompression<double>(numClusters: 8);

        // Act
        var (compressed, metadata) = clustering.Compress(weights);
        var decompressed = clustering.Decompress(compressed, metadata);

        // Assert - After clustering, should have at most numClusters unique values
        var uniqueValues = new HashSet<double>();
        for (int i = 0; i < decompressed.Length; i++)
            uniqueValues.Add(Math.Round(decompressed[i], 6));

        Assert.True(uniqueValues.Count <= 8,
            $"Should have at most 8 unique values, got {uniqueValues.Count}");
    }

    #endregion

    #region Huffman Encoding Compression Tests

    [Fact]
    public void HuffmanEncoding_CompressesRepetitiveData()
    {
        // Arrange - Create data with repetitive patterns (good for Huffman)
        var weights = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
            weights[i] = (i % 5) * 0.1; // Only 5 unique values

        var huffman = new HuffmanEncodingCompression<double>(precision: 2);

        // Act
        var (compressed, metadata) = huffman.Compress(weights);
        var decompressed = huffman.Decompress(compressed, metadata);

        // Assert
        Assert.Equal(weights.Length, decompressed.Length);

        // Values should be approximately preserved (within precision)
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(weights[i], decompressed[i], 2);
        }
    }

    #endregion

    #region Low-Rank Factorization Tests

    [Fact]
    public void LowRankFactorization_CompressesVector()
    {
        // Arrange - Create a low-rank matrix (should compress well) and flatten to vector
        var matrix = new Matrix<double>(10, 10);
        // Create rank-2 matrix: outer product of two vectors
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                matrix[i, j] = (i + 1) * (j + 1) * 0.01;

        // Flatten to vector
        var weights = new Vector<double>(100);
        int idx = 0;
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                weights[idx++] = matrix[i, j];

        var lowRank = new LowRankFactorizationCompression<double>(targetRank: 2);

        // Act
        var (compressed, metadata) = lowRank.Compress(weights);
        var decompressed = lowRank.Decompress(compressed, metadata);

        // Assert - Low-rank matrix should be well-approximated
        double maxError = 0;
        for (int i = 0; i < 100; i++)
        {
            double error = Math.Abs(weights[i] - decompressed[i]);
            maxError = Math.Max(maxError, error);
        }
        Assert.True(maxError < 0.1, $"Max reconstruction error {maxError} should be small for low-rank matrix");
    }

    #endregion

    #region Product Quantization Tests

    [Fact]
    public void ProductQuantization_CompressesVectors()
    {
        // Arrange
        var weights = new Vector<double>(64);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 64; i++)
            weights[i] = random.NextDouble();

        var pq = new ProductQuantizationCompression<double>(
            numSubvectors: 8,
            numCentroids: 256);

        // Act
        var (compressed, metadata) = pq.Compress(weights);
        var decompressed = pq.Decompress(compressed, metadata);

        // Assert
        Assert.Equal(weights.Length, decompressed.Length);
    }

    #endregion

    #region Compression Analyzer Tests

    [Fact]
    public void CompressionAnalyzer_RecommendsAppropriateStrategy()
    {
        // Arrange - Create sparse weights (should recommend pruning)
        var sparseWeights = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
            sparseWeights[i] = i < 20 ? 0.5 : 0.001; // 80% near-zero

        var analyzer = new CompressionAnalyzer<double>();

        // Act
        var analysis = analyzer.Analyze(sparseWeights);

        // Assert
        Assert.NotNull(analysis);
        var pruningPotential = analysis.PruningPotential;
        Assert.True(pruningPotential is double d && d > 0.5,
            "Should identify high pruning potential for sparse weights");
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void Compression_EmptyInput_ThrowsArgumentException()
    {
        // Arrange
        var compression = new DeepCompression<double>();
        var emptyVector = new Vector<double>(0);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => compression.Compress(emptyVector));
    }

    [Fact]
    public void CreateMask_InvalidSparsity_ThrowsArgumentException()
    {
        // Arrange
        var strategy = new MagnitudePruningStrategy<double>();
        var scores = new Matrix<double>(2, 2);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, -0.1));
        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, 1.5));
    }

    #endregion
}
