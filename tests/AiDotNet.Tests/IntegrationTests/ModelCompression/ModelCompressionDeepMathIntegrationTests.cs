using AiDotNet.Enums;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ModelCompression;

/// <summary>
/// Deep math-correctness integration tests for the ModelCompression module.
/// Verifies exact calculations for CompressionMetrics derived metrics,
/// composite fitness scores, quality thresholds, and WeightClusteringCompression
/// round-trip correctness.
/// </summary>
public class ModelCompressionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-6;

    #region CompressionMetrics - CalculateDerivedMetrics

    [Fact]
    public void CompressionMetrics_CompressionRatio_ExactFormula()
    {
        // CompressionRatio = OriginalSize / CompressedSize
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 1000,
            CompressedSize = 250
        };

        metrics.CalculateDerivedMetrics();

        // 1000 / 250 = 4.0
        Assert.Equal(4.0, metrics.CompressionRatio, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_SizeReductionPercentage_ExactFormula()
    {
        // SizeReductionPercentage = (1 - CompressedSize/OriginalSize) * 100
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 400,
            CompressedSize = 100
        };

        metrics.CalculateDerivedMetrics();

        // (1 - 100/400) * 100 = (1 - 0.25) * 100 = 75.0
        Assert.Equal(75.0, metrics.SizeReductionPercentage, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_SizeReduction_50Percent()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 200,
            CompressedSize = 100
        };

        metrics.CalculateDerivedMetrics();

        Assert.Equal(2.0, metrics.CompressionRatio, Tolerance);
        Assert.Equal(50.0, metrics.SizeReductionPercentage, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_InferenceSpeedup_ExactFormula()
    {
        // InferenceSpeedup = OriginalInferenceTimeMs / CompressedInferenceTimeMs
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 50,
            OriginalInferenceTimeMs = 10.0,
            CompressedInferenceTimeMs = 4.0
        };

        metrics.CalculateDerivedMetrics();

        // 10.0 / 4.0 = 2.5
        Assert.Equal(2.5, metrics.InferenceSpeedup, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_AccuracyLoss_ExactFormula()
    {
        // AccuracyLoss = OriginalAccuracy - CompressedAccuracy
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 50,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.93
        };

        metrics.CalculateDerivedMetrics();

        // 0.95 - 0.93 = 0.02
        Assert.Equal(0.02, metrics.AccuracyLoss, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_MemoryBandwidthSavings_EqualsCompressionRatio()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 800,
            CompressedSize = 100
        };

        metrics.CalculateDerivedMetrics();

        // MemoryBandwidthSavings = CompressionRatio
        Assert.Equal(8.0, metrics.MemoryBandwidthSavings, Tolerance);
        Assert.Equal(metrics.CompressionRatio, metrics.MemoryBandwidthSavings, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_ZeroCompressedInferenceTime_NoSpeedup()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 50,
            OriginalInferenceTimeMs = 10.0,
            CompressedInferenceTimeMs = 0.0
        };

        metrics.CalculateDerivedMetrics();

        // CompressedInferenceTime is 0, InferenceSpeedup stays at default (0)
        Assert.Equal(0.0, metrics.InferenceSpeedup, Tolerance);
    }

    #endregion

    #region CompressionMetrics - MeetsQualityThreshold

    [Fact]
    public void CompressionMetrics_MeetsQualityThreshold_GoodCompression()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 1000,
            CompressedSize = 200,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.94
        };
        metrics.CalculateDerivedMetrics();

        // AccuracyLoss = 0.01 = 1% (< 2% threshold)
        // CompressionRatio = 5.0 (>= 2.0 threshold)
        Assert.True(metrics.MeetsQualityThreshold(2.0, 2.0));
    }

    [Fact]
    public void CompressionMetrics_MeetsQualityThreshold_TooMuchAccuracyLoss()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 1000,
            CompressedSize = 100,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.90
        };
        metrics.CalculateDerivedMetrics();

        // AccuracyLoss = 0.05 = 5% (> 2% threshold)
        // CompressionRatio = 10.0 (>= 2.0 threshold)
        Assert.False(metrics.MeetsQualityThreshold(2.0, 2.0));
    }

    [Fact]
    public void CompressionMetrics_MeetsQualityThreshold_InsufficientCompression()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 80,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.95
        };
        metrics.CalculateDerivedMetrics();

        // AccuracyLoss = 0.0 = 0% (< 2% threshold)
        // CompressionRatio = 1.25 (< 2.0 threshold)
        Assert.False(metrics.MeetsQualityThreshold(2.0, 2.0));
    }

    [Fact]
    public void CompressionMetrics_MeetsQualityThreshold_BoundaryValues()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 200,
            CompressedSize = 100,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.93 // 2% loss exactly
        };
        metrics.CalculateDerivedMetrics();

        // AccuracyLoss = 0.02 = 2% (== maxAccuracyLoss, should pass with <=)
        // CompressionRatio = 2.0 (== minCompressionRatio, should pass with >=)
        Assert.True(metrics.MeetsQualityThreshold(2.0, 2.0));
    }

    #endregion

    #region CompressionMetrics - CalculateCompositeFitness

    [Fact]
    public void CompressionMetrics_CompositeFitness_PerfectAccuracy_NoCompression()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 100
        };
        metrics.CalculateDerivedMetrics();

        // AccuracyLoss = 0 => accuracyPreservation = 1.0
        // CompressionRatio = 1.0 => compressionScore = 1 - 1/(1 + 1/10) = 1 - 10/11 = 1/11
        // InferenceSpeedup = 0 => speedScore = 0
        // fitness = (1.0*0.5 + (1/11)*0.3 + 0*0.2) / 1.0
        double expectedCompressionScore = 1.0 - 1.0 / (1.0 + 1.0 / 10.0);
        double expectedFitness = (1.0 * 0.5 + expectedCompressionScore * 0.3 + 0 * 0.2) / 1.0;

        double fitness = metrics.CalculateCompositeFitness();

        Assert.Equal(expectedFitness, fitness, LooseTolerance);
    }

    [Fact]
    public void CompressionMetrics_CompositeFitness_HighCompression()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 1000,
            CompressedSize = 100,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.94,
            OriginalInferenceTimeMs = 10.0,
            CompressedInferenceTimeMs = 5.0
        };
        metrics.CalculateDerivedMetrics();

        // AccuracyLoss = 0.01 => accuracyPreservation = 0.99
        // CompressionRatio = 10.0 => compressionScore = 1 - 1/(1 + 10/10) = 1 - 1/2 = 0.5
        // InferenceSpeedup = 2.0 => speedScore = 1 - 1/(1 + 2) = 1 - 1/3 = 2/3
        double expectedAccuracy = 0.99;
        double expectedCompression = 0.5;
        double expectedSpeed = 2.0 / 3.0;
        double expectedFitness = (expectedAccuracy * 0.5 + expectedCompression * 0.3 + expectedSpeed * 0.2) / 1.0;

        double fitness = metrics.CalculateCompositeFitness();

        Assert.Equal(expectedFitness, fitness, LooseTolerance);
    }

    [Fact]
    public void CompressionMetrics_CompositeFitness_CustomWeights()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 500,
            CompressedSize = 50,
            OriginalAccuracy = 0.9,
            CompressedAccuracy = 0.8,
            OriginalInferenceTimeMs = 20.0,
            CompressedInferenceTimeMs = 2.0
        };
        metrics.CalculateDerivedMetrics();

        // AccuracyLoss = 0.1 => accuracyPreservation = 0.9
        // CompressionRatio = 10.0 => compressionScore = 1 - 1/(1 + 10/10) = 0.5
        // InferenceSpeedup = 10.0 => speedScore = 1 - 1/(1 + 10) = 1 - 1/11 = 10/11

        // Custom weights: accuracy=0.8, compression=0.1, speed=0.1
        double accuracy = 0.9;
        double compression = 0.5;
        double speed = 10.0 / 11.0;
        double totalWeight = 0.8 + 0.1 + 0.1;
        double expected = (accuracy * 0.8 + compression * 0.1 + speed * 0.1) / totalWeight;

        double fitness = metrics.CalculateCompositeFitness(0.8, 0.1, 0.1);

        Assert.Equal(expected, fitness, LooseTolerance);
    }

    [Fact]
    public void CompressionMetrics_CompositeFitness_AccuracyLossClamped()
    {
        // If accuracy loss > 1.0 (shouldn't happen but test clamping)
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 50
        };
        // Manually set extreme accuracy loss
        metrics.AccuracyLoss = 1.5;
        metrics.CompressionRatio = 2.0;

        // accuracyPreservation = 1.0 - 1.5 = -0.5, clamped to 0.0
        double compressionScore = 1.0 - 1.0 / (1.0 + 2.0 / 10.0);
        double expected = (0.0 * 0.5 + compressionScore * 0.3 + 0 * 0.2) / 1.0;

        double fitness = metrics.CalculateCompositeFitness();

        Assert.Equal(expected, fitness, LooseTolerance);
    }

    [Fact]
    public void CompressionMetrics_IsBetterThan_HigherFitnessWins()
    {
        var good = new CompressionMetrics<double>
        {
            OriginalSize = 1000,
            CompressedSize = 100,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.94
        };
        good.CalculateDerivedMetrics();

        var bad = new CompressionMetrics<double>
        {
            OriginalSize = 1000,
            CompressedSize = 900,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.85
        };
        bad.CalculateDerivedMetrics();

        Assert.True(good.IsBetterThan(bad));
        Assert.False(bad.IsBetterThan(good));
    }

    [Fact]
    public void CompressionMetrics_IsBetterThan_NullIsAlwaysWorse()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 50
        };
        metrics.CalculateDerivedMetrics();

        Assert.True(metrics.IsBetterThan(null));
    }

    #endregion

    #region CompressionMetrics - SparseCompressionResult Integration

    [Fact]
    public void SparseCompressionResult_Sparsity_HighDimensional()
    {
        // 4D tensor: [2, 3, 4, 5] = 120 total elements
        // 30 non-zero values => sparsity = 1 - 30/120 = 0.75
        var result = new SparseCompressionResult<double>
        {
            Format = SparseFormat.COO,
            Values = new double[30],
            RowIndices = new int[30],
            ColumnIndices = new int[30],
            OriginalShape = new[] { 2, 3, 4, 5 }
        };

        Assert.Equal(30, result.NonZeroCount);
        Assert.Equal(0.75, result.Sparsity, Tolerance);
    }

    [Fact]
    public void SparseCompressionResult_CompressedSizeBytes_AllFormats()
    {
        // COO format: Values + RowIndices + ColumnIndices + metadata
        var cooResult = new SparseCompressionResult<double>
        {
            Format = SparseFormat.COO,
            Values = new double[10],
            RowIndices = new int[10],
            ColumnIndices = new int[10],
            OriginalShape = new[] { 5, 8 }
        };

        // elementSize = 8 (double)
        // Values: 10 * 8 = 80
        // RowIndices: 10 * 4 = 40
        // ColumnIndices: 10 * 4 = 40
        // Metadata: (2 + 4) * 4 = 24  (shape[2] + 4 metadata ints)
        // Total = 80 + 40 + 40 + 24 = 184
        Assert.Equal(184, cooResult.GetCompressedSizeBytes(8));
    }

    [Fact]
    public void SparseCompressionResult_CompressedSizeBytes_WithSparsityMask()
    {
        var result = new SparseCompressionResult<double>
        {
            Format = SparseFormat.Structured2to4,
            Values = new double[4],
            SparsityMask = new byte[] { 0b1010, 0b0101 },
            SparsityN = 2,
            SparsityM = 4,
            OriginalShape = new[] { 8 }
        };

        // Values: 4 * 8 = 32
        // SparsityMask: 2 * 1 = 2
        // Metadata: (1 + 4) * 4 = 20  (shape[1] + 4 metadata ints)
        // Total = 32 + 2 + 20 = 54
        Assert.Equal(54, result.GetCompressedSizeBytes(8));
    }

    #endregion

    #region WeightClusteringCompression - Round-Trip

    [Fact]
    public void WeightClustering_CompressDecompress_WellSeparatedClusters()
    {
        // Weights with 3 clear clusters: ~0.0, ~0.5, ~1.0
        var weights = new Vector<double>(new double[]
        {
            0.01, 0.02, 0.0, 0.03, -0.01,     // cluster near 0
            0.49, 0.51, 0.50, 0.48, 0.52,      // cluster near 0.5
            0.99, 1.0, 1.01, 0.98, 1.02        // cluster near 1.0
        });

        var compressor = new WeightClusteringCompression<double>(
            numClusters: 3,
            maxIterations: 100,
            tolerance: 1e-8,
            randomSeed: 42);

        var (compressed, metadata) = compressor.Compress(weights);
        var decompressed = compressor.Decompress(compressed, metadata);

        // Decompressed values should be cluster centers, close to original
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.True(Math.Abs(decompressed[i] - weights[i]) < 0.1,
                $"Weight {i}: original={weights[i]}, decompressed={decompressed[i]}, diff={Math.Abs(decompressed[i] - weights[i])}");
        }
    }

    [Fact]
    public void WeightClustering_ClusterCentersConvergeToMean()
    {
        // 2 clusters with well-separated data:
        // Cluster A: [1.0, 1.0, 1.0] => center should converge to 1.0
        // Cluster B: [5.0, 5.0, 5.0] => center should converge to 5.0
        var weights = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 5.0, 5.0, 5.0 });

        var compressor = new WeightClusteringCompression<double>(
            numClusters: 2,
            maxIterations: 100,
            tolerance: 1e-10,
            randomSeed: 42);

        var (compressed, metadata) = compressor.Compress(weights);
        var decompressed = compressor.Decompress(compressed, metadata);

        // All first-group weights should decompress to the same cluster center
        Assert.Equal(decompressed[0], decompressed[1], Tolerance);
        Assert.Equal(decompressed[1], decompressed[2], Tolerance);

        // All second-group weights should decompress to the same cluster center
        Assert.Equal(decompressed[3], decompressed[4], Tolerance);
        Assert.Equal(decompressed[4], decompressed[5], Tolerance);

        // Cluster centers should be close to 1.0 and 5.0
        Assert.True(Math.Abs(decompressed[0] - 1.0) < 0.01);
        Assert.True(Math.Abs(decompressed[3] - 5.0) < 0.01);
    }

    [Fact]
    public void WeightClustering_SingleCluster_AllWeightsBecomeTheMean()
    {
        // With k=1, all weights should map to the single cluster center (the mean)
        var weights = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var compressor = new WeightClusteringCompression<double>(
            numClusters: 1,
            maxIterations: 100,
            tolerance: 1e-10,
            randomSeed: 42);

        var (compressed, metadata) = compressor.Compress(weights);
        var decompressed = compressor.Decompress(compressed, metadata);

        // Mean = (1+2+3+4+5)/5 = 3.0
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(3.0, decompressed[i], Tolerance);
        }
    }

    [Fact]
    public void WeightClustering_KEqualsN_PerfectReconstruction()
    {
        // When number of clusters equals number of unique weights, reconstruction should be exact
        var weights = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });

        var compressor = new WeightClusteringCompression<double>(
            numClusters: 4,
            maxIterations: 100,
            tolerance: 1e-10,
            randomSeed: 42);

        var (compressed, metadata) = compressor.Compress(weights);
        var decompressed = compressor.Decompress(compressed, metadata);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(weights[i], decompressed[i], Tolerance);
        }
    }

    [Fact]
    public void WeightClustering_Reproducible_WithSeed()
    {
        var weights = new Vector<double>(new double[] { 0.1, 0.5, 0.3, 0.9, 0.7 });

        var c1 = new WeightClusteringCompression<double>(numClusters: 2, randomSeed: 99);
        var c2 = new WeightClusteringCompression<double>(numClusters: 2, randomSeed: 99);

        var (compressed1, meta1) = c1.Compress(weights);
        var (compressed2, meta2) = c2.Compress(weights);

        var d1 = c1.Decompress(compressed1, meta1);
        var d2 = c2.Decompress(compressed2, meta2);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(d1[i], d2[i], Tolerance);
        }
    }

    [Fact]
    public void WeightClustering_CompressedSize_SmallerThanOriginal()
    {
        // With 4 clusters and many weights, compressed size should be smaller
        var data = new double[100];
        for (int i = 0; i < 100; i++)
            data[i] = (i % 4) * 0.25; // 4 distinct values
        var weights = new Vector<double>(data);

        var compressor = new WeightClusteringCompression<double>(numClusters: 4, randomSeed: 42);
        var (compressed, metadata) = compressor.Compress(weights);

        long compressedSize = compressor.GetCompressedSize(compressed, metadata);
        long originalSize = weights.Length * sizeof(double); // 100 * 8 = 800 bytes

        Assert.True(compressedSize < originalSize,
            $"Compressed size ({compressedSize}) should be smaller than original ({originalSize})");
    }

    #endregion

    #region WeightClusteringMetadata

    [Fact]
    public void WeightClusteringMetadata_GetMetadataSize_ExactCalculation()
    {
        // 4 cluster centers (double = 8 bytes each) + numClusters (4 bytes) + originalLength (4 bytes)
        var centers = new double[] { 0.0, 0.25, 0.5, 0.75 };
        var metadata = new WeightClusteringMetadata<double>(centers, 4, 100);

        long expectedSize = 4 * 8 + sizeof(int) + sizeof(int); // 32 + 4 + 4 = 40
        Assert.Equal(expectedSize, metadata.GetMetadataSize());
    }

    [Fact]
    public void WeightClusteringMetadata_Properties()
    {
        var centers = new double[] { 1.0, 2.0, 3.0 };
        var metadata = new WeightClusteringMetadata<double>(centers, 3, 50);

        Assert.Equal(3, metadata.NumClusters);
        Assert.Equal(50, metadata.OriginalLength);
        Assert.Equal(3, metadata.ClusterCenters.Length);
        Assert.Equal(CompressionType.WeightClustering, metadata.Type);
    }

    [Fact]
    public void WeightClusteringMetadata_InvalidParameters_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new WeightClusteringMetadata<double>(null, 3, 50));
        Assert.Throws<ArgumentException>(() =>
            new WeightClusteringMetadata<double>(new double[] { 1.0 }, 0, 50));
        Assert.Throws<ArgumentException>(() =>
            new WeightClusteringMetadata<double>(new double[] { 1.0 }, 1, -1));
    }

    #endregion

    #region CompressionMetrics - Edge Cases

    [Fact]
    public void CompressionMetrics_ZeroOriginalSize_NoException()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 0,
            CompressedSize = 50
        };

        // Should not throw, CompressionRatio = 0 (special case)
        metrics.CalculateDerivedMetrics();

        Assert.Equal(0.0, metrics.CompressionRatio, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_EqualSizes_CompressionRatio1()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 500,
            CompressedSize = 500
        };

        metrics.CalculateDerivedMetrics();

        Assert.Equal(1.0, metrics.CompressionRatio, Tolerance);
        Assert.Equal(0.0, metrics.SizeReductionPercentage, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_NoAccuracyLoss_ZeroLoss()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 50,
            OriginalAccuracy = 0.95,
            CompressedAccuracy = 0.95
        };

        metrics.CalculateDerivedMetrics();

        Assert.Equal(0.0, metrics.AccuracyLoss, Tolerance);
    }

    [Fact]
    public void CompressionMetrics_CompositeFitness_OnlyAccuracyWeighted()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 100,
            CompressedSize = 50,
            OriginalAccuracy = 0.9,
            CompressedAccuracy = 0.85
        };
        metrics.CalculateDerivedMetrics();

        // With only accuracy weight = 1.0, others = 0
        // accuracyPreservation = 1.0 - 0.05 = 0.95
        // fitness = 0.95 * 1.0 / 1.0 = 0.95
        double fitness = metrics.CalculateCompositeFitness(1.0, 0.0, 0.0);
        Assert.Equal(0.95, fitness, LooseTolerance);
    }

    [Fact]
    public void CompressionMetrics_CompositeFitness_OnlyCompressionWeighted()
    {
        var metrics = new CompressionMetrics<double>
        {
            OriginalSize = 1000,
            CompressedSize = 100 // 10x compression
        };
        metrics.CalculateDerivedMetrics();

        // compressionScore = 1 - 1/(1 + 10/10) = 1 - 0.5 = 0.5
        // fitness = 0.5 * 1.0 / 1.0 = 0.5
        double fitness = metrics.CalculateCompositeFitness(0.0, 1.0, 0.0);
        Assert.Equal(0.5, fitness, LooseTolerance);
    }

    [Fact]
    public void CompressionMetrics_FromDeepCompressionStats_CorrectMapping()
    {
        var stats = new DeepCompressionStats
        {
            OriginalSizeBytes = 1000,
            CompressedSizeBytes = 200,
            CompressionRatio = 5.0,
            Sparsity = 0.8,
            BitsPerWeight = 4.0
        };

        var metrics = CompressionMetrics<double>.FromDeepCompressionStats(stats, "TestTechnique");

        Assert.Equal(1000, metrics.OriginalSize);
        Assert.Equal(200, metrics.CompressedSize);
        Assert.Equal(5.0, metrics.CompressionRatio, Tolerance);
        Assert.Equal(0.8, metrics.Sparsity, Tolerance);
        Assert.Equal(4.0, metrics.BitsPerWeight, Tolerance);
        Assert.Equal("TestTechnique", metrics.CompressionTechnique);

        // SizeReductionPercentage = (1 - 200/1000) * 100 = 80
        Assert.Equal(80.0, metrics.SizeReductionPercentage, Tolerance);
    }

    #endregion

    #region WeightClusteringCompression - Validation

    [Fact]
    public void WeightClustering_EmptyWeights_Throws()
    {
        var compressor = new WeightClusteringCompression<double>(numClusters: 3);
        Assert.Throws<ArgumentException>(() => compressor.Compress(new Vector<double>(Array.Empty<double>())));
    }

    [Fact]
    public void WeightClustering_NullWeights_Throws()
    {
        var compressor = new WeightClusteringCompression<double>(numClusters: 3);
        Assert.Throws<ArgumentNullException>(() => compressor.Compress(null));
    }

    [Fact]
    public void WeightClustering_InvalidParameters_Throws()
    {
        Assert.Throws<ArgumentException>(() => new WeightClusteringCompression<double>(numClusters: 0));
        Assert.Throws<ArgumentException>(() => new WeightClusteringCompression<double>(maxIterations: 0));
        Assert.Throws<ArgumentException>(() => new WeightClusteringCompression<double>(tolerance: 0));
    }

    [Fact]
    public void WeightClustering_MoreClustersThanWeights_AdjustsDown()
    {
        // 3 weights but k=10 => should adjust to k=3
        var weights = new Vector<double>(new double[] { 1.0, 5.0, 9.0 });

        var compressor = new WeightClusteringCompression<double>(numClusters: 10, randomSeed: 42);
        var (compressed, metadata) = compressor.Compress(weights);
        var decompressed = compressor.Decompress(compressed, metadata);

        // With k=3 and 3 distinct values, should perfectly reconstruct
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(weights[i], decompressed[i], Tolerance);
        }
    }

    #endregion
}
