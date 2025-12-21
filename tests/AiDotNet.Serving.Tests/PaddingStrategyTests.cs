using AiDotNet.LinearAlgebra;
using AiDotNet.Serving.Padding;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for padding strategies.
/// </summary>
public class PaddingStrategyTests
{
    [Fact]
    public void MinimalPaddingStrategy_ShouldPadToMaxLength()
    {
        // Arrange
        var strategy = new MinimalPaddingStrategy();
        var vectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3 }),
            new Vector<double>(new double[] { 4, 5 }),
            new Vector<double>(new double[] { 6, 7, 8, 9 })
        };

        // Act
        var paddedMatrix = strategy.PadBatch(vectors, out var attentionMask);

        // Assert
        Assert.Equal(3, paddedMatrix.Rows); // 3 vectors
        Assert.Equal(4, paddedMatrix.Columns); // Max length is 4
        Assert.NotNull(attentionMask);
        Assert.Equal(3, attentionMask!.Rows);
        Assert.Equal(4, attentionMask.Columns);

        // Check padding values
        Assert.Equal(0.0, paddedMatrix[0, 3]); // First vector padded
        Assert.Equal(0.0, paddedMatrix[1, 2]); // Second vector padded
        Assert.Equal(0.0, paddedMatrix[1, 3]); // Second vector padded

        // Check attention mask
        Assert.Equal(1.0, attentionMask[0, 0]); // Actual data
        Assert.Equal(0.0, attentionMask[0, 3]); // Padding
        Assert.Equal(1.0, attentionMask[2, 3]); // Actual data (last element)
    }

    [Fact]
    public void MinimalPaddingStrategy_ShouldUnpadCorrectly()
    {
        // Arrange
        var strategy = new MinimalPaddingStrategy();
        var originalVectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3 }),
            new Vector<double>(new double[] { 4, 5 }),
            new Vector<double>(new double[] { 6, 7, 8, 9 })
        };

        var paddedMatrix = strategy.PadBatch(originalVectors, out _);
        var originalLengths = new[] { 3, 2, 4 };

        // Act
        var unpaddedVectors = strategy.UnpadBatch(paddedMatrix, originalLengths);

        // Assert
        Assert.Equal(3, unpaddedVectors.Length);
        Assert.Equal(3, unpaddedVectors[0].Length);
        Assert.Equal(2, unpaddedVectors[1].Length);
        Assert.Equal(4, unpaddedVectors[2].Length);

        Assert.Equal(1.0, unpaddedVectors[0][0]);
        Assert.Equal(5.0, unpaddedVectors[1][1]);
        Assert.Equal(9.0, unpaddedVectors[2][3]);
    }

    [Fact]
    public void BucketPaddingStrategy_ShouldPadToNearestBucket()
    {
        // Arrange
        var bucketSizes = new[] { 8, 16, 32, 64 };
        var strategy = new BucketPaddingStrategy(bucketSizes);
        var vectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3 }),
            new Vector<double>(new double[] { 4, 5 }),
            new Vector<double>(new double[] { 6, 7, 8, 9, 10 })
        };

        // Act
        var paddedMatrix = strategy.PadBatch(vectors, out var attentionMask);

        // Assert
        Assert.Equal(3, paddedMatrix.Rows);
        Assert.Equal(8, paddedMatrix.Columns); // Nearest bucket to max length 5 is 8
        Assert.NotNull(attentionMask);
    }

    [Fact]
    public void BucketPaddingStrategy_ShouldUnpadCorrectly()
    {
        // Arrange
        var bucketSizes = new[] { 8, 16, 32, 64 };
        var strategy = new BucketPaddingStrategy(bucketSizes);
        var originalVectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3 }),
            new Vector<double>(new double[] { 4, 5 })
        };

        var paddedMatrix = strategy.PadBatch(originalVectors, out _);
        var originalLengths = new[] { 3, 2 };

        // Act
        var unpaddedVectors = strategy.UnpadBatch(paddedMatrix, originalLengths);

        // Assert
        Assert.Equal(2, unpaddedVectors.Length);
        Assert.Equal(3, unpaddedVectors[0].Length);
        Assert.Equal(2, unpaddedVectors[1].Length);
    }

    [Fact]
    public void FixedSizePaddingStrategy_ShouldPadToFixedSize()
    {
        // Arrange
        var strategy = new FixedSizePaddingStrategy(fixedLength: 10);
        var vectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3 }),
            new Vector<double>(new double[] { 4, 5 }),
            new Vector<double>(new double[] { 6, 7, 8, 9 })
        };

        // Act
        var paddedMatrix = strategy.PadBatch(vectors, out var attentionMask);

        // Assert
        Assert.Equal(3, paddedMatrix.Rows);
        Assert.Equal(10, paddedMatrix.Columns); // Fixed size
        Assert.NotNull(attentionMask);
    }

    [Fact]
    public void FixedSizePaddingStrategy_ShouldThrow_WhenVectorExceedsFixedSize()
    {
        // Arrange
        var strategy = new FixedSizePaddingStrategy(fixedLength: 5);
        var vectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }) // Length 8 > 5
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => strategy.PadBatch(vectors, out _));
    }

    [Fact]
    public void FixedSizePaddingStrategy_ShouldUnpadCorrectly()
    {
        // Arrange
        var strategy = new FixedSizePaddingStrategy(fixedLength: 10);
        var originalVectors = new[]
        {
            new Vector<double>(new double[] { 1, 2, 3 }),
            new Vector<double>(new double[] { 4, 5, 6, 7 })
        };

        var paddedMatrix = strategy.PadBatch(originalVectors, out _);
        var originalLengths = new[] { 3, 4 };

        // Act
        var unpaddedVectors = strategy.UnpadBatch(paddedMatrix, originalLengths);

        // Assert
        Assert.Equal(2, unpaddedVectors.Length);
        Assert.Equal(3, unpaddedVectors[0].Length);
        Assert.Equal(4, unpaddedVectors[1].Length);
    }

    [Fact]
    public void PaddingStrategies_ShouldThrow_WhenVectorsArrayEmpty()
    {
        // Arrange
        var minimalStrategy = new MinimalPaddingStrategy();
        var bucketStrategy = new BucketPaddingStrategy(new[] { 8, 16, 32 });
        var fixedStrategy = new FixedSizePaddingStrategy(10);
        var emptyVectors = Array.Empty<Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => minimalStrategy.PadBatch(emptyVectors, out _));
        Assert.Throws<ArgumentException>(() => bucketStrategy.PadBatch(emptyVectors, out _));
        Assert.Throws<ArgumentException>(() => fixedStrategy.PadBatch(emptyVectors, out _));
    }
}
