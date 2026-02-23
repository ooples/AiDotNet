using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="ScanPatterns{T}"/>.
/// </summary>
public class ScanPatternsTests
{
    [Fact]
    public void BidirectionalScan_ProducesCorrectOutputShape()
    {
        int batchSize = 2;
        int numPatches = 16;
        int dim = 8;
        var patches = CreateRandomTensor(new[] { batchSize, numPatches, dim });

        var result = ScanPatterns<float>.BidirectionalScan(patches);

        Assert.Equal(new[] { batchSize, numPatches, dim * 2 }, result.Shape);
    }

    [Fact]
    public void BidirectionalScan_ForwardHalfMatchesInput()
    {
        int batchSize = 1;
        int numPatches = 4;
        int dim = 2;
        var patches = CreateRandomTensor(new[] { batchSize, numPatches, dim });

        var result = ScanPatterns<float>.BidirectionalScan(patches);

        // First half of feature dim should match original
        for (int p = 0; p < numPatches; p++)
        {
            for (int d = 0; d < dim; d++)
            {
                Assert.Equal(patches[new[] { 0, p, d }], result[new[] { 0, p, d }]);
            }
        }
    }

    [Fact]
    public void BidirectionalScan_ReverseHalfMatchesReversed()
    {
        int batchSize = 1;
        int numPatches = 4;
        int dim = 2;
        var patches = CreateRandomTensor(new[] { batchSize, numPatches, dim });

        var result = ScanPatterns<float>.BidirectionalScan(patches);

        // Second half of feature dim should match reversed sequence
        for (int p = 0; p < numPatches; p++)
        {
            int revP = numPatches - 1 - p;
            for (int d = 0; d < dim; d++)
            {
                Assert.Equal(patches[new[] { 0, revP, d }], result[new[] { 0, p, d + dim }]);
            }
        }
    }

    [Fact]
    public void BidirectionalScan_ThrowsFor2DInput()
    {
        var patches = CreateRandomTensor(new[] { 4, 8 });
        Assert.Throws<ArgumentException>(() => ScanPatterns<float>.BidirectionalScan(patches));
    }

    [Fact]
    public void CrossScan_ProducesFourOutputs()
    {
        int batchSize = 2;
        int height = 4;
        int width = 4;
        int numPatches = height * width;
        int dim = 8;
        var patches = CreateRandomTensor(new[] { batchSize, numPatches, dim });

        var results = ScanPatterns<float>.CrossScan(patches, height, width);

        Assert.Equal(4, results.Count);
    }

    [Fact]
    public void CrossScan_AllOutputsHaveCorrectShape()
    {
        int batchSize = 2;
        int height = 3;
        int width = 4;
        int numPatches = height * width;
        int dim = 8;
        var patches = CreateRandomTensor(new[] { batchSize, numPatches, dim });

        var results = ScanPatterns<float>.CrossScan(patches, height, width);

        foreach (var result in results)
        {
            Assert.Equal(new[] { batchSize, numPatches, dim }, result.Shape);
        }
    }

    [Fact]
    public void CrossScan_FirstDirectionMatchesInput()
    {
        int batchSize = 1;
        int height = 2;
        int width = 3;
        int numPatches = height * width;
        int dim = 4;
        var patches = CreateRandomTensor(new[] { batchSize, numPatches, dim });

        var results = ScanPatterns<float>.CrossScan(patches, height, width);

        // Direction 1 (L→R, T→B) should be identity
        for (int i = 0; i < patches.Length; i++)
        {
            Assert.Equal(patches[i], results[0][i]);
        }
    }

    [Fact]
    public void CrossScan_ThrowsOnMismatchedDimensions()
    {
        var patches = CreateRandomTensor(new[] { 1, 12, 4 });

        Assert.Throws<ArgumentException>(() => ScanPatterns<float>.CrossScan(patches, 3, 5));
    }

    [Fact]
    public void ContinuousScan_ProducesCorrectOutputShape()
    {
        int batchSize = 2;
        int height = 3;
        int width = 4;
        int numPatches = height * width;
        int dim = 8;
        var patches = CreateRandomTensor(new[] { batchSize, numPatches, dim });

        var result = ScanPatterns<float>.ContinuousScan(patches, height, width);

        Assert.Equal(new[] { batchSize, numPatches, dim }, result.Shape);
    }

    [Fact]
    public void ContinuousScan_EvenRowLeftToRight()
    {
        int batchSize = 1;
        int height = 2;
        int width = 3;
        int dim = 1;
        var patches = CreateSequentialTensor(new[] { batchSize, height * width, dim });

        var result = ScanPatterns<float>.ContinuousScan(patches, height, width);

        // Row 0 (even): left to right -> indices 0, 1, 2
        Assert.Equal(patches[new[] { 0, 0, 0 }], result[new[] { 0, 0, 0 }]);
        Assert.Equal(patches[new[] { 0, 1, 0 }], result[new[] { 0, 1, 0 }]);
        Assert.Equal(patches[new[] { 0, 2, 0 }], result[new[] { 0, 2, 0 }]);
    }

    [Fact]
    public void ContinuousScan_OddRowRightToLeft()
    {
        int batchSize = 1;
        int height = 2;
        int width = 3;
        int dim = 1;
        var patches = CreateSequentialTensor(new[] { batchSize, height * width, dim });

        var result = ScanPatterns<float>.ContinuousScan(patches, height, width);

        // Row 1 (odd): right to left -> indices 5, 4, 3
        Assert.Equal(patches[new[] { 0, 5, 0 }], result[new[] { 0, 3, 0 }]);
        Assert.Equal(patches[new[] { 0, 4, 0 }], result[new[] { 0, 4, 0 }]);
        Assert.Equal(patches[new[] { 0, 3, 0 }], result[new[] { 0, 5, 0 }]);
    }

    [Fact]
    public void SpatioTemporalScan_ProducesTwoOutputs()
    {
        int batchSize = 1;
        int height = 2;
        int width = 2;
        int numFrames = 3;
        int totalPatches = height * width * numFrames;
        int dim = 4;
        var frames = CreateRandomTensor(new[] { batchSize, totalPatches, dim });

        var results = ScanPatterns<float>.SpatioTemporalScan(frames, height, width, numFrames);

        Assert.Equal(2, results.Count);
        Assert.Equal(new[] { batchSize, totalPatches, dim }, results[0].Shape);
        Assert.Equal(new[] { batchSize, totalPatches, dim }, results[1].Shape);
    }

    [Fact]
    public void SpatioTemporalScan_SpatialMatchesInput()
    {
        int batchSize = 1;
        int height = 2;
        int width = 2;
        int numFrames = 2;
        int totalPatches = height * width * numFrames;
        int dim = 4;
        var frames = CreateRandomTensor(new[] { batchSize, totalPatches, dim });

        var results = ScanPatterns<float>.SpatioTemporalScan(frames, height, width, numFrames);

        // Spatial scan should be identity
        for (int i = 0; i < frames.Length; i++)
        {
            Assert.Equal(frames[i], results[0][i]);
        }
    }

    [Fact]
    public void SpatioTemporalScan_ThrowsOnDimensionMismatch()
    {
        var frames = CreateRandomTensor(new[] { 1, 10, 4 });

        Assert.Throws<ArgumentException>(() =>
            ScanPatterns<float>.SpatioTemporalScan(frames, 2, 2, 3)); // 2*2*3=12 != 10
    }

    [Fact]
    public void MergeScanOutputs_AveragesCorrectly()
    {
        int batchSize = 1;
        int numPatches = 4;
        int dim = 2;

        var output1 = new Tensor<float>(new[] { batchSize, numPatches, dim });
        var output2 = new Tensor<float>(new[] { batchSize, numPatches, dim });
        output1.Fill(2.0f);
        output2.Fill(4.0f);

        var merged = ScanPatterns<float>.MergeScanOutputs(new List<Tensor<float>> { output1, output2 });

        Assert.Equal(new[] { batchSize, numPatches, dim }, merged.Shape);
        // Average of 2 and 4 should be 3
        for (int i = 0; i < merged.Length; i++)
        {
            Assert.Equal(3.0f, merged[i], 5);
        }
    }

    [Fact]
    public void MergeScanOutputs_ThrowsOnEmpty()
    {
        Assert.Throws<ArgumentException>(() =>
            ScanPatterns<float>.MergeScanOutputs(new List<Tensor<float>>()));
    }

    [Fact]
    public void MergeScanOutputs_ThrowsOnMismatchedShapes()
    {
        var output1 = CreateRandomTensor(new[] { 1, 4, 8 });
        var output2 = CreateRandomTensor(new[] { 1, 4, 16 });

        Assert.Throws<ArgumentException>(() =>
            ScanPatterns<float>.MergeScanOutputs(new List<Tensor<float>> { output1, output2 }));
    }

    [Fact]
    public void CrossScan_MergeRoundTrip_PreservesShape()
    {
        int batchSize = 1;
        int height = 3;
        int width = 3;
        int numPatches = height * width;
        int dim = 4;
        var patches = CreateRandomTensor(new[] { batchSize, numPatches, dim });

        var scans = ScanPatterns<float>.CrossScan(patches, height, width);
        var merged = ScanPatterns<float>.MergeScanOutputs(scans);

        Assert.Equal(new[] { batchSize, numPatches, dim }, merged.Shape);
    }

    [Fact]
    public void BidirectionalScan_Double_ProducesValidOutput()
    {
        int batchSize = 1;
        int numPatches = 8;
        int dim = 4;
        var patches = CreateRandomDoubleTensor(new[] { batchSize, numPatches, dim });

        var result = ScanPatterns<double>.BidirectionalScan(patches);

        Assert.Equal(new[] { batchSize, numPatches, dim * 2 }, result.Shape);
    }

    #region Helpers

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    private static Tensor<float> CreateSequentialTensor(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i;
        }
        return tensor;
    }

    private static Tensor<double> CreateRandomDoubleTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<double>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.NextDouble() * 2 - 1;
        }
        return tensor;
    }

    #endregion
}
