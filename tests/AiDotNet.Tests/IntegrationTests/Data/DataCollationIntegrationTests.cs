using AiDotNet.Data.Collation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

/// <summary>
/// Integration tests for data collation classes:
/// DefaultCollateFunction, PaddingCollateFunction, PackedSequenceCollateFunction,
/// PackedSequenceBatch.
/// </summary>
public class DataCollationIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region DefaultCollateFunction

    [Fact]
    public void DefaultCollate_StacksSameSizeSamples()
    {
        var collate = new DefaultCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 })),
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 4, 5, 6 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(new[] { 2, 3 }, batch.Shape);
        Assert.Equal(1.0, batch[0, 0], Tolerance);
        Assert.Equal(4.0, batch[1, 0], Tolerance);
        Assert.Equal(6.0, batch[1, 2], Tolerance);
    }

    [Fact]
    public void DefaultCollate_2DSamples_StacksCorrectly()
    {
        var collate = new DefaultCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 1, 2, 3, 4 })),
            new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 5, 6, 7, 8 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(new[] { 2, 2, 2 }, batch.Shape);
    }

    [Fact]
    public void DefaultCollate_SingleSample_Works()
    {
        var collate = new DefaultCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 1, 2, 3, 4 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(new[] { 1, 4 }, batch.Shape);
        Assert.Equal(1.0, batch[0, 0], Tolerance);
    }

    [Fact]
    public void DefaultCollate_EmptyList_Throws()
    {
        var collate = new DefaultCollateFunction<double>();
        Assert.Throws<ArgumentException>(() =>
            collate.Collate(new List<Tensor<double>>()));
    }

    [Fact]
    public void DefaultCollate_MismatchedRanks_Throws()
    {
        var collate = new DefaultCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 })),
            new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[] { 4, 5, 6 }))
        };

        Assert.Throws<ArgumentException>(() => collate.Collate(samples));
    }

    [Fact]
    public void DefaultCollate_MismatchedShapes_Throws()
    {
        var collate = new DefaultCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 })),
            new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 4, 5, 6, 7 }))
        };

        Assert.Throws<ArgumentException>(() => collate.Collate(samples));
    }

    #endregion

    #region PaddingCollateFunction

    [Fact]
    public void PaddingCollate_PadsToMaxLength()
    {
        var collate = new PaddingCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 })),
            new Tensor<double>(new[] { 5 }, new Vector<double>(new double[] { 4, 5, 6, 7, 8 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(new[] { 2, 5 }, batch.Shape);
        // First sample padded with zeros
        Assert.Equal(1.0, batch[0, 0], Tolerance);
        Assert.Equal(3.0, batch[0, 2], Tolerance);
        Assert.Equal(0.0, batch[0, 3], Tolerance); // padding
        Assert.Equal(0.0, batch[0, 4], Tolerance); // padding
        // Second sample no padding
        Assert.Equal(8.0, batch[1, 4], Tolerance);
    }

    [Fact]
    public void PaddingCollate_WithCustomPadValue()
    {
        var collate = new PaddingCollateFunction<double>(-1.0);
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 1, 2 })),
            new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 3, 4, 5, 6 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(new[] { 2, 4 }, batch.Shape);
        Assert.Equal(-1.0, batch[0, 2], Tolerance); // custom pad
        Assert.Equal(-1.0, batch[0, 3], Tolerance); // custom pad
    }

    [Fact]
    public void PaddingCollate_WithMaxLength_Truncates()
    {
        var collate = new PaddingCollateFunction<double>(maxLength: 3);
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 5 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5 })),
            new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 6, 7 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(new[] { 2, 3 }, batch.Shape);
        // First sample truncated to 3 elements
        Assert.Equal(1.0, batch[0, 0], Tolerance);
        Assert.Equal(3.0, batch[0, 2], Tolerance);
    }

    [Fact]
    public void PaddingCollate_SameLengthSamples_NoPadding()
    {
        var collate = new PaddingCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 })),
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 4, 5, 6 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(new[] { 2, 3 }, batch.Shape);
        Assert.Equal(1.0, batch[0, 0], Tolerance);
        Assert.Equal(6.0, batch[1, 2], Tolerance);
    }

    [Fact]
    public void PaddingCollate_EmptyList_Throws()
    {
        var collate = new PaddingCollateFunction<double>();
        Assert.Throws<ArgumentException>(() =>
            collate.Collate(new List<Tensor<double>>()));
    }

    #endregion

    #region PackedSequenceCollateFunction

    [Fact]
    public void PackedSequence_PacksSequences()
    {
        var collate = new PackedSequenceCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 })),
            new Tensor<double>(new[] { 5 }, new Vector<double>(new double[] { 4, 5, 6, 7, 8 })),
            new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 9, 10 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(3, batch.BatchSize);
        Assert.Equal(10, batch.Data.Length); // 3 + 5 + 2
        // Sorted by length descending
        Assert.Equal(5, batch.Lengths[0]); // longest first
        Assert.Equal(3, batch.Lengths[1]);
        Assert.Equal(2, batch.Lengths[2]); // shortest last
    }

    [Fact]
    public void PackedSequence_SortedIndices_MapCorrectly()
    {
        var collate = new PackedSequenceCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 1, 2 })),         // index 0, len 2
            new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 3, 4, 5, 6 })),   // index 1, len 4
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 7, 8, 9 }))       // index 2, len 3
        };

        var batch = collate.Collate(samples);

        // Sorted: index 1 (len 4), index 2 (len 3), index 0 (len 2)
        Assert.Equal(1, batch.SortedIndices[0]);
        Assert.Equal(2, batch.SortedIndices[1]);
        Assert.Equal(0, batch.SortedIndices[2]);
    }

    [Fact]
    public void PackedSequence_WithoutSort_PreservesOrder()
    {
        var collate = new PackedSequenceCollateFunction<double>(sortByLength: false);
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 1, 2 })),
            new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 3, 4, 5, 6 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(0, batch.SortedIndices[0]);
        Assert.Equal(1, batch.SortedIndices[1]);
        Assert.Equal(2, batch.Lengths[0]);
        Assert.Equal(4, batch.Lengths[1]);
    }

    [Fact]
    public void PackedSequence_EmptyList_Throws()
    {
        var collate = new PackedSequenceCollateFunction<double>();
        Assert.Throws<ArgumentException>(() =>
            collate.Collate(new List<Tensor<double>>()));
    }

    [Fact]
    public void PackedSequence_SingleSample_Works()
    {
        var collate = new PackedSequenceCollateFunction<double>();
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }))
        };

        var batch = collate.Collate(samples);

        Assert.Equal(1, batch.BatchSize);
        Assert.Equal(3, batch.Lengths[0]);
        Assert.Equal(3, batch.Data.Length);
    }

    #endregion

    #region PackedSequenceBatch

    [Fact]
    public void PackedSequenceBatch_Properties()
    {
        var data = new Tensor<double>(new[] { 6 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        var lengths = new[] { 3, 2, 1 };
        var sortedIndices = new[] { 0, 1, 2 };

        var batch = new PackedSequenceBatch<double>(data, lengths, sortedIndices);

        Assert.Equal(3, batch.BatchSize);
        Assert.Same(data, batch.Data);
        Assert.Equal(new[] { 3, 2, 1 }, batch.Lengths);
        Assert.Equal(new[] { 0, 1, 2 }, batch.SortedIndices);
    }

    #endregion
}
