using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class SparseTensorTests
{
    [Fact]
    public void FromDense_ToDense_RoundTrip()
    {
        var dense = new Tensor<double>(new[] { 2, 3 });
        dense[0, 1] = 2.0;
        dense[1, 2] = -1.5;

        var sparse = SparseTensor<double>.FromDense(dense);
        var roundTrip = sparse.ToDense();

        Assert.Equal(2.0, roundTrip[0, 1], precision: 12);
        Assert.Equal(-1.5, roundTrip[1, 2], precision: 12);
        Assert.Equal(0.0, roundTrip[0, 0], precision: 12);
    }

    [Fact]
    public void Transpose_SwapsIndices()
    {
        var sparse = new SparseTensor<double>(
            2,
            3,
            new[] { 0, 1 },
            new[] { 1, 2 },
            new[] { 2.0, 3.0 });

        var transposed = sparse.Transpose();
        var dense = transposed.ToDense();

        Assert.Equal(3, transposed.Rows);
        Assert.Equal(2, transposed.Columns);
        Assert.Equal(2.0, dense[1, 0], precision: 12);
        Assert.Equal(3.0, dense[2, 1], precision: 12);
    }

    [Fact]
    public void Coalesce_MergesDuplicates()
    {
        var sparse = new SparseTensor<double>(
            2,
            2,
            new[] { 0, 0 },
            new[] { 1, 1 },
            new[] { 1.0, 2.0 });

        var coalesced = sparse.Coalesce();

        Assert.Equal(1, coalesced.NonZeroCount);
        Assert.Equal(3.0, coalesced.Values[0], precision: 12);
        Assert.Equal(0, coalesced.RowIndices[0]);
        Assert.Equal(1, coalesced.ColumnIndices[0]);
    }
}
