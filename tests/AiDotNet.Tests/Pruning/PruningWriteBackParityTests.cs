using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Pruning;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Pruning;

/// <summary>
/// Pins <c>ApplyPruning</c> of every pruning strategy to the element-by-element write-back it replaced: the weights
/// end up holding exactly what the mask produced, in place, for vectors, matrices and tensors, in float and double.
/// </summary>
public class PruningWriteBackParityTests
{
    private static readonly double[] Values =
    {
        0.75, -1.5, 0.0, 3.25, -0.0, 1e-30, -2.5e12, double.NaN, double.PositiveInfinity, 0.1, -0.3, 7.0,
    };

    public static IEnumerable<object[]> DoubleStrategies() => new[]
    {
        new object[] { new MagnitudePruningStrategy<double>() },
        new object[] { new GradientPruningStrategy<double>() },
        new object[] { new LotteryTicketPruningStrategy<double>() },
        new object[] { new StructuredPruningStrategy<double>() },
    };

    public static IEnumerable<object[]> FloatStrategies() => new[]
    {
        new object[] { new MagnitudePruningStrategy<float>() },
        new object[] { new GradientPruningStrategy<float>() },
        new object[] { new LotteryTicketPruningStrategy<float>() },
        new object[] { new StructuredPruningStrategy<float>() },
    };

    private static bool[] Keep(int count) => Enumerable.Range(0, count).Select(i => i % 3 != 1).ToArray();

    [Theory]
    [MemberData(nameof(DoubleStrategies))]
    public void Vector_Double_HoldsTheMaskedValuesInPlace(IPruningStrategy<double> strategy)
    {
        var weights = new Vector<double>(Values.ToArray());
        var mask = new PruningMask<double>(Keep(Values.Length));
        var expected = mask.Apply(new Vector<double>(Values.ToArray())).ToArray();

        strategy.ApplyPruning(weights, mask);

        AssertBitwise(expected, weights.ToArray());
    }

    [Theory]
    [MemberData(nameof(FloatStrategies))]
    public void Vector_Float_HoldsTheMaskedValuesInPlace(IPruningStrategy<float> strategy)
    {
        var values = Values.Select(v => (float)v).ToArray();
        var weights = new Vector<float>(values.ToArray());
        var mask = new PruningMask<float>(Keep(values.Length));
        var expected = mask.Apply(new Vector<float>(values.ToArray())).ToArray();

        strategy.ApplyPruning(weights, mask);

        AssertBitwise(expected, weights.ToArray());
    }

    [Theory]
    [MemberData(nameof(DoubleStrategies))]
    public void Matrix_Double_HoldsTheMaskedValuesInPlace(IPruningStrategy<double> strategy)
    {
        var weights = MatrixOf(Values, 3, 4);
        var mask = new PruningMask<double>(Keep2D(3, 4));
        var applied = mask.Apply(MatrixOf(Values, 3, 4));

        strategy.ApplyPruning(weights, mask);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(applied[i, j]), BitConverter.DoubleToInt64Bits(weights[i, j]));
            }
        }
    }

    [Theory]
    [MemberData(nameof(FloatStrategies))]
    public void Matrix_Float_HoldsTheMaskedValuesInPlace(IPruningStrategy<float> strategy)
    {
        var values = Values.Select(v => (float)v).ToArray();
        var weights = MatrixOf(values, 3, 4);
        var mask = new PruningMask<float>(Keep2D(3, 4));
        var applied = mask.Apply(MatrixOf(values, 3, 4));

        strategy.ApplyPruning(weights, mask);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.Equal(
                    BitConverter.DoubleToInt64Bits(applied[i, j]), BitConverter.DoubleToInt64Bits(weights[i, j]));
            }
        }
    }

    [Theory]
    [MemberData(nameof(DoubleStrategies))]
    public void Tensor_Double_HoldsTheMaskedValuesInPlace_AndIsMarkedChanged(IPruningStrategy<double> strategy)
    {
        var weights = new Tensor<double>(Values.ToArray(), new[] { 3, 4 });
        var mask = new PruningMask<double>(Keep2D(3, 4));
        var expected = mask.Apply(new Tensor<double>(Values.ToArray(), new[] { 3, 4 })).ToArray();
        var versionBefore = weights.Version;

        strategy.ApplyPruning(weights, mask);

        AssertBitwise(expected, weights.ToArray());
        Assert.True(weights.Version == versionBefore + 1, $"one block write bumps the version once (was {versionBefore} -> {weights.Version})");
    }

    [Theory]
    [MemberData(nameof(FloatStrategies))]
    public void Tensor_Float_HoldsTheMaskedValuesInPlace(IPruningStrategy<float> strategy)
    {
        var values = Values.Select(v => (float)v).ToArray();
        var weights = new Tensor<float>(values.ToArray(), new[] { 3, 4 });
        var mask = new PruningMask<float>(Keep2D(3, 4));
        var expected = mask.Apply(new Tensor<float>(values.ToArray(), new[] { 3, 4 })).ToArray();
        var versionBefore = weights.Version;

        strategy.ApplyPruning(weights, mask);

        AssertBitwise(expected, weights.ToArray());
        Assert.True(weights.Version == versionBefore + 1, $"one block write bumps the version once (was {versionBefore} -> {weights.Version})");
    }

    [Fact]
    public void Tensor_SparseMaskResult_IsWrittenBackDensely()
    {
        // A custom mask may return a SparseTensor of the destination's shape: the diagonal of a 2 x 2.
        var weights = new Tensor<double>(new[] { 9.0, 9.0, 9.0, 9.0 }, new[] { 2, 2 });
        var pruned = new SparseTensor<double>(2, 2, new[] { 0, 1 }, new[] { 0, 1 }, new[] { 3.0, 4.0 });

        PruningWriteBack.Copy(pruned, weights);

        AssertBitwise(new[] { 3.0, 0.0, 0.0, 4.0 }, weights.ToArray());
    }

    [Fact]
    public void Tensor_StridedDestinationSharingTheSourceStorage_ReadsEverySourceValueFirst()
    {
        // The destination is the transposed [2, 3] view of a [3, 2] array, the source the same six elements read flat.
        // Destination element 1 is storage slot 2, so a write-back without a snapshot overwrites source value 2 (7)
        // before it is read.
        var backing = new[] { 5.0, 0.0, 7.0, 0.0, 7.0, 0.0 };
        var destination = Tensor<double>.FromMemory(new Memory<double>(backing), new[] { 3, 2 }).Transpose(new[] { 1, 0 });
        Assert.False(destination.IsContiguous, "the test needs a strided destination");
        var source = Tensor<double>.FromMemory(new Memory<double>(backing), new[] { 2, 3 });
        var expected = source.ToArray();

        PruningWriteBack.Copy(source, destination);

        AssertBitwise(expected, destination.ToArray());
    }

    [Fact]
    public void Tensor_StridedView_IsWrittenThroughToItsOwnElements()
    {
        // A transposed view is not contiguous: the write-back must land on the view's elements, not on the first
        // twelve storage slots.
        var storage = new Tensor<double>(Enumerable.Range(1, 12).Select(i => (double)i).ToArray(), new[] { 3, 4 });
        var view = storage.Transpose(new[] { 1, 0 });
        Assert.False(view.IsContiguous, "the test needs a strided view");
        var before = view.ToArray();
        var mask = new PruningMask<double>(Keep2D(4, 3));
        var expected = mask.Apply(new Tensor<double>(before.ToArray(), new[] { 4, 3 })).ToArray();

        new MagnitudePruningStrategy<double>().ApplyPruning(view, mask);

        AssertBitwise(expected, view.ToArray());
    }

    private static bool[,] Keep2D(int rows, int columns)
    {
        var keep = new bool[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                keep[i, j] = (i * columns + j) % 3 != 1;
            }
        }

        return keep;
    }

    private static Matrix<T> MatrixOf<T>(T[] values, int rows, int columns)
    {
        var matrix = new Matrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = values[i * columns + j];
            }
        }

        return matrix;
    }

    private static void AssertBitwise(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(actual[i]));
        }
    }

    private static void AssertBitwise(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            // Float bits, not widened doubles: widening is exact for finite values but need not keep a NaN payload.
            Assert.Equal(
                BitConverter.ToInt32(BitConverter.GetBytes(expected[i]), 0),
                BitConverter.ToInt32(BitConverter.GetBytes(actual[i]), 0));
        }
    }
}
