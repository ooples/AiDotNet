using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SyntheticData;

/// <summary>
/// Pins the shared draw, row and column-preparation helpers of <see cref="SyntheticTensorOps{T}"/> to the
/// element-by-element loops they replaced in the tabular generators. Each reference below is the original loop;
/// both sides read the same seeded generator stream, and the values must be identical.
/// </summary>
public class SyntheticTensorOpsDrawParityTests
{
    private const int Seed = 20240917;

    private static double D<T>(T value) => MathHelper.GetNumericOperations<T>().ToDouble(value);

    private static void AssertSame<T>(IReadOnlyList<T> expected, IReadOnlyList<T> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
            Assert.True(D(expected[i]).Equals(D(actual[i])), $"element {i}: expected {D(expected[i]):R} but was {D(actual[i]):R}");
    }

    /// <summary>The three per-element Box-Muller loops the generators carried.</summary>
    private static T[] ReferenceNormals<T>(int count, Random random, BoxMullerGuard guard)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var values = new T[count];
        for (int i = 0; i < count; i++)
        {
            switch (guard)
            {
                case BoxMullerGuard.Complement:
                {
                    double u1 = 1.0 - random.NextDouble();
                    double u2 = random.NextDouble();
                    double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    values[i] = ops.FromDouble(z);
                    break;
                }
                case BoxMullerGuard.FlooredComplement:
                {
                    double u1 = 1.0 - random.NextDouble();
                    double u2 = random.NextDouble();
                    double normal = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
                    values[i] = ops.FromDouble(normal);
                    break;
                }
                default:
                {
                    double u1 = Math.Max(1e-10, random.NextDouble());
                    double u2 = random.NextDouble();
                    double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    values[i] = ops.FromDouble(z);
                    break;
                }
            }
        }

        return values;
    }

    private static void NormalVectorMatches<T>(int length, BoxMullerGuard guard)
    {
        var expected = ReferenceNormals<T>(length, RandomHelper.CreateSeededRandom(Seed), guard);
        var stream = RandomHelper.CreateSeededRandom(Seed);

        var actual = SeededNoiseTensors.StandardNormalVector<T>(length, stream, guard: guard);

        AssertSame(expected, actual.ToArray());
        // The helper consumed exactly two uniforms per value: the streams are in step afterwards
        var twin = RandomHelper.CreateSeededRandom(Seed);
        for (int i = 0; i < 2 * length; i++) twin.NextDouble();
        Assert.Equal(twin.NextDouble(), stream.NextDouble());
    }

    [Theory]
    [InlineData(0, (int)BoxMullerGuard.Complement)]
    [InlineData(1, (int)BoxMullerGuard.Complement)]
    [InlineData(64, (int)BoxMullerGuard.Complement)]
    [InlineData(64, (int)BoxMullerGuard.FlooredComplement)]
    [InlineData(64, (int)BoxMullerGuard.Floored)]
    [InlineData(7, (int)BoxMullerGuard.Floored)]
    public void StandardNormalVector_MatchesTheElementLoop_Double(int length, int guard)
        => NormalVectorMatches<double>(length, (BoxMullerGuard)guard);

    [Theory]
    [InlineData(64, (int)BoxMullerGuard.Complement)]
    [InlineData(33, (int)BoxMullerGuard.FlooredComplement)]
    [InlineData(33, (int)BoxMullerGuard.Floored)]
    public void StandardNormalVector_MatchesTheElementLoop_Float(int length, int guard)
        => NormalVectorMatches<float>(length, (BoxMullerGuard)guard);

    [Fact]
    public void TheThreeGuards_DrawDifferentValues_WhereTheyShould()
    {
        // Complement and Floored read the first uniform differently (1 - u against u), so their draws differ;
        // the parity cases above would pass vacuously if the guard were ignored
        var complement = SeededNoiseTensors.StandardNormalVector<double>(16, RandomHelper.CreateSeededRandom(Seed)).ToArray();
        var floored = SeededNoiseTensors.StandardNormalVector<double>(16, RandomHelper.CreateSeededRandom(Seed), guard: BoxMullerGuard.Floored).ToArray();

        Assert.NotEqual(complement, floored);
        Assert.All(complement, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }

    [Theory]
    [InlineData(4, 5)]
    [InlineData(1, 1)]
    [InlineData(3, 0)]
    public void StandardNormalTensor_FillsRowMajor_LikeTheElementLoop(int rows, int columns)
    {
        // OCTGAN Noise: z[i] over the flat index
        var expected = ReferenceNormals<double>(rows * columns, RandomHelper.CreateSeededRandom(Seed), BoxMullerGuard.Complement);

        var actual = SeededNoiseTensors.StandardNormal<double>(new[] { rows, columns }, RandomHelper.CreateSeededRandom(Seed));

        Assert.Equal(new[] { rows, columns }, actual.Shape.ToArray());
        AssertSame(expected, actual.ToArray());
    }

    [Theory]
    [InlineData((int)BoxMullerGuard.Complement)]
    [InlineData((int)BoxMullerGuard.FlooredComplement)]
    public void FillStandardNormal_OverwritesEveryElementInFlatOrder_LikeTheElementLoop(int guardValue)
    {
        var guard = (BoxMullerGuard)guardValue;
        // GOGGLE / TVAE / TabSyn FillStandardNormal(Tensor): eps[i] over the flat index, on a tensor that already
        // holds values
        var expected = ReferenceNormals<double>(12, RandomHelper.CreateSeededRandom(Seed), guard);
        var target = new Tensor<double>(Enumerable.Repeat(123.0, 12).ToArray(), new[] { 3, 4 });

        SeededNoiseTensors.FillStandardNormal(target, RandomHelper.CreateSeededRandom(Seed), guard: guard);

        Assert.Equal(new[] { 3, 4 }, target.Shape.ToArray());
        AssertSame(expected, target.ToArray());
        Assert.DoesNotContain(123.0, target.ToArray());
    }

    [Fact]
    public void FillStandardNormal_Float_MatchesTheElementLoop()
    {
        var expected = ReferenceNormals<float>(10, RandomHelper.CreateSeededRandom(Seed), BoxMullerGuard.Complement);
        var target = new Tensor<float>(new[] { 2, 5 });

        SeededNoiseTensors.FillStandardNormal(target, RandomHelper.CreateSeededRandom(Seed));

        AssertSame(expected, target.ToArray());
    }

    private static void PairTensorMatches<T>(int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var random = RandomHelper.CreateSeededRandom(Seed);
        int totalElements = shape.Aggregate(1, (a, b) => a * b);
        // DP-CTGAN / MedGAN / TimeGAN GenerateNoiseBatchTensor
        var noiseData = new T[totalElements];
        for (int i = 0; i < totalElements; i += 2)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            noiseData[i] = ops.FromDouble(r * Math.Cos(theta));
            if (i + 1 < totalElements)
                noiseData[i + 1] = ops.FromDouble(r * Math.Sin(theta));
        }

        var actual = SeededNoiseTensors.StandardNormalPairs<T>(shape, RandomHelper.CreateSeededRandom(Seed));

        Assert.Equal(shape, actual.Shape.ToArray());
        AssertSame(noiseData, actual.ToArray());
    }

    [Theory]
    [InlineData(new[] { 4, 6 })]
    [InlineData(new[] { 3, 5 })]    // odd count: the last sine is dropped
    [InlineData(new[] { 2, 3, 4 })]
    [InlineData(new[] { 1, 1 })]
    [InlineData(new[] { 0, 8 })]
    public void StandardNormalPairTensor_MatchesTheElementLoop_Double(int[] shape) => PairTensorMatches<double>(shape);

    [Theory]
    [InlineData(new[] { 4, 6 })]
    [InlineData(new[] { 3, 3 })]
    public void StandardNormalPairTensor_MatchesTheElementLoop_Float(int[] shape) => PairTensorMatches<float>(shape);

    [Fact]
    public void StandardNormalPairTensor_RejectsAnElementCountThatOverflows()
        => Assert.Throws<OverflowException>(
            () => SeededNoiseTensors.StandardNormalPairs<double>(new[] { 70000, 70000, 2 }, RandomHelper.CreateSeededRandom(Seed)));

    [Theory]
    [InlineData(3, 7)]
    [InlineData(1, 1)]
    public void UniformTensor_MatchesTheElementLoop(int rows, int columns)
    {
        // PATE-GAN Noise: z[i] = FromDouble(NextDouble())
        var random = RandomHelper.CreateSeededRandom(Seed);
        var expectedDouble = new double[rows * columns];
        for (int i = 0; i < expectedDouble.Length; i++) expectedDouble[i] = random.NextDouble();

        var asDouble = SyntheticTensorOps<double>.UniformTensor(new[] { rows, columns }, RandomHelper.CreateSeededRandom(Seed));
        var asFloat = SyntheticTensorOps<float>.UniformTensor(new[] { rows, columns }, RandomHelper.CreateSeededRandom(Seed));

        Assert.Equal(new[] { rows, columns }, asDouble.Shape.ToArray());
        Assert.Equal(expectedDouble, asDouble.ToArray());
        Assert.Equal(expectedDouble.Select(v => (float)v).ToArray(), asFloat.ToArray());
    }

    private static void TimestepEmbeddingMatches<T>(int timestep, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // AutoDiffTab / FinDiff / TabSyn CreateTimestepEmbedding
        var embedding = new Vector<T>(dim);
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            double freq = Math.Exp(-Math.Log(10000.0) * i / halfDim);
            double angle = timestep * freq;
            embedding[i] = ops.FromDouble(Math.Sin(angle));
            if (i + halfDim < dim) embedding[i + halfDim] = ops.FromDouble(Math.Cos(angle));
        }

        var actual = SyntheticTensorOps<T>.SinusoidalTimestepEmbedding(timestep, dim);

        AssertSame(embedding.ToArray(), actual.ToArray());
    }

    [Theory]
    [InlineData(0, 16)]
    [InlineData(1, 16)]
    [InlineData(250, 64)]
    [InlineData(999, 7)]   // odd: the last element stays 0
    [InlineData(5, 1)]
    [InlineData(5, 0)]
    public void SinusoidalTimestepEmbedding_MatchesTheElementLoop_Double(int timestep, int dim)
        => TimestepEmbeddingMatches<double>(timestep, dim);

    [Theory]
    [InlineData(250, 64)]
    [InlineData(999, 7)]
    public void SinusoidalTimestepEmbedding_MatchesTheElementLoop_Float(int timestep, int dim)
        => TimestepEmbeddingMatches<float>(timestep, dim);

    [Fact]
    public void SinusoidalTimestepEmbedding_OddDimension_LeavesTheLastElementZero()
    {
        var embedding = SyntheticTensorOps<double>.SinusoidalTimestepEmbedding(999, 7).ToArray();

        Assert.Equal(0.0, embedding[6]);
        Assert.Contains(embedding.Take(6), v => v != 0.0);
    }

    private static Matrix<double> Data()
    {
        // Column 0 numerical, column 1 categorical codes with repeats, column 2 categorical with fractional codes
        double[,] values =
        {
            { 0.5, 2.0, 1.5 },
            { -1.25, 0.0, 1.5 },
            { 3.0, 10.0, -0.5 },
            { 0.0, 2.0, 1e-7 },
            { 7.5, 1.0, 1.5 },
        };
        var matrix = new Matrix<double>(5, 3);
        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 3; c++) matrix[r, c] = values[r, c];
        return matrix;
    }

    [Fact]
    public void PrepareColumns_MatchesTheRowScan()
    {
        var data = Data();
        var columns = new List<ColumnMetadata>
        {
            new ColumnMetadata("amount", ColumnDataType.Continuous),
            new ColumnMetadata("code", ColumnDataType.Categorical),
            new ColumnMetadata("grade", ColumnDataType.Categorical),
        };
        var statisticsCalls = new List<int>();

        // The scan every generator ran
        var expected = new List<IReadOnlyList<string>?>();
        for (int col = 0; col < columns.Count; col++)
        {
            var meta = columns[col].Clone();
            if (meta.IsNumerical)
            {
                expected.Add(null);
            }
            else
            {
                var categories = new HashSet<string>();
                for (int row = 0; row < data.Rows; row++)
                {
                    var val = data[row, col];
                    categories.Add(val.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }

                expected.Add(categories.OrderBy(c => c, StringComparer.Ordinal).ToList().AsReadOnly());
            }
        }

        var prepared = SyntheticTensorOps<double>.PrepareColumns(data, columns, (_, col, _) => statisticsCalls.Add(col));

        Assert.Equal(3, prepared.Count);
        Assert.Equal(new[] { 0 }, statisticsCalls);
        for (int col = 0; col < 3; col++)
        {
            Assert.Equal(col, prepared[col].ColumnIndex);
            Assert.NotSame(columns[col], prepared[col]);
            if (expected[col] is { } categories)
                Assert.Equal(categories, prepared[col].Categories);
        }

        // Ordinal order, not numeric: "10" sorts before "2"
        Assert.Equal(new[] { "0", "1", "10", "2" }, prepared[1].Categories);
        Assert.Equal(3, prepared[2].Categories.Count);
    }

    [Fact]
    public void PrepareColumns_KeepsDeclaredCategories()
    {
        var declared = new ColumnMetadata("code", ColumnDataType.Categorical, new[] { "x", "y" });

        var prepared = SyntheticTensorOps<double>.PrepareColumns(
            Data(), new List<ColumnMetadata> { new ColumnMetadata("amount", ColumnDataType.Continuous), declared }, (_, _, _) => { });

        Assert.Equal(new[] { "x", "y" }, prepared[1].Categories);
    }

    [Fact]
    public void MatrixGetRow_IsTheCopyTheGeneratorsMadeByHand()
    {
        // The generators' private GetRow copied a row element by element; they now call Matrix.GetRow
        var data = Data();

        var row = data.GetRow(2);

        Assert.Equal(new[] { 3.0, 10.0, -0.5 }, row.ToArray());
        row[0] = 99.0;
        Assert.Equal(3.0, data[2, 0]);
    }

    /// <summary>A generator that returns one fixed uniform, to drive the Box-Muller guards to their edges.</summary>
    private sealed class FixedRandom : Random
    {
        private readonly double _value;
        public FixedRandom(double value) => _value = value;
        public override double NextDouble() => _value;
        protected override double Sample() => _value;
    }

    // The largest double below 1, the top of NextDouble's range.
    private static readonly double JustBelowOne = BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(1.0) - 1);

    [Fact]
    public void FlooredGuard_FloorsAZeroUniform_ToAFiniteDraw()
    {
        // u1 = max(1e-10, 0) = 1e-10 and u2 = 0; without the floor ln(0) makes every draw +Infinity.
        double expected = Math.Sqrt(-2.0 * Math.Log(1e-10)) * Math.Cos(2.0 * Math.PI * 0.0);
        var vector = SeededNoiseTensors.StandardNormalVector<double>(4, new FixedRandom(0.0), guard: BoxMullerGuard.Floored).ToArray();
        var tensor = SeededNoiseTensors.StandardNormal<double>(new[] { 2, 2 }, new FixedRandom(0.0), guard: BoxMullerGuard.Floored).ToArray();
        foreach (var draws in new[] { vector, tensor })
        {
            Assert.All(draws, v => Assert.False(double.IsInfinity(v) || double.IsNaN(v)));
            Assert.All(draws, v => Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(v)));
        }
    }

    [Fact]
    public void FlooredComplementGuard_FloorsTheComplementOfTheLargestUniform()
    {
        // 1 - JustBelowOne is 2^-53, below the 1e-10 floor: the floored draw uses ln(1e-10), the unfloored one
        // ln(2^-53), so the values differ and the floor is observable.
        double u2 = JustBelowOne;
        double expected = Math.Sqrt(-2.0 * Math.Log(1e-10)) * Math.Cos(2.0 * Math.PI * u2);
        double unfloored = Math.Sqrt(-2.0 * Math.Log(1.0 - JustBelowOne)) * Math.Cos(2.0 * Math.PI * u2);
        Assert.NotEqual(expected, unfloored);

        var target = new Tensor<double>(new[] { 3 });
        SeededNoiseTensors.FillStandardNormal(target, new FixedRandom(JustBelowOne), guard: BoxMullerGuard.FlooredComplement);
        var vector = SeededNoiseTensors.StandardNormalVector<double>(3, new FixedRandom(JustBelowOne), guard: BoxMullerGuard.FlooredComplement).ToArray();
        foreach (var draws in new[] { target.ToArray(), vector })
            Assert.All(draws, v => Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(v)));
    }
}
