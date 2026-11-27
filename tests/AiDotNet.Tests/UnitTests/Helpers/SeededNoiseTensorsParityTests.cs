using System;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// Pins <see cref="SeededNoiseTensors.StandardNormal{T}"/> to the per-element loop the video and text-to-speech
/// models each carried: <c>t[i] = NumOps.FromDouble(scale * sqrt(-2 ln(1 - u1)) * cos(2 pi u2))</c>.
/// </summary>
public class SeededNoiseTensorsParityTests
{
    private const int Seed = 20490;

    private static Tensor<T> Reference<T>(int[] shape, Random random, double scale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = ops.FromDouble(scale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }

        return t;
    }

    private static Tensor<T> ReferenceUnscaled<T>(int[] shape, Random random)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = ops.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        return t;
    }

    private static void Matches<T>(int[] shape, double scale)
    {
        var expectedRandom = RandomHelper.CreateSeededRandom(Seed);
        var actualRandom = RandomHelper.CreateSeededRandom(Seed);

        var expected = Reference<T>(shape, expectedRandom, scale);
        var actual = SeededNoiseTensors.StandardNormal<T>(shape, actualRandom, scale);

        Assert.Equal(shape, actual.Shape.ToArray());
        Assert.Equal(expected.ToArray(), actual.ToArray());
        // The helper took exactly as many uniforms as the loop: the next draw of both generators agrees.
        Assert.Equal(expectedRandom.NextDouble(), actualRandom.NextDouble());
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(0.667)]
    [InlineData(0.0)]
    [InlineData(3.5)]
    public void StandardNormal_MatchesTheElementLoop_Double(double scale)
    {
        Matches<double>(new[] { 7 }, scale);
        Matches<double>(new[] { 2, 3, 5 }, scale);
        Matches<double>(new[] { 1, 4, 3, 3 }, scale);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(0.667)]
    [InlineData(0.0)]
    [InlineData(3.5)]
    public void StandardNormal_MatchesTheElementLoop_Float(double scale)
    {
        Matches<float>(new[] { 7 }, scale);
        Matches<float>(new[] { 2, 3, 5 }, scale);
        Matches<float>(new[] { 1, 4, 3, 3 }, scale);
    }

    [Fact]
    public void StandardNormal_DefaultScale_MatchesTheLoopWithoutAScaleFactor()
    {
        // The video models and two of the speech bases had no scale factor at all.
        var shape = new[] { 3, 4, 5 };

        Assert.Equal(
            ReferenceUnscaled<double>(shape, RandomHelper.CreateSeededRandom(Seed)).ToArray(),
            SeededNoiseTensors.StandardNormal<double>(shape, RandomHelper.CreateSeededRandom(Seed)).ToArray());
        Assert.Equal(
            ReferenceUnscaled<float>(shape, RandomHelper.CreateSeededRandom(Seed)).ToArray(),
            SeededNoiseTensors.StandardNormal<float>(shape, RandomHelper.CreateSeededRandom(Seed)).ToArray());
    }

    [Fact]
    public void StandardNormal_EmptyShape_ReturnsAnEmptyTensorAndDrawsNothing()
    {
        var used = RandomHelper.CreateSeededRandom(Seed);
        var untouched = RandomHelper.CreateSeededRandom(Seed);

        var tensor = SeededNoiseTensors.StandardNormal<double>(new[] { 2, 0, 3 }, used);

        Assert.Equal(0, tensor.Length);
        Assert.Equal(new[] { 2, 0, 3 }, tensor.Shape.ToArray());
        Assert.Equal(untouched.NextDouble(), used.NextDouble());
    }

    [Fact]
    public void StandardNormal_SameSeed_SameTensor_DifferentSeed_DifferentTensor()
    {
        var shape = new[] { 4, 4 };
        var first = SeededNoiseTensors.StandardNormal<double>(shape, RandomHelper.CreateSeededRandom(1));
        var again = SeededNoiseTensors.StandardNormal<double>(shape, RandomHelper.CreateSeededRandom(1));
        var other = SeededNoiseTensors.StandardNormal<double>(shape, RandomHelper.CreateSeededRandom(2));

        Assert.Equal(first.ToArray(), again.ToArray());
        Assert.NotEqual(first.ToArray(), other.ToArray());
    }

    /// <summary>Returns a fixed script of uniforms, then repeats the last one.</summary>
    private sealed class ScriptedRandom : Random
    {
        private readonly double[] _values;
        private int _next;

        public ScriptedRandom(params double[] values) => _values = values;

        public override double NextDouble() => _values[Math.Min(_next++, _values.Length - 1)];
    }

    [Theory]
    [InlineData((int)BoxMullerGuard.Complement, 0.25, 0.75)]
    [InlineData((int)BoxMullerGuard.FlooredComplement, 0.25, 0.75)]
    [InlineData((int)BoxMullerGuard.Floored, 0.25, 0.25)]
    public void Guard_DerivesTheFirstUniformAsNamed(int guardValue, double first, double expectedU1)
    {
        var guard = (BoxMullerGuard)guardValue;
        // u2 = 0 makes the cosine exactly 1, so the draw is sqrt(-2 ln u1).
        double draw = SeededNoiseTensors.NextStandardNormal(new ScriptedRandom(first, 0.0), guard);

        Assert.Equal(Math.Sqrt(-2.0 * Math.Log(expectedU1)), draw);
    }

    [Theory]
    [InlineData((int)BoxMullerGuard.FlooredComplement, 1.0)]
    [InlineData((int)BoxMullerGuard.Floored, 0.0)]
    public void FlooredGuards_KeepAZeroUniformFinite(int guardValue, double first)
    {
        var guard = (BoxMullerGuard)guardValue;
        double draw = SeededNoiseTensors.NextStandardNormal(new ScriptedRandom(first, 0.0), guard);

        Assert.Equal(Math.Sqrt(-2.0 * Math.Log(1e-10)), draw);
    }

    [Fact]
    public void Complement_AtUniformZero_IsZero()
    {
        // 1 - 0 = 1, and ln 1 = 0: the complement guard never takes the log of zero.
        Assert.Equal(0.0, SeededNoiseTensors.NextStandardNormal(new ScriptedRandom(0.0, 0.3)));
    }

    [Theory]
    [InlineData((int)BoxMullerGuard.Complement)]
    [InlineData((int)BoxMullerGuard.FlooredComplement)]
    [InlineData((int)BoxMullerGuard.Floored)]
    public void Span_Vector_Tensor_And_Scalar_DrawTheSameSequence(int guardValue)
    {
        var guard = (BoxMullerGuard)guardValue;
        const int n = 11;
        const double scale = 1.0;
        var scalarRandom = RandomHelper.CreateSeededRandom(Seed);
        var expected = new double[n];
        for (int i = 0; i < n; i++) expected[i] = SeededNoiseTensors.NextStandardNormal(scalarRandom, guard);

        var span = new double[n];
        SeededNoiseTensors.WriteStandardNormal<double>(span, RandomHelper.CreateSeededRandom(Seed), scale, guard);
        var vector = SeededNoiseTensors.StandardNormalVector<double>(n, RandomHelper.CreateSeededRandom(Seed), scale, guard);
        var tensor = SeededNoiseTensors.StandardNormal<double>(new[] { n }, RandomHelper.CreateSeededRandom(Seed), scale, guard);
        var filled = new Tensor<double>(new[] { n });
        SeededNoiseTensors.FillStandardNormal(filled, RandomHelper.CreateSeededRandom(Seed), scale, guard);

        Assert.Equal(expected, span);
        Assert.Equal(expected, vector.ToArray());
        Assert.Equal(expected, tensor.ToArray());
        Assert.Equal(expected, filled.ToArray());
    }

    [Fact]
    public void Mean_ShiftsEveryDraw()
    {
        var unshifted = new Tensor<double>(new[] { 9 });
        var shifted = new Tensor<double>(new[] { 9 });
        SeededNoiseTensors.FillStandardNormal(unshifted, RandomHelper.CreateSeededRandom(Seed), 0.02);
        SeededNoiseTensors.FillStandardNormal(shifted, RandomHelper.CreateSeededRandom(Seed), 0.02, mean: 1.0);

        for (int i = 0; i < 9; i++) Assert.Equal(1.0 + unshifted[i], shifted[i]);
    }

    [Theory]
    [InlineData(8, 1.0)]
    [InlineData(7, 1.0)]
    [InlineData(7, 0.02)]
    public void Pairs_MatchTheCosineSineLoop(int count, double scale)
    {
        // The loop CompiledStepRandom and SoundStorm carried: one radius per pair, cosine then sine,
        // and an odd count drops the last sine.
        var reference = RandomHelper.CreateSeededRandom(Seed);
        var expected = new double[count];
        for (int i = 0; i < count; i += 2)
        {
            double radius = scale * Math.Sqrt(-2.0 * Math.Log(1.0 - reference.NextDouble()));
            double angle = 2.0 * Math.PI * reference.NextDouble();
            expected[i] = radius * Math.Cos(angle);
            if (i + 1 < count) expected[i + 1] = radius * Math.Sin(angle);
        }

        var actualRandom = RandomHelper.CreateSeededRandom(Seed);
        var actual = new double[count];
        SeededNoiseTensors.WriteStandardNormalPairs<double>(actual, actualRandom, scale);

        Assert.Equal(expected, actual);
        Assert.Equal(reference.NextDouble(), actualRandom.NextDouble());
        if (scale == 1.0)
            Assert.Equal(expected, SeededNoiseTensors.StandardNormalPairs<double>(new[] { count }, RandomHelper.CreateSeededRandom(Seed)).ToArray());
    }
}
