#if NET5_0_OR_GREATER
using System;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

public class DktDistanceRegressionTests
{
    private static Tensor<T> Kernel<T>(Tensor<T> a, Tensor<T> b, T logLengthScale,
        Tensor<T>? aMask = null, Tensor<T>? bMask = null)
    {
#pragma warning disable SYSLIB0050
        var algorithm = (DKTAlgorithm<T, Matrix<T>, Vector<T>>)FormatterServices.GetUninitializedObject(typeof(DKTAlgorithm<T, Matrix<T>, Vector<T>>));
#pragma warning restore SYSLIB0050
        typeof(DKTAlgorithm<T, Matrix<T>, Vector<T>>).GetField("_kernelParams", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(algorithm, new Vector<T>(new[] { logLengthScale, default! }));
        aMask ??= Tensor<double>.CreateDefault(new[] { a.Shape[0], a.Shape[1] }, 1.0).Cast<T>();
        bMask ??= Tensor<double>.CreateDefault(new[] { b.Shape[0], b.Shape[1] }, 1.0).Cast<T>();
        return Assert.IsType<Tensor<T>>(typeof(DKTAlgorithm<T, Matrix<T>, Vector<T>>)
            .GetMethod("KernelMatrix", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(algorithm, new object[] { a, aMask, b, bMask }));
    }

    [Theory]
    [InlineData(100000000.0)]
    [InlineData(1000000000000.0)]
    public void DoubleNearbyRowsPreserveTheirDistanceAtLargeCommonOffsets(double origin)
    {
        var a = new Tensor<double>(new[] { origin, origin + 2, origin + 4 }, new[] { 1, 3 });
        var b = new Tensor<double>(new[] { origin + 1, origin + 3, origin + 5 }, new[] { 1, 3 });
        Assert.Equal(Math.Exp(-1.5), Kernel(a, b, 0.0)[0, 0], 13);
        Assert.Equal(1, Kernel(a, a, 0.0)[0, 0]);
    }

    [Fact]
    public void FloatNearbyRowsDoNotCollapseToTheSamePoint()
    {
        var a = new Tensor<float>(new[] { 100000f, 100001f, 100002f }, new[] { 1, 3 });
        var b = new Tensor<float>(new[] { 100000.25f, 100001.25f, 100002.25f }, new[] { 1, 3 });
        Assert.Equal((float)Math.Exp(-0.5 * 3 * 0.25 * 0.25), Kernel(a, b, 0f)[0, 0]);
    }

    [Fact]
    public void HalfNormalizesWideDifferencesBeforeSquaring()
    {
        var a = Tensor<Half>.CreateDefault(new[] { 1, 3 }, (Half)60000);
        var b = Tensor<Half>.CreateDefault(new[] { 1, 3 }, (Half)(-60000));
        Half logScale = (Half)Math.Log(60000);
        double delta = 120000 * Math.Exp(-(double)logScale);
        var result = Kernel(a, b, logScale);
        Assert.Equal((Half)Math.Exp(-0.5 * 3 * delta * delta), result[0, 0]);
        Assert.True(result[0, 0] > (Half)0);
        Assert.Equal((Half)1, Kernel(a, a, logScale)[0, 0]);
    }

    [Fact]
    public void DoubleNormalizesBeforeDistanceOrLengthScaleSquaresCanOverflow()
    {
        var a = new Tensor<double>(new[] { 1e200 }, new[] { 1, 1 });
        var b = new Tensor<double>(new[] { 0.0 }, new[] { 1, 1 });
        Assert.Equal(Math.Exp(-0.5), Kernel(a, b, Math.Log(1e200))[0, 0], 12);
    }

    [Fact]
    public void AbsentFeaturesAreMaskedBeforeOverflowingDifferencesAreSquared()
    {
        var a = new Tensor<double>(new[] { 2.0, double.MaxValue }, new[] { 1, 2 });
        var b = new Tensor<double>(new[] { 3.0, -double.MaxValue }, new[] { 1, 2 });
        var mask = new Tensor<double>(new[] { 1.0, 0.0 }, new[] { 1, 2 });
        Assert.Equal(Math.Exp(-0.5), Kernel(a, b, 0.0, mask, mask)[0, 0], 13);
    }

    [Fact]
    public void DecimalNearbyRowsAreSubtractedBeforeWidening()
    {
        const decimal origin = 100000000000000000000m;
        var a = new Tensor<decimal>(new[] { origin }, new[] { 1, 1 });
        var b = new Tensor<decimal>(new[] { origin + 1 }, new[] { 1, 1 });
        Assert.Equal(Math.Exp(-0.5), (double)Kernel(a, b, 0m)[0, 0], 13);
        var extremes = Assert.IsType<Tensor<double>>(typeof(DKTAlgorithm<decimal, Matrix<decimal>, Vector<decimal>>)
            .GetMethod("NativeDifferences", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, new object[]
            {
                new Tensor<decimal>(new[] { decimal.MaxValue }, new[] { 1, 1 }),
                new Tensor<decimal>(new[] { decimal.MinValue }, new[] { 1, 1 }), 1, 1, 1
            }));
        Assert.Equal((double)decimal.MaxValue - (double)decimal.MinValue, extremes[0, 0, 0]);
    }

    [Fact]
    public void PairAndFeatureBlocksMatchAnIndependentScalarReference()
    {
        const int rows = 33, columns = 35, width = 257;
        var a = new Tensor<double>(Enumerable.Range(0, rows * width)
            .Select(i => 1000000 + (i / width) * 0.1 + (i % 11) * 0.01).ToArray(), new[] { rows, width });
        var b = new Tensor<double>(Enumerable.Range(0, columns * width)
            .Select(i => 1000000 + (i / width) * 0.08 + (i % 7) * 0.01).ToArray(), new[] { columns, width });
        var previous = AiDotNetEngine.Current;
        var guard = new BoundedEngine();
        try
        {
            AiDotNetEngine.Current = guard;
            var result = Kernel(a, b, Math.Log(2));
            for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
            {
                double sum = 0;
                for (int d = 0; d < width; d++)
                {
                    double delta = (a[i, d] - b[j, d]) / 2;
                    sum += delta * delta;
                }
                Assert.Equal(Math.Exp(-0.5 * sum), result[i, j], 12);
            }
            Assert.True(guard.PairBlocks > 1);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void EmptyPairAxisProducesAnEmptyKernel()
    {
        var result = Kernel(new Tensor<double>(new[] { 0, 3 }), new Tensor<double>(new[] { 2, 3 }), 0.0);
        Assert.Equal(new[] { 0, 2 }, result.Shape.ToArray());
        Assert.Equal(0, result.Length);
    }

    private sealed class BoundedEngine : CpuEngine
    {
        public int PairBlocks { get; private set; }
        public override Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
        {
            if (a.Shape.Length == 3)
            {
                PairBlocks++;
                Assert.InRange(a.Length, 1, 32768);
                Assert.InRange(b.Length, 1, 32768);
            }
            return base.TensorMultiply(a, b);
        }
    }
}
#endif
