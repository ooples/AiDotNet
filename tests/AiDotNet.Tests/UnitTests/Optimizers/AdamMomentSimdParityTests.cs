using System;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// #1757: the SIMD-vectorized Adam/AMSGrad moment kernel (<see cref="AdamMomentSimd"/>) must be
/// BIT-IDENTICAL to the scalar reference across every size (incl. sub-vector-width tails) and both the
/// plain-Adam and AMSGrad paths, for fp64 and fp32. Any drift here is a real regression, not noise —
/// the whole point of the change is a faster kernel that changes no numbers.
/// </summary>
public class AdamMomentSimdParityTests
{
    private const double B1 = 0.9, B2 = 0.999, Eps = 1e-8, Lr = 0.01;
    private static readonly int Step = 5;
    private static readonly double Om1 = 1.0 - B1, Om2 = 1.0 - B2;
    private static readonly double Bc1 = 1.0 - Math.Pow(B1, Step), Bc2 = 1.0 - Math.Pow(B2, Step);

    // Independent scalar reference == the pre-SIMD inner loop, exactly.
    private static void RefD(Span<double> p, ReadOnlySpan<double> g, Span<double> m, Span<double> v, Span<double> vMax, bool ams)
    {
        for (int i = 0; i < p.Length; i++)
        {
            double gi = g[i];
            double mn = B1 * m[i] + Om1 * gi;
            double vn = B2 * v[i] + Om2 * gi * gi;
            m[i] = mn; v[i] = vn;
            double mh = mn / Bc1, ve;
            if (ams) { double vhn = vn / Bc2, vmp = vMax[i]; double vmn = vhn > vmp ? vhn : vmp; vMax[i] = vmn; ve = vmn; }
            else ve = vn / Bc2;
            p[i] -= Lr * mh / (Math.Sqrt(ve) + Eps);
        }
    }

    private static void RefF(Span<float> p, ReadOnlySpan<float> g, Span<float> m, Span<float> v, Span<float> vMax, bool ams)
    {
        float b1 = (float)B1, b2 = (float)B2, om1 = (float)Om1, om2 = (float)Om2;
        float bc1 = (float)Bc1, bc2 = (float)Bc2, lr = (float)Lr, eps = (float)Eps;
        for (int i = 0; i < p.Length; i++)
        {
            float gi = g[i];
            float mn = b1 * m[i] + om1 * gi;
            float vn = b2 * v[i] + om2 * gi * gi;
            m[i] = mn; v[i] = vn;
            float mh = mn / bc1, ve;
            if (ams) { float vhn = vn / bc2, vmp = vMax[i]; float vmn = vhn > vmp ? vhn : vmp; vMax[i] = vmn; ve = vmn; }
            else ve = vn / bc2;
            p[i] -= lr * mh / ((float)Math.Sqrt(ve) + eps);
        }
    }

    [Theory]
    [InlineData(1)] [InlineData(3)] [InlineData(4)] [InlineData(7)] [InlineData(8)]
    [InlineData(15)] [InlineData(31)] [InlineData(256)] [InlineData(1000)]
    public void Fp64_Simd_IsBitIdentical_ToScalar(int n)
    {
        foreach (bool ams in new[] { false, true })
        {
            var rng = new Random(9001 + n + (ams ? 1 : 0));
            double[] p = Rand(rng, n), g = Rand(rng, n), m = Rand(rng, n), v = RandPos(rng, n), vMax = RandPos(rng, n);

            double[] pR = (double[])p.Clone(), mR = (double[])m.Clone(), vR = (double[])v.Clone(), vmR = (double[])vMax.Clone();
            RefD(pR, g, mR, vR, vmR, ams);

            double[] pS = (double[])p.Clone(), mS = (double[])m.Clone(), vS = (double[])v.Clone(), vmS = (double[])vMax.Clone();
            AdamMomentSimd.Step(pS, g, mS, vS, vmS, B1, B2, Om1, Om2, Bc1, Bc2, Lr, Eps, ams);

            for (int i = 0; i < n; i++)
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(pR[i]), BitConverter.DoubleToInt64Bits(pS[i]));
                Assert.Equal(BitConverter.DoubleToInt64Bits(mR[i]), BitConverter.DoubleToInt64Bits(mS[i]));
                Assert.Equal(BitConverter.DoubleToInt64Bits(vR[i]), BitConverter.DoubleToInt64Bits(vS[i]));
                if (ams) Assert.Equal(BitConverter.DoubleToInt64Bits(vmR[i]), BitConverter.DoubleToInt64Bits(vmS[i]));
            }
        }
    }

    [Theory]
    [InlineData(1)] [InlineData(3)] [InlineData(8)] [InlineData(15)] [InlineData(16)]
    [InlineData(63)] [InlineData(256)] [InlineData(1000)]
    public void Fp32_Simd_IsBitIdentical_ToScalar(int n)
    {
        foreach (bool ams in new[] { false, true })
        {
            var rng = new Random(4200 + n + (ams ? 1 : 0));
            float[] p = RandF(rng, n), g = RandF(rng, n), m = RandF(rng, n), v = RandPosF(rng, n), vMax = RandPosF(rng, n);

            float[] pR = (float[])p.Clone(), mR = (float[])m.Clone(), vR = (float[])v.Clone(), vmR = (float[])vMax.Clone();
            RefF(pR, g, mR, vR, vmR, ams);

            float[] pS = (float[])p.Clone(), mS = (float[])m.Clone(), vS = (float[])v.Clone(), vmS = (float[])vMax.Clone();
            AdamMomentSimd.Step(pS, g, mS, vS, vmS, (float)B1, (float)B2, (float)Om1, (float)Om2, (float)Bc1, (float)Bc2, (float)Lr, (float)Eps, ams);

            for (int i = 0; i < n; i++)
            {
                Assert.Equal(BitConverter.SingleToInt32Bits(pR[i]), BitConverter.SingleToInt32Bits(pS[i]));
                Assert.Equal(BitConverter.SingleToInt32Bits(mR[i]), BitConverter.SingleToInt32Bits(mS[i]));
                Assert.Equal(BitConverter.SingleToInt32Bits(vR[i]), BitConverter.SingleToInt32Bits(vS[i]));
                if (ams) Assert.Equal(BitConverter.SingleToInt32Bits(vmR[i]), BitConverter.SingleToInt32Bits(vmS[i]));
            }
        }
    }

    private static double[] Rand(Random r, int n) { var a = new double[n]; for (int i = 0; i < n; i++) a[i] = (r.NextDouble() - 0.5) * 2.0; return a; }
    private static double[] RandPos(Random r, int n) { var a = new double[n]; for (int i = 0; i < n; i++) a[i] = r.NextDouble() * 2.0 + 1e-6; return a; }
    private static float[] RandF(Random r, int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((r.NextDouble() - 0.5) * 2.0); return a; }
    private static float[] RandPosF(Random r, int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2.0 + 1e-6); return a; }
}
