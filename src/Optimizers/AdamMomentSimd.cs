using System;
using SnVec = System.Numerics.Vector;

namespace AiDotNet.Optimizers;

/// <summary>
/// Hardware-SIMD inner kernel for the fused Adam / AMSGrad moment update (#1757). The tape-step path
/// in <see cref="AdamOptimizer{T,TInput,TOutput}"/> already updates the moments in place over the raw
/// backing arrays (no per-step allocations); these helpers replace its scalar inner loop with
/// <c>System.Numerics.Vector&lt;T&gt;</c> ops.
///
/// <para>Every operation is IEEE elementwise, so the result is <b>bit-identical</b> to the scalar loop:
/// the fp64 path uses <c>Vector.SquareRoot</c> (same as scalar <c>Math.Sqrt</c>), and the fp32 path
/// widens to fp64 for the sqrt then narrows — matching the scalar path's <c>(float)Math.Sqrt(...)</c>
/// exactly rather than a ~1-ULP single-precision sqrt. The AMSGrad running-max uses
/// <c>ConditionalSelect(GreaterThan(...))</c> to match the scalar <c>&gt;</c> ternary (incl. NaN
/// behaviour). The scalar tail handles the sub-vector-width remainder.</para>
///
/// <para><c>Vector&lt;T&gt;</c> here is <c>System.Numerics.Vector&lt;T&gt;</c> — the SnVec alias points
/// at the static helper class; the generic struct is written out because this assembly's unqualified
/// <c>Vector&lt;T&gt;</c> is AiDotNet's linear-algebra vector.</para>
/// </summary>
internal static class AdamMomentSimd
{
    /// <summary>fp64 Adam/AMSGrad moment update, in place over <paramref name="param"/>/<paramref name="m"/>/<paramref name="v"/>.</summary>
    internal static void Step(
        Span<double> param, ReadOnlySpan<double> grad, Span<double> m, Span<double> v, Span<double> vMax,
        double b1, double b2, double oneMinusB1, double oneMinusB2,
        double bc1, double bc2, double lr, double eps, bool useAmsgrad)
    {
        int n = param.Length;
        int width = System.Numerics.Vector<double>.Count;
        int i = 0;
        if (SnVec.IsHardwareAccelerated && n >= width)
        {
            var vb1 = new System.Numerics.Vector<double>(b1);
            var vb2 = new System.Numerics.Vector<double>(b2);
            var v1mb1 = new System.Numerics.Vector<double>(oneMinusB1);
            var v1mb2 = new System.Numerics.Vector<double>(oneMinusB2);
            var vbc1 = new System.Numerics.Vector<double>(bc1);
            var vbc2 = new System.Numerics.Vector<double>(bc2);
            var vlr = new System.Numerics.Vector<double>(lr);
            var veps = new System.Numerics.Vector<double>(eps);
            for (; i <= n - width; i += width)
            {
                var g = new System.Numerics.Vector<double>(grad.Slice(i, width));
                var mNew = vb1 * new System.Numerics.Vector<double>(m.Slice(i, width)) + v1mb1 * g;
                var vNew = vb2 * new System.Numerics.Vector<double>(v.Slice(i, width)) + v1mb2 * g * g;
                mNew.CopyTo(m.Slice(i, width));
                vNew.CopyTo(v.Slice(i, width));
                var mHat = mNew / vbc1;
                System.Numerics.Vector<double> vHatEff;
                if (useAmsgrad)
                {
                    var vHatNow = vNew / vbc2;
                    var vMaxPrev = new System.Numerics.Vector<double>(vMax.Slice(i, width));
                    var vMaxNew = SnVec.ConditionalSelect(SnVec.GreaterThan(vHatNow, vMaxPrev), vHatNow, vMaxPrev);
                    vMaxNew.CopyTo(vMax.Slice(i, width));
                    vHatEff = vMaxNew;
                }
                else
                {
                    vHatEff = vNew / vbc2;
                }
                var pv = new System.Numerics.Vector<double>(param.Slice(i, width));
                pv -= vlr * mHat / (SnVec.SquareRoot(vHatEff) + veps);
                pv.CopyTo(param.Slice(i, width));
            }
        }
        for (; i < n; i++)
        {
            double g = grad[i];
            double mNew = b1 * m[i] + oneMinusB1 * g;
            double vNew = b2 * v[i] + oneMinusB2 * g * g;
            m[i] = mNew;
            v[i] = vNew;
            double mHat = mNew / bc1;
            double vHatEff;
            if (useAmsgrad)
            {
                double vHatNow = vNew / bc2;
                double vMaxPrev = vMax[i];
                double vMaxNew = vHatNow > vMaxPrev ? vHatNow : vMaxPrev;
                vMax[i] = vMaxNew;
                vHatEff = vMaxNew;
            }
            else
            {
                vHatEff = vNew / bc2;
            }
            param[i] -= lr * mHat / (Math.Sqrt(vHatEff) + eps);
        }
    }

    /// <summary>fp32 Adam/AMSGrad moment update. sqrt is done in fp64 (widen→sqrt→narrow) to match the scalar <c>(float)Math.Sqrt</c> bit-for-bit.</summary>
    internal static void Step(
        Span<float> param, ReadOnlySpan<float> grad, Span<float> m, Span<float> v, Span<float> vMax,
        float b1, float b2, float oneMinusB1, float oneMinusB2,
        float bc1, float bc2, float lr, float eps, bool useAmsgrad)
    {
        int n = param.Length;
        int width = System.Numerics.Vector<float>.Count;
        int i = 0;
        if (SnVec.IsHardwareAccelerated && n >= width)
        {
            var vb1 = new System.Numerics.Vector<float>(b1);
            var vb2 = new System.Numerics.Vector<float>(b2);
            var v1mb1 = new System.Numerics.Vector<float>(oneMinusB1);
            var v1mb2 = new System.Numerics.Vector<float>(oneMinusB2);
            var vbc1 = new System.Numerics.Vector<float>(bc1);
            var vbc2 = new System.Numerics.Vector<float>(bc2);
            var vlr = new System.Numerics.Vector<float>(lr);
            var veps = new System.Numerics.Vector<float>(eps);
            for (; i <= n - width; i += width)
            {
                var g = new System.Numerics.Vector<float>(grad.Slice(i, width));
                var mNew = vb1 * new System.Numerics.Vector<float>(m.Slice(i, width)) + v1mb1 * g;
                var vNew = vb2 * new System.Numerics.Vector<float>(v.Slice(i, width)) + v1mb2 * g * g;
                mNew.CopyTo(m.Slice(i, width));
                vNew.CopyTo(v.Slice(i, width));
                var mHat = mNew / vbc1;
                System.Numerics.Vector<float> vHatEff;
                if (useAmsgrad)
                {
                    var vHatNow = vNew / vbc2;
                    var vMaxPrev = new System.Numerics.Vector<float>(vMax.Slice(i, width));
                    var vMaxNew = SnVec.ConditionalSelect(SnVec.GreaterThan(vHatNow, vMaxPrev), vHatNow, vMaxPrev);
                    vMaxNew.CopyTo(vMax.Slice(i, width));
                    vHatEff = vMaxNew;
                }
                else
                {
                    vHatEff = vNew / vbc2;
                }
                // fp64 sqrt then narrow → bit-identical to the scalar `(float)Math.Sqrt(vHatEff)`.
                SnVec.Widen(vHatEff, out var vhLow, out var vhHigh);
                var sqrtVec = SnVec.Narrow(SnVec.SquareRoot(vhLow), SnVec.SquareRoot(vhHigh));
                var pv = new System.Numerics.Vector<float>(param.Slice(i, width));
                pv -= vlr * mHat / (sqrtVec + veps);
                pv.CopyTo(param.Slice(i, width));
            }
        }
        for (; i < n; i++)
        {
            float g = grad[i];
            float mNew = b1 * m[i] + oneMinusB1 * g;
            float vNew = b2 * v[i] + oneMinusB2 * g * g;
            m[i] = mNew;
            v[i] = vNew;
            float mHat = mNew / bc1;
            float vHatEff;
            if (useAmsgrad)
            {
                float vHatNow = vNew / bc2;
                float vMaxPrev = vMax[i];
                float vMaxNew = vHatNow > vMaxPrev ? vHatNow : vMaxPrev;
                vMax[i] = vMaxNew;
                vHatEff = vMaxNew;
            }
            else
            {
                vHatEff = vNew / bc2;
            }
            param[i] -= lr * mHat / ((float)Math.Sqrt(vHatEff) + eps);
        }
    }
}
