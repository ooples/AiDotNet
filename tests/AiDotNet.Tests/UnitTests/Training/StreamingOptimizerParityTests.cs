// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// Parity tests for the 8-bit block-quantized streaming optimizers (#1600 acceptance criterion:
/// "each streaming variant matches its full-precision counterpart within tolerance over N steps").
/// Each streaming optimizer is run against an INDEPENDENT fp64 reference of the SAME canonical
/// update rule, on the same fixed gradient sequence; the test asserts the 8-bit parameter
/// trajectory tracks the fp64 one within a bounded relative error — i.e. the rule is correct AND
/// the per-block 8-bit moment quantization doesn't drift/diverge over many steps.
/// </summary>
public class StreamingOptimizerParityTests
{
    private const int N = 512;     // one 2048-block, so per-block scale behaviour is exercised
    private const int Steps = 60;
    private const double Lr = 1e-2;

    // Per-element fp64 reference update: (param, grad, state[]) -> newParam, mutating state in place.
    private delegate double RefUpdate(double param, double grad, double[] state, int step);

    private static double[] Grad(int step)
    {
        var g = new double[N];
        uint s = (uint)(step * 2654435761u + 12345u);
        for (int i = 0; i < N; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            g[i] = ((s & 0xFFFF) / 65535.0 - 0.5) * 0.2; // ~[-0.1, 0.1]
        }
        return g;
    }

    private static double[] InitParams()
    {
        var p = new double[N];
        uint s = 99u;
        for (int i = 0; i < N; i++) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; p[i] = ((s & 0xFFFF) / 65535.0 - 0.5); }
        return p;
    }

    private static double RelL2(double[] a, double[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { double e = a[i] - b[i]; num += e * e; den += b[i] * b[i]; }
        return Math.Sqrt(num / Math.Max(1e-30, den));
    }

    private void RunParity(string name, IStreamingOptimizer<double> opt, RefUpdate reference, int momentCount, double tol)
    {
        var streamParam = new Tensor<double>(new[] { N });
        var refParam = InitParams();
        for (int i = 0; i < N; i++) streamParam[i] = refParam[i];
        var refState = new double[N * Math.Max(1, momentCount)];

        for (int step = 1; step <= Steps; step++)
        {
            var g = Grad(step);

            opt.BeginStep();
            var gradTensor = new Tensor<double>(new[] { N });
            for (int i = 0; i < N; i++) gradTensor[i] = g[i];
            opt.Apply(streamParam, gradTensor);

            for (int i = 0; i < N; i++)
            {
                var slot = momentCount == 0 ? Array.Empty<double>() : new double[momentCount];
                for (int m = 0; m < momentCount; m++) slot[m] = refState[i * momentCount + m];
                refParam[i] = reference(refParam[i], g[i], slot, step);
                for (int m = 0; m < momentCount; m++) refState[i * momentCount + m] = slot[m];
            }
        }

        var streamArr = new double[N];
        for (int i = 0; i < N; i++) streamArr[i] = streamParam[i];
        double rel = RelL2(streamArr, refParam);
        Assert.True(rel < tol, $"{name}: 8-bit streaming drifted from fp64 reference (relL2 {rel:E3} >= {tol}).");
    }

    [Fact] public void Sgd_TracksFp64()
        => RunParity("SGD", new StreamingSgd8Bit<double>(Lr), (p, g, st, t) => p - Lr * g, 0, 1e-9);

    [Fact] public void Momentum_TracksFp64()
        => RunParity("Momentum", new StreamingMomentum8Bit<double>(Lr, 0.9),
            (p, g, st, t) => { st[0] = 0.9 * st[0] + Lr * g; return p - st[0]; }, 1, 0.03);

    [Fact] public void Nesterov_TracksFp64()
        => RunParity("Nesterov", new StreamingNesterov8Bit<double>(Lr, 0.9),
            (p, g, st, t) => { st[0] = 0.9 * st[0] + Lr * g; return p - (0.9 * st[0] + Lr * g); }, 1, 0.03);

    [Fact] public void RmsProp_TracksFp64()
        => RunParity("RMSProp", new StreamingRmsProp8Bit<double>(Lr, 0.9, 1e-8),
            (p, g, st, t) => { st[0] = 0.9 * st[0] + 0.1 * g * g; return p - Lr * g / (Math.Sqrt(st[0]) + 1e-8); }, 1, 0.05);

    [Fact] public void Adagrad_TracksFp64()
        => RunParity("Adagrad", new StreamingAdagrad8Bit<double>(Lr, 1e-8),
            (p, g, st, t) => { st[0] += g * g; return p - Lr * g / (Math.Sqrt(st[0]) + 1e-8); }, 1, 0.05);

    [Fact] public void Adam_TracksFp64()
        => RunParity("Adam", new StreamingAdam8Bit<double>(Lr), (p, g, st, t) =>
        {
            st[0] = 0.9 * st[0] + 0.1 * g; st[1] = 0.999 * st[1] + 0.001 * g * g;
            double c1 = 1 - Math.Pow(0.9, t), c2 = 1 - Math.Pow(0.999, t);
            return p - Lr * (st[0] / c1) / (Math.Sqrt(st[1] / c2) + 1e-8);
        }, 2, 0.05);

    [Fact] public void AmsGrad_TracksFp64()
        => RunParity("AMSGrad", new StreamingAMSGrad8Bit<double>(Lr, 0.9, 0.999, 1e-8, 0.0), (p, g, st, t) =>
        {
            st[0] = 0.9 * st[0] + 0.1 * g; st[1] = 0.999 * st[1] + 0.001 * g * g; st[2] = Math.Max(st[2], st[1]);
            double c1 = 1 - Math.Pow(0.9, t), c2 = 1 - Math.Pow(0.999, t);
            return p - Lr * (st[0] / c1) / (Math.Sqrt(st[2] / c2) + 1e-8);
        }, 3, 0.05);

    [Fact] public void Nadam_TracksFp64()
        => RunParity("Nadam", new StreamingNadam8Bit<double>(Lr, 0.9, 0.999, 1e-8), (p, g, st, t) =>
        {
            st[0] = 0.9 * st[0] + 0.1 * g; st[1] = 0.999 * st[1] + 0.001 * g * g;
            double c1 = 1 - Math.Pow(0.9, t), c2 = 1 - Math.Pow(0.999, t);
            double mHat = st[0] / c1, vHat = st[1] / c2;
            double nest = 0.9 * mHat + 0.1 * g / c1;
            return p - Lr * nest / (Math.Sqrt(vHat) + 1e-8);
        }, 2, 0.05);

    [Fact] public void AdaMax_TracksFp64()
        => RunParity("AdaMax", new StreamingAdaMax8Bit<double>(Lr, 0.9, 0.999, 1e-8), (p, g, st, t) =>
        {
            st[0] = 0.9 * st[0] + 0.1 * g; st[1] = Math.Max(0.999 * st[1], Math.Abs(g));
            double c1 = 1 - Math.Pow(0.9, t);
            return p - (Lr / c1) * st[0] / (st[1] + 1e-8);
        }, 2, 0.05);

    [Fact] public void AdaDelta_TracksFp64()
        => RunParity("AdaDelta", new StreamingAdaDelta8Bit<double>(Lr, 0.95, 1e-6), (p, g, st, t) =>
        {
            double accG = 0.95 * st[0] + 0.05 * g * g;
            double delta = Math.Sqrt(st[1] + 1e-6) / Math.Sqrt(accG + 1e-6) * g;
            st[1] = 0.95 * st[1] + 0.05 * delta * delta;
            st[0] = accG;
            return p - Lr * delta;
        }, 2, 0.06);

    [Fact] public void AdamW_TracksFp64()
        => RunParity("AdamW", new StreamingAdamW8Bit<double>(Lr, 0.9, 0.999, 1e-8, 0.01), (p, g, st, t) =>
        {
            st[0] = 0.9 * st[0] + 0.1 * g; st[1] = 0.999 * st[1] + 0.001 * g * g;
            double c1 = 1 - Math.Pow(0.9, t), c2 = 1 - Math.Pow(0.999, t);
            p -= Lr * 0.01 * p; // decoupled weight decay
            return p - Lr * (st[0] / c1) / (Math.Sqrt(st[1] / c2) + 1e-8);
        }, 2, 0.05);

    // Lion's update is lr*sign(interp(m,g)); the 8-bit momentum only flips the SIGN near zero, so
    // a looser bound (a few sign disagreements over 60 steps) is expected and still meaningful.
    [Fact] public void Lion_TracksFp64()
        => RunParity("Lion", new StreamingLion8Bit<double>(Lr, 0.9, 0.99, 0.0), (p, g, st, t) =>
        {
            double dir = 0.9 * st[0] + 0.1 * g;
            st[0] = 0.99 * st[0] + 0.01 * g;
            return p - Lr * Math.Sign(dir);
        }, 1, 0.12);

    // FTRL replaces the parameter from its (z, n) accumulators (closed form), so it's per-element.
    // lambda1 is the L1 strength; use a small value so the soft-threshold actually fires.
    [Fact] public void Ftrl_TracksFp64()
        => RunParity("FTRL", new StreamingFtrl8Bit<double>(Lr, 1.0, 1e-3, 1e-3), (p, g, st, t) =>
        {
            double nNew = st[1] + g * g;
            double sigma = (Math.Sqrt(nNew) - Math.Sqrt(st[1])) / Lr;
            double zNew = st[0] + g - sigma * p;
            st[0] = zNew; st[1] = nNew;
            if (Math.Abs(zNew) <= 1e-3) return 0.0;
            double num = -Math.Sign(zNew) * (Math.Abs(zNew) - 1e-3);
            double den = 1e-3 + (Math.Sqrt(nNew) + 1.0) / Lr;
            return num / den;
        }, 2, 0.06);

    // LARS/LAMB need the per-PARAMETER norm (a full-param pre-pass), so they get a dedicated driver.
    [Fact] public void Lars_TracksFp64()
    {
        const double momentum = 0.9, wd = 1e-4, trust = 0.001, eps = 1e-8;
        var opt = new StreamingLars8Bit<double>(Lr, momentum, wd, trust, eps, false);
        var sp = new Tensor<double>(new[] { N }); var rp = InitParams(); var vel = new double[N];
        for (int i = 0; i < N; i++) sp[i] = rp[i];
        for (int step = 1; step <= Steps; step++)
        {
            var g = Grad(step);
            opt.BeginStep(); var gt = new Tensor<double>(new[] { N }); for (int i = 0; i < N; i++) gt[i] = g[i];
            opt.Apply(sp, gt);
            double pn = 0, gn = 0; for (int i = 0; i < N; i++) { pn += rp[i] * rp[i]; gn += g[i] * g[i]; }
            pn = Math.Sqrt(pn); gn = Math.Sqrt(gn);
            double localLr = (pn > 0 && gn > 0) ? Lr * trust * pn / (gn + wd * pn + eps) : Lr;
            for (int i = 0; i < N; i++) { vel[i] = momentum * vel[i] + localLr * (g[i] + wd * rp[i]); rp[i] -= vel[i]; }
        }
        var sa = new double[N]; for (int i = 0; i < N; i++) sa[i] = sp[i];
        double rel = RelL2(sa, rp);
        Assert.True(rel < 0.05, $"LARS: 8-bit drifted from fp64 (relL2 {rel:E3}).");
    }

    [Fact] public void Lamb_TracksFp64()
    {
        const double b1 = 0.9, b2 = 0.999, eps = 1e-6, wd = 0.01, maxTrust = 10.0;
        var opt = new StreamingLamb8Bit<double>(Lr, b1, b2, eps, wd, true, maxTrust, true);
        var sp = new Tensor<double>(new[] { N }); var rp = InitParams(); var m = new double[N]; var v = new double[N];
        for (int i = 0; i < N; i++) sp[i] = rp[i];
        for (int step = 1; step <= Steps; step++)
        {
            var g = Grad(step);
            opt.BeginStep(); var gt = new Tensor<double>(new[] { N }); for (int i = 0; i < N; i++) gt[i] = g[i];
            opt.Apply(sp, gt);
            double c1 = 1 - Math.Pow(b1, step), c2 = 1 - Math.Pow(b2, step);
            double pn = 0, un = 0;
            for (int i = 0; i < N; i++)
            {
                double mm = b1 * m[i] + (1 - b1) * g[i], vv = b2 * v[i] + (1 - b2) * g[i] * g[i];
                double u = (mm / c1) / (Math.Sqrt(vv / c2) + eps) + wd * rp[i];
                pn += rp[i] * rp[i]; un += u * u;
            }
            pn = Math.Sqrt(pn); un = Math.Sqrt(un);
            double tr = (pn > 0 && un > 0) ? Math.Min(pn / un, maxTrust) : 1.0;
            for (int i = 0; i < N; i++)
            {
                m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
                double u = (m[i] / c1) / (Math.Sqrt(v[i] / c2) + eps) + wd * rp[i];
                rp[i] -= Lr * tr * u;
            }
        }
        var sa = new double[N]; for (int i = 0; i < N; i++) sa[i] = sp[i];
        double rel = RelL2(sa, rp);
        Assert.True(rel < 0.05, $"LAMB: 8-bit drifted from fp64 (relL2 {rel:E3}).");
    }

    private static double Dot(double[] a, double[] b) { double s = 0; for (int i = 0; i < a.Length; i++) s += a[i] * b[i]; return s; }

    // fp64 reference two-loop recursion (full-precision (s,y) history) — identical algorithm to
    // StreamingLBFGS.TwoLoopDirection but without the 8-bit history quantization.
    private static double[] RefLbfgsDirection(double[] g, System.Collections.Generic.List<double[]> sH, System.Collections.Generic.List<double[]> yH)
    {
        int n = g.Length, k = sH.Count;
        var q = (double[])g.Clone();
        if (k == 0) return q;
        var alpha = new double[k]; var rho = new double[k];
        for (int i = k - 1; i >= 0; i--)
        {
            double ys = Dot(yH[i], sH[i]); rho[i] = ys != 0 ? 1.0 / ys : 0.0;
            alpha[i] = rho[i] * Dot(sH[i], q);
            for (int j = 0; j < n; j++) q[j] -= alpha[i] * yH[i][j];
        }
        double yy = Dot(yH[k - 1], yH[k - 1]);
        double gamma = yy > 0 ? Dot(sH[k - 1], yH[k - 1]) / yy : 1.0;
        if (gamma <= 0 || double.IsNaN(gamma) || double.IsInfinity(gamma)) gamma = 1.0;
        var r = new double[n]; for (int j = 0; j < n; j++) r[j] = gamma * q[j];
        for (int i = 0; i < k; i++)
        {
            double beta = rho[i] * Dot(yH[i], r);
            for (int j = 0; j < n; j++) r[j] += (alpha[i] - beta) * sH[i][j];
        }
        return r;
    }

    // Streaming L-BFGS (second-order) with an 8-bit-quantized (s,y) history must track an fp64
    // L-BFGS (full-precision history) running the SAME two-loop + fixed step. The only divergence
    // source is the 8-bit history quantization; the test asserts it stays bounded over many steps.
    [Fact]
    public void LBFGS_8bitHistory_TracksFp64()
    {
        const int memSize = 10;
        var opt = new StreamingLBFGS<double>(Lr, memSize);
        var sp = new Tensor<double>(new[] { N });
        var rp = InitParams();
        for (int i = 0; i < N; i++) sp[i] = rp[i];

        var sH = new System.Collections.Generic.List<double[]>();
        var yH = new System.Collections.Generic.List<double[]>();
        double[]? prevGrad = null;

        for (int step = 1; step <= Steps; step++)
        {
            var g = Grad(step);

            opt.BeginStep();
            var gt = new Tensor<double>(new[] { N });
            for (int i = 0; i < N; i++) gt[i] = g[i];
            opt.Apply(sp, gt);
            opt.EndStep();

            // fp64 reference: same two-loop, full-precision history.
            var d = RefLbfgsDirection(g, sH, yH);
            var xNew = new double[N];
            var s = new double[N];
            for (int i = 0; i < N; i++) { xNew[i] = rp[i] - Lr * d[i]; s[i] = xNew[i] - rp[i]; }
            if (prevGrad is not null)
            {
                var y = new double[N];
                for (int i = 0; i < N; i++) y[i] = g[i] - prevGrad[i];
                if (Dot(s, y) > 1e-12)
                {
                    sH.Add(s); yH.Add(y);
                    if (sH.Count > memSize) { sH.RemoveAt(0); yH.RemoveAt(0); }
                }
            }
            rp = xNew;
            prevGrad = (double[])g.Clone();
        }

        var sa = new double[N]; for (int i = 0; i < N; i++) sa[i] = sp[i];
        double rel = RelL2(sa, rp);
        Assert.True(rel < 0.10, $"Streaming L-BFGS (8-bit history) drifted from fp64 L-BFGS (relL2 {rel:E3} >= 0.10).");
    }

    private static double[] GradN(int n, int step)
    {
        var g = new double[n];
        uint s = (uint)(step * 2654435761u + 12345u);
        for (int i = 0; i < n; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            g[i] = ((s & 0xFFFF) / 65535.0 - 0.5) * 0.2;
        }
        return g;
    }

    private static double[] InitParamsN(int n)
    {
        var p = new double[n];
        uint s = 99u;
        for (int i = 0; i < n; i++) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; p[i] = ((s & 0xFFFF) / 65535.0 - 0.5); }
        return p;
    }

    private static double[] RunAdamLarge(int n)
    {
        var opt = new StreamingAdam8Bit<double>(Lr);
        var p = new Tensor<double>(new[] { n });
        var init = InitParamsN(n);
        for (int i = 0; i < n; i++) p[i] = init[i];
        for (int step = 1; step <= Steps; step++)
        {
            var g = GradN(n, step);
            opt.BeginStep();
            var gt = new Tensor<double>(new[] { n });
            for (int i = 0; i < n; i++) gt[i] = g[i];
            opt.Apply(p, gt);
            opt.EndStep();
        }
        var outArr = new double[n];
        for (int i = 0; i < n; i++) outArr[i] = p[i];
        return outArr;
    }

    // The block loop parallelizes over DISJOINT parameter slices, so for a parameter large enough
    // to take the parallel path (40000 > the 1<<15 threshold) the result must (a) still track the
    // fp64 Adam reference within the same tolerance as the small serial case — no drift from a race
    // — and (b) be perfectly reproducible across runs — no nondeterminism from a race.
    [Fact]
    public void Adam_LargeTensor_ParallelBlockLoop_TracksFp64AndIsDeterministic()
    {
        const int n = 40000;

        // fp64 reference (independent of the streaming optimizer).
        var refParam = InitParamsN(n);
        var m = new double[n];
        var v = new double[n];
        for (int step = 1; step <= Steps; step++)
        {
            var g = GradN(n, step);
            double c1 = 1 - Math.Pow(0.9, step), c2 = 1 - Math.Pow(0.999, step);
            for (int i = 0; i < n; i++)
            {
                m[i] = 0.9 * m[i] + 0.1 * g[i];
                v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
                refParam[i] -= Lr * (m[i] / c1) / (Math.Sqrt(v[i] / c2) + 1e-8);
            }
        }

        var run1 = RunAdamLarge(n);
        double rel = RelL2(run1, refParam);
        Assert.True(rel < 0.05, $"Adam (parallel block loop, n={n}) drifted from fp64 reference (relL2 {rel:E3} >= 0.05).");

        // Determinism: a second identical run must be bit-for-bit identical (a data race in the
        // parallel loop would surface here as a mismatch).
        var run2 = RunAdamLarge(n);
        for (int i = 0; i < n; i++)
            Assert.Equal(run1[i], run2[i]);
    }
}
