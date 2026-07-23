using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

/// <summary>
/// Kernel-level micro-test for the ACEStep tape-forward corruption (see
/// ACEStepNanDiag): validates CpuEngine.TensorMatMul double output for the
/// exact problem shape [2, 44100] x [44100, 64] against a naive dot product,
/// under an active GradientTape, repeated across iterations and with/without
/// an active TensorArena — the conditions TrainWithTape creates. Reports any
/// cell that is non-finite or diverges from the naive value.
/// </summary>
internal static class TapeMatMulMicroTest
{
    public static void Run()
    {
        // Use the GLOBAL engine — the one model layers actually dispatch
        // through — not a locally pinned CpuEngine. The first run of this
        // test with `new CpuEngine()` was clean while the model NaN'd; if
        // the global engine differs (GPU auto-detect), that's the variable.
        var engine = AiDotNetEngine.Current;
        Console.WriteLine($"global engine: {engine.GetType().Name}");
        var rng = new Random(11);

        const int m = 2, k = 44100, n = 64;
        var a = new Tensor<double>([m, k]);
        var b = new Tensor<double>([k, n]);
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = (rng.NextDouble() * 2 - 1) * 0.01;

        // Naive reference for a handful of cells (full naive is 5.6M mults x cells; sample grid).
        double Naive(int row, int col)
        {
            double s = 0;
            for (int kk = 0; kk < k; kk++) s += a[row * k + kk] * b[kk * n + col];
            return s;
        }

        void Check(string label, Func<Tensor<double>> compute)
        {
            var c = compute();
            int bad = 0, diverged = 0;
            for (int i = 0; i < c.Length; i++)
            {
                double v = c[i];
                if (double.IsNaN(v) || double.IsInfinity(v)) bad++;
            }
            // Sample-validate 16 cells against naive.
            for (int s = 0; s < 16; s++)
            {
                int row = s % m, col = (s * 7) % n;
                double expect = Naive(row, col);
                double got = c[row * n + col];
                if (Math.Abs(expect - got) > 1e-6 * (1 + Math.Abs(expect))) diverged++;
            }
            Console.WriteLine($"{label}: nonFinite={bad}/{c.Length}, divergedFromNaive={diverged}/16");
        }

        for (int iter = 1; iter <= 3; iter++)
        {
            // Case 1: tape active, no arena (the diag tape-walk condition).
            using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
            {
                Check($"iter {iter} tape, no arena", () => engine.TensorMatMul(a, b));
            }

            // Case 2: tape + arena (the TrainWithTape condition).
            using (var arena = AiDotNet.Tensors.Helpers.TensorArena.Create())
            using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
            {
                Check($"iter {iter} tape + arena", () => engine.TensorMatMul(a, b));
            }

            // Case 3: no tape (the eager Predict condition).
            Check($"iter {iter} eager", () => engine.TensorMatMul(a, b));
        }
    }
}
