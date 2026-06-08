using System.Diagnostics;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Benchmarks;

/// <summary>
/// Lightweight training time + allocation survey across every concrete regression model, used to
/// find real performance bottlenecks (algorithmic + GC) rather than micro-optimizing blindly.
/// Run with: dotnet run -c Release -- survey [n] [p]
/// </summary>
internal static class RegressionSurvey
{
    public static void Run(int n, int p)
    {
        var (x, y) = MakeData(n, p);
        var iface = typeof(IFullModel<,,>).MakeGenericType(typeof(double), typeof(Matrix<double>), typeof(Vector<double>));
        var trainSig = new[] { typeof(Matrix<double>), typeof(Vector<double>) };

        var asm = typeof(AiDotNet.Regression.MultipleRegression<>).Assembly;
        var rows = new List<(string Name, double Ms, long Bytes, string Note)>();

        foreach (var t in asm.GetTypes())
        {
            if (t.Namespace != "AiDotNet.Regression" || !t.IsClass || t.IsAbstract) continue;
            Type closed;
            try { closed = t.IsGenericTypeDefinition ? t.MakeGenericType(typeof(double)) : t; }
            catch { continue; }
            if (!iface.IsAssignableFrom(closed)) continue;

            object? model = TryConstruct(closed);
            if (model is null) continue;
            var train = closed.GetMethod("Train", trainSig);
            if (train is null) continue;

            try
            {
                // Warm up (JIT) on a tiny slice, then measure on the full set.
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                long before = GC.GetAllocatedBytesForCurrentThread();
                var sw = Stopwatch.StartNew();
                train.Invoke(model, new object[] { x, y });
                sw.Stop();
                long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
                rows.Add((closed.Name.Replace("`1", ""), sw.Elapsed.TotalMilliseconds, bytes, ""));
            }
            catch (Exception ex)
            {
                var inner = ex.InnerException ?? ex;
                rows.Add((closed.Name.Replace("`1", ""), -1, 0, inner.GetType().Name));
            }
        }

        Console.WriteLine($"\n=== Regression training survey: n={n}, p={p} ({rows.Count} models) ===");
        Console.WriteLine($"{"model",-40}{"train ms",12}{"alloc MB",12}  note");
        foreach (var r in rows.OrderByDescending(r => r.Ms))
        {
            string ms = r.Ms < 0 ? "ERR" : r.Ms.ToString("F1");
            Console.WriteLine($"{r.Name,-40}{ms,12}{r.Bytes / 1_048_576.0,12:F1}  {r.Note}");
        }
    }

    private static object? TryConstruct(Type closed)
    {
        // Prefer a ctor whose parameters are all optional (options/regularization default to null).
        foreach (var c in closed.GetConstructors().OrderBy(c => c.GetParameters().Length))
        {
            var ps = c.GetParameters();
            if (ps.Length == 0)
            {
                try { return c.Invoke(null); } catch { }
            }
            else if (ps.All(pp => pp.IsOptional))
            {
                try { return c.Invoke(Enumerable.Repeat(Type.Missing, ps.Length).ToArray()); } catch { }
            }
        }

        return null;
    }

    private static (Matrix<double> X, Vector<double> Y) MakeData(int n, int p)
    {
        // Deterministic synthetic regression: y = Xβ + small noise, features ~ U(0,1), y kept
        // positive and in a moderate range so GLM-family links don't immediately diverge.
        var rnd = new Random(12345);
        var xData = new double[n, p];
        var yData = new double[n];
        var beta = new double[p];
        for (int j = 0; j < p; j++) beta[j] = 0.5 + rnd.NextDouble();
        for (int i = 0; i < n; i++)
        {
            double yi = 1.0;
            for (int j = 0; j < p; j++)
            {
                double v = rnd.NextDouble();
                xData[i, j] = v;
                yi += beta[j] * v;
            }

            yData[i] = yi + (rnd.NextDouble() - 0.5) * 0.1;
        }

        var x = new Matrix<double>(n, p);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                x[i, j] = xData[i, j];
        var y = new Vector<double>(yData);
        return (x, y);
    }
}
