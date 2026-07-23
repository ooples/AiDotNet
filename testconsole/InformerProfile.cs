using System;
using System.Diagnostics;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors;

namespace AiDotNetTestConsole;

internal static class InformerProfile
{
    // Deterministic learnable signal: trend + two sinusoids + small noise.
    private static double Signal(int t, Random rng)
        => 0.02 * t
         + 3.0 * Math.Sin(2.0 * Math.PI * t / 50.0)
         + 1.5 * Math.Sin(2.0 * Math.PI * t / 13.0)
         + 0.3 * (rng.NextDouble() * 2 - 1);

    // Mean squared error of the model's H-step forecast over sliding windows,
    // compared against a naive repeat-last-value baseline.
    private static (double model, double baseline) WindowedMse(
        InformerModel<double> model, double[] series, int lookback, int horizon, int stride)
    {
        double sumM = 0, sumB = 0; int n = 0;
        for (int i = lookback; i + horizon <= series.Length; i += stride)
        {
            var window = new Vector<double>(lookback);
            for (int j = 0; j < lookback; j++) window[j] = series[i - lookback + j];
            Vector<double> fc;
            try { fc = model.ForecastHorizon(window); }
            catch { continue; }
            double last = series[i - 1];
            for (int h = 0; h < horizon; h++)
            {
                double actual = series[i + h];
                double em = fc[h] - actual; sumM += em * em;
                double eb = last - actual; sumB += eb * eb;
                n++;
            }
        }
        return n == 0 ? (double.NaN, double.NaN) : (sumM / n, sumB / n);
    }

    public static void RunVerify()
    {
        Console.WriteLine("=== Informer CORRECTNESS verify (double, CPU) ===");
        var opts = new InformerOptions<double>
        {
            LookbackWindow = 96,
            ForecastHorizon = 24,
            EmbeddingDim = 64,
            NumEncoderLayers = 2,
            NumDecoderLayers = 1,
            NumAttentionHeads = 8,
            Epochs = 40,
            BatchSize = 32,
            LearningRate = 1e-3
        };
        const int trainLength = 1200;
        var rng = new Random(7);
        var x = new Matrix<double>(trainLength, 1);
        var y = new Vector<double>(trainLength);
        var series = new double[trainLength];
        for (int i = 0; i < trainLength; i++)
        {
            series[i] = Signal(i, rng);
            x[i, 0] = i;
            y[i] = series[i];
        }

        Console.WriteLine($"opts: L={opts.LookbackWindow} H={opts.ForecastHorizon} d={opts.EmbeddingDim} " +
            $"enc={opts.NumEncoderLayers} dec={opts.NumDecoderLayers} heads={opts.NumAttentionHeads} " +
            $"epochs={opts.Epochs} batch={opts.BatchSize} lr={opts.LearningRate}");

        var model = new InformerModel<double>(opts);
        Console.WriteLine($"ParameterCount = {model.ParameterCount}");

        var (initM, initB) = WindowedMse(model, series, opts.LookbackWindow, opts.ForecastHorizon, 8);
        Console.WriteLine($"UNTRAINED windowed MSE = {initM:F4}  (repeat-last baseline = {initB:F4})");

        var sw = Stopwatch.StartNew();
        model.Train(x, y);
        sw.Stop();
        Console.WriteLine($"Train time = {sw.Elapsed.TotalSeconds:F2} s");

        var (finM, finB) = WindowedMse(model, series, opts.LookbackWindow, opts.ForecastHorizon, 8);
        Console.WriteLine($"TRAINED   windowed MSE = {finM:F4}  (repeat-last baseline = {finB:F4})");

        double ratio = initM > 0 ? finM / initM : double.NaN;
        Console.WriteLine($"ratio final/initial = {ratio:F4}  (need < 0.5)");
        Console.WriteLine($"beats naive repeat-last? {(finM < finB ? "YES" : "NO")}  ({finM:F4} vs {finB:F4})");
        Console.WriteLine(finM < 0.5 * initM && finM < finB
            ? "RESULT: PASS"
            : "RESULT: FAIL");

        // Probe a couple of concrete forecasts.
        int probe = trainLength - opts.ForecastHorizon - 1;
        var w = new Vector<double>(opts.LookbackWindow);
        for (int j = 0; j < opts.LookbackWindow; j++) w[j] = series[probe - opts.LookbackWindow + j];
        var pf = model.ForecastHorizon(w);
        Console.WriteLine($"probe forecast[0..3] = {pf[0]:F3},{pf[1]:F3},{pf[2]:F3}  " +
            $"actual = {series[probe]:F3},{series[probe + 1]:F3},{series[probe + 2]:F3}");
    }

    public static void RunGpu()
    {
        Console.WriteLine("=== Informer GPU utilization run (float32) ===");
        bool gpu = AiDotNet.Tensors.Engines.AiDotNetEngine.AutoDetectAndConfigureGpu();
        Console.WriteLine($"AutoDetectAndConfigureGpu = {gpu}  Engine = {AiDotNet.Tensors.Engines.AiDotNetEngine.Current.GetType().Name}");

        var opts = new InformerOptions<float>
        {
            LookbackWindow = 96,
            ForecastHorizon = 24,
            EmbeddingDim = 64,
            NumEncoderLayers = 2,
            NumDecoderLayers = 1,
            NumAttentionHeads = 8,
            Epochs = 12,
            BatchSize = 64,
            LearningRate = 1e-3
        };
        const int trainLength = 6000;
        var rng = new Random(7);
        var x = new Matrix<float>(trainLength, 1);
        var y = new Vector<float>(trainLength);
        for (int i = 0; i < trainLength; i++)
        {
            float v = (float)Signal(i, rng);
            x[i, 0] = i;
            y[i] = v;
        }

        Console.WriteLine($"opts: L={opts.LookbackWindow} H={opts.ForecastHorizon} d={opts.EmbeddingDim} " +
            $"heads={opts.NumAttentionHeads} epochs={opts.Epochs} batch={opts.BatchSize} trainLen={trainLength}");
        var model = new InformerModel<float>(opts);
        Console.WriteLine($"ParameterCount = {model.ParameterCount}");

        Console.WriteLine("TRAIN_BEGIN");
        Console.Out.Flush();
        var sw = Stopwatch.StartNew();
        model.Train(x, y);
        sw.Stop();
        Console.WriteLine("TRAIN_END");
        Console.WriteLine($"Train time = {sw.Elapsed.TotalSeconds:F2} s");
    }
}
