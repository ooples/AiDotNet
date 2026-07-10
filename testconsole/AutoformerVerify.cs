using System;
using System.Diagnostics;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;

namespace AiDotNetTestConsole;

// Verify harness for the Autoformer tape-training conversion.
// Trains AutoformerModel<double> on a learnable noisy sinusoid+trend and asserts:
//   1. the trained multi-horizon forecast MSE is < 0.5x the untrained model's MSE, and
//   2. it beats a naive repeat-last-value baseline.
// Trains ONE epoch per Train() call so per-epoch MSE is visible (streamed to the redirect file).
// Invoke:  dotnet run -c Release --framework net10.0 -- autoformer-verify
internal static class AutoformerVerify
{
    public static void Run()
    {
        Console.WriteLine("=== Autoformer tape-training verify ===");

        // Reduced embedding dim / single encoder layer so a CPU verify finishes quickly (the
        // 512-dim, 2-encoder default is intractable for a quick CPU run because the time-domain
        // auto-correlation builds O(L) small tensor nodes per attention call). Everything else is
        // a realistic small Autoformer; MovingAverageKernel=25 is the model default.
        var opts = new AutoformerOptions<double>
        {
            LookbackWindow = 32,
            ForecastHorizon = 10,
            EmbeddingDim = 16,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumAttentionHeads = 2,
            MovingAverageKernel = 25,
            BatchSize = 16,
            LearningRate = 0.02,
            Epochs = 1 // one epoch per Train() call; the loop below drives the epoch count
        };
        const int totalEpochs = 12;
        Console.WriteLine($"opts: lookback={opts.LookbackWindow} horizon={opts.ForecastHorizon} " +
            $"embDim={opts.EmbeddingDim} enc={opts.NumEncoderLayers} dec={opts.NumDecoderLayers} " +
            $"heads={opts.NumAttentionHeads} batch={opts.BatchSize} lr={opts.LearningRate} epochs={totalEpochs}");

        // Learnable series: trend + two seasonal components + small noise, ~600 points.
        const int n = 600;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);
        var rng = new Random(1234);
        for (int i = 0; i < n; i++)
        {
            double t = i;
            double val = 0.03 * t
                       + 4.0 * Math.Sin(2.0 * Math.PI * t / 50.0)
                       + 1.5 * Math.Sin(2.0 * Math.PI * t / 12.0)
                       + (rng.NextDouble() * 2 - 1) * 0.25;
            x[i, 0] = t;
            y[i] = val;
        }

        int lookback = opts.LookbackWindow;
        int horizon = opts.ForecastHorizon;

        var model = new AutoformerModel<double>(opts);

        double beforeMse = EvalHorizonMse(model, y, lookback, horizon);
        double baselineMse = NaiveRepeatLastMse(y, lookback, horizon);
        Console.WriteLine($"BEFORE (untrained) horizon MSE : {beforeMse:F6}");
        Console.WriteLine($"BASELINE (repeat-last) horizon MSE: {baselineMse:F6}");
        Console.Out.Flush();

        double afterMse = beforeMse;
        var swAll = Stopwatch.StartNew();
        for (int e = 1; e <= totalEpochs; e++)
        {
            var sw = Stopwatch.StartNew();
            model.Train(x, y); // one epoch (opts.Epochs == 1)
            sw.Stop();
            afterMse = EvalHorizonMse(model, y, lookback, horizon);
            Console.WriteLine($"epoch {e,2}: horizon MSE = {afterMse:F6}   ({sw.Elapsed.TotalSeconds:F1}s)");
            Console.Out.Flush();
        }
        swAll.Stop();
        Console.WriteLine($"total train time: {swAll.Elapsed.TotalSeconds:F1} s");

        double ratio = beforeMse > 0 ? afterMse / beforeMse : double.NaN;
        Console.WriteLine($"AFTER (trained) horizon MSE : {afterMse:F6}");
        Console.WriteLine($"after/before ratio : {ratio:F4}  (want < 0.5)");
        Console.WriteLine($"after < baseline   : {afterMse < baselineMse}  ({afterMse:F6} vs {baselineMse:F6})");

        bool decreased = afterMse < 0.5 * beforeMse;
        bool beatsBaseline = afterMse < baselineMse;
        Console.WriteLine();
        Console.WriteLine(decreased ? "PASS: training loss decreased materially (final < 0.5x initial)"
                                    : "FAIL: training loss did NOT drop below 0.5x initial");
        Console.WriteLine(beatsBaseline ? "PASS: beats naive repeat-last baseline"
                                        : "FAIL: does NOT beat naive repeat-last baseline");
        Console.WriteLine(decreased && beatsBaseline ? "OVERALL: PASS" : "OVERALL: FAIL");
    }

    // Average MSE of the model's full-horizon forecast over valid lookback windows (strided).
    private static double EvalHorizonMse(AutoformerModel<double> model, Vector<double> y, int lookback, int horizon)
    {
        double sum = 0.0;
        long count = 0;
        for (int idx = lookback; idx + horizon <= y.Length; idx += 4)
        {
            var window = new Vector<double>(lookback);
            for (int t = 0; t < lookback; t++) window[t] = y[idx - lookback + t];
            var forecast = model.PredictMultiple(window);
            for (int h = 0; h < horizon; h++)
            {
                double e = forecast[h] - y[idx + h];
                sum += e * e;
                count++;
            }
        }
        return count > 0 ? sum / count : double.NaN;
    }

    // Naive baseline: forecast every horizon step as the last observed value in the window.
    private static double NaiveRepeatLastMse(Vector<double> y, int lookback, int horizon)
    {
        double sum = 0.0;
        long count = 0;
        for (int idx = lookback; idx + horizon <= y.Length; idx += 4)
        {
            double last = y[idx - 1];
            for (int h = 0; h < horizon; h++)
            {
                double e = last - y[idx + h];
                sum += e * e;
                count++;
            }
        }
        return count > 0 ? sum / count : double.NaN;
    }
}
