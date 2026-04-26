using System;
using System.Diagnostics;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries.AnomalyDetection;

namespace AiDotNetTestConsole;

internal static class DeepANTProfile
{
    public static void Run()
    {
        Console.WriteLine("=== DeepANT profile (test-shape: TrainLength=100, default opts) ===");
        var opts = new DeepANTOptions<double>();
        Console.WriteLine($"WindowSize={opts.WindowSize} Epochs={opts.Epochs} BatchSize={opts.BatchSize}");

        // Deterministic synthetic signal — a linear ramp + 1.0-period
        // sinusoid is enough to exercise DeepANT's anomaly-detection
        // path under the profiler without randomness adding variance
        // to the timing measurements.
        const int trainLength = 100;
        var x = new Matrix<double>(trainLength, opts.WindowSize);
        var y = new Vector<double>(trainLength);
        for (int i = 0; i < trainLength; i++)
        {
            for (int j = 0; j < opts.WindowSize; j++) x[i, j] = i + j;
            y[i] = 0.5 * i + 3.0 * Math.Sin(2.0 * Math.PI * i / 20.0);
        }

        var ctorSw = Stopwatch.StartNew();
        var model = new DeepANT<double>(opts);
        ctorSw.Stop();
        Console.WriteLine($"ctor    : {ctorSw.Elapsed.TotalSeconds,8:F3} s");

        var trainSw = Stopwatch.StartNew();
        model.Train(x, y);
        trainSw.Stop();
        Console.WriteLine($"Train   : {trainSw.Elapsed.TotalSeconds,8:F3} s  (CI timeout = 60s)");

        var predictSw = Stopwatch.StartNew();
        var pred = model.Predict(x);
        predictSw.Stop();
        Console.WriteLine($"Predict : {predictSw.Elapsed.TotalSeconds,8:F3} s");

        Console.WriteLine($"TOTAL   : {(ctorSw.Elapsed + trainSw.Elapsed + predictSw.Elapsed).TotalSeconds:F3} s");
    }
}
