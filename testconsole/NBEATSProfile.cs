using System;
using System.Diagnostics;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;

namespace AiDotNetTestConsole;

internal static class NBEATSProfile
{
    public static void Run()
    {
        Console.WriteLine("=== NBEATS profile (test-shape: TrainLength=100, NBEATSModelTests opts) ===");
        // Match NBEATSModelTests.CreateModel() so we profile what CI runs.
        var opts = new NBEATSModelOptions<double>
        {
            NumStacks = 2,
            NumBlocksPerStack = 1,
            LookbackWindow = 10,
            ForecastHorizon = 5,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            MaxTrainingTimeSeconds = 5
        };
        Console.WriteLine(
            $"NumStacks={opts.NumStacks} NumBlocksPerStack={opts.NumBlocksPerStack} " +
            $"HiddenLayerSize={opts.HiddenLayerSize} NumHiddenLayers={opts.NumHiddenLayers} " +
            $"LookbackWindow={opts.LookbackWindow} ForecastHorizon={opts.ForecastHorizon} " +
            $"Epochs={opts.Epochs} BatchSize={opts.BatchSize}");

        // Deterministic synthetic signal — randomness would only add
        // variance to profiler timing without changing what NBEATS does
        // on the forward path.
        const int trainLength = 100;
        var x = new Matrix<double>(trainLength, 1);
        var y = new Vector<double>(trainLength);
        for (int i = 0; i < trainLength; i++)
        {
            double t = i;
            x[i, 0] = t;
            y[i] = 0.5 * t + 3.0 * Math.Sin(2.0 * Math.PI * t / 20.0);
        }

        var ctorSw = Stopwatch.StartNew();
        var model = new NBEATSModel<double>(opts);
        ctorSw.Stop();
        Console.WriteLine($"ctor    : {ctorSw.Elapsed.TotalSeconds,8:F3} s");

        // Guard Train/Predict so a model bug surfaces a structured timing
        // line (matching SVC/NGBoost/DeepANT profiles) instead of
        // hard-aborting the profile command.
        var trainSw = Stopwatch.StartNew();
        try
        {
            model.Train(x, y);
            trainSw.Stop();
            Console.WriteLine($"Train   : {trainSw.Elapsed.TotalSeconds,8:F3} s  (CI timeout = 60s)");
        }
        catch (Exception ex)
        {
            trainSw.Stop();
            Console.WriteLine($"Train   : {ex.GetType().Name} after {trainSw.Elapsed.TotalSeconds:F3}s — {ex.Message}");
            return;
        }

        var predictSw = Stopwatch.StartNew();
        try
        {
            _ = model.Predict(x);
            predictSw.Stop();
            Console.WriteLine($"Predict : {predictSw.Elapsed.TotalSeconds,8:F3} s");
        }
        catch (Exception ex)
        {
            predictSw.Stop();
            Console.WriteLine($"Predict : {ex.GetType().Name} after {predictSw.Elapsed.TotalSeconds:F3}s — {ex.Message}");
            return;
        }

        Console.WriteLine($"TOTAL   : {(ctorSw.Elapsed + trainSw.Elapsed + predictSw.Elapsed).TotalSeconds:F3} s");
    }
}
