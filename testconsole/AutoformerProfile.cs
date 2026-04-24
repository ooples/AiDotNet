using System;
using System.Diagnostics;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;

namespace AiDotNetTestConsole;

internal static class AutoformerProfile
{
    public static void Run()
    {
        Console.WriteLine("=== Autoformer profile (test-shape: TrainLength=100, AutoformerModelTests opts) ===");
        var opts = new AutoformerOptions<double>
        {
            LookbackWindow = 40,
            ForecastHorizon = 10,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            EmbeddingDim = 16,
            NumAttentionHeads = 2,
            Epochs = 3
        };
        Console.WriteLine($"LookbackWindow={opts.LookbackWindow} ForecastHorizon={opts.ForecastHorizon} EmbeddingDim={opts.EmbeddingDim} Epochs={opts.Epochs}");

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
        var model = new AutoformerModel<double>(opts);
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
