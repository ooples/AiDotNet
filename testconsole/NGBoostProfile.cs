using System;
using System.Diagnostics;
using AiDotNet.Classification.Boosting;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

internal static class NGBoostProfile
{
    public static void Run()
    {
        // Match ClassificationModelTestBase.GenerateData defaults so we profile
        // exactly what the failing CI test runs.
        var rng = new Random(42);
        const int trainSamples = 100, testSamples = 30, features = 5, numClasses = 3;
        var x = new Matrix<double>(trainSamples, features);
        var y = new Vector<double>(trainSamples);
        for (int i = 0; i < trainSamples; i++)
        {
            for (int j = 0; j < features; j++) x[i, j] = rng.NextDouble();
            y[i] = rng.Next(numClasses);
        }
        var testX = new Matrix<double>(testSamples, features);
        for (int i = 0; i < testSamples; i++)
            for (int j = 0; j < features; j++) testX[i, j] = rng.NextDouble();

        Console.WriteLine($"=== NGBoost profile (trainSamples={trainSamples}, default opts) ===");

        var ctorSw = Stopwatch.StartNew();
        var model = new NGBoostClassifier<double>();
        ctorSw.Stop();
        Console.WriteLine($"ctor    : {ctorSw.Elapsed.TotalSeconds,8:F3} s");

        var trainSw = Stopwatch.StartNew();
        try { model.Train(x, y); }
        catch (Exception ex) { trainSw.Stop(); Console.WriteLine($"Train   : {ex.GetType().Name} after {trainSw.Elapsed.TotalSeconds:F3}s — {ex.Message}"); return; }
        trainSw.Stop();
        Console.WriteLine($"Train   : {trainSw.Elapsed.TotalSeconds,8:F3} s  (CI timeout = 60s)");

        var predictSw = Stopwatch.StartNew();
        // Discard — we only care about wall-clock time for the profile.
        _ = model.Predict(testX);
        predictSw.Stop();
        Console.WriteLine($"Predict : {predictSw.Elapsed.TotalSeconds,8:F3} s");

        Console.WriteLine($"TOTAL   : {(ctorSw.Elapsed + trainSw.Elapsed + predictSw.Elapsed).TotalSeconds:F3} s");
    }
}
