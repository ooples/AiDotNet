using System;
using System.Diagnostics;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries.AnomalyDetection;

namespace AiDotNetTestConsole;

// Standalone profiling harness: runs LSTMVAE at the exact shape the
// TimeSeriesModelTestBase tests use (TrainLength=100, default LSTMVAEOptions
// — WindowSize=50, HiddenSize=64, LatentDim=20, Epochs=50, BatchSize=32).
// The CI test times out at 60s on this exact configuration; this harness lets
// dotnet-trace pinpoint the hot stack frame.
//
// Run with:
//   dotnet-trace collect --format Speedscope --output lstmvae.speedscope.json ^
//     -- dotnet run --project testconsole -c Release -- lstmvae-profile
internal static class LSTMVAEProfile
{
    public static void Run()
    {
        Console.WriteLine("=== LSTMVAE profile (test-shape: TrainLength=100, default opts) ===");
        var opts = new LSTMVAEOptions<double>();
        Console.WriteLine(
            $"WindowSize={opts.WindowSize} HiddenSize={opts.HiddenSize} LatentDim={opts.LatentDim} " +
            $"Epochs={opts.Epochs} BatchSize={opts.BatchSize} LearningRate={opts.LearningRate}");

        // Match TimeSeriesModelTestBase.TrainLength = 100, identical generator.
        var rng = new Random(42);
        const int trainLength = 100;
        var x = new Matrix<double>(trainLength, 1);
        var y = new Vector<double>(trainLength);
        for (int i = 0; i < trainLength; i++)
        {
            double t = i;
            x[i, 0] = t;
            y[i] = 0.5 * t + 3.0 * Math.Sin(2.0 * Math.PI * t / 20.0) + NextGaussian(rng) * 0.1;
        }

        var ctorSw = Stopwatch.StartNew();
        var model = new LSTMVAE<double>(opts);
        ctorSw.Stop();
        Console.WriteLine($"ctor          : {ctorSw.Elapsed.TotalSeconds,8:F3} s");

        var trainSw = Stopwatch.StartNew();
        try
        {
            model.Train(x, y);
        }
        catch (OutOfMemoryException ex)
        {
            trainSw.Stop();
            Console.WriteLine($"Train         : OOM after {trainSw.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        trainSw.Stop();
        Console.WriteLine($"Train         : {trainSw.Elapsed.TotalSeconds,8:F3} s  (CI timeout = 60s)");

        // LSTMVAE.Predict(Matrix) short-circuits to _trainingSeries
        // for any row index < trainN, so predicting on the same rows
        // we just trained on measures a memoized lookup — not the
        // encoder → reparam → decoder path we actually want to
        // profile. Route through PredictSingle, which always runs
        // the full inference.
        var predictSw = Stopwatch.StartNew();
        var pred = new Vector<double>(trainLength);
        for (int i = 0; i < trainLength; i++)
            pred[i] = model.PredictSingle(x.GetRow(i));
        predictSw.Stop();
        Console.WriteLine($"Predict       : {predictSw.Elapsed.TotalSeconds,8:F3} s  (output length={pred.Length})");

        Console.WriteLine($"TOTAL (incl ctor): {(ctorSw.Elapsed + trainSw.Elapsed + predictSw.Elapsed).TotalSeconds:F3} s");
    }

    // Box-Muller — same as ModelTestHelpers.NextGaussian so the input
    // distribution matches the failing test exactly.
    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
