using System;
using System.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNetTestConsole;

// Standalone profiling harness: runs ChronosBolt once at research-paper defaults
// so dotnet-trace can collect CPU samples. Does NOT go through the smoke test
// harness — keeps Options at paper defaults, which is the entire point of
// profiling. Run with:
//   dotnet-trace collect --format Speedscope --output trace.speedscope.json ^
//     -- dotnet run --project testconsole -c Release -- chronosbolt-profile
internal static class ChronosBoltProfile
{
    public static void Run()
    {
        Console.WriteLine("=== ChronosBolt profile (paper defaults) ===");
        var opts = new ChronosBoltOptions<double>();
        Console.WriteLine(
            $"ContextLength={opts.ContextLength} ForecastHorizon={opts.ForecastHorizon} " +
            $"PatchLength={opts.PatchLength} EncoderHiddenDim={opts.EncoderHiddenDim} " +
            $"DecoderHiddenDim={opts.DecoderHiddenDim} NumEncoderLayers={opts.NumEncoderLayers} " +
            $"NumDecoderLayers={opts.NumDecoderLayers} NumHeads={opts.NumHeads} " +
            $"NumQuantiles={opts.NumQuantiles}");

        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: opts.ContextLength,
            outputSize: opts.ForecastHorizon);

        // Validate patch geometry before any size math (avoids div-by-zero and
        // silently mis-sized buffers when ContextLength is not an exact
        // multiple of PatchLength).
        if (opts.PatchLength <= 0)
            throw new InvalidOperationException(
                $"PatchLength must be positive; got {opts.PatchLength}.");
        if (opts.ContextLength % opts.PatchLength != 0)
            throw new InvalidOperationException(
                $"ContextLength ({opts.ContextLength}) must be an exact multiple of PatchLength ({opts.PatchLength}).");

        // Pre-compute the sizes the layer helper will emit so the OOM has context.
        int numPatches = opts.ContextLength / opts.PatchLength;
        long encInterHidden = (long)numPatches * opts.EncoderHiddenDim;  // input dim to encoder hidden blocks
        long encIntermediate = (long)numPatches * opts.EncoderHiddenDim * 4;
        long encWeightBytes = encInterHidden * encInterHidden * sizeof(double);
        long encFfnWeightBytes = encInterHidden * encIntermediate * sizeof(double);
        int encLayers = opts.NumEncoderLayers;
        Console.WriteLine(
            $"Derived layer dims:\n" +
            $"  numPatches                          = {numPatches}\n" +
            $"  numPatches * encoderHiddenDim       = {encInterHidden}  (feature width into each encoder layer)\n" +
            $"  numPatches * encoderHiddenDim * 4   = {encIntermediate}  (encoder FFN intermediate)\n" +
            $"  one [hidden, hidden] weight tensor  = {encWeightBytes / (1024.0 * 1024.0 * 1024.0),6:F2} GiB (double)\n" +
            $"  one [hidden, intermediate] weight   = {encFfnWeightBytes / (1024.0 * 1024.0 * 1024.0),6:F2} GiB (double)\n" +
            $"  {encLayers} encoder layers × ~5 weight tensors ≈ {((long)encLayers * 5L * encWeightBytes + (long)encLayers * encFfnWeightBytes * 2L) / (1024.0 * 1024.0 * 1024.0),6:F1} GiB just in encoder weights");

        var ctorSw = Stopwatch.StartNew();
        ChronosBolt<double>? model = null;
        try
        {
            model = new ChronosBolt<double>(arch, opts);
        }
        catch (OutOfMemoryException ex)
        {
            ctorSw.Stop();
            Console.WriteLine($"ctor          : OOM after {ctorSw.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        ctorSw.Stop();
        Console.WriteLine($"ctor          : {ctorSw.Elapsed.TotalSeconds,8:F3} s");

        // Paper-scale input: [batch=1, seq=contextLength, features=1].
        var rng = new Random(42);
        var inputData = new double[opts.ContextLength];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() * 2 - 1;
        var input = new Tensor<double>(new[] { 1, opts.ContextLength, 1 }, new Vector<double>(inputData));

        // Predict (2 passes — first includes cold start / lazy init). Each
        // phase has its own OOM guard so paper-scale runs report *which*
        // phase blew up rather than dying at the first unhandled throw.
        Tensor<double>? output = null;
        var predictWarmSw = Stopwatch.StartNew();
        try
        {
            output = model!.Predict(input);
        }
        catch (OutOfMemoryException ex)
        {
            predictWarmSw.Stop();
            Console.WriteLine($"Predict #1    : OOM after {predictWarmSw.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        predictWarmSw.Stop();
        Console.WriteLine($"Predict #1    : {predictWarmSw.Elapsed.TotalSeconds,8:F3} s  (cold)  (output shape=[{string.Join(",", output.Shape.ToArray())}])");

        var predictHotSw = Stopwatch.StartNew();
        try
        {
            _ = model.Predict(input);
        }
        catch (OutOfMemoryException ex)
        {
            predictHotSw.Stop();
            Console.WriteLine($"Predict #2    : OOM after {predictHotSw.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        predictHotSw.Stop();
        Console.WriteLine($"Predict #2    : {predictHotSw.Elapsed.TotalSeconds,8:F3} s  (hot)");

        // Train — 1 iteration, 3 iterations (for warmup/steady-state timing).
        var trainSw = Stopwatch.StartNew();
        try
        {
            model.Train(input, output);
        }
        catch (OutOfMemoryException ex)
        {
            trainSw.Stop();
            Console.WriteLine($"Train #1      : OOM after {trainSw.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        trainSw.Stop();
        Console.WriteLine($"Train #1      : {trainSw.Elapsed.TotalSeconds,8:F3} s  (cold)");

        var trainSw2 = Stopwatch.StartNew();
        try
        {
            model.Train(input, output);
        }
        catch (OutOfMemoryException ex)
        {
            trainSw2.Stop();
            Console.WriteLine($"Train #2      : OOM after {trainSw2.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        trainSw2.Stop();
        Console.WriteLine($"Train #2      : {trainSw2.Elapsed.TotalSeconds,8:F3} s  (hot)");

        var trainSw3 = Stopwatch.StartNew();
        try
        {
            model.Train(input, output);
        }
        catch (OutOfMemoryException ex)
        {
            trainSw3.Stop();
            Console.WriteLine($"Train #3      : OOM after {trainSw3.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        trainSw3.Stop();
        Console.WriteLine($"Train #3      : {trainSw3.Elapsed.TotalSeconds,8:F3} s  (hot)");

        Console.WriteLine($"TOTAL (incl ctor): {(ctorSw.Elapsed + predictWarmSw.Elapsed + predictHotSw.Elapsed + trainSw.Elapsed + trainSw2.Elapsed + trainSw3.Elapsed).TotalSeconds:F3} s");
    }
}
