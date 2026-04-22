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

        // Pre-compute the sizes the layer helper will emit so the OOM has context.
        int numPatches = opts.ContextLength / opts.PatchLength;
        long encInterHidden = (long)numPatches * opts.EncoderHiddenDim;  // input dim to encoder hidden blocks
        long encIntermediate = (long)numPatches * opts.EncoderHiddenDim * 4;
        long encWeightBytes = encInterHidden * encInterHidden * sizeof(double);
        long encFfnWeightBytes = encInterHidden * encIntermediate * sizeof(double);
        Console.WriteLine(
            $"Derived layer dims:\n" +
            $"  numPatches                          = {numPatches}\n" +
            $"  numPatches * encoderHiddenDim       = {encInterHidden}  (feature width into each encoder layer)\n" +
            $"  numPatches * encoderHiddenDim * 4   = {encIntermediate}  (encoder FFN intermediate)\n" +
            $"  one [hidden, hidden] weight tensor  = {encWeightBytes / (1024.0 * 1024.0 * 1024.0),6:F2} GiB (double)\n" +
            $"  one [hidden, intermediate] weight   = {encFfnWeightBytes / (1024.0 * 1024.0 * 1024.0),6:F2} GiB (double)\n" +
            $"  6 encoder layers × ~5 weight tensors ≈ {(6L * 5L * encWeightBytes + 6L * encFfnWeightBytes * 2L) / (1024.0 * 1024.0 * 1024.0),6:F1} GiB just in encoder weights");

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

        // Predict
        var predictSw = Stopwatch.StartNew();
        var output = model!.Predict(input);
        predictSw.Stop();
        Console.WriteLine($"Predict       : {predictSw.Elapsed.TotalSeconds,8:F3} s  (output shape=[{string.Join(",", output.Shape.ToArray())}])");

        // Train
        var trainSw = Stopwatch.StartNew();
        model.Train(input, output);
        trainSw.Stop();
        Console.WriteLine($"Train         : {trainSw.Elapsed.TotalSeconds,8:F3} s");

        Console.WriteLine($"TOTAL forward+backward: {(predictSw.Elapsed + trainSw.Elapsed).TotalSeconds:F3} s");
    }
}
