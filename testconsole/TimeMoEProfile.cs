using System;
using System.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNetTestConsole;

// Standalone profiling harness: runs TimeMoE once at research-paper defaults
// (Shi et al. 2024 "Time-MoE: Billion-Scale Time Series Foundation Models with
// Mixture of Experts") so we can measure the pre-fix bottleneck and verify
// the post-fix speedup. Run with:
//   dotnet run --project testconsole -c Release -- timemoe-profile
internal static class TimeMoEProfile
{
    public static void Run()
    {
        Console.WriteLine("=== TimeMoE profile (paper defaults) ===");
        var opts = new TimeMoEOptions<double>();
        Console.WriteLine(
            $"ContextLength={opts.ContextLength} ForecastHorizon={opts.ForecastHorizon} " +
            $"PatchLength={opts.PatchLength} HiddenDimension={opts.HiddenDimension} " +
            $"IntermediateSize={opts.IntermediateSize} NumLayers={opts.NumLayers} " +
            $"NumExperts={opts.NumExperts} NumHeads={opts.NumHeads}");

        // Compute what the current helper will try to allocate so the OOM has context.
        int numPatches = opts.ContextLength / opts.PatchLength;
        long featureWidth = (long)numPatches * opts.HiddenDimension;
        long intermediateWidth = (long)numPatches * opts.IntermediateSize;
        long attnWeightBytes = featureWidth * featureWidth * sizeof(double);
        long expertFfnBytes = featureWidth * intermediateWidth * sizeof(double);
        long perBlockBytes = 4 * attnWeightBytes + 2L * opts.NumExperts * expertFfnBytes;
        Console.WriteLine(
            $"Derived layer dims (current helper, numPatches * hiddenDim bloat):\n" +
            $"  numPatches                          = {numPatches}\n" +
            $"  numPatches * hiddenDim              = {featureWidth}\n" +
            $"  numPatches * intermediateSize       = {intermediateWidth}\n" +
            $"  one [feat, feat] attn weight        = {attnWeightBytes / (1024.0 * 1024.0 * 1024.0),8:F2} GiB (double)\n" +
            $"  one expert [feat, inter] weight     = {expertFfnBytes / (1024.0 * 1024.0 * 1024.0),8:F2} GiB (double)\n" +
            $"  per MoE block (4 attn + 2*numExperts expert) = {perBlockBytes / (1024.0 * 1024.0 * 1024.0),8:F2} GiB\n" +
            $"  * NumLayers = {opts.NumLayers} blocks = {(double)perBlockBytes * opts.NumLayers / (1024.0 * 1024.0 * 1024.0),8:F1} GiB");

        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: opts.ContextLength,
            outputSize: opts.ForecastHorizon);

        var ctorSw = Stopwatch.StartNew();
        TimeMoE<double>? model = null;
        try
        {
            model = new TimeMoE<double>(arch, opts);
        }
        catch (OutOfMemoryException ex)
        {
            ctorSw.Stop();
            Console.WriteLine($"ctor          : OOM after {ctorSw.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        ctorSw.Stop();
        Console.WriteLine($"ctor          : {ctorSw.Elapsed.TotalSeconds,8:F3} s");

        var rng = new Random(42);
        var inputData = new double[opts.ContextLength];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() * 2 - 1;
        var input = new Tensor<double>(new[] { 1, opts.ContextLength, 1 }, new Vector<double>(inputData));

        var predictSw = Stopwatch.StartNew();
        var output = model!.Predict(input);
        predictSw.Stop();
        Console.WriteLine($"Predict       : {predictSw.Elapsed.TotalSeconds,8:F3} s  (output shape=[{string.Join(",", output.Shape.ToArray())}])");

        var trainSw = Stopwatch.StartNew();
        model.Train(input, output);
        trainSw.Stop();
        Console.WriteLine($"Train         : {trainSw.Elapsed.TotalSeconds,8:F3} s");

        Console.WriteLine($"TOTAL forward+backward: {(predictSw.Elapsed + trainSw.Elapsed).TotalSeconds:F3} s");
    }
}
