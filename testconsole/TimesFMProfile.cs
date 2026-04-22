using System;
using System.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNetTestConsole;

// Standalone profiling harness: runs TimesFM once at research-paper defaults
// (Das et al. 2024 "A decoder-only foundation model for time-series
// forecasting") so we can measure the pre-fix bottleneck and verify the
// post-fix speedup. Run with:
//   dotnet run --project testconsole --framework net10.0 -c Release -- timesfm-profile
internal static class TimesFMProfile
{
    public static void Run()
    {
        Console.WriteLine("=== TimesFM profile (paper defaults) ===");
        var opts = new TimesFMOptions<double>();
        Console.WriteLine(
            $"ContextLength={opts.ContextLength} ForecastHorizon={opts.ForecastHorizon} " +
            $"PatchLength={opts.PatchLength} HiddenDimension={opts.HiddenDimension} " +
            $"NumLayers={opts.NumLayers} NumHeads={opts.NumHeads}");

        int numPatches = opts.ContextLength / opts.PatchLength;
        long featureWidth = (long)numPatches * opts.HiddenDimension;
        long attnWeightBytes = featureWidth * featureWidth * sizeof(double);
        long ffnWeightBytes = featureWidth * (featureWidth * 4L) * sizeof(double);
        long perBlockBytes = 4 * attnWeightBytes + 2 * ffnWeightBytes;
        Console.WriteLine(
            $"Derived layer dims (OLD helper, numPatches * hiddenDim bloat):\n" +
            $"  numPatches                          = {numPatches}\n" +
            $"  numPatches * hiddenDim              = {featureWidth}\n" +
            $"  one [feat, feat] attn weight        = {attnWeightBytes / (1024.0 * 1024.0),8:F2} MiB (double)\n" +
            $"  one FFN [feat, 4*feat] weight       = {ffnWeightBytes / (1024.0 * 1024.0),8:F2} MiB (double)\n" +
            $"  per block (4 attn + 2 FFN)          = {perBlockBytes / (1024.0 * 1024.0),8:F2} MiB\n" +
            $"  * NumLayers = {opts.NumLayers} blocks        = {(double)perBlockBytes * opts.NumLayers / (1024.0 * 1024.0 * 1024.0),8:F2} GiB");

        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: opts.ContextLength,
            outputSize: opts.ForecastHorizon);

        var ctorSw = Stopwatch.StartNew();
        TimesFM<double>? model = null;
        try
        {
            model = new TimesFM<double>(arch, opts);
        }
        catch (OutOfMemoryException ex)
        {
            ctorSw.Stop();
            Console.WriteLine($"ctor          : OOM after {ctorSw.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
            return;
        }
        catch (OverflowException ex)
        {
            ctorSw.Stop();
            Console.WriteLine($"ctor          : OverflowException after {ctorSw.Elapsed.TotalSeconds,8:F3} s  ({ex.Message})");
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
