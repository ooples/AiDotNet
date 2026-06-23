using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.WhisperFamily;

namespace AiDotNetTestConsole;

/// <summary>
/// Bottleneck profile for WhisperTimestamped (#1670). Reproduces the generated
/// invariant scaffold EXACTLY: production-default options (Whisper large-v3:
/// 1280-wide, 32+32 layers, vocab 51866) and the audio test InputShape [1,64,32].
/// Times ctor / first-forward / one Train step separately so the 120s xUnit
/// timeout can be attributed to a concrete phase, then run under dotnet-trace
/// for a CPU flame graph of the dominant phase. Generic over T so the double
/// (current test) and float (proposed float-by-default) paths can be compared.
/// </summary>
internal static class WhisperTimestampedProfile
{
    public static void Run() => RunProfile<double>("double");

    public static void RunFloat() => RunProfile<float>("float");

    private static void RunProfile<T>(string precision)
    {
        int bytes = precision == "float" ? 4 : 8;
        Console.WriteLine($"=== WhisperTimestamped profile [{precision}] (default large-v3 scale, test InputShape [1,64,32]) ===");

        var numOps = MathHelper.GetNumericOperations<T>();
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: 64, inputWidth: 32, inputDepth: 1, outputSize: 4);

        // WT_SCALE=small → a small-footprint config to test whether the double/float
        // gap is op-compute-bound (would persist at small scale) or memory-footprint-
        // bound (would collapse toward the per-op ~3-5x at small scale).
        WhisperTimestampedOptions? options = null;
        if (Environment.GetEnvironmentVariable("WT_SCALE") == "small")
        {
            options = new WhisperTimestampedOptions
            {
                EncoderDim = 256, DecoderDim = 256, NumEncoderLayers = 4,
                NumDecoderLayers = 4, NumAttentionHeads = 8, NumMels = 80,
                VocabSize = 1000, DropoutRate = 0.0,
            };
        }

        var ctorSw = Stopwatch.StartNew();
        var model = options is null ? new WhisperTimestamped<T>(architecture) : new WhisperTimestamped<T>(architecture, options);
        ctorSw.Stop();
        long paramCount = 0;
        try { paramCount = model.GetParameterCount(); }
        catch (Exception ex) { Console.WriteLine($"  (GetParameterCount failed: {ex.GetType().Name}: {ex.Message})"); }
        Console.WriteLine($"ctor          : {ctorSw.Elapsed.TotalSeconds,8:F3} s   params={paramCount:N0}  (~{paramCount * (double)bytes / 1e9:F2} GB {precision} weights)");

        var rng = new Random(42);
        var input = new Tensor<T>(new[] { 1, 64, 32 });
        for (int i = 0; i < input.Length; i++) input[i] = numOps.FromDouble(rng.NextDouble());

        int forwardLoops = Environment.GetEnvironmentVariable("WT_FWD_LOOPS") is { } s && int.TryParse(s, out var n) ? n : 1;
        Tensor<T> output = input;
        for (int it = 0; it < forwardLoops; it++)
        {
            var predictSw = Stopwatch.StartNew();
            output = model.Predict(input);
            predictSw.Stop();
            Console.WriteLine($"Predict[{it}]     : {predictSw.Elapsed.TotalSeconds,8:F3} s   outShape=[{string.Join(",", output.Shape)}]");
        }

        if (Environment.GetEnvironmentVariable("WT_RUN_TRAIN") == "1")
        {
            var target = new Tensor<T>(output.Shape.ToArray());
            for (int i = 0; i < target.Length; i++) target[i] = numOps.FromDouble(rng.NextDouble());
            try
            {
                var trainSw = Stopwatch.StartNew();
                model.Train(input, target);
                trainSw.Stop();
                Console.WriteLine($"one Train step: {trainSw.Elapsed.TotalSeconds,8:F3} s   (xUnit test timeout = 120s)");
            }
            catch (OutOfMemoryException) { Console.WriteLine("one Train step: OUT OF MEMORY (backward bug)"); }
        }
    }
}
