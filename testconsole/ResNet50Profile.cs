using System;
using System.Diagnostics;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNetTestConsole;

internal static class ResNet50Profile
{
    public static void Run()
    {
        Console.WriteLine("=== ResNet50 profile (paper-default: ResNet50 + 224x224x3 + 1000 classes) ===");

        var rng = new Random(42);
        var inputShape = new[] { 1, 3, 224, 224 };
        var outputShape = new[] { 1000 };

        var ctorSw = Stopwatch.StartNew();
        var network = new ResNetNetwork<double>();
        ctorSw.Stop();
        Console.WriteLine($"ctor    : {ctorSw.Elapsed.TotalSeconds,8:F3} s  (params: {network.ParameterCount})");

        var input = new Tensor<double>(inputShape);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
        var target = new Tensor<double>(outputShape);
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();

        // Warmup Predict (allocates buffers, primes JIT, etc.)
        var warmSw = Stopwatch.StartNew();
        var warm = network.Predict(input);
        warmSw.Stop();
        Console.WriteLine($"warm Predict: {warmSw.Elapsed.TotalSeconds,8:F3} s  (output len: {warm.Length})");

        // Predict baseline
        var predictSw = Stopwatch.StartNew();
        var pred = network.Predict(input);
        predictSw.Stop();
        Console.WriteLine($"hot Predict : {predictSw.Elapsed.TotalSeconds,8:F3} s");

        // Train iteration timings
        const int iters = 5;
        Console.WriteLine($"\n--- Train ({iters} iters at default Adam LR) ---");
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            network.Train(input, target);
            sw.Stop();
            Console.WriteLine($"  iter {i}: {sw.Elapsed.TotalSeconds,8:F3} s");
        }

        // Clone timing
        Console.WriteLine("\n--- Clone ---");
        var cloneSw = Stopwatch.StartNew();
        var cloned = network.Clone();
        cloneSw.Stop();
        Console.WriteLine($"Clone    : {cloneSw.Elapsed.TotalSeconds,8:F3} s");

        var clonePredSw = Stopwatch.StartNew();
        var cloneOut = cloned.Predict(input);
        clonePredSw.Stop();
        Console.WriteLine($"clonePred: {clonePredSw.Elapsed.TotalSeconds,8:F3} s");

        Console.WriteLine($"\nDone.");
    }
}
