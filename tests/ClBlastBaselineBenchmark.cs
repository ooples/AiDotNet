// CLBlast Baseline Benchmark - establishes performance floor
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public static class ClBlastBaselineBenchmark
{
    public static void Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║     CLBlast Baseline Kernel Benchmark - Performance Floor    ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        // Enable diagnostics to see which kernel is selected
        DynamicGemmKernel.EnableDiagnostics = true;
        GemmAutoTuner.EnableDiagnostics = true;

        using var engine = new DirectGpuEngine();

        if (!engine.IsAvailable)
        {
            Console.WriteLine("ERROR: GPU not available");
            return;
        }

        Console.WriteLine($"GPU: {engine.DeviceName}");
        Console.WriteLine($"Vendor: {engine.DeviceVendor}");
        Console.WriteLine($"Compute Units: {engine.ComputeUnits}");
        Console.WriteLine($"Theoretical Peak: {engine.ComputeUnits * 64 * 2 * 1.408:F0} GFLOPS (estimate)");
        Console.WriteLine();

        // Test sizes
        int[] sizes = { 512, 1024, 2048, 4096 };
        int warmup = 3;
        int runs = 10;

        Console.WriteLine("Size       Time(ms)   GFLOPS    Efficiency");
        Console.WriteLine("─────────────────────────────────────────────");

        double theoreticalPeak = engine.ComputeUnits * 64 * 2 * 1.408; // ~5196 for RX 5500 XT

        foreach (var size in sizes)
        {
            var A = new float[size * size];
            var B = new float[size * size];
            var rand = new Random(42);
            for (int i = 0; i < A.Length; i++) A[i] = (float)(rand.NextDouble() - 0.5);
            for (int i = 0; i < B.Length; i++) B[i] = (float)(rand.NextDouble() - 0.5);

            // Warmup
            for (int i = 0; i < warmup; i++)
                engine.MatMul(A, B, size, size, size);

            // Benchmark
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < runs; i++)
                engine.MatMul(A, B, size, size, size);
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / runs;
            double flops = 2.0 * size * size * size;
            double gflops = flops / (avgMs * 1e6);
            double efficiency = gflops / theoreticalPeak * 100;

            Console.WriteLine($"{size}x{size}    {avgMs,8:F2}   {gflops,8:F1}   {efficiency,5:F1}%");
        }

        Console.WriteLine();
        Console.WriteLine("Benchmark complete. This is the CLBlast baseline floor.");
    }
}
