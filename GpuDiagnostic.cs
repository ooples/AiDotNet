// Quick GPU diagnostic for AMD GPU
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

class GpuDiagnostic
{
    static void Main()
    {
        Console.WriteLine("=== AiDotNet GPU Diagnostic ===\n");

        // Check current engine
        var engine = AiDotNetEngine.Current;
        Console.WriteLine($"Current Engine: {engine.Name}");
        Console.WriteLine($"Engine Type: {engine.GetType().Name}");

        // Check DirectGpu availability
        Console.WriteLine($"\n--- DirectGpu Status ---");
        var directGpu = engine.DirectGpu;
        if (directGpu != null)
        {
            Console.WriteLine($"DirectGpu Available: {directGpu.IsAvailable}");
            Console.WriteLine($"DirectGpu Type: {directGpu.GetType().Name}");
        }
        else
        {
            Console.WriteLine("DirectGpu: Not configured");
        }

        // Try to detect available backends
        Console.WriteLine("\n--- Backend Detection ---");

        // Check HIP (enable diagnostics for DLL path resolution)
        try
        {
            HipBackend.EnableDiagnostics = true;
            var hipBackend = new HipBackend();
            Console.WriteLine($"HIP Backend Available: {hipBackend.IsAvailable}");
            if (hipBackend.IsAvailable)
            {
                Console.WriteLine($"  - HIP Backend: {hipBackend.BackendName}");
                Console.WriteLine($"  - Device: {hipBackend.DeviceName}");
                Console.WriteLine($"  - Compute Units: {hipBackend.ComputeUnits}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"HIP Backend Error: {ex.Message}");
        }

        // Check OpenCL
        try
        {
            var openclBackend = new OpenClBackend();
            Console.WriteLine($"OpenCL Backend Available: {openclBackend.IsAvailable}");
            if (openclBackend.IsAvailable)
            {
                Console.WriteLine($"  - OpenCL Backend initialized successfully");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"OpenCL Backend Error: {ex.Message}");
        }

        // Quick matrix multiplication test with warmup
        Console.WriteLine("\n--- GEMM Test with Warmup ---");
        try
        {
            // Test multiple sizes
            int[] sizes = { 256, 512, 1024 };

            foreach (var size in sizes)
            {
                Console.WriteLine($"\n  Size: {size}x{size}");

                var a = new float[size * size];
                var b = new float[size * size];
                for (int i = 0; i < a.Length; i++)
                {
                    a[i] = (i % 100) / 100.0f;
                    b[i] = ((i + 50) % 100) / 100.0f;
                }

                // CPU baseline
                var startCpu = DateTime.Now;
                var resultCpu = CpuGemm(a, b, size, size, size);
                var cpuTime = (DateTime.Now - startCpu).TotalMilliseconds;
                Console.WriteLine($"    CPU GEMM: {cpuTime:F2}ms");

                // GPU with warmup
                if (directGpu?.IsAvailable == true)
                {
                    // Warmup iteration (includes JIT)
                    var warmupResult = directGpu.MatMul(a, b, size, size, size);
                    GC.KeepAlive(warmupResult);

                    // Timed iterations
                    const int iterations = 5;
                    var startGpu = DateTime.Now;
                    for (int iter = 0; iter < iterations; iter++)
                    {
                        var resultGpu = directGpu.MatMul(a, b, size, size, size);
                        GC.KeepAlive(resultGpu);
                    }
                    var gpuTime = (DateTime.Now - startGpu).TotalMilliseconds / iterations;
                    Console.WriteLine($"    GPU MatMul (avg of {iterations}): {gpuTime:F2}ms");

                    if (cpuTime > 0 && gpuTime > 0)
                    {
                        var speedup = cpuTime / gpuTime;
                        Console.WriteLine($"    Speedup: {speedup:F1}x {(speedup > 1 ? "(GPU faster)" : "(CPU faster)")}");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"GEMM Test Error: {ex.Message}");
        }

        Console.WriteLine("\n=== Diagnostic Complete ===");
    }

    static float[] CpuGemm(float[] a, float[] b, int m, int k, int n)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int p = 0; p < k; p++)
                {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }
}
