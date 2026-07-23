// Comprehensive GEMM A/B Testing to maximize performance toward theoretical limit
// Tests all kernel variants across all sizes to find optimal configuration

using System;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNetTestConsole;

public static class ComprehensiveGemmAbTest
{
    public static void Run()
    {
        Console.WriteLine("=== Comprehensive GEMM A/B Testing ===");
        Console.WriteLine();
        Console.WriteLine("This test compares all kernel variants across all matrix sizes");
        Console.WriteLine("to identify the optimal configuration for maximum performance.");
        Console.WriteLine();

        try
        {
            using var backend = new OpenClBackend();

            Console.WriteLine($"GPU: {backend.DeviceName}");
            Console.WriteLine($"Vendor: {backend.DeviceVendor}");
            Console.WriteLine($"Compute Units: {backend.ComputeUnits}");
            Console.WriteLine($"Global Memory: {backend.GlobalMemoryBytes / (1024 * 1024)} MB");
            Console.WriteLine();

            // Run with sizes that span the threshold boundaries
            // This helps identify optimal MinSwizzleSize and MinDynamicSize
            var sizes = new[]
            {
                64,    // Very small - should use simple kernel
                128,   // Small - boundary testing
                192,   // Small-medium
                256,   // MinDynamicSize boundary
                320,   // Between thresholds
                384,   // Between thresholds
                448,   // Just below MinSwizzleSize
                512,   // MinSwizzleSize boundary
                640,   // Just above MinSwizzleSize
                768,   // Medium
                1024,  // Standard benchmark size
                1536,  // Large
                2048,  // Large benchmark size
                3072,  // Very large
                4096   // Maximum common size
            };

            Console.WriteLine("Running comprehensive A/B test...");
            Console.WriteLine("This will take several minutes.");
            Console.WriteLine();

            // Enable diagnostics to see kernel selection
            DynamicGemmKernel.EnableDiagnostics = true;

            var result = backend.ComprehensiveAbTest(sizes, warmupRuns: 3, benchmarkRuns: 10);

            // Disable diagnostics after test
            DynamicGemmKernel.EnableDiagnostics = false;

            Console.WriteLine(result);

            // Print recommendations
            Console.WriteLine();
            Console.WriteLine("=== RECOMMENDATIONS ===");
            Console.WriteLine();
            Console.WriteLine("Based on the results above:");
            Console.WriteLine("1. Check which variant wins at each size");
            Console.WriteLine("2. Look for the crossover point where XOR Swizzle becomes better");
            Console.WriteLine("3. Verify correctness (all should show 'OK')");
            Console.WriteLine("4. Note efficiency percentages vs theoretical peak");
            Console.WriteLine();
            Console.WriteLine("Current thresholds:");
            Console.WriteLine($"  MinDynamicSize: {DynamicGemmKernel.MinDynamicSize}");
            Console.WriteLine($"  MinSwizzleSize: {DynamicGemmKernel.MinSwizzleSize}");
            Console.WriteLine();
            Console.WriteLine("To tune thresholds, set environment variables:");
            Console.WriteLine("  AIDOTNET_GEMM_MIN_DYNAMIC_SIZE=<value>");
            Console.WriteLine("  AIDOTNET_GEMM_MIN_SWIZZLE_SIZE=<value>");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
