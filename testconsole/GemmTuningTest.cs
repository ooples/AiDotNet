// Extended GEMM tuning test to find optimal configurations
// Tests the feature-based ARD kernel and persistent database

using System;
using System.Diagnostics;
using System.IO;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNetTestConsole;

public static class GemmTuningTest
{
    /// <summary>
    /// Run the GEMM tuning test.
    /// </summary>
    /// <param name="trialsOverride">Number of trials (if null, prompts user)</param>
    /// <param name="enableDiagnostics">Enable verbose diagnostic output for debugging</param>
    public static void Run(int? trialsOverride = null, bool enableDiagnostics = false)
    {
        Console.WriteLine("=== Extended GEMM Tuning with Feature-Based ARD Kernel ===");
        Console.WriteLine();

        // Enable diagnostics if requested - write to log file to keep console clean
        string logFile = Path.Combine(Path.GetTempPath(), $"gemm_tuning_{DateTime.Now:yyyyMMdd_HHmmss}.log");
        if (enableDiagnostics)
        {
            OpenClBackend.EnableTuningDiagnostics = true;
            GemmAutoTuner.LogFilePath = logFile;
            Console.WriteLine($"[DIAGNOSTICS ENABLED] Verbose output will be written to: {logFile}");
            Console.WriteLine();
        }

        // Test matrix sizes (target shapes for performance comparison)
        var testSizes = new[]
        {
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        };

        int maxTrials;

        if (trialsOverride.HasValue)
        {
            maxTrials = trialsOverride.Value;
            Console.WriteLine($"Running with {maxTrials} trials (from command line)...");
        }
        else
        {
            Console.WriteLine("This test will run extensive Bayesian optimization to find");
            Console.WriteLine("the best GEMM kernel configurations for your GPU.");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("1. Quick test (50 trials per size) - ~5-10 min");
            Console.WriteLine("2. Standard test (100 trials per size) - ~15-30 min");
            Console.WriteLine("3. Extended test (200 trials per size) - ~30-60 min");
            Console.WriteLine("4. Exhaustive test (500 trials per size) - ~1-2 hours");
            Console.WriteLine("5. Custom number of trials");
            Console.WriteLine("0. Cancel");
            Console.WriteLine();
            Console.Write("Select option: ");

            if (!int.TryParse(Console.ReadLine(), out int option) || option == 0)
            {
                Console.WriteLine("Cancelled.");
                return;
            }

            maxTrials = option switch
            {
                1 => 50,
                2 => 100,
                3 => 200,
                4 => 500,
                5 => GetCustomTrials(),
                _ => 50
            };

            if (maxTrials <= 0)
            {
                Console.WriteLine("Invalid number of trials.");
                return;
            }
        }

        Console.WriteLine();
        Console.WriteLine($"Running with {maxTrials} trials per matrix size...");
        Console.WriteLine();

        try
        {
            using var backend = new OpenClBackend();

            Console.WriteLine($"GPU: {backend.DeviceName}");
            Console.WriteLine($"Vendor: {backend.DeviceVendor}");
            Console.WriteLine($"Compute Units: {backend.ComputeUnits}");
            Console.WriteLine($"Global Memory: {backend.GlobalMemoryBytes / (1024 * 1024)} MB");
            Console.WriteLine();

            // Check CLBlast availability for side-by-side comparison
            bool clBlastAvailable = OpenClBackend.IsClBlastAvailable;
            if (clBlastAvailable)
            {
                Console.WriteLine("CLBlast: AVAILABLE - will run side-by-side comparison");
            }
            else
            {
                Console.WriteLine("CLBlast: NOT AVAILABLE - comparison will be skipped");
                Console.WriteLine("         (Install CLBlast and ensure clblast.dll is in PATH)");
            }
            Console.WriteLine();

            var overallBest = new Dictionary<(int, int, int), (GemmConfig Config, double GFlops, string Method)>();
            var clBlastResults = new Dictionary<(int, int, int), double>();

            foreach (var (M, N, K) in testSizes)
            {
                Console.WriteLine(new string('=', 60));
                Console.WriteLine($"Testing {M}x{N}x{K}");
                Console.WriteLine(new string('=', 60));

                // Run BAYESIAN optimization first
                Console.WriteLine();
                Console.WriteLine("--- BAYESIAN OPTIMIZATION ---");
                var swBayes = Stopwatch.StartNew();

                var bayesResults = backend.RunBayesianGemmOptimization(
                    M, N, K,
                    maxTrials: maxTrials,
                    warmupRuns: 3,
                    benchmarkRuns: 5
                );

                swBayes.Stop();
                Console.WriteLine($"Bayesian completed in {swBayes.Elapsed.TotalMinutes:F1} minutes");

                double bayesBest = 0;
                if (bayesResults.Length > 0 && bayesResults[0].IsValid)
                {
                    bayesBest = bayesResults[0].GFlops;
                    Console.WriteLine($"Bayesian Best: {bayesBest:F2} GFLOPS - {bayesResults[0].Config}");
                }

                // Run EXHAUSTIVE optimization
                Console.WriteLine();
                Console.WriteLine("--- EXHAUSTIVE OPTIMIZATION (CLBlast-style) ---");
                var swExhaust = Stopwatch.StartNew();

                var exhaustResults = backend.RunExhaustiveGemmOptimization(
                    M, N, K,
                    warmupRuns: 3,
                    benchmarkRuns: 5
                );

                swExhaust.Stop();
                Console.WriteLine($"Exhaustive completed in {swExhaust.Elapsed.TotalMinutes:F1} minutes");

                double exhaustBest = 0;
                if (exhaustResults.Length > 0 && exhaustResults[0].IsValid)
                {
                    exhaustBest = exhaustResults[0].GFlops;
                    Console.WriteLine($"Exhaustive Best: {exhaustBest:F2} GFLOPS - {exhaustResults[0].Config}");
                }

                // Run CLBlast for comparison if available
                double clBlastGFlops = 0;
                if (clBlastAvailable)
                {
                    Console.WriteLine();
                    Console.WriteLine("--- CLBLAST REFERENCE BENCHMARK ---");

                    // Create test matrices
                    var matrixA = new float[M * K];
                    var matrixB = new float[K * N];
                    var matrixC = new float[M * N];

                    // Initialize with test data
                    var rand = new Random(42);
                    for (int i = 0; i < matrixA.Length; i++) matrixA[i] = (float)(rand.NextDouble() * 2 - 1);
                    for (int i = 0; i < matrixB.Length; i++) matrixB[i] = (float)(rand.NextDouble() * 2 - 1);

                    using var bufferA = backend.AllocateBuffer(matrixA);
                    using var bufferB = backend.AllocateBuffer(matrixB);
                    using var bufferC = backend.AllocateBuffer(matrixC);

                    // Warmup runs
                    for (int w = 0; w < 3; w++)
                    {
                        backend.GemmWithClBlast(bufferA, bufferB, bufferC, M, N, K);
                    }

                    // Benchmark runs
                    double totalMs = 0;
                    int benchRuns = 5;
                    for (int r = 0; r < benchRuns; r++)
                    {
                        double ms = backend.GemmWithClBlast(bufferA, bufferB, bufferC, M, N, K);
                        if (ms > 0)
                            totalMs += ms;
                    }

                    if (totalMs > 0)
                    {
                        double avgMs = totalMs / benchRuns;
                        double ops = 2.0 * M * N * K;
                        clBlastGFlops = (ops / (avgMs / 1000.0)) / 1e9;
                        clBlastResults[(M, N, K)] = clBlastGFlops;
                        Console.WriteLine($"CLBlast: {clBlastGFlops:F2} GFLOPS (avg {avgMs:F2} ms)");
                    }
                    else
                    {
                        Console.WriteLine("CLBlast: Failed to execute");
                    }
                }

                // Determine winner
                Console.WriteLine();
                Console.WriteLine("--- COMPARISON ---");
                double ourBest = Math.Max(bayesBest, exhaustBest);
                string ourMethod = bayesBest > exhaustBest ? "Bayesian" : "Exhaustive";

                Console.WriteLine($"Bayesian:   {bayesBest:F2} GFLOPS");
                Console.WriteLine($"Exhaustive: {exhaustBest:F2} GFLOPS");
                if (clBlastAvailable && clBlastGFlops > 0)
                {
                    Console.WriteLine($"CLBlast:    {clBlastGFlops:F2} GFLOPS (REFERENCE)");
                    double ratio = ourBest / clBlastGFlops * 100;
                    Console.WriteLine();
                    Console.WriteLine($"Our best ({ourMethod}): {ourBest:F2} GFLOPS = {ratio:F1}% of CLBlast");

                    if (ratio >= 95)
                        Console.WriteLine("STATUS: EXCELLENT - Matching CLBlast performance!");
                    else if (ratio >= 80)
                        Console.WriteLine("STATUS: GOOD - Close to CLBlast, room for improvement");
                    else if (ratio >= 60)
                        Console.WriteLine("STATUS: FAIR - Significant gap, optimization needed");
                    else
                        Console.WriteLine("STATUS: POOR - Major optimization work required");
                }
                else
                {
                    Console.WriteLine($"Our best: {ourMethod} ({ourBest:F2} GFLOPS)");
                }

                if (bayesBest > exhaustBest)
                {
                    overallBest[(M, N, K)] = (bayesResults[0].Config, bayesBest, "Bayesian");
                }
                else
                {
                    if (exhaustResults.Length > 0)
                        overallBest[(M, N, K)] = (exhaustResults[0].Config, exhaustBest, "Exhaustive");
                }

                // Show top 5 from each
                Console.WriteLine();
                Console.WriteLine("Top 5 Bayesian:");
                for (int i = 0; i < Math.Min(5, bayesResults.Length); i++)
                {
                    var r = bayesResults[i];
                    if (r.IsValid)
                        Console.WriteLine($"  {i + 1}. {r.GFlops:F2} GFLOPS - {r.Config}");
                }

                Console.WriteLine();
                Console.WriteLine("Top 5 Exhaustive:");
                for (int i = 0; i < Math.Min(5, exhaustResults.Length); i++)
                {
                    var r = exhaustResults[i];
                    if (r.IsValid)
                        Console.WriteLine($"  {i + 1}. {r.GFlops:F2} GFLOPS - {r.Config}");
                }

                Console.WriteLine();
            }

            // Summary
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("FINAL RESULTS SUMMARY");
            Console.WriteLine(new string('=', 60));
            Console.WriteLine();

            // Show CLBlast comparison if available
            if (clBlastResults.Count > 0)
            {
                Console.WriteLine("=== CLBlast vs Our Best Performance ===");
                Console.WriteLine();
                double totalRatio = 0;
                int count = 0;
                foreach (var ((M, N, K), clGflops) in clBlastResults)
                {
                    if (overallBest.TryGetValue((M, N, K), out var our))
                    {
                        double ratio = our.GFlops / clGflops * 100;
                        totalRatio += ratio;
                        count++;
                        Console.WriteLine($"{M}x{N}x{K}: Ours={our.GFlops:F2} vs CLBlast={clGflops:F2} ({ratio:F1}%)");
                    }
                }
                if (count > 0)
                {
                    double avgRatio = totalRatio / count;
                    Console.WriteLine();
                    Console.WriteLine($"AVERAGE: {avgRatio:F1}% of CLBlast performance");
                    if (avgRatio >= 95)
                        Console.WriteLine("OVERALL: EXCELLENT - Matching CLBlast!");
                    else if (avgRatio >= 80)
                        Console.WriteLine("OVERALL: GOOD - Close to CLBlast");
                    else if (avgRatio >= 60)
                        Console.WriteLine("OVERALL: FAIR - Optimization needed");
                    else
                        Console.WriteLine("OVERALL: POOR - Major work required");
                }
                Console.WriteLine();
            }

            Console.WriteLine("=== Best Configurations ===");
            Console.WriteLine();
            foreach (var ((M, N, K), (config, gflops, method)) in overallBest)
            {
                Console.WriteLine($"{M}x{N}x{K}:");
                Console.WriteLine($"  Best: {gflops:F2} GFLOPS (via {method})");
                Console.WriteLine($"  Config: {config}");
                Console.WriteLine();
            }

            // Check database persistence
            Console.WriteLine("Testing database persistence...");
            using (var db = new GemmTuningDatabase())
            {
                var allResults = db.GetAllResults();
                Console.WriteLine($"Database contains {allResults.Count} cached configurations");

                foreach (var (key, (config, gflops)) in allResults)
                {
                    Console.WriteLine($"  {key}: {gflops:F2} GFLOPS - {config}");
                }
            }

            Console.WriteLine();
            Console.WriteLine("Results have been saved to the persistent database.");
            Console.WriteLine("They will be reused on subsequent runs.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    private static int GetCustomTrials()
    {
        Console.Write("Enter number of trials (10-1000): ");
        if (int.TryParse(Console.ReadLine(), out int trials) && trials >= 10 && trials <= 1000)
        {
            return trials;
        }
        return 0;
    }
}
