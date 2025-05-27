using AiDotNet.Compression.Logging;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.Compression.Tests;

/// <summary>
/// Tests for the model compression logging system.
/// </summary>
public class CompressionLoggingTests
{
    /// <summary>
    /// Tests the basic functionality of the compression logger.
    /// </summary>
    public static async Task TestCompressionLogger()
    {
        // Create test directory for logs
        var testLogDir = Path.Combine(Path.GetTempPath(), "AiDotNetTests", "Logs");
        Directory.CreateDirectory(testLogDir);
        
        try
        {
            // Create logging options for testing
            var loggingOptions = new LoggingOptions
            {
                IsEnabled = true,
                MinimumLevel = LoggingLevel.Trace,
                LogDirectory = testLogDir,
                LogToConsole = false, // Don't log to console during tests
                FileNameTemplate = "test-compression-log-{Date}.log"
            };
            
            // Create compression logger
            var logger = new CompressionLogger(
                loggingOptions, 
                "TestModel", 
                "TestCompression");
            
            // Log compression events
            logger.LogCompressionStart(1000000, 4.0);
            logger.LogCompressionProgress("Analysis", 0.25, "Analyzing model structure");
            logger.LogCompressionMetric("TestParameter", 42);
            await Task.Delay(100); // Simulate work
            logger.LogCompressionProgress("Transformation", 0.5, "Applying compression transforms");
            logger.LogCompressionMetric("AnotherMetric", "test value");
            await Task.Delay(100); // Simulate more work
            logger.LogCompressionProgress("Finalization", 0.75, "Finalizing compressed model");
            await Task.Delay(100); // Simulate final work
            logger.LogCompressionComplete(250000, 4.0, -0.01, 2.0);
            
            // Verify log file was created
            var logFiles = Directory.GetFiles(testLogDir, "*.log");
            Console.WriteLine($"Log files created: {logFiles.Length}");
            
            if (logFiles.Length > 0)
            {
                // Basic verification of log content
                var logContent = File.ReadAllText(logFiles[0]);
                var containsStartMessage = logContent.Contains("Starting TestCompression");
                var containsCompleteMessage = logContent.Contains("Completed TestCompression");
                var containsMetrics = logContent.Contains("TestParameter") && logContent.Contains("AnotherMetric");
                
                Console.WriteLine($"Log contains start message: {containsStartMessage}");
                Console.WriteLine($"Log contains complete message: {containsCompleteMessage}");
                Console.WriteLine($"Log contains metrics: {containsMetrics}");
                
                // Get metrics
                var metrics = logger.GetCompressionMetrics();
                Console.WriteLine($"Recorded metrics count: {metrics.Count}");
                
                // Verify that all expected metrics were recorded
                var hasOriginalSize = metrics.ContainsKey("OriginalSizeBytes") && (long)metrics["OriginalSizeBytes"] == 1000000;
                var hasCompressedSize = metrics.ContainsKey("CompressedSizeBytes") && (long)metrics["CompressedSizeBytes"] == 250000;
                var hasCompressionRatio = metrics.ContainsKey("CompressionRatio") && (double)metrics["CompressionRatio"] == 4.0;
                var hasTestParam = metrics.ContainsKey("TestParameter") && (int)metrics["TestParameter"] == 42;
                
                Console.WriteLine($"Has original size: {hasOriginalSize}");
                Console.WriteLine($"Has compressed size: {hasCompressedSize}");
                Console.WriteLine($"Has compression ratio: {hasCompressionRatio}");
                Console.WriteLine($"Has test parameter: {hasTestParam}");
                
                // Overall test result
                var testPassed = containsStartMessage && containsCompleteMessage && containsMetrics &&
                                hasOriginalSize && hasCompressedSize && hasCompressionRatio && hasTestParam;
                
                Console.WriteLine($"Compression logger test result: {(testPassed ? "PASSED" : "FAILED")}");
            }
            else
            {
                Console.WriteLine("FAILED: No log files were created.");
            }
            
            Console.WriteLine("Compression logger test completed.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Compression logger test failed with exception: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
    
    /// <summary>
    /// Tests the compression benchmark logger functionality.
    /// </summary>
    public static void TestCompressionBenchmarkLogger()
    {
        // Create test directory for benchmarks
        var testBenchmarkDir = Path.Combine(Path.GetTempPath(), "AiDotNetTests", "Benchmarks");
        Directory.CreateDirectory(testBenchmarkDir);
        
        try
        {
            // Create logging options for testing
            var loggingOptions = new LoggingOptions
            {
                IsEnabled = true,
                MinimumLevel = LoggingLevel.Information,
                LogDirectory = Path.Combine(testBenchmarkDir, "Logs"),
                LogToConsole = false // Don't log to console during tests
            };
            
            // Create benchmark logger
            var benchmarkLogger = new CompressionBenchmarkLogger(loggingOptions, testBenchmarkDir);
            
            // Create test benchmark results
            var benchmarkResults = new List<CompressionBenchmarkResult>
            {
                new CompressionBenchmarkResult
                {
                    ModelName = "TestModel1",
                    CompressionTechnique = "Quantization",
                    OriginalSizeBytes = 1000000,
                    CompressedSizeBytes = 250000,
                    CompressionRatio = 4.0,
                    AccuracyImpact = -0.01,
                    InferenceSpeedup = 2.0,
                    MemoryReduction = 0.75,
                    HardwareInfo = "Test Hardware"
                },
                new CompressionBenchmarkResult
                {
                    ModelName = "TestModel1",
                    CompressionTechnique = "Pruning",
                    OriginalSizeBytes = 1000000,
                    CompressedSizeBytes = 300000,
                    CompressionRatio = 3.33,
                    AccuracyImpact = -0.02,
                    InferenceSpeedup = 1.5,
                    MemoryReduction = 0.7,
                    HardwareInfo = "Test Hardware"
                },
                new CompressionBenchmarkResult
                {
                    ModelName = "TestModel2",
                    CompressionTechnique = "Quantization",
                    OriginalSizeBytes = 2000000,
                    CompressedSizeBytes = 500000,
                    CompressionRatio = 4.0,
                    AccuracyImpact = -0.015,
                    InferenceSpeedup = 1.8,
                    MemoryReduction = 0.75,
                    HardwareInfo = "Test Hardware"
                }
            };
            
            // Record benchmark results
            foreach (var result in benchmarkResults)
            {
                benchmarkLogger.RecordBenchmarkResult(result);
            }
            
            // Generate benchmark report
            var reportPath = benchmarkLogger.GenerateBenchmarkReport();
            Console.WriteLine($"Benchmark report generated: {reportPath}");
            
            // Verify report file was created
            var reportExists = File.Exists(reportPath);
            Console.WriteLine($"Report file exists: {reportExists}");
            
            if (reportExists)
            {
                // Basic verification of report content
                var reportContent = File.ReadAllText(reportPath);
                var containsModel1 = reportContent.Contains("TestModel1");
                var containsModel2 = reportContent.Contains("TestModel2");
                var containsQuantization = reportContent.Contains("Quantization");
                var containsPruning = reportContent.Contains("Pruning");
                
                Console.WriteLine($"Report contains Model1: {containsModel1}");
                Console.WriteLine($"Report contains Model2: {containsModel2}");
                Console.WriteLine($"Report contains Quantization: {containsQuantization}");
                Console.WriteLine($"Report contains Pruning: {containsPruning}");
                
                // Verify report has recommendations
                var containsRecommendations = reportContent.Contains("Recommendations");
                Console.WriteLine($"Report contains recommendations: {containsRecommendations}");
            }
            
            // Save benchmark results to JSON
            var jsonPath = benchmarkLogger.SaveBenchmarkResults();
            Console.WriteLine($"Benchmark results saved to JSON: {jsonPath}");
            
            // Verify JSON file was created
            var jsonExists = File.Exists(jsonPath);
            Console.WriteLine($"JSON file exists: {jsonExists}");
            
            if (jsonExists)
            {
                // Create a new logger and load the saved results
                var newLogger = new CompressionBenchmarkLogger(loggingOptions, testBenchmarkDir);
                var loadedCount = newLogger.LoadBenchmarkResults(jsonPath);
                
                Console.WriteLine($"Loaded {loadedCount} benchmark results");
                Console.WriteLine($"Expected count: {benchmarkResults.Count}");
                
                // Verify all results were loaded
                var loadingSuccessful = loadedCount == benchmarkResults.Count;
                Console.WriteLine($"Loading successful: {loadingSuccessful}");
                
                // Verify content of loaded results
                var allResults = newLogger.GetAllBenchmarkResults().ToList();
                var modelsMatch = allResults.Select(r => r.ModelName).OrderBy(n => n)
                                .SequenceEqual(benchmarkResults.Select(r => r.ModelName).OrderBy(n => n));
                
                Console.WriteLine($"Loaded models match: {modelsMatch}");
            }
            
            // Overall test result
            var testPassed = reportExists && jsonExists && File.Exists(reportPath) && File.Exists(jsonPath);
            
            Console.WriteLine($"Compression benchmark logger test result: {(testPassed ? "PASSED" : "FAILED")}");
            Console.WriteLine("Compression benchmark logger test completed.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Compression benchmark logger test failed with exception: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
    
    /// <summary>
    /// Runs all logging tests.
    /// </summary>
    public static async Task RunAllTests()
    {
        Console.WriteLine("=== Starting Compression Logging Tests ===");
        
        Console.WriteLine("\n1. Testing CompressionLogger:");
        await TestCompressionLogger();
        
        Console.WriteLine("\n2. Testing CompressionBenchmarkLogger:");
        TestCompressionBenchmarkLogger();
        
        Console.WriteLine("\n=== Compression Logging Tests Complete ===");
    }
}