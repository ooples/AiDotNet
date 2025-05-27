using AiDotNet.Compression.Logging;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace AiDotNet.Compression.Examples;

/// <summary>
/// Example demonstrating how to use the model compression logging system.
/// </summary>
public class CompressionLoggingExample
{
    /// <summary>
    /// Runs the compression logging example.
    /// </summary>
    public static async Task RunExample()
    {
        Console.WriteLine("Model Compression Logging Example");
        Console.WriteLine("================================");
        
        // 1. Initialize logging options
        var loggingOptions = new LoggingOptions
        {
            IsEnabled = true,
            MinimumLevel = LoggingLevel.Debug,
            LogDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Logs"),
            LogToConsole = true,
            IncludeMlContext = true
        };
        
        // 2. Create a compression logger for a quantization operation
        var quantizationLogger = new CompressionLogger(
            loggingOptions, 
            "NeuralNetworkClassifier", 
            CompressionTechnique.Quantization.ToString());
        
        // 3. Log the compression workflow
        quantizationLogger.LogCompressionStart(10_000_000, 4.0);
        
        // Simulate compression stages
        await SimulateCompressionStage(quantizationLogger, "Analysis", 0.0, 0.2);
        await SimulateCompressionStage(quantizationLogger, "Parameter Quantization", 0.2, 0.6);
        await SimulateCompressionStage(quantizationLogger, "Model Reconstruction", 0.6, 0.9);
        await SimulateCompressionStage(quantizationLogger, "Validation", 0.9, 1.0);
        
        // Log additional metrics
        quantizationLogger.LogCompressionMetric("QuantizationPrecision", 8);
        quantizationLogger.LogCompressionMetric("SymmetricQuantization", true);
        quantizationLogger.LogCompressionMetric("CustomScaleFactors", new List<double> { 0.1, 0.2, 0.15 });
        
        // Log completion with some metrics
        quantizationLogger.LogCompressionComplete(
            2_500_000,   // Compressed size (bytes)
            4.0,         // Compression ratio
            -0.015,      // Accuracy impact (-1.5%)
            1.8          // Speedup factor (1.8x faster)
        );
        
        Console.WriteLine("\nBenchmark Example");
        Console.WriteLine("================");
        
        // 4. Create a benchmark logger
        var benchmarkDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Benchmarks");
        var benchmarkLogger = new CompressionBenchmarkLogger(loggingOptions, benchmarkDir);
        
        // 5. Record benchmark results for different techniques
        RecordBenchmarkResults(benchmarkLogger);
        
        // 6. Generate a benchmark report
        var reportPath = benchmarkLogger.GenerateBenchmarkReport();
        Console.WriteLine($"Benchmark report generated at: {reportPath}");
        
        // 7. Save benchmark results to JSON
        var jsonPath = benchmarkLogger.SaveBenchmarkResults();
        Console.WriteLine($"Benchmark results saved to: {jsonPath}");
        
        Console.WriteLine("\nExample Complete!");
    }
    
    private static async Task SimulateCompressionStage(
        CompressionLogger logger, 
        string stageName, 
        double startProgress, 
        double endProgress)
    {
        var progressSteps = 5;
        var stepSize = (endProgress - startProgress) / progressSteps;
        
        for (int i = 0; i <= progressSteps; i++)
        {
            var progress = startProgress + (i * stepSize);
            logger.LogCompressionProgress(
                stageName, 
                progress, 
                $"{stageName} step {i}/{progressSteps}"
            );
            
            // Simulate work
            await Task.Delay(200);
        }
    }
    
    private static void RecordBenchmarkResults(CompressionBenchmarkLogger benchmarkLogger)
    {
        // Record quantization result
        benchmarkLogger.RecordBenchmarkResult(new CompressionBenchmarkResult
        {
            ModelName = "MobileNetV2",
            CompressionTechnique = "Quantization",
            OriginalSizeBytes = 14_000_000,
            CompressedSizeBytes = 3_500_000,
            CompressionRatio = 4.0,
            AccuracyImpact = -0.015, // -1.5%
            InferenceSpeedup = 1.8,
            MemoryReduction = 0.75,
            CompressionTimeMs = 12500,
            HardwareInfo = "CPU: 2.6 GHz Quad Core, RAM: 16GB",
            Notes = "Int8 quantization with per-channel scaling",
            AdditionalMetrics = new Dictionary<string, object>
            {
                ["QuantizationPrecision"] = 8,
                ["SymmetricQuantization"] = true
            }
        });
        
        // Record pruning result
        benchmarkLogger.RecordBenchmarkResult(new CompressionBenchmarkResult
        {
            ModelName = "MobileNetV2",
            CompressionTechnique = "Pruning",
            OriginalSizeBytes = 14_000_000,
            CompressedSizeBytes = 4_200_000,
            CompressionRatio = 3.33,
            AccuracyImpact = -0.008, // -0.8%
            InferenceSpeedup = 1.5,
            MemoryReduction = 0.7,
            CompressionTimeMs = 18200,
            HardwareInfo = "CPU: 2.6 GHz Quad Core, RAM: 16GB",
            Notes = "Magnitude-based pruning with 70% sparsity",
            AdditionalMetrics = new Dictionary<string, object>
            {
                ["Sparsity"] = 0.7,
                ["StructuredPruning"] = false
            }
        });
        
        // Record knowledge distillation result
        benchmarkLogger.RecordBenchmarkResult(new CompressionBenchmarkResult
        {
            ModelName = "MobileNetV2",
            CompressionTechnique = "KnowledgeDistillation",
            OriginalSizeBytes = 14_000_000,
            CompressedSizeBytes = 5_600_000,
            CompressionRatio = 2.5,
            AccuracyImpact = -0.005, // -0.5%
            InferenceSpeedup = 2.3,
            MemoryReduction = 0.6,
            CompressionTimeMs = 120000,
            HardwareInfo = "CPU: 2.6 GHz Quad Core, RAM: 16GB",
            Notes = "Student model with half the channels per layer, trained for 100 epochs",
            AdditionalMetrics = new Dictionary<string, object>
            {
                ["Temperature"] = 4.0,
                ["StudentSizeRatio"] = 0.4
            }
        });
        
        // Record results for a different model
        benchmarkLogger.RecordBenchmarkResult(new CompressionBenchmarkResult
        {
            ModelName = "ResNet50",
            CompressionTechnique = "Quantization",
            OriginalSizeBytes = 97_800_000,
            CompressedSizeBytes = 24_450_000,
            CompressionRatio = 4.0,
            AccuracyImpact = -0.022, // -2.2%
            InferenceSpeedup = 2.1,
            MemoryReduction = 0.75,
            CompressionTimeMs = 28400,
            HardwareInfo = "CPU: 2.6 GHz Quad Core, RAM: 16GB",
            Notes = "Int8 quantization with symmetric scaling",
            AdditionalMetrics = new Dictionary<string, object>
            {
                ["QuantizationPrecision"] = 8,
                ["SymmetricQuantization"] = true
            }
        });
    }
}