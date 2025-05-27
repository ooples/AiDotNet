using System;
using System.Threading.Tasks;

namespace AiDotNet.Compression.Tests;

/// <summary>
/// Entry point for running compression framework tests.
/// </summary>
public class RunTests
{
    /// <summary>
    /// Main entry point for running tests.
    /// </summary>
    /// <param name="args">Command line arguments.</param>
    public static async Task Main(string[] args)
    {
        Console.WriteLine("AiDotNet Model Compression Tests");
        Console.WriteLine("================================");
        
        if (args.Length == 0 || args[0].Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            // Run all tests
            await RunAllTests();
        }
        else
        {
            // Run specific tests based on arguments
            foreach (var arg in args)
            {
                await RunSpecificTest(arg);
            }
        }
        
        Console.WriteLine("\nAll tests completed.");
    }
    
    /// <summary>
    /// Runs all available tests.
    /// </summary>
    private static async Task RunAllTests()
    {
        // Run logging tests
        await CompressionLoggingTests.RunAllTests();
        
        // Add other test groups here as they are implemented
    }
    
    /// <summary>
    /// Runs a specific test based on the provided test name.
    /// </summary>
    /// <param name="testName">Name of the test to run.</param>
    private static async Task RunSpecificTest(string testName)
    {
        switch (testName.ToLowerInvariant())
        {
            case "logging":
                await CompressionLoggingTests.RunAllTests();
                break;
                
            case "compression-logger":
                await CompressionLoggingTests.TestCompressionLogger();
                break;
                
            case "benchmark-logger":
                CompressionLoggingTests.TestCompressionBenchmarkLogger();
                break;
                
            default:
                Console.WriteLine($"Unknown test: {testName}");
                Console.WriteLine("Available tests: logging, compression-logger, benchmark-logger");
                break;
        }
    }
}