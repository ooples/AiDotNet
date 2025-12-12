using AiDotNet.Prototypes;

namespace PhaseATestRunner;

/// <summary>
/// Test runner for Phase A GPU Acceleration Integration Tests and Performance Benchmarks.
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Phase A GPU Acceleration - Test Runner");
        Console.WriteLine("=======================================");
        Console.WriteLine();
        Console.WriteLine("Select test suite to run:");
        Console.WriteLine("1. Integration Tests (validation)");
        Console.WriteLine("2. Performance Benchmarks (Vector operations)");
        Console.WriteLine("3. Run ALL tests and benchmarks");
        Console.WriteLine();
        Console.Write("Enter choice (1-3): ");

        var choice = Console.ReadLine();
        Console.WriteLine();
        Console.WriteLine();

        try
        {
            switch (choice)
            {
                case "1":
                    RunIntegrationTests();
                    break;
                case "2":
                    RunVectorBenchmarks();
                    break;
                case "3":
                    RunAll();
                    break;
                default:
                    Console.WriteLine("Invalid choice. Exiting.");
                    return;
            }

            Console.WriteLine();
            Console.WriteLine("===================================================");
            Console.WriteLine("Phase A testing COMPLETE!");
            Console.WriteLine("===================================================");
        }
        catch (Exception ex)
        {
            Console.WriteLine();
            Console.WriteLine("===================================================");
            Console.WriteLine("ERROR: Test execution failed");
            Console.WriteLine("===================================================");
            Console.WriteLine($"Exception: {ex.GetType().Name}");
            Console.WriteLine($"Message: {ex.Message}");
            Console.WriteLine();
            Console.WriteLine("Stack Trace:");
            Console.WriteLine(ex.StackTrace);
            Environment.Exit(1);
        }
    }

    private static void RunIntegrationTests()
    {
        Console.WriteLine("Running Integration Tests...");
        Console.WriteLine();
        PrototypeIntegrationTests.RunAll();
    }

    private static void RunVectorBenchmarks()
    {
        Console.WriteLine("Running Vector Performance Benchmarks...");
        Console.WriteLine();
        PerformanceBenchmark.RunComparison();
    }

    private static void RunAll()
    {
        Console.WriteLine("Running ALL tests and benchmarks...");
        Console.WriteLine();

        Console.WriteLine("PART 1: Integration Tests");
        Console.WriteLine("==========================");
        RunIntegrationTests();

        Console.WriteLine();
        Console.WriteLine();

        Console.WriteLine("PART 2: Vector Performance Benchmarks");
        Console.WriteLine("======================================");
        RunVectorBenchmarks();
    }
}
