using AiDotNet.Prototypes;

namespace AiDotNet;

/// <summary>
/// Simple test runner for Phase A GPU Acceleration Integration Tests.
/// </summary>
public class RunPhaseATests
{
    public static void Main(string[] args)
    {
        Console.WriteLine("Phase A GPU Acceleration - Integration Test Runner");
        Console.WriteLine("===================================================");
        Console.WriteLine();

        try
        {
            PrototypeIntegrationTests.RunAll();

            Console.WriteLine();
            Console.WriteLine("===================================================");
            Console.WriteLine("Phase A validation complete!");
            Console.WriteLine("===================================================");
        }
        catch (Exception ex)
        {
            Console.WriteLine();
            Console.WriteLine("ERROR: Test execution failed");
            Console.WriteLine($"Exception: {ex.GetType().Name}");
            Console.WriteLine($"Message: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            Environment.Exit(1);
        }
    }
}
