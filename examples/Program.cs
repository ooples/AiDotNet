using AiDotNet.Examples.ConcreteExamples;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Examples;

/// <summary>
/// Main program to run concrete examples.
/// Demonstrates the reasoning framework with real implementations.
/// </summary>
public class Program
{
    public static async Task Main(string[] args)
    {
        Console.WriteLine(@"
╔══════════════════════════════════════════════════════════════════╗
║       AiDotNet Reasoning Framework - Concrete Examples           ║
║                                                                  ║
║  A production-ready reasoning system rivaling DeepSeek-R1       ║
║  and ChatGPT o1/o3 capabilities                                  ║
╚══════════════════════════════════════════════════════════════════╝
");

        // Note: Replace with your actual chat model implementation
        IChatModel chatModel = GetChatModelImplementation();

        if (chatModel == null)
        {
            Console.WriteLine("ERROR: No chat model implementation provided.");
            Console.WriteLine("Please implement IChatModel and update GetChatModelImplementation()");
            return;
        }

        Console.WriteLine("Select an example to run:");
        Console.WriteLine("1. Math Solver (GSM8K problems with verification)");
        Console.WriteLine("2. Code Generation (HumanEval with execution)");
        Console.WriteLine("3. Benchmark Evaluation (Multiple benchmarks)");
        Console.WriteLine("4. RL Training (Standard training)");
        Console.WriteLine("5. STaR Training (Self-Taught Reasoner)");
        Console.WriteLine("6. Run All Examples");
        Console.WriteLine("0. Exit");
        Console.Write("\nYour choice: ");

        var choice = Console.ReadLine();

        try
        {
            switch (choice)
            {
                case "1":
                    await MathSolverExample.RunAsync(chatModel);
                    break;

                case "2":
                    await CodeGenerationExample.RunAsync(chatModel);
                    break;

                case "3":
                    await BenchmarkRunnerExample.RunAsync(chatModel);
                    break;

                case "4":
                    await TrainingExample.RunAsync(chatModel);
                    break;

                case "5":
                    await TrainingExample.RunSTaRTrainingAsync(chatModel);
                    break;

                case "6":
                    await RunAllExamplesAsync(chatModel);
                    break;

                case "0":
                    Console.WriteLine("Goodbye!");
                    return;

                default:
                    Console.WriteLine("Invalid choice. Please try again.");
                    break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ Error running example: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }

        Console.WriteLine("\n\nPress any key to exit...");
        Console.ReadKey();
    }

    private static async Task RunAllExamplesAsync(IChatModel chatModel)
    {
        Console.WriteLine("\n" + new string('=', 80));
        Console.WriteLine("Running all examples...");
        Console.WriteLine(new string('=', 80) + "\n");

        try
        {
            // Example 1: Math Solver
            Console.WriteLine("\n\n### EXAMPLE 1: MATH SOLVER ###");
            await MathSolverExample.RunAsync(chatModel);
            await Task.Delay(2000);

            // Example 2: Code Generation
            Console.WriteLine("\n\n### EXAMPLE 2: CODE GENERATION ###");
            await CodeGenerationExample.RunAsync(chatModel);
            await Task.Delay(2000);

            // Example 3: Benchmark
            Console.WriteLine("\n\n### EXAMPLE 3: BENCHMARK EVALUATION ###");
            await BenchmarkRunnerExample.RunAsync(chatModel);

            Console.WriteLine("\n\n" + new string('=', 80));
            Console.WriteLine("All examples completed!");
            Console.WriteLine(new string('=', 80));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in example execution: {ex.Message}");
        }
    }

    /// <summary>
    /// Get your chat model implementation.
    /// Replace this with your actual implementation (OpenAI, Anthropic, etc.)
    /// </summary>
    private static IChatModel? GetChatModelImplementation()
    {
        // Option 1: Return your OpenAI implementation
        // return new OpenAIChatModel("your-api-key");

        // Option 2: Return your Anthropic implementation
        // return new AnthropicChatModel("your-api-key");

        // Option 3: Use mock for testing (not recommended for real use)
        // return new MockChatModel();

        // For now, return null to indicate no implementation
        Console.WriteLine("\n⚠️  WARNING: Using mock chat model for demonstration.");
        Console.WriteLine("Replace GetChatModelImplementation() with your actual chat model.\n");

        return new MockChatModelForDemo();
    }
}

/// <summary>
/// Mock chat model for demonstration purposes.
/// Replace with your actual implementation!
/// </summary>
internal class MockChatModelForDemo : IChatModel
{
    private readonly Random _random = RandomHelper.CreateSeededRandom(42);

    public async Task<string> GenerateResponseAsync(
        string prompt,
        CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken); // Simulate API call

        // Detect problem type and return mock response
        if (prompt.Contains("step") || prompt.Contains("calculate", StringComparison.OrdinalIgnoreCase))
        {
            return GenerateMathResponse(prompt);
        }

        if (prompt.Contains("def ") || prompt.Contains("function"))
        {
            return GenerateCodeResponse(prompt);
        }

        return "This is a mock response. Please implement IChatModel with your actual chat model provider.";
    }

    private string GenerateMathResponse(string prompt)
    {
        // Try to extract numbers and generate a plausible math solution
        var numbers = System.Text.RegularExpressions.Regex.Matches(prompt, @"\d+")
            .Select(m => int.Parse(m.Value))
            .ToList();

        if (numbers.Count >= 2)
        {
            int result = numbers[0] + numbers[1]; // Simple addition
            return $@"Step 1: Identify the numbers
We have {numbers[0]} and {numbers[1]}.

Step 2: Perform the calculation
{numbers[0]} + {numbers[1]} = {result}

Final Answer: {result}";
        }

        return @"Step 1: Analyze the problem
Step 2: Calculate the result
Final Answer: 42";
    }

    private string GenerateCodeResponse(string prompt)
    {
        return @"```python
def solution(n):
    """"""
    A simple implementation.
    """"""
    return n * 2

# Test
assert solution(5) == 10
```";
    }
}
