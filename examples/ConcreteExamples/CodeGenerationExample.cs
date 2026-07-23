using AiDotNet.Interfaces;
using AiDotNet.Reasoning.DomainSpecific;
using AiDotNet.Reasoning.Verification;
using AiDotNet.Reasoning.Benchmarks.Data;

namespace AiDotNet.Examples.ConcreteExamples;

/// <summary>
/// Concrete example: Code generation with execution and testing.
/// </summary>
public class CodeGenerationExample
{
    public static async Task RunAsync(IChatModel chatModel)
    {
        Console.WriteLine("=== Code Generation Example ===\n");

        var reasoner = new CodeReasoner<double>(chatModel);
        var executor = new CodeExecutionVerifier<double>(timeoutMilliseconds: 5000);

        // Load sample HumanEval problems
        var problems = HumanEvalDataLoader.GetSampleProblems();

        int successCount = 0;
        int testsPassedCount = 0;

        foreach (var problem in problems.Take(3))
        {
            Console.WriteLine($"\nTask: {problem.TaskId}");
            Console.WriteLine($"Prompt:\n{problem.Prompt}");
            Console.WriteLine("\nGenerating code...");

            try
            {
                // Generate code
                var result = await reasoner.GenerateCodeAsync(
                    specification: problem.Prompt,
                    language: "python"
                );

                if (result.Success)
                {
                    successCount++;

                    Console.WriteLine("\nGenerated Code:");
                    Console.WriteLine(result.FinalAnswer);

                    // Extract test cases from the HumanEval test
                    var testCases = ExtractTestCases(problem.Test, problem.EntryPoint);

                    if (testCases.Length > 0)
                    {
                        Console.WriteLine($"\nRunning {testCases.Length} test(s)...");

                        // Combine generated code with tests
                        string codeToTest = result.FinalAnswer + "\n\n" + string.Join("\n", testCases);

                        var executionResult = await executor.VerifyCodeAsync(
                            codeToTest,
                            testCases,
                            "python"
                        );

                        Console.WriteLine($"\nTest Results:");
                        Console.WriteLine($"  Total Tests: {executionResult.TotalTests}");
                        Console.WriteLine($"  Passed: {executionResult.PassedTests}");
                        Console.WriteLine($"  Pass Rate: {executionResult.PassRate:P0}");
                        Console.WriteLine($"  All Passed: {(executionResult.AllTestsPassed ? "✓ YES" : "✗ NO")}");

                        if (executionResult.AllTestsPassed)
                        {
                            testsPassedCount++;
                        }

                        if (!executionResult.AllTestsPassed)
                        {
                            Console.WriteLine($"\nExecution Summary:");
                            Console.WriteLine(executionResult.GetSummary());
                        }
                    }
                    else
                    {
                        Console.WriteLine("\nNo test cases extracted for automatic verification.");
                    }

                    Console.WriteLine($"\nConfidence: {result.ConfidenceScore:P0}");
                }
                else
                {
                    Console.WriteLine($"Failed: {result.ErrorMessage}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }

            Console.WriteLine("\n" + new string('-', 80));
        }

        // Summary
        Console.WriteLine($"\n=== Results ===");
        Console.WriteLine($"Problems Attempted: {problems.Take(3).Count()}");
        Console.WriteLine($"Successfully Generated: {successCount}");
        Console.WriteLine($"All Tests Passed: {testsPassedCount}");
        Console.WriteLine($"Success Rate: {(double)successCount / problems.Take(3).Count():P0}");
    }

    private static string[] ExtractTestCases(string testCode, string entryPoint)
    {
        // Extract assert statements from test code
        var assertPattern = $@"assert\s+candidate\([^\)]*\)[^\n]*";
        var matches = System.Text.RegularExpressions.Regex.Matches(testCode, assertPattern);

        return matches
            .Select(m => m.Value.Replace("candidate", entryPoint))
            .ToArray();
    }
}
