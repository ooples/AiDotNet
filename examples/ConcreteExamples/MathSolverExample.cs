using AiDotNet.Interfaces;
using AiDotNet.Reasoning.DomainSpecific;
using AiDotNet.Reasoning.Verification;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Benchmarks.Data;

namespace AiDotNet.Examples.ConcreteExamples;

/// <summary>
/// Concrete example: Math problem solver with verification.
/// This demonstrates a real, working implementation.
/// </summary>
public class MathSolverExample
{
    public static async Task RunAsync(IChatModel chatModel)
    {
        Console.WriteLine("=== Math Solver Example ===\n");

        // Create the solver with all components
        var reasoner = new MathematicalReasoner<double>(chatModel);
        var verifier = new CalculatorVerifier<double>();
        var criticModel = new CriticModel<double>(chatModel);

        // Load real GSM8K problems
        var problems = GSM8KDataLoader.GetSampleProblems();

        int correctCount = 0;
        int verifiedCount = 0;

        foreach (var problem in problems.Take(5))
        {
            Console.WriteLine($"\nProblem: {problem.Question}");
            Console.WriteLine($"Expected Answer: {problem.FinalAnswer}");
            Console.WriteLine("\nSolving...");

            try
            {
                // Solve with verification
                var result = await reasoner.SolveAsync(
                    problem.Question,
                    useVerification: true,
                    useSelfConsistency: false
                );

                if (result.Success && result.Chain != null)
                {
                    Console.WriteLine("\nReasoning Steps:");
                    for (int i = 0; i < result.Chain.Steps.Count; i++)
                    {
                        var step = result.Chain.Steps[i];
                        Console.WriteLine($"  {i + 1}. {step.Content}");
                        Console.WriteLine($"     Confidence: {step.Score:F2}");
                    }

                    Console.WriteLine($"\nFinal Answer: {result.FinalAnswer}");
                    Console.WriteLine($"Overall Confidence: {result.ConfidenceScore:P0}");

                    // Verify the calculation
                    var verification = await verifier.VerifyAsync(result.Chain, problem.FinalAnswer);
                    Console.WriteLine($"Verification: {(verification.IsValid ? "✓ PASSED" : "✗ FAILED")}");

                    if (verification.IsValid)
                    {
                        verifiedCount++;
                    }

                    // Check if answer is correct
                    bool isCorrect = ExtractNumber(result.FinalAnswer) == ExtractNumber(problem.FinalAnswer);
                    Console.WriteLine($"Correctness: {(isCorrect ? "✓ CORRECT" : "✗ INCORRECT")}");

                    if (isCorrect)
                    {
                        correctCount++;
                    }

                    // Get detailed critique
                    if (result.Chain.Steps.Count > 0)
                    {
                        var context = new ReasoningContext
                        {
                            OriginalQuery = problem.Question,
                            Requirements = new List<string>
                            {
                                "Correct mathematical operations",
                                "Clear step-by-step reasoning",
                                "Proper use of arithmetic"
                            }
                        };

                        var critique = await criticModel.CritiqueStepAsync(
                            result.Chain.Steps[0],
                            context
                        );

                        Console.WriteLine($"\nCritic Score: {critique.OverallScore:F2}");
                        if (critique.Strengths.Count > 0)
                        {
                            Console.WriteLine($"Strength: {critique.Strengths[0]}");
                        }
                    }
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
        Console.WriteLine($"Problems Attempted: {problems.Take(5).Count()}");
        Console.WriteLine($"Correct Answers: {correctCount}/{problems.Take(5).Count()}");
        Console.WriteLine($"Verified: {verifiedCount}/{problems.Take(5).Count()}");
        Console.WriteLine($"Accuracy: {(double)correctCount / problems.Take(5).Count():P0}");
    }

    private static string ExtractNumber(string text)
    {
        var match = System.Text.RegularExpressions.Regex.Match(text, @"-?\d+\.?\d*");
        return match.Success ? match.Value : text;
    }
}
