using AiDotNet.Reasoning.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Reasoning.Benchmarks;

public sealed class HumanEvalBenchmarkExecutionHelpersTests
{
    [Fact]
    public void ExtractDoctestAssertions_WithExpectedOutputs_ProducesAssertStatements()
    {
        var prompt =
            "def has_close_elements(numbers, threshold):\n" +
            "    \"\"\"Example:\n" +
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n" +
            "    False\n" +
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n" +
            "    True\n" +
            "    \"\"\"\n";

        var assertions = HumanEvalBenchmark<double>.ExtractDoctestAssertions(prompt);

        Assert.Equal(2, assertions.Length);
        Assert.Contains("assert (has_close_elements([1.0, 2.0, 3.0], 0.5)) == (False)", assertions);
        Assert.Contains("assert (has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)) == (True)", assertions);
    }

    [Fact]
    public void ComposePythonSubmission_WhenCompletionIsBody_IndentsIntoPrompt()
    {
        var prompt =
            "def add(a, b):\n" +
            "    \"\"\"Add two numbers.\"\"\"\n";

        var submission = HumanEvalBenchmark<double>.ComposePythonSubmission(prompt, "return a + b");

        Assert.Contains("def add(a, b):", submission);
        Assert.Contains("    return a + b", submission);
    }

    [Fact]
    public void ComposePythonSubmission_WhenCompletionContainsDef_ReturnsCompletion()
    {
        var completion =
            "def add(a, b):\n" +
            "    return a + b\n";

        var submission = HumanEvalBenchmark<double>.ComposePythonSubmission("ignored", completion);

        Assert.Contains("def add(a, b):", submission);
        Assert.DoesNotContain("ignored", submission);
    }
}
