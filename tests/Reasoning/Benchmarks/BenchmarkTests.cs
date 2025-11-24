using Xunit;
using AiDotNet.Reasoning.Benchmarks;

namespace AiDotNet.Tests.Reasoning.Benchmarks;

/// <summary>
/// Unit tests for benchmark implementations.
/// </summary>
public class BenchmarkTests
{
    [Fact]
    public async Task GSM8KBenchmark_LoadsProblems()
    {
        // Arrange
        var benchmark = new GSM8KBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 5);

        // Assert
        Assert.NotNull(problems);
        Assert.Equal(5, problems.Count);
        Assert.All(problems, p =>
        {
            Assert.False(string.IsNullOrEmpty(p.Problem));
            Assert.False(string.IsNullOrEmpty(p.CorrectAnswer));
        });
    }

    [Fact]
    public async Task MATHBenchmark_LoadsProblems()
    {
        // Arrange
        var benchmark = new MATHBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.Equal(3, problems.Count);
        Assert.All(problems, p => Assert.Contains("math", p.Category.ToLowerInvariant()));
    }

    [Fact]
    public async Task ARCAGIBenchmark_HasCorrectProblemCount()
    {
        // Arrange
        var benchmark = new ARCAGIBenchmark<double>();

        // Act
        var totalProblems = benchmark.TotalProblems;

        // Assert
        Assert.Equal(800, totalProblems);
    }

    [Fact]
    public async Task MMLUBenchmark_HasCorrectProblemCount()
    {
        // Arrange
        var benchmark = new MMLUBenchmark<double>();

        // Act
        var totalProblems = benchmark.TotalProblems;

        // Assert
        Assert.Equal(15908, totalProblems);
    }

    [Fact]
    public async Task HumanEvalBenchmark_LoadsCodeProblems()
    {
        // Arrange
        var benchmark = new HumanEvalBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.All(problems, p =>
        {
            Assert.Contains("def ", p.Problem.ToLowerInvariant());
            Assert.Equal("code_generation", p.Category);
        });
    }

    [Fact]
    public async Task MBPPBenchmark_LoadsPythonProblems()
    {
        // Arrange
        var benchmark = new MBPPBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.Equal(3, problems.Count);
    }

    [Fact]
    public async Task HellaSwagBenchmark_LoadsCommonsenseProblems()
    {
        // Arrange
        var benchmark = new HellaSwagBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.All(problems, p =>
        {
            Assert.Contains("A)", p.Problem);
            Assert.Contains("B)", p.Problem);
        });
    }

    [Fact]
    public async Task BoolQBenchmark_LoadsYesNoQuestions()
    {
        // Arrange
        var benchmark = new BoolQBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.All(problems, p =>
        {
            var answer = p.CorrectAnswer.ToLowerInvariant();
            Assert.True(answer.Contains("yes") || answer.Contains("no") ||
                       answer.Contains("true") || answer.Contains("false"));
        });
    }

    [Fact]
    public async Task PIQABenchmark_LoadsPhysicalCommonsenseProblems()
    {
        // Arrange
        var benchmark = new PIQABenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.All(problems, p => Assert.Contains("Solution", p.Problem));
    }

    [Fact]
    public async Task WinoGrandeBenchmark_LoadsPronounProblems()
    {
        // Arrange
        var benchmark = new WinoGrandeBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.All(problems, p =>
        {
            Assert.Contains("A)", p.Problem);
            Assert.Contains("B)", p.Problem);
        });
    }

    [Fact]
    public async Task TruthfulQABenchmark_LoadsTruthfulnessQuestions()
    {
        // Arrange
        var benchmark = new TruthfulQABenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.Equal(817, benchmark.TotalProblems);
    }

    [Fact]
    public async Task LogiQABenchmark_LoadsLogicalReasoningProblems()
    {
        // Arrange
        var benchmark = new LogiQABenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.Equal(8678, benchmark.TotalProblems);
    }

    [Fact]
    public async Task DROPBenchmark_LoadsDiscreteReasoningProblems()
    {
        // Arrange
        var benchmark = new DROPBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.Equal(96000, benchmark.TotalProblems);
    }

    [Fact]
    public async Task CommonsenseQABenchmark_LoadsCommonsenseQuestions()
    {
        // Arrange
        var benchmark = new CommonsenseQABenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        // Assert
        Assert.NotNull(problems);
        Assert.All(problems, p =>
        {
            Assert.Contains("A)", p.Problem);
            Assert.Contains("E)", p.Problem); // CommonsenseQA has 5 options
        });
    }

    [Fact]
    public async Task BenchmarkEvaluate_WithMockFunction_ReturnsResult()
    {
        // Arrange
        var benchmark = new GSM8KBenchmark<double>();

        async Task<string> MockEvaluate(string problem)
        {
            await Task.Delay(1);
            return "42";
        }

        // Act
        var result = await benchmark.EvaluateAsync(MockEvaluate, sampleSize: 2);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.TotalEvaluated);
        Assert.True(result.CorrectCount >= 0);
        Assert.NotNull(result.ConfidenceScores);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public async Task BenchmarkLoadProblems_WithDifferentCounts_ReturnsCorrectAmount(int count)
    {
        // Arrange
        var benchmark = new GSM8KBenchmark<double>();

        // Act
        var problems = await benchmark.LoadProblemsAsync(count);

        // Assert
        Assert.Equal(count, problems.Count);
    }

    [Fact]
    public void AllBenchmarks_HaveValidNames()
    {
        // Arrange & Act & Assert
        Assert.Equal("GSM8K", new GSM8KBenchmark<double>().BenchmarkName);
        Assert.Equal("HumanEval", new HumanEvalBenchmark<double>().BenchmarkName);
        Assert.Equal("MATH", new MATHBenchmark<double>().BenchmarkName);
        Assert.Equal("ARC-AGI", new ARCAGIBenchmark<double>().BenchmarkName);
        Assert.Equal("MMLU", new MMLUBenchmark<double>().BenchmarkName);
        Assert.Equal("MBPP", new MBPPBenchmark<double>().BenchmarkName);
        Assert.Equal("HellaSwag", new HellaSwagBenchmark<double>().BenchmarkName);
        Assert.Equal("BoolQ", new BoolQBenchmark<double>().BenchmarkName);
        Assert.Equal("PIQA", new PIQABenchmark<double>().BenchmarkName);
        Assert.Equal("WinoGrande", new WinoGrandeBenchmark<double>().BenchmarkName);
        Assert.Equal("TruthfulQA", new TruthfulQABenchmark<double>().BenchmarkName);
        Assert.Equal("LogiQA", new LogiQABenchmark<double>().BenchmarkName);
        Assert.Equal("DROP", new DROPBenchmark<double>().BenchmarkName);
        Assert.Equal("CommonsenseQA", new CommonsenseQABenchmark<double>().BenchmarkName);
    }

    [Fact]
    public void AllBenchmarks_HaveDescriptions()
    {
        // Arrange
        var benchmarks = new IBenchmark<double>[]
        {
            new GSM8KBenchmark<double>(),
            new HumanEvalBenchmark<double>(),
            new MATHBenchmark<double>(),
            new ARCAGIBenchmark<double>(),
            new MMLUBenchmark<double>()
        };

        // Assert
        Assert.All(benchmarks, b =>
        {
            Assert.False(string.IsNullOrEmpty(b.Description));
            Assert.True(b.Description.Length > 20);
        });
    }
}
