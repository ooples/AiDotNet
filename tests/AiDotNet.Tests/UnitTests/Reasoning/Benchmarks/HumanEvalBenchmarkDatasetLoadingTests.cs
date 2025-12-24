using AiDotNet.Reasoning.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Reasoning.Benchmarks;

public sealed class HumanEvalBenchmarkDatasetLoadingTests
{
    [Fact]
    public async Task LoadProblemsAsync_WhenDatasetEnvVarSet_LoadsFromFile()
    {
        var original = Environment.GetEnvironmentVariable("AIDOTNET_HUMANEVAL_DATASET");

        var tempFile = Path.Combine(Path.GetTempPath(), $"aidotnet-humaneval-{Guid.NewGuid():N}.jsonl");
        try
        {
            File.WriteAllText(
                tempFile,
                string.Join(
                    "\n",
                    new[]
                    {
                        "{\"task_id\":\"HumanEval/0\",\"prompt\":\"def add(a, b):\\n    \\\"\\\"\\\"\\\"\\\"\\\"\\n\",\"canonical_solution\":\"    return a + b\",\"test\":\"def check(candidate):\\n    assert candidate(1,2)==3\\n\",\"entry_point\":\"add\"}",
                        "{\"task_id\":\"HumanEval/1\",\"prompt\":\"def mul(a, b):\\n    \\\"\\\"\\\"\\\"\\\"\\\"\\n\",\"canonical_solution\":\"    return a * b\",\"test\":\"def check(candidate):\\n    assert candidate(2,3)==6\\n\",\"entry_point\":\"mul\"}"
                    }),
                System.Text.Encoding.UTF8);

            Environment.SetEnvironmentVariable("AIDOTNET_HUMANEVAL_DATASET", tempFile);

            var benchmark = new HumanEvalBenchmark<double>();
            var problems = await benchmark.LoadProblemsAsync();

            Assert.Equal(2, problems.Count);
            Assert.Contains(problems, p => p.Id == "HumanEval/0");
            Assert.Contains(problems, p => p.Id == "HumanEval/1");
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_HUMANEVAL_DATASET", original);
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }
}
