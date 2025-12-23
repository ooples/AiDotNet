using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Reasoning.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Reasoning.Benchmarks;

public sealed class CodeXGlueBenchmarkTests
{
    [Fact]
    public async Task EvaluateAsync_ComputesExactMatchAndTokenF1()
    {
        var tempFile = Path.Combine(Path.GetTempPath(), $"codexglue-{Guid.NewGuid():N}.jsonl");

        try
        {
            File.WriteAllText(tempFile,
                "{\"id\":\"0\",\"category\":\"c1\",\"source\":\"alpha beta\",\"target\":\"alpha beta\"}" + Environment.NewLine +
                "{\"id\":\"1\",\"category\":\"c1\",\"source\":\"gamma\",\"target\":\"gamma\"}" + Environment.NewLine);

            var benchmark = new CodeXGlueBenchmark<double>(new CodeXGlueBenchmarkOptions
            {
                DatasetFilePath = tempFile,
                TaskName = "unit-test"
            });

            var result = await benchmark.EvaluateAsync(
                evaluateFunction: prompt => Task.FromResult(prompt == "gamma" ? "wrong" : "alpha beta"),
                sampleSize: null);

            Assert.Equal(2, result.TotalEvaluated);
            Assert.Equal(1, result.CorrectCount);
            Assert.Equal(0.5, Convert.ToDouble(result.Accuracy), 6);
            Assert.Equal(1.0, Convert.ToDouble(result.ProblemResults[0].Metadata["TokenF1"]), 6);
            Assert.Equal(0.0, Convert.ToDouble(result.ProblemResults[1].Metadata["TokenF1"]), 6);
            Assert.True(result.Metrics.TryGetValue("AverageTokenF1", out var f1Obj));
            Assert.Equal(0.5, Convert.ToDouble(f1Obj), 6);
            Assert.True(result.Metrics.TryGetValue("AverageBleu4", out var bleuObj));
            Assert.Equal(0.5, Convert.ToDouble(bleuObj), 6);
            Assert.True(result.Metrics.TryGetValue("AverageRougeL", out var rougeObj));
            Assert.Equal(0.5, Convert.ToDouble(rougeObj), 6);
            Assert.True(result.Metrics.TryGetValue("AverageIdentifierF1", out var idF1Obj));
            Assert.Equal(0.5, Convert.ToDouble(idF1Obj), 6);
            Assert.True(result.Metrics.TryGetValue("AverageCodeBleuLite", out var codeBleuObj));
            Assert.Equal(0.5, Convert.ToDouble(codeBleuObj), 6);
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }
}
