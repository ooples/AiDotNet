using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Reasoning.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.Reasoning.Benchmarks;

public class PredictionModelResultBenchmarkFacadeTests
{
    [Fact]
    public async Task EvaluateBenchmarkAsync_UsesPromptChain_WhenConfigured()
    {
        var benchmark = new GSM8KBenchmark<double>();
        var problems = await benchmark.LoadProblemsAsync(count: 3);

        var answersByProblem = problems.ToDictionary(p => p.Problem, p => p.CorrectAnswer);
        var chain = new ProblemAnswerLookupChain(answersByProblem);

        var options = new PredictionModelResultOptions<double, LinearAlgebra.Matrix<double>, LinearAlgebra.Vector<double>>
        {
            OptimizationResult = new OptimizationResult<double, LinearAlgebra.Matrix<double>, LinearAlgebra.Vector<double>>(),
            NormalizationInfo = new NormalizationInfo<double, LinearAlgebra.Matrix<double>, LinearAlgebra.Vector<double>>(),
            PromptChain = chain
        };

        var modelResult = new PredictionModelResult<double, LinearAlgebra.Matrix<double>, LinearAlgebra.Vector<double>>(options);

        var results = await modelResult.EvaluateBenchmarkAsync(benchmark, sampleSize: 3);

        Assert.Equal(3, results.TotalEvaluated);
        Assert.Equal(3, results.CorrectCount);
        Assert.Equal(1.0, Convert.ToDouble(results.Accuracy), 6);
    }
}

