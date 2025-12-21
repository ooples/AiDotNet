using AiDotNet.PromptEngineering.Chains;

namespace AiDotNet.Tests.Reasoning.Benchmarks;

internal sealed class ProblemAnswerLookupChain : ChainBase<string, string>
{
    private readonly IReadOnlyDictionary<string, string> _answersByProblem;

    public ProblemAnswerLookupChain(IReadOnlyDictionary<string, string> answersByProblem)
        : base(name: "ProblemAnswerLookup", description: "Test-only chain that returns precomputed answers for benchmark problems.")
    {
        _answersByProblem = answersByProblem ?? throw new ArgumentNullException(nameof(answersByProblem));
    }

    protected override string RunCore(string input)
    {
        return _answersByProblem.TryGetValue(input, out var answer) ? answer : string.Empty;
    }

    protected override Task<string> RunCoreAsync(string input, CancellationToken cancellationToken)
    {
        return Task.FromResult(RunCore(input));
    }
}

