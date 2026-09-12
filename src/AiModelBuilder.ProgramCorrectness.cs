using AiDotNet.Configuration;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    private bool _requireProgramTestCaseCorrectness;

    /// <summary>Requires every configured public input/output test to pass for each program candidate.</summary>
    /// <returns>This concrete builder; call before interface-returning Configure methods when chaining.</returns>
    /// <remarks>
    /// Built-in pass-fraction fitness becomes a hard gate without duplicate execution. Script fitness runs
    /// only after public tests pass. Custom providers instead configure a separate checker with ConfigureProgramCorrectness;
    /// the existing options contract forbids combining custom fitness with input/output examples.
    /// Costs include both stages and must share the caller's cost-unit convention.
    /// This does not add held-out tests, guarantee provider honesty, or bypass sandbox and ledger restrictions.
    /// </remarks>
    public AiModelBuilder<T, TInput, TOutput> WithProgramTestCaseCorrectness()
    {
        _requireProgramTestCaseCorrectness = true;
        return this;
    }

    private IProgramFitnessEvaluator ApplyRequiredProgramTestCaseCorrectness(ProgramEvolutionOptions programs,
        IProgramFitnessEvaluator evaluator, ref ProcessProgramExecutionEngine? owned)
    {
        if (!_requireProgramTestCaseCorrectness) return evaluator;
        if (programs.TestCases.Count == 0) throw new InvalidOperationException("Hard public-test correctness requires input/output examples.");
        if (evaluator is SandboxedProgramFitnessEvaluator) return new AllPassProgramFitnessEvaluator(evaluator);
        var runner = ResolveProgramRunner(programs, ref owned);
        return new CorrectnessGatedProgramFitnessEvaluator(new SandboxedProgramFitnessEvaluator(runner, programs.TestCases), evaluator);
    }

    private IProgramExecutionEngine ResolveProgramRunner(ProgramEvolutionOptions programOptions, ref ProcessProgramExecutionEngine? owned)
    {
        if (_programExecutionEngine is not null) return _programExecutionEngine;
        if (owned is not null) return owned;
        if (_programSandboxOptions is not null && programOptions.HasExplicitSandbox)
            throw new ArgumentException("The sandbox is configured twice, through ConfigureProgramSandbox and through " +
                "ProgramEvolutionOptions.Sandbox. Configure it in one place.", nameof(programOptions));
        owned = new ProcessProgramExecutionEngine(_programSandboxOptions ?? programOptions.Sandbox);
        return owned;
    }
}
