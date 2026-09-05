using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;

namespace AiDotNet.Interfaces;

/// <summary>Scores one candidate program and reports the result as an evolution task result.</summary>
/// <remarks>
/// <para>
/// This is the seam through which a program-evolution run learns whether a candidate is any good, and it is
/// deliberately the only place where generated code is executed. The core library never runs untrusted program
/// text in its own process: an implementation hands the source to a sandbox, a container, a separate process, or a
/// remote service of the caller's choosing, which is why no execution dependency is required to use the rest of
/// the substrate. Implementations must be deterministic for a given
/// <see cref="EvolutionEvaluationContext"/>, drawing any randomness from
/// <see cref="EvolutionEvaluationContext.CreateRandom"/>, and must report recoverable problems as
/// <see cref="EvolutionTaskResult.Failed"/> rather than by throwing so the engine can record the diagnostic and
/// apply its retry policy.
/// </para>
/// <para>
/// <see cref="Id"/> names the evaluator and <see cref="VersionHash"/> changes whenever its semantics or its test
/// data change; <c>ProgramEvolutionTask</c> forwards the latter as its evaluator version hash, so a checkpoint
/// written against one test suite refuses to resume against a different one.
/// </para>
/// <para><b>For Beginners:</b> Evolution needs a score for every candidate program, and this interface is where
/// you supply it. A typical implementation writes the program to a temporary file, runs it inside a sandbox with a
/// time limit, compares its output with expected outputs, and returns the fraction that passed. Because the
/// library never runs the code itself, you stay in control of how risky generated code is contained. Return a
/// failed result rather than throwing when a program crashes; a crashing candidate is normal in evolution and
/// should not stop the run.</para>
/// </remarks>
public interface IProgramFitnessEvaluator
{
    /// <summary>Gets a stable evaluator identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash that changes whenever evaluator semantics or test data change.</summary>
    string VersionHash { get; }

    /// <summary>Scores one candidate program.</summary>
    /// <param name="candidate">The program to score.</param>
    /// <param name="context">Deterministic per-evaluation context, including the seed stream and attempt count.</param>
    /// <param name="cancellationToken">A token that cancels the evaluation.</param>
    /// <returns>The terminal result, including quality and any descriptors the evaluator computed itself.</returns>
    ValueTask<EvolutionTaskResult> EvaluateAsync(
        ProgramGenome candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default);
}
