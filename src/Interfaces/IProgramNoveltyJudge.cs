using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;

namespace AiDotNet.Interfaces;

/// <summary>Decides whether a candidate program differs meaningfully from the incumbent it most resembles.</summary>
/// <remarks>
/// <para>
/// This is the last and most expensive rung of a novelty gate, reached only when the cheap structural check and the
/// embedding check have both failed to separate the two programs. An implementation typically asks a language
/// model, but nothing here requires one: a rules engine, a compiler-based comparison, or a human-in-the-loop queue
/// satisfies the contract equally well, and a test double satisfies it with no I/O at all.
/// </para>
/// <para>
/// An implementation must return <see cref="ProgramNoveltyVerdict.Unavailable"/> rather than throwing when it cannot
/// produce an answer, and must never let untrusted program text reach a log or an exception message unbounded and
/// unredacted. Choosing what an unavailable verdict means — admit the candidate or discard it — belongs to the
/// calling policy, not here.
/// </para>
/// <para><b>For Beginners:</b> Two programs can share almost every word and still do genuinely different things, or
/// look different and be the same idea renamed. When the cheap checks cannot tell, this is what gets asked. Because
/// asking costs a model call, a good gate reaches it rarely.</para>
/// </remarks>
public interface IProgramNoveltyJudge
{
    /// <summary>Gets a stable judge identifier.</summary>
    string Id { get; }

    /// <summary>Judges a candidate against the incumbent it most resembles.</summary>
    /// <param name="candidate">The proposed program.</param>
    /// <param name="incumbent">The existing program the candidate most resembles.</param>
    /// <param name="cancellationToken">A token that cancels the judgement.</param>
    /// <returns>The verdict, or <see cref="ProgramNoveltyVerdict.Unavailable"/> when no answer was obtained.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="candidate"/> or <paramref name="incumbent"/> is <c>null</c>.</exception>
    ValueTask<ProgramNoveltyVerdict> JudgeAsync(
        ProgramGenome candidate,
        ProgramGenome incumbent,
        CancellationToken cancellationToken = default);
}
