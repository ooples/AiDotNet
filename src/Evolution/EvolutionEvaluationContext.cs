namespace AiDotNet.Evolution;

/// <summary>Deterministic evaluator context for one candidate.</summary>
/// <remarks>
/// <para>
/// The engine builds one context per evaluation attempt and passes it to
/// <c>IEvolutionTask&lt;TGenome&gt;.EvaluateAsync</c> alongside the candidate. <see cref="SeedStream"/> is derived
/// from the evaluation ID, so <see cref="CreateRandom"/> yields the same PCG sequence (O'Neill, "PCG: A Family of
/// Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation", 2014) regardless of
/// which worker thread runs the evaluation or how many other evaluations finished before it.
/// <see cref="AttemptCount"/> is one for the first attempt and increases each time the engine retries the same
/// candidate after a recoverable failure or timeout.
/// </para>
/// <para><b>For Beginners:</b> When the engine asks your task to score a candidate, this object is the cover sheet
/// that comes with it. It tells you which evaluation this is, gives you a reproducible random-number source, and
/// says how many times this particular candidate has been tried. Always draw randomness from
/// <see cref="CreateRandom"/> rather than <c>new Random()</c>: that is what makes two runs with the same seed
/// produce identical scores, and what lets a resumed checkpoint continue exactly where it stopped. For example, a
/// task that trains a neural network could use the stream to shuffle mini-batches and initialize weights, and could
/// use <see cref="AttemptCount"/> to switch to a smaller batch size on a second attempt after an out-of-memory
/// failure.</para>
/// </remarks>
public sealed class EvolutionEvaluationContext
{
    /// <summary>Initializes an evaluator context.</summary>
    /// <param name="evaluationId">The nonnegative stable evaluation identifier.</param>
    /// <param name="rootSeed">The run's root seed.</param>
    /// <param name="seedStream">The stable stream identifier assigned to this evaluation.</param>
    /// <param name="attemptCount">The one-based attempt number for this candidate.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="evaluationId"/> is negative or <paramref name="attemptCount"/> is not positive.
    /// </exception>
    public EvolutionEvaluationContext(long evaluationId, ulong rootSeed, ulong seedStream, int attemptCount)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        if (attemptCount <= 0) throw new ArgumentOutOfRangeException(nameof(attemptCount));
        EvaluationId = evaluationId;
        RootSeed = rootSeed;
        SeedStream = seedStream;
        AttemptCount = attemptCount;
    }

    /// <summary>Gets the evaluation ID.</summary>
    public long EvaluationId { get; }
    /// <summary>Gets the run root seed.</summary>
    public ulong RootSeed { get; }
    /// <summary>Gets the stable stream identifier.</summary>
    public ulong SeedStream { get; }
    /// <summary>Gets the one-based attempt count.</summary>
    public int AttemptCount { get; }

    /// <summary>Creates a fresh task-local stream whose sequence is independent of worker scheduling.</summary>
    /// <returns>A new <see cref="StableRandom"/> positioned at the start of the stream for this evaluation.</returns>
    public StableRandom CreateRandom() => StableRandom.CreateStream(RootSeed, SeedStream);
}
