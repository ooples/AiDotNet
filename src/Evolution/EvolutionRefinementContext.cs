namespace AiDotNet.Evolution;

/// <summary>Inputs available to an optional candidate refiner.</summary>
/// <remarks>
/// <para>
/// The engine builds one context per proposal before canonicalization and passes it to
/// <see cref="ICandidateRefiner{TGenome}.RefineAsync"/>. <see cref="Random"/> is a dedicated
/// <see cref="StableRandom"/> stream derived from the run seed and <see cref="EvaluationId"/>, distinct from the
/// streams used for variation and evaluation, so a refiner that draws only from it cannot perturb the rest of the
/// run and is reproducible on resume. The type is immutable and carries no archive reference: refinement is a
/// local, per-candidate step.
/// </para>
/// <para><b>For Beginners:</b> A refiner is an optional "polish" stage that runs after a new candidate is proposed
/// and before it is evaluated - for example, a few steps of local search on numeric parameters, or auto-formatting
/// generated code. This context tells the refiner which evaluation slot the candidate belongs to and gives it a
/// private random number stream to use for any stochastic decisions. Drawing from the supplied stream instead of a
/// freshly created <c>System.Random</c> is what keeps a checkpointed run replayable.</para>
/// </remarks>
public sealed class EvolutionRefinementContext
{
    /// <summary>Initializes a refinement context.</summary>
    /// <param name="evaluationId">The non-negative evaluation identifier assigned to the candidate.</param>
    /// <param name="random">The refiner-local stable random stream.</param>
    public EvolutionRefinementContext(long evaluationId, StableRandom random)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        EvaluationId = evaluationId;
        Random = random ?? throw new ArgumentNullException(nameof(random));
    }

    /// <summary>Gets the assigned evaluation ID.</summary>
    public long EvaluationId { get; }
    /// <summary>Gets a refiner-local stable random stream.</summary>
    public StableRandom Random { get; }
}
