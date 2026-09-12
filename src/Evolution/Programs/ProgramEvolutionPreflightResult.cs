namespace AiDotNet.Evolution.Programs;

/// <summary>A bounded seed check, separate from a search run and its evaluation budget.</summary>
/// <remarks>
/// Costs are evaluator-reported units, not money. The correctness and additional fitness receipts are kept
/// separate because arbitrary providers may use different units. This is not held-out validation or a guarantee
/// of future model availability. No program source, provider error text, environment values or credentials are retained.
/// </remarks>
public sealed class ProgramEvolutionPreflightResult
{
    /// <summary>Gets whether the first seed passed correctness, fitness and archive-placement checks.</summary>
    public bool IsReady { get; internal set; }
    /// <summary>Gets a stable outcome code.</summary>
    public string Code { get; internal set; } = "not_checked";
    /// <summary>Gets the exact source-and-language identity of the checked seed.</summary>
    public string SeedGenomeId { get; internal set; } = string.Empty;
    /// <summary>Gets the correctness evaluator's hashed identity.</summary>
    public string? CorrectnessIdentity { get; internal set; }
    /// <summary>Gets the fitness evaluator's hashed identity.</summary>
    public string? FitnessIdentity { get; internal set; }
    /// <summary>Gets the correctness status, or null if not dispatched.</summary>
    public EvolutionEvaluationStatus? CorrectnessStatus { get; internal set; }
    /// <summary>Gets the fitness status, or null if not dispatched or shared with correctness.</summary>
    public EvolutionEvaluationStatus? AdditionalFitnessStatus { get; internal set; }
    /// <summary>Gets reported correctness cost; null means no receipt, not zero consumption.</summary>
    public double? CorrectnessCostUnits { get; internal set; }
    /// <summary>Gets reported additional fitness cost; null means no additional receipt.</summary>
    public double? AdditionalFitnessCostUnits { get; internal set; }
    /// <summary>Gets whether the same evaluation satisfied both correctness and fitness.</summary>
    public bool SharedCorrectnessAndFitness { get; internal set; }
    /// <summary>Gets the number of configured public input/output examples.</summary>
    public int InputOutputCases { get; internal set; }
    /// <summary>Gets whether configured output parents accepted an exclusive temporary write.</summary>
    public bool OutputLocationsChecked { get; internal set; }
}
