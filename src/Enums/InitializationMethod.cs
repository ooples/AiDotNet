namespace AiDotNet.Enums;

/// <summary>
/// Methods for initializing a population.
/// </summary>
public enum InitializationMethod
{
    /// <summary>
    /// Random initialization - create individuals with random genes.
    /// </summary>
    Random,

    /// <summary>
    /// Case-based initialization - create individuals based on known good solutions.
    /// </summary>
    CaseBased,

    /// <summary>
    /// Heuristic initialization - create individuals using problem-specific heuristics.
    /// </summary>
    Heuristic,

    /// <summary>
    /// Diverse initialization - create a diverse set of individuals.
    /// </summary>
    Diverse,

    /// <summary>
    /// Grid initialization - create individuals that systematically cover the solution space.
    /// </summary>
    Grid,

    XavierUniform
}
