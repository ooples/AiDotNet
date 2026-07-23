namespace AiDotNet.Enums;

/// <summary>
/// Methods for selecting individuals for reproduction.
/// </summary>
public enum SelectionMethod
{
    /// <summary>
    /// Tournament selection - randomly select a group of individuals and pick the best.
    /// </summary>
    Tournament,

    /// <summary>
    /// Roulette wheel selection - selection probability proportional to fitness.
    /// </summary>
    RouletteWheel,

    /// <summary>
    /// Rank selection - selection probability based on fitness rank rather than absolute value.
    /// </summary>
    Rank,

    /// <summary>
    /// Truncation selection - select a percentage of the fittest individuals.
    /// </summary>
    Truncation,

    /// <summary>
    /// Uniform selection - all individuals have an equal chance of being selected.
    /// </summary>
    Uniform,

    /// <summary>
    /// Stochastic universal sampling - similar to roulette wheel but with multiple equally spaced pointers.
    /// </summary>
    StochasticUniversalSampling,

    Elitism
}
