namespace AiDotNet.Models.Options;

public abstract class ModelOptions
{
    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>
    /// The random seed value, or null if randomness should not be controlled.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like setting the starting point for a random number generator.
    /// If you set a specific seed value, the "random" decisions the algorithm makes will be the same each time you run it.
    /// This is useful when you want consistent results or when debugging. If left as null (the default),
    /// the algorithm will make truly random decisions each time it runs.</para>
    /// </remarks>
    public int? Seed { get; set; }
}
