namespace AiDotNet.Enums;

/// <summary>
/// Defines compute budget presets for AutoML runs.
/// </summary>
/// <remarks>
/// <para>
/// These presets provide industry-standard defaults for time/trial limits and evaluation rigor.
/// Users can start with a preset and optionally override specific limits via configuration options.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like choosing a "speed vs quality" mode:
/// <list type="bullet">
/// <item><description><see cref="CI"/> is the fastest and is designed for pipelines.</description></item>
/// <item><description><see cref="Fast"/> is for quick local experimentation.</description></item>
/// <item><description><see cref="Standard"/> is the recommended default for most users.</description></item>
/// <item><description><see cref="Thorough"/> spends more time searching for the best result.</description></item>
/// </list>
/// </para>
/// </remarks>
public enum AutoMLBudgetPreset
{
    /// <summary>
    /// A very small, deterministic budget intended for CI pipelines.
    /// </summary>
    CI,

    /// <summary>
    /// A small budget intended for quick local experimentation.
    /// </summary>
    Fast,

    /// <summary>
    /// A balanced budget intended as the default for most use cases.
    /// </summary>
    Standard,

    /// <summary>
    /// A larger budget intended for deeper searches and better model quality.
    /// </summary>
    Thorough
}

