namespace AiDotNet.Enums;

/// <summary>
/// How the step ε of <see cref="NeuSGradientMode.FiniteDifference"/> is chosen (instant-nsr-pl
/// <c>geometry.finite_difference_eps</c>).
/// </summary>
public enum NeuSFiniteDifferenceStep
{
    /// <summary>A constant ε (a float <c>finite_difference_eps</c>, by default 1e-3).</summary>
    Fixed,

    /// <summary>
    /// Neuralangelo's coarse-to-fine step (<c>finite_difference_eps: progressive</c>): the cell size of the finest
    /// active hash-grid level, <c>ε = 2r / (b · σ^(level − 1))</c>, shrinking as the progressive band switches finer
    /// levels on.
    /// </summary>
    Progressive,
}
