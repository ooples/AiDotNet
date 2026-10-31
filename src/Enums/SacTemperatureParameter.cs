namespace AiDotNet.Enums;

/// <summary>
/// What a SAC agent with a learned temperature takes gradient steps on.
/// </summary>
public enum SacTemperatureParameter
{
    /// <summary>
    /// α itself, as the 2018b paper's Algorithm 1 writes it (α ← α − λ∇̂J(α)). The objective is linear in α, so
    /// nothing keeps α positive; a step that would make it non-positive throws.
    /// </summary>
    Alpha = 0,

    /// <summary>log α, as softlearning (the authors' code) learns it, which keeps α positive.</summary>
    LogAlpha = 1,
}
