namespace AiDotNet.Evolution;

/// <summary>Serializable state for <see cref="StableRandom"/>.</summary>
public readonly struct StableRandomState : IEquatable<StableRandomState>
{
    /// <summary>Initializes a captured PCG state.</summary>
    /// <param name="state">The 64-bit generator state.</param>
    /// <param name="increment">The odd 64-bit stream increment.</param>
    public StableRandomState(ulong state, ulong increment)
    {
        if ((increment & 1UL) == 0) throw new ArgumentException("The PCG increment must be odd.", nameof(increment));
        State = state;
        Increment = increment;
    }

    /// <summary>Gets the 64-bit generator state.</summary>
    public ulong State { get; }

    /// <summary>Gets the odd 64-bit stream increment.</summary>
    public ulong Increment { get; }

    /// <inheritdoc/>
    public bool Equals(StableRandomState other) => State == other.State && Increment == other.Increment;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is StableRandomState other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => State.GetHashCode() * 397 ^ Increment.GetHashCode();
}
