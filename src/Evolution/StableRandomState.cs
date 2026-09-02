namespace AiDotNet.Evolution;

/// <summary>Serializable state for <see cref="StableRandom"/>.</summary>
/// <remarks>
/// <para>
/// A PCG generator (O'Neill, 2014, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms
/// for Random Number Generation") is fully described by its 64-bit linear congruential state and its odd 64-bit
/// increment, which selects one of 2^63 independent streams. Capturing these two values with
/// <see cref="StableRandom.CaptureState"/> and later calling <see cref="StableRandom.Restore"/> resumes the exact
/// same output sequence on every supported .NET runtime, which is what makes evolution checkpoints deterministic.
/// The constructor rejects even increments because they shorten the period; the default value of this struct has
/// an increment of zero and is therefore not a valid captured state.
/// </para>
/// <para><b>For Beginners:</b> A random number generator is really a tiny deterministic machine: from the same
/// starting position it always produces the same numbers. This struct is a snapshot of that position, like a
/// bookmark in a very long, fixed list of numbers. The evolution engine stores such bookmarks in its checkpoints so
/// that a run stopped on Monday and resumed on Tuesday proposes exactly the same candidates it would have proposed
/// had it never stopped. You normally never build one by hand; you obtain it from
/// <see cref="StableRandom.CaptureState"/> and hand it back to <see cref="StableRandom.Restore"/>.</para>
/// </remarks>
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
