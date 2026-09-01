using AiDotNet.Validation;

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

/// <summary>
/// A small, checkpointable PCG-XSH-RR generator whose sequence is stable across supported .NET runtimes.
/// </summary>
/// <remarks>This type is intentionally not thread-safe. Fork one stream per candidate or worker.</remarks>
public sealed class StableRandom
{
    private const ulong Multiplier = 6364136223846793005UL;
    private ulong _state;
    private ulong _increment;

    /// <summary>Gets the versioned algorithm identifier stored in checkpoints.</summary>
    public const string AlgorithmId = "pcg-xsh-rr-32-v1";

    /// <summary>Initializes a deterministic stream from a seed and stream selector.</summary>
    /// <param name="seed">The root seed.</param>
    /// <param name="stream">The independent stream selector.</param>
    public StableRandom(ulong seed, ulong stream = 0)
    {
        _state = 0;
        _increment = unchecked((stream << 1) | 1UL);
        NextUInt32();
        _state = unchecked(_state + seed);
        NextUInt32();
    }

    private StableRandom(StableRandomState state)
    {
        _state = state.State;
        _increment = state.Increment;
    }

    /// <summary>Restores a stream from a previously captured state.</summary>
    /// <param name="state">The captured state.</param>
    /// <returns>A generator positioned at the captured point.</returns>
    public static StableRandom Restore(StableRandomState state) => new(state);

    /// <summary>Creates a stream derived solely from a root seed and stable stream identifier.</summary>
    /// <param name="rootSeed">The run's root seed.</param>
    /// <param name="streamId">A stable candidate or operation identifier.</param>
    /// <returns>An independent deterministic stream.</returns>
    public static StableRandom CreateStream(ulong rootSeed, ulong streamId)
    {
        ulong seed = Mix(unchecked(rootSeed + 0x9E3779B97F4A7C15UL * (streamId + 1UL)));
        ulong stream = Mix(unchecked(streamId ^ rootSeed ^ 0xD1B54A32D192ED03UL));
        return new StableRandom(seed, stream);
    }

    /// <summary>Captures the complete state required to resume this exact sequence.</summary>
    public StableRandomState CaptureState() => new(_state, _increment);

    /// <summary>Returns the next uniformly distributed 32-bit value.</summary>
    public uint NextUInt32()
    {
        ulong oldState = _state;
        _state = unchecked(oldState * Multiplier + _increment);
        uint xorShifted = (uint)(((oldState >> 18) ^ oldState) >> 27);
        int rotation = (int)(oldState >> 59);
        return (xorShifted >> rotation) | (xorShifted << ((-rotation) & 31));
    }

    /// <summary>Returns the next uniformly distributed 64-bit value.</summary>
    public ulong NextUInt64() => ((ulong)NextUInt32() << 32) | NextUInt32();

    /// <summary>Returns a value in the half-open interval [0, <paramref name="exclusiveMax"/>).</summary>
    /// <param name="exclusiveMax">The exclusive upper bound.</param>
    public int NextInt(int exclusiveMax)
    {
        Guard.Positive(exclusiveMax);
        uint bound = (uint)exclusiveMax;
        uint threshold = unchecked((uint)(0U - bound)) % bound;
        while (true)
        {
            uint value = NextUInt32();
            if (value >= threshold) return (int)(value % bound);
        }
    }

    /// <summary>Returns a value in the half-open interval [<paramref name="inclusiveMin"/>, <paramref name="exclusiveMax"/>).</summary>
    /// <param name="inclusiveMin">The inclusive lower bound.</param>
    /// <param name="exclusiveMax">The exclusive upper bound.</param>
    public int NextInt(int inclusiveMin, int exclusiveMax)
    {
        if (exclusiveMax <= inclusiveMin) throw new ArgumentOutOfRangeException(nameof(exclusiveMax));
        long width = (long)exclusiveMax - inclusiveMin;
        if (width <= int.MaxValue) return inclusiveMin + NextInt((int)width);

        ulong bound = (ulong)width;
        ulong threshold = unchecked(0UL - bound) % bound;
        while (true)
        {
            ulong value = NextUInt64();
            if (value >= threshold) return (int)(inclusiveMin + (long)(value % bound));
        }
    }

    /// <summary>Returns a value in the half-open interval [0, 1).</summary>
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    /// <summary>Creates a child stream without advancing this stream.</summary>
    /// <param name="streamId">A stable child-stream identifier.</param>
    public StableRandom Fork(ulong streamId) => CreateStream(Mix(_state ^ _increment), streamId);

    private static ulong Mix(ulong value)
    {
        value ^= value >> 30;
        value = unchecked(value * 0xBF58476D1CE4E5B9UL);
        value ^= value >> 27;
        value = unchecked(value * 0x94D049BB133111EBUL);
        return value ^ (value >> 31);
    }
}
