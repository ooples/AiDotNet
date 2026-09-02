using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>
/// A small, checkpointable PCG-XSH-RR generator whose sequence is stable across supported .NET runtimes.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="System.Random"/> is unsuitable for reproducible research runs because its algorithm and seeding differ
/// between .NET Framework and .NET, and its position cannot be captured. This type implements the 64-bit-state,
/// 32-bit-output PCG-XSH-RR generator (O'Neill, 2014, "PCG: A Family of Simple Fast Space-Efficient Statistically Good
/// Algorithms for Random Number Generation", HMC-CS-2014-0905) with an explicit stream increment, so the same seed and
/// stream produce the same sequence on every target framework, and the whole state fits in two 64-bit words that
/// <see cref="CaptureState"/> and <see cref="Restore"/> move in and out of checkpoints. <see cref="CreateStream"/> and
/// <see cref="Fork"/> derive independent streams through a SplitMix64-style finalizer (Steele, Lea, and Flood, 2014,
/// "Fast Splittable Pseudorandom Number Generators"), which is how the engine gives every proposal, refinement, and
/// evaluation its own reproducible randomness without a shared mutable generator.
/// </para>
/// <para><b>For Beginners:</b> Evolution relies on randomness for mutation and selection, but research also needs the
/// exact same run to be repeatable, and a run that was stopped to resume from a checkpoint as if it had never stopped.
/// This generator makes both possible: give it the same seed and it always produces the same numbers, and you can save
/// its position and restore it later. Think of <see cref="CreateStream"/> as opening a fresh, labeled deck of cards for
/// each candidate, shuffled the same way every time the label is the same, so one candidate's draws never disturb
/// another's. Use <see cref="NextInt(int)"/> for indexes, <see cref="NextDouble"/> for probabilities, and
/// <see cref="Fork"/> when a sub-step needs its own deck. Do not share one instance between threads; create or fork a
/// stream per candidate or worker instead.</para>
/// <para>
/// This type is intentionally not thread-safe. Bounded integers use unbiased rejection sampling with O(1) expected
/// draws, and every other operation is O(1). <see cref="AlgorithmId"/> names this exact algorithm and is folded into
/// every checkpoint's compatibility hash, so any change to the generated sequence must be accompanied by a new
/// identifier.
/// </para>
/// </remarks>
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
    /// <exception cref="ArgumentException">
    /// <paramref name="state"/> has an even increment (for example the default <see cref="StableRandomState"/> value) and
    /// therefore was not produced by <see cref="CaptureState"/>.
    /// </exception>
    public static StableRandom Restore(StableRandomState state)
    {
        if ((state.Increment & 1UL) == 0)
            throw new ArgumentException("The captured state must have an odd PCG increment; the default StableRandomState value is not a valid captured state.", nameof(state));
        return new StableRandom(state);
    }

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
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="exclusiveMax"/> is not greater than <paramref name="inclusiveMin"/>.</exception>
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
