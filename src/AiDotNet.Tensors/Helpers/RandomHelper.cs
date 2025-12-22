using System.Security.Cryptography;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides thread-safe random number generation utilities for the entire library.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Random numbers are essential in machine learning for:
/// - Initializing neural network weights
/// - Shuffling training data
/// - Sampling subsets for cross-validation
/// - Adding noise for regularization
///
/// This helper provides a centralized, thread-safe way to generate random numbers
/// that works correctly even when multiple threads are running simultaneously.
/// </para>
/// </remarks>
public static class RandomHelper
{
    /// <summary>
    /// Thread-local random instance for thread-safe random number generation.
    /// Each thread gets its own Random instance to avoid thread-safety issues.
    /// </summary>
    private static readonly ThreadLocal<Random> _threadLocalRandom = new(
        () => new LockedRandom(GenerateCryptographicSeed()));

    /// <summary>
    /// Gets the thread-safe random number generator for the current thread.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to a thread-safe random number generator using ThreadLocal.
    /// Each thread gets its own Random instance, ensuring thread safety without locking.
    /// </para>
    /// <para><b>For Beginners:</b> Use this property whenever you need random numbers in your code.
    /// It's safe to use from multiple threads simultaneously, and each thread will get
    /// consistent random sequences.
    /// </para>
    /// </remarks>
    public static Random ThreadSafeRandom => _threadLocalRandom.Value ?? CreateSecureRandom();

    /// <summary>
    /// Generates a cryptographically secure seed for Random initialization.
    /// Uses RandomNumberGenerator to avoid birthday paradox collisions from GetHashCode().
    /// </summary>
    /// <returns>A cryptographically random integer seed.</returns>
    /// <remarks>
    /// <para>
    /// This method uses <see cref="RandomNumberGenerator"/> to generate truly random bytes,
    /// which are then converted to an integer seed. This avoids the birthday paradox issue
    /// that occurs with <c>Guid.NewGuid().GetHashCode()</c>, which has ~50% collision
    /// probability after only ~77,000 values due to the 32-bit hash space.
    /// </para>
    /// <para><b>For Beginners:</b> When creating Random instances, you need a "seed" - a starting
    /// number that determines the sequence of random numbers. If two Random instances have the
    /// same seed, they produce identical sequences.
    ///
    /// Using cryptographically secure seeds ensures that each Random instance starts with
    /// a truly unique seed, making collisions extremely unlikely even in applications with
    /// many threads or instances.
    /// </para>
    /// </remarks>
    public static int GenerateCryptographicSeed()
    {
        byte[] bytes = new byte[4];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(bytes);
        }
        return BitConverter.ToInt32(bytes, 0);
    }

    /// <summary>
    /// Creates a new thread-safe Random instance with a cryptographically secure seed.
    /// </summary>
    /// <returns>A new thread-safe Random instance with a unique seed.</returns>
    /// <remarks>
    /// <para>
    /// Use this method when you need a dedicated Random instance (e.g., for storing in a field)
    /// rather than using the shared thread-local instance.
    /// </para>
    /// <para><b>For Beginners:</b> Most of the time, you should use <see cref="ThreadSafeRandom"/>
    /// instead. Only use this method when you specifically need your own Random instance,
    /// such as when implementing reproducible sequences with seeds.
    /// </para>
    /// </remarks>
    public static Random CreateSecureRandom()
    {
        return new LockedRandom(GenerateCryptographicSeed());
    }

    /// <summary>
    /// Creates a new thread-safe Random instance with the specified seed for reproducible results.
    /// </summary>
    /// <param name="seed">The seed value to initialize the random number generator.</param>
    /// <returns>A new thread-safe Random instance initialized with the specified seed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you need reproducible random sequences,
    /// such as during testing or when you want experiments to be repeatable.
    /// The same seed will always produce the same sequence of random numbers.
    /// </para>
    /// </remarks>
    public static Random CreateSeededRandom(int seed)
    {
        return new LockedRandom(seed);
    }
}
