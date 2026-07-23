namespace AiDotNet.Initialization;

/// <summary>
/// Eager initialization strategy that initializes weights immediately on construction.
/// </summary>
/// <remarks>
/// <para>
/// This is the traditional initialization approach where weights are allocated and
/// initialized during layer construction. This ensures all weights are ready before
/// any training or inference begins.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EagerInitializationStrategy<T> : InitializationStrategyBase<T>
{
    /// <summary>
    /// Initializes a new instance using the framework's default non-deterministic
    /// RNG.
    /// </summary>
    public EagerInitializationStrategy() : base() { }

    /// <summary>
    /// Initializes a new instance using the supplied <see cref="Random"/> for
    /// reproducible weight initialization. Pass a deterministically-seeded
    /// <see cref="Random"/> to make Xavier/He init produce the same weights
    /// across runs (e.g., for unit tests, multi-seed experiment harnesses,
    /// or reproducible benchmarks).
    /// </summary>
    /// <param name="rng">Seeded random number generator, or <c>null</c> to
    /// fall back to the framework's default thread-safe non-deterministic RNG.</param>
    public EagerInitializationStrategy(Random? rng) : base(rng) { }

    /// <inheritdoc />
    public override IInitializationStrategy<T> WithSeededRandom(Random rng)
        => new EagerInitializationStrategy<T>(rng);

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        XavierNormalInitialize(weights, inputSize, outputSize);
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
