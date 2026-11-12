namespace AiDotNet.Initialization;

/// <summary>
/// PyTorch's default initialization of <c>nn.Conv*d</c> and <c>nn.Linear</c>: <c>kaiming_uniform_</c> with
/// a = √5, which is U(±1/√fan_in) for the weight, and U(±1/√fan_in) for the bias.
/// </summary>
/// <remarks>
/// <para>
/// Reference implementations that never re-initialize their modules train from this distribution, so a port that
/// wants their starting point needs it rather than the gain-scaled Kaiming default of the convolution layers, whose
/// bound is √3 times wider for an identity activation and whose biases start at zero.
/// </para>
/// <para><b>For Beginners:</b> Use this when porting a PyTorch model whose code builds its layers and never calls an
/// init function on them; it reproduces the weights such a model starts training from.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TorchDefaultInitializationStrategy<T> : InitializationStrategyBase<T>
{
    // The bias bound uses the fan-in of the weight initialized just before it, as nn.Conv2d.reset_parameters does;
    // InitializeBiases has no fan-in parameter, and layers initialize their weight first.
    private int _lastFanIn;

    /// <summary>Creates the strategy with the framework's default thread-safe RNG.</summary>
    public TorchDefaultInitializationStrategy() { }

    /// <summary>Creates the strategy with a caller-supplied <see cref="Random"/> source.</summary>
    /// <param name="rng">Seeded RNG (or null for the default thread-safe RNG).</param>
    public TorchDefaultInitializationStrategy(Random? rng) : base(rng) { }

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        _lastFanIn = Math.Max(1, inputSize);
        Fill(weights.AsWritableSpan(), 1.0 / Math.Sqrt(_lastFanIn));
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        if (_lastFanIn == 0)
        {
            ZeroInitializeBiases(biases);
            return;
        }
        Fill(biases.AsWritableSpan(), 1.0 / Math.Sqrt(_lastFanIn));
    }

    private void Fill(Span<T> span, double bound)
    {
        for (int i = 0; i < span.Length; i++)
            span[i] = NumOps.FromDouble((Random.NextDouble() * 2.0 - 1.0) * bound);
    }
}
