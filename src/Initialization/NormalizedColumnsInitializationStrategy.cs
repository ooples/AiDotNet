namespace AiDotNet.Initialization;

/// <summary>
/// OpenAI Baselines' <c>normc_initializer</c>: standard-normal weights, each output unit's incoming column rescaled
/// to a fixed norm, and zero biases.
/// </summary>
/// <remarks>
/// <para>
/// <c>out = randn(in, out); out *= std / sqrt(sum(out², axis=0))</c>. Every unit starts with the same incoming
/// weight norm, <paramref name="std"/>, whatever the fan-in. Baselines' PPO (the code of Schulman et al. 2017)
/// builds its policy and value networks with it: 1.0 on the hidden layers and the value head, 0.01 on the policy
/// head, so the initial policy is close to uniform (or, for a Gaussian, centred on zero).
/// </para>
/// <para>
/// <b>For Beginners:</b> Random starting weights, scaled so every neuron starts equally "loud". A small scale on the
/// last layer makes a new policy start out undecided between its actions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class NormalizedColumnsInitializationStrategy<T> : InitializationStrategyBase<T>
{
    private readonly double _std;

    /// <summary>Creates the initializer.</summary>
    /// <param name="std">Each column's norm (Baselines' <c>std</c>, default 1.0).</param>
    /// <param name="random">The generator to draw from; a seeded one makes the weights reproducible.</param>
    public NormalizedColumnsInitializationStrategy(double std = 1.0, Random? random = null)
        : base(random)
    {
        if (!(std > 0)) throw new ArgumentOutOfRangeException(nameof(std), std, "The column norm must be positive.");
        _std = std;
    }

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    /// <remarks>The weights are laid out <c>[inputSize, outputSize]</c>, as a DenseLayer stores them, so a column is
    /// one output unit's incoming weights.</remarks>
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        if (weights.Length != inputSize * outputSize)
            throw new ArgumentException($"Expected {inputSize}x{outputSize} weights, got {weights.Length}.", nameof(weights));
        var flat = new double[weights.Length];
        for (int i = 0; i < flat.Length; i++) flat[i] = SampleGaussian(0, 1.0);
        var span = weights.AsWritableSpan();
        for (int column = 0; column < outputSize; column++)
        {
            double squares = 0;
            for (int row = 0; row < inputSize; row++) squares += flat[row * outputSize + column] * flat[row * outputSize + column];
            double scale = _std / Math.Sqrt(squares);
            for (int row = 0; row < inputSize; row++)
                span[row * outputSize + column] = NumOps.FromDouble(flat[row * outputSize + column] * scale);
        }
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases) => ZeroInitializeBiases(biases);
}
