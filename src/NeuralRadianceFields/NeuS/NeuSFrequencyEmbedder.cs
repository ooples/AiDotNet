namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// The frequency embedding NeuS and SparseNeuS use (the NeRF embedder with <c>include_input=True</c> and
/// <c>log_sampling=True</c>): <c>γ(x) = (x, sin(2⁰x), cos(2⁰x), …, sin(2^{L−1}x), cos(2^{L−1}x))</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unlike <see cref="Helpers.PositionalEncoder{T}"/> (NeRF's <c>2^l·π</c> frequencies without the identity term),
/// the NeuS embedder keeps the raw input and uses frequencies <c>2^l</c> with no π, and orders the output per frequency
/// as the sine of every input dimension followed by their cosines (reference <c>models/embedder.py</c>). The output
/// width is <c>d·(1 + 2L)</c>; <c>L = 0</c> is the identity.
/// </para>
/// <para>
/// <see cref="Backward"/> is the embedding's vector-Jacobian product, in engine operations, so an SDF network's
/// spatial gradient (the surface normal) is itself differentiable and the Eikonal loss trains the network.
/// </para>
/// </remarks>
internal sealed class NeuSFrequencyEmbedder<T>
{
    private readonly Tensor<T>? _frequencies; // [1, L, 1]

    /// <summary>Creates the embedder.</summary>
    /// <param name="inputDims">The input dimension d.</param>
    /// <param name="frequencies">The number of frequencies L (the reference's <c>multires</c>); 0 is the identity.</param>
    public NeuSFrequencyEmbedder(int inputDims, int frequencies)
    {
        if (inputDims <= 0) throw new ArgumentOutOfRangeException(nameof(inputDims), "The input dimension must be positive.");
        if (frequencies < 0) throw new ArgumentOutOfRangeException(nameof(frequencies), "The frequency count cannot be negative.");
        InputDims = inputDims;
        Frequencies = frequencies;
        if (frequencies > 0)
        {
            var bands = new double[frequencies];
            for (int l = 0; l < frequencies; l++) bands[l] = Math.Pow(2.0, l);
            _frequencies = NeuSHost<T>.FromDoubles(bands, 1, frequencies, 1);
        }
    }

    /// <summary>The input dimension.</summary>
    public int InputDims { get; }

    /// <summary>The number of frequencies.</summary>
    public int Frequencies { get; }

    /// <summary>The output width, <c>d·(1 + 2L)</c>.</summary>
    public int OutputDims => InputDims * (1 + 2 * Frequencies);

    /// <summary>Embeds <c>[N, d]</c> inputs to <c>[N, d·(1 + 2L)]</c>.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="input">The inputs.</param>
    public Tensor<T> Embed(IEngine engine, Tensor<T> input)
    {
        if (_frequencies is null) return input;
        int n = input.Shape[0];
        var angles = Angles(engine, input); // [N, L, d]
        var pairs = engine.TensorConcatenate(new[]
        {
            engine.Reshape(engine.TensorSin(angles), new[] { n, Frequencies, 1, InputDims }),
            engine.Reshape(engine.TensorCos(angles), new[] { n, Frequencies, 1, InputDims }),
        }, 2); // [N, L, 2, d]: per frequency, the sines of every dimension then their cosines
        return engine.TensorConcatenate(new[]
        {
            input,
            engine.Reshape(pairs, new[] { n, 2 * Frequencies * InputDims }),
        }, 1);
    }

    /// <summary>
    /// The vector-Jacobian product: given the gradient of a scalar with respect to the embedding, <c>[N, d(1+2L)]</c>,
    /// its gradient with respect to the input, <c>[N, d]</c>.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="input">The inputs the embedding was taken at.</param>
    /// <param name="embeddingGradient">The gradient with respect to the embedding.</param>
    public Tensor<T> Backward(IEngine engine, Tensor<T> input, Tensor<T> embeddingGradient)
    {
        int n = input.Shape[0];
        var identity = engine.TensorSlice(embeddingGradient, new[] { 0, 0 }, new[] { n, InputDims });
        if (_frequencies is null) return identity;

        var periodic = engine.Reshape(
            engine.TensorSlice(embeddingGradient, new[] { 0, InputDims }, new[] { n, 2 * Frequencies * InputDims }),
            new[] { n, Frequencies, 2, InputDims });
        var gradSin = engine.Reshape(
            engine.TensorSlice(periodic, new[] { 0, 0, 0, 0 }, new[] { n, Frequencies, 1, InputDims }),
            new[] { n, Frequencies, InputDims });
        var gradCos = engine.Reshape(
            engine.TensorSlice(periodic, new[] { 0, 0, 1, 0 }, new[] { n, Frequencies, 1, InputDims }),
            new[] { n, Frequencies, InputDims });
        var angles = Angles(engine, input);
        // d sin(f x)/dx = f cos(f x); d cos(f x)/dx = −f sin(f x).
        var perFrequency = engine.TensorMultiply(
            engine.TensorSubtract(
                engine.TensorMultiply(gradSin, engine.TensorCos(angles)),
                engine.TensorMultiply(gradCos, engine.TensorSin(angles))),
            _frequencies);
        return engine.TensorAdd(identity, engine.ReduceSum(perFrequency, new[] { 1 }, false));
    }

    private Tensor<T> Angles(IEngine engine, Tensor<T> input)
    {
        int n = input.Shape[0];
        var frequencies = _frequencies ?? throw new InvalidOperationException("The identity embedding has no angles.");
        return engine.TensorMultiply(engine.Reshape(input, new[] { n, 1, InputDims }), frequencies); // [N, L, d]
    }
}
