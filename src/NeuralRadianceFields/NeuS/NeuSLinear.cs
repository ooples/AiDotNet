namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// A fully connected layer stored in PyTorch's <c>nn.Linear</c> layout, optionally weight-normalized
/// (<c>nn.utils.weight_norm</c>, Salimans &amp; Kingma 2016), as every NeuS and SparseNeuS MLP layer is.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The weight is <c>[out, in]</c> and <c>y = x Wᵀ + b</c>. With weight normalization the trained parameters are the
/// direction <c>v</c> (<c>weight_v</c>, <c>[out, in]</c>) and the magnitude <c>g</c> (<c>weight_g</c>,
/// <c>[out, 1]</c>), and <c>W = g · v / ‖v‖</c> with the norm taken over each output row. At initialization
/// <c>g = ‖v‖</c>, so the layer starts at exactly the initialized weight. Keeping the PyTorch layout and names lets a
/// reference checkpoint's state dict load tensor for tensor.
/// </para>
/// </remarks>
internal sealed class NeuSLinear<T>
{
    /// <summary>Creates a layer from an initialized weight and bias.</summary>
    /// <param name="weight">The initial weight, <c>[out, in]</c> row-major.</param>
    /// <param name="bias">The initial bias, <c>[out]</c>.</param>
    /// <param name="inputDims">The input width.</param>
    /// <param name="outputDims">The output width.</param>
    /// <param name="weightNorm">Whether to weight-normalize.</param>
    public NeuSLinear(double[] weight, double[] bias, int inputDims, int outputDims, bool weightNorm)
    {
        if (weight.Length != inputDims * outputDims) throw new ArgumentException("The weight has the wrong size.", nameof(weight));
        if (bias.Length != outputDims) throw new ArgumentException("The bias has the wrong size.", nameof(bias));
        InputDims = inputDims;
        OutputDims = outputDims;
        WeightNorm = weightNorm;
        Weight = NeuSHost<T>.FromDoubles(weight, outputDims, inputDims);
        Bias = NeuSHost<T>.FromDoubles(bias, outputDims);
        if (weightNorm)
        {
            var g = new double[outputDims];
            for (int o = 0; o < outputDims; o++)
            {
                double sum = 0;
                for (int i = 0; i < inputDims; i++) sum += weight[o * inputDims + i] * weight[o * inputDims + i];
                g[o] = Math.Sqrt(sum);
            }

            Magnitude = NeuSHost<T>.FromDoubles(g, outputDims, 1);
        }
    }

    /// <summary>The input width.</summary>
    public int InputDims { get; }

    /// <summary>The output width.</summary>
    public int OutputDims { get; }

    /// <summary>Whether the layer is weight-normalized.</summary>
    public bool WeightNorm { get; }

    /// <summary>The weight, or with weight normalization its direction <c>v</c>, <c>[out, in]</c>.</summary>
    public Tensor<T> Weight { get; private set; }

    /// <summary>The weight-normalization magnitude <c>g</c>, <c>[out, 1]</c>, or null without weight normalization.</summary>
    public Tensor<T>? Magnitude { get; private set; }

    /// <summary>The bias, <c>[out]</c>.</summary>
    public Tensor<T> Bias { get; private set; }

    /// <summary>The trainable tensors.</summary>
    public IEnumerable<Tensor<T>> Parameters
    {
        get
        {
            if (Magnitude is not null) yield return Magnitude;
            yield return Weight;
            yield return Bias;
        }
    }

    /// <summary>The parameter count: <c>out·in + out</c>, plus <c>out</c> magnitudes with weight normalization.</summary>
    public int ParameterCount => OutputDims * InputDims + OutputDims + (WeightNorm ? OutputDims : 0);

    /// <summary>The parameters under the reference's state-dict names, prefixed by <paramref name="prefix"/>.</summary>
    /// <param name="prefix">The module path, e.g. <c>"lin0"</c>.</param>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters(string prefix)
    {
        if (Magnitude is not null)
        {
            yield return new KeyValuePair<string, Tensor<T>>(prefix + ".weight_g", Magnitude);
            yield return new KeyValuePair<string, Tensor<T>>(prefix + ".weight_v", Weight);
        }
        else
        {
            yield return new KeyValuePair<string, Tensor<T>>(prefix + ".weight", Weight);
        }

        yield return new KeyValuePair<string, Tensor<T>>(prefix + ".bias", Bias);
    }

    /// <summary>Replaces the parameter named <paramref name="name"/> (as <see cref="NamedParameters"/> lists it).</summary>
    /// <param name="prefix">The module path.</param>
    /// <param name="name">The full parameter name.</param>
    /// <param name="value">The new value; its element count must match.</param>
    /// <returns>Whether the name belongs to this layer.</returns>
    public bool TryLoad(string prefix, string name, Tensor<T> value)
    {
        if (name == prefix + ".bias") { Bias = Checked(value, Bias); return true; }
        if (Magnitude is not null && name == prefix + ".weight_g") { Magnitude = Checked(value, Magnitude); return true; }
        if (name == prefix + (Magnitude is not null ? ".weight_v" : ".weight")) { Weight = Checked(value, Weight); return true; }
        return false;
    }

    /// <summary>The effective weight <c>W</c>, <c>[out, in]</c>: <c>g · v / ‖v‖</c> with weight normalization.</summary>
    /// <param name="engine">The engine.</param>
    public Tensor<T> EffectiveWeight(IEngine engine)
    {
        if (Magnitude is null) return Weight;
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(Weight, Weight), new[] { 1 }, true)); // [out, 1]
        return engine.TensorMultiply(Weight, engine.TensorDivide(Magnitude, norm));
    }

    /// <summary><c>x Wᵀ + b</c> for <c>[N, in]</c> inputs.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="input">The inputs.</param>
    /// <param name="effectiveWeight">The effective weight when the caller already computed it.</param>
    public Tensor<T> Forward(IEngine engine, Tensor<T> input, Tensor<T>? effectiveWeight = null)
    {
        var weight = effectiveWeight ?? EffectiveWeight(engine);
        var linear = engine.TensorMatMul(input, engine.TensorTranspose(weight));
        return engine.TensorAdd(linear, engine.Reshape(Bias, new[] { 1, OutputDims }));
    }

    private static Tensor<T> Checked(Tensor<T> value, Tensor<T> current)
    {
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (value.Length != current.Length)
            throw new ArgumentException($"Expected {current.Length} values, got {value.Length}.", nameof(value));
        return value.Reshape(current.Shape.ToArray());
    }

    /// <summary>
    /// PyTorch's default <c>nn.Linear</c> initialization (Kaiming-uniform with <c>a = √5</c>): weight and bias uniform in
    /// <c>±1/√in</c>.
    /// </summary>
    /// <param name="inputDims">The input width.</param>
    /// <param name="outputDims">The output width.</param>
    /// <param name="random">The generator.</param>
    public static (double[] Weight, double[] Bias) DefaultInit(int inputDims, int outputDims, Random random)
    {
        double bound = 1.0 / Math.Sqrt(inputDims);
        var weight = new double[inputDims * outputDims];
        for (int i = 0; i < weight.Length; i++) weight[i] = (random.NextDouble() * 2.0 - 1.0) * bound;
        var bias = new double[outputDims];
        for (int i = 0; i < bias.Length; i++) bias[i] = (random.NextDouble() * 2.0 - 1.0) * bound;
        return (weight, bias);
    }
}
