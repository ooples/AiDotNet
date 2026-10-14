namespace AiDotNet.Helpers;

/// <summary>
/// Copies PyTorch module parameters (named as in a <c>state_dict</c>) into the repository's layers, converting
/// layouts where they differ. The values come from a caller's reader, so a checkpoint can be served from any
/// container: a converted safetensors file, a JSON fixture, or a dictionary.
/// </summary>
/// <remarks>
/// <para>The reader receives the full state-dict name and the PyTorch shape it must supply, and returns the
/// row-major values. Shapes are checked against the layer before anything is written.</para>
/// <para>Layout conversions: <c>nn.Linear.weight</c> is <c>[out, in]</c> and is transposed into
/// <see cref="DenseLayer{T}"/>'s <c>[in, out]</c>; convolution kernels, LayerNorm, BatchNorm and embedding tables
/// share PyTorch's layout.</para>
/// </remarks>
internal sealed class TorchStateDictLoader<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Func<string, int[], double[]> _read;
    private readonly IEngine _engine;
    private readonly HashSet<string> _loaded = new(StringComparer.Ordinal);

    public TorchStateDictLoader(Func<string, int[], double[]> read, IEngine engine)
    {
        Guard.NotNull(read);
        Guard.NotNull(engine);
        _read = read;
        _engine = engine;
    }

    /// <summary>Every state-dict name read so far.</summary>
    public IReadOnlyCollection<string> Loaded => _loaded;

    private double[] Read(string name, int[] shape)
    {
        var values = _read(name, shape);
        int expected = 1;
        foreach (int d in shape) expected *= d;
        if (values is null || values.Length != expected)
            throw new ArgumentException(
                $"'{name}' must supply {expected} values for shape [{string.Join(", ", shape)}]; got {values?.Length ?? 0}.");
        _loaded.Add(name);
        return values;
    }

    private void Fill(Tensor<T> target, double[] values, string name)
    {
        if (target.Length != values.Length)
            throw new InvalidOperationException($"'{name}': the layer holds {target.Length} values, the checkpoint {values.Length}.");
        for (int i = 0; i < values.Length; i++) target.SetFlat(i, NumOps.FromDouble(values[i]));
        _engine.InvalidatePersistentTensor(target);
    }

    /// <summary><c>nn.Linear</c>: <c>{prefix}.weight [out, in]</c> and, when the layer has one, <c>{prefix}.bias</c>.</summary>
    public void Linear(DenseLayer<T> layer, string prefix)
    {
        var weights = layer.GetWeights();                                      // [in, out]
        int inputs = weights.Shape[0], outputs = weights.Shape[1];
        var values = Read(prefix + ".weight", new[] { outputs, inputs });
        var transposed = new double[values.Length];
        for (int o = 0; o < outputs; o++)
            for (int i = 0; i < inputs; i++) transposed[i * outputs + o] = values[o * inputs + i];
        Fill(weights, transposed, prefix + ".weight");
        if (layer.UseBias) Fill(layer.GetBiases(), Read(prefix + ".bias", new[] { outputs }), prefix + ".bias");
    }

    /// <summary><c>nn.Conv2d</c> with a square kernel: <c>{prefix}.weight [out, in, k, k]</c> and <c>{prefix}.bias</c>.</summary>
    public void Conv2d(ConvolutionalLayer<T> layer, string prefix)
    {
        var kernels = layer.GetFilters();
        Fill(kernels, Read(prefix + ".weight", kernels.Shape.ToArray()), prefix + ".weight");
        var bias = layer.GetBiases();
        if (bias is not null) Fill(bias, Read(prefix + ".bias", new[] { bias.Length }), prefix + ".bias");
    }

    /// <summary>A non-overlapping <c>nn.Conv2d</c> (<c>[out, in, kh, kw]</c>, stride = kernel) held as a dense map
    /// of each flattened patch, which is what it computes.</summary>
    public void PatchConvAsLinear(DenseLayer<T> layer, string prefix, int inChannels, int kernelHeight, int kernelWidth)
    {
        var weights = layer.GetWeights();                                      // [in·kh·kw, out]
        int inputs = weights.Shape[0], outputs = weights.Shape[1];
        if (inputs != inChannels * kernelHeight * kernelWidth)
            throw new InvalidOperationException($"'{prefix}': the dense map reads {inputs} values per patch, not {inChannels * kernelHeight * kernelWidth}.");
        var values = Read(prefix + ".weight", new[] { outputs, inChannels, kernelHeight, kernelWidth });
        var transposed = new double[values.Length];
        for (int o = 0; o < outputs; o++)
            for (int i = 0; i < inputs; i++) transposed[i * outputs + o] = values[o * inputs + i];
        Fill(weights, transposed, prefix + ".weight");
        if (layer.UseBias) Fill(layer.GetBiases(), Read(prefix + ".bias", new[] { outputs }), prefix + ".bias");
    }

    /// <summary><c>nn.LayerNorm</c>: <c>{prefix}.weight</c> and <c>{prefix}.bias</c>.</summary>
    public void LayerNorm(LayerNormalizationLayer<T> layer, string prefix)
    {
        var gamma = layer.GetGammaTensor();
        Fill(gamma, Read(prefix + ".weight", new[] { gamma.Length }), prefix + ".weight");
        Fill(layer.GetBetaTensor(), Read(prefix + ".bias", new[] { gamma.Length }), prefix + ".bias");
    }

    /// <summary><c>nn.BatchNorm*d</c>: affine parameters and the running statistics.</summary>
    public void BatchNorm(BatchNormalizationLayer<T> layer, string prefix)
    {
        var gamma = layer.GetGamma();
        int n = gamma.Length;
        Fill(gamma, Read(prefix + ".weight", new[] { n }), prefix + ".weight");
        Fill(layer.GetBeta(), Read(prefix + ".bias", new[] { n }), prefix + ".bias");
        Fill(layer.GetRunningMean(), Read(prefix + ".running_mean", new[] { n }), prefix + ".running_mean");
        Fill(layer.GetRunningVariance(), Read(prefix + ".running_var", new[] { n }), prefix + ".running_var");
    }

    /// <summary><c>nn.Embedding</c>: <c>{prefix}.weight [rows, dim]</c>.</summary>
    public void Embedding(EmbeddingLayer<T> layer, string prefix)
    {
        var table = layer.GetMaterializedEmbeddingTable();
        Fill(table, Read(prefix + ".weight", table.Shape.ToArray()), prefix + ".weight");
    }

    /// <summary>A 0-dimensional PyTorch parameter (shape <c>[]</c>, e.g. a logit scale) into a one-element tensor.</summary>
    public void Scalar(Tensor<T> target, string name)
    {
        Fill(target, Read(name, Array.Empty<int>()), name);
    }

    /// <summary>A raw parameter tensor in PyTorch's layout (e.g. a relative-position bias table).</summary>
    public void Tensor(Tensor<T> target, string name)
    {
        Fill(target, Read(name, target.Shape.ToArray()), name);
    }
}
