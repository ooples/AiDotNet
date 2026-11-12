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
    private readonly AiDotNet.ModelLoading.TorchTensorSource _source;
    private readonly IEngine _engine;
    private readonly HashSet<string> _loaded = new(StringComparer.Ordinal);

    public TorchStateDictLoader(Func<string, int[], double[]> read, IEngine engine)
        : this(AiDotNet.ModelLoading.TorchTensorSource.From(read ?? throw new ArgumentNullException(nameof(read))), engine)
    {
    }

    /// <summary>A loader copying each tensor of <paramref name="source"/> straight into the layers.</summary>
    public TorchStateDictLoader(AiDotNet.ModelLoading.TorchTensorSource source, IEngine engine)
    {
        Guard.NotNull(source);
        Guard.NotNull(engine);
        _source = source;
        _engine = engine;
    }

    /// <summary>Every state-dict name read so far.</summary>
    public IReadOnlyCollection<string> Loaded => _loaded;

    private double[] Read(string name, int[] shape)
    {
        var values = _source.Read(name, shape);
        int expected = 1;
        foreach (int d in shape) expected *= d;
        if (values.Length != expected)
            throw new ArgumentException(
                $"'{name}' must supply {expected} values for shape [{string.Join(", ", shape)}]; got {values.Length}.");
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

    // Copies the named tensor of torch shape [shape] into the target in stored order.
    private void Copy(Tensor<T> target, string name, int[] shape)
    {
        _source.CopyTo(name, shape, target);
        _loaded.Add(name);
        _engine.InvalidatePersistentTensor(target);
    }

    /// <summary><c>nn.Linear</c>: <c>{prefix}.weight [out, in]</c> and, when the layer has one, <c>{prefix}.bias</c>.</summary>
    public void Linear(DenseLayer<T> layer, string prefix)
    {
        var weights = layer.GetWeights();                                      // [in, out]
        int inputs = weights.Shape[0], outputs = weights.Shape[1];
        _source.CopyTransposedTo(prefix + ".weight", outputs, inputs, weights);
        _loaded.Add(prefix + ".weight");
        _engine.InvalidatePersistentTensor(weights);
        if (layer.UseBias) Copy(layer.GetBiases(), prefix + ".bias", new[] { outputs });
    }

    /// <summary><c>nn.Conv2d</c> with a square kernel: <c>{prefix}.weight [out, in, k, k]</c> and <c>{prefix}.bias</c>.</summary>
    public void Conv2d(ConvolutionalLayer<T> layer, string prefix)
    {
        var kernels = layer.GetFilters();
        Copy(kernels, prefix + ".weight", kernels.Shape.ToArray());
        var bias = layer.GetBiases();
        if (bias is not null) Copy(bias, prefix + ".bias", new[] { bias.Length });
    }

    /// <summary>A non-overlapping <c>nn.Conv2d</c> (<c>[out, in, kh, kw]</c>, stride = kernel) held as a dense map
    /// of each flattened patch, which is what it computes.</summary>
    public void PatchConvAsLinear(DenseLayer<T> layer, string prefix, int inChannels, int kernelHeight, int kernelWidth)
    {
        var weights = layer.GetWeights();                                      // [in·kh·kw, out]
        int inputs = weights.Shape[0], outputs = weights.Shape[1];
        if (inputs != inChannels * kernelHeight * kernelWidth)
            throw new InvalidOperationException($"'{prefix}': the dense map reads {inputs} values per patch, not {inChannels * kernelHeight * kernelWidth}.");
        // [out, in, kh, kw] is [out, in·kh·kw] row-major: the dense map's transpose.
        _source.CopyTransposedTo(prefix + ".weight", new[] { outputs, inChannels, kernelHeight, kernelWidth }, outputs, inputs, weights);
        _loaded.Add(prefix + ".weight");
        _engine.InvalidatePersistentTensor(weights);
        if (layer.UseBias) Copy(layer.GetBiases(), prefix + ".bias", new[] { outputs });
    }

    /// <summary><c>nn.LayerNorm</c>: <c>{prefix}.weight</c> and <c>{prefix}.bias</c>.</summary>
    public void LayerNorm(LayerNormalizationLayer<T> layer, string prefix)
    {
        var gamma = layer.GetGammaTensor();
        Copy(gamma, prefix + ".weight", new[] { gamma.Length });
        Copy(layer.GetBetaTensor(), prefix + ".bias", new[] { gamma.Length });
    }

    /// <summary><c>nn.BatchNorm*d</c>: affine parameters and the running statistics.</summary>
    public void BatchNorm(BatchNormalizationLayer<T> layer, string prefix)
    {
        var gamma = layer.GetGamma();
        int n = gamma.Length;
        Copy(gamma, prefix + ".weight", new[] { n });
        Copy(layer.GetBeta(), prefix + ".bias", new[] { n });
        Copy(layer.GetRunningMean(), prefix + ".running_mean", new[] { n });
        Copy(layer.GetRunningVariance(), prefix + ".running_var", new[] { n });
    }

    /// <summary><c>nn.Embedding</c>: <c>{prefix}.weight [rows, dim]</c>.</summary>
    public void Embedding(EmbeddingLayer<T> layer, string prefix)
    {
        var table = layer.GetMaterializedEmbeddingTable();
        Copy(table, prefix + ".weight", table.Shape.ToArray());
    }

    /// <summary>A 0-dimensional PyTorch parameter (shape <c>[]</c>, e.g. a logit scale) into a one-element tensor.</summary>
    public void Scalar(Tensor<T> target, string name)
    {
        Fill(target, Read(name, Array.Empty<int>()), name);
    }

    /// <summary>A raw parameter tensor in PyTorch's layout (e.g. a relative-position bias table).</summary>
    public void Tensor(Tensor<T> target, string name)
    {
        Copy(target, name, target.Shape.ToArray());
    }
}
