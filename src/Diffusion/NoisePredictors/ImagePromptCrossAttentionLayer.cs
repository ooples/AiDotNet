using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// ImageDream's decoupled image cross-attention (IP-Adapter style): <c>to_out(Attn(q, k, v) + w Attn(q, k_ip, v_ip))</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is <c>MemoryEfficientCrossAttention</c> with <c>with_ip</c> in the official ImageDream code
/// (<c>imagedream/ldm/modules/attention.py</c>). The context is the text tokens followed by <c>ipTokens</c> image
/// tokens; the text tokens go through <c>to_k</c>/<c>to_v</c>, the image tokens through their own <c>to_k_ip</c>/
/// <c>to_v_ip</c>, both attentions share the query <c>to_q</c>, and the image attention is scaled by
/// <c>ip_weight</c> and added before the one output projection <c>to_out.0</c>. Every projection is bias-free except
/// <c>to_out.0</c>, and all start at PyTorch's <c>nn.Linear</c> default.
/// </para>
/// </remarks>
[ElementWiseShape(Note = "Returns the query's [B, S, Q] shape.")]
internal sealed partial class ImagePromptCrossAttentionLayer<T> : LayerBase<T>
{
    private readonly int _queryDimension;
    private readonly int _contextDimension;
    private readonly int _headCount;
    private readonly int _imageTokens;
    private readonly double _imageWeight;

    private readonly DenseLayer<T> _query;
    private readonly DenseLayer<T> _key;
    private readonly DenseLayer<T> _value;
    private readonly DenseLayer<T> _imageKey;
    private readonly DenseLayer<T> _imageValue;
    private readonly DenseLayer<T> _output;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>The number of trailing context tokens that are image tokens (<c>ip_dim</c>).</summary>
    public int ImageTokens => _imageTokens;

    /// <summary>The image attention's scale (<c>ip_weight</c>).</summary>
    public double ImageWeight => _imageWeight;

    public ImagePromptCrossAttentionLayer(int queryDimension, int contextDimension, int headCount, int imageTokens, double imageWeight)
        : base([-1, queryDimension], [-1, queryDimension])
    {
        if (queryDimension <= 0) throw new ArgumentOutOfRangeException(nameof(queryDimension));
        if (contextDimension <= 0) throw new ArgumentOutOfRangeException(nameof(contextDimension));
        if (headCount <= 0 || queryDimension % headCount != 0)
            throw new ArgumentException("Head count must evenly divide the query dimension.", nameof(headCount));
        if (imageTokens <= 0) throw new ArgumentOutOfRangeException(nameof(imageTokens), "Decoupled attention needs image tokens.");
        _queryDimension = queryDimension;
        _contextDimension = contextDimension;
        _headCount = headCount;
        _imageTokens = imageTokens;
        _imageWeight = imageWeight;

        _query = Linear(queryDimension, queryDimension, bias: false);
        _key = Linear(contextDimension, queryDimension, bias: false);
        _value = Linear(contextDimension, queryDimension, bias: false);
        _imageKey = Linear(contextDimension, queryDimension, bias: false);
        _imageValue = Linear(contextDimension, queryDimension, bias: false);
        _output = Linear(queryDimension, queryDimension, bias: true);
        foreach (var layer in ParameterLayers()) RegisterSubLayer(layer);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input) =>
        throw new InvalidOperationException(
            "ImagePromptCrossAttentionLayer requires a context of text tokens followed by image tokens.");

    /// <summary>
    /// Attends <paramref name="query"/> <c>[B, S, Q]</c> to <paramref name="context"/> <c>[B, L + ipTokens, C]</c>: the
    /// first L tokens are text, the last <see cref="ImageTokens"/> are image.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> query, Tensor<T> context)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (query.Rank != 3 || context.Rank != 3)
            throw new ArgumentException("Attention requires query/context tensors of rank 3.");
        if (query.Shape[0] != context.Shape[0])
            throw new ArgumentException("Query and context batch sizes must match.", nameof(context));
        if (query.Shape[2] != _queryDimension)
            throw new ArgumentException($"Expected query width {_queryDimension}, got {query.Shape[2]}.", nameof(query));
        if (context.Shape[2] != _contextDimension)
            throw new ArgumentException($"Expected context width {_contextDimension}, got {context.Shape[2]}.", nameof(context));
        int textTokens = context.Shape[1] - _imageTokens;
        if (textTokens <= 0)
            throw new ArgumentException(
                $"The context has {context.Shape[1]} tokens; ImageDream's needs text tokens followed by {_imageTokens} image tokens.",
                nameof(context));

        var text = Engine.TensorNarrow(context, 1, 0, textTokens).Contiguous();
        var image = Engine.TensorNarrow(context, 1, textTokens, _imageTokens).Contiguous();
        var q = _query.Forward(query);
        var attended = Attend(q, _key.Forward(text), _value.Forward(text));
        var imageAttended = Attend(q, _imageKey.Forward(image), _imageValue.Forward(image));
        var combined = Engine.TensorAdd(attended, Engine.TensorMultiplyScalar(imageAttended, NumOps.FromDouble(_imageWeight)));
        return _output.Forward(combined);
    }

    // softmax(q k^T / sqrt(d_head)) v per head, [B, S, Q] in and out.
    private Tensor<T> Attend(Tensor<T> q, Tensor<T> k, Tensor<T> v)
    {
        int batch = q.Shape[0], queryLength = q.Shape[1], contextLength = k.Shape[1];
        int headDimension = _queryDimension / _headCount;
        Tensor<T> Heads(Tensor<T> x, int length) => Engine.TensorPermute(
            Engine.Reshape(x, [batch, length, _headCount, headDimension]), [0, 2, 1, 3]);
        var attended = Engine.ScaledDotProductAttention(
            Heads(q, queryLength), Heads(k, contextLength), Heads(v, contextLength), mask: null, scale: null, out _);
        attended = Engine.TensorPermute(attended, [0, 2, 1, 3]).Contiguous();
        return Engine.Reshape(attended, [batch, queryLength, _queryDimension]);
    }

    /// <summary>
    /// Loads the official names under <paramref name="prefix"/>: <c>to_q</c>, <c>to_k</c>, <c>to_v</c>,
    /// <c>to_k_ip</c>, <c>to_v_ip</c> (bias-free) and <c>to_out.0</c>.
    /// </summary>
    internal IReadOnlyList<string> LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source, string prefix)
    {
        var engine = AiDotNetEngine.Current;
        TorchParametersLinear(engine, _query, _queryDimension, _queryDimension, source, prefix + "to_q.weight", null);
        TorchParametersLinear(engine, _key, _contextDimension, _queryDimension, source, prefix + "to_k.weight", null);
        TorchParametersLinear(engine, _value, _contextDimension, _queryDimension, source, prefix + "to_v.weight", null);
        TorchParametersLinear(engine, _imageKey, _contextDimension, _queryDimension, source, prefix + "to_k_ip.weight", null);
        TorchParametersLinear(engine, _imageValue, _contextDimension, _queryDimension, source, prefix + "to_v_ip.weight", null);
        TorchParametersLinear(engine, _output, _queryDimension, _queryDimension, source, prefix + "to_out.0.weight", prefix + "to_out.0.bias");
        return new[]
        {
            prefix + "to_q.weight", prefix + "to_k.weight", prefix + "to_v.weight", prefix + "to_k_ip.weight",
            prefix + "to_v_ip.weight", prefix + "to_out.0.weight", prefix + "to_out.0.bias"
        };
    }

    /// <summary>
    /// Loads a <c>weight [out, in]</c> (and a bias when named) into a dense layer; a bias-free layer has no bias to
    /// clear.
    /// </summary>
    internal static void TorchParametersLinear(IEngine engine, DenseLayer<T> layer, int inputs, int outputs,
        AiDotNet.ModelLoading.TorchTensorSource source, string weightName, string? biasName)
    {
        if (layer.UseBias)
        {
            AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(engine, layer, inputs, outputs, source, weightName, biasName);
            return;
        }
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            layer.Forward(new Tensor<T>(new[] { 1, inputs }));
        var weights = layer.GetWeights();
        source.CopyTransposedTo(weightName, new[] { outputs, inputs }, outputs, inputs, weights);
        engine.InvalidatePersistentTensor(weights);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in ParameterLayers()) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in ParameterLayers()) layer.SetTrainingMode(isTraining);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in ParameterLayers()) layer.ResetState();
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        metadata["QueryDimension"] = _queryDimension.ToString(invariant);
        metadata["ContextDimension"] = _contextDimension.ToString(invariant);
        metadata["HeadCount"] = _headCount.ToString(invariant);
        metadata["ImageTokens"] = _imageTokens.ToString(invariant);
        metadata["ImageWeight"] = _imageWeight.ToString("R", invariant);
        return metadata;
    }

    /// <summary>A lazily-allocated <c>nn.Linear</c> with PyTorch's default initialization.</summary>
    internal static DenseLayer<T> Linear(int inputs, int outputs, bool bias)
    {
        var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>(),
            new TorchDefaultInitializationStrategy<T>(LayerInitializationSeedScope.NextRandom()),
            bias ? BiasMode.Always : BiasMode.Never);
        layer.ResolveShapesOnly([inputs]);
        return layer;
    }

    private IEnumerable<ILayer<T>> ParameterLayers()
    {
        yield return _query;
        yield return _key;
        yield return _value;
        yield return _imageKey;
        yield return _imageValue;
        yield return _output;
    }
}
