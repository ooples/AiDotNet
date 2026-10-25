using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>The shape of KaniTTS 2's speaker embedder; the defaults are <c>nineninesix/speaker-emb-tbr</c>
/// (Orange/Speaker-wavLM-tbr: WavLM-Large and a 2048 → 512 → 128 head).</summary>
internal sealed record KaniSpeakerEmbedderConfiguration
{
    public WavLmConfiguration WavLm { get; init; } = new()
    {
        HiddenSize = 1024, Layers = 24, Heads = 16, IntermediateSize = 4096, LayerNormFeatureEncoder = true, StableLayerNorm = true,
    };

    /// <summary>The head's middle width (<c>top_interm_size</c>).</summary>
    public int IntermediateSize { get; init; } = 512;

    /// <summary>The embedding width (<c>embd_size</c>).</summary>
    public int EmbeddingSize { get; init; } = 128;

    /// <summary>The longest reference, in seconds, at 16 kHz (<c>max_duration_sec</c>, 30); longer audio is cut.</summary>
    public double MaxSeconds { get; init; } = 30;
}

/// <summary>
/// KaniTTS 2's speaker embedder (kani_tts/speaker_embedder.py's <c>EmbeddingsModel</c>, after Orange's
/// spk_embeddings.py): the 16 kHz waveform normalized to zero mean and unit (unbiased) standard deviation, WavLM's last
/// hidden state, its mean and (unbiased) standard deviation over time, then a 1×1 convolution, ReLU and a non-affine batch
/// norm twice (2048 → 512 → 128), and L2 normalization.
/// </summary>
/// <remarks>Frozen: evaluation-mode batch norms with the checkpoint's running statistics. Parameter names follow the
/// checkpoint (<c>wavlm.*</c>, <c>top_layers.*</c>).</remarks>
internal sealed class KaniSpeakerEmbedder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public KaniSpeakerEmbedder(IEngine engine, List<LayerBase<T>> layers, KaniSpeakerEmbedderConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        WavLm = new WavLmNetwork<T>(engine, layers, configuration.WavLm);
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        int pooled = 2 * configuration.WavLm.HiddenSize;
        First = Own(new DenseLayer<T>(configuration.IntermediateSize, (IActivationFunction<T>)new IdentityActivation<T>()));
        Second = Own(new DenseLayer<T>(configuration.EmbeddingSize, (IActivationFunction<T>)new IdentityActivation<T>()));
        FirstMean = Own(new BiasFreeLinearLayer<T>(1, configuration.IntermediateSize));
        FirstVariance = Own(new BiasFreeLinearLayer<T>(1, configuration.IntermediateSize));
        SecondMean = Own(new BiasFreeLinearLayer<T>(1, configuration.EmbeddingSize));
        SecondVariance = Own(new BiasFreeLinearLayer<T>(1, configuration.EmbeddingSize));
        _pooled = pooled;
    }

    private readonly int _pooled;

    public KaniSpeakerEmbedderConfiguration Configuration { get; }
    public WavLmNetwork<T> WavLm { get; }
    public DenseLayer<T> First { get; }
    public DenseLayer<T> Second { get; }

    /// <summary>The batch norms' running statistics (frozen buffers).</summary>
    public BiasFreeLinearLayer<T> FirstMean { get; }
    public BiasFreeLinearLayer<T> FirstVariance { get; }
    public BiasFreeLinearLayer<T> SecondMean { get; }
    public BiasFreeLinearLayer<T> SecondVariance { get; }

    /// <summary>The layers that hold the embedder's weights and statistics, all frozen.</summary>
    internal IEnumerable<BiasFreeLinearLayer<T>> Statistics => new[] { FirstMean, FirstVariance, SecondMean, SecondVariance };

    /// <summary>The L2-normalized speaker embedding <c>[EmbeddingSize]</c> of a 16 kHz waveform.</summary>
    public Tensor<T> Embed(Tensor<T> audio16k)
    {
        int max = (int)(Configuration.MaxSeconds * 16000);
        var audio = audio16k.Length > max ? _engine.TensorSlice(_engine.Reshape(audio16k, new[] { audio16k.Length }), new[] { 0 }, new[] { max })
            : _engine.Reshape(audio16k, new[] { audio16k.Length });
        int n = audio.Length;
        if (n < 2) throw new ArgumentException("The reference is too short.", nameof(audio16k));
        // MVN with torch.std's unbiased estimate.
        var mean = _engine.ReduceMean(audio, new[] { 0 }, keepDims: true);
        var centered = _engine.TensorSubtract(audio, _engine.TensorBroadcastTo(mean, new[] { n }));
        var std = _engine.TensorPower(_engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(centered, centered), new[] { 0 }, keepDims: true),
            NumOps.FromDouble(1.0 / (n - 1))), NumOps.FromDouble(0.5));
        var normed = _engine.TensorDivide(centered, _engine.TensorBroadcastTo(_engine.TensorAddScalar(std, NumOps.FromDouble(1e-10)), new[] { n }));
        var hidden = WavLm.HiddenStates(normed)[^1];                                                          // [frames, H]
        int frames = hidden.Shape[0], width = hidden.Shape[1];
        if (frames < 2) throw new ArgumentException("The reference is too short for a standard deviation over time.", nameof(audio16k));
        // Stats pooling: the mean and the unbiased variance (at least 1e-10), its square root.
        var frameMean = _engine.ReduceMean(hidden, new[] { 0 }, keepDims: true);                              // [1, H]
        var deviation = _engine.TensorSubtract(hidden, _engine.TensorBroadcastTo(frameMean, new[] { frames, width }));
        var variance = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(deviation, deviation), new[] { 0 }, keepDims: true),
            NumOps.FromDouble(1.0 / (frames - 1)));
        var spread = _engine.TensorPower(_engine.TensorClampMin(variance, NumOps.FromDouble(1e-10)), NumOps.FromDouble(0.5));
        var stats = _engine.TensorConcatenate(new[] { frameMean, spread }, 1);                                   // [1, 2H]
        var x = BatchNorm(_engine.ReLU(First.Forward(stats)), FirstMean, FirstVariance);
        x = BatchNorm(_engine.ReLU(Second.Forward(x)), SecondMean, SecondVariance);
        // F.normalize: x / max(‖x‖, 1e-12).
        var norm = _engine.TensorPower(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { 1 }, keepDims: true), NumOps.FromDouble(0.5));
        var scaled = _engine.TensorDivide(x, _engine.TensorBroadcastTo(_engine.TensorClampMin(norm, NumOps.FromDouble(1e-12)), x.Shape.ToArray()));
        return _engine.Reshape(scaled, new[] { Configuration.EmbeddingSize });
    }

    // BatchNorm1d(affine=False, eps=1e-3) in evaluation mode.
    private Tensor<T> BatchNorm(Tensor<T> x, BiasFreeLinearLayer<T> mean, BiasFreeLinearLayer<T> variance)
    {
        var shape = x.Shape.ToArray();
        var m = _engine.TensorBroadcastTo(_engine.Reshape(Lfm2Network<T>.Paged(mean.Weights), new[] { 1, shape[1] }), shape);
        var scale = _engine.TensorPower(_engine.TensorAddScalar(_engine.Reshape(Lfm2Network<T>.Paged(variance.Weights), new[] { 1, shape[1] }), NumOps.FromDouble(1e-3)),
            NumOps.FromDouble(-0.5));
        return _engine.TensorMultiply(_engine.TensorSubtract(x, m), _engine.TensorBroadcastTo(scale, shape));
    }

    /// <summary>Loads the embedder's checkpoint (<c>wavlm.*</c>, <c>top_layers.*</c>).</summary>
    public void LoadTorchWeights(Func<string, double[]?> read)
    {
        double[] Need(string name) => read(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        WavLm.LoadTorchWeights((name, shape) => Need("wavlm." + name), name => read("wavlm." + name) is not null);
        var c = Configuration;
        TorchParameters.Linear(_engine, First, _pooled, c.IntermediateSize, Need("top_layers.affine1.weight"), Need("top_layers.affine1.bias"));
        TorchParameters.Linear(_engine, Second, c.IntermediateSize, c.EmbeddingSize, Need("top_layers.affine2.weight"), Need("top_layers.affine2.bias"));
        TorchParameters.Linear(FirstMean, 1, c.IntermediateSize, Need("top_layers.batchnorm1.running_mean"));
        TorchParameters.Linear(FirstVariance, 1, c.IntermediateSize, Need("top_layers.batchnorm1.running_var"));
        TorchParameters.Linear(SecondMean, 1, c.EmbeddingSize, Need("top_layers.batchnorm2.running_mean"));
        TorchParameters.Linear(SecondVariance, 1, c.EmbeddingSize, Need("top_layers.batchnorm2.running_var"));
    }
}
