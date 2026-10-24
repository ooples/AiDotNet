using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// StyleTTS 2's SLM discriminator head (§3.2.2; the reference's <c>WavLMDiscriminator</c>): the stacked hidden states of
/// every WavLM layer <c>[1, hidden · layers, frames]</c> projected by a 1×1 convolution, three 5-tap convolutions with
/// leaky ReLU (0.1), and a 3-tap output convolution, all weight-normalized; the score per frame.
/// </summary>
internal sealed class StyleTts2SlmDiscriminator<T>
{
    private readonly IEngine _engine;

    public StyleTts2SlmDiscriminator(IEngine engine, List<LayerBase<T>> layers, int slmHidden, int slmLayers, int channels)
    {
        _engine = engine;
        Pre = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(slmHidden * slmLayers, channels, 1));
        Convs.Add(StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(channels, channels * 2, 5)));
        Convs.Add(StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(channels * 2, channels * 4, 5)));
        Convs.Add(StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(channels * 4, channels * 4, 5)));
        Post = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(channels * 4, 1, 3));
    }

    public NormedConv1DLayer<T> Pre { get; }
    public List<NormedConv1DLayer<T>> Convs { get; } = new();
    public NormedConv1DLayer<T> Post { get; }

    /// <summary>The per-frame scores <c>[1, frames]</c> of stacked SLM hidden states <c>[1, hidden · layers, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> hidden)
    {
        var x = Pre.Forward(hidden);
        foreach (var conv in Convs) x = StyleTtsOps.LeakyRelu(_engine, conv.Forward(x), 0.1);
        var y = Post.Forward(x);
        return _engine.Reshape(y, new[] { 1, y.Shape[2] });
    }

    public void Load(TorchReader read, string name)
    {
        StyleTtsOps.LoadWeightNorm(Pre, read, name + ".pre");
        for (int i = 0; i < Convs.Count; i++) StyleTtsOps.LoadWeightNorm(Convs[i], read, $"{name}.convs.{i}");
        StyleTtsOps.LoadWeightNorm(Post, read, name + ".conv_post");
    }
}

public partial class StyleTTS2<T>
{
    private CosyVoiceHiFTDiscriminators<T>? _waveDiscriminators;
    private AiDotNet.Audio.Foundations.WavLmNetwork<T>? _slm;
    private readonly List<LayerBase<T>> _slmLayers = new();

    /// <summary>The frozen WavLM the SLM losses read (never trained).</summary>
    internal AiDotNet.Audio.Foundations.WavLmNetwork<T> SlmNetwork => _slm ?? throw NotNative();
    private StyleTts2SlmDiscriminator<T>? _slmDiscriminator;
    private readonly List<LayerBase<T>> _slmDiscriminatorLayers = new();

    /// <summary>The multi-period (periods 2, 3, 5, 7, 11) and multi-resolution spectrogram (1024/120/600, 2048/240/1200,
    /// 512/50/240) discriminators — the reference's <c>mpd</c> and <c>msd</c>, which are CosyVoice's HiFT discriminators.</summary>
    internal CosyVoiceHiFTDiscriminators<T> WaveDiscriminators => _waveDiscriminators ?? throw NotNative();

    internal StyleTts2SlmDiscriminator<T> SlmDiscriminator => _slmDiscriminator ?? throw NotNative();

    private IEnumerable<LayerBase<T>> DiscriminatorLayerList =>
        (_waveDiscriminators?.Layers ?? Array.Empty<LayerBase<T>>()).Concat(_slmDiscriminatorLayers).Concat(_slmLayers);

    private void InitializeDiscriminators()
    {
        var o = _options;
        _waveDiscriminators = new CosyVoiceHiFTDiscriminators<T>(Engine);
        _slmDiscriminator = new StyleTts2SlmDiscriminator<T>(Engine, _slmDiscriminatorLayers, o.SlmHidden, o.SlmLayers, o.SlmDiscriminatorChannels);
        _slm = new AiDotNet.Audio.Foundations.WavLmNetwork<T>(Engine, _slmLayers, o.SlmConfiguration);
    }

    private void AddDiscriminatorLayers() => ComponentLayers.AddRange(DiscriminatorLayerList);

    private void MaterializeDiscriminators()
    {
        var o = _options;
        _waveDiscriminators!.Forward(new Tensor<T>(new[] { 2 * o.HopSize }));
        _slmDiscriminator!.Forward(new Tensor<T>(new[] { 1, o.SlmHidden * o.SlmLayers, 4 }));
        _slm!.HiddenStates(new Tensor<T>(new[] { 800 }));
    }

    /// <summary>
    /// Loads WavLM (a transformers <c>WavLMModel</c> state dict such as <c>microsoft/wavlm-base-plus</c>, safetensors or
    /// <c>.pt</c>; the position convolution's weight norm under either its old <c>weight_g</c>/<c>weight_v</c> names or the
    /// parametrized ones).
    /// </summary>
    public void LoadSlmWeights(string path)
    {
        var names = new HashSet<string>(new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path).Keys, StringComparer.Ordinal);
        var read = StyleTTS<T>.Reader(path, "wavlm.");
        SlmNetwork.LoadTorchWeights((name, shape) => read(name, shape), name => names.Contains(name) || names.Contains("wavlm." + name));
    }

    /// <summary>Loads the reference's <c>mpd</c>, <c>msd</c> and <c>wd</c> state (old-style <c>weight_g</c>/<c>weight_v</c>
    /// or parametrized names).</summary>
    private void LoadDiscriminators(TorchReader read, IReadOnlyDictionary<string, double[]> waveState)
    {
        // The reference names the spectrogram discriminators msd; CosyVoice's loader reads them as mrd.
        var renamed = waveState.ToDictionary(kv => kv.Key.StartsWith("msd.", StringComparison.Ordinal) ? "mrd." + kv.Key.Substring(4) : kv.Key,
            kv => kv.Value);
        WaveDiscriminators.LoadTorch(renamed);
        SlmDiscriminator.Load(read, "wd");
    }
}
