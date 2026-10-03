using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.LanguageIdentification;

/// <summary>
/// An ECAPA-TDNN encoder with a language-classification head: BatchNorm over the embedding and a
/// linear layer to one logit per language. ECAPATDNNLanguageIdentifier and VoxLingua107Identifier
/// both run exactly this network, so it lives here once.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class EcapaTdnnLanguageClassifier<T>
{
    private readonly EcapaTdnnBackbone<T> _backbone;
    private BatchNormalizationLayer<T> _embeddingNorm;
    private DenseLayer<T> _classifier;
    private readonly List<ILayer<T>> _layers;

    /// <summary>Builds the network from an ECAPA-TDNN options object.</summary>
    /// <param name="options">The stage widths, kernels, dilations and bottlenecks.</param>
    /// <param name="numLanguages">Number of output logits.</param>
    public EcapaTdnnLanguageClassifier(ECAPATDNNOptions options, int numLanguages)
        : this(
            options.TdnnChannels,
            options.KernelSizes,
            options.Dilations,
            options.Res2NetScale,
            options.SeChannels,
            options.AttentionChannels,
            options.EmbeddingDimension,
            numLanguages)
    {
    }

    /// <summary>Builds the network from explicit hyper-parameters.</summary>
    public EcapaTdnnLanguageClassifier(
        int tdnnChannels,
        int[] kernelSizes,
        int[] dilations,
        int res2NetScale,
        int seChannels,
        int attentionChannels,
        int embeddingDimension,
        int numLanguages)
    {
        if (tdnnChannels <= 0) throw new ArgumentOutOfRangeException(nameof(tdnnChannels));
        if (dilations is null) throw new ArgumentNullException(nameof(dilations));
        if (numLanguages <= 0) throw new ArgumentOutOfRangeException(nameof(numLanguages));
        if (dilations.Length < 3)
            throw new ArgumentException(
                "Dilations needs the frame-level TDNN block, at least one SE-Res2Block and the MFA convolution.",
                nameof(dilations));

        // Every stage is TdnnChannels wide except the MFA convolution, which keeps the width of the
        // concatenated SE-Res2Block outputs (3 x 1024 = 3072 in the paper).
        int blocks = dilations.Length - 2;
        var channels = new int[dilations.Length];
        for (int i = 0; i < channels.Length - 1; i++) channels[i] = tdnnChannels;
        channels[channels.Length - 1] = tdnnChannels * blocks;

        _backbone = new EcapaTdnnBackbone<T>(
            channels, kernelSizes, dilations, res2NetScale, seChannels, attentionChannels, embeddingDimension);
        _embeddingNorm = new BatchNormalizationLayer<T>();
        _classifier = new DenseLayer<T>(numLanguages, (IActivationFunction<T>)new IdentityActivation<T>());

        _layers = new List<ILayer<T>>(_backbone.Layers) { _embeddingNorm, _classifier };
    }

    /// <summary>
    /// Frames, beyond the first analysis window, in the clip that resolves the network's lazy shapes:
    /// enough for every convolution's kernel and the attentive pooling to see more than one frame.
    /// </summary>
    internal const int ProbeExtraFrames = 8;

    /// <summary>The shortest raw clip whose MFCC features exercise the whole network.</summary>
    internal static Tensor<T> CreateProbeClip(LanguageIdentifierOptions options)
        => new Tensor<T>(new[] { options.FftSize + ProbeExtraFrames * options.HopLength });

    /// <summary>Every layer of the network, in a fixed order, for the owning model to publish.</summary>
    public IReadOnlyList<ILayer<T>> Layers => _layers;

    /// <summary>
    /// Points the network at the model's current layer graph after a deserialize or eager clone
    /// replaced its instances; does nothing while the graph is unchanged.
    /// </summary>
    public void BindTo(IReadOnlyList<ILayer<T>> layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        _backbone.BindTo(layers);
        int encoderCount = _backbone.Layers.Count;
        if (layers.Count != encoderCount + 2
            || layers[encoderCount] is not BatchNormalizationLayer<T> embeddingNorm
            || layers[encoderCount + 1] is not DenseLayer<T> classifier)
        {
            throw new InvalidOperationException(
                "The layer graph does not match the ECAPA-TDNN language classifier layout: expected the " +
                $"{encoderCount} encoder layers, a BatchNormalizationLayer and a DenseLayer, found {layers.Count} layers.");
        }

        bool same = _layers.Count == layers.Count;
        for (int i = 0; same && i < layers.Count; i++)
        {
            same = ReferenceEquals(_layers[i], layers[i]);
        }

        if (same) return;

        _embeddingNorm = embeddingNorm;
        _classifier = classifier;
        _layers.Clear();
        _layers.AddRange(layers);
    }

    /// <summary>
    /// Maps time-major features <c>[T, F]</c> to logits <c>[numLanguages]</c>, or a batch
    /// <c>[B, T, F]</c> to <c>[B, numLanguages]</c>.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> timeMajorFeatures)
    {
        bool unbatched = timeMajorFeatures.Shape.Length == 2;
        var embedding = _backbone.Forward(EcapaTdnnBackbone<T>.ToChannelFirst(timeMajorFeatures));
        var logits = _classifier.Forward(_embeddingNorm.Forward(embedding));
        return unbatched
            ? AiDotNetEngine.Current.Reshape(logits, new[] { logits.Shape[logits.Shape.Length - 1] })
            : logits;
    }
}