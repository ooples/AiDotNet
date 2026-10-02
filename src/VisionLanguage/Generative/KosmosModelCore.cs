using AiDotNet.ComputerVision;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// The model logic KOSMOS-1 and KOSMOS-2 share: a CLIP ViT, an image resampler and a MAGNETO decoder. The
/// sequence layout is <c>&lt;s&gt; &lt;image&gt; [K image embeddings] &lt;/image&gt; text...</c>, and training is
/// next-token cross-entropy.
/// </summary>
internal sealed class KosmosModelCore<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly KosmosOptions _options;

    public KosmosModelCore(KosmosOptions options, bool perceiver, int resamplerDepth, KosmosPositionEncoding positions)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (options.VocabSize < 4) throw new ArgumentException("KOSMOS needs a vocabulary of at least 4 tokens.", nameof(options));
        Vision = new ClipVisionTransformerLayer<T>(options.VisionDim, options.NumVisionLayers, options.VisionHeads, options.PatchSize, options.ImageSize);
        Resampler = new KosmosImageResamplerLayer<T>(options.VisionDim, options.DecoderDim, options.NumImageTokens,
            options.NumHeads, resamplerDepth, perceiver);
        Decoder = new MagnetoDecoderLayer<T>(options.VocabSize, options.DecoderDim, options.NumDecoderLayers, options.NumHeads,
            options.DecoderFeedForwardDim, HeaderLength + options.MaxSequenceLength + options.MaxGenerationLength, positions);
    }

    public ClipVisionTransformerLayer<T> Vision { get; }
    public KosmosImageResamplerLayer<T> Resampler { get; }
    public MagnetoDecoderLayer<T> Decoder { get; }

    public IEnumerable<ILayer<T>> Layers()
    {
        yield return Vision;
        yield return Resampler;
        // The decoder reads the token sequence; the resampler's output enters it through the image slots. It is not a
        // step of this sequence, so a model registers it as an auxiliary stream (see DecoderStream).
    }

    /// <summary>
    /// The decoder as an auxiliary stream: it owns the token embeddings and every MAGNETO block, which must be trained,
    /// counted and checkpointed like the layers above, but it is not a sequential step after the resampler.
    /// </summary>
    public List<ILayer<T>> DecoderStream() => new() { Decoder };

    /// <summary><c>&lt;s&gt; &lt;image&gt; [K slots] &lt;/image&gt;</c>.</summary>
    public int HeaderLength => _options.NumImageTokens + 3;

    private List<int> Header()
    {
        var ids = new List<int> { _options.BosTokenId, _options.ResolvedImageStartTokenId };
        for (int i = 0; i < _options.NumImageTokens; i++) ids.Add(_options.ResolvedImageStartTokenId);
        ids.Add(_options.ResolvedImageEndTokenId);
        return ids;
    }

    /// <summary>The <c>K</c> image embeddings of a preprocessed image, <c>[K, d]</c>.</summary>
    public Tensor<T> ImageEmbeddings(Tensor<T> image) => Resampler.Forward(Vision.Forward(image));

    /// <summary>Next-token logits <c>[HeaderLength + prompt, vocab]</c> for a preprocessed image and prompt ids.</summary>
    public Tensor<T> Logits(Tensor<T> image, IReadOnlyList<int> prompt)
    {
        var ids = Header();
        ids.AddRange(prompt.Take(_options.MaxSequenceLength).Select(Clamp));
        return Decoder.Forward(ids.ToArray(), ImageEmbeddings(image), 2);
    }

    private int Clamp(int id) => Math.Min(Math.Max(id, 0), _options.VocabSize - 1);

    /// <summary>
    /// The training objective for a preprocessed image and a target:
    /// <list type="bullet">
    /// <item>Token ids <c>[T]</c> are a caption. The loss is the mean next-token cross-entropy of the caption
    /// after the image block.</item>
    /// <item>A <c>[HeaderLength, vocab]</c> target is a soft distribution per position of the image-only
    /// layout. The loss is the mean soft cross-entropy <c>-sum(t log softmax(logits))</c>.</item>
    /// </list>
    /// </summary>
    public Tensor<T> Loss(Tensor<T> image, Tensor<T> target)
    {
        if (target.Rank == 2 && target.Shape[0] == HeaderLength && target.Shape[1] == _options.VocabSize)
            return NeuralNetworkBase<T>.SoftTargetCrossEntropy(Logits(image, Array.Empty<int>()), target, 1.0 / HeaderLength);

        var caption = NeuralNetworkBase<T>.TokenIds(target, Clamp, _options.MaxSequenceLength);
        if (caption.Length == 0) throw new ArgumentException("A KOSMOS caption needs at least one token.", nameof(target));
        // The logit at position HeaderLength - 1 + t predicts caption token t.
        return NeuralNetworkBase<T>.TokenCrossEntropy(Logits(image, caption), caption, HeaderLength - 1);
    }

    /// <summary>Greedy decoding: appends the most likely token until EOS or <c>MaxGenerationLength</c> tokens.</summary>
    public Tensor<T> Generate(Tensor<T> image, IReadOnlyList<int> prompt)
    {
        var embeddings = ImageEmbeddings(image);
        var ids = Header();
        ids.AddRange(prompt.Take(_options.MaxSequenceLength).Select(Clamp));
        var generated = NeuralNetworkBase<T>.GreedyDecode(
            context => Decoder.Forward(context.ToArray(), embeddings, 2), ids, _options.MaxGenerationLength,
            (token, _) => token == _options.EosTokenId);
        return NeuralNetworkBase<T>.TokenTensor(generated);
    }
}
