using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Grounding DINO as a detector a model runs: the network its options describe, the official image normalization,
/// detection (<c>predict</c>), and the paper's loss on grounded phrases. Grounding DINO and Grounded SAM 2 both run it.
/// </summary>
internal sealed class GroundingDinoDetector<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public GroundingDinoDetector(IEngine engine, List<LayerBase<T>> layers, GroundingDINOOptions options, int seed)
    {
        if (options.VocabSize < 30522)
            throw new ArgumentException("Grounding DINO reads BERT-base-uncased token ids; VocabSize must be at least 30522.", nameof(options));
        _engine = engine;
        Options = options;
        Network = new GroundingDinoNetwork<T>(engine, layers, Configuration(options));
        Network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        Network.SeedRandom(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1));
    }

    public GroundingDINOOptions Options { get; }

    public GroundingDinoNetwork<T> Network { get; }

    /// <summary>The network configuration the options describe.</summary>
    public static GroundingDinoConfiguration Configuration(GroundingDINOOptions o) => new()
    {
        Backbone = new SwinConfiguration
        {
            EmbedDim = o.VisionDim, Depths = (int[])o.SwinDepths.Clone(), Heads = (int[])o.SwinHeads.Clone(),
            WindowSize = o.SwinWindowSize, DropPathRate = o.SwinDropPathRate,
        },
        Text = new BertConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.TextHiddenSize, Layers = o.TextLayers, Heads = o.TextHeads,
            Intermediate = o.TextIntermediateSize,
        },
        Dim = o.DecoderDim, EncoderLayers = o.NumVisionLayers, DecoderLayers = o.NumDecoderLayers, EncoderHeads = o.NumHeads,
        DecoderHeads = o.NumHeads, EncoderFfn = o.FeedForwardDim, DecoderFfn = o.FeedForwardDim, Levels = o.NumFeatureLevels,
        EncoderPoints = o.NumSamplingPoints, DecoderPoints = o.NumSamplingPoints, Queries = o.NumQueryPositions,
        MaxTextLength = o.MaxSequenceLength, Dropout = o.DropoutRate, FusionDropPath = o.FusionDropPathRate,
    };

    /// <summary>An RGB image <c>[3, H, W]</c> as a batch of one, or a batch <c>[B, 3, H, W]</c> as given.</summary>
    public static Tensor<T> Batched(IEngine engine, Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (image.Rank == 3) image = engine.Reshape(image, new[] { 1, image.Shape[0], image.Shape[1], image.Shape[2] });
        if (image.Rank != 4 || image.Shape[1] != 3)
            throw new ArgumentException($"Expected RGB images [3, H, W] or [B, 3, H, W]; got [{string.Join(", ", image.Shape.ToArray())}].", nameof(image));
        return image;
    }

    /// <summary>The official transform: images in 0..1 normalized by the options' mean and deviation (ImageNet's).</summary>
    public Tensor<T> Normalize(Tensor<T> images)
    {
        var mean = new Tensor<T>(new[] { 1, 3, 1, 1 });
        var scale = new Tensor<T>(new[] { 1, 3, 1, 1 });
        for (int c = 0; c < 3; c++)
        {
            mean[0, c, 0, 0] = NumOps.FromDouble(Options.ImageMean[c]);
            scale[0, c, 0, 0] = NumOps.FromDouble(1.0 / Options.ImageStd[c]);
        }
        var shape = images._shape;
        return _engine.TensorMultiply(_engine.TensorSubtract(images, _engine.TensorBroadcastTo(mean, shape)), _engine.TensorBroadcastTo(scale, shape));
    }

    /// <summary>The image tokens before they meet the text, every level flattened and concatenated, <c>[B, tokens, Dim]</c>.</summary>
    public Tensor<T> ImageTokens(Tensor<T> images)
    {
        var maps = Network.ProjectImage(Normalize(images));
        int b = maps[0].Shape[0], d = maps[0].Shape[1];
        return _engine.TensorConcatenate(
            maps.Select(m => _engine.TensorPermute(_engine.Reshape(m, new[] { b, d, m.Shape[2] * m.Shape[3] }), new[] { 0, 2, 1 })).ToArray(), 1);
    }

    /// <summary>The official <c>predict</c> on one image (a batch of one) and a caption.</summary>
    public List<GroundingDetection> Ground(Tensor<T> image, string caption)
    {
        if (image.Shape[0] != 1) throw new ArgumentException("Grounding reads one image at a time.", nameof(image));
        var prompt = GroundingDinoPrompt.Create(caption);
        GroundingDinoOutput<T> output;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            output = Network.Forward(Normalize(image), prompt.IdMatrix, prompt.Mask);
        return prompt.Detections(output, 0, Options.ConfidenceThreshold, Options.TextThreshold);
    }

    /// <summary>The <see cref="GroundingVLMOptions.MaxDetections"/> highest-scoring detections, in query order.</summary>
    public List<GroundingDetection> Strongest(IReadOnlyList<GroundingDetection> detections)
    {
        if (detections.Count <= Options.MaxDetections) return detections.ToList();
        var keep = new HashSet<GroundingDetection>(detections.OrderByDescending(d => d.Score).Take(Options.MaxDetections));
        return detections.Where(keep.Contains).ToList();
    }

    /// <summary>The caption and its boxes as the network and the loss read them.</summary>
    public static (GroundingDinoPrompt Prompt, GroundingTarget Target) Supervision(string caption, IReadOnlyList<GroundingBox> boxes)
    {
        if (boxes is null) throw new ArgumentNullException(nameof(boxes));
        var prompt = GroundingDinoPrompt.Create(caption);
        var targets = new double[boxes.Count][];
        var phrases = new int[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
        {
            var box = boxes[i];
            if (!(box.X2 > box.X1 && box.Y2 > box.Y1))
                throw new ArgumentException($"Box {i} (\"{box.Phrase}\") has no area.", nameof(boxes));
            targets[i] = new[] { (box.X1 + box.X2) / 2, (box.Y1 + box.Y2) / 2, box.X2 - box.X1, box.Y2 - box.Y1 };
            phrases[i] = prompt.PhraseIndex(box.Phrase);
        }
        return (prompt, new GroundingTarget(targets, phrases));
    }

    /// <summary>The paper's loss on one image (a batch of one), recorded on the tape.</summary>
    public Tensor<T> Loss(Tensor<T> image, GroundingDinoPrompt prompt, GroundingTarget target)
    {
        if (image.Shape[0] != 1) throw new ArgumentException("A grounding step reads one image and its caption.", nameof(image));
        var ids = prompt.IdMatrix;
        var mask = prompt.Mask;
        return new GroundingDinoLoss<T>(_engine).Compute(Network.Forward(Normalize(image), ids, mask), ids, mask, new[] { target });
    }
}
