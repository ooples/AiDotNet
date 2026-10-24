using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ComputerVision.Segmentation.Mamba;

/// <summary>The sizes of a Vision Mamba segmenter (the official <c>upernet_vim_tiny_24_512_slide_60k</c> by default,
/// apart from the image size, which the owner sets).</summary>
internal sealed record VimSegmentationConfiguration
{
    public VimConfiguration Backbone { get; init; } = new() { ClassToken = false, ImageHeight = 512, ImageWidth = 512 };

    /// <summary>The layers whose mixer outputs feed the neck.</summary>
    public int[] OutIndices { get; init; } = { 5, 11, 17, 23 };

    public int Classes { get; init; } = 150;

    /// <summary>The decode head's width (the backbone's for Vim-Tiny).</summary>
    public int HeadChannels { get; init; } = 192;

    public int AuxiliaryChannels { get; init; } = 256;
    public int[] PoolScales { get; init; } = { 1, 2, 3, 6 };
    public double HeadDropout { get; init; } = 0.1;

    /// <summary>The auxiliary head's weight in the training loss.</summary>
    public double AuxiliaryWeight { get; init; } = 0.4;
}

/// <summary>
/// Vision Mamba for semantic segmentation as the official code assembles it (hustvl/Vim <c>seg/</c>): the Vim
/// backbone without a class token, the mixer outputs of four layers reshaped to maps, a BEiT-style multi-level neck
/// that turns them into strides 4, 8, 16 and 32, mmsegmentation's UPerNet decode head, and an FCN auxiliary head on
/// the third map used only for the training loss.
/// </summary>
/// <remarks>
/// For 16-pixel patches the neck is: two 2×2 stride-2 transposed convolutions with BatchNorm and GELU between them
/// (×4), one transposed convolution (×2), the identity, and a 2×2 max-pool. For 8-pixel patches it is one transposed
/// convolution, the identity, and 2×2 and 4×4 max-pools.
/// </remarks>
internal sealed class VimSegmentationNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _owned = new();

    public VimSegmentationNetwork(IEngine engine, List<LayerBase<T>> layers, VimSegmentationConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Backbone.ClassToken) throw new ArgumentException("The segmentation backbone has no class token.");
        if (c.Backbone.PatchSize is not (16 or 8)) throw new ArgumentException("The neck is defined for 16- and 8-pixel patches.");
        if (c.OutIndices.Length != 4) throw new ArgumentException("The neck takes four layer outputs.");
        var sink = new List<LayerBase<T>>();
        Backbone = new VimNetwork<T>(engine, sink, c.Backbone);
        int d = c.Backbone.Dim;
        LayerBase<T> Own(LayerBase<T> layer)
        {
            sink.Add(layer);
            return layer;
        }
        TransposedConvolution2x2<T> Up() => new(engine, Own, d, d);
        if (c.Backbone.PatchSize == 16)
        {
            Up1 = Up();
            UpNorm = (BatchNormalizationLayer<T>)Own(new BatchNormalizationLayer<T>(d, 1e-5, 0.9));
            Up2 = Up();
            Up3 = Up();
        }
        else
        {
            Up1 = Up();
        }
        Head = new UPerNetHead<T>(engine, sink, Enumerable.Repeat(d, 4).ToArray(), c.HeadChannels, c.Classes, c.PoolScales, c.HeadDropout);
        Auxiliary = new FcnHead<T>(engine, sink, d, c.AuxiliaryChannels, c.Classes, c.HeadDropout);
        layers.AddRange(sink);
        _owned.AddRange(sink);
    }

    public VimSegmentationConfiguration Configuration { get; }
    public VimNetwork<T> Backbone { get; }

    /// <summary>The first neck branch's (first) transposed convolution.</summary>
    public TransposedConvolution2x2<T> Up1 { get; }
    public BatchNormalizationLayer<T>? UpNorm { get; }

    /// <summary>The first branch's second transposed convolution (16-pixel patches).</summary>
    public TransposedConvolution2x2<T>? Up2 { get; }

    /// <summary>The second branch's transposed convolution (16-pixel patches).</summary>
    public TransposedConvolution2x2<T>? Up3 { get; }

    public UPerNetHead<T> Head { get; }
    public FcnHead<T> Auxiliary { get; }

    /// <summary>The layers this network added to its owner's list, in order.</summary>
    public IReadOnlyList<LayerBase<T>> OwnedLayers => _owned;

    /// <summary>The neck's four maps, finest first.</summary>
    public List<Tensor<T>> Features(Tensor<T> images, bool training, Random random)
    {
        var maps = Backbone.LayerOutputs(images, Configuration.OutIndices, training, random);
        if (Configuration.Backbone.PatchSize == 16)
        {
            var up = Up1.Forward(maps[0]);
            up = ActivationHelper.ExactGelu(UpNorm!.Forward(up), _engine);
            return new List<Tensor<T>> { Up2!.Forward(up), Up3!.Forward(maps[1]), maps[2], CvTensorOps<T>.MaxPoolPadded(maps[3], 2, 2, 0) };
        }
        return new List<Tensor<T>>
        {
            Up1.Forward(maps[0]), maps[1], CvTensorOps<T>.MaxPoolPadded(maps[2], 2, 2, 0), CvTensorOps<T>.MaxPoolPadded(maps[3], 4, 4, 0),
        };
    }

    /// <summary>The decode logits and, for training, the auxiliary logits, both resized to the image.</summary>
    public (Tensor<T> Logits, Tensor<T>? Auxiliary) Forward(Tensor<T> images, bool training, Random random, bool withAuxiliary)
    {
        int h = images.Shape[2], w = images.Shape[3];
        var features = Features(images, training, random);
        var logits = SpatialResampling.Resize(_engine, Head.Forward(features, training, random), h, w);
        var auxiliary = withAuxiliary ? SpatialResampling.Resize(_engine, Auxiliary.Forward(features[2], training, random), h, w) : null;
        return (logits, auxiliary);
    }

    /// <summary>Loads the official model's state dict: <c>backbone.*</c> (with the neck's <c>fpn*</c>),
    /// <c>decode_head.*</c> and <c>auxiliary_head.*</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        int d = Configuration.Backbone.Dim;
        Backbone.LoadTorchWeights(read, "backbone.");
        var w = new[] { d, d, 2, 2 };
        var b = new[] { d };
        if (Configuration.Backbone.PatchSize == 16)
        {
            Up1.Load(read("backbone.fpn1.0.weight", w), read("backbone.fpn1.0.bias", b));
            TorchParameters.BatchNorm(UpNorm!, read("backbone.fpn1.1.weight", b), read("backbone.fpn1.1.bias", b),
                read("backbone.fpn1.1.running_mean", b), read("backbone.fpn1.1.running_var", b));
            Up2!.Load(read("backbone.fpn1.3.weight", w), read("backbone.fpn1.3.bias", b));
            Up3!.Load(read("backbone.fpn2.0.weight", w), read("backbone.fpn2.0.bias", b));
        }
        else
        {
            Up1.Load(read("backbone.fpn1.0.weight", w), read("backbone.fpn1.0.bias", b));
        }
        Head.Load(read, "decode_head");
        Auxiliary.Load(read, "auxiliary_head");
    }

    /// <summary>
    /// The official initialization: the Vim backbone's, then (VisionMambaSeg.init_weights) every linear layer
    /// re-drawn from N(0, 0.02²) truncated at ±2 with a zero bias; the neck's transposed convolutions keep PyTorch's
    /// default (U(±1/√fan-in), fan-in = out·4) and its BatchNorm scale 1 and shift 0; the heads mmseg's.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        int d = Configuration.Backbone.Dim;
        Backbone.InitializeForSegmentation(random);
        foreach (var deconvolution in new[] { Up1, Up2, Up3 }.Where(x => x is not null)) deconvolution!.Initialize(random);
        if (UpNorm is not null)
            TorchParameters.BatchNorm(UpNorm, Enumerable.Repeat(1.0, d).ToArray(), new double[d], new double[d], Enumerable.Repeat(1.0, d).ToArray());
        Head.Initialize(random);
        Auxiliary.Initialize(random);
    }
}
