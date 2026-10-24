using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The shape of Grounding DINO; the defaults are Grounding DINO-T (Swin-T and BERT-base).</summary>
internal sealed record GroundingDinoConfiguration
{
    /// <summary>Swin-T, with the official backbone's stochastic depth (0.2, SwinTransformer's default, which build_swin_transformer keeps).</summary>
    public SwinConfiguration Backbone { get; init; } = new() { DropPathRate = 0.2 };

    public BertConfiguration Text { get; init; } = new();

    public int Dim { get; init; } = 256;

    public int EncoderLayers { get; init; } = 6;

    public int DecoderLayers { get; init; } = 6;

    public int EncoderHeads { get; init; } = 8;

    public int DecoderHeads { get; init; } = 8;

    public int EncoderFfn { get; init; } = 2048;

    public int DecoderFfn { get; init; } = 2048;

    /// <summary>Feature levels: the backbone's outputs plus stride-2 convolutions of the last (4).</summary>
    public int Levels { get; init; } = 4;

    public int EncoderPoints { get; init; } = 4;

    public int DecoderPoints { get; init; } = 4;

    /// <summary>Object queries selected from the encoder output (900).</summary>
    public int Queries { get; init; } = 900;

    /// <summary>Text tokens the contrastive head scores (256).</summary>
    public int MaxTextLength { get; init; } = 256;

    public double LayerNormEpsilon { get; init; } = 1e-5;

    public double PositionTemperature { get; init; } = 20;

    public int GroupNormGroups { get; init; } = 32;

    /// <summary>Dropout in the encoder and decoder layers (0: the official configuration's dropout, text_dropout and fusion_dropout).</summary>
    public double Dropout { get; init; } = 0.0;

    /// <summary>Stochastic depth on the fusion layers' residual branches (0.1).</summary>
    public double FusionDropPath { get; init; } = 0.1;
}

/// <summary>
/// Grounding DINO (Liu et al. 2023): an open-set detector that finds the image regions a text prompt names. As the
/// official network (IDEA-Research/GroundingDINO; checked against Hugging Face's port), it is a Swin image backbone and a
/// BERT text backbone; a feature enhancer of six layers, each a vision-text bi-attention fusion, a text self-attention
/// and a multi-scale deformable image self-attention; language-guided query selection of the 900 encoder tokens most
/// similar to the text; and a cross-modality decoder of six layers (query self-attention, text cross-attention,
/// deformable image cross-attention) with iterative box refinement. Each query is scored against each text token by a
/// dot product (the contrastive head).
/// </summary>
internal sealed partial class GroundingDinoNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private Random _dropRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(29);

    internal sealed record Attention(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output, int Heads);

    internal sealed record Deformable(DenseLayer<T> Offsets, DenseLayer<T> Weights, DenseLayer<T> Value, DenseLayer<T> Output, int Heads, int Points);

    internal sealed record Fusion(LayerNormalizationLayer<T> VisionNorm, LayerNormalizationLayer<T> TextNorm, DenseLayer<T> VisionProj,
        DenseLayer<T> TextProj, DenseLayer<T> VisionValue, DenseLayer<T> TextValue, DenseLayer<T> VisionOut, DenseLayer<T> TextOut,
        BiasFreeLinearLayer<T> VisionGamma, BiasFreeLinearLayer<T> TextGamma);

    internal sealed record EncoderLayer(Fusion Fusion,
        Attention TextAttention, DenseLayer<T> TextFc1, DenseLayer<T> TextFc2, LayerNormalizationLayer<T> TextNormBefore, LayerNormalizationLayer<T> TextNormAfter,
        Deformable ImageAttention, LayerNormalizationLayer<T> ImageNorm, DenseLayer<T> ImageFc1, DenseLayer<T> ImageFc2, LayerNormalizationLayer<T> ImageFinalNorm);

    internal sealed record DecoderLayer(Attention SelfAttention, LayerNormalizationLayer<T> SelfNorm, Attention TextAttention,
        LayerNormalizationLayer<T> TextNorm, Deformable ImageAttention, LayerNormalizationLayer<T> ImageNorm,
        DenseLayer<T> Fc1, DenseLayer<T> Fc2, LayerNormalizationLayer<T> FinalNorm);

    public GroundingDinoNetwork(IEngine engine, List<LayerBase<T>> layers, GroundingDinoConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Dim % c.EncoderHeads != 0 || c.Dim % c.DecoderHeads != 0 || c.Dim % (c.EncoderHeads / 2) != 0)
            throw new ArgumentException("The width must divide by the head counts.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm(int dim) => Add(new LayerNormalizationLayer<T>(dim, c.LayerNormEpsilon));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        Attention Attend(int heads) => new(Linear(c.Dim), Linear(c.Dim), Linear(c.Dim), Linear(c.Dim), heads);
        Deformable Sample(int heads, int points)
            => new(Linear(heads * c.Levels * points * 2), Linear(heads * c.Levels * points), Linear(c.Dim), Linear(c.Dim), heads, points);

        Backbone = new SwinNetwork<T>(engine, layers, c.Backbone);
        Text = new BertNetwork<T>(engine, layers, c.Text);
        TextProjection = Linear(c.Dim);
        var outChannels = Backbone.OutChannels;
        for (int l = 0; l < c.Levels; l++)
        {
            bool fromBackbone = l < outChannels.Length;
            InputProjections.Add(Add(new DenseLayer<T>(c.Dim, (IActivationFunction<T>)new IdentityActivation<T>())));
            InputNorms.Add(Add(new GroupNormalizationLayer<T>(c.GroupNormGroups, c.Dim, 1e-5)));
            InputKernel.Add(fromBackbone ? 1 : 3);
        }
        LevelEmbedding = Values(c.Levels * c.Dim);
        QueryEmbedding = Values(c.Queries * c.Dim);
        int half = c.EncoderFfn / 2;
        for (int i = 0; i < c.EncoderLayers; i++)
        {
            var fusion = new Fusion(Norm(c.Dim), Norm(c.Dim), Linear(half), Linear(half), Linear(half), Linear(half), Linear(c.Dim), Linear(c.Dim),
                Values(c.Dim), Values(c.Dim));
            EncoderLayers.Add(new EncoderLayer(fusion,
                Attend(c.EncoderHeads / 2), Linear(half), Linear(c.Dim), Norm(c.Dim), Norm(c.Dim),
                Sample(c.EncoderHeads, c.EncoderPoints), Norm(c.Dim), Linear(c.EncoderFfn), Linear(c.Dim), Norm(c.Dim)));
        }
        EncoderOutput = Linear(c.Dim);
        EncoderOutputNorm = Norm(c.Dim);
        EncoderBoxHead = new[] { Linear(c.Dim), Linear(c.Dim), Linear(4) };
        for (int i = 0; i < c.DecoderLayers; i++)
            DecoderLayers.Add(new DecoderLayer(Attend(c.DecoderHeads), Norm(c.Dim), Attend(c.DecoderHeads), Norm(c.Dim),
                Sample(c.DecoderHeads, c.DecoderPoints), Norm(c.Dim), Linear(c.DecoderFfn), Linear(c.Dim), Norm(c.Dim)));
        DecoderNorm = Norm(c.Dim);
        ReferenceHead = new[] { Linear(c.Dim), Linear(c.Dim) };
        // decoder_bbox_embed_share: one box head serves every decoder layer.
        BoxHead = new[] { Linear(c.Dim), Linear(c.Dim), Linear(4) };
    }

    public GroundingDinoConfiguration Configuration { get; }

    public SwinNetwork<T> Backbone { get; }

    public BertNetwork<T> Text { get; }

    public DenseLayer<T> TextProjection { get; }

    /// <summary>Per level: a 1x1 (backbone levels) or 3x3 stride-2 (extra levels) convolution as a linear layer over
    /// its patches, then GroupNorm.</summary>
    public List<DenseLayer<T>> InputProjections { get; } = new();

    public List<GroupNormalizationLayer<T>> InputNorms { get; } = new();

    public List<int> InputKernel { get; } = new();

    public BiasFreeLinearLayer<T> LevelEmbedding { get; }

    public BiasFreeLinearLayer<T> QueryEmbedding { get; }

    public List<EncoderLayer> EncoderLayers { get; } = new();

    public DenseLayer<T> EncoderOutput { get; }

    public LayerNormalizationLayer<T> EncoderOutputNorm { get; }

    public DenseLayer<T>[] EncoderBoxHead { get; }

    public List<DecoderLayer> DecoderLayers { get; } = new();

    public LayerNormalizationLayer<T> DecoderNorm { get; }

    public DenseLayer<T>[] ReferenceHead { get; }

    public DenseLayer<T>[] BoxHead { get; }

    /// <summary>Whether dropout and stochastic depth are active.</summary>
    public bool Training
    {
        get => _training;
        set
        {
            _training = value;
            Backbone.Training = value;
            Text.Training = value;
        }
    }

    private bool _training;

    /// <summary>Seeds the dropout and stochastic-depth draws.</summary>
    public void SeedRandom(Random random)
    {
        _dropRandom = random;
        Backbone.SeedDropPath(random);
    }
}
