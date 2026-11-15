using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Configuration of <see cref="TripoSRModel{T}"/>: the DINO image encoder, the triplane transformer, the triplane
/// upsampler, the triplane NeRF decoder and the renderer, with the values of the official TripoSR configuration.
/// </summary>
/// <remarks>
/// <para>
/// Every default is the value in the released TripoSR configuration (<c>config.yaml</c> of
/// VAST-AI-Research/TripoSR, which the paper's Sec. 3 describes): a DINO ViT-B/16 image encoder at 512x512, 32x32x3
/// learnable triplane tokens of width 1024, a 16-layer transformer with 16 heads of width 64 and cross-attention to
/// the 768-wide image tokens, a stride-2 transposed convolution to 40-channel 64x64 triplanes, and a NeRF MLP of
/// 9 hidden SiLU layers of width 64 over the concatenated plane features, with <c>exp(density - 1)</c> density.
/// </para>
/// <para>
/// <b>For Beginners:</b> The defaults build the full-size TripoSR network (about 419 million weights, allocated
/// lazily). For experiments or tests, shrink the widths and depths; the architecture stays the same.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var tiny = new TripoSROptions
/// {
///     ImageSize = 32, PatchSize = 16, EncoderLayers = 1, EncoderHeads = 2, EncoderHeadDim = 4,
///     TriplaneTokenResolution = 2, TriplaneTokenChannels = 8, TransformerLayers = 1,
///     TransformerHeads = 2, TransformerHeadDim = 4, TransformerNormGroups = 2,
///     TriplaneChannels = 4, DecoderHiddenWidth = 8, DecoderHiddenLayers = 2, Seed = 42
/// };
/// var model = new TripoSRModel&lt;float&gt;(tiny);
/// </code>
/// </example>
public class TripoSROptions : ModelOptions
{
    private const string Config = "TripoSR config.yaml (Tochilkin et al. 2024, Sec. 3; VAST-AI-Research/TripoSR)";

    #region Image encoder

    /// <summary>The side of the square conditioning image the encoder sees (<c>cond_image_size</c>).</summary>
    [PaperDefault(512, Source = Config + ": cond_image_size 512")]
    public int ImageSize { get; set; } = 512;

    /// <summary>The ViT patch size (DINO ViT-B/16).</summary>
    [PaperDefault(16, Source = Config + ": image_tokenizer facebook/dino-vitb16, patch 16")]
    public int PatchSize { get; set; } = 16;

    /// <summary>The number of ViT encoder layers (ViT-B: 12).</summary>
    [PaperDefault(12, Source = Config + ": facebook/dino-vitb16, num_hidden_layers 12")]
    public int EncoderLayers { get; set; } = 12;

    /// <summary>The number of ViT attention heads (ViT-B: 12).</summary>
    [PaperDefault(12, Source = Config + ": facebook/dino-vitb16, num_attention_heads 12")]
    public int EncoderHeads { get; set; } = 12;

    /// <summary>The width of one ViT attention head; the encoder width is heads times this (ViT-B: 768).</summary>
    [PaperDefault(64, Source = Config + ": facebook/dino-vitb16, hidden_size 768 over 12 heads")]
    public int EncoderHeadDim { get; set; } = 64;

    /// <summary>The ViT MLP width as a multiple of the encoder width (ViT-B: 3072 = 4 x 768).</summary>
    [PaperDefault(4, Source = Config + ": facebook/dino-vitb16, intermediate_size 3072")]
    public int EncoderMlpRatio { get; set; } = 4;

    /// <summary>
    /// The side of the patch grid the learned position embeddings are stored at. DINO was pretrained at 224x224
    /// (a 14x14 grid); like the official code (<c>interpolate_pos_encoding=True</c>) the embeddings are resized
    /// bicubically to the input's patch grid at every forward.
    /// </summary>
    [PaperDefault(14, Source = Config + ": facebook/dino-vitb16, image_size 224 over patch 16")]
    public int EncoderPositionGridSize { get; set; } = 14;

    /// <summary>The ViT layer-norm epsilon.</summary>
    [PaperDefault(1e-12, Source = Config + ": facebook/dino-vitb16, layer_norm_eps 1e-12")]
    public double EncoderLayerNormEpsilon { get; set; } = 1e-12;

    /// <summary>The per-channel mean the image tokenizer subtracts from an RGB image in [0, 1] (ImageNet).</summary>
    [PaperDefault(new[] { 0.485, 0.456, 0.406 }, Source = "TripoSR tsr/models/tokenizers/image.py: image_mean")]
    public double[] ImageMean { get; set; } = { 0.485, 0.456, 0.406 };

    /// <summary>The per-channel standard deviation the image tokenizer divides by (ImageNet).</summary>
    [PaperDefault(new[] { 0.229, 0.224, 0.225 }, Source = "TripoSR tsr/models/tokenizers/image.py: image_std")]
    public double[] ImageStd { get; set; } = { 0.229, 0.224, 0.225 };

    #endregion

    #region Triplane transformer

    /// <summary>The side of each learnable triplane token plane (<c>tokenizer.plane_size</c>).</summary>
    [PaperDefault(32, Source = Config + ": tokenizer.plane_size 32")]
    public int TriplaneTokenResolution { get; set; } = 32;

    /// <summary>The width of each triplane token (<c>tokenizer.num_channels</c>).</summary>
    [PaperDefault(1024, Source = Config + ": tokenizer.num_channels 1024")]
    public int TriplaneTokenChannels { get; set; } = 1024;

    /// <summary>The number of transformer blocks (<c>backbone.num_layers</c>).</summary>
    [PaperDefault(16, Source = Config + ": backbone.num_layers 16")]
    public int TransformerLayers { get; set; } = 16;

    /// <summary>The number of attention heads per block (<c>backbone.num_attention_heads</c>).</summary>
    [PaperDefault(16, Source = Config + ": backbone.num_attention_heads 16")]
    public int TransformerHeads { get; set; } = 16;

    /// <summary>The width of one attention head (<c>backbone.attention_head_dim</c>).</summary>
    [PaperDefault(64, Source = Config + ": backbone.attention_head_dim 64")]
    public int TransformerHeadDim { get; set; } = 64;

    /// <summary>The group count of the transformer's input GroupNorm (<c>Transformer1D.norm_num_groups</c>).</summary>
    [PaperDefault(32, Source = "TripoSR tsr/models/transformer/transformer_1d.py: norm_num_groups 32")]
    public int TransformerNormGroups { get; set; } = 32;

    #endregion

    #region Triplane upsampler and decoder

    /// <summary>The channels of each upsampled triplane (<c>post_processor.out_channels</c>).</summary>
    [PaperDefault(40, Source = Config + ": post_processor.out_channels 40")]
    public int TriplaneChannels { get; set; } = 40;

    /// <summary>The width of the NeRF decoder MLP (<c>decoder.n_neurons</c>).</summary>
    [PaperDefault(64, Source = Config + ": decoder.n_neurons 64")]
    public int DecoderHiddenWidth { get; set; } = 64;

    /// <summary>The number of hidden layers of the NeRF decoder MLP (<c>decoder.n_hidden_layers</c>).</summary>
    [PaperDefault(9, Source = Config + ": decoder.n_hidden_layers 9")]
    public int DecoderHiddenLayers { get; set; } = 9;

    /// <summary>
    /// The decoder's hidden activation (<c>decoder.activation: silu</c>; <see cref="ActivationFunction.Swish"/> is
    /// SiLU). Supported: ReLU, Swish, GELU, Softplus, Tanh, Sigmoid, Identity, Linear.
    /// </summary>
    [PaperDefault(ActivationFunction.Swish, Source = Config + ": decoder.activation silu")]
    public ActivationFunction DecoderActivation { get; set; } = ActivationFunction.Swish;

    /// <summary>How the three plane features are combined (<c>renderer.feature_reduction</c>).</summary>
    [PaperDefault(TriplaneFeatureReduction.Concat, Source = Config + ": renderer.feature_reduction concat")]
    public TriplaneFeatureReduction FeatureReduction { get; set; } = TriplaneFeatureReduction.Concat;

    /// <summary>The density activation (<c>renderer.density_activation</c>).</summary>
    [PaperDefault(TriplaneDensityActivation.Exp, Source = Config + ": renderer.density_activation exp")]
    public TriplaneDensityActivation DensityActivation { get; set; } = TriplaneDensityActivation.Exp;

    /// <summary>The bias added to the raw density before its activation (<c>renderer.density_bias</c>).</summary>
    [PaperDefault(-1.0, Source = Config + ": renderer.density_bias -1.0")]
    public double DensityBias { get; set; } = -1.0;

    #endregion

    #region Renderer and geometry

    /// <summary>
    /// The half-extent of the cube [-r, r]^3 the triplanes span (<c>renderer.radius</c>, slightly above
    /// 0.5 sqrt(3)).
    /// </summary>
    [PaperDefault(0.87, Source = Config + ": renderer.radius 0.87")]
    public double Radius { get; set; } = 0.87;

    /// <summary>The number of samples per ray (<c>renderer.num_samples_per_ray</c>).</summary>
    [PaperDefault(128, Source = Config + ": renderer.num_samples_per_ray 128")]
    public int SamplesPerRay { get; set; } = 128;

    /// <summary>How many points are decoded per field query batch (<c>renderer.chunk_size</c>).</summary>
    [PaperDefault(8192, Source = "TripoSR tsr/system.py: renderer.set_chunk_size(8192) default")]
    public int QueryChunkSize { get; set; } = 8192;

    /// <summary>The side of the dense density grid sampled for geometry (<c>extract_mesh</c> resolution).</summary>
    [PaperDefault(256, Source = "TripoSR tsr/system.py: extract_mesh resolution 256")]
    public int GridResolution { get; set; } = 256;

    /// <summary>The density above which a grid sample is occupied (<c>extract_mesh</c> threshold).</summary>
    [PaperDefault(25.0, Source = "TripoSR tsr/system.py: extract_mesh threshold 25.0")]
    public double DensityThreshold { get; set; } = 25.0;

    /// <summary>The side of a rendered novel view, in pixels.</summary>
    [PaperDefault(256, Source = "TripoSR run.py: render height and width 256")]
    public int RenderResolution { get; set; } = 256;

    /// <summary>The distance of the orbiting render camera from the origin.</summary>
    [PaperDefault(1.9, Source = "TripoSR run.py: get_spherical_cameras camera_distance 1.9")]
    public double CameraDistance { get; set; } = 1.9;

    /// <summary>The vertical field of view of the render camera, in degrees.</summary>
    [PaperDefault(40.0, Source = "TripoSR run.py: get_spherical_cameras fovy_deg 40.0")]
    public double FieldOfViewDegrees { get; set; } = 40.0;

    #endregion

    /// <summary>The encoder width, heads times head width.</summary>
    internal int EncoderWidth => EncoderHeads * EncoderHeadDim;

    /// <summary>The transformer's inner width, heads times head width.</summary>
    internal int TransformerWidth => TransformerHeads * TransformerHeadDim;

    /// <summary>The side of each upsampled triplane (the token plane doubled by the stride-2 upsampler).</summary>
    internal int TriplaneResolution => 2 * TriplaneTokenResolution;
}
