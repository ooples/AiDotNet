using AiDotNet.Diffusion.ThreeD;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.NeuralNetworks;

/// <summary>The feed-forward activation of a CLIP vision transformer.</summary>
public enum ClipVisionActivation
{
    /// <summary>OpenAI CLIP's <c>quick_gelu</c>, <c>x sigmoid(1.702 x)</c> (ViT-L/14 and the other OpenAI models).</summary>
    QuickGelu = 0,

    /// <summary>The exact GELU, <c>0.5 x (1 + erf(x / sqrt 2))</c>, as open_clip's LAION models (ViT-H/14) use.</summary>
    Gelu = 1
}

/// <summary>The state-dict naming a CLIP vision transformer's weights load under.</summary>
public enum ClipWeightLayout
{
    /// <summary>
    /// Hugging Face <c>CLIPVisionModelWithProjection</c>: <c>vision_model.embeddings.*</c>,
    /// <c>vision_model.encoder.layers.{i}.*</c>, <c>vision_model.post_layernorm</c> and <c>visual_projection.weight</c>.
    /// </summary>
    HuggingFace = 0,

    /// <summary>
    /// open_clip <c>VisionTransformer</c>: <c>visual.conv1</c>, <c>visual.transformer.resblocks.{i}.*</c> (fused
    /// <c>attn.in_proj_weight</c>), <c>visual.ln_post</c> and <c>visual.proj</c>.
    /// </summary>
    OpenClip = 1
}

/// <summary>A released CLIP image encoder.</summary>
public enum ClipVisionPreset
{
    /// <summary>
    /// open_clip ViT-H/14 (<c>laion2b_s32b_b79k</c>): 224 px, patch 14, width 1280, 32 layers, 16 heads, MLP 5120,
    /// exact GELU, projection to 1024. ImageDream's image encoder (<c>FrozenOpenCLIPImageEmbedder</c>).
    /// </summary>
    ViTH14OpenClip = 0,

    /// <summary>
    /// OpenAI ViT-L/14 as Hugging Face's <c>CLIPVisionModelWithProjection</c> (<c>openai/clip-vit-large-patch14</c>):
    /// 224 px, patch 14, width 1024, 24 layers, 16 heads, MLP 4096, quick GELU, projection to 768.
    /// </summary>
    ViTL14OpenAI = 1
}

/// <summary>The shape and conventions of a <see cref="ClipVisionEncoder{T}"/>.</summary>
public sealed class ClipVisionEncoderOptions
{
    /// <summary>The input side in pixels (224 for both presets).</summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>The patch side (14).</summary>
    public int PatchSize { get; set; } = 14;

    /// <summary>The transformer width.</summary>
    public int Width { get; set; } = 1280;

    /// <summary>The number of transformer layers.</summary>
    public int Layers { get; set; } = 32;

    /// <summary>The number of attention heads.</summary>
    public int Heads { get; set; } = 16;

    /// <summary>The feed-forward width.</summary>
    public int MlpWidth { get; set; } = 5120;

    /// <summary>The width of the projected image embedding (<c>image_embeds</c>); zero for no projection.</summary>
    public int ProjectionDim { get; set; } = 1024;

    /// <summary>The feed-forward activation.</summary>
    public ClipVisionActivation Activation { get; set; } = ClipVisionActivation.Gelu;

    /// <summary>The state-dict naming the weights load under.</summary>
    public ClipWeightLayout WeightLayout { get; set; } = ClipWeightLayout.OpenClip;

    /// <summary>The LayerNorm epsilon (1e-5).</summary>
    public double LayerNormEpsilon { get; set; } = 1e-5;

    /// <summary>The per-channel normalization mean (OpenAI CLIP's, which open_clip also uses).</summary>
    public double[] Mean { get; set; } = { 0.48145466, 0.4578275, 0.40821073 };

    /// <summary>The per-channel normalization standard deviation.</summary>
    public double[] Std { get; set; } = { 0.26862954, 0.26130258, 0.27577711 };

    /// <summary>The options of a released encoder.</summary>
    public static ClipVisionEncoderOptions ForPreset(ClipVisionPreset preset) => preset switch
    {
        ClipVisionPreset.ViTH14OpenClip => new ClipVisionEncoderOptions(),
        ClipVisionPreset.ViTL14OpenAI => new ClipVisionEncoderOptions
        {
            Width = 1024,
            Layers = 24,
            Heads = 16,
            MlpWidth = 4096,
            ProjectionDim = 768,
            Activation = ClipVisionActivation.QuickGelu,
            WeightLayout = ClipWeightLayout.HuggingFace
        },
        _ => throw new ArgumentOutOfRangeException(nameof(preset), preset, "Unknown CLIP preset.")
    };
}

/// <summary>
/// A CLIP vision transformer (Radford et al., 2021) that exposes every hidden state, the penultimate hidden states
/// ImageDream's resampler reads, and the projected <c>image_embeds</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// One implementation with two released presets (<see cref="ClipVisionPreset"/>): open_clip ViT-H/14, which ImageDream
/// encodes its image prompt with, and OpenAI ViT-L/14 (Hugging Face <c>CLIPVisionModelWithProjection</c>). The network is
/// the shared <c>ClipVisionNetwork</c>: patch embedding, class token and position embeddings, a LayerNorm, pre-norm
/// transformer layers, and a LayerNorm on the class token, followed here by the bias-free projection.
/// </para>
/// <para>
/// The penultimate hidden states are ImageDream's <c>encode_image_with_transformer</c>: the output of the
/// second-to-last transformer layer for every token (class token first), <c>[B, 1 + (ImageSize/PatchSize)², Width]</c>,
/// with no final LayerNorm.
/// </para>
/// <para>
/// The weights must be loaded (<see cref="LoadTorchWeights"/>) for the features to mean anything; untrained, the
/// encoder is the right architecture with random weights.
/// </para>
/// <para><b>For Beginners:</b> CLIP turns a picture into numbers that describe what is in it. Diffusion models read
/// those numbers to know what the picture looks like.</para>
/// </remarks>
/// <example>
/// <code>
/// var encoder = new ClipVisionEncoder&lt;float&gt;(ClipVisionEncoderOptions.ForPreset(ClipVisionPreset.ViTL14OpenAI));
/// var image = Tensor&lt;float&gt;.CreateRandom(1, 3, 224, 224);
/// var embeds = encoder.EncodeImage(image);                 // [1, 768]
/// var tokens = encoder.EncodePenultimateHiddenStates(image); // [1, 257, 1024]
/// </code>
/// </example>
public sealed class ClipVisionEncoder<T> : IVisualEncoder<T>, IParameterSource<T>
{
    private readonly IEngine _engine;
    private readonly ClipVisionEncoderOptions _options;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly ClipVisionNetwork<T> _network;
    private readonly BiasFreeLinearLayer<T>? _projection;

    /// <summary>Creates the encoder; ViT-H/14 (open_clip) by default.</summary>
    /// <param name="options">The encoder's shape and conventions; <see cref="ClipVisionPreset.ViTH14OpenClip"/> when null.</param>
    /// <param name="seed">Optional seed for the initial (untrained) weights.</param>
    public ClipVisionEncoder(ClipVisionEncoderOptions? options = null, int? seed = null)
    {
        _options = options ?? ClipVisionEncoderOptions.ForPreset(ClipVisionPreset.ViTH14OpenClip);
        var o = _options;
        if (o.ImageSize <= 0 || o.PatchSize <= 0 || o.Width <= 0 || o.Layers < 2 || o.Heads <= 0 || o.MlpWidth <= 0 || o.ProjectionDim < 0)
            throw new ArgumentException("The CLIP sizes must be positive, with at least two layers (the penultimate states).", nameof(options));
        if (o.Mean is null || o.Std is null || o.Mean.Length != 3 || o.Std.Length != 3)
            throw new ArgumentException("Mean and Std need three values (RGB).", nameof(options));
        _engine = AiDotNetEngine.Current;
        _network = new ClipVisionNetwork<T>(_engine, _layers, new ClipVisionConfiguration
        {
            ImageSize = o.ImageSize,
            PatchSize = o.PatchSize,
            Channels = 3,
            Hidden = o.Width,
            Layers = o.Layers,
            Heads = o.Heads,
            Intermediate = o.MlpWidth,
            LayerNormEpsilon = o.LayerNormEpsilon,
            Activation = o.Activation
        });
        if (o.ProjectionDim > 0)
        {
            _projection = new BiasFreeLinearLayer<T>(o.Width, o.ProjectionDim);
            _layers.Add(_projection);
        }
        _network.InitializeLikeReference(seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom());
    }

    /// <summary>The encoder's options.</summary>
    public ClipVisionEncoderOptions Options => _options;

    /// <inheritdoc />
    public int EmbeddingDimension => _options.ProjectionDim > 0 ? _options.ProjectionDim : _options.Width;

    /// <summary>The width of the hidden states (the transformer width).</summary>
    public int HiddenDimension => _options.Width;

    /// <summary>The number of tokens per image: the class token and one per patch.</summary>
    public int TokenCount => 1 + (_options.ImageSize / _options.PatchSize) * (_options.ImageSize / _options.PatchSize);

    /// <inheritdoc />
    public int ImageSize => _options.ImageSize;

    /// <inheritdoc />
    public int ImageChannels => 3;

    /// <summary>
    /// CLIP's preprocessing of images in [-1, 1] (the VAE's range) <c>[B, 3, H, W]</c> or <c>[3, H, W]</c>: to [0, 1],
    /// an antialiased bicubic resize of the shorter side to <see cref="ImageSize"/> and a centre crop (open_clip's and
    /// Hugging Face's <c>Resize</c> + <c>CenterCrop</c>), then the per-channel mean and standard deviation.
    /// </summary>
    public Tensor<T> Preprocess(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var images = image.Rank == 3 ? _engine.Reshape(image, new[] { 1, image.Shape[0], image.Shape[1], image.Shape[2] }) : image;
        if (images.Rank != 4 || images.Shape[1] != 3)
            throw new ArgumentException($"CLIP reads [B, 3, H, W] images; got [{string.Join(", ", image.Shape.ToArray())}].", nameof(image));
        var ops = MathHelper.GetNumericOperations<T>();
        var unit = _engine.TensorMultiplyScalar(_engine.TensorAddScalar(images, ops.One), ops.FromDouble(0.5));
        int size = _options.ImageSize, height = images.Shape[2], width = images.Shape[3];
        if (height != size || width != size)
        {
            double scale = (double)size / Math.Min(height, width);
            int resizedHeight = Math.Max(size, (int)Math.Round(height * scale)), resizedWidth = Math.Max(size, (int)Math.Round(width * scale));
            unit = AntialiasedResize<T>.Resize(unit, resizedHeight, resizedWidth, ResizeFilter.Bicubic);
            int top = (resizedHeight - size) / 2, left = (resizedWidth - size) / 2;
            unit = _engine.TensorNarrow(_engine.TensorNarrow(unit, 2, top, size), 3, left, size).Contiguous();
        }
        var mean = new Tensor<double>(new[] { 1, 3, 1, 1 }, new Vector<double>(_options.Mean)).Cast<T>();
        var inverseStd = new Tensor<double>(new[] { 1, 3, 1, 1 }, new Vector<double>(_options.Std.Select(s => 1.0 / s).ToArray())).Cast<T>();
        return _engine.TensorMultiply(_engine.TensorSubtract(unit, _engine.TensorBroadcastTo(mean, unit._shape)),
            _engine.TensorBroadcastTo(inverseStd, unit._shape));
    }

    /// <summary>
    /// Every hidden state <c>[B, TokenCount, Width]</c> of preprocessed images: entry 0 is the embeddings after the first
    /// LayerNorm, entry i the output of layer i.
    /// </summary>
    public IReadOnlyList<Tensor<T>> EncodeHiddenStatesPreprocessed(Tensor<T> pixels)
    {
        if (pixels is null) throw new ArgumentNullException(nameof(pixels));
        return _network.Forward(pixels);
    }

    /// <summary>The penultimate hidden states <c>[B, TokenCount, Width]</c> of images in [-1, 1] (ImageDream's local
    /// image features: <c>hidden[-2]</c>, the output of the second-to-last layer).</summary>
    public Tensor<T> EncodePenultimateHiddenStates(Tensor<T> image)
    {
        var states = EncodeHiddenStatesPreprocessed(Preprocess(image));
        return states[states.Count - 2];
    }

    /// <summary>
    /// The projected image embedding <c>image_embeds [B, EmbeddingDimension]</c> of images in [-1, 1]: the final
    /// LayerNorm of the last layer's class token, through the projection when there is one.
    /// </summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        var states = EncodeHiddenStatesPreprocessed(Preprocess(image));
        var last = states[states.Count - 1];
        int batch = last.Shape[0];
        var cls = _engine.Reshape(_engine.TensorNarrow(last, 1, 0, 1).Contiguous(), new[] { batch, _options.Width });
        var pooled = _network.PostNorm.Forward(cls);
        return _projection is null ? pooled : _projection.Forward(pooled);
    }

    /// <summary>
    /// Loads released weights under their official names (<see cref="ClipWeightLayout"/>): open_clip's
    /// <c>{prefix}conv1.weight</c> ... <c>{prefix}proj</c> with <paramref name="prefix"/> <c>visual.</c> by default, or
    /// Hugging Face's <c>vision_model.*</c> and <c>visual_projection.weight</c> (the projection's name has no prefix of its
    /// own; <paramref name="prefix"/> is prepended to both).
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and expected shape.</param>
    /// <param name="prefix">The state-dict prefix; null for the layout's default (<c>visual.</c> or empty).</param>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string? prefix = null)
    {
        if (read is null) throw new ArgumentNullException(nameof(read));
        LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource.From(read), prefix);
    }

    internal void LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source, string? prefix = null)
    {
        var o = _options;
        if (o.WeightLayout == ClipWeightLayout.OpenClip)
        {
            string p = prefix ?? "visual.";
            _network.LoadOpenClipWeights(source, p);
            // open_clip's proj is [width, embed] and applied as x @ proj: already the [in, out] layout.
            if (_projection is not null)
            {
                var weights = AiDotNet.TextToSpeech.CodecBased.TorchParameters.WeightsToWrite(_projection, o.Width, o.ProjectionDim);
                source.CopyTo(p + "proj", new[] { o.Width, o.ProjectionDim }, weights);
                AiDotNet.TextToSpeech.CodecBased.TorchParameters.Commit(_projection, weights);
            }
            return;
        }

        string root = prefix ?? string.Empty;
        _network.LoadTorchWeights(source, root + "vision_model.");
        if (_projection is not null)
            AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(_projection, o.Width, o.ProjectionDim, source, root + "visual_projection.weight");
    }

    /// <inheritdoc />
    public long ParameterCount => _layers.Sum(layer => layer.ParameterCount);

    /// <inheritdoc />
    public Vector<T> GetParameters() => Vector<T>.Concatenate(_layers.Select(layer => layer.GetParameters()).ToArray());

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"The CLIP encoder has {ParameterCount} parameters; got {parameters.Length}.", nameof(parameters));
        int offset = 0;
        foreach (var layer in _layers)
        {
            int count = checked((int)layer.ParameterCount);
            layer.SetParameters(parameters.GetSubVector(offset, count));
            offset += count;
        }
    }
}
