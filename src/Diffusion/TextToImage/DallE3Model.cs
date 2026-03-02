using System.Diagnostics.CodeAnalysis;
using System.Text.RegularExpressions;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// DALL-E 3 style text-to-image generation model with advanced prompt understanding
/// and high-fidelity image generation capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implementation provides DALL-E 3 style capabilities including prompt expansion,
/// text rendering, style control, and high-quality image generation at multiple sizes.
/// </para>
/// </remarks>
public class DallE3Model<T> : LatentDiffusionModelBase<T>, IDallE3Model<T>
{
    #region Constants

    /// <remarks>4000 characters allows for detailed prompts including style, composition, and quality descriptors.</remarks>
    private const int MAX_PROMPT_LENGTH = 4000;

    /// <remarks>4 latent channels match the standard SD VAE architecture.</remarks>
    private const int LATENT_CHANNELS = 4;

    /// <remarks>8x spatial downsampling via the VAE encoder (3 downsample blocks with factor 2 each).</remarks>
    private const int VAE_SCALE_FACTOR = 8;

    /// <remarks>1024 pixels is the standard DALL-E 3 square output resolution.</remarks>
    private const int STANDARD_SIZE = 1024;

    /// <remarks>1792 pixels for landscape wide format, maintaining a ~16:9 aspect ratio with 1024 height.</remarks>
    private const int WIDE_WIDTH = 1792;

    /// <remarks>1792 pixels for portrait tall format, maintaining a ~9:16 aspect ratio with 1024 width.</remarks>
    private const int TALL_HEIGHT = 1792;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly IReadOnlyList<DallE3ImageSize> _supportedSizes;
    private readonly int? _userSeed;

    /// <summary>
    /// Timeout for regex operations to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    // Safety patterns for content filtering - using regex for more accurate matching
    private static readonly Regex[] UnsafePatterns =
    [
        // Violence patterns - match word boundaries to avoid false positives
        new Regex(@"\b(violen(ce|t)|gore|blood(y|shed)?|murder|kill(ing)?|weapon|assault|attack(ing)?)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout),
        // Explicit content patterns
        new Regex(@"\b(explicit|nude|naked|porn(ograph(y|ic))?|nsfw|sexual|erotic)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout),
        // Harmful content patterns
        new Regex(@"\b(harm(ful)?|dangerous|toxic|poison|self[-\s]?harm|suicide)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout),
        // Illegal activity patterns
        new Regex(@"\b(illegal|drug(s)?|contraband|trafficking|smuggl(e|ing))\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout),
        // Hate speech patterns
        new Regex(@"\b(hate(ful)?|racist|discrimination|bigot(ry)?|slur)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout)
    ];

    // Category names corresponding to each pattern
    private static readonly string[] UnsafeCategories = new[]
    {
        "violence",
        "explicit_content",
        "harmful_content",
        "illegal_activity",
        "hate_speech"
    };

    #endregion

    #region IDallE3Model Properties

    /// <inheritdoc/>
    public IReadOnlyList<DallE3ImageSize> SupportedSizes => _supportedSizes;

    /// <inheritdoc/>
    public int MaxPromptLength => MAX_PROMPT_LENGTH;

    /// <inheritdoc/>
    public bool SupportsEditing => true;

    /// <inheritdoc/>
    public bool SupportsVariations => true;

    #endregion

    #region ILatentDiffusionModel Properties

    /// <inheritdoc/>
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc/>
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc/>
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc/>
    public override int LatentChannels => LATENT_CHANNELS;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the DallE3Model class.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="scheduler">Optional noise scheduler.</param>
    /// <param name="conditioner">Optional conditioning module for text encoding.</param>
    /// <param name="seed">Optional seed for reproducibility.</param>
    public DallE3Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(options, scheduler, architecture)
    {
        _conditioner = conditioner;
        _userSeed = seed;

        _supportedSizes = new List<DallE3ImageSize>
        {
            DallE3ImageSize.Square1024,
            DallE3ImageSize.Wide1792x1024,
            DallE3ImageSize.Tall1024x1792
        };

        InitializeLayers(conditioner);
    }

    /// <summary>
    /// Initializes the model layers.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(IConditioningModule<T>? conditioner)
    {
        _vae = new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS);
        _unet = new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: conditioner?.EmbeddingDimension ?? 768,
            numHeads: 8);
    }

    #endregion

    #region IDallE3Model Implementation

    /// <summary>
    /// Gets a thread-safe random seed using RandomHelper.
    /// </summary>
    private int GetNextRandomSeed()
    {
        return RandomHelper.ThreadSafeRandom.Next();
    }

    /// <inheritdoc/>
    public Tensor<T> Generate(
        string prompt,
        DallE3ImageSize size = DallE3ImageSize.Square1024,
        DallE3Quality quality = DallE3Quality.Standard,
        DallE3Style style = DallE3Style.Vivid,
        int? seed = null)
    {
        ValidatePrompt(prompt);

        var (width, height) = GetDimensionsForSize(size);
        var expandedPrompt = ApplyStyleToPrompt(prompt, style);
        var numSteps = quality == DallE3Quality.HD ? 50 : 30;

        return GenerateFromText(
            expandedPrompt,
            negativePrompt: GetDefaultNegativePrompt(quality),
            width: width,
            height: height,
            numInferenceSteps: numSteps,
            guidanceScale: style == DallE3Style.Vivid ? 8.0 : 6.0,
            seed: seed ?? GetNextRandomSeed());
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor<T>> GenerateMultiple(
        string prompt,
        int count = 4,
        DallE3ImageSize size = DallE3ImageSize.Square1024,
        DallE3Quality quality = DallE3Quality.Standard,
        DallE3Style style = DallE3Style.Vivid)
    {
        count = Math.Max(1, Math.Min(4, count));
        var results = new List<Tensor<T>>();

        for (int i = 0; i < count; i++)
        {
            var image = Generate(prompt, size, quality, style, GetNextRandomSeed());
            results.Add(image);
        }

        return results;
    }

    /// <inheritdoc/>
    public (Tensor<T> Image, string RevisedPrompt) GenerateWithPrompt(
        string prompt,
        DallE3ImageSize size = DallE3ImageSize.Square1024,
        DallE3Quality quality = DallE3Quality.Standard,
        DallE3Style style = DallE3Style.Vivid)
    {
        var revisedPrompt = ExpandPrompt(prompt, style);
        var image = Generate(revisedPrompt, size, quality, style);
        return (image, revisedPrompt);
    }

    /// <inheritdoc/>
    public Tensor<T> Edit(
        Tensor<T> image,
        Tensor<T> mask,
        string prompt,
        DallE3ImageSize size = DallE3ImageSize.Square1024)
    {
        ValidatePrompt(prompt);
        var (width, height) = GetDimensionsForSize(size);

        return Inpaint(
            inputImage: image,
            mask: mask,
            prompt: prompt,
            numInferenceSteps: 50);
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor<T>> CreateVariations(
        Tensor<T> image,
        int count = 4,
        double variationStrength = 0.5,
        DallE3ImageSize size = DallE3ImageSize.Square1024)
    {
        count = Math.Max(1, Math.Min(4, count));
        variationStrength = Math.Max(0.0, Math.Min(1.0, variationStrength));
        var (width, height) = GetDimensionsForSize(size);

        var results = new List<Tensor<T>>();
        for (int i = 0; i < count; i++)
        {
            var variation = ImageToImage(
                inputImage: image,
                prompt: string.Empty,
                strength: variationStrength,
                numInferenceSteps: 30,
                seed: GetNextRandomSeed());
            results.Add(variation);
        }

        return results;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateWithText(
        string prompt,
        string textToRender,
        string textPlacement = "center",
        DallE3ImageSize size = DallE3ImageSize.Square1024)
    {
        var enhancedPrompt = $"{prompt}, with text \"{textToRender}\" prominently displayed {textPlacement}, clear readable text";
        return Generate(enhancedPrompt, size, DallE3Quality.HD, DallE3Style.Vivid);
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateWithStyle(
        string prompt,
        string artisticStyle,
        DallE3ImageSize size = DallE3ImageSize.Square1024,
        DallE3Quality quality = DallE3Quality.Standard)
    {
        var stylePrompt = MapArtisticStyle(artisticStyle);
        var enhancedPrompt = $"{prompt}, {stylePrompt}";
        return Generate(enhancedPrompt, size, quality, DallE3Style.Vivid);
    }

    /// <inheritdoc/>
    public Tensor<T> Upscale(
        Tensor<T> image,
        int scaleFactor = 2,
        bool enhanceDetails = true)
    {
        scaleFactor = Math.Max(2, Math.Min(4, scaleFactor));

        // Get current dimensions
        var shape = image.Shape;
        if (shape.Length < 3)
        {
            throw new ArgumentException("Image must have at least 3 dimensions [C, H, W]");
        }

        var channels = shape[^3];
        var height = shape[^2];
        var width = shape[^1];

        var newHeight = height * scaleFactor;
        var newWidth = width * scaleFactor;
        var newShape = new int[shape.Length];
        Array.Copy(shape, newShape, shape.Length);
        newShape[^2] = newHeight;
        newShape[^1] = newWidth;

        // Bilinear upscale followed by optional enhancement
        var upscaled = BilinearUpscale(image, newHeight, newWidth);

        if (enhanceDetails)
        {
            upscaled = ImageToImage(
                inputImage: upscaled,
                prompt: "high quality, detailed, sharp",
                strength: 0.2,
                numInferenceSteps: 20);
        }

        return upscaled;
    }

    /// <inheritdoc/>
    public Tensor<T> Outpaint(
        Tensor<T> image,
        string direction,
        int extensionPixels,
        string? prompt = null)
    {
        if (extensionPixels <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(extensionPixels),
                extensionPixels,
                "Extension pixels must be a positive value.");
        }

        var shape = image.Shape;
        var channels = shape[^3];
        var height = shape[^2];
        var width = shape[^1];

        int newWidth = width;
        int newHeight = height;
        int offsetX = 0;
        int offsetY = 0;

        switch (direction.ToLowerInvariant())
        {
            case "left":
                newWidth = width + extensionPixels;
                offsetX = extensionPixels;
                break;
            case "right":
                newWidth = width + extensionPixels;
                break;
            case "top":
                newHeight = height + extensionPixels;
                offsetY = extensionPixels;
                break;
            case "bottom":
                newHeight = height + extensionPixels;
                break;
            case "all":
                newWidth = width + 2 * extensionPixels;
                newHeight = height + 2 * extensionPixels;
                offsetX = extensionPixels;
                offsetY = extensionPixels;
                break;
            default:
                throw new ArgumentException($"Unknown direction: {direction}");
        }

        // Create expanded canvas and mask
        var expanded = CreateExpandedCanvas(image, newWidth, newHeight, offsetX, offsetY);
        var mask = CreateOutpaintMask(width, height, newWidth, newHeight, offsetX, offsetY);

        return Inpaint(
            inputImage: expanded,
            mask: mask,
            prompt: prompt ?? "continue the scene naturally",
            numInferenceSteps: 50);
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateForUseCase(
        string prompt,
        string useCase,
        DallE3ImageSize size = DallE3ImageSize.Square1024)
    {
        var useCasePrompt = useCase.ToLowerInvariant() switch
        {
            "social_media" => $"{prompt}, social media style, eye-catching, vibrant colors",
            "product_photo" => $"{prompt}, professional product photography, clean background, studio lighting",
            "illustration" => $"{prompt}, digital illustration, artistic style",
            "concept_art" => $"{prompt}, concept art, detailed environment design",
            "stock_photo" => $"{prompt}, stock photo style, professional, versatile",
            _ => prompt
        };

        return Generate(useCasePrompt, size, DallE3Quality.HD, DallE3Style.Natural);
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateTileable(
        string prompt,
        DallE3ImageSize size = DallE3ImageSize.Square1024)
    {
        var tilePrompt = $"{prompt}, seamless tileable pattern, repeating texture";
        return Generate(tilePrompt, size, DallE3Quality.HD, DallE3Style.Natural);
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateWithComposition(
        string prompt,
        IEnumerable<(string Element, string Position, double Prominence)> compositionGuide,
        DallE3ImageSize size = DallE3ImageSize.Square1024)
    {
        var compositionParts = compositionGuide.Select(c =>
            $"{c.Element} positioned {c.Position} with {c.Prominence:P0} prominence");
        var compositionPrompt = $"{prompt}, composition: {string.Join(", ", compositionParts)}";

        return Generate(compositionPrompt, size, DallE3Quality.HD, DallE3Style.Vivid);
    }

    /// <inheritdoc/>
    public (bool IsSafe, IEnumerable<string> FlaggedCategories) CheckPromptSafety(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return (true, Array.Empty<string>());
        }

        var flagged = new List<string>();

        // Check each pattern category using compiled regex for accuracy and performance
        for (int i = 0; i < UnsafePatterns.Length; i++)
        {
            if (UnsafePatterns[i].IsMatch(prompt))
            {
                flagged.Add(UnsafeCategories[i]);
            }
        }

        return (flagged.Count == 0, flagged);
    }

    /// <inheritdoc/>
    public string ExpandPrompt(
        string simplePrompt,
        DallE3Style style = DallE3Style.Vivid)
    {
        var styleDescription = style == DallE3Style.Vivid
            ? "dramatic, hyper-realistic, vibrant colors, professional photography"
            : "natural, realistic, soft lighting, organic feel";

        return $"{simplePrompt}, {styleDescription}, high quality, detailed, 8k resolution";
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor<T>> GenerateConsistentSet(
        string basePrompt,
        IEnumerable<string> variations,
        int consistencySeed,
        DallE3ImageSize size = DallE3ImageSize.Square1024)
    {
        var results = new List<Tensor<T>>();
        var baseSeed = consistencySeed;

        foreach (var variation in variations)
        {
            var combinedPrompt = $"{basePrompt}, {variation}";
            var image = Generate(combinedPrompt, size, DallE3Quality.HD, DallE3Style.Vivid, baseSeed);
            results.Add(image);
            baseSeed++; // Increment for slight variation while maintaining consistency
        }

        return results;
    }

    /// <inheritdoc/>
    public (T PredictedQuality, IEnumerable<string> Suggestions) EstimateQuality(string prompt)
    {
        var suggestions = new List<string>();
        var qualityScore = 0.7; // Base score

        // Check prompt length
        if (prompt.Length < 20)
        {
            suggestions.Add("Consider adding more detail to your prompt");
            qualityScore -= 0.1;
        }

        // Check for style descriptors
        if (!prompt.Contains("style", StringComparison.OrdinalIgnoreCase) &&
            !prompt.Contains("quality", StringComparison.OrdinalIgnoreCase))
        {
            suggestions.Add("Adding style or quality descriptors may improve results");
            qualityScore -= 0.05;
        }

        // Check for lighting descriptions
        if (!prompt.Contains("light", StringComparison.OrdinalIgnoreCase))
        {
            suggestions.Add("Consider specifying lighting conditions");
        }

        // Check for composition hints
        if (!prompt.Contains("composition", StringComparison.OrdinalIgnoreCase) &&
            !prompt.Contains("angle", StringComparison.OrdinalIgnoreCase))
        {
            suggestions.Add("Specifying composition or camera angle can help");
        }

        qualityScore = Math.Max(0.0, Math.Min(1.0, qualityScore));
        return (NumOps.FromDouble(qualityScore), suggestions);
    }

    #endregion

    #region Helper Methods

    private static (int Width, int Height) GetDimensionsForSize(DallE3ImageSize size)
    {
        return size switch
        {
            DallE3ImageSize.Square1024 => (STANDARD_SIZE, STANDARD_SIZE),
            DallE3ImageSize.Wide1792x1024 => (WIDE_WIDTH, STANDARD_SIZE),
            DallE3ImageSize.Tall1024x1792 => (STANDARD_SIZE, TALL_HEIGHT),
            _ => (STANDARD_SIZE, STANDARD_SIZE)
        };
    }

    private void ValidatePrompt(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            throw new ArgumentException("Prompt cannot be empty", nameof(prompt));
        }

        if (prompt.Length > MAX_PROMPT_LENGTH)
        {
            throw new ArgumentException($"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters", nameof(prompt));
        }

        var (isSafe, flagged) = CheckPromptSafety(prompt);
        if (!isSafe)
        {
            throw new ArgumentException($"Prompt contains potentially unsafe content: {string.Join(", ", flagged)}");
        }
    }

    private static string ApplyStyleToPrompt(string prompt, DallE3Style style)
    {
        return style switch
        {
            DallE3Style.Vivid => $"{prompt}, vivid colors, dramatic lighting, hyper-realistic details",
            DallE3Style.Natural => $"{prompt}, natural lighting, realistic, organic feel",
            _ => prompt
        };
    }

    private static string GetDefaultNegativePrompt(DallE3Quality quality)
    {
        var baseNegative = "blurry, low quality, distorted, deformed, ugly, bad anatomy";
        return quality == DallE3Quality.HD
            ? $"{baseNegative}, noise, artifacts, compression"
            : baseNegative;
    }

    private static string MapArtisticStyle(string style)
    {
        return style.ToLowerInvariant() switch
        {
            "photorealistic" => "photorealistic, ultra realistic, 8k, professional photography",
            "oil_painting" => "oil painting style, artistic brushstrokes, traditional art",
            "watercolor" => "watercolor painting, soft edges, artistic",
            "digital_art" => "digital art, vibrant colors, modern illustration",
            "anime" => "anime style, Japanese animation aesthetic",
            "sketch" => "pencil sketch, hand drawn, artistic",
            "3d_render" => "3D render, CGI, raytracing, realistic lighting",
            _ => style
        };
    }

    private Tensor<T> BilinearUpscale(Tensor<T> image, int newHeight, int newWidth)
    {
        var shape = image.Shape;
        var channels = shape[^3];
        var oldHeight = shape[^2];
        var oldWidth = shape[^1];

        var newShape = new int[shape.Length];
        Array.Copy(shape, newShape, shape.Length);
        newShape[^2] = newHeight;
        newShape[^1] = newWidth;

        var result = new Tensor<T>(newShape);
        var scaleY = (double)oldHeight / newHeight;
        var scaleX = (double)oldWidth / newWidth;

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    var srcY = y * scaleY;
                    var srcX = x * scaleX;
                    var y0 = (int)Math.Floor(srcY);
                    var x0 = (int)Math.Floor(srcX);
                    var y1 = Math.Min(y0 + 1, oldHeight - 1);
                    var x1 = Math.Min(x0 + 1, oldWidth - 1);
                    var fy = srcY - y0;
                    var fx = srcX - x0;

                    // Get the four corner values
                    var v00 = NumOps.ToDouble(image.Data.Span[c * oldHeight * oldWidth + y0 * oldWidth + x0]);
                    var v01 = NumOps.ToDouble(image.Data.Span[c * oldHeight * oldWidth + y0 * oldWidth + x1]);
                    var v10 = NumOps.ToDouble(image.Data.Span[c * oldHeight * oldWidth + y1 * oldWidth + x0]);
                    var v11 = NumOps.ToDouble(image.Data.Span[c * oldHeight * oldWidth + y1 * oldWidth + x1]);

                    // Bilinear interpolation
                    var interpolated = v00 * (1 - fx) * (1 - fy) +
                                       v01 * fx * (1 - fy) +
                                       v10 * (1 - fx) * fy +
                                       v11 * fx * fy;

                    result.Data.Span[c * newHeight * newWidth + y * newWidth + x] = NumOps.FromDouble(interpolated);
                }
            }
        }

        return result;
    }

    private Tensor<T> CreateExpandedCanvas(Tensor<T> image, int newWidth, int newHeight, int offsetX, int offsetY)
    {
        var shape = image.Shape;
        var channels = shape[^3];
        var height = shape[^2];
        var width = shape[^1];

        var newShape = new int[shape.Length];
        Array.Copy(shape, newShape, shape.Length);
        newShape[^2] = newHeight;
        newShape[^1] = newWidth;

        var result = new Tensor<T>(newShape);

        // Copy original image to offset position
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var srcIdx = c * height * width + y * width + x;
                    var dstIdx = c * newHeight * newWidth + (y + offsetY) * newWidth + (x + offsetX);
                    result.Data.Span[dstIdx] = image.Data.Span[srcIdx];
                }
            }
        }

        return result;
    }

    private Tensor<T> CreateOutpaintMask(int origWidth, int origHeight, int newWidth, int newHeight, int offsetX, int offsetY)
    {
        var mask = new Tensor<T>([1, newHeight, newWidth]);
        var one = NumOps.FromDouble(1.0);
        var zero = NumOps.Zero;

        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                var isOriginalArea = x >= offsetX && x < offsetX + origWidth &&
                                     y >= offsetY && y < offsetY + origHeight;
                mask.Data.Span[y * newWidth + x] = isOriginalArea ? zero : one;
            }
        }

        return mask;
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[i] = unetParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[unetParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
            throw new ArgumentException($"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.");

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[unetCount + i];
        }

        _unet.SetParameters(unetParams);
        _vae.SetParameters(vaeParams);
    }

    /// <inheritdoc/>
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc/>
    public override IDiffusionModel<T> Clone()
    {
        // Create a new DallE3Model with the same conditioner
        var cloned = new DallE3Model<T>(
            options: null,
            scheduler: null,
            conditioner: _conditioner);

        // Copy UNet parameters to the cloned model's UNet
        cloned._unet.SetParameters(_unet.GetParameters());

        // Copy VAE parameters to the cloned model's VAE
        cloned._vae.SetParameters(_vae.GetParameters());

        return cloned;
    }

    #endregion

    #region DiffusionModelBase Overrides

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DallE3Model",
            ModelType = ModelType.NeuralNetwork,
            Description = "DALL-E 3 style text-to-image generation model",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["architecture"] = "unclip-diffusion-v3",
                ["base_model"] = "DALL-E 3",
                ["text_encoder"] = "T5-XXL + CLIP",
                ["supported_sizes"] = _supportedSizes.Select(s => s.ToString()).ToList(),
                ["max_prompt_length"] = MAX_PROMPT_LENGTH,
                ["supports_editing"] = SupportsEditing,
                ["supports_variations"] = SupportsVariations,
                ["latent_channels"] = LATENT_CHANNELS,
                ["vae_scale_factor"] = VAE_SCALE_FACTOR
            }
        };
    }

    #endregion
}
