using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents the available image sizes for DALL-E 3 generation.
/// </summary>
public enum DallE3ImageSize
{
    /// <summary>Square 1024x1024 image.</summary>
    Square1024,
    /// <summary>Wide 1792x1024 image (landscape).</summary>
    Wide1792x1024,
    /// <summary>Tall 1024x1792 image (portrait).</summary>
    Tall1024x1792
}

/// <summary>
/// Represents the quality settings for DALL-E 3 generation.
/// </summary>
public enum DallE3Quality
{
    /// <summary>Standard quality - faster generation.</summary>
    Standard,
    /// <summary>HD quality - more detail and consistency.</summary>
    HD
}

/// <summary>
/// Represents the style settings for DALL-E 3 generation.
/// </summary>
public enum DallE3Style
{
    /// <summary>Vivid style - hyper-real and dramatic.</summary>
    Vivid,
    /// <summary>Natural style - more natural, less hyper-real.</summary>
    Natural
}

/// <summary>
/// Defines the contract for DALL-E 3-style text-to-image generation models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DALL-E 3 represents a significant advancement in text-to-image generation,
/// with improved prompt following, text rendering, and overall image quality.
/// It uses a combination of diffusion models and language understanding.
/// </para>
/// <para><b>For Beginners:</b> DALL-E 3 creates images from text descriptions!
///
/// Key capabilities:
/// - High-fidelity image generation from text prompts
/// - Accurate text rendering within images
/// - Complex scene composition with multiple objects
/// - Style control (vivid vs natural)
/// - Multiple aspect ratios and sizes
///
/// Architecture concepts:
/// 1. Text Encoder: Understands and expands prompts
/// 2. Diffusion Model: Generates images through iterative denoising
/// 3. Safety Systems: Filters inappropriate content
/// 4. Quality Enhancement: Upscaling and refinement
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("DallE3Model")]
public interface IDallE3Model<T>
{
    /// <summary>
    /// Gets the supported image sizes.
    /// </summary>
    IReadOnlyList<DallE3ImageSize> SupportedSizes { get; }

    /// <summary>
    /// Gets the maximum prompt length in characters.
    /// </summary>
    int MaxPromptLength { get; }

    /// <summary>
    /// Gets whether the model supports image editing (inpainting).
    /// </summary>
    bool SupportsEditing { get; }

    /// <summary>
    /// Gets whether the model supports image variations.
    /// </summary>
    bool SupportsVariations { get; }

    /// <summary>
    /// Generates an image from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the desired image.</param>
    /// <param name="size">Output image size.</param>
    /// <param name="quality">Quality setting.</param>
    /// <param name="style">Style setting.</param>
    /// <param name="seed">Optional seed for reproducibility.</param>
    /// <returns>Generated image tensor [channels, height, width].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The main function - describe what you want!
    ///
    /// Tips for good prompts:
    /// - Be specific about subject, style, and composition
    /// - Include lighting and mood descriptions
    /// - Mention artistic style if desired (e.g., "oil painting", "digital art")
    /// - Describe spatial relationships clearly
    /// </para>
    /// </remarks>
    Tensor<T> Generate(
        string prompt,
        DallE3ImageSize size = DallE3ImageSize.Square1024,
        DallE3Quality quality = DallE3Quality.Standard,
        DallE3Style style = DallE3Style.Vivid,
        int? seed = null);

    /// <summary>
    /// Generates multiple images from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the desired images.</param>
    /// <param name="count">Number of images to generate (1-4).</param>
    /// <param name="size">Output image size.</param>
    /// <param name="quality">Quality setting.</param>
    /// <param name="style">Style setting.</param>
    /// <returns>Collection of generated image tensors.</returns>
    IEnumerable<Tensor<T>> GenerateMultiple(
        string prompt,
        int count = 4,
        DallE3ImageSize size = DallE3ImageSize.Square1024,
        DallE3Quality quality = DallE3Quality.Standard,
        DallE3Style style = DallE3Style.Vivid);

    /// <summary>
    /// Generates an image with the revised/expanded prompt returned.
    /// </summary>
    /// <param name="prompt">Original text prompt.</param>
    /// <param name="size">Output image size.</param>
    /// <param name="quality">Quality setting.</param>
    /// <param name="style">Style setting.</param>
    /// <returns>Generated image and the expanded prompt used.</returns>
    /// <remarks>
    /// <para>
    /// DALL-E 3 internally expands prompts for better results.
    /// This method returns both the image and the expanded prompt
    /// so you can see how your prompt was interpreted.
    /// </para>
    /// </remarks>
    (Tensor<T> Image, string RevisedPrompt) GenerateWithPrompt(
        string prompt,
        DallE3ImageSize size = DallE3ImageSize.Square1024,
        DallE3Quality quality = DallE3Quality.Standard,
        DallE3Style style = DallE3Style.Vivid);

    /// <summary>
    /// Edits an existing image based on a prompt and mask.
    /// </summary>
    /// <param name="image">Original image to edit.</param>
    /// <param name="mask">Mask indicating areas to edit (white = edit, black = keep).</param>
    /// <param name="prompt">Description of what to generate in masked areas.</param>
    /// <param name="size">Output image size.</param>
    /// <returns>Edited image.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Change specific parts of an image!
    ///
    /// The mask tells the model where to make changes:
    /// - White (255) areas will be regenerated based on prompt
    /// - Black (0) areas will be preserved from the original
    /// - Gray areas blend between original and generated
    /// </para>
    /// </remarks>
    Tensor<T> Edit(
        Tensor<T> image,
        Tensor<T> mask,
        string prompt,
        DallE3ImageSize size = DallE3ImageSize.Square1024);

    /// <summary>
    /// Generates variations of an existing image.
    /// </summary>
    /// <param name="image">Source image to create variations of.</param>
    /// <param name="count">Number of variations to generate.</param>
    /// <param name="variationStrength">How different from original (0-1).</param>
    /// <param name="size">Output image size.</param>
    /// <returns>Collection of image variations.</returns>
    IEnumerable<Tensor<T>> CreateVariations(
        Tensor<T> image,
        int count = 4,
        double variationStrength = 0.5,
        DallE3ImageSize size = DallE3ImageSize.Square1024);

    /// <summary>
    /// Generates an image with specific text rendered in it.
    /// </summary>
    /// <param name="prompt">Overall image description.</param>
    /// <param name="textToRender">Exact text to appear in the image.</param>
    /// <param name="textPlacement">Where to place text: "top", "center", "bottom", "overlay".</param>
    /// <param name="size">Output image size.</param>
    /// <returns>Generated image with text.</returns>
    /// <remarks>
    /// <para>
    /// DALL-E 3 has improved text rendering capabilities compared to earlier models.
    /// Use this method when you need specific text to appear in the image.
    /// </para>
    /// </remarks>
    Tensor<T> GenerateWithText(
        string prompt,
        string textToRender,
        string textPlacement = "center",
        DallE3ImageSize size = DallE3ImageSize.Square1024);

    /// <summary>
    /// Generates an image in a specific artistic style.
    /// </summary>
    /// <param name="prompt">Content description.</param>
    /// <param name="artisticStyle">Style: "photorealistic", "oil_painting", "watercolor", "digital_art", "anime", "sketch", "3d_render".</param>
    /// <param name="size">Output image size.</param>
    /// <param name="quality">Quality setting.</param>
    /// <returns>Generated image in specified style.</returns>
    Tensor<T> GenerateWithStyle(
        string prompt,
        string artisticStyle,
        DallE3ImageSize size = DallE3ImageSize.Square1024,
        DallE3Quality quality = DallE3Quality.Standard);

    /// <summary>
    /// Upscales an image to higher resolution.
    /// </summary>
    /// <param name="image">Image to upscale.</param>
    /// <param name="scaleFactor">Upscale factor (2 or 4).</param>
    /// <param name="enhanceDetails">Whether to enhance details during upscaling.</param>
    /// <returns>Upscaled image.</returns>
    Tensor<T> Upscale(
        Tensor<T> image,
        int scaleFactor = 2,
        bool enhanceDetails = true);

    /// <summary>
    /// Outpaints an image, extending it beyond its original boundaries.
    /// </summary>
    /// <param name="image">Original image.</param>
    /// <param name="direction">Direction to extend: "left", "right", "top", "bottom", "all".</param>
    /// <param name="extensionPixels">How many pixels to extend.</param>
    /// <param name="prompt">Optional prompt to guide the extension.</param>
    /// <returns>Extended image.</returns>
    Tensor<T> Outpaint(
        Tensor<T> image,
        string direction,
        int extensionPixels,
        string? prompt = null);

    /// <summary>
    /// Generates an image optimized for a specific use case.
    /// </summary>
    /// <param name="prompt">Image description.</param>
    /// <param name="useCase">Use case: "social_media", "product_photo", "illustration", "concept_art", "stock_photo".</param>
    /// <param name="size">Output image size.</param>
    /// <returns>Generated image optimized for use case.</returns>
    Tensor<T> GenerateForUseCase(
        string prompt,
        string useCase,
        DallE3ImageSize size = DallE3ImageSize.Square1024);

    /// <summary>
    /// Generates a seamlessly tileable image.
    /// </summary>
    /// <param name="prompt">Pattern or texture description.</param>
    /// <param name="size">Output image size.</param>
    /// <returns>Tileable image.</returns>
    /// <remarks>
    /// <para>
    /// Useful for creating textures, wallpapers, and backgrounds
    /// that can be repeated without visible seams.
    /// </para>
    /// </remarks>
    Tensor<T> GenerateTileable(
        string prompt,
        DallE3ImageSize size = DallE3ImageSize.Square1024);

    /// <summary>
    /// Generates an image with controlled composition.
    /// </summary>
    /// <param name="prompt">Overall description.</param>
    /// <param name="compositionGuide">Composition elements with positions.</param>
    /// <param name="size">Output image size.</param>
    /// <returns>Generated image following composition guide.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Control where things appear in your image!
    ///
    /// Composition guide format:
    /// [("subject", "center", 0.5), ("background", "back", 0.2), ("accent", "bottom-right", 0.3)]
    /// </para>
    /// </remarks>
    Tensor<T> GenerateWithComposition(
        string prompt,
        IEnumerable<(string Element, string Position, double Prominence)> compositionGuide,
        DallE3ImageSize size = DallE3ImageSize.Square1024);

    /// <summary>
    /// Checks if a prompt is likely to be rejected for safety reasons.
    /// </summary>
    /// <param name="prompt">Prompt to check.</param>
    /// <returns>Whether the prompt is safe and any flagged categories.</returns>
    (bool IsSafe, IEnumerable<string> FlaggedCategories) CheckPromptSafety(string prompt);

    /// <summary>
    /// Expands a simple prompt into a more detailed description.
    /// </summary>
    /// <param name="simplePrompt">Brief description.</param>
    /// <param name="style">Desired style for expansion.</param>
    /// <returns>Expanded, detailed prompt.</returns>
    string ExpandPrompt(
        string simplePrompt,
        DallE3Style style = DallE3Style.Vivid);

    /// <summary>
    /// Generates a consistent set of images (same character/scene, different poses/angles).
    /// </summary>
    /// <param name="basePrompt">Base description of the subject.</param>
    /// <param name="variations">List of variation descriptions (poses, angles, etc.).</param>
    /// <param name="consistencySeed">Seed for maintaining consistency.</param>
    /// <param name="size">Output image size.</param>
    /// <returns>Collection of consistent images.</returns>
    IEnumerable<Tensor<T>> GenerateConsistentSet(
        string basePrompt,
        IEnumerable<string> variations,
        int consistencySeed,
        DallE3ImageSize size = DallE3ImageSize.Square1024);

    /// <summary>
    /// Estimates the generation quality before actually generating.
    /// </summary>
    /// <param name="prompt">Prompt to evaluate.</param>
    /// <returns>Predicted quality score and improvement suggestions.</returns>
    (T PredictedQuality, IEnumerable<string> Suggestions) EstimateQuality(string prompt);
}
