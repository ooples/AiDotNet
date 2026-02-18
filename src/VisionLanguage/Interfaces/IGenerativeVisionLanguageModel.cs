namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for generative vision-language models that produce text output from visual input.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Generative VLMs take an image (and optionally a text prompt) and produce text output
/// such as captions, answers to visual questions, or descriptions. Architectures include:
/// <list type="bullet">
/// <item>Q-Former bridges (BLIP-2, InstructBLIP) - lightweight adapter between frozen encoders</item>
/// <item>Encoder-decoder (GIT, CoCa, PaLI) - ViT encoder + autoregressive text decoder</item>
/// <item>Perceiver resampler (Flamingo, IDEFICS) - latent queries cross-attend to vision features</item>
/// <item>Causal multimodal (KOSMOS) - visual tokens embedded directly in causal LM</item>
/// <item>Unified generation (Emu) - single model for understanding + generation</item>
/// </list>
/// </para>
/// </remarks>
public interface IGenerativeVisionLanguageModel<T> : IVisualEncoder<T>
{
    /// <summary>
    /// Generates output token logits from an image, optionally conditioned on a text prompt.
    /// </summary>
    /// <param name="image">Image tensor in [channels, height, width] format.</param>
    /// <param name="prompt">Optional text prompt (question, instruction, or prefix for captioning).</param>
    /// <returns>Output tensor of token logits/embeddings for text generation.</returns>
    Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null);

    /// <summary>
    /// Gets the maximum number of tokens the model can generate.
    /// </summary>
    int MaxGenerationLength { get; }

    /// <summary>
    /// Gets the dimensionality of the decoder embedding space.
    /// </summary>
    int DecoderEmbeddingDim { get; }
}
