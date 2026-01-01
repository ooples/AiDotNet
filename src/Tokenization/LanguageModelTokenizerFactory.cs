using AiDotNet.Enums;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization;

/// <summary>
/// Factory for creating tokenizers appropriate for different language model backbones.
/// </summary>
/// <remarks>
/// <para>
/// Different language models use different tokenization schemes:
/// <list type="bullet">
/// <item><description>OPT, Chinchilla: GPT-style BPE tokenization</description></item>
/// <item><description>Flan-T5: T5-style SentencePiece tokenization</description></item>
/// <item><description>LLaMA, Vicuna, Mistral: LLaMA-style SentencePiece tokenization</description></item>
/// <item><description>Phi, Qwen: GPT-style BPE with custom vocabulary</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Each language model was trained with a specific tokenizer.
/// Using the wrong tokenizer will produce garbage results. This factory creates
/// a basic tokenizer with the correct special tokens for each model type.
///
/// For production use, you should load the actual pretrained tokenizer from
/// HuggingFace using <see cref="HuggingFace.AutoTokenizer"/>.
/// </para>
/// </remarks>
public static class LanguageModelTokenizerFactory
{
    /// <summary>
    /// Creates a tokenizer appropriate for the specified language model backbone.
    /// </summary>
    /// <param name="backbone">The language model backbone type.</param>
    /// <param name="corpus">Optional training corpus. If null, uses a minimal English corpus.</param>
    /// <param name="vocabSize">Vocabulary size for training. Default is 1000 for quick testing.</param>
    /// <returns>A tokenizer configured for the specified backbone.</returns>
    /// <remarks>
    /// <para>
    /// <b>IMPORTANT:</b> This creates a minimal tokenizer suitable for testing and development.
    /// For production use with pretrained ONNX models, you MUST load the actual pretrained
    /// tokenizer that matches your model weights.
    /// </para>
    /// <para>
    /// Use <see cref="HuggingFace.AutoTokenizer.FromPretrained(string, string?)"/> to load
    /// the correct pretrained tokenizer for production use.
    /// </para>
    /// </remarks>
    public static ITokenizer CreateForBackbone(
        LanguageModelBackbone backbone,
        IEnumerable<string>? corpus = null,
        int vocabSize = 1000)
    {
        corpus ??= GetDefaultCorpus();

        return backbone switch
        {
            // GPT-style models (OPT, Chinchilla, Phi, Qwen)
            LanguageModelBackbone.OPT or
            LanguageModelBackbone.Chinchilla or
            LanguageModelBackbone.Phi or
            LanguageModelBackbone.Qwen => CreateGptStyleTokenizer(corpus, vocabSize),

            // T5-style models (Flan-T5)
            LanguageModelBackbone.FlanT5 => CreateT5StyleTokenizer(corpus, vocabSize),

            // LLaMA-style models (LLaMA, Vicuna, Mistral)
            LanguageModelBackbone.LLaMA or
            LanguageModelBackbone.Vicuna or
            LanguageModelBackbone.Mistral => CreateLlamaStyleTokenizer(corpus, vocabSize),

            _ => throw new NotSupportedException($"No default tokenizer available for backbone: {backbone}")
        };
    }

    /// <summary>
    /// Gets the recommended HuggingFace model name for loading a pretrained tokenizer.
    /// </summary>
    /// <param name="backbone">The language model backbone type.</param>
    /// <returns>The HuggingFace model identifier for loading the tokenizer.</returns>
    /// <remarks>
    /// <para>
    /// Use this with <see cref="HuggingFace.AutoTokenizer.FromPretrained(string, string?)"/>
    /// to load the correct pretrained tokenizer.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// var modelName = LanguageModelTokenizerFactory.GetHuggingFaceModelName(LanguageModelBackbone.LLaMA);
    /// var tokenizer = AutoTokenizer.FromPretrained(modelName);
    /// </code>
    /// </para>
    /// </remarks>
    public static string GetHuggingFaceModelName(LanguageModelBackbone backbone)
    {
        return backbone switch
        {
            LanguageModelBackbone.OPT => "facebook/opt-1.3b",
            LanguageModelBackbone.FlanT5 => "google/flan-t5-base",
            LanguageModelBackbone.LLaMA => "meta-llama/Llama-2-7b-hf",
            LanguageModelBackbone.Vicuna => "lmsys/vicuna-7b-v1.5",
            LanguageModelBackbone.Mistral => "mistralai/Mistral-7B-v0.1",
            LanguageModelBackbone.Chinchilla => "EleutherAI/gpt-neox-20b",
            LanguageModelBackbone.Phi => "microsoft/phi-2",
            LanguageModelBackbone.Qwen => "Qwen/Qwen-7B",
            _ => throw new NotSupportedException($"No HuggingFace model known for backbone: {backbone}")
        };
    }

    /// <summary>
    /// Gets the special tokens configuration for a language model backbone.
    /// </summary>
    /// <param name="backbone">The language model backbone type.</param>
    /// <returns>The special tokens configuration.</returns>
    public static SpecialTokens GetSpecialTokens(LanguageModelBackbone backbone)
    {
        return backbone switch
        {
            LanguageModelBackbone.OPT or
            LanguageModelBackbone.Chinchilla or
            LanguageModelBackbone.Phi or
            LanguageModelBackbone.Qwen => SpecialTokens.Gpt(),

            LanguageModelBackbone.FlanT5 => SpecialTokens.T5(),

            LanguageModelBackbone.LLaMA or
            LanguageModelBackbone.Vicuna or
            LanguageModelBackbone.Mistral => CreateLlamaSpecialTokens(),

            _ => SpecialTokens.Default()
        };
    }

    private static SpecialTokens CreateLlamaSpecialTokens()
    {
        return new SpecialTokens
        {
            UnkToken = "<unk>",
            PadToken = "<pad>",
            BosToken = "<s>",
            EosToken = "</s>",
            ClsToken = string.Empty,
            SepToken = string.Empty,
            MaskToken = string.Empty
        };
    }

    private static BpeTokenizer CreateGptStyleTokenizer(IEnumerable<string> corpus, int vocabSize)
    {
        return BpeTokenizer.Train(
            corpus,
            vocabSize,
            SpecialTokens.Gpt(),
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+");
    }

    private static SentencePieceTokenizer CreateT5StyleTokenizer(IEnumerable<string> corpus, int vocabSize)
    {
        return SentencePieceTokenizer.Train(
            corpus,
            vocabSize,
            SpecialTokens.T5());
    }

    private static SentencePieceTokenizer CreateLlamaStyleTokenizer(IEnumerable<string> corpus, int vocabSize)
    {
        return SentencePieceTokenizer.Train(
            corpus,
            vocabSize,
            CreateLlamaSpecialTokens());
    }

    private static IEnumerable<string> GetDefaultCorpus()
    {
        return new[]
        {
            "a photo of a cat",
            "a photo of a dog",
            "a picture of a bird",
            "an image of a car",
            "a drawing of a house",
            "the quick brown fox jumps over the lazy dog",
            "hello world",
            "artificial intelligence",
            "machine learning",
            "neural network",
            "deep learning",
            "natural language processing",
            "computer vision",
            "image classification",
            "object detection",
            "semantic segmentation"
        };
    }
}
