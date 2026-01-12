namespace AiDotNet.Enums;

/// <summary>
/// Defines the language model backbone types used in multimodal neural networks.
/// </summary>
/// <remarks>
/// <para>
/// This enum specifies which language model architecture is used as the backbone for text generation
/// and understanding in multimodal models like BLIP-2, LLaVA, and Flamingo. The backbone determines
/// the model's capacity, vocabulary, and generation capabilities.
/// </para>
/// <para><b>For Beginners:</b> Think of the language model backbone as the "brain" that processes
/// and generates text in vision-language models.
///
/// When a model like BLIP-2 needs to describe an image or answer a question about it:
/// 1. The vision encoder extracts features from the image
/// 2. The Q-Former/adapter bridges vision and language
/// 3. The language model backbone generates the actual text response
///
/// Different backbones have different strengths:
/// - <b>OPT</b>: Good for general text generation, used in BLIP-2
/// - <b>FlanT5</b>: Better for instruction-following, used in BLIP-2
/// - <b>LLaMA</b>: Efficient and powerful, used in LLaVA
/// - <b>Vicuna</b>: LLaMA fine-tuned for conversations, used in LLaVA
/// - <b>Mistral</b>: Fast and efficient, newer alternative for LLaVA
/// - <b>Chinchilla</b>: Used in Flamingo, optimized for multimodal learning
///
/// The choice affects model size, speed, and quality of text generation.
/// </para>
/// </remarks>
public enum LanguageModelBackbone
{
    /// <summary>
    /// OPT (Open Pre-trained Transformer) by Meta AI.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> OPT is a family of decoder-only language models from Meta AI
    /// that range from 125M to 175B parameters. It's commonly used in BLIP-2.
    ///
    /// Key characteristics:
    /// - Decoder-only architecture (like GPT)
    /// - Good for general text generation
    /// - Available in various sizes (OPT-2.7B is common for BLIP-2)
    /// - Hidden dimension: 2560 for OPT-2.7B
    /// </para>
    /// </remarks>
    OPT,

    /// <summary>
    /// Flan-T5 by Google - instruction-tuned T5 model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Flan-T5 is Google's T5 model fine-tuned on a mixture of
    /// instruction-following tasks. It's an encoder-decoder model.
    ///
    /// Key characteristics:
    /// - Encoder-decoder architecture
    /// - Excellent at following instructions
    /// - Better for question-answering tasks
    /// - Hidden dimension: 2048 for Flan-T5-XL
    /// - Commonly used in BLIP-2 for instruction-following variants
    /// </para>
    /// </remarks>
    FlanT5,

    /// <summary>
    /// LLaMA (Large Language Model Meta AI) by Meta.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LLaMA is Meta's efficient open-source language model
    /// that achieves strong performance with fewer parameters.
    ///
    /// Key characteristics:
    /// - Decoder-only architecture
    /// - Very efficient for its size
    /// - Base model for many fine-tuned variants
    /// - Commonly used in LLaVA
    /// - Available in 7B, 13B, 33B, 65B sizes
    /// </para>
    /// </remarks>
    LLaMA,

    /// <summary>
    /// Vicuna - LLaMA fine-tuned on conversational data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vicuna is LLaMA fine-tuned on user conversations,
    /// making it better at natural dialogue and instruction-following.
    ///
    /// Key characteristics:
    /// - Based on LLaMA architecture
    /// - Fine-tuned for conversation
    /// - Better at following complex instructions
    /// - Popular choice for LLaVA-1.5+
    /// </para>
    /// </remarks>
    Vicuna,

    /// <summary>
    /// Mistral - efficient open-source language model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mistral is a newer, highly efficient language model
    /// that outperforms LLaMA-2 on many benchmarks despite being smaller.
    ///
    /// Key characteristics:
    /// - Uses sliding window attention for efficiency
    /// - Strong performance at 7B parameter scale
    /// - Good for resource-constrained scenarios
    /// - Increasingly used in newer LLaVA variants
    /// </para>
    /// </remarks>
    Mistral,

    /// <summary>
    /// Chinchilla by DeepMind - compute-optimal language model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Chinchilla is DeepMind's language model optimized for
    /// the right balance of model size and training data.
    ///
    /// Key characteristics:
    /// - 70B parameters trained on 1.4T tokens
    /// - Optimized for training efficiency
    /// - Used as backbone in Flamingo
    /// - Excellent multimodal learning capabilities
    /// </para>
    /// </remarks>
    Chinchilla,

    /// <summary>
    /// Phi by Microsoft - small but capable language model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Phi models are Microsoft's small language models
    /// that achieve impressive performance for their size.
    ///
    /// Key characteristics:
    /// - Very small (1.3B to 3B parameters)
    /// - Trained on high-quality "textbook" data
    /// - Fast inference on limited hardware
    /// - Good for lightweight multimodal applications
    /// </para>
    /// </remarks>
    Phi,

    /// <summary>
    /// Qwen by Alibaba - multilingual language model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Qwen is Alibaba's multilingual language model
    /// with strong Chinese and English capabilities.
    ///
    /// Key characteristics:
    /// - Strong multilingual support
    /// - Good for international applications
    /// - Available in various sizes
    /// - Used in Qwen-VL for vision-language tasks
    /// </para>
    /// </remarks>
    Qwen,

    /// <summary>
    /// RoBERTa - robustly optimized BERT approach.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RoBERTa is a strong encoder-only language model
    /// based on BERT with improved training settings.
    ///
    /// Key characteristics:
    /// - Encoder-only architecture
    /// - Byte-level BPE tokenizer
    /// - Common backbone for document understanding models like LayoutLMv3
    /// </para>
    /// </remarks>
    RoBERTa
}
