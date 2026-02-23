namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for instruction-tuned vision-language models that generate conversational responses
/// from visual input conditioned on user instructions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Instruction-tuned VLMs extend generative VLMs with instruction-following capabilities.
/// They are fine-tuned on visual instruction data to enable conversational AI about images.
/// Architectures include:
/// <list type="bullet">
/// <item>MLP projection (LLaVA, InternVL): ViT → MLP connector → LLM</item>
/// <item>Q-Former projection (MiniGPT-4): ViT → Q-Former → linear → LLM</item>
/// <item>Cross-attention resampler (Qwen-VL): ViT → resampler → LLM</item>
/// <item>Visual expert (CogVLM): ViT → visual expert modules in every LLM layer</item>
/// </list>
/// </para>
/// </remarks>
public interface IInstructionTunedVLM<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Generates a response in a multi-turn chat context with visual input.
    /// </summary>
    /// <param name="image">Image tensor in [channels, height, width] format.</param>
    /// <param name="conversationHistory">Previous turns as (role, content) pairs.</param>
    /// <param name="userMessage">The new user message/instruction.</param>
    /// <returns>Output tensor of token logits for the assistant's response.</returns>
    Tensor<T> Chat(Tensor<T> image, IEnumerable<(string Role, string Content)> conversationHistory, string userMessage);

    /// <summary>
    /// Gets the name of the language model backbone (e.g., "LLaMA", "Vicuna", "Qwen2", "InternLM2").
    /// </summary>
    string LanguageModelName { get; }
}
