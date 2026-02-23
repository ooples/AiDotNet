namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for vision-language models with chain-of-thought reasoning capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reasoning VLMs extend instruction-tuned VLMs with explicit thinking/reasoning steps
/// before producing a final answer. They are trained with reinforcement learning and/or
/// chain-of-thought supervision to decompose complex visual reasoning tasks.
/// Architectures include:
/// <list type="bullet">
/// <item>QVQ/Kimi-VL: MoE language models with visual reasoning alignment</item>
/// <item>Skywork R1V: Cross-modal transfer of text reasoning to vision</item>
/// <item>LLaVA-CoT: Chain-of-thought fine-tuning on visual instruction data</item>
/// </list>
/// </para>
/// </remarks>
public interface IReasoningVLM<T> : IInstructionTunedVLM<T>
{
    /// <summary>
    /// Generates a response with explicit chain-of-thought reasoning steps.
    /// </summary>
    /// <param name="image">Image tensor in [channels, height, width] format.</param>
    /// <param name="question">The visual reasoning question to answer.</param>
    /// <returns>Output tensor of token logits including reasoning chain and final answer.</returns>
    Tensor<T> ReasonWithChainOfThought(Tensor<T> image, string question);

    /// <summary>
    /// Gets the name of the reasoning approach (e.g., "CoT", "RL-Aligned", "MoE-Reasoning").
    /// </summary>
    string ReasoningApproach { get; }
}
