namespace AiDotNet.Enums;

/// <summary>
/// What a Phi-4-multimodal speech training step updates (Abdin et al. 2025, §2.2.2).
/// </summary>
/// <remarks>
/// <para>
/// The paper trains speech in two stages on a frozen Phi-4-Mini: pre-training aligns the audio encoder and projector on
/// ASR data (<see cref="EncoderAlignment"/>), then post-training updates the projector and the speech LoRA adapters with
/// the encoder frozen (<see cref="ProjectorAndLora"/>). Both assume a pretrained language model; a model built here
/// starts from random weights, and a frozen random decoder cannot fit a transcript, so <see cref="Joint"/> is the
/// default. It is NOT the paper's recipe.
/// </para>
/// </remarks>
public enum Phi4AudioTrainingStage
{
    /// <summary>Every parameter, the language model's included. The default, for a model that starts from random
    /// weights.</summary>
    Joint,

    /// <summary>§2.2.2 speech pre-training: the audio encoder and projector, with the language model frozen.</summary>
    EncoderAlignment,

    /// <summary>§2.2.2 speech post-training: the projector and the speech LoRA adapters, with the audio encoder and the
    /// language model frozen.</summary>
    ProjectorAndLora,
}
