namespace AiDotNet.Enums;

/// <summary>
/// Which parameters a training step of an LLM-based speech recognizer (a speech encoder, an adapter and a pretrained
/// LLM) updates.
/// </summary>
/// <remarks>
/// <para>
/// The papers behind these models fine-tune from a pretrained LLM: they keep its weights fixed and train the encoder,
/// the adapter and LoRA adapters on the LLM (<see cref="EncoderAdapterAndLora"/>). A model built here starts from
/// random weights; a frozen random LLM, whose output head cannot sharpen its predictions, would leave nothing able to
/// fit a transcript, so <see cref="AllParameters"/> is the default. It is NOT the papers' recipe.
/// </para>
/// </remarks>
public enum SpeechLlmTrainingScope
{
    /// <summary>Every parameter, the LLM's included. The default, for a model that starts from random weights.</summary>
    AllParameters,

    /// <summary>The papers' recipe: the encoder, the adapter and the LLM's LoRA adapters; the LLM's own weights stay
    /// fixed.</summary>
    EncoderAdapterAndLora,
}
