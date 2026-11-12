namespace AiDotNet.Enums;

/// <summary>
/// What a Granite-speech training step optimizes (Saon et al. 2025, §II-C and §II-E).
/// </summary>
/// <remarks>
/// <para>
/// The paper trains in two stages: the conformer encoder alone with self-conditioned CTC (<see cref="EncoderCtc"/>),
/// then the Q-Former projector and LoRA adapters on the frozen encoder and LLM with the transcript's cross-entropy
/// (<see cref="ProjectorAndLora"/>). The second stage assumes a pretrained LLM; a model built here starts from random
/// weights, and a frozen random LLM cannot fit a transcript, so <see cref="Joint"/> is the default. It is NOT the
/// paper's recipe.
/// </para>
/// </remarks>
public enum GraniteSpeechTrainingStage
{
    /// <summary>The transcript's cross-entropy with every parameter trained. The default, for a model that starts from
    /// random weights.</summary>
    Joint,

    /// <summary>§II-C: the encoder alone, with self-conditioned CTC (0.2 on the middle layer) over the CTC vocabulary;
    /// the training target holds CTC labels.</summary>
    EncoderCtc,

    /// <summary>§II-E: the transcript's cross-entropy training the Q-Former projector and the LoRA adapters on the
    /// query and value projections; the encoder and the LLM's own weights stay fixed.</summary>
    ProjectorAndLora,
}
