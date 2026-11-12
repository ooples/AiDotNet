namespace AiDotNet.Enums;

/// <summary>
/// Which modules a Fun-ASR-nano training step updates: the report's supervised fine-tuning stages.
/// </summary>
/// <remarks>
/// <para>
/// The Fun-ASR report (arXiv:2509.12508, §4.2) fine-tunes in stages. Stage 1 trains only the adaptor. Stage 2 also
/// trains the audio encoder. Stage 3 freezes both and trains LoRA adapters on the LLM. Stage 4 trains the encoder, the
/// adaptor and the LoRA adapters together. Stage 5 adds a CTC decoder on the frozen encoder and trains only that. The
/// LLM's own weights are never updated: every stage presupposes a pretrained Qwen3. <see cref="Joint"/> is the default
/// because a model built here starts from random weights, which those stages would leave frozen; it is NOT the report's
/// recipe.
/// </para>
/// </remarks>
public enum FunASRNanoTrainingStage
{
    /// <summary>Every parameter, the LLM's and the CTC decoder's included, in one step. The default, for a model that
    /// starts from random weights; not one of the report's stages.</summary>
    Joint,

    /// <summary>Stage 1: only the adaptor, aligning the encoder's output with the LLM's embedding space.</summary>
    AdaptorAlignment,

    /// <summary>Stage 2: the audio encoder and the adaptor.</summary>
    EncoderAndAdaptor,

    /// <summary>Stage 3: only the LLM's LoRA adapters; the encoder and the adaptor are frozen.</summary>
    LlmLora,

    /// <summary>Stage 4: the encoder, the adaptor and the LLM's LoRA adapters together.</summary>
    FullFineTuning,

    /// <summary>Stage 5: only the CTC decoder on the frozen encoder; targets are ids in the CTC vocabulary.</summary>
    CtcDecoder,
}
