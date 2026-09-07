namespace AiDotNet.Attributes;

/// <summary>
/// Which stage of training a declared paper recipe describes.
/// </summary>
/// <remarks>
/// <para>
/// Papers overwhelmingly state more than one recipe. Of 122 surveyed, 25 give a different optimizer
/// or rate per stage: LLaVA pre-trains at 2e-3 with a batch of 128 and fine-tunes at 2e-5 with 32;
/// SpeechT5 pre-trains with warmup-then-linear-decay and fine-tunes with a triangular cyclical
/// schedule; DocPedia peaks at 1e-3 then 1e-5. A single recipe per model cannot hold any of that,
/// and picking one silently discards the rest.
/// </para>
/// <para><b>For Beginners:</b> Large models are usually trained in stages — first on a huge generic
/// dataset (pre-training), then on a smaller specific one (fine-tuning). Each stage has its own
/// settings, and using the pre-training settings for a fine-tune is usually far too aggressive.
/// </para>
/// </remarks>
public enum TrainingPhase
{
    /// <summary>No phase. Only meaningful as "does not inherit"; never a valid Phase.</summary>
    /// <remarks>
    /// Exists because a nullable enum cannot be an attribute argument, so the absence of an
    /// inherited phase needs a member of its own rather than null.
    /// </remarks>
    Unspecified = 0,

    /// <summary>Training from scratch, or the first large-scale stage.</summary>
    /// <remarks>
    /// The default, because a model asked to train with no further context is training from
    /// scratch. A paper that states only one recipe is recorded here.
    /// </remarks>
    PreTraining,

    /// <summary>Adapting an already-trained model to a specific task or dataset.</summary>
    FineTuning,

    /// <summary>Training a student against a teacher's outputs.</summary>
    /// <remarks>
    /// Distinct from fine-tuning because the recipes genuinely differ: Distil-Whisper trains with
    /// its own schedule, batch and temperature rather than Whisper's.
    /// </remarks>
    Distillation,

    /// <summary>Aligning or instruction-tuning an already-capable model.</summary>
    Alignment,
}
