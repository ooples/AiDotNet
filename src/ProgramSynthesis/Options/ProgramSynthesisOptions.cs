using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.ProgramSynthesis.Options;

/// <summary>
/// Configuration options for enabling Program Synthesis / Code Tasks via the primary builder/result APIs.
/// </summary>
/// <remarks>
/// <para>
/// This options type is intended for use with <c>PredictionModelBuilder</c> to configure a code model and
/// safe defaults (tokenization and architecture) without requiring users to construct low-level engines.
/// </para>
/// <para><b>For Beginners:</b> These settings control which built-in code model to use and how big it is.
/// You can usually accept the defaults and only change the language and model kind.
/// </para>
/// </remarks>
public class ProgramSynthesisOptions
{
    /// <summary>
    /// Gets or sets the built-in code model kind to configure when no explicit model is provided.
    /// </summary>
    public ProgramSynthesisModelKind ModelKind { get; set; } = ProgramSynthesisModelKind.CodeT5;

    /// <summary>
    /// Gets or sets the target programming language for the configured model.
    /// </summary>
    public ProgramLanguage TargetLanguage { get; set; } = ProgramLanguage.Generic;

    /// <summary>
    /// Gets or sets the default code task type used for the configured architecture.
    /// </summary>
    public CodeTask DefaultTask { get; set; } = CodeTask.Generation;

    /// <summary>
    /// Gets or sets the synthesis approach type used for the configured architecture.
    /// </summary>
    public SynthesisType SynthesisType { get; set; } = SynthesisType.Neural;

    /// <summary>
    /// Gets or sets the maximum sequence length (in tokens) that the model can process.
    /// </summary>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the vocabulary size used by the configured architecture.
    /// </summary>
    public int VocabularySize { get; set; } = 50000;

    /// <summary>
    /// Gets or sets the number of encoder layers used by the configured architecture.
    /// </summary>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of decoder layers used by the configured architecture.
    /// </summary>
    /// <remarks>
    /// This is only applicable for encoder-decoder models (e.g., CodeT5). For encoder-only models,
    /// this value is ignored.
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets an optional tokenizer to use for code tasks. If null, the builder will use an existing configured tokenizer or a safe default.
    /// </summary>
    public ITokenizer? Tokenizer { get; set; }
}
