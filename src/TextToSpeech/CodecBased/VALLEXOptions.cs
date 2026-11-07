namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The languages of VALL-E X's language-ID table.</summary>
public enum VallEXLanguage
{
    /// <summary>English (the reference's id 0).</summary>
    English = 0,

    /// <summary>Mandarin Chinese (the reference's id 1).</summary>
    Mandarin = 1,

    /// <summary>Japanese (id 2 in the community checkpoint; absent from the two-language paper configuration).</summary>
    Japanese = 2,
}

/// <summary>Options for VALL-E X (Zhang et al. 2023, "Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural
/// Codec Language Modeling").</summary>
/// <remarks>
/// <para>
/// VALL-E X is VALL-E (<see cref="VALLEOptions"/>) trained on two languages with a language ID. The paper states the
/// Transformers' sizes (§5.2: 12 layers, attention width 1024, feed-forward 4096), EnCodec with eight codebooks of 1024
/// codes at 75 Hz (§3.2) and the schedule (§5.2: a peak learning rate of 5e-4 after 8,000 warm-up steps, 800k steps);
/// the rest is VALL-E's, which the paper extends. Its phonemes come from an unreleased lexicon (BigCiDian); this model
/// reads English through <see cref="FrontEnd.EnglishG2P"/> and Mandarin through <see cref="FrontEnd.MandarinG2P"/>
/// (VALL-E X's runnable reference, Plachtaa/VALL-E-X), in one table.
/// </para>
/// <para><b>For Beginners:</b> These options set VALL-E X's sizes, codec and training. The defaults reproduce the
/// paper; <see cref="LanguagePlacement"/> switches to the reference reproduction's language embedding.</para>
/// </remarks>
public class VALLEXOptions : VallEModelOptions
{
    /// <summary>Creates the paper's options.</summary>
    public VALLEXOptions()
    {
        WarmupSteps = 8_000;
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public VALLEXOptions(VALLEXOptions other)
        : base(other)
    {
        LanguagePlacement = other.LanguagePlacement;
        LanguageCount = other.LanguageCount;
        UseCommunityPhoneTokenizer = other.UseCommunityPhoneTokenizer;
        SynthesisLanguage = other.SynthesisLanguage;
    }

    /// <summary>Language embedding rows: two for the paper, three for the community checkpoint.</summary>
    public int LanguageCount { get; set; } = 2;

    /// <summary>Use the community's 70-token character BPE and unframed phoneme sequence.</summary>
    public bool UseCommunityPhoneTokenizer { get; set; }

    /// <summary>Explicit target language for synthesis, or null to infer English/Mandarin from phonemes.</summary>
    public VallEXLanguage? SynthesisLanguage { get; set; }

    /// <summary>Original Plachtaa/VALL-E-X checkpoint axes and decoding choices; this is a community reproduction.</summary>
    public static VALLEXOptions CommunityCheckpoint() => new()
    {
        TextTokens = 2048,
        LanguageCount = 3,
        LanguagePlacement = VallELanguagePlacement.TextTokens,
        PrependBos = true,
        StopOnMostLikelyEndToken = false,
        UseCommunityPhoneTokenizer = true,
    };

    /// <summary>
    /// Where the language embedding is added: the AR model's acoustic-token embeddings (the paper, §3.3, the default)
    /// or the phoneme embeddings of both models (Plachtaa/VALL-E-X).
    /// </summary>
    public VallELanguagePlacement LanguagePlacement { get; set; } = VallELanguagePlacement.AcousticTokens;
}
