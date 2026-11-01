using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Keyed synonym-substitution text watermark: at every word from a synonym group, the key and the preceding word
/// choose which synonym the text uses.
/// </summary>
/// <remarks>
/// <para>
/// This is the synonym-substitution family of natural-language watermarks (Topkara, Topkara and Atallah 2006) with a
/// keyed, context-dependent choice and a z-test detector; see <see cref="KeyedSubstitutionWatermarker{T}"/> for the
/// embedding and the statistic. It works on finished text, so it can mark the output of any model.
/// </para>
/// <para>
/// The default groups are words that can replace each other in nearly every context and start with the same kind of
/// sound, so a preceding "a" or "an" stays correct (for example "rarely" and "seldom", "toward" and "towards",
/// "additional" and "extra"). Pass your own groups to cover the vocabulary of your domain. Ordinary prose contains
/// only a few sites per hundred words, so this watermark needs long text: with two-member groups, 36 distinct sites
/// reach full confidence.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker swaps words with their synonyms in a pattern that only your key can
/// recognise. The meaning stays the same, but text marked with your key always agrees with the key's choice, while
/// other text agrees only about half the time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("The Hiding Virtues of Ambiguity: Quantifiably Resilient Watermarking of Natural Language Text through Synonym Substitutions",
    "https://doi.org/10.1145/1161366.1161397",
    Year = 2006,
    Authors = "Umut Topkara, Mercan Topkara, Mikhail J. Atallah")]
public class LexicalWatermarker<T> : KeyedSubstitutionWatermarker<T>
{
    /// <summary>The default synonym groups.</summary>
    public static IReadOnlyList<IReadOnlyList<string>> DefaultSynonymGroups { get; } = new IReadOnlyList<string>[]
    {
        new[] { "rarely", "seldom" },
        new[] { "toward", "towards" },
        new[] { "afterward", "afterwards" },
        new[] { "among", "amongst" },
        new[] { "anyway", "anyhow" },
        new[] { "additional", "extra" },
        new[] { "perhaps", "maybe" },
        new[] { "therefore", "thus", "hence" },
        new[] { "furthermore", "moreover" },
        new[] { "nevertheless", "nonetheless" },
        new[] { "quickly", "rapidly", "swiftly" },
        new[] { "usually", "typically", "normally" },
        new[] { "mainly", "chiefly", "primarily" },
        new[] { "mostly", "largely" },
        new[] { "currently", "presently" },
        new[] { "certainly", "surely" },
        new[] { "primary", "main" },
        new[] { "dreamed", "dreamt" },
        new[] { "spelled", "spelt" },
    };

    /// <inheritdoc />
    public override string ModuleName => "LexicalWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Keyed synonym watermark";

    /// <inheritdoc />
    protected override int SeedSalt => 0x4C455831; // "LEX1"

    /// <summary>
    /// Initializes a new lexical watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Validated and kept for the shared API; every synonym site is embedded.</param>
    /// <param name="key">The secret key; <c>null</c> uses the published <see cref="WatermarkKey.Public"/> key.</param>
    /// <param name="synonymGroups">The synonym groups; <c>null</c> uses <see cref="DefaultSynonymGroups"/>.</param>
    public LexicalWatermarker(double watermarkStrength = 0.5, string? key = null,
        IReadOnlyList<IReadOnlyList<string>>? synonymGroups = null)
        : base(synonymGroups ?? DefaultSynonymGroups, watermarkStrength, key)
    {
    }
}
