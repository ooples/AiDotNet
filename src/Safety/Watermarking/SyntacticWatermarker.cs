using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Keyed form-transformation text watermark: at every negation or pronoun-auxiliary pair that can be written either
/// in full or contracted ("do not" / "don't", "they are" / "they're"), the key and the preceding word choose the form.
/// </summary>
/// <remarks>
/// <para>
/// Natural-language watermarking by meaning-preserving transformations was introduced by Atallah et al. (2001), who
/// applied transformations to parse trees. This watermarker uses one transformation that needs no parser:
/// contraction and expansion. Only unambiguous pairs are used ("it's" is left alone, since it may stand for "it is"
/// or "it has"). Embedding and detection follow <see cref="KeyedSubstitutionWatermarker{T}"/>: the keyed choice at
/// each site, and a z-test on how many sites agree with it.
/// </para>
/// <para>
/// Embedding changes the register of a text slightly, since it mixes contracted and full forms. Conversational and
/// technical prose usually offers several sites per hundred words; 36 distinct sites reach full confidence.
/// </para>
/// <para>
/// <b>For Beginners:</b> many phrases can be written two ways, "do not" or "don't". This watermarker picks one way at
/// each place using the secret key. Text marked with your key always matches the key's picks; other text matches only
/// about half the time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Natural Language Watermarking: Design, Analysis, and a Proof-of-Concept Implementation",
    "https://doi.org/10.1007/3-540-45496-9_14",
    Year = 2001,
    Authors = "Mikhail J. Atallah, Victor Raskin, Michael Crogan, Christian Hempelmann, Florian Kerschbaum, Dina Mohamed, Sanket Naik")]
public class SyntacticWatermarker<T> : KeyedSubstitutionWatermarker<T>
{
    /// <summary>The contraction pairs used as substitution sites.</summary>
    public static IReadOnlyList<IReadOnlyList<string>> ContractionGroups { get; } = new IReadOnlyList<string>[]
    {
        new[] { "do not", "don't" },
        new[] { "does not", "doesn't" },
        new[] { "did not", "didn't" },
        new[] { "is not", "isn't" },
        new[] { "are not", "aren't" },
        new[] { "was not", "wasn't" },
        new[] { "were not", "weren't" },
        new[] { "cannot", "can't" },
        new[] { "will not", "won't" },
        new[] { "would not", "wouldn't" },
        new[] { "should not", "shouldn't" },
        new[] { "could not", "couldn't" },
        new[] { "has not", "hasn't" },
        new[] { "have not", "haven't" },
        new[] { "had not", "hadn't" },
        new[] { "must not", "mustn't" },
        new[] { "i am", "i'm" },
        new[] { "you are", "you're" },
        new[] { "we are", "we're" },
        new[] { "they are", "they're" },
        new[] { "i will", "i'll" },
        new[] { "you will", "you'll" },
        new[] { "we will", "we'll" },
        new[] { "they will", "they'll" },
    };

    /// <inheritdoc />
    public override string ModuleName => "SyntacticWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Keyed contraction watermark";

    /// <inheritdoc />
    protected override int SeedSalt => 0x53594E31; // "SYN1"

    /// <summary>
    /// Initializes a new syntactic (contraction) watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Validated and kept for the shared API; every site is embedded.</param>
    /// <param name="key">The secret key; <c>null</c> uses the published <see cref="WatermarkKey.Public"/> key.</param>
    public SyntacticWatermarker(double watermarkStrength = 0.5, string? key = null)
        : base(ContractionGroups, watermarkStrength, key)
    {
    }
}
