namespace AiDotNet.Attributes;

/// <summary>
/// How a declared hyperparameter was arrived at in the paper that states it.
/// </summary>
/// <remarks>
/// <para>
/// Not every number in a paper is a value the authors chose. Some are searched over, some vary by
/// dataset, and some are inherited from a cited work rather than stated. Treating all three as
/// though the paper had simply said "use this" is how a reproduction quietly acquires a number
/// nobody ever recommended.
/// </para>
/// <para>
/// Before this existed, the only way to be honest about a searched or per-dataset value was to omit
/// it and explain in prose — which is why five of the first hundred declarations state an optimizer
/// and no rate at all. iTransformer searches {1e-3, 5e-4, 1e-4}; BERT searches {5e-5, 3e-5, 2e-5};
/// TimesNet sets a rate per dataset in a table; DeepAR tunes it manually per dataset. All four are
/// now recordable rather than lost.
/// </para>
/// </remarks>
public enum RecipeProvenance
{
    /// <summary>The paper states this value directly as the one it used.</summary>
    Stated,

    /// <summary>
    /// The paper searched over several values; the declared one is representative rather than
    /// prescribed. List the alternatives in <c>SearchedValues</c>.
    /// </summary>
    Searched,

    /// <summary>The paper gives a different value per dataset, and the declared one is an example.</summary>
    PerDataset,

    /// <summary>
    /// The paper does not state this value; it comes from a work the paper says it follows.
    /// </summary>
    /// <remarks>
    /// Used where a paper defers — BigVGAN says its optimizer and scheduler follow HiFi-GAN, and
    /// WavLM says its hyperparameters are adapted from HuBERT. The chain belongs in the record, not
    /// in a reader's head, and the <c>Source</c> must name both ends of it.
    /// </remarks>
    DerivedFromCitedWork,
}
