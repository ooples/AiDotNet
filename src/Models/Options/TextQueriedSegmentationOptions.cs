namespace AiDotNet.Models.Options;

/// <summary>
/// Options shared by segmentation models that classify pixels against text embeddings from a
/// CLIP-style text encoder: the open-vocabulary models (CAT-Seg, SAN, SED, Mask-Adapter,
/// Open-Vocabulary SAM) and the referring models (LISA, VideoLISA, OMG-LLaVA, PixelLM).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These models never learn a fixed list of classes. They turn each class name
/// (or a sentence) into a vector with a text encoder, turn each pixel into a vector of the same size,
/// and score a pixel against a name by how closely the two vectors point the same way.
/// </para>
/// </remarks>
public abstract class TextQueriedSegmentationOptions : SegmentationModelOptions
{
    /// <summary>
    /// Gets or sets the softmax temperature <c>tau</c> that divides each cosine similarity between a
    /// pixel embedding and a text embedding.
    /// </summary>
    /// <value>
    /// Defaults to 0.01, the inverse of CLIP's logit scale of 100 (Radford et al. 2021), which these
    /// models reuse when they score pixels or masks against CLIP text embeddings.
    /// </value>
    public double Temperature { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the names of the classes the plain <c>Predict</c>/<c>Train</c> path scores against,
    /// one per output channel.
    /// </summary>
    /// <value>
    /// <c>null</c> by default, which names the channels "class 0" ... "class N-1". Set it to the dataset's
    /// category names (for example the 150 ADE20K names) so the closed-set output means those classes.
    /// Its length must equal <see cref="SegmentationModelOptions.NumClasses"/>.
    /// </value>
    public string[]? ClassNames { get; set; }

    /// <summary>
    /// Copies the text-query settings from <paramref name="other"/>.
    /// </summary>
    protected void CopyTextQuerySettings(TextQueriedSegmentationOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Temperature = other.Temperature;
        ClassNames = other.ClassNames is null ? null : (string[])other.ClassNames.Clone();
    }

    /// <summary>
    /// Returns <see cref="ClassNames"/>, or "class 0" ... "class N-1" when it is not set.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// <see cref="ClassNames"/> is set but its length is not <see cref="SegmentationModelOptions.NumClasses"/>.
    /// </exception>
    internal IReadOnlyList<string> ResolveClassNames()
    {
        if (ClassNames is null)
            return Enumerable.Range(0, NumClasses).Select(i => "class " + i.ToString(CultureInfo.InvariantCulture)).ToArray();
        if (ClassNames.Length != NumClasses)
            throw new ArgumentException(
                $"{GetType().Name}.{nameof(ClassNames)} has {ClassNames.Length} names but {nameof(NumClasses)} is {NumClasses}.",
                OptionsParameterName);
        return ClassNames;
    }

    /// <summary>
    /// Throws when <see cref="Temperature"/> is not a positive finite number.
    /// </summary>
    internal void ValidateTextQuerySettings()
    {
        if (double.IsNaN(Temperature) || double.IsInfinity(Temperature) || Temperature <= 0.0)
            throw new ArgumentException(
                $"{GetType().Name}.{nameof(Temperature)} is {Temperature.ToString(CultureInfo.InvariantCulture)}, " +
                "but it must be a positive finite number.",
                OptionsParameterName);
    }
}
