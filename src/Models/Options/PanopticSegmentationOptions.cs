namespace AiDotNet.Models.Options;

/// <summary>
/// Shared hyperparameters for the panoptic segmentation family — the query-based mask
/// transformers that assign every pixel both a class and an instance.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These models all work the same way: a fixed number of learned
/// "queries" each go looking for one object or region in the image. The settings here are the
/// ones every member of the family has, and each model's own options class ships the values
/// from its paper, so you do not normally set any of them.
/// </para>
/// <para>
/// The encoder size variant is deliberately NOT here. Each model has its own enum for it
/// (<c>EoMTModelSize</c>, <c>Mask2FormerModelSize</c>, and so on) because the variants a paper
/// publishes differ per model, so the property lives on each leaf with its own type.
/// </para>
/// </remarks>
public abstract class PanopticSegmentationOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the number of semantic classes the model predicts. Default: per model,
    /// typically 150 for ADE20K.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different kinds of thing the model can name. 150 is
    /// the ADE20K benchmark's class count, which is what most of these papers report against.</para>
    /// </remarks>
    public int NumClasses { get; set; }

    /// <summary>
    /// Gets or sets the number of object queries. Default: per model, typically 100.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The maximum number of separate objects the model can find in
    /// one image. Each query is a slot that either latches onto something or comes back empty.</para>
    /// </remarks>
    public int NumQueries { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate used through the transformer. Default: per model,
    /// typically 0.1. Zero disables dropout.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training the model randomly ignores this fraction of
    /// its own internal signals, which stops it leaning too hard on any one of them.</para>
    /// </remarks>
    public double DropRate { get; set; }

    /// <summary>
    /// Throws if a dimension every panoptic model requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Only <see cref="NumClasses"/> and <see cref="NumQueries"/> are required: every member of
    /// this family predicts a fixed class count over a fixed number of queries.
    /// <see cref="DropRate"/> is deliberately not required — zero is a legitimate setting that
    /// means "no dropout", so requiring it positive would reject a valid configuration.
    /// </para>
    /// <para>
    /// <b>A base may only require what every member has.</b> Requiring more than that makes a
    /// model throw at its own published defaults, with the user having configured nothing.
    /// </para>
    /// </remarks>
    protected void ValidateCore()
    {
        Require(NumClasses, nameof(NumClasses));
        Require(NumQueries, nameof(NumQueries));
    }
}
