using System.Globalization;

namespace AiDotNet.Models.Options;

/// <summary>
/// The hyperparameters every segmentation model in the library has: how many classes it predicts,
/// and how much dropout its encoder uses.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A segmentation model labels every pixel (or point, or voxel) of its input.
/// Two settings apply to all of them regardless of the architecture: how many different labels it
/// can choose between, and how much of the network is randomly switched off during training to stop
/// it memorising the training images. Everything else — patch sizes, embedding widths, decoder
/// shapes — differs per model and lives on that model's own options class.
/// </para>
/// <para>
/// Both values were previously constructor parameters on all 62 segmentation models, which is the
/// defect <see href="https://github.com/ooples/AiDotNet/issues/2090">issue #2090</see> describes:
/// the options object existed and was stored, but the values that actually shape the model arrived
/// past it and could not be set through it. They are declared here rather than repeated on each
/// leaf because every one of the 62 has both, and every one of them called them by these names.
/// </para>
/// <para>
/// A model's <i>size variant</i> is deliberately NOT here. Each model names its own variants with
/// its own enum (<c>SegFormerModelSize</c>, <c>SAMModelSize</c>, and so on), so there is no common
/// type to declare; the variant stays a property on the leaf options class.
/// </para>
/// </remarks>
public abstract class SegmentationModelOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the number of classes the model predicts per pixel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The number of different labels the model can assign. A model trained
    /// on ADE20K predicts 150, one trained on Cityscapes 19, and a model that only separates a
    /// single object from its background predicts 1.
    /// </para>
    /// <para>
    /// Each model's parameterless constructor sets this to the class count of the dataset its paper
    /// reports results on, so the default reproduces the published configuration.
    /// </para>
    /// </remarks>
    public int NumClasses { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate applied inside the encoder during training. Zero disables
    /// dropout.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training the model randomly ignores this fraction of its own
    /// internal signals, which stops it leaning too hard on any one of them. It has no effect at
    /// prediction time. Zero is a normal, meaningful setting — many of these models publish no
    /// dropout at all — so it is accepted rather than rejected as "unset".
    /// </para>
    /// </remarks>
    public double DropRate { get; set; }

    /// <summary>
    /// Throws when the values shared by every segmentation model cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="NumClasses"/> is not positive, or when <see cref="DropRate"/> is not
    /// a fraction in [0, 1).
    /// </exception>
    /// <remarks>
    /// <para>
    /// Each leaf exposes its own <c>Validate()</c> that calls this, rather than this being called
    /// automatically: a leaf with extra requirements adds them alongside, and the coverage guard in
    /// <c>OptionsDefaultsValidateTests</c> can see that each class opted in.
    /// </para>
    /// <para>
    /// <see cref="DropRate"/> is range-checked rather than required, because zero is meaningful and
    /// <c>Require</c> rejects zero. One is excluded at the top: dropping every signal leaves the
    /// encoder with nothing to pass on.
    /// </para>
    /// </remarks>
    protected void ValidateSegmentationCore()
    {
        Require(NumClasses, nameof(NumClasses));

        if (double.IsNaN(DropRate) || DropRate < 0.0 || DropRate >= 1.0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.{nameof(DropRate)} is "
                    + $"{DropRate.ToString(CultureInfo.InvariantCulture)}, but it must be at least 0 "
                    + "and less than 1. Zero disables dropout.",
                OptionsParameterName);
        }
    }
}
