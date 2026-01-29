namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the TabNet model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These options control how TabNet selects and transforms features.
/// Think of them as knobs that adjust how complex the model is and how much it should
/// focus on different parts of your data.
/// </para>
/// </remarks>
public class TabNetOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Dimension of the decision and attention steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "width" of the hidden layers.
    /// Bigger values make the model more powerful but slower to train.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Number of decision steps in the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabNet looks at your data in multiple rounds.
    /// More steps means it can refine its decision, but it also increases cost.
    /// </para>
    /// </remarks>
    public int NumDecisionSteps { get; set; } = 3;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout randomly turns off neurons during training
    /// to prevent overfitting. Higher values add more regularization.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;
}
