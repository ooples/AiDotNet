namespace AiDotNet.Models.Results;

/// <summary>
/// Represents a conformal prediction set for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// A conformal prediction set contains the class indices that are guaranteed (under standard conformal assumptions)
/// to contain the true class with the configured confidence level.
/// </para>
/// <para><b>For Beginners:</b> Instead of returning a single class label, conformal prediction can return a set of
/// possible classes. When the model is uncertain, the set tends to be larger. When the model is confident, the set
/// often contains a single class.</para>
/// </remarks>
public sealed class ClassificationConformalPredictionSet
{
    /// <summary>
    /// Gets the predicted class index sets per sample.
    /// </summary>
    /// <remarks>
    /// The outer array corresponds to each input sample. Each inner array contains the class indices included in the
    /// conformal prediction set for that sample.
    /// </remarks>
    public int[][] ClassIndices { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ClassificationConformalPredictionSet"/> class.
    /// </summary>
    /// <param name="classIndices">The predicted class index sets per sample.</param>
    public ClassificationConformalPredictionSet(int[][] classIndices)
    {
        ClassIndices = classIndices ?? throw new ArgumentNullException(nameof(classIndices));
    }
}

