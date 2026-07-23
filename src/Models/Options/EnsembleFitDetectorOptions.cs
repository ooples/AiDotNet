namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Ensemble Fit Detector, which combines multiple model fitness detectors
/// to provide more robust and accurate recommendations for algorithm selection.
/// </summary>
/// <remarks>
/// <para>
/// An ensemble fit detector evaluates how well different algorithms might perform on a given dataset
/// by combining the opinions of multiple specialized detectors. Each detector in the ensemble focuses on
/// different aspects of the data and problem characteristics.
/// </para>
/// <para><b>For Beginners:</b> Think of this as getting advice from a panel of experts instead of just one person.
/// Each expert (detector) specializes in recognizing different patterns in your data. By combining their opinions,
/// you get more reliable recommendations about which AI algorithms will work best for your specific problem.
/// It's like asking several doctors for a diagnosis instead of relying on just one opinion.</para>
/// </remarks>
public class EnsembleFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the weights applied to each detector in the ensemble.
    /// </summary>
    /// <value>A list of weights for each detector, defaulting to an empty list which will use equal weights.</value>
    /// <remarks>
    /// <para>
    /// These weights determine how much influence each individual detector has on the final recommendations.
    /// Higher weights give more importance to certain detectors. If left empty, all detectors will be weighted equally.
    /// The weights are automatically normalized so they sum to 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you control how much you trust each "expert" in your panel.
    /// For example, if you set weights to [2.0, 1.0, 1.0], the first expert's opinion counts twice as much as
    /// each of the others. If you leave this empty (the default), all experts' opinions are treated equally.
    /// You might want to adjust these weights if you know certain detectors work better for your type of data.</para>
    /// </remarks>
    public List<double> DetectorWeights { get; set; } = new List<double>();

    /// <summary>
    /// Gets or sets the maximum number of algorithm recommendations to return.
    /// </summary>
    /// <value>The maximum number of recommendations, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// This limits how many algorithm recommendations the ensemble detector will provide, sorted by their
    /// predicted fitness score. This helps focus on the most promising algorithms rather than overwhelming
    /// the user with too many options.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many suggestions you'll get from the system.
    /// With the default value of 5, you'll receive the top five AI algorithms that are predicted to work best
    /// for your data, ranked from most to least promising. This saves you time by focusing on the most likely
    /// successful approaches rather than showing you every possible option.</para>
    /// </remarks>
    public int MaxRecommendations { get; set; } = 5;
}
