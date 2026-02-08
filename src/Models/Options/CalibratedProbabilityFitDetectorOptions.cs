namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Calibrated Probability Fit Detector, which evaluates how well a model's 
/// predicted probabilities match actual outcomes.
/// </summary>
/// <remarks>
/// <para>
/// Probability calibration measures whether a model's confidence (predicted probability) aligns with its actual accuracy.
/// For example, when a well-calibrated model predicts events with 80% confidence, those events should occur about 80% of the time.
/// This detector helps identify models that are overconfident or underconfident in their predictions.
/// </para>
/// <para><b>For Beginners:</b> This class contains settings for a tool that checks if your AI model's confidence levels 
/// are trustworthy. Imagine a weather forecaster who predicts a 70% chance of rain - if it actually rains on 70% of the 
/// days when they make this prediction, they're well-calibrated. Similarly, we want AI models that say "I'm 90% sure" 
/// to be right about 90% of the time. This detector helps identify if your model is overconfident (saying it's very sure 
/// when it shouldn't be) or underconfident (not expressing enough confidence when it should).</para>
/// </remarks>
public class CalibratedProbabilityFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the number of bins used to group predictions for calibration assessment.
    /// </summary>
    /// <value>The number of calibration bins, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// Predictions are grouped into bins (e.g., 0-0.1, 0.1-0.2, etc.) to compare predicted probabilities 
    /// with actual outcomes. More bins provide finer granularity but require more data for reliable estimates.
    /// </para>
    /// <para><b>For Beginners:</b> To check calibration, we group predictions into "bins" based on their confidence levels. 
    /// The default setting (10) means we'll create 10 equal-sized groups: predictions with 0-10% confidence, 10-20% confidence, 
    /// and so on. For each group, we check if the actual success rate matches the predicted confidence. More bins give you 
    /// more detailed analysis but require more data to be reliable - like how a poll of 1,000 people gives more reliable 
    /// results than a poll of 100 people.</para>
    /// </remarks>
    public int NumCalibrationBins { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum calibration error threshold for a model to be considered well-calibrated.
    /// </summary>
    /// <value>The good fit threshold as a decimal between 0 and 1, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// This threshold represents the maximum acceptable average difference between predicted probabilities and 
    /// actual outcomes for a model to be considered well-calibrated. Lower values indicate stricter requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how close your model's confidence needs to match reality 
    /// for us to consider it "well-calibrated." The default value (0.05) means that, on average, your model's predicted 
    /// probabilities should be within 5 percentage points of the actual outcomes. For example, when your model predicts 
    /// a 75% chance, the actual frequency should be between 70% and 80% for good calibration. A lower threshold means 
    /// we're demanding more precision from your model.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the calibration error threshold above which a model is considered poorly calibrated.
    /// </summary>
    /// <value>The overfit threshold as a decimal between 0 and 1, defaulting to 0.15 (15%).</value>
    /// <remarks>
    /// <para>
    /// If the average difference between predicted probabilities and actual outcomes exceeds this threshold,
    /// the model is considered to have significant calibration issues that should be addressed.
    /// </para>
    /// <para><b>For Beginners:</b> While the GoodFitThreshold tells us when calibration is good, this setting tells 
    /// us when calibration is definitely problematic. The default value (0.15) means that if your model's predicted 
    /// probabilities differ from actual outcomes by more than 15 percentage points on average, we'll flag it as having 
    /// poor calibration. For instance, if your model consistently predicts a 60% chance but the actual frequency is 80% 
    /// or 40%, that's a sign that your model's confidence levels need adjustment.</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.15;

    /// <summary>
    /// Gets or sets the maximum allowed calibration error before the detector reports a critical issue.
    /// </summary>
    /// <value>The maximum calibration error as a decimal between 0 and 1, defaulting to 1.0 (100%).</value>
    /// <remarks>
    /// <para>
    /// This value represents an upper bound on acceptable calibration error. If the calibration error exceeds this value,
    /// it indicates a fundamental problem with the model's probability estimates that requires immediate attention.
    /// </para>
    /// <para><b>For Beginners:</b> This is a safety limit that flags severely miscalibrated models. The default value (1.0) 
    /// is the maximum possible error - it would mean your model is completely wrong about its probabilities (like consistently 
    /// predicting 100% chance when the actual frequency is 0%). In practice, you might want to set this lower, perhaps to 0.5, 
    /// to catch serious calibration problems earlier. Think of it as a "red alert" threshold that tells you when your model's 
    /// confidence levels are fundamentally unreliable.</para>
    /// </remarks>
    public double MaxCalibrationError { get; set; } = 1.0;
}
