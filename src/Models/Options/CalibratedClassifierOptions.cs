using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for calibrated classifiers.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Calibration improves the probability estimates from classifiers.
///
/// Choose your method based on your needs:
/// - <b>Platt Scaling</b>: Good for SVMs and linear models. Assumes S-shaped transformation.
/// - <b>Isotonic Regression</b>: Non-parametric, very flexible. Best with lots of data.
/// - <b>Beta Calibration</b>: More flexible than Platt, handles asymmetric distortions.
/// - <b>Temperature Scaling</b>: Simple and effective for neural networks.
///
/// Key settings:
/// - Use cross-validation (CrossValidationFolds > 1) for better calibration with less data
/// - More CV folds = better calibration but slower training
/// - Set Seed for reproducibility
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CalibratedClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the calibration method.
    /// </summary>
    /// <value>Default is PlattScaling.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>PlattScaling: Fits a sigmoid curve. Good default choice.</description></item>
    /// <item><description>IsotonicRegression: Non-parametric. Needs more data but very flexible.</description></item>
    /// <item><description>BetaCalibration: Like Platt but more flexible. Good for asymmetric distortions.</description></item>
    /// <item><description>TemperatureScaling: Single parameter. Best for neural networks.</description></item>
    /// </list>
    /// </remarks>
    public ProbabilityCalibrationMethod CalibrationMethod { get; set; } = ProbabilityCalibrationMethod.PlattScaling;

    /// <summary>
    /// Gets or sets the number of cross-validation folds.
    /// </summary>
    /// <value>Default is 5. Set to 1 to use holdout validation instead.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cross-validation gives better calibration because it uses
    /// all data for both training and calibration (out-of-fold predictions). Set to 1
    /// only if you have a lot of data or need faster training.</para>
    /// </remarks>
    public int CrossValidationFolds { get; set; } = 5;

    /// <summary>
    /// Gets or sets the fraction of data to use for calibration when not using cross-validation.
    /// </summary>
    /// <value>Default is 0.2 (20% holdout for calibration).</value>
    /// <remarks>
    /// Only used when CrossValidationFolds = 1.
    /// </remarks>
    public double CalibrationSetFraction { get; set; } = 0.2;
}
