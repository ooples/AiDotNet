namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the AdaBoost R2 regression algorithm.
/// </summary>
/// <remarks>
/// <para>
/// AdaBoost R2 is an ensemble learning method that combines multiple decision trees
/// to create a more powerful regression model.
/// </para>
/// <para><b>For Beginners:</b> AdaBoost (Adaptive Boosting) is like having a team of experts
/// (decision trees) working together to solve a problem. Each expert specializes in fixing
/// the mistakes made by previous experts. The "R2" indicates this is a version designed
/// specifically for regression problems (predicting continuous values like prices or temperatures)
/// rather than classification problems (categorizing data into groups).
/// </para>
/// <para>
/// This class inherits from <see cref="DecisionTreeOptions"/>, which means it includes all the
/// configuration options for decision trees plus additional options specific to AdaBoost R2.
/// </para>
/// </remarks>
public class AdaBoostR2RegressionOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the number of decision tree estimators (weak learners) to use in the ensemble.
    /// </summary>
    /// <value>
    /// The number of estimators, defaulting to 50.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This setting controls how many "experts" (decision trees) 
    /// will be in your team. More trees can lead to better predictions but will take longer to train
    /// and use more memory. The default value of 50 is a good starting point for most problems.
    /// </para>
    /// <para>
    /// Increasing this value may improve model accuracy up to a point, but with diminishing returns.
    /// Very high values might lead to overfitting (when the model performs well on training data
    /// but poorly on new data).
    /// </para>
    /// </remarks>
    public int NumberOfEstimators { get; set; } = 50;
}
