namespace AiDotNet.Models.Options;

/// <summary>
/// Represents the options for Monte Carlo cross-validation.
/// </summary>
/// <remarks>
/// <para>
/// This class extends the base CrossValidationOptions with additional properties specific to Monte Carlo cross-validation.
/// </para>
/// <para><b>For Beginners:</b> Monte Carlo cross-validation options help you customize how the Monte Carlo method splits and tests your data.
/// 
/// What this class does:
/// - Inherits all the basic cross-validation options (like number of folds)
/// - Adds a new option to set the size of the validation set
/// 
/// This is useful because:
/// - It allows you to control how much of your data is used for validation in each Monte Carlo iteration
/// - You can adjust this to find the right balance between your training and validation set sizes
/// 
/// Think of it like deciding how to split a deck of cards for a card game - this option lets you choose 
/// how many cards go into each pile (training and validation) for each round of Monte Carlo testing.
/// </para>
/// </remarks>
public class MonteCarloValidationOptions : CrossValidationOptions
{
    /// <summary>
    /// Gets or sets the size of the validation set as a proportion of the total dataset.
    /// </summary>
    /// <value>
    /// A double value between 0 and 1, representing the fraction of the dataset to use for validation.
    /// The default value is 0.2 (20% of the data used for validation).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property determines what portion of the data will be used for validation in each Monte Carlo iteration.
    /// </para>
    /// <para><b>For Beginners:</b> This sets how much of your data is used to test your model in each round.
    /// 
    /// What it does:
    /// - A value of 0.2 means 20% of your data will be used for testing (validation) and 80% for training
    /// - You can adjust this value to change the split between training and validation data
    /// 
    /// For example:
    /// - If you have 1000 data points and set this to 0.2, each round will use 200 points for testing and 800 for training
    /// - If you set it to 0.3, each round will use 300 points for testing and 700 for training
    /// 
    /// Choose a value that gives you enough data for both training and testing your model effectively.
    /// </para>
    /// </remarks>
    public double ValidationSize { get; set; } = 0.2;
}
