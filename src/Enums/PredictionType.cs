namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of prediction task that a machine learning model performs.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This enum helps you tell the library what kind of prediction you're trying to make.
/// Think of it as telling the AI system what type of question you're asking:
/// 
/// - Are you asking a yes/no question? Use Binary.
/// - Are you asking "how much" or "what value"? Use Regression.
/// 
/// Choosing the right prediction type helps the AI model understand what you're trying to accomplish
/// and use the appropriate techniques for your specific problem.
/// </remarks>
public enum PredictionType
{
    /// <summary>
    /// Represents a binary classification task where the output is one of two possible classes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this when your prediction has only two possible outcomes, like:
    /// - Yes or No
    /// - True or False
    /// - Spam or Not Spam
    /// - Positive or Negative
    /// 
    /// Binary predictions typically output a probability between 0 and 1, where:
    /// - Values closer to 0 indicate the first class (e.g., "No")
    /// - Values closer to 1 indicate the second class (e.g., "Yes")
    /// 
    /// Examples: Email spam detection, disease diagnosis, fraud detection
    /// </remarks>
    Binary,

    /// <summary>
    /// Represents a regression task where the output is a continuous numerical value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this when your prediction is a number that can take any value within a range, like:
    /// - Price of a house
    /// - Temperature tomorrow
    /// - Number of sales next month
    /// - Age of a person from their photo
    /// 
    /// Unlike Binary prediction, Regression doesn't have fixed categories - it predicts
    /// actual numerical values that can be any number (like 42.5, 1000, or -3.14).
    /// 
    /// Examples: Price prediction, weather forecasting, age estimation, stock market prediction
    /// </remarks>
    Regression
}