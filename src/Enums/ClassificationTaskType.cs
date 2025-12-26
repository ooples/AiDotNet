namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of classification task being performed.
/// </summary>
/// <remarks>
/// <para>
/// Classification task types determine how the model interprets the target variable
/// and how predictions are structured. Different task types require different output
/// formats and loss functions.
/// </para>
/// <para><b>For Beginners:</b> Classification is about putting things into categories.
///
/// Think of it like sorting mail:
/// - Binary: Is this spam or not spam? (2 categories)
/// - MultiClass: Is this a bill, letter, package, or advertisement? (multiple exclusive categories)
/// - MultiLabel: Mark all that apply: urgent, personal, work-related (multiple overlapping labels)
/// - Ordinal: Rate satisfaction: Poor, Fair, Good, Excellent (ordered categories)
///
/// The task type tells the model what kind of answer you're expecting.
/// </para>
/// </remarks>
public enum ClassificationTaskType
{
    /// <summary>
    /// Binary classification with exactly two classes (e.g., spam/not-spam, positive/negative).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Binary classification is the simplest form of classification where each sample belongs
    /// to exactly one of two classes. The output is typically a single probability value
    /// representing the likelihood of the positive class.
    /// </para>
    /// <para><b>For Beginners:</b> Binary classification answers yes/no questions.
    ///
    /// Examples:
    /// - Is this email spam? (Yes/No)
    /// - Will this customer churn? (Yes/No)
    /// - Is this transaction fraudulent? (Yes/No)
    /// - Does this patient have the disease? (Yes/No)
    ///
    /// The model outputs a probability between 0 and 1, and you typically use a
    /// threshold (usually 0.5) to make the final yes/no decision.
    /// </para>
    /// </remarks>
    Binary = 0,

    /// <summary>
    /// Multi-class classification where each sample belongs to exactly one of multiple classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multi-class classification extends binary classification to multiple mutually exclusive
    /// classes. The output is a probability distribution over all classes, and the predicted
    /// class is typically the one with the highest probability.
    /// </para>
    /// <para><b>For Beginners:</b> Multi-class classification picks one category from many options.
    ///
    /// Examples:
    /// - What animal is in this image? (Cat, Dog, Bird, Fish, etc.)
    /// - What genre is this movie? (Action, Comedy, Drama, Horror, etc.)
    /// - What digit is this? (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    /// - What language is this text? (English, Spanish, French, etc.)
    ///
    /// Key difference from multi-label: each sample can only belong to ONE class.
    /// A movie can't be both "Comedy" and "Drama" in multi-class classification.
    /// </para>
    /// </remarks>
    MultiClass = 1,

    /// <summary>
    /// Multi-label classification where each sample can belong to multiple classes simultaneously.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multi-label classification allows each sample to be assigned to multiple classes
    /// at the same time. The output is a set of binary predictions (one for each possible label),
    /// indicating which labels apply to the sample.
    /// </para>
    /// <para><b>For Beginners:</b> Multi-label classification assigns multiple tags to each item.
    ///
    /// Examples:
    /// - What topics are in this article? (Politics, Sports, Technology - multiple can apply)
    /// - What objects are in this image? (Person, Car, Tree - multiple can appear)
    /// - What genres does this movie belong to? (Action-Comedy, Drama-Romance)
    /// - What symptoms does this patient have? (Fever, Cough, Fatigue)
    ///
    /// Key difference from multi-class: an item CAN belong to multiple categories at once.
    /// A movie can be both "Comedy" AND "Drama" in multi-label classification.
    /// </para>
    /// </remarks>
    MultiLabel = 2,

    /// <summary>
    /// Ordinal classification where classes have a natural ordering.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Ordinal classification is similar to multi-class classification, but the classes
    /// have an inherent order. The model should respect this ordering, penalizing
    /// predictions that are "far" from the true class more than predictions that are "close."
    /// </para>
    /// <para><b>For Beginners:</b> Ordinal classification predicts categories that have a natural order.
    ///
    /// Examples:
    /// - Customer satisfaction: Poor, Fair, Good, Excellent (ordered from worst to best)
    /// - Product rating: 1 star, 2 stars, 3 stars, 4 stars, 5 stars
    /// - Education level: High School, Bachelor's, Master's, PhD
    /// - Pain level: None, Mild, Moderate, Severe
    ///
    /// The key insight is that predicting "Fair" when the answer is "Good" is less wrong
    /// than predicting "Poor" - they're both wrong, but one is closer to the truth.
    ///
    /// Unlike regular multi-class where all mistakes are equally wrong, ordinal classification
    /// considers how far off the prediction is from the correct answer.
    /// </para>
    /// </remarks>
    Ordinal = 3
}
