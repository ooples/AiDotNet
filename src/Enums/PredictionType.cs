namespace AiDotNet.Enums;

/// <summary>
/// Defines the different types of predictions that can be made by machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This enum helps categorize the different kinds of outputs a machine learning model can produce.
/// Each prediction type requires different evaluation metrics and has different interpretation methods.
/// </para>
/// </remarks>
public enum PredictionType
{
    /// <summary>
    /// Predicting continuous numerical values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Regression predictions estimate a number on a continuous scale, like house prices,
    /// temperature, or a person's age. These models try to predict exact values rather than categories.
    /// 
    /// Examples:
    /// - Predicting a house price ($350,000)
    /// - Estimating a person's income
    /// - Forecasting temperature
    /// 
    /// Common evaluation metrics include MAE, RMSE, and R² (R-squared).
    /// </para>
    /// </remarks>
    Regression,

    /// <summary>
    /// Predicting one of two possible categories or classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Binary classification predictions categorize data into one of two classes,
    /// like yes/no, spam/not spam, or positive/negative.
    /// 
    /// Examples:
    /// - Email spam detection (spam or not spam)
    /// - Medical diagnosis (disease present or not)
    /// - Fraud detection (fraudulent or legitimate)
    /// 
    /// Common evaluation metrics include accuracy, precision, recall, and F1 score.
    /// </para>
    /// </remarks>
    BinaryClassification,

    /// <summary>
    /// Predicting one of three or more possible categories or classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiclass classification predictions categorize data into three or more classes.
    /// 
    /// Examples:
    /// - Identifying which animal appears in an image
    /// - Classifying documents by topic
    /// - Determining which language a text is written in
    /// 
    /// Common evaluation metrics include accuracy, precision, recall, and F1 score,
    /// but they're often calculated per class or using strategies like one-vs-rest.
    /// </para>
    /// </remarks>
    MulticlassClassification,

    /// <summary>
    /// Predicting values that change over time in a sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Time series predictions forecast how values will change over time.
    /// They account for temporal patterns like trends, seasonality, and cycles.
    /// 
    /// Examples:
    /// - Stock price forecasting
    /// - Weather prediction
    /// - Sales forecasting
    /// - Energy demand prediction
    /// 
    /// Common evaluation metrics include MAE, RMSE, and specialized metrics like
    /// MAPE and Theil's U Statistic. Time series predictions often use special intervals
    /// like forecast intervals that account for increasing uncertainty over time.
    /// </para>
    /// </remarks>
    TimeSeries,

    /// <summary>
    /// Predicting probabilities for multiple classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Probabilistic classification assigns a probability to each possible class,
    /// rather than just picking the most likely class.
    /// 
    /// Examples:
    /// - Predicting the probability of different diseases based on symptoms
    /// - Estimating the likelihood of different customer actions
    /// - Assigning confidence levels to different possible interpretations of an image
    /// 
    /// Common evaluation metrics include log loss, Brier score, and calibration plots.
    /// </para>
    /// </remarks>
    ProbabilisticClassification,

    /// <summary>
    /// Predicting multiple labels for a single instance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-label classification assigns multiple categories to a single item,
    /// unlike multiclass classification where each item belongs to exactly one category.
    /// 
    /// Examples:
    /// - Tagging an article with multiple topics (e.g., both "Technology" and "Business")
    /// - Identifying multiple objects in a single image
    /// - Classifying a movie into multiple genres
    /// 
    /// Common evaluation metrics include Hamming loss, subset accuracy, and F1 score variants.
    /// </para>
    /// </remarks>
    MultiLabelClassification,

    /// <summary>
    /// Predicting distributions rather than single values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Distribution prediction estimates a range of possible values along with their probabilities,
    /// providing a complete picture of uncertainty.
    /// 
    /// Examples:
    /// - Predicting a probability distribution for future temperature
    /// - Estimating the range of possible sales figures with confidence levels
    /// - Forecasting demand with uncertainty quantification
    /// 
    /// Common evaluation metrics include negative log-likelihood, Kullback-Leibler divergence,
    /// and calibration metrics.
    /// </para>
    /// </remarks>
    Distribution,

    /// <summary>
    /// Predicting the optimal ordering of items.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ranking predictions focus on the correct order of items rather than
    /// their absolute scores or categories.
    /// 
    /// Examples:
    /// - Search result ranking
    /// - Product recommendations
    /// - Content personalization
    /// 
    /// Common evaluation metrics include Mean Average Precision (MAP), Normalized Discounted
    /// Cumulative Gain (NDCG), and Mean Reciprocal Rank (MRR).
    /// </para>
    /// </remarks>
    Ranking
}