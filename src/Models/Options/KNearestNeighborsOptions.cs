namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the K-Nearest Neighbors algorithm, which makes predictions based on the
/// values of the K closest data points in the training set.
/// </summary>
/// <remarks>
/// <para>
/// K-Nearest Neighbors (KNN) is a simple but powerful non-parametric algorithm that can be used for both
/// classification and regression. For regression, it predicts the value of a new data point by averaging
/// the values of its K nearest neighbors in the training data. The algorithm doesn't build an explicit
/// model during training - instead, it stores the training data and performs calculations at prediction time,
/// making it an example of "lazy learning."
/// </para>
/// <para><b>For Beginners:</b> K-Nearest Neighbors (KNN) is one of the simplest machine learning algorithms
/// to understand. It works on a very intuitive principle: things that are similar tend to have similar
/// outcomes.
/// 
/// Imagine you want to predict the price of a house. With KNN, you would:
/// 1. Find the K houses in your training data that are most similar to the house you're trying to price
///    (based on features like size, location, number of bedrooms, etc.)
/// 2. Take the average price of those K houses as your prediction
/// 
/// The "K" in KNN is simply how many neighbors you consider when making your prediction. If K=5, you look
/// at the 5 most similar houses.
/// 
/// KNN is different from many other algorithms because it doesn't build a complex model during training.
/// Instead, it simply remembers all the training examples and uses them directly when making predictions.
/// This makes it easy to understand but can make it slower for predictions on large datasets.
/// 
/// This class inherits from NonLinearRegressionOptions, so all the general non-linear regression settings
/// are also available. The additional setting specific to KNN lets you configure how many neighbors to
/// consider when making predictions.</para>
/// </remarks>
public class KNearestNeighborsOptions : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the number of nearest neighbors to consider when making predictions.
    /// </summary>
    /// <value>The number of neighbors, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines how many of the closest training examples are used to generate a prediction.
    /// A smaller K value makes the model more sensitive to local patterns but may lead to overfitting and
    /// sensitivity to noise. A larger K value produces smoother predictions but may miss important local
    /// patterns in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many "neighbors" (similar data points) the
    /// algorithm considers when making a prediction. With the default value of 5, the algorithm will find
    /// the 5 most similar examples in your training data and average their values to make its prediction.
    /// 
    /// Choosing the right value for K involves a trade-off:
    /// 
    /// - Smaller K values (like 1 or 3): The model becomes more detailed and can capture very specific
    ///   patterns in your data. However, it also becomes more sensitive to noise or unusual data points.
    ///   Think of it like asking just a few neighbors about local restaurants - you might get very specific
    ///   recommendations, but if one neighbor has unusual tastes, it could skew your results.
    /// 
    /// - Larger K values (like 10 or 20): The model makes smoother, more generalized predictions that are
    ///   less affected by individual outliers. However, it might miss important local patterns. This is like
    ///   asking many neighbors about restaurants - you'll get a more general consensus, but might miss some
    ///   hidden gems that only a few people know about.
    /// 
    /// The default value of 5 is a good starting point for many problems. You might want to try different
    /// values and see which gives the best performance for your specific dataset. As a rule of thumb, an
    /// odd number for K can be helpful when doing classification (to avoid ties in voting), but for
    /// regression tasks like predicting continuous values, both even and odd values work fine.</para>
    /// </remarks>
    public int K { get; set; } = 5;
}
