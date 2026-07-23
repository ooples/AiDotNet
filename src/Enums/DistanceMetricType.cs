namespace AiDotNet.Enums;

/// <summary>
/// Represents different methods for measuring the distance or similarity between data points.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Distance metrics are ways to measure how similar or different two data points are.
/// 
/// Think of distance metrics like different ways to measure the distance between two cities:
/// - As the crow flies (straight line)
/// - By following streets (Manhattan)
/// - By considering terrain and obstacles (more complex metrics)
/// 
/// In machine learning, we use these metrics to:
/// - Group similar items together (clustering)
/// - Find nearest neighbors
/// - Measure how well our model is performing
/// 
/// Different distance metrics work better for different types of data and problems.
/// For example, Euclidean distance works well for continuous numerical data,
/// while Jaccard distance is better for comparing sets.
/// </para>
/// </remarks>
public enum DistanceMetricType
{
    /// <summary>
    /// The straight-line distance between two points in Euclidean space (also known as L2 distance).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Euclidean distance is the most common and intuitive distance measure - it's the straight-line 
    /// distance between two points "as the crow flies."
    /// 
    /// Formula: sqrt((x2-x1)² + (y2-y1)² + ...)
    /// 
    /// Think of it as measuring distance with a ruler in a straight line.
    /// 
    /// Best used for:
    /// - Continuous numerical data
    /// - When features have similar scales
    /// - When the data space is relatively low-dimensional
    /// 
    /// Examples:
    /// - Distance between physical locations
    /// - Similarity between customer profiles with numerical attributes
    /// - Image similarity when using pixel values
    /// 
    /// Note: Euclidean distance is sensitive to the scale of the features, so normalization
    /// is often required before using this metric.
    /// </para>
    /// </remarks>
    Euclidean,

    /// <summary>
    /// The sum of absolute differences between coordinates (also known as L1 distance or taxicab distance).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Manhattan distance measures the distance between two points by summing the absolute differences 
    /// of their coordinates.
    /// 
    /// Formula: |x2-x1| + |y2-y1| + ...
    /// 
    /// Think of it as the distance a taxi would drive in a city with a grid layout, where you can only 
    /// travel along the streets (horizontal and vertical movements).
    /// 
    /// Best used for:
    /// - Grid-based problems
    /// - When diagonal movement costs the same as horizontal/vertical
    /// - Data with discrete or binary features
    /// - When you want to reduce the influence of outliers
    /// 
    /// Examples:
    /// - Navigation in city streets
    /// - Feature comparison when features are independent
    /// - Image processing with pixel-wise comparisons
    /// 
    /// Manhattan distance is less sensitive to outliers than Euclidean distance.
    /// </para>
    /// </remarks>
    Manhattan,

    /// <summary>
    /// Measures the cosine of the angle between two vectors, focusing on orientation rather than magnitude.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cosine similarity measures the angle between two vectors, ignoring their magnitude (length).
    /// 
    /// Think of it as comparing the direction two people are facing, regardless of how far they've walked.
    /// Two people facing north are similar (cosine = 1), even if one walked 1 mile and the other 100 miles.
    /// People facing opposite directions have maximum dissimilarity (cosine = -1).
    /// 
    /// Formula: cos(?) = (A·B)/(||A||·||B||)
    /// 
    /// Best used for:
    /// - Text documents (comparing document topics regardless of length)
    /// - Recommendation systems
    /// - High-dimensional sparse data
    /// - When the magnitude of vectors doesn't matter, only their direction
    /// 
    /// Examples:
    /// - Document similarity in text analysis
    /// - User preference matching in recommendations
    /// - Image feature comparison
    /// 
    /// Note: Cosine similarity ranges from -1 (completely opposite) to 1 (exactly the same),
    /// with 0 indicating orthogonality (no correlation). To convert to a distance, use 1 - cosine_similarity.
    /// </para>
    /// </remarks>
    Cosine,

    /// <summary>
    /// Measures dissimilarity between sets by comparing elements they share versus elements they don't.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Jaccard distance measures how different two sets are by comparing what they have in common 
    /// versus what they don't.
    /// 
    /// Think of it as comparing two shopping lists:
    /// - How many items appear on both lists?
    /// - How many items appear on at least one list?
    /// - The ratio of these gives you the similarity
    /// 
    /// Formula: 1 - |AnB|/|A?B| (1 minus the size of intersection divided by size of union)
    /// 
    /// Best used for:
    /// - Binary data (presence/absence)
    /// - Set comparisons
    /// - Categorical data
    /// - Sparse binary vectors
    /// 
    /// Examples:
    /// - Comparing which movies two users have watched
    /// - Measuring similarity between text documents based on shared words
    /// - Comparing species presence/absence in ecological samples
    /// - Comparing shopping baskets
    /// 
    /// Jaccard distance ranges from 0 (identical sets) to 1 (completely different sets).
    /// </para>
    /// </remarks>
    Jaccard,

    /// <summary>
    /// Counts the number of positions at which corresponding elements differ (used for strings or binary vectors).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hamming distance counts how many positions have different values when comparing two strings 
    /// or sequences of equal length.
    /// 
    /// Think of it as comparing two multiple-choice tests and counting how many answers are different.
    /// 
    /// For example, comparing "CART" and "PART":
    /// - Position 1: C vs P (different) ? +1
    /// - Position 2: A vs A (same) ? +0
    /// - Position 3: R vs R (same) ? +0
    /// - Position 4: T vs T (same) ? +0
    /// - Hamming distance = 1
    /// 
    /// Best used for:
    /// - Strings of equal length
    /// - Binary vectors
    /// - Error detection in communication
    /// - Genetic sequence comparison
    /// 
    /// Examples:
    /// - DNA sequence comparison
    /// - Error detection in data transmission
    /// - Comparing fixed-length binary feature vectors
    /// - Spell checking for words of the same length
    /// 
    /// Note: Hamming distance requires sequences of equal length.
    /// </para>
    /// </remarks>
    Hamming,

    /// <summary>
    /// Measures distance while accounting for correlations between variables and their relative importance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mahalanobis distance is an advanced metric that takes into account how variables 
    /// are related to each other (their correlations).
    /// 
    /// Unlike Euclidean distance, which treats all dimensions equally, Mahalanobis distance:
    /// - Gives less weight to variables that have high variance
    /// - Accounts for correlations between variables
    /// - Adjusts for the shape of the data distribution
    /// 
    /// Think of it as measuring distance while considering the natural "shape" of your data:
    /// - If you have height and weight data, these are correlated (taller people tend to weigh more)
    /// - Mahalanobis distance accounts for this correlation when measuring similarity
    /// 
    /// Formula: sqrt((x-µ)? S?¹ (x-µ)) where S is the covariance matrix
    /// 
    /// Best used for:
    /// - Multivariate data with correlated features
    /// - Outlier detection
    /// - Classification problems
    /// - When features have different scales and variances
    /// 
    /// Examples:
    /// - Anomaly detection in multivariate systems
    /// - Face recognition
    /// - Quality control in manufacturing
    /// 
    /// Note: Mahalanobis distance requires computing the covariance matrix of your data,
    /// which can be computationally expensive for high-dimensional data.
    /// </para>
    /// </remarks>
    Mahalanobis
}
