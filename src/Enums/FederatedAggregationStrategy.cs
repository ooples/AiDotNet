namespace AiDotNet.Enums;

/// <summary>
/// Defines strategies for aggregating models in federated learning.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Federated learning trains models across many devices without sharing 
/// raw data. This enum defines how to combine updates from all those separate devices into 
/// one improved model.
/// </para>
/// </remarks>
public enum FederatedAggregationStrategy
{
    /// <summary>
    /// Simple averaging of all client models.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Takes the average of all device models - treats every device equally, 
    /// like taking the average of test scores from all students.
    /// </remarks>
    FederatedAverage,

    /// <summary>
    /// Weighted average based on data size.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Devices with more data get more influence - like giving more weight 
    /// to teachers who have taught more students.
    /// </remarks>
    WeightedFederatedAverage,

    /// <summary>
    /// Median-based aggregation for robustness.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses the median instead of average - protects against devices that 
    /// might send bad updates, accidentally or maliciously.
    /// </remarks>
    MedianAggregation,

    /// <summary>
    /// Trimmed mean to remove outliers.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Removes the most extreme updates before averaging - like ignoring 
    /// the highest and lowest scores before calculating class average.
    /// </remarks>
    TrimmedMean,

    /// <summary>
    /// Krum aggregation for Byzantine tolerance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Selects updates that are most similar to others - protects against 
    /// devices trying to poison the model with bad data.
    /// </remarks>
    Krum,

    /// <summary>
    /// Multi-Krum for selecting multiple good updates.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Like Krum but selects several good updates instead of just one - 
    /// provides better performance while maintaining safety.
    /// </remarks>
    MultiKrum,

    /// <summary>
    /// Momentum-based federated aggregation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Remembers previous updates and uses them to smooth out learning - 
    /// like using past motion to predict future movement.
    /// </remarks>
    FederatedMomentum,

    /// <summary>
    /// Adaptive aggregation based on performance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Gives more weight to devices whose models perform better - like 
    /// listening more to experts who have proven track records.
    /// </remarks>
    AdaptiveAggregation,

    /// <summary>
    /// Secure aggregation with encryption.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Combines updates in a way that keeps individual updates private - 
    /// like counting votes without seeing individual ballots.
    /// </remarks>
    SecureAggregation,

    /// <summary>
    /// Hierarchical aggregation for large networks.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Aggregates in groups first, then combines groups - like having 
    /// regional managers summarize before reporting to headquarters.
    /// </remarks>
    HierarchicalAggregation,

    /// <summary>
    /// Clustering-based aggregation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Groups similar devices and aggregates within groups first - useful 
    /// when devices have different types of data.
    /// </remarks>
    ClusterAggregation,

    /// <summary>
    /// Personalized aggregation per client.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each device gets a slightly different model based on its needs - 
    /// like customizing a general recipe to local tastes.
    /// </remarks>
    PersonalizedAggregation,

    /// <summary>
    /// Asynchronous aggregation for stragglers.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Doesn't wait for all devices to finish - updates the model as 
    /// updates arrive, good for unreliable networks.
    /// </remarks>
    AsynchronousAggregation,

    /// <summary>
    /// Differential privacy aggregation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Adds carefully designed noise to protect individual privacy while 
    /// still learning useful patterns.
    /// </remarks>
    DifferentialPrivacy,

    /// <summary>
    /// Custom aggregation strategy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Allows you to implement your own method for combining federated updates.
    /// </remarks>
    Custom
}