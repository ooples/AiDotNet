using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements SMOTE combined with Tomek Links (SMOTETomek) for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// SMOTETomek first applies SMOTE to oversample the minority class, then applies Tomek Links
/// removal to clean up the decision boundary. This is less aggressive than SMOTEENN.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a two-step process:
///
/// Step 1 (SMOTE): Create synthetic minority samples
/// - Increases minority class size
/// - Fills in the minority class region
///
/// Step 2 (Tomek Links): Remove specific boundary pairs
/// - Only removes samples that form Tomek links
/// - More targeted than ENN
/// - Preserves more data
///
/// Comparison with SMOTEENN:
/// - SMOTETomek: Removes fewer samples, more conservative
/// - SMOTEENN: Removes more samples, more aggressive cleaning
///
/// Visual example:
/// ```
/// Original:    M M M M M M M M M m m
/// After SMOTE: M M M M M M M M M m m m m m m m
///                              ^         ^
///                              Tomek link pair
/// After Tomek: M M M M M M M M   m m m m m m m
///                              ^
///                      Only the majority sample of the pair removed
/// ```
///
/// When to use:
/// - When you want to balance but preserve more data
/// - When ENN is too aggressive for your dataset
/// - As a middle ground between pure SMOTE and SMOTEENN
///
/// References:
/// - Batista et al. (2004). "A Study of the Behavior of Several Methods for Balancing
///   Machine Learning Training Data"
/// </para>
/// </remarks>
public class SMOTETomek<T> : IResamplingStrategy<T>
{
    private readonly SMOTE<T> _smote;
    private readonly TomekLinks<T> _tomek;
    private ResamplingStatistics<T>? _lastStatistics;

    /// <summary>
    /// Gets the name of this resampling strategy.
    /// </summary>
    public string Name => "SMOTETomek";

    /// <summary>
    /// Initializes a new instance of the SMOTETomek class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio for SMOTE. Default is 1.0 (balanced).</param>
    /// <param name="kNeighbors">Number of neighbors for SMOTE synthesis. Default is 5.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Example usage:
    ///
    /// <code>
    /// // Default settings
    /// var smoteTomek = new SMOTETomek&lt;double&gt;();
    ///
    /// // Custom SMOTE settings
    /// var smoteTomek = new SMOTETomek&lt;double&gt;(
    ///     samplingStrategy: 0.8,  // 80% balance
    ///     kNeighbors: 7           // More neighbors
    /// );
    ///
    /// // Apply to data
    /// var (newX, newY) = smoteTomek.Resample(trainX, trainY);
    /// </code>
    /// </para>
    /// </remarks>
    public SMOTETomek(double samplingStrategy = 1.0, int kNeighbors = 5, int? seed = null)
    {
        _smote = new SMOTE<T>(samplingStrategy, kNeighbors, seed);
        _tomek = new TomekLinks<T>();
    }

    /// <summary>
    /// Resamples the dataset using SMOTE followed by Tomek Links removal.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <returns>The resampled features and labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Applies SMOTE to create synthetic minority samples
    /// 2. Applies Tomek Links to remove boundary pairs
    /// 3. Returns the balanced and cleaned dataset
    /// </para>
    /// </remarks>
    public (Matrix<T> resampledX, Vector<T> resampledY) Resample(Matrix<T> x, Vector<T> y)
    {
        // Initialize statistics
        _lastStatistics = new ResamplingStatistics<T>
        {
            TotalOriginalSamples = x.Rows
        };

        // Step 1: Apply SMOTE
        var (smoteX, smoteY) = _smote.Resample(x, y);
        var smoteStats = _smote.GetStatistics();

        // Record original class counts
        foreach (var kvp in smoteStats.OriginalClassCounts)
        {
            _lastStatistics.OriginalClassCounts[kvp.Key] = kvp.Value;
        }

        // Step 2: Apply Tomek Links
        var (finalX, finalY) = _tomek.Resample(smoteX, smoteY);
        var tomekStats = _tomek.GetStatistics();

        // Calculate final statistics
        _lastStatistics.TotalResampledSamples = finalX.Rows;

        foreach (var kvp in tomekStats.ResampledClassCounts)
        {
            _lastStatistics.ResampledClassCounts[kvp.Key] = kvp.Value;

            // Calculate net samples added/removed
            int original = _lastStatistics.OriginalClassCounts.ContainsKey(kvp.Key)
                ? _lastStatistics.OriginalClassCounts[kvp.Key]
                : 0;
            int final = kvp.Value;
            int diff = final - original;

            if (diff > 0)
            {
                _lastStatistics.SamplesAddedPerClass[kvp.Key] = diff;
                _lastStatistics.SamplesRemovedPerClass[kvp.Key] = 0;
            }
            else
            {
                _lastStatistics.SamplesAddedPerClass[kvp.Key] = 0;
                _lastStatistics.SamplesRemovedPerClass[kvp.Key] = -diff;
            }
        }

        return (finalX, finalY);
    }

    /// <summary>
    /// Gets statistics about the last resampling operation.
    /// </summary>
    public ResamplingStatistics<T> GetStatistics()
    {
        return _lastStatistics ?? new ResamplingStatistics<T>();
    }
}
