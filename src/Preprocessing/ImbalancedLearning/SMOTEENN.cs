using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements SMOTE combined with Edited Nearest Neighbors (SMOTEENN) for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// SMOTEENN first applies SMOTE to oversample the minority class, then applies ENN to
/// remove noisy and borderline samples from both classes. This combination often produces
/// better results than either method alone.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a two-step process:
///
/// Step 1 (SMOTE): Create synthetic minority samples
/// - Increases minority class size
/// - Fills in the minority class region
///
/// Step 2 (ENN): Clean up the boundary
/// - Removes samples misclassified by their neighbors
/// - Cleans both majority AND synthetic minority samples
/// - Creates a cleaner decision boundary
///
/// Why combine them:
/// - SMOTE alone can create noisy synthetic samples
/// - ENN cleans up these noisy samples
/// - Result is a balanced AND clean dataset
///
/// Visual example:
/// ```
/// Original:    M M M M M M M M M m m
/// After SMOTE: M M M M M M M M M m m m m m m m
///                                     ^ synthetic
/// After ENN:   M M M M M M     m m m m m m
///                       ^     ^
///               Noisy samples removed from both classes
/// ```
///
/// References:
/// - Batista et al. (2004). "A Study of the Behavior of Several Methods for Balancing
///   Machine Learning Training Data"
/// </para>
/// </remarks>
public class SMOTEENN<T> : IResamplingStrategy<T>
{
    private readonly SMOTE<T> _smote;
    private readonly EditedNearestNeighbors<T> _enn;
    private ResamplingStatistics<T>? _lastStatistics;

    /// <summary>
    /// Gets the name of this resampling strategy.
    /// </summary>
    public string Name => "SMOTEENN";

    /// <summary>
    /// Initializes a new instance of the SMOTEENN class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio for SMOTE. Default is 1.0 (balanced).</param>
    /// <param name="kNeighborsSMOTE">Number of neighbors for SMOTE synthesis. Default is 5.</param>
    /// <param name="kNeighborsENN">Number of neighbors for ENN editing. Default is 3.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Example usage:
    ///
    /// <code>
    /// // Default settings
    /// var smoteenn = new SMOTEENN&lt;double&gt;();
    ///
    /// // Custom settings
    /// var smoteenn = new SMOTEENN&lt;double&gt;(
    ///     kNeighborsSMOTE: 7,  // More neighbors for synthesis
    ///     kNeighborsENN: 5     // More neighbors for cleaning
    /// );
    ///
    /// // Apply to data
    /// var (newX, newY) = smoteenn.Resample(trainX, trainY);
    /// </code>
    /// </para>
    /// </remarks>
    public SMOTEENN(
        double samplingStrategy = 1.0,
        int kNeighborsSMOTE = 5,
        int kNeighborsENN = 3,
        int? seed = null)
    {
        _smote = new SMOTE<T>(samplingStrategy, kNeighborsSMOTE, seed);
        _enn = new EditedNearestNeighbors<T>(kNeighborsENN);
    }

    /// <summary>
    /// Resamples the dataset using SMOTE followed by ENN.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <returns>The resampled features and labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Applies SMOTE to create synthetic minority samples
    /// 2. Applies ENN to clean up noisy samples from both classes
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

        // Step 2: Apply ENN
        var (finalX, finalY) = _enn.Resample(smoteX, smoteY);
        var ennStats = _enn.GetStatistics();

        // Calculate final statistics
        _lastStatistics.TotalResampledSamples = finalX.Rows;

        foreach (var kvp in ennStats.ResampledClassCounts)
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
