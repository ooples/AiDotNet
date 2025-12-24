using AiDotNet.Interfaces;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// Static factory class for creating data samplers with beginner-friendly methods.
/// </summary>
/// <remarks>
/// <para>
/// Samplers provides factory methods for creating various sampling strategies used
/// during training. These samplers control which data points are selected and in what order.
/// </para>
/// <para><b>For Beginners:</b> Sampling strategies determine how you pick data from your dataset.
/// Different strategies can help with:
/// - Class imbalance (use weighted sampling)
/// - Curriculum learning (start with easy examples, progress to hard)
/// - Active learning (focus on uncertain examples)
///
/// **Common Patterns:**
/// ```csharp
/// // Random sampling (default, good for most cases)
/// var sampler = Samplers.Random(dataSize);
///
/// // Balanced sampling for imbalanced classes
/// var sampler = Samplers.Balanced(labels, numClasses);
///
/// // Curriculum learning (easy to hard)
/// var sampler = Samplers.Curriculum(difficulties, totalEpochs);
/// ```
/// </para>
/// </remarks>
public static class Samplers
{
    #region Basic Samplers

    /// <summary>
    /// Creates a random sampler that shuffles data each epoch.
    /// </summary>
    /// <param name="dataSize">The total number of samples.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A random sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the default and most common sampler.
    /// It randomly shuffles your data each epoch, which helps the model generalize better.
    /// </para>
    /// </remarks>
    public static RandomSampler Random(int dataSize, int? seed = null)
    {
        return new RandomSampler(dataSize, seed);
    }

    /// <summary>
    /// Creates a sequential sampler that iterates through data in order.
    /// </summary>
    /// <param name="dataSize">The total number of samples.</param>
    /// <returns>A sequential sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you want to iterate through data
    /// in the same order every time. Useful for validation/testing or when order matters.
    /// </para>
    /// </remarks>
    public static SequentialSampler Sequential(int dataSize)
    {
        return new SequentialSampler(dataSize);
    }

    /// <summary>
    /// Creates a subset sampler that samples from specific indices.
    /// </summary>
    /// <param name="indices">The indices to sample from.</param>
    /// <param name="shuffle">Whether to shuffle the subset indices.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A subset sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you only want to train on a portion
    /// of your data, or when you've pre-computed a specific sampling order.
    /// </para>
    /// </remarks>
    public static SubsetSampler Subset(IEnumerable<int> indices, bool shuffle = false, int? seed = null)
    {
        return new SubsetSampler(indices, shuffle, seed);
    }

    #endregion

    #region Stratified and Weighted Samplers

    /// <summary>
    /// Creates a stratified sampler that maintains class proportions in each batch.
    /// </summary>
    /// <param name="labels">The class labels for each sample.</param>
    /// <param name="numClasses">The number of classes.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A stratified sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you want each batch to have
    /// the same proportion of classes as your full dataset. This helps prevent
    /// batches that are all one class.
    /// </para>
    /// </remarks>
    public static StratifiedSampler Stratified(
        IReadOnlyList<int> labels,
        int numClasses,
        int? seed = null)
    {
        return new StratifiedSampler(labels, numClasses, seed);
    }

    /// <summary>
    /// Creates a weighted sampler that samples based on per-sample weights.
    /// </summary>
    /// <typeparam name="T">The numeric type for weights.</typeparam>
    /// <param name="weights">The weight for each sample (higher = more likely to be sampled).</param>
    /// <param name="numSamples">Number of samples to draw per epoch.</param>
    /// <param name="replacement">Whether to sample with replacement. Default is true.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A weighted sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when some samples are more important than others.
    /// Higher weights make a sample more likely to be selected.
    /// </para>
    /// </remarks>
    public static WeightedSampler<T> Weighted<T>(
        IEnumerable<T> weights,
        int numSamples,
        bool replacement = true,
        int? seed = null)
    {
        return new WeightedSampler<T>(weights, numSamples, replacement, seed);
    }

    /// <summary>
    /// Creates a balanced sampler that oversamples minority classes.
    /// </summary>
    /// <param name="labels">The class labels for each sample.</param>
    /// <param name="numClasses">The number of classes.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A weighted sampler configured for class balancing.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for imbalanced datasets where some classes
    /// have many fewer examples than others. The sampler will select minority class
    /// examples more often to balance the training.
    ///
    /// **Example:**
    /// ```csharp
    /// // If you have 1000 samples of class A and 100 samples of class B,
    /// // this sampler will pick class B samples ~10x more often
    /// var sampler = Samplers.Balanced(labels, numClasses: 2);
    /// ```
    /// </para>
    /// </remarks>
    public static WeightedSampler<double> Balanced(
        IReadOnlyList<int> labels,
        int numClasses,
        int? seed = null)
    {
        var weights = WeightedSampler<double>.CreateBalancedWeights(labels, numClasses);
        return new WeightedSampler<double>(weights, labels.Count, replacement: false, seed);
    }

    #endregion

    #region Advanced Samplers

    /// <summary>
    /// Creates a curriculum learning sampler that starts with easy samples.
    /// </summary>
    /// <param name="difficulties">Difficulty score for each sample (0 = easiest, 1 = hardest).</param>
    /// <param name="totalEpochs">Total number of epochs for curriculum completion. Default is 100.</param>
    /// <param name="strategy">The curriculum progression strategy.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A curriculum sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Curriculum learning trains on easy examples first,
    /// then gradually introduces harder ones. This often leads to better and faster learning.
    ///
    /// **Example:**
    /// ```csharp
    /// // Difficulty scores: 0 = easy, 1 = hard
    /// var difficulties = ComputeDifficultyScores(data);
    /// var sampler = Samplers.Curriculum(difficulties);  // Uses default 100 epochs
    ///
    /// // In early epochs, mainly easy samples
    /// // In later epochs, mix of easy and hard samples
    /// ```
    /// </para>
    /// </remarks>
    public static CurriculumSampler<double> Curriculum(
        IEnumerable<double> difficulties,
        int totalEpochs = 100,
        CurriculumSampler<double>.CurriculumStrategy strategy = CurriculumSampler<double>.CurriculumStrategy.Linear,
        int? seed = null)
    {
        return new CurriculumSampler<double>(difficulties, totalEpochs, strategy, seed);
    }

    /// <summary>
    /// Creates a curriculum learning sampler that starts with easy samples.
    /// </summary>
    /// <typeparam name="T">The numeric type for difficulty scores.</typeparam>
    /// <param name="difficulties">Difficulty score for each sample (0 = easiest, 1 = hardest).</param>
    /// <param name="totalEpochs">Total number of epochs for curriculum completion.</param>
    /// <param name="strategy">The curriculum progression strategy.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A curriculum sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Curriculum learning trains on easy examples first,
    /// then gradually introduces harder ones. This often leads to better and faster learning.
    ///
    /// **Example:**
    /// ```csharp
    /// // Difficulty scores: 0 = easy, 1 = hard
    /// var difficulties = ComputeDifficultyScores(data);
    /// var sampler = Samplers.Curriculum(difficulties, totalEpochs: 100);
    ///
    /// // In early epochs, mainly easy samples
    /// // In later epochs, mix of easy and hard samples
    /// ```
    /// </para>
    /// </remarks>
    public static CurriculumSampler<T> Curriculum<T>(
        IEnumerable<T> difficulties,
        int totalEpochs,
        CurriculumSampler<T>.CurriculumStrategy strategy = CurriculumSampler<T>.CurriculumStrategy.Linear,
        int? seed = null)
    {
        return new CurriculumSampler<T>(difficulties, totalEpochs, strategy, seed);
    }

    /// <summary>
    /// Creates a self-paced learning sampler with default parameters.
    /// </summary>
    /// <param name="datasetSize">The total number of samples in the dataset.</param>
    /// <param name="initialLambda">Starting pace parameter (lower = stricter selection). Default is 0.1.</param>
    /// <param name="lambdaGrowthRate">How much lambda increases each epoch. Default is 0.1.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A self-paced sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like curriculum learning, but the difficulty is determined
    /// by the model's loss on each sample. Samples the model finds easy are included first.
    /// Call UpdateLoss() or UpdateLosses() after each batch to update sample losses.
    /// </para>
    /// </remarks>
    public static SelfPacedSampler<double> SelfPaced(
        int datasetSize,
        double initialLambda = 0.1,
        double lambdaGrowthRate = 0.1,
        int? seed = null)
    {
        return new SelfPacedSampler<double>(datasetSize, initialLambda, lambdaGrowthRate, seed);
    }

    /// <summary>
    /// Creates a self-paced learning sampler that adapts based on model performance.
    /// </summary>
    /// <typeparam name="T">The numeric type for losses.</typeparam>
    /// <param name="datasetSize">The total number of samples in the dataset.</param>
    /// <param name="initialLambda">Starting pace parameter (lower = stricter selection).</param>
    /// <param name="lambdaGrowthRate">How much lambda increases each epoch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A self-paced sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like curriculum learning, but the difficulty is determined
    /// by the model's loss on each sample. Samples the model finds easy are included first.
    /// Call UpdateLoss() or UpdateLosses() after each batch to update sample losses.
    /// </para>
    /// </remarks>
    public static SelfPacedSampler<T> SelfPaced<T>(
        int datasetSize,
        T initialLambda,
        T lambdaGrowthRate,
        int? seed = null)
    {
        return new SelfPacedSampler<T>(datasetSize, initialLambda, lambdaGrowthRate, seed);
    }

    /// <summary>
    /// Creates an importance sampler that prioritizes high-loss samples.
    /// </summary>
    /// <param name="datasetSize">The total number of samples in the dataset.</param>
    /// <param name="smoothingFactor">Smoothing factor to prevent extreme sampling (0.1-0.5 recommended).</param>
    /// <param name="stabilize">Whether to clip extreme importance values.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>An importance sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Importance sampling focuses training on samples the model
    /// currently gets wrong (high loss). This can speed up training by focusing on hard examples.
    /// Call SetImportances() or UpdateImportance() after each batch to update importance scores.
    /// </para>
    /// </remarks>
    public static ImportanceSampler<double> Importance(
        int datasetSize,
        double smoothingFactor = 0.2,
        bool stabilize = true,
        int? seed = null)
    {
        return new ImportanceSampler<double>(datasetSize, smoothingFactor, stabilize, seed);
    }

    /// <summary>
    /// Creates an importance sampler that prioritizes high-loss samples.
    /// </summary>
    /// <typeparam name="T">The numeric type for importance scores.</typeparam>
    /// <param name="datasetSize">The total number of samples in the dataset.</param>
    /// <param name="smoothingFactor">Smoothing factor to prevent extreme sampling (0.1-0.5 recommended).</param>
    /// <param name="stabilize">Whether to clip extreme importance values.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>An importance sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Importance sampling focuses training on samples the model
    /// currently gets wrong (high loss). This can speed up training by focusing on hard examples.
    /// Call SetImportances() or UpdateImportance() after each batch to update importance scores.
    /// </para>
    /// </remarks>
    public static ImportanceSampler<T> Importance<T>(
        int datasetSize,
        double smoothingFactor = 0.2,
        bool stabilize = true,
        int? seed = null)
    {
        return new ImportanceSampler<T>(datasetSize, smoothingFactor, stabilize, seed);
    }

    /// <summary>
    /// Creates an active learning sampler that prioritizes uncertain samples.
    /// </summary>
    /// <param name="datasetSize">The total number of samples in the dataset.</param>
    /// <param name="strategy">The active learning selection strategy.</param>
    /// <param name="diversityWeight">Weight for diversity in hybrid strategy (0-1).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>An active learning sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Active learning prioritizes samples where the model is most
    /// uncertain. This is especially useful when you can only label a limited number of samples.
    /// Call MarkAsLabeled() to mark samples that have been labeled, and UpdateUncertainty()
    /// to update uncertainty scores after each batch.
    ///
    /// **Common strategies:**
    /// - Uncertainty: Focus on samples with highest uncertainty (entropy, margin, etc.)
    /// - Diversity: Focus on diverse samples using clustering
    /// - Hybrid: Combine uncertainty and diversity
    /// </para>
    /// </remarks>
    public static ActiveLearningSampler<double> ActiveLearning(
        int datasetSize,
        ActiveLearningSampler<double>.SelectionStrategy strategy = ActiveLearningSampler<double>.SelectionStrategy.Uncertainty,
        double diversityWeight = 0.3,
        int? seed = null)
    {
        return new ActiveLearningSampler<double>(datasetSize, strategy, diversityWeight, seed);
    }

    /// <summary>
    /// Creates an active learning sampler that prioritizes uncertain samples.
    /// </summary>
    /// <typeparam name="T">The numeric type for uncertainty scores.</typeparam>
    /// <param name="datasetSize">The total number of samples in the dataset.</param>
    /// <param name="strategy">The active learning selection strategy.</param>
    /// <param name="diversityWeight">Weight for diversity in hybrid strategy (0-1).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>An active learning sampler.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Active learning prioritizes samples where the model is most
    /// uncertain. This is especially useful when you can only label a limited number of samples.
    /// Call MarkAsLabeled() to mark samples that have been labeled, and UpdateUncertainty()
    /// to update uncertainty scores after each batch.
    ///
    /// **Common strategies:**
    /// - Uncertainty: Focus on samples with highest uncertainty (entropy, margin, etc.)
    /// - Diversity: Focus on diverse samples using clustering
    /// - Hybrid: Combine uncertainty and diversity
    /// </para>
    /// </remarks>
    public static ActiveLearningSampler<T> ActiveLearning<T>(
        int datasetSize,
        ActiveLearningSampler<T>.SelectionStrategy strategy = ActiveLearningSampler<T>.SelectionStrategy.Uncertainty,
        double diversityWeight = 0.3,
        int? seed = null)
    {
        return new ActiveLearningSampler<T>(datasetSize, strategy, diversityWeight, seed);
    }

    #endregion
}
