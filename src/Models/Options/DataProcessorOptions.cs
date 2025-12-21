namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for data processing operations such as splitting datasets, normalization, and feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The DataProcessorOptions class provides settings that control how data is prepared before being used for
/// training, validating, and testing machine learning models. These settings affect data splitting ratios,
/// randomization, and preprocessing order.
/// </para>
/// <para><b>For Beginners:</b> Think of this class as your data preparation recipe. It tells the system how to
/// divide your data for different purposes (like training and testing), whether to shuffle it randomly, and
/// what order to perform certain data cleaning steps. Just like preparing ingredients before cooking, these
/// options help you prepare your data before training an AI model.</para>
/// </remarks>
public class DataProcessorOptions
{
    /// <summary>
    /// Gets or sets the percentage of data to use for training the model.
    /// </summary>
    /// <value>The training split percentage as a decimal between 0 and 1, defaulting to 0.7 (70%).</value>
    /// <remarks>
    /// <para>
    /// This value determines what portion of your dataset will be used to train the model. The training
    /// data is what the model learns from during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This is like deciding how much of your study time to spend learning new material.
    /// With the default value of 0.7, 70% of your data will be used to teach the model patterns and relationships.
    /// This is typically the largest portion because the model needs plenty of examples to learn from.</para>
    /// </remarks>
    public double TrainingSplitPercentage { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the percentage of data to use for validating the model during training.
    /// </summary>
    /// <value>The validation split percentage as a decimal between 0 and 1, defaulting to 0.15 (15%).</value>
    /// <remarks>
    /// <para>
    /// This value determines what portion of your dataset will be used to validate the model during training.
    /// Validation data helps tune the model's parameters and prevent overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> Think of validation data as practice tests while studying. With the default
    /// value of 0.15, 15% of your data will be used to check how well the model is learning and to make adjustments.
    /// The model doesn't learn directly from this data, but you use the results to guide the learning process,
    /// like adjusting your study strategy based on practice test results.</para>
    /// </remarks>
    public double ValidationSplitPercentage { get; set; } = 0.15;

    /// <summary>
    /// Gets or sets the percentage of data to reserve for final testing of the model.
    /// </summary>
    /// <value>The testing split percentage as a decimal between 0 and 1, defaulting to 0.15 (15%).</value>
    /// <remarks>
    /// <para>
    /// This value determines what portion of your dataset will be held back for final testing of the model.
    /// Testing data provides an unbiased evaluation of the model's performance on unseen data.
    /// </para>
    /// <para><b>For Beginners:</b> Testing data is like the final exam that tests what you've learned.
    /// With the default value of 0.15, 15% of your data will be set aside and only used at the very end
    /// to see how well your model performs on data it has never seen before. This gives you an honest
    /// assessment of how well your model will work in the real world.</para>
    /// </remarks>
    public double TestingSplitPercentage { get; set; } = 0.15;

    /// <summary>
    /// Gets or sets the seed value for random operations, ensuring reproducible results.
    /// </summary>
    /// <value>The random seed integer, defaulting to 42.</value>
    /// <remarks>
    /// <para>
    /// The random seed controls the randomization process when shuffling or splitting data. Using the same
    /// seed value ensures that random operations produce the same results across different runs.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as the starting point for a random number generator.
    /// Using the same seed (like the default 42) means you'll get the same "random" results every time you run
    /// your code. This is important for reproducing your results and debugging. If you want truly different
    /// random results each time, you can change this value or set it to a value based on the current time.</para>
    /// </remarks>
    public int RandomSeed { get; set; } = 42;

    /// <summary>
    /// Gets or sets whether data should be shuffled before splitting into train/validation/test sets.
    /// </summary>
    /// <value>True to shuffle before splitting, false to preserve the original order; defaults to true.</value>
    /// <remarks>
    /// <para>
    /// Most supervised ML problems benefit from shuffling before splitting to reduce accidental ordering bias.
    /// However, for time-series and other sequential datasets, shuffling can invalidate the evaluation because it leaks
    /// future information into the training set. In those cases, set this to false to preserve chronological order.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// If your data is time-ordered (like daily sales), you usually want to keep it in order when splitting so the model
    /// trains on the past and is tested on the future.
    /// </para>
    /// </remarks>
    public bool ShuffleBeforeSplit { get; set; } = true;

    /// <summary>
    /// Gets or sets whether data normalization should be performed before feature selection.
    /// </summary>
    /// <value>True to normalize before feature selection, false to normalize after; defaults to true.</value>
    /// <remarks>
    /// <para>
    /// This setting determines the order of preprocessing operations. When true, data is normalized (scaled)
    /// before features are selected. When false, feature selection happens first, followed by normalization.
    /// </para>
    /// <para><b>For Beginners:</b> Imagine you're preparing ingredients for cooking. This setting decides whether
    /// you first wash all vegetables (normalize) and then pick the best ones (feature selection), or if you
    /// first pick the vegetables you want and then wash only those. The default (true) means normalize everything
    /// first, which often works better because normalization can help identify which features are most important.</para>
    /// </remarks>
    public bool NormalizeBeforeFeatureSelection { get; set; } = true;
}
