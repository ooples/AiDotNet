using AiDotNet.Augmentation.Interfaces;
using AiDotNet.Augmentation.Policies;

namespace AiDotNet.Augmentation.TTA;

/// <summary>
/// Configuration for Test-Time Augmentation - a technique to improve prediction accuracy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Test-Time Augmentation (TTA) is a simple but powerful trick
/// to make your model's predictions more accurate. Here's how it works:
///
/// <b>Without TTA:</b>
/// 1. You give the model one image
/// 2. The model makes one prediction
/// 3. Done - but what if that prediction is slightly wrong?
///
/// <b>With TTA:</b>
/// 1. You give the model one image
/// 2. The system creates 5 variations (flipped, rotated slightly, etc.)
/// 3. The model makes 5 predictions (one for each variation)
/// 4. The predictions are averaged together
/// 5. The final answer is more reliable because it's based on multiple views
///
/// <b>Real-world analogy:</b> Instead of asking one doctor for a diagnosis, you ask
/// 5 doctors and take their consensus. You're more likely to get the right answer.
///
/// <b>Trade-off:</b> TTA makes predictions more accurate but also 5x slower (since
/// you're making 5 predictions instead of 1). Use it when accuracy matters more than speed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (usually double or float).</typeparam>
/// <typeparam name="TData">The data type being augmented (e.g., ImageTensor for images).</typeparam>
public class TestTimeAugmentationConfiguration<T, TData> : ITTAConfiguration<T, TData>
{
    /// <summary>
    /// Gets or sets the augmentation pipeline that creates variations of your input.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This defines what transformations to apply to your input.
    /// Common choices for images:
    /// - Horizontal flip (mirror image left-to-right)
    /// - Small rotations (-5 to +5 degrees)
    /// - Slight zoom in/out
    ///
    /// If you don't set this, TTA will still work but with no augmentations applied,
    /// which means it just makes the same prediction multiple times (not useful).
    /// </para>
    /// </remarks>
    public IAugmentationPolicy<T, TData>? Pipeline { get; set; }

    /// <summary>
    /// Gets or sets whether Test-Time Augmentation is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c> (enabled)</para>
    /// <para>Set to <c>false</c> to temporarily disable TTA without removing the configuration.</para>
    /// </remarks>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Gets or sets how many augmented versions to create.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>5</c> (research-backed sweet spot)</para>
    /// <para><b>For Beginners:</b> This controls how many variations of your input to test.
    ///
    /// - <b>Higher values (8-10):</b> More accurate but slower. Use for competitions or critical predictions.
    /// - <b>Lower values (3-4):</b> Faster but less benefit. Use when speed matters.
    /// - <b>Default (5):</b> Good balance for most use cases.
    ///
    /// Research shows diminishing returns beyond 10 augmentations.
    /// </para>
    /// </remarks>
    public int NumberOfAugmentations { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to include the original (unaugmented) input in predictions.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c> (include original)</para>
    /// <para><b>For Beginners:</b> When enabled, the model sees both:
    /// - The original image (unchanged)
    /// - Plus N augmented versions
    ///
    /// This ensures the "ground truth" view is always considered alongside the variations.
    /// Almost always leave this enabled unless you have a specific reason not to.
    /// </para>
    /// </remarks>
    public bool IncludeOriginal { get; set; } = true;

    /// <summary>
    /// Gets or sets how to combine predictions from all the augmented versions.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>Mean</c> (average predictions together)</para>
    /// <para><b>For Beginners:</b> See <see cref="PredictionAggregationMethod"/> for details
    /// on each option. Mean (averaging) works well for most use cases.</para>
    /// </remarks>
    public PredictionAggregationMethod AggregationMethod { get; set; } = PredictionAggregationMethod.Mean;

    /// <summary>
    /// Gets or sets a random seed for reproducible augmentations.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>null</c> (random augmentations each time)</para>
    /// <para><b>For Beginners:</b> Setting a seed means you get the same augmentations every time.
    /// This is useful for debugging or when you need reproducible results.
    /// Leave as null for production use where variety is beneficial.
    /// </para>
    /// </remarks>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to reverse spatial transformations in predictions.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c> (apply inverse transforms)</para>
    /// <para><b>For Beginners:</b> This is important for tasks that output locations, like:
    /// - Object detection (bounding boxes)
    /// - Segmentation (pixel masks)
    /// - Pose estimation (keypoint locations)
    ///
    /// If you flip an image and detect a cat on the left side, the inverse transform
    /// moves that detection back to the right side (where the cat really is).
    ///
    /// For simple classification ("is this a cat?"), this setting doesn't matter.
    /// </para>
    /// </remarks>
    public bool ApplyInverseTransforms { get; set; } = true;

    /// <summary>
    /// Gets or sets a minimum confidence threshold for including predictions.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>null</c> (include all predictions)</para>
    /// <para><b>For Beginners:</b> If set, predictions with confidence below this threshold
    /// are ignored when computing the final answer. For example, if set to 0.5, any
    /// prediction with less than 50% confidence is thrown out.
    ///
    /// Leave as null unless you're seeing bad predictions from heavily augmented inputs.
    /// </para>
    /// </remarks>
    public double? ConfidenceThreshold { get; set; }

    /// <summary>
    /// Creates a new Test-Time Augmentation configuration with industry-standard defaults.
    /// </summary>
    /// <remarks>
    /// <para><b>Default settings:</b>
    /// - 5 augmentations (research-backed optimal count)
    /// - Mean aggregation (most robust for general use)
    /// - Original included (best practice)
    /// - Inverse transforms enabled (correct for spatial outputs)
    /// </para>
    /// </remarks>
    public TestTimeAugmentationConfiguration()
    {
    }

    /// <summary>
    /// Creates a Test-Time Augmentation configuration with a specific pipeline.
    /// </summary>
    /// <param name="pipeline">The augmentation pipeline to use.</param>
    /// <param name="numberOfAugmentations">How many augmented versions to create (default: 5).</param>
    public TestTimeAugmentationConfiguration(IAugmentationPolicy<T, TData> pipeline, int numberOfAugmentations = 5)
    {
        Pipeline = pipeline;
        NumberOfAugmentations = numberOfAugmentations;
    }

    /// <summary>
    /// Gets the configuration as a dictionary for logging or serialization.
    /// </summary>
    /// <returns>A dictionary containing all configuration values.</returns>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>
        {
            ["isEnabled"] = IsEnabled,
            ["numberOfAugmentations"] = NumberOfAugmentations,
            ["includeOriginal"] = IncludeOriginal,
            ["aggregationMethod"] = AggregationMethod.ToString(),
            ["applyInverseTransforms"] = ApplyInverseTransforms
        };

        if (Seed.HasValue)
        {
            config["seed"] = Seed.Value;
        }

        if (ConfidenceThreshold.HasValue)
        {
            config["confidenceThreshold"] = ConfidenceThreshold.Value;
        }

        if (Pipeline is not null)
        {
            config["pipeline"] = Pipeline.GetConfiguration();
        }

        return config;
    }
}

/// <summary>
/// Fluent builder for creating Test-Time Augmentation configurations step by step.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This builder lets you configure TTA using a readable, chainable syntax:
/// <code>
/// var tta = new TestTimeAugmentationBuilder&lt;double, ImageTensor&lt;double&gt;&gt;()
///     .Add(new HorizontalFlip&lt;double, ImageTensor&lt;double&gt;&gt;())
///     .Add(new SmallRotation&lt;double, ImageTensor&lt;double&gt;&gt;())
///     .WithNumberOfAugmentations(8)
///     .WithAggregation(PredictionAggregationMethod.Median)
///     .Build();
/// </code>
/// Each method returns the builder, so you can chain calls together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public class TestTimeAugmentationBuilder<T, TData>
{
    private readonly AugmentationPipeline<T, TData> _pipeline = new();
    private int _numberOfAugmentations = 5;
    private bool _includeOriginal = true;
    private PredictionAggregationMethod _aggregationMethod = PredictionAggregationMethod.Mean;
    private int? _seed;
    private bool _applyInverseTransforms = true;
    private double? _confidenceThreshold;

    /// <summary>
    /// Adds an augmentation transform to the pipeline.
    /// </summary>
    /// <param name="augmentation">The augmentation to add (e.g., HorizontalFlip, Rotation).</param>
    /// <returns>This builder for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this multiple times to add different transformations.
    /// Common choices:
    /// - HorizontalFlip: Mirror the image left-to-right
    /// - Rotation: Rotate slightly (-5 to +5 degrees)
    /// - Scale: Zoom in or out slightly
    /// </para>
    /// </remarks>
    public TestTimeAugmentationBuilder<T, TData> Add(IAugmentation<T, TData> augmentation)
    {
        _pipeline.Add(augmentation);
        return this;
    }

    /// <summary>
    /// Adds a choice of augmentations where only one is randomly selected each time.
    /// </summary>
    /// <param name="augmentations">The augmentations to choose from.</param>
    /// <returns>This builder for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of applying all augmentations, this randomly
    /// picks ONE from the list each time. Useful for variety without overwhelming the input.
    /// </para>
    /// </remarks>
    public TestTimeAugmentationBuilder<T, TData> OneOf(params IAugmentation<T, TData>[] augmentations)
    {
        _pipeline.OneOf(augmentations);
        return this;
    }

    /// <summary>
    /// Sets how many augmented versions to create.
    /// </summary>
    /// <param name="count">Number of augmented samples (default: 5, range: 3-10 recommended).</param>
    /// <returns>This builder for method chaining.</returns>
    public TestTimeAugmentationBuilder<T, TData> WithNumberOfAugmentations(int count)
    {
        _numberOfAugmentations = count;
        return this;
    }

    /// <summary>
    /// Sets whether to include the original (unaugmented) input.
    /// </summary>
    /// <param name="include">True to include original input (default: true).</param>
    /// <returns>This builder for method chaining.</returns>
    public TestTimeAugmentationBuilder<T, TData> IncludeOriginal(bool include = true)
    {
        _includeOriginal = include;
        return this;
    }

    /// <summary>
    /// Sets how predictions from augmented versions are combined.
    /// </summary>
    /// <param name="method">The aggregation method (default: Mean).</param>
    /// <returns>This builder for method chaining.</returns>
    public TestTimeAugmentationBuilder<T, TData> WithAggregation(PredictionAggregationMethod method)
    {
        _aggregationMethod = method;
        return this;
    }

    /// <summary>
    /// Sets a random seed for reproducible augmentations.
    /// </summary>
    /// <param name="seed">The random seed value.</param>
    /// <returns>This builder for method chaining.</returns>
    public TestTimeAugmentationBuilder<T, TData> WithSeed(int seed)
    {
        _seed = seed;
        return this;
    }

    /// <summary>
    /// Sets whether to apply inverse transforms to spatial predictions.
    /// </summary>
    /// <param name="apply">True to apply inverse transforms (default: true).</param>
    /// <returns>This builder for method chaining.</returns>
    public TestTimeAugmentationBuilder<T, TData> WithInverseTransforms(bool apply = true)
    {
        _applyInverseTransforms = apply;
        return this;
    }

    /// <summary>
    /// Sets a confidence threshold for filtering low-confidence predictions.
    /// </summary>
    /// <param name="threshold">Minimum confidence (0.0 to 1.0) to include a prediction.</param>
    /// <returns>This builder for method chaining.</returns>
    public TestTimeAugmentationBuilder<T, TData> WithConfidenceThreshold(double threshold)
    {
        _confidenceThreshold = threshold;
        return this;
    }

    /// <summary>
    /// Builds the final Test-Time Augmentation configuration.
    /// </summary>
    /// <returns>A configured TestTimeAugmentationConfiguration ready for use.</returns>
    public TestTimeAugmentationConfiguration<T, TData> Build()
    {
        return new TestTimeAugmentationConfiguration<T, TData>
        {
            Pipeline = _pipeline,
            NumberOfAugmentations = _numberOfAugmentations,
            IncludeOriginal = _includeOriginal,
            AggregationMethod = _aggregationMethod,
            Seed = _seed,
            ApplyInverseTransforms = _applyInverseTransforms,
            ConfidenceThreshold = _confidenceThreshold
        };
    }
}

/// <summary>
/// Contains the result of a Test-Time Augmentation prediction, including individual and aggregated predictions.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class gives you both:
/// 1. The final combined prediction (what you usually want)
/// 2. All the individual predictions (for debugging or analysis)
///
/// You also get uncertainty information - if all 5 predictions were similar, you can be
/// confident in the result. If they varied wildly, you might want to be more cautious.
/// </para>
/// </remarks>
/// <typeparam name="TOutput">The type of prediction output (e.g., Vector for class probabilities).</typeparam>
public class TestTimeAugmentationResult<TOutput>
{
    /// <summary>
    /// Gets the final combined prediction after aggregating all augmented predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main result you'll use. It's the combination
    /// of all individual predictions based on the aggregation method (mean, median, vote, etc.).</para>
    /// </remarks>
    public TOutput AggregatedPrediction { get; }

    /// <summary>
    /// Gets all the individual predictions from each augmented version.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows what the model predicted for each variation.
    /// Useful for:
    /// - Debugging: See if one augmentation is causing problems
    /// - Analysis: Understand how much predictions vary
    /// - Visualization: Show uncertainty in results
    /// </para>
    /// </remarks>
    public IReadOnlyList<TOutput> IndividualPredictions { get; }

    /// <summary>
    /// Gets the confidence score of the aggregated prediction (if available).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A value between 0 and 1 indicating how confident the
    /// model is in its prediction. Higher = more confident. May be null if the model
    /// doesn't provide confidence scores.</para>
    /// </remarks>
    public double? Confidence { get; }

    /// <summary>
    /// Gets the standard deviation of predictions, measuring uncertainty.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how much the predictions varied:
    /// - Low standard deviation: Predictions were consistent (good!)
    /// - High standard deviation: Predictions varied a lot (less reliable)
    ///
    /// Use this to decide how much to trust the result.
    /// </para>
    /// </remarks>
    public double? StandardDeviation { get; }

    /// <summary>
    /// Creates a new Test-Time Augmentation result.
    /// </summary>
    /// <param name="aggregated">The combined prediction.</param>
    /// <param name="individual">All individual predictions.</param>
    /// <param name="confidence">Optional confidence score.</param>
    /// <param name="standardDeviation">Optional standard deviation of predictions.</param>
    public TestTimeAugmentationResult(
        TOutput aggregated,
        IReadOnlyList<TOutput> individual,
        double? confidence = null,
        double? standardDeviation = null)
    {
        AggregatedPrediction = aggregated;
        IndividualPredictions = individual;
        Confidence = confidence;
        StandardDeviation = standardDeviation;
    }
}
