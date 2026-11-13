using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Randomly masks (sets to zero) features in tabular data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Feature dropout randomly "hides" some features during training
/// by setting them to zero. This forces the model to learn more robust representations
/// that don't rely too heavily on any single feature.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>To prevent overfitting to specific features</item>
/// <item>To simulate missing data scenarios</item>
/// <item>When you want the model to be robust to feature absence</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>When all features are critical and cannot be missing</item>
/// <item>When features have strong interdependencies that break when one is missing</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FeatureDropout<T> : TabularAugmenterBase<T>
{
    /// <summary>
    /// Gets the probability of dropping each feature.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.1 (10% of features dropped)</para>
    /// <para>Higher values drop more features. Typical range: 0.05 to 0.3</para>
    /// </remarks>
    public double DropoutRate { get; }

    /// <summary>
    /// Gets or sets the indices of features that can be dropped.
    /// </summary>
    /// <remarks>
    /// <para>If null, any feature can be dropped.</para>
    /// </remarks>
    public int[]? FeatureIndices { get; set; }

    /// <summary>
    /// Gets or sets the value to use for dropped features.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.0</para>
    /// <para>Can be set to other values like -1 or the feature mean.</para>
    /// </remarks>
    public double DropValue { get; set; } = 0.0;

    /// <summary>
    /// Creates a new feature dropout augmentation.
    /// </summary>
    /// <param name="dropoutRate">Probability of dropping each feature (default: 0.1).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="featureIndices">Optional array of feature indices that can be dropped.</param>
    public FeatureDropout(
        double dropoutRate = 0.1,
        double probability = 0.5,
        int[]? featureIndices = null) : base(probability)
    {
        if (dropoutRate < 0 || dropoutRate > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(dropoutRate), "Dropout rate must be between 0 and 1.");
        }

        DropoutRate = dropoutRate;
        FeatureIndices = featureIndices;
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int rows = GetSampleCount(data);
        int cols = GetFeatureCount(data);

        // Determine which features can be dropped
        var droppableFeatures = FeatureIndices ?? Enumerable.Range(0, cols).ToArray();

        // Each listed feature has an independent chance of being dropped (draws in row, feature order)
        var features = droppableFeatures.Where(f => f >= 0 && f < cols).ToArray();
        var values = Tensor<T>.FromMatrix(result);
        if (rows == 0 || features.Length == 0)
        {
            return result;
        }

        var engine = Engine;
        // Each listed feature's hits are accumulated into its column; a column drops if any listing hit.
        var dropped = engine.TensorLessThan(
            AugmentationRandom<T>.Unit(context, [rows, features.Length]), NumOps.FromDouble(DropoutRate));
        var hits = engine.TensorIndexAdd(
            new Tensor<T>([rows, cols]), 1, new Tensor<int>(features, [features.Length]), dropped);
        var mask = engine.TensorGreaterThan(hits, NumOps.Zero);
        var fill = new Tensor<T>([rows, cols]);
        engine.TensorFill(fill, NumOps.FromDouble(DropValue));
        return engine.TensorWhere(mask, fill, values).ToMatrix();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["dropoutRate"] = DropoutRate;
        parameters["dropValue"] = DropValue;
        if (FeatureIndices is not null)
        {
            parameters["featureIndices"] = FeatureIndices;
        }
        return parameters;
    }
}
