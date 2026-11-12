namespace AiDotNet.Data.Transforms.Numeric;

/// <summary>
/// Scales values to a target range using min-max normalization.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// The formula is: x_scaled = (x - min) / (max - min) * (targetMax - targetMin) + targetMin
/// </para>
/// <para><b>For Beginners:</b> Min-max scaling squeezes your data into a specific range
/// (default [0, 1]). This is commonly used for pixel values or when your algorithm
/// requires bounded inputs.
/// </para>
/// </remarks>
public class MinMaxScaleTransform<T> : ITransform<T[], T[]>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Tensor<T> _min;
    private readonly Tensor<T> _max;
    private readonly double _targetMin;
    private readonly double _targetMax;

    /// <summary>
    /// Creates a min-max scaler from a reference dataset.
    /// </summary>
    /// <param name="referenceData">The reference data to compute min/max from.</param>
    /// <param name="targetMin">The minimum value of the target range. Default is 0.</param>
    /// <param name="targetMax">The maximum value of the target range. Default is 1.</param>
    public MinMaxScaleTransform(T[][] referenceData, double targetMin = 0.0, double targetMax = 1.0)
    {
        if (referenceData is null)
        {
            throw new ArgumentNullException(nameof(referenceData));
        }

        if (referenceData.Length == 0)
        {
            throw new ArgumentException("Reference data cannot be empty.", nameof(referenceData));
        }

        if (targetMin >= targetMax)
        {
            throw new ArgumentException("Target min must be less than target max.");
        }

        _targetMin = targetMin;
        _targetMax = targetMax;

        int featureCount = referenceData[0].Length;
        for (int i = 0; i < referenceData.Length; i++)
        {
            if (referenceData[i] is null)
            {
                throw new ArgumentException($"Reference data row {i} is null.", nameof(referenceData));
            }

            if (referenceData[i].Length != featureCount)
            {
                throw new ArgumentException(
                    $"Reference data row {i} has {referenceData[i].Length} elements, expected {featureCount}.",
                    nameof(referenceData));
            }
        }

        if (featureCount == 0)
        {
            _min = new Tensor<T>(new[] { 0 });
            _max = new Tensor<T>(new[] { 0 });
        }
        else
        {
            // Column-wise min and max (min = -max(-x)) with engine axis reductions
            var engine = AiDotNetEngine.Current;
            var data = TransformTensors.Rows(referenceData, featureCount);
            var axis0 = new[] { 0 };
            _max = engine.ReduceMax(data, axis0, false);
            _min = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(data), axis0, false));
        }
    }

    /// <summary>
    /// Creates a min-max scaler from pre-computed statistics that maps to the range [0, 1].
    /// </summary>
    /// <param name="min">The minimum for each feature.</param>
    /// <param name="max">The maximum for each feature.</param>
    public MinMaxScaleTransform(T[] min, T[] max)
        : this(min, max, 0.0, 1.0)
    {
    }

    /// <summary>
    /// Creates a min-max scaler from pre-computed statistics.
    /// </summary>
    /// <param name="min">The minimum for each feature.</param>
    /// <param name="max">The maximum for each feature.</param>
    /// <param name="targetMin">The minimum value of the target range.</param>
    /// <param name="targetMax">The maximum value of the target range.</param>
    public MinMaxScaleTransform(T[] min, T[] max, double targetMin, double targetMax)
    {
        if (min is null)
        {
            throw new ArgumentNullException(nameof(min));
        }

        if (max is null)
        {
            throw new ArgumentNullException(nameof(max));
        }

        if (min.Length != max.Length)
        {
            throw new ArgumentException(
                $"Min length ({min.Length}) must match max length ({max.Length}).",
                nameof(max));
        }

        if (targetMin >= targetMax)
        {
            throw new ArgumentException("Target min must be less than target max.");
        }

        _min = TransformTensors.FromStatistics<T>(min);
        _max = TransformTensors.FromStatistics<T>(max);
        _targetMin = targetMin;
        _targetMax = targetMax;
    }

    /// <inheritdoc/>
    public T[] Apply(T[] input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        int len = Math.Min(input.Length, _min.Length);
        var result = new T[input.Length];
        T targetRange = NumOps.FromDouble(_targetMax - _targetMin);
        T targetMin = NumOps.FromDouble(_targetMin);

        if (len > 0)
        {
            // ((x - min) / (max - min)) * targetRange + targetMin; constant features map to targetMin
            var engine = AiDotNetEngine.Current;
            var shape = new[] { len };
            var min = TransformTensors.Prefix(_min, len);
            var range = engine.TensorSubtract(TransformTensors.Prefix(_max, len), min);
            var values = TransformTensors.Prefix(input, len);
            var scaled = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(engine.TensorDivide(engine.TensorSubtract(values, min), range), targetRange),
                targetMin);
            var constant = engine.TensorLessThan(engine.TensorAbs(range), NumOps.FromDouble(1e-12));
            var targetMinFill = new Tensor<T>(shape);
            targetMinFill.Fill(targetMin);
            engine.TensorWhere(constant, targetMinFill, scaled).AsSpan().CopyTo(result.AsSpan(0, len));
        }

        // Features beyond the fitted statistics pass through unchanged
        Array.Copy(input, len, result, len, input.Length - len);

        return result;
    }
}
