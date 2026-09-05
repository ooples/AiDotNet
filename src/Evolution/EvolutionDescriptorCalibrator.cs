using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>
/// Collects order-independent observed bounds and freezes them into a fixed descriptor definition.
/// </summary>
/// <remarks>
/// <para>
/// The calibrator keeps only the running minimum, maximum, and count of the finite values passed to
/// <see cref="Observe"/>, so it uses O(1) memory and O(1) time per observation, and the frozen bounds depend
/// only on the set of values, never on the order in which they arrived. <see cref="Freeze"/> widens the
/// observed span by an optional relative padding and returns an immutable
/// <see cref="EvolutionDescriptorDefinition"/> carrying the configured bin count and out-of-range policy. When
/// every observation was the same value, the bounds are nudged apart by one representable double on each side
/// (or on one side only at the extreme of the double range) so the definition still has a positive, finite
/// span. Freezing with no observations, or with padding that pushes a bound to infinity, throws.
/// </para>
/// <para><b>For Beginners:</b> A MAP-Elites archive needs to know the range of each behaviour descriptor
/// (for example the slowest and fastest inference time you expect) before it can divide that range into bins,
/// but you often do not know those ranges in advance. This helper solves that: feed it the descriptor values
/// from a warm-up batch of candidates, then call <see cref="Freeze"/> once to turn the observed low and high
/// into a fixed definition the archive uses for the rest of the run. Passing a small <c>relativePadding</c>
/// such as 0.1 leaves 10 percent of head-room on each side so later candidates slightly outside the warm-up
/// range still land in a bin instead of being clamped or rejected. It is like measuring the shortest and
/// tallest students in a class before deciding where to draw the height brackets on a chart.</para>
/// </remarks>
public sealed class EvolutionDescriptorCalibrator
{
    private readonly string _name;
    private readonly int _binCount;
    private readonly EvolutionOutOfRangePolicy _policy;
    private double _minimum = double.PositiveInfinity;
    private double _maximum = double.NegativeInfinity;
    private long _observations;

    /// <summary>Initializes a descriptor calibrator.</summary>
    /// <param name="name">The unique descriptor name; surrounding whitespace is trimmed.</param>
    /// <param name="binCount">The positive number of bins the frozen definition uses inside its bounds.</param>
    /// <param name="outOfRangePolicy">How the frozen definition treats values outside its bounds.</param>
    public EvolutionDescriptorCalibrator(
        string name,
        int binCount,
        EvolutionOutOfRangePolicy outOfRangePolicy = EvolutionOutOfRangePolicy.Clamp)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.Positive(binCount);
        if (!Enum.IsDefined(typeof(EvolutionOutOfRangePolicy), outOfRangePolicy))
            throw new ArgumentOutOfRangeException(nameof(outOfRangePolicy));
        _name = name.Trim();
        _binCount = binCount;
        _policy = outOfRangePolicy;
    }

    /// <summary>Gets the number of finite observations.</summary>
    public long ObservationCount => _observations;

    /// <summary>Adds one finite observation.</summary>
    /// <param name="value">The observed descriptor value.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="value"/> is NaN or infinite.</exception>
    public void Observe(double value)
    {
        if (!EvolutionDescriptorDefinition.IsFinite(value))
            throw new ArgumentOutOfRangeException(nameof(value), "Calibration values must be finite.");
        _minimum = Math.Min(_minimum, value);
        _maximum = Math.Max(_maximum, value);
        if (_observations == long.MaxValue) throw new InvalidOperationException("The calibration observation count overflowed.");
        _observations++;
    }

    /// <summary>Freezes the observed range into an immutable definition.</summary>
    /// <param name="relativePadding">Optional nonnegative fractional padding around the observed span.</param>
    /// <returns>A definition whose bounds cover every observed value plus the requested padding.</returns>
    /// <exception cref="InvalidOperationException">
    /// No value has been observed, or the padded bounds cannot be represented as finite doubles.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="relativePadding"/> is negative or not finite.</exception>
    public EvolutionDescriptorDefinition Freeze(double relativePadding = 0)
    {
        if (_observations == 0) throw new InvalidOperationException("At least one observation is required before freezing bounds.");
        if (!EvolutionDescriptorDefinition.IsFinite(relativePadding) || relativePadding < 0)
            throw new ArgumentOutOfRangeException(nameof(relativePadding));

        double minimum;
        double maximum;
        if (_minimum == _maximum)
        {
            if (_minimum == double.MaxValue)
            {
                minimum = NextDown(_minimum);
                maximum = _minimum;
            }
            else if (_minimum == -double.MaxValue)
            {
                minimum = _minimum;
                maximum = NextUp(_minimum);
            }
            else
            {
                minimum = NextDown(_minimum);
                maximum = NextUp(_maximum);
            }
        }
        else
        {
            double span = _maximum - _minimum;
            if (!EvolutionDescriptorDefinition.IsFinite(span))
                throw new InvalidOperationException("The observed descriptor span cannot be represented as a finite double.");
            double padding = span * relativePadding;
            minimum = _minimum - padding;
            maximum = _maximum + padding;
            if (!EvolutionDescriptorDefinition.IsFinite(minimum) || !EvolutionDescriptorDefinition.IsFinite(maximum))
                throw new InvalidOperationException("The padded descriptor bounds cannot be represented as finite doubles.");
        }
        return new EvolutionDescriptorDefinition(_name, minimum, maximum, _binCount, _policy);
    }

    private static double NextUp(double value)
    {
        if (value == 0) return double.Epsilon;
        long bits = BitConverter.DoubleToInt64Bits(value);
        return BitConverter.Int64BitsToDouble(value > 0 ? bits + 1 : bits - 1);
    }

    private static double NextDown(double value)
    {
        if (value == 0) return -double.Epsilon;
        long bits = BitConverter.DoubleToInt64Bits(value);
        return BitConverter.Int64BitsToDouble(value > 0 ? bits - 1 : bits + 1);
    }
}
