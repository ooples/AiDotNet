using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>
/// Collects order-independent observed bounds and freezes them into a fixed descriptor definition.
/// </summary>
public sealed class EvolutionDescriptorCalibrator
{
    private readonly string _name;
    private readonly int _binCount;
    private readonly EvolutionOutOfRangePolicy _policy;
    private double _minimum = double.PositiveInfinity;
    private double _maximum = double.NegativeInfinity;
    private long _observations;

    /// <summary>Initializes a descriptor calibrator.</summary>
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
