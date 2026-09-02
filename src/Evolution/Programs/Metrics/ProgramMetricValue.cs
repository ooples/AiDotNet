using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Metrics;

/// <summary>One reported metric value, tagged as a number, a boolean flag, or free text.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve evaluator contract is an untyped dictionary, so <c>{"score": 0.8, "timeout": True,
/// "stderr": "ZeroDivisionError"}</c> is a single legal metrics payload. That mixture is the reason
/// <c>metrics_utils.safe_numeric_average</c> exists at all, and the reason it has to special-case
/// <c>bool</c>: Python treats <c>True</c> as the integer 1, so averaging a timeout flag into a score would give a
/// program that crashed a mid-range fitness. This type keeps the same expressiveness while making the distinction
/// explicit at the type level, so nothing has to be guessed from a runtime type test.
/// </para>
/// <para>
/// Values are immutable and are compared by value. <see cref="TryGetNumber"/> is the single accessor an
/// aggregation needs: it succeeds for <see cref="ProgramMetricValueKind.Number"/> and, when
/// <paramref name="allowTextConversion"/> is requested, for text that parses as a finite invariant-culture number.
/// It deliberately fails for a flag, because a flag is not a score.
/// </para>
/// <para><b>For Beginners:</b> This is one entry of the score card your evaluation code fills in. Use
/// <see cref="Number"/> for anything you want the search to optimise, <see cref="Flag"/> for a yes/no fact such as
/// "the program timed out", and <see cref="Text"/> for a message such as the exception that was thrown. Marking
/// them apart means the search will never mistake "it timed out" for "it scored one point".</para>
/// </remarks>
public sealed class ProgramMetricValue : IEquatable<ProgramMetricValue>
{
    /// <summary>The largest text value accepted, in characters.</summary>
    public const int MaxTextLength = 4_096;

    private readonly double _number;
    private readonly bool _flag;
    private readonly string _text;

    private ProgramMetricValue(ProgramMetricValueKind kind, double number, bool flag, string text)
    {
        Kind = kind;
        _number = number;
        _flag = flag;
        _text = text;
    }

    /// <summary>Gets which kind of value this instance carries.</summary>
    public ProgramMetricValueKind Kind { get; }

    /// <summary>Creates a numeric metric value.</summary>
    /// <param name="value">The measurement, which may be non-finite so that a broken evaluator can be reported.</param>
    /// <returns>A value whose <see cref="Kind"/> is <see cref="ProgramMetricValueKind.Number"/>.</returns>
    public static ProgramMetricValue Number(double value) =>
        new(ProgramMetricValueKind.Number, value, flag: false, text: string.Empty);

    /// <summary>Creates a boolean flag metric value.</summary>
    /// <param name="value">The flag state, such as whether the evaluation timed out.</param>
    /// <returns>A value whose <see cref="Kind"/> is <see cref="ProgramMetricValueKind.Flag"/>.</returns>
    public static ProgramMetricValue Flag(bool value) =>
        new(ProgramMetricValueKind.Flag, value ? 1.0 : 0.0, value, text: string.Empty);

    /// <summary>Creates a free-text metric value.</summary>
    /// <param name="value">The text, which is truncated to <see cref="MaxTextLength"/> characters.</param>
    /// <returns>A value whose <see cref="Kind"/> is <see cref="ProgramMetricValueKind.Text"/>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
    public static ProgramMetricValue Text(string value)
    {
        Guard.NotNull(value);
        string bounded = value.Length > MaxTextLength ? value.Substring(0, MaxTextLength) : value;
        return new ProgramMetricValue(ProgramMetricValueKind.Text, double.NaN, flag: false, text: bounded);
    }

    /// <summary>Gets the flag state, or <c>false</c> when this value is not a flag.</summary>
    public bool FlagValue => Kind == ProgramMetricValueKind.Flag && _flag;

    /// <summary>Gets the text, or an empty string when this value is not text.</summary>
    public string TextValue => Kind == ProgramMetricValueKind.Text ? _text : string.Empty;

    /// <summary>Attempts to read this value as a number.</summary>
    /// <param name="allowTextConversion">
    /// Whether text that parses as a finite invariant-culture number is accepted; flags are never accepted.
    /// </param>
    /// <param name="value">The numeric value when the attempt succeeds; otherwise <c>0</c>.</param>
    /// <returns><c>true</c> when a numeric value was produced.</returns>
    /// <remarks>
    /// A numeric value is returned exactly as reported, including <c>NaN</c> and infinity, so a caller can decide
    /// how to report a broken measurement rather than having it silently disappear. Text conversion accepts only
    /// finite results, which keeps the outcome identical on every target framework - unlike Python's <c>float()</c>,
    /// which also accepts the literals <c>nan</c> and <c>inf</c>.
    /// </remarks>
    public bool TryGetNumber(bool allowTextConversion, out double value)
    {
        if (Kind == ProgramMetricValueKind.Number)
        {
            value = _number;
            return true;
        }

        if (allowTextConversion && Kind == ProgramMetricValueKind.Text &&
            double.TryParse(_text, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) &&
            !double.IsNaN(parsed) && !double.IsInfinity(parsed))
        {
            value = parsed;
            return true;
        }

        value = 0.0;
        return false;
    }

    /// <inheritdoc/>
    public bool Equals(ProgramMetricValue? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        if (Kind != other.Kind) return false;
        return Kind switch
        {
            ProgramMetricValueKind.Number => _number.Equals(other._number),
            ProgramMetricValueKind.Flag => _flag == other._flag,
            _ => string.Equals(_text, other._text, StringComparison.Ordinal)
        };
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as ProgramMetricValue);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = (hash * 31) + (int)Kind;
            hash = (hash * 31) + Kind switch
            {
                ProgramMetricValueKind.Number => _number.GetHashCode(),
                ProgramMetricValueKind.Flag => _flag ? 1 : 0,
                _ => StringComparer.Ordinal.GetHashCode(_text)
            };
            return hash;
        }
    }

    /// <summary>Returns the kind and a bounded rendering of the value.</summary>
    /// <returns>A short diagnostic label that never exceeds the text bound.</returns>
    public override string ToString() => Kind switch
    {
        ProgramMetricValueKind.Number => _number.ToString("R", CultureInfo.InvariantCulture),
        ProgramMetricValueKind.Flag => _flag ? "true" : "false",
        _ => "\"" + (_text.Length > 32 ? _text.Substring(0, 32) : _text) + "\""
    };
}
