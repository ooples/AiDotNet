namespace AiDotNet.Enums;

/// <summary>Distinguishes the three kinds of value an evaluator may report under one metric name.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve evaluator returns an untyped dictionary, so a single metrics payload routinely mixes
/// real scores with boolean flags such as <c>timeout</c> and with free text such as an error message. Scalarization
/// treats those three cases very differently, and the difference is invisible in an untyped dictionary: a flag that
/// is averaged in as <c>1.0</c> can hand a crashed program a mid-range fitness. Making the kind explicit is what
/// lets the aggregation report exactly which values it used and which it refused.
/// </para>
/// <para><b>For Beginners:</b> When your scoring code reports results, some entries are real numbers you want
/// averaged, some are yes/no flags such as "did it time out", and some are messages. This says which of the three
/// a particular entry is, so a flag or a message is never quietly treated as a score.</para>
/// </remarks>
public enum ProgramMetricValueKind
{
    /// <summary>A real-valued measurement that may participate in an aggregation.</summary>
    Number = 0,

    /// <summary>A boolean flag such as a timeout indicator, which is never averaged as a score.</summary>
    Flag = 1,

    /// <summary>Free text such as an error message or a label.</summary>
    Text = 2
}
