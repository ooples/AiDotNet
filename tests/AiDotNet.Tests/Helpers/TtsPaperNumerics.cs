namespace AiDotNet.Tests;

/// <summary>Finite-value checks shared by paper regressions on all supported test frameworks.</summary>
internal static class TtsPaperNumerics
{
    internal static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}
