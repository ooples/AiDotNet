namespace AiDotNet.Tests;

/// <summary>Finite-value checks shared by paper regressions on all supported test frameworks.</summary>
internal static class TtsPaperNumerics
{
    internal static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    internal static long Fingerprint(IEnumerable<double> values)
    {
        long hash = unchecked((long)0xcbf29ce484222325UL), count = 0;
        foreach (double value in values)
        {
            hash = unchecked((hash ^ BitConverter.DoubleToInt64Bits(value)) * (long)0x100000001b3UL);
            count++;
        }
        return unchecked((hash ^ count) * (long)0x100000001b3UL);
    }
}
