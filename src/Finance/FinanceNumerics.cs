namespace AiDotNet.Finance;

/// <summary>
/// Numeric guards shared by the finance models that must survive every element type.
/// </summary>
internal static class FinanceNumerics
{
    /// <summary>The smallest positive normal Half, 2^-14.</summary>
    private const double HalfSmallestNormal = 6.103515625e-5;

    /// <summary>
    /// <paramref name="epsilon"/> as a <typeparamref name="T"/> guard that is still positive: in Half a
    /// guard such as 1e-8 or 1e-10 rounds to zero (or to a subnormal whose reciprocal overflows), so it
    /// is raised to the smallest normal Half. float and double keep the value.
    /// </summary>
    internal static T Tolerance<T>(double epsilon) =>
        MathHelper.GetNumericOperations<T>().FromDouble(
            typeof(T) == typeof(Half) ? Math.Max(epsilon, HalfSmallestNormal) : epsilon);
}
