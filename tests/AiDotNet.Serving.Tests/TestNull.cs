#nullable disable

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Produces a null reference typed as a non-nullable parameter so a test can drive the guard it is
/// asserting on.
/// </summary>
/// <remarks>
/// A test that proves "this method rejects null" has to actually hand the method a null. Writing that
/// as <c>null</c> at the call site needs a suppression there, and suppressing it with the
/// null-forgiving operator is not allowed in this repository. This file is compiled with nullable
/// analysis off, so the null is produced here, once, where it is deliberate and documented, and every
/// call site stays free of both the operator and a pragma.
/// </remarks>
internal static class TestNull
{
    /// <summary>Returns null typed as <typeparamref name="T"/>.</summary>
    internal static T Of<T>() where T : class => null;

    /// <summary>
    /// Hands <paramref name="value"/> to a non-nullable parameter unchanged, for a theory whose data
    /// includes a null case.
    /// </summary>
    internal static T Pass<T>(T value) where T : class => value;

    /// <summary>
    /// Casts <paramref name="value"/> to <typeparamref name="T"/>, letting a null result through for a
    /// reference type, so a reflection helper can return the null its caller is asserting on.
    /// </summary>
    internal static T Cast<T>(object value) => (T)value;
}
