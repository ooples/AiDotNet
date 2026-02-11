#nullable disable
// Nullable disabled because these tests deliberately pass null through non-nullable parameters
// to demonstrate that NRT annotations are compile-time only.

using Xunit;

namespace AiDotNet.Tests.UnitTests.Documentation;

/// <summary>
/// Educational tests that prove Nullable Reference Types (NRT) are a compile-time-only feature.
/// These tests demonstrate WHY runtime null checks (Guard.NotNull) are still necessary
/// even when NRT annotations are enabled.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> C# 8+ introduced "nullable reference types" which let you annotate
/// parameters as non-nullable (e.g., <c>string name</c> instead of <c>string? name</c>).
///
/// However, these annotations are ONLY enforced at compile time as warnings.
/// At runtime, null can still be passed to any reference type parameter because:
/// - Callers can suppress warnings with <c>#nullable disable</c> or the <c>!</c> operator
/// - Code compiled without NRT enabled doesn't see the annotations at all
/// - Reflection, serialization, and interop can inject nulls
///
/// These tests prove that runtime guards are essential for robust code.
/// </para>
/// </remarks>
public class NullableReferenceTypeTests
{
    /// <summary>
    /// Demonstrates that a method with a non-nullable parameter can receive null at runtime.
    /// </summary>
    [Fact]
    public void NonNullableParameter_CanReceiveNull_AtRuntime()
    {
        // This compiles without warning because we have #nullable disable at the top.
        // At runtime, the string parameter IS null despite the method signature saying it's non-nullable.
        string result = AcceptNonNullableString(null);
        Assert.Null(result);
    }

    /// <summary>
    /// Demonstrates that a non-nullable property can be set to null via an object initializer
    /// when NRT is disabled.
    /// </summary>
    [Fact]
    public void NonNullableProperty_CanBeNull_WhenSetFromDisabledContext()
    {
        var holder = new StringHolder { Value = null };
        Assert.Null(holder.Value);
    }

    /// <summary>
    /// Demonstrates that an array of non-nullable type contains null default elements.
    /// </summary>
    [Fact]
    public void NonNullableArray_ContainsNullDefaults()
    {
        // string[] is an array of "non-nullable" strings,
        // but default(string) is null, so each element starts as null.
        var array = new string[3];
        Assert.Null(array[0]);
        Assert.Null(array[1]);
        Assert.Null(array[2]);
    }

    /// <summary>
    /// Demonstrates that casting through object erases NRT information.
    /// </summary>
    [Fact]
    public void CastingThroughObject_ErasesNullability()
    {
        object boxed = null;
        // This cast succeeds at runtime even though the target type is "non-nullable".
        string value = (string)boxed;
        Assert.Null(value);
    }

    // Helper method with a non-nullable parameter (when #nullable is enabled elsewhere)
    private static string AcceptNonNullableString(string value)
    {
        // Without a guard, this silently returns null.
        // With Guard.NotNull(value), this would throw immediately with a clear message.
        return value;
    }

    // Helper class with a non-nullable property
    private sealed class StringHolder
    {
        public string Value { get; set; } = "";
    }
}
