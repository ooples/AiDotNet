using System.Collections.Generic;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Covers the optimizer-type compatibility parsing used by <see cref="Checkpoint{T,TInput,TOutput}"/>
/// when <c>Type.GetType</c> can't resolve a saved optimizer type (renamed/versioned assembly) and it
/// falls back to a bare <c>FullName</c> comparison. The parse must NOT truncate generic type names,
/// whose assembly-qualified form embeds type-argument assembly-qualified names (with commas) inside
/// <c>[[...]]</c>.
/// </summary>
public class CheckpointOptimizerTypeCompatTests
{
    // ExtractTypeFullName is a static internal helper that does not use the type parameters, so it can be
    // invoked through any concrete instantiation of the generic Checkpoint type.
    private static string Extract(string assemblyQualifiedName)
        => Checkpoint<double, double, double>.ExtractTypeFullName(assemblyQualifiedName);

    [Fact]
    public void ExtractTypeFullName_NonGenericType_MatchesFullName()
    {
        var type = typeof(int);
        // The bare-FullName fallback compares expectedFullName to actualType.FullName; that equality must
        // hold when the saved name is the type's own assembly-qualified name.
        Assert.Equal(type.FullName, Extract(type.AssemblyQualifiedName!));
    }

    [Fact]
    public void ExtractTypeFullName_GenericTypeWithCommasInArguments_IsNotTruncated()
    {
        // Dictionary<string,int>'s assembly-qualified name embeds the args' own assembly-qualified names
        // (each containing commas) inside [[...]]. A naive Split(',')[0] would truncate at the first inner
        // comma; the depth-aware parse must return the entire type FullName.
        var type = typeof(Dictionary<string, int>);
        string extracted = Extract(type.AssemblyQualifiedName!);

        Assert.Equal(type.FullName, extracted);
        // Sanity: the generic-argument block survived (a truncating parse would have dropped it).
        Assert.Contains("[[", extracted);
    }

    [Fact]
    public void ExtractTypeFullName_NoTopLevelComma_ReturnsInputTrimmed()
    {
        // A bare FullName (no assembly qualification) has no top-level comma; it should round-trip as-is.
        Assert.Equal("Some.Namespace.MyType", Extract("Some.Namespace.MyType"));
    }
}
