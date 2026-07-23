using System;
using System.Text.RegularExpressions;

namespace AiDotNet.Generators;

/// <summary>
/// Rewrites occurrences of <c>double</c> that appear as a GENERIC TYPE ARGUMENT to <c>float</c>,
/// leaving every other <c>double</c> (the C# keyword) untouched. Used by
/// <see cref="TestScaffoldGenerator"/> to emit <c>&lt;float&gt;</c> scaffolds for training-perf-bound
/// models (#1679) without mangling unrelated <c>double</c> tokens in the same fragment.
///
/// <para>This is deliberately a separate, Roslyn-free, pure-string type so it can be shared-compiled
/// into the test assembly (<c>AiDotNet.Tests</c>) and unit-tested directly — the generator itself is
/// referenced only as an analyzer and is otherwise not callable from tests.</para>
///
/// <para><b>What gets rewritten</b> — <c>double</c> bounded on the left by a generic-argument
/// delimiter (<c>&lt;</c> or <c>,</c>, allowing whitespace) and on the right by a generic-argument
/// terminator (<c>&gt;</c>, <c>,</c>, <c>[</c> for arrays, or <c>?</c> for nullable). That covers
/// <c>&lt;double&gt;</c>, <c>&lt;double,T&gt;</c>, <c>&lt;T,double&gt;</c>, <c>Dictionary&lt;string, double&gt;</c>,
/// <c>List&lt;double[]&gt;</c>, <c>&lt;double?&gt;</c>, and nested generics like
/// <c>List&lt;Tensor&lt;double&gt;&gt;</c>.</para>
///
/// <para><b>What is left alone</b> — the <c>double</c> keyword anywhere else: <c>double x = 0.0;</c>,
/// <c>(double)y</c>, <c>double.IsNaN(z)</c>, <c>typeof(double)</c>, and identifiers that merely
/// contain the text (<c>myDouble</c>). A <c>where T : double</c> constraint (which is not even legal
/// C#) is also untouched because <c>double</c> there is preceded by <c>:</c>, not <c>&lt;</c>/<c>,</c>.</para>
/// </summary>
internal static class GeneratedTestFloatify
{
    // Bounded so we only ever touch a generic type ARGUMENT, never the `double` keyword.
    // .NET supports variable-length lookbehind, so the leading "\s*" is fine.
    // 1s timeout per the repo's ReDoS policy (the pattern is linear, so it never actually trips).
    private static readonly Regex GenericDoubleArg = new Regex(
        @"(?<=[<,]\s*)double(?=\s*[>,\[?])",
        RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    /// <summary>Returns <paramref name="code"/> with every generic <c>double</c> argument rewritten to <c>float</c>.</summary>
    internal static string Floatify(string code)
    {
        if (string.IsNullOrEmpty(code)) return code;
        return GenericDoubleArg.Replace(code, "float");
    }

    /// <summary>True if <see cref="Floatify"/> would change <paramref name="code"/> (i.e. it contains a generic <c>double</c> argument).</summary>
    internal static bool ContainsGenericDoubleArg(string code)
        => !string.IsNullOrEmpty(code) && GenericDoubleArg.IsMatch(code);
}
