using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Checks that each options property's declared default matches the default its own documentation
/// states.
/// </summary>
/// <remarks>
/// <para>
/// This is the tractable half of #2090's paper-fidelity item. A table of published values for
/// hundreds of models cannot be produced without a source for each one, and inventing them would be
/// worse than having no table: a citation nobody checked is indistinguishable from a correct one.
/// What CAN be verified with no external knowledge is whether the code agrees with the claim it
/// already makes — <c>&lt;value&gt;Defaults to 0.001 (arXiv:2404.05892, Appendix H).&lt;/value&gt;</c>
/// next to <c>= 0.001</c>.
/// </para>
/// <para>
/// A disagreement is a real defect rather than a formatting nit. The doc comment is what a caller
/// reads to decide whether to override a value, so a stale one sends them to change something that
/// is already different, or to leave something alone that is not what they think it is. Both
/// instances found when this guard was written were stale docs rather than wrong code, which is the
/// direction that silently misleads.
/// </para>
/// <para>
/// <b>Source, not IL.</b> Doc comments do not survive compilation into the assembly this suite
/// loads, so unlike the unread-property guard this one has to read the files. It accepts the usual
/// cost of that: it sees text, not semantics.
/// </para>
/// </remarks>
public class DocumentedDefaultsFidelityTests
{
    private readonly ITestOutputHelper _output;

    public DocumentedDefaultsFidelityTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Properties whose declared default contradicts their documented one.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Lower this as they are reconciled; never raise it.</b> A rise means a default was changed
    /// without its documentation, or documentation was written that the code does not honour.
    /// </para>
    /// <para>
    /// Established at 0: the two found when this guard was written were fixed with it.
    /// <c>LocallyWeightedRegressionOptions.Bandwidth</c> documented a default of 1.0 while declaring
    /// 0.0 — the code is right and the doc predated it, because 0.0 is a deliberate sentinel
    /// selecting Cleveland and Devlin's adaptive span, which the model calls "an escape hatch".
    /// <c>AffinityPropagationOptions.Damping</c> documented 0.5 while declaring 0.8.
    /// </para>
    /// </remarks>
    private const int FidelityBaseline = 0;

    /// <summary>
    /// Matches an explicitly stated default, tolerating thousands separators.
    /// </summary>
    /// <remarks>
    /// Deliberately narrow. A first draft took the first number anywhere in the value text and
    /// reported 97 mismatches, nearly all of them range descriptions ("between 0 and 1, defaulting
    /// to 0.3") rather than defaults. Three of the five survivors were then <c>10,000</c> read as
    /// <c>10</c>, which is why the separator is handled here rather than left to chance.
    /// </remarks>
    private static readonly Regex DocumentedDefault = new(
        @"default(?:s|ing)?\s*(?:value\s*(?:of|is))?\s*(?:to|:|is|=)?\s*"
        + @"(?:<c>\s*)?(-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex ValueBlock = new(
        @"///\s*<value>(.*?)</value>", RegexOptions.Singleline | RegexOptions.Compiled);

    private static readonly Regex NumericProperty = new(
        @"public\s+(?:int|double|float)\s+(\w+)\s*\{\s*get;\s*set;\s*\}\s*=\s*([^;]+);",
        RegexOptions.Compiled);

    private static readonly Regex Literal = new(
        @"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[fdmFDM]?$", RegexOptions.Compiled);

    [Fact(Timeout = 600000)]
    public async Task DeclaredDefaultsMatchTheirDocumentation_DoesNotRegress()
    {
        await Task.Yield();

        string root = LocateSourceRoot();
        var mismatches = new List<string>();
        int checkedCount = 0;

        foreach (string file in Directory.EnumerateFiles(root, "*Options.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}Generated{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            {
                continue;
            }

            string[] lines;
            try
            {
                lines = File.ReadAllLines(file);
            }
            catch
            {
                continue;
            }

            for (int i = 0; i < lines.Length; i++)
            {
                var property = NumericProperty.Match(lines[i]);
                if (!property.Success) continue;

                string declared = property.Groups[2].Value.Trim().TrimEnd('f', 'd', 'm', 'F', 'D', 'M');
                if (!Literal.IsMatch(declared)) continue;

                // the contiguous doc block immediately above
                int start = i - 1;
                var doc = new List<string>();
                while (start >= 0 && lines[start].TrimStart().StartsWith("///", StringComparison.Ordinal))
                {
                    doc.Add(lines[start]);
                    start--;
                }

                if (doc.Count == 0) continue;
                doc.Reverse();

                var value = ValueBlock.Match(string.Join("\n", doc));
                if (!value.Success) continue;

                var documented = DocumentedDefault.Match(value.Groups[1].Value);
                if (!documented.Success) continue;

                checkedCount++;

                string text = documented.Groups[1].Value.Replace(",", string.Empty);
                if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double documentedValue)
                    || !double.TryParse(declared, NumberStyles.Float, CultureInfo.InvariantCulture, out double declaredValue))
                {
                    continue;
                }

                if (Math.Abs(declaredValue - documentedValue) <= 1e-12) continue;

                mismatches.Add(
                    $"{Path.GetFileName(file)}.{property.Groups[1].Value}: "
                    + $"declares {declared} but documents {documented.Groups[1].Value}");
            }
        }

        _output.WriteLine($"Properties whose doc names a default: {checkedCount}");
        _output.WriteLine($"Disagreeing with their documentation: {mismatches.Count}");
        foreach (string mismatch in mismatches.OrderBy(m => m, StringComparer.Ordinal))
        {
            _output.WriteLine("  " + mismatch);
        }

        Assert.True(
            mismatches.Count <= FidelityBaseline,
            $"Documented-default mismatches rose from {FidelityBaseline} to {mismatches.Count}. "
            + "A property's declared default contradicts the default its own documentation states, "
            + "so a caller reading the doc is told the wrong number:"
            + Environment.NewLine + string.Join(Environment.NewLine, mismatches)
            + Environment.NewLine
            + "If this instead shows a DROP, lower FidelityBaseline so the progress is recorded in "
            + "the diff rather than silently absorbed.");
    }

    private static string LocateSourceRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Combine(directory.FullName, "src");
            if (Directory.Exists(candidate) && Directory.Exists(Path.Combine(candidate, "NeuralNetworks")))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException(
            "Could not locate the repository's src directory from " + AppContext.BaseDirectory
            + ". This guard reads source because doc comments do not survive compilation.");
    }
}
