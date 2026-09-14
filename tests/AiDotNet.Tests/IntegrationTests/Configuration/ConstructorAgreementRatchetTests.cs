using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Ratchets defect form (4) of issue #2090: one model whose two constructors describe it
/// differently.
/// </summary>
/// <remarks>
/// <para>
/// A model with both an ONNX and a native constructor could assign the same backing field from
/// <c>_options.X</c> in one and a hardcoded literal in the other. The ONNX constructors take an
/// options parameter, so they were discarding values they had been handed: a caller configuring
/// that path had their settings silently ignored, and the model described itself two ways.
/// </para>
/// <para>
/// No other guard sees this. The unread ratchet counts properties nothing reads, and these ARE read
/// — by the sibling constructor — so they never appeared there. The constructor ratchets count
/// parameters, not field assignments. Every instance closed on this branch was found by reading
/// code, which is why the count is now fixed in place here.
/// </para>
/// <para>
/// <b>Source text rather than IL.</b> The two assignments compile to indistinguishable field stores
/// once the literal is folded, so the distinction only exists before compilation. That makes this
/// the one guard in this directory that must read source, and it accepts the usual cost: it sees
/// syntax, not semantics.
/// </para>
/// </remarks>
public class ConstructorAgreementRatchetTests
{
    private readonly ITestOutputHelper _output;

    public ConstructorAgreementRatchetTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Fields assigned from options in one constructor and from a literal in another.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Lower this as instances are fixed; never raise it.</b> A rise means a constructor was
    /// added or edited to hardcode a value its sibling reads from options.
    /// </para>
    /// <para>
    /// Established at 4 after 129 fields across 33 files were migrated to read their options. The
    /// four that remain are deliberate and must NOT be "fixed": <c>DropRate = 0.0</c> in the ONNX
    /// inference constructors of Mask2Former, MixedQueryTransformer, OneFormer and XDecoder.
    /// Dropout must be off at inference, so applying the declared 0.1 there would introduce dropout
    /// into inference — a real defect dressed as consistency. They are listed by name below so a
    /// future sweep does not silently absorb them.
    /// </para>
    /// </remarks>
    private const int AgreementBaseline = 4;

    /// <summary>
    /// Sites where a literal is correct and the options value would be wrong.
    /// </summary>
    private static readonly HashSet<string> DeliberateDivergence = new(StringComparer.Ordinal)
    {
        "Mask2Former._dropRate",
        "MixedQueryTransformer._dropRate",
        "OneFormer._dropRate",
        "XDecoder._dropRate",
    };

    private static readonly Regex LiteralAssignment = new(
        @"^\s*(_\w+)\s*=\s*(?:[0-9][0-9_.eE+\-]*[fdmFDM]?|true|false)\s*;\s*$",
        RegexOptions.Compiled);

    private static readonly Regex OptionsAssignment = new(
        @"^\s*(_\w+)\s*=\s*_?[Oo]ptions\.(\w+)\s*;\s*$", RegexOptions.Compiled);

    private static readonly Regex ConstructorOpen = new(
        @"^\s*(?:public|private|protected|internal)\s+\w+\s*\(", RegexOptions.Compiled);

    [Fact(Timeout = 300000)]
    public async Task ConstructorsDoNotDisagreeAboutOptions_DoesNotRegress()
    {
        await Task.Yield();

        string root = LocateSourceRoot();
        var divergences = new List<string>();

        foreach (string file in Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories))
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

            string typeName = Path.GetFileNameWithoutExtension(file);
            var spans = ConstructorSpans(lines, typeName);
            if (spans.Count < 2) continue;

            // field -> constructors assigning it from a literal / from options
            var literal = new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);
            var fromOptions = new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);

            for (int index = 0; index < spans.Count; index++)
            {
                var (start, end) = spans[index];
                for (int line = start; line <= end && line < lines.Length; line++)
                {
                    var match = LiteralAssignment.Match(lines[line]);
                    if (match.Success)
                    {
                        if (!literal.TryGetValue(match.Groups[1].Value, out var set))
                        {
                            set = new HashSet<int>();
                            literal[match.Groups[1].Value] = set;
                        }

                        set.Add(index);
                        continue;
                    }

                    match = OptionsAssignment.Match(lines[line]);
                    if (match.Success)
                    {
                        if (!fromOptions.TryGetValue(match.Groups[1].Value, out var set))
                        {
                            set = new HashSet<int>();
                            fromOptions[match.Groups[1].Value] = set;
                        }

                        set.Add(index);
                    }
                }
            }

            foreach (var pair in literal.OrderBy(p => p.Key, StringComparer.Ordinal))
            {
                if (!fromOptions.TryGetValue(pair.Key, out var optionCtors)) continue;

                // only a constructor that hardcodes AND never reads the option is a divergence
                if (!pair.Value.Except(optionCtors).Any()) continue;

                string site = $"{typeName}.{pair.Key}";
                if (DeliberateDivergence.Contains(site)) continue;

                divergences.Add(site);
            }
        }

        _output.WriteLine($"Constructors disagreeing about an option: {divergences.Count}");
        foreach (string divergence in divergences.OrderBy(d => d, StringComparer.Ordinal))
        {
            _output.WriteLine("  " + divergence);
        }

        Assert.True(
            divergences.Count <= AgreementBaseline,
            $"Constructor disagreements rose from {AgreementBaseline} to {divergences.Count}. "
            + "A constructor assigns a backing field from a hardcoded literal while a sibling "
            + "constructor reads the same field from options, so the model describes itself two "
            + "ways and one path silently discards what the caller configured:"
            + Environment.NewLine + string.Join(Environment.NewLine, divergences)
            + Environment.NewLine
            + "If this instead shows a DROP, lower AgreementBaseline so the progress is recorded "
            + "in the diff rather than silently absorbed.");
    }

    /// <summary>
    /// Line spans of each constructor body, by brace depth.
    /// </summary>
    private static List<(int Start, int End)> ConstructorSpans(string[] lines, string typeName)
    {
        var spans = new List<(int, int)>();
        var signature = new Regex(
            @"^\s*(?:public|private|protected|internal)\s+" + Regex.Escape(typeName) + @"\s*\(",
            RegexOptions.Compiled);

        int i = 0;
        while (i < lines.Length)
        {
            if (!signature.IsMatch(lines[i]))
            {
                i++;
                continue;
            }

            int j = i;
            int parens = 0;
            bool sawParen = false;
            while (j < lines.Length)
            {
                parens += lines[j].Count(c => c == '(') - lines[j].Count(c => c == ')');
                if (lines[j].IndexOf('(') >= 0) sawParen = true;
                if (sawParen && parens <= 0) break;
                j++;
            }

            while (j < lines.Length && lines[j].IndexOf('{') < 0) j++;
            if (j >= lines.Length) break;

            int depth = 0;
            int k = j;
            while (k < lines.Length)
            {
                depth += lines[k].Count(c => c == '{') - lines[k].Count(c => c == '}');
                if (depth <= 0) break;
                k++;
            }

            spans.Add((j, k));
            i = k + 1;
        }

        return spans;
    }

    /// <summary>
    /// Walks up from the test assembly to the repository's src directory.
    /// </summary>
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
            + ". This guard reads source because the defect it measures does not survive "
            + "compilation.");
    }
}
