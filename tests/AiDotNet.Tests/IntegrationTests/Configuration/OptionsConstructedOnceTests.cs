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
/// Fails if a constructor builds its options object twice — once for the base initializer and
/// again for its own field — which hands the base class and the derived class two different
/// objects whenever the caller passes no options.
/// </summary>
/// <remarks>
/// <para>
/// The shape this rejects:
/// </para>
/// <code>
///     : base(options ?? new KMeansOptions&lt;T&gt;())      // instance A
///     {
///         _options = options ?? new KMeansOptions&lt;T&gt;();  // instance B
/// </code>
/// <para>
/// The <c>??</c> evaluates on both lines, so on the options-null path the base holds one object
/// and the derived class another. Both start as freshly-constructed defaults, so they agree until
/// something mutates one — at which point the halves of the model disagree about their own
/// configuration and nothing reports it.
/// </para>
/// <para>
/// <b>This is not hypothetical.</b> <c>MatryoshkaEmbedding</c> had exactly this shape: its base
/// built a <c>TransformerEmbeddingOptions</c> (<c>EmbeddingDimension</c> 768) while the derived
/// class built a <c>MatryoshkaEmbeddingOptions</c> (1536). The base's copy sizes every layer, so a
/// model documented and tested as 1536 wide was built 768 wide and three
/// <c>MatryoshkaEmbeddingTests</c> failed on the mismatch. Issue #2228 then found the same shape
/// at 34 more sites.
/// </para>
/// <para>
/// <b>Why a guard rather than trusting the fix.</b> The defect is invisible on the path tests
/// normally take: constructing a model with an explicit options object produces one instance and
/// nothing looks wrong. Counting how the affected models are built in this suite,
/// <c>KMeans</c>, <c>DBSCAN</c> and <c>AgglomerativeClustering</c> are never constructed without
/// arguments at all, so no existing test exercises the path this concerns. That is precisely how
/// 34 sites accumulated, and it is why the protection has to be a source rule rather than a
/// behavioural test.
/// </para>
/// <para>
/// <b>Source, not IL.</b> <c>options ?? new X()</c> twice and one shared instance compile to
/// different code, but telling them apart in IL means recognising two <c>newobj</c> sites behind
/// a branch — far more fragile than matching the two lines that spell the mistake out.
/// </para>
/// </remarks>
public class OptionsConstructedOnceTests
{
    private readonly ITestOutputHelper _output;

    public OptionsConstructedOnceTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Constructors that build their options twice.
    /// </summary>
    /// <remarks>
    /// Zero, and it should stay zero — this is a rule, not a ratchet with a backlog behind it.
    /// The 34 sites issue #2228 found were all corrected in the change that added this guard, so
    /// any rise is a newly written instance rather than an un-migrated one.
    /// </remarks>
    private const int AllowedDoubleConstruction = 0;

    /// <summary>Matches <c>: base(options ?? new SomethingOptions&lt;T&gt;()</c>.</summary>
    private static readonly Regex BaseInitializer = new(
        @":\s*base\(\s*options\s*\?\?\s*new\s+\w+", RegexOptions.Compiled);

    /// <summary>Matches <c>_options = options ?? new SomethingOptions&lt;T&gt;();</c>.</summary>
    private static readonly Regex FieldAssignment = new(
        @"_options\s*=\s*options\s*\?\?\s*new\s+\w+", RegexOptions.Compiled);

    [Fact(Timeout = 600000)]
    public async Task NoConstructorBuildsItsOptionsTwice()
    {
        await Task.Yield();

        string root = LocateSourceRoot();
        var offenders = new List<string>();
        int scanned = 0;

        foreach (string file in Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}Generated{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            {
                continue;
            }

            string text;
            try
            {
                text = File.ReadAllText(file);
            }
            catch
            {
                continue;
            }

            scanned++;

            // Both halves must be present. A file with only the base initializer is fine: it
            // forwards the caller's object and keeps no field of its own.
            if (!BaseInitializer.IsMatch(text) || !FieldAssignment.IsMatch(text)) continue;

            // Path.GetRelativePath does not exist on net471, which this project also targets.
            string relative = file.StartsWith(root, StringComparison.Ordinal)
                ? file.Substring(root.Length).TrimStart(Path.DirectorySeparatorChar)
                : file;
            offenders.Add(relative.Replace('\\', '/'));
        }

        _output.WriteLine($"Source files scanned: {scanned}");
        _output.WriteLine($"Constructors building their options twice: {offenders.Count}");
        foreach (string offender in offenders.OrderBy(o => o, StringComparer.Ordinal))
        {
            _output.WriteLine("  " + offender);
        }

        Assert.True(
            offenders.Count <= AllowedDoubleConstruction,
            $"{offenders.Count} constructor(s) build their options object twice, so the base class "
            + "and the derived class hold different instances when the caller passes null:"
            + Environment.NewLine
            + string.Join(Environment.NewLine, offenders.Select(o => "  " + o))
            + Environment.NewLine
            + "Materialize once instead — `: base(options ??= new XOptions<T>())` in the "
            + "initializer and `_options = options;` in the body. `??=` carries non-null flow "
            + "state into the constructor body, so the assignment needs no null-forgiving operator.");
    }

    private static string LocateSourceRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Combine(directory.FullName, "src");
            if (Directory.Exists(candidate) && Directory.Exists(Path.Combine(candidate, "Clustering")))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException(
            "Could not locate the repository's src directory from " + AppContext.BaseDirectory
            + ". This guard reads source because the defect is a source shape.");
    }
}
