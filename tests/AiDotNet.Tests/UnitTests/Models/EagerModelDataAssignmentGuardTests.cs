using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// No shipped model may build its metadata bytes eagerly (#2096, #1830).
/// </summary>
/// <remarks>
/// <para>
/// <see cref="AiDotNet.Models.ModelMetadata{T}"/> materializes <c>ModelData</c> on first read via
/// <c>ModelDataProvider</c>, but nothing stops a model from assigning <c>ModelData</c> directly
/// again. That is precisely how this drifted: #1830 fixed the sites that existed then, and 290 more
/// were either missed or added afterwards, including 15 in the crash-risk form
/// <c>ModelData = this.Serialize()</c> that #1830 was specifically about.
/// </para>
/// <para>
/// The existing <c>ModelMetadataLazyModelDataTests</c> prove the mechanism defers correctly. They
/// cannot prove the models use it, and a per-model behavioural test would cover only the models
/// someone remembered to list. Reading the source is the only check that scales to every model and
/// catches the next one added.
/// </para>
/// <para>
/// Serializing is not merely wasted work. <c>Serialize()</c> runs
/// <c>ModelPersistenceGuard.EnforceBeforeSerialize()</c>, so an eager assignment makes
/// <c>AiModelBuilder.BuildAsync</c> a licensed operation; and where the call is wrapped to swallow
/// the licence error, the caller receives an empty array indistinguishable from a model that
/// genuinely has no weights.
/// </para>
/// </remarks>
public class EagerModelDataAssignmentGuardTests
{
    /// <summary>
    /// Assignments to <c>ModelData</c>, whether in an object initializer or through an instance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Two forms, because catching only the first leaves a hole exactly where the next regression
    /// would land: <c>ModelData = ...</c> at the start of a line is the object-initializer form all
    /// 290 converted sites used, and <c>something.ModelData = ...</c> is the post-construction form
    /// several base classes already use for their JSON config blobs. None of those serialize weights
    /// today, but nothing stopped one from being written that way, and a guard blind to it would
    /// report success.
    /// </para>
    /// <para>
    /// <c>SerializedModelData</c> cannot match either branch: the first is anchored at line start,
    /// and the second requires the dot to sit immediately before <c>ModelData</c>. Comment lines are
    /// filtered separately so the prose in ModelMetadata.cs that quotes the old pattern is not read
    /// as code.
    /// </para>
    /// </remarks>
    private static readonly Regex EagerAssignment = new(
        @"(?:^[ \t]*|\.)ModelData[ \t]*=[ \t]*(?<rhs>.+)$",
        RegexOptions.Compiled);

    private static readonly Regex Serializes = new(
        @"\b(Serialize|SerializeForMetadata|SafeSerialize|SafeSerializeMaterializedModel)[ \t]*\(",
        RegexOptions.Compiled);

    [Fact]
    public void NoShippedModelAssignsModelDataEagerly()
    {
        string sourceRoot = LocateSourceRoot();
        var files = Directory.GetFiles(sourceRoot, "*.cs", SearchOption.AllDirectories);

        // Non-vacuity: an empty or tiny scan must fail rather than report success. Without this a
        // wrong path would make the whole guard pass while checking nothing.
        Assert.True(
            files.Length > 500,
            $"Only {files.Length} source files found under '{sourceRoot}'; the scan is not reaching "
            + "the library and this guard would pass without checking anything.");

        var offenders = new List<string>();

        foreach (string file in files)
        {
            string[] lines = File.ReadAllLines(file);
            for (int i = 0; i < lines.Length; i++)
            {
                string trimmed = lines[i].TrimStart();
                if (trimmed.StartsWith("//", StringComparison.Ordinal)
                    || trimmed.StartsWith("*", StringComparison.Ordinal))
                {
                    continue;
                }

                Match match = EagerAssignment.Match(lines[i]);
                if (!match.Success || !Serializes.IsMatch(match.Groups["rhs"].Value))
                {
                    continue;
                }

                offenders.Add(
                    $"{Path.GetRelativePath(sourceRoot, file).Replace('\\', '/')}:{i + 1}  {trimmed}");
            }
        }

        Assert.True(
            offenders.Count == 0,
            $"{offenders.Count} site(s) still serialize into ModelData eagerly. Every AiModelResult "
            + "construction captures metadata for the model it wraps, so each of these serializes a "
            + "full model on every BuildAsync and discards the bytes unread. Use "
            + "'ModelDataProvider = () => <expr>' instead, which produces the bytes on first read:"
            + Environment.NewLine
            + string.Join(Environment.NewLine, offenders.Take(25))
            + (offenders.Count > 25 ? $"{Environment.NewLine}... and {offenders.Count - 25} more" : string.Empty));
    }

    /// <summary>
    /// The library's <c>src</c> directory, found by walking up to the solution file.
    /// </summary>
    /// <remarks>
    /// Throws rather than skipping when the tree cannot be found. A guard that quietly opts out
    /// when it cannot see the source is the same silent pass it exists to prevent.
    /// </remarks>
    private static string LocateSourceRoot()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            string candidate = Path.Combine(current.FullName, "src");
            if (File.Exists(Path.Combine(current.FullName, "AiDotNet.sln")) && Directory.Exists(candidate))
            {
                return candidate;
            }

            current = current.Parent;
        }

        throw new DirectoryNotFoundException(
            $"Could not find AiDotNet.sln walking up from '{AppContext.BaseDirectory}', so the "
            + "source tree this guard reads is unavailable.");
    }
}
