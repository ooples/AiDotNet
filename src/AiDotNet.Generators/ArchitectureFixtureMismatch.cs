using System;
using System.Text.RegularExpressions;

namespace AiDotNet.Generators;

/// <summary>
/// Decides whether a generated test fixture's <c>InputShape</c> contradicts the architecture its
/// constructor pins (diagnostic <c>ADNTEST002</c>).
/// </summary>
/// <remarks>
/// <para>
/// This lives apart from <c>TestScaffoldGenerator</c> so the decision can be exercised directly.
/// Driving the generator end to end would need a synthetic model that happens to match one of the
/// hundreds of hardcoded per-model constructor pins, which tests the pin table rather than the rule.
/// </para>
/// <para>
/// There is deliberately NO domain or model-kind exemption. A pinned architecture is not
/// decorative: <c>ResolveLazyLayerShapes</c> performs an architecture-driven warm-up, so a lazy
/// layer resolves its weights from the DECLARED shape while the forward runs on the fixture. A
/// gate that excused "the axes mean something else for this family" was tried and was wrong - the
/// eleven models it would have excused all declared a rank-2 architecture while being fed rank-3
/// mel spectrograms, which is the defect, not an alternative reading of it.
/// </para>
/// </remarks>
public static class ArchitectureFixtureMismatch
{
    /// <summary>ReDoS guard. These patterns run once per emitted fixture over generated text.</summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    private static readonly Regex ArchitecturePin = new Regex(
        @"inputHeight:\s*(\d+)[\s\S]{0,80}?inputWidth:\s*(\d+)",
        RegexOptions.None,
        RegexTimeout);

    /// <remarks>
    /// The terminator is REQUIRED. With it optional this matched a PREFIX of a longer shape - a
    /// 4-element [B, C, H, W] fixture matched its first three entries, so the comparison used
    /// [C, H] instead of [H, W] and reported nonsense such as "AVID fixture [.., 3, 32]". Anchoring
    /// to the close brace/bracket restricts this to genuine 3-element [C, H, W] fixtures, which is
    /// the only form the check is valid for.
    /// </remarks>
    private static readonly Regex FixtureInputShape = new Regex(
        @"InputShape\s*=>\s*(?:ResolveModelDeclaredInputShape\s*\(\s*)?(?:new\s*\[\]\s*\{|\[)\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:\}|\])\s*\)?",
        RegexOptions.None,
        RegexTimeout);

    /// <summary>
    /// Reports whether <paramref name="generatedFixture"/> pins an architecture whose spatial size
    /// disagrees with the <c>InputShape</c> it feeds.
    /// </summary>
    /// <param name="generatedFixture">The emitted fixture source.</param>
    /// <param name="architectureHeight">The pinned <c>inputHeight</c>, when a mismatch is found.</param>
    /// <param name="architectureWidth">The pinned <c>inputWidth</c>, when a mismatch is found.</param>
    /// <param name="fixtureHeight">The fixture's height axis, when a mismatch is found.</param>
    /// <param name="fixtureWidth">The fixture's width axis, when a mismatch is found.</param>
    /// <returns><c>true</c> only when both facts are present, comparable, and disagree.</returns>
    public static bool TryDetect(
        string generatedFixture,
        out int architectureHeight,
        out int architectureWidth,
        out int fixtureHeight,
        out int fixtureWidth)
    {
        architectureHeight = 0;
        architectureWidth = 0;
        fixtureHeight = 0;
        fixtureWidth = 0;

        if (string.IsNullOrEmpty(generatedFixture))
        {
            return false;
        }

        Match architecture;
        Match fixture;
        try
        {
            architecture = ArchitecturePin.Match(generatedFixture);
            fixture = FixtureInputShape.Match(generatedFixture);
        }
        catch (RegexMatchTimeoutException)
        {
            // A fixture large enough to time out is not evidence of a mismatch, and a source
            // generator must not fail a build over its own diagnostic. Stay silent.
            return false;
        }

        if (!architecture.Success || !fixture.Success)
        {
            return false;
        }

        architectureHeight = int.Parse(architecture.Groups[1].Value);
        architectureWidth = int.Parse(architecture.Groups[2].Value);
        fixtureHeight = int.Parse(fixture.Groups[1].Value);
        fixtureWidth = int.Parse(fixture.Groups[2].Value);

        return architectureHeight != fixtureHeight || architectureWidth != fixtureWidth;
    }
}
