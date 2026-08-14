using System.Text.RegularExpressions;

namespace AiDotNet.Generators;

/// <summary>
/// Extracts a generated vision model's explicitly configured spatial contract.
/// </summary>
/// <remarks>
/// Test-scaffold constructors sometimes intentionally replace a paper-scale model with a bounded
/// architecture. The input fixture must follow that constructor instead of independently falling
/// back to the generic vision size; otherwise attention and activation costs can grow quadratically
/// while the test appears to exercise the bounded model.
/// </remarks>
internal static class GeneratedVisionFixtureShape
{
    private static readonly Regex InputHeightPattern = new(
        @"\binputHeight\s*:\s*(\d+)",
        RegexOptions.CultureInvariant);

    private static readonly Regex InputWidthPattern = new(
        @"\binputWidth\s*:\s*(\d+)",
        RegexOptions.CultureInvariant);

    /// <summary>
    /// Reads literal <c>inputHeight</c> and <c>inputWidth</c> named arguments from a generated
    /// constructor expression.
    /// </summary>
    internal static bool TryGetExplicitArchitectureSpatialSize(
        string constructorExpression,
        out int height,
        out int width)
    {
        height = 0;
        width = 0;

        if (string.IsNullOrWhiteSpace(constructorExpression))
            return false;

        Match heightMatch = InputHeightPattern.Match(constructorExpression);
        Match widthMatch = InputWidthPattern.Match(constructorExpression);
        if (!heightMatch.Success || !widthMatch.Success)
            return false;

        return int.TryParse(heightMatch.Groups[1].Value, out height)
            && int.TryParse(widthMatch.Groups[1].Value, out width)
            && height > 0
            && width > 0;
    }
}
