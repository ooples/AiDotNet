using System.IO;
using System.Linq;
using System.Text;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Encoding;

/// <summary>
/// Tests to ensure UTF-8 encoding integrity across the codebase.
/// Prevents corruption of special characters like ×, ², ³, ≈, √, ±
/// </summary>
public class Utf8EncodingTests
{
    private const char ReplacementCharacter = '\uFFFD'; // U+FFFD

    private static string GetRelativePathCompat(string relativeTo, string path)
    {
#if NET462 || NET471
        // Path.GetRelativePath not available in .NET Framework 4.6.2/4.7.1
        if (path.StartsWith(relativeTo, StringComparison.OrdinalIgnoreCase))
        {
            return path.Substring(relativeTo.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        }
        return path;
#else
        return Path.GetRelativePath(relativeTo, path);
#endif
    }

    /// <summary>
    /// Ensures no UTF-8 replacement characters exist in source files.
    /// The replacement character (U+FFFD) indicates encoding corruption.
    /// </summary>
    [Fact]
    public void SourceFiles_ShouldNotContainReplacementCharacter()
    {
        var rootDir = GetRepositoryRoot();
        var csFiles = Directory.GetFiles(rootDir, "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains("bin") && !f.Contains("obj") && !f.Contains(".git") && !f.Contains(".worktrees") && !f.Contains("worktrees"))
            .ToList();

        var filesWithIssues = new List<(string file, int count)>();

        foreach (var file in csFiles)
        {
            var content = File.ReadAllText(file, System.Text.Encoding.UTF8);
            var count = content.Count(c => c == ReplacementCharacter);

            if (count > 0)
            {
                filesWithIssues.Add((GetRelativePathCompat(rootDir, file), count));
            }
        }

        if (filesWithIssues.Any())
        {
            var message = new StringBuilder();
            message.AppendLine("UTF-8 encoding corruption detected:");
            message.AppendLine();

            foreach (var (file, count) in filesWithIssues)
            {
                message.AppendLine($"  {file}: {count} replacement character(s)");
            }

            message.AppendLine();
            message.AppendLine("Common corrupted characters:");
            message.AppendLine($"  {ReplacementCharacter} should be × (multiplication, U+00D7)");
            message.AppendLine($"  {ReplacementCharacter} should be ² (superscript 2, U+00B2)");
            message.AppendLine($"  {ReplacementCharacter} should be ³ (superscript 3, U+00B3)");
            message.AppendLine($"  {ReplacementCharacter} should be ≈ (approximately, U+2248)");
            message.AppendLine($"  {ReplacementCharacter} should be √ (square root, U+221A)");
            message.AppendLine($"  {ReplacementCharacter} should be ± (plus-minus, U+00B1)");
            message.AppendLine();
            message.AppendLine("Run: python3 scripts/fix-encoding.py");

            Assert.Fail(message.ToString());
        }
    }

    private static string GetRepositoryRoot()
    {
        var dir = Directory.GetCurrentDirectory();
        while (!Directory.Exists(Path.Combine(dir, ".git")))
        {
            var parent = Directory.GetParent(dir);
            if (parent == null)
                return Directory.GetCurrentDirectory();
            dir = parent.FullName;
        }
        return dir;
    }
}
