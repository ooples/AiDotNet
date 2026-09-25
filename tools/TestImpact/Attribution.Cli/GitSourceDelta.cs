using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;
using AiDotNet.TestImpact;

internal static partial class GitSourceDelta
{
    public static SourceDelta Read(string repository, string before, string after)
    {
        string oldCommit = Git(repository, "rev-parse", "--verify", before + "^{commit}").Trim();
        string newCommit = Git(repository, "rev-parse", "--verify", after + "^{commit}").Trim();
        if (before != oldCommit || after != newCommit) throw new InvalidDataException("Diff inputs must be exact commit identities.");
        var oldLines = new List<ChangedLines>();
        var newLines = new List<ChangedLines>();
        bool unmapped = false;
        string[] paths = Git(repository, "diff", "--no-ext-diff", "--no-renames", "--name-only", "-z", before, after, "--")
            .Split('\0', StringSplitOptions.RemoveEmptyEntries);
        foreach (string path in paths)
        {
            if (!path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase)) { unmapped = true; continue; }
            string patch = Git(repository, "-c", "diff.algorithm=myers", "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--unified=0", before, after, "--", path);
            bool hunk = false;
            foreach (string line in patch.Split('\n'))
            {
                Match match = Hunk().Match(line);
                if (!match.Success) continue;
                hunk = true;
                oldLines.Add(new(path, Number(match, 1), match.Groups[2].Success ? Number(match, 2) : 1));
                newLines.Add(new(path, Number(match, 3), match.Groups[4].Success ? Number(match, 4) : 1));
            }
            // Binary diffs, mode-only edits and empty-file changes are not no-ops.
            if (!hunk || patch.Contains("old mode ", StringComparison.Ordinal)) unmapped = true;
        }
        return new(before, after, oldLines.ToArray(), newLines.ToArray(), unmapped);
    }

    private static int Number(Match match, int group) => int.Parse(match.Groups[group].Value, CultureInfo.InvariantCulture);
    [GeneratedRegex(@"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", RegexOptions.CultureInvariant)]
    private static partial Regex Hunk();

    private static string Git(string repository, params string[] arguments)
    {
        var start = new ProcessStartInfo("git") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
        start.ArgumentList.Add("-C"); start.ArgumentList.Add(repository);
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using Process process = Process.Start(start) ?? throw new IOException("Cannot start git diff.");
        Task<string> errors = process.StandardError.ReadToEndAsync();
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0) throw new IOException(errors.GetAwaiter().GetResult());
        return output;
    }
}
