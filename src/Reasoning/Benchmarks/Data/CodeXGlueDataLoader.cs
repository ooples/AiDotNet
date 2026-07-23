using Newtonsoft.Json.Linq;

namespace AiDotNet.Reasoning.Benchmarks.Data;

/// <summary>
/// Loads CodeXGLUE-style datasets from JSONL (one JSON object per line).
/// </summary>
/// <remarks>
/// <para>
/// CodeXGLUE is a suite of code understanding/generation datasets. This loader intentionally does not ship any datasets
/// with the repository; callers provide a path to a local JSONL file.
/// </para>
/// <para><b>For Beginners:</b> A JSONL file is a text file where each line is one JSON record.
/// That makes it easy to stream or partially load large datasets.
/// </para>
/// </remarks>
public static class CodeXGlueDataLoader
{
    /// <summary>
    /// Loads a CodeXGLUE dataset from a JSONL file.
    /// </summary>
    /// <param name="filePath">Path to a JSONL file.</param>
    /// <param name="sourceField">Field name for the input (prompt/source code/text).</param>
    /// <param name="targetField">Field name for the expected output.</param>
    /// <param name="idField">Optional field name for the record identifier.</param>
    /// <param name="categoryField">Optional field name for category grouping.</param>
    public static Task<List<CodeXGlueProblem>> LoadFromFileAsync(
        string filePath,
        string sourceField = "source",
        string targetField = "target",
        string idField = "id",
        string categoryField = "category")
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"CodeXGLUE dataset not found: {filePath}");
        }

        var problems = new List<CodeXGlueProblem>();
        var lines = File.ReadAllLines(filePath); // net4xx compatible

        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            try
            {
                var json = JObject.Parse(line);

                var source = json[sourceField]?.ToString() ?? string.Empty;
                var target = json[targetField]?.ToString() ?? string.Empty;

                if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(target))
                {
                    continue;
                }

                problems.Add(new CodeXGlueProblem
                {
                    Id = json[idField]?.ToString() ?? string.Empty,
                    Source = source,
                    Target = target,
                    Category = json[categoryField]?.ToString() ?? string.Empty
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"CodeXGLUE: Failed to parse line: {ex.Message}");
            }
        }

        return Task.FromResult(problems);
    }
}

/// <summary>
/// A single CodeXGLUE problem record (task-agnostic).
/// </summary>
public sealed class CodeXGlueProblem
{
    /// <summary>
    /// Optional identifier from the dataset.
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Input text/code ("source").
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Expected output text/code ("target").
    /// </summary>
    public string Target { get; set; } = string.Empty;

    /// <summary>
    /// Optional category for grouping metrics.
    /// </summary>
    public string Category { get; set; } = string.Empty;
}

