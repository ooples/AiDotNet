using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using System.Text.RegularExpressions;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;

namespace AiDotNet.Serving.Sandboxing.Execution;

public static class CompilationDiagnosticParser
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    public static List<CompilationDiagnostic> Parse(ProgramLanguage language, string compileOutput)
    {
        if (string.IsNullOrWhiteSpace(compileOutput))
        {
            return new List<CompilationDiagnostic>();
        }

        return language switch
        {
            ProgramLanguage.C => ParseGccJson(compileOutput, tool: "gcc"),
            ProgramLanguage.CPlusPlus => ParseGccJson(compileOutput, tool: "g++"),
            ProgramLanguage.Rust => ParseRustcJson(compileOutput),
            ProgramLanguage.Java => ParseJavacText(compileOutput),
            ProgramLanguage.CSharp => ParseDotNetBuildText(compileOutput),
            _ => new List<CompilationDiagnostic>()
        };
    }

    private static List<CompilationDiagnostic> ParseGccJson(string json, string tool)
    {
        var diagnostics = new List<CompilationDiagnostic>();

        foreach (var root in EnumerateJsonRoots(json))
        {
            diagnostics.AddRange(ParseGccJsonRoot(root, tool));
        }

        return diagnostics;
    }

    private static IEnumerable<JsonElement> EnumerateJsonRoots(string payload)
    {
        var trimmed = payload.Trim();

        if (TryParseJsonDocument(trimmed, out var doc))
        {
            using (doc)
            {
                yield return doc.RootElement.Clone();
            }

            yield break;
        }

        foreach (var line in payload.Split('\n'))
        {
            var candidate = line.Trim();
            if (candidate.Length == 0)
            {
                continue;
            }

            if (!TryParseJsonDocument(candidate, out var lineDoc))
            {
                continue;
            }

            using (lineDoc)
            {
                yield return lineDoc.RootElement.Clone();
            }
        }
    }

    private static bool TryParseJsonDocument(string json, [NotNullWhen(true)] out JsonDocument? document)
    {
        document = null;

        try
        {
            document = JsonDocument.Parse(json);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static IEnumerable<CompilationDiagnostic> ParseGccJsonRoot(JsonElement root, string tool)
    {
        if (root.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in root.EnumerateArray())
            {
                foreach (var diag in ParseGccJsonRoot(item, tool))
                {
                    yield return diag;
                }
            }

            yield break;
        }

        if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("diagnostics", out var diagnostics) && diagnostics.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in diagnostics.EnumerateArray())
            {
                foreach (var diag in ParseGccJsonRoot(item, tool))
                {
                    yield return diag;
                }
            }

            yield break;
        }

        if (root.ValueKind != JsonValueKind.Object)
        {
            yield break;
        }

        var severity = CompilationDiagnosticSeverity.Error;
        if (root.TryGetProperty("kind", out var kindProp))
        {
            severity = MapSeverity(kindProp.GetString());
        }
        else if (root.TryGetProperty("level", out var levelProp))
        {
            severity = MapSeverity(levelProp.GetString());
        }

        var message = root.TryGetProperty("message", out var messageProp)
            ? messageProp.ValueKind == JsonValueKind.String
                ? messageProp.GetString()
                : messageProp.ToString()
            : null;

        var code = root.TryGetProperty("option", out var optionProp) && optionProp.ValueKind == JsonValueKind.String
            ? optionProp.GetString()
            : null;

        (string? filePath, int? line, int? column) = ExtractGccLocation(root);

        if (string.IsNullOrWhiteSpace(message))
        {
            yield break;
        }

        yield return new CompilationDiagnostic
        {
            Severity = severity,
            Message = message!,
            Code = code,
            FilePath = filePath,
            Line = line,
            Column = column,
            Tool = tool
        };
    }

    private static (string? FilePath, int? Line, int? Column) ExtractGccLocation(JsonElement root)
    {
        if (root.TryGetProperty("locations", out var locations) && locations.ValueKind == JsonValueKind.Array)
        {
            foreach (var loc in locations.EnumerateArray())
            {
                if (loc.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                if (!loc.TryGetProperty("caret", out var caret) || caret.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                var file = caret.TryGetProperty("file", out var fileProp) && fileProp.ValueKind == JsonValueKind.String
                    ? fileProp.GetString()
                    : null;

                var line = caret.TryGetProperty("line", out var lineProp) && lineProp.TryGetInt32(out var l)
                    ? l
                    : (int?)null;

                var column = caret.TryGetProperty("column", out var columnProp) && columnProp.TryGetInt32(out var c)
                    ? c
                    : (int?)null;

                return (file, line, column);
            }
        }

        return (null, null, null);
    }

    private static List<CompilationDiagnostic> ParseRustcJson(string payload)
    {
        var diagnostics = new List<CompilationDiagnostic>();

        foreach (var root in EnumerateJsonRoots(payload))
        {
            if (root.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            if (!root.TryGetProperty("message", out var messageProp) || messageProp.ValueKind != JsonValueKind.String)
            {
                continue;
            }

            var level = root.TryGetProperty("level", out var levelProp) && levelProp.ValueKind == JsonValueKind.String
                ? levelProp.GetString()
                : null;

            var severity = MapSeverity(level);

            var code = root.TryGetProperty("code", out var codeProp) && codeProp.ValueKind == JsonValueKind.Object &&
                       codeProp.TryGetProperty("code", out var codeValue) && codeValue.ValueKind == JsonValueKind.String
                ? codeValue.GetString()
                : null;

            (string? filePath, int? line, int? column) = ExtractRustcLocation(root);

            diagnostics.Add(new CompilationDiagnostic
            {
                Severity = severity,
                Message = messageProp.GetString()!,
                Code = code,
                FilePath = filePath,
                Line = line,
                Column = column,
                Tool = "rustc"
            });
        }

        return diagnostics;
    }

    private static (string? FilePath, int? Line, int? Column) ExtractRustcLocation(JsonElement root)
    {
        if (!root.TryGetProperty("spans", out var spans) || spans.ValueKind != JsonValueKind.Array)
        {
            return (null, null, null);
        }

        JsonElement? primary = null;

        foreach (var span in spans.EnumerateArray())
        {
            if (span.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            if (span.TryGetProperty("is_primary", out var isPrimary) && isPrimary.ValueKind == JsonValueKind.True)
            {
                primary = span;
                break;
            }

            primary ??= span;
        }

        if (primary is null)
        {
            return (null, null, null);
        }

        var filePath = primary.Value.TryGetProperty("file_name", out var fileProp) && fileProp.ValueKind == JsonValueKind.String
            ? fileProp.GetString()
            : null;

        var line = primary.Value.TryGetProperty("line_start", out var lineProp) && lineProp.TryGetInt32(out var l)
            ? l
            : (int?)null;

        var column = primary.Value.TryGetProperty("column_start", out var colProp) && colProp.TryGetInt32(out var c)
            ? c
            : (int?)null;

        return (filePath, line, column);
    }

    private static List<CompilationDiagnostic> ParseJavacText(string text)
    {
        var diagnostics = new List<CompilationDiagnostic>();
        var regex = new Regex(
            @"^(?<file>[^:\r\n]+):(?<line>\d+):\s*(?<severity>error|warning):\s*(?<message>.+)$",
            RegexOptions.Multiline | RegexOptions.CultureInvariant,
            RegexTimeout);

        foreach (Match match in regex.Matches(text))
        {
            var severity = MapSeverity(match.Groups["severity"].Value);

            diagnostics.Add(new CompilationDiagnostic
            {
                Severity = severity,
                Message = match.Groups["message"].Value.Trim(),
                FilePath = match.Groups["file"].Value.Trim(),
                Line = TryParseInt(match.Groups["line"].Value),
                Tool = "javac"
            });
        }

        return diagnostics;
    }

    private static List<CompilationDiagnostic> ParseDotNetBuildText(string text)
    {
        var diagnostics = new List<CompilationDiagnostic>();

        var regex = new Regex(
            @"^(?<file>.+?)\((?<line>\d+),(?<col>\d+)\):\s*(?<severity>error|warning)\s*(?<code>[A-Z]{2}\d+):\s*(?<message>.+)$",
            RegexOptions.Multiline | RegexOptions.CultureInvariant,
            RegexTimeout);

        foreach (Match match in regex.Matches(text))
        {
            diagnostics.Add(new CompilationDiagnostic
            {
                Severity = MapSeverity(match.Groups["severity"].Value),
                Message = match.Groups["message"].Value.Trim(),
                Code = match.Groups["code"].Value.Trim(),
                FilePath = match.Groups["file"].Value.Trim(),
                Line = TryParseInt(match.Groups["line"].Value),
                Column = TryParseInt(match.Groups["col"].Value),
                Tool = "dotnet"
            });
        }

        return diagnostics;
    }

    private static int? TryParseInt(string value) =>
        int.TryParse(value, out var parsed) ? parsed : (int?)null;

    private static CompilationDiagnosticSeverity MapSeverity(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return CompilationDiagnosticSeverity.Error;
        }

        return value.Trim().ToLowerInvariant() switch
        {
            "warning" => CompilationDiagnosticSeverity.Warning,
            "note" => CompilationDiagnosticSeverity.Info,
            "help" => CompilationDiagnosticSeverity.Info,
            "info" => CompilationDiagnosticSeverity.Info,
            _ => CompilationDiagnosticSeverity.Error
        };
    }
}
