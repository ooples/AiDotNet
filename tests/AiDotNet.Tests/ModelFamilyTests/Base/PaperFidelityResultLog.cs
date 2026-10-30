using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// The kind of paper-fidelity check a <see cref="PaperFidelityResultLog"/> record describes.
/// </summary>
public enum PaperFidelityCheck
{
    /// <summary>A generated option-sensitivity case: one options property of one model.</summary>
    OptionSensitivity,

    /// <summary>A learns-a-toy-task test on a family base.</summary>
    ToyTask,

    /// <summary>A generated paper-defaults test: one options type's constructor defaults.</summary>
    PaperDefaults,
}

/// <summary>
/// Writes one machine-readable line per paper-fidelity case when the
/// <c>AIDOTNET_PAPER_FIDELITY_RESULTS</c> environment variable names a directory.
/// </summary>
/// <remarks>
/// <para>
/// The paper-fidelity workflows (.github/workflows/paper-fidelity-nightly.yml and the pull-request job in
/// sonarcloud.yml) set the variable; every other run leaves it unset and nothing is written. Each test host
/// appends to its own <c>paper-fidelity-&lt;process id&gt;.jsonl</c>, one JSON object per line:
/// </para>
/// <code>
/// {"check":"OptionSensitivity","fixture":"...DQNAgentTests","model":"AiDotNet...DQNAgent`1","property":"WarmupSteps","outcome":"Insensitive","failure":"..."}
/// </code>
/// <para>
/// <c>outcome</c> is the measured verdict, recorded before the runtime baseline is applied, so a case that is
/// skipped as a known defect still reports what it measured. <c>failure</c> is null for a passing outcome.
/// A case whose measurement threw is recorded with outcome <c>Error</c> and the exception, and a null
/// <c>model</c> when the model was never built. tools/PaperFidelity/ConvertTo-RuntimeFidelityBaseline.ps1 turns
/// these records into <see cref="RuntimeFidelityBaseline"/> entries.
/// </para>
/// </remarks>
public static class PaperFidelityResultLog
{
    /// <summary>The environment variable naming the directory records are written to.</summary>
    public const string DirectoryVariable = "AIDOTNET_PAPER_FIDELITY_RESULTS";

    /// <summary>The outcome recorded for a case whose measurement threw.</summary>
    public const string ErrorOutcome = "Error";

    private static readonly object s_gate = new();

    private static readonly string s_fileName =
        "paper-fidelity-" + Process.GetCurrentProcess().Id.ToString(CultureInfo.InvariantCulture) + ".jsonl";

    /// <summary>
    /// Appends one record, or does nothing when <see cref="DirectoryVariable"/> is unset. Never throws: a
    /// result that cannot be written must not change the verdict of the test that produced it.
    /// </summary>
    /// <param name="check">Which check produced the record.</param>
    /// <param name="fixtureType">The concrete test class.</param>
    /// <param name="modelKey">The model's generic type definition name, or null when no model was built.</param>
    /// <param name="property">The option, test or options-type name the case is about.</param>
    /// <param name="outcome">The measured verdict.</param>
    /// <param name="failure">What was observed when the outcome is not a pass; null otherwise.</param>
    public static void Record(PaperFidelityCheck check, Type fixtureType, string? modelKey, string property, string outcome, string? failure)
    {
        string? directory = Environment.GetEnvironmentVariable(DirectoryVariable);
        if (string.IsNullOrWhiteSpace(directory) || fixtureType is null)
        {
            return;
        }

        var line = new StringBuilder(256)
            .Append("{\"check\":").Append(Quote(check.ToString()))
            .Append(",\"fixture\":").Append(Quote(fixtureType.FullName ?? fixtureType.Name))
            .Append(",\"model\":").Append(modelKey is null ? "null" : Quote(modelKey))
            .Append(",\"property\":").Append(Quote(property))
            .Append(",\"outcome\":").Append(Quote(outcome))
            .Append(",\"failure\":").Append(failure is null ? "null" : Quote(failure))
            .Append('}')
            .ToString();

        try
        {
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, s_fileName);
            lock (s_gate)
            {
                File.AppendAllText(path, line + "\n", new UTF8Encoding(false));
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private static string Quote(string value)
    {
        var sb = new StringBuilder(value.Length + 2).Append('"');
        foreach (char c in value)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < ' ')
                    {
                        sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        sb.Append(c);
                    }

                    break;
            }
        }

        return sb.Append('"').ToString();
    }
}
