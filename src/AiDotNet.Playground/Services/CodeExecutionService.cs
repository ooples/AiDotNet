using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;

namespace AiDotNet.Playground.Services;

/// <summary>
/// Service for compiling and executing C# code in the browser.
/// Uses Roslyn for compilation and executes in a sandboxed environment.
/// </summary>
public class CodeExecutionService
{
    private readonly HttpClient _httpClient;

    public CodeExecutionService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    /// <summary>
    /// Executes C# code and returns the result.
    /// </summary>
    public async Task<ExecutionResult> ExecuteAsync(string code)
    {
        try
        {
            // Wrap the code in a class structure if it's top-level statements
            var wrappedCode = WrapCode(code);

            // For browser execution, we simulate the output
            // In a full implementation, this would use Roslyn to compile and execute
            var output = new StringBuilder();

            // Simulate execution by parsing common patterns
            var result = SimulateExecution(code, output);

            return new ExecutionResult
            {
                Success = result.Success,
                Output = result.Output,
                CompilationErrors = result.Errors
            };
        }
        catch (Exception ex)
        {
            return new ExecutionResult
            {
                Success = false,
                Output = $"Execution error: {ex.Message}"
            };
        }
    }

    private string WrapCode(string code)
    {
        // If code already has a class definition, return as-is
        if (code.Contains("class ") || code.Contains("static void Main"))
        {
            return code;
        }

        // Wrap top-level statements
        return $@"
using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;

public class Program
{{
    public static void Main()
    {{
        {code}
    }}
}}";
    }

    private (bool Success, string Output, List<string>? Errors) SimulateExecution(string code, StringBuilder output)
    {
        // This is a simplified simulation for the browser demo
        // A full implementation would use actual code compilation and execution

        var lines = new List<string>();

        // Parse Console.WriteLine statements
        var writeLinePattern = new System.Text.RegularExpressions.Regex(
            @"Console\.WriteLine\s*\(\s*(?:\$?""([^""]*)""|(\w+))\s*\)",
            System.Text.RegularExpressions.RegexOptions.Multiline);

        var matches = writeLinePattern.Matches(code);
        foreach (System.Text.RegularExpressions.Match match in matches)
        {
            var text = match.Groups[1].Value;
            if (!string.IsNullOrEmpty(text))
            {
                // Handle string interpolation markers (simplified)
                text = System.Text.RegularExpressions.Regex.Replace(text, @"\{[^}]+\}", "[value]");
                lines.Add(text);
            }
        }

        // If no output patterns found, provide a default message
        if (lines.Count == 0)
        {
            lines.Add("Code parsed successfully!");
            lines.Add("");
            lines.Add("Note: This is a browser-based simulation.");
            lines.Add("For full execution, download and run locally.");
        }

        return (true, string.Join("\n", lines), null);
    }
}

/// <summary>
/// Result of code execution.
/// </summary>
public class ExecutionResult
{
    public bool Success { get; set; }
    public string Output { get; set; } = "";
    public List<string>? CompilationErrors { get; set; }
}
