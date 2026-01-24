using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Playground;

/// <summary>
/// Tests that verify all playground examples compile successfully.
/// These tests extract code snippets from ExampleService.cs and compile them using Roslyn.
/// </summary>
public class PlaygroundExampleCompilationTests
{
    private readonly ITestOutputHelper _output;
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(5);

    public PlaygroundExampleCompilationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Verifies that all playground examples compile without errors.
    /// </summary>
    [Fact]
    [Trait("Category", "Playground")]
    public void AllPlaygroundExamples_ShouldCompile()
    {
        // Get the path to ExampleService.cs
        var exampleServicePath = FindExampleServicePath();
        if (string.IsNullOrEmpty(exampleServicePath))
        {
            // In CI environments, fail the test if the file is not found
            // This prevents silent pass when paths are misconfigured
            var isCi = Environment.GetEnvironmentVariable("CI") is not null ||
                       Environment.GetEnvironmentVariable("TF_BUILD") is not null ||
                       Environment.GetEnvironmentVariable("GITHUB_ACTIONS") is not null;
            if (isCi)
            {
                Assert.Fail("Could not find ExampleService.cs - this should not happen in CI");
            }
            _output.WriteLine("WARNING: Could not find ExampleService.cs - skipping test in local environment");
            return;
        }

        var content = File.ReadAllText(exampleServicePath);
        var examples = ExtractCodeExamples(content);

        _output.WriteLine($"Found {examples.Count} code examples to test");

        var failures = new List<(string Id, List<string> Errors)>();
        var successes = 0;

        foreach (var (id, code) in examples)
        {
            var errors = CompileCode(id, code);
            if (errors.Count > 0)
            {
                failures.Add((id, errors));
                _output.WriteLine($"FAIL: {id}");
                foreach (var error in errors.Take(3)) // Show first 3 errors
                {
                    _output.WriteLine($"  - {error}");
                }
            }
            else
            {
                successes++;
                _output.WriteLine($"PASS: {id}");
            }
        }

        _output.WriteLine($"\nResults: {successes}/{examples.Count} passed, {failures.Count} failed");

        if (failures.Count > 0)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"{failures.Count} playground examples failed to compile:");
            foreach (var (id, errors) in failures.Take(10)) // Show first 10 failures
            {
                sb.AppendLine($"\n  {id}:");
                foreach (var error in errors.Take(3))
                {
                    sb.AppendLine($"    - {error}");
                }
            }
            if (failures.Count > 10)
            {
                sb.AppendLine($"\n  ... and {failures.Count - 10} more failures");
            }

            Assert.Fail(sb.ToString());
        }
    }

    /// <summary>
    /// Extracts code examples from the ExampleService.cs content.
    /// </summary>
    private static List<(string Id, string Code)> ExtractCodeExamples(string content)
    {
        var examples = new List<(string Id, string Code)>();

        // Find all Id = "xxx" occurrences
        var idPattern = new Regex(
            @"Id\s*=\s*""([^""]+)""",
            RegexOptions.None,
            RegexTimeout);

        var idMatches = idPattern.Matches(content);

        foreach (Match idMatch in idMatches)
        {
            var id = idMatch.Groups[1].Value;
            var startPos = idMatch.Index;

            // Find the Code = @"..." that follows this Id
            // Look for Code = @" starting from after the Id
            var searchStart = startPos;
            var codeStart = content.IndexOf("Code = @\"", searchStart, StringComparison.Ordinal);

            if (codeStart == -1 || codeStart > startPos + 2000) // Code should be within 2000 chars of Id
            {
                continue;
            }

            // Find the end of the verbatim string
            // We need to find @" and then find the closing " (not "")
            var verbatimStart = codeStart + 9; // Length of "Code = @\""
            var codeEnd = FindVerbatimStringEnd(content, verbatimStart);

            if (codeEnd == -1)
            {
                continue;
            }

            var code = content.Substring(verbatimStart, codeEnd - verbatimStart);

            // Unescape the verbatim string (double quotes become single quotes)
            code = code.Replace("\"\"", "\"");

            examples.Add((id, code));
        }

        return examples;
    }

    /// <summary>
    /// Finds the end of a verbatim string literal.
    /// In verbatim strings, "" is an escaped quote, while a single " ends the string.
    /// </summary>
    private static int FindVerbatimStringEnd(string content, int startPos)
    {
        var pos = startPos;
        while (pos < content.Length)
        {
            var quotePos = content.IndexOf('"', pos);
            if (quotePos == -1)
            {
                return -1;
            }

            // Check if this is an escaped quote (followed by another quote)
            if (quotePos + 1 < content.Length && content[quotePos + 1] == '"')
            {
                // Skip the escaped quote
                pos = quotePos + 2;
                continue;
            }

            // This is the end of the verbatim string
            return quotePos;
        }

        return -1;
    }

    /// <summary>
    /// Compiles the code and returns any errors.
    /// </summary>
    private List<string> CompileCode(string id, string code)
    {
        var errors = new List<string>();

        try
        {
            // Wrap user code in a class structure (similar to playground execution)
            var wrappedCode = WrapUserCode(code);

            // Create syntax tree
            var syntaxTree = CSharpSyntaxTree.ParseText(wrappedCode);

            // Get references
            var references = GetMetadataReferences();

            // Create compilation
            var compilation = CSharpCompilation.Create(
                $"PlaygroundExample_{id}",
                new[] { syntaxTree },
                references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
                    .WithOptimizationLevel(OptimizationLevel.Release)
                    .WithAllowUnsafe(false));

            // Get diagnostics
            var diagnostics = compilation.GetDiagnostics()
                .Where(d => d.Severity == DiagnosticSeverity.Error)
                .ToList();

            foreach (var diagnostic in diagnostics)
            {
                errors.Add(diagnostic.GetMessage());
            }
        }
        catch (Exception ex)
        {
            errors.Add($"Exception during compilation: {ex.Message}");
        }

        return errors;
    }

    /// <summary>
    /// Wraps user code in a compilable class structure.
    /// </summary>
    private static string WrapUserCode(string code)
    {
        // Extract using statements from user code
        var lines = code.Split('\n');
        var usings = new List<string>();
        var codeLines = new List<string>();

        foreach (var line in lines)
        {
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("using ") && (trimmed.EndsWith(";") || trimmed.TrimEnd().EndsWith(";")))
            {
                usings.Add(line);
            }
            else if (!string.IsNullOrWhiteSpace(trimmed) || codeLines.Count > 0)
            {
                codeLines.Add(line);
            }
        }

        var sb = new StringBuilder();

        // Default usings - include common AiDotNet namespaces
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine("using System.Linq;");
        sb.AppendLine("using System.Text;");
        sb.AppendLine("using System.Threading.Tasks;");
        sb.AppendLine("using System.IO;");
        sb.AppendLine("using AiDotNet;");
        sb.AppendLine("using AiDotNet.Data.Loaders;");
        sb.AppendLine("using AiDotNet.Enums;");
        sb.AppendLine("using AiDotNet.Models;");
        sb.AppendLine("using AiDotNet.Models.Options;");
        sb.AppendLine("using AiDotNet.Tensors;");
        sb.AppendLine("using AiDotNet.Tensors.Helpers;");
        sb.AppendLine("using AiDotNet.Tensors.LinearAlgebra;");

        // User usings (deduplicated)
        // Only skip exact "using System;" to avoid filtering System.* namespaces
        foreach (var u in usings.Distinct())
        {
            if (!u.Trim().Equals("using System;", StringComparison.Ordinal))
            {
                sb.AppendLine(u);
            }
        }

        sb.AppendLine();
        sb.AppendLine("public class UserProgram");
        sb.AppendLine("{");
        sb.AppendLine("    public static async Task<string> ExecuteAsync()");
        sb.AppendLine("    {");
        sb.AppendLine("        var __output = new StringWriter();");
        sb.AppendLine("        var __originalOut = Console.Out;");
        sb.AppendLine("        Console.SetOut(__output);");
        sb.AppendLine("        try");
        sb.AppendLine("        {");

        // Indent user code
        foreach (var line in codeLines)
        {
            sb.AppendLine("            " + line);
        }

        sb.AppendLine("        }");
        sb.AppendLine("        finally");
        sb.AppendLine("        {");
        sb.AppendLine("            Console.SetOut(__originalOut);");
        sb.AppendLine("        }");
        sb.AppendLine("        return __output.ToString();");
        sb.AppendLine("    }");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Gets metadata references for compilation.
    /// Includes core runtime assemblies, loaded assemblies, and AiDotNet assemblies from the output directory.
    /// </summary>
    private static List<MetadataReference> GetMetadataReferences()
    {
        var references = new List<MetadataReference>();
        var addedPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Get the runtime directory
        var runtimeDir = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        // Core runtime assemblies
        var coreAssemblies = new[]
        {
            "System.Runtime.dll",
            "System.Console.dll",
            "System.Collections.dll",
            "System.Linq.dll",
            "System.Threading.Tasks.dll",
            "System.Private.CoreLib.dll",
            "netstandard.dll",
            "System.dll",
            "System.Core.dll",
            "System.Text.RegularExpressions.dll",
            "System.Collections.Concurrent.dll",
            "System.Memory.dll",
            "System.Numerics.Vectors.dll"
        };

        foreach (var assembly in coreAssemblies)
        {
            var path = Path.Combine(runtimeDir, assembly);
            if (File.Exists(path) && addedPaths.Add(path))
            {
                references.Add(MetadataReference.CreateFromFile(path));
            }
        }

        // Add all loaded assemblies that might be needed for AiDotNet code
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            try
            {
                if (!assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location) && addedPaths.Add(assembly.Location))
                {
                    references.Add(MetadataReference.CreateFromFile(assembly.Location));
                }
            }
            catch
            {
                // Skip assemblies that can't be loaded
            }
        }

        // Scan the output directory for AiDotNet assemblies that may not be loaded yet
        // This ensures all AiDotNet types are available for compilation even if not yet used
        var outputDir = AppDomain.CurrentDomain.BaseDirectory;
        try
        {
            foreach (var dllPath in Directory.GetFiles(outputDir, "AiDotNet*.dll"))
            {
                if (addedPaths.Add(dllPath))
                {
                    try
                    {
                        references.Add(MetadataReference.CreateFromFile(dllPath));
                    }
                    catch
                    {
                        // Skip assemblies that can't be loaded as metadata references
                    }
                }
            }
        }
        catch
        {
            // Skip if we can't access the output directory
        }

        return references;
    }

    /// <summary>
    /// Finds the path to ExampleService.cs by searching from the test project location.
    /// </summary>
    private static string? FindExampleServicePath()
    {
        // Start from the test assembly location and navigate up
        var currentDir = AppDomain.CurrentDomain.BaseDirectory;

        // Try to find the src directory by navigating up
        var dir = new DirectoryInfo(currentDir);
        while (dir is not null)
        {
            var srcPath = Path.Combine(dir.FullName, "src", "AiDotNet.Playground", "Services", "ExampleService.cs");
            if (File.Exists(srcPath))
            {
                return srcPath;
            }

            // Also check parallel to tests directory
            var testsParent = Path.Combine(dir.FullName, "AiDotNet.Playground", "Services", "ExampleService.cs");
            if (File.Exists(testsParent))
            {
                return testsParent;
            }

            dir = dir.Parent;
        }

        // Fallback: try common relative development paths
        var fallbackPaths = new[]
        {
            "../../../../../src/AiDotNet.Playground/Services/ExampleService.cs",
            "../../../../../../src/AiDotNet.Playground/Services/ExampleService.cs"
        };

        foreach (var path in fallbackPaths)
        {
            if (File.Exists(path))
            {
                return path;
            }
        }

        return null;
    }
}
