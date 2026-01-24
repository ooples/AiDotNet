using System.Reflection;
using System.Runtime.Loader;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.Functions.Worker;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.Extensions.Logging;

// Force load AiDotNet assemblies by referencing them
using AiDotNet;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Playground.Functions;

// Static initializer to force assembly loading
file static class AssemblyLoader
{
    static AssemblyLoader()
    {
        // Reference a type from AiDotNet.Tensors to force load the assembly
        _ = typeof(Tensor<>);
    }
}

public class ExecuteCode
{
    private readonly ILogger<ExecuteCode> _logger;
    private static readonly TimeSpan ExecutionTimeout = TimeSpan.FromSeconds(10);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    // Force load AiDotNet assemblies on class initialization
    private static readonly Type TensorType = typeof(AiDotNet.Tensors.LinearAlgebra.Tensor<>);
    private static readonly Type AiModelBuilderType = typeof(AiDotNet.AiModelBuilder<,,>);

    public ExecuteCode(ILogger<ExecuteCode> logger)
    {
        _logger = logger;
    }

    [Function("execute")]
    public async Task<IActionResult> Run(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)] HttpRequest req)
    {
        // Add CORS headers
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Origin", "*");
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Methods", "POST, OPTIONS");
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Headers", "Content-Type");

        // Handle preflight request
        if (req.Method == "OPTIONS")
        {
            return new OkResult();
        }

        try
        {
            using var reader = new StreamReader(req.Body);
            var requestBody = await reader.ReadToEndAsync();
            var request = JsonSerializer.Deserialize<CodeExecutionRequest>(requestBody, JsonOptions);

            if (request?.Code is null || string.IsNullOrWhiteSpace(request.Code))
            {
                return new BadRequestObjectResult(new CodeExecutionResponse
                {
                    Success = false,
                    Error = "No code provided"
                });
            }

            _logger.LogInformation("Executing code: {CodeLength} characters", request.Code.Length);

            var result = await CompileAndExecuteAsync(request.Code);

            return new OkObjectResult(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing code");
            return new OkObjectResult(new CodeExecutionResponse
            {
                Success = false,
                Error = $"Execution error: {ex.Message}"
            });
        }
    }

    private async Task<CodeExecutionResponse> CompileAndExecuteAsync(string code)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            // Wrap user code in a class structure
            var wrappedCode = WrapUserCode(code);

            // Create syntax tree
            var syntaxTree = CSharpSyntaxTree.ParseText(wrappedCode);

            // Get references
            var references = GetMetadataReferences();

            // Create compilation
            var compilation = CSharpCompilation.Create(
                "UserCodeAssembly",
                new[] { syntaxTree },
                references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
                    .WithOptimizationLevel(OptimizationLevel.Release)
                    .WithAllowUnsafe(false));

            // Compile to memory
            using var ms = new MemoryStream();
            var emitResult = compilation.Emit(ms);

            if (!emitResult.Success)
            {
                var errors = emitResult.Diagnostics
                    .Where(d => d.Severity == DiagnosticSeverity.Error)
                    .Select(d => d.GetMessage())
                    .ToList();

                return new CodeExecutionResponse
                {
                    Success = false,
                    Error = "Compilation failed",
                    CompilationErrors = errors,
                    ExecutionTime = stopwatch.ElapsedMilliseconds
                };
            }

            ms.Seek(0, SeekOrigin.Begin);

            // Load and execute the assembly
            var output = await ExecuteAssemblyAsync(ms);

            stopwatch.Stop();

            return new CodeExecutionResponse
            {
                Success = true,
                Output = output,
                ExecutionTime = stopwatch.ElapsedMilliseconds
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            return new CodeExecutionResponse
            {
                Success = false,
                Error = ex.Message,
                ExecutionTime = stopwatch.ElapsedMilliseconds
            };
        }
    }

    private static string WrapUserCode(string code)
    {
        // Extract using statements from user code
        var lines = code.Split('\n');
        var usings = new List<string>();
        var codeLines = new List<string>();

        foreach (var line in lines)
        {
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("using ") && trimmed.EndsWith(";"))
            {
                usings.Add(line);
            }
            else if (!string.IsNullOrWhiteSpace(trimmed) || codeLines.Count > 0)
            {
                codeLines.Add(line);
            }
        }

        // Build the wrapped code
        var sb = new StringBuilder();

        // Default usings
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine("using System.Linq;");
        sb.AppendLine("using System.Text;");
        sb.AppendLine("using System.Threading.Tasks;");
        sb.AppendLine("using System.IO;");

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

    private static List<MetadataReference> GetMetadataReferences()
    {
        var references = new List<MetadataReference>();

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
            if (File.Exists(path))
            {
                references.Add(MetadataReference.CreateFromFile(path));
            }
        }

        // Add all loaded assemblies that might be needed for AiDotNet code
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            try
            {
                if (!assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location))
                {
                    references.Add(MetadataReference.CreateFromFile(assembly.Location));
                }
            }
            catch
            {
                // Skip assemblies that can't be loaded
            }
        }

        return references;
    }

    /// <summary>
    /// Executes the compiled assembly in a collectible AssemblyLoadContext to allow unloading.
    /// </summary>
    /// <remarks>
    /// Note: The timeout implementation detects when execution exceeds the limit but cannot
    /// forcibly cancel the user's task since it doesn't receive a CancellationToken.
    /// Timed-out tasks will continue running in the background until they complete naturally.
    /// For true cancellation, process isolation would be required.
    /// </remarks>
    private async Task<string> ExecuteAssemblyAsync(MemoryStream assemblyStream)
    {
        // Create a cancellation token for timeout
        using var cts = new CancellationTokenSource(ExecutionTimeout);

        // Use a collectible AssemblyLoadContext to allow unloading after execution
        // This prevents memory leaks from repeated code executions
        var loadContext = new CollectibleAssemblyLoadContext();

        try
        {
            // Load assembly into the collectible context
            assemblyStream.Seek(0, SeekOrigin.Begin);
            var assembly = loadContext.LoadFromStream(assemblyStream);
            var type = assembly.GetType("UserProgram");

            if (type is null)
            {
                return "Error: Could not find UserProgram class";
            }

            var method = type.GetMethod("ExecuteAsync", BindingFlags.Public | BindingFlags.Static);

            if (method is null)
            {
                return "Error: Could not find ExecuteAsync method";
            }

            // Execute with timeout
            // Note: Task.WhenAny detects timeout but cannot cancel the user's task
            // since user code doesn't receive a CancellationToken
            var task = (Task<string>?)method.Invoke(null, null);

            if (task is null)
            {
                return "Error: Method did not return a task";
            }

            var completedTask = await Task.WhenAny(task, Task.Delay(ExecutionTimeout, cts.Token));

            if (completedTask == task)
            {
                return await task;
            }
            else
            {
                return "Error: Execution timed out after 10 seconds";
            }
        }
        catch (OperationCanceledException)
        {
            return "Error: Execution timed out after 10 seconds";
        }
        catch (TargetInvocationException ex)
        {
            return $"Runtime error: {ex.InnerException?.Message ?? ex.Message}";
        }
        catch (Exception ex)
        {
            return $"Execution error: {ex.Message}";
        }
        finally
        {
            // Unload the assembly context to free memory
            loadContext.Unload();
        }
    }

    /// <summary>
    /// A collectible AssemblyLoadContext that allows assemblies to be unloaded after execution.
    /// This prevents memory leaks from repeatedly loading compiled user code.
    /// </summary>
    private class CollectibleAssemblyLoadContext : AssemblyLoadContext
    {
        public CollectibleAssemblyLoadContext() : base(isCollectible: true)
        {
        }

        protected override Assembly? Load(AssemblyName assemblyName)
        {
            // Return null to fall back to default loading behavior for dependencies
            return null;
        }
    }
}

public class CodeExecutionRequest
{
    public string? Code { get; set; }
}

public class CodeExecutionResponse
{
    public bool Success { get; set; }
    public string? Output { get; set; }
    public string? Error { get; set; }
    public List<string>? CompilationErrors { get; set; }
    public long ExecutionTime { get; set; }
}
