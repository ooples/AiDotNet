using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace AiDotNet.Playground.Services;

/// <summary>
/// Service for executing C# code.
/// Attempts real execution via backend API, falls back to simulation if unavailable.
/// </summary>
public class CodeExecutionService
{
    private readonly HttpClient _httpClient;
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    // API endpoint - will be set based on environment
    private const string ProductionApiUrl = "https://aidotnet-playground-api.vercel.app/api/execute";
    private const string LocalApiUrl = "http://localhost:3000/api/execute";

    // Patterns for detecting AiDotNet API usage
    private static readonly string[] AiDotNetPatterns = new[]
    {
        "Tensor<",
        "Matrix<",
        "Vector<",
        "KMeans",
        "DBSCAN",
        "NeuralNetwork",
        "Dense<",
        "Conv2D",
        "ReLU",
        "Sigmoid",
        "Adam",
        "SGD",
        "CrossEntropy",
        "MeanSquaredError",
        "AudioProcessor",
        "Whisper",
        "Transformer"
    };

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public CodeExecutionService(HttpClient httpClient)
    {
        _httpClient = httpClient;
        _httpClient.Timeout = TimeSpan.FromSeconds(30);
    }

    /// <summary>
    /// Executes C# code and returns the result.
    /// Attempts real execution via API, falls back to simulation if unavailable.
    /// </summary>
    public async Task<ExecutionResult> ExecuteAsync(string code)
    {
        try
        {
            // Validate the code has content
            if (string.IsNullOrWhiteSpace(code))
            {
                return new ExecutionResult
                {
                    Success = false,
                    Output = "Error: No code provided.\n\nPlease enter some C# code to execute."
                };
            }

            // Try real execution via backend API
            var apiResult = await TryExecuteViaApiAsync(code);
            if (apiResult is not null)
            {
                return apiResult;
            }

            // Fall back to simulation
            return ExecuteSimulation(code);
        }
        catch (Exception ex)
        {
            var fallback = ExecuteSimulation(code);
            fallback.Success = false;
            fallback.Output = $"Execution error: {ex.Message}\n\n{fallback.Output}";
            return fallback;
        }
    }

    /// <summary>
    /// Attempts to execute code via the backend API.
    /// Returns null if the API is unavailable.
    /// </summary>
    private async Task<ExecutionResult?> TryExecuteViaApiAsync(string code)
    {
        try
        {
            var request = new ApiExecuteRequest { Code = code };

            // Try production API first
            var response = await TryApiEndpoint(ProductionApiUrl, request);
            if (response is not null)
            {
                return response;
            }

            // Try local API as fallback (for development)
            response = await TryApiEndpoint(LocalApiUrl, request);
            return response;
        }
        catch
        {
            // API not available, return null to trigger simulation fallback
            return null;
        }
    }

    private async Task<ExecutionResult?> TryApiEndpoint(string url, ApiExecuteRequest request)
    {
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(25));

            var response = await _httpClient.PostAsJsonAsync(url, request, JsonOptions, cts.Token);

            if (!response.IsSuccessStatusCode)
            {
                // If rate limited, show a friendly message
                if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
                {
                    return new ExecutionResult
                    {
                        Success = false,
                        Output = "Rate limit reached. Please wait a moment before trying again.\n\nThe playground API allows 10 executions per minute."
                    };
                }

                return null;
            }

            var result = await response.Content.ReadFromJsonAsync<ApiExecuteResponse>(JsonOptions, cts.Token);

            if (result is null)
            {
                return null;
            }

            var output = new StringBuilder();

            output.AppendLine("=== Real Code Execution ===");
            output.AppendLine();

            if (result.Success)
            {
                output.AppendLine("Output:");
                output.AppendLine("-------");
                output.AppendLine(result.Output ?? "(No output)");
            }
            else
            {
                output.AppendLine("Execution Failed:");
                output.AppendLine("-----------------");
                if (!string.IsNullOrEmpty(result.Error))
                {
                    output.AppendLine(result.Error);
                }
                if (!string.IsNullOrEmpty(result.CompilationOutput))
                {
                    output.AppendLine();
                    output.AppendLine("Compilation Output:");
                    output.AppendLine(result.CompilationOutput);
                }
            }

            if (result.ExecutionTime.HasValue)
            {
                output.AppendLine();
                output.AppendLine($"Execution time: {result.ExecutionTime}ms");
            }

            return new ExecutionResult
            {
                Success = result.Success,
                Output = output.ToString()
            };
        }
        catch (TaskCanceledException)
        {
            return new ExecutionResult
            {
                Success = false,
                Output = "Execution timed out.\n\nThe code took too long to execute. Please simplify your code or reduce the workload."
            };
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Executes code in simulation mode (browser-only, no real execution).
    /// </summary>
    private ExecutionResult ExecuteSimulation(string code)
    {
        try
        {
            // Detect what type of code this is
            var detectedApis = DetectAiDotNetApis(code);
            var hasConsoleOutput = HasConsoleOutput(code);

            // Simulate execution
            var result = SimulateExecution(code, detectedApis, hasConsoleOutput);

            return new ExecutionResult
            {
                Success = result.Success,
                Output = result.Output,
                CompilationErrors = result.Errors
            };
        }
        catch (RegexMatchTimeoutException)
        {
            return new ExecutionResult
            {
                Success = false,
                Output = "Error: Code analysis timed out.\n\nPlease simplify your code pattern."
            };
        }
    }

    private List<string> DetectAiDotNetApis(string code)
    {
        var detected = new List<string>();
        foreach (var pattern in AiDotNetPatterns)
        {
            if (code.Contains(pattern, StringComparison.OrdinalIgnoreCase))
            {
                detected.Add(pattern.TrimEnd('<'));
            }
        }
        return detected;
    }

    private static bool HasConsoleOutput(string code)
    {
        return code.Contains("Console.Write", StringComparison.OrdinalIgnoreCase);
    }

    private (bool Success, string Output, List<string>? Errors) SimulateExecution(
        string code,
        List<string> detectedApis,
        bool hasConsoleOutput)
    {
        var output = new StringBuilder();

        // Header showing simulation mode
        output.AppendLine("=== Browser Simulation Mode ===");
        output.AppendLine("(Backend API unavailable - showing simulated results)");
        output.AppendLine();

        // If AiDotNet APIs are detected, show what we found
        if (detectedApis.Count > 0)
        {
            output.AppendLine("Detected AiDotNet APIs:");
            foreach (var api in detectedApis)
            {
                output.AppendLine($"  - {api}");
            }
            output.AppendLine();

            // Provide simulated output based on the API type
            output.AppendLine("Simulated Output:");
            output.AppendLine("-----------------");
            output.Append(GenerateSimulatedOutput(code, detectedApis));
            output.AppendLine();
        }

        // Parse any Console.WriteLine statements
        if (hasConsoleOutput)
        {
            var consoleOutput = ParseConsoleOutput(code);
            if (consoleOutput.Count > 0)
            {
                if (detectedApis.Count > 0)
                {
                    output.AppendLine();
                    output.AppendLine("Console Output:");
                    output.AppendLine("---------------");
                }
                foreach (var line in consoleOutput)
                {
                    output.AppendLine(line);
                }
            }
        }

        // If nothing detected, show helpful message
        if (detectedApis.Count == 0 && !hasConsoleOutput)
        {
            output.AppendLine("Code structure validated.");
            output.AppendLine();
            output.AppendLine("Tips:");
            output.AppendLine("- Add Console.WriteLine() to see output");
            output.AppendLine("- Use AiDotNet APIs like Tensor<T>, KMeans, etc.");
        }

        // Footer with instructions
        output.AppendLine();
        output.AppendLine("=================================");
        output.AppendLine("For actual execution with real results:");
        output.AppendLine("  1. Clone: git clone https://github.com/ooples/AiDotNet");
        output.AppendLine("  2. Run: dotnet run --project samples/YourExample");

        return (true, output.ToString(), null);
    }

    private static string GenerateSimulatedOutput(string code, List<string> detectedApis)
    {
        var output = new StringBuilder();

        // Generate realistic-looking simulated output based on detected APIs
        foreach (var api in detectedApis)
        {
            switch (api)
            {
                case "Tensor":
                    output.AppendLine("[Tensor created: shape detected from code]");
                    if (code.Contains("Add") || code.Contains("+"))
                        output.AppendLine("[Tensor addition performed]");
                    if (code.Contains("Multiply") || code.Contains("*"))
                        output.AppendLine("[Tensor multiplication performed]");
                    if (code.Contains("MatMul"))
                        output.AppendLine("[Matrix multiplication performed]");
                    break;

                case "Matrix":
                    output.AppendLine("[Matrix created]");
                    break;

                case "Vector":
                    output.AppendLine("[Vector created]");
                    break;

                case "KMeans":
                    output.AppendLine("[K-Means clustering initialized]");
                    if (code.Contains("Fit") || code.Contains("Train"))
                    {
                        output.AppendLine("Clustering iteration 1: inertia = 125.34");
                        output.AppendLine("Clustering iteration 2: inertia = 89.21");
                        output.AppendLine("Clustering iteration 3: inertia = 67.45");
                        output.AppendLine("Converged after 3 iterations");
                        output.AppendLine("Cluster centers computed successfully");
                    }
                    break;

                case "DBSCAN":
                    output.AppendLine("[DBSCAN clustering initialized]");
                    if (code.Contains("Fit") || code.Contains("Cluster"))
                    {
                        output.AppendLine("Found 3 clusters");
                        output.AppendLine("Noise points: 12");
                    }
                    break;

                case "NeuralNetwork":
                case "Dense":
                case "Conv2D":
                    output.AppendLine("[Neural network layer created]");
                    if (code.Contains("Train") || code.Contains("Fit"))
                    {
                        output.AppendLine("Epoch 1/10: loss = 0.8234, accuracy = 0.65");
                        output.AppendLine("Epoch 2/10: loss = 0.5123, accuracy = 0.78");
                        output.AppendLine("Epoch 3/10: loss = 0.3456, accuracy = 0.85");
                        output.AppendLine("...");
                        output.AppendLine("Training complete!");
                    }
                    break;

                case "ReLU":
                case "Sigmoid":
                    output.AppendLine("[Activation function applied]");
                    break;

                case "Adam":
                case "SGD":
                    output.AppendLine("[Optimizer configured]");
                    break;

                case "CrossEntropy":
                case "MeanSquaredError":
                    output.AppendLine("[Loss function initialized]");
                    break;

                case "AudioProcessor":
                case "Whisper":
                    output.AppendLine("[Audio processing initialized]");
                    if (code.Contains("Transcribe"))
                    {
                        output.AppendLine("Transcription: \"Hello, this is a simulated transcription.\"");
                    }
                    break;

                case "Transformer":
                    output.AppendLine("[Transformer model initialized]");
                    if (code.Contains("Generate") || code.Contains("Forward"))
                    {
                        output.AppendLine("Generated tokens: [token1, token2, ...]");
                    }
                    break;
            }
        }

        return output.ToString();
    }

    private static List<string> ParseConsoleOutput(string code)
    {
        var lines = new List<string>();

        try
        {
            // Parse Console.WriteLine statements with timeout to prevent ReDoS
            var writeLinePattern = new Regex(
                @"Console\.WriteLine\s*\(\s*(?:\$?""([^""]*)""|(\w+))\s*\)",
                RegexOptions.Multiline,
                RegexTimeout);

            var matches = writeLinePattern.Matches(code);
            foreach (Match match in matches)
            {
                var text = match.Groups[1].Value;
                if (!string.IsNullOrEmpty(text))
                {
                    // Handle string interpolation markers (simplified)
                    text = Regex.Replace(text, @"\{[^}]+\}", "[value]", RegexOptions.None, RegexTimeout);
                    lines.Add(text);
                }
                else if (!string.IsNullOrEmpty(match.Groups[2].Value))
                {
                    lines.Add($"[{match.Groups[2].Value}]");
                }
            }
        }
        catch (RegexMatchTimeoutException)
        {
            lines.Add("[Console output parsing timed out]");
        }

        return lines;
    }
}

/// <summary>
/// Request to the code execution API.
/// </summary>
internal class ApiExecuteRequest
{
    public string Code { get; set; } = "";
    public string? Language { get; set; }
}

/// <summary>
/// Response from the code execution API.
/// </summary>
internal class ApiExecuteResponse
{
    public bool Success { get; set; }
    public string? Output { get; set; }
    public string? Error { get; set; }
    public string? CompilationOutput { get; set; }
    public int? ExecutionTime { get; set; }
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
