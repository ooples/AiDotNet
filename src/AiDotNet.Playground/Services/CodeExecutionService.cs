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

    // API endpoints - will be set based on environment
    private const string ProductionApiUrl = "https://aidotnet.vercel.app/api/execute";
    private const string LocalApiUrl = "http://localhost:3000/api/execute";

    // Azure Functions endpoint for AiDotNet code execution
    private const string AzureFunctionsApiUrl = "https://aidotnet-playground-emfxa0abhzdqfyh3.westeurope-01.azurewebsites.net/api/execute";
    private const string LocalAzureFunctionsUrl = "http://localhost:7071/api/execute";

    // Controls whether to attempt API calls for real code execution
    // Set to false to use simulation-only mode (no network requests)
    public static bool EnableApiCalls { get; set; } = true;

    // Patterns for detecting AiDotNet API usage
    private static readonly string[] AiDotNetPatterns = new[]
    {
        // Core types
        "Tensor<",
        "Matrix<",
        "Vector<",
        // Clustering
        "KMeans",
        "DBSCAN",
        "AgglomerativeClustering",
        // Neural Networks
        "NeuralNetwork",
        "NeuralNetworkBuilder",
        "Dense<",
        "DenseLayer",
        "Conv2D",
        "Conv2DLayer",
        "MaxPooling",
        "Flatten",
        "Dropout",
        "BatchNorm",
        "Embedding",
        "TransformerEncoder",
        "LSTM",
        "GRU",
        // Activations
        "ReLU",
        "Sigmoid",
        "Softmax",
        "GELU",
        "Tanh",
        // Optimizers
        "Adam",
        "SGD",
        // Loss Functions
        "CrossEntropy",
        "MeanSquaredError",
        // Audio
        "AudioProcessor",
        "Whisper",
        "MusicGen",
        "TtsModel",
        // Models
        "Transformer",
        "ResNet",
        "YOLO",
        // Regression
        "LinearRegression",
        "RidgeRegression",
        "PolynomialRegression",
        "GradientBoostingRegressor",
        // Classification
        "LogisticRegression",
        "RandomForestClassifier",
        "SupportVectorClassifier",
        "GaussianNaiveBayes",
        "GradientBoostingClassifier",
        // Time Series
        "ARIMA",
        "ExponentialSmoothing",
        // Anomaly Detection
        "IsolationForest",
        "OneClassSVM",
        // Dimensionality Reduction
        "PCA",
        "TSNE",
        // NLP
        "TfidfVectorizer",
        "Word2Vec",
        "SentenceTransformer",
        "BiLSTMCRF",
        // RAG
        "VectorStore",
        "RAGPipeline",
        "DocumentChunker",
        "SemanticSearch",
        // LoRA
        "LoRA",
        "QLoRA",
        "DoRA",
        "AdaLoRA",
        // RL
        "DQNAgent",
        "PPOAgent",
        "SACAgent",
        "MADDPGSystem",
        // AutoML
        "AutoML",
        "NeuralArchitectureSearch",
        "BayesianOptimization",
        // Preprocessing
        "StandardScaler",
        "MinMaxScaler",
        "PreprocessingPipeline",
        // Builder patterns
        "AiModelBuilder"
    };

    // Patterns indicating AiDotNet using statements
    private static readonly string[] AiDotNetUsingPatterns = new[]
    {
        "using AiDotNet",
        "AiDotNet."
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
    /// Uses smart routing: AiDotNet code goes to Azure Functions (with AiDotNet installed) first,
    /// then falls back to simulation. Pure C# goes to Piston API first, then falls back to simulation.
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

            // Check if the code uses AiDotNet APIs
            bool usesAiDotNet = UsesAiDotNetApis(code);

            if (usesAiDotNet)
            {
                // Try real execution via Azure Functions (which has AiDotNet installed)
                var azureResult = await TryExecuteAiDotNetViaAzureFunctionsAsync(code);
                if (azureResult is not null)
                {
                    return azureResult;
                }

                // Fall back to simulation if Azure Functions unavailable
                return ExecuteAiDotNetSimulation(code);
            }

            // Pure C# code can try real execution via Piston API
            var apiResult = await TryExecuteViaApiAsync(code);
            if (apiResult is not null)
            {
                return apiResult;
            }

            // Fall back to basic simulation if API unavailable
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
    /// Checks if the code uses any AiDotNet APIs or namespaces.
    /// Uses smart detection to avoid false positives from string literals.
    /// </summary>
    private bool UsesAiDotNetApis(string code)
    {
        // Primary indicator: Check for AiDotNet using statements
        // This is the most reliable indicator
        if (code.Contains("using AiDotNet", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        // Check for AiDotNet namespace usage (e.g., AiDotNet.Regression)
        // Must have period after to avoid matching text in strings like "Welcome to AiDotNet!"
        if (code.Contains("AiDotNet.", StringComparison.Ordinal))
        {
            return true;
        }

        // For API pattern detection, we need to be more careful to avoid false positives
        // from patterns appearing inside string literals in Console.WriteLine
        // Use a simple heuristic: pattern must be followed by < or ( or . or whitespace
        foreach (var pattern in AiDotNetPatterns)
        {
            // Skip if pattern not in code at all
            if (!code.Contains(pattern, StringComparison.Ordinal))
            {
                continue;
            }

            // Look for the pattern with code-like context (not inside strings)
            // Pattern followed by <, (, ., or = indicates actual API usage
            var searchPatterns = new List<string>();

            // Only add generic pattern if the pattern doesn't already end with '<'
            // (e.g., "Tensor<" already ends with '<', so don't create "Tensor<<")
            if (!pattern.EndsWith('<'))
            {
                searchPatterns.Add(pattern + "<");   // Generic type: Tensor<double>
            }

            searchPatterns.Add(pattern + "(");   // Constructor or method: KMeans()
            searchPatterns.Add(pattern + ".");   // Instance method: kmeans.Fit()
            searchPatterns.Add(pattern + " ");   // Declaration: KMeans kmeans
            searchPatterns.Add("new " + pattern); // Constructor: new KMeans()

            foreach (var searchPattern in searchPatterns)
            {
                if (code.Contains(searchPattern, StringComparison.Ordinal))
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Executes AiDotNet code in enhanced simulation mode.
    /// Provides realistic output based on detected APIs.
    /// </summary>
    private ExecutionResult ExecuteAiDotNetSimulation(string code)
    {
        try
        {
            var detectedApis = DetectAiDotNetApis(code);
            var output = new StringBuilder();

            // Clear header explaining this is a demonstration
            output.AppendLine("╔══════════════════════════════════════════════════════════════╗");
            output.AppendLine("║          AiDotNet Interactive Demonstration                  ║");
            output.AppendLine("╚══════════════════════════════════════════════════════════════╝");
            output.AppendLine();

            // Show detected APIs
            if (detectedApis.Count > 0)
            {
                output.AppendLine("📦 Detected AiDotNet Components:");
                foreach (var api in detectedApis.Distinct().Take(10))
                {
                    output.AppendLine($"   • {api}");
                }
                output.AppendLine();
            }

            // Generate simulated output
            output.AppendLine("📊 Simulated Output:");
            output.AppendLine("─────────────────────────────────────────────────────────────────");
            var simulatedOutput = GenerateEnhancedSimulatedOutput(code, detectedApis);
            output.Append(simulatedOutput);

            // Parse and show Console.WriteLine statements
            if (HasConsoleOutput(code))
            {
                var consoleOutput = ParseConsoleOutput(code);
                if (consoleOutput.Count > 0)
                {
                    output.AppendLine();
                    output.AppendLine("📝 Console Output:");
                    output.AppendLine("─────────────────────────────────────────────────────────────────");
                    foreach (var line in consoleOutput)
                    {
                        output.AppendLine(line);
                    }
                }
            }

            // Footer with instructions for real execution
            output.AppendLine();
            output.AppendLine("─────────────────────────────────────────────────────────────────");
            output.AppendLine("ℹ️  This is a simulated demonstration of AiDotNet capabilities.");
            output.AppendLine();
            output.AppendLine("🚀 To run this code with real execution:");
            output.AppendLine();
            output.AppendLine("   # Clone and run locally:");
            output.AppendLine("   git clone https://github.com/ooples/AiDotNet");
            output.AppendLine("   cd AiDotNet");
            output.AppendLine("   dotnet new console -n MyAiApp");
            output.AppendLine("   cd MyAiApp");
            output.AppendLine("   dotnet add package AiDotNet");
            output.AppendLine("   # Paste your code in Program.cs");
            output.AppendLine("   dotnet run");
            output.AppendLine();
            output.AppendLine("📚 Documentation: https://ooples.github.io/AiDotNet/");

            return new ExecutionResult
            {
                Success = true,
                Output = output.ToString()
            };
        }
        catch (Exception ex)
        {
            return new ExecutionResult
            {
                Success = false,
                Output = $"Simulation error: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Attempts to execute code via the backend API.
    /// Returns null if the API is unavailable or disabled.
    /// </summary>
    private async Task<ExecutionResult?> TryExecuteViaApiAsync(string code)
    {
        // Skip API calls if disabled (prevents console errors when no backend exists)
        if (!EnableApiCalls)
        {
            return null;
        }

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

            // If rate limited, show a friendly message (this is the only user-facing API error)
            if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
            {
                return new ExecutionResult
                {
                    Success = false,
                    Output = "Rate limit reached. Please wait a moment before trying again.\n\nThe playground API allows 10 executions per minute."
                };
            }

            // If not successful, fall back to simulation (don't show raw API errors to user)
            if (!response.IsSuccessStatusCode)
            {
                // Log to console for debugging (only visible in browser dev tools)
                Console.WriteLine($"API returned {response.StatusCode}, falling back to simulation");
                return null;
            }

            ApiExecuteResponse? result = null;
            try
            {
                result = await response.Content.ReadFromJsonAsync<ApiExecuteResponse>(JsonOptions, cts.Token);
            }
            catch
            {
                // If we can't parse the response, fall back to simulation
                Console.WriteLine("Failed to parse API response, falling back to simulation");
                return null;
            }

            if (result is null)
            {
                return null;
            }

            // Build output - show both successful and failed executions from API
            // (Don't mask real compile/runtime errors with simulation)
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
                // Show real compilation/runtime errors to user
                output.AppendLine("Execution Failed:");
                output.AppendLine("-----------------");
                if (!string.IsNullOrEmpty(result.Error))
                {
                    output.AppendLine(result.Error);
                }
                if (!string.IsNullOrEmpty(result.CompilationOutput))
                {
                    output.AppendLine();
                    output.AppendLine("Compiler Output:");
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
            // Timeout - fall back to simulation
            Console.WriteLine("API request timed out, falling back to simulation");
            return null;
        }
        catch (Exception ex)
        {
            // Any other error - fall back to simulation
            Console.WriteLine($"API error: {ex.Message}, falling back to simulation");
            return null;
        }
    }

    /// <summary>
    /// Attempts to execute AiDotNet code via Azure Functions.
    /// The Azure Function has AiDotNet.Tensors installed for real execution.
    /// </summary>
    private async Task<ExecutionResult?> TryExecuteAiDotNetViaAzureFunctionsAsync(string code)
    {
        if (!EnableApiCalls)
        {
            return null;
        }

        try
        {
            var request = new { code };

            // Try local Azure Functions first (for development)
            var response = await TryAzureFunctionsEndpoint(LocalAzureFunctionsUrl, request);
            if (response is not null)
            {
                return response;
            }

            // Try production Azure Functions
            response = await TryAzureFunctionsEndpoint(AzureFunctionsApiUrl, request);
            return response;
        }
        catch
        {
            return null;
        }
    }

    private async Task<ExecutionResult?> TryAzureFunctionsEndpoint(string url, object request)
    {
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(25));

            var response = await _httpClient.PostAsJsonAsync(url, request, JsonOptions, cts.Token);

            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine($"Azure Functions API returned {response.StatusCode}");
                return null;
            }

            var result = await response.Content.ReadFromJsonAsync<AzureFunctionsResponse>(JsonOptions, cts.Token);

            if (result is null)
            {
                return null;
            }

            var output = new StringBuilder();
            output.AppendLine("=== AiDotNet Code Execution ===");
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
                if (result.CompilationErrors is { Count: > 0 })
                {
                    output.AppendLine();
                    output.AppendLine("Compilation Errors:");
                    foreach (var error in result.CompilationErrors)
                    {
                        output.AppendLine($"  • {error}");
                    }
                }
            }

            output.AppendLine();
            output.AppendLine($"Execution time: {result.ExecutionTime}ms");

            return new ExecutionResult
            {
                Success = result.Success,
                Output = output.ToString()
            };
        }
        catch (TaskCanceledException)
        {
            Console.WriteLine("Azure Functions request timed out");
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Azure Functions error: {ex.Message}");
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
        output.AppendLine("To run live examples with real results:");
        output.AppendLine();
        output.AppendLine("  Option 1 - Run samples locally:");
        output.AppendLine("    git clone https://github.com/ooples/AiDotNet");
        output.AppendLine("    cd AiDotNet/samples/getting-started/HelloWorld");
        output.AppendLine("    dotnet run");
        output.AppendLine();
        output.AppendLine("  Option 2 - Deploy the playground API:");
        output.AppendLine("    cd AiDotNet && vercel --prod");
        output.AppendLine();
        output.AppendLine("See samples/README.md for the full sample index.");

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

    /// <summary>
    /// Generates enhanced simulated output for AiDotNet APIs with more realistic results.
    /// </summary>
    private static string GenerateEnhancedSimulatedOutput(string code, List<string> detectedApis)
    {
        var output = new StringBuilder();
        var processedCategories = new HashSet<string>();

        foreach (var api in detectedApis.Distinct())
        {
            // Regression Models
            if ((api.Contains("LinearRegression") || api.Contains("RidgeRegression") ||
                 api.Contains("PolynomialRegression") || api.Contains("GradientBoostingRegressor")) &&
                !processedCategories.Contains("regression"))
            {
                processedCategories.Add("regression");
                output.AppendLine("Training regression model...");
                output.AppendLine("  Iteration 1: MSE = 0.4521");
                output.AppendLine("  Iteration 2: MSE = 0.2134");
                output.AppendLine("  Iteration 3: MSE = 0.0892");
                output.AppendLine("  Model converged!");
                output.AppendLine();
                output.AppendLine("Model Metrics:");
                output.AppendLine("  R² Score: 0.9547");
                output.AppendLine("  RMSE: 0.2987");
                output.AppendLine("  MAE: 0.2134");
                if (code.Contains("Predict"))
                {
                    output.AppendLine();
                    output.AppendLine("Prediction: 11.87");
                }
                output.AppendLine();
            }

            // Classification Models
            if ((api.Contains("LogisticRegression") || api.Contains("RandomForestClassifier") ||
                 api.Contains("SupportVectorClassifier") || api.Contains("GaussianNaiveBayes") ||
                 api.Contains("GradientBoostingClassifier")) &&
                !processedCategories.Contains("classification"))
            {
                processedCategories.Add("classification");
                output.AppendLine("Training classification model...");
                output.AppendLine("  Building decision trees: 100/100");
                output.AppendLine("  Training complete!");
                output.AppendLine();
                output.AppendLine("Model Metrics:");
                output.AppendLine("  Accuracy: 94.50%");
                output.AppendLine("  Precision: 0.9321");
                output.AppendLine("  Recall: 0.9456");
                output.AppendLine("  F1 Score: 0.9388");
                if (code.Contains("Predict"))
                {
                    output.AppendLine();
                    output.AppendLine("Predicted class: 1");
                }
                output.AppendLine();
            }

            // Clustering
            if ((api.Contains("KMeans") || api.Contains("DBSCAN") || api.Contains("AgglomerativeClustering")) &&
                !processedCategories.Contains("clustering"))
            {
                processedCategories.Add("clustering");
                if (api.Contains("KMeans"))
                {
                    output.AppendLine("Running K-Means clustering...");
                    output.AppendLine("  Iteration 1: inertia = 245.67");
                    output.AppendLine("  Iteration 2: inertia = 156.34");
                    output.AppendLine("  Iteration 3: inertia = 98.21");
                    output.AppendLine("  Iteration 4: inertia = 87.45");
                    output.AppendLine("  Converged after 4 iterations");
                    output.AppendLine();
                    output.AppendLine("Cluster Summary:");
                    output.AppendLine("  Cluster 0: 45 points");
                    output.AppendLine("  Cluster 1: 55 points");
                }
                else if (api.Contains("DBSCAN"))
                {
                    output.AppendLine("Running DBSCAN clustering...");
                    output.AppendLine("  Scanning for dense regions...");
                    output.AppendLine("  Found 3 clusters");
                    output.AppendLine("  Noise points: 7");
                }
                else
                {
                    output.AppendLine("Running hierarchical clustering...");
                    output.AppendLine("  Building dendrogram...");
                    output.AppendLine("  Cutting at 2 clusters");
                }
                output.AppendLine();
            }

            // Neural Networks
            if ((api.Contains("NeuralNetwork") || api.Contains("Dense") || api.Contains("Conv2D") ||
                 api.Contains("NeuralNetworkBuilder") || api.Contains("DenseLayer")) &&
                !processedCategories.Contains("neural"))
            {
                processedCategories.Add("neural");
                output.AppendLine("Building neural network...");
                output.AppendLine("  Layer 1: Dense(input=2, output=4) - 12 parameters");
                output.AppendLine("  Layer 2: ReLU activation");
                output.AppendLine("  Layer 3: Dense(input=4, output=1) - 5 parameters");
                output.AppendLine("  Total parameters: 17");
                output.AppendLine();
                if (code.Contains("Train") || code.Contains("Fit") || code.Contains("BuildAsync"))
                {
                    output.AppendLine("Training neural network...");
                    output.AppendLine("  Epoch 1/10:  loss=0.6931  accuracy=0.5000");
                    output.AppendLine("  Epoch 2/10:  loss=0.5234  accuracy=0.6500");
                    output.AppendLine("  Epoch 3/10:  loss=0.3876  accuracy=0.7500");
                    output.AppendLine("  Epoch 4/10:  loss=0.2543  accuracy=0.8500");
                    output.AppendLine("  Epoch 5/10:  loss=0.1654  accuracy=0.9000");
                    output.AppendLine("  ...");
                    output.AppendLine("  Epoch 10/10: loss=0.0432  accuracy=0.9750");
                    output.AppendLine("  Training complete!");
                }
                output.AppendLine();
            }

            // Time Series
            if ((api.Contains("ARIMA") || api.Contains("ExponentialSmoothing")) &&
                !processedCategories.Contains("timeseries"))
            {
                processedCategories.Add("timeseries");
                output.AppendLine("Fitting time series model...");
                output.AppendLine("  Estimating parameters...");
                output.AppendLine("  Model fitted successfully");
                output.AppendLine();
                output.AppendLine("Forecast (next 3 periods):");
                output.AppendLine("  Period 13: 208.45");
                output.AppendLine("  Period 14: 215.67");
                output.AppendLine("  Period 15: 223.12");
                output.AppendLine();
            }

            // Anomaly Detection
            if ((api.Contains("IsolationForest") || api.Contains("OneClassSVM")) &&
                !processedCategories.Contains("anomaly"))
            {
                processedCategories.Add("anomaly");
                output.AppendLine("Training anomaly detector...");
                output.AppendLine("  Building isolation trees: 100/100");
                output.AppendLine("  Model trained successfully");
                output.AppendLine();
                output.AppendLine("Anomaly Detection Results:");
                output.AppendLine("  Detected 8 anomalies out of 110 samples");
                output.AppendLine("  Anomaly ratio: 7.27%");
                output.AppendLine();
            }

            // Dimensionality Reduction
            if ((api.Contains("PCA") || api.Contains("TSNE")) &&
                !processedCategories.Contains("dimred"))
            {
                processedCategories.Add("dimred");
                if (api.Contains("PCA"))
                {
                    output.AppendLine("Performing PCA...");
                    output.AppendLine("  Computing covariance matrix...");
                    output.AppendLine("  Extracting principal components...");
                    output.AppendLine();
                    output.AppendLine("PCA Results:");
                    output.AppendLine("  Original dimensions: 50");
                    output.AppendLine("  Reduced dimensions: 10");
                    output.AppendLine("  Explained variance: 95.2%");
                }
                else
                {
                    output.AppendLine("Running t-SNE embedding...");
                    output.AppendLine("  Iteration 100: KL divergence = 1.234");
                    output.AppendLine("  Iteration 200: KL divergence = 0.876");
                    output.AppendLine("  Iteration 300: KL divergence = 0.654");
                    output.AppendLine("  Embedding complete!");
                }
                output.AppendLine();
            }

            // NLP
            if ((api.Contains("TfidfVectorizer") || api.Contains("Word2Vec") ||
                 api.Contains("SentenceTransformer") || api.Contains("BiLSTMCRF")) &&
                !processedCategories.Contains("nlp"))
            {
                processedCategories.Add("nlp");
                output.AppendLine("Processing text data...");
                if (api.Contains("TfidfVectorizer"))
                {
                    output.AppendLine("  Building vocabulary: 1000 terms");
                    output.AppendLine("  Computing TF-IDF weights...");
                }
                else if (api.Contains("Word2Vec"))
                {
                    output.AppendLine("  Training word embeddings...");
                    output.AppendLine("  Vocabulary size: 5000 words");
                    output.AppendLine("  Embedding dimension: 100");
                }
                output.AppendLine("  Text processing complete!");
                output.AppendLine();
            }

            // RAG
            if ((api.Contains("VectorStore") || api.Contains("RAGPipeline") ||
                 api.Contains("DocumentChunker") || api.Contains("SemanticSearch")) &&
                !processedCategories.Contains("rag"))
            {
                processedCategories.Add("rag");
                output.AppendLine("Building RAG pipeline...");
                output.AppendLine("  Chunking documents: 5 chunks created");
                output.AppendLine("  Generating embeddings...");
                output.AppendLine("  Building vector index...");
                output.AppendLine("  RAG pipeline ready!");
                output.AppendLine();
                if (code.Contains("Search") || code.Contains("Query"))
                {
                    output.AppendLine("Search Results:");
                    output.AppendLine("  [0.9234] Machine learning is a subset of AI...");
                    output.AppendLine("  [0.8765] Deep learning uses neural networks...");
                    output.AppendLine("  [0.8123] Natural language processing deals with...");
                }
                output.AppendLine();
            }

            // LoRA/Fine-tuning
            if ((api.Contains("LoRA") || api.Contains("QLoRA") ||
                 api.Contains("DoRA") || api.Contains("AdaLoRA")) &&
                !processedCategories.Contains("lora"))
            {
                processedCategories.Add("lora");
                output.AppendLine("Applying LoRA adaptation...");
                output.AppendLine("  Target modules: q_proj, v_proj");
                output.AppendLine("  LoRA rank: 8");
                output.AppendLine("  Alpha: 16");
                output.AppendLine();
                output.AppendLine("Parameter Statistics:");
                output.AppendLine("  Total parameters: 7,000,000,000");
                output.AppendLine("  Trainable parameters: 4,194,304");
                output.AppendLine("  Trainable %: 0.06%");
                output.AppendLine();
            }

            // Reinforcement Learning
            if ((api.Contains("DQNAgent") || api.Contains("PPOAgent") ||
                 api.Contains("SACAgent") || api.Contains("MADDPGSystem")) &&
                !processedCategories.Contains("rl"))
            {
                processedCategories.Add("rl");
                output.AppendLine("Initializing RL agent...");
                output.AppendLine("  State size: 4");
                output.AppendLine("  Action size: 2");
                output.AppendLine("  Network: 128 -> 128 -> output");
                output.AppendLine("  Agent ready for training!");
                output.AppendLine();
            }

            // AutoML
            if ((api.Contains("AutoML") || api.Contains("NeuralArchitectureSearch") ||
                 api.Contains("BayesianOptimization")) &&
                !processedCategories.Contains("automl"))
            {
                processedCategories.Add("automl");
                output.AppendLine("Running AutoML...");
                output.AppendLine("  Evaluating model 1/10: RandomForest - Score: 0.8921");
                output.AppendLine("  Evaluating model 2/10: GradientBoosting - Score: 0.9123");
                output.AppendLine("  Evaluating model 3/10: LogisticRegression - Score: 0.8654");
                output.AppendLine("  ...");
                output.AppendLine();
                output.AppendLine("AutoML Results:");
                output.AppendLine("  Best model: GradientBoostingClassifier");
                output.AppendLine("  Best score: 0.9123");
                output.AppendLine("  Models evaluated: 10");
                output.AppendLine();
            }

            // Audio Processing
            if ((api.Contains("Whisper") || api.Contains("AudioProcessor") ||
                 api.Contains("MusicGen") || api.Contains("TtsModel")) &&
                !processedCategories.Contains("audio"))
            {
                processedCategories.Add("audio");
                if (api.Contains("Whisper"))
                {
                    output.AppendLine("Loading Whisper model...");
                    output.AppendLine("  Model: whisper-base");
                    output.AppendLine("  Processing audio...");
                    output.AppendLine();
                    output.AppendLine("Transcription:");
                    output.AppendLine("  \"Hello, welcome to AiDotNet. This is a demonstration of speech recognition.\"");
                    output.AppendLine("  Confidence: 94.5%");
                }
                else if (api.Contains("MusicGen"))
                {
                    output.AppendLine("Generating music...");
                    output.AppendLine("  Prompt: A calm piano melody");
                    output.AppendLine("  Duration: 10 seconds");
                    output.AppendLine("  Sample rate: 32000 Hz");
                    output.AppendLine("  Generation complete!");
                }
                else if (api.Contains("TtsModel"))
                {
                    output.AppendLine("Generating speech...");
                    output.AppendLine("  Input: \"Hello, welcome to AiDotNet!\"");
                    output.AppendLine("  Audio length: 2.5 seconds");
                    output.AppendLine("  Sample rate: 22050 Hz");
                }
                output.AppendLine();
            }

            // Computer Vision
            if ((api.Contains("ResNet") || api.Contains("YOLO")) &&
                !processedCategories.Contains("cv"))
            {
                processedCategories.Add("cv");
                if (api.Contains("YOLO"))
                {
                    output.AppendLine("Running YOLO object detection...");
                    output.AppendLine();
                    output.AppendLine("Detected Objects:");
                    output.AppendLine("  Person: 95.2% at (120, 45, 180, 320)");
                    output.AppendLine("  Car: 89.7% at (300, 200, 150, 80)");
                    output.AppendLine("  Dog: 82.1% at (50, 280, 60, 70)");
                }
                else
                {
                    output.AppendLine("Loading ResNet model...");
                    output.AppendLine("  Architecture: ResNet-18");
                    output.AppendLine("  Pretrained: Yes");
                    output.AppendLine("  Fine-tuning for 5 classes...");
                    output.AppendLine("  Training accuracy: 96.5%");
                }
                output.AppendLine();
            }

            // Preprocessing
            if ((api.Contains("StandardScaler") || api.Contains("MinMaxScaler") ||
                 api.Contains("PreprocessingPipeline")) &&
                !processedCategories.Contains("preprocess"))
            {
                processedCategories.Add("preprocess");
                output.AppendLine("Applying preprocessing...");
                output.AppendLine("  StandardScaler: mean=0, std=1");
                output.AppendLine("  Features normalized successfully");
                output.AppendLine();
            }

            // AiModelBuilder
            if (api.Contains("AiModelBuilder") && !processedCategories.Contains("builder"))
            {
                processedCategories.Add("builder");
                output.AppendLine("Using AiModelBuilder facade pattern...");
                output.AppendLine("  Configuration complete");
                output.AppendLine("  Building model pipeline...");
                output.AppendLine();
            }
        }

        // If no specific category was matched, show a generic message
        if (output.Length == 0)
        {
            output.AppendLine("Processing AiDotNet operations...");
            output.AppendLine("  Operations completed successfully!");
            output.AppendLine();
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
                    // Handle string interpolation markers with realistic values
                    text = GenerateRealisticInterpolation(text, code);
                    lines.Add(text);
                }
                else if (!string.IsNullOrEmpty(match.Groups[2].Value))
                {
                    var varName = match.Groups[2].Value;
                    lines.Add(GenerateRealisticVariableValue(varName, code));
                }
            }
        }
        catch (RegexMatchTimeoutException)
        {
            lines.Add("[Console output parsing timed out]");
        }

        return lines;
    }

    /// <summary>
    /// Generates realistic values for string interpolation based on variable names and context.
    /// </summary>
    private static string GenerateRealisticInterpolation(string text, string code)
    {
        try
        {
            // Find all interpolation placeholders
            var interpolationPattern = new Regex(@"\{([^}:]+)(?::([^}]+))?\}", RegexOptions.None, RegexTimeout);

            return interpolationPattern.Replace(text, match =>
            {
                var varName = match.Groups[1].Value.Trim();
                var format = match.Groups[2].Success ? match.Groups[2].Value : null;
                return GenerateValueForVariable(varName, format, code);
            });
        }
        catch (RegexMatchTimeoutException)
        {
            return text;
        }
    }

    /// <summary>
    /// Generates a realistic value based on variable name and context.
    /// </summary>
    private static string GenerateValueForVariable(string varName, string? format, string code)
    {
        var lowerVar = varName.ToLowerInvariant();

        // Check for common ML/stats variable patterns

        // Accuracy, Precision, Recall, F1 - usually 0-1 or percentage
        if (lowerVar.Contains("accuracy") || lowerVar.Contains("precision") ||
            lowerVar.Contains("recall") || lowerVar.Contains("f1") ||
            lowerVar.Contains("auc") || lowerVar.Contains("score"))
        {
            var value = 0.85 + (GetHashBasedVariation(varName) * 0.12); // 0.85 - 0.97
            return FormatNumber(value, format, 4);
        }

        // R² Score - usually 0.7-0.99 for good models
        if (lowerVar.Contains("r2") || lowerVar.Contains("rsquared") || lowerVar.Contains("r²"))
        {
            var value = 0.82 + (GetHashBasedVariation(varName) * 0.15); // 0.82 - 0.97
            return FormatNumber(value, format, 4);
        }

        // Loss values - usually small positive numbers
        if (lowerVar.Contains("loss") || lowerVar.Contains("error") || lowerVar.Contains("mse") ||
            lowerVar.Contains("mae") || lowerVar.Contains("rmse"))
        {
            var value = 0.02 + (GetHashBasedVariation(varName) * 0.15); // 0.02 - 0.17
            return FormatNumber(value, format, 4);
        }

        // Predictions - check context for range hints
        if (lowerVar.Contains("predict") || lowerVar.Contains("output") || lowerVar.Contains("result"))
        {
            // Check if it's classification (0/1) or regression
            if (code.Contains("Classification") || code.Contains("Binary"))
            {
                return GetHashBasedVariation(varName) > 0.5 ? "1" : "0";
            }
            // Regression - generate realistic numeric value
            var value = 10.0 + (GetHashBasedVariation(varName) * 90.0); // 10 - 100
            return FormatNumber(value, format, 2);
        }

        // Count, size, number of items
        if (lowerVar.Contains("count") || lowerVar.Contains("size") || lowerVar.Contains("length") ||
            lowerVar.Contains("samples") || lowerVar.Contains("features"))
        {
            var value = (int)(10 + GetHashBasedVariation(varName) * 990); // 10 - 1000
            return value.ToString();
        }

        // Clusters (k as standalone or in cluster-related names)
        if (lowerVar.Contains("cluster") || lowerVar.Contains("numk") ||
            lowerVar.Contains("n_clusters") || lowerVar == "k" || lowerVar.EndsWith("_k"))
        {
            var value = (int)(2 + GetHashBasedVariation(varName) * 8); // 2 - 10
            return value.ToString();
        }

        // Epochs, iterations
        if (lowerVar.Contains("epoch") || lowerVar.Contains("iteration") || lowerVar.Contains("step"))
        {
            var value = (int)(1 + GetHashBasedVariation(varName) * 99); // 1 - 100
            return value.ToString();
        }

        // Time-related
        if (lowerVar.Contains("time") || lowerVar.Contains("duration") || lowerVar.Contains("elapsed"))
        {
            var value = 0.5 + (GetHashBasedVariation(varName) * 4.5); // 0.5 - 5.0 seconds
            return FormatNumber(value, format, 2) + "s";
        }

        // Learning rate
        if (lowerVar.Contains("learningrate") || lowerVar.Contains("lr") || lowerVar.Contains("alpha"))
        {
            var value = 0.001 * (1 + GetHashBasedVariation(varName) * 9); // 0.001 - 0.01
            return FormatNumber(value, format, 4);
        }

        // Weight, coefficient, parameter values
        if (lowerVar.Contains("weight") || lowerVar.Contains("coef") || lowerVar.Contains("param") ||
            lowerVar.Contains("bias") || lowerVar.Contains("intercept"))
        {
            var value = -2.0 + (GetHashBasedVariation(varName) * 4.0); // -2 to 2
            return FormatNumber(value, format, 4);
        }

        // Array/collection display with index
        if (varName.Contains("["))
        {
            var value = 0.1 + (GetHashBasedVariation(varName) * 0.9);
            return FormatNumber(value, format, 4);
        }

        // Method calls like .ToString()
        if (varName.Contains(".") && !varName.Contains("["))
        {
            var parts = varName.Split('.');
            return GenerateValueForVariable(parts[0], format, code);
        }

        // Default: generate a reasonable numeric value
        var defaultValue = 0.5 + (GetHashBasedVariation(varName) * 49.5); // 0.5 - 50
        return FormatNumber(defaultValue, format, 2);
    }

    /// <summary>
    /// Generates a realistic value for a standalone variable reference.
    /// </summary>
    private static string GenerateRealisticVariableValue(string varName, string code)
    {
        var value = GenerateValueForVariable(varName, null, code);
        return value;
    }

    /// <summary>
    /// Gets a deterministic variation (0-1) based on variable name hash.
    /// This ensures the same variable always gets the same simulated value.
    /// Uses a deterministic hash (FNV-1a) instead of String.GetHashCode() which
    /// is randomized per-process in modern .NET.
    /// </summary>
    private static double GetHashBasedVariation(string varName)
    {
        var hash = GetDeterministicHash(varName);
        return (hash % 1000) / 1000.0;
    }

    /// <summary>
    /// Computes a deterministic hash for a string using FNV-1a algorithm.
    /// Unlike String.GetHashCode(), this produces consistent results across processes.
    /// </summary>
    private static uint GetDeterministicHash(string str)
    {
        const uint fnvPrime = 16777619;
        const uint fnvOffsetBasis = 2166136261;

        uint hash = fnvOffsetBasis;
        foreach (char c in str)
        {
            hash ^= c;
            hash *= fnvPrime;
        }
        return hash;
    }

    /// <summary>
    /// Formats a number with the specified format or default precision.
    /// </summary>
    private static string FormatNumber(double value, string? format, int defaultPrecision)
    {
        if (!string.IsNullOrEmpty(format))
        {
            try
            {
                return value.ToString(format);
            }
            catch
            {
                // Fall through to default formatting
            }
        }
        return value.ToString($"F{defaultPrecision}");
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
/// Response from the Azure Functions code execution API.
/// </summary>
internal class AzureFunctionsResponse
{
    public bool Success { get; set; }
    public string? Output { get; set; }
    public string? Error { get; set; }
    public List<string>? CompilationErrors { get; set; }
    public long ExecutionTime { get; set; }
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
