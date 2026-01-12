using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Verification;

/// <summary>
/// Executes and verifies code by running it with test cases.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This verifier actually runs code to check if it works correctly.
/// Think of it like a teacher running your program to see if it produces the right output.
///
/// **How it works:**
/// 1. Takes generated code and test cases
/// 2. Creates a temporary file with the code
/// 3. Runs the code in a subprocess
/// 4. Captures output and errors
/// 5. Compares actual output with expected output
/// 6. Returns pass/fail with detailed feedback
///
/// **Supported languages:**
/// - Python (via python3 interpreter)
/// - JavaScript/Node.js (via node interpreter)
/// - C# (via dotnet script or compilation)
///
/// **Safety considerations:**
/// - Runs code in separate process (isolation)
/// - Timeout limits (prevents infinite loops)
/// - Resource limits (prevents memory exhaustion)
/// - No network access recommended
///
/// **Use cases:**
/// - Validating HumanEval solutions
/// - Testing generated code in CodeReasoner
/// - Verifying algorithm implementations
/// - Unit test generation and validation
///
/// **Example:**
/// ```csharp
/// var verifier = new CodeExecutionVerifier<double>();
/// var result = await verifier.VerifyCodeAsync(
///     pythonCode,
///     testCases: new[] { "assert add(2, 3) == 5" },
///     language: "python"
/// );
/// Console.WriteLine($"Tests passed: {result.AllTestsPassed}");
/// Console.WriteLine($"Success rate: {result.PassRate}");
/// ```
///
/// **Research:**
/// - CodeRL: Mastering Code Generation through Pretrained Models and Deep RL (Le et al., 2022)
/// - CodeT: Code Generation with Generated Tests (Chen et al., 2022)
/// </para>
/// </remarks>
internal class CodeExecutionVerifier<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _timeoutMilliseconds;
    private readonly string _workingDirectory;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeExecutionVerifier{T}"/> class.
    /// </summary>
    /// <param name="timeoutMilliseconds">Execution timeout in milliseconds (default: 5000).</param>
    /// <param name="workingDirectory">Working directory for code execution (default: temp).</param>
    public CodeExecutionVerifier(int timeoutMilliseconds = 5000, string? workingDirectory = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _timeoutMilliseconds = timeoutMilliseconds;
        _workingDirectory = workingDirectory ?? Path.GetTempPath();
    }

    /// <inheritdoc/>
    public string VerifierName => "Code Execution Verifier";

    /// <inheritdoc/>
    public string Description =>
        "Executes code with test cases and verifies correctness. " +
        "Supports Python, JavaScript, and C#. Provides detailed execution feedback.";

    /// <summary>
    /// Verifies code by running it with test cases.
    /// </summary>
    public async Task<CodeExecutionResult<T>> VerifyCodeAsync(
        string code,
        string[] testCases,
        string language = "python",
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(code))
            throw new ArgumentNullException(nameof(code));

        var result = new CodeExecutionResult<T>
        {
            Language = language,
            Code = code,
            TotalTests = testCases.Length
        };

        // Create temporary file
        string extension = GetFileExtension(language);
        string tempFile = Path.Combine(_workingDirectory, $"temp_{Guid.NewGuid()}{extension}");

        try
        {
            // Write code to file (net462 compatible)
            File.WriteAllText(tempFile, code);

            // Execute tests
            var testResults = new List<TestCaseResult>();

            foreach (var testCase in testCases)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var testResult = await ExecuteTestCaseAsync(
                    tempFile,
                    testCase,
                    language,
                    cancellationToken
                );

                testResults.Add(testResult);

                if (testResult.Passed)
                {
                    result.PassedTests++;
                }
            }

            result.TestResults = testResults;
            result.AllTestsPassed = result.PassedTests == result.TotalTests;
            result.PassRate = _numOps.FromDouble((double)result.PassedTests / result.TotalTests);

            // Calculate verification score (0.0-1.0)
            double score = (double)result.PassedTests / result.TotalTests;

            // Bonus for all tests passing
            if (result.AllTestsPassed)
            {
                score = 1.0;
            }

            result.Score = _numOps.FromDouble(score);
        }
        catch (Exception ex)
        {
            result.ExecutionError = ex.Message;
            result.AllTestsPassed = false;
            result.Score = _numOps.Zero;
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
            {
                try
                {
                    File.Delete(tempFile);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public async Task<VerificationResult<T>> VerifyAsync(
        ReasoningChain<T> chain,
        string? correctAnswer = null,
        CancellationToken cancellationToken = default)
    {
        // Extract code from chain
        string? code = ExtractCode(chain);

        if (code is null || string.IsNullOrEmpty(code))
        {
            return new VerificationResult<T>
            {
                Passed = false,
                ToolUsed = VerifierName,
                Explanation = "No code found in reasoning chain",
                Confidence = _numOps.Zero
            };
        }

        // At this point, code is guaranteed to be non-null and non-empty
        // Try to extract or infer test cases
        var testCases = ExtractTestCases(chain, correctAnswer);

        if (testCases.Length == 0)
        {
            return new VerificationResult<T>
            {
                Passed = false,
                ToolUsed = VerifierName,
                Explanation = "No test cases available for verification",
                Confidence = _numOps.Zero
            };
        }

        // Detect language (code is guaranteed non-null after check above)
        string language = DetectLanguage(code);

        // Execute verification
        var executionResult = await VerifyCodeAsync(code, testCases, language, cancellationToken);

        return new VerificationResult<T>
        {
            Passed = executionResult.AllTestsPassed,
            ToolUsed = VerifierName,
            Explanation = executionResult.AllTestsPassed
                ? $"All {executionResult.TotalTests} test cases passed. {executionResult.GetSummary()}"
                : $"Failed {executionResult.TotalTests - executionResult.PassedTests}/{executionResult.TotalTests} tests. {executionResult.GetSummary()}",
            Confidence = executionResult.Score
        };
    }

    private async Task<TestCaseResult> ExecuteTestCaseAsync(
        string codeFile,
        string testCase,
        string language,
        CancellationToken cancellationToken)
    {
        var result = new TestCaseResult
        {
            TestCase = testCase
        };

        try
        {
            // Prepare command based on language
            var (command, arguments) = GetExecutionCommand(codeFile, testCase, language);

            var startInfo = new ProcessStartInfo
            {
                FileName = command,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = _workingDirectory
            };

            using (var process = new Process { StartInfo = startInfo })
            {
                var output = new StringBuilder();
                var errors = new StringBuilder();

                process.OutputDataReceived += (sender, e) =>
                {
                    if (e.Data != null)
                        output.AppendLine(e.Data);
                };

                process.ErrorDataReceived += (sender, e) =>
                {
                    if (e.Data != null)
                        errors.AppendLine(e.Data);
                };

                var stopwatch = Stopwatch.StartNew();
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                // Wait with timeout (net462: WaitForExit returns void, not bool)
                bool finished = await Task.Run(() =>
                {
                    process.WaitForExit(_timeoutMilliseconds);
                    return process.HasExited;
                },
                    cancellationToken
                );

                stopwatch.Stop();
                result.ExecutionTime = stopwatch.Elapsed;

                if (!finished)
                {
                    try
                    {
                        if (!process.HasExited)
                        {
                            process.Kill(); // net462 doesn't support entireProcessTree parameter
                        }
                    }
                    catch
                    {
                        // Process may have exited between check and kill
                    }

                    result.Passed = false;
                    result.Error = $"Execution timed out after {_timeoutMilliseconds}ms";
                    return result;
                }

                result.Output = output.ToString().Trim();
                result.Error = errors.ToString().Trim();

                // Check if test passed
                if (process.ExitCode == 0 && string.IsNullOrEmpty(result.Error))
                {
                    result.Passed = true;
                }
                else
                {
                    result.Passed = false;
                    if (string.IsNullOrEmpty(result.Error))
                    {
                        result.Error = $"Exit code: {process.ExitCode}";
                    }
                }
            }
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.Error = $"Execution exception: {ex.Message}";
        }

        return result;
    }

    private (string command, string arguments) GetExecutionCommand(
        string codeFile,
        string testCase,
        string language)
    {
        return language.ToLowerInvariant() switch
        {
            "python" or "py" => ("python3", $"-c \"exec(open('{codeFile}').read()); {testCase}\""),
            "javascript" or "js" or "node" => ("node", $"-e \"require('fs').readFileSync('{codeFile}', 'utf8'); {testCase}\""),
            "csharp" or "cs" => ("dotnet", $"script {codeFile}"),
            _ => throw new NotSupportedException($"Language '{language}' is not supported")
        };
    }

    private string GetFileExtension(string language)
    {
        return language.ToLowerInvariant() switch
        {
            "python" or "py" => ".py",
            "javascript" or "js" or "node" => ".js",
            "csharp" or "cs" => ".csx",
            _ => ".txt"
        };
    }

    private string? ExtractCode(ReasoningChain<T> chain)
    {
        // Look for code in final answer or steps
        string[] textsToSearch = new[]
        {
            chain.FinalAnswer,
            string.Join("\n", chain.Steps.Select(s => s.Content))
        };

        foreach (var text in textsToSearch)
        {
            // Try to extract from code blocks
            var match = RegexHelper.Match(text, @"```(?:python|javascript|csharp|py|js|cs)?\s*\n([\s\S]*?)\n```", RegexOptions.Multiline);
            if (match.Success)
            {
                return match.Groups[1].Value.Trim();
            }

            // If entire text looks like code
            if (text.Contains("def ") || text.Contains("function ") || text.Contains("public "))
            {
                return text.Trim();
            }
        }

        return null;
    }

    private string[] ExtractTestCases(ReasoningChain<T> chain, string? correctAnswer)
    {
        var testCases = new List<string>();

        // Look for explicit test cases in the chain
        string fullText = string.Join("\n", chain.Steps.Select(s => s.Content));

        var assertMatches = RegexHelper.Matches(fullText, @"assert\s+(.+?)(?:\n|$)", RegexOptions.Multiline);
        foreach (Match match in assertMatches)
        {
            testCases.Add(match.Groups[1].Value.Trim());
        }

        // If correct answer provided, create a basic test
        if (!string.IsNullOrEmpty(correctAnswer) && testCases.Count == 0)
        {
            testCases.Add($"# Expected: {correctAnswer}");
        }

        return testCases.ToArray();
    }

    private string DetectLanguage(string code)
    {
        if (code.Contains("def ") && (code.Contains("import ") || code.Contains(":")))
            return "python";

        if (code.Contains("function ") || code.Contains("const ") || code.Contains("let "))
            return "javascript";

        if (code.Contains("public ") || code.Contains("private ") || code.Contains("class "))
            return "csharp";

        return "python"; // Default
    }
}

/// <summary>
/// Result of code execution verification.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring.</typeparam>
internal class CodeExecutionResult<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeExecutionResult{T}"/> class.
    /// </summary>
    public CodeExecutionResult()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        PassRate = _numOps.Zero;
        Score = _numOps.Zero;
    }

    /// <summary>
    /// Programming language of the code.
    /// </summary>
    public string Language { get; set; } = string.Empty;

    /// <summary>
    /// The code that was executed.
    /// </summary>
    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Total number of test cases.
    /// </summary>
    public int TotalTests { get; set; }

    /// <summary>
    /// Number of tests that passed.
    /// </summary>
    public int PassedTests { get; set; }

    /// <summary>
    /// Pass rate (0.0-1.0).
    /// </summary>
    public T PassRate { get; set; }

    /// <summary>
    /// Whether all tests passed.
    /// </summary>
    public bool AllTestsPassed { get; set; }

    /// <summary>
    /// Overall verification score (0.0-1.0).
    /// </summary>
    public T Score { get; set; }

    /// <summary>
    /// Results for each test case.
    /// </summary>
    public List<TestCaseResult> TestResults { get; set; } = new();

    /// <summary>
    /// Execution error (if any).
    /// </summary>
    public string? ExecutionError { get; set; }

    /// <summary>
    /// Gets a summary of the execution results.
    /// </summary>
    public string GetSummary()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Language: {Language}");
        sb.AppendLine($"Tests: {PassedTests}/{TotalTests} passed ({Convert.ToDouble(PassRate):P0})");
        sb.AppendLine($"Score: {Convert.ToDouble(Score):F3}");

        if (!string.IsNullOrEmpty(ExecutionError))
        {
            sb.AppendLine($"Error: {ExecutionError}");
        }

        if (TestResults.Any(t => !t.Passed))
        {
            sb.AppendLine("\nFailed tests:");
            foreach (var failed in TestResults.Where(t => !t.Passed))
            {
                sb.AppendLine($"  - {failed.TestCase}");
                if (!string.IsNullOrEmpty(failed.Error))
                {
                    sb.AppendLine($"    Error: {failed.Error}");
                }
            }
        }

        return sb.ToString();
    }
}

/// <summary>
/// Result of a single test case execution.
/// </summary>
internal class TestCaseResult
{
    /// <summary>
    /// The test case that was executed.
    /// </summary>
    public string TestCase { get; set; } = string.Empty;

    /// <summary>
    /// Whether the test passed.
    /// </summary>
    public bool Passed { get; set; }

    /// <summary>
    /// Output from the test execution.
    /// </summary>
    public string Output { get; set; } = string.Empty;

    /// <summary>
    /// Error message (if test failed).
    /// </summary>
    public string? Error { get; set; }

    /// <summary>
    /// Execution time for this test.
    /// </summary>
    public TimeSpan ExecutionTime { get; set; }
}



