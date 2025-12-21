#if !NET6_0_OR_GREATER
#pragma warning disable CS8600, CS8601, CS8602, CS8603, CS8604
#endif
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Newtonsoft.Json;
#if !NET6_0_OR_GREATER
using AiDotNet.TrainingMonitoring;
#endif

namespace AiDotNet.TrainingMonitoring.ExperimentTracking;

/// <summary>
/// Local file-based experiment tracker providing MLflow-compatible functionality.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> ExperimentTracker stores all your experiment data locally
/// in a structured format. It's like having your own MLflow server without needing
/// to set up any infrastructure.
///
/// Directory structure:
/// <code>
/// tracking_uri/
/// |-- experiments/
/// |   |-- experiment_name/
/// |   |   |-- metadata.json
/// |   |   |-- runs/
/// |   |   |   |-- run_id/
/// |   |   |   |   |-- info.json
/// |   |   |   |   |-- params.json
/// |   |   |   |   |-- metrics/
/// |   |   |   |   |   |-- metric_name.json
/// |   |   |   |   |-- artifacts/
/// </code>
///
/// Example usage:
/// <code>
/// // Create tracker
/// var tracker = new ExperimentTracker("./mlruns");
///
/// // Set experiment
/// tracker.SetExperiment("image-classification");
///
/// // Start a run
/// var run = tracker.StartRun("baseline-resnet50");
///
/// // Log parameters
/// tracker.LogParameters(new Dictionary&lt;string, string&gt;
/// {
///     ["learning_rate"] = "0.001",
///     ["batch_size"] = "32",
///     ["epochs"] = "100"
/// });
///
/// // During training
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     var (loss, accuracy) = TrainEpoch();
///     tracker.LogMetrics(new Dictionary&lt;string, double&gt;
///     {
///         ["loss"] = loss,
///         ["accuracy"] = accuracy
///     }, step: epoch);
/// }
///
/// // Log model
/// tracker.LogModel("./model.onnx", "best_model");
///
/// // End run
/// tracker.EndRun();
/// </code>
/// </remarks>
public class ExperimentTracker : IExperimentTracker
{
    private readonly ConcurrentDictionary<string, ExperimentInfo> _experiments = new();
    private readonly ConcurrentDictionary<string, RunInfo> _runs = new();
    private readonly ConcurrentDictionary<string, ConcurrentDictionary<string, List<MetricValue>>> _metricHistory = new();
    private readonly object _fileLock = new();
    private string? _activeExperiment;
    private string? _activeRunId;
    private long _currentStep;
    private bool _disposed;

    /// <inheritdoc />
    public string TrackingUri { get; }

    /// <inheritdoc />
    public string? ActiveExperiment => _activeExperiment;

    /// <inheritdoc />
    public string? ActiveRunId => _activeRunId;

    /// <summary>
    /// Gets or sets the path to the git executable.
    /// </summary>
    /// <remarks>
    /// <b>Security Note:</b> For enhanced security, specify the full path to git
    /// rather than relying on PATH resolution. Common locations:
    /// - Windows: <c>C:\Program Files\Git\bin\git.exe</c>
    /// - Linux/macOS: <c>/usr/bin/git</c>
    ///
    /// If left as default ("git"), the executable is resolved via the system PATH.
    /// Ensure your PATH only contains trusted directories if using the default.
    /// </remarks>
    public static string GitPath { get; set; } = GetDefaultGitPath();

    private string ExperimentsPath => Path.Combine(TrackingUri, "experiments");

    /// <summary>
    /// Creates a new experiment tracker.
    /// </summary>
    /// <param name="trackingUri">Base directory for tracking data.</param>
    public ExperimentTracker(string? trackingUri = null)
    {
        TrackingUri = trackingUri ?? Path.Combine(Environment.CurrentDirectory, "mlruns");
        Directory.CreateDirectory(ExperimentsPath);
        LoadExistingExperiments();
    }

    /// <inheritdoc />
    public ExperimentInfo CreateExperiment(string name, string? description = null, Dictionary<string, string>? tags = null)
    {
        if (_experiments.TryGetValue(name, out var existing))
        {
            return existing;
        }

        var experimentPath = Path.Combine(ExperimentsPath, SanitizeName(name));
        Directory.CreateDirectory(experimentPath);
        Directory.CreateDirectory(Path.Combine(experimentPath, "runs"));

        var experiment = new ExperimentInfo
        {
            ExperimentId = Guid.NewGuid().ToString("N"),
            Name = name,
            Description = description,
            ArtifactLocation = Path.Combine(experimentPath, "artifacts"),
            Tags = tags ?? new Dictionary<string, string>()
        };

        _experiments[name] = experiment;
        SaveExperimentMetadata(experiment);

        return experiment;
    }

    /// <inheritdoc />
    public ExperimentInfo? GetExperiment(string name)
    {
        _experiments.TryGetValue(name, out var experiment);
        return experiment;
    }

    /// <inheritdoc />
    public List<ExperimentInfo> ListExperiments()
    {
        return _experiments.Values.Where(e => !e.IsDeleted).ToList();
    }

    /// <inheritdoc />
    public void SetExperiment(string name)
    {
        if (!_experiments.ContainsKey(name))
        {
            CreateExperiment(name);
        }
        _activeExperiment = name;
    }

    /// <inheritdoc />
    public RunInfo StartRun(string? runName = null, Dictionary<string, string>? tags = null, string? description = null)
    {
        if (string.IsNullOrEmpty(_activeExperiment))
        {
            SetExperiment("Default");
        }

        var runId = GenerateRunId();
        var experiment = _experiments[_activeExperiment!];

        var run = new RunInfo
        {
            RunId = runId,
            RunName = runName ?? $"run_{DateTime.Now:yyyyMMdd_HHmmss}",
            ExperimentId = experiment.ExperimentId,
            ExperimentName = experiment.Name,
            Description = description,
            User = Environment.UserName,
            Tags = tags ?? new Dictionary<string, string>(),
            Source = GetSourceInfo()
        };

        _runs[runId] = run;
        _metricHistory[runId] = new ConcurrentDictionary<string, List<MetricValue>>();
        _activeRunId = runId;
        _currentStep = 0;

        // Update experiment
        experiment.RunCount++;
        experiment.LastUpdatedAt = DateTime.UtcNow;

        SaveRunInfo(run);
        SaveExperimentMetadata(experiment);

        return run;
    }

    /// <inheritdoc />
    public void EndRun(RunStatus status = RunStatus.Completed)
    {
        var runId = _activeRunId;
        if (string.IsNullOrEmpty(runId))
            return;

        if (_runs.TryGetValue(runId, out var run))
        {
            run.Status = status;
            run.EndTime = DateTime.UtcNow;
            SaveRunInfo(run);
        }

        _activeRunId = null;
        _currentStep = 0;
    }

    /// <inheritdoc />
    public RunInfo? GetRun(string runId)
    {
        _runs.TryGetValue(runId, out var run);
        return run;
    }

    /// <inheritdoc />
    public List<RunInfo> ListRuns(string? experimentName = null, string? filter = null, string? orderBy = null, int maxResults = 100)
    {
        var query = _runs.Values.Where(r => !r.IsDeleted).AsEnumerable();

        if (!string.IsNullOrEmpty(experimentName))
        {
            query = query.Where(r => r.ExperimentName == experimentName);
        }

        // Apply simple filter (basic support for common patterns)
        var filterValue = filter;
        if (!string.IsNullOrEmpty(filterValue))
        {
            query = ApplyFilter(query, filterValue);
        }

        // Apply ordering
        var orderByValue = orderBy;
        if (!string.IsNullOrEmpty(orderByValue))
        {
            query = ApplyOrdering(query, orderByValue);
        }
        else
        {
            query = query.OrderByDescending(r => r.StartTime);
        }

        return query.Take(maxResults).ToList();
    }

    /// <inheritdoc />
    public void LogParameter(string key, string value)
    {
        var runId = _activeRunId;
        if (string.IsNullOrEmpty(runId))
            throw new InvalidOperationException("No active run. Call StartRun() first.");

        var run = _runs[runId];
        run.Parameters[key] = value;
        SaveRunParameters(run);
    }

    /// <inheritdoc />
    public void LogParameters(Dictionary<string, string> parameters)
    {
        var runId = _activeRunId;
        if (string.IsNullOrEmpty(runId))
            throw new InvalidOperationException("No active run. Call StartRun() first.");

        var run = _runs[runId];
        foreach (var kvp in parameters)
        {
            run.Parameters[kvp.Key] = kvp.Value;
        }
        SaveRunParameters(run);
    }

    /// <inheritdoc />
    public void LogMetric(string key, double value, long? step = null)
    {
        var runId = _activeRunId;
        if (string.IsNullOrEmpty(runId))
            throw new InvalidOperationException("No active run. Call StartRun() first.");

        var actualStep = step ?? _currentStep++;
        var run = _runs[runId];
        var metricValue = new MetricValue
        {
            Key = key,
            Value = value,
            Step = actualStep,
            Timestamp = DateTime.UtcNow
        };

        // Update latest value
        run.Metrics[key] = value;

        // Add to history
        var history = _metricHistory[runId];
        var metricList = history.GetOrAdd(key, _ => new List<MetricValue>());
        lock (metricList)
        {
            metricList.Add(metricValue);
        }

        SaveMetric(runId, key, metricList);
    }

    /// <inheritdoc />
    public void LogMetrics(Dictionary<string, double> metrics, long? step = null)
    {
        var actualStep = step ?? _currentStep++;

        foreach (var kvp in metrics)
        {
            LogMetric(kvp.Key, kvp.Value, actualStep);
        }
    }

    /// <inheritdoc />
    public void SetTag(string key, string value)
    {
        var runId = _activeRunId;
        if (string.IsNullOrEmpty(runId))
            throw new InvalidOperationException("No active run. Call StartRun() first.");

        var run = _runs[runId];
        run.Tags[key] = value;
        SaveRunInfo(run);
    }

    /// <inheritdoc />
    public void SetTags(Dictionary<string, string> tags)
    {
        var runId = _activeRunId;
        if (string.IsNullOrEmpty(runId))
            throw new InvalidOperationException("No active run. Call StartRun() first.");

        var run = _runs[runId];
        foreach (var kvp in tags)
        {
            run.Tags[kvp.Key] = kvp.Value;
        }
        SaveRunInfo(run);
    }

    /// <inheritdoc />
    public void LogArtifact(string localPath, string? artifactPath = null)
    {
        var runId = _activeRunId;
        if (string.IsNullOrEmpty(runId))
            throw new InvalidOperationException("No active run. Call StartRun() first.");

        var run = _runs[runId];
        var artifactDir = GetArtifactDir(run.RunId);
        var targetDir = string.IsNullOrEmpty(artifactPath) ? artifactDir : Path.Combine(artifactDir, artifactPath);
        Directory.CreateDirectory(targetDir);

        var fileName = Path.GetFileName(localPath);
        var targetPath = Path.Combine(targetDir, fileName);
        File.Copy(localPath, targetPath, overwrite: true);

#if NET6_0_OR_GREATER
        var relativePath = Path.GetRelativePath(artifactDir, targetPath);
#else
        var relativePath = FrameworkPolyfills.GetRelativePath(artifactDir, targetPath);
#endif
        if (!run.Artifacts.Contains(relativePath))
        {
            run.Artifacts.Add(relativePath);
            SaveRunInfo(run);
        }
    }

    /// <inheritdoc />
    public void LogArtifacts(string localDir, string? artifactPath = null)
    {
        if (!Directory.Exists(localDir))
            throw new DirectoryNotFoundException($"Directory not found: {localDir}");

        foreach (var file in Directory.GetFiles(localDir, "*", SearchOption.AllDirectories))
        {
#if NET6_0_OR_GREATER
            var relativePath = Path.GetRelativePath(localDir, file);
#else
            var relativePath = FrameworkPolyfills.GetRelativePath(localDir, file);
#endif
            var targetSubPath = string.IsNullOrEmpty(artifactPath)
                ? Path.GetDirectoryName(relativePath)
                : Path.Combine(artifactPath, Path.GetDirectoryName(relativePath) ?? string.Empty);

            LogArtifact(file, targetSubPath);
        }
    }

    /// <inheritdoc />
    public void LogModel(string modelPath, string modelName, Dictionary<string, object>? metadata = null)
    {
        var runId = _activeRunId;
        if (string.IsNullOrEmpty(runId))
            throw new InvalidOperationException("No active run. Call StartRun() first.");

        var artifactDir = GetArtifactDir(runId);
        var modelDir = Path.Combine(artifactDir, "models", modelName);
        Directory.CreateDirectory(modelDir);

        // Copy model files
        if (File.Exists(modelPath))
        {
            var targetPath = Path.Combine(modelDir, Path.GetFileName(modelPath));
            File.Copy(modelPath, targetPath, overwrite: true);
        }
        else if (Directory.Exists(modelPath))
        {
            foreach (var file in Directory.GetFiles(modelPath, "*", SearchOption.AllDirectories))
            {
#if NET6_0_OR_GREATER
                var relativePath = Path.GetRelativePath(modelPath, file);
#else
                var relativePath = FrameworkPolyfills.GetRelativePath(modelPath, file);
#endif
                var targetPath = Path.Combine(modelDir, relativePath);
                var targetDir = Path.GetDirectoryName(targetPath);
                if (!string.IsNullOrEmpty(targetDir))
                {
                    Directory.CreateDirectory(targetDir);
                }
                File.Copy(file, targetPath, overwrite: true);
            }
        }

        // Save model metadata
        var modelMeta = new Dictionary<string, object>
        {
            ["model_name"] = modelName,
            ["logged_at"] = DateTime.UtcNow,
            ["run_id"] = runId,
            ["source_path"] = modelPath
        };

        if (metadata is not null)
        {
            foreach (var kvp in metadata)
            {
                modelMeta[kvp.Key] = kvp.Value;
            }
        }

        var metaPath = Path.Combine(modelDir, "model_metadata.json");
        File.WriteAllText(metaPath, JsonConvert.SerializeObject(modelMeta, Formatting.Indented));

        var run = _runs[runId];
        run.Artifacts.Add($"models/{modelName}");
        SaveRunInfo(run);
    }

    /// <inheritdoc />
    public List<MetricValue> GetMetricHistory(string runId, string metricKey)
    {
        if (_metricHistory.TryGetValue(runId, out var metrics) &&
            metrics.TryGetValue(metricKey, out var history))
        {
            return history.ToList();
        }

        // Try loading from disk
        var run = GetRun(runId);
        if (run is null) return new List<MetricValue>();

        var metricPath = GetMetricPath(runId, metricKey);
        if (File.Exists(metricPath))
        {
            var json = File.ReadAllText(metricPath);
            return JsonConvert.DeserializeObject<List<MetricValue>>(json) ?? new List<MetricValue>();
        }

        return new List<MetricValue>();
    }

    /// <inheritdoc />
    public RunComparison CompareRuns(params string[] runIds)
    {
        var comparison = new RunComparison();

        foreach (var runId in runIds)
        {
            var run = GetRun(runId);
            if (run is not null)
            {
                comparison.Runs.Add(run);
            }
        }

        // Compare parameters
        var allParams = comparison.Runs.SelectMany(r => r.Parameters.Keys).Distinct();
        foreach (var paramKey in allParams)
        {
            comparison.ParameterComparison[paramKey] = comparison.Runs
                .Where(r => r.Parameters.ContainsKey(paramKey))
                .ToDictionary(r => r.RunId, r => r.Parameters[paramKey]);
        }

        // Compare metrics
        var allMetrics = comparison.Runs.SelectMany(r => r.Metrics.Keys).Distinct();
        foreach (var metricKey in allMetrics)
        {
            comparison.MetricComparison[metricKey] = comparison.Runs
                .Where(r => r.Metrics.ContainsKey(metricKey))
                .ToDictionary(r => r.RunId, r => r.Metrics[metricKey]);
        }

        return comparison;
    }

    /// <inheritdoc />
    public List<RunInfo> SearchRuns(
        IEnumerable<string>? experimentNames = null,
        string? filter = null,
        string? orderBy = null,
        int maxResults = 100)
    {
        var query = _runs.Values.Where(r => !r.IsDeleted).AsEnumerable();

        if (experimentNames is not null && experimentNames.Any())
        {
            var names = experimentNames.ToHashSet();
            query = query.Where(r => names.Contains(r.ExperimentName));
        }

        var filterValue = filter;
        if (!string.IsNullOrEmpty(filterValue))
        {
            query = ApplyFilter(query, filterValue);
        }

        var orderByValue = orderBy;
        if (!string.IsNullOrEmpty(orderByValue))
        {
            query = ApplyOrdering(query, orderByValue);
        }
        else
        {
            query = query.OrderByDescending(r => r.StartTime);
        }

        return query.Take(maxResults).ToList();
    }

    /// <inheritdoc />
    public void DeleteRun(string runId)
    {
        if (_runs.TryGetValue(runId, out var run))
        {
            run.IsDeleted = true;
            SaveRunInfo(run);
        }
    }

    /// <inheritdoc />
    public void RestoreRun(string runId)
    {
        if (_runs.TryGetValue(runId, out var run))
        {
            run.IsDeleted = false;
            SaveRunInfo(run);
        }
    }

    /// <inheritdoc />
    public void DeleteExperiment(string experimentName)
    {
        if (_experiments.TryGetValue(experimentName, out var experiment))
        {
            experiment.IsDeleted = true;
            SaveExperimentMetadata(experiment);

            // Mark all runs as deleted
            foreach (var run in _runs.Values.Where(r => r.ExperimentName == experimentName))
            {
                run.IsDeleted = true;
                SaveRunInfo(run);
            }
        }
    }

    private void LoadExistingExperiments()
    {
        if (!Directory.Exists(ExperimentsPath))
            return;

        foreach (var expDir in Directory.GetDirectories(ExperimentsPath))
        {
            var metaPath = Path.Combine(expDir, "metadata.json");
            if (File.Exists(metaPath))
            {
                try
                {
                    var json = File.ReadAllText(metaPath);
                    var experiment = JsonConvert.DeserializeObject<ExperimentInfo>(json);
                    if (experiment is not null)
                    {
                        _experiments[experiment.Name] = experiment;
                        LoadExperimentRuns(expDir, experiment);
                    }
                }
                catch (Exception)
                {
                    // Skip invalid experiments
                }
            }
        }
    }

    private void LoadExperimentRuns(string experimentDir, ExperimentInfo experiment)
    {
        var runsDir = Path.Combine(experimentDir, "runs");
        if (!Directory.Exists(runsDir))
            return;

        foreach (var runDir in Directory.GetDirectories(runsDir))
        {
            var infoPath = Path.Combine(runDir, "info.json");
            if (File.Exists(infoPath))
            {
                try
                {
                    var json = File.ReadAllText(infoPath);
                    var run = JsonConvert.DeserializeObject<RunInfo>(json);
                    if (run is not null)
                    {
                        _runs[run.RunId] = run;
                        _metricHistory[run.RunId] = new ConcurrentDictionary<string, List<MetricValue>>();

                        // Load metrics
                        var metricsDir = Path.Combine(runDir, "metrics");
                        if (Directory.Exists(metricsDir))
                        {
                            foreach (var metricFile in Directory.GetFiles(metricsDir, "*.json"))
                            {
                                var metricKey = Path.GetFileNameWithoutExtension(metricFile);
                                var metricJson = File.ReadAllText(metricFile);
                                var history = JsonConvert.DeserializeObject<List<MetricValue>>(metricJson);
                                if (history is not null)
                                {
                                    _metricHistory[run.RunId][metricKey] = history;
                                }
                            }
                        }
                    }
                }
                catch (Exception)
                {
                    // Skip invalid runs
                }
            }
        }
    }

    private void SaveExperimentMetadata(ExperimentInfo experiment)
    {
        var expDir = Path.Combine(ExperimentsPath, SanitizeName(experiment.Name));
        Directory.CreateDirectory(expDir);
        var metaPath = Path.Combine(expDir, "metadata.json");

        lock (_fileLock)
        {
            File.WriteAllText(metaPath, JsonConvert.SerializeObject(experiment, Formatting.Indented));
        }
    }

    private void SaveRunInfo(RunInfo run)
    {
        var runDir = GetRunDir(run);
        Directory.CreateDirectory(runDir);
        var infoPath = Path.Combine(runDir, "info.json");

        lock (_fileLock)
        {
            File.WriteAllText(infoPath, JsonConvert.SerializeObject(run, Formatting.Indented));
        }
    }

    private void SaveRunParameters(RunInfo run)
    {
        var runDir = GetRunDir(run);
        var paramsPath = Path.Combine(runDir, "params.json");

        lock (_fileLock)
        {
            File.WriteAllText(paramsPath, JsonConvert.SerializeObject(run.Parameters, Formatting.Indented));
        }
    }

    private void SaveMetric(string runId, string metricKey, List<MetricValue> history)
    {
        var run = _runs[runId];
        var metricsDir = Path.Combine(GetRunDir(run), "metrics");
        Directory.CreateDirectory(metricsDir);
        var metricPath = Path.Combine(metricsDir, $"{SanitizeName(metricKey)}.json");

        lock (_fileLock)
        {
            File.WriteAllText(metricPath, JsonConvert.SerializeObject(history, Formatting.Indented));
        }
    }

    private string GetRunDir(RunInfo run)
    {
        var expDir = Path.Combine(ExperimentsPath, SanitizeName(run.ExperimentName));
        return Path.Combine(expDir, "runs", run.RunId);
    }

    private string GetArtifactDir(string runId)
    {
        var run = _runs[runId];
        return Path.Combine(GetRunDir(run), "artifacts");
    }

    private string GetMetricPath(string runId, string metricKey)
    {
        var run = _runs[runId];
        var metricsDir = Path.Combine(GetRunDir(run), "metrics");
        return Path.Combine(metricsDir, $"{SanitizeName(metricKey)}.json");
    }

    private static string SanitizeName(string name)
    {
        foreach (var c in Path.GetInvalidFileNameChars())
        {
            name = name.Replace(c, '_');
        }
        return name;
    }

    private static string GenerateRunId()
    {
        return $"{DateTime.UtcNow:yyyyMMddHHmmss}_{Guid.NewGuid():N}"[..20];
    }

    private static SourceInfo GetSourceInfo()
    {
        var source = new SourceInfo
        {
            SourceType = "LOCAL"
        };

        try
        {
            // Try to get git info
            var startInfo = new ProcessStartInfo
            {
                FileName = GitPath,
                Arguments = "rev-parse HEAD",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process is not null)
            {
                source.GitCommit = process.StandardOutput.ReadToEnd().Trim();
                process.WaitForExit(1000);
            }

            startInfo.Arguments = "rev-parse --abbrev-ref HEAD";
            using var branchProcess = Process.Start(startInfo);
            if (branchProcess is not null)
            {
                source.GitBranch = branchProcess.StandardOutput.ReadToEnd().Trim();
                branchProcess.WaitForExit(1000);
            }
        }
        catch
        {
            // Git not available
        }

        return source;
    }

    private static IEnumerable<RunInfo> ApplyFilter(IEnumerable<RunInfo> query, string filter)
    {
        // Simple filter parsing for common patterns
        // Format: "metrics.accuracy > 0.9" or "params.learning_rate = '0.001'" or "tags.env = 'prod'"

#if NET6_0_OR_GREATER
        var parts = filter.Split(' ', 3, StringSplitOptions.RemoveEmptyEntries);
#else
        var parts = filter.SplitWithOptions(' ', 3, StringSplitOptions.RemoveEmptyEntries);
#endif
        if (parts.Length < 3) return query;

        var field = parts[0];
        var op = parts[1];
        var valueStr = parts[2].Trim('\'', '"');

        if (field.StartsWith("metrics."))
        {
            var metricKey = field.Substring(8);
            if (double.TryParse(valueStr, out var value))
            {
                query = op switch
                {
                    ">" => query.Where(r => r.Metrics.TryGetValue(metricKey, out var v) && v > value),
                    ">=" => query.Where(r => r.Metrics.TryGetValue(metricKey, out var v) && v >= value),
                    "<" => query.Where(r => r.Metrics.TryGetValue(metricKey, out var v) && v < value),
                    "<=" => query.Where(r => r.Metrics.TryGetValue(metricKey, out var v) && v <= value),
                    "=" or "==" => query.Where(r => r.Metrics.TryGetValue(metricKey, out var v) && Math.Abs(v - value) < 0.0001),
                    _ => query
                };
            }
        }
        else if (field.StartsWith("params."))
        {
            var paramKey = field.Substring(7);
            query = op switch
            {
                "=" or "==" => query.Where(r => r.Parameters.TryGetValue(paramKey, out var v) && v == valueStr),
                "!=" => query.Where(r => !r.Parameters.TryGetValue(paramKey, out var v) || v != valueStr),
                _ => query
            };
        }
        else if (field.StartsWith("tags."))
        {
            var tagKey = field.Substring(5);
            query = op switch
            {
                "=" or "==" => query.Where(r => r.Tags.TryGetValue(tagKey, out var v) && v == valueStr),
                "!=" => query.Where(r => !r.Tags.TryGetValue(tagKey, out var v) || v != valueStr),
                _ => query
            };
        }
        else if (field == "status")
        {
            if (Enum.TryParse<RunStatus>(valueStr, true, out var status))
            {
                query = op switch
                {
                    "=" or "==" => query.Where(r => r.Status == status),
                    "!=" => query.Where(r => r.Status != status),
                    _ => query
                };
            }
        }

        return query;
    }

    private static IEnumerable<RunInfo> ApplyOrdering(IEnumerable<RunInfo> query, string orderBy)
    {
        // Format: "metrics.loss ASC" or "start_time DESC"
#if NET6_0_OR_GREATER
        var parts = orderBy.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
#else
        var parts = orderBy.SplitWithOptions(' ', 2, StringSplitOptions.RemoveEmptyEntries);
#endif
        var field = parts[0];
        var descending = parts.Length > 1 && parts[1].Equals("DESC", StringComparison.OrdinalIgnoreCase);

        if (field.StartsWith("metrics."))
        {
            var metricKey = field.Substring(8);
            return descending
                ? query.OrderByDescending(r => r.Metrics.GetValueOrDefault(metricKey, 0))
                : query.OrderBy(r => r.Metrics.GetValueOrDefault(metricKey, 0));
        }

        return field switch
        {
            "start_time" => descending ? query.OrderByDescending(r => r.StartTime) : query.OrderBy(r => r.StartTime),
            "end_time" => descending ? query.OrderByDescending(r => r.EndTime) : query.OrderBy(r => r.EndTime),
            "run_name" => descending ? query.OrderByDescending(r => r.RunName) : query.OrderBy(r => r.RunName),
            _ => query
        };
    }

    /// <summary>
    /// Disposes the tracker.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // End any active run
        if (!string.IsNullOrEmpty(_activeRunId))
        {
            EndRun(RunStatus.Completed);
        }
    }

    /// <summary>
    /// Gets the default path for git based on the current platform.
    /// </summary>
    private static string GetDefaultGitPath()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            // Try common Git installation paths on Windows
            var windowsPaths = new[]
            {
                @"C:\Program Files\Git\bin\git.exe",
                @"C:\Program Files (x86)\Git\bin\git.exe"
            };
            foreach (var path in windowsPaths)
            {
                if (File.Exists(path))
                    return path;
            }
        }
        else
        {
            // Try standard Unix paths
            var unixPaths = new[] { "/usr/bin/git", "/usr/local/bin/git" };
            foreach (var path in unixPaths)
            {
                if (File.Exists(path))
                    return path;
            }
        }

        // Fall back to PATH resolution
        return "git";
    }
}
