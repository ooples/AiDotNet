using AiDotNet.Interfaces;
using AiDotNet.Models;
using System.Text.Json;

namespace AiDotNet.ExperimentTracking;

/// <summary>
/// Implementation of experiment tracking system for managing ML experiments and runs.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This is a complete implementation of MLflow-like experiment tracking.
/// It helps you organize and track all your machine learning experiments in one place.
///
/// Features include:
/// - Creating and managing experiments
/// - Starting and tracking training runs
/// - Logging parameters, metrics, and artifacts
/// - Searching and comparing runs
/// - Persistent storage of all experiment data
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ExperimentTracker<T> : IExperimentTracker<T>
{
    private readonly string _storageDirectory;
    private readonly Dictionary<string, Experiment> _experiments;
    private readonly Dictionary<string, ExperimentRun<T>> _runs;
    private readonly object _lock = new();

    /// <summary>
    /// Initializes a new instance of the ExperimentTracker class.
    /// </summary>
    /// <param name="storageDirectory">Directory to store experiment data. Defaults to "./mlruns".</param>
    public ExperimentTracker(string? storageDirectory = null)
    {
        _storageDirectory = storageDirectory ?? Path.Combine(Directory.GetCurrentDirectory(), "mlruns");
        _experiments = new Dictionary<string, Experiment>();
        _runs = new Dictionary<string, ExperimentRun<T>>();

        // Create storage directory if it doesn't exist
        if (!Directory.Exists(_storageDirectory))
        {
            Directory.CreateDirectory(_storageDirectory);
        }

        // Load existing experiments and runs
        LoadExistingData();
    }

    /// <summary>
    /// Creates a new experiment to organize related training runs.
    /// </summary>
    public string CreateExperiment(string name, string? description = null, Dictionary<string, string>? tags = null)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Experiment name cannot be null or empty.", nameof(name));

        lock (_lock)
        {
            // Check if experiment with this name already exists
            var existing = _experiments.Values.FirstOrDefault(e => e.Name == name);
            if (existing != null)
            {
                return existing.ExperimentId;
            }

            var experiment = new Experiment(name, description, tags);
            _experiments[experiment.ExperimentId] = experiment;

            // Persist experiment
            SaveExperiment(experiment);

            return experiment.ExperimentId;
        }
    }

    /// <summary>
    /// Starts a new training run within an experiment.
    /// </summary>
    public IExperimentRun<T> StartRun(string experimentId, string? runName = null, Dictionary<string, string>? tags = null)
    {
        if (string.IsNullOrWhiteSpace(experimentId))
            throw new ArgumentException("Experiment ID cannot be null or empty.", nameof(experimentId));

        lock (_lock)
        {
            if (!_experiments.ContainsKey(experimentId))
                throw new ArgumentException($"Experiment with ID '{experimentId}' not found.", nameof(experimentId));

            var run = new ExperimentRun<T>(experimentId, runName, tags);
            _runs[run.RunId] = run;

            // Update experiment timestamp
            _experiments[experimentId].Touch();

            // Persist run
            SaveRun(run);

            return run;
        }
    }

    /// <summary>
    /// Gets an existing experiment by its ID.
    /// </summary>
    public IExperiment GetExperiment(string experimentId)
    {
        lock (_lock)
        {
            if (!_experiments.TryGetValue(experimentId, out var experiment))
                throw new ArgumentException($"Experiment with ID '{experimentId}' not found.", nameof(experimentId));

            return experiment;
        }
    }

    /// <summary>
    /// Gets an existing run by its ID.
    /// </summary>
    public IExperimentRun<T> GetRun(string runId)
    {
        lock (_lock)
        {
            if (!_runs.TryGetValue(runId, out var run))
                throw new ArgumentException($"Run with ID '{runId}' not found.", nameof(runId));

            return run;
        }
    }

    /// <summary>
    /// Lists all experiments, optionally filtered by criteria.
    /// </summary>
    public IEnumerable<IExperiment> ListExperiments(string? filter = null)
    {
        lock (_lock)
        {
            IEnumerable<Experiment> experiments = _experiments.Values;

            if (!string.IsNullOrWhiteSpace(filter))
            {
                // Simple filter implementation - can be extended
                experiments = experiments.Where(e =>
                    e.Name.Contains(filter, StringComparison.OrdinalIgnoreCase) ||
                    (e.Description?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false));
            }

            return experiments.OrderByDescending(e => e.LastUpdatedAt).ToList();
        }
    }

    /// <summary>
    /// Lists all runs in an experiment, optionally filtered by criteria.
    /// </summary>
    public IEnumerable<IExperimentRun<T>> ListRuns(string experimentId, string? filter = null)
    {
        lock (_lock)
        {
            if (!_experiments.ContainsKey(experimentId))
                throw new ArgumentException($"Experiment with ID '{experimentId}' not found.", nameof(experimentId));

            IEnumerable<ExperimentRun<T>> runs = _runs.Values
                .Where(r => r.ExperimentId == experimentId);

            if (!string.IsNullOrWhiteSpace(filter))
            {
                runs = runs.Where(r =>
                    (r.RunName?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false) ||
                    r.Status.Contains(filter, StringComparison.OrdinalIgnoreCase));
            }

            return runs.OrderByDescending(r => r.StartTime).ToList();
        }
    }

    /// <summary>
    /// Deletes an experiment and all its associated runs.
    /// </summary>
    public void DeleteExperiment(string experimentId)
    {
        lock (_lock)
        {
            if (!_experiments.ContainsKey(experimentId))
                throw new ArgumentException($"Experiment with ID '{experimentId}' not found.", nameof(experimentId));

            // Delete all runs in this experiment
            var runsToDelete = _runs.Values
                .Where(r => r.ExperimentId == experimentId)
                .Select(r => r.RunId)
                .ToList();

            foreach (var runId in runsToDelete)
            {
                DeleteRun(runId);
            }

            // Delete experiment
            _experiments.Remove(experimentId);

            // Delete experiment directory
            var experimentDir = GetExperimentDirectory(experimentId);
            if (Directory.Exists(experimentDir))
            {
                Directory.Delete(experimentDir, true);
            }
        }
    }

    /// <summary>
    /// Deletes a specific run.
    /// </summary>
    public void DeleteRun(string runId)
    {
        lock (_lock)
        {
            if (!_runs.ContainsKey(runId))
                throw new ArgumentException($"Run with ID '{runId}' not found.", nameof(runId));

            _runs.Remove(runId);

            // Delete run directory
            var runDir = GetRunDirectory(runId);
            if (Directory.Exists(runDir))
            {
                Directory.Delete(runDir, true);
            }
        }
    }

    /// <summary>
    /// Searches for runs across all experiments based on criteria.
    /// </summary>
    public IEnumerable<IExperimentRun<T>> SearchRuns(string filter, int maxResults = 100)
    {
        lock (_lock)
        {
            var matchingRuns = _runs.Values.AsEnumerable();

            if (!string.IsNullOrWhiteSpace(filter))
            {
                matchingRuns = matchingRuns.Where(r =>
                    (r.RunName?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false) ||
                    r.Status.Contains(filter, StringComparison.OrdinalIgnoreCase) ||
                    r.Tags.Any(t => t.Key.Contains(filter, StringComparison.OrdinalIgnoreCase) ||
                                   t.Value.Contains(filter, StringComparison.OrdinalIgnoreCase)));
            }

            return matchingRuns
                .OrderByDescending(r => r.StartTime)
                .Take(maxResults)
                .ToList();
        }
    }

    #region Private Helper Methods

    private void LoadExistingData()
    {
        if (!Directory.Exists(_storageDirectory))
            return;

        // Load experiments
        var experimentDirs = Directory.GetDirectories(_storageDirectory);
        foreach (var expDir in experimentDirs)
        {
            var metaFile = Path.Combine(expDir, "meta.json");
            if (File.Exists(metaFile))
            {
                try
                {
                    var json = File.ReadAllText(metaFile);
                    var experiment = JsonSerializer.Deserialize<Experiment>(json);
                    if (experiment != null)
                    {
                        _experiments[experiment.ExperimentId] = experiment;

                        // Load runs for this experiment
                        LoadRunsForExperiment(experiment.ExperimentId);
                    }
                }
                catch
                {
                    // Skip corrupted experiment data
                }
            }
        }
    }

    private void LoadRunsForExperiment(string experimentId)
    {
        var experimentDir = GetExperimentDirectory(experimentId);
        var runDirs = Directory.GetDirectories(experimentDir);

        foreach (var runDir in runDirs)
        {
            var metaFile = Path.Combine(runDir, "meta.json");
            if (File.Exists(metaFile))
            {
                try
                {
                    var json = File.ReadAllText(metaFile);
                    var run = JsonSerializer.Deserialize<ExperimentRun<T>>(json);
                    if (run != null)
                    {
                        _runs[run.RunId] = run;
                    }
                }
                catch
                {
                    // Skip corrupted run data
                }
            }
        }
    }

    private void SaveExperiment(Experiment experiment)
    {
        var experimentDir = GetExperimentDirectory(experiment.ExperimentId);
        Directory.CreateDirectory(experimentDir);

        var metaFile = Path.Combine(experimentDir, "meta.json");
        var json = JsonSerializer.Serialize(experiment, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(metaFile, json);
    }

    private void SaveRun(ExperimentRun<T> run)
    {
        var runDir = GetRunDirectory(run.RunId);
        Directory.CreateDirectory(runDir);

        var metaFile = Path.Combine(runDir, "meta.json");
        var json = JsonSerializer.Serialize(run, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(metaFile, json);
    }

    private string GetExperimentDirectory(string experimentId)
    {
        return Path.Combine(_storageDirectory, experimentId);
    }

    private string GetRunDirectory(string runId)
    {
        var run = _runs[runId];
        var experimentDir = GetExperimentDirectory(run.ExperimentId);
        return Path.Combine(experimentDir, runId);
    }

    #endregion
}
