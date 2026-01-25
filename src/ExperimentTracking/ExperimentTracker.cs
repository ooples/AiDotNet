using AiDotNet.Interfaces;
using AiDotNet.Models;
using Newtonsoft.Json;

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
public class ExperimentTracker<T> : ExperimentTrackerBase<T>
{
    private readonly Dictionary<string, Experiment> _experiments;
    private readonly Dictionary<string, ExperimentRun<T>> _runs;

    /// <summary>
    /// Initializes a new instance of the ExperimentTracker class.
    /// </summary>
    /// <param name="storageDirectory">Directory to store experiment data. Defaults to "./mlruns".</param>
    public ExperimentTracker(string? storageDirectory = null) : base(storageDirectory)
    {
        _experiments = new Dictionary<string, Experiment>();
        _runs = new Dictionary<string, ExperimentRun<T>>();

        // Load existing experiments and runs
        LoadExistingData();
    }

    /// <summary>
    /// Creates a new experiment to organize related training runs.
    /// </summary>
    public override string CreateExperiment(string name, string? description = null, Dictionary<string, string>? tags = null)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Experiment name cannot be null or empty.", nameof(name));

        lock (SyncLock)
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
    public override IExperimentRun<T> StartRun(string experimentId, string? runName = null, Dictionary<string, string>? tags = null)
    {
        if (string.IsNullOrWhiteSpace(experimentId))
            throw new ArgumentException("Experiment ID cannot be null or empty.", nameof(experimentId));

        lock (SyncLock)
        {
            if (!_experiments.TryGetValue(experimentId, out var experiment))
                throw new ArgumentException($"Experiment with ID '{experimentId}' not found.", nameof(experimentId));

            var run = new ExperimentRun<T>(experimentId, runName, tags);
            _runs[run.RunId] = run;

            // Update experiment timestamp and persist both
            experiment.Touch();
            SaveExperiment(experiment);
            SaveRun(run);

            return run;
        }
    }

    /// <summary>
    /// Gets an existing experiment by its ID.
    /// </summary>
    public override IExperiment GetExperiment(string experimentId)
    {
        if (experimentId == null)
            throw new ArgumentNullException(nameof(experimentId));

        lock (SyncLock)
        {
            if (!_experiments.TryGetValue(experimentId, out var experiment))
                throw new ArgumentException($"Experiment with ID '{experimentId}' not found.", nameof(experimentId));

            return experiment;
        }
    }

    /// <summary>
    /// Gets an existing run by its ID.
    /// </summary>
    public override IExperimentRun<T> GetRun(string runId)
    {
        if (runId == null)
            throw new ArgumentNullException(nameof(runId));

        lock (SyncLock)
        {
            if (!_runs.TryGetValue(runId, out var run))
                throw new ArgumentException($"Run with ID '{runId}' not found.", nameof(runId));

            return run;
        }
    }

    /// <summary>
    /// Lists all experiments, optionally filtered by criteria.
    /// </summary>
    public override IEnumerable<IExperiment> ListExperiments(string? filter = null)
    {
        lock (SyncLock)
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
    public override IEnumerable<IExperimentRun<T>> ListRuns(string experimentId, string? filter = null)
    {
        if (experimentId == null)
            throw new ArgumentNullException(nameof(experimentId));

        lock (SyncLock)
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
    public override void DeleteExperiment(string experimentId)
    {
        if (experimentId == null)
            throw new ArgumentNullException(nameof(experimentId));

        lock (SyncLock)
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

            // Delete experiment directory - continue even if deletion fails
            // (the in-memory state is already updated)
            var experimentDir = GetExperimentDirectoryPath(experimentId);
            try
            {
                if (Directory.Exists(experimentDir))
                {
                    Directory.Delete(experimentDir, true);
                }
            }
            catch (IOException ex)
            {
                // Directory may be in use or have permission issues
                // The experiment is already removed from memory, so log and continue
                Console.WriteLine($"[ExperimentTracker] Failed to delete experiment directory '{experimentDir}': {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Deletes a specific run.
    /// </summary>
    public override void DeleteRun(string runId)
    {
        if (runId == null)
            throw new ArgumentNullException(nameof(runId));

        lock (SyncLock)
        {
            if (!_runs.ContainsKey(runId))
                throw new ArgumentException($"Run with ID '{runId}' not found.", nameof(runId));

            // Get run directory BEFORE removing from dictionary
            // (GetRunDirectory accesses _runs[runId])
            var runDir = GetRunDirectory(runId);

            // Remove from tracking
            _runs.Remove(runId);

            // Delete run directory - continue even if deletion fails
            // (the in-memory state is already updated)
            try
            {
                if (Directory.Exists(runDir))
                {
                    Directory.Delete(runDir, true);
                }
            }
            catch (IOException)
            {
                // Directory may be in use or have permission issues
                // The run is already removed from memory, so log and continue
            }
        }
    }

    /// <summary>
    /// Searches for runs across all experiments based on criteria.
    /// </summary>
    public override IEnumerable<IExperimentRun<T>> SearchRuns(string filter, int maxResults = 100)
    {
        if (maxResults <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxResults), "Max results must be greater than zero.");

        lock (SyncLock)
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
        if (!Directory.Exists(StorageDirectory))
            return;

        // Load experiments - map directories to meta file paths
        var metaFiles = Directory.GetDirectories(StorageDirectory)
            .Select(expDir => Path.Combine(expDir, "meta.json"))
            .Where(File.Exists);

        foreach (var metaFile in metaFiles)
        {
            try
            {
                // Validate path is within storage directory
                ValidatePathWithinDirectory(metaFile, StorageDirectory);

                var json = File.ReadAllText(metaFile);
                var experiment = DeserializeFromJson<Experiment>(json);
                if (experiment != null)
                {
                    _experiments[experiment.ExperimentId] = experiment;

                    // Load runs for this experiment
                    LoadRunsForExperiment(experiment.ExperimentId);
                }
            }
            catch (IOException ex)
            {
                // Skip experiment with unreadable data
                Console.WriteLine($"[ExperimentTracker] Failed to read experiment file '{metaFile}': {ex.Message}");
            }
            catch (JsonException ex)
            {
                // Skip experiment with corrupted data
                Console.WriteLine($"[ExperimentTracker] Failed to deserialize experiment file '{metaFile}': {ex.Message}");
            }
        }
    }

    private void LoadRunsForExperiment(string experimentId)
    {
        var experimentDir = GetExperimentDirectoryPath(experimentId);

        // Map run directories to meta file paths
        var metaFiles = Directory.GetDirectories(experimentDir)
            .Select(runDir => Path.Combine(runDir, "meta.json"))
            .Where(File.Exists);

        foreach (var metaFile in metaFiles)
        {
            try
            {
                // Validate path is within storage directory
                ValidatePathWithinDirectory(metaFile, StorageDirectory);

                var json = File.ReadAllText(metaFile);
                var run = DeserializeFromJson<ExperimentRun<T>>(json);
                if (run != null)
                {
                    _runs[run.RunId] = run;
                }
            }
            catch (IOException ex)
            {
                // Skip run with unreadable data
                Console.WriteLine($"[ExperimentTracker] Failed to read run file '{metaFile}': {ex.Message}");
            }
            catch (JsonException ex)
            {
                // Skip run with corrupted data
                Console.WriteLine($"[ExperimentTracker] Failed to deserialize run file '{metaFile}': {ex.Message}");
            }
        }
    }

    private void SaveExperiment(Experiment experiment)
    {
        var experimentDir = GetExperimentDirectoryPath(experiment.ExperimentId);
        Directory.CreateDirectory(experimentDir);

        var metaFile = Path.Combine(experimentDir, "meta.json");
        ValidatePathWithinDirectory(metaFile, StorageDirectory);

        var json = SerializeToJson(experiment);
        File.WriteAllText(metaFile, json);
    }

    private void SaveRun(ExperimentRun<T> run)
    {
        var runDir = GetRunDirectory(run.RunId);
        Directory.CreateDirectory(runDir);

        var metaFile = Path.Combine(runDir, "meta.json");
        ValidatePathWithinDirectory(metaFile, StorageDirectory);

        var json = SerializeToJson(run);
        File.WriteAllText(metaFile, json);
    }

    private string GetRunDirectory(string runId)
    {
        if (!_runs.TryGetValue(runId, out var run))
            throw new InvalidOperationException($"Run with ID '{runId}' not found in tracker. This is an internal error.");

        var experimentDir = GetExperimentDirectoryPath(run.ExperimentId);
        var sanitizedRunId = GetSanitizedFileName(runId);
        var path = Path.Combine(experimentDir, sanitizedRunId);
        ValidatePathWithinDirectory(path, StorageDirectory);
        return path;
    }

    #endregion
}
