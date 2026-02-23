using System.Collections;
using AiDotNet.Evaluation.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.Evaluation.Results.Core;

/// <summary>
/// A collection of metrics, providing easy access by name and category.
/// </summary>
/// <remarks>
/// <para>
/// MetricCollection provides a convenient way to store, access, and iterate over evaluation metrics.
/// It supports indexing by name, filtering by category, and various output formats.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like a dictionary of metrics. You can access any metric by name
/// (e.g., collection["Accuracy"]) or iterate through all metrics. It also groups metrics by
/// category (Classification, Regression, etc.) for organized reporting.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for metric values.</typeparam>
public class MetricCollection<T> : IEnumerable<MetricWithCI<T>>
{
    private readonly Dictionary<string, MetricWithCI<T>> _metrics;
    private readonly Dictionary<string, List<string>> _categoryIndex;

    /// <summary>
    /// Gets the number of metrics in the collection.
    /// </summary>
    public int Count => _metrics.Count;

    /// <summary>
    /// Gets all metric names in the collection.
    /// </summary>
    public IReadOnlyCollection<string> Names => _metrics.Keys;

    /// <summary>
    /// Gets all categories in the collection.
    /// </summary>
    public IReadOnlyCollection<string> Categories => _categoryIndex.Keys;

    /// <summary>
    /// Initializes a new empty metric collection.
    /// </summary>
    public MetricCollection()
    {
        _metrics = new Dictionary<string, MetricWithCI<T>>(StringComparer.OrdinalIgnoreCase);
        _categoryIndex = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Initializes a metric collection from an existing dictionary.
    /// </summary>
    public MetricCollection(IEnumerable<MetricWithCI<T>> metrics) : this()
    {
        foreach (var metric in metrics)
        {
            Add(metric);
        }
    }

    /// <summary>
    /// Gets a metric by name.
    /// </summary>
    /// <param name="name">The metric name (case-insensitive).</param>
    /// <returns>The metric, or null if not found.</returns>
    public MetricWithCI<T>? this[string name]
    {
        get => _metrics.TryGetValue(name, out var metric) ? metric : null;
    }

    /// <summary>
    /// Adds a metric to the collection.
    /// </summary>
    /// <param name="metric">The metric to add.</param>
    /// <exception cref="ArgumentException">Thrown if a metric with the same name already exists.</exception>
    public void Add(MetricWithCI<T> metric)
    {
        if (string.IsNullOrEmpty(metric.Name))
        {
            throw new ArgumentException("Metric must have a name.", nameof(metric));
        }

        if (_metrics.ContainsKey(metric.Name))
        {
            throw new ArgumentException($"A metric with name '{metric.Name}' already exists in the collection. Use AddOrUpdate() to replace existing metrics.", nameof(metric));
        }

        _metrics[metric.Name] = metric;

        // Update category index
        var category = metric.Category ?? "Uncategorized";
        if (!_categoryIndex.TryGetValue(category, out var categoryList))
        {
            categoryList = new List<string>();
            _categoryIndex[category] = categoryList;
        }

        if (!categoryList.Contains(metric.Name, StringComparer.OrdinalIgnoreCase))
        {
            categoryList.Add(metric.Name);
        }
    }

    /// <summary>
    /// Adds multiple metrics to the collection.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if any metric with the same name already exists.</exception>
    public void AddRange(IEnumerable<MetricWithCI<T>> metrics)
    {
        foreach (var metric in metrics)
        {
            Add(metric);
        }
    }

    /// <summary>
    /// Adds or updates a metric in the collection.
    /// </summary>
    /// <param name="metric">The metric to add or update.</param>
    public void AddOrUpdate(MetricWithCI<T> metric)
    {
        if (string.IsNullOrEmpty(metric.Name))
        {
            throw new ArgumentException("Metric must have a name.", nameof(metric));
        }

        _metrics[metric.Name] = metric;

        // Update category index
        var category = metric.Category ?? "Uncategorized";
        if (!_categoryIndex.TryGetValue(category, out var categoryList))
        {
            categoryList = new List<string>();
            _categoryIndex[category] = categoryList;
        }

        if (!categoryList.Contains(metric.Name, StringComparer.OrdinalIgnoreCase))
        {
            categoryList.Add(metric.Name);
        }
    }

    /// <summary>
    /// Tries to get a metric by name.
    /// </summary>
    /// <param name="name">The metric name.</param>
    /// <param name="metric">The metric if found.</param>
    /// <returns>True if the metric was found.</returns>
    public bool TryGetMetric(string name, out MetricWithCI<T>? metric)
    {
        return _metrics.TryGetValue(name, out metric);
    }

    /// <summary>
    /// Checks if a metric exists in the collection.
    /// </summary>
    public bool Contains(string name)
    {
        return _metrics.ContainsKey(name);
    }

    /// <summary>
    /// Gets all metrics in a specific category.
    /// </summary>
    /// <param name="category">The category name.</param>
    /// <returns>Metrics in the category, or empty if category not found.</returns>
    public IEnumerable<MetricWithCI<T>> GetByCategory(string category)
    {
        if (!_categoryIndex.TryGetValue(category, out var names))
        {
            return Enumerable.Empty<MetricWithCI<T>>();
        }

        return names.Select(n => _metrics[n]);
    }

    /// <summary>
    /// Gets the best metric by the specified direction.
    /// </summary>
    /// <param name="direction">Whether higher or lower is better.</param>
    /// <returns>The best metric, or null if collection is empty.</returns>
    public MetricWithCI<T>? GetBest(MetricDirection direction = MetricDirection.HigherIsBetter)
    {
        if (_metrics.Count == 0) return null;

        var numOps = MathHelper.GetNumericOperations<T>();
        MetricWithCI<T>? best = null;

        foreach (var metric in _metrics.Values)
        {
            if (best == null)
            {
                best = metric;
                continue;
            }

            var comparison = numOps.Compare(metric.Value, best.Value);
            if ((direction == MetricDirection.HigherIsBetter && comparison > 0) ||
                (direction == MetricDirection.LowerIsBetter && comparison < 0))
            {
                best = metric;
            }
        }

        return best;
    }

    /// <summary>
    /// Gets the worst metric by the specified direction.
    /// </summary>
    public MetricWithCI<T>? GetWorst(MetricDirection direction = MetricDirection.HigherIsBetter)
    {
        if (_metrics.Count == 0) return null;

        var numOps = MathHelper.GetNumericOperations<T>();
        MetricWithCI<T>? worst = null;

        foreach (var metric in _metrics.Values)
        {
            if (worst == null)
            {
                worst = metric;
                continue;
            }

            var comparison = numOps.Compare(metric.Value, worst.Value);
            if ((direction == MetricDirection.HigherIsBetter && comparison < 0) ||
                (direction == MetricDirection.LowerIsBetter && comparison > 0))
            {
                worst = metric;
            }
        }

        return worst;
    }

    /// <summary>
    /// Filters metrics by a predicate.
    /// </summary>
    public IEnumerable<MetricWithCI<T>> Where(Func<MetricWithCI<T>, bool> predicate)
    {
        return _metrics.Values.Where(predicate);
    }

    /// <summary>
    /// Gets all valid metrics (non-NaN, non-infinite values).
    /// </summary>
    public IEnumerable<MetricWithCI<T>> GetValidMetrics()
    {
        return _metrics.Values.Where(m => m.IsValid);
    }

    /// <summary>
    /// Converts the collection to a dictionary.
    /// </summary>
    public Dictionary<string, T> ToDictionary()
    {
        return _metrics.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value.Value,
            StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Converts the collection to a dictionary grouped by category.
    /// </summary>
    public Dictionary<string, Dictionary<string, T>> ToDictionaryByCategory()
    {
        var result = new Dictionary<string, Dictionary<string, T>>(StringComparer.OrdinalIgnoreCase);

        foreach (var category in _categoryIndex.Keys)
        {
            result[category] = GetByCategory(category).ToDictionary(
                m => m.Name,
                m => m.Value,
                StringComparer.OrdinalIgnoreCase);
        }

        return result;
    }

    /// <summary>
    /// Formats all metrics as a table string.
    /// </summary>
    public string ToTableString(int decimalPlaces = 4, bool includeCI = true)
    {
        if (_metrics.Count == 0) return "No metrics available.";

        var lines = new List<string>();
        var maxNameLength = _metrics.Keys.Max(k => k.Length);

        lines.Add($"{"Metric".PadRight(maxNameLength)} | Value");
        lines.Add(new string('-', maxNameLength + 30));

        foreach (var category in _categoryIndex.Keys.OrderBy(c => c))
        {
            lines.Add($"[{category}]");
            foreach (var metric in GetByCategory(category))
            {
                var value = metric.Format(decimalPlaces, includeCI);
                lines.Add($"  {metric.Name.PadRight(maxNameLength - 2)} | {value}");
            }
        }

        return string.Join(Environment.NewLine, lines);
    }

    /// <summary>
    /// Returns an enumerator that iterates through the metrics.
    /// </summary>
    public IEnumerator<MetricWithCI<T>> GetEnumerator()
    {
        return _metrics.Values.GetEnumerator();
    }

    /// <summary>
    /// Returns an enumerator that iterates through the metrics.
    /// </summary>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
