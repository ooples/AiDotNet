namespace AiDotNet.Data.Structures;

/// <summary>
/// A meta-learning task implementation for hierarchical meta-learning scenarios.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// HierarchicalMetaLearningTask extends the basic task functionality by adding support
/// for hierarchical task structures. This is essential for meta-learning algorithms that
/// need to handle tasks with multiple levels of abstraction or subtask relationships.
/// </para>
/// <para>
/// <b>Key Features:</b>
/// - Hierarchical task relationships (parent/child tasks)
/// - Task taxonomy tracking
/// - Level-specific adaptation strategies
/// - Knowledge inheritance between levels
/// - Multi-scale learning
/// </para>
/// <para>
/// <b>Common Use Cases:</b>
/// - Hierarchical classification (e.g., animal → mammal → dog)
/// - Multi-level reinforcement learning
/// - Curriculum learning with task hierarchies
/// - Transfer learning with task taxonomies
/// - Meta-learning on structured problem spaces
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a hierarchical task
/// var task = new HierarchicalMetaLearningTask&lt;double&gt;(
///     "animal_classification",
///     level: 2,
///     parentTask: "mammal_classification"
/// );
///
/// // Set up data as usual
/// task.SupportInput = supportData;
/// task.SupportOutput = supportLabels;
/// task.QueryInput = queryData;
/// task.QueryOutput = queryLabels;
///
/// // Define hierarchy
/// task.SetParentLevel(1);  // Parent is at level 1
/// task.AddChildTask("dog_breed_classification", level: 3);
/// task.SetTaskCategory("fine_grained_classification");
/// </code>
/// </example>
public class HierarchicalMetaLearningTask<T, TInput, TOutput> : MetaLearningTaskBase<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private int _level;
    private string? _parentTaskId;
    private readonly List<string> _childTaskIds;
    private string? _taskCategory;
    private readonly Dictionary<string, object> _levelMetadata;

    /// <summary>
    /// Initializes a new instance of the HierarchicalMetaLearningTask class.
    /// </summary>
    /// <param name="name">The name of the task.</param>
    /// <param name="level">The hierarchical level of this task (0 = root level).</param>
    /// <param name="parentTaskId">Optional ID of the parent task.</param>
    public HierarchicalMetaLearningTask(string name, int level = 0, string? parentTaskId = null)
        : base(name)
    {
        _level = level;
        _parentTaskId = parentTaskId;
        _childTaskIds = new List<string>();
        _levelMetadata = new Dictionary<string, object>();

        // Add hierarchy information to metadata
        SetMetadata("hierarchical_level", level);
        if (!string.IsNullOrEmpty(parentTaskId))
            SetMetadata("parent_task", parentTaskId);
    }

    /// <summary>
    /// Gets or sets the hierarchical level of this task.
    /// </summary>
    /// <value>
    /// The level in the hierarchy (0 = root, higher numbers = more specific).
    /// </value>
    public int Level
    {
        get => _level;
        set
        {
            _level = value;
            SetMetadata("hierarchical_level", value);
        }
    }

    /// <summary>
    /// Gets or sets the ID of the parent task.
    /// </summary>
    /// <value>
    /// The parent task identifier, or null if this is a root-level task.
    /// </value>
    public string? ParentTaskId
    {
        get => _parentTaskId;
        set
        {
            _parentTaskId = value;
            if (!string.IsNullOrEmpty(value))
                SetMetadata("parent_task", value);
            else
                Metadata?.Remove("parent_task");
        }
    }

    /// <summary>
    /// Gets the collection of child task IDs.
    /// </summary>
    /// <value>
    /// Read-only collection of child task identifiers.
    /// </value>
    public IReadOnlyCollection<string> ChildTaskIds => _childTaskIds.AsReadOnly();

    /// <summary>
    /// Gets or sets the task category within the hierarchy.
    /// </summary>
    /// <value>
    /// The category label (e.g., "coarse", "medium", "fine_grained").
    /// </value>
    public string? TaskCategory
    {
        get => _taskCategory;
        set
        {
            _taskCategory = value;
            if (!string.IsNullOrEmpty(value))
                SetMetadata("task_category", value);
            else
                Metadata?.Remove("task_category");
        }
    }

    /// <summary>
    /// Adds a child task to this task.
    /// </summary>
    /// <param name="childTaskId">The ID of the child task.</param>
    /// <param name="childLevel">The hierarchical level of the child task.</param>
    /// <exception cref="ArgumentException">Thrown when child level is not greater than parent level.</exception>
    public void AddChildTask(string childTaskId, int? childLevel = null)
    {
        if (string.IsNullOrEmpty(childTaskId))
            throw new ArgumentException("Child task ID cannot be null or empty.", nameof(childTaskId));

        if (childLevel.HasValue && childLevel.Value <= _level)
            throw new ArgumentException($"Child level ({childLevel.Value}) must be greater than parent level ({_level}).");

        if (!_childTaskIds.Contains(childTaskId))
        {
            _childTaskIds.Add(childTaskId);
            SetMetadata($"child_task_{_childTaskIds.Count - 1}", childTaskId);
            if (childLevel.HasValue)
                SetMetadata($"child_level_{childTaskId}", childLevel.Value);
        }
    }

    /// <summary>
    /// Removes a child task from this task.
    /// </summary>
    /// <param name="childTaskId">The ID of the child task to remove.</param>
    /// <returns>True if the child was removed, false if it wasn't found.</returns>
    public bool RemoveChildTask(string childTaskId)
    {
        if (_childTaskIds.Remove(childTaskId))
        {
            // Remove from metadata
            var keysToRemove = Metadata?.Keys.Where(k => k.Contains(childTaskId)).ToList();
            if (keysToRemove != null)
            {
                foreach (var key in keysToRemove)
                    Metadata?.Remove(key);
            }
            return true;
        }
        return false;
    }

    /// <summary>
    /// Sets metadata specific to this hierarchical level.
    /// </summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="value">The metadata value.</param>
    public void SetLevelMetadata(string key, object value)
    {
        if (string.IsNullOrEmpty(key))
            throw new ArgumentException("Key cannot be null or empty.", nameof(key));

        _levelMetadata[$"level_{_level}_{key}"] = value;
        SetMetadata($"hierarchy_metadata_{key}", value);
    }

    /// <summary>
    /// Gets level-specific metadata.
    /// </summary>
    /// <typeparam name="TValue">The type of value to retrieve.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <returns>The value if found, otherwise default(TValue).</returns>
    public TValue? GetLevelMetadata<TValue>(string key)
    {
        var fullKey = $"level_{_level}_{key}";
        if (_levelMetadata.TryGetValue(fullKey, out var value) && value is TValue)
            return (TValue)value;

        return default;
    }

    /// <summary>
    /// Gets the depth of the hierarchy from this task to the root.
    /// </summary>
    /// <returns>The number of levels from this task to the root.</returns>
    public int GetDepthToRoot()
    {
        return _level;  // Since root is level 0
    }

    /// <summary>
    /// Gets the maximum depth of the subtree rooted at this task.
    /// </summary>
    /// <returns>The maximum number of levels in the subtree.</returns>
    public int GetSubtreeDepth()
    {
        if (_childTaskIds.Count == 0)
            return 0;

        return 1;  // At least one level below if there are children
    }

    /// <summary>
    /// Checks if this task is a root-level task.
    /// </summary>
    /// <returns>True if this task has no parent (level 0).</returns>
    public bool IsRoot()
    {
        return _level == 0 && string.IsNullOrEmpty(_parentTaskId);
    }

    /// <summary>
    /// Checks if this task is a leaf task (has no children).
    /// </summary>
    /// <returns>True if this task has no child tasks.</returns>
    public bool IsLeaf()
    {
        return _childTaskIds.Count == 0;
    }

    /// <summary>
    /// Creates a string representation with hierarchical information.
    /// </summary>
    /// <returns>String containing hierarchical details and configuration.</returns>
    public override string ToString()
    {
        var name = string.IsNullOrEmpty(Name) ? "HierarchicalMetaLearningTask" : Name;
        var result = $"{name} (Level: {_level}";

        if (!string.IsNullOrEmpty(_parentTaskId))
            result += $", Parent: {_parentTaskId}";

        if (_childTaskIds.Count > 0)
            result += $", Children: {_childTaskIds.Count}";

        if (!string.IsNullOrEmpty(_taskCategory))
            result += $", Category: {_taskCategory}";

        result += ")";

        return result;
    }
}