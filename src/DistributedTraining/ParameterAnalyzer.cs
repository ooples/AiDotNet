using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Analyzes model parameters and creates optimized groupings for distributed communication.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Think of ParameterAnalyzer as a smart packing assistant. When shipping items, you don't
/// want to send thousands of tiny packages - it's inefficient! Instead, you group small
/// items together into larger boxes.
/// </para>
/// <para>
/// Similarly, when communicating parameters across GPUs:
/// - Sending many small parameter arrays is slow (lots of communication overhead)
/// - Grouping small parameters together reduces the number of messages
/// - This analyzer figures out the best way to group parameters for efficiency
/// </para>
/// <para>
/// For example, instead of sending 1000 separate bias vectors (each with 1 parameter),
/// we might group them into 10 larger chunks (each with 100 parameters).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
public class ParameterAnalyzer<T>
{
    private readonly int _minimumGroupSize;
    private readonly int _worldSize;

    /// <summary>
    /// Threshold multiplier for merging remaining parameters into the last group.
    /// If remaining parameters are less than minimum group size * this threshold,
    /// they are merged with the last group to avoid creating a tiny final group.
    /// </summary>
    private const double REMAINING_PARAMS_MERGE_THRESHOLD = 1.5;

    /// <summary>
    /// Divisor for calculating base group size for distributed training.
    /// Base group size = total parameters / (world size * this divisor).
    /// This ensures each process gets multiple groups for better load balancing.
    /// </summary>
    private const int DISTRIBUTION_GROUP_DIVISOR = 4;

    /// <summary>
    /// Threshold divisor for merging small final groups.
    /// Groups smaller than (base group size / this divisor) are merged with
    /// the previous group to avoid inefficiently small parameter groups.
    /// </summary>
    private const int SMALL_GROUP_MERGE_DIVISOR = 2;

    /// <summary>
    /// Represents a group of parameters that should be communicated together.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is like a shipping box that contains multiple items. Each ParameterGroup
    /// represents a chunk of parameters that will be sent together in one communication.
    /// </para>
    /// </remarks>
    public class ParameterGroup
    {
        /// <summary>
        /// The starting index of this group in the full parameter vector.
        /// </summary>
        public int StartIndex { get; set; }

        /// <summary>
        /// The number of parameters in this group.
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// A descriptive name for this parameter group (e.g., "Layer1.Weights").
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Indicates whether this group was created by merging smaller groups.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b>
        /// True if this group contains multiple small parameter arrays that were
        /// combined for efficiency. False if it represents a single large parameter array.
        /// </para>
        /// </remarks>
        public bool IsMerged { get; set; }
    }

    /// <summary>
    /// Creates a new parameter analyzer with the specified settings.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This creates the analyzer that will figure out how to group parameters.
    /// You tell it:
    /// - minimumGroupSize: The smallest acceptable group size (smaller groups get merged)
    /// - worldSize: How many processes are sharing the work (affects optimal group sizes)
    /// </para>
    /// </remarks>
    /// <param name="minimumGroupSize">Minimum size for a parameter group (smaller groups will be merged)</param>
    /// <param name="worldSize">Number of processes in the distributed group</param>
    public ParameterAnalyzer(int minimumGroupSize = 1024, int worldSize = 1)
    {
        if (minimumGroupSize <= 0)
        {
            throw new ArgumentException("Minimum group size must be positive.", nameof(minimumGroupSize));
        }

        if (worldSize <= 0)
        {
            throw new ArgumentException("World size must be positive.", nameof(worldSize));
        }

        _minimumGroupSize = minimumGroupSize;
        _worldSize = worldSize;
    }

    /// <summary>
    /// Analyzes a model's parameters and creates optimized groupings.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This method looks at all the parameters in your model and decides how to
    /// group them for efficient communication. It returns a list of ParameterGroups,
    /// each representing parameters that should be sent together.
    /// </para>
    /// <para>
    /// The analyzer:
    /// 1. Identifies natural parameter boundaries (e.g., weights vs biases)
    /// 2. Merges small groups that are below the minimum size
    /// 3. Ensures groups are aligned with process boundaries for even distribution
    /// </para>
    /// </remarks>
    /// <typeparam name="TInput">The input type of the model</typeparam>
    /// <typeparam name="TOutput">The output type of the model</typeparam>
    /// <param name="model">The model to analyze</param>
    /// <returns>A list of optimized parameter groups</returns>
    public List<ParameterGroup> AnalyzeModel<TInput, TOutput>(IFullModel<T, TInput, TOutput> model)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        var parameters = model.GetParameters();
        return AnalyzeParameters(parameters);
    }

    /// <summary>
    /// Analyzes a parameter vector and creates optimized groupings.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is the core analysis method. It takes a long list of parameters
    /// and intelligently groups them for efficient communication.
    /// </para>
    /// <para>
    /// Strategy:
    /// 1. Start with natural boundaries (we assume every N parameters belong together)
    /// 2. Merge groups that are too small
    /// 3. Align group boundaries to make distribution across processes easier
    /// </para>
    /// </remarks>
    /// <param name="parameters">The parameter vector to analyze</param>
    /// <returns>A list of optimized parameter groups</returns>
    public List<ParameterGroup> AnalyzeParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        var groups = new List<ParameterGroup>();
        int totalParams = parameters.Length;

        if (totalParams == 0)
        {
            return groups;
        }

        // Simple strategy: Create groups of at least minimum size
        // This is a basic implementation; a more sophisticated version could:
        // - Detect layer boundaries
        // - Group parameters by type (weights vs biases)
        // - Optimize for specific communication patterns

        int currentIndex = 0;
        int groupId = 0;

        while (currentIndex < totalParams)
        {
            // Calculate group size
            int remainingParams = totalParams - currentIndex;
            int groupSize = Math.Min(_minimumGroupSize, remainingParams);

            // If the remaining parameters are slightly larger than minimum,
            // make the last group slightly larger rather than creating a tiny final group
            if (remainingParams > _minimumGroupSize && remainingParams < _minimumGroupSize * REMAINING_PARAMS_MERGE_THRESHOLD)
            {
                groupSize = remainingParams;
            }

            // Create the group
            groups.Add(new ParameterGroup
            {
                StartIndex = currentIndex,
                Size = groupSize,
                Name = $"ParameterGroup_{groupId}",
                IsMerged = false
            });

            currentIndex += groupSize;
            groupId++;
        }

        return groups;
    }

    /// <summary>
    /// Analyzes parameters and creates groups optimized for even distribution across processes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// When distributing work across multiple processes, we want each process to get
    /// roughly the same amount of work. This method creates groups that divide evenly.
    /// </para>
    /// <para>
    /// For example, if you have 10,000 parameters and 4 processes:
    /// - Each process should get ~2,500 parameters
    /// - We create groups sized to divide evenly by 4
    /// </para>
    /// </remarks>
    /// <param name="parameters">The parameter vector to analyze</param>
    /// <returns>A list of parameter groups optimized for distribution</returns>
    public List<ParameterGroup> AnalyzeForDistribution(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        var groups = new List<ParameterGroup>();
        int totalParams = parameters.Length;

        if (totalParams == 0)
        {
            return groups;
        }

        // Calculate optimal group size for even distribution
        // We want each group to be:
        // 1. At least the minimum size
        // 2. Divisible by world size (or close to it)
        // 3. An even division of total parameters

        int baseGroupSize = Math.Max(_minimumGroupSize, totalParams / (_worldSize * DISTRIBUTION_GROUP_DIVISOR));

        // Round to nearest multiple of world size for better alignment
        if (baseGroupSize > _worldSize)
        {
            baseGroupSize = (baseGroupSize / _worldSize) * _worldSize;
        }

        int currentIndex = 0;
        int groupId = 0;

        while (currentIndex < totalParams)
        {
            int remainingParams = totalParams - currentIndex;
            int groupSize = Math.Min(baseGroupSize, remainingParams);

            // If this would be the last group and it's small, merge with previous
            // Groups smaller than half the base group size are merged to avoid inefficiently small groups
            if (remainingParams - groupSize > 0 && remainingParams - groupSize < baseGroupSize / SMALL_GROUP_MERGE_DIVISOR)
            {
                groupSize = remainingParams;
            }

            groups.Add(new ParameterGroup
            {
                StartIndex = currentIndex,
                Size = groupSize,
                Name = $"DistributedGroup_{groupId}",
                IsMerged = false
            });

            currentIndex += groupSize;
            groupId++;
        }

        return groups;
    }

    /// <summary>
    /// Calculates statistics about parameter distribution for a model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This gives you information about how parameters would be distributed,
    /// helping you understand the efficiency of the grouping strategy.
    /// </para>
    /// </remarks>
    /// <param name="groups">The parameter groups to analyze</param>
    /// <returns>A dictionary of statistics (e.g., "TotalGroups", "AverageGroupSize"), or an empty dictionary if groups is null or empty.</returns>
    public Dictionary<string, double> CalculateDistributionStats(List<ParameterGroup> groups)
    {
        if (groups == null || groups.Count == 0)
        {
            return new Dictionary<string, double>();
        }

        var stats = new Dictionary<string, double>
        {
            ["TotalGroups"] = groups.Count,
            ["TotalParameters"] = groups.Sum(g => g.Size),
            ["AverageGroupSize"] = groups.Average(g => (double)g.Size),
            ["MinGroupSize"] = groups.Min(g => g.Size),
            ["MaxGroupSize"] = groups.Max(g => g.Size),
            ["MergedGroups"] = groups.Count(g => g.IsMerged),
            ["GroupsPerProcess"] = groups.Count / (double)_worldSize
        };

        // Calculate group size variance (measure of how evenly sized groups are)
        double avgSize = stats["AverageGroupSize"];
        double variance = groups.Average(g => Math.Pow(g.Size - avgSize, 2));
        stats["GroupSizeVariance"] = variance;
        stats["GroupSizeStdDev"] = Math.Sqrt(variance);

        return stats;
    }

    /// <summary>
    /// Validates that parameter groups cover all parameters without gaps or overlaps.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is a safety check to make sure our grouping is correct. It verifies:
    /// - Every parameter is in exactly one group
    /// - Groups don't overlap
    /// - There are no gaps between groups
    /// </para>
    /// </remarks>
    /// <param name="groups">The parameter groups to validate</param>
    /// <param name="totalParameterCount">The total number of parameters</param>
    /// <returns>True if grouping is valid, false otherwise</returns>
    /// <exception cref="InvalidOperationException">Thrown if validation fails with details</exception>
    public bool ValidateGrouping(List<ParameterGroup> groups, int totalParameterCount)
    {
        if (groups == null || groups.Count == 0)
        {
            throw new InvalidOperationException("Parameter groups cannot be null or empty.");
        }

        // Sort groups by start index
        var sortedGroups = groups.OrderBy(g => g.StartIndex).ToList();

        // Check first group starts at 0
        if (sortedGroups[0].StartIndex != 0)
        {
            throw new InvalidOperationException(
                $"First parameter group must start at index 0, but starts at {sortedGroups[0].StartIndex}.");
        }

        // Check for gaps and overlaps
        for (int i = 0; i < sortedGroups.Count - 1; i++)
        {
            int currentEnd = sortedGroups[i].StartIndex + sortedGroups[i].Size;
            int nextStart = sortedGroups[i + 1].StartIndex;

            if (currentEnd < nextStart)
            {
                throw new InvalidOperationException(
                    $"Gap detected between groups {i} and {i + 1}. " +
                    $"Group {i} ends at {currentEnd}, Group {i + 1} starts at {nextStart}.");
            }

            if (currentEnd > nextStart)
            {
                throw new InvalidOperationException(
                    $"Overlap detected between groups {i} and {i + 1}. " +
                    $"Group {i} ends at {currentEnd}, Group {i + 1} starts at {nextStart}.");
            }
        }

        // Check last group covers all parameters
        var lastGroup = sortedGroups[sortedGroups.Count - 1];
        int lastEnd = lastGroup.StartIndex + lastGroup.Size;

        if (lastEnd != totalParameterCount)
        {
            throw new InvalidOperationException(
                $"Parameter groups don't cover all parameters. " +
                $"Last group ends at {lastEnd}, but total parameter count is {totalParameterCount}.");
        }

        return true;
    }
}
