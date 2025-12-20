using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Models;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.Reasoning.Training;

/// <summary>
/// Collects and manages training data for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This collects examples of reasoning chains with their rewards
/// to train the model to reason better.
///
/// **What is Training Data Collection?**
/// In RL for reasoning, we need to collect:
/// 1. Reasoning chains (sequences of thoughts)
/// 2. Rewards for each chain (how good was the reasoning?)
/// 3. Correct answers (for supervised learning)
/// 4. Step-by-step rewards (for process supervision)
///
/// **Example training sample:**
/// ```
/// Problem: "What is 15 × 12?"
///
/// Chain 1 (correct):
/// Step 1: "Break down: 15 × 12 = 15 × 10 + 15 × 2" (reward: 0.9)
/// Step 2: "Calculate: 150 + 30" (reward: 0.9)
/// Step 3: "Final answer: 180" (reward: 1.0)
/// Chain reward: 0.93, Outcome reward: 1.0
///
/// Chain 2 (incorrect):
/// Step 1: "15 × 12 is approximately 15 × 10 = 150" (reward: 0.5)
/// Step 2: "Final answer: 150" (reward: 0.3)
/// Chain reward: 0.4, Outcome reward: 0.0
/// ```
///
/// **Why collect training data?**
/// - Improve reasoning through examples
/// - Learn which reasoning patterns work
/// - Fine-tune models with RL
/// - Build reward models (PRM/ORM)
/// - Enable self-improvement
///
/// **Data collection strategies:**
///
/// *1. Expert demonstrations:*
/// Collect human-written reasoning chains (expensive but high quality)
///
/// *2. Model sampling:*
/// Generate multiple chains from the model, keep good ones
///
/// *3. Iterative refinement:*
/// Collect initial chains → Train → Collect better chains → Repeat
///
/// *4. Curriculum learning:*
/// Start with easy problems, gradually increase difficulty
///
/// **Data quality considerations:**
/// - Diversity: Collect varied reasoning approaches
/// - Balance: Mix correct and incorrect examples
/// - Coverage: Include different problem types
/// - Difficulty distribution: Easy to hard
/// - Step granularity: Right level of detail
///
/// **Storage formats:**
/// - JSON: Human-readable, good for analysis
/// - Binary: Faster, more compact
/// - Database: Scalable, queryable
/// - HuggingFace datasets: Standard format
///
/// **Usage example:**
/// ```csharp
/// var collector = new TrainingDataCollector<double>();
///
/// // Collect a training sample
/// var chain = /* reasoning chain from solving a problem */;
/// var sample = new TrainingSample<double>
/// {
///     Problem = "What is 15 × 12?",
///     ReasoningChain = chain,
///     CorrectAnswer = "180",
///     ChainReward = 0.93,
///     OutcomeReward = 1.0
/// };
///
/// collector.AddSample(sample);
///
/// // Save for training
/// await collector.SaveToFileAsync("training_data.json");
///
/// // Later: Load for training
/// await collector.LoadFromFileAsync("training_data.json");
/// var batches = collector.GetBatches(batchSize: 32);
/// ```
///
/// **Research:**
/// - "Let's Verify Step by Step" (Lightman et al., 2023) - PRM training data
/// - "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
/// - "STaR: Self-Taught Reasoner" (Zelikman et al., 2022) - Bootstrapping
/// </para>
/// </remarks>
internal class TrainingDataCollector<T>
{
    private readonly List<TrainingSample<T>> _samples;
    private readonly INumericOperations<T> _numOps;
    private readonly Dictionary<string, int> _categoryCount;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the <see cref="TrainingDataCollector{T}"/> class.
    /// </summary>
    /// <param name="randomSeed">Optional seed for reproducible shuffling. Default is 42.</param>
    public TrainingDataCollector(int? randomSeed = 42)
    {
        _samples = new List<TrainingSample<T>>();
        _numOps = MathHelper.GetNumericOperations<T>();
        _categoryCount = new Dictionary<string, int>();
        _random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Total number of samples collected.
    /// </summary>
    public int SampleCount => _samples.Count;

    /// <summary>
    /// Gets statistics about the collected data.
    /// </summary>
    public DataStatistics<T> Statistics => CalculateStatistics();

    /// <summary>
    /// Adds a training sample to the collection.
    /// </summary>
    public void AddSample(TrainingSample<T> sample)
    {
        if (sample == null)
            throw new ArgumentNullException(nameof(sample));

        _samples.Add(sample);

        // Track category
        string category = sample.Category ?? "uncategorized";
        if (!_categoryCount.ContainsKey(category))
            _categoryCount[category] = 0;
        _categoryCount[category]++;
    }

    /// <summary>
    /// Adds multiple samples at once.
    /// </summary>
    public void AddSamples(IEnumerable<TrainingSample<T>> samples)
    {
        foreach (var sample in samples)
        {
            AddSample(sample);
        }
    }

    /// <summary>
    /// Gets all samples as a read-only list.
    /// </summary>
    public IReadOnlyList<TrainingSample<T>> GetAllSamples()
    {
        return _samples.AsReadOnly();
    }

    /// <summary>
    /// Clears all collected samples and category counts.
    /// </summary>
    public void Clear()
    {
        _samples.Clear();
        _categoryCount.Clear();
    }

    /// <summary>
    /// Removes low-quality samples based on reward threshold.
    /// </summary>
    public int FilterByQuality(double minReward)
    {
        int initialCount = _samples.Count;
        _samples.RemoveAll(s => Convert.ToDouble(s.ChainReward) < minReward);
        return initialCount - _samples.Count;
    }

    /// <summary>
    /// Balances the dataset by limiting samples per category.
    /// </summary>
    public void BalanceCategories(int maxPerCategory)
    {
        var balanced = new List<TrainingSample<T>>();
        var categoryGroups = _samples.GroupBy(s => s.Category ?? "uncategorized");

        foreach (var group in categoryGroups)
        {
            balanced.AddRange(group.Take(maxPerCategory));
        }

        _samples.Clear();
        _samples.AddRange(balanced);
    }

    /// <summary>
    /// Gets samples in batches for training.
    /// </summary>
    public List<List<TrainingSample<T>>> GetBatches(int batchSize, bool shuffle = true)
    {
        var samples = shuffle ? _samples.OrderBy(_ => _random.Next()).ToList() : _samples.ToList();
        var batches = new List<List<TrainingSample<T>>>();

        for (int i = 0; i < samples.Count; i += batchSize)
        {
            var batch = samples.Skip(i).Take(batchSize).ToList();
            batches.Add(batch);
        }

        return batches;
    }

    /// <summary>
    /// Splits data into train/validation/test sets.
    /// </summary>
    public (List<TrainingSample<T>> train, List<TrainingSample<T>> validation, List<TrainingSample<T>> test)
        SplitData(double trainRatio = 0.8, double validationRatio = 0.1)
    {
        if (trainRatio + validationRatio >= 1.0)
            throw new ArgumentException("Train + validation ratios must be < 1.0");

        var shuffled = _samples.OrderBy(_ => _random.Next()).ToList();
        int trainSize = (int)(shuffled.Count * trainRatio);
        int validationSize = (int)(shuffled.Count * validationRatio);

        var train = shuffled.Take(trainSize).ToList();
        var validation = shuffled.Skip(trainSize).Take(validationSize).ToList();
        var test = shuffled.Skip(trainSize + validationSize).ToList();

        return (train, validation, test);
    }

    /// <summary>
    /// Filters samples to only include those with correct final answers.
    /// </summary>
    public List<TrainingSample<T>> GetCorrectSamples()
    {
        return _samples.Where(s => Convert.ToDouble(s.OutcomeReward) >= 0.9).ToList();
    }

    /// <summary>
    /// Gets samples with diverse reasoning approaches.
    /// </summary>
    public List<TrainingSample<T>> GetDiverseSamples(int count)
    {
        // Simple diversity: Select samples with different problem types
        var diverse = new List<TrainingSample<T>>();
        var seenProblems = new HashSet<string>();

        foreach (var sample in _samples.OrderByDescending(s => Convert.ToDouble(s.ChainReward)))
        {
            string problemHash = sample.Problem.Length > 50
                ? sample.Problem.Substring(0, 50)
                : sample.Problem;

            if (!seenProblems.Contains(problemHash))
            {
                diverse.Add(sample);
                seenProblems.Add(problemHash);

                if (diverse.Count >= count)
                    break;
            }
        }

        return diverse;
    }

    /// <summary>
    /// Saves training data to a JSON file.
    /// </summary>
    public async Task SaveToFileAsync(string filePath, CancellationToken cancellationToken = default)
    {
        var settings = new JsonSerializerSettings
        {
            Formatting = Formatting.Indented,
            NullValueHandling = NullValueHandling.Ignore
        };

        var json = JsonConvert.SerializeObject(_samples, settings);
        File.WriteAllText(filePath, json);  // net462 compatible
        await Task.CompletedTask;  // Maintain async signature
    }

    /// <summary>
    /// Loads training data from a JSON file.
    /// </summary>
    public async Task LoadFromFileAsync(string filePath, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Training data file not found: {filePath}");

        var json = File.ReadAllText(filePath);  // net462 compatible
        var samples = JsonConvert.DeserializeObject<List<TrainingSample<T>>>(json);

        if (samples != null)
        {
            _samples.Clear();
            _samples.AddRange(samples);
        }

        await Task.CompletedTask;  // Maintain async signature
    }

    /// <summary>
    /// Exports data in HuggingFace dataset format.
    /// </summary>
    public async Task ExportToHuggingFaceFormatAsync(
        string outputDirectory,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(outputDirectory);

        var (train, validation, test) = SplitData();

        await SaveSplitAsync(Path.Combine(outputDirectory, "train.json"), train, cancellationToken);
        await SaveSplitAsync(Path.Combine(outputDirectory, "validation.json"), validation, cancellationToken);
        await SaveSplitAsync(Path.Combine(outputDirectory, "test.json"), test, cancellationToken);
    }

    private async Task SaveSplitAsync(
        string filePath,
        List<TrainingSample<T>> samples,
        CancellationToken cancellationToken)
    {
        var settings = new JsonSerializerSettings
        {
            Formatting = Formatting.Indented
        };

        var json = JsonConvert.SerializeObject(samples, settings);
        File.WriteAllText(filePath, json);  // net462 compatible
        await Task.CompletedTask;  // Maintain async signature
    }

    private DataStatistics<T> CalculateStatistics()
    {
        if (_samples.Count == 0)
        {
            return new DataStatistics<T>
            {
                TotalSamples = 0,
                AverageChainReward = _numOps.Zero,
                AverageOutcomeReward = _numOps.Zero,
                CorrectCount = 0,
                CorrectPercentage = 0.0
            };
        }

        var chainRewards = _samples.Select(s => Convert.ToDouble(s.ChainReward)).ToList();
        var outcomeRewards = _samples.Select(s => Convert.ToDouble(s.OutcomeReward)).ToList();
        int correctCount = _samples.Count(s => Convert.ToDouble(s.OutcomeReward) >= 0.9);

        return new DataStatistics<T>
        {
            TotalSamples = _samples.Count,
            AverageChainReward = _numOps.FromDouble(chainRewards.Average()),
            AverageOutcomeReward = _numOps.FromDouble(outcomeRewards.Average()),
            CorrectCount = correctCount,
            CorrectPercentage = (double)correctCount / _samples.Count,
            CategoryDistribution = new Dictionary<string, int>(_categoryCount),
            AverageStepsPerChain = _samples.Average(s => s.ReasoningChain?.Steps.Count ?? 0)
        };
    }
}

/// <summary>
/// Represents a single training sample.
/// </summary>
internal class TrainingSample<T>
{
    /// <summary>
    /// The problem or query.
    /// </summary>
    public string Problem { get; set; } = string.Empty;

    /// <summary>
    /// The reasoning chain generated.
    /// </summary>
    public ReasoningChain<T>? ReasoningChain { get; set; }

    /// <summary>
    /// The correct answer (if known).
    /// </summary>
    public string? CorrectAnswer { get; set; }

    /// <summary>
    /// Process reward (quality of reasoning steps).
    /// </summary>
    public T ChainReward { get; set; } = default!;

    /// <summary>
    /// Outcome reward (correctness of final answer).
    /// </summary>
    public T OutcomeReward { get; set; } = default!;

    /// <summary>
    /// Problem category or domain.
    /// </summary>
    public string? Category { get; set; }

    /// <summary>
    /// Difficulty level.
    /// </summary>
    public string? Difficulty { get; set; }

    /// <summary>
    /// Source of the sample (benchmark name, etc.).
    /// </summary>
    public string? Source { get; set; }

    /// <summary>
    /// Timestamp when collected.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Additional metadata.
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Statistics about collected training data.
/// </summary>
internal class DataStatistics<T>
{
    /// <summary>
    /// Total number of samples collected.
    /// </summary>
    public int TotalSamples { get; set; }

    /// <summary>
    /// Average reward across reasoning chains.
    /// </summary>
    public T AverageChainReward { get; set; } = default!;

    /// <summary>
    /// Average reward for the final outcomes.
    /// </summary>
    public T AverageOutcomeReward { get; set; } = default!;

    /// <summary>
    /// Number of samples considered correct.
    /// </summary>
    public int CorrectCount { get; set; }

    /// <summary>
    /// Percentage of correct samples.
    /// </summary>
    public double CorrectPercentage { get; set; }

    /// <summary>
    /// Distribution of samples by category.
    /// </summary>
    public Dictionary<string, int> CategoryDistribution { get; set; } = new();

    /// <summary>
    /// Average number of steps per reasoning chain.
    /// </summary>
    public double AverageStepsPerChain { get; set; }

    /// <summary>
    /// Returns a formatted, human-readable summary of the current statistics.
    /// </summary>
    public override string ToString()
    {
        return $@"Training Data Statistics:
Total Samples: {TotalSamples}
Average Chain Reward: {Convert.ToDouble(AverageChainReward):F3}
Average Outcome Reward: {Convert.ToDouble(AverageOutcomeReward):F3}
Correct: {CorrectCount}/{TotalSamples} ({CorrectPercentage:P1})
Average Steps: {AverageStepsPerChain:F1}
Categories: {string.Join(", ", CategoryDistribution.Select(kvp => $"{kvp.Key}: {kvp.Value}"))}";
    }
}
