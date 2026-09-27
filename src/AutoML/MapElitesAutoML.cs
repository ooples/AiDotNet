using System.Collections.ObjectModel;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.AutoML;

/// <summary>
/// Built-in supervised AutoML strategy that searches a deterministic MAP-Elites archive.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
/// <remarks>
/// <para>
/// Quality is the configured validation metric. The built-in behavior descriptors are model family
/// and normalized configuration complexity. Final-test data is not accepted by this API and cannot
/// influence selection or archive membership.
/// </para>
/// <para>
/// <see cref="AutoMLModelBase{T,TInput,TOutput}.TrialLimit"/> is the maximum number of unique,
/// expensive training attempts. Duplicate specifications are rejected before training and do not
/// consume that budget. Evaluation is intentionally sequential because the shared AutoML trial
/// executor owns best-model and trial-history state; proposal randomness still comes from stable,
/// candidate-local streams.
/// </para>
/// <para>
/// The search drives the generic <see cref="EvolutionEngine{TGenome}"/> over a
/// <see cref="MapElitesArchive{TGenome}"/> whose grid has one row per candidate model family and
/// <see cref="MapElitesAutoMLOptions.ComplexityBinCount"/> columns. Each step selects a parent elite through
/// <see cref="CuriosityEvolutionSelectionPolicy{TGenome}"/>, which favors parents whose offspring improved the
/// archive, then either samples a fresh family with probability
/// <see cref="MapElitesAutoMLOptions.ExplorationProbability"/> or mutates the parent's hyperparameters, optionally
/// borrowing individual values from an inspiring elite of the same family drawn from up to
/// <see cref="MapElitesAutoMLOptions.InspirationCount"/> archive members. The proposal is trained and validated
/// once and kept only if it beats the incumbent of its cell. This is MAP-Elites as described by Mouret and Clune,
/// "Illuminating search spaces by mapping elites" (2015, arXiv:1504.04909). Seed and proposal streams are PCG
/// generators (O'Neill, 2014) exposed through <see cref="StableRandom"/>, so a run is reproducible across
/// processes and platforms. Wall-clock cost is dominated by the at most <c>TrialLimit</c> model fits; archive
/// bookkeeping is at most O(n log n) in the number of occupied cells per insertion, which is negligible by
/// comparison.
/// </para>
/// <para><b>For Beginners:</b> Ordinary AutoML keeps a single best model. This strategy keeps a small grid of
/// "best in class" models instead, one slot for every model family at every level of configuration complexity,
/// and still returns the overall winner the way any other AutoML strategy does. The extra diversity matters when
/// the winner is not what you can deploy: if the top model is a heavy ensemble, the archive also tells you the
/// best simple model that was found. Most users never construct this class directly; set
/// <c>AutoMLOptions.SearchStrategy = AutoMLSearchStrategy.MapElites</c>, pass those options to
/// <c>ConfigureAutoML</c> on the model builder, and read <see cref="Archive"/> after the run to inspect the grid.
/// Construct it yourself only when you need custom <see cref="MapElitesAutoMLOptions"/> such as a different seed
/// or bin count, then plug the instance in through the <c>ConfigureAutoML(IAutoMLModel)</c> overload.</para>
/// </remarks>
[ModelMetadataExempt]
public sealed class MapElitesAutoML<T, TInput, TOutput> :
    BuiltInSupervisedAutoMLModelBase<T, TInput, TOutput>,
    IMapElitesAutoMLModel<T, TInput, TOutput>
{
    private const string ModelFamilyDescriptor = "model_family";
    private const string ComplexityDescriptor = "configuration_complexity";
    private readonly MapElitesAutoMLOptions _options;
    private IReadOnlyList<MapElitesAutoMLArchiveEntry> _archive =
        Array.AsReadOnly(Array.Empty<MapElitesAutoMLArchiveEntry>());
    private long _suggestionCounter;
    private int _searchStarted;

    /// <summary>Initializes a MAP-Elites AutoML search strategy.</summary>
    /// <param name="options">Optional quality-diversity settings. Values are validated and snapshotted.</param>
    /// <param name="random">
    /// Optional RNG used only by inherited trial features such as unseeded cross-validation shuffling.
    /// MAP-Elites proposal streams are derived from <see cref="MapElitesAutoMLOptions.Seed"/>.
    /// </param>
    public MapElitesAutoML(MapElitesAutoMLOptions? options = null, Random? random = null)
        : base(random ?? new StableSystemRandom(StableRandom.CreateStream(options?.Seed ?? 1234UL, 0x43564D4CUL)))
    {
        _options = (options ?? new MapElitesAutoMLOptions()).SnapshotAndValidate();
    }

    /// <inheritdoc/>
    public IReadOnlyList<MapElitesAutoMLArchiveEntry> Archive => _archive;

    /// <inheritdoc/>
    public string ArchiveStateHash { get; private set; } = string.Empty;

    /// <inheritdoc/>
    public override async Task<IFullModel<T, TInput, TOutput>> SearchAsync(
        TInput inputs,
        TOutput targets,
        TInput validationInputs,
        TOutput validationTargets,
        TimeSpan timeLimit,
        CancellationToken cancellationToken = default)
    {
        if (TrialLimit <= 0) throw new InvalidOperationException("TrialLimit must be positive for MAP-Elites AutoML.");
        if (timeLimit <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(timeLimit));
        if (timeLimit.TotalMilliseconds > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(timeLimit), "The time limit exceeds the cross-target cancellation range.");
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (validationInputs is null) throw new ArgumentNullException(nameof(validationInputs));
        if (validationTargets is null) throw new ArgumentNullException(nameof(validationTargets));
        if (typeof(TInput) != typeof(Matrix<T>) || typeof(TOutput) != typeof(Vector<T>))
        {
            throw new NotSupportedException(
                $"MAP-Elites AutoML currently supports Matrix<T>/Vector<T> supervised tasks. " +
                $"Received {typeof(TInput).Name}/{typeof(TOutput).Name}.");
        }
        if (Interlocked.Exchange(ref _searchStarted, 1) != 0)
            throw new InvalidOperationException("A MapElitesAutoML instance can run only one search.");

        Status = AutoMLStatus.Running;
        TimeLimit = timeLimit;

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            EnsureDefaultOptimizationMetric(targets);
            EnsureDefaultCandidateModels(inputs, targets);

            SearchDefinition definition = CaptureSearchDefinition();
            if (definition.ModelTypes.Count == 0)
                throw new InvalidOperationException("No candidate models are configured for MAP-Elites AutoML.");

            EvolutionOptimizationDirection direction = _maximize
                ? EvolutionOptimizationDirection.Maximize
                : EvolutionOptimizationDirection.Minimize;
            string dataHash = ComputeDataHash(
                inputs,
                targets,
                validationInputs,
                validationTargets,
                cancellationToken);
            DateTime deadline = DateTime.UtcNow.Add(timeLimit);
            var task = new MapElitesTask(
                this,
                definition,
                direction,
                dataHash,
                inputs,
                targets,
                validationInputs,
                validationTargets);
            var variation = new MapElitesVariation(definition, _options);
            int seedCount = Math.Min(TrialLimit, Math.Max(_options.InitialPopulationSize, definition.ModelTypes.Count));
            MapElitesGenome[] seeds = CreateSeeds(definition, seedCount, _options.Seed);
            int maxProposals = ComputeMaxProposals(TrialLimit, seedCount, _options.MaxProposalMultiplier);

            var descriptors = new[]
            {
                new EvolutionDescriptorDefinition(
                    ModelFamilyDescriptor,
                    -0.5,
                    definition.ModelTypes.Count - 0.5,
                    definition.ModelTypes.Count,
                    EvolutionOutOfRangePolicy.Clamp),
                new EvolutionDescriptorDefinition(
                    ComplexityDescriptor,
                    0.0,
                    1.0,
                    _options.ComplexityBinCount,
                    EvolutionOutOfRangePolicy.Clamp)
            };
            long descriptorGridSizeLong = (long)definition.ModelTypes.Count * _options.ComplexityBinCount;
            if (descriptorGridSizeLong > 10_000_000)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(_options.ComplexityBinCount),
                    "The model-family and complexity descriptor grid cannot exceed 10,000,000 cells.");
            }
            int descriptorGridSize = (int)descriptorGridSizeLong;
            int effectiveArchiveCapacity = _options.ArchiveCapacity == 0
                ? 0
                : Math.Min(_options.ArchiveCapacity, descriptorGridSize);

            var engineOptions = new EvolutionEngineOptions
            {
                RunId = "automl-map-elites",
                Seed = _options.Seed,
                MaxEvaluationAttempts = TrialLimit,
                MaxProposals = maxProposals,
                MaxGenerations = maxProposals,
                ProposalBatchSize = 1,
                MaxDegreeOfParallelism = 1,
                ExecutionMode = EvolutionExecutionMode.Deterministic,
                FailurePolicy = EvolutionFailurePolicy.Continue,
                MaxRetries = 0,
                TimeLimit = timeLimit,
                EnableEvaluationCache = true,
                DeduplicateFailedCandidates = true,
                IslandCount = _options.IslandCount,
                MigrationInterval = _options.MigrationInterval,
                MigrantsPerIsland = _options.MigrantsPerIsland,
                InspirationCount = _options.InspirationCount
            };

            var engine = new EvolutionEngine<MapElitesGenome>(
                task,
                variation,
                _ => new MapElitesArchive<MapElitesGenome>(
                    descriptors,
                    direction,
                    effectiveArchiveCapacity),
                engineOptions,
                selection: new CuriosityEvolutionSelectionPolicy<MapElitesGenome>());

            EvolutionRunResult<MapElitesGenome> result = await engine
                .RunAsync(seeds, cancellationToken)
                .ConfigureAwait(false);

            _archive = Array.AsReadOnly(result.Islands
                .SelectMany(island => island.Entries)
                .OrderBy(entry => entry.Cell.StableKey, StringComparer.Ordinal)
                .Select(ToPublicArchiveEntry)
                .ToArray());
            ArchiveStateHash = result.StateHash;

            if (BestModel is null)
            {
                IReadOnlyList<TrialResult> history = GetTrialHistory();
                int failed = history.Count(item => !item.Success);
                throw new InvalidOperationException(
                    $"MAP-Elites AutoML found no valid model within the budget. " +
                    $"Expensive trials recorded: {history.Count}; failed: {failed}.");
            }

            await TrySelectEnsembleAsBestAsync(
                inputs,
                targets,
                validationInputs,
                validationTargets,
                deadline,
                cancellationToken).ConfigureAwait(false);

            Status = AutoMLStatus.Completed;
            return BestModel;
        }
        catch (OperationCanceledException)
        {
            Status = AutoMLStatus.Cancelled;
            throw;
        }
        catch (Exception)
        {
            Status = AutoMLStatus.Failed;
            throw;
        }
    }

    /// <inheritdoc/>
    public override Task<Dictionary<string, object>> SuggestNextTrialAsync()
    {
        SearchDefinition definition = CaptureSearchDefinition();
        if (definition.ModelTypes.Count == 0)
            throw new InvalidOperationException("No candidate models are configured for MAP-Elites AutoML.");

        long suggestion = Interlocked.Increment(ref _suggestionCounter) - 1;
        ulong streamId = unchecked((ulong)suggestion);
        StableRandom random = StableRandom.CreateStream(_options.Seed, streamId);
        Type modelType = definition.ModelTypes[(int)(streamId % (ulong)definition.ModelTypes.Count)];
        MapElitesGenome genome = CreateRandomGenome(modelType, definition.SearchSpaces[modelType], random);
        return Task.FromResult(genome.ToTrialParameters());
    }

    private SearchDefinition CaptureSearchDefinition()
    {
        List<Type> modelTypes;
        Dictionary<string, ParameterRange> overrides;
        lock (_lock)
        {
            modelTypes = _candidateModels
                .Where(type => type is not null)
                .Distinct()
                .ToList();
            overrides = _searchSpace.ToDictionary(
                item => item.Key,
                item => (ParameterRange)item.Value.Clone(),
                StringComparer.Ordinal);
        }

        var spaces = new Dictionary<Type, IReadOnlyDictionary<string, ParameterRange>>();
        foreach (Type modelType in modelTypes)
        {
            Dictionary<string, ParameterRange> merged = GetDefaultSearchSpace(modelType)
                .ToDictionary(
                    item => item.Key,
                    item => (ParameterRange)item.Value.Clone(),
                    StringComparer.Ordinal);
            foreach (KeyValuePair<string, ParameterRange> item in overrides)
                merged[item.Key] = (ParameterRange)item.Value.Clone();
            spaces.Add(modelType, new ReadOnlyDictionary<string, ParameterRange>(merged));
        }

        return new SearchDefinition(modelTypes, spaces);
    }

    private static MapElitesGenome[] CreateSeeds(SearchDefinition definition, int count, ulong rootSeed)
    {
        var result = new MapElitesGenome[count];
        for (int index = 0; index < count; index++)
        {
            Type modelType = definition.ModelTypes[index % definition.ModelTypes.Count];
            StableRandom random = StableRandom.CreateStream(rootSeed, unchecked((ulong)index));
            result[index] = CreateRandomGenome(modelType, definition.SearchSpaces[modelType], random);
        }
        return result;
    }

    private static MapElitesGenome CreateRandomGenome(
        Type modelType,
        IReadOnlyDictionary<string, ParameterRange> searchSpace,
        StableRandom random)
    {
        var adapter = new StableSystemRandom(random);
        Dictionary<string, object> parameters = AutoMLParameterSampler.Sample(adapter, searchSpace);
        return new MapElitesGenome(modelType, parameters);
    }

    private static int ComputeMaxProposals(int trialLimit, int seedCount, int multiplier)
    {
        long value = Math.Max(seedCount, (long)trialLimit * multiplier);
        return value > int.MaxValue ? int.MaxValue : (int)value;
    }

    private static MapElitesAutoMLArchiveEntry ToPublicArchiveEntry(
        EvolutionArchiveEntry<MapElitesGenome> entry)
    {
        double score = entry.Evaluation.Quality
            ?? throw new InvalidOperationException("Archive elites always carry a completed quality value.");
        return new MapElitesAutoMLArchiveEntry(
            entry.Evaluation.EvaluationId,
            entry.Evaluation.GenomeId,
            entry.Candidate.CanonicalGenome.Genome.ModelType,
            entry.Candidate.CanonicalGenome.Genome.Parameters,
            score,
            entry.Evaluation.Descriptors,
            entry.Cell.Bins);
    }

    private static string ComputeDataHash(
        TInput inputs,
        TOutput targets,
        TInput validationInputs,
        TOutput validationTargets,
        CancellationToken cancellationToken)
    {
        if (inputs is not Matrix<T> trainX || targets is not Vector<T> trainY ||
            validationInputs is not Matrix<T> validationX || validationTargets is not Vector<T> validationY)
        {
            throw new NotSupportedException(
                "MAP-Elites AutoML currently supports Matrix<T>/Vector<T> supervised tasks.");
        }

        using (SHA256 hash = SHA256.Create())
        {
            AppendMatrix(hash, "train-inputs", trainX, cancellationToken);
            AppendVector(hash, "train-targets", trainY, cancellationToken);
            AppendMatrix(hash, "validation-inputs", validationX, cancellationToken);
            AppendVector(hash, "validation-targets", validationY, cancellationToken);
            hash.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
            byte[] digest = hash.Hash
                ?? throw new InvalidOperationException("The SHA-256 data hash was not finalized.");
            return ToHex(digest);
        }
    }

    private static void AppendMatrix(
        SHA256 hash,
        string name,
        Matrix<T> matrix,
        CancellationToken cancellationToken)
    {
        AppendHashComponent(hash, name);
        AppendHashComponent(hash, matrix.Rows.ToString(CultureInfo.InvariantCulture));
        AppendHashComponent(hash, matrix.Columns.ToString(CultureInfo.InvariantCulture));
        for (int row = 0; row < matrix.Rows; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int column = 0; column < matrix.Columns; column++)
                AppendHashComponent(hash, FormatValue(matrix[row, column]));
        }
    }

    private static void AppendVector(
        SHA256 hash,
        string name,
        Vector<T> vector,
        CancellationToken cancellationToken)
    {
        AppendHashComponent(hash, name);
        AppendHashComponent(hash, vector.Length.ToString(CultureInfo.InvariantCulture));
        for (int index = 0; index < vector.Length; index++)
        {
            if ((index & 255) == 0) cancellationToken.ThrowIfCancellationRequested();
            AppendHashComponent(hash, FormatValue(vector[index]));
        }
    }

    private static void AppendHashComponent(HashAlgorithm hash, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        byte[] length = BitConverter.GetBytes(bytes.Length);
        if (BitConverter.IsLittleEndian) Array.Reverse(length);
        hash.TransformBlock(length, 0, length.Length, length, 0);
        hash.TransformBlock(bytes, 0, bytes.Length, bytes, 0);
    }

    private static string ToHex(byte[] hash)
    {
        var builder = new StringBuilder(hash.Length * 2);
        foreach (byte value in hash)
            builder.Append(value.ToString("x2", CultureInfo.InvariantCulture));
        return builder.ToString();
    }

    private static string FormatValue(object? value)
    {
        if (value is null) return "null";
        if (value is Type type) return "type:" + (type.AssemblyQualifiedName ?? type.FullName ?? type.Name);
        Type valueType = value.GetType();
        if (valueType.IsEnum)
        {
            return "enum:" + (valueType.AssemblyQualifiedName ?? valueType.FullName ?? valueType.Name) + ":" +
                Enum.Format(valueType, value, "D");
        }
        if (!IsSupportedImmutableValue(value))
            throw new NotSupportedException(
                $"MAP-Elites AutoML parameters must be immutable scalar values. Type '{valueType.FullName}' is not supported.");
        string formatted = value is IFormattable formattable
            ? formattable.ToString(null, CultureInfo.InvariantCulture) ?? string.Empty
            : value.ToString() ?? string.Empty;
        return (valueType.AssemblyQualifiedName ?? valueType.FullName ?? valueType.Name) + ":" + formatted;
    }

    private static bool IsSupportedImmutableValue(object value)
    {
        Type type = value.GetType();
        if (type.IsEnum || value is Type || value is string || value is Guid ||
            value is TimeSpan || value is DateTimeOffset)
            return true;
        return Type.GetTypeCode(type) != TypeCode.Object;
    }

    private static double ComputeComplexity(
        MapElitesGenome genome,
        IReadOnlyDictionary<string, ParameterRange> searchSpace)
    {
        if (searchSpace.Count == 0) return 0;
        double total = 0;
        int count = 0;
        foreach (KeyValuePair<string, ParameterRange> item in searchSpace.OrderBy(item => item.Key, StringComparer.Ordinal))
        {
            if (!genome.Parameters.TryGetValue(item.Key, out object? value)) continue;
            total += NormalizeParameter(value, item.Value);
            count++;
        }
        if (count == 0) return 0;
        return Math.Max(0, Math.Min(1, total / count));
    }

    private static double NormalizeParameter(object value, ParameterRange range)
    {
        if (range.Type == ParameterType.Boolean) return value is bool enabled && enabled ? 1 : 0;
        if (range.Type == ParameterType.Categorical)
        {
            if (range.CategoricalValues is null || range.CategoricalValues.Count <= 1) return 0;
            int index = range.CategoricalValues.FindIndex(candidate => Equals(candidate, value));
            return index < 0 ? 0 : (double)index / (range.CategoricalValues.Count - 1);
        }
        if (range.Type != ParameterType.Integer && range.Type != ParameterType.Float &&
            range.Type != ParameterType.Continuous)
            return 0;

        try
        {
            double minimum = range.MinValue is null ? 0 : Convert.ToDouble(range.MinValue, CultureInfo.InvariantCulture);
            double maximum = range.MaxValue is null ? minimum + 1 : Convert.ToDouble(range.MaxValue, CultureInfo.InvariantCulture);
            double current = Convert.ToDouble(value, CultureInfo.InvariantCulture);
            if (maximum < minimum) (minimum, maximum) = (maximum, minimum);
            if (range.UseLogScale && minimum > 0 && maximum > 0 && current > 0)
            {
                minimum = Math.Log(minimum);
                maximum = Math.Log(maximum);
                current = Math.Log(current);
            }
            if (maximum <= minimum) return 0;
            return Math.Max(0, Math.Min(1, (current - minimum) / (maximum - minimum)));
        }
        catch (Exception)
        {
            return 0;
        }
    }

    private string ComputeEvaluationPolicyHash()
    {
        if (CrossValidationOptions is null)
            return EvolutionHash.Compute("automl-single-validation-split-v1");

        var components = new List<string>
        {
            "automl-cross-validation-v1",
            CrossValidationOptions.NumberOfFolds.ToString(CultureInfo.InvariantCulture),
            CrossValidationOptions.ValidationType.ToString(),
            CrossValidationOptions.RandomSeed?.ToString(CultureInfo.InvariantCulture) ??
                "map-elites-stable-inherited-stream",
            CrossValidationOptions.ShuffleData ? "1" : "0"
        };
        foreach (MetricType metric in CrossValidationOptions.MetricsToCompute ?? Array.Empty<MetricType>())
            components.Add(metric.ToString());
        return EvolutionHash.Combine(components);
    }

    private sealed class SearchDefinition
    {
        public SearchDefinition(
            IEnumerable<Type> modelTypes,
            IReadOnlyDictionary<Type, IReadOnlyDictionary<string, ParameterRange>> searchSpaces)
        {
            ModelTypes = Array.AsReadOnly(modelTypes.ToArray());
            SearchSpaces = new ReadOnlyDictionary<Type, IReadOnlyDictionary<string, ParameterRange>>(
                searchSpaces.ToDictionary(item => item.Key, item => item.Value));
            ModelIndices = new ReadOnlyDictionary<Type, int>(ModelTypes
                .Select((type, index) => new { type, index })
                .ToDictionary(item => item.type, item => item.index));
            VersionHash = ComputeVersionHash();
        }

        public IReadOnlyList<Type> ModelTypes { get; }
        public IReadOnlyDictionary<Type, IReadOnlyDictionary<string, ParameterRange>> SearchSpaces { get; }
        public IReadOnlyDictionary<Type, int> ModelIndices { get; }
        public string VersionHash { get; }

        private string ComputeVersionHash()
        {
            var components = new List<string> { "automl-map-elites-space-v1" };
            foreach (Type modelType in ModelTypes)
            {
                components.Add(modelType.AssemblyQualifiedName ?? modelType.FullName ?? modelType.Name);
                foreach (KeyValuePair<string, ParameterRange> item in SearchSpaces[modelType]
                    .OrderBy(item => item.Key, StringComparer.Ordinal))
                {
                    ParameterRange range = item.Value;
                    components.Add(item.Key);
                    components.Add(((int)range.Type).ToString(CultureInfo.InvariantCulture));
                    components.Add(FormatValue(range.MinValue));
                    components.Add(FormatValue(range.MaxValue));
                    components.Add(range.Step?.ToString("R", CultureInfo.InvariantCulture) ?? "null");
                    components.Add(range.UseLogScale ? "1" : "0");
                    components.Add(FormatValue(range.DefaultValue));
                    if (range.CategoricalValues is not null)
                    {
                        foreach (object value in range.CategoricalValues)
                            components.Add(FormatValue(value));
                    }
                }
            }
            return EvolutionHash.Combine(components);
        }
    }

    private sealed class MapElitesGenome : IImmutableEvolutionGenome<MapElitesGenome>
    {
        private readonly ReadOnlyDictionary<string, object> _parameters;

        public MapElitesGenome(Type modelType, IReadOnlyDictionary<string, object> parameters)
        {
            ModelType = modelType ?? throw new ArgumentNullException(nameof(modelType));
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));
            var copy = new Dictionary<string, object>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, object> item in parameters)
            {
                if (string.IsNullOrWhiteSpace(item.Key))
                    throw new ArgumentException("Parameter names cannot be empty.", nameof(parameters));
                if (item.Key == ModelTypeKey)
                    throw new ArgumentException($"'{ModelTypeKey}' is stored separately from genome parameters.", nameof(parameters));
                _ = FormatValue(item.Value);
                copy.Add(item.Key, item.Value);
            }
            _parameters = new ReadOnlyDictionary<string, object>(copy);
            var identity = new List<string>
            {
                "automl-map-elites-genome-v1",
                modelType.AssemblyQualifiedName ?? modelType.FullName ?? modelType.Name
            };
            foreach (KeyValuePair<string, object> item in copy.OrderBy(item => item.Key, StringComparer.Ordinal))
            {
                identity.Add(item.Key);
                identity.Add(FormatValue(item.Value));
            }
            Id = EvolutionHash.Combine(identity);
        }

        private MapElitesGenome(MapElitesGenome source)
        {
            ModelType = source.ModelType;
            _parameters = new ReadOnlyDictionary<string, object>(
                source._parameters.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal));
            Id = source.Id;
        }

        public Type ModelType { get; }
        public IReadOnlyDictionary<string, object> Parameters => _parameters;
        public string Id { get; }

        public MapElitesGenome CreateOwnedSnapshot() => new(this);

        public Dictionary<string, object> ToTrialParameters()
        {
            var result = _parameters.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal);
            result[ModelTypeKey] = ModelType;
            return result;
        }
    }

    private sealed class MapElitesTask : IEvolutionTask<MapElitesGenome>
    {
        private readonly MapElitesAutoML<T, TInput, TOutput> _owner;
        private readonly SearchDefinition _definition;
        private readonly EvolutionOptimizationDirection _direction;
        private readonly TInput _trainInputs;
        private readonly TOutput _trainTargets;
        private readonly TInput _validationInputs;
        private readonly TOutput _validationTargets;

        public MapElitesTask(
            MapElitesAutoML<T, TInput, TOutput> owner,
            SearchDefinition definition,
            EvolutionOptimizationDirection direction,
            string dataHash,
            TInput trainInputs,
            TOutput trainTargets,
            TInput validationInputs,
            TOutput validationTargets)
        {
            _owner = owner;
            _definition = definition;
            _direction = direction;
            _trainInputs = trainInputs;
            _trainTargets = trainTargets;
            _validationInputs = validationInputs;
            _validationTargets = validationTargets;
            VersionHash = EvolutionHash.Combine(new[] { "automl-map-elites-task-v1", definition.VersionHash });
            EvaluatorVersionHash = EvolutionHash.Combine(new[]
            {
                "automl-map-elites-evaluator-v1",
                dataHash,
                owner._optimizationMetric.ToString(),
                direction.ToString(),
                owner.ComputeEvaluationPolicyHash()
            });
        }

        public string Id => "automl-map-elites";
        public string VersionHash { get; }
        public string EvaluatorVersionHash { get; }

        public ValueTask<EvolutionCanonicalGenome<MapElitesGenome>> CanonicalizeAsync(
            MapElitesGenome genome,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!_definition.SearchSpaces.TryGetValue(genome.ModelType, out IReadOnlyDictionary<string, ParameterRange>? space))
                throw new ArgumentException("The genome uses a model type outside the configured search space.", nameof(genome));
            if (genome.Parameters.Keys.Any(key => !space.ContainsKey(key)))
                throw new ArgumentException("The genome contains a parameter outside its model search space.", nameof(genome));
            var snapshot = new MapElitesGenome(genome.ModelType, genome.Parameters);
            return new ValueTask<EvolutionCanonicalGenome<MapElitesGenome>>(
                new EvolutionCanonicalGenome<MapElitesGenome>(snapshot, snapshot.Id));
        }

        public async ValueTask<EvolutionTaskResult> EvaluateAsync(
            EvolutionCandidate<MapElitesGenome> candidate,
            EvolutionEvaluationContext context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (candidate.EvaluationId != context.EvaluationId)
                throw new InvalidOperationException("Candidate and evaluation context identifiers must match.");

            MapElitesGenome genome = candidate.CanonicalGenome.Genome;
            int historyCount = _owner.GetTrialHistory().Count;
            double score = await _owner.ExecuteTrialAsync(
                genome.ModelType,
                genome.ToTrialParameters(),
                _trainInputs,
                _trainTargets,
                _validationInputs,
                _validationTargets,
                cancellationToken).ConfigureAwait(false);

            List<TrialResult> history = _owner.GetTrialHistory();
            if (history.Count <= historyCount || !history[history.Count - 1].Success ||
                double.IsNaN(score) || double.IsInfinity(score))
                return EvolutionTaskResult.Failed("automl_trial_failed", "The candidate model could not be trained and validated.");

            var descriptors = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                [ModelFamilyDescriptor] = _definition.ModelIndices[genome.ModelType],
                [ComplexityDescriptor] = ComputeComplexity(genome, _definition.SearchSpaces[genome.ModelType])
            };
            return EvolutionTaskResult.Completed(score, descriptors, _direction, costUnits: 1);
        }
    }

    private sealed class MapElitesVariation : IVariationOperator<MapElitesGenome>
    {
        private readonly SearchDefinition _definition;
        private readonly MapElitesAutoMLOptions _options;

        public MapElitesVariation(SearchDefinition definition, MapElitesAutoMLOptions options)
        {
            _definition = definition;
            _options = options;
            VersionHash = EvolutionHash.Combine(new[]
            {
                "automl-map-elites-variation-v1",
                definition.VersionHash,
                options.MutationProbability.ToString("R", CultureInfo.InvariantCulture),
                options.ExplorationProbability.ToString("R", CultureInfo.InvariantCulture)
            });
        }

        public string Id => "automl-map-elites-variation";
        public string VersionHash { get; }

        public ValueTask<MapElitesGenome> ProposeAsync(
            EvolutionVariationContext<MapElitesGenome> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            StableRandom random = context.Random;
            if (random.NextDouble() < _options.ExplorationProbability)
            {
                Type exploratoryType = _definition.ModelTypes[random.NextInt(_definition.ModelTypes.Count)];
                return new ValueTask<MapElitesGenome>(CreateRandomGenome(
                    exploratoryType,
                    _definition.SearchSpaces[exploratoryType],
                    random));
            }

            MapElitesGenome parent = context.Parent.Candidate.CanonicalGenome.Genome;
            IReadOnlyDictionary<string, ParameterRange> space = _definition.SearchSpaces[parent.ModelType];
            var adapter = new StableSystemRandom(random);
            MapElitesGenome? inspiration = context.Inspirations
                .Select(entry => entry.Candidate.CanonicalGenome.Genome)
                .Where(genome => genome.ModelType == parent.ModelType)
                .OrderBy(genome => genome.Id, StringComparer.Ordinal)
                .FirstOrDefault();

            var parameters = new Dictionary<string, object>(StringComparer.Ordinal);
            bool changed = false;
            foreach (KeyValuePair<string, ParameterRange> item in space.OrderBy(item => item.Key, StringComparer.Ordinal))
            {
                object value;
                if (parent.Parameters.TryGetValue(item.Key, out object? parentValue))
                    value = parentValue;
                else
                    value = AutoMLParameterSampler.Sample(adapter,
                        new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { [item.Key] = item.Value })[item.Key];

                if (inspiration is not null && inspiration.Parameters.TryGetValue(item.Key, out object? inspiredValue) &&
                    random.NextDouble() < 0.25)
                    value = inspiredValue;

                if (random.NextDouble() < _options.MutationProbability)
                {
                    object mutated = AutoMLParameterSampler.Sample(adapter,
                        new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { [item.Key] = item.Value })[item.Key];
                    changed |= !Equals(value, mutated);
                    value = mutated;
                }
                changed |= !parent.Parameters.TryGetValue(item.Key, out object? original) || !Equals(original, value);
                parameters.Add(item.Key, value);
            }

            if (!changed && space.Count > 0)
            {
                KeyValuePair<string, ParameterRange> forced = space
                    .OrderBy(item => item.Key, StringComparer.Ordinal)
                    .ElementAt(random.NextInt(space.Count));
                for (int attempt = 0; attempt < 8; attempt++)
                {
                    object mutated = AutoMLParameterSampler.Sample(adapter,
                        new Dictionary<string, ParameterRange>(StringComparer.Ordinal) { [forced.Key] = forced.Value })[forced.Key];
                    if (!Equals(parameters[forced.Key], mutated))
                    {
                        parameters[forced.Key] = mutated;
                        changed = true;
                        break;
                    }
                }
            }

            if (!changed && _definition.ModelTypes.Count > 1)
            {
                int current = _definition.ModelIndices[parent.ModelType];
                int offset = 1 + random.NextInt(_definition.ModelTypes.Count - 1);
                Type alternate = _definition.ModelTypes[(current + offset) % _definition.ModelTypes.Count];
                return new ValueTask<MapElitesGenome>(CreateRandomGenome(
                    alternate,
                    _definition.SearchSpaces[alternate],
                    random));
            }

            return new ValueTask<MapElitesGenome>(new MapElitesGenome(parent.ModelType, parameters));
        }
    }

    private sealed class StableSystemRandom : Random
    {
        private readonly StableRandom _random;

        public StableSystemRandom(StableRandom random)
        {
            _random = random ?? throw new ArgumentNullException(nameof(random));
        }

        protected override double Sample() => _random.NextDouble();
        public override double NextDouble() => _random.NextDouble();
        public override int Next() => _random.NextInt(int.MaxValue);
        public override int Next(int maxValue) => maxValue == 0 ? 0 : _random.NextInt(maxValue);
        public override int Next(int minValue, int maxValue) =>
            minValue == maxValue ? minValue : _random.NextInt(minValue, maxValue);

        public override void NextBytes(byte[] buffer)
        {
            if (buffer is null) throw new ArgumentNullException(nameof(buffer));
            for (int index = 0; index < buffer.Length; index++)
                buffer[index] = (byte)_random.NextUInt32();
        }
    }

}
