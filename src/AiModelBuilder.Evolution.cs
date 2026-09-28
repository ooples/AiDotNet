using AiDotNet.Agentic.Embeddings;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.ArtifactStore;
using AiDotNet.Evolution.Programs.Metrics;
using AiDotNet.Evolution.Programs.Novelty;
using AiDotNet.Evolution.Programs.Outputs;
using AiDotNet.Evolution.Programs.Provenance;
using AiDotNet.Interfaces;
using AiDotNet.Models.Results;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Validation;

namespace AiDotNet;

/// <summary>
/// Evolution extensions for AiModelBuilder.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>The number of bins the default program-length behaviour axis is divided into.</summary>
    private const int DefaultProgramLengthBins = 16;

    /// <summary>How much larger than the longest seed the default program-length axis reaches.</summary>
    private const int DefaultProgramLengthGrowthFactor = 4;

    /// <summary>The narrowest the default program-length axis may be, in characters.</summary>
    private const int MinimumProgramLengthAxisSpan = 512;

    private EvolutionOptions? _evolutionOptions;
    private EvolutionSeedOptions? _evolutionSeedOptions;

    /// <summary>
    /// The whole typed-genome run, captured while <c>TGenome</c> is still in scope.
    /// </summary>
    /// <remarks>
    /// The genome type is a parameter of <c>ConfigureEvolution</c>, not of this builder, so the components cannot be
    /// stored in strongly-typed fields. Capturing the run in a delegate at configure time is what lets the build
    /// path drive it later without reflection: no <c>MakeGenericMethod</c>, no boxing of the components, and a
    /// compile-time check that the task, the operator and the codec all agree on one genome type. The same shape is
    /// used by <c>_imageDataLoader</c> for a loader whose type parameters do not match the builder's.
    /// </remarks>
    private Func<CancellationToken, Task<EvolutionRunOutcome>>? _evolutionRunner;

    /// <summary>The seed genomes for a typed run, held as <see cref="object"/> for the same reason as the runner.</summary>
    private object? _evolutionSeeds;

    private ProgramEvolutionOptions? _programEvolutionOptions;
    private IEmbeddingClient? _embeddingClient;
    private IChatClient<T>? _chatClient;
    private ChatClientOptions? _chatClientOptions;
    private ProgramSandboxOptions? _programSandboxOptions;
    private IProgramExecutionEngine? _programExecutionEngine;

    /// <summary>
    /// Configures the settings an evolution run uses without yet naming what is being evolved.
    /// </summary>
    /// <param name="options">Budgets, islands, the archive grid, persistence, and tracing.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This is the settings-only entry point, and the one a YAML configuration file targets. Pair it with
    /// <see cref="ConfigureProgramEvolution"/> to evolve source code, or use the
    /// <c>ConfigureEvolution&lt;TGenome&gt;</c> overloads to evolve a candidate type of your own — those accept the
    /// same options object as an argument, so you rarely need to call both.
    /// </para>
    /// <para>
    /// The options are validated and copied here, so a mistake is reported on this line rather than after the first
    /// evaluation, and changing the object afterwards cannot alter the run.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "how long and how hard to search" step.
    /// <code>
    /// var result = await new AiModelBuilder&lt;double, double[], double&gt;()
    ///     .ConfigureEvolution(new EvolutionOptions { MaxEvaluationAttempts = 200, IslandCount = 4 })
    ///     .ConfigureChatClient(myChatClient)
    ///     .ConfigureProgramEvolution(programOptions)
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">A setting is invalid or two descriptors share a name.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEvolution(EvolutionOptions options)
    {
        Guard.NotNull(options);
        _evolutionOptions = options.SnapshotAndValidate();
        return this;
    }

    /// <summary>
    /// Configures evolution of a candidate type of your own, without checkpointing.
    /// </summary>
    /// <typeparam name="TGenome">The immutable candidate type being evolved.</typeparam>
    /// <param name="task">Canonicalizes and scores a candidate.</param>
    /// <param name="variation">Proposes a child from a parent and its inspirations.</param>
    /// <param name="options">Run settings; when <see langword="null"/> the settings from a previous
    /// <see cref="ConfigureEvolution(EvolutionOptions)"/> call are used, or the defaults.</param>
    /// <param name="selection">Optional parent and inspiration policy; overrides <c>EvolutionOptions.SelectionPolicy</c>.</param>
    /// <param name="refiner">Optional inner optimizer applied to a proposal before it is scored.</param>
    /// <param name="migration">Optional island migration policy; the default exchanges elites around a ring.</param>
    /// <param name="observer">Optional observer for run events, in addition to the trace writer.</param>
    /// <param name="genomeDistance">
    /// Optional structural distance between two candidates. Supply one whenever
    /// <see cref="EvolutionOptions.NoveltyDistanceThreshold"/> is positive; without it the engine refuses the run,
    /// because a novelty gate has no way to tell two candidates apart.
    /// </param>
    /// <param name="archiveFactory">
    /// Optional factory for a distinct empty archive per island. When supplied, it replaces the MAP-Elites archive
    /// described by <see cref="EvolutionOptions.Descriptors"/>.
    /// </param>
    /// <param name="winnerModelFactory">
    /// Optional typed adapter that turns the best genome into the model carried by the built result. When omitted,
    /// the result remains genome-only.
    /// </param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This overload cannot checkpoint, because writing a checkpoint means serializing genomes and only an
    /// <see cref="IEvolutionGenomeCodec{TGenome}"/> knows how. Asking for <c>Resume</c>, a positive
    /// <c>CheckpointInterval</c>, or a <c>CheckpointDirectory</c> here is rejected on this line rather than
    /// silently ignored; use the overload that takes a codec instead.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when what you are evolving is not source code — a schedule, a layout, a
    /// set of rules. You supply two things: a task that says how good a candidate is, and a variation operator that
    /// says how to make a new candidate from an existing one. Seed the search with
    /// <see cref="ConfigureEvolutionSeeds{TGenome}"/> and read the winner from
    /// <c>AiModelResult.GetEvolutionRunResult&lt;TGenome&gt;()</c>.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="task"/> or <paramref name="variation"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// The options request checkpointing or resume, or they configure no archive descriptors.
    /// </exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEvolution<TGenome>(
        IEvolutionTask<TGenome> task,
        IVariationOperator<TGenome> variation,
        EvolutionOptions? options = null,
        ISelectionPolicy<TGenome>? selection = null,
        ICandidateRefiner<TGenome>? refiner = null,
        IMigrationPolicy<TGenome>? migration = null,
        IEvolutionObserver<TGenome>? observer = null,
        IGenomeDistance<TGenome>? genomeDistance = null,
        Func<int, IEvolutionArchive<TGenome>>? archiveFactory = null,
        Func<TGenome, IFullModel<T, TInput, TOutput>>? winnerModelFactory = null)
    {
        Guard.NotNull(task);
        Guard.NotNull(variation);
        EvolutionOptions effective = ResolveTypedEvolutionOptions(options, archiveFactory is not null);
        if (effective.Resume || effective.CheckpointInterval > 0 || effective.CheckpointDirectory is not null)
        {
            throw new ArgumentException(
                "Checkpointing and resume need an IEvolutionGenomeCodec<TGenome> so genomes can be written to and " +
                "read back from a checkpoint. Use the ConfigureEvolution overload that takes a genome codec, or " +
                "clear Resume, CheckpointInterval and CheckpointDirectory on the options.",
                nameof(options));
        }

        _evolutionOptions = effective;
        _evolutionRunner = cancellationToken => RunTypedEvolutionAsync(
            task, variation, null, selection, refiner, migration, observer, null, genomeDistance,
            archiveFactory, winnerModelFactory, cancellationToken);
        return this;
    }

    /// <summary>
    /// Configures evolution of a candidate type of your own, with checkpointing and resume available.
    /// </summary>
    /// <typeparam name="TGenome">The immutable candidate type being evolved.</typeparam>
    /// <param name="task">Canonicalizes and scores a candidate.</param>
    /// <param name="variation">Proposes a child from a parent and its inspirations.</param>
    /// <param name="genomeCodec">Serializes a genome to text and back, which is what makes a checkpoint possible.</param>
    /// <param name="options">Run settings; when <see langword="null"/> the settings from a previous
    /// <see cref="ConfigureEvolution(EvolutionOptions)"/> call are used, or the defaults.</param>
    /// <param name="selection">Optional parent and inspiration policy; overrides <c>EvolutionOptions.SelectionPolicy</c>.</param>
    /// <param name="refiner">Optional inner optimizer applied to a proposal before it is scored.</param>
    /// <param name="migration">Optional island migration policy; the default exchanges elites around a ring.</param>
    /// <param name="observer">Optional observer for run events, in addition to the trace writer.</param>
    /// <param name="checkpointStore">
    /// Optional store; when <see langword="null"/> and the options ask for checkpointing, a JSON file store is
    /// created under the run's output directory.
    /// </param>
    /// <param name="genomeDistance">
    /// Optional structural distance between two candidates. Supply one whenever
    /// <see cref="EvolutionOptions.NoveltyDistanceThreshold"/> is positive; without it the engine refuses the run,
    /// because a novelty gate has no way to tell two candidates apart.
    /// </param>
    /// <param name="archiveFactory">
    /// Optional factory for a distinct empty archive per island. When supplied, it replaces the MAP-Elites archive
    /// described by <see cref="EvolutionOptions.Descriptors"/>.
    /// </param>
    /// <param name="winnerModelFactory">
    /// Optional typed adapter that turns the best genome into the model carried by the built result. When omitted,
    /// the result remains genome-only.
    /// </param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// A checkpoint can only be resumed by a run whose compatibility hash matches, which covers the task, the
    /// operator, the codec, the archive definition and every option that can change the search. Resuming with a
    /// changed configuration is refused rather than silently producing a different run.
    /// </para>
    /// <para><b>For Beginners:</b> Use this overload for long runs. Set <c>CheckpointInterval</c> so progress is
    /// saved as it goes, and <c>Resume</c> to pick up where a previous run stopped. The codec is the small adapter
    /// that turns one of your candidates into text and back — for evolving programs, the library ships one.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="task"/>, <paramref name="variation"/>, or <paramref name="genomeCodec"/> is <see langword="null"/>.
    /// </exception>
    /// <exception cref="ArgumentException">The options configure no archive descriptors.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEvolution<TGenome>(
        IEvolutionTask<TGenome> task,
        IVariationOperator<TGenome> variation,
        IEvolutionGenomeCodec<TGenome> genomeCodec,
        EvolutionOptions? options = null,
        ISelectionPolicy<TGenome>? selection = null,
        ICandidateRefiner<TGenome>? refiner = null,
        IMigrationPolicy<TGenome>? migration = null,
        IEvolutionObserver<TGenome>? observer = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        IGenomeDistance<TGenome>? genomeDistance = null,
        Func<int, IEvolutionArchive<TGenome>>? archiveFactory = null,
        Func<TGenome, IFullModel<T, TInput, TOutput>>? winnerModelFactory = null)
    {
        Guard.NotNull(task);
        Guard.NotNull(variation);
        Guard.NotNull(genomeCodec);
        _evolutionOptions = ResolveTypedEvolutionOptions(options, archiveFactory is not null);
        _evolutionRunner = cancellationToken => RunTypedEvolutionAsync(
            task, variation, genomeCodec, selection, refiner, migration, observer, checkpointStore, genomeDistance,
            archiveFactory, winnerModelFactory, cancellationToken);
        return this;
    }

    /// <summary>
    /// Configures the programs a program-evolution run starts from, as plain source text.
    /// </summary>
    /// <param name="options">The seed sources, one per starting candidate.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// These are added ahead of anything already on <c>ProgramEvolutionOptions.SeedPrograms</c>, so a configuration
    /// file can extend a program's built-in seeds instead of restating them.
    /// </para>
    /// <para><b>For Beginners:</b> Give the search a working starting program, even a slow one — it improves what it
    /// is given far faster than it invents something from nothing.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">A seed source is empty or white space.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEvolutionSeeds(EvolutionSeedOptions options)
    {
        Guard.NotNull(options);
        _evolutionSeedOptions = options.SnapshotAndValidate();
        return this;
    }

    /// <summary>
    /// Configures the candidates a typed evolution run starts from.
    /// </summary>
    /// <typeparam name="TGenome">The candidate type; it must match the one given to <c>ConfigureEvolution</c>.</typeparam>
    /// <param name="seeds">The starting candidates, consumed in order before any variation proposal.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// The sequence is materialized here, so a lazy enumerator is not re-enumerated during the run and cannot yield
    /// different candidates the second time. Order matters: seeds are evaluated first, in the order given, which is
    /// part of what makes a run with a fixed seed reproducible.
    /// </para>
    /// <para><b>For Beginners:</b> Call this with one or more starting candidates. Several genuinely different
    /// starting points are better than several similar ones, because the archive keeps them apart and develops each.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="seeds"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="seeds"/> contains a <see langword="null"/> entry.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEvolutionSeeds<TGenome>(IEnumerable<TGenome> seeds)
    {
        Guard.NotNull(seeds);
        var materialized = new List<TGenome>();
        foreach (TGenome seed in seeds)
        {
            if (seed is null) throw new ArgumentException("Seeds cannot contain a null entry.", nameof(seeds));
            materialized.Add(seed);
        }

        _evolutionSeeds = materialized;
        return this;
    }

    /// <summary>
    /// Configures evolution of source code: the OpenEvolve-equivalent entry point.
    /// </summary>
    /// <param name="options">Seeds, bounds, prompting, sandboxing, and how a candidate program is scored.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// A program-evolution run needs three things beyond these options: a chat client to propose edits
    /// (<see cref="ConfigureChatClient"/>), a way to score a candidate — either <c>TestCases</c> or an
    /// <c>EvaluatorScript</c> on the options — and somewhere to run it, which defaults to an out-of-process sandbox
    /// unless <see cref="ConfigureProgramExecutionEngine"/> supplies one. Run settings come from
    /// <see cref="ConfigureEvolution(EvolutionOptions)"/> when it was called and from
    /// <c>ProgramEvolutionOptions.Engine</c> otherwise.
    /// </para>
    /// <para>
    /// When no archive descriptors were configured the builder maps one documented behaviour axis — program length
    /// in sixteen clamped bins, spanning four times the longest seed program and at least 512 characters — because
    /// that axis needs no domain knowledge and an archive with no axes keeps only a single overall winner.
    /// </para>
    /// <para><b>For Beginners:</b> This is the one-call equivalent of running OpenEvolve.
    /// <code>
    /// var options = new ProgramEvolutionOptions { Language = ProgramLanguage.Python, TaskDescription = "Sort faster." };
    /// options.SeedPrograms.Add(File.ReadAllText("solution.py"));
    ///
    /// var result = await new AiModelBuilder&lt;double, double[], double&gt;()
    ///     .ConfigureChatClient(myChatClient)
    ///     .ConfigureProgramEvolution(options)
    ///     .BuildAsync();
    ///
    /// string? best = result.ProgramEvolution?.BestProgram?.Source;
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">A setting is invalid, or a seed program is blank or oversized.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureProgramEvolution(ProgramEvolutionOptions options)
    {
        Guard.NotNull(options);
        options.Validate();
        _programEvolutionOptions = options.Clone();
        return this;
    }

    /// <summary>
    /// Configures the language model that proposes program edits.
    /// </summary>
    /// <param name="chatClient">The client that talks to the model.</param>
    /// <param name="options">Optional retry policy, timeout, filters, and recording settings.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// The client is wrapped according to <paramref name="options"/> before it is used, so retries, a per-call
    /// timeout, any filters, and recording apply to every call the run makes. Setting
    /// <c>ChatClientOptions.RecordingMode</c> to <c>Record</c> once and to <c>Replay</c> afterwards is what makes an
    /// experiment repeatable offline and at no cost.
    /// </para>
    /// <para><b>For Beginners:</b> This is where you plug in your AI model. Everything about how to call it — how
    /// many times to retry, how long to wait, whether to save the answers — lives in the options, so the rest of the
    /// configuration does not change when you switch models.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="chatClient"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">A retry, timeout, or recording setting is invalid.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureChatClient(
        IChatClient<T> chatClient,
        ChatClientOptions? options = null)
    {
        Guard.NotNull(chatClient);
        options?.Validate();
        _chatClientOptions = options?.Clone();
        _chatClient = chatClient;
        return this;
    }

    /// <summary>
    /// Configures the embedding model used to spot candidates that are near-duplicates of ones already tried.
    /// </summary>
    /// <param name="embeddingClient">The client that turns program text into vectors.</param>
    /// <param name="cacheCapacity">
    /// How many embeddings to remember so a repeated candidate costs nothing; zero disables the cache.
    /// </param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Optional, and consulted only when <c>ProgramEvolutionOptions.Novelty</c> is set. Even then the cheap
    /// structural comparison runs first and settles most candidates without a single call, so this is paid for only
    /// where the structural rung was inconclusive. The client is wrapped in a content-keyed cache, so the same
    /// program is never embedded twice.
    /// </para>
    /// <para><b>For Beginners:</b> An embedding turns code into a list of numbers so two programs can be compared
    /// for meaning rather than for exact text. You only need this if you want duplicate detection that catches a
    /// rewrite which does the same thing in different words.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="embeddingClient"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="cacheCapacity"/> is negative.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEmbeddingClient(
        IEmbeddingClient embeddingClient,
        int cacheCapacity = CachingEmbeddingClient.DefaultCapacity)
    {
        Guard.NotNull(embeddingClient);
        if (cacheCapacity < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(cacheCapacity), cacheCapacity, "Value cannot be negative.");
        }

        _embeddingClient = cacheCapacity == 0
            ? embeddingClient
            : new CachingEmbeddingClient(embeddingClient, cacheCapacity);
        return this;
    }

    /// <summary>
    /// Configures several language models as one weighted ensemble.
    /// </summary>
    /// <param name="clients">The member clients; each call picks one of them.</param>
    /// <param name="weights">
    /// Optional relative weights, one per client and in the same order. When <see langword="null"/> every member is
    /// equally likely.
    /// </param>
    /// <param name="ensembleOptions">Optional selection seed, fallback behaviour, and ensemble model identifier.</param>
    /// <param name="options">Optional retry policy, timeout, filters, and recording applied around the ensemble.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Selection is seeded, so the same run picks the same members in the same order — an ensemble does not cost you
    /// reproducibility. With <c>FallbackOnError</c> left on, a member that fails is skipped and another answers, so
    /// one flaky provider does not end a long run.
    /// </para>
    /// <para><b>For Beginners:</b> Use this to mix a strong expensive model with a cheap fast one. Give the
    /// expensive one a smaller weight and most calls go to the cheap model, with the strong one contributing often
    /// enough to matter.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="clients"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// The list is empty or holds a <see langword="null"/>, or the weight count does not match the client count.
    /// </exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureChatClientEnsemble(
        IReadOnlyList<IChatClient<T>> clients,
        IReadOnlyList<double>? weights = null,
        WeightedEnsembleChatClientOptions? ensembleOptions = null,
        ChatClientOptions? options = null)
    {
        Guard.NotNull(clients);
        if (clients.Count == 0)
            throw new ArgumentException("An ensemble needs at least one chat client.", nameof(clients));
        if (weights is not null && weights.Count != clients.Count)
            throw new ArgumentException(
                "Supply one weight per client, in the same order, or no weights at all.", nameof(weights));

        var members = new List<ChatClientEnsembleMember<T>>();
        for (int index = 0; index < clients.Count; index++)
        {
            IChatClient<T> client = clients[index];
            if (client is null) throw new ArgumentException("An ensemble member cannot be null.", nameof(clients));
            members.Add(new ChatClientEnsembleMember<T>(client, weights is null ? 1.0 : weights[index]));
        }

        options?.Validate();
        _chatClientOptions = options?.Clone();
        _chatClient = new WeightedEnsembleChatClient<T>(members, ensembleOptions);
        return this;
    }

    /// <summary>
    /// Configures the execution boundary and resource limits applied to generated programs.
    /// </summary>
    /// <param name="options">The sandbox mode, limits, and interpreters; <see langword="null"/> uses the defaults.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Generated programs are untrusted code. The default mode runs each one in a separate short-lived process with
    /// a time limit, a memory limit, and bounded output, and the same limits apply to a caller-supplied evaluator
    /// script. These options are ignored when <see cref="ConfigureProgramExecutionEngine"/> supplies an engine of
    /// your own, because that engine then owns the boundary.
    /// </para>
    /// <para><b>For Beginners:</b> The most useful setting is <c>Limits.TimeLimitSeconds</c>: a generated program
    /// with an accidental infinite loop is normal, and the limit is what keeps it from stalling the run.</para>
    /// </remarks>
    /// <exception cref="ArgumentException">A limit or interpreter specification is invalid.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureProgramSandbox(ProgramSandboxOptions? options = null)
    {
        ProgramSandboxOptions resolved = (options ?? new ProgramSandboxOptions()).Clone();
        resolved.Validate();
        _programSandboxOptions = resolved;
        return this;
    }

    /// <summary>
    /// Configures the engine that runs generated programs, replacing the built-in sandbox.
    /// </summary>
    /// <param name="engine">The execution engine to use.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Supply an engine when the built-in process sandbox is not the right boundary — a container, a remote worker,
    /// or a test double. The engine you pass owns isolation and resource limits from then on, so the settings from
    /// <see cref="ConfigureProgramSandbox"/> no longer apply.
    /// </para>
    /// <para><b>For Beginners:</b> Most users never call this. It exists so a test can run programs without
    /// launching real processes, and so a hardened deployment can route execution somewhere it controls.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is <see langword="null"/>.</exception>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureProgramExecutionEngine(IProgramExecutionEngine engine)
    {
        Guard.NotNull(engine);
        _programExecutionEngine = engine;
        return this;
    }

    /// <summary>Resolves the options a typed run should use and enforces the archive requirement.</summary>
    /// <param name="options">The options passed to the overload, or <see langword="null"/> to reuse the configured ones.</param>
    /// <returns>A validated copy.</returns>
    private EvolutionOptions ResolveTypedEvolutionOptions(EvolutionOptions? options, bool hasCustomArchiveFactory)
    {
        EvolutionOptions effective = options is null
            ? _evolutionOptions ?? new EvolutionOptions().SnapshotAndValidate()
            : options.SnapshotAndValidate();

        if (!hasCustomArchiveFactory && effective.Descriptors.Count == 0)
        {
            throw new ArgumentException(
                "An evolution archive needs at least one behaviour axis. Add an EvolutionDescriptorDefinition to " +
                "EvolutionOptions.Descriptors naming a descriptor your task returns, the range it spans, and how " +
                "many bins that range is divided into.",
                nameof(options));
        }

        return effective;
    }

    /// <summary>Runs a typed-genome evolution to completion and projects the result.</summary>
    /// <typeparam name="TGenome">The candidate type being evolved.</typeparam>
    /// <param name="task">The configured task.</param>
    /// <param name="variation">The configured variation operator.</param>
    /// <param name="genomeCodec">The codec, or <see langword="null"/> when checkpointing was not requested.</param>
    /// <param name="selection">The caller's selection policy, or <see langword="null"/> to build one from the options.</param>
    /// <param name="refiner">The caller's refiner, or <see langword="null"/>.</param>
    /// <param name="migration">The caller's migration policy, or <see langword="null"/> for the ring default.</param>
    /// <param name="observer">The caller's observer, or <see langword="null"/>.</param>
    /// <param name="checkpointStore">The caller's checkpoint store, or <see langword="null"/> to derive one.</param>
    /// <param name="genomeDistance">The caller's structural distance metric, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Propagated into proposals, refinement, and evaluation.</param>
    /// <returns>The redacted summary and the engine's own typed result.</returns>
    private async Task<EvolutionRunOutcome> RunTypedEvolutionAsync<TGenome>(
        IEvolutionTask<TGenome> task,
        IVariationOperator<TGenome> variation,
        IEvolutionGenomeCodec<TGenome>? genomeCodec,
        ISelectionPolicy<TGenome>? selection,
        ICandidateRefiner<TGenome>? refiner,
        IMigrationPolicy<TGenome>? migration,
        IEvolutionObserver<TGenome>? observer,
        IEvolutionCheckpointStore? checkpointStore,
        IGenomeDistance<TGenome>? genomeDistance,
        Func<int, IEvolutionArchive<TGenome>>? archiveFactory,
        Func<TGenome, IFullModel<T, TInput, TOutput>>? winnerModelFactory,
        CancellationToken cancellationToken)
    {
        EvolutionOptions effective = _evolutionOptions ?? new EvolutionOptions().SnapshotAndValidate();
        IReadOnlyList<TGenome> seeds = ResolveTypedSeeds<TGenome>();
        return await RunEvolutionAsync(
            effective, task, variation, genomeCodec, selection, refiner, migration, observer, checkpointStore,
            genomeDistance, archiveFactory, winnerModelFactory, seeds, EvolutionRunSummary.DefaultEliteCount,
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Casts the configured seeds to the genome type this run uses.</summary>
    /// <typeparam name="TGenome">The candidate type being evolved.</typeparam>
    /// <returns>The seeds, or an empty list when none were configured.</returns>
    /// <exception cref="InvalidOperationException">The configured seeds are of a different genome type.</exception>
    private IReadOnlyList<TGenome> ResolveTypedSeeds<TGenome>()
    {
        if (_evolutionSeeds is null) return new List<TGenome>();
        if (_evolutionSeeds is IReadOnlyList<TGenome> typed) return typed;
        throw new InvalidOperationException(
            "ConfigureEvolutionSeeds was called with a different genome type than ConfigureEvolution. Both must " +
            $"use the same type; this run evolves '{typeof(TGenome).Name}'.");
    }

    /// <summary>Builds the engine, runs it, and turns the result into a redacted summary.</summary>
    /// <typeparam name="TGenome">The candidate type being evolved.</typeparam>
    /// <param name="options">The validated run settings.</param>
    /// <param name="task">The task that scores a candidate.</param>
    /// <param name="variation">The operator that proposes a child.</param>
    /// <param name="genomeCodec">The codec, or <see langword="null"/> when checkpointing was not requested.</param>
    /// <param name="selection">The caller's selection policy, or <see langword="null"/>.</param>
    /// <param name="refiner">The caller's refiner, or <see langword="null"/>.</param>
    /// <param name="migration">The caller's migration policy, or <see langword="null"/>.</param>
    /// <param name="observer">The caller's observer, or <see langword="null"/>.</param>
    /// <param name="checkpointStore">The caller's checkpoint store, or <see langword="null"/> to derive one.</param>
    /// <param name="genomeDistance">
    /// The caller's structural distance metric, or <see langword="null"/>. Required by the engine whenever
    /// <see cref="EvolutionOptions.NoveltyDistanceThreshold"/> is positive.
    /// </param>
    /// <param name="seeds">The starting candidates.</param>
    /// <param name="maxElites">How many elites the summary retains.</param>
    /// <param name="cancellationToken">Propagated into the run.</param>
    /// <returns>The redacted summary and the engine's own typed result.</returns>
    private static async Task<EvolutionRunOutcome> RunEvolutionAsync<TGenome>(
        EvolutionOptions options,
        IEvolutionTask<TGenome> task,
        IVariationOperator<TGenome> variation,
        IEvolutionGenomeCodec<TGenome>? genomeCodec,
        ISelectionPolicy<TGenome>? selection,
        ICandidateRefiner<TGenome>? refiner,
        IMigrationPolicy<TGenome>? migration,
        IEvolutionObserver<TGenome>? observer,
        IEvolutionCheckpointStore? checkpointStore,
        IGenomeDistance<TGenome>? genomeDistance,
        Func<int, IEvolutionArchive<TGenome>>? archiveFactory,
        Func<TGenome, IFullModel<T, TInput, TOutput>>? winnerModelFactory,
        IReadOnlyList<TGenome> seeds,
        int maxElites,
        CancellationToken cancellationToken)
    {
        EvolutionRunLocations locations = ResolveEvolutionLocations(options);
        EvolutionTraceObserver<TGenome>? tracer = null;
        try
        {
            if (locations.TracePath is { } tracePath)
            {
                tracer = new EvolutionTraceObserver<TGenome>(
                    options.CreateTraceOptions(tracePath),
                    options.RunId,
                    new List<EvolutionDescriptorDefinition>(options.Descriptors));
            }

            IEvolutionObserver<TGenome>? effectiveObserver = tracer is null
                ? observer
                : observer is null ? tracer : new FanOutEvolutionObserver<TGenome>(observer, tracer);

            IEvolutionCheckpointStore? store = checkpointStore;
            if (store is null && genomeCodec is not null && locations.CheckpointPath is { } checkpointPath)
            {
                store = new JsonEvolutionCheckpointStore(checkpointPath);
            }

            var engine = new EvolutionEngine<TGenome>(
                task,
                variation,
                archiveFactory ?? (_ => options.CreateArchive<TGenome>()),
                options.ToEngineOptions(),
                selection,
                refiner,
                migration,
                effectiveObserver,
                store,
                genomeCodec,
                genomeDistance);

            DateTimeOffset started = DateTimeOffset.UtcNow;
            EvolutionRunResult<TGenome> run = await engine.RunAsync(seeds, cancellationToken).ConfigureAwait(false);
            DateTimeOffset finished = DateTimeOffset.UtcNow;
            IFullModel<T, TInput, TOutput>? winningModel = null;
            if (winnerModelFactory is not null)
            {
                EvolutionArchiveEntry<TGenome> winner = run.Best ?? throw new InvalidOperationException(
                    "The winner model factory cannot run because evolution completed without an archived genome.");
                winningModel = winnerModelFactory(winner.Candidate.CanonicalGenome.Genome)
                    ?? throw new InvalidOperationException("The winner model factory returned null.");
            }

            EvolutionRunSummary summary = EvolutionRunSummary.Create(
                options.RunId, engine.CompatibilityHash, run, started, finished, maxElites);
            summary.OutputDirectory = locations.OutputDirectory;
            summary.CheckpointPath = store is null ? null : locations.CheckpointPath;
            if (tracer is not null)
            {
                tracer.Flush();
                summary.TracePath = tracer.TracePath;
                summary.TraceRecordCount = tracer.Summary.RecordsWritten;
            }

            return new EvolutionRunOutcome(summary, run, winningModel: winningModel);
        }
        finally
        {
            tracer?.Dispose();
            CleanUpDerivedOutput(options, locations);
        }
    }

    /// <summary>Runs a program-evolution search assembled from the configured options.</summary>
    /// <param name="cancellationToken">Propagated into proposals, refinement, and evaluation.</param>
    /// <returns>The redacted summary, the program result, and the engine's own typed result.</returns>
    /// <exception cref="InvalidOperationException">No chat client, or no way to score a candidate, was configured.</exception>
    private async Task<EvolutionRunOutcome> RunProgramEvolutionAsync(CancellationToken cancellationToken)
    {
        ProgramEvolutionOptions programOptions = _programEvolutionOptions
            ?? throw new InvalidOperationException("ConfigureProgramEvolution has not been called.");

        IChatClient<T> configuredClient = _chatClient
            ?? throw new InvalidOperationException(
                "Program evolution proposes edits with a language model, so it needs a chat client. Call " +
                "ConfigureChatClient(...) or ConfigureChatClientEnsemble(...) before BuildAsync().");

        if (_evolutionSeedOptions is { } seedOptions)
        {
            for (int index = 0; index < seedOptions.ProgramSources.Count; index++)
            {
                programOptions.SeedPrograms.Insert(index, seedOptions.ProgramSources[index]);
            }
        }

        if (programOptions.SeedPrograms.Count == 0)
        {
            throw new InvalidOperationException(
                "Program evolution starts from at least one program. Add one to " +
                "ProgramEvolutionOptions.SeedPrograms or call ConfigureEvolutionSeeds(...).");
        }

        EvolutionOptions options = ResolveProgramEvolutionOptions(programOptions);
        ProcessProgramExecutionEngine? ownedEngine = null;

        // The provenance sink buffers records, so it is owned here and disposed in the finally below. Leaving that
        // to the garbage collector loses whatever had not reached its flush threshold, which is the whole audit
        // trail on a short run.
        JsonLinesProposalProvenanceSink? provenanceSink = null;
        try
        {
            IProgramFitnessEvaluator evaluator = CreateProgramEvaluator(programOptions, out ownedEngine);

            // Duplicate rejection. The structural rung costs no network call and no model, so it is the metric a
            // program run gets by default; an embedding rung is added only when a client was supplied to score the
            // candidates the structural rung could not settle.
            IGenomeDistance<ProgramGenome>? genomeDistance = null;
            if (programOptions.Novelty is { } noveltyOptions)
            {
                genomeDistance = _embeddingClient is null
                    ? new ProgramTokenSetDistance()
                    : new EmbeddingCosineGenomeDistance(_embeddingClient);
                evaluator = new NoveltyGatingProgramFitnessEvaluator(
                    evaluator,
                    new ProgramNoveltyPolicy(noveltyOptions, new ProgramTokenSetDistance(), _embeddingClient));
            }

            var task = new ProgramEvolutionTask(evaluator, programOptions.CreateDescriptorSet(), programOptions);
            IChatClient<T> client = _chatClientOptions is null
                ? configuredClient
                : ChatClientPipelineFactory.Create(configuredClient, _chatClientOptions);

            string? runRoot = programOptions.Engine.OutputDirectory;

            // Per-proposal audit trail. The sink writes beneath the run directory, bounded and redacted, and stays
            // uncreated unless the caller turned it on.
            if (programOptions.Provenance.Enabled && runRoot is not null)
            {
                provenanceSink = new JsonLinesProposalProvenanceSink(
                    Path.Combine(runRoot, "provenance"), programOptions.Provenance);
            }

            var variation = new LlmProgramVariationOperator<T>(
                client, programOptions, programOptions.Variation, "llm-program-variation", null,
                provenanceSink, programOptions.Provenance);

            // Best-program files. Without this a finished run leaves nothing on disk to open.
            ProgramRunOutputObserver? outputObserver = null;
            if (programOptions.RunOutput is { } outputOptions && runRoot is not null)
            {
                outputObserver = new ProgramRunOutputObserver(
                    new ProgramRunOutputWriter(runRoot, outputOptions));
            }

            // Artifact retention. The engine keeps artifacts only long enough to feed one follow-up proposal, so
            // without a store there is no way to ask after the run why a candidate scored as it did.
            ProgramArtifactStoreObserver? artifactObserver = null;
            if (programOptions.ArtifactStore is { } artifactOptions && runRoot is not null)
            {
                // The cadence is passed through so the retention the caller configured is actually applied during
                // the run. Without it the store knows what to keep and nothing ever asks it to.
                artifactObserver = new ProgramArtifactStoreObserver(
                    new FileSystemProgramArtifactStore(Path.Combine(runRoot, "artifacts"), artifactOptions),
                    artifactOptions.PurgeEveryStores);
            }

            // The engine takes one observer, so the two are folded together when both are configured.
            IEvolutionObserver<ProgramGenome>? programObserver = outputObserver is null
                ? artifactObserver
                : artifactObserver is null
                    ? outputObserver
                    : new FanOutEvolutionObserver<ProgramGenome>(outputObserver, artifactObserver);

            EvolutionRunOutcome outcome = await RunEvolutionAsync(
                options,
                task,
                variation,
                new ProgramGenomeCodec(),
                null,
                null,
                null,
                programObserver,
                null,
                genomeDistance,
                null,
                null,
                programOptions.CreateSeedGenomes(),
                programOptions.IncludeEliteSourceCount,
                cancellationToken).ConfigureAwait(false);

            var typedRun = (EvolutionRunResult<ProgramGenome>)outcome.RunResult;

            // The observer cannot see the archives during the run, because the engine owns them and its events carry
            // candidates rather than archive views. The final result does expose them, so the run-end write happens
            // here with the full frontier available.
            if (outputObserver is not null && programOptions.RunOutput is { WriteAtRunEnd: true })
            {
                foreach (IEvolutionArchiveView<ProgramGenome> island in typedRun.Islands)
                {
                    outputObserver.AddArchive(island);
                }

                outputObserver.WriteNow();
            }

            ProgramEvolutionResult programResult = ProgramEvolutionResult.Create(
                typedRun, programOptions, variation.GetUsage(), outcome.Summary.CheckpointPath);
            outcome.Summary.LlmUsage = programResult.LlmUsage;
            return new EvolutionRunOutcome(outcome.Summary, outcome.RunResult, programResult);
        }
        finally
        {
            // Dispose flushes whatever the sink still holds, so the audit trail survives a short run and a
            // cancelled one alike.
            provenanceSink?.Dispose();
            ownedEngine?.Dispose();
        }
    }

    /// <summary>Chooses the run settings a program-evolution run uses and fills in a default behaviour axis.</summary>
    /// <param name="programOptions">The validated program options.</param>
    /// <returns>Validated run settings with at least one archive descriptor.</returns>
    private EvolutionOptions ResolveProgramEvolutionOptions(ProgramEvolutionOptions programOptions)
    {
        EvolutionOptions options = _evolutionOptions
            ?? EvolutionOptions.FromEngineOptions(programOptions.Engine).SnapshotAndValidate();

        // Configuring novelty has to reach the engine's archive-side gate, which is the only place a near-duplicate
        // can be refused BEFORE it costs an evaluation. The engine's threshold is off by default, so without this
        // the option would be accepted, a distance metric would be supplied, and nothing would ever be rejected.
        // A caller who set the engine threshold themselves is not overridden.
        if (programOptions.Novelty is { } novelty &&
            options.NoveltyDistanceThreshold <= 0 &&
            novelty.StructuralNoveltyThreshold > 0)
        {
            options.NoveltyDistanceThreshold = novelty.StructuralNoveltyThreshold;
        }

        if (options.Descriptors.Count > 0) return options;

        // No behaviour axes were configured. Program length is the one axis every program has without any domain
        // knowledge, so map it rather than collapsing the archive to a single overall winner. The matching
        // descriptor is added to the program options too when the caller configured none, because the archive
        // definition and the value the task computes have to agree on the name.
        if (programOptions.Descriptors.Count == 0)
        {
            programOptions.Descriptors.Add(new ProgramLengthDescriptor());
            options.Descriptors.Add(new EvolutionDescriptorDefinition(
                ProgramLengthDescriptor.DefaultName,
                0,
                ResolveDefaultLengthAxisMaximum(programOptions),
                DefaultProgramLengthBins,
                EvolutionOutOfRangePolicy.Clamp));
            return options;
        }

        // The caller named the behaviours they care about but not the range each one takes, which is the ordinary
        // case: you know you want to map branching factor, not that it runs from 0.1 to 0.6 on this problem. A
        // program descriptor is a pure function of the program text, so the seeds can be measured here, for free and
        // before anything runs, and the grid derived from what they actually produced.
        foreach (EvolutionDescriptorDefinition definition in CalibrateProgramDescriptors(programOptions))
        {
            options.Descriptors.Add(definition);
        }

        return options;
    }

    /// <summary>Derives one archive axis per configured program descriptor by measuring the seed programs.</summary>
    /// <param name="programOptions">The validated program options, whose descriptors and seeds set the scale.</param>
    /// <returns>One definition per configured descriptor, in the order they were configured.</returns>
    /// <remarks>
    /// <para>
    /// Previously only the first configured descriptor became an axis, and it was given the program-length range
    /// whatever it measured, so a descriptor reporting a ratio between zero and one was mapped onto an axis spanning
    /// thousands of characters and every candidate landed in the first bin. The archive then held one overall winner
    /// and the search lost the diversity pressure it exists for, silently and with no error.
    /// </para>
    /// <para>
    /// Measuring the seeds is free because descriptors read only the genome, and it is deterministic because the
    /// seeds are a fixed, ordered list the caller already supplied. Axes grow in whole bins if the search reaches
    /// past the seeded span, so a narrow seed population costs a little growth rather than a wrong grid.
    /// </para>
    /// </remarks>
    internal static IReadOnlyList<EvolutionDescriptorDefinition> CalibrateProgramDescriptors(
        ProgramEvolutionOptions programOptions)
    {
        ProgramDescriptorSet set = programOptions.CreateDescriptorSet();
        var observations = new List<IReadOnlyDictionary<string, double>>(programOptions.SeedPrograms.Count);
        foreach (string source in programOptions.SeedPrograms)
        {
            observations.Add(set.Compute(new ProgramGenome(source, programOptions.Language)));
        }

        var calibration = new EvolutionDescriptorCalibrationOptions { BinCount = DefaultProgramLengthBins };
        return EvolutionDescriptorCalibration.FromObservations(observations, set.Names, calibration);
    }

    /// <summary>Chooses the upper bound of the default program-length behaviour axis.</summary>
    /// <param name="programOptions">The validated program options, whose seeds set the scale.</param>
    /// <returns>A bound wide enough for real growth and narrow enough for the bins to separate candidates.</returns>
    /// <remarks>
    /// Spanning the whole of <c>MaxProgramChars</c> would be useless: a hundred-character program and a program four
    /// times its size would share the first bin, so the archive would keep one overall winner and the search would
    /// lose its diversity pressure. The seeds are the only evidence available about the size a solution actually
    /// has, so the axis spans four times the longest seed — room for substantial growth — with a floor that keeps a
    /// very short seed from producing bins a single added line would overflow, and the configured
    /// <c>MaxProgramChars</c> as the ceiling.
    /// </remarks>
    private static double ResolveDefaultLengthAxisMaximum(ProgramEvolutionOptions programOptions)
    {
        int longestSeed = 0;
        foreach (string source in programOptions.SeedPrograms)
        {
            int length = ProgramGenome.Normalize(source).Length;
            if (length > longestSeed) longestSeed = length;
        }

        int span = longestSeed * DefaultProgramLengthGrowthFactor;
        if (span < MinimumProgramLengthAxisSpan) span = MinimumProgramLengthAxisSpan;
        return span > programOptions.MaxProgramChars ? programOptions.MaxProgramChars : span;
    }

    /// <summary>Builds the evaluator a program-evolution run scores candidates with.</summary>
    /// <param name="programOptions">The validated program options.</param>
    /// <param name="ownedEngine">Receives the sandbox this method created, so the caller can dispose it.</param>
    /// <returns>An evaluator over the configured test cases or evaluator script.</returns>
    /// <exception cref="InvalidOperationException">Neither test cases nor an evaluator script were configured.</exception>
    private IProgramFitnessEvaluator CreateProgramEvaluator(
        ProgramEvolutionOptions programOptions,
        out ProcessProgramExecutionEngine? ownedEngine)
    {
        ownedEngine = null;
        bool hasScript = !string.IsNullOrWhiteSpace(programOptions.EvaluatorScript);
        if (programOptions.TestCases.Count == 0 && !hasScript)
        {
            throw new InvalidOperationException(
                "Program evolution needs a way to score a candidate. Add input/output examples to " +
                "ProgramEvolutionOptions.TestCases, or set ProgramEvolutionOptions.EvaluatorScript.");
        }

        IProgramExecutionEngine? executionEngine = _programExecutionEngine;
        if (executionEngine is null)
        {
            // Two places can describe the sandbox, and quietly preferring one would leave the other configured,
            // validated, and ignored. Saying so is better than picking.
            if (_programSandboxOptions is not null && programOptions.HasExplicitSandbox)
            {
                throw new ArgumentException(
                    "The sandbox is configured twice, through ConfigureProgramSandbox and through " +
                    "ProgramEvolutionOptions.Sandbox, and the two settings differ in effect. Configure it in one " +
                    "place.",
                    nameof(programOptions));
            }

            ProgramSandboxOptions sandbox = _programSandboxOptions ?? programOptions.Sandbox;
            ownedEngine = new ProcessProgramExecutionEngine(sandbox);
            executionEngine = ownedEngine;
        }

        return hasScript
            ? new ScriptProgramFitnessEvaluator(
                executionEngine,
                programOptions.EvaluatorScript ?? string.Empty,
                programOptions.Script,
                "program-script-evaluator",

                // Without this an evaluator script written for the reference implementation, which prints a flat
                // metric dictionary and no "quality", fails outright instead of scoring.
                new ProgramMetricAggregator(programOptions.Metrics))
            : new SandboxedProgramFitnessEvaluator(executionEngine, programOptions.TestCases);
    }

    /// <summary>Works out where this run's checkpoint and trace files go.</summary>
    /// <param name="options">The validated run settings.</param>
    /// <returns>The resolved directory and file paths, and whether the directory was derived rather than named.</returns>
    private static EvolutionRunLocations ResolveEvolutionLocations(EvolutionOptions options)
    {
        string? root = options.OutputDirectory;
        bool derived = false;
        if (root is null && options.NeedsOutputLocation())
        {
            root = Path.Combine(
                Path.GetTempPath(), "aidotnet-evolve", EvolutionOutputLayout.CreateStem(options.RunId));
            derived = true;
        }

        if (root is null) return new EvolutionRunLocations(null, null, null, false);

        var layout = new EvolutionOutputLayout(root, options.RunId);
        string? checkpointPath = null;
        if (options.Resume || options.CheckpointInterval > 0 || options.CheckpointDirectory is not null)
        {
            checkpointPath = options.CheckpointDirectory is { } directory
                ? Path.Combine(directory, EvolutionOutputLayout.CreateStem(options.RunId) + ".checkpoint.json")
                : layout.CheckpointPath;
        }

        string? tracePath = null;
        if (options.Trace.Enabled)
        {
            tracePath = options.Trace.Path ?? layout.TracePath(options.Trace.Format, options.Trace.Compress);
        }

        return new EvolutionRunLocations(layout.Root, checkpointPath, tracePath, derived);
    }

    /// <summary>Removes a directory the builder derived itself when the caller asked not to retain it.</summary>
    /// <param name="options">The validated run settings.</param>
    /// <param name="locations">The resolved locations for this run.</param>
    /// <remarks>
    /// A directory the caller named is never deleted, whatever <c>RetainOutput</c> says, because it is theirs and
    /// may hold more than this run. Deletion is best-effort: a file still held open by an antivirus scanner or by a
    /// reader must not turn a finished run into a failed one.
    /// </remarks>
    private static void CleanUpDerivedOutput(EvolutionOptions options, EvolutionRunLocations locations)
    {
        if (options.RetainOutput || !locations.IsDerived || locations.OutputDirectory is not { } directory) return;
        try
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            // Leaving a temporary directory behind is a smaller problem than failing a completed run.
        }
    }

    /// <summary>Runs the configured evolution and returns its search result and optional materialized winner model.</summary>
    /// <param name="cancellationToken">The token supplied to <c>BuildAsync</c>.</param>
    /// <returns>A result whose evolution properties describe the finished run.</returns>
    /// <remarks>
    /// A typed run remains genome-only unless its configure call supplied a winner model factory. Program evolution
    /// is always genome-only because generated source is returned for review rather than silently executed as a model.
    /// </remarks>
    private async Task<AiModelResult<T, TInput, TOutput>> BuildEvolutionInternalAsync(CancellationToken cancellationToken)
    {
        EvolutionRunOutcome outcome = _evolutionRunner is not null
            ? await _evolutionRunner(cancellationToken).ConfigureAwait(false)
            : await RunProgramEvolutionAsync(cancellationToken).ConfigureAwait(false);

        var options = new AiModelResultOptions<T, TInput, TOutput>
        {
            // The result type obtains its model from BestSolution. It remains null for the default genome-only mode,
            // and carries the one model materialized from the winner when a typed factory was configured.
            OptimizationResult = new OptimizationResult<T, TInput, TOutput>
            {
                BestSolution = outcome.WinningModel
            },
            EvolutionSummary = outcome.Summary,
            ProgramEvolution = outcome.ProgramResult,
            EvolutionRunResult = outcome.RunResult,
            AllowNondeterminism = _allowNondeterminism
        };

        return new AiModelResult<T, TInput, TOutput>(options);
    }

    /// <summary>Where one run's files live, and whether the builder chose the location itself.</summary>
    private readonly struct EvolutionRunLocations
    {
        public EvolutionRunLocations(string? outputDirectory, string? checkpointPath, string? tracePath, bool isDerived)
        {
            OutputDirectory = outputDirectory;
            CheckpointPath = checkpointPath;
            TracePath = tracePath;
            IsDerived = isDerived;
        }

        public string? OutputDirectory { get; }

        public string? CheckpointPath { get; }

        public string? TracePath { get; }

        public bool IsDerived { get; }
    }

    /// <summary>What one finished run hands back to the build path.</summary>
    private sealed class EvolutionRunOutcome
    {
        public EvolutionRunOutcome(
            EvolutionRunSummary summary,
            object runResult,
            ProgramEvolutionResult? programResult = null,
            IFullModel<T, TInput, TOutput>? winningModel = null)
        {
            Summary = summary;
            RunResult = runResult;
            ProgramResult = programResult;
            WinningModel = winningModel;
        }

        public EvolutionRunSummary Summary { get; }

        /// <summary>The engine's <c>EvolutionRunResult&lt;TGenome&gt;</c>, boxed because TGenome is not in scope here.</summary>
        public object RunResult { get; }

        public ProgramEvolutionResult? ProgramResult { get; }

        public IFullModel<T, TInput, TOutput>? WinningModel { get; }
    }

    /// <summary>Delivers each run event to two observers, so a caller's observer and the trace writer coexist.</summary>
    /// <typeparam name="TGenome">The candidate type being evolved.</typeparam>
    private sealed class FanOutEvolutionObserver<TGenome> : IEvolutionObserver<TGenome>
    {
        private readonly IEvolutionObserver<TGenome> _first;
        private readonly IEvolutionObserver<TGenome> _second;

        public FanOutEvolutionObserver(IEvolutionObserver<TGenome> first, IEvolutionObserver<TGenome> second)
        {
            _first = first;
            _second = second;
        }

        /// <inheritdoc/>
        public async ValueTask OnEventAsync(
            EvolutionEvent<TGenome> evolutionEvent,
            CancellationToken cancellationToken = default)
        {
            await _first.OnEventAsync(evolutionEvent, cancellationToken).ConfigureAwait(false);
            await _second.OnEventAsync(evolutionEvent, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <inheritdoc/>
    EvolutionOptions? IConfiguredView<T, TInput, TOutput>.ConfiguredEvolution => _evolutionOptions;

    /// <inheritdoc/>
    EvolutionSeedOptions? IConfiguredView<T, TInput, TOutput>.ConfiguredEvolutionSeeds => _evolutionSeedOptions;

    /// <inheritdoc/>
    int IConfiguredView<T, TInput, TOutput>.ConfiguredEvolutionSeedCount =>
        _evolutionSeeds is System.Collections.ICollection collection ? collection.Count : 0;

    /// <inheritdoc/>
    ProgramEvolutionOptions? IConfiguredView<T, TInput, TOutput>.ConfiguredProgramEvolution => _programEvolutionOptions;

    /// <inheritdoc/>
    IChatClient<T>? IConfiguredView<T, TInput, TOutput>.ConfiguredChatClient => _chatClient;

    /// <inheritdoc/>
    ChatClientOptions? IConfiguredView<T, TInput, TOutput>.ConfiguredChatClientOptions => _chatClientOptions;

    /// <inheritdoc/>
    IEmbeddingClient? IConfiguredView<T, TInput, TOutput>.ConfiguredEmbeddingClient => _embeddingClient;

    /// <inheritdoc/>
    ProgramSandboxOptions? IConfiguredView<T, TInput, TOutput>.ConfiguredProgramSandbox => _programSandboxOptions;

    /// <inheritdoc/>
    IProgramExecutionEngine? IConfiguredView<T, TInput, TOutput>.ConfiguredProgramExecutionEngine =>
        _programExecutionEngine;

    /// <inheritdoc/>
    bool IConfiguredView<T, TInput, TOutput>.HasConfiguredEvolutionRun => _evolutionRunner is not null;
}
