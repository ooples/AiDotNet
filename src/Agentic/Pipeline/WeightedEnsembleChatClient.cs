using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Validation;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>Routes each chat call to one of several models, chosen by weight from a reproducible random stream.</summary>
/// <typeparam name="T">The numeric type of the member clients.</typeparam>
/// <remarks>
/// <para>
/// Mixing a strong model with a cheap one is how a long search stays affordable: the expensive model contributes
/// the occasional breakthrough and the cheap one contributes volume, in whatever ratio the weights describe. This
/// client makes that mixture a drop-in <see cref="IChatClient{T}"/>, so every layer above it — the prompt
/// builder, the variation operator, the recording pipeline — is unaware there is more than one model.
/// </para>
/// <para>
/// Three things distinguish it from the reference OpenEvolve ensemble. Selection is drawn from a seeded
/// <see cref="StableRandom"/> owned by this client, so it is reproducible and cannot be disturbed by anything
/// else in the process; upstream seeds Python's shared module-level generator from the first model's
/// configuration and shares it with the rest of the program. A member that fails hands its turn to the remaining
/// members in weight order instead of failing the iteration; upstream has no fallback at all. And
/// <see cref="GetAllResponsesAsync"/> calls every member with bounded concurrency, where upstream's
/// equivalent loops sequentially.
/// </para>
/// <para>
/// Every call is accounted for: <see cref="GetSelectionCounts"/> reports how often each member was chosen,
/// <see cref="GetUsage"/> aggregates the tokens they reported, and a response that arrives without a model id is
/// stamped with the answering member's name, so the record always says who actually answered.
/// </para>
/// <para><b>For Beginners:</b> This lets you use several AI models as if they were one. Give each model a weight
/// and it gets picked that often; if the picked model errors, the next one is tried automatically. Because the
/// picking is seeded, running your program twice picks the same models in the same order — which is what makes
/// two experiments comparable. You can also ask every model at once when you want several opinions.</para>
/// </remarks>
public sealed class WeightedEnsembleChatClient<T> : IChatClient<T>
{
    /// <summary>The model id reported when no other is configured.</summary>
    public const string DefaultModelId = "ensemble";

    private readonly object _gate = new();
    private readonly List<ChatClientEnsembleMember<T>> _members;
    private readonly double[] _cumulative;
    private readonly int[] _fallbackOrder;
    private readonly long[] _selections;
    private readonly WeightedEnsembleChatClientOptions _options;
    private readonly StableRandom _selector;
    private long _totalInputTokens;
    private long _totalOutputTokens;
    private long _totalCalls;

    /// <summary>Initializes a weighted ensemble.</summary>
    /// <param name="members">The models to route between; must contain at least one member.</param>
    /// <param name="options">Selection, fallback, and reporting settings; <c>null</c> uses the defaults.</param>
    /// <exception cref="ArgumentNullException"><paramref name="members"/> or any element is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="members"/> is empty or two members share a name.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    public WeightedEnsembleChatClient(
        IReadOnlyList<ChatClientEnsembleMember<T>> members,
        WeightedEnsembleChatClientOptions? options = null)
    {
        Guard.NotNull(members);
        if (members.Count == 0)
        {
            throw new ArgumentException("A weighted ensemble needs at least one member.", nameof(members));
        }

        WeightedEnsembleChatClientOptions optionsCopy = (options ?? new WeightedEnsembleChatClientOptions()).Clone();
        optionsCopy.Validate();

        var copy = new List<ChatClientEnsembleMember<T>>(members.Count);
        var names = new HashSet<string>(StringComparer.Ordinal);
        double total = 0;
        foreach (ChatClientEnsembleMember<T> member in members)
        {
            Guard.NotNull(member);
            if (!names.Add(member.Name))
            {
                // Duplicate names would merge two members in the statistics and in
                // the reported model id, hiding which one actually answered.
                throw new ArgumentException(
                    $"Two ensemble members share the name '{member.Name}'. Give each member a distinct name.",
                    nameof(members));
            }

            copy.Add(member);
            total += member.Weight;
        }

        if (total <= 0 || double.IsNaN(total) || double.IsInfinity(total))
        {
            // Upstream divides by this sum and raises ZeroDivisionError at the
            // first call; member weights are validated as positive, so reaching
            // here means the total overflowed.
            throw new ArgumentException("The ensemble's total member weight must be a positive, finite number.", nameof(members));
        }

        _members = copy;
        _options = optionsCopy;
        _selector = StableRandom.CreateStream(optionsCopy.Seed, optionsCopy.Stream);
        _selections = new long[copy.Count];

        _cumulative = new double[copy.Count];
        double running = 0;
        for (int index = 0; index < copy.Count; index++)
        {
            running += copy[index].Weight / total;
            _cumulative[index] = running;
        }

        // Floating-point accumulation can leave the last bound a hair under 1.0,
        // which would make a draw very close to 1.0 fall through the search.
        _cumulative[copy.Count - 1] = 1.0;

        var order = new int[copy.Count];
        for (int index = 0; index < order.Length; index++) order[index] = index;
        Array.Sort(order, (left, right) =>
        {
            int byWeight = copy[right].Weight.CompareTo(copy[left].Weight);
            return byWeight != 0 ? byWeight : left.CompareTo(right);
        });
        _fallbackOrder = order;

        ModelId = optionsCopy.ModelId ?? DefaultModelId;
    }

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>Gets the members this ensemble routes between, in declaration order.</summary>
    public IReadOnlyList<ChatClientEnsembleMember<T>> Members => _members;

    /// <summary>Gets a copy of the settings this ensemble was constructed with.</summary>
    /// <returns>An independent copy; mutating it does not affect the client.</returns>
    public WeightedEnsembleChatClientOptions GetOptions() => _options.Clone();

    /// <summary>Gets how many calls this ensemble has served.</summary>
    public long TotalCalls => Interlocked.Read(ref _totalCalls);

    /// <summary>Gets how many times each member was selected, keyed by member name.</summary>
    /// <returns>A snapshot of the selection counts.</returns>
    public IReadOnlyDictionary<string, long> GetSelectionCounts()
    {
        var counts = new Dictionary<string, long>(StringComparer.Ordinal);
        lock (_gate)
        {
            for (int index = 0; index < _members.Count; index++) counts[_members[index].Name] = _selections[index];
        }

        return counts;
    }

    /// <summary>Gets the token usage reported by every member so far, summed.</summary>
    /// <returns>The aggregated usage; counts are zero when no member reported any.</returns>
    public ChatUsage GetUsage()
    {
        long input = Interlocked.Read(ref _totalInputTokens);
        long output = Interlocked.Read(ref _totalOutputTokens);
        return new ChatUsage(ClampToInt(input), ClampToInt(output));
    }

    /// <inheritdoc/>
    public async Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        cancellationToken.ThrowIfCancellationRequested();

        int chosen = SelectMemberIndex();
        Interlocked.Increment(ref _totalCalls);

        var failures = new List<string>();
        Exception? lastError = null;
        foreach (int index in AttemptOrder(chosen))
        {
            ChatClientEnsembleMember<T> member = _members[index];
            try
            {
                ChatResponse response = await member.Client
                    .GetResponseAsync(messages, EffectiveOptions(member, options), cancellationToken)
                    .ConfigureAwait(false);
                RecordUsage(response);
                return Stamp(response, member);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
#pragma warning disable CA1031
            catch (Exception exception)
#pragma warning restore CA1031
            {
                // Only the exception's type is retained: a provider message can
                // carry an endpoint or a key fragment, and this text reaches logs.
                lastError = exception;
                failures.Add(member.Name + ": " + exception.GetType().Name);
                if (!_options.FallbackOnError) break;
            }
        }

        var message = new StringBuilder("Every ensemble member that was tried failed (");
        for (int index = 0; index < failures.Count; index++)
        {
            if (index > 0) message.Append("; ");
            message.Append(failures[index]);
        }

        message.Append(").");
        throw new InvalidOperationException(message.ToString(), lastError);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);

        // A partially consumed stream cannot be replayed against another member,
        // so streaming selects once and does not fall back.
        int chosen = SelectMemberIndex();
        Interlocked.Increment(ref _totalCalls);
        ChatClientEnsembleMember<T> member = _members[chosen];

        await foreach (ChatResponseUpdate update in member.Client
            .GetStreamingResponseAsync(messages, EffectiveOptions(member, options), cancellationToken)
            .ConfigureAwait(false))
        {
            if (update.Usage is { } usage) RecordUsage(usage);
            yield return update;
        }
    }

    /// <summary>Asks every member the same question, with bounded concurrency, and returns all answers.</summary>
    /// <param name="messages">The conversation to send.</param>
    /// <param name="options">Per-call settings layered above each member's own settings.</param>
    /// <param name="cancellationToken">Token used to cancel the calls.</param>
    /// <returns>
    /// One entry per member, in declaration order. A member that failed contributes <c>null</c> rather than
    /// failing the whole request, so a panel of judges survives one unavailable provider.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="messages"/> is <c>null</c>.</exception>
    public async Task<IReadOnlyList<ChatResponse?>> GetAllResponsesAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);

        var results = new ChatResponse?[_members.Count];
        using var limiter = new SemaphoreSlim(_options.MaxParallelism, _options.MaxParallelism);
        var running = new List<Task>(_members.Count);
        for (int index = 0; index < _members.Count; index++)
        {
            int position = index;
            running.Add(CallMemberAsync(position, messages, options, results, limiter, cancellationToken));
        }

        await Task.WhenAll(running).ConfigureAwait(false);
        return results;
    }

    private async Task CallMemberAsync(
        int index,
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options,
        ChatResponse?[] results,
        SemaphoreSlim limiter,
        CancellationToken cancellationToken)
    {
        await limiter.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ChatClientEnsembleMember<T> member = _members[index];
            Interlocked.Increment(ref _totalCalls);
            try
            {
                ChatResponse response = await member.Client
                    .GetResponseAsync(messages, EffectiveOptions(member, options), cancellationToken)
                    .ConfigureAwait(false);
                RecordUsage(response);
                results[index] = Stamp(response, member);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
#pragma warning disable CA1031
            catch (Exception)
#pragma warning restore CA1031
            {
                // One unavailable judge must not discard the answers of the others.
                results[index] = null;
            }
        }
        finally
        {
            limiter.Release();
        }
    }

    private IEnumerable<int> AttemptOrder(int chosen)
    {
        yield return chosen;
        if (!_options.FallbackOnError) yield break;
        foreach (int index in _fallbackOrder)
        {
            if (index != chosen) yield return index;
        }
    }

    private int SelectMemberIndex()
    {
        double draw;
        lock (_gate)
        {
            // One generator, so the sequence of selections for a run is a function
            // of the seed alone rather than of thread interleaving.
            draw = _selector.NextDouble();
            int selected = FindIndex(draw);
            _selections[selected]++;
            return selected;
        }
    }

    private int FindIndex(double draw)
    {
        for (int index = 0; index < _cumulative.Length; index++)
        {
            if (draw < _cumulative[index]) return index;
        }

        return _cumulative.Length - 1;
    }

    private ChatOptions? EffectiveOptions(ChatClientEnsembleMember<T> member, ChatOptions? perCall) =>
        ChatOptionsMerge.Merge(perCall, member.ChatOptions, _options.DefaultChatOptions);

    private ChatResponse Stamp(ChatResponse response, ChatClientEnsembleMember<T> member)
    {
        if (!_options.StampAnsweringMember || response.ModelId is { } existing && existing.Length > 0)
        {
            return response;
        }

        return new ChatResponse(response.Message, response.FinishReason, response.Usage, member.Name);
    }

    private void RecordUsage(ChatResponse response)
    {
        if (response.Usage is { } usage) RecordUsage(usage);
    }

    private void RecordUsage(ChatUsage usage)
    {
        Interlocked.Add(ref _totalInputTokens, usage.InputTokens);
        Interlocked.Add(ref _totalOutputTokens, usage.OutputTokens);
    }

    private static int ClampToInt(long value) => value > int.MaxValue ? int.MaxValue : (int)value;
}
