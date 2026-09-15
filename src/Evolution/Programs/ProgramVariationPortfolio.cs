namespace AiDotNet.Evolution.Programs;

/// <summary>An opt-in, cost-validated adaptive program portfolio usable through ProgramEvolutionOptions.CustomVariation.</summary>
/// <remarks>Every child must provide checkpointed proposal costs in the policy's units. Supply metered mutation,
/// crossover, restart, local-refinement or consumer LLM backends without changing the facade. Child usage is summed
/// once; missing provider usage is not a zero-cost claim. This does not relax the facade's coordinated-ledger-resume
/// restriction. Instantiate fresh children for a new run, or restore both the portfolio and its matching ledger.</remarks>
public sealed class ProgramVariationPortfolio : IProgramVariationOperator,
    IOutcomeAwareVariationOperator<ProgramGenome>, IEvolutionProposalCostProvider, IProgramResourceLedgerProvider
{
    private readonly IProgramVariationOperator[] _operators;
    private readonly AdaptiveVariationPortfolio<ProgramGenome> _portfolio;

    /// <summary>Creates an explicitly cost-normalized portfolio, retaining nonzero exploration.</summary>
    public ProgramVariationPortfolio(IEnumerable<IProgramVariationOperator> operators,
        EvolutionOperatorRewardPolicy rewardPolicy, double explorationProbability = 0.1)
    {
        if (operators is null) throw new ArgumentNullException(nameof(operators));
        if (rewardPolicy is null) throw new ArgumentNullException(nameof(rewardPolicy));
        if (rewardPolicy.CostBasis != EvolutionOperatorCostBasis.ProposalAndEvaluation)
            throw new ArgumentException("Program portfolios require proposal-plus-evaluation cost credit.", nameof(rewardPolicy));
        _operators = operators.Take(257).ToArray();
        if (_operators.Length == 0 || _operators[0] is not IProgramResourceLedgerProvider first ||
            _operators.Any(op => op is not IProgramResourceLedgerProvider owner || !ReferenceEquals(first.Ledger, owner.Ledger)))
            throw new ArgumentException("Every program portfolio child must identify the same live resource ledger.", nameof(operators));
        Ledger = first.Ledger ?? throw new ArgumentException("A live shared ledger is required.", nameof(operators));
        _portfolio = new(_operators, rewardPolicy, explorationProbability);
        CostUnitVersionHash = rewardPolicy.CostUnitVersionHash;
    }

    /// <inheritdoc/>
    public string Id => _portfolio.Id;
    /// <inheritdoc/>
    public string VersionHash => _portfolio.VersionHash;
    /// <inheritdoc/>
    public string CostUnitVersionHash { get; }
    /// <inheritdoc/>
    public EvolutionResourceLedger Ledger { get; }
    /// <summary>Gets detached per-child learned statistics.</summary>
    public IReadOnlyList<EvolutionOperatorStatistics> Statistics => _portfolio.Statistics;
    /// <summary>Gets the last committed typed credit, or null before feedback and after restore.</summary>
    public EvolutionOperatorCredit? LastCredit => _portfolio.LastCredit;
    /// <summary>Gets failed diagnostic notification attempts, which never alter learning.</summary>
    public long CreditNotificationFailures => _portfolio.CreditNotificationFailures;
    /// <summary>Notifies each registered handler once per committed outcome; not replayed after restore.</summary>
    public event Action<EvolutionOperatorCredit>? CreditCommitted
    {
        add => _portfolio.CreditCommitted += value;
        remove => _portfolio.CreditCommitted -= value;
    }
    /// <inheritdoc/>
    public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default) => _portfolio.ProposeAsync(context, cancellationToken);
    /// <inheritdoc/>
    public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) => _portfolio.Observe(evaluation, insertionResult);
    /// <inheritdoc/>
    public string CaptureState() => _portfolio.CaptureState();
    /// <inheritdoc/>
    public void RestoreState(string state) => _portfolio.RestoreState(state);
    /// <inheritdoc/>
    public EvolutionProposalCost GetProposalCost(long generation) => _portfolio.GetProposalCost(generation);
    /// <inheritdoc/>
    public ProgramEvolutionLlmUsage GetUsage()
    {
        var usage = _operators.Select(op => op.GetUsage() ?? throw new InvalidOperationException("A child omitted its usage snapshot.")).ToArray();
        return new(usage.Sum(u => u.Proposals), usage.Sum(u => u.ChatCalls), usage.Sum(u => u.Retries),
            usage.Sum(u => u.AbandonedProposals), usage.Sum(u => u.ProviderErrors), usage.Sum(u => u.InputTokens), usage.Sum(u => u.OutputTokens));
    }
}
