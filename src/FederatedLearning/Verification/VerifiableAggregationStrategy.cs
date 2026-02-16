using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.Verification;

/// <summary>
/// Decorator that wraps any <see cref="IAggregationStrategy{TModel}"/> with proof verification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class wraps an existing aggregation strategy (like FedAvg or Krum)
/// and adds verification: before aggregating updates, it checks cryptographic proofs from each
/// client. Only clients that pass verification are included in the aggregation.</para>
///
/// <para><b>How to use:</b></para>
/// <list type="bullet">
/// <item><description>Create your normal aggregation strategy (e.g., FedAvg).</description></item>
/// <item><description>Wrap it: <c>new VerifiableAggregationStrategy&lt;TModel&gt;(fedAvg, options)</c>.</description></item>
/// <item><description>Use the wrapper wherever you'd use the original strategy.</description></item>
/// <item><description>Clients must register proofs before aggregation via <see cref="RegisterClientProof"/>.</description></item>
/// </list>
///
/// <para><b>Integration with existing FL:</b> The existing Byzantine-robust aggregators (Krum, Bulyan,
/// Median) detect statistical anomalies. This adds cryptographic guarantees on top — first
/// verify proofs, then pass verified updates to the inner aggregator.</para>
/// </remarks>
/// <typeparam name="TModel">The type of model being aggregated.</typeparam>
public class VerifiableAggregationStrategy<TModel> : IAggregationStrategy<TModel>
{
    private readonly IAggregationStrategy<TModel> _innerStrategy;
    private readonly VerificationOptions _options;
    private readonly Dictionary<int, ClientProofBundle> _clientProofs;
    private readonly List<int> _rejectedClients;
    private int _currentRound;

    /// <summary>
    /// Gets the list of client IDs rejected in the current round.
    /// </summary>
    public IReadOnlyList<int> RejectedClients => _rejectedClients;

    /// <summary>
    /// Gets the verification options.
    /// </summary>
    public VerificationOptions Options => _options;

    /// <summary>
    /// Initializes a new instance of <see cref="VerifiableAggregationStrategy{TModel}"/>.
    /// </summary>
    /// <param name="innerStrategy">The aggregation strategy to wrap with verification.</param>
    /// <param name="options">Verification options.</param>
    public VerifiableAggregationStrategy(
        IAggregationStrategy<TModel> innerStrategy,
        VerificationOptions? options = null)
    {
        _innerStrategy = innerStrategy ?? throw new ArgumentNullException(nameof(innerStrategy));
        _options = options ?? new VerificationOptions();
        _clientProofs = new Dictionary<int, ClientProofBundle>();
        _rejectedClients = new List<int>();
        _currentRound = 0;
    }

    /// <summary>
    /// Registers a client's proof bundle for verification during aggregation.
    /// </summary>
    /// <param name="clientId">The client identifier.</param>
    /// <param name="proofBundle">The client's proofs.</param>
    public void RegisterClientProof(int clientId, ClientProofBundle proofBundle)
    {
        if (proofBundle is null)
        {
            throw new ArgumentNullException(nameof(proofBundle));
        }

        _clientProofs[clientId] = proofBundle;
    }

    /// <summary>
    /// Sets the current training round (for proof verification context).
    /// </summary>
    /// <param name="round">The round number.</param>
    public void SetRound(int round)
    {
        _currentRound = round;
        _rejectedClients.Clear();
        _clientProofs.Clear();
    }

    /// <inheritdoc/>
    public TModel Aggregate(Dictionary<int, TModel> clientModels, Dictionary<int, double> clientWeights)
    {
        if (clientModels is null)
        {
            throw new ArgumentNullException(nameof(clientModels));
        }

        if (clientWeights is null)
        {
            throw new ArgumentNullException(nameof(clientWeights));
        }

        if (_options.Level == VerificationLevel.None)
        {
            // No verification — pass through directly
            return _innerStrategy.Aggregate(clientModels, clientWeights);
        }

        // Filter clients based on proof verification
        var verifiedModels = new Dictionary<int, TModel>();
        var verifiedWeights = new Dictionary<int, double>();

        foreach (var kvp in clientModels)
        {
            int clientId = kvp.Key;

            if (!_clientProofs.ContainsKey(clientId))
            {
                if (_options.RejectFailedClients)
                {
                    _rejectedClients.Add(clientId);
                    continue;
                }
            }
            else
            {
                var bundle = _clientProofs[clientId];
                bool verified = VerifyBundle(clientId, bundle);

                if (!verified && _options.RejectFailedClients)
                {
                    _rejectedClients.Add(clientId);
                    continue;
                }
            }

            verifiedModels[clientId] = kvp.Value;
            if (clientWeights.ContainsKey(clientId))
            {
                verifiedWeights[clientId] = clientWeights[clientId];
            }
        }

        if (verifiedModels.Count == 0)
        {
            throw new InvalidOperationException(
                "No clients passed verification. Cannot aggregate empty set.");
        }

        return _innerStrategy.Aggregate(verifiedModels, verifiedWeights);
    }

    /// <inheritdoc/>
    public string GetStrategyName()
    {
        return $"Verifiable({_innerStrategy.GetStrategyName()}, Level={_options.Level})";
    }

    private bool VerifyBundle(int clientId, ClientProofBundle bundle)
    {
        // Check commitment
        if (_options.Level >= VerificationLevel.CommitmentOnly)
        {
            if (bundle.CommitmentProof is null || bundle.CommitmentProof.ProofData.Length == 0)
            {
                return false;
            }
        }

        // Check norm bound
        if (_options.Level >= VerificationLevel.NormBound)
        {
            if (bundle.NormProof is null || bundle.NormProof.ProofData.Length == 0)
            {
                return false;
            }
        }

        // Check element bound
        if (_options.Level >= VerificationLevel.ElementBound)
        {
            if (bundle.BoundednessProof is null || bundle.BoundednessProof.ProofData.Length == 0)
            {
                return false;
            }
        }

        // Check loss threshold
        if (_options.Level >= VerificationLevel.LossThreshold)
        {
            if (bundle.LossProof is null || bundle.LossProof.ProofData.Length == 0)
            {
                return false;
            }
        }

        return true;
    }
}

/// <summary>
/// Contains all proofs that a client must provide for a given verification level.
/// </summary>
public class ClientProofBundle
{
    /// <summary>Gets or sets the gradient commitment proof.</summary>
    public VerificationProof? CommitmentProof { get; set; }

    /// <summary>Gets or sets the gradient norm bound proof.</summary>
    public VerificationProof? NormProof { get; set; }

    /// <summary>Gets or sets the element boundedness proof.</summary>
    public VerificationProof? BoundednessProof { get; set; }

    /// <summary>Gets or sets the loss threshold proof.</summary>
    public VerificationProof? LossProof { get; set; }

    /// <summary>Gets or sets the computation integrity proof.</summary>
    public VerificationProof? IntegrityProof { get; set; }
}
