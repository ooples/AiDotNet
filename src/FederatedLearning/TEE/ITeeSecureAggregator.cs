using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// Performs model aggregation inside a TEE enclave boundary.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In standard federated learning, the server aggregates client model updates
/// in plaintext — the server can see every client's update. TEE-based aggregation runs inside a secure
/// enclave: clients encrypt their updates so only the enclave can decrypt them, the enclave aggregates
/// in plaintext internally, and returns the sealed result. The host OS never sees individual updates.</para>
///
/// <para><b>Flow:</b></para>
/// <list type="number">
/// <item><description>Server generates attestation quote and sends it to clients.</description></item>
/// <item><description>Clients verify the quote (proof the enclave is genuine).</description></item>
/// <item><description>Clients encrypt their updates with a key shared only with the enclave.</description></item>
/// <item><description>Enclave receives encrypted updates, decrypts inside, aggregates, seals result.</description></item>
/// <item><description>Server unseals the aggregated model inside the enclave and publishes it.</description></item>
/// </list>
///
/// <para><b>Performance:</b> 10-100x faster than homomorphic encryption because aggregation runs in
/// plaintext inside the enclave — the overhead is only encryption/decryption at ingress/egress.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public interface ITeeSecureAggregator<T>
{
    /// <summary>
    /// Initializes the aggregator for a new round.
    /// </summary>
    /// <param name="roundNumber">Current federated learning round number.</param>
    /// <param name="expectedClients">Number of clients expected this round.</param>
    void BeginRound(int roundNumber, int expectedClients);

    /// <summary>
    /// Submits an encrypted client model update to the enclave for aggregation.
    /// </summary>
    /// <param name="clientId">Unique client identifier.</param>
    /// <param name="encryptedUpdate">Client model update encrypted with the enclave's session key.</param>
    /// <param name="weight">Aggregation weight for this client (typically proportional to dataset size).</param>
    void SubmitEncryptedUpdate(int clientId, byte[] encryptedUpdate, double weight);

    /// <summary>
    /// Performs weighted aggregation of all submitted updates inside the TEE enclave.
    /// </summary>
    /// <returns>Aggregated model parameters (decrypted inside enclave, sealed for transport).</returns>
    Tensor<T> Aggregate();

    /// <summary>
    /// Gets the attestation quote for this aggregation enclave so clients can verify it.
    /// </summary>
    /// <param name="roundData">Round-specific data to bind into the quote (e.g., round number hash).</param>
    /// <returns>Raw attestation quote bytes.</returns>
    byte[] GetAttestationQuote(byte[] roundData);

    /// <summary>
    /// Generates a session key that clients use to encrypt their updates for this round.
    /// The key is sealed to the enclave and only accessible inside the TEE.
    /// </summary>
    /// <returns>Public portion of the session key (clients encrypt with this; enclave decrypts internally).</returns>
    byte[] GenerateSessionKey();

    /// <summary>
    /// Gets a value indicating whether all expected clients have submitted their updates.
    /// </summary>
    bool AllUpdatesReceived { get; }

    /// <summary>
    /// Gets the number of updates received so far this round.
    /// </summary>
    int UpdatesReceived { get; }
}
