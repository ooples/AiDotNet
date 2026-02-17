using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// Performs weighted model aggregation inside a TEE enclave boundary.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the core component that makes TEE-based federated learning work.
/// Instead of aggregating model updates in regular memory (where the server OS can see them),
/// this class runs aggregation inside a hardware-protected enclave:</para>
///
/// <list type="number">
/// <item><description>The aggregator initializes inside the TEE and generates a session key.</description></item>
/// <item><description>Clients encrypt their model updates with the session key.</description></item>
/// <item><description>The enclave decrypts updates internally (the host OS cannot see plaintext).</description></item>
/// <item><description>Weighted averaging is performed inside the enclave.</description></item>
/// <item><description>The aggregated result is returned (optionally sealed for transport).</description></item>
/// </list>
///
/// <para><b>Performance:</b> TEE aggregation is 10-100x faster than homomorphic encryption because
/// the enclave operates on plaintext internally. The only overhead is AES encryption/decryption
/// at the enclave boundary.</para>
///
/// <para><b>Security model:</b> The host OS and hypervisor cannot read or modify data inside the
/// enclave. Even a compromised server cannot extract individual client updates. Only the
/// aggregated result leaves the enclave.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class TeeSecureAggregation<T> : FederatedLearningComponentBase<T>, ITeeSecureAggregator<T>
{
    private readonly ITeeProvider<T> _provider;
    private readonly TeeOptions _options;

    private int _roundNumber;
    private int _expectedClients;
    private byte[] _sessionKey = Array.Empty<byte>();
    private byte[] _sessionNonceSeed = Array.Empty<byte>();
    private readonly Dictionary<int, (byte[] EncryptedData, double Weight)> _pendingUpdates = new();
    private bool _initialized;

    /// <summary>
    /// Initializes a new instance of <see cref="TeeSecureAggregation{T}"/>.
    /// </summary>
    /// <param name="provider">The TEE provider managing the enclave.</param>
    /// <param name="options">TEE configuration options.</param>
    public TeeSecureAggregation(ITeeProvider<T> provider, TeeOptions options)
    {
        _provider = provider ?? throw new ArgumentNullException(nameof(provider));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!_provider.IsInitialized)
        {
            _provider.Initialize(options);
        }

        _initialized = true;
    }

    /// <inheritdoc/>
    public bool AllUpdatesReceived => _pendingUpdates.Count >= _expectedClients;

    /// <inheritdoc/>
    public int UpdatesReceived => _pendingUpdates.Count;

    /// <inheritdoc/>
    public void BeginRound(int roundNumber, int expectedClients)
    {
        if (expectedClients <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(expectedClients), "Expected clients must be positive.");
        }

        EnsureInitialized();

        _roundNumber = roundNumber;
        _expectedClients = expectedClients;
        _pendingUpdates.Clear();

        // Generate a fresh session key for this round
        _sessionKey = new byte[32]; // AES-256
        _sessionNonceSeed = new byte[12]; // Nonce seed for deterministic per-client nonces
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(_sessionKey);
            rng.GetBytes(_sessionNonceSeed);
        }
    }

    /// <inheritdoc/>
    public void SubmitEncryptedUpdate(int clientId, byte[] encryptedUpdate, double weight)
    {
        EnsureInitialized();

        if (encryptedUpdate is null || encryptedUpdate.Length == 0)
        {
            throw new ArgumentException("Encrypted update must not be null or empty.", nameof(encryptedUpdate));
        }

        if (weight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weight), "Weight must be positive.");
        }

        if (_pendingUpdates.ContainsKey(clientId))
        {
            throw new InvalidOperationException($"Client {clientId} has already submitted an update for round {_roundNumber}.");
        }

        _pendingUpdates[clientId] = (encryptedUpdate, weight);
    }

    /// <inheritdoc/>
    public Tensor<T> Aggregate()
    {
        EnsureInitialized();

        if (_pendingUpdates.Count == 0)
        {
            throw new InvalidOperationException("No client updates submitted for aggregation.");
        }

        // Decrypt all client updates inside the enclave
        var decryptedUpdates = new List<(double[] Values, double Weight)>();
        double totalWeight = 0.0;

        foreach (var kvp in _pendingUpdates)
        {
            byte[] plaintext = DecryptClientUpdate(kvp.Value.EncryptedData);
            double[] values = DeserializeDoubleArray(plaintext);
            decryptedUpdates.Add((values, kvp.Value.Weight));
            totalWeight += kvp.Value.Weight;
        }

        if (decryptedUpdates.Count == 0 || totalWeight <= 0)
        {
            throw new InvalidOperationException("No valid updates to aggregate.");
        }

        // Weighted averaging inside the enclave
        int paramCount = decryptedUpdates[0].Values.Length;
        var aggregated = new double[paramCount];

        for (int i = 0; i < decryptedUpdates.Count; i++)
        {
            double[] values = decryptedUpdates[i].Values;
            double weight = decryptedUpdates[i].Weight / totalWeight;

            if (values.Length != paramCount)
            {
                throw new InvalidOperationException(
                    $"Client update dimension mismatch: expected {paramCount}, got {values.Length}.");
            }

            for (int j = 0; j < paramCount; j++)
            {
                aggregated[j] += values[j] * weight;
            }
        }

        // Clear plaintext data from enclave memory
        for (int i = 0; i < decryptedUpdates.Count; i++)
        {
            Array.Clear(decryptedUpdates[i].Values, 0, decryptedUpdates[i].Values.Length);
        }

        // Convert to tensor
        var result = new Tensor<T>(new[] { paramCount });
        for (int i = 0; i < paramCount; i++)
        {
            result[i] = NumOps.FromDouble(aggregated[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public byte[] GetAttestationQuote(byte[] roundData)
    {
        EnsureInitialized();

        if (roundData is null)
        {
            throw new ArgumentNullException(nameof(roundData));
        }

        return _provider.GenerateAttestationQuote(roundData);
    }

    /// <inheritdoc/>
    public byte[] GenerateSessionKey()
    {
        EnsureInitialized();
        EnsureSessionKey();

        // Return the public portion of the session key
        // In a real implementation, this would use ECDH or similar key exchange
        // For now, we return the session key sealed by the enclave so it can only be
        // used within the TEE boundary. The client would encrypt with this key.
        return _provider.SealData(_sessionKey);
    }

    /// <summary>
    /// Encrypts model parameters for submission to this aggregator.
    /// Clients call this to prepare their updates.
    /// </summary>
    /// <param name="parameters">Model parameters to encrypt.</param>
    /// <returns>Encrypted parameter bytes.</returns>
    public byte[] EncryptForSubmission(Tensor<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        EnsureSessionKey();

        byte[] serialized = SerializeTensor(parameters);
        return EncryptWithSessionKey(serialized);
    }

    private byte[] DecryptClientUpdate(byte[] encryptedData)
    {
        return TeeAesHelper.Decrypt(_sessionKey, encryptedData);
    }

    private byte[] EncryptWithSessionKey(byte[] plaintext)
    {
        return TeeAesHelper.Encrypt(_sessionKey, plaintext);
    }

    private byte[] SerializeTensor(Tensor<T> tensor)
    {
        int totalElements = ComputeTotalElements(tensor);
        var bytes = new byte[totalElements * 8]; // 8 bytes per double

        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(tensor[i]);
            var doubleBytes = BitConverter.GetBytes(val);
            Buffer.BlockCopy(doubleBytes, 0, bytes, i * 8, 8);
        }

        return bytes;
    }

    private static double[] DeserializeDoubleArray(byte[] data)
    {
        if (data.Length % 8 != 0)
        {
            throw new ArgumentException("Data length must be a multiple of 8 (double size).", nameof(data));
        }

        int count = data.Length / 8;
        var result = new double[count];

        for (int i = 0; i < count; i++)
        {
            result[i] = BitConverter.ToDouble(data, i * 8);
        }

        return result;
    }

    private static int ComputeTotalElements(Tensor<T> tensor)
    {
        int total = 1;
        for (int d = 0; d < tensor.Rank; d++)
        {
            total *= tensor.Shape[d];
        }

        return total;
    }

    private void EnsureSessionKey()
    {
        if (_sessionKey is null || _sessionKey.Length == 0)
        {
            _sessionKey = new byte[32]; // AES-256
            using (var rng = RandomNumberGenerator.Create())
            {
                rng.GetBytes(_sessionKey);
            }
        }
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("TEE secure aggregation has not been initialized.");
        }
    }
}
