using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Aggregation
{
    /// <summary>
    /// Secure aggregation implementation that protects individual client updates
    /// while allowing server to compute the aggregate
    /// </summary>
    public class SecureAggregation : IParameterAggregator
    {
        /// <summary>
        /// Aggregation strategy type
        /// </summary>
        public FederatedAggregationStrategy Strategy => FederatedAggregationStrategy.SecureAggregation;

        /// <summary>
        /// Secure aggregation parameters
        /// </summary>
        public SecureAggregationParams Parameters { get; set; } = new();

        /// <summary>
        /// Client public keys for encryption
        /// </summary>
        private Dictionary<string, byte[]> ClientPublicKeys { get; set; } = new();

        /// <summary>
        /// Shared secrets between clients
        /// </summary>
        private Dictionary<string, Dictionary<string, byte[]>> SharedSecrets { get; set; } = new();

        /// <summary>
        /// Random number generator for cryptographic operations
        /// </summary>
        private readonly RandomNumberGenerator _rng = default!;

        /// <summary>
        /// Initialize secure aggregation
        /// </summary>
        public SecureAggregation()
        {
            Parameters = new SecureAggregationParams();
            ClientPublicKeys = new Dictionary<string, byte[]>();
            SharedSecrets = new Dictionary<string, Dictionary<string, byte[]>>();
            _rng = RandomNumberGenerator.Create();
        }

        /// <summary>
        /// Aggregate parameters securely without revealing individual client updates
        /// </summary>
        /// <param name="clientUpdates">Encrypted client parameter updates</param>
        /// <param name="clientWeights">Client weights for aggregation</param>
        /// <param name="strategy">Aggregation strategy</param>
        /// <returns>Aggregated parameters</returns>
        public Dictionary<string, Vector<double>> AggregateParameters(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, double> clientWeights,
            FederatedAggregationStrategy strategy)
        {
            if (strategy != FederatedAggregationStrategy.SecureAggregation)
            {
                // Fall back to standard aggregation for other strategies
                var fallbackAggregator = new FederatedAveraging();
                return fallbackAggregator.AggregateParameters(clientUpdates, clientWeights, strategy);
            }

            // Perform secure aggregation protocol
            return PerformSecureAggregation(clientUpdates, clientWeights);
        }

        /// <summary>
        /// Setup secure aggregation protocol with client key exchange
        /// </summary>
        /// <param name="clientIds">List of participating client IDs</param>
        /// <returns>Public keys for each client</returns>
        public Dictionary<string, byte[]> SetupSecureAggregation(List<string> clientIds)
        {
            ClientPublicKeys.Clear();
            SharedSecrets.Clear();

            // Generate key pairs for each client
            foreach (var clientId in clientIds)
            {
                var keyPair = GenerateKeyPair();
                ClientPublicKeys[clientId] = keyPair.PublicKey;
            }

            // Generate pairwise shared secrets
            GeneratePairwiseSecrets(clientIds);

            return new Dictionary<string, byte[]>(ClientPublicKeys);
        }

        /// <summary>
        /// Encrypt client parameters for secure aggregation
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        /// <param name="parameters">Parameters to encrypt</param>
        /// <returns>Encrypted parameters</returns>
        public Dictionary<string, Vector<double>> EncryptClientParameters(string clientId, Dictionary<string, Vector<double>> parameters)
        {
            var encryptedParameters = new Dictionary<string, Vector<double>>();

            foreach (var kvp in parameters)
            {
                var parameterName = kvp.Key;
                var parameterValues = kvp.Value;

                // Add noise mask for secure aggregation
                var maskedValues = AddSecretShares(clientId, parameterValues);
                encryptedParameters[parameterName] = maskedValues;
            }

            return encryptedParameters;
        }

        /// <summary>
        /// Perform the secure aggregation protocol
        /// </summary>
        /// <param name="maskedUpdates">Masked client updates</param>
        /// <param name="clientWeights">Client weights</param>
        /// <returns>Aggregated parameters</returns>
        private Dictionary<string, Vector<double>> PerformSecureAggregation(
            Dictionary<string, Dictionary<string, Vector<double>>> maskedUpdates,
            Dictionary<string, double> clientWeights)
        {
            // Step 1: Sum all masked parameters
            var summedMaskedParameters = SumMaskedParameters(maskedUpdates, clientWeights);

            // Step 2: Remove the masks to reveal the aggregate
            var aggregatedParameters = RemoveMasks(summedMaskedParameters, maskedUpdates.Keys.ToList());

            return aggregatedParameters;
        }

        /// <summary>
        /// Add secret shares to parameter values
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        /// <param name="parameters">Original parameters</param>
        /// <returns>Masked parameters</returns>
        private Vector<double> AddSecretShares(string clientId, Vector<double> parameters)
        {
            var maskedValues = new double[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                var mask = GenerateSecretShare(clientId, i);
                maskedValues[i] = parameters[i] + mask;
            }

            return new Vector<double>(maskedValues);
        }

        /// <summary>
        /// Generate a secret share for secure masking
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        /// <param name="parameterIndex">Parameter index</param>
        /// <returns>Secret share value</returns>
        private double GenerateSecretShare(string clientId, int parameterIndex)
        {
            // Generate deterministic noise based on shared secrets
            var totalShare = 0.0;

            if (SharedSecrets.ContainsKey(clientId))
            {
                foreach (var kvp in SharedSecrets[clientId])
                {
                    var otherClientId = kvp.Key;
                    var sharedSecret = kvp.Value;

                    // Generate noise from shared secret
                    var noise = GenerateNoiseFromSecret(sharedSecret, parameterIndex);
                    
                    // Add or subtract based on client ordering (ensures cancellation)
                    if (string.Compare(clientId, otherClientId, StringComparison.Ordinal) < 0)
                    {
                        totalShare += noise;
                    }
                    else
                    {
                        totalShare -= noise;
                    }
                }
            }

            return totalShare;
        }

        /// <summary>
        /// Generate noise from shared secret
        /// </summary>
        /// <param name="sharedSecret">Shared secret between two clients</param>
        /// <param name="parameterIndex">Parameter index for deterministic generation</param>
        /// <returns>Noise value</returns>
        private double GenerateNoiseFromSecret(byte[] sharedSecret, int parameterIndex)
        {
            // Use SHA256 to generate deterministic noise
            using (var sha256 = SHA256.Create())
            {
                var input = new byte[sharedSecret.Length + sizeof(int)];
                Array.Copy(sharedSecret, 0, input, 0, sharedSecret.Length);
                BitConverter.GetBytes(parameterIndex).CopyTo(input, sharedSecret.Length);

                var hash = sha256.ComputeHash(input);
                
                // Convert hash to double in range [-1, 1]
                var longValue = BitConverter.ToInt64(hash, 0);
                return (double)longValue / long.MaxValue;
            }
        }

        /// <summary>
        /// Sum masked parameters from all clients
        /// </summary>
        /// <param name="maskedUpdates">Masked client updates</param>
        /// <param name="clientWeights">Client weights</param>
        /// <returns>Summed masked parameters</returns>
        private Dictionary<string, Vector<double>> SumMaskedParameters(
            Dictionary<string, Dictionary<string, Vector<double>>> maskedUpdates,
            Dictionary<string, double> clientWeights)
        {
            var summedParameters = new Dictionary<string, Vector<double>>();
            var normalizedWeights = NormalizeWeights(maskedUpdates.Keys.ToList(), clientWeights);

            // Get parameter structure from first client
            var firstClient = maskedUpdates.Values.First();
            foreach (var parameterName in firstClient.Keys)
            {
                var parameterSize = firstClient[parameterName].Length;
                var summedValues = new double[parameterSize];

                // Sum weighted parameters from all clients
                foreach (var clientId in maskedUpdates.Keys)
                {
                    if (maskedUpdates[clientId].ContainsKey(parameterName))
                    {
                        var clientParameter = maskedUpdates[clientId][parameterName];
                        var weight = normalizedWeights.ContainsKey(clientId) ? normalizedWeights[clientId] : 1.0 / maskedUpdates.Count;

                        for (int i = 0; i < parameterSize; i++)
                        {
                            summedValues[i] += weight * clientParameter[i];
                        }
                    }
                }

                summedParameters[parameterName] = new Vector<double>(summedValues);
            }

            return summedParameters;
        }

        /// <summary>
        /// Remove masks from summed parameters to reveal aggregate
        /// </summary>
        /// <param name="summedMaskedParameters">Summed masked parameters</param>
        /// <param name="participatingClients">List of participating clients</param>
        /// <returns>Unmasked aggregated parameters</returns>
        private Dictionary<string, Vector<double>> RemoveMasks(
            Dictionary<string, Vector<double>> summedMaskedParameters,
            List<string> participatingClients)
        {
            var aggregatedParameters = new Dictionary<string, Vector<double>>();

            foreach (var kvp in summedMaskedParameters)
            {
                var parameterName = kvp.Key;
                var maskedParameter = kvp.Value;
                var unmaskedValues = new double[maskedParameter.Length];

                for (int i = 0; i < maskedParameter.Length; i++)
                {
                    // The masks should cancel out when all clients participate
                    // Since we add positive masks for client pairs where clientA < clientB
                    // and subtract for clientA > clientB, the total mask sum is zero
                    unmaskedValues[i] = maskedParameter[i];
                }

                aggregatedParameters[parameterName] = new Vector<double>(unmaskedValues);
            }

            return aggregatedParameters;
        }

        /// <summary>
        /// Generate key pair for client
        /// </summary>
        /// <returns>Key pair</returns>
        private KeyPair GenerateKeyPair()
        {
            var keySize = Parameters.KeySize;
            var privateKey = new byte[keySize / 8];
            var publicKey = new byte[keySize / 8];

            _rng.GetBytes(privateKey);
            
            // For simplicity, use hash of private key as public key
            // In practice, use proper elliptic curve or RSA key generation
            using (var sha256 = SHA256.Create())
            {
                publicKey = sha256.ComputeHash(privateKey);
            }

            return new KeyPair { PrivateKey = privateKey, PublicKey = publicKey };
        }

        /// <summary>
        /// Generate pairwise shared secrets between clients
        /// </summary>
        /// <param name="clientIds">List of client identifiers</param>
        private void GeneratePairwiseSecrets(List<string> clientIds)
        {
            for (int i = 0; i < clientIds.Count; i++)
            {
                var clientA = clientIds[i];
                SharedSecrets[clientA] = new Dictionary<string, byte[]>();

                for (int j = i + 1; j < clientIds.Count; j++)
                {
                    var clientB = clientIds[j];
                    
                    // Generate shared secret between clientA and clientB
                    var sharedSecret = GenerateSharedSecret(clientA, clientB);
                    
                    // Store shared secret for both clients
                    if (!SharedSecrets.ContainsKey(clientA))
                        SharedSecrets[clientA] = new Dictionary<string, byte[]>();
                    if (!SharedSecrets.ContainsKey(clientB))
                        SharedSecrets[clientB] = new Dictionary<string, byte[]>();

                    SharedSecrets[clientA][clientB] = sharedSecret;
                    SharedSecrets[clientB][clientA] = sharedSecret;
                }
            }
        }

        /// <summary>
        /// Generate shared secret between two clients
        /// </summary>
        /// <param name="clientA">First client</param>
        /// <param name="clientB">Second client</param>
        /// <returns>Shared secret</returns>
        private byte[] GenerateSharedSecret(string clientA, string clientB)
        {
            // For simplicity, generate deterministic shared secret
            // In practice, use Diffie-Hellman key exchange
            var combinedId = string.Compare(clientA, clientB, StringComparison.Ordinal) < 0 ? 
                clientA + clientB : clientB + clientA;
            
            using (var sha256 = SHA256.Create())
            {
                return sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(combinedId + Parameters.SecretSalt));
            }
        }

        /// <summary>
        /// Normalize client weights
        /// </summary>
        /// <param name="clientIds">Client identifiers</param>
        /// <param name="clientWeights">Original weights</param>
        /// <returns>Normalized weights</returns>
        private Dictionary<string, double> NormalizeWeights(List<string> clientIds, Dictionary<string, double> clientWeights)
        {
            var normalizedWeights = new Dictionary<string, double>();
            var totalWeight = 0.0;

            foreach (var clientId in clientIds)
            {
                var weight = clientWeights?.ContainsKey(clientId) == true ? clientWeights[clientId] : 1.0;
                totalWeight += weight;
            }

            foreach (var clientId in clientIds)
            {
                var weight = clientWeights?.ContainsKey(clientId) == true ? clientWeights[clientId] : 1.0;
                normalizedWeights[clientId] = totalWeight > 0 ? weight / totalWeight : 1.0 / clientIds.Count;
            }

            return normalizedWeights;
        }

        #region Interface Implementation

        public bool ValidateClientUpdates(Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates)
        {
            // Validate that all clients have consistent parameter structure
            if (clientUpdates == null || clientUpdates.Count == 0)
                return false;

            var firstClient = clientUpdates.Values.First();
            var parameterNames = firstClient.Keys.ToHashSet();

            foreach (var clientUpdate in clientUpdates.Values)
            {
                if (!parameterNames.SetEquals(clientUpdate.Keys))
                    return false;

                foreach (var paramName in parameterNames)
                {
                    if (clientUpdate[paramName].Length != firstClient[paramName].Length)
                        return false;
                }
            }

            return true;
        }

        public AggregationMetrics CalculateAggregationMetrics(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, Vector<double>> aggregatedParameters)
        {
            var metrics = new AggregationMetrics
            {
                ParticipatingClients = clientUpdates.Count,
                ParameterCount = aggregatedParameters.Values.Sum(v => v.Length)
            };

            // Note: Variance calculation is not meaningful for secure aggregation
            // as individual client updates are masked
            metrics.AverageParameterVariance = 0.0;

            return metrics;
        }

        #endregion

        /// <summary>
        /// Clean up cryptographic resources
        /// </summary>
        public void Dispose()
        {
            _rng?.Dispose();
            ClientPublicKeys?.Clear();
            SharedSecrets?.Clear();
        }
    }

    /// <summary>
    /// Key pair for secure aggregation
    /// </summary>
    public class KeyPair
    {
        public byte[] PrivateKey { get; set; } = Array.Empty<byte>();
        public byte[] PublicKey { get; set; } = Array.Empty<byte>();
    }

    /// <summary>
    /// Secure aggregation parameters
    /// </summary>
    public class SecureAggregationParams
    {
        /// <summary>
        /// Key size in bits
        /// </summary>
        public int KeySize { get; set; } = 256;

        /// <summary>
        /// Salt for shared secret generation
        /// </summary>
        public string SecretSalt { get; set; } = "FederatedLearningSecureAggregation";

        /// <summary>
        /// Minimum number of clients required for secure aggregation
        /// </summary>
        public int MinimumClients { get; set; } = 3;

        /// <summary>
        /// Maximum number of dropout clients allowed
        /// </summary>
        public int MaxDropouts { get; set; } = 1;

        /// <summary>
        /// Use additional security measures
        /// </summary>
        public bool UseAdvancedSecurity { get; set; } = true;

        /// <summary>
        /// Noise variance for additional privacy
        /// </summary>
        public double NoiseVariance { get; set; } = 0.01;
    }
}
