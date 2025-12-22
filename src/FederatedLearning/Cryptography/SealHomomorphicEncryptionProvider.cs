using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Microsoft.Research.SEAL;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// Homomorphic encryption provider implemented using Microsoft SEAL (.NET wrapper).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This provider enables "sum then decrypt" aggregation:
/// clients are simulated as encrypting their weighted updates, the server adds ciphertexts, and the trusted key holder decrypts the result.
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
public sealed class SealHomomorphicEncryptionProvider<T> : HomomorphicEncryptionProviderBase<T>
{
    public override Vector<T> AggregateEncryptedWeightedAverage(
        Dictionary<int, Vector<T>> clientParameters,
        Dictionary<int, double> clientWeights,
        Vector<T> globalBaseline,
        IReadOnlyList<int> encryptedIndices,
        HomomorphicEncryptionOptions options)
    {
        if (clientParameters == null || clientParameters.Count == 0)
        {
            throw new ArgumentException("Client parameters cannot be null or empty.", nameof(clientParameters));
        }

        if (clientWeights == null || clientWeights.Count == 0)
        {
            throw new ArgumentException("Client weights cannot be null or empty.", nameof(clientWeights));
        }

        if (globalBaseline == null)
        {
            throw new ArgumentNullException(nameof(globalBaseline));
        }

        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        int n = globalBaseline.Length;
        var indices = (encryptedIndices ?? Array.Empty<int>())
            .Distinct()
            .Where(i => i >= 0 && i < n)
            .OrderBy(i => i)
            .ToArray();

        var result = globalBaseline.Clone();
        if (indices.Length == 0)
        {
            return result;
        }

        double totalWeight = 0.0;
        foreach (var clientId in clientParameters.Keys)
        {
            if (!clientWeights.TryGetValue(clientId, out var w))
            {
                throw new ArgumentException($"Missing weight for client {clientId}.", nameof(clientWeights));
            }

            totalWeight += w;
        }

        if (totalWeight <= 0.0)
        {
            throw new ArgumentException("Total weight must be positive.", nameof(clientWeights));
        }

        switch (options.Scheme)
        {
            case HomomorphicEncryptionScheme.Ckks:
                AggregateCkks(clientParameters, clientWeights, indices, options, totalWeight, result);
                return result;

            case HomomorphicEncryptionScheme.Bfv:
                AggregateBfv(clientParameters, clientWeights, indices, options, totalWeight, result);
                return result;

            default:
                throw new InvalidOperationException($"Unknown HE scheme '{options.Scheme}'. Supported values: Ckks, Bfv.");
        }
    }

    public override string GetProviderName() => "SEAL";

    private void AggregateCkks(
        Dictionary<int, Vector<T>> clientParameters,
        Dictionary<int, double> clientWeights,
        int[] indices,
        HomomorphicEncryptionOptions options,
        double totalWeight,
        Vector<T> output)
    {
        var parms = new EncryptionParameters(SchemeType.CKKS)
        {
            PolyModulusDegree = (ulong)Math.Max(2048, options.PolyModulusDegree)
        };

        parms.CoeffModulus = CreateCkksCoeffModulus(parms.PolyModulusDegree, options.CkksCoeffModulusBits);

        using var context = new SEALContext(parms);
        using var keygen = new KeyGenerator(context);
        keygen.CreatePublicKey(out PublicKey publicKey);
        using var pk = publicKey;
        using var secretKey = keygen.SecretKey;
        using var encryptor = new Encryptor(context, pk);
        using var evaluator = new Evaluator(context);
        using var decryptor = new Decryptor(context, secretKey);
        using var encoder = new CKKSEncoder(context);

        double scale = options.CkksScale;
        if (scale <= 0.0)
        {
            scale = Math.Pow(2.0, 40);
        }

        int slotCount = (int)Math.Min(int.MaxValue, encoder.SlotCount);
        int n = output.Length;
        var mask = new bool[n];
        foreach (var idx in indices)
        {
            mask[idx] = true;
        }

        int blocks = (n + slotCount - 1) / slotCount;
        for (int b = 0; b < blocks; b++)
        {
            int offset = b * slotCount;
            int len = Math.Min(slotCount, n - offset);

            using var sumCipher = new Ciphertext();
            var isFirstClient = true;

            foreach (var (clientId, parameters) in clientParameters.OrderBy(k => k.Key))
            {
                double weight = clientWeights[clientId];
                var values = new double[slotCount];
                for (int i = 0; i < len; i++)
                {
                    int idx = offset + i;
                    if (!mask[idx])
                    {
                        continue;
                    }

                    double v = NumOps.ToDouble(parameters[idx]);
                    values[i] = v * weight;
                }

                using var plain = new Plaintext();
                encoder.Encode(values, scale, plain);

                if (isFirstClient)
                {
                    encryptor.Encrypt(plain, sumCipher);
                    isFirstClient = false;
                }
                else
                {
                    using var cipher = new Ciphertext();
                    encryptor.Encrypt(plain, cipher);
                    evaluator.AddInplace(sumCipher, cipher);
                }
            }

            if (isFirstClient)
            {
                continue;
            }

            using var sumPlain = new Plaintext();
            decryptor.Decrypt(sumCipher, sumPlain);
            var decoded = new List<double>();
            encoder.Decode(sumPlain, decoded);

            for (int i = 0; i < len; i++)
            {
                int idx = offset + i;
                if (!mask[idx])
                {
                    continue;
                }

                double avg = decoded[i] / totalWeight;
                output[idx] = NumOps.FromDouble(avg);
            }
        }
    }

    private void AggregateBfv(
        Dictionary<int, Vector<T>> clientParameters,
        Dictionary<int, double> clientWeights,
        int[] indices,
        HomomorphicEncryptionOptions options,
        double totalWeight,
        Vector<T> output)
    {
        int degree = Math.Max(2048, options.PolyModulusDegree);
        var parms = new EncryptionParameters(SchemeType.BFV)
        {
            PolyModulusDegree = (ulong)degree
        };

        parms.CoeffModulus = CoeffModulus.BFVDefault(parms.PolyModulusDegree);
        parms.PlainModulus = PlainModulus.Batching(parms.PolyModulusDegree, Math.Max(16, options.BfvPlainModulusBitSize));

        using var context = new SEALContext(parms);
        using var keygen = new KeyGenerator(context);
        keygen.CreatePublicKey(out PublicKey publicKey);
        using var pk = publicKey;
        using var secretKey = keygen.SecretKey;
        using var encryptor = new Encryptor(context, pk);
        using var evaluator = new Evaluator(context);
        using var decryptor = new Decryptor(context, secretKey);
        using var encoder = new BatchEncoder(context);

        ulong plainModulus = context.FirstContextData.Parms.PlainModulus.Value;
        double scale = options.BfvFixedPointScale;
        if (scale <= 0.0)
        {
            scale = 1_000.0;
        }

        int slotCount = (int)Math.Min(int.MaxValue, encoder.SlotCount);
        int n = output.Length;
        var mask = new bool[n];
        foreach (var idx in indices)
        {
            mask[idx] = true;
        }

        int blocks = (n + slotCount - 1) / slotCount;
        for (int b = 0; b < blocks; b++)
        {
            int offset = b * slotCount;
            int len = Math.Min(slotCount, n - offset);

            using var sumCipher = new Ciphertext();
            var isFirstClient = true;

            foreach (var (clientId, parameters) in clientParameters.OrderBy(k => k.Key))
            {
                double weight = clientWeights[clientId];
                var values = new ulong[slotCount];
                for (int i = 0; i < len; i++)
                {
                    int idx = offset + i;
                    if (!mask[idx])
                    {
                        continue;
                    }

                    double v = NumOps.ToDouble(parameters[idx]) * weight;
                    long scaled = (long)Math.Round(v * scale);
                    long mod = scaled % (long)plainModulus;
                    if (mod < 0)
                    {
                        mod += (long)plainModulus;
                    }
                    values[i] = (ulong)mod;
                }

                using var plain = new Plaintext();
                encoder.Encode(values, plain);

                if (isFirstClient)
                {
                    encryptor.Encrypt(plain, sumCipher);
                    isFirstClient = false;
                }
                else
                {
                    using var cipher = new Ciphertext();
                    encryptor.Encrypt(plain, cipher);
                    evaluator.AddInplace(sumCipher, cipher);
                }
            }

            if (isFirstClient)
            {
                continue;
            }

            using var sumPlain = new Plaintext();
            decryptor.Decrypt(sumCipher, sumPlain);
            var decoded = new List<ulong>();
            encoder.Decode(sumPlain, decoded);

            for (int i = 0; i < len; i++)
            {
                int idx = offset + i;
                if (!mask[idx])
                {
                    continue;
                }

                long signed = decoded[i] > plainModulus / 2 ? (long)decoded[i] - (long)plainModulus : (long)decoded[i];
                double avg = (signed / scale) / totalWeight;
                output[idx] = NumOps.FromDouble(avg);
            }
        }
    }

    private static IEnumerable<Modulus> CreateCkksCoeffModulus(ulong polyModulusDegree, IReadOnlyList<int>? configuredBits)
    {
        var candidates = new List<int[]>();

        if (configuredBits != null && configuredBits.Count > 0)
        {
            candidates.Add(configuredBits.Where(b => b > 0).ToArray());
        }

        candidates.Add(polyModulusDegree switch
        {
            <= 2048 => new[] { 27, 27 },
            <= 4096 => new[] { 36, 36, 37 },
            <= 8192 => new[] { 60, 40, 40, 60 },
            <= 16384 => new[] { 60, 50, 50, 60 },
            _ => new[] { 60, 60, 60, 60 }
        });

        candidates.Add(new[] { 30, 30, 30 });
        candidates.Add(new[] { 20, 20, 20 });

        foreach (var bits in candidates)
        {
            if (bits.Length == 0)
            {
                continue;
            }

            try
            {
                return CoeffModulus.Create(polyModulusDegree, bits);
            }
            catch (ArgumentException)
            {
                // Try next candidate.
            }
            catch (InvalidOperationException)
            {
                // Try next candidate.
            }
        }

        throw new ArgumentException("Unable to select valid CKKS coefficient modulus parameters.", nameof(configuredBits));
    }
}
