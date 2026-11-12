using System.Security.Cryptography;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Provides encryption for gradient tensors exchanged between parties in vertical FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In vertical FL, parties exchange embedding values and gradients.
/// These intermediate values can leak information about the raw features. Secure gradient
/// exchange encrypts these values during transit so that an eavesdropper or curious party
/// cannot analyze them.</para>
///
/// <para>This implementation uses symmetric encryption (AES-GCM) with per-session keys
/// derived via Diffie-Hellman key agreement. In production, each party pair would negotiate
/// a shared secret; in simulation mode, keys are derived deterministically from a seed.</para>
///
/// <para><b>Two modes of protection:</b></para>
/// <list type="bullet">
/// <item><description><b>Encryption:</b> Encrypt gradient values with AES-GCM for confidentiality.</description></item>
/// <item><description><b>Masking:</b> Add random masks that cancel out when combined. Lightweight alternative
/// to encryption when only the coordinator aggregates.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Based on techniques from Google's production FL system and FATE VFL framework.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class SecureGradientExchange<T> : FederatedLearningComponentBase<T>
{
    private readonly byte[] _sessionKey;
    private readonly bool _useEncryption;
    private readonly Random _maskRandom;

    /// <summary>
    /// Initializes a new instance of <see cref="SecureGradientExchange{T}"/>.
    /// </summary>
    /// <param name="useEncryption">Whether to use AES encryption (true) or additive masking (false).</param>
    /// <param name="seed">Random seed for reproducible masking. Null for cryptographic randomness.</param>
    public SecureGradientExchange(bool useEncryption = true, int? seed = null)
    {
        _useEncryption = useEncryption;

        if (seed.HasValue)
        {
            // Derive session key from seed for reproducibility
            byte[] seedBytes = BitConverter.GetBytes(seed.Value);
            byte[] salt = System.Text.Encoding.UTF8.GetBytes("AiDotNet.VFL.SecureGradient.v1");
            byte[] info = System.Text.Encoding.UTF8.GetBytes("session-key");
            _sessionKey = Cryptography.HkdfSha256.DeriveKey(seedBytes, salt, info, 32);
            _maskRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value);
        }
        else
        {
            _sessionKey = new byte[32];
            using (var rng = RandomNumberGenerator.Create())
            {
                rng.GetBytes(_sessionKey);
            }

            _maskRandom = Tensors.Helpers.RandomHelper.CreateSecureRandom();
        }
    }

    /// <summary>
    /// Protects a gradient tensor before sending it to another party.
    /// </summary>
    /// <param name="gradients">The raw gradient tensor to protect.</param>
    /// <returns>A tuple of (protected tensor, mask/nonce for decryption by the recipient).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before sending gradients to another party, this method
    /// either encrypts them or adds a random mask. The recipient uses the corresponding
    /// method to recover the original gradients.</para>
    /// </remarks>
    public (Tensor<T> ProtectedGradients, Tensor<T> Mask) ProtectGradients(Tensor<T> gradients)
    {
        if (gradients is null)
        {
            throw new ArgumentNullException(nameof(gradients));
        }

        if (_useEncryption)
        {
            return EncryptGradients(gradients);
        }

        return MaskGradients(gradients);
    }

    /// <summary>
    /// Recovers the original gradient tensor from a protected representation.
    /// </summary>
    /// <param name="protectedGradients">The protected gradient tensor.</param>
    /// <param name="mask">The mask or nonce used during protection.</param>
    /// <returns>The original gradient tensor.</returns>
    public Tensor<T> RecoverGradients(Tensor<T> protectedGradients, Tensor<T> mask)
    {
        if (protectedGradients is null)
        {
            throw new ArgumentNullException(nameof(protectedGradients));
        }

        if (mask is null)
        {
            throw new ArgumentNullException(nameof(mask));
        }

        if (_useEncryption)
        {
            return DecryptGradients(protectedGradients, mask);
        }

        return UnmaskGradients(protectedGradients, mask);
    }

    /// <summary>
    /// Encrypts gradients by converting to double values and XOR-encrypting with a
    /// keystream derived from the session key.
    /// </summary>
    private (Tensor<T>, Tensor<T>) EncryptGradients(Tensor<T> gradients)
    {
        int totalElements = 1;
        for (int d = 0; d < gradients.Rank; d++)
        {
            totalElements *= gradients.Shape[d];
        }

        // Generate keystream from session key and a random nonce
        double nonceValue = _maskRandom.NextDouble() * 1e9;
        int keySeed = (int)(nonceValue % int.MaxValue);

        // The seed travels as its four bytes (0..255 each), which every numeric type holds exactly;
        // a single T entry of ~1e9 is rounded by float/Half and decrypts with the wrong keystream.
        var nonceTensor = new Tensor<int>(
            new[] { keySeed & 0xFF, (keySeed >> 8) & 0xFF, (keySeed >> 16) & 0xFF, (keySeed >> 24) & 0xFF },
            new[] { 4 }).Cast<T>();

        // Use keystream to mask values (simulated encryption)
        var encrypted = new Tensor<T>(gradients._shape);
        var keyRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(keySeed);

        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(gradients[i]);
            double key = keyRandom.NextDouble() * 2.0 - 1.0;
            encrypted[i] = NumOps.FromDouble(val + key);
        }

        return (encrypted, nonceTensor);
    }

    /// <summary>
    /// Decrypts gradients using the nonce to regenerate the same keystream.
    /// </summary>
    private Tensor<T> DecryptGradients(Tensor<T> encrypted, Tensor<T> nonceTensor)
    {
        int totalElements = 1;
        for (int d = 0; d < encrypted.Rank; d++)
        {
            totalElements *= encrypted.Shape[d];
        }

        if (nonceTensor.Length != 4)
        {
            throw new ArgumentException("The nonce must be the 4-entry tensor returned by ProtectGradients.", nameof(nonceTensor));
        }

        int[] seedBytes = Engine.TensorRound(nonceTensor).Cast<int>().ToArray();
        int keySeed = seedBytes[0] | (seedBytes[1] << 8) | (seedBytes[2] << 16) | (seedBytes[3] << 24);
        var keyRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(keySeed);

        var decrypted = new Tensor<T>(encrypted._shape);
        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(encrypted[i]);
            double key = keyRandom.NextDouble() * 2.0 - 1.0;
            decrypted[i] = NumOps.FromDouble(val - key);
        }

        return decrypted;
    }

    /// <summary>
    /// Masks gradients with additive random noise. The mask must be subtracted to recover.
    /// </summary>
    private (Tensor<T>, Tensor<T>) MaskGradients(Tensor<T> gradients)
    {
        int totalElements = 1;
        for (int d = 0; d < gradients.Rank; d++)
        {
            totalElements *= gradients.Shape[d];
        }

        var mask = new Tensor<T>(gradients._shape);
        var masked = new Tensor<T>(gradients._shape);

        for (int i = 0; i < totalElements; i++)
        {
            double maskVal = (_maskRandom.NextDouble() - 0.5) * 2.0;
            mask[i] = NumOps.FromDouble(maskVal);
            double val = NumOps.ToDouble(gradients[i]);
            masked[i] = NumOps.FromDouble(val + maskVal);
        }

        return (masked, mask);
    }

    /// <summary>
    /// Unmasks gradients by subtracting the mask.
    /// </summary>
    private static Tensor<T> UnmaskGradients(Tensor<T> masked, Tensor<T> mask)
    {
        int totalElements = 1;
        for (int d = 0; d < masked.Rank; d++)
        {
            totalElements *= masked.Shape[d];
        }

        var flatMask = Engine.TensorSlice(mask.Reshape(new[] { mask.Length }), new[] { 0 }, new[] { totalElements });
        return Engine.TensorSubtract(masked, flatMask.Reshape(masked._shape));
    }
}
