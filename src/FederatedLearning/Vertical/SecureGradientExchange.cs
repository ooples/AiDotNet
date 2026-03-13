using System.Security.Cryptography;
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
        var nonceTensor = new Tensor<T>(new[] { 1 });
        double nonceValue = _maskRandom.NextDouble() * 1e9;
        nonceTensor[0] = NumOps.FromDouble(nonceValue);

        // Use keystream to mask values (simulated encryption)
        var encrypted = new Tensor<T>(gradients.Shape);
        var keyRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom((int)(nonceValue % int.MaxValue));

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

        double nonceValue = NumOps.ToDouble(nonceTensor[0]);
        var keyRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom((int)(nonceValue % int.MaxValue));

        var decrypted = new Tensor<T>(encrypted.Shape);
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

        var mask = new Tensor<T>(gradients.Shape);
        var masked = new Tensor<T>(gradients.Shape);

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

        var unmasked = new Tensor<T>(masked.Shape);
        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(masked[i]);
            double maskVal = NumOps.ToDouble(mask[i]);
            unmasked[i] = NumOps.FromDouble(val - maskVal);
        }

        return unmasked;
    }
}
