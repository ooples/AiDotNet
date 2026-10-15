using System;
#if NETSTANDARD2_0
using Org.BouncyCastle.Asn1.Nist;
using Org.BouncyCastle.Crypto.Agreement;
using Org.BouncyCastle.Crypto.Generators;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Security;
#endif
using System.Security.Cryptography;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// An ephemeral elliptic-curve Diffie-Hellman key pair, for deriving pairwise secrets between simulated clients.
/// </summary>
/// <remarks>
/// On every target except .NET Standard 2.0 this wraps <see cref="ECDiffieHellman"/> unchanged:
/// <c>ECDiffieHellman.Create()</c> and <c>DeriveKeyMaterial</c>, whose default key derivation is SHA-256 of the shared
/// secret. .NET Standard 2.0 has no <c>ECDiffieHellman</c>, so there the same agreement (NIST P-521, SHA-256 of the
/// shared x-coordinate) runs on BouncyCastle, already a dependency for license verification.
/// </remarks>
internal sealed class EphemeralKeyPair : IDisposable
{
#if NETSTANDARD2_0
    private static readonly ECDomainParameters Domain = CreateDomain();
    private readonly ECPrivateKeyParameters _private;
    private readonly ECPublicKeyParameters _public;

    private EphemeralKeyPair()
    {
        var generator = new ECKeyPairGenerator();
        generator.Init(new ECKeyGenerationParameters(Domain, new SecureRandom()));
        var pair = generator.GenerateKeyPair();
        _private = (ECPrivateKeyParameters)pair.Private;
        _public = (ECPublicKeyParameters)pair.Public;
    }

    private static ECDomainParameters CreateDomain()
    {
        var curve = NistNamedCurves.GetByName("P-521");
        return new ECDomainParameters(curve.Curve, curve.G, curve.N, curve.H, curve.GetSeed());
    }

    /// <summary>Derives SHA-256 of the shared secret with <paramref name="other"/>'s public key.</summary>
    public byte[] DeriveSharedSecret(EphemeralKeyPair other)
    {
        var agreement = new ECDHBasicAgreement();
        agreement.Init(_private);
        byte[] secret = agreement.CalculateAgreement(other._public).ToByteArrayUnsigned();
        // Left-pad to the field size, as the platform implementations hash the fixed-width x-coordinate.
        int fieldBytes = (Domain.Curve.FieldSize + 7) / 8;
        var padded = new byte[fieldBytes];
        Buffer.BlockCopy(secret, 0, padded, fieldBytes - secret.Length, secret.Length);
        try
        {
            using var sha = SHA256.Create();
            return sha.ComputeHash(padded);
        }
        finally
        {
            Array.Clear(secret, 0, secret.Length);
            Array.Clear(padded, 0, padded.Length);
        }
    }

    public void Dispose()
    {
    }
#else
    private readonly ECDiffieHellman _key = ECDiffieHellman.Create();

    /// <summary>Derives SHA-256 of the shared secret with <paramref name="other"/>'s public key.</summary>
    public byte[] DeriveSharedSecret(EphemeralKeyPair other) => _key.DeriveKeyMaterial(other._key.PublicKey);

    public void Dispose() => _key.Dispose();
#endif

    /// <summary>Generates a fresh key pair.</summary>
    public static EphemeralKeyPair Create() => new();
}
