using System;
using System.Security.Cryptography;
using Org.BouncyCastle.Asn1.Nist;
using Org.BouncyCastle.Crypto.Agreement;
using Org.BouncyCastle.Crypto.Parameters;
using BcBigInteger = Org.BouncyCastle.Math.BigInteger;
#if NETSTANDARD2_0
using Org.BouncyCastle.Crypto.Generators;
using Org.BouncyCastle.Security;
#endif

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// An ephemeral elliptic-curve Diffie-Hellman key pair, for deriving pairwise secrets between simulated clients.
/// </summary>
/// <remarks>
/// On every target except .NET Standard 2.0 this wraps <see cref="ECDiffieHellman"/> unchanged:
/// <c>ECDiffieHellman.Create()</c> (NIST P-521) and <c>DeriveKeyMaterial</c>, whose default key derivation is SHA-256
/// of the shared secret. .NET Standard 2.0 has no <c>ECDiffieHellman</c>, so there the same agreement runs on
/// BouncyCastle (<see cref="P521Sha256Agreement"/>), already a dependency for license verification.
/// </remarks>
internal sealed class EphemeralKeyPair : IDisposable
{
#if NETSTANDARD2_0
    private ECPrivateKeyParameters? _private;
    private ECPublicKeyParameters? _public;

    private EphemeralKeyPair()
    {
        var generator = new ECKeyPairGenerator();
        generator.Init(new ECKeyGenerationParameters(P521Sha256Agreement.Domain, new SecureRandom()));
        var pair = generator.GenerateKeyPair();
        _private = (ECPrivateKeyParameters)pair.Private;
        _public = (ECPublicKeyParameters)pair.Public;
    }

    /// <summary>Derives SHA-256 of the shared secret with <paramref name="other"/>'s public key.</summary>
    public byte[] DeriveSharedSecret(EphemeralKeyPair other)
    {
        var privateKey = _private ?? throw new ObjectDisposedException(nameof(EphemeralKeyPair));
        var peer = other._public ?? throw new ObjectDisposedException(nameof(EphemeralKeyPair));
        return P521Sha256Agreement.Derive(privateKey.D, peer.Q.Normalize());
    }

    /// <summary>
    /// Ends the key's lifetime: drops the private scalar and public point, after which the pair refuses to derive.
    /// BouncyCastle's <c>BigInteger</c> is immutable and cannot be zeroed, so dropping the reference is what this
    /// implementation can do; the other targets dispose their <see cref="ECDiffieHellman"/>.
    /// </summary>
    public void Dispose()
    {
        _private = null;
        _public = null;
    }
#else
    private ECDiffieHellman? _key = ECDiffieHellman.Create();

    /// <summary>Derives SHA-256 of the shared secret with <paramref name="other"/>'s public key.</summary>
    public byte[] DeriveSharedSecret(EphemeralKeyPair other)
    {
        var key = _key ?? throw new ObjectDisposedException(nameof(EphemeralKeyPair));
        var peer = other._key ?? throw new ObjectDisposedException(nameof(EphemeralKeyPair));
        return key.DeriveKeyMaterial(peer.PublicKey);
    }

    public void Dispose()
    {
        _key?.Dispose();
        _key = null;
    }
#endif

    /// <summary>Generates a fresh key pair.</summary>
    public static EphemeralKeyPair Create() => new();
}

/// <summary>
/// NIST P-521 ECDH followed by SHA-256 of the shared x-coordinate, left-padded to the field width: the derivation
/// <c>ECDiffieHellman.DeriveKeyMaterial</c> uses by default. Compiled on every target so the equivalence with the
/// platform implementation is tested where the platform implementation exists.
/// </summary>
internal static class P521Sha256Agreement
{
    internal static readonly ECDomainParameters Domain = CreateDomain();

    private static ECDomainParameters CreateDomain()
    {
        var curve = NistNamedCurves.GetByName("P-521");
        return new ECDomainParameters(curve.Curve, curve.G, curve.N, curve.H, curve.GetSeed());
    }

    /// <summary>SHA-256 of the x-coordinate of <paramref name="privateScalar"/> · <paramref name="peerPoint"/>.</summary>
    internal static byte[] Derive(BcBigInteger privateScalar, Org.BouncyCastle.Math.EC.ECPoint peerPoint)
    {
        var agreement = new ECDHBasicAgreement();
        agreement.Init(new ECPrivateKeyParameters(privateScalar, Domain));
        byte[] secret = agreement.CalculateAgreement(new ECPublicKeyParameters(peerPoint, Domain)).ToByteArrayUnsigned();
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

    /// <summary><see cref="Derive(BcBigInteger, Org.BouncyCastle.Math.EC.ECPoint)"/> from raw big-endian key bytes.</summary>
    internal static byte[] Derive(byte[] privateScalar, byte[] peerX, byte[] peerY)
    {
        var point = Domain.Curve.CreatePoint(new BcBigInteger(1, peerX), new BcBigInteger(1, peerY));
        return Derive(new BcBigInteger(1, privateScalar), point);
    }
}
