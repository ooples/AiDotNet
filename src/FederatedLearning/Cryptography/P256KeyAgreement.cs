using System.Security.Cryptography;
using Org.BouncyCastle.Asn1.Nist;
using Org.BouncyCastle.Crypto.Agreement;
using Org.BouncyCastle.Crypto.Generators;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Security;
using BcBigInteger = Org.BouncyCastle.Math.BigInteger;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// A NIST P-256 key pair in raw form: the 32-byte big-endian private scalar and the 64-byte public point <c>X || Y</c>.
/// </summary>
/// <remarks>
/// Secure aggregation needs the raw private scalar because the masking key <c>s_u^SK</c> is Shamir-shared so the
/// server can reconstruct it when client <c>u</c> drops out (Bonawitz et al. 2017, round 1 ShareKeys).
/// </remarks>
internal sealed class P256KeyPair
{
    internal P256KeyPair(byte[] privateScalar, byte[] publicKey)
    {
        PrivateScalar = privateScalar;
        PublicKey = publicKey;
    }

    /// <summary>The private scalar <c>d</c>, 32 bytes big-endian.</summary>
    internal byte[] PrivateScalar { get; }

    /// <summary>The public point <c>Q = d * G</c> as <c>X || Y</c>, 32 bytes each, big-endian.</summary>
    internal byte[] PublicKey { get; }

    /// <summary>Zeroes the private scalar.</summary>
    internal void Clear() => Array.Clear(PrivateScalar, 0, PrivateScalar.Length);
}

/// <summary>
/// Elliptic-curve Diffie-Hellman key agreement over NIST P-256: the <c>KA.gen</c> and <c>KA.agree</c> of Bonawitz
/// et al. (2017), section 3.3, instantiated as in their prototype (ECDH over P-256 followed by SHA-256).
/// </summary>
/// <remarks>
/// <para>
/// <c>KA.agree</c> returns SHA-256 of the shared point's x-coordinate. That is what
/// <see cref="ECDiffieHellman.DeriveKeyMaterial(ECDiffieHellmanPublicKey)"/> computes by default, which this class uses
/// on every target that has <see cref="ECCurve"/> (.NET Framework 4.7 and later, .NET Core). .NET Framework 4.6.2 and
/// .NET Standard 2.0 lack named-curve ECDH, so there the same function runs on BouncyCastle
/// (<see cref="P256Sha256Agreement"/>), which the library already depends on. Callers expand the result with
/// HKDF-SHA256 (<see cref="HkdfSha256"/>, RFC 5869) into the actual keys and seeds.
/// </para>
/// <para><b>For Beginners:</b> Two clients each publish a public key. Combining your own private key with the other
/// client's public key gives a secret that both of you compute identically and nobody else can compute.</para>
/// </remarks>
internal static class P256KeyAgreement
{
    internal const int CoordinateLength = 32;
    internal const int PublicKeyLength = 2 * CoordinateLength;

    /// <summary>Generates a fresh key pair from the operating system's cryptographic random number generator.</summary>
    internal static P256KeyPair Generate()
    {
#if NETSTANDARD2_0 || NET462
        return P256Sha256Agreement.Generate();
#else
        using var ecdh = ECDiffieHellman.Create(ECCurve.NamedCurves.nistP256);
        var parameters = ecdh.ExportParameters(includePrivateParameters: true);
        var d = parameters.D ?? throw new CryptographicException("The platform did not export the P-256 private scalar.");
        var x = parameters.Q.X ?? throw new CryptographicException("The platform did not export the P-256 public point.");
        var y = parameters.Q.Y ?? throw new CryptographicException("The platform did not export the P-256 public point.");
        var publicKey = new byte[PublicKeyLength];
        CopyLeftPadded(x, publicKey, 0);
        CopyLeftPadded(y, publicKey, CoordinateLength);
        var scalar = new byte[CoordinateLength];
        CopyLeftPadded(d, scalar, 0);
        Array.Clear(d, 0, d.Length);
        return new P256KeyPair(scalar, publicKey);
#endif
    }

    /// <summary>
    /// <c>KA.agree</c>: SHA-256 of the x-coordinate of <paramref name="privateScalar"/> times the peer's public point.
    /// </summary>
    /// <param name="privateScalar">This party's 32-byte private scalar.</param>
    /// <param name="ownPublicKey">This party's 64-byte public key (the platform API imports a full key pair).</param>
    /// <param name="peerPublicKey">The peer's 64-byte public key; it is validated to lie on P-256.</param>
    internal static byte[] Agree(byte[] privateScalar, byte[] ownPublicKey, byte[] peerPublicKey)
    {
        ValidateScalar(privateScalar);
        ValidatePublicKey(ownPublicKey, nameof(ownPublicKey));
        ValidatePublicKey(peerPublicKey, nameof(peerPublicKey));

#if NETSTANDARD2_0 || NET462
        return P256Sha256Agreement.Derive(privateScalar, peerPublicKey);
#else
        var ownParameters = new ECParameters
        {
            Curve = ECCurve.NamedCurves.nistP256,
            D = (byte[])privateScalar.Clone(),
            Q = new ECPoint { X = Slice(ownPublicKey, 0), Y = Slice(ownPublicKey, CoordinateLength) },
        };
        var peerParameters = new ECParameters
        {
            Curve = ECCurve.NamedCurves.nistP256,
            Q = new ECPoint { X = Slice(peerPublicKey, 0), Y = Slice(peerPublicKey, CoordinateLength) },
        };

        try
        {
            using var own = ECDiffieHellman.Create(ownParameters);
            using var peer = ECDiffieHellman.Create(peerParameters);
            return own.DeriveKeyMaterial(peer.PublicKey);
        }
        finally
        {
            if (ownParameters.D is not null)
            {
                Array.Clear(ownParameters.D, 0, ownParameters.D.Length);
            }
        }
#endif
    }

    /// <summary>
    /// Validates that <paramref name="publicKey"/> is a 64-byte point on P-256 (not the identity), so a malformed key
    /// from a peer fails with a clear message instead of a platform-specific error.
    /// </summary>
    internal static void ValidatePublicKey(byte[] publicKey, string paramName)
    {
        if (publicKey is null)
        {
            throw new ArgumentNullException(paramName);
        }

        if (publicKey.Length != PublicKeyLength)
        {
            throw new ArgumentException($"A P-256 public key must be {PublicKeyLength} bytes (X || Y).", paramName);
        }

        if (!P256Sha256Agreement.IsOnCurve(publicKey))
        {
            throw new ArgumentException("The public key is not a valid point on NIST P-256.", paramName);
        }
    }

    private static void ValidateScalar(byte[] privateScalar)
    {
        if (privateScalar is null)
        {
            throw new ArgumentNullException(nameof(privateScalar));
        }

        if (privateScalar.Length != CoordinateLength)
        {
            throw new ArgumentException($"A P-256 private scalar must be {CoordinateLength} bytes.", nameof(privateScalar));
        }
    }

    internal static void CopyLeftPadded(byte[] source, byte[] destination, int offset)
    {
        if (source.Length > CoordinateLength)
        {
            throw new CryptographicException("A P-256 field element is longer than 32 bytes.");
        }

        int pad = CoordinateLength - source.Length;
        Array.Clear(destination, offset, pad);
        Buffer.BlockCopy(source, 0, destination, offset + pad, source.Length);
    }

    private static byte[] Slice(byte[] source, int offset)
    {
        var result = new byte[CoordinateLength];
        Buffer.BlockCopy(source, offset, result, 0, CoordinateLength);
        return result;
    }
}

/// <summary>
/// NIST P-256 ECDH followed by SHA-256 of the 32-byte x-coordinate, on BouncyCastle: the derivation
/// <c>ECDiffieHellman.DeriveKeyMaterial</c> uses by default. Compiled on every target so its equivalence with the
/// platform implementation is tested wherever the platform implementation exists.
/// </summary>
internal static class P256Sha256Agreement
{
    internal static readonly ECDomainParameters Domain = CreateDomain();

    private static ECDomainParameters CreateDomain()
    {
        var curve = NistNamedCurves.GetByName("P-256");
        return new ECDomainParameters(curve.Curve, curve.G, curve.N, curve.H, curve.GetSeed());
    }

    /// <summary>Generates a key pair with BouncyCastle's <see cref="SecureRandom"/>.</summary>
    internal static P256KeyPair Generate()
    {
        var generator = new ECKeyPairGenerator();
        generator.Init(new ECKeyGenerationParameters(Domain, new SecureRandom()));
        var pair = generator.GenerateKeyPair();
        var privateKey = (ECPrivateKeyParameters)pair.Private;
        var publicPoint = ((ECPublicKeyParameters)pair.Public).Q.Normalize();

        var scalar = new byte[P256KeyAgreement.CoordinateLength];
        P256KeyAgreement.CopyLeftPadded(privateKey.D.ToByteArrayUnsigned(), scalar, 0);
        var publicKey = new byte[P256KeyAgreement.PublicKeyLength];
        P256KeyAgreement.CopyLeftPadded(publicPoint.AffineXCoord.ToBigInteger().ToByteArrayUnsigned(), publicKey, 0);
        P256KeyAgreement.CopyLeftPadded(
            publicPoint.AffineYCoord.ToBigInteger().ToByteArrayUnsigned(), publicKey, P256KeyAgreement.CoordinateLength);
        return new P256KeyPair(scalar, publicKey);
    }

    /// <summary>
    /// Returns true when <paramref name="privateScalar"/> is the private key of <paramref name="publicKey"/>
    /// (<c>Q = d * G</c>); used to check a key reconstructed from Shamir shares before unmasking with it.
    /// </summary>
    internal static bool IsKeyPair(byte[] privateScalar, byte[] publicKey)
    {
        var d = new BcBigInteger(1, privateScalar);
        if (d.SignValue <= 0 || d.CompareTo(Domain.N) >= 0)
        {
            return false;
        }

        var q = Domain.G.Multiply(d).Normalize();
        var expected = new byte[P256KeyAgreement.PublicKeyLength];
        P256KeyAgreement.CopyLeftPadded(q.AffineXCoord.ToBigInteger().ToByteArrayUnsigned(), expected, 0);
        P256KeyAgreement.CopyLeftPadded(
            q.AffineYCoord.ToBigInteger().ToByteArrayUnsigned(), expected, P256KeyAgreement.CoordinateLength);

        int difference = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            difference |= expected[i] ^ publicKey[i];
        }

        return difference == 0;
    }

    /// <summary>Returns true when the 64-byte <c>X || Y</c> encodes a point on P-256 other than the identity.</summary>
    internal static bool IsOnCurve(byte[] publicKey)
    {
        var x = new BcBigInteger(1, publicKey, 0, P256KeyAgreement.CoordinateLength);
        var y = new BcBigInteger(1, publicKey, P256KeyAgreement.CoordinateLength, P256KeyAgreement.CoordinateLength);
        var p = Domain.Curve.Field.Characteristic;
        if (x.CompareTo(p) >= 0 || y.CompareTo(p) >= 0)
        {
            return false;
        }

        var point = Domain.Curve.CreatePoint(x, y);
        return !point.IsInfinity && point.IsValid();
    }

    /// <summary>SHA-256 of the x-coordinate of <paramref name="privateScalar"/> times the peer point.</summary>
    internal static byte[] Derive(byte[] privateScalar, byte[] peerPublicKey)
    {
        var x = new BcBigInteger(1, peerPublicKey, 0, P256KeyAgreement.CoordinateLength);
        var y = new BcBigInteger(1, peerPublicKey, P256KeyAgreement.CoordinateLength, P256KeyAgreement.CoordinateLength);
        var peerPoint = Domain.Curve.CreatePoint(x, y);

        var agreement = new ECDHBasicAgreement();
        agreement.Init(new ECPrivateKeyParameters(new BcBigInteger(1, privateScalar), Domain));
        byte[] secret = agreement.CalculateAgreement(new ECPublicKeyParameters(peerPoint, Domain)).ToByteArrayUnsigned();
        var padded = new byte[P256KeyAgreement.CoordinateLength];
        try
        {
            P256KeyAgreement.CopyLeftPadded(secret, padded, 0);
            using var sha = SHA256.Create();
            return sha.ComputeHash(padded);
        }
        finally
        {
            Array.Clear(secret, 0, secret.Length);
            Array.Clear(padded, 0, padded.Length);
        }
    }
}
