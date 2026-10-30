using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// The secret key of a keyed watermark, and the pseudo-random values every keyed watermarker derives from it.
/// </summary>
/// <remarks>
/// <para>
/// Every keyed watermarker in this namespace (text, audio and image) derives its watermark from the key through
/// HMAC-SHA256, so the key is the whole secret: without it the pattern cannot be predicted, removed on purpose or
/// forged, and a detector holding a different key sees only noise. Two derivations are offered:
/// </para>
/// <list type="bullet">
/// <item><description><see cref="SignPattern"/>: a +/-1 sequence, the bits of HMAC-SHA256(key, salt || stream ||
/// block), for spread-spectrum marks in audio and images.</description></item>
/// <item><description><see cref="Seed"/>: a 64-bit seed, the first eight bytes of HMAC-SHA256(key, salt || values),
/// for per-context choices such as the green list of a text watermark.</description></item>
/// </list>
/// <para>
/// A key is created from a string. <see cref="Public"/> is a key whose text is published in this library: anyone can
/// reproduce, and therefore forge, a watermark made with it. It exists so that a watermarker constructed without a
/// key is deterministic (it never draws a random key that nobody can later detect with); an application marking its
/// own content must use a secret key of its own, and <see cref="RequireSecret"/> turns a missing or public key into a
/// configuration error.
/// </para>
/// <para>
/// <b>For Beginners:</b> the key works like a password. Whoever embeds a watermark and whoever later checks for it
/// must use the same key. Keep it secret and never use the public key for content you need to prove is yours; a
/// long random value, such as 32 random bytes written in base64, is a good key.
/// </para>
/// </remarks>
public sealed class WatermarkKey
{
    /// <summary>The text of <see cref="Public"/>. It is published, so marks made with it prove nothing.</summary>
    public const string PublicKeyText = "aidotnet-public-watermark-key";

    private readonly byte[] _bytes;

    /// <summary>
    /// Creates a key from its text.
    /// </summary>
    /// <param name="key">The key text; it is encoded as UTF-8.</param>
    /// <exception cref="ArgumentException"><paramref name="key"/> is null, empty or whitespace.</exception>
    public WatermarkKey(string key)
    {
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("A watermark key must not be null, empty or whitespace.", nameof(key));
        _bytes = Encoding.UTF8.GetBytes(key);
        IsPublic = string.Equals(key, PublicKeyText, StringComparison.Ordinal);
    }

    /// <summary>The published key, used by watermarkers constructed without a key.</summary>
    public static WatermarkKey Public { get; } = new WatermarkKey(PublicKeyText);

    /// <summary>Whether this is the published <see cref="Public"/> key, whose marks anyone can forge.</summary>
    public bool IsPublic { get; }

    /// <summary>
    /// The key for <paramref name="key"/>, or <see cref="Public"/> when it is <c>null</c>.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="key"/> is empty or whitespace.</exception>
    public static WatermarkKey FromOptional(string? key) => key is null ? Public : new WatermarkKey(key);

    /// <summary>
    /// The key for a setting that must hold a secret key of the application's own.
    /// </summary>
    /// <param name="key">The configured key text.</param>
    /// <param name="settingName">The name of the setting, used in the error message.</param>
    /// <exception cref="InvalidOperationException">
    /// The key is missing, blank, or the published <see cref="PublicKeyText"/>: a watermark anyone can reproduce
    /// cannot show that content is the application's own.
    /// </exception>
    public static WatermarkKey RequireSecret(string? key, string settingName)
    {
        if (string.IsNullOrWhiteSpace(key) || string.Equals(key, PublicKeyText, StringComparison.Ordinal))
            throw new InvalidOperationException(
                $"{settingName} must be a secret watermark key of your own (not blank and not the published " +
                $"public key), so the watermark cannot be forged. Use a long random value, for example 32 random bytes in base64.");
        return new WatermarkKey(key ?? string.Empty);
    }

    /// <summary>
    /// The key's +/-1 pattern for one stream: the bits of HMAC-SHA256(key, salt || stream || block), 256 per block,
    /// least significant bit first; a 0 bit is -1.
    /// </summary>
    /// <param name="salt">Separates the patterns of different watermarkers that share a key.</param>
    /// <param name="stream">Selects an independent pattern, for example per tile or per audio frame.</param>
    /// <param name="count">The pattern length.</param>
    public double[] SignPattern(int salt, int stream, int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), "The pattern length must not be negative.");
        var pattern = new double[count];
        var message = new byte[12];
        WriteInt32(message, 0, salt);
        WriteInt32(message, 4, stream);
        using var hmac = new HMACSHA256(_bytes);
        for (int block = 0; block * 256 < count; block++)
        {
            WriteInt32(message, 8, block);
            var digest = hmac.ComputeHash(message);
            for (int bit = 0; bit < 256 && block * 256 + bit < count; bit++)
                pattern[block * 256 + bit] = ((digest[bit >> 3] >> (bit & 7)) & 1) == 0 ? -1.0 : 1.0;
        }

        return pattern;
    }

    /// <summary>
    /// A 64-bit pseudo-random seed for one context: the first eight bytes (little-endian) of
    /// HMAC-SHA256(key, salt || values[0] || values[1] || ...).
    /// </summary>
    /// <param name="salt">Separates the seeds of different watermarkers that share a key.</param>
    /// <param name="values">The context, for example the preceding token ids.</param>
    public ulong Seed(int salt, IReadOnlyList<int> values)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        var message = new byte[4 + 4 * values.Count];
        WriteInt32(message, 0, salt);
        for (int i = 0; i < values.Count; i++) WriteInt32(message, 4 + 4 * i, values[i]);
        using var hmac = new HMACSHA256(_bytes);
        var digest = hmac.ComputeHash(message);
        ulong seed = 0;
        for (int i = 7; i >= 0; i--) seed = (seed << 8) | digest[i];
        return seed;
    }

    /// <summary>
    /// A uniform value in [0, 1) for <paramref name="item"/> under <paramref name="seed"/>: the SplitMix64 finalizer of
    /// seed + item * golden-ratio increment, top 53 bits. Cheap enough to evaluate for every token of a vocabulary;
    /// it is keyed through <paramref name="seed"/>, which only the key holder can compute.
    /// </summary>
    public static double Uniform(ulong seed, int item)
    {
        unchecked
        {
            ulong z = seed + ((ulong)(uint)item + 1UL) * 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            return (z >> 11) * (1.0 / 9007199254740992.0);
        }
    }

    private static void WriteInt32(byte[] buffer, int offset, int value)
    {
        buffer[offset] = (byte)value;
        buffer[offset + 1] = (byte)(value >> 8);
        buffer[offset + 2] = (byte)(value >> 16);
        buffer[offset + 3] = (byte)(value >> 24);
    }
}

/// <summary>
/// A watermarker whose mark is derived from a <see cref="WatermarkKey"/>.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> code that embeds watermarks for an application checks <see cref="UsesPublicKey"/> and
/// refuses to mark content with the published key, since such a mark proves nothing about who made the content.
/// </remarks>
public interface IKeyedWatermarker
{
    /// <summary>Whether the watermarker uses the published <see cref="WatermarkKey.Public"/> key.</summary>
    bool UsesPublicKey { get; }
}
