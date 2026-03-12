using System.Security.Cryptography;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// Deterministic pseudorandom generator (PRG) based on HMAC-SHA256.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A PRG expands a short random seed into a long stream of random-looking bytes.
/// We use it to generate per-parameter mask values from a 32-byte pairwise seed.
/// </remarks>
internal sealed class HmacSha256Prg : IDisposable
{
    private readonly HMACSHA256 _hmac;
    private byte[] _buffer = Array.Empty<byte>();
    private int _bufferOffset;
    private ulong _counter;
    private bool _disposed;

    public HmacSha256Prg(byte[] key)
    {
        if (key == null || key.Length == 0)
        {
            throw new ArgumentException("Key cannot be null or empty.", nameof(key));
        }

        _hmac = new HMACSHA256(key);
        _bufferOffset = 0;
        _counter = 0;
    }

    public double NextUnitIntervalDouble()
    {
        ulong value = NextUInt64() >> 11; // Keep 53 bits for IEEE 754 mantissa.
        return value / (double)(1UL << 53);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _hmac.Dispose();

        if (_buffer.Length > 0)
        {
            Array.Clear(_buffer, 0, _buffer.Length);
        }

        _buffer = Array.Empty<byte>();
    }

    private ulong NextUInt64()
    {
        EnsureBufferHasBytes(required: 8);
        ulong value = BitConverter.ToUInt64(_buffer, _bufferOffset);
        _bufferOffset += 8;
        return value;
    }

    private void EnsureBufferHasBytes(int required)
    {
        if (_bufferOffset + required <= _buffer.Length)
        {
            return;
        }

        var counterBytes = BitConverter.GetBytes(_counter++);
        _buffer = _hmac.ComputeHash(counterBytes);
        _bufferOffset = 0;
    }
}

