using System;
using System.IO;

namespace AiDotNet.Helpers;

/// <summary>
/// A read-only, seekable view of the byte range [start, start + length) of a seekable stream, so a section of a large
/// file (an optimizer payload, the state block inside it) can be parsed as if it were the whole stream, with
/// <see cref="Length"/> and end-of-stream at the section's end, and without copying it into an array.
/// </summary>
/// <remarks>
/// Every read seeks the underlying stream to this view's position first, so several views of one stream, and other
/// readers of it, do not disturb each other. Disposing the view leaves the underlying stream open. Not thread-safe.
/// </remarks>
internal sealed class ReadOnlyWindowStream : Stream
{
    private readonly Stream _inner;
    private readonly long _start;
    private readonly long _length;
    private long _position;

    /// <summary>Creates a view of <paramref name="length"/> bytes of <paramref name="inner"/> from <paramref name="start"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="inner"/> cannot both read and seek.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range does not lie inside <paramref name="inner"/>.</exception>
    public ReadOnlyWindowStream(Stream inner, long start, long length)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        if (!inner.CanRead || !inner.CanSeek)
            throw new ArgumentException("The underlying stream must be readable and seekable.", nameof(inner));
        if (start < 0 || length < 0 || start > inner.Length - length)
            throw new ArgumentOutOfRangeException(nameof(length),
                $"The range [{start}, {start} + {length}) is outside a stream of {inner.Length} bytes.");
        _start = start;
        _length = length;
    }

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => _length;

    public override long Position
    {
        get => _position;
        set
        {
            if (value < 0) throw new ArgumentOutOfRangeException(nameof(value));
            _position = value;
        }
    }

    public override int Read(byte[] buffer, int offset, int count)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (offset < 0 || count < 0 || offset > buffer.Length - count) throw new ArgumentOutOfRangeException(nameof(count));
        long remaining = _length - _position;
        if (remaining <= 0 || count == 0) return 0;
        int toRead = (int)Math.Min(count, remaining);
        _inner.Position = _start + _position;
        int read = _inner.Read(buffer, offset, toRead);
        _position += read;
        return read;
    }

    public override long Seek(long offset, SeekOrigin origin)
    {
        long target = origin switch
        {
            SeekOrigin.Begin => offset,
            SeekOrigin.Current => _position + offset,
            SeekOrigin.End => _length + offset,
            _ => throw new ArgumentOutOfRangeException(nameof(origin)),
        };
        if (target < 0) throw new IOException("An attempt was made to move the position before the beginning of the stream.");
        _position = target;
        return _position;
    }

    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException("The view is read-only.");
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException("The view is read-only.");
}
