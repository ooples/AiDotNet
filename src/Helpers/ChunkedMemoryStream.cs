using System;
using System.Collections.Generic;
using System.IO;

namespace AiDotNet.Helpers;

/// <summary>
/// An in-memory stream backed by a list of blocks, so it can hold more than the 2 GB a <see cref="MemoryStream"/>
/// (one int-indexed array) can. Used where a whole layer or model is serialized in memory: a large layer's state
/// overflowed MemoryStream ("Array dimensions exceeded supported range") and made the model impossible to clone.
/// Blocks start small and double up to 64 MB, so a small payload costs about what a MemoryStream would.
/// Supports write, seek, and read; not thread-safe.
/// </summary>
internal sealed class ChunkedMemoryStream : Stream
{
    private const int FirstBlockSize = 64 * 1024;
    private const int MaxBlockSize = 64 * 1024 * 1024;

    private readonly List<byte[]> _blocks = new();
    private readonly List<long> _blockStarts = new();
    private long _capacity;
    private long _length;
    private long _position;

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => true;
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

    public override void Flush() { }

    private void EnsureCapacity(long required)
    {
        while (_capacity < required)
        {
            int size = _blocks.Count == 0
                ? FirstBlockSize
                : (int)Math.Min(MaxBlockSize, (long)_blocks[_blocks.Count - 1].Length * 2);
            _blockStarts.Add(_capacity);
            _blocks.Add(new byte[size]);
            _capacity += size;
        }
    }

    private int BlockIndexOf(long position)
    {
        // Few blocks (it takes ~40 to pass 2 GB), so a backward scan is cheap and usually hits the last block.
        for (int i = _blockStarts.Count - 1; i >= 0; i--)
            if (_blockStarts[i] <= position) return i;
        return 0;
    }

    public override int Read(byte[] buffer, int offset, int count)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (offset < 0 || count < 0 || offset + count > buffer.Length) throw new ArgumentOutOfRangeException(nameof(count));
        long available = _length - _position;
        if (available <= 0) return 0;
        int toRead = (int)Math.Min(count, available);
        int done = 0;
        while (done < toRead)
        {
            int block = BlockIndexOf(_position);
            int within = (int)(_position - _blockStarts[block]);
            int n = Math.Min(toRead - done, _blocks[block].Length - within);
            Buffer.BlockCopy(_blocks[block], within, buffer, offset + done, n);
            done += n;
            _position += n;
        }
        return done;
    }

    public override void Write(byte[] buffer, int offset, int count)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (offset < 0 || count < 0 || offset + count > buffer.Length) throw new ArgumentOutOfRangeException(nameof(count));
        EnsureCapacity(_position + count);
        int done = 0;
        while (done < count)
        {
            int block = BlockIndexOf(_position);
            int within = (int)(_position - _blockStarts[block]);
            int n = Math.Min(count - done, _blocks[block].Length - within);
            Buffer.BlockCopy(buffer, offset + done, _blocks[block], within, n);
            done += n;
            _position += n;
        }
        if (_position > _length) _length = _position;
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
        if (target < 0) throw new IOException("Seek before the beginning of the stream.");
        _position = target;
        return _position;
    }

    public override void SetLength(long value)
    {
        if (value < 0) throw new ArgumentOutOfRangeException(nameof(value));
        if (value < _length)
        {
            // Zero what is cut off so a later extension reads zeros, as MemoryStream does.
            long clearFrom = value;
            while (clearFrom < _length)
            {
                int block = BlockIndexOf(clearFrom);
                int within = (int)(clearFrom - _blockStarts[block]);
                int n = (int)Math.Min(_length - clearFrom, _blocks[block].Length - within);
                Array.Clear(_blocks[block], within, n);
                clearFrom += n;
            }
        }
        else
        {
            EnsureCapacity(value);
        }
        _length = value;
        if (_position > _length) _position = _length;
    }
}
