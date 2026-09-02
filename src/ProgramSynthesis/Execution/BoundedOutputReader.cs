using System.Text;

namespace AiDotNet.ProgramSynthesis.Execution;

/// <summary>
/// Drains one redirected child-process stream into a fixed-size buffer, discarding everything beyond the cap while
/// still reading to the end so the child never blocks on a full pipe.
/// </summary>
/// <remarks>
/// Two properties matter here and both are deliberate. The reader keeps consuming after the cap is reached instead
/// of stopping, because a writer blocked on a full pipe would hang until the wall-clock limit expired and would
/// then be reported as a timeout rather than as the noisy program it is. And <see cref="Snapshot"/> takes the same
/// lock the pump uses, so a caller that gives up waiting for the drain (after a kill that left a stream open) can
/// still read a consistent, non-torn view of what arrived.
/// </remarks>
internal sealed class BoundedOutputReader
{
    private const int BufferSize = 4096;

    private readonly object _gate = new();
    private readonly StringBuilder _builder;
    private readonly int _maxChars;
    private bool _truncated;

    /// <summary>Initializes a reader that keeps at most <paramref name="maxChars"/> characters.</summary>
    /// <param name="maxChars">The character cap; zero drains the stream while keeping nothing.</param>
    public BoundedOutputReader(int maxChars)
    {
        _maxChars = maxChars < 0 ? 0 : maxChars;
        _builder = new StringBuilder(Math.Min(_maxChars, BufferSize));
    }

    /// <summary>Reads the stream to completion, retaining at most the configured number of characters.</summary>
    /// <param name="reader">The redirected stream to drain.</param>
    /// <param name="cancellationToken">A token that stops the pump between chunks.</param>
    /// <returns>A task that completes when the stream ends, the token is signalled, or the pipe breaks.</returns>
    public async Task PumpAsync(StreamReader reader, CancellationToken cancellationToken)
    {
        if (reader is null)
        {
            return;
        }

        var buffer = new char[BufferSize];
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                int read = await reader.ReadAsync(buffer, 0, buffer.Length).ConfigureAwait(false);
                if (read <= 0)
                {
                    return;
                }

                Append(buffer, read);
            }
        }
        catch (ObjectDisposedException)
        {
            // The process was killed and its redirected stream was torn down; keep whatever arrived.
        }
        catch (IOException)
        {
            // The pipe broke when the child died; keep whatever arrived.
        }
        catch (InvalidOperationException)
        {
            // The stream was closed concurrently with a pending read; keep whatever arrived.
        }
    }

    /// <summary>Takes a consistent copy of the captured text and whether anything was discarded.</summary>
    /// <returns>The retained text and a flag reporting that output exceeded the cap.</returns>
    public (string Text, bool Truncated) Snapshot()
    {
        lock (_gate)
        {
            return (_builder.ToString(), _truncated);
        }
    }

    private void Append(char[] buffer, int count)
    {
        lock (_gate)
        {
            if (_maxChars == 0)
            {
                _truncated = true;
                return;
            }

            int remaining = _maxChars - _builder.Length;
            if (remaining <= 0)
            {
                _truncated = true;
                return;
            }

            int toAppend = Math.Min(count, remaining);
            _builder.Append(buffer, 0, toAppend);
            if (toAppend < count)
            {
                _truncated = true;
            }
        }
    }
}
