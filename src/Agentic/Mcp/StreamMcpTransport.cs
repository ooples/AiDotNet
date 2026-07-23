using System.IO;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// An <see cref="IMcpTransport"/> that speaks newline-delimited JSON-RPC 2.0 over a reader/writer pair — the
/// framing MCP uses over stdio. Wire the writer to a child MCP server process's standard input and the reader
/// to its standard output to drive a local server process.
/// </summary>
/// <remarks>
/// <para>
/// Each request is written as one JSON line and flushed; responses are read line by line until the one whose
/// <c>id</c> matches the request (interleaved notifications and unrelated responses are skipped). Calls are
/// serialized with an internal lock so the request/response correlation is safe under concurrent callers.
/// </para>
/// <para><b>For Beginners:</b> The way to talk to an MCP server that runs as a separate program on your
/// machine: messages go in and out as lines of text over its input/output streams. Point this at that
/// program's streams and the client works the same as over HTTP.
/// </para>
/// </remarks>
public sealed class StreamMcpTransport : IMcpTransport, IDisposable
{
    private readonly TextReader _reader;
    private readonly TextWriter _writer;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private int _id;

    /// <summary>
    /// Initializes a new stream transport over the given reader/writer.
    /// </summary>
    /// <param name="reader">The incoming-message reader (e.g., a process's standard output).</param>
    /// <param name="writer">The outgoing-message writer (e.g., a process's standard input).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="reader"/> or <paramref name="writer"/> is <c>null</c>.</exception>
    public StreamMcpTransport(TextReader reader, TextWriter writer)
    {
        Guard.NotNull(reader);
        Guard.NotNull(writer);
        _reader = reader;
        _writer = writer;
    }

    /// <inheritdoc/>
    public async Task<JObject> SendRequestAsync(string method, JObject? parameters, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(method);

        var id = Interlocked.Increment(ref _id);
        var envelope = McpJsonRpc.BuildRequest(id, method, parameters);

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await _writer.WriteLineAsync(envelope.ToString(Formatting.None)).ConfigureAwait(false);
            await _writer.FlushAsync().ConfigureAwait(false);

            while (true)
            {
                // ReadLineAsync has no cancellation path of its own, and a
                // stalled MCP peer would otherwise hang this call forever
                // while it holds _gate (blocking every later caller). Race the
                // read against a token-cancelled delay so cancellation always
                // gets the caller out.
                var readTask = _reader.ReadLineAsync();
                var completed = await Task.WhenAny(
                    readTask,
                    Task.Delay(Timeout.Infinite, cancellationToken)).ConfigureAwait(false);
                if (completed != readTask)
                {
                    throw new OperationCanceledException(cancellationToken);
                }

                var line = await readTask.ConfigureAwait(false);
                if (line is null)
                {
                    throw new McpException("MCP stream closed before a response was received.");
                }

                if (line.Trim().Length == 0)
                {
                    continue;
                }

                JObject message;
                try
                {
                    message = JObject.Parse(line);
                }
                catch (JsonException ex)
                {
                    throw new McpException("Malformed JSON-RPC message: " + ex.Message);
                }

                var responseId = (int?)message["id"];
                if (responseId is null || responseId.Value != id)
                {
                    // A notification or a response to a different request — skip and keep reading.
                    continue;
                }

                return McpJsonRpc.ExtractResult(message);
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc/>
    public void Dispose() => _gate.Dispose();
}
