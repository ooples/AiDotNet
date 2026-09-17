using System.Buffers.Binary;
using System.IO.Pipes;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Evolve.Cli;

/// <summary>Bounded, current-user-only local IPC; no shell commands, file paths or arbitrary method dispatch.</summary>
internal sealed class LocalRunControl : IAsyncDisposable
{
    private const int MaximumResponseBytes = 64 * 1024;
    private static readonly Encoding Utf8 = new UTF8Encoding(false, true);
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Func<string, string> _handle;
    private readonly string _name;
    private readonly Task _serve;
    private int _disposed;

    internal LocalRunControl(string session, Func<string, string> handle)
    {
        _name = PipeName(session);
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        // Bind before returning so duplicate session names fail before any evaluation starts.
        try { _serve = ServeAsync(CreateServer()); }
        catch { _shutdown.Dispose(); throw; }
    }

    internal static string PipeName(string session)
    {
        if (session.Length is < 1 or > 64 || session.Any(c => !char.IsAsciiLetterOrDigit(c) && c is not '-' and not '_'))
            throw new ArgumentException("--session must contain 1..64 ASCII letters, digits, hyphens or underscores.");
        return "aidotnet-evolve-" + Convert.ToHexString(SHA256.HashData(Utf8.GetBytes(session))).ToLowerInvariant();
    }

    private NamedPipeServerStream CreateServer() => new(_name, PipeDirection.InOut, 1, PipeTransmissionMode.Byte,
        PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly | PipeOptions.FirstPipeInstance);

    private async Task ServeAsync(NamedPipeServerStream first)
    {
        NamedPipeServerStream? pipe = first;
        try
        {
            while (pipe is not null && !_shutdown.IsCancellationRequested)
            {
                using (pipe)
                {
                    await pipe.WaitForConnectionAsync(_shutdown.Token).ConfigureAwait(false);
                    using var request = CancellationTokenSource.CreateLinkedTokenSource(_shutdown.Token);
                    request.CancelAfter(TimeSpan.FromSeconds(3));
                    try
                    {
                        string command = await ReadAsync(pipe, 32, request.Token).ConfigureAwait(false);
                        string response = command is "inspect" or "pause" or "cancel"
                            ? _handle(command) : "{\"error\":\"unsupported_command\"}";
                        await WriteAsync(pipe, response, MaximumResponseBytes, request.Token).ConfigureAwait(false);
                    }
                    catch (Exception exception) when (exception is IOException or OperationCanceledException or DecoderFallbackException)
                    {
                        // A disconnected, stalled or malformed client cannot stop the optimization run.
                    }
                }
                pipe = null;
                if (!_shutdown.IsCancellationRequested) pipe = CreateServer();
            }
        }
        catch (OperationCanceledException) when (_shutdown.IsCancellationRequested) { }
        finally { pipe?.Dispose(); }
    }

    internal static async Task<string> SendAsync(string session, string command, CancellationToken token = default)
    {
        if (command is not ("inspect" or "pause" or "cancel")) throw new ArgumentException("Unsupported control command.");
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token);
        timeout.CancelAfter(TimeSpan.FromSeconds(5));
        using var pipe = new NamedPipeClientStream(".", PipeName(session), PipeDirection.InOut,
            PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly);
        await pipe.ConnectAsync(timeout.Token).ConfigureAwait(false);
        await WriteAsync(pipe, command, 32, timeout.Token).ConfigureAwait(false);
        return await ReadAsync(pipe, MaximumResponseBytes, timeout.Token).ConfigureAwait(false);
    }

    private static async Task<string> ReadAsync(Stream stream, int maximum, CancellationToken token)
    {
        var header = new byte[4];
        await stream.ReadExactlyAsync(header, token).ConfigureAwait(false);
        int length = BinaryPrimitives.ReadInt32LittleEndian(header);
        if (length < 1 || length > maximum) throw new IOException("Invalid control frame length.");
        var payload = new byte[length];
        await stream.ReadExactlyAsync(payload, token).ConfigureAwait(false);
        return Utf8.GetString(payload);
    }

    private static async Task WriteAsync(Stream stream, string value, int maximum, CancellationToken token)
    {
        int length = Utf8.GetByteCount(value);
        if (length < 1 || length > maximum) throw new IOException("Control response exceeds its bound.");
        var frame = new byte[4 + length];
        BinaryPrimitives.WriteInt32LittleEndian(frame, length);
        Utf8.GetBytes(value, frame.AsSpan(4));
        await stream.WriteAsync(frame, token).ConfigureAwait(false);
        await stream.FlushAsync(token).ConfigureAwait(false);
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _shutdown.Cancel();
        try { await _serve.ConfigureAwait(false); }
        finally { _shutdown.Dispose(); }
    }
}
