using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Validation;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>An <see cref="IChatClient{T}"/> that reaches a model by running a command-line program.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Not every model is an HTTP endpoint. A locally installed agent, a vendor's own CLI, or a wrapper script that
/// holds credentials this process should never see are all reached by running a command, and this is the client for
/// that shape. The conversation is written to the program's standard input and its standard output is the reply.
/// </para>
/// <para>
/// Three things make that safe enough to run thousands of times in an evolution run. The prompt goes on standard
/// input rather than in an argument, so a long conversation cannot overflow the operating system's command-length
/// limit or end up in a process listing. Every call is bounded by a timeout and the process is killed, with its
/// children, when it expires, so one hung model does not stall a run. And
/// <see cref="ProcessChatClientOptions.MaxBudgetUsd"/> is passed to the program on every call, because a subprocess
/// model can be an agent that decides for itself how much work a prompt deserves.
/// </para>
/// <para>
/// A non-zero exit code becomes an exception carrying the exit code and a bounded excerpt of standard error, which
/// is what a caller needs to tell "the command is missing" from "the model refused".
/// </para>
/// <para><b>For Beginners:</b> Use this when your model is a program you run rather than a web service. Give it the
/// command and, if the program supports one, a spending cap per call.</para>
/// </remarks>
public sealed class ProcessChatClient<T> : IChatClient<T>
{
    private const int MaxErrorExcerpt = 2_000;

    /// <summary>UTF-8 without a byte-order mark, used for both directions of the command's pipes.</summary>
    private static readonly UTF8Encoding Utf8 = new(encoderShouldEmitUTF8Identifier: false);

    private readonly ProcessChatClientOptions _options;
    private readonly IReadOnlyList<string> _arguments;
    private long _calls;

    /// <summary>Initializes a client over a command.</summary>
    /// <param name="options">The command, its arguments, and the per-call bounds.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A setting is invalid, or the composed command line is too long.</exception>
    public ProcessChatClient(ProcessChatClientOptions options)
    {
        Guard.NotNull(options);
        ProcessChatClientOptions effective = options.Clone();
        effective.Validate();
        _options = effective;
        _arguments = effective.BuildArguments();
    }

    /// <inheritdoc/>
    public string ModelId => _options.ModelId;

    /// <summary>Gets how many times the command has been run.</summary>
    public long Calls => Interlocked.Read(ref _calls);

    /// <summary>Gets the arguments passed on every call, budget included.</summary>
    public IReadOnlyList<string> Arguments => _arguments;

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">The command could not start, or it exited non-zero.</exception>
    /// <exception cref="TimeoutException">The command did not finish within the configured timeout.</exception>
    public async Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _calls);

        var startInfo = new ProcessStartInfo
        {
            FileName = _options.FileName,
            WorkingDirectory = _options.WorkingDirectory ?? string.Empty,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,

            // Program text is not ASCII in general, and the console code page would mangle it in both directions.
            // A code-evolution tool that silently corrupts non-ASCII source is worse than one that cannot run.
            StandardOutputEncoding = Utf8,
            StandardErrorEncoding = Utf8
        };
#if NET5_0_OR_GREATER
        startInfo.StandardInputEncoding = Utf8;
#endif
#if NET5_0_OR_GREATER
        foreach (string argument in _arguments) startInfo.ArgumentList.Add(argument);
#else
        // The older framework has no argument list, so each value is quoted into one string. Every argument here is
        // caller-supplied configuration rather than model output, and the prompt never reaches the command line.
        startInfo.Arguments = string.Join(" ", _arguments.Select(Quote));
#endif

        using var process = new Process { StartInfo = startInfo };
        try
        {
            if (!process.Start())
                throw new InvalidOperationException("The command '" + _options.FileName + "' did not start.");
        }
        catch (Exception exception) when (exception is System.ComponentModel.Win32Exception or IOException)
        {
            throw new InvalidOperationException(
                "The command '" + _options.FileName + "' could not be started.", exception);
        }

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(_options.Timeout);

        Task<string> output = process.StandardOutput.ReadToEndAsync();
        Task<string> error = process.StandardError.ReadToEndAsync();

        try
        {
            await SendPromptAsync(process, RenderConversation(messages), timeout.Token).ConfigureAwait(false);
            await WaitForExitAsync(process, timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            KillTree(process);
            throw new TimeoutException(
                "The command '" + _options.FileName + "' did not finish within " + _options.Timeout + ".");
        }
        catch (OperationCanceledException)
        {
            KillTree(process);
            throw;
        }

        string text = Bound(await output.ConfigureAwait(false));
        string diagnostics = await error.ConfigureAwait(false);
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                "The command '" + _options.FileName + "' exited with code " +
                process.ExitCode.ToString(System.Globalization.CultureInfo.InvariantCulture) + ": " +
                Excerpt(diagnostics) + ".");
        }

        return new ChatResponse(new ChatMessage(ChatRole.Assistant, text), ChatFinishReason.Stop,
            usage: null, modelId: ModelId);
    }

    /// <inheritdoc/>
    /// <remarks>The command answers all at once, so the whole reply arrives as a single update.</remarks>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ChatResponse response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        yield return new ChatResponseUpdate(ChatRole.Assistant, response.Text, finishReason: ChatFinishReason.Stop);
    }

    /// <summary>Renders the conversation as the plain text the command reads from standard input.</summary>
    /// <remarks>
    /// Each message is labelled with its role, because a command that receives an unlabelled wall of text cannot tell
    /// the instructions from the question. The format is deliberately plain rather than JSON so a shell script can
    /// consume it without a parser.
    /// </remarks>
    private static string RenderConversation(IReadOnlyList<ChatMessage> messages)
    {
        var text = new StringBuilder();
        foreach (ChatMessage message in messages)
        {
            if (message is null) continue;
            text.Append(message.Role.ToString().ToUpperInvariant()).Append(':').Append('\n');
            text.Append(message.Text).Append('\n').Append('\n');
        }
        return text.ToString();
    }

    private string Bound(string value) =>
        value.Length <= _options.MaxOutputChars ? value : value.Substring(0, _options.MaxOutputChars);

    private static string Excerpt(string value)
    {
        string trimmed = value.Trim();
        if (trimmed.Length == 0) return "no error output";
        return trimmed.Length <= MaxErrorExcerpt ? trimmed : trimmed.Substring(0, MaxErrorExcerpt) + "...";
    }

    /// <summary>Writes the prompt to the command's input, under the same deadline as everything else.</summary>
    /// <remarks>
    /// <para>
    /// The write has to be cancellable, because it is the step most likely to block: a redirected input pipe holds
    /// only a few kilobytes, and an evolution prompt is far larger, so a command that never reads its input stops the
    /// write partway. Left unguarded that is an unbounded wait with the process still running, past the very timeout
    /// this class promises, and no kill.
    /// </para>
    /// <para>
    /// A command that closes its input early is not an error - it read as much as it needed - so that case falls
    /// through to the exit code and the output, which are what actually decide the outcome.
    /// </para>
    /// </remarks>
    private static async Task SendPromptAsync(Process process, string prompt, CancellationToken cancellationToken)
    {
        try
        {
            Task write = process.StandardInput.WriteAsync(prompt);
            var cancelled = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            using (cancellationToken.Register(() => cancelled.TrySetResult(true)))
            {
                if (await Task.WhenAny(write, cancelled.Task).ConfigureAwait(false) != write)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                }
            }

            await write.ConfigureAwait(false);
            process.StandardInput.Close();
        }
        catch (IOException)
        {
            TryCloseInput(process);
        }
        catch (ObjectDisposedException)
        {
            TryCloseInput(process);
        }
    }

    private static void TryCloseInput(Process process)
    {
        try
        {
            process.StandardInput.Close();
        }
        catch (IOException)
        {
            // The pipe is already gone, which is the state this was trying to reach.
        }
        catch (ObjectDisposedException)
        {
            // As above.
        }
    }

    /// <summary>Waits for the command to exit, throwing when the token fires first.</summary>
    /// <remarks>
    /// The older target framework has no asynchronous wait, so it subscribes to the exit event instead of blocking a
    /// thread pool thread for the length of a model call.
    /// </remarks>
    private static async Task WaitForExitAsync(Process process, CancellationToken cancellationToken)
    {
#if NET5_0_OR_GREATER
        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
#else
        var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        void OnExited(object? sender, EventArgs args) => completion.TrySetResult(true);

        process.EnableRaisingEvents = true;
        process.Exited += OnExited;
        try
        {
            if (process.HasExited) return;
            using (cancellationToken.Register(() => completion.TrySetCanceled()))
            {
                await completion.Task.ConfigureAwait(false);
            }
        }
        finally
        {
            process.Exited -= OnExited;
        }
#endif
    }

#if !NET5_0_OR_GREATER
    /// <summary>Quotes one argument for the single command-line string the older framework takes.</summary>
    private static string Quote(string argument)
    {
        if (argument.Length > 0 && argument.IndexOfAny(new[] { ' ', '\t', '"' }) < 0) return argument;
        return "\"" + argument.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }
#endif

    /// <summary>Kills the command and anything it started, so a timeout does not leave work running.</summary>
    private static void KillTree(Process process)
    {
        try
        {
            if (process.HasExited) return;
#if NET5_0_OR_GREATER
            process.Kill(entireProcessTree: true);
#else
            process.Kill();
#endif
        }
        catch (InvalidOperationException)
        {
            // The process ended between the check and the kill, which is the outcome the kill wanted.
        }
        catch (System.ComponentModel.Win32Exception)
        {
            // The operating system refused the kill; there is nothing further this code can do about it.
        }
    }
}
