using System.Diagnostics;
using System.Linq;
using System.Text;

namespace AiDotNet.Serving.Sandboxing.Docker;

public sealed class DockerRunner : IDockerRunner
{
    private static readonly string DockerExecutablePath = ResolveDockerExecutablePath();
    private static readonly string SafePath = BuildSafePath();

    public async Task<DockerCommandResult> RunAsync(
        string arguments,
        string? stdIn,
        TimeSpan timeout,
        int maxStdOutChars,
        int maxStdErrChars,
        CancellationToken cancellationToken)
    {
        if (arguments is null)
        {
            throw new ArgumentNullException(nameof(arguments));
        }

        if (string.IsNullOrWhiteSpace(arguments))
        {
            throw new ArgumentException("Docker arguments are required.", nameof(arguments));
        }

        if (timeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be > 0.");
        }

        if (maxStdOutChars < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxStdOutChars), "MaxStdOutChars must be >= 0.");
        }

        if (maxStdErrChars < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxStdErrChars), "MaxStdErrChars must be >= 0.");
        }

        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        linkedCts.CancelAfter(timeout);

        var psi = new ProcessStartInfo
        {
            FileName = DockerExecutablePath,
            Arguments = arguments,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = stdIn is not null,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        // Avoid searching for executables in attacker-controlled PATH segments. Constrain the child process PATH to a
        // fixed set of typical system directories (and known docker install directories on Windows).
        psi.Environment["PATH"] = SafePath;

        using var process = new Process { StartInfo = psi };

        try
        {
            if (!process.Start())
            {
                throw new InvalidOperationException("Failed to start docker process.");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Docker is not available or could not be started.", ex);
        }

        var stdOutTask = ReadWithLimitAsync(process.StandardOutput, maxStdOutChars, linkedCts.Token);
        var stdErrTask = ReadWithLimitAsync(process.StandardError, maxStdErrChars, linkedCts.Token);
        Task? stdInTask = null;

        if (stdIn is not null)
        {
            stdInTask = WriteStdInAsync(process, stdIn, linkedCts.Token);
        }

        try
        {
            await process.WaitForExitAsync(linkedCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // Ignore kill failures.
            }

            throw;
        }

        var (stdOut, stdOutTruncated) = await stdOutTask.ConfigureAwait(false);
        var (stdErr, stdErrTruncated) = await stdErrTask.ConfigureAwait(false);

        if (stdInTask is not null)
        {
            await stdInTask.ConfigureAwait(false);
        }

        return new DockerCommandResult
        {
            ExitCode = process.ExitCode,
            StdOut = stdOut,
            StdErr = stdErr,
            StdOutTruncated = stdOutTruncated,
            StdErrTruncated = stdErrTruncated
        };
    }

    private static async Task WriteStdInAsync(Process process, string stdIn, CancellationToken cancellationToken)
    {
        try
        {
            await process.StandardInput.WriteAsync(stdIn.AsMemory(), cancellationToken).ConfigureAwait(false);
            await process.StandardInput.FlushAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            // Ignore STDIN failures (process may exit early).
        }
        finally
        {
            try
            {
                process.StandardInput.Close();
            }
            catch
            {
                // Ignore close failures.
            }
        }
    }

    private static async Task<(string Output, bool Truncated)> ReadWithLimitAsync(
        StreamReader reader,
        int maxChars,
        CancellationToken cancellationToken)
    {
        if (maxChars == 0)
        {
            var buffer = new char[4096];
            var sawOutput = false;
            while (true)
            {
                var read = await reader.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken).ConfigureAwait(false);
                if (read <= 0)
                {
                    break;
                }

                sawOutput = true;
            }

            return (string.Empty, Truncated: sawOutput);
        }

        var sb = new StringBuilder(Math.Min(maxChars, 4096));
        var buf = new char[4096];
        var remaining = maxChars;
        var truncated = false;

        while (true)
        {
            int read = await reader.ReadAsync(buf.AsMemory(0, buf.Length), cancellationToken).ConfigureAwait(false);
            if (read <= 0)
            {
                break;
            }

            if (remaining > 0)
            {
                int toAppend = Math.Min(read, remaining);
                sb.Append(buf, 0, toAppend);
                if (read > toAppend)
                {
                    truncated = true;
                }

                remaining -= toAppend;
            }
            else
            {
                truncated = true;
            }
        }

        return (sb.ToString(), truncated);
    }

    private static string ResolveDockerExecutablePath()
    {
        if (OperatingSystem.IsWindows())
        {
            var candidates = new[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Docker", "Docker", "resources", "bin", "docker.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Docker", "Docker", "resources", "bin", "docker.exe")
            };

            foreach (var candidate in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (!string.IsNullOrWhiteSpace(candidate) && File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return "docker";
        }

        var unixCandidates = new[]
        {
            "/usr/bin/docker",
            "/bin/docker",
            "/usr/local/bin/docker",
            "/snap/bin/docker"
        };

        foreach (var candidate in unixCandidates)
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return "docker";
    }

    private static string BuildSafePath()
    {
        if (OperatingSystem.IsWindows())
        {
            var system32 = Environment.GetFolderPath(Environment.SpecialFolder.System);
            var windows = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
            var programFilesDocker = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Docker", "Docker", "resources", "bin");
            var programFilesDockerX86 = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Docker", "Docker", "resources", "bin");

            return string.Join(
                ";",
                new[] { system32, windows, programFilesDocker, programFilesDockerX86 }
                    .Where(static p => !string.IsNullOrWhiteSpace(p))
                    .Select(static p => p.Trim())
                    .Distinct(StringComparer.OrdinalIgnoreCase));
        }

        return "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
    }
}
