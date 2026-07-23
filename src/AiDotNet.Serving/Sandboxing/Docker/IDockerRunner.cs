namespace AiDotNet.Serving.Sandboxing.Docker;

public interface IDockerRunner
{
    Task<DockerCommandResult> RunAsync(
        string arguments,
        string? stdIn,
        TimeSpan timeout,
        int maxStdOutChars,
        int maxStdErrChars,
        CancellationToken cancellationToken);
}
