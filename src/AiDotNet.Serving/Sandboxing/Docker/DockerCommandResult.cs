namespace AiDotNet.Serving.Sandboxing.Docker;

public sealed class DockerCommandResult
{
    public required int ExitCode { get; init; }

    public required string StdOut { get; init; }

    public required string StdErr { get; init; }

    public bool StdOutTruncated { get; init; }

    public bool StdErrTruncated { get; init; }
}
