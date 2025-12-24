namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class SqlValue
{
    public required SqlValueKind Kind { get; init; }

    public long? IntegerValue { get; init; }

    public double? RealValue { get; init; }

    public bool? BooleanValue { get; init; }

    public string? TextValue { get; init; }

    public string? BlobBase64 { get; init; }
}

