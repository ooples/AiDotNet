namespace AiDotNet.ProgramSynthesis.Execution;

/// <summary>
/// Classifies SQL execution failures in a structured, machine-readable way.
/// </summary>
public enum SqlExecuteErrorCode
{
    None = 0,
    InvalidRequest = 1,
    QueryRequired = 2,
    UnsupportedDialect = 3,
    DialectMismatch = 4,
    DialectNotAllowedForTier = 5,
    UnknownDbId = 6,
    UnknownDatasetId = 7,
    MultiStatementNotAllowed = 8,
    TimeoutOrCanceled = 9,
    ExecutionFailed = 10,
    DialectNotConfigured = 11
}
