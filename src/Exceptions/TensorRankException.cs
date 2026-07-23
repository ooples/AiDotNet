namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when a tensor's rank doesn't match the expected rank.
/// </summary>
public class TensorRankException : AiDotNetException
{
    /// <summary>
    /// The expected rank of the tensor.
    /// </summary>
    public int ExpectedRank { get; }

    /// <summary>
    /// The actual rank of the tensor.
    /// </summary>
    public int ActualRank { get; }

    /// <summary>
    /// The component where the rank mismatch occurred.
    /// </summary>
    public string Component { get; }

    /// <summary>
    /// The operation being performed when the mismatch was detected.
    /// </summary>
    public string Operation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorRankException"/> class.
    /// </summary>
    /// <param name="expectedRank">The expected rank of the tensor.</param>
    /// <param name="actualRank">The actual rank of the tensor.</param>
    /// <param name="component">The component where the rank mismatch occurred.</param>
    /// <param name="operation">The operation being performed when the mismatch was detected.</param>
    public TensorRankException(int expectedRank, int actualRank, string component, string operation)
        : base(FormatMessage(expectedRank, actualRank, component, operation))
    {
        ExpectedRank = expectedRank;
        ActualRank = actualRank;
        Component = component;
        Operation = operation;
    }

    private static string FormatMessage(int expectedRank, int actualRank, string component, string operation)
    {
        return $"Rank mismatch in {component} during {operation}: " +
               $"Expected rank {expectedRank}, but got {actualRank}.";
    }
}
