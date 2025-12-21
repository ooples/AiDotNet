using AiDotNet.Autodiff;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Interface for regression models (for testing purposes).
/// </summary>
public interface IRegressionModel<T>
{
    bool SupportsJitCompilation { get; }

    ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes);
}
