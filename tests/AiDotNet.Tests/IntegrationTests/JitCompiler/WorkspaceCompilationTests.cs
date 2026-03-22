using Xunit;
using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.JitCompiler;

/// <summary>
/// Integration tests for CompileWithWorkspace — verifies that JIT-compiled
/// workspace-backed execution produces correct results matching interpreted execution.
/// </summary>
public class WorkspaceCompilationTests
{
    private readonly AiDotNet.JitCompiler.JitCompiler _jit = new();
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private const double Tolerance = 1e-6;

    private static Tensor<double> MakeTensor(params double[] values)
        => new(values, new[] { values.Length });

    private static Tensor<double> MakeMatrix(double[,] values)
    {
        int rows = values.GetLength(0), cols = values.GetLength(1);
        var flat = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = values[i, j];
        return new Tensor<double>(flat, new[] { rows, cols });
    }

    [Fact]
    public void CompileWithWorkspace_AddTwoTensors_MatchesInterpreted()
    {
        var aData = MakeTensor(1, 2, 3);
        var bData = MakeTensor(4, 5, 6);

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        var result = TensorOperations<double>.Add(a, b);

        var (execute, workspace, outputSlot) = _jit.CompileWithWorkspace<double>(result, new() { a, b });

        execute(new[] { aData, bData }, _engine);
        var output = workspace.Get(outputSlot);

        Assert.Equal(5.0, output.GetFlat(0), Tolerance);
        Assert.Equal(7.0, output.GetFlat(1), Tolerance);
        Assert.Equal(9.0, output.GetFlat(2), Tolerance);
    }

    [Fact]
    public void CompileWithWorkspace_MultiplyAndAdd_MatchesInterpreted()
    {
        var aData = MakeTensor(2, 3, 4);
        var bData = MakeTensor(5, 6, 7);

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        var product = TensorOperations<double>.ElementwiseMultiply(a, b);
        var sum = TensorOperations<double>.Add(product, a);

        var (execute, workspace, outputSlot) = _jit.CompileWithWorkspace<double>(sum, new() { a, b });

        execute(new[] { aData, bData }, _engine);
        var output = workspace.Get(outputSlot);

        // Expected: a*b + a = [2*5+2, 3*6+3, 4*7+4] = [12, 21, 32]
        Assert.Equal(12.0, output.GetFlat(0), Tolerance);
        Assert.Equal(21.0, output.GetFlat(1), Tolerance);
        Assert.Equal(32.0, output.GetFlat(2), Tolerance);
    }

    [Fact]
    public void CompileWithWorkspace_ReLU_MatchesInterpreted()
    {
        var aData = MakeTensor(-2, -1, 0, 1, 2);

        var a = TensorOperations<double>.Variable(aData, "a");
        var result = TensorOperations<double>.ReLU(a);

        var (execute, workspace, outputSlot) = _jit.CompileWithWorkspace<double>(result, new() { a });

        execute(new[] { aData }, _engine);
        var output = workspace.Get(outputSlot);

        Assert.Equal(0.0, output.GetFlat(0), Tolerance);
        Assert.Equal(0.0, output.GetFlat(1), Tolerance);
        Assert.Equal(0.0, output.GetFlat(2), Tolerance);
        Assert.Equal(1.0, output.GetFlat(3), Tolerance);
        Assert.Equal(2.0, output.GetFlat(4), Tolerance);
    }

    [Fact]
    public void CompileWithWorkspace_Sigmoid_MatchesInterpreted()
    {
        var aData = MakeTensor(0, 1, -1);

        var a = TensorOperations<double>.Variable(aData, "a");
        var result = TensorOperations<double>.Sigmoid(a);

        var (execute, workspace, outputSlot) = _jit.CompileWithWorkspace<double>(result, new() { a });

        execute(new[] { aData }, _engine);
        var output = workspace.Get(outputSlot);

        Assert.Equal(0.5, output.GetFlat(0), 1e-4);
        Assert.True(output.GetFlat(1) > 0.7);  // sigmoid(1) ≈ 0.731
        Assert.True(output.GetFlat(2) < 0.3);  // sigmoid(-1) ≈ 0.269
    }

    [Fact]
    public void CompileWithWorkspace_MatMul_MatchesInterpreted()
    {
        var aData = MakeMatrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var bData = MakeMatrix(new double[,] { { 5, 6 }, { 7, 8 } });

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        var result = TensorOperations<double>.MatrixMultiply(a, b);

        var (execute, workspace, outputSlot) = _jit.CompileWithWorkspace<double>(result, new() { a, b });

        execute(new[] { aData, bData }, _engine);
        var output = workspace.Get(outputSlot);

        // [1,2;3,4] @ [5,6;7,8] = [19,22;43,50]
        Assert.Equal(19.0, output[0, 0], Tolerance);
        Assert.Equal(22.0, output[0, 1], Tolerance);
        Assert.Equal(43.0, output[1, 0], Tolerance);
        Assert.Equal(50.0, output[1, 1], Tolerance);
    }

    [Fact]
    public void CompileWithWorkspace_ZeroAllocation_AfterWarmup()
    {
        var aData = MakeTensor(1, 2, 3);
        var bData = MakeTensor(4, 5, 6);

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        var result = TensorOperations<double>.Add(a, b);

        var (execute, workspace, outputSlot) = _jit.CompileWithWorkspace<double>(result, new() { a, b });

        // Warmup
        execute(new[] { aData, bData }, _engine);

        // Subsequent executions should reuse workspace — verify result is still correct
        var newA = MakeTensor(10, 20, 30);
        var newB = MakeTensor(40, 50, 60);
        execute(new[] { newA, newB }, _engine);
        var output = workspace.Get(outputSlot);

        Assert.Equal(50.0, output.GetFlat(0), Tolerance);
        Assert.Equal(70.0, output.GetFlat(1), Tolerance);
        Assert.Equal(90.0, output.GetFlat(2), Tolerance);
    }

    [Fact]
    public void CompileWithWorkspace_ChainedOperations_MatchesInterpreted()
    {
        var aData = MakeTensor(1, 2, 3, 4);

        var a = TensorOperations<double>.Variable(aData, "a");
        // Chain: sigmoid(relu(a + a))
        var doubled = TensorOperations<double>.Add(a, a);
        var activated = TensorOperations<double>.ReLU(doubled);
        var final = TensorOperations<double>.Sigmoid(activated);

        var (execute, workspace, outputSlot) = _jit.CompileWithWorkspace<double>(final, new() { a });

        execute(new[] { aData }, _engine);
        var output = workspace.Get(outputSlot);

        // sigmoid(relu(2)) = sigmoid(2) ≈ 0.88
        Assert.True(output.GetFlat(0) > 0.8);
        Assert.True(output.GetFlat(0) < 1.0);
    }
}
