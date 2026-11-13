using System;
using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

/// <summary>
/// Element types the linear-algebra kernels do not run natively (integers, decimal) go through double; these pin what
/// that route may and may not do to them.
/// </summary>
public class LinalgDispatchExactnessTests
{
    [Fact]
    [Trait("category", "unit")]
    public void Diagonal_of_a_long_matrix_keeps_every_digit()
    {
        // 2^53 + 1 has no double representation; a round trip through double returns 2^53.
        const long exact = 9007199254740993L;
        var matrix = new Tensor<long>(new[] { exact, 1L, 2L, 3L }, new[] { 2, 2 });

        var diagonal = LinalgDispatch<long>.Diagonal(matrix);

        Assert.Equal(new[] { exact, 3L }, diagonal.ToArray());
    }

    [Fact]
    [Trait("category", "unit")]
    public void Lu_factor_of_an_integer_matrix_is_refused_rather_than_truncated()
    {
        // [[2,1],[1,1]] has the multiplier 0.5, which an int factor would hold as 0.
        var matrix = new Tensor<int>(new[] { 2, 1, 1, 1 }, new[] { 2, 2 });

        Assert.Throws<NotSupportedException>(() => LinalgDispatch<int>.LuFactor(matrix));
    }

    [Fact]
    [Trait("category", "unit")]
    public void Cramer_rejects_a_singular_decimal_matrix()
    {
        // decimal's determinant is computed in double; its residue on this singular matrix is double round-off, far
        // above decimal's own epsilon, so the bound must be double's.
        var a = new Matrix<decimal>(3, 3);
        a[0, 0] = 1; a[0, 1] = 2; a[0, 2] = 3;
        a[1, 0] = 4; a[1, 1] = 5; a[1, 2] = 6;
        a[2, 0] = 7; a[2, 1] = 8; a[2, 2] = 9;
        var cramer = new CramerDecomposition<decimal>(a);

        Assert.Throws<InvalidOperationException>(() => cramer.Solve(new Vector<decimal>(new[] { 1m, 2m, 3m })));
    }

    [Fact]
    [Trait("category", "unit")]
    public void Class_index_labels_are_rechecked_outside_a_compiled_plan()
    {
        // A compiled plan replays the one-hot but not the host-side range check, so the fused step re-runs it on each
        // batch's labels; an index past the class count must still throw.
        var loss = new CrossEntropyLoss<double>();
        var predicted = new Tensor<double>(new[] { 0.2, 0.3, 0.5, 0.6, 0.3, 0.1 }, new[] { 2, 3 });
        loss.ComputeTapeLoss(predicted, new Tensor<double>(new[] { 2.0, 0.0 }, new[] { 2 }));

        loss.ValidateIndexTargetOutsidePlan(new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 }));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => loss.ValidateIndexTargetOutsidePlan(new Tensor<double>(new[] { 1.0, 3.0 }, new[] { 2 })));
    }
}
