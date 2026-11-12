using System.Collections.Generic;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="TETNPAlgorithm{T, TInput, TOutput}"/> against Ashman et al., "Translation-Equivariant
/// Transformer Neural Processes" (ICML 2024): attention depends on the INPUT displacements x_i - x_j, so translating
/// every input leaves the refined representations unchanged, and no regularizer is needed for it.
/// </summary>
public class TETNPPaperTests
{
    private static TETNPAlgorithm<double, Matrix<double>, Vector<double>> Algorithm()
        => new(new TETNPOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
        {
            RepresentationDim = 4, NumHeads = 2, NumFrequencyBands = 3, Seed = 8,
        });

    private static List<Vector<double>> Reps() => new()
    {
        new Vector<double>(new[] { 0.3, -0.1, 0.8, 0.2 }),
        new Vector<double>(new[] { -0.5, 0.4, 0.1, 0.9 }),
        new Vector<double>(new[] { 0.7, 0.6, -0.3, -0.2 }),
    };

    [Fact]
    public void Defaults_UseNoEquivarianceRegularizer()
    {
        Assert.Equal(0.0, new TETNPOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2)).EquivarianceRegWeight);
    }

    [Fact]
    public void Attention_IsTranslationEquivariant_InTheInputs()
    {
        var algorithm = Algorithm();
        var x = new Matrix<double>(new[,] { { 0.1, 0.5 }, { 1.2, -0.4 }, { -0.7, 0.3 } });
        var shifted = new Matrix<double>(new[,] { { 3.1, 3.5 }, { 4.2, 2.6 }, { 2.3, 3.3 } });

        var a = algorithm.ApplyRelativeSelfAttention(Reps(), x);
        var b = algorithm.ApplyRelativeSelfAttention(Reps(), shifted);

        for (int i = 0; i < 3; i++)
            for (int d = 0; d < 4; d++)
                Assert.Equal(a[i][d], b[i][d], 10);
    }

    [Fact]
    public void Attention_DependsOnTheInputs()
    {
        // The positions are the inputs, not the model's outputs: moving one input must change the result.
        var algorithm = Algorithm();
        var x = new Matrix<double>(new[,] { { 0.1, 0.5 }, { 1.2, -0.4 }, { -0.7, 0.3 } });
        var moved = new Matrix<double>(new[,] { { 0.1, 0.5 }, { 2.2, -0.4 }, { -0.7, 0.3 } });

        var a = algorithm.ApplyRelativeSelfAttention(Reps(), x);
        var b = algorithm.ApplyRelativeSelfAttention(Reps(), moved);

        bool differ = false;
        for (int i = 0; i < 3; i++)
            for (int d = 0; d < 4; d++)
                differ |= System.Math.Abs(a[i][d] - b[i][d]) > 1e-9;
        Assert.True(differ);
    }
}
