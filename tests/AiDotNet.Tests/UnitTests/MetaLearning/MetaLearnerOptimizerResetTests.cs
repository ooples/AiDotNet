using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// A meta-learner clears the state of the optimizers it was given whenever it starts over: on <c>Reset</c> and when
/// its meta-model is replaced. The checks once tested the optimizers for the data-loader <c>IResettable</c>
/// interface, which no optimizer implements, so neither reset ever reached them.
/// </summary>
public class MetaLearnerOptimizerResetTests
{
    private sealed class CountingAdam : AdamOptimizer<double, Matrix<double>, Tensor<double>>
    {
        public CountingAdam() : base(null) { }

        public int Resets { get; private set; }

        public override void Reset()
        {
            Resets++;
            base.Reset();
        }
    }

    private static (ANILAlgorithm<double, Matrix<double>, Tensor<double>> Learner, CountingAdam Meta, CountingAdam Inner) Create()
    {
        var meta = new CountingAdam();
        var inner = new CountingAdam();
        var options = new ANILOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(3, 3))
        {
            NumClasses = 2,
            FeatureDimension = 3,
            MetaOptimizer = meta,
            InnerOptimizer = inner,
        };
        return (new ANILAlgorithm<double, Matrix<double>, Tensor<double>>(options), meta, inner);
    }

    [Fact]
    public void Reset_ResetsTheSuppliedOptimizers()
    {
        var (learner, meta, inner) = Create();
        int metaBefore = meta.Resets, innerBefore = inner.Resets;

        learner.Reset();

        Assert.Equal(metaBefore + 1, meta.Resets);
        Assert.Equal(innerBefore + 1, inner.Resets);
    }

    [Fact]
    public void SetMetaModel_ResetsTheSuppliedOptimizers()
    {
        var (learner, meta, inner) = Create();
        int metaBefore = meta.Resets, innerBefore = inner.Resets;

        learner.SetMetaModel(new LinearEmbeddingModel(3, 3));

        Assert.Equal(metaBefore + 1, meta.Resets);
        Assert.Equal(innerBefore + 1, inner.Resets);
    }
}
