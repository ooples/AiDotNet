using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.UnitTests.MetaLearning.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

public sealed class MetaLearnerParameterOwnershipTests
{
    private static MAMLAlgorithm<double, Tensor<double>, Tensor<double>> Learner(SimpleMockModel model)
        => new(new MAMLOptions<double, Tensor<double>, Tensor<double>>(model));

    [Fact]
    public void DefaultOwnership_ExposesTheInnerModelExactlyOnce_AndWritesItsStorage()
    {
        var model = new SimpleMockModel(3);
        model.SetParameters(new Vector<double>(new[] { 1.25, -2.5, 3.75 }));
        var learner = Learner(model);

        Assert.Equal(3, learner.ParameterCount);
        Assert.Equal(model.GetParameters().ToArray(), learner.GetParameters().ToArray());
        Assert.Equal(3, learner.ParameterCount);

        var updated = new Vector<double>(new[] { -4.0, 5.0, 6.0 });
        learner.SetParameters(updated);
        Assert.Equal(updated.ToArray(), model.GetParameters().ToArray());

        model.SetParameters(new Vector<double>(new[] { 7.0, 8.0, -9.0 }));
        Assert.Equal(model.GetParameters().ToArray(), learner.GetParameters().ToArray());
    }

    [Fact]
    public void SetMetaModel_RebindsCountReadsAndWrites_ToTheReplacementOnly()
    {
        var original = new SimpleMockModel(3);
        var learner = Learner(original);
        var originalParameters = original.GetParameters().ToArray();
        // Populate the lazy registry before replacing the model with a different shape.
        Assert.Equal(3, learner.GetParameters().Length);
        Assert.Equal(3, learner.ParameterCount);

        var replacement = new SimpleMockModel(2);
        replacement.SetParameters(new Vector<double>(new[] { 10.0, -11.0 }));
        learner.SetMetaModel(replacement);

        Assert.Same(replacement, learner.BaseModel);
        Assert.Equal(2, learner.ParameterCount);
        Assert.Equal(replacement.GetParameters().ToArray(), learner.GetParameters().ToArray());
        var updated = new Vector<double>(new[] { -12.0, 13.0 });
        learner.SetParameters(updated);
        Assert.Equal(updated.ToArray(), replacement.GetParameters().ToArray());
        Assert.Equal(originalParameters, original.GetParameters().ToArray());
    }
}
