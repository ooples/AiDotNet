using System.Reflection;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;
using static AiDotNet.Tests.UnitTests.VisionLanguage.JointVisionLanguageStateTests;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class JointVisionLanguageEncodingTests
{
    public JointVisionLanguageEncodingTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void JointStates_UseBothImageAndInstruction_WithoutMutatingInputs()
    {
        using var model = new SmallLlava();
        model.SetTrainingMode(false);
        var image = Values(new[] { 2, 3, 4, 4 });
        var queries = Values(new[] { 2, 8 });
        var imageBefore = image.ToArray();
        var queriesBefore = queries.ToArray();
        var first = model.EncodeJointHiddenStates(image, new[] { 1, 2, 3 }, queries);
        var repeated = model.EncodeJointHiddenStates(image, new[] { 1, 2, 3 }, queries);
        Assert.Equal(new[] { 2, 10, 8 }, first.Shape.ToArray());
        Assert.Equal(first.ToArray(), repeated.ToArray());
        var changedInstruction = model.EncodeJointHiddenStates(image, new[] { 3, 2, 1 }, queries);
        var changedImage = Values(new[] { 2, 3, 4, 4 });
        changedImage[3] += 0.37f;
        var changedVisual = model.EncodeJointHiddenStates(changedImage, new[] { 1, 2, 3 }, queries);
        AssertQueryDifference(first, changedInstruction);
        AssertQueryDifference(first, changedVisual);
        Assert.Equal(imageBefore, image.ToArray());
        Assert.Equal(queriesBefore, queries.ToArray());
    }

    [Fact]
    public void JointStates_ConnectImageQueriesAndStandalonePositionsToRealGradients()
    {
        using var model = new SmallLlava();
        var image = Values(new[] { 2, 3, 4, 4 });
        var queries = Values(new[] { 2, 8 });
        var parameters = new[] { image, queries, Field(model, "_visionClsToken"),
            Field(model, "_visionPositionalEmbeddings"), Field(model, "_textPositionalEmbeddings") };
        using var tape = new GradientTape<float>();
        var joint = model.EncodeJointHiddenStates(image, new[] { 1, 2, 3 }, queries);
        var loss = model.WeightedSum(joint);
        var gradients = tape.ComputeGradients(loss, parameters);
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient));
            Assert.NotNull(gradient);
            Assert.Equal(parameter.Shape.ToArray(), gradient.Shape.ToArray());
            Assert.Contains(gradient.ToArray(), value => Math.Abs(value) > 1e-8f);
            Assert.All(gradient.ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        }
    }

    [Fact]
    public void DeclaredNativeOwner_ExposesStandaloneLiveTensorsAfterItsFirstJointForward()
    {
        using var model = new SmallLlava();
        model.EncodeJointHiddenStates(Values(new[] { 3, 4, 4 }), new[] { 1, 2, 3 });
        var parameters = Collect(model);
        foreach (var name in new[] { "_visionClsToken", "_visionPositionalEmbeddings", "_textPositionalEmbeddings" })
            Assert.Contains(parameters, parameter => ReferenceEquals(parameter, Field(model, name)));
        Assert.Equal(parameters.Count, parameters.Distinct().Count());
    }

    [Fact]
    public void TextPositionAddition_HasExactGradientForUsedRowsOnly()
    {
        using var model = new SmallLlava();
        var positions = Field(model, "_textPositionalEmbeddings");
        var embed = typeof(LLaVANeuralNetwork<float>).GetMethod("EmbedTextTokens", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(embed);
        using var tape = new GradientTape<float>();
        var output = Assert.IsType<Tensor<float>>(embed.Invoke(model, new object[] { new List<int> { 1, 2, 3 } }));
        var gradients = tape.ComputeGradients(model.WeightedSum(output), new[] { positions });
        Assert.True(gradients.TryGetValue(positions, out var gradient));
        Assert.NotNull(gradient);
        var expected = new float[positions.Length];
        for (int i = 0; i < 3 * 8; i++) expected[i] = (i + 1) * 0.007f;
        Assert.Equal(expected, gradient.ToArray());
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(64)]
    public void JointStates_RejectIdsOutsideTheConfiguredVocabulary(int token)
    {
        using var model = new SmallLlava();
        var error = Assert.Throws<ArgumentOutOfRangeException>(() =>
            model.EncodeJointHiddenStates(Values(new[] { 3, 4, 4 }), new[] { token }));
        Assert.Equal("tokenIds", error.ParamName);
    }

    [Fact]
    public void JointStates_EnforceCombinedSequenceLimitAtTheExactBoundary()
    {
        using var model = new SmallLlava();
        var image = Values(new[] { 3, 4, 4 });
        var queries = Values(new[] { 2, 8 });
        var accepted = model.EncodeJointHiddenStates(image, Enumerable.Repeat(1, 57).ToArray(), queries);
        Assert.Equal(new[] { 64, 8 }, accepted.Shape.ToArray());
        var error = Assert.Throws<ArgumentException>(() =>
            model.EncodeJointHiddenStates(image, Enumerable.Repeat(1, 58).ToArray(), queries));
        Assert.Equal("tokenIds", error.ParamName);
    }

    private static Tensor<float> Field(SmallLlava model, string name)
    {
        var field = typeof(LLaVANeuralNetwork<float>).GetField(name, BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(field);
        return Assert.IsType<Tensor<float>>(field.GetValue(model));
    }

    private static void AssertQueryDifference(Tensor<float> before, Tensor<float> after)
    {
        // Compare the trailing joint states, not the image/text input rows that changed directly.
        double difference = 0;
        for (int token = 8; token < 10; token++)
            for (int feature = 0; feature < 8; feature++)
                difference += Math.Abs(before[0, token, feature] - after[0, token, feature]);
        Assert.True(difference > 1e-6, $"Trailing query states did not respond to the changed modality: {difference}.");
    }
}
