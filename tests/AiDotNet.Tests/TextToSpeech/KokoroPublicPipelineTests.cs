using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class KokoroPublicPipelineTests
{
    [Fact(Timeout = 300000)]
    public async Task SeededNativeTraining_PreservesPredictionsAcrossCloneAndFreshRestoration()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        using var model = KokoroNativeFixture.Create<float>();
        var input = new Tensor<float>(new[] { 4 }, new Vector<float>(new[] { 12f, 24f, 36f, 48f }));
        var initial = model.Predict(input).ToArray();
        Assert.Equal(initial, model.Predict(input).ToArray());
        var target = new Tensor<float>(new[] { initial.Length }, new Vector<float>(Enumerable.Range(0, initial.Length)
            .Select(index => (float)(.1 * Math.Sin(index * .13))).ToArray()));
        for (int step = 0; step < 3; step++) model.Train(input, target);
        var trained = model.Predict(input).ToArray();
        Assert.Contains(initial.Zip(trained, (before, after) => Math.Abs(before - after)), difference => difference > 1e-8);
        Assert.Equal(trained, model.Predict(input).ToArray());
        using (var copy = (Kokoro<float>)model.DeepCopy())
        {
            Assert.Equal(model.ParameterCount, copy.ParameterCount);
            Assert.Equal(trained, copy.Predict(input).ToArray());
        }
        var state = model.Serialize();
        using var restored = KokoroNativeFixture.Create<float>();
        restored.Deserialize(state);
        Assert.Equal(model.ParameterCount, restored.ParameterCount);
        Assert.Equal(trained, restored.Predict(input).ToArray());
    }
}
