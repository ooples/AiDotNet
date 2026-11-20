using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NER;

public class WordCharEmbeddingLayerTests
{
    [Fact]
    public void PaddedWordsKeepTheirCharacterLayoutAndGradientsSeparate()
    {
        const int wordDim = 3, charDim = 2, hidden = 4, vocabulary = 6;
        using var layer = new WordCharEmbeddingLayer<double>(
            vocabulary, wordDim, vocabulary, charDim, hidden, sequenceLength: 3, maxWordLength: 3);
        var input = new Tensor<double>(new double[]
        {
            2, 2, 3, 0,
            3, 4, 0, 0,
            0, 0, 0, 0,
        }, [3, 4]);

        var together = layer.Forward(input).ToArray();
        Assert.Equal(3 * (wordDim + hidden), together.Length);
        // The character recurrence runs independently per word; batching words must not
        // move a character into another word or turn a padded slot into an embedding.
        for (int row = 0; row < 2; row++)
        {
            layer.ResetState();
            var word = new Tensor<double>(input.ToArray().Skip(row * 4).Take(4).ToArray(), [1, 4]);
            var alone = layer.Forward(word).ToArray();
            for (int feature = 0; feature < alone.Length; feature++)
                Assert.Equal(alone[feature], together[row * alone.Length + feature], 10);
        }
        Assert.All(together.Skip(2 * (wordDim + hidden)), value => Assert.Equal(0.0, value));

        var characterEmbedding = layer.GetSubLayers().OfType<DenseLayer<double>>()
            .Single(child => !ReferenceEquals(child, layer.WordEmbedding));
        var weights = characterEmbedding.GetTrainableParameters().Single(parameter => parameter.Rank == 2);
        layer.ResetState();
        using var tape = new GradientTape<double>();
        var output = layer.Forward(input);
        var engine = AiDotNetEngine.Current;
        // Only the first word's character features contribute to this objective.
        var firstCharacterFeatures = engine.TensorSlice(output, [0, wordDim], [1, hidden]);
        var loss = engine.ReduceSum(engine.TensorMultiply(firstCharacterFeatures, firstCharacterFeatures), null);
        var gradients = tape.ComputeGradients(loss, new[] { weights });
        Assert.True(gradients.TryGetValue(weights, out var gradient), "The character embedding must stay on the tape.");
        Assert.All(gradient!.ToArray(), value => Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        foreach (int used in new[] { 2, 3 })
            Assert.True(Enumerable.Range(0, charDim).Any(column => gradient[used, column] != 0.0),
                $"No gradient reached character {used} of the first word.");
        foreach (int unused in new[] { 0, 1, 4, 5 })
            for (int column = 0; column < charDim; column++)
                Assert.Equal(0.0, gradient[unused, column]);
    }
}
