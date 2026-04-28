using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Engines;

internal static class CodeTransformerLayerFactory
{
    internal static void AddEmbeddingAndPosition<T>(IList<ILayer<T>> layers, CodeSynthesisArchitecture<T> architecture)
    {
        layers.Add(new EmbeddingLayer<T>(
            vocabularySize: architecture.VocabularySize,
            embeddingDimension: architecture.ModelDimension));

        if (architecture.UsePositionalEncoding)
        {
            layers.Add(new PositionalEncodingLayer<T>(architecture.MaxSequenceLength, architecture.ModelDimension));
        }

        AddDropoutIfEnabled(layers, architecture);
    }

    internal static void AddGraphConvolutionIfEnabled<T>(IList<ILayer<T>> layers, CodeSynthesisArchitecture<T> architecture)
    {
        if (!architecture.UseDataFlow)
        {
            return;
        }

        layers.Add(new GraphConvolutionalLayer<T>(
            architecture.ModelDimension,
            architecture.ModelDimension,
            (IActivationFunction<T>)new IdentityActivation<T>()));
    }

    internal static void AddEncoderBlocks<T>(IList<ILayer<T>> layers, CodeSynthesisArchitecture<T> architecture)
    {
        for (int i = 0; i < architecture.NumEncoderLayers; i++)
        {
            layers.Add(new MultiHeadAttentionLayer<T>(architecture.NumHeads, (architecture.ModelDimension) / (architecture.NumHeads), 
                activationFunction: new IdentityActivation<T>()));

            layers.Add(new LayerNormalizationLayer<T>());

            layers.Add(new DenseLayer<T>(
                outputSize: architecture.FeedForwardDimension,
                activationFunction: new GELUActivation<T>()));

            layers.Add(new DenseLayer<T>(
                outputSize: architecture.ModelDimension,
                activationFunction: new IdentityActivation<T>()));

            layers.Add(new LayerNormalizationLayer<T>());
            AddDropoutIfEnabled(layers, architecture);
        }
    }

    internal static void AddDecoderBlocks<T>(IList<ILayer<T>> layers, CodeSynthesisArchitecture<T> architecture)
    {
        for (int i = 0; i < architecture.NumDecoderLayers; i++)
        {
            layers.Add(new MultiHeadAttentionLayer<T>(architecture.NumHeads, (architecture.ModelDimension) / (architecture.NumHeads), 
                activationFunction: new IdentityActivation<T>()));

            layers.Add(new LayerNormalizationLayer<T>());

            layers.Add(new MultiHeadAttentionLayer<T>(architecture.NumHeads, (architecture.ModelDimension) / (architecture.NumHeads), 
                activationFunction: new IdentityActivation<T>()));

            layers.Add(new LayerNormalizationLayer<T>());

            layers.Add(new DenseLayer<T>(
                outputSize: architecture.FeedForwardDimension,
                activationFunction: new GELUActivation<T>()));

            layers.Add(new DenseLayer<T>(
                outputSize: architecture.ModelDimension,
                activationFunction: new IdentityActivation<T>()));

            layers.Add(new LayerNormalizationLayer<T>());
            AddDropoutIfEnabled(layers, architecture);
        }
    }

    internal static void AddOutputProjection<T>(IList<ILayer<T>> layers, CodeSynthesisArchitecture<T> architecture)
    {
        layers.Add(new DenseLayer<T>(
            outputSize: architecture.VocabularySize,
            activationFunction: new IdentityActivation<T>()));
    }

    private static void AddDropoutIfEnabled<T>(IList<ILayer<T>> layers, CodeSynthesisArchitecture<T> architecture)
    {
        if (architecture.DropoutRate > 0)
        {
            layers.Add(new DropoutLayer<T>(architecture.DropoutRate));
        }
    }
}

