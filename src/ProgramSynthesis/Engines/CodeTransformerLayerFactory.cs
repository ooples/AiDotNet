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

        // GraphConvolutionalLayer.Forward requires an adjacency matrix (Kipf &
        // Welling 2017 — a graph convolution is undefined without a graph). The
        // code-synthesis encoder runs on a plain token stream that may carry no
        // explicit data-flow edges, so it opts into the self-loop identity
        // fallback (implicitIdentityWhenUnset: true). A caller with a real
        // data-flow graph (GraphCodeBERT, Guo et al. 2020) can still override it
        // via SetAdjacencyMatrix. The flag is set at construction, so it survives
        // Clone (which re-runs the layer constructors).
        layers.Add(new GraphConvolutionalLayer<T>(
            architecture.ModelDimension,
            architecture.ModelDimension,
            (IActivationFunction<T>)new IdentityActivation<T>(),
            implicitIdentityWhenUnset: true));
    }

    internal static void AddEncoderBlocks<T>(IList<ILayer<T>> layers, CodeSynthesisArchitecture<T> architecture)
    {
        // BERT-class encoder (Devlin et al. 2018; CodeBERT, Feng et al. 2020): each
        // layer is a RESIDUAL Transformer encoder block — multi-head self-attention
        // with a residual skip + LayerNorm, then a position-wise GELU feed-forward
        // with a residual skip + LayerNorm (Vaswani et al. 2017, §3). The previous
        // hand-stacked MHA→LN→FFN→LN sequence had NO residual connections, so
        // gradients could not flow through the depth and Training_ShouldReduceLoss
        // failed (loss flat / drifting up). TransformerEncoderBlock implements the
        // skip connections; GELU keeps the FFN faithful to BERT/CodeBERT.
        for (int i = 0; i < architecture.NumEncoderLayers; i++)
        {
            layers.Add(new TransformerEncoderBlock<T>(
                hiddenSize: architecture.ModelDimension,
                numHeads: architecture.NumHeads,
                ffnDim: architecture.FeedForwardDimension,
                dropoutRate: architecture.DropoutRate,
                ffnActivation: new GELUActivation<T>()));
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

