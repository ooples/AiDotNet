using System;
using System.IO;
using System.Text;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Diffusion;

/// <summary>
/// Input validation and compatibility contracts of the diffusion components: a U-ViT time embedding must be shared or
/// per sample, E-MMDiT keeps the constructor earlier releases bound to, and a VAE checkpoint of an older layer layout
/// is refused by name.
/// </summary>
public class DiffusionContractTests
{
    [Fact]
    public void UViT_refuses_time_embeddings_that_are_neither_shared_nor_per_sample()
    {
        var predictor = new UViTNoisePredictor<double>(inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 4,
            patchSize: 2, contextDim: 32, latentSpatialSize: 8, seed: 1);
        var sample = new Tensor<double>(new[] { 3, 4, 8, 8 });

        var ex = Assert.Throws<ArgumentException>(() =>
            predictor.PredictNoiseWithEmbedding(sample, new Tensor<double>(new[] { 2, predictor.TimeEmbeddingDim })));
        Assert.Contains("one per sample (3)", ex.Message);

        // One shared embedding, and one per sample, are both accepted.
        Assert.Equal(sample.Shape.ToArray(),
            predictor.PredictNoiseWithEmbedding(sample, new Tensor<double>(new[] { 1, predictor.TimeEmbeddingDim })).Shape.ToArray());
        Assert.Equal(sample.Shape.ToArray(),
            predictor.PredictNoiseWithEmbedding(sample, new Tensor<double>(new[] { 3, predictor.TimeEmbeddingDim })).Shape.ToArray());
    }

    [Fact]
    public void EMMDiT_keeps_its_three_parameter_constructor()
    {
        // Binary compatibility: the constructor earlier releases compiled against still exists.
        Assert.NotNull(typeof(EMMDiTPredictor<double>).GetConstructor(new[] { typeof(int), typeof(int), typeof(int?) }));
    }

    [Fact]
    public void A_checkpoint_of_an_older_VAE_layout_is_refused_by_name()
    {
        var vae = new ImprovedVideoVAE<double>(inputChannels: 3, latentChannels: 4, baseChannels: 8, seed: 1);
        using var current = new MemoryStream();
        vae.SaveState(current);

        // The same header and weights in format 1, which predates revisions: revision 1, the 16x layout.
        var bytes = current.ToArray();
        using var legacy = new MemoryStream();
        using (var writer = new BinaryWriter(legacy, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(1);
            writer.Write(bytes, 8, bytes.Length - 8);
        }
        legacy.Position = 0;

        var ex = Assert.Throws<InvalidOperationException>(() => vae.LoadState(legacy));
        Assert.Contains("revision 1", ex.Message);
        Assert.Contains("revision 2", ex.Message);

        // A checkpoint of the current layout loads.
        current.Position = 0;
        var copy = new ImprovedVideoVAE<double>(inputChannels: 3, latentChannels: 4, baseChannels: 8, seed: 2);
        copy.LoadState(current);
        Assert.Equal(vae.GetParameters().ToArray(), copy.GetParameters().ToArray());
    }
}
