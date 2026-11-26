using System;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Audio.StableAudio;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Stable Audio's sinusoidal timing and timestep embeddings in Half against a double reference, and the RNN-T
/// transcription duration for a batched waveform.
/// </summary>
public class StableAudioRnntReviewTests
{
    // Half rounds each sinusoid (at most 1 in magnitude) to within 2^-11; the angles themselves must not be
    // formed in Half (spacing 0.5 near 1000, 2^-5 near 47).
    private const double HalfTolerance = 1.0 / 1024;

    private static double H(Half value) => MathHelper.GetNumericOperations<Half>().ToDouble(value);

    private static double Frequency(int i, int denominator) => Math.Pow(10000.0, -2.0 * i / denominator);

    [Fact]
    public void TimestepEmbedding_InHalf_MatchesTheDoubleSinusoids()
    {
        const int dim = 64;
        const double timestep = 0.987;
        var embedding = StableAudioModel<Half>.TimestepEmbedding(AiDotNetEngine.Current, dim, timestep);

        Assert.Equal(new[] { 1, dim }, embedding.Shape.ToArray());
        for (int i = 0; i < dim / 2; i++)
        {
            double angle = 1000 * timestep * Frequency(i, dim);
            Assert.Equal(Math.Sin(angle), H(embedding[0, i]), HalfTolerance);
            Assert.Equal(Math.Cos(angle), H(embedding[0, dim / 2 + i]), HalfTolerance);
        }
    }

    [Fact]
    public void TimingEmbedding_InHalf_MatchesTheDoubleSinusoids()
    {
        const int dim = 64;
        const double seconds = 47.0;
        var embedding = StableAudioModel<Half>.TimingEmbedding(AiDotNetEngine.Current, dim, seconds);

        int quarter = dim / 4, half = dim / 2, threeQuarter = 3 * dim / 4;
        for (int i = 0; i < quarter; i++)
        {
            double f = Frequency(i, half);
            Assert.Equal(Math.Sin(seconds * f), H(embedding[0, i]), HalfTolerance);
            Assert.Equal(Math.Cos(seconds * f), H(embedding[0, quarter + i]), HalfTolerance);
            // The span is start (0) plus end (the duration).
            Assert.Equal(Math.Sin(seconds * f), H(embedding[0, half + i]), HalfTolerance);
            Assert.Equal(Math.Cos(seconds * f), H(embedding[0, threeQuarter + i]), HalfTolerance);
        }
    }

    private static RNNTransducer<double> SmallTransducer()
    {
        var options = new RNNTransducerOptions
        {
            NumMels = 16,
            EncoderDim = 16,
            NumEncoderLayers = 1,
            NumEncoderHeads = 2,
            PredictionDim = 8,
            NumPredictionLayers = 1,
            EmbeddingDim = 8,
            JointDim = 8,
            VocabSize = 30,
            DropoutRate = 0.0,
        };
        return new RNNTransducer<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: options.NumMels, outputSize: options.VocabSize), options);
    }

    private static Tensor<double> Waveform(int rows, int samples)
    {
        var data = new double[rows * samples];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = 0.3 * Math.Sin(2 * Math.PI * 440.0 * (i % samples) / 16000.0) + 0.01 * (i / samples);
        }

        return rows == 1 ? new Tensor<double>(data, new[] { samples }) : new Tensor<double>(data, new[] { rows, samples });
    }

    [Fact]
    public void Transcribe_BatchedWaveform_ReportsTheDurationOfItsTimeAxisAndDecodesTheFirstRow()
    {
        var model = SmallTransducer();
        var batch = Waveform(2, 3200);
        var firstRow = Waveform(1, 3200);

        var batched = model.Transcribe(batch, includeTimestamps: true);
        var single = model.Transcribe(firstRow, includeTimestamps: true);

        // 3200 samples at 16 kHz; the batch axis (2) is not time.
        Assert.Equal(0.2, batched.DurationSeconds, 12);
        Assert.Equal(0.2, single.DurationSeconds, 12);
        Assert.Equal(single.Text, batched.Text);
        foreach (var segment in batched.Segments)
        {
            Assert.Equal(0.2, segment.EndTime, 12);
        }
    }
}
