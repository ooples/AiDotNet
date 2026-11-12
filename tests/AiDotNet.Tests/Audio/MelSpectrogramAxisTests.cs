using System;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

public class MelSpectrogramAxisTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OutputPreservesBatchAndEndsWithFramesThenMelBins(bool batched, bool offsetCompression)
    {
        var extractor = new MelSpectrogram<double>(sampleRate: 16000, nFft: 400, hopLength: 160,
            nMels: 8, logMel: true, logOffset: offsetCompression ? .01 : (double?)null);
        var waveform = new Tensor<double>(batched ? new[] { 2, 6400 } : new[] { 6400 });
        for (int i = 0; i < waveform.Length; i++) waveform[i] = .1 * Math.Sin(.03 * i);
        var mel = extractor.Forward(waveform);
        Assert.Equal(batched ? 3 : 2, mel.Rank);
        if (batched) Assert.Equal(2, mel.Shape[0]);
        Assert.Equal(8, mel.Shape[^1]);
        Assert.True(mel.Shape[^2] > 8);
    }
}
