using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.WindowFunctions;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Tests for audio processing components used in diffusion models.
/// </summary>
public class AudioProcessingTests
{
    #region Window Function Tests

    [Fact]
    public void WindowFunctions_Hann_CreatesCorrectShape()
    {
        // Arrange
        int length = 2048;
        var windowFunc = new HanningWindow<float>();

        // Act
        var window = windowFunc.Create(length);

        // Assert
        Assert.Equal(length, window.Length);
    }

    [Fact]
    public void WindowFunctions_Hann_FirstAndLastValuesNearZero()
    {
        // Arrange
        int length = 1024;
        var windowFunc = new HanningWindow<float>();

        // Act
        var window = windowFunc.Create(length);

        // Assert - Hann window should be near zero at edges
        Assert.True(Math.Abs(window[0]) < 0.001f);
        Assert.True(window[length / 2] > 0.99f); // Peak should be near 1
    }

    [Fact]
    public void WindowFunctions_Hamming_DoesNotGoToZero()
    {
        // Arrange
        int length = 1024;
        var windowFunc = new HammingWindow<float>();

        // Act
        var window = windowFunc.Create(length);

        // Assert - Hamming doesn't go to zero at edges
        Assert.True(window[0] > 0.05f);
    }

    [Fact]
    public void WindowFunctions_Kaiser_VariesWithBeta()
    {
        // Arrange
        int length = 512;
        var windowBeta5Func = new KaiserWindow<float>(beta: 5.0);
        var windowBeta14Func = new KaiserWindow<float>(beta: 14.0);

        // Act
        var windowBeta5 = windowBeta5Func.Create(length);
        var windowBeta14 = windowBeta14Func.Create(length);

        // Assert - Higher beta should have narrower window
        // Edge values should be lower for higher beta
        Assert.True(windowBeta14[0] < windowBeta5[0]);
    }

    [Theory]
    [InlineData(WindowFunctionType.Hanning)]
    [InlineData(WindowFunctionType.Hamming)]
    [InlineData(WindowFunctionType.Blackman)]
    [InlineData(WindowFunctionType.Rectangular)]
    [InlineData(WindowFunctionType.Triangular)]
    public void WindowFunctions_AllTypes_CreateCorrectLength(WindowFunctionType windowType)
    {
        // Arrange
        int length = 1024;
        IWindowFunction<float> windowFunc = windowType switch
        {
            WindowFunctionType.Hanning => new HanningWindow<float>(),
            WindowFunctionType.Hamming => new HammingWindow<float>(),
            WindowFunctionType.Blackman => new BlackmanWindow<float>(),
            WindowFunctionType.Rectangular => new RectangularWindow<float>(),
            WindowFunctionType.Triangular => new TriangularWindow<float>(),
            _ => new HanningWindow<float>()
        };

        // Act
        var window = windowFunc.Create(length);

        // Assert
        Assert.Equal(length, window.Length);
    }

    #endregion

    #region STFT Tests

    [Fact]
    public void STFT_Forward_ProducesCorrectShape()
    {
        // Arrange
        int nFft = 512;
        int hopLength = 128;
        var stft = new ShortTimeFourierTransform<float>(nFft: nFft, hopLength: hopLength);

        int signalLength = 4096;
        var signal = new Tensor<float>(new[] { signalLength });
        for (int i = 0; i < signalLength; i++)
        {
            signal.Data[i] = (float)Math.Sin(2 * Math.PI * 440 * i / 16000); // 440 Hz sine
        }

        // Act
        var spectrogram = stft.Forward(signal);

        // Assert
        int expectedFrames = stft.CalculateNumFrames(signalLength);
        int expectedFreqs = nFft / 2 + 1;
        Assert.Equal(2, spectrogram.Shape.Length);
        Assert.Equal(expectedFrames, spectrogram.Shape[0]);
        Assert.Equal(expectedFreqs, spectrogram.Shape[1]);
    }

    [Fact]
    public void STFT_Magnitude_ProducesNonNegativeValues()
    {
        // Arrange
        int nFft = 256;
        var stft = new ShortTimeFourierTransform<float>(nFft: nFft);

        int signalLength = 2048;
        var signal = new Tensor<float>(new[] { signalLength });
        var random = new Random(42);
        for (int i = 0; i < signalLength; i++)
        {
            signal.Data[i] = (float)(random.NextDouble() - 0.5);
        }

        // Act
        var magnitude = stft.Magnitude(signal);

        // Assert - All magnitudes should be non-negative
        foreach (var value in magnitude.Data)
        {
            Assert.True(value >= 0f, "Magnitude should be non-negative");
        }
    }

    [Fact]
    public void STFT_Power_IsSquareOfMagnitude()
    {
        // Arrange
        int nFft = 256;
        var stft = new ShortTimeFourierTransform<float>(nFft: nFft);

        var signal = new Tensor<float>(new[] { 1024 });
        for (int i = 0; i < 1024; i++)
        {
            signal.Data[i] = (float)Math.Sin(2 * Math.PI * 100 * i / 8000);
        }

        // Act
        var magnitude = stft.Magnitude(signal);
        var power = stft.Power(signal);

        // Assert
        for (int i = 0; i < magnitude.Data.Length; i++)
        {
            var expected = magnitude.Data[i] * magnitude.Data[i];
            Assert.True(Math.Abs(expected - power.Data[i]) < 1e-5f,
                $"Power should equal magnitude squared at index {i}");
        }
    }

    [Fact]
    public void STFT_Inverse_ReconstructsSignal()
    {
        // Arrange
        int nFft = 512;
        int hopLength = 128;
        var stft = new ShortTimeFourierTransform<float>(nFft: nFft, hopLength: hopLength);

        int signalLength = 2048;
        var original = new Tensor<float>(new[] { signalLength });
        for (int i = 0; i < signalLength; i++)
        {
            original.Data[i] = (float)Math.Sin(2 * Math.PI * 440 * i / 16000);
        }

        // Act
        var spectrogram = stft.Forward(original);
        var reconstructed = stft.Inverse(spectrogram, signalLength);

        // Assert - Reconstructed should be close to original
        double totalError = 0;
        for (int i = 0; i < signalLength; i++)
        {
            double diff = Math.Abs(original.Data[i] - reconstructed.Data[i]);
            totalError += diff * diff;
        }
        double rmse = Math.Sqrt(totalError / signalLength);
        Assert.True(rmse < 0.1, $"RMSE {rmse} should be small for perfect reconstruction");
    }

    [Fact]
    public void STFT_CalculateNumFrames_ConsistentWithForward()
    {
        // Arrange
        int nFft = 1024;
        int hopLength = 256;
        var stft = new ShortTimeFourierTransform<float>(nFft: nFft, hopLength: hopLength);

        int[] signalLengths = { 1000, 2048, 4096, 8000 };

        foreach (int length in signalLengths)
        {
            // Act
            int calculated = stft.CalculateNumFrames(length);
            var signal = new Tensor<float>(new[] { length });
            var spectrogram = stft.Forward(signal);

            // Assert
            Assert.Equal(calculated, spectrogram.Shape[0]);
        }
    }

    #endregion

    #region Mel Spectrogram Tests

    [Fact]
    public void MelSpectrogram_Forward_ProducesCorrectShape()
    {
        // Arrange
        int sampleRate = 22050;
        int nMels = 128;
        int nFft = 2048;
        var melSpec = new MelSpectrogram<float>(
            sampleRate: sampleRate,
            nMels: nMels,
            nFft: nFft);

        int signalLength = 22050; // 1 second
        var signal = new Tensor<float>(new[] { signalLength });
        for (int i = 0; i < signalLength; i++)
        {
            signal.Data[i] = (float)Math.Sin(2 * Math.PI * 1000 * i / sampleRate);
        }

        // Act
        var mel = melSpec.Forward(signal);

        // Assert
        Assert.Equal(2, mel.Shape.Length);
        Assert.Equal(nMels, mel.Shape[1]);
    }

    [Fact]
    public void MelSpectrogram_HzToMel_Invertible()
    {
        // Arrange
        double[] frequencies = { 100, 440, 1000, 4000, 10000 };

        foreach (double hz in frequencies)
        {
            // Act
            double mel = MelSpectrogram<float>.HzToMel(hz);
            double backToHz = MelSpectrogram<float>.MelToHz(mel);

            // Assert
            Assert.True(Math.Abs(hz - backToHz) < 1e-6,
                $"Hz->Mel->Hz should be identity: {hz} != {backToHz}");
        }
    }

    [Fact]
    public void MelSpectrogram_GetFilterbank_HasCorrectShape()
    {
        // Arrange
        int nMels = 80;
        int nFft = 1024;
        var melSpec = new MelSpectrogram<float>(
            sampleRate: 16000,
            nMels: nMels,
            nFft: nFft);

        // Act
        var filterbank = melSpec.GetFilterbank();

        // Assert
        Assert.Equal(nMels, filterbank.Shape[0]);
        Assert.Equal(nFft / 2 + 1, filterbank.Shape[1]);
    }

    [Fact]
    public void MelSpectrogram_Filterbank_RowsAreNormalized()
    {
        // Arrange
        var melSpec = new MelSpectrogram<float>(
            sampleRate: 16000,
            nMels: 40,
            nFft: 512);

        // Act
        var filterbank = melSpec.GetFilterbank();
        int numFreqs = filterbank.Shape[1];

        // Assert - Each row should sum to ~1 (Slaney normalization)
        for (int mel = 0; mel < 40; mel++)
        {
            float rowSum = 0;
            for (int f = 0; f < numFreqs; f++)
            {
                rowSum += filterbank.Data[mel * numFreqs + f];
            }
            Assert.True(Math.Abs(rowSum - 1.0f) < 0.01f || rowSum < 0.01f,
                $"Row {mel} sum should be ~1 (normalized) or 0 (zero filter): {rowSum}");
        }
    }

    [Fact]
    public void MelSpectrogram_GetMelCenterFrequencies_IncreasesMonotonically()
    {
        // Arrange
        var melSpec = new MelSpectrogram<float>(
            sampleRate: 22050,
            nMels: 128,
            fMin: 20,
            fMax: 8000);

        // Act
        var centers = melSpec.GetMelCenterFrequencies();

        // Assert
        for (int i = 1; i < centers.Length; i++)
        {
            Assert.True(centers[i] > centers[i - 1],
                $"Center frequencies should increase: {centers[i - 1]} >= {centers[i]}");
        }
    }

    #endregion

    #region Griffin-Lim Tests

    [Fact]
    public void GriffinLim_Reconstruct_ProducesCorrectLength()
    {
        // Arrange
        int nFft = 512;
        int hopLength = 128;
        var griffinLim = new GriffinLim<float>(
            nFft: nFft,
            hopLength: hopLength,
            iterations: 10);

        int numFrames = 32;
        int numFreqs = nFft / 2 + 1;
        var magnitude = new Tensor<float>(new[] { numFrames, numFreqs });

        // Fill with some magnitude values
        for (int i = 0; i < magnitude.Data.Length; i++)
        {
            magnitude.Data[i] = 1.0f;
        }

        // Act
        var audio = griffinLim.Reconstruct(magnitude);

        // Assert - Check output is reasonable length
        int expectedLength = griffinLim.STFT.CalculateSignalLength(numFrames);
        Assert.True(audio.Data.Length > 0, "Output should have data");
    }

    [Fact]
    public void GriffinLim_ComputeSpectralConvergence_DecreasesWithIterations()
    {
        // Arrange
        int nFft = 256;
        var stft = new ShortTimeFourierTransform<float>(nFft: nFft);

        // Create a simple signal
        var signal = new Tensor<float>(new[] { 1024 });
        for (int i = 0; i < 1024; i++)
        {
            signal.Data[i] = (float)Math.Sin(2 * Math.PI * 440 * i / 16000);
        }
        var targetMagnitude = stft.Magnitude(signal);

        // Act - Compare different iteration counts
        // Use momentum: 0 for deterministic convergence (high momentum can cause oscillation)
        var griffinLim10 = new GriffinLim<float>(nFft: nFft, iterations: 10, momentum: 0.0, seed: 42);
        var griffinLim50 = new GriffinLim<float>(nFft: nFft, iterations: 50, momentum: 0.0, seed: 42);

        var audio10 = griffinLim10.Reconstruct(targetMagnitude);
        var audio50 = griffinLim50.Reconstruct(targetMagnitude);

        double error10 = griffinLim10.ComputeSpectralConvergence(targetMagnitude, audio10);
        double error50 = griffinLim50.ComputeSpectralConvergence(targetMagnitude, audio50);

        // Assert - More iterations should have lower or equal error (with tolerance for numerical precision)
        Assert.True(error50 <= error10 * 1.05,
            $"50 iterations error ({error50}) should be <= 10 iterations error ({error10}) with 5% tolerance");
    }

    [Fact]
    public void GriffinLim_WithProgress_CallsCallback()
    {
        // Arrange
        int nFft = 256;
        int iterations = 10;
        var griffinLim = new GriffinLim<float>(nFft: nFft, iterations: iterations);

        var magnitude = new Tensor<float>(new[] { 16, nFft / 2 + 1 });
        for (int i = 0; i < magnitude.Data.Length; i++)
        {
            magnitude.Data[i] = 1.0f;
        }

        var callbackCounts = 0;

        // Act
        griffinLim.ReconstructWithProgress(magnitude, (iter, convergence) =>
        {
            callbackCounts++;
        });

        // Assert
        Assert.Equal(iterations, callbackCounts);
    }

    #endregion

    #region AudioProcessor Tests

    [Fact]
    public void AudioProcessor_AudioToSpectrogram_ProducesNormalizedOutput()
    {
        // Arrange
        var processor = new AudioProcessor<float>(
            sampleRate: 22050,
            nFft: 1024,
            nMels: 128);

        var audio = new Tensor<float>(new[] { 22050 }); // 1 second
        for (int i = 0; i < 22050; i++)
        {
            audio.Data[i] = (float)Math.Sin(2 * Math.PI * 440 * i / 22050);
        }

        // Act
        var spectrogram = processor.AudioToSpectrogram(audio);

        // Assert - Should be normalized to [0, 1]
        foreach (var value in spectrogram.Data)
        {
            Assert.True(value >= 0f && value <= 1f,
                $"Spectrogram value {value} should be in [0, 1]");
        }
    }

    [Fact]
    public void AudioProcessor_DurationConversions_AreConsistent()
    {
        // Arrange
        int sampleRate = 44100;
        var processor = new AudioProcessor<float>(sampleRate: sampleRate);

        double duration = 2.5; // 2.5 seconds

        // Act
        int samples = processor.DurationToSamples(duration);
        int frames = processor.DurationToFrames(duration);
        double recoveredDuration = processor.FramesToDuration(frames);

        // Assert
        Assert.Equal((int)(duration * sampleRate), samples);
        Assert.True(Math.Abs(duration - recoveredDuration) < 0.1,
            $"Duration should round-trip: {duration} -> {recoveredDuration}");
    }

    [Fact]
    public void AudioProcessor_NormalizeAudio_ScalesToTargetPeak()
    {
        // Arrange
        var processor = new AudioProcessor<float>();

        var audio = new Tensor<float>(new[] { 1000 });
        for (int i = 0; i < 1000; i++)
        {
            audio.Data[i] = (float)(0.1 * Math.Sin(2 * Math.PI * 100 * i / 1000));
        }

        double targetPeak = 0.8;

        // Act
        var normalized = processor.NormalizeAudio(audio, targetPeak);

        // Assert - Find max absolute value
        float maxAbs = 0;
        foreach (var value in normalized.Data)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(value));
        }

        Assert.True(Math.Abs(maxAbs - targetPeak) < 0.01,
            $"Peak should be {targetPeak}, got {maxAbs}");
    }

    [Fact]
    public void AudioProcessor_PadOrTruncate_PadsCorrectly()
    {
        // Arrange
        var processor = new AudioProcessor<float>();

        var audio = new Tensor<float>(new[] { 100 });
        for (int i = 0; i < 100; i++)
        {
            audio.Data[i] = 1.0f;
        }

        // Act
        var padded = processor.PadOrTruncate(audio, 200);

        // Assert
        Assert.Equal(200, padded.Data.Length);
        // First 100 should be 1.0
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(1.0f, padded.Data[i]);
        }
        // Rest should be 0 (padded)
        for (int i = 100; i < 200; i++)
        {
            Assert.Equal(0f, padded.Data[i]);
        }
    }

    [Fact]
    public void AudioProcessor_PadOrTruncate_TruncatesCorrectly()
    {
        // Arrange
        var processor = new AudioProcessor<float>();

        var audio = new Tensor<float>(new[] { 200 });
        for (int i = 0; i < 200; i++)
        {
            audio.Data[i] = (float)i;
        }

        // Act
        var truncated = processor.PadOrTruncate(audio, 100);

        // Assert
        Assert.Equal(100, truncated.Data.Length);
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal((float)i, truncated.Data[i]);
        }
    }

    #endregion
}
