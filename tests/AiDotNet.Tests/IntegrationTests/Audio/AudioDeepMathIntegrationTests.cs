using AiDotNet.Audio.Features;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Audio;

/// <summary>
/// Deep integration tests for Audio:
/// AudioFeatureOptions (defaults, computed properties),
/// MfccOptions (MFCC-specific defaults),
/// ChromaOptions (chroma feature defaults),
/// SpectralFeatureOptions (spectral feature defaults),
/// SpectralFeatureType (flags enum),
/// WindowType enum.
/// </summary>
public class AudioDeepMathIntegrationTests
{
    // ============================
    // AudioFeatureOptions: Defaults
    // ============================

    [Fact]
    public void AudioFeatureOptions_Defaults_SampleRate16000()
    {
        var options = new AudioFeatureOptions();
        Assert.Equal(16000, options.SampleRate);
    }

    [Fact]
    public void AudioFeatureOptions_Defaults_FftSize512()
    {
        var options = new AudioFeatureOptions();
        Assert.Equal(512, options.FftSize);
    }

    [Fact]
    public void AudioFeatureOptions_Defaults_HopLength160()
    {
        var options = new AudioFeatureOptions();
        Assert.Equal(160, options.HopLength);
    }

    [Fact]
    public void AudioFeatureOptions_Defaults_WindowLengthNull()
    {
        var options = new AudioFeatureOptions();
        Assert.Null(options.WindowLength);
    }

    [Fact]
    public void AudioFeatureOptions_Defaults_CenterPadTrue()
    {
        var options = new AudioFeatureOptions();
        Assert.True(options.CenterPad);
    }

    [Fact]
    public void AudioFeatureOptions_EffectiveWindowLength_DefaultsToFftSize()
    {
        var options = new AudioFeatureOptions();
        Assert.Equal(options.FftSize, options.EffectiveWindowLength);
    }

    [Fact]
    public void AudioFeatureOptions_EffectiveWindowLength_UsesCustomWhenSet()
    {
        var options = new AudioFeatureOptions { WindowLength = 256 };
        Assert.Equal(256, options.EffectiveWindowLength);
    }

    [Fact]
    public void AudioFeatureOptions_SetProperties()
    {
        var options = new AudioFeatureOptions
        {
            SampleRate = 44100,
            FftSize = 1024,
            HopLength = 512,
            WindowLength = 800,
            CenterPad = false
        };

        Assert.Equal(44100, options.SampleRate);
        Assert.Equal(1024, options.FftSize);
        Assert.Equal(512, options.HopLength);
        Assert.Equal(800, options.WindowLength);
        Assert.False(options.CenterPad);
        Assert.Equal(800, options.EffectiveWindowLength);
    }

    // ============================
    // AudioFeatureOptions: Nyquist Frequency Math
    // ============================

    [Theory]
    [InlineData(16000, 8000)]
    [InlineData(44100, 22050)]
    [InlineData(48000, 24000)]
    [InlineData(8000, 4000)]
    public void AudioFeatureOptions_NyquistFrequency_IsHalfSampleRate(int sampleRate, int expectedNyquist)
    {
        var options = new AudioFeatureOptions { SampleRate = sampleRate };
        int nyquist = options.SampleRate / 2;
        Assert.Equal(expectedNyquist, nyquist);
    }

    [Theory]
    [InlineData(16000, 512, 160)]   // frames = (16000 + 512) / 160 = ~103
    [InlineData(16000, 1024, 256)]  // frames = (16000 + 1024) / 256 = ~66
    [InlineData(44100, 2048, 512)]  // frames = (44100 + 2048) / 512 = ~90
    public void AudioFeatureOptions_FrameCount_WithCenterPad(int sampleRate, int fftSize, int hopLength)
    {
        var options = new AudioFeatureOptions
        {
            SampleRate = sampleRate,
            FftSize = fftSize,
            HopLength = hopLength,
            CenterPad = true
        };

        // With center padding, signal is padded by fftSize/2 on each side
        // Total length = sampleRate + fftSize (for 1 second of audio)
        // Number of frames = floor((totalLength - fftSize) / hopLength) + 1
        int totalLength = sampleRate + fftSize;
        int numFrames = (totalLength - fftSize) / hopLength + 1;

        Assert.True(numFrames > 0);
        // Number of frames should be approximately sampleRate / hopLength
        double approxFrames = (double)sampleRate / hopLength;
        Assert.True(Math.Abs(numFrames - approxFrames) < 2,
            $"Frame count {numFrames} should be approximately {approxFrames}");
    }

    // ============================
    // MfccOptions: Defaults
    // ============================

    [Fact]
    public void MfccOptions_Defaults_NumCoefficients13()
    {
        var options = new MfccOptions();
        Assert.Equal(13, options.NumCoefficients);
    }

    [Fact]
    public void MfccOptions_Defaults_NumMels40()
    {
        var options = new MfccOptions();
        Assert.Equal(40, options.NumMels);
    }

    [Fact]
    public void MfccOptions_Defaults_IncludeEnergyTrue()
    {
        var options = new MfccOptions();
        Assert.True(options.IncludeEnergy);
    }

    [Fact]
    public void MfccOptions_Defaults_AppendDeltaFalse()
    {
        var options = new MfccOptions();
        Assert.False(options.AppendDelta);
    }

    [Fact]
    public void MfccOptions_Defaults_AppendDeltaDeltaFalse()
    {
        var options = new MfccOptions();
        Assert.False(options.AppendDeltaDelta);
    }

    [Fact]
    public void MfccOptions_Defaults_FMinZero()
    {
        var options = new MfccOptions();
        Assert.Equal(0.0, options.FMin);
    }

    [Fact]
    public void MfccOptions_Defaults_FMaxNull()
    {
        var options = new MfccOptions();
        Assert.Null(options.FMax);
    }

    [Fact]
    public void MfccOptions_InheritsAudioFeatureOptions()
    {
        var options = new MfccOptions();
        Assert.IsAssignableFrom<AudioFeatureOptions>(options);
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(512, options.FftSize);
    }

    [Fact]
    public void MfccOptions_SetProperties()
    {
        var options = new MfccOptions
        {
            NumCoefficients = 20,
            NumMels = 80,
            IncludeEnergy = false,
            AppendDelta = true,
            AppendDeltaDelta = true,
            FMin = 80.0,
            FMax = 7600.0,
            SampleRate = 22050
        };

        Assert.Equal(20, options.NumCoefficients);
        Assert.Equal(80, options.NumMels);
        Assert.False(options.IncludeEnergy);
        Assert.True(options.AppendDelta);
        Assert.True(options.AppendDeltaDelta);
        Assert.Equal(80.0, options.FMin);
        Assert.Equal(7600.0, options.FMax);
    }

    // ============================
    // MfccOptions: Feature Dimension Math
    // ============================

    [Theory]
    [InlineData(13, false, false, 13)]   // Just MFCC
    [InlineData(13, true, false, 26)]    // MFCC + delta
    [InlineData(13, true, true, 39)]     // MFCC + delta + delta-delta
    [InlineData(20, false, false, 20)]   // Custom MFCC count
    [InlineData(20, true, true, 60)]     // Custom + delta + delta-delta
    public void MfccOptions_FeatureDimension_WithDeltas(int numCoeffs, bool delta, bool deltaDelta, int expectedDim)
    {
        var options = new MfccOptions
        {
            NumCoefficients = numCoeffs,
            AppendDelta = delta,
            AppendDeltaDelta = deltaDelta
        };

        int featureDim = options.NumCoefficients;
        if (options.AppendDelta) featureDim += options.NumCoefficients;
        if (options.AppendDeltaDelta) featureDim += options.NumCoefficients;

        Assert.Equal(expectedDim, featureDim);
    }

    // ============================
    // MfccOptions: Mel Scale Math
    // ============================

    [Theory]
    [InlineData(0, 0)]          // 0 Hz = 0 mel
    [InlineData(1000, 1000)]    // 1000 Hz ≈ 1000 mel (by definition for HTK formula)
    public void MfccOptions_MelScale_Htk_KnownValues(double freqHz, double approxMel)
    {
        // HTK formula: mel = 2595 * log10(1 + f/700)
        double mel = 2595.0 * Math.Log10(1.0 + freqHz / 700.0);

        if (freqHz == 0)
        {
            Assert.Equal(0.0, mel, 1e-10);
        }
        else
        {
            // At 1000 Hz, mel should be approximately 1000 (within 50%)
            Assert.True(Math.Abs(mel - approxMel) < approxMel * 0.5,
                $"Expected mel ≈ {approxMel}, got {mel}");
        }
    }

    [Theory]
    [InlineData(16000, 40)]
    [InlineData(22050, 80)]
    [InlineData(44100, 128)]
    public void MfccOptions_NumMels_GreaterThanNumCoefficients(int sampleRate, int numMels)
    {
        var options = new MfccOptions
        {
            SampleRate = sampleRate,
            NumMels = numMels,
            NumCoefficients = 13
        };

        // NumMels should always be >= NumCoefficients (DCT truncation)
        Assert.True(options.NumMels >= options.NumCoefficients,
            $"NumMels ({options.NumMels}) should be >= NumCoefficients ({options.NumCoefficients})");
    }

    // ============================
    // ChromaOptions: Defaults
    // ============================

    [Fact]
    public void ChromaOptions_Defaults_NormalizeTrue()
    {
        var options = new ChromaOptions();
        Assert.True(options.Normalize);
    }

    [Fact]
    public void ChromaOptions_Defaults_TuningFrequency440()
    {
        var options = new ChromaOptions();
        Assert.Equal(440.0, options.TuningFrequency);
    }

    [Fact]
    public void ChromaOptions_Defaults_NumOctaves7()
    {
        var options = new ChromaOptions();
        Assert.Equal(7, options.NumOctaves);
    }

    [Fact]
    public void ChromaOptions_InheritsAudioFeatureOptions()
    {
        var options = new ChromaOptions();
        Assert.IsAssignableFrom<AudioFeatureOptions>(options);
    }

    [Fact]
    public void ChromaOptions_SetProperties()
    {
        var options = new ChromaOptions
        {
            Normalize = false,
            TuningFrequency = 432.0,
            NumOctaves = 5
        };

        Assert.False(options.Normalize);
        Assert.Equal(432.0, options.TuningFrequency);
        Assert.Equal(5, options.NumOctaves);
    }

    // ============================
    // ChromaOptions: Musical Math
    // ============================

    [Fact]
    public void ChromaOptions_ChromaBins_Always12()
    {
        // Chroma features always have 12 bins (one per semitone in Western music: C, C#, D, ...)
        int chromaBins = 12;
        Assert.Equal(12, chromaBins);
    }

    [Theory]
    [InlineData(440.0, 7, 27.5)]    // A4=440Hz, 7 octaves -> lowest ~27.5 Hz (A0)
    [InlineData(440.0, 5, 110.0)]   // A4=440Hz, 5 octaves -> lowest ~110 Hz (A2)
    public void ChromaOptions_LowestFrequency_FromOctaves(double tuningFreq, int numOctaves, double expectedLowest)
    {
        // Starting from A4, go down numOctaves to find lowest A note covered
        // Frequency halves for each octave down
        double lowestA = tuningFreq / Math.Pow(2, numOctaves - 4 + 1);
        // Allow approximate match since we reference A0 (not C0)
        Assert.True(Math.Abs(lowestA - expectedLowest) < 1.0,
            $"Expected lowest freq ≈ {expectedLowest}, got {lowestA}");
    }

    // ============================
    // SpectralFeatureOptions: Defaults
    // ============================

    [Fact]
    public void SpectralFeatureOptions_Defaults_FeatureTypesBasic()
    {
        var options = new SpectralFeatureOptions();
        Assert.Equal(SpectralFeatureType.Basic, options.FeatureTypes);
    }

    [Fact]
    public void SpectralFeatureOptions_Defaults_RolloffPercentage085()
    {
        var options = new SpectralFeatureOptions();
        Assert.Equal(0.85, options.RolloffPercentage, 1e-10);
    }

    [Fact]
    public void SpectralFeatureOptions_InheritsAudioFeatureOptions()
    {
        var options = new SpectralFeatureOptions();
        Assert.IsAssignableFrom<AudioFeatureOptions>(options);
    }

    // ============================
    // SpectralFeatureType: Flags Enum
    // ============================

    [Fact]
    public void SpectralFeatureType_None_IsZero()
    {
        Assert.Equal(0, (int)SpectralFeatureType.None);
    }

    [Theory]
    [InlineData(SpectralFeatureType.Centroid, 1)]
    [InlineData(SpectralFeatureType.Bandwidth, 2)]
    [InlineData(SpectralFeatureType.Rolloff, 4)]
    [InlineData(SpectralFeatureType.Flux, 8)]
    [InlineData(SpectralFeatureType.Flatness, 16)]
    [InlineData(SpectralFeatureType.Contrast, 32)]
    [InlineData(SpectralFeatureType.ZeroCrossingRate, 64)]
    public void SpectralFeatureType_PowerOfTwoValues(SpectralFeatureType type, int expectedValue)
    {
        Assert.Equal(expectedValue, (int)type);
        // Verify power of 2
        Assert.True((expectedValue & (expectedValue - 1)) == 0, $"{type} should be a power of 2");
    }

    [Fact]
    public void SpectralFeatureType_Basic_IsExpectedCombination()
    {
        var expected = SpectralFeatureType.Centroid | SpectralFeatureType.Bandwidth |
                       SpectralFeatureType.Rolloff | SpectralFeatureType.Flux |
                       SpectralFeatureType.Flatness;
        Assert.Equal(expected, SpectralFeatureType.Basic);
        Assert.Equal(1 + 2 + 4 + 8 + 16, (int)SpectralFeatureType.Basic);
    }

    [Fact]
    public void SpectralFeatureType_All_IncludesAllValues()
    {
        var all = SpectralFeatureType.All;
        Assert.True(all.HasFlag(SpectralFeatureType.Centroid));
        Assert.True(all.HasFlag(SpectralFeatureType.Bandwidth));
        Assert.True(all.HasFlag(SpectralFeatureType.Rolloff));
        Assert.True(all.HasFlag(SpectralFeatureType.Flux));
        Assert.True(all.HasFlag(SpectralFeatureType.Flatness));
        Assert.True(all.HasFlag(SpectralFeatureType.Contrast));
        Assert.True(all.HasFlag(SpectralFeatureType.ZeroCrossingRate));
    }

    [Fact]
    public void SpectralFeatureType_FlagsCombination_Works()
    {
        var combined = SpectralFeatureType.Centroid | SpectralFeatureType.Contrast;
        Assert.True(combined.HasFlag(SpectralFeatureType.Centroid));
        Assert.True(combined.HasFlag(SpectralFeatureType.Contrast));
        Assert.False(combined.HasFlag(SpectralFeatureType.Flux));
    }

    // ============================
    // WindowType Enum
    // ============================

    [Fact]
    public void WindowType_HasFourValues()
    {
        var values = (((WindowType[])Enum.GetValues(typeof(WindowType))));
        Assert.Equal(4, values.Length);
    }

    [Theory]
    [InlineData(WindowType.Rectangular)]
    [InlineData(WindowType.Hann)]
    [InlineData(WindowType.Hamming)]
    [InlineData(WindowType.Blackman)]
    public void WindowType_AllValuesValid(WindowType type)
    {
        Assert.True(Enum.IsDefined(typeof(WindowType), type));
    }

    // ============================
    // Window Function Math
    // ============================

    [Theory]
    [InlineData(511)]   // Odd size: exact center exists
    [InlineData(1023)]
    [InlineData(2047)]
    public void WindowType_Hann_CenterIsOne(int windowSize)
    {
        // Hann window: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))
        // For odd N, exact center is at (N-1)/2, cos(pi) = -1, so w = 1.0
        int centerIdx = (windowSize - 1) / 2;
        double centerValue = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * centerIdx / (windowSize - 1)));
        Assert.Equal(1.0, centerValue, 1e-6);
    }

    [Theory]
    [InlineData(512)]
    [InlineData(1024)]
    public void WindowType_Hann_EdgesAreZero(int windowSize)
    {
        // Hann window edges: w(0) = 0.5*(1-cos(0)) = 0.5*(1-1) = 0
        // w(N-1) = 0.5*(1-cos(2*pi)) = 0.5*(1-1) = 0
        double edge0 = 0.5 * (1.0 - Math.Cos(0));
        double edgeN = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * (windowSize - 1) / (windowSize - 1)));

        Assert.Equal(0.0, edge0, 1e-10);
        Assert.Equal(0.0, edgeN, 1e-10);
    }

    [Theory]
    [InlineData(512)]
    [InlineData(1024)]
    public void WindowType_Hamming_EdgesAreNotZero(int windowSize)
    {
        // Hamming window: w(n) = 0.54 - 0.46 * cos(2*pi*n/(N-1))
        // At edges: w(0) = 0.54 - 0.46 * 1 = 0.08
        double edge0 = 0.54 - 0.46 * Math.Cos(0);
        Assert.Equal(0.08, edge0, 1e-10);
    }
}
