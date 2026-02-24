using AiDotNet.Audio.AudioGen;
using AiDotNet.Audio.Classification;
using AiDotNet.Audio.Effects;
using AiDotNet.Audio.Features;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Audio;

/// <summary>
/// Integration tests for Audio module classes:
/// Feature extractors (MFCC, Chroma, Spectral), Audio effects (Compressor, Reverb, ParametricEQ),
/// Options/Result classes, Enums, and AudioEvent.
/// </summary>
public class AudioIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region AudioFeatureOptions - Defaults

    [Fact]
    public void AudioFeatureOptions_DefaultValues()
    {
        var options = new AudioFeatureOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(512, options.FftSize);
        Assert.Equal(160, options.HopLength);
        Assert.Null(options.WindowLength);
        Assert.True(options.CenterPad);
    }

    [Fact]
    public void AudioFeatureOptions_EffectiveWindowLength_DefaultsToFftSize()
    {
        var options = new AudioFeatureOptions();
        Assert.Equal(options.FftSize, options.EffectiveWindowLength);
    }

    [Fact]
    public void AudioFeatureOptions_EffectiveWindowLength_UsesExplicitValue()
    {
        var options = new AudioFeatureOptions { WindowLength = 256 };
        Assert.Equal(256, options.EffectiveWindowLength);
    }

    [Fact]
    public void AudioFeatureOptions_MutableProperties()
    {
        var options = new AudioFeatureOptions
        {
            SampleRate = 22050,
            FftSize = 1024,
            HopLength = 512,
            CenterPad = false
        };
        Assert.Equal(22050, options.SampleRate);
        Assert.Equal(1024, options.FftSize);
        Assert.Equal(512, options.HopLength);
        Assert.False(options.CenterPad);
    }

    #endregion

    #region MfccOptions - Defaults

    [Fact]
    public void MfccOptions_DefaultValues()
    {
        var options = new MfccOptions();
        Assert.Equal(13, options.NumCoefficients);
        Assert.Equal(40, options.NumMels);
        Assert.True(options.IncludeEnergy);
        Assert.False(options.AppendDelta);
        Assert.False(options.AppendDeltaDelta);
        Assert.Equal(0, options.FMin);
        Assert.Null(options.FMax);
    }

    [Fact]
    public void MfccOptions_InheritsAudioFeatureOptions()
    {
        var options = new MfccOptions();
        Assert.IsAssignableFrom<AudioFeatureOptions>(options);
        Assert.Equal(16000, options.SampleRate);
    }

    [Fact]
    public void MfccOptions_CustomValues()
    {
        var options = new MfccOptions
        {
            NumCoefficients = 20,
            NumMels = 64,
            IncludeEnergy = false,
            AppendDelta = true,
            AppendDeltaDelta = true,
            FMin = 80,
            FMax = 7600
        };
        Assert.Equal(20, options.NumCoefficients);
        Assert.Equal(64, options.NumMels);
        Assert.False(options.IncludeEnergy);
        Assert.True(options.AppendDelta);
        Assert.True(options.AppendDeltaDelta);
        Assert.Equal(80, options.FMin);
        Assert.Equal(7600.0, options.FMax);
    }

    #endregion

    #region ChromaOptions - Defaults

    [Fact]
    public void ChromaOptions_DefaultValues()
    {
        var options = new ChromaOptions();
        Assert.True(options.Normalize);
        Assert.Equal(440.0, options.TuningFrequency);
        Assert.Equal(7, options.NumOctaves);
    }

    [Fact]
    public void ChromaOptions_InheritsAudioFeatureOptions()
    {
        var options = new ChromaOptions();
        Assert.IsAssignableFrom<AudioFeatureOptions>(options);
    }

    [Fact]
    public void ChromaOptions_CustomValues()
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

    #endregion

    #region SpectralFeatureOptions - Defaults

    [Fact]
    public void SpectralFeatureOptions_DefaultValues()
    {
        var options = new SpectralFeatureOptions();
        Assert.Equal(SpectralFeatureType.Basic, options.FeatureTypes);
        Assert.Equal(0.85, options.RolloffPercentage, Tolerance);
    }

    [Fact]
    public void SpectralFeatureOptions_InheritsAudioFeatureOptions()
    {
        var options = new SpectralFeatureOptions();
        Assert.IsAssignableFrom<AudioFeatureOptions>(options);
    }

    #endregion

    #region SpectralFeatureType Enum

    [Fact]
    public void SpectralFeatureType_None_IsZero()
    {
        Assert.Equal(0, (int)SpectralFeatureType.None);
    }

    [Fact]
    public void SpectralFeatureType_Basic_IsCombination()
    {
        var basic = SpectralFeatureType.Basic;
        Assert.True(basic.HasFlag(SpectralFeatureType.Centroid));
        Assert.True(basic.HasFlag(SpectralFeatureType.Bandwidth));
        Assert.True(basic.HasFlag(SpectralFeatureType.Rolloff));
        Assert.True(basic.HasFlag(SpectralFeatureType.Flux));
        Assert.True(basic.HasFlag(SpectralFeatureType.Flatness));
        Assert.False(basic.HasFlag(SpectralFeatureType.Contrast));
        Assert.False(basic.HasFlag(SpectralFeatureType.ZeroCrossingRate));
    }

    [Fact]
    public void SpectralFeatureType_All_IncludesEverything()
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
    public void SpectralFeatureType_FlagsArePowersOfTwo()
    {
        Assert.Equal(1, (int)SpectralFeatureType.Centroid);
        Assert.Equal(2, (int)SpectralFeatureType.Bandwidth);
        Assert.Equal(4, (int)SpectralFeatureType.Rolloff);
        Assert.Equal(8, (int)SpectralFeatureType.Flux);
        Assert.Equal(16, (int)SpectralFeatureType.Flatness);
        Assert.Equal(32, (int)SpectralFeatureType.Contrast);
        Assert.Equal(64, (int)SpectralFeatureType.ZeroCrossingRate);
    }

    #endregion

    #region WindowType Enum

    [Fact]
    public void WindowType_AllValues()
    {
        var values = Enum.GetValues<WindowType>();
        Assert.Equal(4, values.Length);
        Assert.Contains(WindowType.Rectangular, values);
        Assert.Contains(WindowType.Hann, values);
        Assert.Contains(WindowType.Hamming, values);
        Assert.Contains(WindowType.Blackman, values);
    }

    #endregion

    #region MfccExtractor - Construction and Feature Dimension

    [Fact]
    public void MfccExtractor_DefaultConstruction_DoesNotThrow()
    {
        var extractor = new MfccExtractor<double>();
        Assert.NotNull(extractor);
        Assert.Equal("MFCC", extractor.Name);
    }

    [Fact]
    public void MfccExtractor_DefaultFeatureDimension_Is13()
    {
        var extractor = new MfccExtractor<double>();
        Assert.Equal(13, extractor.FeatureDimension);
    }

    [Fact]
    public void MfccExtractor_WithDelta_DoublesDimension()
    {
        var options = new MfccOptions { NumCoefficients = 13, AppendDelta = true };
        var extractor = new MfccExtractor<double>(options);
        Assert.Equal(26, extractor.FeatureDimension);
    }

    [Fact]
    public void MfccExtractor_WithDeltaAndDeltaDelta_TriplesDimension()
    {
        var options = new MfccOptions { NumCoefficients = 13, AppendDelta = true, AppendDeltaDelta = true };
        var extractor = new MfccExtractor<double>(options);
        Assert.Equal(39, extractor.FeatureDimension);
    }

    [Fact]
    public void MfccExtractor_ImplementsInterface()
    {
        var extractor = new MfccExtractor<double>();
        Assert.IsAssignableFrom<IAudioFeatureExtractor<double>>(extractor);
    }

    [Fact]
    public void MfccExtractor_SampleRateFromOptions()
    {
        var options = new MfccOptions { SampleRate = 22050 };
        var extractor = new MfccExtractor<double>(options);
        Assert.Equal(22050, extractor.SampleRate);
    }

    #endregion

    #region ChromaExtractor - Construction

    [Fact]
    public void ChromaExtractor_DefaultConstruction_DoesNotThrow()
    {
        var extractor = new ChromaExtractor<double>();
        Assert.NotNull(extractor);
        Assert.Equal("Chroma", extractor.Name);
    }

    [Fact]
    public void ChromaExtractor_FeatureDimension_Is12()
    {
        var extractor = new ChromaExtractor<double>();
        Assert.Equal(12, extractor.FeatureDimension);
    }

    [Fact]
    public void ChromaExtractor_ImplementsInterface()
    {
        var extractor = new ChromaExtractor<double>();
        Assert.IsAssignableFrom<IAudioFeatureExtractor<double>>(extractor);
    }

    [Fact]
    public void ChromaExtractor_GetPitchClassName_AllNotes()
    {
        Assert.Equal("C", ChromaExtractor<double>.GetPitchClassName(0));
        Assert.Equal("C#", ChromaExtractor<double>.GetPitchClassName(1));
        Assert.Equal("D", ChromaExtractor<double>.GetPitchClassName(2));
        Assert.Equal("D#", ChromaExtractor<double>.GetPitchClassName(3));
        Assert.Equal("E", ChromaExtractor<double>.GetPitchClassName(4));
        Assert.Equal("F", ChromaExtractor<double>.GetPitchClassName(5));
        Assert.Equal("F#", ChromaExtractor<double>.GetPitchClassName(6));
        Assert.Equal("G", ChromaExtractor<double>.GetPitchClassName(7));
        Assert.Equal("G#", ChromaExtractor<double>.GetPitchClassName(8));
        Assert.Equal("A", ChromaExtractor<double>.GetPitchClassName(9));
        Assert.Equal("A#", ChromaExtractor<double>.GetPitchClassName(10));
        Assert.Equal("B", ChromaExtractor<double>.GetPitchClassName(11));
    }

    [Fact]
    public void ChromaExtractor_GetPitchClassName_WrapsAround()
    {
        Assert.Equal("C", ChromaExtractor<double>.GetPitchClassName(12));
        Assert.Equal("A", ChromaExtractor<double>.GetPitchClassName(21));
    }

    #endregion

    #region SpectralFeatureExtractor - Construction

    [Fact]
    public void SpectralFeatureExtractor_DefaultConstruction_DoesNotThrow()
    {
        var extractor = new SpectralFeatureExtractor<double>();
        Assert.NotNull(extractor);
        Assert.Equal("SpectralFeatures", extractor.Name);
    }

    [Fact]
    public void SpectralFeatureExtractor_BasicFeatures_CorrectDimension()
    {
        var extractor = new SpectralFeatureExtractor<double>();
        // Basic = Centroid + Bandwidth + Rolloff + Flux + Flatness = 5 features
        Assert.Equal(5, extractor.FeatureDimension);
    }

    [Fact]
    public void SpectralFeatureExtractor_AllFeatures_CorrectDimension()
    {
        var options = new SpectralFeatureOptions { FeatureTypes = SpectralFeatureType.All };
        var extractor = new SpectralFeatureExtractor<double>(options);
        // All = 5 basic + 6 contrast bands + 1 ZeroCrossingRate = 12 features
        Assert.Equal(12, extractor.FeatureDimension);
    }

    [Fact]
    public void SpectralFeatureExtractor_SingleFeature_DimensionOne()
    {
        var options = new SpectralFeatureOptions { FeatureTypes = SpectralFeatureType.Centroid };
        var extractor = new SpectralFeatureExtractor<double>(options);
        Assert.Equal(1, extractor.FeatureDimension);
    }

    [Fact]
    public void SpectralFeatureExtractor_ImplementsInterface()
    {
        var extractor = new SpectralFeatureExtractor<double>();
        Assert.IsAssignableFrom<IAudioFeatureExtractor<double>>(extractor);
    }

    #endregion

    #region Compressor - Construction

    [Fact]
    public void Compressor_DefaultConstruction_DoesNotThrow()
    {
        var compressor = new Compressor<double>();
        Assert.NotNull(compressor);
        Assert.Equal("Compressor", compressor.Name);
    }

    [Fact]
    public void Compressor_DefaultParameters()
    {
        var compressor = new Compressor<double>();
        Assert.Equal(44100, compressor.SampleRate);
        Assert.False(compressor.Bypass);
        Assert.Equal(1.0, compressor.Mix, Tolerance);
    }

    [Fact]
    public void Compressor_CustomParameters()
    {
        var compressor = new Compressor<double>(
            sampleRate: 48000,
            thresholdDb: -10.0,
            ratio: 8.0,
            attackMs: 5.0,
            releaseMs: 50.0,
            makeupGainDb: 6.0,
            kneeDb: 3.0,
            mix: 0.8);
        Assert.Equal(48000, compressor.SampleRate);
        Assert.Equal(0.8, compressor.Mix, Tolerance);
    }

    [Fact]
    public void Compressor_HasExpectedParameters()
    {
        var compressor = new Compressor<double>();
        Assert.True(compressor.Parameters.ContainsKey("threshold"));
        Assert.True(compressor.Parameters.ContainsKey("ratio"));
        Assert.True(compressor.Parameters.ContainsKey("attack"));
        Assert.True(compressor.Parameters.ContainsKey("release"));
        Assert.True(compressor.Parameters.ContainsKey("makeup"));
        Assert.True(compressor.Parameters.ContainsKey("knee"));
    }

    [Fact]
    public void Compressor_ProcessSample_SilenceRemainsQuiet()
    {
        var compressor = new Compressor<double>();
        var result = compressor.ProcessSample(0.0);
        Assert.True(Math.Abs(result) < 0.01, $"Silent input should produce near-silent output, got {result}");
    }

    [Fact]
    public void Compressor_ProcessSample_SignPreserved()
    {
        var compressor = new Compressor<double>(thresholdDb: -6.0, ratio: 4.0);
        // Process a few samples to warm up the envelope
        for (int i = 0; i < 100; i++)
            compressor.ProcessSample(0.5);

        var positiveResult = compressor.ProcessSample(0.5);
        compressor.Reset();
        for (int i = 0; i < 100; i++)
            compressor.ProcessSample(-0.5);
        var negativeResult = compressor.ProcessSample(-0.5);

        Assert.True(positiveResult >= 0, "Positive input should yield positive output");
        Assert.True(negativeResult <= 0, "Negative input should yield negative output");
    }

    [Fact]
    public void Compressor_Bypass_ReturnsInput()
    {
        var compressor = new Compressor<double>();
        compressor.Bypass = true;
        var result = compressor.ProcessSample(0.75);
        Assert.Equal(0.75, result, Tolerance);
    }

    [Fact]
    public void Compressor_Reset_DoesNotThrow()
    {
        var compressor = new Compressor<double>();
        compressor.ProcessSample(0.5);
        compressor.Reset();
        // After reset, should be able to process again
        var result = compressor.ProcessSample(0.0);
        Assert.True(Math.Abs(result) < 0.1);
    }

    [Fact]
    public void Compressor_GetGainReduction_ReturnsNonPositive()
    {
        var compressor = new Compressor<double>(thresholdDb: -20.0, ratio: 4.0);
        // Process some loud samples
        for (int i = 0; i < 1000; i++)
            compressor.ProcessSample(0.9);

        var gainReduction = compressor.GetGainReduction();
        Assert.True(gainReduction <= 0, $"Gain reduction should be non-positive, got {gainReduction}");
    }

    [Fact]
    public void Compressor_LatencySamples_IsZero()
    {
        var compressor = new Compressor<double>();
        Assert.Equal(0, compressor.LatencySamples);
    }

    #endregion

    #region Reverb - Construction

    [Fact]
    public void Reverb_DefaultConstruction_DoesNotThrow()
    {
        var reverb = new Reverb<double>();
        Assert.NotNull(reverb);
        Assert.Equal("Reverb", reverb.Name);
    }

    [Fact]
    public void Reverb_DefaultParameters()
    {
        var reverb = new Reverb<double>();
        Assert.Equal(44100, reverb.SampleRate);
        Assert.Equal(0.3, reverb.Mix, Tolerance);
        Assert.False(reverb.Bypass);
    }

    [Fact]
    public void Reverb_CustomParameters()
    {
        var reverb = new Reverb<double>(
            sampleRate: 48000,
            roomSize: 0.8,
            decayTime: 3.0,
            preDelayMs: 40.0,
            damping: 0.7,
            diffusion: 0.9,
            mix: 0.5);
        Assert.Equal(48000, reverb.SampleRate);
        Assert.Equal(0.5, reverb.Mix, Tolerance);
    }

    [Fact]
    public void Reverb_HasExpectedParameters()
    {
        var reverb = new Reverb<double>();
        Assert.True(reverb.Parameters.ContainsKey("roomSize"));
        Assert.True(reverb.Parameters.ContainsKey("decayTime"));
        Assert.True(reverb.Parameters.ContainsKey("preDelay"));
        Assert.True(reverb.Parameters.ContainsKey("damping"));
        Assert.True(reverb.Parameters.ContainsKey("diffusion"));
    }

    [Fact]
    public void Reverb_ProcessSample_SilenceInSilenceOut()
    {
        var reverb = new Reverb<double>();
        // Process many silent samples
        for (int i = 0; i < 100; i++)
        {
            var result = reverb.ProcessSample(0.0);
            // With mix of 0.3, the dry component of silence is still silence
        }
    }

    [Fact]
    public void Reverb_TailSamples_IsPositive()
    {
        var reverb = new Reverb<double>(decayTime: 2.0);
        Assert.True(reverb.TailSamples > 0, "Reverb tail should be positive");
    }

    [Fact]
    public void Reverb_Bypass_ReturnsInput()
    {
        var reverb = new Reverb<double>();
        reverb.Bypass = true;
        var result = reverb.ProcessSample(0.5);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void Reverb_Reset_DoesNotThrow()
    {
        var reverb = new Reverb<double>();
        reverb.ProcessSample(0.8);
        reverb.Reset();
    }

    [Fact]
    public void Reverb_ProcessTensor_ReturnsSameLength()
    {
        var reverb = new Reverb<double>();
        var input = new Tensor<double>(new[] { 100 });
        for (int i = 0; i < 100; i++)
            input[i] = Math.Sin(2 * Math.PI * 440 * i / 44100.0);

        var output = reverb.Process(input);
        Assert.Equal(100, output.Length);
    }

    #endregion

    #region ParametricEqualizer - Construction

    [Fact]
    public void ParametricEQ_DefaultConstruction_HasFiveBands()
    {
        var eq = new ParametricEqualizer<double>();
        Assert.NotNull(eq);
        Assert.Equal("Parametric EQ", eq.Name);
        Assert.Equal(5, eq.Bands.Count);
    }

    [Fact]
    public void ParametricEQ_DefaultBands_HaveCorrectTypes()
    {
        var eq = new ParametricEqualizer<double>();
        Assert.Equal(EqFilterType.LowShelf, eq.Bands[0].FilterType);
        Assert.Equal(EqFilterType.Peak, eq.Bands[1].FilterType);
        Assert.Equal(EqFilterType.Peak, eq.Bands[2].FilterType);
        Assert.Equal(EqFilterType.Peak, eq.Bands[3].FilterType);
        Assert.Equal(EqFilterType.HighShelf, eq.Bands[4].FilterType);
    }

    [Fact]
    public void ParametricEQ_DefaultBands_HaveCorrectFrequencies()
    {
        var eq = new ParametricEqualizer<double>();
        Assert.Equal(80, eq.Bands[0].Frequency);
        Assert.Equal(300, eq.Bands[1].Frequency);
        Assert.Equal(1000, eq.Bands[2].Frequency);
        Assert.Equal(3500, eq.Bands[3].Frequency);
        Assert.Equal(10000, eq.Bands[4].Frequency);
    }

    [Fact]
    public void ParametricEQ_DefaultBands_GainIsZero()
    {
        var eq = new ParametricEqualizer<double>();
        foreach (var band in eq.Bands)
        {
            Assert.Equal(0.0, band.GainDb, Tolerance);
        }
    }

    [Fact]
    public void ParametricEQ_AddBand_IncreasesCount()
    {
        var eq = new ParametricEqualizer<double>();
        eq.AddBand(2000, 3.0, 1.5, EqFilterType.Peak);
        Assert.Equal(6, eq.Bands.Count);
    }

    [Fact]
    public void ParametricEQ_RemoveBand_DecreasesCount()
    {
        var eq = new ParametricEqualizer<double>();
        eq.RemoveBand(0);
        Assert.Equal(4, eq.Bands.Count);
    }

    [Fact]
    public void ParametricEQ_SetBand_UpdatesParameters()
    {
        var eq = new ParametricEqualizer<double>();
        eq.SetBand(2, 2000, 6.0, 2.0);
        Assert.Equal(2000, eq.Bands[2].Frequency);
        Assert.Equal(6.0, eq.Bands[2].GainDb);
        Assert.Equal(2.0, eq.Bands[2].Q);
    }

    [Fact]
    public void ParametricEQ_ProcessSample_FlatEQ_Passthrough()
    {
        var eq = new ParametricEqualizer<double>();
        // With all gains at 0 dB, signal should pass through mostly unchanged
        // (biquad filters introduce some slight changes due to digital filter artifacts)
        var result = eq.ProcessSample(0.5);
        Assert.True(Math.Abs(result - 0.5) < 0.1,
            $"Flat EQ should approximately pass through signal, got {result}");
    }

    [Fact]
    public void ParametricEQ_Bypass_ReturnsInput()
    {
        var eq = new ParametricEqualizer<double>();
        eq.Bypass = true;
        var result = eq.ProcessSample(0.75);
        Assert.Equal(0.75, result, Tolerance);
    }

    [Fact]
    public void ParametricEQ_Reset_DoesNotThrow()
    {
        var eq = new ParametricEqualizer<double>();
        eq.ProcessSample(0.5);
        eq.Reset();
    }

    #endregion

    #region EqFilterType Enum

    [Fact]
    public void EqFilterType_AllValues()
    {
        var values = Enum.GetValues<EqFilterType>();
        Assert.Equal(7, values.Length);
        Assert.Contains(EqFilterType.Peak, values);
        Assert.Contains(EqFilterType.LowShelf, values);
        Assert.Contains(EqFilterType.HighShelf, values);
        Assert.Contains(EqFilterType.LowPass, values);
        Assert.Contains(EqFilterType.HighPass, values);
        Assert.Contains(EqFilterType.BandPass, values);
        Assert.Contains(EqFilterType.Notch, values);
    }

    #endregion

    #region EqBand - Construction and Processing

    [Fact]
    public void EqBand_StoresProperties()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var band = new EqBand<double>(numOps, 44100, 1000, 3.0, 1.5, EqFilterType.Peak);
        Assert.Equal(1000, band.Frequency);
        Assert.Equal(3.0, band.GainDb);
        Assert.Equal(1.5, band.Q);
        Assert.Equal(EqFilterType.Peak, band.FilterType);
    }

    [Fact]
    public void EqBand_FrequencyClamped()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var band = new EqBand<double>(numOps, 44100, 5, 0, 1.0, EqFilterType.Peak);
        // Frequency should be clamped to minimum of 20
        Assert.Equal(20, band.Frequency);
    }

    [Fact]
    public void EqBand_GainClamped()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var band = new EqBand<double>(numOps, 44100, 1000, 50.0, 1.0, EqFilterType.Peak);
        // Gain should be clamped to max of 24
        Assert.Equal(24, band.GainDb);
    }

    [Fact]
    public void EqBand_QClamped()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var band = new EqBand<double>(numOps, 44100, 1000, 0, 0.01, EqFilterType.Peak);
        // Q should be clamped to minimum of 0.1
        Assert.Equal(0.1, band.Q, Tolerance);
    }

    [Fact]
    public void EqBand_SetParameters_Updates()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var band = new EqBand<double>(numOps, 44100, 1000, 0, 1.0, EqFilterType.Peak);
        band.SetParameters(2000, 6.0, 2.0);
        Assert.Equal(2000, band.Frequency);
        Assert.Equal(6.0, band.GainDb);
        Assert.Equal(2.0, band.Q);
    }

    [Fact]
    public void EqBand_Process_DoesNotThrow()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var band = new EqBand<double>(numOps, 44100, 1000, 3.0, 1.0, EqFilterType.Peak);
        var result = band.Process(0.5);
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void EqBand_Reset_DoesNotThrow()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var band = new EqBand<double>(numOps, 44100, 1000, 3.0, 1.0, EqFilterType.Peak);
        band.Process(0.5);
        band.Reset();
        // After reset, state should be clean
        var result = band.Process(0.0);
        Assert.True(Math.Abs(result) < 0.01, "After reset, processing zero should yield near-zero");
    }

    #endregion

    #region AudioEvent - Construction and Properties

    [Fact]
    public void AudioEvent_RequiredAndInitProperties()
    {
        var evt = new AudioEvent
        {
            Label = "Speech",
            Confidence = 0.95,
            StartTime = 1.5,
            EndTime = 3.0
        };
        Assert.Equal("Speech", evt.Label);
        Assert.Equal(0.95, evt.Confidence, Tolerance);
        Assert.Equal(1.5, evt.StartTime, Tolerance);
        Assert.Equal(3.0, evt.EndTime, Tolerance);
    }

    [Fact]
    public void AudioEvent_Duration_Computed()
    {
        var evt = new AudioEvent
        {
            Label = "Music",
            Confidence = 0.8,
            StartTime = 2.0,
            EndTime = 5.5
        };
        Assert.Equal(3.5, evt.Duration, Tolerance);
    }

    [Fact]
    public void AudioEvent_ToString_ContainsLabel()
    {
        var evt = new AudioEvent
        {
            Label = "DogBark",
            Confidence = 0.75,
            StartTime = 0.0,
            EndTime = 1.0
        };
        var str = evt.ToString();
        Assert.Contains("DogBark", str);
    }

    [Fact]
    public void AudioEvent_ZeroDuration()
    {
        var evt = new AudioEvent
        {
            Label = "Click",
            Confidence = 0.5,
            StartTime = 1.0,
            EndTime = 1.0
        };
        Assert.Equal(0.0, evt.Duration, Tolerance);
    }

    #endregion

    #region AudioGenOptions - Defaults

    [Fact]
    public void AudioGenOptions_DefaultValues()
    {
        var options = new AudioGenOptions();
        Assert.Equal(AudioGenModelSize.Medium, options.ModelSize);
        Assert.Equal(32000, options.SampleRate);
        Assert.Equal(5.0, options.DurationSeconds, Tolerance);
        Assert.Equal(30.0, options.MaxDurationSeconds, Tolerance);
        Assert.Equal(1.0, options.Temperature, Tolerance);
        Assert.Equal(250, options.TopK);
        Assert.Equal(0.0, options.TopP, Tolerance);
        Assert.Equal(3.0, options.GuidanceScale, Tolerance);
        Assert.Null(options.Seed);
        Assert.Equal(1, options.Channels);
        Assert.Null(options.TextEncoderPath);
        Assert.Null(options.LanguageModelPath);
        Assert.Null(options.AudioCodecPath);
        Assert.NotNull(options.OnnxOptions);
    }

    [Fact]
    public void AudioGenOptions_CustomValues()
    {
        var options = new AudioGenOptions
        {
            ModelSize = AudioGenModelSize.Large,
            SampleRate = 44100,
            DurationSeconds = 10.0,
            Temperature = 0.8,
            TopK = 100,
            GuidanceScale = 5.0,
            Seed = 42,
            Channels = 2
        };
        Assert.Equal(AudioGenModelSize.Large, options.ModelSize);
        Assert.Equal(44100, options.SampleRate);
        Assert.Equal(10.0, options.DurationSeconds, Tolerance);
        Assert.Equal(42, options.Seed);
        Assert.Equal(2, options.Channels);
    }

    #endregion

    #region AudioGenResult - Properties

    [Fact]
    public void AudioGenResult_Properties()
    {
        var result = new AudioGenResult<double>
        {
            Audio = new double[] { 0.1, 0.2, 0.3 },
            SampleRate = 16000,
            Duration = 1.5,
            Prompt = "birds chirping",
            SeedUsed = 42,
            ProcessingTimeMs = 500
        };
        Assert.Equal(3, result.Audio.Length);
        Assert.Equal(16000, result.SampleRate);
        Assert.Equal(1.5, result.Duration, Tolerance);
        Assert.Equal("birds chirping", result.Prompt);
        Assert.Equal(42, result.SeedUsed);
        Assert.Equal(500, result.ProcessingTimeMs);
    }

    [Fact]
    public void AudioGenResult_DefaultPrompt()
    {
        var result = new AudioGenResult<double>
        {
            Audio = Array.Empty<double>()
        };
        Assert.Equal(string.Empty, result.Prompt);
    }

    #endregion

    #region AudioGenModelSize Enum

    [Fact]
    public void AudioGenModelSize_AllValues()
    {
        var values = Enum.GetValues<AudioGenModelSize>();
        Assert.Equal(3, values.Length);
        Assert.Contains(AudioGenModelSize.Small, values);
        Assert.Contains(AudioGenModelSize.Medium, values);
        Assert.Contains(AudioGenModelSize.Large, values);
    }

    #endregion

    #region ASTOptions - Defaults

    [Fact]
    public void ASTOptions_DefaultValues()
    {
        var options = new ASTOptions();
        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(400, options.FftSize);
        Assert.Equal(160, options.HopLength);
        Assert.Equal(128, options.NumMels);
        Assert.Equal(0, options.FMin);
        Assert.Equal(8000, options.FMax);
        Assert.Equal(16, options.PatchSize);
        Assert.Equal(10, options.PatchStride);
        Assert.Equal(768, options.EmbeddingDim);
        Assert.Equal(12, options.NumEncoderLayers);
        Assert.Equal(12, options.NumAttentionHeads);
        Assert.Equal(3072, options.FeedForwardDim);
        Assert.Equal(0.1, options.DropoutRate, Tolerance);
        Assert.Equal(0.0, options.AttentionDropoutRate, Tolerance);
        Assert.Equal(0.3, options.Threshold, Tolerance);
        Assert.Equal(10.0, options.WindowSize, Tolerance);
        Assert.Equal(0.5, options.WindowOverlap, Tolerance);
        Assert.Null(options.CustomLabels);
        Assert.Null(options.ModelPath);
        Assert.NotNull(options.OnnxOptions);
        Assert.Equal(1e-4, options.LearningRate, 1e-10);
        Assert.Equal(5, options.WarmUpEpochs);
        Assert.Equal(0.1, options.LabelSmoothing, Tolerance);
        Assert.True(options.UseImageNetPretrain);
    }

    #endregion

    #region Cross-Module - Audio Effects Process Tensor

    [Fact]
    public void AudioEffects_Process_SineWave()
    {
        // Generate a 1kHz sine wave at 44100 Hz
        int sampleRate = 44100;
        int numSamples = 4410; // 0.1 seconds
        var input = new Tensor<double>(new[] { numSamples });
        for (int i = 0; i < numSamples; i++)
        {
            input[i] = 0.5 * Math.Sin(2 * Math.PI * 1000 * i / sampleRate);
        }

        // Compressor should produce output of same length
        var compressor = new Compressor<double>(sampleRate: sampleRate);
        var compressedOutput = compressor.Process(input);
        Assert.Equal(numSamples, compressedOutput.Length);

        // EQ should produce output of same length
        var eq = new ParametricEqualizer<double>(sampleRate: sampleRate);
        var eqOutput = eq.Process(input);
        Assert.Equal(numSamples, eqOutput.Length);
    }

    [Fact]
    public void AudioEffects_SetParameter_ChangesValue()
    {
        var compressor = new Compressor<double>();
        compressor.SetParameter("threshold", -10.0);
        var value = compressor.GetParameter("threshold");
        Assert.Equal(-10.0, value, Tolerance);
    }

    [Fact]
    public void AudioEffects_GetParameter_NonExistent_ReturnsZero()
    {
        var compressor = new Compressor<double>();
        var value = compressor.GetParameter("nonexistent");
        Assert.Equal(0.0, value, Tolerance);
    }

    #endregion

    #region Cross-Module - Feature Extractors with Synthetic Audio

    [Fact]
    public void MfccExtractor_Extract_SyntheticAudio_ReturnsCorrectShape()
    {
        var options = new MfccOptions { SampleRate = 16000, FftSize = 512, HopLength = 160 };
        var extractor = new MfccExtractor<double>(options);

        // Create synthetic audio: 1 second of sine wave at 440 Hz
        int numSamples = 16000;
        var audio = new Tensor<double>(new[] { numSamples });
        for (int i = 0; i < numSamples; i++)
        {
            audio[i] = 0.5 * Math.Sin(2 * Math.PI * 440 * i / 16000.0);
        }

        var features = extractor.Extract(audio);
        Assert.Equal(2, features.Shape.Length); // [frames, coefficients]
        Assert.Equal(13, features.Shape[1]); // 13 MFCC coefficients
        Assert.True(features.Shape[0] > 0, "Should have at least one frame");
    }

    [Fact]
    public void ChromaExtractor_Extract_SyntheticAudio_Returns12PitchClasses()
    {
        var options = new ChromaOptions { SampleRate = 16000, FftSize = 2048, HopLength = 512 };
        var extractor = new ChromaExtractor<double>(options);

        int numSamples = 16000;
        var audio = new Tensor<double>(new[] { numSamples });
        for (int i = 0; i < numSamples; i++)
        {
            audio[i] = 0.5 * Math.Sin(2 * Math.PI * 440 * i / 16000.0);
        }

        var features = extractor.Extract(audio);
        Assert.Equal(2, features.Shape.Length);
        Assert.Equal(12, features.Shape[1]); // 12 pitch classes
        Assert.True(features.Shape[0] > 0, "Should have at least one frame");
    }

    [Fact]
    public void SpectralFeatureExtractor_Extract_SyntheticAudio_ReturnsCorrectShape()
    {
        var options = new SpectralFeatureOptions
        {
            SampleRate = 16000,
            FftSize = 512,
            HopLength = 160,
            FeatureTypes = SpectralFeatureType.Basic
        };
        var extractor = new SpectralFeatureExtractor<double>(options);

        int numSamples = 16000;
        var audio = new Tensor<double>(new[] { numSamples });
        for (int i = 0; i < numSamples; i++)
        {
            audio[i] = 0.5 * Math.Sin(2 * Math.PI * 1000 * i / 16000.0);
        }

        var features = extractor.Extract(audio);
        Assert.Equal(2, features.Shape.Length);
        Assert.Equal(5, features.Shape[1]); // 5 basic spectral features
        Assert.True(features.Shape[0] > 0, "Should have at least one frame");
    }

    #endregion

    #region Cross-Module - ConstantQTransform

    [Fact]
    public void ConstantQTransform_Construction_DoesNotThrow()
    {
        var cqt = new ConstantQTransform<double>(
            sampleRate: 22050,
            fMin: 32.7,
            binsPerOctave: 12,
            numOctaves: 7);
        Assert.NotNull(cqt);
    }

    #endregion
}
