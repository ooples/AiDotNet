namespace AiDotNet.Augmentation;

/// <summary>
/// Unified configuration for data augmentation with industry-standard defaults.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Data augmentation creates variations of your training data
/// (like flipping images, adding noise, or shuffling words) to help your model learn better
/// and become more robust. This configuration controls how augmentation works.</para>
///
/// <para><b>Key features:</b>
/// <list type="bullet">
/// <item>Automatic data-type detection (image, tabular, audio, text, video)</item>
/// <item>Industry-standard defaults that work well out-of-the-box</item>
/// <item>Test-Time Augmentation (TTA) enabled by default for better predictions</item>
/// <item>Modality-specific settings for fine-tuning</item>
/// </list>
/// </para>
///
/// <para><b>Example - Simple usage with defaults:</b></para>
/// <code>
/// var result = builder
///     .ConfigureModel(myModel)
///     .ConfigureAugmentation()  // Uses auto-detected defaults
///     .Build(X, y);
/// </code>
///
/// <para><b>Example - Custom configuration:</b></para>
/// <code>
/// var config = new AugmentationConfig
/// {
///     EnableTTA = true,
///     TTANumAugmentations = 8,
///     ImageSettings = new ImageAugmentationSettings
///     {
///         EnableFlips = true,
///         EnableRotation = true,
///         RotationRange = 20.0
///     }
/// };
/// builder.ConfigureAugmentation(config);
/// </code>
/// </remarks>
public class AugmentationConfig
{
    // === Core Settings ===

    /// <summary>
    /// Gets or sets whether augmentation is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Set to <c>false</c> to disable augmentation entirely without removing configuration.</para>
    /// </remarks>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the global probability of applying any augmentation.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.5</c> (50% chance)</para>
    /// <para><b>For Beginners:</b> This controls how often augmentations are applied.
    /// At 0.5, each augmentation has a 50% chance of being applied to each sample.
    /// Higher values (0.8) mean more aggressive augmentation.
    /// </para>
    /// </remarks>
    public double Probability { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the random seed for reproducible augmentations.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>null</c> (random each time)</para>
    /// <para>Set a specific seed for reproducible results (useful for debugging).</para>
    /// </remarks>
    public int? Seed { get; set; }

    // === Test-Time Augmentation (TTA) Settings ===

    /// <summary>
    /// Gets or sets whether Test-Time Augmentation is enabled during inference.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c> (recommended for production)</para>
    /// <para><b>For Beginners:</b> TTA improves prediction accuracy by making multiple
    /// predictions on augmented versions of your input and combining them. This typically
    /// improves accuracy by 1-3% at the cost of slower inference.</para>
    /// </remarks>
    public bool EnableTTA { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of augmented samples for TTA.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>5</c> (research-backed optimal count)</para>
    /// <para><b>For Beginners:</b> More augmentations = more accurate but slower.
    /// Research shows diminishing returns beyond 10.</para>
    /// </remarks>
    public int TTANumAugmentations { get; set; } = 5;

    /// <summary>
    /// Gets or sets how to aggregate TTA predictions.
    /// </summary>
    /// <remarks>
    /// <para>Default: <see cref="PredictionAggregationMethod.Mean"/></para>
    /// <para>Options: Mean (average), Median (robust to outliers), Vote (for classification),
    /// Max, Min, WeightedMean, GeometricMean.</para>
    /// </remarks>
    public PredictionAggregationMethod TTAAggregation { get; set; } = PredictionAggregationMethod.Mean;

    /// <summary>
    /// Gets or sets whether to include the original (non-augmented) sample in TTA.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Almost always leave this enabled to include the "ground truth" view.</para>
    /// </remarks>
    public bool TTAIncludeOriginal { get; set; } = true;

    // === Modality-Specific Settings ===

    /// <summary>
    /// Gets or sets image-specific augmentation settings.
    /// </summary>
    /// <remarks>
    /// <para>If null, auto-configured with industry-standard defaults when image data is detected.</para>
    /// </remarks>
    public ImageAugmentationSettings? ImageSettings { get; set; }

    /// <summary>
    /// Gets or sets tabular-specific augmentation settings.
    /// </summary>
    /// <remarks>
    /// <para>If null, auto-configured with industry-standard defaults when tabular data is detected.</para>
    /// </remarks>
    public TabularAugmentationSettings? TabularSettings { get; set; }

    /// <summary>
    /// Gets or sets audio-specific augmentation settings.
    /// </summary>
    /// <remarks>
    /// <para>If null, auto-configured with industry-standard defaults when audio data is detected.</para>
    /// </remarks>
    public AudioAugmentationSettings? AudioSettings { get; set; }

    /// <summary>
    /// Gets or sets text-specific augmentation settings.
    /// </summary>
    /// <remarks>
    /// <para>If null, auto-configured with industry-standard defaults when text data is detected.</para>
    /// </remarks>
    public TextAugmentationSettings? TextSettings { get; set; }

    /// <summary>
    /// Gets or sets video-specific augmentation settings.
    /// </summary>
    /// <remarks>
    /// <para>If null, auto-configured with industry-standard defaults when video data is detected.</para>
    /// </remarks>
    public VideoAugmentationSettings? VideoSettings { get; set; }

    /// <summary>
    /// Creates a new augmentation configuration with industry-standard defaults.
    /// </summary>
    public AugmentationConfig()
    {
    }

    /// <summary>
    /// Creates an augmentation configuration for image data with standard defaults.
    /// </summary>
    /// <returns>A configuration optimized for image augmentation.</returns>
    public static AugmentationConfig ForImages()
    {
        return new AugmentationConfig
        {
            ImageSettings = new ImageAugmentationSettings()
        };
    }

    /// <summary>
    /// Creates an augmentation configuration for tabular data with standard defaults.
    /// </summary>
    /// <returns>A configuration optimized for tabular augmentation.</returns>
    public static AugmentationConfig ForTabular()
    {
        return new AugmentationConfig
        {
            TabularSettings = new TabularAugmentationSettings()
        };
    }

    /// <summary>
    /// Creates an augmentation configuration for audio data with standard defaults.
    /// </summary>
    /// <returns>A configuration optimized for audio augmentation.</returns>
    public static AugmentationConfig ForAudio()
    {
        return new AugmentationConfig
        {
            AudioSettings = new AudioAugmentationSettings()
        };
    }

    /// <summary>
    /// Creates an augmentation configuration for text data with standard defaults.
    /// </summary>
    /// <returns>A configuration optimized for text augmentation.</returns>
    public static AugmentationConfig ForText()
    {
        return new AugmentationConfig
        {
            TextSettings = new TextAugmentationSettings()
        };
    }

    /// <summary>
    /// Creates an augmentation configuration for video data with standard defaults.
    /// </summary>
    /// <returns>A configuration optimized for video augmentation.</returns>
    public static AugmentationConfig ForVideo()
    {
        return new AugmentationConfig
        {
            VideoSettings = new VideoAugmentationSettings()
        };
    }

    /// <summary>
    /// Gets the configuration as a dictionary for logging or serialization.
    /// </summary>
    /// <returns>A dictionary containing all configuration values.</returns>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>
        {
            ["isEnabled"] = IsEnabled,
            ["probability"] = Probability,
            ["enableTTA"] = EnableTTA,
            ["ttaNumAugmentations"] = TTANumAugmentations,
            ["ttaAggregation"] = TTAAggregation.ToString(),
            ["ttaIncludeOriginal"] = TTAIncludeOriginal
        };

        if (Seed.HasValue)
        {
            config["seed"] = Seed.Value;
        }

        if (ImageSettings is not null)
        {
            config["imageSettings"] = ImageSettings.GetConfiguration();
        }

        if (TabularSettings is not null)
        {
            config["tabularSettings"] = TabularSettings.GetConfiguration();
        }

        if (AudioSettings is not null)
        {
            config["audioSettings"] = AudioSettings.GetConfiguration();
        }

        if (TextSettings is not null)
        {
            config["textSettings"] = TextSettings.GetConfiguration();
        }

        if (VideoSettings is not null)
        {
            config["videoSettings"] = VideoSettings.GetConfiguration();
        }

        return config;
    }
}

/// <summary>
/// Image-specific augmentation settings with industry-standard defaults.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how images are transformed during
/// training. Defaults are based on best practices from Albumentations and torchvision.</para>
/// </remarks>
public class ImageAugmentationSettings
{
    /// <summary>
    /// Gets or sets whether horizontal flip is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c> (probability 0.5)</para>
    /// <para>Mirrors the image left-to-right. Good for most tasks except where orientation matters.</para>
    /// </remarks>
    public bool EnableFlips { get; set; } = true;

    /// <summary>
    /// Gets or sets whether vertical flip is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c></para>
    /// <para>Mirrors the image top-to-bottom. Usually only useful for satellite/medical images.</para>
    /// </remarks>
    public bool EnableVerticalFlip { get; set; }

    /// <summary>
    /// Gets or sets whether rotation is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Rotates the image by a random angle within the specified range.</para>
    /// </remarks>
    public bool EnableRotation { get; set; } = true;

    /// <summary>
    /// Gets or sets the rotation range in degrees.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>15.0</c> (degrees, +/- from horizontal)</para>
    /// <para><b>For Beginners:</b> A value of 15 means rotation between -15 and +15 degrees.</para>
    /// </remarks>
    public double RotationRange { get; set; } = 15.0;

    /// <summary>
    /// Gets or sets whether color jitter (brightness, contrast, saturation) is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Randomly adjusts color properties to simulate different lighting conditions.</para>
    /// </remarks>
    public bool EnableColorJitter { get; set; } = true;

    /// <summary>
    /// Gets or sets the brightness adjustment range.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.2</c> (+/- 20% brightness)</para>
    /// </remarks>
    public double BrightnessRange { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the contrast adjustment range.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.2</c> (+/- 20% contrast)</para>
    /// </remarks>
    public double ContrastRange { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the saturation adjustment range.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.2</c> (+/- 20% saturation)</para>
    /// </remarks>
    public double SaturationRange { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether Gaussian noise is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Adds random noise to simulate sensor noise or low-light conditions.</para>
    /// </remarks>
    public bool EnableGaussianNoise { get; set; } = true;

    /// <summary>
    /// Gets or sets the standard deviation of Gaussian noise.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.01</c> (for normalized images [0,1])</para>
    /// </remarks>
    public double NoiseStdDev { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether Gaussian blur is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c></para>
    /// <para>Applies blur to simulate out-of-focus images or motion blur.</para>
    /// </remarks>
    public bool EnableGaussianBlur { get; set; }

    /// <summary>
    /// Gets or sets whether Cutout/CoarseDropout is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c></para>
    /// <para>Randomly masks out rectangular regions. Good for regularization.</para>
    /// </remarks>
    public bool EnableCutout { get; set; }

    /// <summary>
    /// Gets or sets whether MixUp is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c></para>
    /// <para>Blends two images together with interpolated labels. Powerful regularizer.</para>
    /// </remarks>
    public bool EnableMixUp { get; set; }

    /// <summary>
    /// Gets or sets the MixUp alpha parameter.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.2</c></para>
    /// <para>Controls the Beta distribution for mixing ratio. Lower = more original image.</para>
    /// </remarks>
    public double MixUpAlpha { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether CutMix is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c></para>
    /// <para>Cuts a patch from one image and pastes it onto another. Often better than MixUp.</para>
    /// </remarks>
    public bool EnableCutMix { get; set; }

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        return new Dictionary<string, object>
        {
            ["enableFlips"] = EnableFlips,
            ["enableVerticalFlip"] = EnableVerticalFlip,
            ["enableRotation"] = EnableRotation,
            ["rotationRange"] = RotationRange,
            ["enableColorJitter"] = EnableColorJitter,
            ["brightnessRange"] = BrightnessRange,
            ["contrastRange"] = ContrastRange,
            ["saturationRange"] = SaturationRange,
            ["enableGaussianNoise"] = EnableGaussianNoise,
            ["noiseStdDev"] = NoiseStdDev,
            ["enableGaussianBlur"] = EnableGaussianBlur,
            ["enableCutout"] = EnableCutout,
            ["enableMixUp"] = EnableMixUp,
            ["mixUpAlpha"] = MixUpAlpha,
            ["enableCutMix"] = EnableCutMix
        };
    }
}

/// <summary>
/// Tabular-specific augmentation settings with industry-standard defaults.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how tabular (spreadsheet-like) data
/// is augmented. Useful for improving model generalization on structured data.</para>
/// </remarks>
public class TabularAugmentationSettings
{
    /// <summary>
    /// Gets or sets whether MixUp for tabular data is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Blends feature vectors from different samples. Effective for tabular data.</para>
    /// </remarks>
    public bool EnableMixUp { get; set; } = true;

    /// <summary>
    /// Gets or sets the MixUp alpha parameter.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.2</c></para>
    /// </remarks>
    public double MixUpAlpha { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether feature noise is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Adds small random noise to numerical features.</para>
    /// </remarks>
    public bool EnableFeatureNoise { get; set; } = true;

    /// <summary>
    /// Gets or sets the standard deviation of feature noise.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.01</c></para>
    /// </remarks>
    public double NoiseStdDev { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether feature dropout is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c></para>
    /// <para>Randomly zeros out features during training. Good regularizer.</para>
    /// </remarks>
    public bool EnableFeatureDropout { get; set; }

    /// <summary>
    /// Gets or sets the feature dropout rate.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.1</c> (10% of features dropped)</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether SMOTE is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c> (only enable for imbalanced data)</para>
    /// <para>Synthetic Minority Over-sampling Technique for class imbalance.</para>
    /// </remarks>
    public bool EnableSmote { get; set; }

    /// <summary>
    /// Gets or sets the number of nearest neighbors for SMOTE.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>5</c></para>
    /// </remarks>
    public int SmoteK { get; set; } = 5;

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        return new Dictionary<string, object>
        {
            ["enableMixUp"] = EnableMixUp,
            ["mixUpAlpha"] = MixUpAlpha,
            ["enableFeatureNoise"] = EnableFeatureNoise,
            ["noiseStdDev"] = NoiseStdDev,
            ["enableFeatureDropout"] = EnableFeatureDropout,
            ["dropoutRate"] = DropoutRate,
            ["enableSmote"] = EnableSmote,
            ["smoteK"] = SmoteK
        };
    }
}

/// <summary>
/// Audio-specific augmentation settings with industry-standard defaults.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how audio data is augmented.
/// Defaults are based on best practices from audiomentations and torchaudio.</para>
/// </remarks>
public class AudioAugmentationSettings
{
    /// <summary>
    /// Gets or sets whether pitch shifting is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Changes the pitch without changing tempo. Good for speech/music tasks.</para>
    /// </remarks>
    public bool EnablePitchShift { get; set; } = true;

    /// <summary>
    /// Gets or sets the pitch shift range in semitones.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>2.0</c> (+/- 2 semitones)</para>
    /// </remarks>
    public double PitchShiftRange { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets whether time stretching is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Changes tempo without changing pitch.</para>
    /// </remarks>
    public bool EnableTimeStretch { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum time stretch factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.8</c> (80% speed, slower)</para>
    /// </remarks>
    public double MinTimeStretch { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the maximum time stretch factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>1.2</c> (120% speed, faster)</para>
    /// </remarks>
    public double MaxTimeStretch { get; set; } = 1.2;

    /// <summary>
    /// Gets or sets whether background noise is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Adds random noise to simulate different recording conditions.</para>
    /// </remarks>
    public bool EnableNoise { get; set; } = true;

    /// <summary>
    /// Gets or sets the signal-to-noise ratio in decibels.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>20.0</c> (dB)</para>
    /// <para>Higher values mean less noise relative to signal.</para>
    /// </remarks>
    public double NoiseSNR { get; set; } = 20.0;

    /// <summary>
    /// Gets or sets whether volume change is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Randomly adjusts volume to simulate different recording levels.</para>
    /// </remarks>
    public bool EnableVolumeChange { get; set; } = true;

    /// <summary>
    /// Gets or sets the volume change range in decibels.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>6.0</c> (+/- 6 dB)</para>
    /// </remarks>
    public double VolumeChangeRange { get; set; } = 6.0;

    /// <summary>
    /// Gets or sets whether time shift is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Shifts the audio forward or backward in time.</para>
    /// </remarks>
    public bool EnableTimeShift { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum time shift as a fraction of audio length.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.1</c> (10% of audio length)</para>
    /// </remarks>
    public double MaxTimeShift { get; set; } = 0.1;

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        return new Dictionary<string, object>
        {
            ["enablePitchShift"] = EnablePitchShift,
            ["pitchShiftRange"] = PitchShiftRange,
            ["enableTimeStretch"] = EnableTimeStretch,
            ["minTimeStretch"] = MinTimeStretch,
            ["maxTimeStretch"] = MaxTimeStretch,
            ["enableNoise"] = EnableNoise,
            ["noiseSNR"] = NoiseSNR,
            ["enableVolumeChange"] = EnableVolumeChange,
            ["volumeChangeRange"] = VolumeChangeRange,
            ["enableTimeShift"] = EnableTimeShift,
            ["maxTimeShift"] = MaxTimeShift
        };
    }
}

/// <summary>
/// Text-specific augmentation settings with industry-standard defaults.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how text data is augmented.
/// Defaults are based on best practices from nlpaug and TextAttack.</para>
/// </remarks>
public class TextAugmentationSettings
{
    /// <summary>
    /// Gets or sets whether synonym replacement is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Replaces words with their synonyms.</para>
    /// </remarks>
    public bool EnableSynonymReplacement { get; set; } = true;

    /// <summary>
    /// Gets or sets the fraction of words to replace with synonyms.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.1</c> (10% of words)</para>
    /// </remarks>
    public double SynonymReplacementRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether random deletion is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Randomly deletes words from the text.</para>
    /// </remarks>
    public bool EnableRandomDeletion { get; set; } = true;

    /// <summary>
    /// Gets or sets the fraction of words to delete.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.1</c> (10% of words)</para>
    /// </remarks>
    public double DeletionRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether random swap is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Randomly swaps the positions of two words.</para>
    /// </remarks>
    public bool EnableRandomSwap { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of word swaps to perform.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>2</c></para>
    /// </remarks>
    public int NumSwaps { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether random insertion is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c></para>
    /// <para>Randomly inserts synonyms of random words.</para>
    /// </remarks>
    public bool EnableRandomInsertion { get; set; }

    /// <summary>
    /// Gets or sets the fraction of words to insert.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.1</c> (10% of original length)</para>
    /// </remarks>
    public double InsertionRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether back-translation is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>false</c> (requires external translation service)</para>
    /// <para>Translates text to another language and back for paraphrasing.</para>
    /// </remarks>
    public bool EnableBackTranslation { get; set; }

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        return new Dictionary<string, object>
        {
            ["enableSynonymReplacement"] = EnableSynonymReplacement,
            ["synonymReplacementRate"] = SynonymReplacementRate,
            ["enableRandomDeletion"] = EnableRandomDeletion,
            ["deletionRate"] = DeletionRate,
            ["enableRandomSwap"] = EnableRandomSwap,
            ["numSwaps"] = NumSwaps,
            ["enableRandomInsertion"] = EnableRandomInsertion,
            ["insertionRate"] = InsertionRate,
            ["enableBackTranslation"] = EnableBackTranslation
        };
    }
}

/// <summary>
/// Video-specific augmentation settings with industry-standard defaults.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how video data is augmented.
/// Video augmentation applies both spatial (image-based) and temporal (time-based) transforms.</para>
/// </remarks>
public class VideoAugmentationSettings
{
    /// <summary>
    /// Gets or sets whether temporal crop is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Randomly selects a temporal segment from the video.</para>
    /// </remarks>
    public bool EnableTemporalCrop { get; set; } = true;

    /// <summary>
    /// Gets or sets the fraction of frames to keep.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.8</c> (keep 80% of frames)</para>
    /// </remarks>
    public double CropRatio { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets whether temporal flip (reverse) is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Reverses the frame order. Good for action recognition.</para>
    /// </remarks>
    public bool EnableTemporalFlip { get; set; } = true;

    /// <summary>
    /// Gets or sets whether frame dropout is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Randomly drops frames from the video.</para>
    /// </remarks>
    public bool EnableFrameDropout { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for frames.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.1</c> (drop 10% of frames)</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether speed change is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Randomly speeds up or slows down the video.</para>
    /// </remarks>
    public bool EnableSpeedChange { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum speed factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.8</c> (80% speed, slower)</para>
    /// </remarks>
    public double MinSpeed { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the maximum speed factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>1.2</c> (120% speed, faster)</para>
    /// </remarks>
    public double MaxSpeed { get; set; } = 1.2;

    /// <summary>
    /// Gets or sets whether spatial transforms (image augmentations) are applied to frames.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// <para>Applies image augmentations (flip, color jitter, etc.) to all frames consistently.</para>
    /// </remarks>
    public bool EnableSpatialTransforms { get; set; } = true;

    /// <summary>
    /// Gets or sets the image augmentation settings for spatial transforms.
    /// </summary>
    /// <remarks>
    /// <para>If null, uses default ImageAugmentationSettings.</para>
    /// </remarks>
    public ImageAugmentationSettings? SpatialSettings { get; set; }

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>
        {
            ["enableTemporalCrop"] = EnableTemporalCrop,
            ["cropRatio"] = CropRatio,
            ["enableTemporalFlip"] = EnableTemporalFlip,
            ["enableFrameDropout"] = EnableFrameDropout,
            ["dropoutRate"] = DropoutRate,
            ["enableSpeedChange"] = EnableSpeedChange,
            ["minSpeed"] = MinSpeed,
            ["maxSpeed"] = MaxSpeed,
            ["enableSpatialTransforms"] = EnableSpatialTransforms
        };

        if (SpatialSettings is not null)
        {
            config["spatialSettings"] = SpatialSettings.GetConfiguration();
        }

        return config;
    }
}

/// <summary>
/// Enum representing detected data modality types.
/// </summary>
public enum DataModality
{
    /// <summary>Unknown or undetected data type.</summary>
    Unknown,
    /// <summary>Image data (2D or 3D tensors representing pixels).</summary>
    Image,
    /// <summary>Tabular data (matrices with rows and columns).</summary>
    Tabular,
    /// <summary>Audio data (1D waveforms or spectrograms).</summary>
    Audio,
    /// <summary>Text data (strings or token sequences).</summary>
    Text,
    /// <summary>Video data (sequences of images/frames).</summary>
    Video
}

/// <summary>
/// Utility for auto-detecting data modality from input types.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This helper automatically determines what kind of data
/// you're working with (images, tables, audio, etc.) so the right augmentations can be applied.</para>
/// </remarks>
public static class DataModalityDetector
{
    /// <summary>
    /// Detects the data modality from the input type.
    /// </summary>
    /// <typeparam name="TInput">The input data type.</typeparam>
    /// <returns>The detected data modality.</returns>
    public static DataModality Detect<TInput>()
    {
        var inputType = typeof(TInput);
        var typeName = inputType.Name;
        var fullName = inputType.FullName ?? string.Empty;

        // Check for ImageTensor (our image representation)
        if (typeName.StartsWith("ImageTensor", StringComparison.Ordinal) ||
            fullName.Contains("Image", StringComparison.OrdinalIgnoreCase))
        {
            return DataModality.Image;
        }

        // Check for Matrix (tabular data)
        if (typeName.StartsWith("Matrix", StringComparison.Ordinal) ||
            typeName.Equals("DataFrame", StringComparison.Ordinal))
        {
            return DataModality.Tabular;
        }

        // Check for audio types
        if (fullName.Contains("Audio", StringComparison.OrdinalIgnoreCase) ||
            fullName.Contains("Waveform", StringComparison.OrdinalIgnoreCase) ||
            fullName.Contains("Spectrogram", StringComparison.OrdinalIgnoreCase))
        {
            return DataModality.Audio;
        }

        // Check for text types
        if (inputType == typeof(string) ||
            inputType == typeof(string[]) ||
            fullName.Contains("Text", StringComparison.OrdinalIgnoreCase) ||
            fullName.Contains("Token", StringComparison.OrdinalIgnoreCase))
        {
            return DataModality.Text;
        }

        // Check for video types (array of images/tensors)
        if (inputType.IsArray && inputType.GetElementType() is not null)
        {
            var elementType = inputType.GetElementType();
            if (elementType is not null && elementType.Name.StartsWith("ImageTensor", StringComparison.Ordinal))
            {
                return DataModality.Video;
            }
        }

        if (fullName.Contains("Video", StringComparison.OrdinalIgnoreCase) ||
            fullName.Contains("Frame", StringComparison.OrdinalIgnoreCase))
        {
            return DataModality.Video;
        }

        // Check for generic Tensor - return Unknown to require explicit configuration
        // since a Tensor could be image, audio spectrogram, or other modalities
        if (typeName.StartsWith("Tensor", StringComparison.Ordinal))
        {
            return DataModality.Unknown;
        }

        // Check for Vector (could be tabular)
        if (typeName.StartsWith("Vector", StringComparison.Ordinal))
        {
            return DataModality.Tabular;
        }

        return DataModality.Unknown;
    }
}
