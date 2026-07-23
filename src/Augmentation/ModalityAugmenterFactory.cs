using AiDotNet.Augmentation.Audio;
using AiDotNet.Augmentation.Image;
using AiDotNet.Augmentation.Tabular;
using AiDotNet.Augmentation.Text;
using AiDotNet.Augmentation.Video;

namespace AiDotNet.Augmentation;

/// <summary>
/// Factory translating modality-specific settings blocks (image / tabular /
/// audio / text / video) on <see cref="AugmentationConfig"/> into a typed
/// <see cref="AugmentationPipeline{T,TData}"/> assembled from the built-in
/// augmenters under <c>src/Augmentation/{Image,Audio,Tabular,Text,Video}</c>.
/// </summary>
/// <remarks>
/// <para>
/// Resolves review #1368 C6WKu: the modality settings (e.g.
/// <see cref="ImageAugmentationSettings.EnableRotation"/>) were stored on
/// <see cref="AugmentationConfig"/> but never translated into an
/// <see cref="IAugmentation{T,TData}"/> — the surface was documentation-only
/// unless the user supplied their own <see cref="AugmentationConfig.CustomAugmenter"/>.
/// This factory wires each modality to its built-in augmenter family so
/// callers can configure augmentation purely through settings.
/// </para>
/// <para>
/// Each builder returns null if no augmenter in the settings block is
/// enabled — letting the caller distinguish "nothing to do" from "wired
/// but empty pipeline" and avoid pointless Apply traversals.
/// </para>
/// </remarks>
internal static class ModalityAugmenterFactory
{
    /// <summary>
    /// Builds an image-modality pipeline (<see cref="ImageTensor{T}"/> data)
    /// from <paramref name="s"/>. Honors <c>EnableFlips</c>,
    /// <c>EnableRotation</c>, <c>EnableColorJitter</c>, <c>EnableGaussianNoise</c>,
    /// and <c>EnableGaussianBlur</c> — the augmenters with concrete
    /// <see cref="ImageTensor{T}"/>-typed implementations in
    /// <c>src/Augmentation/Image</c>. Cutout / MixUp / CutMix have settings
    /// flags but no <see cref="ImageTensor{T}"/>-typed implementation yet
    /// (their classes are written against generic tensor types) — they're
    /// skipped here and surface a future-work hook.
    /// </summary>
    public static AugmentationPipeline<T, AiDotNet.Augmentation.Image.ImageTensor<T>>? BuildImageAugmenter<T>(
        ImageAugmentationSettings s,
        double globalProbability)
    {
        if (s is null) return null;
        var pipeline = new AugmentationPipeline<T, AiDotNet.Augmentation.Image.ImageTensor<T>>("ImageAugmenter");
        int added = 0;
        if (s.EnableFlips)
        {
            pipeline.Add(new HorizontalFlip<T>(probability: globalProbability));
            added++;
        }
        if (s.EnableRotation)
        {
            pipeline.Add(new Rotation<T>(
                minAngle: -s.RotationRange,
                maxAngle: s.RotationRange,
                probability: globalProbability));
            added++;
        }
        if (s.EnableColorJitter)
        {
            pipeline.Add(new ColorJitter<T>(
                brightnessRange: s.BrightnessRange,
                contrastRange: s.ContrastRange,
                saturationRange: s.SaturationRange,
                probability: globalProbability));
            added++;
        }
        if (s.EnableGaussianNoise)
        {
            pipeline.Add(new GaussianNoise<T>(
                minStd: s.NoiseStdDev,
                maxStd: s.NoiseStdDev * 5.0,
                probability: globalProbability));
            added++;
        }
        if (s.EnableGaussianBlur)
        {
            pipeline.Add(new GaussianBlur<T>(probability: globalProbability));
            added++;
        }
        return added > 0 ? pipeline : null;
    }

    /// <summary>
    /// Builds a tabular-modality pipeline (<see cref="AiDotNet.Tensors.LinearAlgebra.Matrix{T}"/> data) from
    /// <paramref name="s"/>. Honors <c>EnableFeatureNoise</c>,
    /// <c>EnableFeatureDropout</c>, and <c>EnableMixUp</c> via the
    /// <c>src/Augmentation/Tabular</c> family. SMOTE family requires labeled
    /// data and is not wired through the single-instance Apply pathway.
    /// </summary>
    public static AugmentationPipeline<T, AiDotNet.Tensors.LinearAlgebra.Matrix<T>>? BuildTabularAugmenter<T>(
        TabularAugmentationSettings s,
        double globalProbability)
    {
        if (s is null) return null;
        var pipeline = new AugmentationPipeline<T, AiDotNet.Tensors.LinearAlgebra.Matrix<T>>("TabularAugmenter");
        int added = 0;
        if (s.EnableFeatureNoise)
        {
            pipeline.Add(new FeatureNoise<T>(noiseStdDev: s.NoiseStdDev, probability: globalProbability));
            added++;
        }
        if (s.EnableFeatureDropout)
        {
            pipeline.Add(new FeatureDropout<T>(dropoutRate: s.DropoutRate, probability: globalProbability));
            added++;
        }
        if (s.EnableMixUp)
        {
            pipeline.Add(new TabularMixUp<T>(alpha: s.MixUpAlpha, probability: globalProbability));
            added++;
        }
        return added > 0 ? pipeline : null;
    }

    /// <summary>
    /// Builds an audio-modality pipeline (<see cref="Tensor{T}"/> waveform
    /// data) from <paramref name="s"/>. Honors all five built-in audio
    /// augmenters under <c>src/Augmentation/Audio</c>.
    /// </summary>
    public static AugmentationPipeline<T, AiDotNet.Tensors.LinearAlgebra.Tensor<T>>? BuildAudioAugmenter<T>(
        AudioAugmentationSettings s,
        double globalProbability)
    {
        if (s is null) return null;
        var pipeline = new AugmentationPipeline<T, AiDotNet.Tensors.LinearAlgebra.Tensor<T>>("AudioAugmenter");
        int added = 0;
        if (s.EnablePitchShift)
        {
            pipeline.Add(new PitchShift<T>(
                minSemitones: -s.PitchShiftRange,
                maxSemitones: s.PitchShiftRange,
                probability: globalProbability));
            added++;
        }
        if (s.EnableTimeStretch)
        {
            pipeline.Add(new TimeStretch<T>(
                minRate: s.MinTimeStretch,
                maxRate: s.MaxTimeStretch,
                probability: globalProbability));
            added++;
        }
        if (s.EnableNoise)
        {
            // NoiseSNR is a single fixed-SNR target; spread by ±10dB so the
            // augmenter's min/max contract is non-degenerate. Clamp the
            // lower bound at 0 dB to avoid passing a negative SNR (which
            // would push noise above the signal and produce uniformly
            // destroyed audio) — most consumers calling
            // AudioAugmentationSettings.NoiseSNR mean a snr-floor target,
            // not a wraparound (review #1368 C8ekA). Upper bound +∞ is
            // fine (clean signal).
            pipeline.Add(new AudioNoise<T>(
                minSnrDb: System.Math.Max(0.0, s.NoiseSNR - 10.0),
                maxSnrDb: s.NoiseSNR + 10.0,
                probability: globalProbability));
            added++;
        }
        if (s.EnableVolumeChange)
        {
            pipeline.Add(new VolumeChange<T>(
                minGainDb: -s.VolumeChangeRange,
                maxGainDb: s.VolumeChangeRange,
                probability: globalProbability));
            added++;
        }
        if (s.EnableTimeShift)
        {
            pipeline.Add(new TimeShift<T>(
                minShiftFraction: -s.MaxTimeShift,
                maxShiftFraction: s.MaxTimeShift,
                probability: globalProbability));
            added++;
        }
        return added > 0 ? pipeline : null;
    }

    /// <summary>
    /// Builds a text-modality pipeline (<see cref="string"/> array data) from
    /// <paramref name="s"/>. Honors all four built-in text augmenters under
    /// <c>src/Augmentation/Text</c>. Back-translation requires an external
    /// translation service and is not wired automatically.
    /// </summary>
    public static AugmentationPipeline<T, string[]>? BuildTextAugmenter<T>(
        TextAugmentationSettings s,
        double globalProbability)
    {
        if (s is null) return null;
        var pipeline = new AugmentationPipeline<T, string[]>("TextAugmenter");
        int added = 0;
        if (s.EnableSynonymReplacement)
        {
            pipeline.Add(new SynonymReplacement<T>(
                replacementFraction: s.SynonymReplacementRate,
                probability: globalProbability));
            added++;
        }
        if (s.EnableRandomDeletion)
        {
            pipeline.Add(new RandomDeletion<T>(
                deletionProbability: s.DeletionRate,
                probability: globalProbability));
            added++;
        }
        if (s.EnableRandomSwap)
        {
            pipeline.Add(new RandomSwap<T>(
                numSwaps: s.NumSwaps,
                probability: globalProbability));
            added++;
        }
        if (s.EnableRandomInsertion)
        {
            pipeline.Add(new RandomInsertion<T>(probability: globalProbability));
            added++;
        }
        return added > 0 ? pipeline : null;
    }

    /// <summary>
    /// Builds a video-modality pipeline (<see cref="ImageTensor{T}"/> array
    /// data) from <paramref name="s"/>. Honors temporal augmenters
    /// (<c>EnableTemporalCrop</c>, <c>EnableTemporalFlip</c>,
    /// <c>EnableFrameDropout</c>, <c>EnableSpeedChange</c>) and the spatial
    /// per-frame jitter via <c>VideoColorJitter</c>.
    /// </summary>
    public static AugmentationPipeline<T, AiDotNet.Augmentation.Image.ImageTensor<T>[]>? BuildVideoAugmenter<T>(
        VideoAugmentationSettings s,
        double globalProbability)
    {
        if (s is null) return null;
        var pipeline = new AugmentationPipeline<T, AiDotNet.Augmentation.Image.ImageTensor<T>[]>("VideoAugmenter");
        int added = 0;
        if (s.EnableTemporalCrop)
        {
            pipeline.Add(new TemporalCrop<T>(
                minCropRatio: System.Math.Max(0.01, s.CropRatio - 0.1),
                maxCropRatio: System.Math.Min(1.0, s.CropRatio + 0.1),
                probability: globalProbability));
            added++;
        }
        if (s.EnableTemporalFlip)
        {
            pipeline.Add(new TemporalFlip<T>(probability: globalProbability));
            added++;
        }
        if (s.EnableFrameDropout)
        {
            pipeline.Add(new FrameDropout<T>(
                dropoutRate: s.DropoutRate,
                probability: globalProbability));
            added++;
        }
        if (s.EnableSpeedChange)
        {
            pipeline.Add(new SpeedChange<T>(
                minSpeed: s.MinSpeed,
                maxSpeed: s.MaxSpeed,
                probability: globalProbability));
            added++;
        }
        if (s.EnableSpatialTransforms)
        {
            var spatial = s.SpatialSettings ?? new ImageAugmentationSettings();
            pipeline.Add(new VideoColorJitter<T>(
                brightnessRange: spatial.BrightnessRange,
                contrastRange: spatial.ContrastRange,
                saturationRange: spatial.SaturationRange,
                probability: globalProbability));
            added++;
        }
        return added > 0 ? pipeline : null;
    }
}
