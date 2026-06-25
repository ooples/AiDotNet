---
title: "Augmentation"
description: "All 250 public types in the AiDotNet.augmentation namespace, organized by kind."
section: "API Reference"
---

**250** public types in this namespace, organized by kind.

## Models & Types (192)

| Type | Summary |
|:-----|:--------|
| [`AdaIN<T>`](/docs/reference/wiki/augmentation/adain/) | Adaptive Instance Normalization (AdaIN) for style transfer augmentation. |
| [`AdasynAugmenter<T>`](/docs/reference/wiki/augmentation/adasynaugmenter/) | Implements ADASYN (Adaptive Synthetic Sampling) for imbalanced datasets. |
| [`Affine<T>`](/docs/reference/wiki/augmentation/affine/) | Applies random affine transformations (rotation, scale, shear, translation) to an image. |
| [`AudioNoise<T>`](/docs/reference/wiki/augmentation/audionoise/) | Adds background noise to audio data. |
| [`AugMax<T>`](/docs/reference/wiki/augmentation/augmax/) | AugMax (Wang et al., 2021) - adversarial composition of random augmentations. |
| [`AugmentationAppliedEventArgs<T>`](/docs/reference/wiki/augmentation/augmentationappliedeventargs/) | Event arguments raised when an augmentation is applied. |
| [`AugmentationContext<T>`](/docs/reference/wiki/augmentation/augmentationcontext/) | Provides runtime context for augmentation operations including random state, training mode, and spatial targets. |
| [`AugmentationPipeline<T, TData>`](/docs/reference/wiki/augmentation/augmentationpipeline/) | Represents a pipeline of augmentations that are applied in sequence or composition. |
| [`AugmentationPolicyEntry<T>`](/docs/reference/wiki/augmentation/augmentationpolicyentry/) | An entry in an augmentation policy. |
| [`AugmentationPolicy<T>`](/docs/reference/wiki/augmentation/augmentationpolicy/) | Defines a serializable augmentation policy with named transforms and parameters. |
| [`AugmentationRecommendation`](/docs/reference/wiki/augmentation/augmentationrecommendation/) | Represents a recommendation for an augmentation with its configuration. |
| [`AugmentationScheduler<T>`](/docs/reference/wiki/augmentation/augmentationscheduler/) | Schedules augmentation strength changes during training. |
| [`AugmentationSearchSpace`](/docs/reference/wiki/augmentation/augmentationsearchspace/) | Represents the search space for augmentation hyperparameters. |
| [`AugmentationValidationResult`](/docs/reference/wiki/augmentation/augmentationvalidationresult/) | Result of validating augmentations for a task. |
| [`AugmentedSample<T, TData>`](/docs/reference/wiki/augmentation/augmentedsample/) | Represents a sample with its data and associated spatial targets. |
| [`AutoAugment<T>`](/docs/reference/wiki/augmentation/autoaugment/) | AutoAugment (Cubuk et al., 2018) - applies learned augmentation policies. |
| [`AutoContrast<T>`](/docs/reference/wiki/augmentation/autocontrast/) | Maximizes image contrast by stretching the intensity range to fill [0, max]. |
| [`BBoxClipToImage<T>`](/docs/reference/wiki/augmentation/bboxcliptoimage/) | Clips bounding boxes to image boundaries. |
| [`BBoxSafeRandomCrop<T>`](/docs/reference/wiki/augmentation/bboxsaferandomcrop/) | Random crop that ensures bounding boxes remain valid after cropping. |
| [`BiasFieldCorrection<T>`](/docs/reference/wiki/augmentation/biasfieldcorrection/) | Simulates or corrects MRI bias field (intensity non-uniformity). |
| [`BorderlineSmoteAugmenter<T>`](/docs/reference/wiki/augmentation/borderlinesmoteaugmenter/) | Implements Borderline-SMOTE for imbalanced datasets, focusing on samples near the decision boundary. |
| [`BoundingBox<T>`](/docs/reference/wiki/augmentation/boundingbox/) | Represents a bounding box annotation for object detection. |
| [`BoxBlur<T>`](/docs/reference/wiki/augmentation/boxblur/) | Applies simple box (mean) blur using a uniform kernel. |
| [`Brightness<T>`](/docs/reference/wiki/augmentation/brightness/) | Adjusts the brightness of an image by adding a random offset to all pixel values. |
| [`CLAHE<T>`](/docs/reference/wiki/augmentation/clahe/) | Applies Contrast Limited Adaptive Histogram Equalization (CLAHE). |
| [`CenterCropOrPad<T>`](/docs/reference/wiki/augmentation/centercroporpad/) | Crops or pads an image to reach the target size, centering the content. |
| [`CenterCrop<T>`](/docs/reference/wiki/augmentation/centercrop/) | Crops the center region of an image to a specified size. |
| [`ChannelDropout<T>`](/docs/reference/wiki/augmentation/channeldropout/) | Randomly zeros out one or more color channels. |
| [`ChannelShuffle<T>`](/docs/reference/wiki/augmentation/channelshuffle/) | Randomly shuffles the order of color channels. |
| [`Closing<T>`](/docs/reference/wiki/augmentation/closing/) | Morphological closing (dilation followed by erosion) - removes small dark spots. |
| [`Clouds<T>`](/docs/reference/wiki/augmentation/clouds/) | Simulates cloud overlay on the image. |
| [`CoarseDropout<T>`](/docs/reference/wiki/augmentation/coarsedropout/) | Drops rectangular regions from the image (similar to Cutout/Random Erasing). |
| [`ColorConstancy<T>`](/docs/reference/wiki/augmentation/colorconstancy/) | Applies color constancy correction using the Gray World assumption or Max-RGB method. |
| [`ColorJitter<T>`](/docs/reference/wiki/augmentation/colorjitter/) | Applies random combinations of brightness, contrast, saturation, and hue adjustments. |
| [`Compose<T, TData>`](/docs/reference/wiki/augmentation/compose/) | Applies multiple augmentations sequentially. |
| [`Contrast<T>`](/docs/reference/wiki/augmentation/contrast/) | Adjusts the contrast of an image by scaling pixel values around the mean. |
| [`CopyPaste<T>`](/docs/reference/wiki/augmentation/copypaste/) | Copy-Paste augmentation (Ghiasi et al., 2020) - copies random regions from a source image and pastes them onto the target image. |
| [`CropNonEmptyMaskIfExists<T>`](/docs/reference/wiki/augmentation/cropnonemptymaskifexists/) | Crops the image to a region that contains non-empty mask content. |
| [`CutMix<T>`](/docs/reference/wiki/augmentation/cutmix/) | Cuts a rectangular region from one image and pastes it onto another (CutMix augmentation). |
| [`Cutout<T>`](/docs/reference/wiki/augmentation/cutout/) | Randomly masks out (cuts out) rectangular regions of an image. |
| [`DADA<T>`](/docs/reference/wiki/augmentation/dada/) | DADA - Differentiable Automatic Data Augmentation (Li et al., 2020). |
| [`DatasetCharacteristics`](/docs/reference/wiki/augmentation/datasetcharacteristics/) | Represents metadata about a dataset for augmentation recommendations. |
| [`Defocus<T>`](/docs/reference/wiki/augmentation/defocus/) | Simulates defocus blur using a circular (disc) kernel. |
| [`Denormalize<T>`](/docs/reference/wiki/augmentation/denormalize/) | Reverses normalization of an image tensor, restoring original pixel value ranges. |
| [`Dilate<T>`](/docs/reference/wiki/augmentation/dilate/) | Morphological dilation - expands bright regions using a structuring element. |
| [`Downscale<T>`](/docs/reference/wiki/augmentation/downscale/) | Downscales then upscales the image to simulate resolution degradation. |
| [`ElasticTransform<T>`](/docs/reference/wiki/augmentation/elastictransform/) | Applies elastic deformation to an image (Simard et al., 2003). |
| [`Emboss<T>`](/docs/reference/wiki/augmentation/emboss/) | Applies an emboss filter to create a 3D shadow effect. |
| [`Equalize<T>`](/docs/reference/wiki/augmentation/equalize/) | Equalizes the image histogram per channel (same as HistogramEqualization with per-channel mode). |
| [`Erode<T>`](/docs/reference/wiki/augmentation/erode/) | Morphological erosion - shrinks bright regions using a structuring element. |
| [`FDA<T>`](/docs/reference/wiki/augmentation/fda/) | Fourier Domain Adaptation (Yang & Soatto, 2020) - swaps low-frequency spectral components between source and target images for domain adaptation. |
| [`FMix<T>`](/docs/reference/wiki/augmentation/fmix/) | FMix augmentation (Harris et al., 2020) - mixes images using Fourier-space masks. |
| [`FancyPCA<T>`](/docs/reference/wiki/augmentation/fancypca/) | Fancy PCA color augmentation as described in AlexNet (Krizhevsky et al. |
| [`FastAutoAugment<T>`](/docs/reference/wiki/augmentation/fastautoaugment/) | Fast AutoAugment (Lim et al., 2019) - efficient augmentation policy search using density matching. |
| [`FeatureDropout<T>`](/docs/reference/wiki/augmentation/featuredropout/) | Randomly masks (sets to zero) features in tabular data. |
| [`FeatureNoise<T>`](/docs/reference/wiki/augmentation/featurenoise/) | Adds Gaussian noise to numerical features in tabular data. |
| [`FiveCrop<T>`](/docs/reference/wiki/augmentation/fivecrop/) | Extracts five crops from an image: four corners and center. |
| [`Fog<T>`](/docs/reference/wiki/augmentation/fog/) | Simulates fog effect by blending the image with a fog layer. |
| [`FrameDropout<T>`](/docs/reference/wiki/augmentation/framedropout/) | Randomly drops frames from video. |
| [`Frost<T>`](/docs/reference/wiki/augmentation/frost/) | Simulates frost/ice crystal patterns on the image. |
| [`GammaCorrection<T>`](/docs/reference/wiki/augmentation/gammacorrection/) | Applies gamma correction to adjust image brightness non-linearly. |
| [`GaussianBlur<T>`](/docs/reference/wiki/augmentation/gaussianblur/) | Applies Gaussian blur to an image. |
| [`GaussianNoise<T>`](/docs/reference/wiki/augmentation/gaussiannoise/) | Adds Gaussian noise to an image. |
| [`GhostingArtifact<T>`](/docs/reference/wiki/augmentation/ghostingartifact/) | Simulates MRI ghosting artifacts caused by motion during acquisition. |
| [`GlassBlur<T>`](/docs/reference/wiki/augmentation/glassblur/) | Applies glass-like blur by randomly displacing pixels then smoothing. |
| [`GrayscaleToRgb<T>`](/docs/reference/wiki/augmentation/grayscaletorgb/) | Converts a grayscale image to RGB by replicating the single channel across all three channels. |
| [`GridDistortion<T>`](/docs/reference/wiki/augmentation/griddistortion/) | Applies grid-based distortion to an image. |
| [`GridDropout<T>`](/docs/reference/wiki/augmentation/griddropout/) | Drops cells from a regular grid overlay on the image. |
| [`GridMask<T>`](/docs/reference/wiki/augmentation/gridmask/) | GridMask augmentation (Chen et al. |
| [`HideAndSeek<T>`](/docs/reference/wiki/augmentation/hideandseek/) | Hide-and-Seek augmentation (Singh & Lee 2017) that randomly hides grid patches. |
| [`HistogramColorTransfer<T>`](/docs/reference/wiki/augmentation/histogramcolortransfer/) | Transfers color distribution from a reference image using histogram matching. |
| [`HistogramEqualization<T>`](/docs/reference/wiki/augmentation/histogramequalization/) | Applies standard histogram equalization to improve image contrast. |
| [`HistogramMatching<T>`](/docs/reference/wiki/augmentation/histogrammatching/) | Matches the histogram of an image to a reference histogram or target distribution. |
| [`HorizontalFlip<T>`](/docs/reference/wiki/augmentation/horizontalflip/) | Flips an image horizontally (left-right mirror). |
| [`HueSaturationValue<T>`](/docs/reference/wiki/augmentation/huesaturationvalue/) | Randomly adjusts hue, saturation, and value in HSV space. |
| [`HyperparameterDefinition`](/docs/reference/wiki/augmentation/hyperparameterdefinition/) | Represents a hyperparameter definition for AutoML search. |
| [`ISONoise<T>`](/docs/reference/wiki/augmentation/isonoise/) | Simulates camera sensor noise (ISO noise) with separate color and intensity components. |
| [`ImageCompression<T>`](/docs/reference/wiki/augmentation/imagecompression/) | General compression artifact simulation that randomly selects between JPEG and WebP styles. |
| [`ImagePreprocessor<T>`](/docs/reference/wiki/augmentation/imagepreprocessor/) | Unified preprocessing pipeline builder for chaining image transformations. |
| [`ImageTensor<T>`](/docs/reference/wiki/augmentation/imagetensor/) | Represents an image as a tensor with image-specific metadata and operations. |
| [`IntensityNormalization<T>`](/docs/reference/wiki/augmentation/intensitynormalization/) | Medical image intensity normalization using z-score or min-max normalization. |
| [`Invert<T>`](/docs/reference/wiki/augmentation/invert/) | Inverts all pixel values in the image (creates a negative). |
| [`JpegCompression<T>`](/docs/reference/wiki/augmentation/jpegcompression/) | Simulates JPEG compression artifacts by quantizing DCT coefficients. |
| [`KSpaceMotion<T>`](/docs/reference/wiki/augmentation/kspacemotion/) | Simulates K-space motion artifacts in MRI by introducing phase shifts. |
| [`Keypoint<T>`](/docs/reference/wiki/augmentation/keypoint/) | Represents a keypoint annotation for pose estimation and landmark detection. |
| [`LabelMixingEventArgs<T>`](/docs/reference/wiki/augmentation/labelmixingeventargs/) | Event arguments for label mixing operations in augmentations like Mixup and CutMix. |
| [`LongestMaxSize<T>`](/docs/reference/wiki/augmentation/longestmaxsize/) | Resizes the image so that the longest edge equals the specified max size, preserving aspect ratio. |
| [`MaskDropout<T>`](/docs/reference/wiki/augmentation/maskdropout/) | Drops random object masks from the segmentation mask. |
| [`MedianBlur<T>`](/docs/reference/wiki/augmentation/medianblur/) | Applies median filter blur, effective at removing salt-and-pepper noise while preserving edges. |
| [`MinIoURandomCrop<T>`](/docs/reference/wiki/augmentation/miniourandomcrop/) | Random crop with minimum IoU guarantee for bounding boxes (SSD-style). |
| [`MixUp<T>`](/docs/reference/wiki/augmentation/mixup/) | Blends two images together by weighted averaging (MixUp augmentation). |
| [`MorphologicalGradient<T>`](/docs/reference/wiki/augmentation/morphologicalgradient/) | Morphological gradient - the difference between dilation and erosion, highlighting edges. |
| [`Mosaic9<T>`](/docs/reference/wiki/augmentation/mosaic9/) | 9-image mosaic variant that arranges images in a 3x3 grid. |
| [`Mosaic<T>`](/docs/reference/wiki/augmentation/mosaic/) | YOLO-style 4-image mosaic augmentation (Bochkovskiy et al., 2020). |
| [`MotionBlur<T>`](/docs/reference/wiki/augmentation/motionblur/) | Applies directional motion blur simulating camera or object motion. |
| [`MultiplicativeNoise<T>`](/docs/reference/wiki/augmentation/multiplicativenoise/) | Applies multiplicative Gaussian noise: output = input * (1 + noise). |
| [`NearMissUnderSampler<T>`](/docs/reference/wiki/augmentation/nearmissundersampler/) |  |
| [`Normalize<T>`](/docs/reference/wiki/augmentation/normalize/) | Normalizes an image tensor with per-channel mean and standard deviation. |
| [`OneOf<T, TData>`](/docs/reference/wiki/augmentation/oneof/) | Randomly selects and applies exactly one augmentation from a set. |
| [`Opening<T>`](/docs/reference/wiki/augmentation/opening/) | Morphological opening (erosion followed by dilation) - removes small bright spots. |
| [`OpticalDistortion<T>`](/docs/reference/wiki/augmentation/opticaldistortion/) | Simulates barrel and pincushion lens distortion. |
| [`PadToSquare<T>`](/docs/reference/wiki/augmentation/padtosquare/) | Pads an image to make it square while preserving the original content centered. |
| [`Pad<T>`](/docs/reference/wiki/augmentation/pad/) | Pads an image with configurable padding amounts and fill modes. |
| [`Perspective<T>`](/docs/reference/wiki/augmentation/perspective/) | Applies a random perspective transformation to an image. |
| [`PiecewiseAffine<T>`](/docs/reference/wiki/augmentation/piecewiseaffine/) | Applies piecewise affine transformation by dividing the image into triangular regions. |
| [`PitchShift<T>`](/docs/reference/wiki/augmentation/pitchshift/) | Shifts the pitch of audio without changing tempo using WSOLA (Waveform Similarity Overlap-Add). |
| [`PixelDropout<T>`](/docs/reference/wiki/augmentation/pixeldropout/) | Randomly sets individual pixels to a fill value. |
| [`PlasmaTransform<T>`](/docs/reference/wiki/augmentation/plasmatransform/) | Plasma fractal transformation - applies a procedural plasma-like color perturbation. |
| [`PoissonNoise<T>`](/docs/reference/wiki/augmentation/poissonnoise/) | Adds Poisson (shot) noise that scales with pixel intensity, simulating photon counting noise. |
| [`PolicySearchSpace`](/docs/reference/wiki/augmentation/policysearchspace/) | Represents a complete search space for augmentation policies. |
| [`Posterize<T>`](/docs/reference/wiki/augmentation/posterize/) | Reduces the number of bits per color channel, creating a poster-like effect. |
| [`PuzzleMix<T>`](/docs/reference/wiki/augmentation/puzzlemix/) | PuzzleMix (Kim et al., 2020) - optimal mixing using saliency and local statistics. |
| [`Rain<T>`](/docs/reference/wiki/augmentation/rain/) | Simulates rain streaks on the image. |
| [`RandAugment<T>`](/docs/reference/wiki/augmentation/randaugment/) | RandAugment (Cubuk et al., 2019) - applies N random augmentations at magnitude M. |
| [`RandomBrightnessContrast<T>`](/docs/reference/wiki/augmentation/randombrightnesscontrast/) | Randomly adjusts brightness and contrast simultaneously. |
| [`RandomCrop<T>`](/docs/reference/wiki/augmentation/randomcrop/) | Randomly crops a region from the image. |
| [`RandomDeletion<T>`](/docs/reference/wiki/augmentation/randomdeletion/) | Randomly deletes words from text. |
| [`RandomErasing<T>`](/docs/reference/wiki/augmentation/randomerasing/) | Random Erasing augmentation (Zhong et al. |
| [`RandomGamma<T>`](/docs/reference/wiki/augmentation/randomgamma/) | Applies random gamma correction with a random gamma value within a range. |
| [`RandomInsertion<T>`](/docs/reference/wiki/augmentation/randominsertion/) | Randomly inserts synonyms of existing words into the text. |
| [`RandomResizedCrop<T>`](/docs/reference/wiki/augmentation/randomresizedcrop/) | Randomly crops and resizes a region of the image (PyTorch-style RandomResizedCrop). |
| [`RandomSizedBBoxSafeCrop<T>`](/docs/reference/wiki/augmentation/randomsizedbboxsafecrop/) | Random-sized crop with bounding box safety, followed by resize to target size. |
| [`RandomSwap<T>`](/docs/reference/wiki/augmentation/randomswap/) | Randomly swaps the positions of words in text. |
| [`RandomToneCurve<T>`](/docs/reference/wiki/augmentation/randomtonecurve/) | Applies a random tone curve transformation to adjust image tonality. |
| [`RandomUnderSampler<T>`](/docs/reference/wiki/augmentation/randomundersampler/) | Implements random undersampling to balance imbalanced datasets. |
| [`ResizeMix<T>`](/docs/reference/wiki/augmentation/resizemix/) | ResizeMix augmentation - resizes one image and pastes it onto another. |
| [`ResizeWithAspectRatio<T>`](/docs/reference/wiki/augmentation/resizewithaspectratio/) | Resizes an image to fit within target dimensions while preserving the aspect ratio, then pads to reach the exact target size. |
| [`Resize<T>`](/docs/reference/wiki/augmentation/resize/) | Resizes an image to a target size using configurable interpolation. |
| [`RgbShift<T>`](/docs/reference/wiki/augmentation/rgbshift/) | Independently shifts each RGB channel by a random amount. |
| [`RgbToBgr<T>`](/docs/reference/wiki/augmentation/rgbtobgr/) | Converts between RGB and BGR color channel orderings. |
| [`RgbToGrayscale<T>`](/docs/reference/wiki/augmentation/rgbtograyscale/) | Converts an RGB image to grayscale using configurable channel weights. |
| [`RgbToHls<T>`](/docs/reference/wiki/augmentation/rgbtohls/) | Converts an image between RGB and HLS (Hue, Lightness, Saturation) color spaces. |
| [`RgbToHsv<T>`](/docs/reference/wiki/augmentation/rgbtohsv/) | Converts an image between RGB and HSV color spaces. |
| [`RgbToLab<T>`](/docs/reference/wiki/augmentation/rgbtolab/) | Converts an image between RGB and CIE L*a*b* color space. |
| [`RgbToXyz<T>`](/docs/reference/wiki/augmentation/rgbtoxyz/) | Converts an image between RGB and CIE XYZ color space. |
| [`RgbToYuv<T>`](/docs/reference/wiki/augmentation/rgbtoyuv/) | Converts an image between RGB and YUV color spaces. |
| [`Rotation<T>`](/docs/reference/wiki/augmentation/rotation/) | Rotates an image by a random angle within a specified range. |
| [`RowShuffle<T>`](/docs/reference/wiki/augmentation/rowshuffle/) | Shuffles rows within a batch of tabular data. |
| [`SaliencyMix<T>`](/docs/reference/wiki/augmentation/saliencymix/) | SaliencyMix - saliency-guided image mixing that pastes salient regions. |
| [`SaltAndPepperNoise<T>`](/docs/reference/wiki/augmentation/saltandpeppernoise/) | Adds salt-and-pepper (impulse) noise by randomly setting pixels to black or white. |
| [`SamplePairing<T>`](/docs/reference/wiki/augmentation/samplepairing/) | Sample Pairing augmentation (Inoue, 2018) - averages two images together. |
| [`Saturation<T>`](/docs/reference/wiki/augmentation/saturation/) | Adjusts the saturation (color intensity) of an image. |
| [`Scale<T>`](/docs/reference/wiki/augmentation/scale/) | Scales an image by a random factor within a specified range. |
| [`SegmentationMask<T>`](/docs/reference/wiki/augmentation/segmentationmask/) | Represents a segmentation mask for pixel-level annotations. |
| [`Shadow<T>`](/docs/reference/wiki/augmentation/shadow/) | Adds random shadow regions to the image. |
| [`Sharpen<T>`](/docs/reference/wiki/augmentation/sharpen/) | Applies a sharpening filter using a 3x3 kernel. |
| [`SkeletonDefinition`](/docs/reference/wiki/augmentation/skeletondefinition/) | Represents a skeleton definition for pose estimation. |
| [`SliceSelection<T>`](/docs/reference/wiki/augmentation/sliceselection/) | Randomly selects a slice from a multi-channel volume (simulates 3D volume slice selection). |
| [`SmallestMaxSize<T>`](/docs/reference/wiki/augmentation/smallestmaxsize/) | Resizes the image so that the shortest edge equals the specified max size, preserving aspect ratio. |
| [`SmoteAugmenter<T>`](/docs/reference/wiki/augmentation/smoteaugmenter/) | Implements SMOTE (Synthetic Minority Over-sampling Technique) for imbalanced datasets. |
| [`SmoteEnnAugmenter<T>`](/docs/reference/wiki/augmentation/smoteennaugmenter/) | Implements SMOTE-ENN combination for imbalanced datasets. |
| [`SmoteTomekAugmenter<T>`](/docs/reference/wiki/augmentation/smotetomekaugmenter/) | Implements SMOTE-Tomek combination for imbalanced datasets. |
| [`SnapMix<T>`](/docs/reference/wiki/augmentation/snapmix/) | SnapMix (Huang et al., 2021) - semantically proportional mixing for fine-grained recognition. |
| [`Snow<T>`](/docs/reference/wiki/augmentation/snow/) | Simulates snow effect on the image. |
| [`Solarize<T>`](/docs/reference/wiki/augmentation/solarize/) | Inverts all pixel values above a threshold, creating a solarization effect. |
| [`SomeOf<T, TData>`](/docs/reference/wiki/augmentation/someof/) | Randomly selects and applies N augmentations from a set. |
| [`SpatialTransform<T>`](/docs/reference/wiki/augmentation/spatialtransform/) | Applies spatial transformations (flips, rotations) consistently to all frames. |
| [`Spatter<T>`](/docs/reference/wiki/augmentation/spatter/) | Simulates spatter effects (mud, rain drops) on the image. |
| [`SpeckleNoise<T>`](/docs/reference/wiki/augmentation/specklenoise/) | Adds multiplicative speckle noise: output = input + input * noise. |
| [`SpeedChange<T>`](/docs/reference/wiki/augmentation/speedchange/) | Changes the playback speed of video by resampling frames. |
| [`SpikeArtifact<T>`](/docs/reference/wiki/augmentation/spikeartifact/) | Simulates MRI spike (Herringbone) artifacts caused by electrical interference in k-space. |
| [`StackSlices<T>`](/docs/reference/wiki/augmentation/stackslices/) | Stacks adjacent 2D slices to create multi-channel input (useful for 3D volume processing). |
| [`StyleMix<T>`](/docs/reference/wiki/augmentation/stylemix/) | Style mixing augmentation - transfers statistical style (mean/variance) from a reference image. |
| [`SunFlare<T>`](/docs/reference/wiki/augmentation/sunflare/) | Simulates sun flare / lens flare effect on the image. |
| [`Superpixels<T>`](/docs/reference/wiki/augmentation/superpixels/) | Replaces random regions with their superpixel (average color) representation. |
| [`SvmSmoteAugmenter<T>`](/docs/reference/wiki/augmentation/svmsmoteaugmenter/) | Implements SVM-SMOTE for imbalanced datasets, using SVM decision boundary to identify borderline samples. |
| [`SynonymReplacement<T>`](/docs/reference/wiki/augmentation/synonymreplacement/) | Replaces random words with their synonyms. |
| [`TabularMixUp<T>`](/docs/reference/wiki/augmentation/tabularmixup/) | Applies MixUp augmentation to tabular data by linearly interpolating between samples. |
| [`TemporalCrop<T>`](/docs/reference/wiki/augmentation/temporalcrop/) | Randomly crops a temporal segment from video. |
| [`TemporalFlip<T>`](/docs/reference/wiki/augmentation/temporalflip/) | Reverses the order of frames in a video. |
| [`TenCrop<T>`](/docs/reference/wiki/augmentation/tencrop/) | Extracts ten crops from an image: five crops plus their horizontal flips. |
| [`TestTimeAugmentationResult<TOutput>`](/docs/reference/wiki/augmentation/testtimeaugmentationresult/) | Contains the result of a Test-Time Augmentation prediction, including individual and aggregated predictions. |
| [`TextureOverlay<T>`](/docs/reference/wiki/augmentation/textureoverlay/) | Overlays a procedurally generated random texture on the image. |
| [`ThinPlateSpline<T>`](/docs/reference/wiki/augmentation/thinplatespline/) | Applies thin plate spline (TPS) transformation to an image. |
| [`TimeShift<T>`](/docs/reference/wiki/augmentation/timeshift/) | Shifts audio forward or backward in time. |
| [`TimeStretch<T>`](/docs/reference/wiki/augmentation/timestretch/) | Stretches or compresses audio in time without changing pitch. |
| [`ToFloat<T>`](/docs/reference/wiki/augmentation/tofloat/) | Converts an image tensor to floating-point representation with configurable scaling. |
| [`ToGray<T>`](/docs/reference/wiki/augmentation/togray/) | Converts to grayscale with random channel weights, outputting 3 channels. |
| [`ToSepia<T>`](/docs/reference/wiki/augmentation/tosepia/) | Applies a sepia tone filter to the image. |
| [`ToTensor<T>`](/docs/reference/wiki/augmentation/totensor/) | Converts an image to a normalized tensor with values in [0, 1]. |
| [`TokenMix<T>`](/docs/reference/wiki/augmentation/tokenmix/) | TokenMix - token-level mixing for transformer architectures. |
| [`TomekLinksAugmenter<T>`](/docs/reference/wiki/augmentation/tomeklinksaugmenter/) | Implements Tomek Links removal for cleaning decision boundaries in imbalanced datasets. |
| [`TransMix<T>`](/docs/reference/wiki/augmentation/transmix/) | TransMix (Chen et al., 2022) - attention-guided mixing for Vision Transformers. |
| [`TrivialAugment<T>`](/docs/reference/wiki/augmentation/trivialaugment/) | TrivialAugmentWide (Muller & Hutter, 2021) - applies a single random augmentation with uniformly sampled magnitude. |
| [`UniformAugment<T>`](/docs/reference/wiki/augmentation/uniformaugment/) | UniformAugment - applies a random number of randomly selected augmentations with uniformly sampled magnitudes. |
| [`UnsharpMask<T>`](/docs/reference/wiki/augmentation/unsharpmask/) | Applies unsharp masking to sharpen the image. |
| [`VerticalFlip<T>`](/docs/reference/wiki/augmentation/verticalflip/) | Flips an image vertically (top-bottom mirror). |
| [`VideoColorJitter<T>`](/docs/reference/wiki/augmentation/videocolorjitter/) | Applies color jitter (brightness, contrast, saturation) to video frames. |
| [`VolumeChange<T>`](/docs/reference/wiki/augmentation/volumechange/) | Randomly changes the volume (gain) of audio. |
| [`WebPCompression<T>`](/docs/reference/wiki/augmentation/webpcompression/) | Simulates WebP compression artifacts (similar to JPEG but with different characteristics). |
| [`WindowLevel<T>`](/docs/reference/wiki/augmentation/windowlevel/) | CT/MRI window/level adjustment for medical image augmentation. |
| [`ZoomBlur<T>`](/docs/reference/wiki/augmentation/zoomblur/) | Applies radial/zoom blur emanating from the image center. |

## Base Classes (13)

| Type | Summary |
|:-----|:--------|
| [`AudioAugmenterBase<T>`](/docs/reference/wiki/augmentation/audioaugmenterbase/) | Base class for audio data augmentations. |
| [`AugmentationBase<T, TData>`](/docs/reference/wiki/augmentation/augmentationbase/) | Abstract base class for all augmentations providing common functionality. |
| [`ImageAugmenterBase<T>`](/docs/reference/wiki/augmentation/imageaugmenterbase/) | Base class for image data augmentations. |
| [`ImageMixingAugmenterBase<T>`](/docs/reference/wiki/augmentation/imagemixingaugmenterbase/) | Base class for image augmentations that mix multiple images together. |
| [`LabelMixingAugmentationBase<T, TData>`](/docs/reference/wiki/augmentation/labelmixingaugmentationbase/) | Base class for label-mixing augmentations like Mixup and CutMix. |
| [`SpatialAugmentationBase<T, TData>`](/docs/reference/wiki/augmentation/spatialaugmentationbase/) | Base class for augmentations that transform spatial targets (bounding boxes, keypoints, masks). |
| [`SpatialImageAugmenterBase<T>`](/docs/reference/wiki/augmentation/spatialimageaugmenterbase/) | Base class for image augmentations that transform spatial targets. |
| [`SpatialVideoAugmenterBase<T>`](/docs/reference/wiki/augmentation/spatialvideoaugmenterbase/) | Base class for spatial video augmentations that apply image transforms to all frames. |
| [`TabularAugmenterBase<T>`](/docs/reference/wiki/augmentation/tabularaugmenterbase/) | Base class for tabular data augmentations. |
| [`TabularMixingAugmenterBase<T>`](/docs/reference/wiki/augmentation/tabularmixingaugmenterbase/) | Base class for tabular augmentations that mix multiple samples together. |
| [`TemporalAugmenterBase<T>`](/docs/reference/wiki/augmentation/temporalaugmenterbase/) | Base class for temporal video augmentations that modify the frame sequence. |
| [`TextAugmenterBase<T>`](/docs/reference/wiki/augmentation/textaugmenterbase/) | Base class for text data augmentations. |
| [`VideoAugmenterBase<T>`](/docs/reference/wiki/augmentation/videoaugmenterbase/) | Base class for video data augmentations. |

## Interfaces (11)

| Type | Summary |
|:-----|:--------|
| [`IAugmentationFactory<T, TData>`](/docs/reference/wiki/augmentation/iaugmentationfactory/) | Factory for creating augmentations from configuration. |
| [`IAugmentationPolicy`](/docs/reference/wiki/augmentation/iaugmentationpolicy/) | Non-generic base interface for augmentation policies. |
| [`IAugmentationPolicy<T, TData>`](/docs/reference/wiki/augmentation/iaugmentationpolicy-2/) | Interface for composable augmentation policies. |
| [`IAugmentationRecommender`](/docs/reference/wiki/augmentation/iaugmentationrecommender/) | Interface for recommending augmentations based on task and data characteristics. |
| [`IAugmentationSearcher<T, TData>`](/docs/reference/wiki/augmentation/iaugmentationsearcher/) | Interface for AutoML search algorithms over augmentation spaces. |
| [`IAugmentation<T, TData>`](/docs/reference/wiki/augmentation/iaugmentation/) | Base interface for all data augmentations across domains (image, audio, tabular). |
| [`IAutoMLAugmentation<T, TData>`](/docs/reference/wiki/augmentation/iautomlaugmentation/) | Interface for augmentations that expose their hyperparameter search space. |
| [`IAutoMLPolicy<T, TData>`](/docs/reference/wiki/augmentation/iautomlpolicy/) | Interface for policies that expose their complete search space. |
| [`ILabelMixingAugmentation<T, TData>`](/docs/reference/wiki/augmentation/ilabelmixingaugmentation/) | Interface for augmentations that modify labels (e.g., Mixup, CutMix). |
| [`ISpatialAugmentation<T, TData>`](/docs/reference/wiki/augmentation/ispatialaugmentation/) | Interface for augmentations that can transform spatial targets (bounding boxes, keypoints, segmentation masks). |
| [`IUnderSampler<T>`](/docs/reference/wiki/augmentation/iundersampler/) | Interface for undersampling techniques that reduce the majority class. |

## Enums (23)

| Type | Summary |
|:-----|:--------|
| [`AugmentOp`](/docs/reference/wiki/augmentation/augmentop/) | Augmentation operation type for AutoAugment policies. |
| [`AugmentationOrder`](/docs/reference/wiki/augmentation/augmentationorder/) | Specifies how multiple augmentations are applied in a pipeline. |
| [`AugmentationTaskType`](/docs/reference/wiki/augmentation/augmentationtasktype/) | Specifies the type of machine learning task for augmentation recommendations. |
| [`AutoAugmentPolicy`](/docs/reference/wiki/augmentation/autoaugmentpolicy/) | AutoAugment policy preset. |
| [`BorderMode`](/docs/reference/wiki/augmentation/bordermode/) | Specifies how to handle pixels that fall outside the image bounds during transformation. |
| [`BoundingBoxFormat`](/docs/reference/wiki/augmentation/boundingboxformat/) | Specifies the format of bounding box coordinates. |
| [`ChannelOrder`](/docs/reference/wiki/augmentation/channelorder/) | Specifies the channel ordering of an image tensor. |
| [`ColorConstancyMethod`](/docs/reference/wiki/augmentation/colorconstancymethod/) | Color constancy correction method. |
| [`ColorSpace`](/docs/reference/wiki/augmentation/colorspace/) | Specifies the color space of an image. |
| [`DataModality`](/docs/reference/wiki/augmentation/datamodality/) | Enum representing detected data modality types. |
| [`HyperparameterType`](/docs/reference/wiki/augmentation/hyperparametertype/) | Specifies the type of hyperparameter. |
| [`IntensityNormMethod`](/docs/reference/wiki/augmentation/intensitynormmethod/) | Intensity normalization method. |
| [`InterpolationMode`](/docs/reference/wiki/augmentation/interpolationmode/) | Specifies the interpolation method for image resizing. |
| [`MaskEncoding`](/docs/reference/wiki/augmentation/maskencoding/) | Specifies the encoding format for segmentation masks. |
| [`MaskType`](/docs/reference/wiki/augmentation/masktype/) | Specifies the type of segmentation mask. |
| [`MixingStrategy`](/docs/reference/wiki/augmentation/mixingstrategy/) | Specifies the mixing strategy used for label mixing. |
| [`NearMissVersion<T>`](/docs/reference/wiki/augmentation/nearmissversion/) | NearMiss variant versions. |
| [`NoiseType`](/docs/reference/wiki/augmentation/noisetype/) | Types of audio noise. |
| [`PaddingMode`](/docs/reference/wiki/augmentation/paddingmode/) | Specifies the padding mode for filling new pixels. |
| [`PredictionAggregationMethod`](/docs/reference/wiki/augmentation/predictionaggregationmethod/) | Specifies how to combine predictions from multiple augmented versions of the same input. |
| [`RemovalStrategy<T>`](/docs/reference/wiki/augmentation/removalstrategy/) | Strategy for removing samples in Tomek links. |
| [`ScheduleType`](/docs/reference/wiki/augmentation/scheduletype/) | Schedule type for augmentation strength. |
| [`SpatterMode`](/docs/reference/wiki/augmentation/spattermode/) | Spatter effect type. |

## Options & Configuration (8)

| Type | Summary |
|:-----|:--------|
| [`AudioAugmentationSettings`](/docs/reference/wiki/augmentation/audioaugmentationsettings/) | Audio-specific augmentation settings with industry-standard defaults. |
| [`AugmentationConfig`](/docs/reference/wiki/augmentation/augmentationconfig/) | Unified configuration for data augmentation with industry-standard defaults. |
| [`AugmentationConfig<T, TInput>`](/docs/reference/wiki/augmentation/augmentationconfig-2/) | Strongly-typed augmentation configuration parameterised by the model's numeric type and input type. |
| [`ImageAugmentationSettings`](/docs/reference/wiki/augmentation/imageaugmentationsettings/) | Image-specific augmentation settings with industry-standard defaults. |
| [`SampledConfiguration`](/docs/reference/wiki/augmentation/sampledconfiguration/) | Represents a sampled configuration from the search space. |
| [`TabularAugmentationSettings`](/docs/reference/wiki/augmentation/tabularaugmentationsettings/) | Tabular-specific augmentation settings with industry-standard defaults. |
| [`TextAugmentationSettings`](/docs/reference/wiki/augmentation/textaugmentationsettings/) | Text-specific augmentation settings with industry-standard defaults. |
| [`VideoAugmentationSettings`](/docs/reference/wiki/augmentation/videoaugmentationsettings/) | Video-specific augmentation settings with industry-standard defaults. |

## Helpers & Utilities (3)

| Type | Summary |
|:-----|:--------|
| [`DataModalityDetector`](/docs/reference/wiki/augmentation/datamodalitydetector/) | Utility for auto-detecting data modality from input types. |
| [`ImagePresets<T>`](/docs/reference/wiki/augmentation/imagepresets/) | Preset preprocessing configurations for common models. |
| [`PolicyRegistry<T>`](/docs/reference/wiki/augmentation/policyregistry/) | Registry for built-in and custom augmentation policies. |

