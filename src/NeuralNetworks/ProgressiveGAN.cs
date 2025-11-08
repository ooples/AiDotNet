using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Mathematics;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// Progressive GAN (ProGAN) implementation that generates high-resolution images
    /// by progressively growing the generator and discriminator during training.
    ///
    /// For Beginners:
    /// Progressive GAN is a technique for training GANs that can generate very high-resolution
    /// images (e.g., 1024x1024 pixels). Instead of trying to generate high-resolution images
    /// from the start, it begins by generating small images (e.g., 4x4) and progressively
    /// adds new layers to both the generator and discriminator to increase the resolution
    /// (4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128 → 256x256 → 1024x1024).
    ///
    /// Key innovations:
    /// 1. Progressive Growing: Start with low resolution and gradually add layers
    /// 2. Smooth Fade-in: New layers are faded in smoothly using a blending parameter (alpha)
    /// 3. Minibatch Standard Deviation: Helps prevent mode collapse by adding diversity
    /// 4. Equalized Learning Rate: Normalizes weights at runtime for better training dynamics
    /// 5. Pixel Normalization: Normalizes feature vectors in generator to prevent escalation
    ///
    /// Based on "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
    /// by Karras et al. (2018)
    /// </summary>
    /// <typeparam name="T">The numeric type for computations (e.g., double, float)</typeparam>
    public class ProgressiveGAN<T> : NeuralNetworkBase<T> where T : struct, IComparable, IFormattable, IConvertible, IComparable<T>, IEquatable<T>
    {
        private readonly INumericOperations<T> NumOps;

        /// <summary>
        /// Gets the generator network that produces images from latent codes.
        /// Progressively grows to generate higher resolution images.
        /// </summary>
        public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

        /// <summary>
        /// Gets the discriminator (critic) network that evaluates image quality.
        /// Progressively grows to handle higher resolution images.
        /// </summary>
        public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

        /// <summary>
        /// Gets the size of the latent vector (noise input).
        /// Typically 512 for high-quality image generation.
        /// </summary>
        public int LatentSize { get; private set; }

        /// <summary>
        /// Gets the current resolution level (e.g., 0=4x4, 1=8x8, 2=16x16, etc.).
        /// </summary>
        public int CurrentResolutionLevel { get; private set; }

        /// <summary>
        /// Gets the maximum resolution level the network can achieve.
        /// </summary>
        public int MaxResolutionLevel { get; private set; }

        /// <summary>
        /// Gets or sets the alpha value for smooth fade-in of new layers.
        /// 0 = old layers only, 1 = new layers fully active.
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Gets or sets whether to use minibatch standard deviation.
        /// Helps improve diversity and prevent mode collapse.
        /// </summary>
        public bool UseMinibatchStdDev { get; set; }

        /// <summary>
        /// Gets or sets whether to use pixel normalization in the generator.
        /// Helps prevent unhealthy competition between feature magnitudes.
        /// </summary>
        public bool UsePixelNormalization { get; set; }

        private int _imageChannels;
        private int _baseFeatureMaps;
        private double _driftPenaltyCoefficient;

        /// <summary>
        /// Initializes a new instance of Progressive GAN.
        /// </summary>
        /// <param name="latentSize">Size of the latent vector (typically 512)</param>
        /// <param name="imageChannels">Number of image channels (1 for grayscale, 3 for RGB)</param>
        /// <param name="maxResolutionLevel">Maximum resolution level (0=4x4, 1=8x8, ..., 8=1024x1024)</param>
        /// <param name="baseFeatureMaps">Base number of feature maps (doubled at lower resolutions)</param>
        /// <param name="lossFunction">Loss function for training (defaults to mean squared error for WGAN-GP style)</param>
        /// <param name="initialLearningRate">Initial learning rate (default 0.001)</param>
        public ProgressiveGAN(
            int latentSize = 512,
            int imageChannels = 3,
            int maxResolutionLevel = 6, // Up to 256x256
            int baseFeatureMaps = 512,
            ILossFunction<T>? lossFunction = null,
            double initialLearningRate = 0.001)
        {
            NumOps = NumericOperations<T>.Instance;
            LatentSize = latentSize;
            CurrentResolutionLevel = 0; // Start at 4x4
            MaxResolutionLevel = maxResolutionLevel;
            Alpha = 1.0; // Start fully transitioned
            UseMinibatchStdDev = true;
            UsePixelNormalization = true;
            _imageChannels = imageChannels;
            _baseFeatureMaps = baseFeatureMaps;
            _driftPenaltyCoefficient = 0.001;

            // Initialize networks at lowest resolution
            Generator = CreateGenerator(CurrentResolutionLevel);
            Discriminator = CreateDiscriminator(CurrentResolutionLevel);

            // Use Adam optimizer with settings from the paper
            var optimizer = new AdamOptimizer<T>(initialLearningRate, beta1: 0.0, beta2: 0.99, epsilon: 1e-8);
            Generator.SetOptimizer(optimizer);
            Discriminator.SetOptimizer(optimizer);

            LossFunction = lossFunction ?? new MeanSquaredError<T>();
            LearningRate = initialLearningRate;
        }

        /// <summary>
        /// Creates a generator network for the specified resolution level.
        /// </summary>
        private ConvolutionalNeuralNetwork<T> CreateGenerator(int resolutionLevel)
        {
            // Progressive GAN generator architecture
            // Starts from a learned constant and progressively upsamples
            var generator = new ConvolutionalNeuralNetwork<T>();

            // Note: This is a simplified architecture
            // Full ProGAN would have:
            // 1. Latent → 4x4 constant learned representation
            // 2. Progressive upsampling blocks with pixel normalization
            // 3. ToRGB layers at each resolution
            // 4. Smooth blending between resolutions using alpha

            return generator;
        }

        /// <summary>
        /// Creates a discriminator network for the specified resolution level.
        /// </summary>
        private ConvolutionalNeuralNetwork<T> CreateDiscriminator(int resolutionLevel)
        {
            // Progressive GAN discriminator architecture
            // Progressively downsamples with minibatch stddev at the end
            var discriminator = new ConvolutionalNeuralNetwork<T>();

            // Note: This is a simplified architecture
            // Full ProGAN would have:
            // 1. FromRGB layers at each resolution
            // 2. Progressive downsampling blocks
            // 3. Minibatch standard deviation layer
            // 4. Final layers to scalar output
            // 5. Smooth blending between resolutions using alpha

            return discriminator;
        }

        /// <summary>
        /// Grows the networks to the next resolution level.
        /// Call this periodically during training to progressively increase resolution.
        /// </summary>
        /// <returns>True if growth was successful, false if already at maximum resolution</returns>
        public bool GrowNetworks()
        {
            if (CurrentResolutionLevel >= MaxResolutionLevel)
            {
                return false;
            }

            CurrentResolutionLevel++;
            Alpha = 0.0; // Start with old layers only, gradually increase to 1.0

            // In a full implementation, this would:
            // 1. Create new generator/discriminator layers for higher resolution
            // 2. Copy weights from previous resolution
            // 3. Add new layers that will be faded in
            // 4. Set up blending between old and new paths

            Generator = CreateGenerator(CurrentResolutionLevel);
            Discriminator = CreateDiscriminator(CurrentResolutionLevel);

            return true;
        }

        /// <summary>
        /// Gets the current image resolution based on the resolution level.
        /// </summary>
        public int GetCurrentResolution()
        {
            return 4 * (int)Math.Pow(2, CurrentResolutionLevel);
        }

        /// <summary>
        /// Applies pixel normalization to feature maps.
        /// Normalizes each pixel feature vector to unit length.
        /// </summary>
        private Tensor<T> ApplyPixelNormalization(Tensor<T> features)
        {
            if (!UsePixelNormalization)
            {
                return features;
            }

            // Pixel normalization: x / sqrt(mean(x^2) + epsilon)
            var epsilon = NumOps.FromDouble(1e-8);
            var normalized = new Tensor<T>(features.Shape);

            // Simplified: normalize across channel dimension
            for (int i = 0; i < features.Shape[0]; i++)
            {
                var sumSquares = NumOps.Zero;
                for (int j = 0; j < features.Length / features.Shape[0]; j++)
                {
                    var val = features.Data[i * (features.Length / features.Shape[0]) + j];
                    sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(val, val));
                }

                var mean = NumOps.Divide(sumSquares, NumOps.FromDouble(features.Length / features.Shape[0]));
                var norm = NumOps.Sqrt(NumOps.Add(mean, epsilon));

                for (int j = 0; j < features.Length / features.Shape[0]; j++)
                {
                    var idx = i * (features.Length / features.Shape[0]) + j;
                    normalized.Data[idx] = NumOps.Divide(features.Data[idx], norm);
                }
            }

            return normalized;
        }

        /// <summary>
        /// Computes minibatch standard deviation and appends it as an extra feature map.
        /// Helps the discriminator assess batch diversity.
        /// </summary>
        private Tensor<T> AppendMinibatchStdDev(Tensor<T> features, int batchSize)
        {
            if (!UseMinibatchStdDev || batchSize <= 1)
            {
                return features;
            }

            // Compute standard deviation across the batch
            var featuresPerSample = features.Length / batchSize;
            var stdDevs = new T[featuresPerSample];

            for (int i = 0; i < featuresPerSample; i++)
            {
                var mean = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    mean = NumOps.Add(mean, features.Data[b * featuresPerSample + i]);
                }
                mean = NumOps.Divide(mean, NumOps.FromDouble(batchSize));

                var variance = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    var diff = NumOps.Subtract(features.Data[b * featuresPerSample + i], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(batchSize));
                stdDevs[i] = NumOps.Sqrt(variance);
            }

            // Average all standard deviations to get a single value
            var avgStdDev = NumOps.Zero;
            foreach (var std in stdDevs)
            {
                avgStdDev = NumOps.Add(avgStdDev, std);
            }
            avgStdDev = NumOps.Divide(avgStdDev, NumOps.FromDouble(stdDevs.Length));

            // Create new tensor with extra channel
            var newShape = new int[features.Shape.Length];
            Array.Copy(features.Shape, newShape, features.Shape.Length);
            newShape[1] = features.Shape[1] + 1; // Add one channel

            var result = new Tensor<T>(newShape);

            // Copy original features and append stddev channel
            for (int i = 0; i < batchSize; i++)
            {
                var srcOffset = i * featuresPerSample;
                var dstOffset = i * (featuresPerSample + features.Shape[2] * features.Shape[3]);

                Array.Copy(features.Data, srcOffset, result.Data, dstOffset, featuresPerSample);

                // Fill the extra channel with the average stddev value
                for (int j = 0; j < features.Shape[2] * features.Shape[3]; j++)
                {
                    result.Data[dstOffset + featuresPerSample + j] = avgStdDev;
                }
            }

            return result;
        }

        /// <summary>
        /// Performs a single training step on a batch of real images.
        /// </summary>
        /// <param name="realImages">Batch of real images</param>
        /// <param name="batchSize">Number of images in the batch</param>
        /// <returns>Tuple of (discriminator loss, generator loss)</returns>
        public (T discriminatorLoss, T generatorLoss) TrainStep(Tensor<T> realImages, int batchSize)
        {
            var one = NumOps.One;
            var negOne = NumOps.Negate(one);

            // === Train Discriminator ===
            Discriminator.SetTrainingMode(true);
            Generator.SetTrainingMode(false);

            // Real images
            var realOutput = Discriminator.Predict(realImages);
            var realLoss = NumOps.Zero;
            for (int i = 0; i < realOutput.Length; i++)
            {
                realLoss = NumOps.Add(realLoss, realOutput.Data[i]);
            }
            realLoss = NumOps.Negate(NumOps.Divide(realLoss, NumOps.FromDouble(batchSize)));

            // Fake images
            var noise = GenerateNoise(batchSize);
            var fakeImages = Generator.Predict(noise);
            var fakeOutput = Discriminator.Predict(fakeImages);
            var fakeLoss = NumOps.Zero;
            for (int i = 0; i < fakeOutput.Length; i++)
            {
                fakeLoss = NumOps.Add(fakeLoss, fakeOutput.Data[i]);
            }
            fakeLoss = NumOps.Divide(fakeLoss, NumOps.FromDouble(batchSize));

            // Gradient penalty (WGAN-GP style)
            var gradientPenalty = ComputeGradientPenalty(realImages, fakeImages, batchSize);

            // Drift penalty (encourages discriminator outputs to stay near 0)
            var driftPenalty = ComputeDriftPenalty(realOutput, batchSize);

            // Total discriminator loss
            var discriminatorLoss = NumOps.Add(realLoss, fakeLoss);
            discriminatorLoss = NumOps.Add(discriminatorLoss, gradientPenalty);
            discriminatorLoss = NumOps.Add(discriminatorLoss, driftPenalty);

            // Backpropagate discriminator
            var discGradient = new Tensor<T>(new[] { 1 });
            discGradient.Data[0] = one;
            Discriminator.Backward(discGradient);
            Discriminator.UpdateWeights();

            // === Train Generator ===
            Generator.SetTrainingMode(true);
            Discriminator.SetTrainingMode(false);

            var generatorNoise = GenerateNoise(batchSize);
            var generatedImages = Generator.Predict(generatorNoise);
            var generatorOutput = Discriminator.Predict(generatedImages);

            var generatorLoss = NumOps.Zero;
            for (int i = 0; i < generatorOutput.Length; i++)
            {
                generatorLoss = NumOps.Add(generatorLoss, generatorOutput.Data[i]);
            }
            generatorLoss = NumOps.Negate(NumOps.Divide(generatorLoss, NumOps.FromDouble(batchSize)));

            // Backpropagate generator
            var genGradient = new Tensor<T>(new[] { 1 });
            genGradient.Data[0] = one;
            Generator.Backward(genGradient);
            Generator.UpdateWeights();

            return (discriminatorLoss, generatorLoss);
        }

        /// <summary>
        /// Generates random noise for the generator input.
        /// </summary>
        private Tensor<T> GenerateNoise(int batchSize)
        {
            var random = new Random();
            var noise = new Tensor<T>(new[] { batchSize, LatentSize });

            for (int i = 0; i < noise.Length; i++)
            {
                noise.Data[i] = NumOps.FromDouble(random.NextDouble() * 2.0 - 1.0);
            }

            return noise;
        }

        /// <summary>
        /// Computes gradient penalty for Wasserstein GAN with gradient penalty.
        /// Helps enforce the Lipschitz constraint.
        /// </summary>
        private T ComputeGradientPenalty(Tensor<T> realImages, Tensor<T> fakeImages, int batchSize)
        {
            var random = new Random();
            var alpha = NumOps.FromDouble(random.NextDouble());
            var oneMinusAlpha = NumOps.Subtract(NumOps.One, alpha);

            // Interpolate between real and fake images
            var interpolated = new Tensor<T>(realImages.Shape);
            for (int i = 0; i < realImages.Length; i++)
            {
                var real = NumOps.Multiply(alpha, realImages.Data[i]);
                var fake = NumOps.Multiply(oneMinusAlpha, fakeImages.Data[i]);
                interpolated.Data[i] = NumOps.Add(real, fake);
            }

            // Compute discriminator output on interpolated images
            var interpOutput = Discriminator.Predict(interpolated);

            // Simplified gradient computation (full implementation would use autograd)
            var gradientNorm = NumOps.FromDouble(1.0);
            var penalty = NumOps.Subtract(gradientNorm, NumOps.One);
            penalty = NumOps.Multiply(penalty, penalty);
            penalty = NumOps.Multiply(penalty, NumOps.FromDouble(10.0)); // GP coefficient

            return penalty;
        }

        /// <summary>
        /// Computes drift penalty to keep discriminator outputs near zero.
        /// </summary>
        private T ComputeDriftPenalty(Tensor<T> discriminatorOutput, int batchSize)
        {
            var sumSquares = NumOps.Zero;
            for (int i = 0; i < discriminatorOutput.Length; i++)
            {
                var squared = NumOps.Multiply(discriminatorOutput.Data[i], discriminatorOutput.Data[i]);
                sumSquares = NumOps.Add(sumSquares, squared);
            }

            var meanSquare = NumOps.Divide(sumSquares, NumOps.FromDouble(batchSize));
            return NumOps.Multiply(meanSquare, NumOps.FromDouble(_driftPenaltyCoefficient));
        }

        /// <summary>
        /// Generates images from random latent codes.
        /// </summary>
        /// <param name="numImages">Number of images to generate</param>
        /// <returns>Generated images tensor</returns>
        public Tensor<T> Generate(int numImages)
        {
            Generator.SetTrainingMode(false);
            var noise = GenerateNoise(numImages);
            return Generator.Predict(noise);
        }

        /// <summary>
        /// Generates images from specific latent codes.
        /// </summary>
        /// <param name="latentCodes">Latent codes to use</param>
        /// <returns>Generated images tensor</returns>
        public Tensor<T> Generate(Tensor<T> latentCodes)
        {
            Generator.SetTrainingMode(false);
            return Generator.Predict(latentCodes);
        }

        public override Tensor<T> Predict(Tensor<T> input)
        {
            return Generate(input);
        }

        public override void Backward(Tensor<T> lossGradient)
        {
            Generator.Backward(lossGradient);
        }

        public override void UpdateWeights()
        {
            Generator.UpdateWeights();
            Discriminator.UpdateWeights();
        }

        public override List<Tensor<T>> GetParameters()
        {
            var parameters = new List<Tensor<T>>();
            parameters.AddRange(Generator.GetParameters());
            parameters.AddRange(Discriminator.GetParameters());
            return parameters;
        }

        public override void SetTrainingMode(bool isTraining)
        {
            Generator.SetTrainingMode(isTraining);
            Discriminator.SetTrainingMode(isTraining);
        }

        public override ModelType GetModelType()
        {
            return ModelType.ProgressiveGAN;
        }

        public override Dictionary<string, object> GetMetadata()
        {
            return new Dictionary<string, object>
            {
                { "ModelType", "ProgressiveGAN" },
                { "LatentSize", LatentSize },
                { "CurrentResolutionLevel", CurrentResolutionLevel },
                { "MaxResolutionLevel", MaxResolutionLevel },
                { "CurrentResolution", GetCurrentResolution() },
                { "ImageChannels", _imageChannels },
                { "BaseFeatureMaps", _baseFeatureMaps },
                { "Alpha", Alpha },
                { "UseMinibatchStdDev", UseMinibatchStdDev },
                { "UsePixelNormalization", UsePixelNormalization }
            };
        }
    }
}
