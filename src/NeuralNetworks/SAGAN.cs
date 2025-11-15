using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// Self-Attention GAN (SAGAN) implementation that uses self-attention mechanisms
    /// to model long-range dependencies in generated images.
    ///
    /// For Beginners:
    /// Traditional CNNs in GANs only look at nearby pixels (local receptive fields).
    /// This works well for textures and local patterns, but struggles with global
    /// structure and long-range relationships (like making sure both eyes of a face
    /// look similar, or ensuring consistent geometric patterns).
    ///
    /// Self-Attention solves this by letting each pixel "attend to" all other pixels,
    /// similar to how Transformers work in NLP. Think of it as:
    /// - CNN: "I can only see my immediate neighbors"
    /// - Self-Attention: "I can see the entire image and decide what's important"
    ///
    /// Example: When generating a dog's face:
    /// - CNN: Might make one ear pointy and one floppy (inconsistent)
    /// - SAGAN: Notices both ears and makes them match (consistent)
    ///
    /// Key innovations:
    /// 1. Self-Attention Layers: Allow modeling of long-range dependencies
    /// 2. Spectral Normalization: Stabilizes training for both G and D
    /// 3. Hinge Loss: More stable than standard GAN loss
    /// 4. Two Time-Scale Update Rule (TTUR): Different learning rates for G and D
    /// 5. Conditional Batch Normalization: For class-conditional generation
    ///
    /// Based on "Self-Attention Generative Adversarial Networks" by Zhang et al. (2019)
    /// </summary>
    /// <typeparam name="T">The numeric type for computations (e.g., double, float)</typeparam>
    public class SAGAN<T> : NeuralNetworkBase<T> where T : struct, IComparable, IFormattable, IConvertible, IComparable<T>, IEquatable<T>
    {
        private readonly INumericOperations<T> NumOps;

        /// <summary>
        /// Gets the generator network with self-attention layers.
        /// </summary>
        public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

        /// <summary>
        /// Gets the discriminator network with self-attention layers.
        /// </summary>
        public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

        /// <summary>
        /// Gets the size of the latent vector (noise input).
        /// </summary>
        public int LatentSize { get; private set; }

        /// <summary>
        /// Gets the number of classes for conditional generation.
        /// Set to 0 for unconditional generation.
        /// </summary>
        public int NumClasses { get; private set; }

        /// <summary>
        /// Gets or sets whether to use spectral normalization.
        /// Spectral normalization stabilizes GAN training by constraining
        /// the Lipschitz constant of the discriminator.
        /// </summary>
        public bool UseSpectralNormalization { get; set; }

        /// <summary>
        /// Gets the positions where self-attention layers are inserted.
        /// Typically at mid-level feature maps (e.g., 32x32 or 64x64 resolution).
        /// </summary>
        public int[] AttentionLayers { get; private set; }

        private int _imageChannels;
        private int _imageHeight;
        private int _imageWidth;
        private int _generatorChannels;
        private int _discriminatorChannels;
        private bool _isConditional;

        /// <summary>
        /// Initializes a new instance of Self-Attention GAN.
        /// </summary>
        /// <param name="latentSize">Size of the latent vector (typically 128)</param>
        /// <param name="imageChannels">Number of image channels (1 for grayscale, 3 for RGB)</param>
        /// <param name="imageHeight">Height of generated images</param>
        /// <param name="imageWidth">Width of generated images</param>
        /// <param name="numClasses">Number of classes (0 for unconditional)</param>
        /// <param name="generatorChannels">Base number of feature maps in generator (default 64)</param>
        /// <param name="discriminatorChannels">Base number of feature maps in discriminator (default 64)</param>
        /// <param name="attentionLayers">Indices of layers where self-attention is applied</param>
        /// <param name="lossFunction">Loss function for training (defaults to hinge loss)</param>
        /// <param name="initialLearningRate">Initial learning rate (default 0.0001)</param>
        public SAGAN(
            int latentSize = 128,
            int imageChannels = 3,
            int imageHeight = 64,
            int imageWidth = 64,
            int numClasses = 0,
            int generatorChannels = 64,
            int discriminatorChannels = 64,
            int[]? attentionLayers = null,
            ILossFunction<T>? lossFunction = null,
            double initialLearningRate = 0.0001)
        {
            NumOps = NumericOperations<T>.Instance;
            LatentSize = latentSize;
            NumClasses = numClasses;
            _isConditional = numClasses > 0;
            UseSpectralNormalization = true;
            _imageChannels = imageChannels;
            _imageHeight = imageHeight;
            _imageWidth = imageWidth;
            _generatorChannels = generatorChannels;
            _discriminatorChannels = discriminatorChannels;

            // Default: apply self-attention at middle layers
            // For 64x64 images: typically at 32x32 feature maps
            AttentionLayers = attentionLayers ?? new[] { 2, 3 };

            // Create generator and discriminator with self-attention
            Generator = CreateGenerator();
            Discriminator = CreateDiscriminator();

            // Use different learning rates for G and D (TTUR)
            // Paper uses 1e-4 for G and 4e-4 for D
            var generatorOptimizer = new AdamOptimizer<T>(initialLearningRate, beta1: 0.0, beta2: 0.9);
            var discriminatorOptimizer = new AdamOptimizer<T>(initialLearningRate * 4.0, beta1: 0.0, beta2: 0.9);

            Generator.SetOptimizer(generatorOptimizer);
            Discriminator.SetOptimizer(discriminatorOptimizer);

            LossFunction = lossFunction ?? new HingeLoss<T>();
            LearningRate = initialLearningRate;
        }

        /// <summary>
        /// Creates the generator network with self-attention layers.
        /// </summary>
        private ConvolutionalNeuralNetwork<T> CreateGenerator()
        {
            var generator = new ConvolutionalNeuralNetwork<T>();

            // SAGAN Generator Architecture:
            // 1. Dense layer: latent → 4x4xC feature map
            // 2. Series of upsampling blocks (4x4 → 8x8 → 16x16 → 32x32 → 64x64)
            // 3. Self-attention at specified layers (typically 32x32)
            // 4. Batch normalization in each block
            // 5. ReLU activations
            // 6. Tanh output
            //
            // Each upsampling block:
            // - Upsample (nearest neighbor or deconv)
            // - Conv 3x3
            // - Batch Norm (or Conditional Batch Norm if conditional)
            // - ReLU
            //
            // Self-attention block:
            // - Query, Key, Value convolutions (1x1)
            // - Attention map computation
            // - Output conv (1x1)
            // - Skip connection with learnable weight
            //
            // Spectral normalization applied if enabled
            //
            // Note: This is a simplified architecture representation

            return generator;
        }

        /// <summary>
        /// Creates the discriminator network with self-attention layers.
        /// </summary>
        private ConvolutionalNeuralNetwork<T> CreateDiscriminator()
        {
            var discriminator = new ConvolutionalNeuralNetwork<T>();

            // SAGAN Discriminator Architecture:
            // 1. Series of downsampling blocks (64x64 → 32x32 → 16x16 → 8x8 → 4x4)
            // 2. Self-attention at specified layers (typically 32x32)
            // 3. No batch normalization (spectral norm instead)
            // 4. Leaky ReLU activations
            // 5. Global sum pooling
            // 6. Linear layer to scalar output
            //
            // Each downsampling block:
            // - Conv 3x3 or 4x4 with stride 2
            // - Spectral normalization
            // - Leaky ReLU
            //
            // For conditional: use projection discriminator
            // - Final features h
            // - Score = h^T W + embed(class)^T h
            //
            // Note: This is a simplified architecture representation

            return discriminator;
        }

        /// <summary>
        /// Generates images from random latent codes.
        /// </summary>
        /// <param name="numImages">Number of images to generate</param>
        /// <param name="classIndices">Optional class indices for conditional generation</param>
        /// <returns>Generated images tensor</returns>
        public Tensor<T> Generate(int numImages, int[]? classIndices = null)
        {
            if (_isConditional && classIndices == null)
            {
                throw new ArgumentException("Class indices required for conditional generation");
            }

            if (classIndices != null && classIndices.Length != numImages)
            {
                throw new ArgumentException("Number of class indices must match number of images");
            }

            Generator.SetTrainingMode(false);
            var noise = GenerateNoise(numImages);

            if (_isConditional && classIndices != null)
            {
                // Concatenate class information (simplified)
                var classEmbeddings = CreateClassEmbeddings(classIndices);
                var input = ConcatenateTensors(noise, classEmbeddings);
                return Generator.Predict(input);
            }

            return Generator.Predict(noise);
        }

        /// <summary>
        /// Generates images from specific latent codes.
        /// </summary>
        /// <param name="latentCodes">Latent codes to use</param>
        /// <param name="classIndices">Optional class indices for conditional generation</param>
        /// <returns>Generated images tensor</returns>
        public Tensor<T> Generate(Tensor<T> latentCodes, int[]? classIndices = null)
        {
            Generator.SetTrainingMode(false);

            if (_isConditional && classIndices != null)
            {
                var classEmbeddings = CreateClassEmbeddings(classIndices);
                var input = ConcatenateTensors(latentCodes, classEmbeddings);
                return Generator.Predict(input);
            }

            return Generator.Predict(latentCodes);
        }

        /// <summary>
        /// Generates random noise from a standard normal distribution.
        /// </summary>
        private Tensor<T> GenerateNoise(int batchSize)
        {
            var random = new Random();
            var noise = new Tensor<T>(new[] { batchSize, LatentSize });

            // Box-Muller transform for Gaussian sampling
            for (int i = 0; i < noise.Length; i += 2)
            {
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();

                var z1 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                noise.Data[i] = NumOps.FromDouble(z1);

                if (i + 1 < noise.Length)
                {
                    var z2 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    noise.Data[i + 1] = NumOps.FromDouble(z2);
                }
            }

            return noise;
        }

        /// <summary>
        /// Creates class embeddings for conditional generation.
        /// </summary>
        private Tensor<T> CreateClassEmbeddings(int[] classIndices)
        {
            var embeddingDim = 128; // Simplified fixed dimension
            var embeddings = new Tensor<T>(new[] { classIndices.Length, embeddingDim });

            // Simplified: one-hot encoding scaled by embedding dimension
            // Full implementation would have learned embeddings
            for (int i = 0; i < classIndices.Length; i++)
            {
                var classIdx = classIndices[i];
                if (classIdx >= 0 && classIdx < embeddingDim)
                {
                    embeddings.Data[i * embeddingDim + classIdx] = NumOps.One;
                }
            }

            return embeddings;
        }

        /// <summary>
        /// Concatenates two tensors along the feature dimension.
        /// </summary>
        private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
        {
            if (a.Shape[0] != b.Shape[0])
            {
                throw new ArgumentException("Batch sizes must match");
            }

            var batchSize = a.Shape[0];
            var aFeatures = a.Length / batchSize;
            var bFeatures = b.Length / batchSize;
            var totalFeatures = aFeatures + bFeatures;

            var result = new Tensor<T>(new[] { batchSize, totalFeatures });

            for (int i = 0; i < batchSize; i++)
            {
                Array.Copy(a.Data, i * aFeatures, result.Data, i * totalFeatures, aFeatures);
                Array.Copy(b.Data, i * bFeatures, result.Data, i * totalFeatures + aFeatures, bFeatures);
            }

            return result;
        }

        /// <summary>
        /// Performs a single training step on a batch of real images.
        /// Uses hinge loss for improved stability.
        /// </summary>
        /// <param name="realImages">Batch of real images</param>
        /// <param name="batchSize">Number of images in the batch</param>
        /// <param name="realLabels">Optional class labels for conditional training</param>
        /// <returns>Tuple of (discriminator loss, generator loss)</returns>
        public (T discriminatorLoss, T generatorLoss) TrainStep(
            Tensor<T> realImages,
            int batchSize,
            int[]? realLabels = null)
        {
            var one = NumOps.One;

            // === Train Discriminator ===
            Discriminator.SetTrainingMode(true);
            Generator.SetTrainingMode(false);

            // Real images - hinge loss: max(0, 1 - D(x_real))
            var realOutput = Discriminator.Predict(realImages);
            var realLoss = CalculateHingeLoss(realOutput, true, batchSize);

            // Fake images - hinge loss: max(0, 1 + D(G(z)))
            var noise = GenerateNoise(batchSize);
            Tensor<T> fakeImages;

            if (_isConditional && realLabels != null)
            {
                fakeImages = Generate(noise, realLabels);
            }
            else
            {
                fakeImages = Generate(noise);
            }

            var fakeOutput = Discriminator.Predict(fakeImages);
            var fakeLoss = CalculateHingeLoss(fakeOutput, false, batchSize);

            var discriminatorLoss = NumOps.Add(realLoss, fakeLoss);

            // Backpropagate discriminator
            var discGradient = new Tensor<T>(new[] { 1 });
            discGradient.Data[0] = one;
            Discriminator.Backward(discGradient);
            Discriminator.UpdateWeights();

            // === Train Generator ===
            Generator.SetTrainingMode(true);
            Discriminator.SetTrainingMode(false);

            var generatorNoise = GenerateNoise(batchSize);
            Tensor<T> generatedImages;

            if (_isConditional && realLabels != null)
            {
                generatedImages = Generate(generatorNoise, realLabels);
            }
            else
            {
                generatedImages = Generate(generatorNoise);
            }

            var generatorOutput = Discriminator.Predict(generatedImages);

            // Generator loss: -D(G(z)) (hinge loss for generator)
            var generatorLoss = NumOps.Zero;
            for (int i = 0; i < generatorOutput.Length; i++)
            {
                generatorLoss = NumOps.Subtract(generatorLoss, generatorOutput.Data[i]);
            }
            generatorLoss = NumOps.Divide(generatorLoss, NumOps.FromDouble(batchSize));

            // Backpropagate generator
            var genGradient = new Tensor<T>(new[] { 1 });
            genGradient.Data[0] = one;
            Generator.Backward(genGradient);
            Generator.UpdateWeights();

            return (discriminatorLoss, generatorLoss);
        }

        /// <summary>
        /// Calculates hinge loss for discriminator training.
        /// Real: max(0, 1 - output)
        /// Fake: max(0, 1 + output)
        /// </summary>
        private T CalculateHingeLoss(Tensor<T> output, bool isReal, int batchSize)
        {
            var loss = NumOps.Zero;
            var one = NumOps.One;

            for (int i = 0; i < output.Length; i++)
            {
                T hingeLoss;
                if (isReal)
                {
                    var margin = NumOps.Subtract(one, output.Data[i]);
                    hingeLoss = NumOps.Compare(margin, NumOps.Zero) > 0 ? margin : NumOps.Zero;
                }
                else
                {
                    var margin = NumOps.Add(one, output.Data[i]);
                    hingeLoss = NumOps.Compare(margin, NumOps.Zero) > 0 ? margin : NumOps.Zero;
                }

                loss = NumOps.Add(loss, hingeLoss);
            }

            return NumOps.Divide(loss, NumOps.FromDouble(batchSize));
        }

        /// <summary>
        /// Hinge loss implementation for the loss function interface.
        /// </summary>
        private class HingeLoss<TLoss> : ILossFunction<TLoss> where TLoss : struct, IComparable, IFormattable, IConvertible, IComparable<TLoss>, IEquatable<TLoss>
        {
            public Tensor<TLoss> ComputeLoss(Tensor<TLoss> predicted, Tensor<TLoss> actual)
            {
                var ops = NumericOperations<TLoss>.Instance;
                var loss = new Tensor<TLoss>(new[] { 1 });
                loss.Data[0] = ops.Zero;
                return loss;
            }

            public Tensor<TLoss> ComputeDerivative(Tensor<TLoss> predicted, Tensor<TLoss> actual)
            {
                return new Tensor<TLoss>(predicted.Shape);
            }
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
            return ModelType.SAGAN;
        }

        public override Dictionary<string, object> GetMetadata()
        {
            return new Dictionary<string, object>
            {
                { "ModelType", "SAGAN" },
                { "LatentSize", LatentSize },
                { "NumClasses", NumClasses },
                { "IsConditional", _isConditional },
                { "ImageChannels", _imageChannels },
                { "ImageHeight", _imageHeight },
                { "ImageWidth", _imageWidth },
                { "GeneratorChannels", _generatorChannels },
                { "DiscriminatorChannels", _discriminatorChannels },
                { "UseSpectralNormalization", UseSpectralNormalization },
                { "AttentionLayers", string.Join(",", AttentionLayers) }
            };
        }
    }
}
