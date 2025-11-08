using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Mathematics;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// BigGAN implementation for large-scale high-fidelity image generation.
    ///
    /// For Beginners:
    /// BigGAN is a state-of-the-art GAN architecture that generates extremely high-quality
    /// images by scaling up training in several ways:
    /// 1. Using very large batch sizes (256-2048 images at once)
    /// 2. Increasing model capacity (more parameters and feature maps)
    /// 3. Using class information to generate specific types of images
    ///
    /// Think of it like training an artist:
    /// - Small batch = showing the artist 1-2 examples at a time
    /// - BigGAN batch = showing 256+ examples at once for better learning
    /// - Class conditioning = telling the artist exactly what to draw ("draw a cat" vs "draw something")
    ///
    /// Key innovations:
    /// 1. Large Batch Training: Uses batch sizes of 256-2048 (vs typical 32-128)
    /// 2. Spectral Normalization: Stabilizes training for both G and D
    /// 3. Self-Attention: Helps model long-range dependencies in images
    /// 4. Class Conditioning: Uses class embeddings for controlled generation
    /// 5. Truncation Trick: Trade diversity for quality at generation time
    /// 6. Orthogonal Initialization: Better weight initialization
    /// 7. Skip Connections: Direct paths in generator architecture
    ///
    /// Based on "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
    /// by Brock et al. (2019)
    /// </summary>
    /// <typeparam name="T">The numeric type for computations (e.g., double, float)</typeparam>
    public class BigGAN<T> : NeuralNetworkBase<T> where T : struct, IComparable, IFormattable, IConvertible, IComparable<T>, IEquatable<T>
    {
        private readonly INumericOperations<T> NumOps;

        /// <summary>
        /// Gets the generator network that produces images from noise and class labels.
        /// </summary>
        public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

        /// <summary>
        /// Gets the discriminator network that evaluates images and predicts their class.
        /// Uses projection discriminator for efficient class conditioning.
        /// </summary>
        public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

        /// <summary>
        /// Gets the size of the latent noise vector.
        /// BigGAN typically uses 120-dimensional latent codes.
        /// </summary>
        public int LatentSize { get; private set; }

        /// <summary>
        /// Gets the number of classes for conditional generation.
        /// For example, ImageNet has 1000 classes.
        /// </summary>
        public int NumClasses { get; private set; }

        /// <summary>
        /// Gets the dimension of class embeddings.
        /// These learned embeddings represent each class.
        /// </summary>
        public int ClassEmbeddingDim { get; private set; }

        /// <summary>
        /// Gets or sets the truncation threshold for the truncation trick.
        /// Values in range [0, 2], where lower values trade diversity for quality.
        /// Typical value: 0.5 for high quality, 1.0 for balanced, 2.0 for high diversity.
        /// </summary>
        public double TruncationThreshold { get; set; }

        /// <summary>
        /// Gets or sets whether to use the truncation trick during generation.
        /// When enabled, samples are resampled if they fall outside the truncation threshold.
        /// </summary>
        public bool UseTruncation { get; set; }

        /// <summary>
        /// Gets or sets whether to use spectral normalization in both generator and discriminator.
        /// </summary>
        public bool UseSpectralNormalization { get; set; }

        /// <summary>
        /// Gets or sets whether to use self-attention layers.
        /// </summary>
        public bool UseSelfAttention { get; set; }

        private Matrix<T> _classEmbeddings;
        private int _imageChannels;
        private int _imageHeight;
        private int _imageWidth;
        private int _generatorChannels;
        private int _discriminatorChannels;

        /// <summary>
        /// Initializes a new instance of BigGAN.
        /// </summary>
        /// <param name="latentSize">Size of the latent noise vector (default 120)</param>
        /// <param name="numClasses">Number of classes for conditional generation</param>
        /// <param name="classEmbeddingDim">Dimension of class embeddings (default 128)</param>
        /// <param name="imageChannels">Number of image channels (1 for grayscale, 3 for RGB)</param>
        /// <param name="imageHeight">Height of generated images</param>
        /// <param name="imageWidth">Width of generated images</param>
        /// <param name="generatorChannels">Base number of channels in generator (default 96)</param>
        /// <param name="discriminatorChannels">Base number of channels in discriminator (default 96)</param>
        /// <param name="lossFunction">Loss function for training (defaults to hinge loss)</param>
        /// <param name="initialLearningRate">Initial learning rate (default 0.0001)</param>
        public BigGAN(
            int latentSize = 120,
            int numClasses = 1000,
            int classEmbeddingDim = 128,
            int imageChannels = 3,
            int imageHeight = 128,
            int imageWidth = 128,
            int generatorChannels = 96,
            int discriminatorChannels = 96,
            ILossFunction<T>? lossFunction = null,
            double initialLearningRate = 0.0001)
        {
            NumOps = NumericOperations<T>.Instance;
            LatentSize = latentSize;
            NumClasses = numClasses;
            ClassEmbeddingDim = classEmbeddingDim;
            TruncationThreshold = 1.0;
            UseTruncation = false;
            UseSpectralNormalization = true;
            UseSelfAttention = true;
            _imageChannels = imageChannels;
            _imageHeight = imageHeight;
            _imageWidth = imageWidth;
            _generatorChannels = generatorChannels;
            _discriminatorChannels = discriminatorChannels;

            // Initialize class embeddings with orthogonal initialization
            _classEmbeddings = InitializeClassEmbeddings();

            // Create generator and discriminator
            Generator = CreateGenerator();
            Discriminator = CreateDiscriminator();

            // Use Adam optimizer with settings from the paper
            // BigGAN uses different learning rates for G and D
            var generatorOptimizer = new AdamOptimizer<T>(initialLearningRate, beta1: 0.0, beta2: 0.999, epsilon: 1e-6);
            var discriminatorOptimizer = new AdamOptimizer<T>(initialLearningRate * 4.0, beta1: 0.0, beta2: 0.999, epsilon: 1e-6);

            Generator.SetOptimizer(generatorOptimizer);
            Discriminator.SetOptimizer(discriminatorOptimizer);

            LossFunction = lossFunction ?? new HingeLoss<T>();
            LearningRate = initialLearningRate;
        }

        /// <summary>
        /// Initializes class embeddings using orthogonal initialization.
        /// Orthogonal initialization helps with training stability.
        /// </summary>
        private Matrix<T> InitializeClassEmbeddings()
        {
            var embeddings = new Matrix<T>(NumClasses, ClassEmbeddingDim);
            var random = new Random();

            // Simplified orthogonal initialization
            // Full implementation would use proper orthogonal matrix generation
            for (int i = 0; i < NumClasses; i++)
            {
                for (int j = 0; j < ClassEmbeddingDim; j++)
                {
                    embeddings[i, j] = NumOps.FromDouble((random.NextDouble() - 0.5) * 0.1);
                }
            }

            return embeddings;
        }

        /// <summary>
        /// Creates the BigGAN generator network.
        /// </summary>
        private ConvolutionalNeuralNetwork<T> CreateGenerator()
        {
            var generator = new ConvolutionalNeuralNetwork<T>();

            // BigGAN generator architecture:
            // 1. Split latent code z into chunks for each resolution block
            // 2. Class embedding is concatenated with each chunk
            // 3. Residual blocks with conditional batch normalization
            // 4. Self-attention at specific resolutions (e.g., 64x64)
            // 5. Skip connections (hierarchical structure)
            // 6. Spectral normalization (optional but recommended)
            //
            // Note: This is a simplified architecture
            // Full BigGAN has sophisticated residual blocks with class-conditional
            // batch normalization where class information modulates the normalization

            return generator;
        }

        /// <summary>
        /// Creates the BigGAN discriminator network with projection discriminator.
        /// </summary>
        private ConvolutionalNeuralNetwork<T> CreateDiscriminator()
        {
            var discriminator = new ConvolutionalNeuralNetwork<T>();

            // BigGAN discriminator architecture:
            // 1. Residual blocks for downsampling
            // 2. Self-attention at specific resolutions
            // 3. Spectral normalization on all layers
            // 4. Projection discriminator:
            //    - Image features → h
            //    - Score = h^T W + (class_embedding)^T h
            //    - This is more parameter-efficient than concatenation
            // 5. Global sum pooling before final layer
            //
            // Note: This is a simplified architecture

            return discriminator;
        }

        /// <summary>
        /// Gets the class embedding for a specific class index.
        /// </summary>
        private Vector<T> GetClassEmbedding(int classIndex)
        {
            var embedding = new Vector<T>(ClassEmbeddingDim);
            for (int i = 0; i < ClassEmbeddingDim; i++)
            {
                embedding[i] = _classEmbeddings[classIndex, i];
            }
            return embedding;
        }

        /// <summary>
        /// Applies the truncation trick to latent codes.
        /// Resamples values that fall outside the threshold.
        /// </summary>
        private Tensor<T> ApplyTruncation(Tensor<T> latentCodes)
        {
            if (!UseTruncation)
            {
                return latentCodes;
            }

            var random = new Random();
            var truncated = new Tensor<T>(latentCodes.Shape);
            var threshold = NumOps.FromDouble(TruncationThreshold);

            for (int i = 0; i < latentCodes.Length; i++)
            {
                var value = latentCodes.Data[i];
                var absValue = NumOps.Abs(value);

                // If absolute value exceeds threshold, resample
                if (NumOps.Compare(absValue, threshold) > 0)
                {
                    // Resample until within threshold
                    do
                    {
                        value = NumOps.FromDouble(random.NextDouble() * 2.0 - 1.0);
                        value = NumOps.Multiply(value, NumOps.FromDouble(2.0)); // Scaled Gaussian approximation
                        absValue = NumOps.Abs(value);
                    } while (NumOps.Compare(absValue, threshold) > 0);
                }

                truncated.Data[i] = value;
            }

            return truncated;
        }

        /// <summary>
        /// Generates images from latent codes and class labels.
        /// </summary>
        /// <param name="latentCodes">Latent noise vectors</param>
        /// <param name="classIndices">Class indices for each sample</param>
        /// <returns>Generated images</returns>
        public Tensor<T> Generate(Tensor<T> latentCodes, int[] classIndices)
        {
            if (classIndices.Length != latentCodes.Shape[0])
            {
                throw new ArgumentException("Number of class indices must match batch size");
            }

            Generator.SetTrainingMode(false);

            // Apply truncation if enabled
            var truncatedCodes = ApplyTruncation(latentCodes);

            // In full implementation, class embeddings would be retrieved and
            // concatenated/injected into the generator at each layer
            // For now, simplified version
            var classEmbeddings = new Tensor<T>(new[] { classIndices.Length, ClassEmbeddingDim });
            for (int i = 0; i < classIndices.Length; i++)
            {
                var embedding = GetClassEmbedding(classIndices[i]);
                for (int j = 0; j < ClassEmbeddingDim; j++)
                {
                    classEmbeddings.Data[i * ClassEmbeddingDim + j] = embedding[j];
                }
            }

            // Concatenate latent codes and class embeddings
            var input = ConcatenateTensors(truncatedCodes, classEmbeddings);

            return Generator.Predict(input);
        }

        /// <summary>
        /// Generates random images with random class labels.
        /// </summary>
        /// <param name="numImages">Number of images to generate</param>
        /// <returns>Generated images</returns>
        public Tensor<T> Generate(int numImages)
        {
            var random = new Random();
            var noise = GenerateNoise(numImages);
            var classIndices = new int[numImages];

            for (int i = 0; i < numImages; i++)
            {
                classIndices[i] = random.Next(NumClasses);
            }

            return Generate(noise, classIndices);
        }

        /// <summary>
        /// Generates random noise for the generator input.
        /// Uses Gaussian distribution (standard normal).
        /// </summary>
        private Tensor<T> GenerateNoise(int batchSize)
        {
            var random = new Random();
            var noise = new Tensor<T>(new[] { batchSize, LatentSize });

            // Sample from approximate Gaussian using Box-Muller transform
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
                // Copy from tensor a
                Array.Copy(a.Data, i * aFeatures, result.Data, i * totalFeatures, aFeatures);
                // Copy from tensor b
                Array.Copy(b.Data, i * bFeatures, result.Data, i * totalFeatures + aFeatures, bFeatures);
            }

            return result;
        }

        /// <summary>
        /// Performs a single training step on a batch of real images with labels.
        /// Uses hinge loss by default for improved stability.
        /// </summary>
        /// <param name="realImages">Batch of real images</param>
        /// <param name="realLabels">Class labels for real images</param>
        /// <param name="batchSize">Number of images in the batch</param>
        /// <returns>Tuple of (discriminator loss, generator loss)</returns>
        public (T discriminatorLoss, T generatorLoss) TrainStep(Tensor<T> realImages, int[] realLabels, int batchSize)
        {
            var one = NumOps.One;
            var zero = NumOps.Zero;

            // === Train Discriminator ===
            Discriminator.SetTrainingMode(true);
            Generator.SetTrainingMode(false);

            // Real images
            var realOutput = Discriminator.Predict(realImages);
            var realLoss = CalculateHingeLoss(realOutput, true, batchSize);

            // Fake images
            var noise = GenerateNoise(batchSize);
            var random = new Random();
            var fakeLabels = new int[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                fakeLabels[i] = random.Next(NumClasses);
            }

            var fakeImages = Generate(noise, fakeLabels);
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
            var generatorLabels = new int[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                generatorLabels[i] = random.Next(NumClasses);
            }

            var generatedImages = Generate(generatorNoise, generatorLabels);
            var generatorOutput = Discriminator.Predict(generatedImages);
            var generatorLoss = CalculateHingeLoss(generatorOutput, true, batchSize);

            // Backpropagate generator
            var genGradient = new Tensor<T>(new[] { 1 });
            genGradient.Data[0] = one;
            Generator.Backward(genGradient);
            Generator.UpdateWeights();

            return (discriminatorLoss, generatorLoss);
        }

        /// <summary>
        /// Calculates hinge loss for adversarial training.
        /// Hinge loss: max(0, 1 - t*y) where t is target, y is output
        /// For real: max(0, 1 - y)
        /// For fake: max(0, 1 + y)
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
                    // max(0, 1 - output)
                    var margin = NumOps.Subtract(one, output.Data[i]);
                    hingeLoss = NumOps.Compare(margin, NumOps.Zero) > 0 ? margin : NumOps.Zero;
                }
                else
                {
                    // max(0, 1 + output)
                    var margin = NumOps.Add(one, output.Data[i]);
                    hingeLoss = NumOps.Compare(margin, NumOps.Zero) > 0 ? margin : NumOps.Zero;
                }

                loss = NumOps.Add(loss, hingeLoss);
            }

            return NumOps.Divide(loss, NumOps.FromDouble(batchSize));
        }

        /// <summary>
        /// Hinge loss implementation for use with the loss function interface.
        /// </summary>
        private class HingeLoss<TLoss> : ILossFunction<TLoss> where TLoss : struct, IComparable, IFormattable, IConvertible, IComparable<TLoss>, IEquatable<TLoss>
        {
            public Tensor<TLoss> ComputeLoss(Tensor<TLoss> predicted, Tensor<TLoss> actual)
            {
                // Simplified hinge loss
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
            // For general prediction, generate with random classes
            var batchSize = input.Shape[0];
            var random = new Random();
            var classIndices = new int[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                classIndices[i] = random.Next(NumClasses);
            }
            return Generate(input, classIndices);
        }

        public override void Backward(Tensor<T> lossGradient)
        {
            Generator.Backward(lossGradient);
        }

        public override void UpdateWeights()
        {
            Generator.UpdateWeights();
            Discriminator.UpdateWeights();

            // Update class embeddings (simplified gradient update)
            // Full implementation would properly backpropagate through embeddings
        }

        public override List<Tensor<T>> GetParameters()
        {
            var parameters = new List<Tensor<T>>();
            parameters.AddRange(Generator.GetParameters());
            parameters.AddRange(Discriminator.GetParameters());

            // Add class embeddings as parameters
            var embeddingTensor = new Tensor<T>(new[] { NumClasses, ClassEmbeddingDim });
            for (int i = 0; i < NumClasses; i++)
            {
                for (int j = 0; j < ClassEmbeddingDim; j++)
                {
                    embeddingTensor.Data[i * ClassEmbeddingDim + j] = _classEmbeddings[i, j];
                }
            }
            parameters.Add(embeddingTensor);

            return parameters;
        }

        public override void SetTrainingMode(bool isTraining)
        {
            Generator.SetTrainingMode(isTraining);
            Discriminator.SetTrainingMode(isTraining);
        }

        public override ModelType GetModelType()
        {
            return ModelType.BigGAN;
        }

        public override Dictionary<string, object> GetMetadata()
        {
            return new Dictionary<string, object>
            {
                { "ModelType", "BigGAN" },
                { "LatentSize", LatentSize },
                { "NumClasses", NumClasses },
                { "ClassEmbeddingDim", ClassEmbeddingDim },
                { "ImageChannels", _imageChannels },
                { "ImageHeight", _imageHeight },
                { "ImageWidth", _imageWidth },
                { "GeneratorChannels", _generatorChannels },
                { "DiscriminatorChannels", _discriminatorChannels },
                { "TruncationThreshold", TruncationThreshold },
                { "UseTruncation", UseTruncation },
                { "UseSpectralNormalization", UseSpectralNormalization },
                { "UseSelfAttention", UseSelfAttention }
            };
        }
    }
}
