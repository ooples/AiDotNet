using AiDotNet.LinearAlgebra;
using System;
using System.Linq;

namespace AiDotNet.MultimodalAI.Encoders
{
    /// <summary>
    /// Encoder for image modality
    /// </summary>
    public class ImageEncoder : ModalityEncoder
    {
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly int _channels;
        private readonly bool _useColorHistogram;
        private readonly bool _useEdgeDetection;

        /// <summary>
        /// Initializes a new instance of ImageEncoder
        /// </summary>
        /// <param name="outputDimension">Output dimension of the encoder</param>
        /// <param name="imageWidth">Expected image width</param>
        /// <param name="imageHeight">Expected image height</param>
        /// <param name="channels">Number of color channels (1 for grayscale, 3 for RGB)</param>
        /// <param name="useColorHistogram">Whether to use color histogram features</param>
        /// <param name="useEdgeDetection">Whether to use edge detection features</param>
        public ImageEncoder(int outputDimension, int imageWidth = 224, int imageHeight = 224, 
                          int channels = 3, bool useColorHistogram = true, bool useEdgeDetection = true)
            : base("image", outputDimension)
        {
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
            _channels = channels;
            _useColorHistogram = useColorHistogram;
            _useEdgeDetection = useEdgeDetection;
        }

        /// <summary>
        /// Encodes image input into a vector representation
        /// </summary>
        /// <param name="input">Image input as array or tensor</param>
        /// <returns>Encoded vector representation</returns>
        public override Vector<double> Encode(object input)
        {
            if (!ValidateInput(input))
                throw new ArgumentException("Invalid input type for image encoder");

            var preprocessed = Preprocess(input);
            var imageData = preprocessed as double[,,] ?? ConvertToArray3D(preprocessed);

            // Extract features
            var features = new Vector<double>(GetFeatureSize());
            int featureIndex = 0;

            // Extract color histogram features
            if (_useColorHistogram)
            {
                var histogram = ExtractColorHistogram(imageData);
                for (int i = 0; i < histogram.Dimension; i++)
                {
                    features[featureIndex++] = histogram[i];
                }
            }

            // Extract edge features
            if (_useEdgeDetection)
            {
                var edges = ExtractEdgeFeatures(imageData);
                for (int i = 0; i < edges.Dimension; i++)
                {
                    features[featureIndex++] = edges[i];
                }
            }

            // Extract spatial features (simplified - would use CNN in real implementation)
            var spatial = ExtractSpatialFeatures(imageData);
            for (int i = 0; i < spatial.Dimension && featureIndex < features.Dimension; i++)
            {
                features[featureIndex++] = spatial[i];
            }

            // Normalize
            features = Normalize(features);

            // Project to output dimension if needed
            if (features.Dimension != _outputDimension)
            {
                var projectionMatrix = CreateProjectionMatrix(features.Dimension, _outputDimension);
                features = Project(features, projectionMatrix);
            }

            return features;
        }

        /// <summary>
        /// Preprocesses image input
        /// </summary>
        /// <param name="input">Raw image input</param>
        /// <returns>Preprocessed image as 3D array</returns>
        public override object Preprocess(object input)
        {
            double[,,] imageArray = input switch
            {
                double[,,] arr3d => arr3d,
                float[,,] arr3d => ConvertToDouble(arr3d),
                byte[,,] arr3d => ConvertToDouble(arr3d),
                Tensor<double> tensor when tensor.Rank == 3 => ConvertTensorToArray3D(tensor),
                _ => throw new ArgumentException("Unsupported image format")
            };

            // Resize if necessary (simplified - would use proper interpolation)
            if (imageArray.GetLength(0) != _imageHeight || imageArray.GetLength(1) != _imageWidth)
            {
                imageArray = ResizeImage(imageArray, _imageHeight, _imageWidth);
            }

            // Normalize pixel values to [0, 1]
            imageArray = NormalizePixelValues(imageArray);

            return imageArray;
        }

        /// <summary>
        /// Validates the input for image encoding
        /// </summary>
        /// <param name="input">Input to validate</param>
        /// <returns>True if valid</returns>
        protected override bool ValidateInput(object input)
        {
            return input is double[,,] || input is float[,,] || input is byte[,,] || 
                   (input is Tensor<double> tensor && tensor.Rank == 3);
        }

        /// <summary>
        /// Extracts color histogram features
        /// </summary>
        private Vector<double> ExtractColorHistogram(double[,,] image)
        {
            int bins = 16; // Number of bins per channel
            var histogram = new Vector<double>(bins * _channels);

            for (int c = 0; c < _channels; c++)
            {
                for (int y = 0; y < image.GetLength(0); y++)
                {
                    for (int x = 0; x < image.GetLength(1); x++)
                    {
                        int bin = (int)(image[y, x, c] * (bins - 1));
                        histogram[c * bins + bin] += 1.0;
                    }
                }
            }

            // Normalize histogram
            double total = image.GetLength(0) * image.GetLength(1);
            for (int i = 0; i < histogram.Dimension; i++)
            {
                histogram[i] /= total;
            }

            return histogram;
        }

        /// <summary>
        /// Extracts edge features using simplified edge detection
        /// </summary>
        private Vector<double> ExtractEdgeFeatures(double[,,] image)
        {
            int gridSize = 8; // Divide image into 8x8 grid
            var edges = new Vector<double>(gridSize * gridSize);

            int cellHeight = image.GetLength(0) / gridSize;
            int cellWidth = image.GetLength(1) / gridSize;

            for (int gy = 0; gy < gridSize; gy++)
            {
                for (int gx = 0; gx < gridSize; gx++)
                {
                    double edgeStrength = 0;
                    int count = 0;

                    // Calculate edge strength in this grid cell
                    for (int y = gy * cellHeight; y < Math.Min((gy + 1) * cellHeight, image.GetLength(0) - 1); y++)
                    {
                        for (int x = gx * cellWidth; x < Math.Min((gx + 1) * cellWidth, image.GetLength(1) - 1); x++)
                        {
                            // Simple edge detection using gradients
                            double dx = 0, dy = 0;
                            for (int c = 0; c < _channels; c++)
                            {
                                dx += Math.Abs(image[y, x + 1, c] - image[y, x, c]);
                                dy += Math.Abs(image[y + 1, x, c] - image[y, x, c]);
                            }
                            edgeStrength += Math.Sqrt(dx * dx + dy * dy);
                            count++;
                        }
                    }

                    edges[gy * gridSize + gx] = count > 0 ? edgeStrength / count : 0;
                }
            }

            return edges;
        }

        /// <summary>
        /// Extracts spatial features (simplified version)
        /// </summary>
        private Vector<double> ExtractSpatialFeatures(double[,,] image)
        {
            int poolSize = 16; // Pooling size
            int pooledHeight = image.GetLength(0) / poolSize;
            int pooledWidth = image.GetLength(1) / poolSize;
            
            var features = new Vector<double>(pooledHeight * pooledWidth * _channels);
            int index = 0;

            for (int c = 0; c < _channels; c++)
            {
                for (int py = 0; py < pooledHeight; py++)
                {
                    for (int px = 0; px < pooledWidth; px++)
                    {
                        double sum = 0;
                        int count = 0;

                        // Average pooling
                        for (int y = py * poolSize; y < Math.Min((py + 1) * poolSize, image.GetLength(0)); y++)
                        {
                            for (int x = px * poolSize; x < Math.Min((px + 1) * poolSize, image.GetLength(1)); x++)
                            {
                                sum += image[y, x, c];
                                count++;
                            }
                        }

                        features[index++] = count > 0 ? sum / count : 0;
                    }
                }
            }

            return features;
        }

        /// <summary>
        /// Gets the total feature size
        /// </summary>
        private int GetFeatureSize()
        {
            int size = 0;
            
            if (_useColorHistogram)
                size += 16 * _channels; // 16 bins per channel
            
            if (_useEdgeDetection)
                size += 64; // 8x8 grid
            
            // Spatial features
            size += ((_imageHeight / 16) * (_imageWidth / 16) * _channels);
            
            return size;
        }

        /// <summary>
        /// Converts various array types to double[,,]
        /// </summary>
        private double[,,] ConvertToDouble<T>(T[,,] array) where T : struct
        {
            var result = new double[array.GetLength(0), array.GetLength(1), array.GetLength(2)];
            for (int i = 0; i < array.GetLength(0); i++)
                for (int j = 0; j < array.GetLength(1); j++)
                    for (int k = 0; k < array.GetLength(2); k++)
                        result[i, j, k] = Convert.ToDouble(array[i, j, k]);
            return result;
        }

        /// <summary>
        /// Converts object to 3D array
        /// </summary>
        private double[,,] ConvertToArray3D(object input)
        {
            if (input is double[,,] arr)
                return arr;
            
            throw new ArgumentException("Cannot convert input to 3D array");
        }

        /// <summary>
        /// Converts tensor to 3D array
        /// </summary>
        private double[,,] ConvertTensorToArray3D(Tensor<double> tensor)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor<double> must be rank 3");

            var shape = tensor.Shape;
            var result = new double[shape[0], shape[1], shape[2]];
            
            for (int i = 0; i < shape[0]; i++)
                for (int j = 0; j < shape[1]; j++)
                    for (int k = 0; k < shape[2]; k++)
                        result[i, j, k] = tensor[i, j, k];
            
            return result;
        }

        /// <summary>
        /// Resizes image (simplified nearest neighbor)
        /// </summary>
        private double[,,] ResizeImage(double[,,] image, int newHeight, int newWidth)
        {
            var resized = new double[newHeight, newWidth, image.GetLength(2)];
            double yRatio = (double)image.GetLength(0) / newHeight;
            double xRatio = (double)image.GetLength(1) / newWidth;

            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    int srcY = (int)(y * yRatio);
                    int srcX = (int)(x * xRatio);
                    
                    for (int c = 0; c < image.GetLength(2); c++)
                    {
                        resized[y, x, c] = image[srcY, srcX, c];
                    }
                }
            }

            return resized;
        }

        /// <summary>
        /// Normalizes pixel values to [0, 1]
        /// </summary>
        private double[,,] NormalizePixelValues(double[,,] image)
        {
            double min = double.MaxValue;
            double max = double.MinValue;

            // Find min and max
            for (int y = 0; y < image.GetLength(0); y++)
                for (int x = 0; x < image.GetLength(1); x++)
                    for (int c = 0; c < image.GetLength(2); c++)
                    {
                        min = Math.Min(min, image[y, x, c]);
                        max = Math.Max(max, image[y, x, c]);
                    }

            // Normalize
            double range = max - min;
            if (range > 0)
            {
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        for (int c = 0; c < image.GetLength(2); c++)
                            image[y, x, c] = (image[y, x, c] - min) / range;
            }

            return image;
        }

        /// <summary>
        /// Creates a random projection matrix
        /// </summary>
        private Matrix<double> CreateProjectionMatrix(int inputDim, int outputDim)
        {
            var random = new Random(42);
            var matrix = new Matrix<double>(outputDim, inputDim);
            
            double scale = Math.Sqrt(2.0 / (inputDim + outputDim));
            
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    matrix[i, j] = (random.NextDouble() * 2 - 1) * scale;
                }
            }
            
            return matrix;
        }
    }
}