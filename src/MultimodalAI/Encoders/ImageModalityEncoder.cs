using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.MultimodalAI.Encoders
{
    /// <summary>
    /// Image-specific modality encoder for processing image data
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class ImageModalityEncoder<T> : ModalityEncoderBase<T>
    {
        private readonly int _patchSize;
        private readonly bool _useColorHistogram;
        private readonly bool _useTextureFeatures;
        
        /// <summary>
        /// Initializes a new instance of ImageModalityEncoder
        /// </summary>
        /// <param name="outputDimension">Output dimension of the encoder (default: 512)</param>
        /// <param name="patchSize">Size of patches for feature extraction (default: 16)</param>
        /// <param name="useColorHistogram">Whether to extract color histogram features (default: true)</param>
        /// <param name="useTextureFeatures">Whether to extract texture features (default: true)</param>
        /// <param name="encoder">Optional custom neural network encoder. If null, a default encoder will be created when needed.</param>
        public ImageModalityEncoder(int outputDimension = 512, int patchSize = 16,
            bool useColorHistogram = true, bool useTextureFeatures = true,
            INeuralNetworkModel<T>? encoder = null) 
            : base("Image", outputDimension, encoder)
        {
            _patchSize = patchSize;
            _useColorHistogram = useColorHistogram;
            _useTextureFeatures = useTextureFeatures;
        }

        /// <summary>
        /// Encodes image data into a vector representation
        /// </summary>
        /// <param name="input">Image data as 2D/3D array or Tensor</param>
        /// <returns>Encoded vector representation</returns>
        public override Vector<T> Encode(object input)
        {
            if (!ValidateInput(input))
            {
                throw new ArgumentException($"Invalid input type for image encoding. Expected array or Tensor, got {input?.GetType()?.Name ?? "null"}");
            }

            // Preprocess the input
            var preprocessed = Preprocess(input);
            var imageData = preprocessed as ImageData ?? throw new InvalidOperationException("Preprocessing failed");

            // Extract features
            var features = ExtractImageFeatures(imageData);
            
            // Project to output dimension if needed
            if (features.Length != OutputDimension)
            {
                features = ProjectToOutputDimension(features);
            }

            // Normalize the output
            return Normalize(features);
        }

        /// <summary>
        /// Preprocesses raw image input
        /// </summary>
        public override object Preprocess(object input)
        {
            ImageData imageData;

            switch (input)
            {
                case Array array2D when array2D.Rank == 2:
                    imageData = new ImageData
                    {
                        Width = array2D.GetLength(1),
                        Height = array2D.GetLength(0),
                        Channels = 1,
                        Data = Flatten2DArray(array2D)
                    };
                    break;
                    
                case Array array3D when array3D.Rank == 3:
                    imageData = new ImageData
                    {
                        Width = array3D.GetLength(2),
                        Height = array3D.GetLength(1),
                        Channels = array3D.GetLength(0),
                        Data = Flatten3DArray(array3D)
                    };
                    break;
                    
                case Tensor<T> tensor:
                    // Assume tensor shape is [C, H, W] or [H, W]
                    if (tensor.Rank == 2)
                    {
                        imageData = new ImageData
                        {
                            Width = tensor.Shape[1],
                            Height = tensor.Shape[0],
                            Channels = 1,
                            Data = tensor.ToArray()
                        };
                    }
                    else if (tensor.Rank == 3)
                    {
                        imageData = new ImageData
                        {
                            Width = tensor.Shape[2],
                            Height = tensor.Shape[1],
                            Channels = tensor.Shape[0],
                            Data = tensor.ToArray()
                        };
                    }
                    else
                    {
                        throw new ArgumentException($"Tensor must have rank 2 or 3, got rank {tensor.Rank}");
                    }
                    break;
                    
                default:
                    throw new ArgumentException($"Unsupported input type: {input?.GetType()?.Name ?? "null"}");
            }

            // Normalize pixel values to [0, 1]
            imageData.Data = NormalizePixelValues(imageData.Data);

            return imageData;
        }

        /// <summary>
        /// Validates the input for image encoding
        /// </summary>
        protected override bool ValidateInput(object input)
        {
            return input is Array array && (array.Rank == 2 || array.Rank == 3) ||
                   input is Tensor<T>;
        }

        /// <summary>
        /// Extracts image features from preprocessed data
        /// </summary>
        private Vector<T> ExtractImageFeatures(ImageData imageData)
        {
            var features = new System.Collections.Generic.List<T>();

            // Global statistics
            features.AddRange(ExtractGlobalStatistics(imageData));

            if (_useColorHistogram && imageData.Channels >= 3)
            {
                // Color histogram features
                features.AddRange(ExtractColorHistogram(imageData));
            }

            if (_useTextureFeatures)
            {
                // Texture features
                features.AddRange(ExtractTextureFeatures(imageData));
            }

            // Spatial features
            features.AddRange(ExtractSpatialFeatures(imageData));

            return new Vector<T>(features.ToArray());
        }

        /// <summary>
        /// Extracts global image statistics
        /// </summary>
        private T[] ExtractGlobalStatistics(ImageData imageData)
        {
            var features = new System.Collections.Generic.List<T>();

            for (int c = 0; c < imageData.Channels; c++)
            {
                var channelData = GetChannelData(imageData, c);
                
                // Mean
                T mean = ComputeMean(channelData);
                features.Add(mean);

                // Standard deviation
                T stdDev = ComputeStandardDeviation(channelData, mean);
                features.Add(stdDev);

                // Min and Max
                features.Add(ComputeMin(channelData));
                features.Add(ComputeMax(channelData));
            }

            return features.ToArray();
        }

        /// <summary>
        /// Extracts color histogram features
        /// </summary>
        private T[] ExtractColorHistogram(ImageData imageData)
        {
            int numBins = 16; // Reduced for efficiency
            var histogram = new T[numBins * imageData.Channels];

            for (int c = 0; c < imageData.Channels; c++)
            {
                var channelData = GetChannelData(imageData, c);
                var channelHist = ComputeHistogram(channelData, numBins);
                
                for (int b = 0; b < numBins; b++)
                {
                    histogram[c * numBins + b] = channelHist[b];
                }
            }

            // Normalize histogram
            T sum = ComputeSum(histogram);
            if (_numericOps.GreaterThan(sum, _numericOps.Zero))
            {
                for (int i = 0; i < histogram.Length; i++)
                {
                    histogram[i] = _numericOps.Divide(histogram[i], sum);
                }
            }

            return histogram;
        }

        /// <summary>
        /// Extracts texture features using simple edge detection
        /// </summary>
        private T[] ExtractTextureFeatures(ImageData imageData)
        {
            var features = new System.Collections.Generic.List<T>();

            // Convert to grayscale if needed
            var grayData = imageData.Channels == 1 ? imageData.Data : ConvertToGrayscale(imageData);

            // Compute gradients
            var (gradX, gradY) = ComputeGradients(grayData, imageData.Width, imageData.Height);

            // Gradient magnitude statistics
            var magnitudes = new T[gradX.Length];
            for (int i = 0; i < gradX.Length; i++)
            {
                var sqrX = _numericOps.Multiply(gradX[i], gradX[i]);
                var sqrY = _numericOps.Multiply(gradY[i], gradY[i]);
                magnitudes[i] = _numericOps.Sqrt(_numericOps.Add(sqrX, sqrY));
            }

            features.Add(ComputeMean(magnitudes));
            features.Add(ComputeRMS(magnitudes));
            features.Add(ComputeMax(magnitudes));

            // Edge density
            T edgeThreshold = _numericOps.FromDouble(0.1);
            int edgeCount = magnitudes.Count(m => _numericOps.GreaterThan(m, edgeThreshold));
            T edgeDensity = _numericOps.Divide(_numericOps.FromDouble(edgeCount), _numericOps.FromDouble(magnitudes.Length));
            features.Add(edgeDensity);

            return features.ToArray();
        }

        /// <summary>
        /// Extracts spatial features by dividing image into patches
        /// </summary>
        private T[] ExtractSpatialFeatures(ImageData imageData)
        {
            var features = new System.Collections.Generic.List<T>();

            int patchesX = Math.Max(1, imageData.Width / _patchSize);
            int patchesY = Math.Max(1, imageData.Height / _patchSize);

            for (int py = 0; py < patchesY; py++)
            {
                for (int px = 0; px < patchesX; px++)
                {
                    var patchFeatures = ExtractPatchFeatures(imageData, px, py, patchesX, patchesY);
                    features.AddRange(patchFeatures);
                }
            }

            return features.ToArray();
        }

        /// <summary>
        /// Extracts features from a single patch
        /// </summary>
        private T[] ExtractPatchFeatures(ImageData imageData, int patchX, int patchY, int patchesX, int patchesY)
        {
            var features = new System.Collections.Generic.List<T>();

            int startX = patchX * imageData.Width / patchesX;
            int endX = (patchX + 1) * imageData.Width / patchesX;
            int startY = patchY * imageData.Height / patchesY;
            int endY = (patchY + 1) * imageData.Height / patchesY;

            for (int c = 0; c < imageData.Channels; c++)
            {
                T sum = _numericOps.Zero;
                int count = 0;

                for (int y = startY; y < endY; y++)
                {
                    for (int x = startX; x < endX; x++)
                    {
                        int idx = (c * imageData.Height + y) * imageData.Width + x;
                        sum = _numericOps.Add(sum, imageData.Data[idx]);
                        count++;
                    }
                }

                features.Add(_numericOps.Divide(sum, _numericOps.FromDouble(count)));
            }

            return features.ToArray();
        }

        /// <summary>
        /// Projects features to the desired output dimension
        /// </summary>
        private Vector<T> ProjectToOutputDimension(Vector<T> features)
        {
            if (features.Length == OutputDimension)
                return features;

            var result = new T[OutputDimension];

            if (features.Length > OutputDimension)
            {
                // Use PCA-like approach (simplified)
                int step = features.Length / OutputDimension;
                for (int i = 0; i < OutputDimension; i++)
                {
                    int idx = Math.Min(i * step, features.Length - 1);
                    result[i] = features[idx];
                }
            }
            else
            {
                // Pad with zeros
                for (int i = 0; i < features.Length; i++)
                {
                    result[i] = features[i];
                }
            }

            return new Vector<T>(result);
        }

        /// <summary>
        /// Helper methods
        /// </summary>
        private T[] Flatten2DArray(Array array)
        {
            int height = array.GetLength(0);
            int width = array.GetLength(1);
            var result = new T[height * width];
            
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var value = array.GetValue(y, x);
                    result[y * width + x] = ConvertToT(value);
                }
            }
            
            return result;
        }

        private T[] Flatten3DArray(Array array)
        {
            int channels = array.GetLength(0);
            int height = array.GetLength(1);
            int width = array.GetLength(2);
            var result = new T[channels * height * width];
            
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        var value = array.GetValue(c, y, x);
                        result[(c * height + y) * width + x] = ConvertToT(value);
                    }
                }
            }
            
            return result;
        }

        private T[] NormalizePixelValues(T[] pixels)
        {
            T min = ComputeMin(pixels);
            T max = ComputeMax(pixels);
            T range = _numericOps.Subtract(max, min);

            if (_numericOps.GreaterThan(range, _numericOps.Zero))
            {
                return pixels.Select(p => _numericOps.Divide(_numericOps.Subtract(p, min), range)).ToArray();
            }
            return pixels;
        }

        private T[] GetChannelData(ImageData imageData, int channel)
        {
            int pixelsPerChannel = imageData.Width * imageData.Height;
            var channelData = new T[pixelsPerChannel];
            
            int offset = channel * pixelsPerChannel;
            Array.Copy(imageData.Data, offset, channelData, 0, pixelsPerChannel);
            
            return channelData;
        }

        private T[] ConvertToGrayscale(ImageData imageData)
        {
            int numPixels = imageData.Width * imageData.Height;
            var grayscale = new T[numPixels];

            for (int i = 0; i < numPixels; i++)
            {
                T gray = _numericOps.Zero;
                for (int c = 0; c < imageData.Channels; c++)
                {
                    gray = _numericOps.Add(gray, imageData.Data[c * numPixels + i]);
                }
                grayscale[i] = _numericOps.Divide(gray, _numericOps.FromDouble(imageData.Channels));
            }

            return grayscale;
        }

        private T[] ComputeHistogram(T[] data, int numBins)
        {
            var histogram = new T[numBins];
            for (int i = 0; i < numBins; i++)
            {
                histogram[i] = _numericOps.Zero;
            }
            
            foreach (var value in data)
            {
                // Convert value to [0, 1] range then to bin index
                var normalized = value; // Assume already normalized
                var binFloat = _numericOps.Multiply(normalized, _numericOps.FromDouble(numBins));
                int bin = Math.Min(_numericOps.ToInt32(binFloat), numBins - 1);
                histogram[bin] = _numericOps.Add(histogram[bin], _numericOps.One);
            }

            return histogram;
        }

        private (T[], T[]) ComputeGradients(T[] data, int width, int height)
        {
            var gradX = new T[data.Length];
            var gradY = new T[data.Length];
            
            // Initialize with zeros
            for (int i = 0; i < data.Length; i++)
            {
                gradX[i] = _numericOps.Zero;
                gradY[i] = _numericOps.Zero;
            }

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;
                    
                    // Compute X gradient (Sobel-like)
                    if (x > 0 && x < width - 1)
                    {
                        gradX[idx] = _numericOps.Subtract(data[y * width + (x + 1)], data[y * width + (x - 1)]);
                    }

                    // Compute Y gradient
                    if (y > 0 && y < height - 1)
                    {
                        gradY[idx] = _numericOps.Subtract(data[(y + 1) * width + x], data[(y - 1) * width + x]);
                    }
                }
            }

            return (gradX, gradY);
        }

        /// <summary>
        /// Converts a value to type T
        /// </summary>
        private T ConvertToT(object value)
        {
            if (value is T tValue)
                return tValue;
            
            // Use numeric operations to convert
            if (value is IConvertible convertible)
            {
                double doubleValue = Convert.ToDouble(convertible);
                return _numericOps.FromDouble(doubleValue);
            }
            
            throw new ArgumentException($"Cannot convert {value?.GetType()?.Name ?? "null"} to {typeof(T).Name}");
        }

        /// <summary>
        /// Computes the mean of an array
        /// </summary>
        private T ComputeMean(T[] values)
        {
            if (values.Length == 0)
                return _numericOps.Zero;
                
            T sum = _numericOps.Zero;
            foreach (var value in values)
            {
                sum = _numericOps.Add(sum, value);
            }
            return _numericOps.Divide(sum, _numericOps.FromDouble(values.Length));
        }

        /// <summary>
        /// Computes the standard deviation of an array
        /// </summary>
        private T ComputeStandardDeviation(T[] values, T mean)
        {
            if (values.Length == 0)
                return _numericOps.Zero;
                
            T sumSquaredDiff = _numericOps.Zero;
            foreach (var value in values)
            {
                T diff = _numericOps.Subtract(value, mean);
                sumSquaredDiff = _numericOps.Add(sumSquaredDiff, _numericOps.Multiply(diff, diff));
            }
            
            T variance = _numericOps.Divide(sumSquaredDiff, _numericOps.FromDouble(values.Length));
            return _numericOps.Sqrt(variance);
        }

        /// <summary>
        /// Computes the minimum value in an array
        /// </summary>
        private T ComputeMin(T[] values)
        {
            if (values.Length == 0)
                return _numericOps.Zero;
                
            T min = values[0];
            for (int i = 1; i < values.Length; i++)
            {
                if (_numericOps.LessThan(values[i], min))
                    min = values[i];
            }
            return min;
        }

        /// <summary>
        /// Computes the maximum value in an array
        /// </summary>
        private T ComputeMax(T[] values)
        {
            if (values.Length == 0)
                return _numericOps.Zero;
                
            T max = values[0];
            for (int i = 1; i < values.Length; i++)
            {
                if (_numericOps.GreaterThan(values[i], max))
                    max = values[i];
            }
            return max;
        }

        /// <summary>
        /// Computes the sum of an array
        /// </summary>
        private T ComputeSum(T[] values)
        {
            T sum = _numericOps.Zero;
            foreach (var value in values)
            {
                sum = _numericOps.Add(sum, value);
            }
            return sum;
        }

        /// <summary>
        /// Computes the RMS (root mean square) of an array
        /// </summary>
        private T ComputeRMS(T[] values)
        {
            if (values.Length == 0)
                return _numericOps.Zero;
                
            T sumSquared = _numericOps.Zero;
            foreach (var value in values)
            {
                sumSquared = _numericOps.Add(sumSquared, _numericOps.Multiply(value, value));
            }
            
            T meanSquared = _numericOps.Divide(sumSquared, _numericOps.FromDouble(values.Length));
            return _numericOps.Sqrt(meanSquared);
        }

        /// <summary>
        /// Internal class for storing image data
        /// </summary>
        private class ImageData
        {
            public int Width { get; set; }
            public int Height { get; set; }
            public int Channels { get; set; }
            public T[] Data { get; set; } = Array.Empty<T>();
        }
    }
}