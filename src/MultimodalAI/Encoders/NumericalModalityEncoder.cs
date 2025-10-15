using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;

namespace AiDotNet.MultimodalAI.Encoders
{
    /// <summary>
    /// Numerical data modality encoder for processing numerical/tabular data
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class NumericalModalityEncoder<T> : ModalityEncoderBase<T>
    {
        private readonly bool _useFeatureEngineering;
        private readonly bool _useInteractionFeatures;
        private readonly int _polynomialDegree;
        private readonly T _missingValueIndicator;
        private readonly NormalizationMethod _normalizationMethod;
        
        // Feature statistics for normalization
        private T[] _featureMeans = Array.Empty<T>();
        private T[] _featureStdDevs = Array.Empty<T>();
        private T[] _featureMins = Array.Empty<T>();
        private T[] _featureMaxs = Array.Empty<T>();
        private bool _statisticsComputed;
        
        /// <summary>
        /// Normalization methods for numerical features
        /// </summary>
        public enum NormalizationMethod
        {
            None,
            StandardScore,
            MinMax,
            Robust
        }
        
        /// <summary>
        /// Initializes a new instance of NumericalModalityEncoder
        /// </summary>
        /// <param name="outputDimension">Output dimension of the encoder (default: 128)</param>
        /// <param name="useFeatureEngineering">Whether to create engineered features (default: true)</param>
        /// <param name="useInteractionFeatures">Whether to create interaction features (default: true)</param>
        /// <param name="polynomialDegree">Degree for polynomial features (default: 2)</param>
        /// <param name="normalizationMethod">Method for normalizing features (default: StandardScore)</param>
        /// <param name="missingValueIndicator">Value to use for missing data (default: -999)</param>
        /// <param name="encoder">Optional custom neural network encoder. If null, a default encoder will be created when needed.</param>
        public NumericalModalityEncoder(int outputDimension = 128, bool useFeatureEngineering = true,
            bool useInteractionFeatures = true, int polynomialDegree = 2,
            NormalizationMethod normalizationMethod = NormalizationMethod.StandardScore,
            double missingValueIndicator = -999, INeuralNetworkModel<T>? encoder = null) 
            : base("Numerical", outputDimension, encoder)
        {
            _useFeatureEngineering = useFeatureEngineering;
            _useInteractionFeatures = useInteractionFeatures;
            _polynomialDegree = Math.Max(1, Math.Min(polynomialDegree, 3)); // Limit to reasonable range
            _normalizationMethod = normalizationMethod;
            _missingValueIndicator = _numericOps.FromDouble(missingValueIndicator);
            _statisticsComputed = false;
        }

        /// <summary>
        /// Encodes numerical data into a vector representation
        /// </summary>
        /// <param name="input">Numerical data as array, Vector, or Tensor</param>
        /// <returns>Encoded vector representation</returns>
        public override Vector<T> Encode(object input)
        {
            if (!ValidateInput(input))
            {
                throw new ArgumentException($"Invalid input type for numerical encoding. Expected numerical array or Vector/Tensor, got {input?.GetType()?.Name ?? "null"}");
            }

            // Preprocess the input
            var preprocessed = Preprocess(input);
            var numericData = preprocessed as T[] ?? throw new InvalidOperationException("Preprocessing failed");

            // Extract features
            var features = ExtractNumericalFeatures(numericData);
            
            // Project to output dimension if needed
            if (features.Length != OutputDimension)
            {
                features = ProjectToOutputDimension(features);
            }

            // Final normalization
            return Normalize(features);
        }

        /// <summary>
        /// Preprocesses raw numerical input
        /// </summary>
        public override object Preprocess(object input)
        {
            T[] data;

            switch (input)
            {
                case T[] genericArray:
                    data = (T[])genericArray.Clone();
                    break;
                case double[] doubleArray:
                    data = doubleArray.Select(d => _numericOps.FromDouble(d)).ToArray();
                    break;
                case float[] floatArray:
                    data = floatArray.Select(f => _numericOps.FromDouble(f)).ToArray();
                    break;
                case int[] intArray:
                    data = intArray.Select(i => _numericOps.FromDouble(i)).ToArray();
                    break;
                case Vector<T> vector:
                    data = vector.ToArray();
                    break;
                case Vector<double> doubleVector:
                    data = doubleVector.ToArray().Select(d => _numericOps.FromDouble(d)).ToArray();
                    break;
                case Vector<float> floatVector:
                    data = floatVector.ToArray().Select(f => _numericOps.FromDouble(f)).ToArray();
                    break;
                case Tensor<T> tensor:
                    if (tensor.Rank != 1)
                        throw new ArgumentException($"Tensor must be 1D for numerical encoding, got rank {tensor.Rank}");
                    data = tensor.ToArray();
                    break;
                case Tensor<double> doubleTensor:
                    if (doubleTensor.Rank != 1)
                        throw new ArgumentException($"Tensor must be 1D for numerical encoding, got rank {doubleTensor.Rank}");
                    data = doubleTensor.ToArray().Select(d => _numericOps.FromDouble(d)).ToArray();
                    break;
                case Tensor<float> floatTensor:
                    if (floatTensor.Rank != 1)
                        throw new ArgumentException($"Tensor must be 1D for numerical encoding, got rank {floatTensor.Rank}");
                    data = floatTensor.ToArray().Select(f => _numericOps.FromDouble(f)).ToArray();
                    break;
                default:
                    throw new ArgumentException($"Unsupported input type: {input?.GetType()?.Name ?? "null"}");
            }

            // Handle missing values
            data = HandleMissingValues(data);

            // Compute statistics if needed
            if (!_statisticsComputed && _normalizationMethod != NormalizationMethod.None)
            {
                ComputeStatistics(data);
            }

            // Apply normalization
            data = ApplyNormalization(data);

            return data;
        }

        /// <summary>
        /// Validates the input for numerical encoding
        /// </summary>
        protected override bool ValidateInput(object input)
        {
            return input is T[] || input is double[] || input is float[] || input is int[] ||
                   input is Vector<T> || input is Vector<double> || input is Vector<float> ||
                   input is Tensor<T> || input is Tensor<double> || input is Tensor<float>;
        }

        /// <summary>
        /// Extracts features from numerical data
        /// </summary>
        private Vector<T> ExtractNumericalFeatures(T[] data)
        {
            var features = new List<double>();

            // Convert T[] to double[] for feature extraction
            var doubleData = data.Select(x => Convert.ToDouble(x)).ToArray();

            // Original features
            features.AddRange(doubleData);

            if (_useFeatureEngineering)
            {
                // Statistical aggregations
                features.AddRange(ExtractStatisticalFeatures(doubleData));

                // Polynomial features
                if (_polynomialDegree > 1)
                {
                    features.AddRange(ExtractPolynomialFeatures(doubleData));
                }

                // Interaction features
                if (_useInteractionFeatures && doubleData.Length > 1)
                {
                    features.AddRange(ExtractInteractionFeatures(doubleData));
                }

                // Binned features
                features.AddRange(ExtractBinnedFeatures(doubleData));
            }

            // Convert back to T[]
            var tFeatures = features.Select(f => _numericOps.FromDouble(f)).ToArray();
            return new Vector<T>(tFeatures);
        }

        /// <summary>
        /// Extracts statistical features from the data
        /// </summary>
        private double[] ExtractStatisticalFeatures(double[] data)
        {
            var features = new List<double>();

            if (data.Length == 0)
                return features.ToArray();

            // Basic statistics
            double mean = data.Average();
            double variance = data.Select(x => Math.Pow(x - mean, 2)).Average();
            double stdDev = Math.Sqrt(variance);
            
            features.Add(mean);
            features.Add(stdDev);
            features.Add(data.Min());
            features.Add(data.Max());
            features.Add(data.Max() - data.Min()); // Range

            // Higher moments
            if (stdDev > 0)
            {
                double skewness = data.Select(x => Math.Pow((x - mean) / stdDev, 3)).Average();
                double kurtosis = data.Select(x => Math.Pow((x - mean) / stdDev, 4)).Average() - 3;
                features.Add(skewness);
                features.Add(kurtosis);
            }
            else
            {
                features.Add(0); // Skewness
                features.Add(0); // Kurtosis
            }

            // Percentiles
            var sorted = data.OrderBy(x => x).ToArray();
            features.Add(GetPercentile(sorted, 0.25)); // Q1
            features.Add(GetPercentile(sorted, 0.50)); // Median
            features.Add(GetPercentile(sorted, 0.75)); // Q3

            return features.ToArray();
        }

        /// <summary>
        /// Extracts polynomial features
        /// </summary>
        private double[] ExtractPolynomialFeatures(double[] data)
        {
            var features = new List<double>();

            // Square features
            foreach (var value in data)
            {
                features.Add(value * value);
            }

            // Cubic features if degree >= 3
            if (_polynomialDegree >= 3)
            {
                foreach (var value in data)
                {
                    features.Add(value * value * value);
                }
            }

            return features.ToArray();
        }

        /// <summary>
        /// Extracts interaction features between pairs of features
        /// </summary>
        private double[] ExtractInteractionFeatures(double[] data)
        {
            var features = new List<double>();
            int maxInteractions = Math.Min(data.Length * (data.Length - 1) / 2, 50); // Limit interactions
            int added = 0;

            for (int i = 0; i < data.Length && added < maxInteractions; i++)
            {
                for (int j = i + 1; j < data.Length && added < maxInteractions; j++)
                {
                    // Multiplication interaction
                    features.Add(data[i] * data[j]);
                    
                    // Difference interaction
                    features.Add(data[i] - data[j]);
                    
                    // Ratio interaction (with protection against division by zero)
                    if (Math.Abs(data[j]) > 1e-8)
                    {
                        features.Add(data[i] / data[j]);
                    }
                    else
                    {
                        features.Add(0);
                    }
                    
                    added++;
                }
            }

            return features.ToArray();
        }

        /// <summary>
        /// Creates binned/discretized features
        /// </summary>
        private double[] ExtractBinnedFeatures(double[] data)
        {
            var features = new List<double>();
            int numBins = 5;

            foreach (var value in data)
            {
                // Create one-hot encoded bins
                var bins = new double[numBins];
                double normalizedValue = (value - data.Min()) / (data.Max() - data.Min() + 1e-8);
                int binIndex = Math.Min((int)(normalizedValue * numBins), numBins - 1);
                bins[binIndex] = 1.0;
                features.AddRange(bins);
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

            // Convert to double for processing
            var doubleFeatures = features.ToArray().Select(f => Convert.ToDouble(f)).ToArray();
            var result = new double[OutputDimension];

            if (doubleFeatures.Length > OutputDimension)
            {
                // Use feature selection based on variance
                var featureVariances = new List<(int Index, double Variance)>();
                
                for (int i = 0; i < doubleFeatures.Length; i++)
                {
                    // Simple variance estimate using feature value as proxy
                    featureVariances.Add((i, Math.Abs(doubleFeatures[i])));
                }

                // Select top features by variance
                var selectedIndices = featureVariances
                    .OrderByDescending(f => f.Variance)
                    .Take(OutputDimension)
                    .OrderBy(f => f.Index)
                    .Select(f => f.Index)
                    .ToList();

                for (int i = 0; i < OutputDimension; i++)
                {
                    result[i] = doubleFeatures[selectedIndices[i]];
                }
            }
            else
            {
                // Copy existing features
                for (int i = 0; i < doubleFeatures.Length; i++)
                {
                    result[i] = doubleFeatures[i];
                }
                
                // Pad with learned representations
                var random = new Random(42);
                for (int i = doubleFeatures.Length; i < OutputDimension; i++)
                {
                    // Create synthetic features based on existing ones
                    int idx1 = random.Next(doubleFeatures.Length);
                    int idx2 = random.Next(doubleFeatures.Length);
                    result[i] = (doubleFeatures[idx1] + doubleFeatures[idx2]) / 2.0;
                }
            }

            // Convert back to T
            var tResult = result.Select(r => _numericOps.FromDouble(r)).ToArray();
            return new Vector<T>(tResult);
        }

        /// <summary>
        /// Handles missing values in the data
        /// </summary>
        private T[] HandleMissingValues(T[] data)
        {
            var result = new T[data.Length];
            
            for (int i = 0; i < data.Length; i++)
            {
                double doubleValue = Convert.ToDouble(data[i]);
                if (double.IsNaN(doubleValue) || double.IsInfinity(doubleValue))
                {
                    result[i] = _missingValueIndicator;
                }
                else
                {
                    result[i] = data[i];
                }
            }

            return result;
        }

        /// <summary>
        /// Computes statistics for normalization
        /// </summary>
        private void ComputeStatistics(T[] data)
        {
            int n = data.Length;
            _featureMeans = new T[n];
            _featureStdDevs = new T[n];
            _featureMins = new T[n];
            _featureMaxs = new T[n];

            // For this simple implementation, compute element-wise statistics
            for (int i = 0; i < n; i++)
            {
                _featureMeans[i] = data[i];
                _featureStdDevs[i] = _numericOps.One; // Default to 1 for single sample
                _featureMins[i] = data[i];
                _featureMaxs[i] = data[i];
            }

            _statisticsComputed = true;
        }

        /// <summary>
        /// Applies normalization to the data
        /// </summary>
        private T[] ApplyNormalization(T[] data)
        {
            if (_normalizationMethod == NormalizationMethod.None)
                return data;

            var normalized = new T[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                switch (_normalizationMethod)
                {
                    case NormalizationMethod.StandardScore:
                        if (_statisticsComputed && i < _featureStdDevs.Length && _numericOps.GreaterThan(_featureStdDevs[i], _numericOps.Zero))
                        {
                            T diff = _numericOps.Subtract(data[i], _featureMeans[i]);
                            normalized[i] = _numericOps.Divide(diff, _featureStdDevs[i]);
                        }
                        else
                        {
                            normalized[i] = data[i];
                        }
                        break;

                    case NormalizationMethod.MinMax:
                        if (_statisticsComputed && i < _featureMaxs.Length)
                        {
                            T range = _numericOps.Subtract(_featureMaxs[i], _featureMins[i]);
                            if (_numericOps.GreaterThan(range, _numericOps.Zero))
                            {
                                T diff = _numericOps.Subtract(data[i], _featureMins[i]);
                                normalized[i] = _numericOps.Divide(diff, range);
                            }
                            else
                            {
                                normalized[i] = _numericOps.FromDouble(0.5); // Center if no range
                            }
                        }
                        else
                        {
                            normalized[i] = data[i];
                        }
                        break;

                    case NormalizationMethod.Robust:
                        // Simplified robust scaling using median and IQR approximation
                        T absValue = !_numericOps.LessThan(data[i], _numericOps.Zero) ? data[i] : _numericOps.Negate(data[i]);
                        T denominator = _numericOps.Add(_numericOps.One, absValue);
                        normalized[i] = _numericOps.Divide(data[i], denominator);
                        break;

                    default:
                        normalized[i] = data[i];
                        break;
                }
            }

            return normalized;
        }

        /// <summary>
        /// Gets the percentile value from sorted data
        /// </summary>
        private double GetPercentile(double[] sortedData, double percentile)
        {
            if (sortedData.Length == 0)
                return 0;

            double index = percentile * (sortedData.Length - 1);
            int lower = (int)Math.Floor(index);
            int upper = (int)Math.Ceiling(index);

            if (lower == upper)
                return sortedData[lower];

            double weight = index - lower;
            return sortedData[lower] * (1 - weight) + sortedData[upper] * weight;
        }

        /// <summary>
        /// Trains the encoder on training data (for learning statistics)
        /// </summary>
        public void Train(Vector<T>[] trainingData)
        {
            if (trainingData == null || trainingData.Length == 0)
                return;

            int numFeatures = trainingData[0].Length;
            _featureMeans = new T[numFeatures];
            _featureStdDevs = new T[numFeatures];
            _featureMins = new T[numFeatures];
            _featureMaxs = new T[numFeatures];

            // Initialize min/max
            for (int i = 0; i < numFeatures; i++)
            {
                _featureMins[i] = _numericOps.FromDouble(double.MaxValue);
                _featureMaxs[i] = _numericOps.FromDouble(double.MinValue);
            }

            // Compute means and min/max
            var tempMeans = new T[numFeatures];
            var tempMins = new T[numFeatures];
            var tempMaxs = new T[numFeatures];
            
            for (int i = 0; i < numFeatures; i++)
            {
                tempMins[i] = _numericOps.FromDouble(double.MaxValue);
                tempMaxs[i] = _numericOps.FromDouble(double.MinValue);
                tempMeans[i] = _numericOps.Zero;
            }

            foreach (var sample in trainingData)
            {
                for (int i = 0; i < numFeatures && i < sample.Length; i++)
                {
                    tempMeans[i] = _numericOps.Add(tempMeans[i], sample[i]);
                    if (_numericOps.LessThan(sample[i], tempMins[i]))
                        tempMins[i] = sample[i];
                    if (_numericOps.GreaterThan(sample[i], tempMaxs[i]))
                        tempMaxs[i] = sample[i];
                }
            }

            for (int i = 0; i < numFeatures; i++)
            {
                _featureMeans[i] = _numericOps.Divide(tempMeans[i], _numericOps.FromDouble(trainingData.Length));
                _featureMins[i] = tempMins[i];
                _featureMaxs[i] = tempMaxs[i];
            }

            // Compute standard deviations
            var tempStdDevs = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                tempStdDevs[i] = _numericOps.Zero;
            }

            foreach (var sample in trainingData)
            {
                for (int i = 0; i < numFeatures && i < sample.Length; i++)
                {
                    T diff = _numericOps.Subtract(sample[i], _featureMeans[i]);
                    tempStdDevs[i] = _numericOps.Add(tempStdDevs[i], _numericOps.Multiply(diff, diff));
                }
            }

            for (int i = 0; i < numFeatures; i++)
            {
                T variance = _numericOps.Divide(tempStdDevs[i], _numericOps.FromDouble(trainingData.Length));
                _featureStdDevs[i] = _numericOps.Sqrt(variance);
            }

            _statisticsComputed = true;
        }
    }
}