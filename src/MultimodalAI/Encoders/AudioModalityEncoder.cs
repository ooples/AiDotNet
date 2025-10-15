using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MultimodalAI.Encoders
{
    /// <summary>
    /// Audio-specific modality encoder for processing audio data
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class AudioModalityEncoder<T> : ModalityEncoderBase<T>
    {
        private readonly int _sampleRate;
        private readonly int _frameSize;
        private readonly bool _useSpectralFeatures;
        
        /// <summary>
        /// Initializes a new instance of AudioModalityEncoder
        /// </summary>
        /// <param name="outputDimension">Output dimension of the encoder (default: 256)</param>
        /// <param name="sampleRate">Sample rate of audio data (default: 16000)</param>
        /// <param name="frameSize">Frame size for feature extraction (default: 512)</param>
        /// <param name="useSpectralFeatures">Whether to extract spectral features (default: true)</param>
        /// <param name="encoder">Optional custom neural network encoder. If null, a default encoder will be created when needed.</param>
        public AudioModalityEncoder(int outputDimension = 256, int sampleRate = 16000, 
            int frameSize = 512, bool useSpectralFeatures = true, INeuralNetworkModel<T>? encoder = null) 
            : base("Audio", outputDimension, encoder)
        {
            _sampleRate = sampleRate;
            _frameSize = frameSize;
            _useSpectralFeatures = useSpectralFeatures;
        }

        /// <summary>
        /// Encodes audio data into a vector representation
        /// </summary>
        /// <param name="input">Audio data as double[], float[], or Tensor</param>
        /// <returns>Encoded vector representation</returns>
        public override Vector<T> Encode(object input)
        {
            if (!ValidateInput(input))
            {
                throw new ArgumentException($"Invalid input type for audio encoding. Expected double[], float[], or Tensor<double>, got {input?.GetType()?.Name ?? "null"}");
            }

            // Preprocess the input
            var preprocessed = Preprocess(input);
            var audioData = preprocessed as T[] ?? throw new InvalidOperationException("Preprocessing failed");

            // Extract features
            var features = ExtractAudioFeatures(audioData);
            
            // Project to output dimension if needed
            if (features.Length != OutputDimension)
            {
                features = ProjectToOutputDimension(features);
            }

            // Normalize the output
            return Normalize(features);
        }

        /// <summary>
        /// Preprocesses raw audio input
        /// </summary>
        public override object Preprocess(object input)
        {
            T[] audioData;

            switch (input)
            {
                case T[] genericArray:
                    audioData = (T[])genericArray.Clone();
                    break;
                case double[] doubleArray:
                    audioData = doubleArray.Select(d => _numericOps.FromDouble(d)).ToArray();
                    break;
                case float[] floatArray:
                    audioData = floatArray.Select(f => _numericOps.FromDouble(f)).ToArray();
                    break;
                case int[] intArray:
                    audioData = intArray.Select(i => _numericOps.FromDouble(i)).ToArray();
                    break;
                case Tensor<T> tensor:
                    audioData = tensor.ToArray();
                    break;
                case Tensor<double> doubleTensor:
                    audioData = doubleTensor.ToArray().Select(d => _numericOps.FromDouble(d)).ToArray();
                    break;
                case Tensor<float> floatTensor:
                    audioData = floatTensor.ToArray().Select(f => _numericOps.FromDouble(f)).ToArray();
                    break;
                default:
                    throw new ArgumentException($"Unsupported input type: {input?.GetType()?.Name ?? "null"}");
            }

            // Apply preprocessing steps
            audioData = RemoveDCOffset(audioData);
            audioData = NormalizeAmplitude(audioData);
            
            if (_useSpectralFeatures)
            {
                audioData = ApplyPreEmphasis(audioData, _numericOps.FromDouble(0.97));
            }

            return audioData;
        }

        /// <summary>
        /// Validates the input for audio encoding
        /// </summary>
        protected override bool ValidateInput(object input)
        {
            return input is T[] || input is double[] || input is float[] || input is int[] ||
                   input is Tensor<T> || input is Tensor<double> || input is Tensor<float>;
        }

        /// <summary>
        /// Extracts audio features from preprocessed data
        /// </summary>
        private Vector<T> ExtractAudioFeatures(T[] audioData)
        {
            var features = new System.Collections.Generic.List<T>();

            // Time domain features
            features.AddRange(ExtractTimeDomainFeatures(audioData));

            if (_useSpectralFeatures)
            {
                // Frequency domain features
                features.AddRange(ExtractSpectralFeatures(audioData));
            }

            // Statistical features
            features.AddRange(ExtractStatisticalFeatures(audioData));

            return new Vector<T>(features.ToArray());
        }

        /// <summary>
        /// Extracts time domain features
        /// </summary>
        private T[] ExtractTimeDomainFeatures(T[] audioData)
        {
            var features = new System.Collections.Generic.List<T>();

            // Zero Crossing Rate
            T zcr = CalculateZeroCrossingRate(audioData);
            features.Add(zcr);

            // Energy
            T energy = ComputeEnergy(audioData);
            features.Add(energy);

            // Root Mean Square
            T rms = _numericOps.Sqrt(energy);
            features.Add(rms);

            // Peak amplitude
            T peak = ComputePeakAmplitude(audioData);
            features.Add(peak);

            return features.ToArray();
        }

        /// <summary>
        /// Extracts spectral features using basic FFT
        /// </summary>
        private T[] ExtractSpectralFeatures(T[] audioData)
        {
            var features = new System.Collections.Generic.List<T>();

            // Simple spectral analysis (placeholder for full FFT)
            // In production, you would use a proper FFT library
            int numBins = Math.Min(32, audioData.Length / 2);
            var spectrum = ComputeSimpleSpectrum(audioData, numBins);

            // Spectral centroid
            T centroid = CalculateSpectralCentroid(spectrum);
            features.Add(centroid);

            // Spectral spread
            T spread = CalculateSpectralSpread(spectrum, centroid);
            features.Add(spread);

            // Spectral flux
            T flux = CalculateSpectralFlux(spectrum);
            features.Add(flux);

            // Add spectral bins (reduced set)
            features.AddRange(spectrum.Take(Math.Min(16, spectrum.Length)));

            return features.ToArray();
        }

        /// <summary>
        /// Extracts statistical features
        /// </summary>
        private T[] ExtractStatisticalFeatures(T[] audioData)
        {
            var features = new System.Collections.Generic.List<T>();

            // Mean
            T sum = _numericOps.Zero;
            foreach (var value in audioData)
            {
                sum = _numericOps.Add(sum, value);
            }
            T mean = _numericOps.Divide(sum, _numericOps.FromDouble(audioData.Length));
            features.Add(mean);

            // Standard deviation
            T varianceSum = _numericOps.Zero;
            foreach (var value in audioData)
            {
                T diff = _numericOps.Subtract(value, mean);
                varianceSum = _numericOps.Add(varianceSum, _numericOps.Multiply(diff, diff));
            }
            T variance = _numericOps.Divide(varianceSum, _numericOps.FromDouble(audioData.Length));
            T stdDev = _numericOps.Sqrt(variance);
            features.Add(stdDev);

            // Simplified skewness (third moment)
            T skewnessSum = _numericOps.Zero;
            foreach (var value in audioData)
            {
                T diff = _numericOps.Subtract(value, mean);
                T normalizedDiff = _numericOps.GreaterThan(stdDev, _numericOps.Zero) 
                    ? _numericOps.Divide(diff, stdDev) 
                    : _numericOps.Zero;
                skewnessSum = _numericOps.Add(skewnessSum, 
                    _numericOps.Multiply(_numericOps.Multiply(normalizedDiff, normalizedDiff), normalizedDiff));
            }
            T skewness = _numericOps.Divide(skewnessSum, _numericOps.FromDouble(audioData.Length));
            features.Add(skewness);

            // Simplified kurtosis (fourth moment - 3)
            T kurtosisSum = _numericOps.Zero;
            foreach (var value in audioData)
            {
                T diff = _numericOps.Subtract(value, mean);
                T normalizedDiff = _numericOps.GreaterThan(stdDev, _numericOps.Zero) 
                    ? _numericOps.Divide(diff, stdDev) 
                    : _numericOps.Zero;
                T fourth = _numericOps.Multiply(_numericOps.Multiply(normalizedDiff, normalizedDiff), 
                                              _numericOps.Multiply(normalizedDiff, normalizedDiff));
                kurtosisSum = _numericOps.Add(kurtosisSum, fourth);
            }
            T kurtosis = _numericOps.Subtract(_numericOps.Divide(kurtosisSum, _numericOps.FromDouble(audioData.Length)), 
                                            _numericOps.FromDouble(3));
            features.Add(kurtosis);

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
                // Downsample by averaging groups
                int groupSize = features.Length / OutputDimension;
                for (int i = 0; i < OutputDimension; i++)
                {
                    int start = i * groupSize;
                    int end = Math.Min(start + groupSize, features.Length);
                    T sum = _numericOps.Zero;
                    for (int j = start; j < end; j++)
                    {
                        sum = _numericOps.Add(sum, features[j]);
                    }
                    result[i] = _numericOps.Divide(sum, _numericOps.FromDouble(end - start));
                }
            }
            else
            {
                // Upsample by interpolation
                double scale = (double)(features.Length - 1) / (OutputDimension - 1);
                for (int i = 0; i < OutputDimension; i++)
                {
                    double pos = i * scale;
                    int lower = (int)pos;
                    int upper = Math.Min(lower + 1, features.Length - 1);
                    double frac = pos - lower;
                    
                    T lowerValue = features[lower];
                    T upperValue = features[upper];
                    T fracT = _numericOps.FromDouble(frac);
                    T oneFrac = _numericOps.FromDouble(1 - frac);
                    
                    result[i] = _numericOps.Add(
                        _numericOps.Multiply(lowerValue, oneFrac),
                        _numericOps.Multiply(upperValue, fracT)
                    );
                }
            }

            return new Vector<T>(result);
        }

        /// <summary>
        /// Removes DC offset from audio signal
        /// </summary>
        private T[] RemoveDCOffset(T[] audio)
        {
            T sum = _numericOps.Zero;
            foreach (var sample in audio)
            {
                sum = _numericOps.Add(sum, sample);
            }
            T mean = _numericOps.Divide(sum, _numericOps.FromDouble(audio.Length));
            
            return audio.Select(x => _numericOps.Subtract(x, mean)).ToArray();
        }

        /// <summary>
        /// Normalizes audio amplitude to [-1, 1] range
        /// </summary>
        private T[] NormalizeAmplitude(T[] audio)
        {
            T maxAbs = ComputePeakAmplitude(audio);
            if (_numericOps.GreaterThan(maxAbs, _numericOps.Zero))
            {
                return audio.Select(x => _numericOps.Divide(x, maxAbs)).ToArray();
            }
            return audio;
        }

        /// <summary>
        /// Applies pre-emphasis filter
        /// </summary>
        private T[] ApplyPreEmphasis(T[] audio, T coefficient)
        {
            var result = new T[audio.Length];
            result[0] = audio[0];
            for (int i = 1; i < audio.Length; i++)
            {
                result[i] = _numericOps.Subtract(audio[i], _numericOps.Multiply(coefficient, audio[i - 1]));
            }
            return result;
        }

        /// <summary>
        /// Calculates zero crossing rate
        /// </summary>
        private T CalculateZeroCrossingRate(T[] audio)
        {
            int crossings = 0;
            for (int i = 1; i < audio.Length; i++)
            {
                bool currentPositive = !_numericOps.LessThan(audio[i], _numericOps.Zero);
                bool previousPositive = !_numericOps.LessThan(audio[i - 1], _numericOps.Zero);
                if (currentPositive != previousPositive)
                {
                    crossings++;
                }
            }
            return _numericOps.Divide(_numericOps.FromDouble(crossings), _numericOps.FromDouble(audio.Length - 1));
        }

        /// <summary>
        /// Computes a simple spectrum (placeholder for FFT)
        /// </summary>
        private T[] ComputeSimpleSpectrum(T[] audio, int numBins)
        {
            var spectrum = new T[numBins];
            int windowSize = audio.Length / numBins;

            for (int i = 0; i < numBins; i++)
            {
                int start = i * windowSize;
                int end = Math.Min(start + windowSize, audio.Length);
                T energy = _numericOps.Zero;
                for (int j = start; j < end; j++)
                {
                    energy = _numericOps.Add(energy, _numericOps.Multiply(audio[j], audio[j]));
                }
                T avgEnergy = _numericOps.Divide(energy, _numericOps.FromDouble(end - start));
                spectrum[i] = _numericOps.Sqrt(avgEnergy);
            }

            return spectrum;
        }

        /// <summary>
        /// Calculates spectral centroid
        /// </summary>
        private T CalculateSpectralCentroid(T[] spectrum)
        {
            T weightedSum = _numericOps.Zero;
            T magnitudeSum = _numericOps.Zero;

            for (int i = 0; i < spectrum.Length; i++)
            {
                T weighted = _numericOps.Multiply(_numericOps.FromDouble(i), spectrum[i]);
                weightedSum = _numericOps.Add(weightedSum, weighted);
                magnitudeSum = _numericOps.Add(magnitudeSum, spectrum[i]);
            }

            return _numericOps.GreaterThan(magnitudeSum, _numericOps.Zero) 
                ? _numericOps.Divide(weightedSum, magnitudeSum) 
                : _numericOps.Zero;
        }

        /// <summary>
        /// Calculates spectral spread
        /// </summary>
        private T CalculateSpectralSpread(T[] spectrum, T centroid)
        {
            T weightedVariance = _numericOps.Zero;
            T magnitudeSum = _numericOps.Zero;

            for (int i = 0; i < spectrum.Length; i++)
            {
                T deviation = _numericOps.Subtract(_numericOps.FromDouble(i), centroid);
                T deviationSquared = _numericOps.Multiply(deviation, deviation);
                T weighted = _numericOps.Multiply(deviationSquared, spectrum[i]);
                weightedVariance = _numericOps.Add(weightedVariance, weighted);
                magnitudeSum = _numericOps.Add(magnitudeSum, spectrum[i]);
            }

            return _numericOps.GreaterThan(magnitudeSum, _numericOps.Zero) 
                ? _numericOps.Sqrt(_numericOps.Divide(weightedVariance, magnitudeSum))
                : _numericOps.Zero;
        }

        /// <summary>
        /// Calculates spectral flux
        /// </summary>
        private T CalculateSpectralFlux(T[] spectrum)
        {
            // For a single frame, return the sum of squared magnitudes
            T sum = _numericOps.Zero;
            foreach (var value in spectrum)
            {
                sum = _numericOps.Add(sum, _numericOps.Multiply(value, value));
            }
            return sum;
        }

        /// <summary>
        /// Computes energy of the audio signal
        /// </summary>
        private T ComputeEnergy(T[] audioData)
        {
            T sum = _numericOps.Zero;
            foreach (var sample in audioData)
            {
                sum = _numericOps.Add(sum, _numericOps.Multiply(sample, sample));
            }
            return _numericOps.Divide(sum, _numericOps.FromDouble(audioData.Length));
        }

        /// <summary>
        /// Computes peak amplitude of the audio signal
        /// </summary>
        private T ComputePeakAmplitude(T[] audioData)
        {
            T max = _numericOps.Zero;
            foreach (var sample in audioData)
            {
                T abs = !_numericOps.LessThan(sample, _numericOps.Zero) ? sample : _numericOps.Negate(sample);
                if (_numericOps.GreaterThan(abs, max))
                    max = abs;
            }
            return max;
        }
    }
}