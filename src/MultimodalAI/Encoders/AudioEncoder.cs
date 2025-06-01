using AiDotNet.LinearAlgebra;
using System;
using System.Linq;

namespace AiDotNet.MultimodalAI.Encoders
{
    /// <summary>
    /// Encoder for audio modality
    /// </summary>
    public class AudioEncoder : ModalityEncoder
    {
        private readonly int _sampleRate;
        private readonly int _windowSize;
        private readonly int _hopLength;
        private readonly bool _useMfcc;
        private readonly bool _useSpectralFeatures;
        private readonly int _numMfccCoefficients;

        /// <summary>
        /// Initializes a new instance of AudioEncoder
        /// </summary>
        /// <param name="outputDimension">Output dimension of the encoder</param>
        /// <param name="sampleRate">Sample rate of audio</param>
        /// <param name="windowSize">Window size for feature extraction</param>
        /// <param name="hopLength">Hop length for sliding window</param>
        /// <param name="useMfcc">Whether to use MFCC features</param>
        /// <param name="useSpectralFeatures">Whether to use spectral features</param>
        /// <param name="numMfccCoefficients">Number of MFCC coefficients</param>
        public AudioEncoder(int outputDimension, int sampleRate = 16000, int windowSize = 1024, 
                          int hopLength = 512, bool useMfcc = true, bool useSpectralFeatures = true,
                          int numMfccCoefficients = 13)
            : base("audio", outputDimension)
        {
            _sampleRate = sampleRate;
            _windowSize = windowSize;
            _hopLength = hopLength;
            _useMfcc = useMfcc;
            _useSpectralFeatures = useSpectralFeatures;
            _numMfccCoefficients = numMfccCoefficients;
        }

        /// <summary>
        /// Encodes audio input into a vector representation
        /// </summary>
        /// <param name="input">Audio input as array</param>
        /// <returns>Encoded vector representation</returns>
        public override Vector<double> Encode(object input)
        {
            if (!ValidateInput(input))
                throw new ArgumentException("Invalid input type for audio encoder");

            var preprocessed = Preprocess(input);
            var audioData = preprocessed as double[] ?? ConvertToDoubleArray(preprocessed);

            // Extract features
            var features = new Vector<double>(GetFeatureSize());
            int featureIndex = 0;

            // Extract temporal features
            var temporal = ExtractTemporalFeatures(audioData);
            for (int i = 0; i < temporal.Dimension; i++)
            {
                features[featureIndex++] = temporal[i];
            }

            // Extract spectral features
            if (_useSpectralFeatures)
            {
                var spectral = ExtractSpectralFeatures(audioData);
                for (int i = 0; i < spectral.Dimension && featureIndex < features.Dimension; i++)
                {
                    features[featureIndex++] = spectral[i];
                }
            }

            // Extract MFCC features
            if (_useMfcc)
            {
                var mfcc = ExtractMfccFeatures(audioData);
                for (int i = 0; i < mfcc.Dimension && featureIndex < features.Dimension; i++)
                {
                    features[featureIndex++] = mfcc[i];
                }
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
        /// Preprocesses audio input
        /// </summary>
        /// <param name="input">Raw audio input</param>
        /// <returns>Preprocessed audio as double array</returns>
        public override object Preprocess(object input)
        {
            double[] audioArray = input switch
            {
                double[] arr => arr,
                float[] arr => arr.Select(x => (double)x).ToArray(),
                short[] arr => arr.Select(x => x / 32768.0).ToArray(),
                byte[] arr => arr.Select(x => (x - 128) / 128.0).ToArray(),
                _ => throw new ArgumentException("Unsupported audio format")
            };

            // Normalize audio to [-1, 1]
            audioArray = NormalizeAudio(audioArray);

            // Apply pre-emphasis filter
            audioArray = ApplyPreEmphasis(audioArray, 0.97);

            return audioArray;
        }

        /// <summary>
        /// Validates the input for audio encoding
        /// </summary>
        /// <param name="input">Input to validate</param>
        /// <returns>True if valid</returns>
        protected override bool ValidateInput(object input)
        {
            return input is double[] || input is float[] || input is short[] || input is byte[];
        }

        /// <summary>
        /// Extracts temporal features from audio
        /// </summary>
        private Vector<double> ExtractTemporalFeatures(double[] audio)
        {
            var features = new Vector<double>(6);
            
            // Zero crossing rate
            features[0] = CalculateZeroCrossingRate(audio);
            
            // Energy
            features[1] = CalculateEnergy(audio);
            
            // RMS
            features[2] = CalculateRMS(audio);
            
            // Temporal centroid
            features[3] = CalculateTemporalCentroid(audio);
            
            // Max amplitude
            features[4] = audio.Max(Math.Abs);
            
            // Dynamic range
            features[5] = audio.Max() - audio.Min();
            
            return features;
        }

        /// <summary>
        /// Extracts spectral features from audio
        /// </summary>
        private Vector<double> ExtractSpectralFeatures(double[] audio)
        {
            int numFrames = (audio.Length - _windowSize) / _hopLength + 1;
            var features = new Vector<double>(5 * numFrames); // 5 features per frame
            int featureIndex = 0;

            for (int frame = 0; frame < numFrames; frame++)
            {
                int start = frame * _hopLength;
                var windowedAudio = ApplyWindow(audio, start, _windowSize);
                var spectrum = ComputeSpectrum(windowedAudio);

                // Spectral centroid
                features[featureIndex++] = CalculateSpectralCentroid(spectrum);
                
                // Spectral spread
                features[featureIndex++] = CalculateSpectralSpread(spectrum);
                
                // Spectral flux
                if (frame > 0)
                {
                    var prevStart = (frame - 1) * _hopLength;
                    var prevWindow = ApplyWindow(audio, prevStart, _windowSize);
                    var prevSpectrum = ComputeSpectrum(prevWindow);
                    features[featureIndex++] = CalculateSpectralFlux(spectrum, prevSpectrum);
                }
                else
                {
                    features[featureIndex++] = 0;
                }
                
                // Spectral rolloff
                features[featureIndex++] = CalculateSpectralRolloff(spectrum, 0.85);
                
                // Spectral entropy
                features[featureIndex++] = CalculateSpectralEntropy(spectrum);
            }

            return features;
        }

        /// <summary>
        /// Extracts MFCC features from audio (simplified)
        /// </summary>
        private Vector<double> ExtractMfccFeatures(double[] audio)
        {
            int numFrames = (audio.Length - _windowSize) / _hopLength + 1;
            var mfccFeatures = new Vector<double>(_numMfccCoefficients * numFrames);
            int featureIndex = 0;

            for (int frame = 0; frame < numFrames; frame++)
            {
                int start = frame * _hopLength;
                var windowedAudio = ApplyWindow(audio, start, _windowSize);
                var spectrum = ComputeSpectrum(windowedAudio);
                
                // Simplified MFCC calculation
                var melSpectrum = ConvertToMelScale(spectrum);
                var logMelSpectrum = melSpectrum.Select(x => Math.Log(x + 1e-10)).ToArray();
                var mfcc = ApplyDCT(logMelSpectrum, _numMfccCoefficients);

                for (int i = 0; i < _numMfccCoefficients; i++)
                {
                    mfccFeatures[featureIndex++] = mfcc[i];
                }
            }

            return mfccFeatures;
        }

        /// <summary>
        /// Calculates zero crossing rate
        /// </summary>
        private double CalculateZeroCrossingRate(double[] audio)
        {
            int crossings = 0;
            for (int i = 1; i < audio.Length; i++)
            {
                if (Math.Sign(audio[i]) != Math.Sign(audio[i - 1]))
                    crossings++;
            }
            return (double)crossings / audio.Length;
        }

        /// <summary>
        /// Calculates energy of audio signal
        /// </summary>
        private double CalculateEnergy(double[] audio)
        {
            return audio.Sum(x => x * x) / audio.Length;
        }

        /// <summary>
        /// Calculates RMS of audio signal
        /// </summary>
        private double CalculateRMS(double[] audio)
        {
            return Math.Sqrt(CalculateEnergy(audio));
        }

        /// <summary>
        /// Calculates temporal centroid
        /// </summary>
        private double CalculateTemporalCentroid(double[] audio)
        {
            double weightedSum = 0;
            double energySum = 0;
            
            for (int i = 0; i < audio.Length; i++)
            {
                double energy = audio[i] * audio[i];
                weightedSum += i * energy;
                energySum += energy;
            }
            
            return energySum > 0 ? weightedSum / energySum : 0;
        }

        /// <summary>
        /// Applies window function to audio segment
        /// </summary>
        private double[] ApplyWindow(double[] audio, int start, int length)
        {
            var windowed = new double[length];
            for (int i = 0; i < length && start + i < audio.Length; i++)
            {
                // Hamming window
                double window = 0.54 - 0.46 * Math.Cos(2 * Math.PI * i / (length - 1));
                windowed[i] = audio[start + i] * window;
            }
            return windowed;
        }

        /// <summary>
        /// Computes magnitude spectrum (simplified DFT)
        /// </summary>
        private double[] ComputeSpectrum(double[] audio)
        {
            int fftSize = _windowSize / 2 + 1;
            var spectrum = new double[fftSize];
            
            // Simplified DFT (would use FFT in real implementation)
            for (int k = 0; k < fftSize; k++)
            {
                double real = 0, imag = 0;
                for (int n = 0; n < audio.Length; n++)
                {
                    double angle = -2 * Math.PI * k * n / audio.Length;
                    real += audio[n] * Math.Cos(angle);
                    imag += audio[n] * Math.Sin(angle);
                }
                spectrum[k] = Math.Sqrt(real * real + imag * imag);
            }
            
            return spectrum;
        }

        /// <summary>
        /// Calculates spectral centroid
        /// </summary>
        private double CalculateSpectralCentroid(double[] spectrum)
        {
            double weightedSum = 0;
            double magnitudeSum = 0;
            
            for (int i = 0; i < spectrum.Length; i++)
            {
                weightedSum += i * spectrum[i];
                magnitudeSum += spectrum[i];
            }
            
            return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
        }

        /// <summary>
        /// Calculates spectral spread
        /// </summary>
        private double CalculateSpectralSpread(double[] spectrum)
        {
            double centroid = CalculateSpectralCentroid(spectrum);
            double weightedVariance = 0;
            double magnitudeSum = 0;
            
            for (int i = 0; i < spectrum.Length; i++)
            {
                double diff = i - centroid;
                weightedVariance += diff * diff * spectrum[i];
                magnitudeSum += spectrum[i];
            }
            
            return magnitudeSum > 0 ? Math.Sqrt(weightedVariance / magnitudeSum) : 0;
        }

        /// <summary>
        /// Calculates spectral flux
        /// </summary>
        private double CalculateSpectralFlux(double[] spectrum, double[] prevSpectrum)
        {
            double flux = 0;
            for (int i = 0; i < Math.Min(spectrum.Length, prevSpectrum.Length); i++)
            {
                double diff = spectrum[i] - prevSpectrum[i];
                if (diff > 0)
                    flux += diff * diff;
            }
            return Math.Sqrt(flux);
        }

        /// <summary>
        /// Calculates spectral rolloff
        /// </summary>
        private double CalculateSpectralRolloff(double[] spectrum, double threshold)
        {
            double totalEnergy = spectrum.Sum();
            double cumulativeEnergy = 0;
            
            for (int i = 0; i < spectrum.Length; i++)
            {
                cumulativeEnergy += spectrum[i];
                if (cumulativeEnergy >= threshold * totalEnergy)
                    return i;
            }
            
            return spectrum.Length - 1;
        }

        /// <summary>
        /// Calculates spectral entropy
        /// </summary>
        private double CalculateSpectralEntropy(double[] spectrum)
        {
            double totalEnergy = spectrum.Sum();
            if (totalEnergy == 0) return 0;
            
            double entropy = 0;
            for (int i = 0; i < spectrum.Length; i++)
            {
                if (spectrum[i] > 0)
                {
                    double probability = spectrum[i] / totalEnergy;
                    entropy -= probability * Math.Log(probability, 2);
                }
            }
            
            return entropy;
        }

        /// <summary>
        /// Converts spectrum to mel scale (simplified)
        /// </summary>
        private double[] ConvertToMelScale(double[] spectrum)
        {
            int numMelBins = 40;
            var melSpectrum = new double[numMelBins];
            
            for (int i = 0; i < numMelBins; i++)
            {
                // Simplified mel filterbank
                int start = i * spectrum.Length / numMelBins;
                int end = Math.Min((i + 1) * spectrum.Length / numMelBins, spectrum.Length);
                
                for (int j = start; j < end; j++)
                {
                    melSpectrum[i] += spectrum[j];
                }
            }
            
            return melSpectrum;
        }

        /// <summary>
        /// Applies DCT for MFCC calculation (simplified)
        /// </summary>
        private double[] ApplyDCT(double[] input, int numCoefficients)
        {
            var dct = new double[numCoefficients];
            
            for (int k = 0; k < numCoefficients; k++)
            {
                for (int n = 0; n < input.Length; n++)
                {
                    dct[k] += input[n] * Math.Cos(Math.PI * k * (n + 0.5) / input.Length);
                }
                dct[k] *= Math.Sqrt(2.0 / input.Length);
            }
            
            return dct;
        }

        /// <summary>
        /// Normalizes audio to [-1, 1]
        /// </summary>
        private double[] NormalizeAudio(double[] audio)
        {
            double max = audio.Max(Math.Abs);
            if (max > 0)
            {
                return audio.Select(x => x / max).ToArray();
            }
            return audio;
        }

        /// <summary>
        /// Applies pre-emphasis filter
        /// </summary>
        private double[] ApplyPreEmphasis(double[] audio, double coefficient)
        {
            var filtered = new double[audio.Length];
            filtered[0] = audio[0];
            
            for (int i = 1; i < audio.Length; i++)
            {
                filtered[i] = audio[i] - coefficient * audio[i - 1];
            }
            
            return filtered;
        }

        /// <summary>
        /// Converts to double array
        /// </summary>
        private double[] ConvertToDoubleArray(object input)
        {
            return input switch
            {
                double[] arr => arr,
                float[] arr => arr.Select(x => (double)x).ToArray(),
                _ => throw new ArgumentException("Cannot convert input to double array")
            };
        }

        /// <summary>
        /// Gets the total feature size
        /// </summary>
        private int GetFeatureSize()
        {
            int size = 6; // Temporal features
            
            int numFrames = 10; // Approximate number of frames
            
            if (_useSpectralFeatures)
                size += 5 * numFrames; // 5 spectral features per frame
            
            if (_useMfcc)
                size += _numMfccCoefficients * numFrames;
            
            return size;
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