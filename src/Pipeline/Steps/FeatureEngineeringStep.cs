using AiDotNet.Enums;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Pipeline.Steps
{
    /// <summary>
    /// Pipeline step for feature engineering and transformation
    /// </summary>
    public class FeatureEngineeringStep : PipelineStepBase
    {
        private readonly List<FeatureTransformation> _transformations = default!;
        private Dictionary<int, double[]>? _polynomialCoefficients = default!;
        private Dictionary<string, int>? _interactionFeatureMap = default!;
        private double[]? _logOffsets;
        private double[]? _powerExponents;
        private int _originalFeatureCount;

        /// <summary>
        /// Enum for feature transformation types
        /// </summary>
        public enum FeatureTransformationType
        {
            Polynomial,
            Logarithmic,
            Exponential,
            SquareRoot,
            Reciprocal,
            Absolute,
            Power,
            Trigonometric,
            Binning,
            Interaction,
            Difference,
            Ratio,
            Custom
        }

        /// <summary>
        /// Represents a feature transformation
        /// </summary>
        public class FeatureTransformation
        {
            public FeatureTransformationType Type { get; set; }
            public int[]? FeatureIndices { get; set; }
            public Dictionary<string, object>? Parameters { get; set; }
            public Func<double[], double>? CustomTransform { get; set; }
        }

        /// <summary>
        /// Gets the list of transformations to apply
        /// </summary>
        public IReadOnlyList<FeatureTransformation> Transformations => _transformations.AsReadOnly();

        /// <summary>
        /// Initializes a new instance of the FeatureEngineeringStep class
        /// </summary>
        /// <param name="name">Optional name for this step</param>
        public FeatureEngineeringStep(string? name = null) : base(name ?? "FeatureEngineering")
        {
            Position = PipelinePosition.FeatureEngineering;
            _transformations = new List<FeatureTransformation>();
            SupportsParallelExecution = true;
        }

        /// <summary>
        /// Adds a polynomial transformation
        /// </summary>
        /// <param name="featureIndices">Indices of features to transform</param>
        /// <param name="degree">Polynomial degree</param>
        /// <param name="includeInteractions">Whether to include interaction terms</param>
        public void AddPolynomialFeatures(int[] featureIndices, int degree = 2, bool includeInteractions = false)
        {
            _transformations.Add(new FeatureTransformation
            {
                Type = FeatureTransformationType.Polynomial,
                FeatureIndices = featureIndices,
                Parameters = new Dictionary<string, object>
                {
                    ["Degree"] = degree,
                    ["IncludeInteractions"] = includeInteractions
                }
            });
        }

        /// <summary>
        /// Adds a logarithmic transformation
        /// </summary>
        /// <param name="featureIndices">Indices of features to transform</param>
        /// <param name="offset">Offset to add before log transformation</param>
        public void AddLogTransform(int[] featureIndices, double offset = 1e-8)
        {
            _transformations.Add(new FeatureTransformation
            {
                Type = FeatureTransformationType.Logarithmic,
                FeatureIndices = featureIndices,
                Parameters = new Dictionary<string, object> { ["Offset"] = offset }
            });
        }

        /// <summary>
        /// Adds interaction features between specified feature pairs
        /// </summary>
        /// <param name="featurePairs">Pairs of feature indices to create interactions</param>
        public void AddInteractionFeatures(List<(int, int)> featurePairs)
        {
            _transformations.Add(new FeatureTransformation
            {
                Type = FeatureTransformationType.Interaction,
                Parameters = new Dictionary<string, object> { ["FeaturePairs"] = featurePairs }
            });
        }

        /// <summary>
        /// Adds a custom transformation
        /// </summary>
        /// <param name="featureIndices">Indices of features to transform</param>
        /// <param name="transform">Custom transformation function</param>
        /// <param name="name">Name of the custom transformation</param>
        public void AddCustomTransform(int[] featureIndices, Func<double[], double> transform, string name)
        {
            _transformations.Add(new FeatureTransformation
            {
                Type = FeatureTransformationType.Custom,
                FeatureIndices = featureIndices,
                CustomTransform = transform,
                Parameters = new Dictionary<string, object> { ["Name"] = name }
            });
        }

        /// <summary>
        /// Adds binning transformation
        /// </summary>
        /// <param name="featureIndex">Index of feature to bin</param>
        /// <param name="numberOfBins">Number of bins</param>
        /// <param name="strategy">Binning strategy (uniform, quantile)</param>
        public void AddBinning(int featureIndex, int numberOfBins, string strategy = "uniform")
        {
            _transformations.Add(new FeatureTransformation
            {
                Type = FeatureTransformationType.Binning,
                FeatureIndices = new[] { featureIndex },
                Parameters = new Dictionary<string, object>
                {
                    ["NumberOfBins"] = numberOfBins,
                    ["Strategy"] = strategy
                }
            });
        }

        /// <summary>
        /// Core fitting logic that prepares transformations
        /// </summary>
        protected override void FitCore(double[][] inputs, double[]? targets)
        {
            _originalFeatureCount = inputs[0].Length;
            _polynomialCoefficients = new Dictionary<int, double[]>();
            _interactionFeatureMap = new Dictionary<string, int>();
            _logOffsets = new double[_originalFeatureCount];
            _powerExponents = new double[_originalFeatureCount];

            // Prepare transformation parameters based on data
            foreach (var transformation in _transformations)
            {
                switch (transformation.Type)
                {
                    case FeatureTransformationType.Logarithmic:
                        if (transformation.FeatureIndices != null)
                        {
                            foreach (var idx in transformation.FeatureIndices)
                            {
                                // Calculate appropriate offset for log transform
                                var minValue = inputs.Select(row => row[idx]).Min();
                                var offset = transformation.Parameters?["Offset"] as double? ?? 1e-8;
                                _logOffsets[idx] = minValue <= 0 ? Math.Abs(minValue) + offset : offset;
                            }
                        }
                        break;

                    case FeatureTransformationType.Binning:
                        if (transformation.FeatureIndices != null && transformation.Parameters != null)
                        {
                            var featureIndex = transformation.FeatureIndices[0];
                            var numberOfBins = (int)transformation.Parameters["NumberOfBins"];
                            var strategy = transformation.Parameters["Strategy"] as string;

                            var values = inputs.Select(row => row[featureIndex]).OrderBy(v => v).ToArray();
                            var binEdges = CalculateBinEdges(values, numberOfBins, strategy);
                            transformation.Parameters["BinEdges"] = binEdges;
                        }
                        break;
                }
            }

            UpdateMetadata("OriginalFeatureCount", _originalFeatureCount.ToString());
            UpdateMetadata("TransformationCount", _transformations.Count.ToString());
        }

        /// <summary>
        /// Core transformation logic that applies feature engineering
        /// </summary>
        protected override double[][] TransformCore(double[][] inputs)
        {
            var transformedFeatures = new List<double[]>();

            // Start with original features
            for (int j = 0; j < _originalFeatureCount; j++)
            {
                transformedFeatures.Add(inputs.Select(row => row[j]).ToArray());
            }

            // Apply each transformation
            foreach (var transformation in _transformations)
            {
                var newFeatures = ApplyTransformation(inputs, transformation);
                transformedFeatures.AddRange(newFeatures);
            }

            // Transpose back to row-major format
            var result = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                result[i] = new double[transformedFeatures.Count];
                for (int j = 0; j < transformedFeatures.Count; j++)
                {
                    result[i][j] = transformedFeatures[j][i];
                }
            }

            UpdateMetadata("OutputFeatureCount", transformedFeatures.Count.ToString());
            return result;
        }

        /// <summary>
        /// Applies a specific transformation
        /// </summary>
        private List<double[]> ApplyTransformation(double[][] inputs, FeatureTransformation transformation)
        {
            var newFeatures = new List<double[]>();

            switch (transformation.Type)
            {
                case FeatureTransformationType.Polynomial:
                    newFeatures.AddRange(ApplyPolynomialTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.Logarithmic:
                    newFeatures.AddRange(ApplyLogTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.Exponential:
                    newFeatures.AddRange(ApplyExponentialTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.SquareRoot:
                    newFeatures.AddRange(ApplySqrtTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.Reciprocal:
                    newFeatures.AddRange(ApplyReciprocalTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.Absolute:
                    newFeatures.AddRange(ApplyAbsoluteTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.Trigonometric:
                    newFeatures.AddRange(ApplyTrigonometricTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.Interaction:
                    newFeatures.AddRange(ApplyInteractionTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.Binning:
                    newFeatures.AddRange(ApplyBinningTransform(inputs, transformation));
                    break;

                case FeatureTransformationType.Custom:
                    newFeatures.AddRange(ApplyCustomTransform(inputs, transformation));
                    break;
            }

            return newFeatures;
        }

        /// <summary>
        /// Applies polynomial transformation
        /// </summary>
        private List<double[]> ApplyPolynomialTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();
            var degree = (int)(transformation.Parameters?["Degree"] ?? 2);
            var includeInteractions = (bool)(transformation.Parameters?["IncludeInteractions"] ?? false);

            if (transformation.FeatureIndices != null)
            {
                foreach (var idx in transformation.FeatureIndices)
                {
                    for (int d = 2; d <= degree; d++)
                    {
                        var feature = new double[inputs.Length];
                        for (int i = 0; i < inputs.Length; i++)
                        {
                            feature[i] = Math.Pow(inputs[i][idx], d);
                        }
                        features.Add(feature);
                    }
                }

                if (includeInteractions && transformation.FeatureIndices.Length > 1)
                {
                    // Add interaction terms
                    for (int j = 0; j < transformation.FeatureIndices.Length - 1; j++)
                    {
                        for (int k = j + 1; k < transformation.FeatureIndices.Length; k++)
                        {
                            var feature = new double[inputs.Length];
                            for (int i = 0; i < inputs.Length; i++)
                            {
                                feature[i] = inputs[i][transformation.FeatureIndices[j]] * 
                                           inputs[i][transformation.FeatureIndices[k]];
                            }
                            features.Add(feature);
                        }
                    }
                }
            }

            return features;
        }

        /// <summary>
        /// Applies logarithmic transformation
        /// </summary>
        private List<double[]> ApplyLogTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();

            if (transformation.FeatureIndices != null && _logOffsets != null)
            {
                foreach (var idx in transformation.FeatureIndices)
                {
                    var feature = new double[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        feature[i] = Math.Log(inputs[i][idx] + _logOffsets[idx]);
                    }
                    features.Add(feature);
                }
            }

            return features;
        }

        /// <summary>
        /// Applies exponential transformation
        /// </summary>
        private List<double[]> ApplyExponentialTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();

            if (transformation.FeatureIndices != null)
            {
                foreach (var idx in transformation.FeatureIndices)
                {
                    var feature = new double[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        // Clip to prevent overflow
                        feature[i] = Math.Exp(Math.Min(inputs[i][idx], 700));
                    }
                    features.Add(feature);
                }
            }

            return features;
        }

        /// <summary>
        /// Applies square root transformation
        /// </summary>
        private List<double[]> ApplySqrtTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();

            if (transformation.FeatureIndices != null)
            {
                foreach (var idx in transformation.FeatureIndices)
                {
                    var feature = new double[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        feature[i] = Math.Sqrt(Math.Abs(inputs[i][idx]));
                    }
                    features.Add(feature);
                }
            }

            return features;
        }

        /// <summary>
        /// Applies reciprocal transformation
        /// </summary>
        private List<double[]> ApplyReciprocalTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();
            var epsilon = transformation.Parameters?["Epsilon"] as double? ?? 1e-8;

            if (transformation.FeatureIndices != null)
            {
                foreach (var idx in transformation.FeatureIndices)
                {
                    var feature = new double[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        feature[i] = 1.0 / (inputs[i][idx] + epsilon);
                    }
                    features.Add(feature);
                }
            }

            return features;
        }

        /// <summary>
        /// Applies absolute value transformation
        /// </summary>
        private List<double[]> ApplyAbsoluteTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();

            if (transformation.FeatureIndices != null)
            {
                foreach (var idx in transformation.FeatureIndices)
                {
                    var feature = new double[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        feature[i] = Math.Abs(inputs[i][idx]);
                    }
                    features.Add(feature);
                }
            }

            return features;
        }

        /// <summary>
        /// Applies trigonometric transformations
        /// </summary>
        private List<double[]> ApplyTrigonometricTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();
            var functions = transformation.Parameters?["Functions"] as string[] ?? new[] { "sin", "cos" };

            if (transformation.FeatureIndices != null)
            {
                foreach (var idx in transformation.FeatureIndices)
                {
                    foreach (var func in functions)
                    {
                        var feature = new double[inputs.Length];
                        for (int i = 0; i < inputs.Length; i++)
                        {
                            feature[i] = func.ToLower() switch
                            {
                                "sin" => Math.Sin(inputs[i][idx]),
                                "cos" => Math.Cos(inputs[i][idx]),
                                "tan" => Math.Tan(inputs[i][idx]),
                                _ => inputs[i][idx]
                            };
                        }
                        features.Add(feature);
                    }
                }
            }

            return features;
        }

        /// <summary>
        /// Applies interaction transformation
        /// </summary>
        private List<double[]> ApplyInteractionTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();
            var featurePairs = transformation.Parameters?["FeaturePairs"] as List<(int, int)>;

            if (featurePairs != null)
            {
                foreach (var (idx1, idx2) in featurePairs)
                {
                    var feature = new double[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        feature[i] = inputs[i][idx1] * inputs[i][idx2];
                    }
                    features.Add(feature);
                }
            }

            return features;
        }

        /// <summary>
        /// Applies binning transformation
        /// </summary>
        private List<double[]> ApplyBinningTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();

            if (transformation.FeatureIndices != null && transformation.Parameters != null)
            {
                var binEdges = transformation.Parameters["BinEdges"] as double[];
                if (binEdges != null)
                {
                    foreach (var idx in transformation.FeatureIndices)
                    {
                        var feature = new double[inputs.Length];
                        for (int i = 0; i < inputs.Length; i++)
                        {
                            feature[i] = GetBinIndex(inputs[i][idx], binEdges);
                        }
                        features.Add(feature);
                    }
                }
            }

            return features;
        }

        /// <summary>
        /// Applies custom transformation
        /// </summary>
        private List<double[]> ApplyCustomTransform(double[][] inputs, FeatureTransformation transformation)
        {
            var features = new List<double[]>();

            if (transformation.CustomTransform != null && transformation.FeatureIndices != null)
            {
                var feature = new double[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    var values = transformation.FeatureIndices.Select(idx => inputs[i][idx]).ToArray();
                    feature[i] = transformation.CustomTransform(values);
                }
                features.Add(feature);
            }

            return features;
        }

        /// <summary>
        /// Calculates bin edges for binning transformation
        /// </summary>
        private double[] CalculateBinEdges(double[] sortedValues, int numberOfBins, string? strategy)
        {
            var edges = new double[numberOfBins + 1];

            if (strategy == "quantile")
            {
                // Quantile-based binning
                for (int i = 0; i <= numberOfBins; i++)
                {
                    var percentile = (double)i / numberOfBins;
                    var index = (int)(percentile * (sortedValues.Length - 1));
                    edges[i] = sortedValues[index];
                }
            }
            else // uniform
            {
                // Uniform binning
                var min = sortedValues[0];
                var max = sortedValues[sortedValues.Length - 1];
                var width = (max - min) / numberOfBins;

                for (int i = 0; i <= numberOfBins; i++)
                {
                    edges[i] = min + i * width;
                }
            }

            return edges;
        }

        /// <summary>
        /// Gets the bin index for a value
        /// </summary>
        private int GetBinIndex(double value, double[] binEdges)
        {
            for (int i = 0; i < binEdges.Length - 1; i++)
            {
                if (value >= binEdges[i] && value < binEdges[i + 1])
                {
                    return i;
                }
            }
            return binEdges.Length - 2; // Last bin
        }

        /// <summary>
        /// Indicates whether this step requires fitting before transformation
        /// </summary>
        protected override bool RequiresFitting()
        {
            // Some transformations need fitting (e.g., binning), others don't
            return _transformations.Any(t => 
                t.Type == FeatureTransformationType.Binning ||
                t.Type == FeatureTransformationType.Logarithmic);
        }
    }
}