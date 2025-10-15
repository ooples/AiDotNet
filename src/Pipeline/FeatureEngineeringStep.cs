using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Generic feature engineering pipeline step
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class FeatureEngineeringStep<T> : PipelineStepBase<T>
    {
        private readonly FeatureEngineeringConfig<T> config;
        private List<Func<Vector<T>, T>>? generatedFeatures;
        private Dictionary<string, object> featureMetadata;
        private int originalFeatureCount;
        
        public FeatureEngineeringStep(FeatureEngineeringConfig<T> config) 
            : base("FeatureEngineering", MathHelper.GetNumericOperations<T>())
        {
            this.config = config ?? throw new ArgumentNullException(nameof(config));
            this.featureMetadata = new Dictionary<string, object>();
            
            IsCacheable = true;
            SupportsParallelExecution = true;
        }
        
        protected override void FitCore(Matrix<T> inputs, Vector<T>? targets)
        {
            originalFeatureCount = inputs.Columns;
            generatedFeatures = new List<Func<Vector<T>, T>>();
            
            if (config.AutoGenerate)
            {
                GenerateAutomaticFeatures(inputs, targets);
            }
            
            if (config.GeneratePolynomialFeatures)
            {
                GeneratePolynomialFeatures();
            }
            
            if (config.GenerateInteractionFeatures)
            {
                GenerateInteractionFeatures();
            }
            
            if (config.CustomFeatureGenerators != null)
            {
                generatedFeatures.AddRange(config.CustomFeatureGenerators);
            }
            
            UpdateMetadata("OriginalFeatures", originalFeatureCount.ToString());
            UpdateMetadata("GeneratedFeatures", generatedFeatures.Count.ToString());
            UpdateMetadata("TotalFeatures", (originalFeatureCount + generatedFeatures.Count).ToString());
        }
        
        protected override Matrix<T> TransformCore(Matrix<T> inputs)
        {
            if (generatedFeatures == null || generatedFeatures.Count == 0)
            {
                return inputs;
            }
            
            // Create new matrix with additional columns for generated features
            var newColumns = inputs.Columns + generatedFeatures.Count;
            var transformed = new Matrix<T>(inputs.Rows, newColumns);
            
            // Copy original features
            for (int i = 0; i < inputs.Rows; i++)
            {
                for (int j = 0; j < inputs.Columns; j++)
                {
                    transformed[i, j] = inputs[i, j];
                }
            }
            
            // Add generated features
            for (int i = 0; i < inputs.Rows; i++)
            {
                var row = inputs.GetRow(i);
                
                for (int j = 0; j < generatedFeatures.Count; j++)
                {
                    try
                    {
                        transformed[i, inputs.Columns + j] = generatedFeatures[j](row);
                    }
                    catch
                    {
                        // If feature generation fails, use zero
                        transformed[i, inputs.Columns + j] = NumOps.Zero;
                    }
                }
            }
            
            return transformed;
        }
        
        private void GenerateAutomaticFeatures(Matrix<T> inputs, Vector<T>? targets)
        {
            // Analyze feature statistics to generate relevant features
            for (int i = 0; i < originalFeatureCount; i++)
            {
                var columnData = inputs.GetColumn(i);
                var stats = CalculateColumnStatistics(columnData);
                
                // Generate log transform for positive skewed features
                if (stats.Skewness > 1.0 && stats.Min > 0)
                {
                    int featureIndex = i;
                    generatedFeatures?.Add(row => 
                    {
                        if (row == null) throw new ArgumentNullException(nameof(row));
                        var value = Convert.ToDouble(row[featureIndex]);
                        return (T)Convert.ChangeType(Math.Log(value + 1), typeof(T));
                    });
                    featureMetadata[$"log_{i}"] = "Log transform";
                }
                
                // Generate square root for features with high variance
                if (stats.Variance > 10 * stats.Mean && stats.Min >= 0)
                {
                    int featureIndex = i;
                    generatedFeatures?.Add(row => 
                    {
                        if (row == null) throw new ArgumentNullException(nameof(row));
                        var value = Convert.ToDouble(row[featureIndex]);
                        return (T)Convert.ChangeType(Math.Sqrt(value), typeof(T));
                    });
                    featureMetadata[$"sqrt_{i}"] = "Square root transform";
                }
                
                // Generate reciprocal for features not close to zero
                if (Math.Abs(stats.Min) > 0.1)
                {
                    int featureIndex = i;
                    generatedFeatures?.Add(row => 
                    {
                        if (row == null) throw new ArgumentNullException(nameof(row));
                        var value = Convert.ToDouble(row[featureIndex]);
                        return (T)Convert.ChangeType(1.0 / (value + Math.Sign(value) * 0.1), typeof(T));
                    });
                    featureMetadata[$"reciprocal_{i}"] = "Reciprocal transform";
                }
            }
        }
        
        private void GeneratePolynomialFeatures()
        {
            for (int i = 0; i < originalFeatureCount; i++)
            {
                int featureIndex = i;
                
                // Square terms
                generatedFeatures?.Add(row => 
                {
                    if (row == null) throw new ArgumentNullException(nameof(row));
                    return NumOps.Multiply(row[featureIndex], row[featureIndex]);
                });
                featureMetadata[$"poly2_{i}"] = "Polynomial degree 2";
                
                if (config.PolynomialDegree >= 3)
                {
                    // Cubic terms
                    generatedFeatures?.Add(row => 
                    {
                        if (row == null) throw new ArgumentNullException(nameof(row));
                        var squared = NumOps.Multiply(row[featureIndex], row[featureIndex]);
                        return NumOps.Multiply(squared, row[featureIndex]);
                    });
                    featureMetadata[$"poly3_{i}"] = "Polynomial degree 3";
                }
            }
        }
        
        private void GenerateInteractionFeatures()
        {
            // Generate pairwise interactions for top features
            var maxInteractions = Math.Min(config.MaxInteractionFeatures, (originalFeatureCount * (originalFeatureCount - 1)) / 2);
            var interactionCount = 0;
            
            for (int i = 0; i < originalFeatureCount - 1 && interactionCount < maxInteractions; i++)
            {
                for (int j = i + 1; j < originalFeatureCount && interactionCount < maxInteractions; j++)
                {
                    int feat1 = i;
                    int feat2 = j;
                    
                    generatedFeatures?.Add(row => 
                    {
                        if (row == null) throw new ArgumentNullException(nameof(row));
                        return NumOps.Multiply(row[feat1], row[feat2]);
                    });
                    featureMetadata[$"interaction_{i}_{j}"] = $"Interaction between features {i} and {j}";
                    interactionCount++;
                }
            }
        }
        
        private ColumnStatistics CalculateColumnStatistics(Vector<T> column)
        {
            var values = column.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
            
            var mean = values.Average();
            var variance = values.Select(v => Math.Pow(v - mean, 2)).Average();
            var stdDev = Math.Sqrt(variance);
            var min = values.Min();
            var max = values.Max();
            
            // Calculate skewness
            var skewness = 0.0;
            if (stdDev > 0)
            {
                var n = values.Length;
                var sum = values.Select(v => Math.Pow((v - mean) / stdDev, 3)).Sum();
                skewness = sum * n / ((n - 1) * (n - 2));
            }
            
            return new ColumnStatistics
            {
                Mean = mean,
                Variance = variance,
                StandardDeviation = stdDev,
                Min = min,
                Max = max,
                Skewness = skewness
            };
        }
        
        private class ColumnStatistics
        {
            public double Mean { get; set; }
            public double Variance { get; set; }
            public double StandardDeviation { get; set; }
            public double Min { get; set; }
            public double Max { get; set; }
            public double Skewness { get; set; }
        }
    }
    
    /// <summary>
    /// Configuration for feature engineering
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class FeatureEngineeringConfig<T>
    {
        /// <summary>
        /// Whether to automatically generate features based on data characteristics
        /// </summary>
        public bool AutoGenerate { get; set; } = true;
        
        /// <summary>
        /// Whether to generate polynomial features
        /// </summary>
        public bool GeneratePolynomialFeatures { get; set; } = true;
        
        /// <summary>
        /// Maximum polynomial degree
        /// </summary>
        public int PolynomialDegree { get; set; } = 2;
        
        /// <summary>
        /// Whether to generate interaction features
        /// </summary>
        public bool GenerateInteractionFeatures { get; set; } = true;
        
        /// <summary>
        /// Maximum number of interaction features to generate
        /// </summary>
        public int MaxInteractionFeatures { get; set; } = 10;
        
        /// <summary>
        /// Custom feature generators
        /// </summary>
        public List<Func<Vector<T>, T>>? CustomFeatureGenerators { get; set; }
    }
}