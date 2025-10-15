using AiDotNet.Enums;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Pipeline.Steps
{
    /// <summary>
    /// Pipeline step for validating data quality and integrity
    /// </summary>
    public class DataValidationStep : PipelineStepBase
    {
        private readonly List<string> _validationErrors = default!;
        private readonly List<string> _validationWarnings = default!;
        private double _missingValueThreshold;
        private double _outlierThreshold;
        private bool _checkForDuplicates;
        private bool _checkForConstantFeatures;
        private bool _checkForInfiniteValues;
        private bool _checkForNaNValues;
        private double[]? _featureMeans;
        private double[]? _featureStdDevs;
        private HashSet<int>? _invalidFeatureIndices;

        /// <summary>
        /// Gets the validation errors found during the last validation
        /// </summary>
        public IReadOnlyList<string> ValidationErrors => _validationErrors.AsReadOnly();

        /// <summary>
        /// Gets the validation warnings found during the last validation
        /// </summary>
        public IReadOnlyList<string> ValidationWarnings => _validationWarnings.AsReadOnly();

        /// <summary>
        /// Gets whether the last validation passed
        /// </summary>
        public bool IsValid => _validationErrors.Count == 0;

        /// <summary>
        /// Initializes a new instance of the DataValidationStep class
        /// </summary>
        /// <param name="name">Optional name for this step</param>
        public DataValidationStep(string? name = null) : base(name ?? "DataValidation")
        {
            Position = PipelinePosition.Preprocessing;
            _validationErrors = new List<string>();
            _validationWarnings = new List<string>();
            _missingValueThreshold = 0.1; // 10% default
            _outlierThreshold = 3.0; // 3 standard deviations
            _checkForDuplicates = true;
            _checkForConstantFeatures = true;
            _checkForInfiniteValues = true;
            _checkForNaNValues = true;

            // Set default parameters
            SetParameter("MissingValueThreshold", _missingValueThreshold);
            SetParameter("OutlierThreshold", _outlierThreshold);
            SetParameter("CheckForDuplicates", _checkForDuplicates);
            SetParameter("CheckForConstantFeatures", _checkForConstantFeatures);
            SetParameter("CheckForInfiniteValues", _checkForInfiniteValues);
            SetParameter("CheckForNaNValues", _checkForNaNValues);
        }

        /// <summary>
        /// Core fitting logic that analyzes the data for validation
        /// </summary>
        protected override void FitCore(double[][] inputs, double[]? targets)
        {
            _validationErrors.Clear();
            _validationWarnings.Clear();
            _invalidFeatureIndices = new HashSet<int>();

            int numSamples = inputs.Length;
            int numFeatures = inputs[0].Length;

            // Calculate feature statistics
            _featureMeans = new double[numFeatures];
            _featureStdDevs = new double[numFeatures];

            for (int j = 0; j < numFeatures; j++)
            {
                var featureValues = inputs.Select(row => row[j]).ToArray();
                _featureMeans[j] = StatisticsHelper<double>.Mean(featureValues);
                _featureStdDevs[j] = StatisticsHelper<double>.StandardDeviation(featureValues);
            }

            // Perform validation checks
            ValidateMissingValues(inputs);
            ValidateInfiniteAndNaNValues(inputs);
            ValidateConstantFeatures(inputs);
            ValidateDuplicates(inputs);
            ValidateOutliers(inputs);
            ValidateTargets(targets);

            // Update metadata
            UpdateMetadata("TotalErrors", _validationErrors.Count.ToString());
            UpdateMetadata("TotalWarnings", _validationWarnings.Count.ToString());
            UpdateMetadata("InvalidFeatures", string.Join(",", _invalidFeatureIndices));
        }

        /// <summary>
        /// Core transformation logic that applies validation
        /// </summary>
        protected override double[][] TransformCore(double[][] inputs)
        {
            _validationErrors.Clear();
            _validationWarnings.Clear();

            // Validate the input data
            ValidateTransformData(inputs);

            if (!IsValid)
            {
                throw new InvalidOperationException($"Data validation failed with {_validationErrors.Count} errors. " +
                    $"First error: {_validationErrors.FirstOrDefault()}");
            }

            // Return the data unchanged (validation step doesn't modify data)
            return inputs;
        }

        /// <summary>
        /// Validates missing values in the data
        /// </summary>
        private void ValidateMissingValues(double[][] inputs)
        {
            if (!_checkForNaNValues) return;

            int numFeatures = inputs[0].Length;
            for (int j = 0; j < numFeatures; j++)
            {
                int missingCount = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    if (double.IsNaN(inputs[i][j]))
                    {
                        missingCount++;
                    }
                }

                double missingRatio = (double)missingCount / inputs.Length;
                if (missingRatio > _missingValueThreshold)
                {
                    _validationErrors.Add($"Feature {j} has {missingRatio:P1} missing values, " +
                        $"exceeding threshold of {_missingValueThreshold:P1}");
                    _invalidFeatureIndices?.Add(j);
                }
                else if (missingCount > 0)
                {
                    _validationWarnings.Add($"Feature {j} has {missingCount} missing values ({missingRatio:P1})");
                }
            }
        }

        /// <summary>
        /// Validates for infinite and NaN values
        /// </summary>
        private void ValidateInfiniteAndNaNValues(double[][] inputs)
        {
            if (!_checkForInfiniteValues && !_checkForNaNValues) return;

            for (int i = 0; i < inputs.Length; i++)
            {
                for (int j = 0; j < inputs[i].Length; j++)
                {
                    if (_checkForInfiniteValues && double.IsInfinity(inputs[i][j]))
                    {
                        _validationErrors.Add($"Infinite value found at sample {i}, feature {j}");
                        _invalidFeatureIndices?.Add(j);
                    }

                    if (_checkForNaNValues && double.IsNaN(inputs[i][j]) && !_checkForNaNValues)
                    {
                        _validationErrors.Add($"NaN value found at sample {i}, feature {j}");
                        _invalidFeatureIndices?.Add(j);
                    }
                }
            }
        }

        /// <summary>
        /// Validates for constant features
        /// </summary>
        private void ValidateConstantFeatures(double[][] inputs)
        {
            if (!_checkForConstantFeatures || _featureStdDevs == null) return;

            for (int j = 0; j < _featureStdDevs.Length; j++)
            {
                if (Math.Abs(_featureStdDevs[j]) < 1e-10)
                {
                    _validationWarnings.Add($"Feature {j} has zero variance (constant feature)");
                    _invalidFeatureIndices?.Add(j);
                }
            }
        }

        /// <summary>
        /// Validates for duplicate samples
        /// </summary>
        private void ValidateDuplicates(double[][] inputs)
        {
            if (!_checkForDuplicates) return;

            var uniqueSamples = new HashSet<string>();
            int duplicateCount = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                var sampleKey = string.Join(",", inputs[i].Select(v => v.ToString("G17")));
                if (!uniqueSamples.Add(sampleKey))
                {
                    duplicateCount++;
                }
            }

            if (duplicateCount > 0)
            {
                double duplicateRatio = (double)duplicateCount / inputs.Length;
                if (duplicateRatio > 0.1) // More than 10% duplicates
                {
                    _validationWarnings.Add($"Found {duplicateCount} duplicate samples ({duplicateRatio:P1})");
                }
            }
        }

        /// <summary>
        /// Validates for outliers
        /// </summary>
        private void ValidateOutliers(double[][] inputs)
        {
            if (_featureMeans == null || _featureStdDevs == null) return;

            int totalOutliers = 0;
            for (int j = 0; j < inputs[0].Length; j++)
            {
                if (Math.Abs(_featureStdDevs[j]) < 1e-10) continue; // Skip constant features

                int featureOutliers = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    double zScore = Math.Abs((inputs[i][j] - _featureMeans[j]) / _featureStdDevs[j]);
                    if (zScore > _outlierThreshold)
                    {
                        featureOutliers++;
                        totalOutliers++;
                    }
                }

                if (featureOutliers > 0)
                {
                    double outlierRatio = (double)featureOutliers / inputs.Length;
                    if (outlierRatio > 0.05) // More than 5% outliers in a feature
                    {
                        _validationWarnings.Add($"Feature {j} has {featureOutliers} outliers ({outlierRatio:P1})");
                    }
                }
            }

            if (totalOutliers > 0)
            {
                UpdateMetadata("TotalOutliers", totalOutliers.ToString());
            }
        }

        /// <summary>
        /// Validates target values if provided
        /// </summary>
        private void ValidateTargets(double[]? targets)
        {
            if (targets == null) return;

            int nanCount = 0;
            int infCount = 0;

            for (int i = 0; i < targets.Length; i++)
            {
                if (double.IsNaN(targets[i])) nanCount++;
                if (double.IsInfinity(targets[i])) infCount++;
            }

            if (nanCount > 0)
            {
                _validationErrors.Add($"Target contains {nanCount} NaN values");
            }

            if (infCount > 0)
            {
                _validationErrors.Add($"Target contains {infCount} infinite values");
            }

            // Check for constant target
            var targetStdDev = StatisticsHelper<double>.StandardDeviation(targets);
            if (Math.Abs(targetStdDev) < 1e-10)
            {
                _validationWarnings.Add("Target has zero variance (constant target)");
            }
        }

        /// <summary>
        /// Validates data during transformation
        /// </summary>
        private void ValidateTransformData(double[][] inputs)
        {
            // Quick validation for transform data
            if (_checkForNaNValues)
            {
                for (int i = 0; i < Math.Min(inputs.Length, 100); i++) // Check first 100 samples
                {
                    for (int j = 0; j < inputs[i].Length; j++)
                    {
                        if (double.IsNaN(inputs[i][j]))
                        {
                            _validationErrors.Add($"NaN value found in transform data at sample {i}, feature {j}");
                            return;
                        }
                    }
                }
            }

            if (_checkForInfiniteValues)
            {
                for (int i = 0; i < Math.Min(inputs.Length, 100); i++) // Check first 100 samples
                {
                    for (int j = 0; j < inputs[i].Length; j++)
                    {
                        if (double.IsInfinity(inputs[i][j]))
                        {
                            _validationErrors.Add($"Infinite value found in transform data at sample {i}, feature {j}");
                            return;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Sets a single parameter value
        /// </summary>
        protected override void SetParameter(string name, object value)
        {
            base.SetParameter(name, value);

            switch (name)
            {
                case "MissingValueThreshold":
                    _missingValueThreshold = Convert.ToDouble(value);
                    break;
                case "OutlierThreshold":
                    _outlierThreshold = Convert.ToDouble(value);
                    break;
                case "CheckForDuplicates":
                    _checkForDuplicates = Convert.ToBoolean(value);
                    break;
                case "CheckForConstantFeatures":
                    _checkForConstantFeatures = Convert.ToBoolean(value);
                    break;
                case "CheckForInfiniteValues":
                    _checkForInfiniteValues = Convert.ToBoolean(value);
                    break;
                case "CheckForNaNValues":
                    _checkForNaNValues = Convert.ToBoolean(value);
                    break;
            }

            ResetFittedState();
        }

        /// <summary>
        /// Gets metadata about this pipeline step
        /// </summary>
        public override Dictionary<string, string> GetMetadata()
        {
            var metadata = base.GetMetadata();
            metadata["ValidationErrors"] = _validationErrors.Count.ToString();
            metadata["ValidationWarnings"] = _validationWarnings.Count.ToString();
            metadata["IsValid"] = IsValid.ToString();
            return metadata;
        }

        /// <summary>
        /// Indicates whether this step requires fitting before transformation
        /// </summary>
        protected override bool RequiresFitting()
        {
            return false; // Validation can be performed without fitting
        }
    }
}