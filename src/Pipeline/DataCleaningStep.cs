using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.OutlierRemoval;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Generic data cleaning pipeline step for handling missing values, outliers, and data quality issues
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class DataCleaningStep<T> : PipelineStepBase<T>
    {
        private readonly DataCleaningConfig config;
        private Dictionary<int, T> imputationValues;
        private HashSet<int> rowsToRemove;
        private IOutlierRemoval<T, Matrix<T>, (Matrix<T>, Vector<T>)>? outlierRemover;
        
        public DataCleaningStep(DataCleaningConfig config) 
            : base("DataCleaning", MathHelper.GetNumericOperations<T>())
        {
            this.config = config ?? throw new ArgumentNullException(nameof(config));
            this.imputationValues = new Dictionary<int, T>();
            this.rowsToRemove = new HashSet<int>();
            
            IsCacheable = true;
            SupportsParallelExecution = true;
        }
        
        protected override void FitCore(Matrix<T> inputs, Vector<T>? targets)
        {
            // Calculate imputation values for each feature
            if (config.HandleMissingValues)
            {
                for (int j = 0; j < inputs.Columns; j++)
                {
                    var column = inputs.GetColumn(j);
                    var validValues = new List<T>();
                    
                    for (int i = 0; i < column.Length; i++)
                    {
                        var value = column[i];
                        if (!IsInvalidValue(value))
                        {
                            validValues.Add(value);
                        }
                    }
                    
                    if (validValues.Count > 0)
                    {
                        imputationValues[j] = config.ImputationStrategy switch
                        {
                            ImputationStrategy.Mean => CalculateMean(validValues),
                            ImputationStrategy.Median => CalculateMedian(validValues),
                            ImputationStrategy.Mode => CalculateMode(validValues),
                            ImputationStrategy.Zero => NumOps.Zero,
                            _ => CalculateMean(validValues)
                        };
                    }
                    else
                    {
                        imputationValues[j] = NumOps.Zero;
                    }
                }
            }
            
            // Initialize outlier detector
            if (config.HandleOutliers)
            {
                outlierRemover = config.OutlierMethod switch
                {
                    OutlierDetectionMethod.IQR => new IQROutlierRemoval<T, Matrix<T>, (Matrix<T>, Vector<T>)>(),
                    OutlierDetectionMethod.ZScore => new ZScoreOutlierRemoval<T, Matrix<T>, (Matrix<T>, Vector<T>)>(),
                    OutlierDetectionMethod.MAD => new MADOutlierRemoval<T, Matrix<T>, (Matrix<T>, Vector<T>)>(),
                    _ => new NoOutlierRemoval<T, Matrix<T>, (Matrix<T>, Vector<T>)>()
                };
                
                // No need to call Fit - outlier removers calculate statistics during RemoveOutliers
            }
            
            // Identify rows with too many missing values
            if (config.RemoveRowsWithMissingThreshold > 0)
            {
                for (int i = 0; i < inputs.Rows; i++)
                {
                    var row = inputs.GetRow(i);
                    var missingCount = 0;
                    
                    for (int j = 0; j < row.Length; j++)
                    {
                        if (IsInvalidValue(row[j]))
                        {
                            missingCount++;
                        }
                    }
                    
                    var missingRatio = (double)missingCount / row.Length;
                    if (missingRatio > config.RemoveRowsWithMissingThreshold)
                    {
                        rowsToRemove.Add(i);
                    }
                }
            }
            
            UpdateMetadata("ImputationValues", imputationValues.Count.ToString());
            UpdateMetadata("RowsMarkedForRemoval", rowsToRemove.Count.ToString());
            UpdateMetadata("OutlierDetectionMethod", config.OutlierMethod.ToString());
        }
        
        protected override Matrix<T> TransformCore(Matrix<T> inputs)
        {
            var workingData = inputs.DeepCopy();
            
            // Handle missing values
            if (config.HandleMissingValues)
            {
                for (int i = 0; i < workingData.Rows; i++)
                {
                    for (int j = 0; j < workingData.Columns; j++)
                    {
                        if (IsInvalidValue(workingData[i, j]) && imputationValues.ContainsKey(j))
                        {
                            workingData[i, j] = imputationValues[j];
                        }
                    }
                }
            }
            
            // Remove rows with too many missing values
            if (rowsToRemove.Count > 0)
            {
                var keepRows = new List<int>();
                for (int i = 0; i < workingData.Rows; i++)
                {
                    if (!rowsToRemove.Contains(i))
                    {
                        keepRows.Add(i);
                    }
                }
                
                var cleanedMatrix = new Matrix<T>(keepRows.Count, workingData.Columns);
                for (int i = 0; i < keepRows.Count; i++)
                {
                    for (int j = 0; j < workingData.Columns; j++)
                    {
                        cleanedMatrix[i, j] = workingData[keepRows[i], j];
                    }
                }
                workingData = cleanedMatrix;
            }
            
            // Handle outliers
            if (config.HandleOutliers && outlierRemover != null)
            {
                var (cleanedData, _) = outlierRemover.RemoveOutliers(workingData, null);
                workingData = cleanedData;
            }
            
            // Handle duplicates
            if (config.RemoveDuplicates)
            {
                workingData = RemoveDuplicateRows(workingData);
            }
            
            // Validate data integrity
            if (config.ValidateDataIntegrity)
            {
                ValidateDataIntegrity(workingData);
            }
            
            UpdateMetadata("CleanedRows", workingData.Rows.ToString());
            UpdateMetadata("CleanedColumns", workingData.Columns.ToString());
            
            return workingData;
        }
        
        private bool IsInvalidValue(T value)
        {
            var doubleValue = Convert.ToDouble(value);
            return double.IsNaN(doubleValue) || double.IsInfinity(doubleValue);
        }
        
        private T CalculateMean(List<T> values)
        {
            if (values.Count == 0) return NumOps.Zero;
            
            var sum = NumOps.Zero;
            foreach (var value in values)
            {
                sum = NumOps.Add(sum, value);
            }
            
            return NumOps.Divide(sum, NumOps.FromDouble(values.Count));
        }
        
        private T CalculateMedian(List<T> values)
        {
            if (values.Count == 0) return NumOps.Zero;
            
            var sorted = values.OrderBy(v => v).ToList();
            var middle = sorted.Count / 2;
            
            if (sorted.Count % 2 == 0)
            {
                var a = sorted[middle - 1];
                var b = sorted[middle];
                return NumOps.Divide(NumOps.Add(a, b), NumOps.FromDouble(2));
            }
            else
            {
                return sorted[middle];
            }
        }
        
        private T CalculateMode(List<T> values)
        {
            if (values.Count == 0) return NumOps.Zero;
            
            var grouped = values.GroupBy(v => v)
                .OrderByDescending(g => g.Count())
                .FirstOrDefault();
                
            return grouped?.Key ?? NumOps.Zero;
        }
        
        private Matrix<T> RemoveDuplicateRows(Matrix<T> matrix)
        {
            var uniqueRows = new List<Vector<T>>();
            var seenRows = new HashSet<string>();
            
            for (int i = 0; i < matrix.Rows; i++)
            {
                var row = matrix.GetRow(i);
                var rowHash = string.Join(",", row.ToArray().Select(v => v.ToString()));
                
                if (!seenRows.Contains(rowHash))
                {
                    seenRows.Add(rowHash);
                    uniqueRows.Add(row);
                }
            }
            
            var result = new Matrix<T>(uniqueRows.Count, matrix.Columns);
            for (int i = 0; i < uniqueRows.Count; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[i, j] = uniqueRows[i][j];
                }
            }
            
            return result;
        }
        
        private void ValidateDataIntegrity(Matrix<T> data)
        {
            // Check for any remaining invalid values
            var invalidCount = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                for (int j = 0; j < data.Columns; j++)
                {
                    if (IsInvalidValue(data[i, j]))
                    {
                        invalidCount++;
                    }
                }
            }
            
            if (invalidCount > 0)
            {
                UpdateMetadata("DataIntegrityWarning", $"{invalidCount} invalid values remain after cleaning");
            }
            else
            {
                UpdateMetadata("DataIntegrityStatus", "All values valid");
            }
        }
    }
}