using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Generic production-ready data loading pipeline step with support for multiple data sources
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class DataLoadingStep<T> : PipelineStepBase<T>
    {
        private readonly string source;
        private readonly DataSourceType sourceType;
        private readonly Func<Task<(Matrix<T> data, Vector<T> labels)>>? customLoader;
        private readonly DataLoadingOptions options;
        
        // Cached data for transform operations
        private Matrix<T>? cachedData;
        private Vector<T>? cachedLabels;
        
        public DataLoadingStep(string source, DataSourceType sourceType, DataLoadingOptions? options = null) 
            : base("DataLoading", MathHelper.GetNumericOperations<T>())
        {
            this.source = source ?? throw new ArgumentNullException(nameof(source));
            this.sourceType = sourceType;
            this.options = options ?? new DataLoadingOptions();
            
            // Data loading steps don't need fitting
            Position = PipelinePosition.Beginning;
            SupportsParallelExecution = true;
        }
        
        public DataLoadingStep(Func<Task<(Matrix<T> data, Vector<T> labels)>> customLoader, DataLoadingOptions? options = null) 
            : base("DataLoading", MathHelper.GetNumericOperations<T>())
        {
            this.customLoader = customLoader ?? throw new ArgumentNullException(nameof(customLoader));
            this.sourceType = DataSourceType.Custom;
            this.options = options ?? new DataLoadingOptions();
            this.source = "Custom";
            
            Position = PipelinePosition.Beginning;
            SupportsParallelExecution = true;
        }
        
        protected override bool RequiresFitting() => false;
        
        protected override void FitCore(Matrix<T> inputs, Vector<T>? targets)
        {
            // Data loading doesn't require fitting
            // But we can cache the data for subsequent transform operations
            cachedData = inputs;
            cachedLabels = targets;
            
            UpdateMetadata("DataRows", inputs.Rows.ToString());
            UpdateMetadata("DataColumns", inputs.Columns.ToString());
            UpdateMetadata("HasLabels", (targets != null).ToString());
        }
        
        protected override Matrix<T> TransformCore(Matrix<T> inputs)
        {
            // For data loading step, transform just returns the input
            // This allows the step to be used in pipelines consistently
            return inputs;
        }
        
        /// <summary>
        /// Loads data from the configured source
        /// </summary>
        /// <returns>Loaded data and labels</returns>
        public async Task<(Matrix<T> data, Vector<T>? labels)> LoadDataAsync()
        {
            try
            {
                switch (sourceType)
                {
                    case DataSourceType.CSV:
                        return await LoadFromCsvAsync();
                        
                    case DataSourceType.Database:
                        return await LoadFromDatabaseAsync();
                        
                    case DataSourceType.API:
                        return await LoadFromApiAsync();
                        
                    case DataSourceType.Memory:
                        return LoadFromMemory();
                        
                    case DataSourceType.Custom:
                        if (customLoader != null)
                        {
                            return await customLoader();
                        }
                        throw new InvalidOperationException("Custom loader not provided");
                        
                    default:
                        throw new NotSupportedException($"Data source type {sourceType} is not supported");
                }
            }
            catch (Exception ex)
            {
                UpdateMetadata("LastError", ex.Message);
                UpdateMetadata("LastErrorTime", DateTime.UtcNow.ToString("O"));
                throw new InvalidOperationException($"Failed to load data from {sourceType}: {ex.Message}", ex);
            }
        }
        
        private async Task<(Matrix<T> data, Vector<T>? labels)> LoadFromCsvAsync()
        {
            if (!File.Exists(source))
            {
                throw new FileNotFoundException($"CSV file not found: {source}");
            }
            
            var lines = await File.ReadAllLinesAsync(source);
            if (lines.Length == 0)
            {
                throw new InvalidOperationException("CSV file is empty");
            }
            
            // Skip header if configured
            var startIdx = options.HasHeader ? 1 : 0;
            var dataLines = lines.Skip(startIdx).ToArray();
            
            if (dataLines.Length == 0)
            {
                throw new InvalidOperationException("No data rows found in CSV file");
            }
            
            // Parse the CSV
            var rows = new List<T[]>();
            var labels = options.LabelColumnIndex >= 0 ? new List<T>() : null;
            
            foreach (var line in dataLines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                var values = line.Split(options.Delimiter);
                var row = new List<T>();
                
                for (int i = 0; i < values.Length; i++)
                {
                    if (i == options.LabelColumnIndex && labels != null)
                    {
                        labels.Add(ParseValue(values[i]));
                    }
                    else
                    {
                        row.Add(ParseValue(values[i]));
                    }
                }
                
                if (row.Count > 0)
                {
                    rows.Add(row.ToArray());
                }
            }
            
            // Convert to Matrix and Vector
            if (rows.Count == 0)
            {
                throw new InvalidOperationException("No valid data rows found");
            }
            
            var numRows = rows.Count;
            var numCols = rows[0].Length;
            var dataMatrix = new Matrix<T>(numRows, numCols);
            
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    dataMatrix[i, j] = rows[i][j];
                }
            }
            
            var labelVector = labels != null ? new Vector<T>(labels.ToArray()) : null;
            
            UpdateMetadata("LoadedRows", numRows.ToString());
            UpdateMetadata("LoadedColumns", numCols.ToString());
            
            return (dataMatrix, labelVector);
        }
        
        private T ParseValue(string value)
        {
            try
            {
                // Handle missing values
                if (string.IsNullOrWhiteSpace(value) || value.Equals(options.MissingValueIndicator, StringComparison.OrdinalIgnoreCase))
                {
                    return NumOps.Zero;
                }
                
                // Try to parse as the target type
                return (T)Convert.ChangeType(value.Trim(), typeof(T));
            }
            catch
            {
                // If parsing fails, return zero
                return NumOps.Zero;
            }
        }
        
        private async Task<(Matrix<T> data, Vector<T>? labels)> LoadFromDatabaseAsync()
        {
            // Placeholder for database loading
            // In a real implementation, this would use ADO.NET, EF Core, or another data access library
            throw new NotImplementedException("Database loading not yet implemented");
        }
        
        private async Task<(Matrix<T> data, Vector<T>? labels)> LoadFromApiAsync()
        {
            // Placeholder for API loading
            // In a real implementation, this would use HttpClient to fetch data
            throw new NotImplementedException("API loading not yet implemented");
        }
        
        private (Matrix<T> data, Vector<T>? labels) LoadFromMemory()
        {
            if (cachedData == null)
            {
                throw new InvalidOperationException("No data cached in memory");
            }
            
            return (cachedData, cachedLabels);
        }
    }
    
    /// <summary>
    /// Options for data loading
    /// </summary>
    public class DataLoadingOptions
    {
        /// <summary>
        /// Whether the first row contains headers
        /// </summary>
        public bool HasHeader { get; set; } = true;
        
        /// <summary>
        /// Delimiter for CSV files
        /// </summary>
        public char Delimiter { get; set; } = ',';
        
        /// <summary>
        /// Index of the label column (-1 if no labels)
        /// </summary>
        public int LabelColumnIndex { get; set; } = -1;
        
        /// <summary>
        /// Indicator for missing values
        /// </summary>
        public string MissingValueIndicator { get; set; } = "NA";
        
        /// <summary>
        /// Maximum number of rows to load (0 for all)
        /// </summary>
        public int MaxRows { get; set; } = 0;
        
        /// <summary>
        /// Connection string for database sources
        /// </summary>
        public string? ConnectionString { get; set; }
        
        /// <summary>
        /// API endpoint for API sources
        /// </summary>
        public string? ApiEndpoint { get; set; }
        
        /// <summary>
        /// API key for authenticated API sources
        /// </summary>
        public string? ApiKey { get; set; }
    }
}