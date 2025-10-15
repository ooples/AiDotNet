namespace AiDotNet.Enums
{
    /// <summary>
    /// Specifies the type of data source for loading data
    /// </summary>
    public enum DataSourceType
    {
        /// <summary>
        /// Comma-separated values file
        /// </summary>
        CSV,
        
        /// <summary>
        /// JSON format
        /// </summary>
        JSON,
        
        /// <summary>
        /// Parquet format
        /// </summary>
        Parquet,
        
        /// <summary>
        /// Excel file
        /// </summary>
        Excel,
        
        /// <summary>
        /// SQL database
        /// </summary>
        SQL,
        
        /// <summary>
        /// Generic database connection
        /// </summary>
        Database,
        
        /// <summary>
        /// HDF5 format
        /// </summary>
        HDF5,
        
        /// <summary>
        /// Binary format
        /// </summary>
        Binary,
        
        /// <summary>
        /// Custom data loader
        /// </summary>
        Custom
    }
}