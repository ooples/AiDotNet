using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents the distribution of predictions
    /// </summary>
    public class PredictionDistribution
    {
        /// <summary>
        /// Gets or sets the bin edges for the histogram
        /// </summary>
        public List<double> BinEdges { get; set; } = new List<double>();
        
        /// <summary>
        /// Gets or sets the counts for each bin
        /// </summary>
        public List<int> BinCounts { get; set; } = new List<int>();
        
        /// <summary>
        /// Gets or sets the mean prediction value
        /// </summary>
        public double Mean { get; set; }
        
        /// <summary>
        /// Gets or sets the standard deviation
        /// </summary>
        public double StandardDeviation { get; set; }
        
        /// <summary>
        /// Gets or sets the skewness
        /// </summary>
        public double Skewness { get; set; }
        
        /// <summary>
        /// Gets or sets the kurtosis
        /// </summary>
        public double Kurtosis { get; set; }
    }
}