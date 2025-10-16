using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Deployment.Techniques
{
    /// <summary>
    /// Calibration data for quantization.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class CalibrationData<T>
    {
        /// <summary>
        /// Gets or sets the minimum values per layer.
        /// </summary>
        public Dictionary<string, T> MinValues { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the maximum values per layer.
        /// </summary>
        public Dictionary<string, T> MaxValues { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the histograms per layer.
        /// </summary>
        public Dictionary<string, Vector<T>> Histograms { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the number of calibration samples processed.
        /// </summary>
        public int SampleCount { get; set; }
    }
}