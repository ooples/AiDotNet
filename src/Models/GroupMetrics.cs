using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Models
{
    /// <summary>
    /// Metrics for a specific group in fairness analysis
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class GroupMetrics<T>
    {
        private readonly INumericOperations<T> _ops;
        
        /// <summary>
        /// Group identifier
        /// </summary>
        public string GroupId { get; set; }
        
        /// <summary>
        /// Positive prediction rate
        /// </summary>
        public T PositiveRate { get; set; }
        
        /// <summary>
        /// True positive rate
        /// </summary>
        public T TruePositiveRate { get; set; }
        
        /// <summary>
        /// False positive rate
        /// </summary>
        public T FalsePositiveRate { get; set; }
        
        /// <summary>
        /// Group size
        /// </summary>
        public int GroupSize { get; set; }
        
        /// <summary>
        /// Initializes a new instance of GroupMetrics
        /// </summary>
        public GroupMetrics()
        {
            _ops = MathHelper.GetNumericOperations<T>();
            GroupId = string.Empty;
            PositiveRate = _ops.Zero;
            TruePositiveRate = _ops.Zero;
            FalsePositiveRate = _ops.Zero;
            GroupSize = 0;
        }
        
        /// <summary>
        /// Initializes a new instance with specified group ID
        /// </summary>
        public GroupMetrics(string groupId)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            GroupId = groupId;
            PositiveRate = _ops.Zero;
            TruePositiveRate = _ops.Zero;
            FalsePositiveRate = _ops.Zero;
            GroupSize = 0;
        }
    }
}