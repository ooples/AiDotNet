using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// ML Pipeline implementation
    /// </summary>
    public class MLPipeline<T, TInput, TOutput>
    {
        private readonly List<IPipelineStep> steps;
        private readonly PipelineConfiguration configuration;
        
        public MLPipeline(List<IPipelineStep> steps, PipelineConfiguration configuration)
        {
            this.steps = steps;
            this.configuration = configuration;
        }
        
        /// <summary>
        /// Run the pipeline asynchronously
        /// </summary>
        public async Task<PipelineExecutionResult<T, TInput, TOutput>> RunAsync()
        {
            var result = new PipelineExecutionResult<T, TInput, TOutput>
            {
                Success = true,
                StartTime = DateTime.UtcNow
            };
            
            var context = new PipelineContext<T, TInput, TOutput>();
            
            try
            {
                foreach (var step in steps)
                {
                    if (configuration.EnableLogging)
                    {
                        Console.WriteLine($"Executing step: {step.Name}");
                    }
                    
                    await step.ExecuteAsync(context);
                    
                    if (context.HasError)
                    {
                        result.Success = false;
                        result.Error = context.Error;
                        break;
                    }
                }
                
                result.EndTime = DateTime.UtcNow;
                result.Metrics = context.Metrics;
                result.Model = context.Model;
                result.DeploymentInfo = context.DeploymentInfo;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Error = ex.Message;
            }
            
            return result;
        }
    }
    
    /// <summary>
    /// Pipeline execution result
    /// </summary>
    public class PipelineExecutionResult<T, TInput, TOutput>
    {
        public bool Success { get; set; }
        public string Error { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public Dictionary<string, double> Metrics { get; set; } = new Dictionary<string, double>();
        public IFullModel<T, TInput, TOutput> Model { get; set; }
        public DeploymentInfo DeploymentInfo { get; set; }
    }
    
    /// <summary>
    /// Pipeline execution context
    /// </summary>
    public class PipelineContext<T, TInput, TOutput>
    {
        public bool HasError { get; set; }
        public string Error { get; set; }
        public Dictionary<string, object> Data { get; set; } = new Dictionary<string, object>();
        public Dictionary<string, double> Metrics { get; set; } = new Dictionary<string, double>();
        public IFullModel<T, TInput, TOutput> Model { get; set; }
        public DeploymentInfo DeploymentInfo { get; set; }
    }
    
    /// <summary>
    /// Deployment information
    /// </summary>
    public class DeploymentInfo
    {
        public string Endpoint { get; set; }
        public string Platform { get; set; }
        public DateTime DeployedAt { get; set; }
        public Dictionary<string, string> Configuration { get; set; } = new Dictionary<string, string>();
    }
}