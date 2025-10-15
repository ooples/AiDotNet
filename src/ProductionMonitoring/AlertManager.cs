using AiDotNet.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Manages alerts and notifications for production monitoring
    /// </summary>
    public class AlertManager
    {
        private readonly List<IAlertChannel> _alertChannels = default!;
        private readonly Dictionary<string, AlertRule> _alertRules = default!;
        private readonly Dictionary<string, AlertState> _alertStates = default!;
        private readonly ActionBlock<MonitoringAlert> _alertProcessor = default!;
        private readonly AlertConfiguration _configuration = default!;
        private readonly object _lockObject = new object();

        public AlertManager(AlertConfiguration? configuration = null)
        {
            _configuration = configuration ?? new AlertConfiguration();
            _alertChannels = new List<IAlertChannel>();
            _alertRules = new Dictionary<string, AlertRule>();
            _alertStates = new Dictionary<string, AlertState>();
            
            // Create alert processor with parallelism
            _alertProcessor = new ActionBlock<MonitoringAlert>(
                async alert => await ProcessAlertAsync(alert),
                new ExecutionDataflowBlockOptions
                {
                    MaxDegreeOfParallelism = _configuration.MaxParallelAlerts,
                    BoundedCapacity = _configuration.AlertQueueCapacity
                });
        }

        /// <summary>
        /// Registers an alert channel (email, webhook, etc.)
        /// </summary>
        public void RegisterAlertChannel(IAlertChannel channel)
        {
            lock (_lockObject)
            {
                _alertChannels.Add(channel);
            }
        }

        /// <summary>
        /// Adds an alert rule
        /// </summary>
        public void AddAlertRule(AlertRule rule)
        {
            lock (_lockObject)
            {
                _alertRules[rule.RuleId] = rule;
                _alertStates[rule.RuleId] = new AlertState { RuleId = rule.RuleId };
            }
        }

        /// <summary>
        /// Sends an alert
        /// </summary>
        public async Task SendAlertAsync(MonitoringAlert alert)
        {
            // Apply filtering and throttling
            if (!ShouldSendAlert(alert))
            {
                return;
            }

            // Queue alert for processing
            await _alertProcessor.SendAsync(alert);
        }

        /// <summary>
        /// Evaluates alert rules against current metrics
        /// </summary>
        public async Task EvaluateRulesAsync(Dictionary<string, double> metrics)
        {
            List<AlertRule> rulesToEvaluate;
            lock (_lockObject)
            {
                rulesToEvaluate = _alertRules.Values.Where(r => r.IsEnabled).ToList();
            }

            foreach (var rule in rulesToEvaluate)
            {
                if (await EvaluateRuleAsync(rule, metrics))
                {
                    var alert = CreateAlertFromRule(rule, metrics);
                    await SendAlertAsync(alert);
                }
            }
        }

        /// <summary>
        /// Gets alert history
        /// </summary>
        public List<AlertHistoryEntry> GetAlertHistory(DateTime? startDate = null, DateTime? endDate = null, string? severity = null)
        {
            lock (_lockObject)
            {
                var history = _alertStates.Values
                    .SelectMany(state => state.History)
                    .Where(h => (!startDate.HasValue || h.Timestamp >= startDate.Value) &&
                               (!endDate.HasValue || h.Timestamp <= endDate.Value) &&
                               (string.IsNullOrEmpty(severity) || h.Severity == severity))
                    .OrderByDescending(h => h.Timestamp)
                    .ToList();

                return history;
            }
        }

        /// <summary>
        /// Gets alert statistics
        /// </summary>
        public AlertStatistics GetAlertStatistics(DateTime? startDate = null, DateTime? endDate = null)
        {
            var history = GetAlertHistory(startDate, endDate);
            
            return new AlertStatistics
            {
                TotalAlerts = history.Count,
                AlertsBySeverity = history.GroupBy(h => h.Severity)
                    .ToDictionary(g => g.Key, g => g.Count()),
                AlertsByType = history.GroupBy(h => h.AlertType)
                    .ToDictionary(g => g.Key, g => g.Count()),
                AverageResponseTime = history
                    .Where(h => h.ResponseTime.HasValue)
                    .Select(h => h.ResponseTime!.Value.TotalSeconds)
                    .DefaultIfEmpty(0)
                    .Average(),
                TopAlertingRules = history.Where(h => !string.IsNullOrEmpty(h.RuleId))
                    .GroupBy(h => h.RuleId)
                    .OrderByDescending(g => g.Count())
                    .Take(10)
                    .Select(g => new { RuleId = g.Key, Count = g.Count() })
                    .ToDictionary(x => x.RuleId, x => x.Count),
                StartDate = startDate ?? history.LastOrDefault()?.Timestamp ?? DateTime.UtcNow,
                EndDate = endDate ?? history.FirstOrDefault()?.Timestamp ?? DateTime.UtcNow
            };
        }

        /// <summary>
        /// Acknowledges an alert
        /// </summary>
        public async Task AcknowledgeAlertAsync(string alertId, string acknowledgedBy, string? notes = null)
        {
            lock (_lockObject)
            {
                foreach (var state in _alertStates.Values)
                {
                    var entry = state.History.FirstOrDefault(h => h.AlertId == alertId);
                    if (entry != null)
                    {
                        entry.IsAcknowledged = true;
                        entry.AcknowledgedBy = acknowledgedBy;
                        entry.AcknowledgedAt = DateTime.UtcNow;
                        entry.Notes = notes;
                        entry.ResponseTime = entry.AcknowledgedAt - entry.Timestamp;
                        break;
                    }
                }
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Mutes alerts for a specific rule or type
        /// </summary>
        public void MuteAlerts(string ruleIdOrType, TimeSpan duration, string? reason = null)
        {
            var muteUntil = DateTime.UtcNow.Add(duration);
            
            lock (_lockObject)
            {
                if (_alertRules.ContainsKey(ruleIdOrType))
                {
                    _alertStates[ruleIdOrType].MutedUntil = muteUntil;
                    _alertStates[ruleIdOrType].MuteReason = reason;
                }
                else
                {
                    // Mute by type
                    foreach (var state in _alertStates.Values)
                    {
                        state.TypeMutes[ruleIdOrType] = muteUntil;
                    }
                }
            }
        }

        // Private methods

        private async Task ProcessAlertAsync(MonitoringAlert alert)
        {
            try
            {
                // Create alert entry
                var entry = new AlertHistoryEntry
                {
                    AlertId = Guid.NewGuid().ToString(),
                    Timestamp = alert.Timestamp,
                    AlertType = alert.AlertType,
                    Severity = alert.Severity,
                    Message = alert.Message,
                    Context = alert.Context,
                    RuleId = GetRuleIdForAlert(alert)
                };

                // Store in history
                lock (_lockObject)
                {
                    var ruleId = entry.RuleId ?? "system";
                    if (!_alertStates.ContainsKey(ruleId))
                    {
                        _alertStates[ruleId] = new AlertState { RuleId = ruleId };
                    }
                    
                    _alertStates[ruleId].History.Add(entry);
                    
                    // Trim history
                    if (_alertStates[ruleId].History.Count > _configuration.MaxHistoryPerRule)
                    {
                        _alertStates[ruleId].History.RemoveAt(0);
                    }
                }

                // Send through channels
                var channels = GetChannelsForAlert(alert);
                var sendTasks = channels.Select(channel => SendThroughChannelAsync(channel, alert, entry)).ToList();
                await Task.WhenAll(sendTasks);
            }
            catch (Exception ex)
            {
                // Log error but don't throw
                Console.WriteLine($"Error processing alert: {ex.Message}");
            }
        }

        private async Task SendThroughChannelAsync(IAlertChannel channel, MonitoringAlert alert, AlertHistoryEntry entry)
        {
            try
            {
                var enrichedAlert = EnrichAlert(alert, entry);
                await channel.SendAlertAsync(enrichedAlert);
            }
            catch (Exception ex)
            {
                // Log channel error
                Console.WriteLine($"Error sending alert through {channel.GetType().Name}: {ex.Message}");
            }
        }

        private bool ShouldSendAlert(MonitoringAlert alert)
        {
            lock (_lockObject)
            {
                // Check global rate limiting
                var recentAlerts = _alertStates.Values
                    .SelectMany(s => s.History)
                    .Count(h => h.Timestamp > DateTime.UtcNow.AddMinutes(-1));
                
                if (recentAlerts >= _configuration.MaxAlertsPerMinute)
                {
                    return false;
                }

                // Check muting
                var ruleId = GetRuleIdForAlert(alert);
                if (!string.IsNullOrEmpty(ruleId) && _alertStates.ContainsKey(ruleId))
                {
                    var state = _alertStates[ruleId];
                    
                    // Check if rule is muted
                    if (state.MutedUntil.HasValue && state.MutedUntil.Value > DateTime.UtcNow)
                    {
                        return false;
                    }
                    
                    // Check if type is muted
                    if (state.TypeMutes.ContainsKey(alert.AlertType) && 
                        state.TypeMutes[alert.AlertType] > DateTime.UtcNow)
                    {
                        return false;
                    }
                    
                    // Check throttling
                    if (state.LastAlertTime.HasValue)
                    {
                        var timeSinceLastAlert = DateTime.UtcNow - state.LastAlertTime.Value;
                        if (timeSinceLastAlert < GetThrottleDuration(alert.Severity))
                        {
                            return false;
                        }
                    }
                    
                    state.LastAlertTime = DateTime.UtcNow;
                }

                // Check severity filtering
                if (!_configuration.EnabledSeverities.Contains(alert.Severity))
                {
                    return false;
                }

                return true;
            }
        }

        private TimeSpan GetThrottleDuration(string severity)
        {
            return severity switch
            {
                "Critical" => _configuration.CriticalAlertThrottle,
                "Error" => _configuration.ErrorAlertThrottle,
                "Warning" => _configuration.WarningAlertThrottle,
                _ => _configuration.InfoAlertThrottle
            };
        }

        private Task<bool> EvaluateRuleAsync(AlertRule rule, Dictionary<string, double> metrics)
        {
            if (!metrics.ContainsKey(rule.MetricName))
            {
                return Task.FromResult(false);
            }

            var value = metrics[rule.MetricName];
            var threshold = rule.Threshold;

            bool conditionMet = rule.Operator switch
            {
                ">" => value > threshold,
                ">=" => value >= threshold,
                "<" => value < threshold,
                "<=" => value <= threshold,
                "==" => Math.Abs(value - threshold) < 0.0001,
                "!=" => Math.Abs(value - threshold) >= 0.0001,
                _ => false
            };

            if (conditionMet)
            {
                lock (_lockObject)
                {
                    var state = _alertStates[rule.RuleId];
                    state.ConsecutiveMatches++;

                    // Check if we've met the required consecutive matches
                    return Task.FromResult(state.ConsecutiveMatches >= rule.ConsecutiveMatches);
                }
            }
            else
            {
                lock (_lockObject)
                {
                    _alertStates[rule.RuleId].ConsecutiveMatches = 0;
                }
            }

            return Task.FromResult(false);
        }

        private MonitoringAlert CreateAlertFromRule(AlertRule rule, Dictionary<string, double> metrics)
        {
            return new MonitoringAlert
            {
                AlertType = rule.AlertType,
                Severity = rule.Severity,
                Message = string.Format(rule.MessageTemplate, metrics[rule.MetricName], rule.Threshold),
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>
                {
                    ["RuleId"] = rule.RuleId,
                    ["MetricName"] = rule.MetricName,
                    ["MetricValue"] = metrics[rule.MetricName],
                    ["Threshold"] = rule.Threshold,
                    ["Operator"] = rule.Operator
                }
            };
        }

        private string? GetRuleIdForAlert(MonitoringAlert alert)
        {
            if (alert.Context != null && alert.Context.ContainsKey("RuleId"))
            {
                return alert.Context["RuleId"]?.ToString();
            }
            return null;
        }

        private List<IAlertChannel> GetChannelsForAlert(MonitoringAlert alert)
        {
            lock (_lockObject)
            {
                return _alertChannels
                    .Where(c => c.SupportsAlertType(alert.AlertType) && 
                               c.SupportsSeverity(alert.Severity))
                    .ToList();
            }
        }

        private MonitoringAlert EnrichAlert(MonitoringAlert alert, AlertHistoryEntry entry)
        {
            // Add additional context
            var enrichedContext = new Dictionary<string, object>(alert.Context ?? new Dictionary<string, object>())
            {
                ["AlertId"] = entry.AlertId,
                ["Environment"] = _configuration.Environment,
                ["Service"] = _configuration.ServiceName,
                ["Host"] = Environment.MachineName
            };

            return new MonitoringAlert
            {
                AlertType = alert.AlertType,
                Severity = alert.Severity,
                Message = alert.Message,
                Timestamp = alert.Timestamp,
                Context = enrichedContext
            };
        }

        // Helper classes

        public class AlertConfiguration
        {
            public int MaxParallelAlerts { get; set; } = 5;
            public int AlertQueueCapacity { get; set; } = 1000;
            public int MaxAlertsPerMinute { get; set; } = 100;
            public int MaxHistoryPerRule { get; set; } = 1000;
            public List<string> EnabledSeverities { get; set; } = new List<string> { "Critical", "Error", "Warning", "Info" };
            public TimeSpan CriticalAlertThrottle { get; set; } = TimeSpan.FromMinutes(1);
            public TimeSpan ErrorAlertThrottle { get; set; } = TimeSpan.FromMinutes(5);
            public TimeSpan WarningAlertThrottle { get; set; } = TimeSpan.FromMinutes(15);
            public TimeSpan InfoAlertThrottle { get; set; } = TimeSpan.FromMinutes(30);
            public string Environment { get; set; } = "Production";
            public string ServiceName { get; set; } = "AiDotNet";
        }

        public class AlertRule
        {
            public string RuleId { get; set; } = string.Empty;
            public string Name { get; set; } = string.Empty;
            public string Description { get; set; } = string.Empty;
            public string MetricName { get; set; } = string.Empty;
            public string Operator { get; set; } = string.Empty; // >, >=, <, <=, ==, !=
            public double Threshold { get; set; }
            public int ConsecutiveMatches { get; set; } = 1;
            public string AlertType { get; set; } = string.Empty;
            public string Severity { get; set; } = string.Empty;
            public string MessageTemplate { get; set; } = string.Empty;
            public bool IsEnabled { get; set; } = true;
            public Dictionary<string, object> Metadata { get; set; } = new();
        }

        public class AlertState
        {
            public string RuleId { get; set; } = string.Empty;
            public int ConsecutiveMatches { get; set; }
            public DateTime? LastAlertTime { get; set; }
            public DateTime? MutedUntil { get; set; }
            public string? MuteReason { get; set; }
            public Dictionary<string, DateTime> TypeMutes { get; set; } = new Dictionary<string, DateTime>();
            public List<AlertHistoryEntry> History { get; set; } = new List<AlertHistoryEntry>();
        }

        public class AlertHistoryEntry
        {
            public string AlertId { get; set; } = string.Empty;
            public DateTime Timestamp { get; set; }
            public string AlertType { get; set; } = string.Empty;
            public string Severity { get; set; } = string.Empty;
            public string Message { get; set; } = string.Empty;
            public Dictionary<string, object> Context { get; set; } = new();
            public string? RuleId { get; set; }
            public bool IsAcknowledged { get; set; }
            public string? AcknowledgedBy { get; set; }
            public DateTime? AcknowledgedAt { get; set; }
            public TimeSpan? ResponseTime { get; set; }
            public string? Notes { get; set; }
        }

        public class AlertStatistics
        {
            public int TotalAlerts { get; set; }
            public Dictionary<string, int> AlertsBySeverity { get; set; } = new();
            public Dictionary<string, int> AlertsByType { get; set; } = new();
            public double AverageResponseTime { get; set; }
            public Dictionary<string, int> TopAlertingRules { get; set; } = new();
            public DateTime StartDate { get; set; }
            public DateTime EndDate { get; set; }
        }
    }

    /// <summary>
    /// Interface for alert channels
    /// </summary>
    public interface IAlertChannel
    {
        string ChannelName { get; }
        Task SendAlertAsync(MonitoringAlert alert);
        bool SupportsAlertType(string alertType);
        bool SupportsSeverity(string severity);
    }

    /// <summary>
    /// Console alert channel for development/debugging
    /// </summary>
    public class ConsoleAlertChannel : IAlertChannel
    {
        public string ChannelName => "Console";

        public async Task SendAlertAsync(MonitoringAlert alert)
        {
            var color = alert.Severity switch
            {
                "Critical" => ConsoleColor.Red,
                "Error" => ConsoleColor.DarkRed,
                "Warning" => ConsoleColor.Yellow,
                _ => ConsoleColor.Gray
            };

            var originalColor = Console.ForegroundColor;
            Console.ForegroundColor = color;
            
            Console.WriteLine($"[{alert.Timestamp:yyyy-MM-dd HH:mm:ss}] [{alert.Severity}] {alert.AlertType}: {alert.Message}");
            
            if (alert.Context != null && alert.Context.Any())
            {
                Console.WriteLine($"  Context: {string.Join(", ", alert.Context.Select(kv => $"{kv.Key}={kv.Value}"))}");
            }
            
            Console.ForegroundColor = originalColor;
            
            await Task.CompletedTask;
        }

        public bool SupportsAlertType(string alertType) => true;
        public bool SupportsSeverity(string severity) => true;
    }
}
