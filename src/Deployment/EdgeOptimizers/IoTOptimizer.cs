using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Deployment.EdgeOptimizers
{
    /// <summary>
    /// IoT device-specific model optimizer for edge computing scenarios.
    /// </summary>
    public class IoTOptimizer<TInput, TOutput, TMetadata> : ModelOptimizer<TInput, TOutput, TMetadata>
    {
        public override string Name => "IoT Optimizer";
        public override DeploymentTarget Target => DeploymentTarget.IoT;

        private readonly Dictionary<string, IoTDeviceProfile> _deviceProfiles = new Dictionary<string, IoTDeviceProfile>();

        public IoTOptimizer()
        {
            InitializeDeviceProfiles();
            ConfigureForIoT();
        }

        private void InitializeDeviceProfiles()
        {
            DeviceProfiles = new Dictionary<string, IoTDeviceProfile>
            {
                ["RaspberryPi"] = new IoTDeviceProfile
                {
                    DeviceName = "Raspberry Pi",
                    CPU = "ARM Cortex-A72",
                    RAM = 8192, // 8 GB max
                    Storage = 32768, // 32 GB typical
                    MaxModelSize = 500, // 500 MB
                    SupportedFormats = new[] { "Tensor<double>Flow Lite", "ONNX", "OpenVINO" },
                    Accelerators = new[] { "VideoCore GPU", "Neural Compute Stick" }
                },
                ["JetsonNano"] = new IoTDeviceProfile
                {
                    DeviceName = "NVIDIA Jetson Nano",
                    CPU = "ARM Cortex-A57",
                    RAM = 4096, // 4 GB
                    Storage = 16384, // 16 GB
                    MaxModelSize = 1000, // 1 GB
                    SupportedFormats = new[] { "Tensor<double>RT", "Tensor<double>Flow", "ONNX" },
                    Accelerators = new[] { "128-core Maxwell GPU", "CUDA" }
                },
                ["Arduino"] = new IoTDeviceProfile
                {
                    DeviceName = "Arduino",
                    CPU = "ATmega328P",
                    RAM = 2, // 2 KB
                    Storage = 32, // 32 KB
                    MaxModelSize = 0.025, // 25 KB
                    SupportedFormats = new[] { "Tensor<double>Flow Lite Micro", "Custom" },
                    Accelerators = new string[] { }
                },
                ["ESP32"] = new IoTDeviceProfile
                {
                    DeviceName = "ESP32",
                    CPU = "Xtensa LX6",
                    RAM = 520, // 520 KB
                    Storage = 4096, // 4 MB
                    MaxModelSize = 2, // 2 MB
                    SupportedFormats = new[] { "Tensor<double>Flow Lite Micro", "ESP-DL" },
                    Accelerators = new[] { "ESP32-S3 AI Acceleration" }
                },
                ["CoralEdgeTPU"] = new IoTDeviceProfile
                {
                    DeviceName = "Coral Edge TPU",
                    CPU = "ARM Cortex-A53",
                    RAM = 1024, // 1 GB
                    Storage = 8192, // 8 GB
                    MaxModelSize = 100, // 100 MB
                    SupportedFormats = new[] { "Tensor<double>Flow Lite", "Edge TPU Model" },
                    Accelerators = new[] { "Edge TPU Coprocessor" }
                }
            };
            
            foreach (var profile in profiles)
            {
                _deviceProfiles.Add(profile.Key, profile.Value);
            }
        }

        private void ConfigureForIoT()
        {
            Configuration.MaxModelSize = 50.0; // 50 MB default for IoT
            Configuration.MaxLatency = 100.0; // 100 ms for real-time response
            Configuration.MaxMemory = 100.0; // 100 MB max memory
            Configuration.PlatformSpecificSettings["DeviceProfile"] = "RaspberryPi";
            Configuration.PlatformSpecificSettings["PowerMode"] = "Balanced";
            Configuration.PlatformSpecificSettings["EnableEdgeComputing"] = true;
            Configuration.PlatformSpecificSettings["EnableDistributedInference"] = false;
        }

        public override async Task<IModel<TInput, TOutput, TMetadata>> OptimizeAsync(IModel<TInput, TOutput, TMetadata> model, OptimizationOptions options)
        {
            var deviceProfile = options.CustomOptions.ContainsKey("DeviceProfile")
                ? (string)options.CustomOptions["DeviceProfile"]
                : "RaspberryPi";

            var profile = DeviceProfiles[deviceProfile];
            var optimizedModel = model;

            // Apply extreme optimization for constrained devices
            if (profile.RAM < 1000) // Less than 1 GB RAM
            {
                optimizedModel = await ApplyExtremeOptimizationAsync(optimizedModel, profile);
            }
            else
            {
                // Standard IoT optimizations
                if (options.EnableQuantization)
                {
                    optimizedModel = await ApplyIoTQuantizationAsync(optimizedModel, profile);
                }

                if (options.EnablePruning)
                {
                    optimizedModel = await ApplyIoTPruningAsync(optimizedModel, profile);
                }
            }

            // Device-specific optimizations
            optimizedModel = await ApplyDeviceSpecificOptimizationsAsync(optimizedModel, profile, options);

            // Power optimization
            var powerMode = Configuration.PlatformSpecificSettings["PowerMode"].ToString();
            optimizedModel = await OptimizeForPowerConsumptionAsync(optimizedModel, powerMode);

            // Edge computing optimizations
            if ((bool)Configuration.PlatformSpecificSettings["EnableEdgeComputing"])
            {
                optimizedModel = await OptimizeForEdgeComputingAsync(optimizedModel);
            }

            return optimizedModel;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> ApplyExtremeOptimizationAsync(IModel<TInput, TOutput, TMetadata> model, IoTDeviceProfile profile)
        {
            // Simulate extreme optimization for microcontrollers
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Convert to fixed-point arithmetic
            // 2. Remove all unnecessary operations
            // 3. Implement custom kernels for specific operations
            // 4. Use lookup tables where possible
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> ApplyIoTQuantizationAsync(IModel<TInput, TOutput, TMetadata> model, IoTDeviceProfile profile)
        {
            // Simulate IoT-specific quantization
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Use INT8 quantization for most operations
            // 2. Apply post-training quantization
            // 3. Optimize for specific hardware accelerators
            // 4. Balance accuracy vs. performance
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> ApplyIoTPruningAsync(IModel<TInput, TOutput, TMetadata> model, IoTDeviceProfile profile)
        {
            // Simulate IoT-specific pruning
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Remove small weight values
            // 2. Eliminate redundant connections
            // 3. Simplify model architecture
            // 4. Reduce number of parameters significantly
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> ApplyDeviceSpecificOptimizationsAsync(IModel<TInput, TOutput, TMetadata> model, IoTDeviceProfile profile, OptimizationOptions options)
        {
            switch (profile.DeviceName)
            {
                case "Raspberry Pi":
                    return await OptimizeForRaspberryPiAsync(model);
                case "NVIDIA Jetson Nano":
                    return await OptimizeForJetsonAsync(model);
                case "Arduino":
                    return await OptimizeForArduinoAsync(model);
                case "ESP32":
                    return await OptimizeForESP32Async(model);
                case "Coral Edge TPU":
                    return await OptimizeForEdgeTPUAsync(model);
                default:
                    return model;
            }
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> OptimizeForRaspberryPiAsync(IModel<TInput, TOutput, TMetadata> model)
        {
            // Simulate Raspberry Pi optimization
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Optimize for ARM NEON instructions
            // 2. Enable VideoCore GPU acceleration
            // 3. Configure for multi-threading on quad-core CPU
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> OptimizeForJetsonAsync(IModel<TInput, TOutput, TMetadata> model)
        {
            // Simulate Jetson optimization
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Convert to Tensor<double>RT format
            // 2. Enable CUDA acceleration
            // 3. Optimize for Maxwell GPU architecture
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> OptimizeForArduinoAsync(IModel<TInput, TOutput, TMetadata> model)
        {
            // Simulate Arduino optimization
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Convert to Tensor<double>Flow Lite Micro
            // 2. Use fixed-point operations only
            // 3. Implement custom memory management
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> OptimizeForESP32Async(IModel<TInput, TOutput, TMetadata> model)
        {
            // Simulate ESP32 optimization
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Use ESP-DL framework
            // 2. Optimize for dual-core processing
            // 3. Leverage ESP32-S3 AI acceleration if available
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> OptimizeForEdgeTPUAsync(IModel<TInput, TOutput, TMetadata> model)
        {
            // Simulate Edge TPU optimization
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Compile model for Edge TPU
            // 2. Ensure all operations are Edge TPU compatible
            // 3. Optimize for TPU's matrix multiplication units
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> OptimizeForPowerConsumptionAsync(IModel<TInput, TOutput, TMetadata> model, string powerMode)
        {
            // Simulate power optimization
            await Task.Delay(100);

            // In a real implementation, this would adjust based on power mode:
            // - "LowPower": Aggressive optimization, lower accuracy
            // - "Balanced": Moderate optimization
            // - "Performance": Minimal optimization, best accuracy
            return model;
        }

        private async Task<IModel<TInput, TOutput, TMetadata>> OptimizeForEdgeComputingAsync(IModel<TInput, TOutput, TMetadata> model)
        {
            // Simulate edge computing optimization
            await Task.Delay(100);

            // In a real implementation, this would:
            // 1. Enable model partitioning for distributed inference
            // 2. Implement caching strategies
            // 3. Optimize for intermittent connectivity
            return model;
        }

        public override async Task<DeploymentPackage> CreateDeploymentPackageAsync(IModel<TInput, TOutput, TMetadata> model, string targetPath)
        {
            var deviceProfile = Configuration.PlatformSpecificSettings["DeviceProfile"].ToString();
            var profile = DeviceProfiles[deviceProfile];
            
            var package = new DeploymentPackage
            {
                PackagePath = targetPath,
                Format = $"IoT-{deviceProfile}",
                Metadata = new Dictionary<string, object>
                {
                    ["Platform"] = "IoT",
                    ["DeviceProfile"] = deviceProfile,
                    ["Timestamp"] = DateTime.UtcNow,
                    ["PowerMode"] = Configuration.PlatformSpecificSettings["PowerMode"]
                }
            };

            // Create directory structure
            var modelsDir = Path.Combine(targetPath, "models");
            var firmwareDir = Path.Combine(targetPath, "firmware");
            var configDir = Path.Combine(targetPath, "config");
            var scriptsDir = Path.Combine(targetPath, "scripts");
            var docsDir = Path.Combine(targetPath, "docs");

            Directory.CreateDirectory(modelsDir);
            Directory.CreateDirectory(firmwareDir);
            Directory.CreateDirectory(configDir);
            Directory.CreateDirectory(scriptsDir);
            Directory.CreateDirectory(docsDir);

            // Save optimized model in appropriate format
            package.ModelPath = await SaveOptimizedModelAsync(model, modelsDir, profile);

            // Create deployment scripts
            await CreateDeploymentScriptsAsync(scriptsDir, profile);
            package.Artifacts["Scripts"] = scriptsDir;

            // Create firmware if needed
            if (profile.RAM < 1000) // Microcontroller
            {
                await CreateFirmwareAsync(firmwareDir, profile);
                package.Artifacts["Firmware"] = firmwareDir;
            }

            // Create configuration files
            await CreateConfigurationFilesAsync(configDir, profile);
            package.ConfigPath = Path.Combine(configDir, "device_config.json");

            // Create documentation
            await CreateDocumentationAsync(docsDir, profile);
            package.Artifacts["Documentation"] = docsDir;

            // Calculate package size
            var allFiles = Directory.GetFiles(targetPath, "*", SearchOption.AllDirectories);
            package.PackageSize = allFiles.Sum(f => new FileInfo(f).Length) / (1024.0 * 1024.0);

            return package;
        }

        private async Task<string> SaveOptimizedModelAsync(IModel<TInput, TOutput, TMetadata> model, string modelsDir, IoTDeviceProfile profile)
        {
            string modelPath;
            
            if (profile.DeviceName == "Arduino" || profile.DeviceName == "ESP32")
            {
                // Save as C array for microcontrollers
                modelPath = Path.Combine(modelsDir, "model_data.h");
                var cArray = GenerateCArrayModel();
                await FileAsyncHelper.WriteAllTextAsync(modelPath, cArray);
            }
            else if (profile.SupportedFormats.Contains("Tensor<double>RT"))
            {
                // Save as Tensor<double>RT engine
                modelPath = Path.Combine(modelsDir, "model.engine");
                // await ConvertToTensorRTAsync(model, modelPath);
            }
            else
            {
                // Default to Tensor<double>Flow Lite
                modelPath = Path.Combine(modelsDir, "model.tflite");
                // await ConvertToTFLiteAsync(model, modelPath);
            }

            return modelPath;
        }

        private async Task CreateDeploymentScriptsAsync(string scriptsDir, IoTDeviceProfile profile)
        {
            // Create installation script
            var installScript = GenerateInstallScript(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(scriptsDir, "install.sh"), installScript);

            // Create deployment script
            var deployScript = GenerateDeploymentScript(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(scriptsDir, "deploy.sh"), deployScript);

            // Create monitoring script
            var monitorScript = GenerateMonitoringScript(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(scriptsDir, "monitor.py"), monitorScript);

            // Create update script
            var updateScript = GenerateUpdateScript(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(scriptsDir, "update.sh"), updateScript);
        }

        private async Task CreateFirmwareAsync(string firmwareDir, IoTDeviceProfile profile)
        {
            if (profile.DeviceName == "Arduino")
            {
                var arduinoSketch = GenerateArduinoSketch();
                await FileAsyncHelper.WriteAllTextAsync(Path.Combine(firmwareDir, "inference.ino"), arduinoSketch);
            }
            else if (profile.DeviceName == "ESP32")
            {
                var esp32Code = GenerateESP32Firmware();
                await FileAsyncHelper.WriteAllTextAsync(Path.Combine(firmwareDir, "main.cpp"), esp32Code);
                
                var platformioConfig = GeneratePlatformIOConfig();
                await FileAsyncHelper.WriteAllTextAsync(Path.Combine(firmwareDir, "platformio.ini"), platformioConfig);
            }
        }

        private async Task CreateConfigurationFilesAsync(string configDir, IoTDeviceProfile profile)
        {
            // Device configuration
            var deviceConfig = GenerateDeviceConfig(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(configDir, "device_config.json"), deviceConfig);

            // Network configuration
            var networkConfig = GenerateNetworkConfig();
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(configDir, "network_config.json"), networkConfig);

            // Security configuration
            var securityConfig = GenerateSecurityConfig();
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(configDir, "security_config.json"), securityConfig);

            // Power management configuration
            var powerConfig = GeneratePowerConfig();
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(configDir, "power_config.json"), powerConfig);
        }

        private async Task CreateDocumentationAsync(string docsDir, IoTDeviceProfile profile)
        {
            // Deployment guide
            var deploymentGuide = GenerateDeploymentGuide(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(docsDir, "deployment_guide.md"), deploymentGuide);

            // Hardware requirements
            var hardwareReqs = GenerateHardwareRequirements(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(docsDir, "hardware_requirements.md"), hardwareReqs);

            // API documentation
            var apiDocs = GenerateAPIDocs(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(docsDir, "api_documentation.md"), apiDocs);

            // Troubleshooting guide
            var troubleshooting = GenerateTroubleshootingGuide(profile);
            await FileAsyncHelper.WriteAllTextAsync(Path.Combine(docsDir, "troubleshooting.md"), troubleshooting);
        }

        private string GenerateCArrayModel()
        {
            return @"
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Quantized model data
const unsigned char model_data[] = {
    0x54, 0x46, 0x4c, 0x33, // TFLite magic number
    // ... model data bytes ...
};

const unsigned int model_data_len = sizeof(model_data);

#endif // MODEL_DATA_H
";
        }

        private string GenerateInstallScript(IoTDeviceProfile profile)
        {
            return $@"#!/bin/bash
# Installation script for {profile.DeviceName}

echo ""Installing dependencies for {profile.DeviceName}...""

# Update package manager
{(profile.DeviceName.Contains("Raspberry") ? "sudo apt-get update" : "echo 'Manual installation required'")}

# Install required packages
{GeneratePackageInstallCommands(profile)}

# Install Python dependencies
pip3 install -r requirements.txt

# Setup model directory
mkdir -p /opt/aimodel
cp -r models/* /opt/aimodel/

# Setup systemd service
sudo cp services/aimodel.service /etc/systemd/system/
sudo systemctl enable aimodel.service

echo ""Installation complete!""
";
        }

        private string GenerateDeploymentScript(IoTDeviceProfile profile)
        {
            return @"#!/bin/bash
# Deployment script

echo ""Deploying model to device...""

# Stop existing service
sudo systemctl stop aimodel.service

# Copy new model
cp models/model.tflite /opt/aimodel/

# Update configuration
cp config/*.json /etc/aimodel/

# Start service
sudo systemctl start aimodel.service

# Verify deployment
sleep 5
if systemctl is-active --quiet aimodel.service; then
    echo ""Deployment successful!""
else
    echo ""Deployment failed! Check logs: sudo journalctl -u aimodel.service""
    exit 1
fi
";
        }

        private string GenerateMonitoringScript(IoTDeviceProfile profile)
        {
            return @"
import time
import psutil
import json
import requests
from datetime import datetime

class DeviceMonitor:
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'temperature': [],
            'inference_count': 0,
            'inference_time': []
        }
    
    def collect_metrics(self):
        # CPU usage
        self.metrics['cpu_usage'].append(psutil.cpu_percent(interval=1))
        
        # Memory usage
        mem = psutil.virtual_memory()
        self.metrics['memory_usage'].append(mem.percent)
        
        # Temperature (platform specific)
        temp = self.get_temperature()
        if temp:
            self.metrics['temperature'].append(temp)
        
        # Keep only last 100 measurements
        for key in ['cpu_usage', 'memory_usage', 'temperature']:
            if len(self.metrics[key]) > 100:
                self.metrics[key] = self.metrics[key][-100:]
    
    def get_temperature(self):
        try:
            # Raspberry Pi temperature
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read()) / 1000.0
        except:
            return None
    
    def report_metrics(self):
        report = {
            'timestamp': datetime.now().isoformat(),
            'device_id': self.get_device_id(),
            'metrics': {
                'cpu_avg': sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'memory_avg': sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'temperature_avg': sum(self.metrics['temperature']) / len(self.metrics['temperature']) if self.metrics['temperature'] else 0,
                'inference_count': self.metrics['inference_count']
            }
        }
        
        # Send to monitoring endpoint
        try:
            requests.post('http://monitoring.local/metrics', json=report)
        except:
            print(f""Failed to send metrics: {json.dumps(report, indent=2)}"")
    
    def get_device_id(self):
        # Get unique device identifier
        return 'iot-device-001'
    
    def run(self):
        while True:
            self.collect_metrics()
            if int(time.time()) % 60 == 0:  # Report every minute
                self.report_metrics()
            time.sleep(5)

if __name__ == '__main__':
    monitor = DeviceMonitor()
    monitor.run()
";
        }

        private string GenerateUpdateScript(IoTDeviceProfile profile)
        {
            return @"#!/bin/bash
# OTA Update Script

echo ""Checking for updates...""

# Configuration
UPDATE_SERVER=""https://updates.example.com""
DEVICE_ID=$(cat /etc/machine-id)
CURRENT_VERSION=$(cat /opt/aimodel/version.txt)

# Check for updates
LATEST_VERSION=$(curl -s ""$UPDATE_SERVER/latest-version"")

if [ ""$CURRENT_VERSION"" != ""$LATEST_VERSION"" ]; then
    echo ""Update available: $LATEST_VERSION""
    
    # Download update
    wget -O /tmp/update.tar.gz ""$UPDATE_SERVER/download/$LATEST_VERSION""
    
    # Verify checksum
    EXPECTED_CHECKSUM=$(curl -s ""$UPDATE_SERVER/checksum/$LATEST_VERSION"")
    ACTUAL_CHECKSUM=$(sha256sum /tmp/update.tar.gz | cut -d' ' -f1)
    
    if [ ""$EXPECTED_CHECKSUM"" != ""$ACTUAL_CHECKSUM"" ]; then
        echo ""Checksum verification failed!""
        exit 1
    fi
    
    # Apply update
    tar -xzf /tmp/update.tar.gz -C /tmp/
    bash /tmp/update/install.sh
    
    # Update version
    echo ""$LATEST_VERSION"" > /opt/aimodel/version.txt
    
    echo ""Update completed successfully!""
else
    echo ""Already up to date.""
fi
";
        }

        private string GenerateArduinoSketch()
        {
            return @"
#include <Tensor<double>FlowLite.h>
#include ""tensorflow/lite/micro/micro_interpreter.h""
#include ""tensorflow/lite/micro/micro_mutable_op_resolver.h""
#include ""tensorflow/lite/schema/schema_generated.h""
#include ""model_data.h""

// Globals
const int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
    Serial.begin(115200);
    
    // Load model
    const tflite::Model* model = tflite::GetModel(model_data);
    
    // Create interpreter
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;
    
    // Allocate tensors
    interpreter->AllocateTensors();
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println(""Model loaded successfully!"");
}

void loop() {
    // Read sensor data
    float sensor_data = analogRead(A0) / 1023.0;
    
    // Set input
    input->data.f[0] = sensor_data;
    
    // Run inference
    unsigned long start_time = micros();
    interpreter->Invoke();
    unsigned long inference_time = micros() - start_time;
    
    // Get output
    float prediction = output->data.f[0];
    
    // Print results
    Serial.print(""Inference time: "");
    Serial.print(inference_time);
    Serial.print("" us, Prediction: "");
    Serial.println(prediction);
    
    delay(1000);
}
";
        }

        private string GenerateESP32Firmware()
        {
            return @"
#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include ""tensorflow/lite/micro/micro_interpreter.h""
#include ""tensorflow/lite/micro/micro_mutable_op_resolver.h""
#include ""model_data.h""

// WiFi credentials
const char* ssid = ""your-ssid"";
const char* password = ""your-password"";

// Tensor<double>Flow Lite
const int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
    Serial.begin(115200);
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println(""Connecting to WiFi..."");
    }
    Serial.println(""Connected!"");
    
    // Initialize model
    initializeModel();
}

void initializeModel() {
    const tflite::Model* model = tflite::GetModel(model_data);
    
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddRelu();
    resolver.AddSoftmax();
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;
    
    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);
}

void loop() {
    // Collect sensor data
    float sensor_values[10];
    for (int i = 0; i < 10; i++) {
        sensor_values[i] = analogRead(34 + i) / 4095.0;
    }
    
    // Prepare input
    for (int i = 0; i < 10; i++) {
        input->data.f[i] = sensor_values[i];
    }
    
    // Run inference
    unsigned long start = millis();
    interpreter->Invoke();
    unsigned long inference_time = millis() - start;
    
    // Process output
    int predicted_class = 0;
    float max_score = output->data.f[0];
    for (int i = 1; i < output->dims->data[1]; i++) {
        if (output->data.f[i] > max_score) {
            max_score = output->data.f[i];
            predicted_class = i;
        }
    }
    
    // Send results
    sendResults(predicted_class, max_score, inference_time);
    
    delay(5000); // Run every 5 seconds
}

void sendResults(int prediction, float confidence, unsigned long time_ms) {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(""http://api.example.com/inference"");
        http.addHeader(""Content-Type"", ""application/json"");
        
        String payload = ""{""device_id"":""esp32-001"",""prediction"":"" + 
                        String(prediction) + "",""confidence"":"" + 
                        String(confidence) + "",""inference_time"":"" + 
                        String(time_ms) + ""}"";
        
        int httpCode = http.POST(payload);
        http.end();
    }
}
";
        }

        private string GeneratePlatformIOConfig()
        {
            return @"
[env:esp32]
platform = espressif32
board = esp32dev
framework = arduino

lib_deps = 
    tensorflow/lite/micro

build_flags = 
    -DESP32
    -O3
    -Wno-error=unused-function
    -Wno-error=unused-variable

monitor_speed = 115200
";
        }

        private string GenerateDeviceConfig(IoTDeviceProfile profile)
        {
            var config = new
            {
                device = new
                {
                    id = "iot-device-001",
                    profile = profile.DeviceName,
                    capabilities = profile.Accelerators,
                    resources = new
                    {
                        cpu = profile.CPU,
                        ram_mb = profile.RAM,
                        storage_mb = profile.Storage
                    }
                },
                model = new
                {
                    format = profile.SupportedFormats[0],
                    max_size_mb = profile.MaxModelSize,
                    optimization_level = "high"
                },
                inference = new
                {
                    batch_size = 1,
                    timeout_ms = 1000,
                    max_concurrent = 1,
                    cache_predictions = true
                },
                monitoring = new
                {
                    enabled = true,
                    interval_seconds = 60,
                    metrics = new[] { "cpu", "memory", "temperature", "inference_count", "latency" }
                }
            };

            return System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
        }

        private string GenerateNetworkConfig()
        {
            var config = new
            {
                connectivity = new
                {
                    primary = "wifi",
                    fallback = "cellular",
                    wifi = new
                    {
                        ssid = "iot-network",
                        security = "wpa2",
                        reconnect_attempts = 5
                    },
                    cellular = new
                    {
                        apn = "iot.carrier.com",
                        enabled = false
                    }
                },
                protocols = new
                {
                    mqtt = new
                    {
                        enabled = true,
                        broker = "mqtt.example.com",
                        port = 1883,
                        topics = new
                        {
                            telemetry = "devices/{device_id}/telemetry",
                            commands = "devices/{device_id}/commands",
                            config = "devices/{device_id}/config"
                        }
                    },
                    http = new
                    {
                        enabled = true,
                        endpoint = "https://api.example.com/v1/devices",
                        auth_type = "bearer"
                    }
                },
                edge_computing = new
                {
                    enabled = true,
                    local_discovery = true,
                    mesh_network = false
                }
            };

            return System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
        }

        private string GenerateSecurityConfig()
        {
            var config = new
            {
                authentication = new
                {
                    method = "certificate",
                    certificate_path = "/etc/ssl/device.crt",
                    key_path = "/etc/ssl/device.key",
                    ca_path = "/etc/ssl/ca.crt"
                },
                encryption = new
                {
                    data_at_rest = true,
                    data_in_transit = true,
                    algorithm = "AES-256-GCM"
                },
                secure_boot = new
                {
                    enabled = true,
                    verify_firmware = true
                },
                access_control = new
                {
                    whitelist_ips = new[] { "10.0.0.0/8", "192.168.0.0/16" },
                    max_connections = 10
                }
            };

            return System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
        }

        private string GeneratePowerConfig()
        {
            var config = new
            {
                power_management = new
                {
                    mode = Configuration.PlatformSpecificSettings["PowerMode"],
                    sleep_enabled = true,
                    wake_on_event = true,
                    idle_timeout_seconds = 300
                },
                performance_profiles = new
                {
                    low_power = new
                    {
                        cpu_frequency_mhz = 400,
                        inference_interval_ms = 5000,
                        batch_size = 1
                    },
                    balanced = new
                    {
                        cpu_frequency_mhz = 800,
                        inference_interval_ms = 1000,
                        batch_size = 1
                    },
                    performance = new
                    {
                        cpu_frequency_mhz = 1200,
                        inference_interval_ms = 100,
                        batch_size = 4
                    }
                },
                battery = new
                {
                    monitoring_enabled = true,
                    low_battery_threshold = 20,
                    critical_battery_threshold = 5,
                    shutdown_on_critical = true
                }
            };

            return System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
        }

        private string GenerateDeploymentGuide(IoTDeviceProfile profile)
        {
            return $@"# IoT Deployment Guide for {profile.DeviceName}

## Overview
This guide covers the deployment of optimized AI models to {profile.DeviceName} devices.

## Prerequisites
- {profile.DeviceName} with {profile.RAM} MB RAM
- Operating System: {GetRecommendedOS(profile)}
- Network connectivity (WiFi/Ethernet)
- Power supply

## Installation Steps

1. **Prepare the Device**
   ```bash
   # Update system
   {GetUpdateCommand(profile)}
   
   # Install dependencies
   bash scripts/install.sh
   ```

2. **Deploy the Model**
   ```bash
   # Copy files to device
   scp -r deployment/* user@device:/home/user/
   
   # Run deployment script
   ssh user@device 'bash /home/user/scripts/deploy.sh'
   ```

3. **Configure the Service**
   - Edit `/etc/aimodel/device_config.json` for device-specific settings
   - Configure network in `/etc/aimodel/network_config.json`
   - Set security options in `/etc/aimodel/security_config.json`

4. **Start the Service**
   ```bash
   sudo systemctl start aimodel.service
   sudo systemctl enable aimodel.service
   ```

5. **Verify Operation**
   ```bash
   # Check service status
   sudo systemctl status aimodel.service
   
   # View logs
   sudo journalctl -u aimodel.service -f
   
   # Test inference
   curl http://localhost:8080/predict -d '{{""input"": [1.0, 2.0, 3.0]}}'
   ```

## Performance Tuning

### CPU Optimization
- Enable all cores: `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Set CPU frequency: Edit power configuration

### Memory Optimization
- Increase swap: `sudo dphys-swapfile swapoff && sudo nano /etc/dphys-swapfile`
- Optimize memory usage in device configuration

### Network Optimization
- Use wired connection when possible
- Configure QoS for inference traffic
- Enable local caching

## Monitoring

- Real-time metrics: `python3 scripts/monitor.py`
- Remote monitoring: Configure MQTT/HTTP endpoints
- Alerts: Set thresholds in monitoring configuration

## Troubleshooting

See `troubleshooting.md` for common issues and solutions.
";
        }

        private string GenerateHardwareRequirements(IoTDeviceProfile profile)
        {
            return $@"# Hardware Requirements for {profile.DeviceName}

## Minimum Requirements
- CPU: {profile.CPU}
- RAM: {profile.RAM} MB
- Storage: {profile.Storage} MB
- Network: WiFi 802.11n or Ethernet

## Recommended Configuration
- RAM: {profile.RAM * 2} MB for optimal performance
- Storage: {profile.Storage * 2} MB for logs and updates
- Cooling: Active cooling for sustained workloads
- Power: Stable 5V power supply (2A minimum)

## Supported Accelerators
{string.Join("\n", profile.Accelerators.Select(a => $"- {a}"))}

## Performance Expectations
- Model Size: Up to {profile.MaxModelSize} MB
- Inference Latency: {EstimateLatency(null!)} ms (CPU only)
- Power Consumption: 2-10W depending on workload

## Optional Components
- USB Neural Compute Stick for acceleration
- External storage for model versioning
- UPS for uninterrupted operation
- Temperature/humidity sensors for environmental monitoring
";
        }

        private string GenerateAPIDocs(IoTDeviceProfile profile)
        {
            return @"# API Documentation

## REST API

### Health Check
```
GET /health
Response: {""status"": ""healthy"", ""uptime"": 3600}
```

### Inference
```
POST /predict
Content-Type: application/json
Body: {""input"": [1.0, 2.0, 3.0]}
Response: {""prediction"": 0.85, ""class"": 1, ""latency_ms"": 45}
```

### Metrics
```
GET /metrics
Response: {
  ""cpu_usage"": 25.5,
  ""memory_usage"": 45.2,
  ""temperature"": 55.0,
  ""inference_count"": 1000,
  ""average_latency"": 42.3
}
```

## MQTT Topics

### Telemetry
```
Topic: devices/{device_id}/telemetry
Payload: {""temperature"": 55.0, ""cpu"": 25.5, ""memory"": 45.2}
```

### Commands
```
Topic: devices/{device_id}/commands
Payload: {""command"": ""update_model"", ""url"": ""https://.../""}
```

### Configuration
```
Topic: devices/{device_id}/config
Payload: {""inference_interval"": 1000, ""batch_size"": 1}
```

## WebSocket API

### Connection
```
ws://device-ip:8081/ws
```

### Real-time Inference
```
Send: {""type"": ""inference"", ""data"": [1.0, 2.0, 3.0]}
Receive: {""type"": ""result"", ""prediction"": 0.85, ""latency"": 45}
```

## Error Codes
- 400: Invalid input format
- 413: Input too large
- 500: Inference error
- 503: Model not loaded
";
        }

        private string GenerateTroubleshootingGuide(IoTDeviceProfile profile)
        {
            return @"# Troubleshooting Guide

## Common Issues

### Model fails to load
- Check available memory: `free -h`
- Verify model file integrity: `sha256sum models/model.tflite`
- Check permissions: `ls -la /opt/aimodel/`
- Review logs: `sudo journalctl -u aimodel.service`

### High inference latency
- Check CPU throttling: `vcgencmd measure_clock arm`
- Monitor temperature: `vcgencmd measure_temp`
- Reduce model complexity or batch size
- Enable hardware acceleration if available

### Network connectivity issues
- Test connection: `ping -c 4 google.com`
- Check firewall rules: `sudo iptables -L`
- Verify MQTT broker: `mosquitto_sub -h broker -t test`
- Review network configuration

### Out of memory errors
- Increase swap space
- Reduce model size through quantization
- Limit concurrent inferences
- Monitor memory usage: `htop`

### Service crashes
- Check core dumps: `ls /var/crash/`
- Review system logs: `dmesg | tail -50`
- Verify power supply stability
- Test with minimal configuration

## Diagnostic Commands

```bash
# System information
uname -a
cat /proc/cpuinfo
cat /proc/meminfo

# Service status
systemctl status aimodel.service
journalctl -u aimodel.service -n 100

# Resource usage
top -b -n 1
iostat -x 1
netstat -tulpn

# Model verification
python3 -c ""import tensorflow as tf; print(tf.__version__)""
```

## Performance Optimization

1. **Enable GPU/Accelerator**
   - Install drivers
   - Configure runtime
   - Verify acceleration

2. **Optimize Power Settings**
   - Disable CPU throttling
   - Set performance governor
   - Configure wake locks

3. **Network Optimization**
   - Use local inference when possible
   - Batch requests
   - Enable compression

## Getting Help

- Logs location: `/var/log/aimodel/`
- Community forum: https://forum.example.com
- Support email: support@example.com
";
        }

        private string GeneratePackageInstallCommands(IoTDeviceProfile profile)
        {
            switch (profile.DeviceName)
            {
                case "Raspberry Pi":
                    return @"sudo apt-get install -y python3-pip python3-dev libatlas-base-dev
sudo pip3 install tensorflow-lite numpy";
                    
                case "NVIDIA Jetson Nano":
                    return @"sudo apt-get install -y python3-pip python3-dev
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow";
                    
                default:
                    return "# Manual installation required for this platform";
            }
        }

        private string GetRecommendedOS(IoTDeviceProfile profile)
        {
            return profile.DeviceName switch
            {
                "Raspberry Pi" => "Raspberry Pi OS (64-bit)",
                "NVIDIA Jetson Nano" => "JetPack 4.6+",
                "Arduino" => "Arduino IDE 1.8+",
                "ESP32" => "ESP-IDF 4.4+",
                "Coral Edge TPU" => "Mendel Linux",
                _ => "Linux (Debian-based)"
            };
        }

        private string GetUpdateCommand(IoTDeviceProfile profile)
        {
            return profile.DeviceName switch
            {
                "Raspberry Pi" => "sudo apt-get update && sudo apt-get upgrade -y",
                "NVIDIA Jetson Nano" => "sudo apt update && sudo apt upgrade -y",
                "ESP32" => "pio update",
                _ => "# Platform-specific update command"
            };
        }

        protected override double EstimateLatency(IModel<TInput, TOutput, TMetadata> model)
        {
            var deviceProfile = Configuration.PlatformSpecificSettings["DeviceProfile"].ToString();
            var profile = DeviceProfiles[deviceProfile];
            
            var baseLatency = base.EstimateLatency(model);
            
            // Adjust for IoT hardware limitations
            if (profile.RAM < 100) // Microcontroller
            {
                baseLatency *= 10; // Much slower on microcontrollers
            }
            else if (profile.Accelerators.Any(a => a.Contains("GPU") || a.Contains("TPU")))
            {
                baseLatency *= 0.5; // Faster with accelerators
            }
            else
            {
                baseLatency *= 2; // Generally slower on IoT devices
            }

            return baseLatency;
        }

        protected override double EstimateMemoryRequirements(IModel<TInput, TOutput, TMetadata> model)
        {
            var deviceProfile = Configuration.PlatformSpecificSettings["DeviceProfile"].ToString();
            var profile = DeviceProfiles[deviceProfile];
            
            var baseMemory = base.EstimateMemoryRequirements(model);
            
            // Constrain to device capabilities
            return Math.Min(baseMemory, profile.RAM * 0.8); // Use max 80% of available RAM
        }

        private class IoTDeviceProfile
        {
            public string DeviceName { get; set; } = string.Empty;
            public string CPU { get; set; } = string.Empty;
            public double RAM { get; set; } // MB
            public double Storage { get; set; } // MB
            public double MaxModelSize { get; set; } // MB
            public string[] SupportedFormats { get; set; } = new string[0];
            public string[] Accelerators { get; set; } = new string[0];
        }
    }
}
