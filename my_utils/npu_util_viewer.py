from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio


from furiosa.device import list_devices


class WARBOYDevice:
    def __init__(self):
        pass

    @classmethod
    async def create(cls):
        self = cls()
        self.warboy_devices = await list_devices()
        self.last_pc = {}
        self.time_count = 0
        return self

    async def __call__(self):
        power_info, util_info, temper_info, devices = (
            await self._get_warboy_device_status()
        )
        self.time_count += 1
        return power_info, util_info, temper_info, self.time_count, devices

    async def _get_warboy_device_status(self,):
        status = [[] for _ in range(4)]

        for device in self.warboy_devices:
            warboy_name = str(device)
            device_idx = warboy_name[3:]
            per_counters = device.performance_counters()

            if len(per_counters) != 0:
                fetcher = device.get_hwmon_fetcher()
                temper = await fetcher.read_temperatures()
                peak_device_temper = int(str(temper[0]).split(" ")[-1]) // 1000
                power_info = str((await fetcher.read_powers_average())[0])
                p = int(float(power_info.split(" ")[-1]) / 1000000.0)

                status[0].append(p)
                status[2].append(peak_device_temper)
                status[3].append(device_idx)

            t_utils = 0.0
            for pc in per_counters:
                pe_name = str(pc[0])
                cur_pc = pc[1]

                if pe_name in self.last_pc:
                    result = cur_pc.calculate_utilization(self.last_pc[pe_name])
                    util = result.npu_utilization()
                    if not ("0-1" in pe_name):
                        util /= 2.0
                    t_utils += util

                self.last_pc[pe_name] = cur_pc

            if len(per_counters) != 0:
                t_utils = int(t_utils * 100.0)
                status[1].append(t_utils)
        return status


app = FastAPI()

HTML_CONTENT = """
<!DOCTYPE html>
<html>
    <head>
        <title>NPU Utilization Monitor</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                margin: 0; 
                padding: 20px; 
                font-family: Arial, sans-serif; 
                background: #1a1a1a;
                color: #ffffff;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
            }
            .chart { 
                width: 100%; 
                height: 400px; 
                margin-bottom: 20px;
                background: #2d2d2d;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .stats { 
                display: flex; 
                justify-content: space-around; 
                margin-bottom: 20px; 
            }
            .stat-box {
                flex: 1;
                margin: 0 10px;
                padding: 20px;
                border-radius: 10px;
                background: #2d2d2d;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .stat-value { 
                font-size: 32px; 
                font-weight: bold; 
                margin: 10px 0;
                color: #4285f4;
            }
            h1 { 
                color: #4285f4;
                text-align: center;
                margin-bottom: 30px;
            }
            h3 { 
                color: #ffffff;
                margin: 0;
                opacity: 0.9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Null AI NPU Utilization Monitor</h1>
            <div class="stats">
                <div class="stat-box">
                    <h3>NPU Utilization</h3>
                    <div id="npu-value" class="stat-value">0%</div>
                </div>
                <div class="stat-box">
                    <h3>Average Utilization</h3>
                    <div id="avg-value" class="stat-value">0%</div>
                </div>
            </div>
            <div id="chart" class="chart"></div>
        </div>
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            // Initialize the plot with NPU utilization only
            const npuValues = [];
            let avgNpu = 0;
            
            const trace1 = {
                y: [],
                name: 'NPU',
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#4285f4',
                    width: 3,
                    shape: 'spline',
                    smoothing: 1.3
                },
                hoverinfo: 'y+name'
            };

            const avgLine = {
                y: [],
                name: 'Average',
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#fbbc04',
                    width: 2,
                    dash: 'dash'
                },
                hoverinfo: 'y+name'
            };
            
            const layout = {
                title: {
                    text: 'NPU Utilization Monitor',
                    font: {
                        size: 24,
                        color: '#ffffff'
                    }
                },
                yaxis: {
                    title: 'Utilization (%)',
                    range: [0, 100],
                    gridcolor: '#404040',
                    zerolinecolor: '#404040',
                    tickfont: { 
                        size: 12,
                        color: '#ffffff'
                    },
                    titlefont: {
                        color: '#ffffff'
                    }
                },
                xaxis: {
                    gridcolor: '#404040',
                    zerolinecolor: '#404040',
                    showticklabels: true,
                    tickfont: { 
                        size: 10,
                        color: '#ffffff'
                    },
                    tickmode: 'array',
                    ticktext: [],
                    tickvals: [],
                    title: 'Time (s)',
                    titlefont: {
                        color: '#ffffff'
                    }
                },
                paper_bgcolor: '#2d2d2d',
                plot_bgcolor: '#2d2d2d',
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1.1,
                    orientation: 'h',
                    bgcolor: 'rgba(45,45,45,0.9)',
                    font: { 
                        size: 12,
                        color: '#ffffff'
                    }
                },
                margin: { t: 50, r: 20, l: 50, b: 20 },
                shapes: [{
                    type: 'rect',
                    xref: 'paper',
                    yref: 'paper',
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                    line: {
                        color: '#404040',
                        width: 1
                    }
                }]
            };
            
            const config = {
                responsive: true,
                displayModeBar: false
            };
            
            Plotly.newPlot('chart', [trace1, avgLine], layout, config);
            
            const maxPoints = 1000;  // Maximum number of points to store
            const avgWindowSize = 300;  // Number of points for moving average (30 seconds)
            
            // Keep track of the data point index
            let dataIndex = 0;
            
            // Calculate moving average
            function calculateMovingAverage(arr) {
                // Get the window data for the last 30 seconds
                const windowData = arr.slice(Math.max(0, arr.length - avgWindowSize));
                
                // Return 0 if no data
                if (windowData.length === 0) return 0;
                
                // Calculate average including all values
                return windowData.reduce((a, b) => a + b, 0) / windowData.length;
            }
            
            ws.onmessage = function(event) {
                if (!event.data) return;
                
                const data = JSON.parse(event.data);
                dataIndex++;
                
                // Update NPU value with animation
                const updateValue = (elementId, newValue) => {
                    const element = document.getElementById(elementId);
                    const currentValue = parseFloat(element.textContent);
                    
                    // 현재 값과 새로운 값이 같으면 업데이트하지 않음
                    if (currentValue === newValue) return;
                    
                    element.textContent = newValue.toFixed(1) + '%';
                };
                
                updateValue('npu-value', data.npu);
                
                // Update average calculation - include all values including zeros
                npuValues.push(data.npu);
                if (npuValues.length > maxPoints) {
                    npuValues.shift();  // Remove oldest point if we exceed maxPoints
                }
                
                // Calculate moving average with all values
                avgNpu = calculateMovingAverage(npuValues);
                updateValue('avg-value', avgNpu);
                
                // Update plot with smooth transition
                Plotly.extendTraces('chart', {
                    y: [[data.npu]]
                }, [0]);
                
                // Update average line
                const avgArray = Array(npuValues.length).fill(avgNpu);
                Plotly.restyle('chart', {
                    'y': [npuValues, avgArray]
                });
                
                // Update x-axis ticks every 5 seconds (50 data points)
                if (dataIndex % 50 === 0) {
                    const tickvals = [];
                    const ticktext = [];
                    for (let i = 0; i <= dataIndex; i += 50) {
                        tickvals.push(i);
                        ticktext.push((i/10).toString());  // Convert to seconds (assuming 100ms interval)
                    }
                    Plotly.relayout('chart', {
                        'xaxis.range': [0, dataIndex],
                        'xaxis.tickmode': 'array',
                        'xaxis.tickvals': tickvals,
                        'xaxis.ticktext': ticktext
                    });
                } else {
                    Plotly.relayout('chart', {
                        'xaxis.range': [0, dataIndex]
                    });
                }
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize WARBOYDevice
    warboy = await WARBOYDevice.create()
    
    try:
        while True:
            # Get device status
            power_info, util_info, temper_info, time_count, devices = await warboy()
            
            # Send NPU utilization data
            if util_info:  # Check if we have any utilization data
                await websocket.send_json({
                    "npu": util_info[1]  # Assuming we want the first device's utilization
                })
            
            # Wait a bit before next update
            await asyncio.sleep(0.1)  # 100ms interval
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
