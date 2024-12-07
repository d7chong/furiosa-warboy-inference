from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import logging

from furiosa.device import list_devices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WARBOYDevice:
    def __init__(self):
        self.warboy_devices = []
        self.last_pc = {}
        self.time_count = 0

    @classmethod
    async def create(cls):
        self = cls()
        self.warboy_devices = await list_devices()
        if not self.warboy_devices:
            logger.error("No WARBOY devices detected.")
        else:
            device_names = [str(device) for device in self.warboy_devices]
            logger.info(f"Detected devices: {device_names}")
        return self

    async def __call__(self):
        power_info, util_info, temper_info, devices = await self._get_warboy_device_status()
        self.time_count += 1
        return power_info, util_info, temper_info, self.time_count, devices

    async def _get_warboy_device_status(self):
        status = [[] for _ in range(4)]  # [power_info, util_info, temper_info, devices]

        for device in self.warboy_devices:
            warboy_name = str(device)
            device_id = warboy_name  # Use full device name as the unique identifier
            per_counters = device.performance_counters()

            if not per_counters:
                # logger.warning(f"No performance counters for device {warboy_name}")
                continue

            try:
                fetcher = device.get_hwmon_fetcher()
                temper = await fetcher.read_temperatures()
                if (temper):
                    peak_device_temper = int(str(temper[0]).split(" ")[-1]) // 1000
                    status[2].append(peak_device_temper)
                else:
                    peak_device_temper = 0
                    status[2].append(peak_device_temper)
                    # logger.warning(f"No temperature data for device {warboy_name}")

                power_info = str((await fetcher.read_powers_average())[0])
                p = int(float(power_info.split(" ")[-1]) / 1000000.0)

                status[0].append(p)
                status[3].append(device_id)  # Append the full device name

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

                if per_counters:
                    t_utils = int(t_utils * 100.0)
                    status[1].append(t_utils)
            except Exception as e:
                logger.error(f"Error while fetching status for device {warboy_name}: {e}")

        # logger.info(f"Utilization Info: {status[1]}")
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
                background-color: #1e1e1e; /* Slightly lighter dark background for contrast */
                color: #ffffff; /* Light text */
                font-family: 'Roboto', sans-serif; /* Modern, clean font */
                margin: 0;
                padding: 20px;
                display: grid;
                grid-template-rows: auto 1fr; /* Add two rows: header and main content */
                grid-template-columns: 1fr 2fr 1fr; /* Define 1x3 grid for header */
                align-items: center; /* Vertically center items in header */
                gap: 10px 0; /* Reduce vertical spacing between header and main content */
                height: 100vh;
                box-sizing: border-box;
            }
            h1 {
                margin: 20px 0;
                color: #ff0000; /* Changed from #ffffff to red */
                font-size: 3em;
                text-align: center;
                letter-spacing: 2px;
                font-weight: 300;
                justify-self: center; /* Center the title in the grid */
            }
            /* Removed #device-names styles */
            #average-dashboard {
                width: 90%;
                max-width: 1200px;
                margin-bottom: 30px;
                padding: 20px;
                background-color: #2c2c2c; /* Darker dashboard background */
                border: none; /* Removed border for a cleaner look */
                border-radius: 12px;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
                display: flex;
                flex-wrap: wrap;
                gap: 25px;
                justify-content: center;
                grid-row: 2;
                margin-top: 10px; /* Reduce top margin for closer spacing */
            }
            .average-item {
                flex: 1 1 180px;
                background-color: #3a3a3a;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s;
            }
            .average-item:hover {
                transform: translateY(-5px);
            }
            .average-item h3 {
                margin: 0;
                font-size: 1.3em;
                color: #ffffff;
                font-weight: 400;
            }
            .average-item p {
                margin: 10px 0 0 0;
                font-size: 2em; /* Increased font size */
                /* Color will be set dynamically via JavaScript */
                font-weight: bold;
            }
            #chart {
                width: 90%;
                max-width: 1200px;
                height: 650px;
                background-color: #2c2c2c; /* Dark chart background */
                border: none;
                border-radius: 12px;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
                width: 100%; /* Ensure chart takes full width */
            }
            #average-dashboard {
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                /* existing styles... */
            }
            .average-item {
                flex: 1;
                /* ...existing styles... */
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .average-item p {
                /* ...existing styles... */
                text-align: center;
            }
            /* Adjust grid layout to have three rows: header, average-dashboard, and chart */
            display: grid;
            grid-template-rows: auto auto 1fr; /* Header, Average Dashboard, Main Content */
            grid-template-columns: 1fr 2fr 1fr; /* Define 1x3 grid for header */
            align-items: center; /* Vertically center items in header */
            gap: 10px 0; /* Reduce vertical spacing between rows */
            height: 100vh;
            box-sizing: border-box;
        </style>
        <style>
            /* ...existing styles... */
            body {
                /* Ensure grid layout has three rows: header, average-dashboard, chart */
                display: grid;
                grid-template-rows: auto auto 1fr; /* Header, Average Dashboard, Main Content */
                grid-template-columns: 1fr 2fr 1fr; /* Define 1x3 grid for header */
                align-items: center; /* Vertically center items in header */
                gap: 10px 0; /* Reduce vertical spacing between rows */
                height: 100vh;
                box-sizing: border-box;
            }

            /* Ensure header elements are in the first row */
            div.header {
                grid-row: 1;
                display: contents;
            }

            /* Style for average-dashboard to be in the second row and second column */
            #average-dashboard {
                grid-row: 2;
                grid-column: 2; /* Position in the second column */
                /* ...existing styles... */
                margin-top: 10px; /* Reduce top margin for closer spacing */
                display: flex;
                justify-content: center; /* Center the dashboard horizontally */
                align-items: center; /* Optional: center items vertically */
            }

            /* Ensure the chart is in the third row and second column */
            #chart {
                grid-row: 3;
                grid-column: 2; /* Position in the second column */
                /* ...existing styles... */
                width: 100%; /* Ensure chart takes full width */
                display: flex;
                justify-content: center; /* Center the chart horizontally */
            }
            /* Remove redundant grid declarations */
            /* ...existing styles... */
        </style>
        <style>
                /* ...existing styles... */
                /* Remove duplicate and conflicting styles */
                /* Consolidate #average-dashboard styles */
                #average-dashboard {
                    grid-row: 2;
                    grid-column: 2; /* Position in the second column */
                    width: 90%;
                    max-width: 1200px;
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #2c2c2c; /* Darker dashboard background */
                    border: none; /* Removed border for a cleaner look */
                    border-radius: 12px;
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
                    display: flex;
                    flex-wrap: wrap;
                    gap: 25px;
                    justify-content: center;
                    margin-top: 10px; /* Reduce top margin for closer spacing */
                    flex-direction: row; /* Ensure items are in a row */
                }
                
                /* Consolidate .average-item styles */
                .average-item {
                    flex: 1 1 180px;
                    background-color: #3a3a3a;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    transition: transform 0.3s;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                
                .average-item:hover {
                    transform: translateY(-5px);
                }
                
                .average-item h3 {
                    margin: 0;
                    font-size: 1.3em;
                    color: #ffffff;
                    font-weight: 400;
                }
                
                .average-item p {
                    margin: 10px 0 0 0;
                    font-size: 2em; /* Increased font size */
                    /* Color will be set dynamically via JavaScript */
                    font-weight: bold;
                    text-align: center;
                }
                
                /* Remove conflicting width settings for #chart */
                #chart {
                    grid-row: 3;
                    grid-column: 2; /* Position in the second column */
                    width: 100%; /* Ensure chart takes full width */
                    height: 650px;
                    background-color: #2c2c2c; /* Dark chart background */
                    border: none;
                    border-radius: 12px;
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
                    display: flex;
                    justify-content: center; /* Center the chart horizontally */
                }
                
                /* Remove any redundant display properties */
                /* ...existing styles... */
            </style>
    </head>
    <body>
        <div class="header" style="display: contents;">
            <!-- First Image in (0,0) -->
            <img src="https://images.crunchbase.com/image/upload/c_pad,h_256,w_256,f_auto,q_auto:eco,dpr_1/ac6c6784959948e1aa377e8b01cfed51" alt="New Image" style="height: 80px; margin: 0 auto;">
        
            <!-- Title in (0,1) -->
            <h1>WARBOY NPU Utilization</h1>
        
            <!-- Second Image in (0,2) -->
            <img src="https://dli5ezlttyahz.cloudfront.net/fai-headerWARBOY.png" alt="WARBOY Logo" style="height: 80px; margin: 0 auto;">
        </div>
        
        <!-- Average Utilization Dashboard moved below the header grid -->
        <div id="average-dashboard">
            <!-- Average items will be dynamically inserted here -->
        </div>
        <div id="chart"></div>
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);

            const traces = {}; // Store traces for each device
            const traceIndices = {}; // Map trace names to their Plotly indices
            let trackingStarted = false; // Flag to start tracking after first positive value
            let baseTime = null; // Base time to set the first x-value to 0

            // Object to store sum and count for each device to calculate averages
            const averages = {};
            const currentUtilizations = {}; // Added to store current utilizations

            // Object to store device name colors for consistency
            const deviceColors = {};

            const layout = {
                title: '',
                plot_bgcolor: '#2c2c2c', // Dark plot background
                paper_bgcolor: '#1e1e1e', // Dark paper background
                font: {
                    color: '#ffffff' // Light font color
                },
                xaxis: {
                    title: 'Time (s)',
                    showgrid: true,
                    gridcolor: '#444444',
                    showline: true,
                    linecolor: '#ffffff',
                    zeroline: false,
                    tickfont: { size: 18 }, // Increased from 14 to 18
                    autorange: true // Enable autorange for dynamic expansion
                    // range: [0, baseTime || 100] // Removed to allow dynamic expansion
                },
                yaxis: {
                    title: 'Utilization (%)',
                    range: [0, 100], // Disallow negative values
                    showgrid: true,
                    gridcolor: '#444444',
                    showline: true,
                    linecolor: '#ffffff',
                    zeroline: false,
                    tickcolor: '#ffffff',
                    tickfont: { size: 18 } // Increased from 14 to 18
                },
                showlegend: true, // Enable legend
                legend: {
                    x: 1,
                    xanchor: 'right',
                    bgcolor: 'rgba(0,0,0,0)',
                    bordercolor: '#444444',
                    font: {
                        color: '#ffffff',
                        size: 14
                    }
                },
                margin: { l: 70, r: 70, t: 50, b: 70 }
            };

            const data = []; // Array to hold all traces
            Plotly.newPlot('chart', data, layout);

            const averageLines = {}; // Object to store average line traces

            // Function to update the average dashboard
            function updateAverageDashboard(currentTime) { // Added currentTime parameter
                const dashboard = document.getElementById('average-dashboard');
                dashboard.innerHTML = ''; // Clear existing content

                for (const device in averages) {
                    const avg = averages[device].count > 0 ? (averages[device].sum / averages[device].count).toFixed(2) : '0.00';
                    const current = currentUtilizations[device] !== undefined ? currentUtilizations[device] : '0.00';
                    const color = deviceColors[device] || '#00ff00';

                    // Current Utilization Box
                    const currentBox = document.createElement('div');
                    currentBox.className = 'average-item';

                    const currentTitle = document.createElement('h3');
                    currentTitle.textContent = `${device} Current`;
                    currentTitle.style.color = color; // Set device-specific color

                    const currentValue = document.createElement('p');
                    currentValue.style.color = color;
                    currentValue.textContent = `${current}%`;

                    currentBox.appendChild(currentTitle);
                    currentBox.appendChild(currentValue);

                    // Average Utilization Box
                    const averageBox = document.createElement('div');
                    averageBox.className = 'average-item';

                    const averageTitle = document.createElement('h3');
                    averageTitle.textContent = `${device} Average`;
                    averageTitle.style.color = '#ffff00'; // Set average line color to yellow

                    const averageValue = document.createElement('p');
                    averageValue.style.color = '#ffff00'; // Set average value color to yellow
                    averageValue.textContent = `${avg}%`;

                    averageBox.appendChild(averageTitle);
                    averageBox.appendChild(averageValue);

                    // Append both boxes to dashboard
                    dashboard.appendChild(currentBox);
                    dashboard.appendChild(averageBox);

                    // Update average lines in Plotly
                    if (!averageLines[device]) {
                        averageLines[device] = {
                            x: [0, currentTime],
                            y: [avg, avg],
                            name: `${device} Avg`,
                            type: 'scatter',
                            mode: 'lines',
                            line: { dash: 'dash', width: 2, color: '#ffff00' }, // Set average line color to yellow
                            fill: 'none',
                            connectgaps: true
                        };
                        data.push(averageLines[device]);
                        Plotly.addTraces('chart', averageLines[device]);
                        traceIndices[`${device} Avg`] = data.length - 1;
                    } else {
                        const avgVal = parseFloat(avg);
                        // Extend the average line to span the entire width
                        averageLines[device].x = [0, currentTime];
                        averageLines[device].y = [avgVal, avgVal];
                        Plotly.update('chart', { x: averageLines[device].x, y: averageLines[device].y }, {}, [traceIndices[`${device} Avg`]]);
                    }
                }

                // No need to set xaxis.range as autorange is enabled
            }

            ws.onmessage = function(event) {
                try {
                    const message = JSON.parse(event.data);
                    console.log("WebSocket Data Received:", message);

                    const time = message.time_count; // Use integer time_count
                    const devices = message.devices || [];
                    const utilization = message.util_info || [];

                    // Skip processing if no valid device data
                    if (devices.length === 1 && devices[0] === "No Data") {
                        console.warn("No valid device data received.");
                        return; // Skip further processing
                    }

                    devices.forEach((device, index) => {
                        let traceName = device || `Device-${index}`;
                        // Remove "pe-" prefix if present
                        if (traceName.startsWith("pe-")) {
                            traceName = traceName.slice(3);
                        }

                        if (!traces[traceName]) {
                            // Assign a color to the device
                            const color = getColor(index);
                            deviceColors[traceName] = color;

                            // Create a new trace for the device if it doesn't exist
                            traces[traceName] = {
                                x: [],
                                y: [],
                                name: traceName,
                                type: 'scatter',
                                mode: 'lines', // Smooth curves without markers
                                line: { shape: 'spline', width: 3, color: color } // Increased width for better visibility
                            };
                            data.push(traces[traceName]);
                            Plotly.addTraces('chart', traces[traceName]);
                            traceIndices[traceName] = data.length - 1; // Assign the new trace index correctly
                            console.log(`Added new trace for ${traceName} at index ${traceIndices[traceName]}:`, traces[traceName]);
                        }

                        const util = utilization[index];
                        if (util !== undefined) {
                            // Initialize trackingStarted on the first positive value
                            if (!trackingStarted && util > 0) {
                                baseTime = time;
                                trackingStarted = true;
                                console.log("Started tracking utilization data.");
                            }

                            // Only plot if tracking has started
                            if (trackingStarted) {
                                // Calculate relative time for x-axis
                                const relativeTime = baseTime !== null ? (time - baseTime) : 0;

                                const newX = [relativeTime];
                                const newY = [util];
                                const traceIndex = traceIndices[traceName]; // Use the correct trace index
                                Plotly.extendTraces('chart', { x: [newX], y: [newY] }, [traceIndex]); // Wrap in arrays
                                console.log(`Extended trace for ${traceName} with x: ${relativeTime}, y: ${util}`);

                                // Update averages
                                if (!averages[traceName]) {
                                    averages[traceName] = { sum: 0, count: 0 };
                                }
                                averages[traceName].sum += util;
                                averages[traceName].count += 1;
                                currentUtilizations[traceName] = util; // Added to store current utilization
                                updateAverageDashboard(relativeTime); // Pass relativeTime

                                // Optional: Limit the number of points to a reasonable value
                                const maxPoints = 1000;
                                const currentTrace = traces[traceName];
                                if (currentTrace.x.length > maxPoints) {
                                    // Remove the fixed x-axis range logic
                                    // Plotly.relayout('chart', {
                                    //     'xaxis.range': [currentTrace.x[currentTrace.x.length - maxPoints], currentTrace.x[currentTrace.x.length - 1]]
                                    // });
                                }
                            }
                        }
                    });

                    console.log("Plotly data updated:", data);
                } catch (err) {
                    console.error("Error processing WebSocket message:", err);
                }
            };

            ws.onerror = function(error) {
                console.error("WebSocket error:", error);
            };

            // Function to generate distinct colors for each device
            function getColor(index) {
                const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];
                return colors[index % colors.length];
            }
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

    warboy = await WARBOYDevice.create()

    try:
        while True:
            power_info, util_info, temper_info, time_count, devices = await warboy()

            # Debug logging for server-side output
            # logger.info(f"Time Count: {time_count}")
            # logger.info(f"Devices: {devices}")
            # logger.info(f"Power Info: {power_info}")
            # logger.info(f"Utilization Info: {util_info}")
            # logger.info(f"Temperature Info: {temper_info}")
            

            if util_info and devices:
                await websocket.send_json({
                    "time_count": time_count,
                    "devices": devices,
                    "util_info": util_info
                })
            else:
                # logger.warning("No valid device data available; sending default values.")
                await websocket.send_json({
                    "time_count": time_count,
                    "devices": ["No Data"],
                    "util_info": [0]  # Default value for missing utilization
                });

            await asyncio.sleep(0.1)  # 100ms interval

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
