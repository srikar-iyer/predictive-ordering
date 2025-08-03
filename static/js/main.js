/**
 * Main JavaScript file for the Predictive Ordering System Flask app
 */

// Helper function for making API calls
function fetchApiData(endpoint) {
    return fetch(endpoint)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Parse the JSON string back to an object if it's a string
            if (typeof data === 'string') {
                return JSON.parse(data);
            }
            return data;
        });
}

// Function to display a Plotly chart from API data
function displayChart(endpoint, elementId, loadingMessage = 'Loading chart...') {
    // Show loading indicator
    document.getElementById(elementId).innerHTML = `
        <div class="loading-container">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">${loadingMessage}</span>
            </div>
        </div>
    `;
    
    // Fetch and display the chart
    fetchApiData(endpoint)
        .then(chartData => {
            // Clear loading indicator
            document.getElementById(elementId).innerHTML = '';
            
            // Set larger size for all charts
            if (!chartData.layout.height) {
                chartData.layout.height = 800;
            }
            if (!chartData.layout.width) {
                chartData.layout.width = 1200;
            }
            
            // Improve legend positioning and appearance
            if (!chartData.layout.legend || !chartData.layout.legend.orientation) {
                chartData.layout.legend = {
                    orientation: "h",
                    yanchor: "bottom",
                    y: -0.2,
                    xanchor: "center",
                    x: 0.5,
                    bgcolor: "rgba(255,255,255,0.8)",
                    bordercolor: "rgba(0,0,0,0.1)",
                    borderwidth: 1,
                    font: { size: 12 }
                };
            }
            
            // Add scroll feature to container
            const container = document.getElementById(elementId);
            container.style.overflowX = 'auto';
            container.style.overflowY = 'auto';
            
            // Render the chart
            Plotly.newPlot(elementId, chartData.data, chartData.layout);
        })
        .catch(error => {
            console.error(`Error loading chart from ${endpoint}:`, error);
            document.getElementById(elementId).innerHTML = `
                <div class="alert alert-danger">
                    Error loading chart data: ${error.message}
                </div>
            `;
        });
}

// Function to export chart data to CSV
function exportToCsv(chartId, filename) {
    const chart = document.getElementById(chartId);
    if (!chart || !chart.data) {
        alert('No chart data available to export');
        return;
    }
    
    // Construct CSV content
    let csvContent = 'data:text/csv;charset=utf-8,';
    
    // Add headers
    const headers = [];
    chart.data.forEach(trace => {
        headers.push(trace.name || 'Series');
    });
    csvContent += headers.join(',') + '\n';
    
    // Add data
    const maxLength = Math.max(...chart.data.map(trace => trace.y.length));
    for (let i = 0; i < maxLength; i++) {
        const row = [];
        chart.data.forEach(trace => {
            row.push(trace.y[i] !== undefined ? trace.y[i] : '');
        });
        csvContent += row.join(',') + '\n';
    }
    
    // Create download link
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `${filename}.csv`);
    document.body.appendChild(link);
    
    // Trigger download
    link.click();
    
    // Clean up
    document.body.removeChild(link);
}

// Add event listener for filter forms
document.addEventListener('DOMContentLoaded', function() {
    // Find all filter forms and add submit event listener
    const filterForms = document.querySelectorAll('.control-panel');
    filterForms.forEach(form => {
        const applyButton = form.querySelector('#apply-filters');
        if (applyButton) {
            applyButton.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Get filter values
                const storeSelect = form.querySelector('#store-select');
                const productSelect = form.querySelector('#product-select');
                const startDate = form.querySelector('#start-date');
                const endDate = form.querySelector('#end-date');
                
                // Build query parameters
                const params = new URLSearchParams();
                if (storeSelect && storeSelect.value !== 'all') {
                    params.append('store', storeSelect.value);
                }
                if (productSelect && productSelect.value !== 'all') {
                    params.append('product', productSelect.value);
                }
                if (startDate && startDate.value) {
                    params.append('start_date', startDate.value);
                }
                if (endDate && endDate.value) {
                    params.append('end_date', endDate.value);
                }
                
                // Refresh charts on the page
                const charts = document.querySelectorAll('.chart-container');
                charts.forEach(chart => {
                    const chartId = chart.id;
                    if (chartId) {
                        const endpoint = `/api/${chartId.replace('-chart', '')}-data`;
                        displayChart(`${endpoint}?${params.toString()}`, chartId);
                    }
                });
            });
        }
    });
    
    // Toggle confidence intervals on forecast charts
    const toggleConfidence = document.getElementById('toggle-confidence');
    if (toggleConfidence) {
        toggleConfidence.addEventListener('change', function() {
            const chartId = 'forecast-chart';
            const chart = document.getElementById(chartId);
            if (chart && chart.data && chart.data.length >= 3) {
                const visibility = this.checked ? true : 'legendonly';
                Plotly.restyle(chartId, {'visible': visibility}, [1, 2]);
            }
        });
    }
    
    // Toggle actual sales on forecast charts
    const toggleActual = document.getElementById('toggle-actual');
    if (toggleActual) {
        toggleActual.addEventListener('change', function() {
            const chartId = 'forecast-chart';
            const chart = document.getElementById(chartId);
            if (chart && chart.data && chart.data.length >= 4) {
                const visibility = this.checked ? true : 'legendonly';
                Plotly.restyle(chartId, {'visible': visibility}, [3]);
            }
        });
    }
    
    // Set up export buttons
    const exportButtons = document.querySelectorAll('[id^="export-"][id$="-png"]');
    exportButtons.forEach(button => {
        button.addEventListener('click', function() {
            const chartId = this.id.replace('export-', '').replace('-png', '-chart');
            const chart = document.getElementById(chartId);
            if (chart) {
                Plotly.downloadImage(chartId, {
                    format: 'png', 
                    filename: chartId.replace('-chart', '')
                });
            }
        });
    });
    
    const csvExportButtons = document.querySelectorAll('[id^="export-"][id$="-csv"]');
    csvExportButtons.forEach(button => {
        button.addEventListener('click', function() {
            const chartId = this.id.replace('export-', '').replace('-csv', '-chart');
            const filename = chartId.replace('-chart', '');
            exportToCsv(chartId, filename);
        });
    });
});

// Function to update KPI indicators
function updateKpiValues(kpiData) {
    Object.keys(kpiData).forEach(kpiId => {
        const element = document.getElementById(kpiId);
        if (element) {
            element.textContent = kpiData[kpiId];
            
            // Update color classes if needed
            const value = parseFloat(kpiData[kpiId]);
            if (!isNaN(value)) {
                // Remove existing classes
                element.classList.remove('positive-impact', 'negative-impact', 'neutral-impact');
                
                // Add appropriate class based on value
                if (value > 0) {
                    element.classList.add('positive-impact');
                } else if (value < 0) {
                    element.classList.add('negative-impact');
                } else {
                    element.classList.add('neutral-impact');
                }
            }
        }
    });
}

// Function to create table rows from data
function populateTable(tableId, data, columns, formatter = null) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    table.innerHTML = '';
    
    data.forEach(item => {
        const row = document.createElement('tr');
        
        columns.forEach(col => {
            const cell = document.createElement('td');
            
            if (formatter && formatter[col]) {
                cell.innerHTML = formatter[col](item[col]);
            } else {
                cell.textContent = item[col];
            }
            
            row.appendChild(cell);
        });
        
        table.appendChild(row);
    });
}