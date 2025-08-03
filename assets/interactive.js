/**
 * Client-side JavaScript for interactive features in the 
 * Pizza Predictive Ordering System dashboard.
 */

if (!window.dash_clientside) {
    window.dash_clientside = {};
}

// Create namespace for interactivity functions
window.dash_clientside.interactivity = {
    
    /**
     * Handle enhanced tooltips
     * @param {Object} tooltipData - Data for the tooltip
     * @returns {HTMLElement} - HTML element for the tooltip
     */
    handleTooltip: function(tooltipData) {
        if (!tooltipData) return null;
        
        try {
            // Parse tooltip data
            const data = JSON.parse(tooltipData);
            
            // Create enhanced tooltip
            const tooltip = document.createElement('div');
            tooltip.className = 'enhanced-tooltip';
            // Use enhanced-tooltip class instead of inline styles
            tooltip.className = 'enhanced-tooltip';
            tooltip.style.left = `${data.x + 10}px`;
            tooltip.style.top = `${data.y + 10}px`;
            
            // Add content
            if (data.title) {
                const title = document.createElement('div');
                title.className = 'tooltip-title';
                title.innerHTML = data.title;
                tooltip.appendChild(title);
            }
            
            // Add data points
            if (data.points && data.points.length > 0) {
                const content = document.createElement('div');
                
                data.points.forEach(point => {
                    const row = document.createElement('div');
                    row.className = 'tooltip-row';
                    
                    const label = document.createElement('span');
                    label.className = 'tooltip-label';
                    label.innerHTML = point.label + ':';
                    
                    const value = document.createElement('span');
                    value.innerHTML = point.value;
                    
                    row.appendChild(label);
                    row.appendChild(value);
                    content.appendChild(row);
                });
                
                tooltip.appendChild(content);
            }
            
            return tooltip;
        } catch (e) {
            console.error("Error handling tooltip:", e);
            return null;
        }
    },
    
    /**
     * Handle drill down clicks on charts
     * @param {string} chartClickData - JSON string with chart click data
     * @returns {Object} - Processing result
     */
    handleDrillDown: function(chartClickData) {
        if (!chartClickData) return {processed: false};
        
        try {
            // Parse chart click data
            const clickData = JSON.parse(chartClickData);
            if (!clickData || !clickData.points || clickData.points.length === 0) {
                return {processed: false};
            }
            
            // Update the drill-down-data store
            const drillDownData = {
                chart_type: clickData.chartType || "Unknown Chart",
                point_data: clickData.points[0]
            };
            
            // Store the data in the hidden div
            const drillDownStore = document.getElementById('drill-down-data');
            if (drillDownStore) {
                // Store data in the proper dataset attribute
                drillDownStore.dataset.data = JSON.stringify(drillDownData);
                
                // Update the data property for Dash callbacks
                if ('__component' in drillDownStore && drillDownStore.__component.setProps) {
                    drillDownStore.__component.setProps({data: JSON.stringify(drillDownData)});
                }
                
                // Trigger click on hidden button to open the modal
                const drillDownTrigger = document.getElementById('drill-down-trigger');
                if (drillDownTrigger) {
                    drillDownTrigger.click();
                }
            }
            
            return {processed: true};
        } catch (e) {
            console.error("Error processing drill-down:", e);
            return {processed: false, error: e.toString()};
        }
    },
    
    /**
     * Synchronize zoom between linked charts
     * @param {string} zoomData - JSON string with zoom event data
     * @returns {Object} - Processing result
     */
    syncZoom: function(zoomData) {
        if (!zoomData) return {processed: false};
        
        try {
            // Parse zoom event data
            const zoomEvent = JSON.parse(zoomData);
            if (!zoomEvent || !zoomEvent.sourceId) {
                return {processed: false};
            }
            
            const sourceId = zoomEvent.sourceId;
            const xRange = zoomEvent.xRange;
            const yRange = zoomEvent.yRange;
            
            // Find all connected charts
            if (zoomEvent.linkedCharts && zoomEvent.linkedCharts.length > 0) {
                zoomEvent.linkedCharts.forEach(chartId => {
                    const chartElement = document.getElementById(chartId);
                    if (!chartElement) return;
                    
                    // For Plotly charts, make sure they are fully initialized
                    if (!chartElement._fullLayout || !chartElement._fullData) {
                        console.log(`Chart ${chartId} not fully initialized, skipping sync operation`);
                        return;
                    }
                    
                    // Apply zoom to connected chart
                    if (chartElement._fullLayout) {
                        // Only sync x-axis for time series charts
                        if (zoomEvent.syncX) {
                            Plotly.relayout(chartId, {
                                'xaxis.range': xRange
                            });
                        }
                        
                        // Sync both axes for non-time series charts if specified
                        if (zoomEvent.syncY) {
                            Plotly.relayout(chartId, {
                                'yaxis.range': yRange
                            });
                        }
                    }
                });
            }
            
            return {processed: true};
        } catch (e) {
            console.error("Error synchronizing zoom:", e);
            return {processed: false, error: e.toString()};
        }
    },
    
    /**
     * Apply crossfiltering between charts
     * @param {string} filterData - JSON string with filter event data
     * @returns {Object} - Processing result
     */
    applyCrossfilter: function(filterData) {
        if (!filterData) return {processed: false};
        
        try {
            // Parse filter event data
            const filterEvent = JSON.parse(filterData);
            if (!filterEvent || !filterEvent.sourceId || !filterEvent.selectedData) {
                return {processed: false};
            }
            
            const sourceId = filterEvent.sourceId;
            const selectedData = filterEvent.selectedData;
            
            // Apply filter to connected charts
            if (filterEvent.linkedCharts && filterEvent.linkedCharts.length > 0) {
                filterEvent.linkedCharts.forEach(chartId => {
                    const chartElement = document.getElementById(chartId);
                    if (!chartElement) return;
                    
                    // For Plotly charts, make sure they are fully initialized
                    if (!chartElement._fullLayout || !chartElement._fullData) {
                        console.log(`Chart ${chartId} not fully initialized, skipping filter operation`);
                        return;
                    }
                    
                    // Extract dimension values from selected data
                    let dimensionValues = [];
                    if (filterEvent.dimension === 'x') {
                        dimensionValues = selectedData.points.map(p => p.x);
                    } else if (filterEvent.dimension === 'y') {
                        dimensionValues = selectedData.points.map(p => p.y);
                    } else if (filterEvent.dimension === 'customdata') {
                        // Handle filtering by custom data dimensions
                        dimensionValues = selectedData.points.map(p => 
                            p.customdata ? p.customdata[filterEvent.customdataIndex || 0] : null
                        );
                    }
                    
                    // Apply filter to this chart
                    for (let i = 0; i < chartElement._fullData.length; i++) {
                        const trace = chartElement._fullData[i];
                        if (!trace.x || !trace.y) continue;
                        
                        // Create a mask of points to highlight
                        const mask = trace.x.map((x, idx) => {
                            let value;
                            
                            if (filterEvent.dimension === 'x') {
                                value = x;
                            } else if (filterEvent.dimension === 'y') {
                                value = trace.y[idx];
                            } else if (filterEvent.dimension === 'customdata') {
                                value = trace.customdata ? trace.customdata[idx][filterEvent.customdataIndex || 0] : null;
                            }
                            
                            return dimensionValues.includes(value);
                        });
                        
                        // Update trace opacity
                        Plotly.restyle(chartId, {
                            'opacity': mask.map(m => m ? 1.0 : 0.3),
                            'selectedpoints': mask.map((m, i) => m ? i : null).filter(i => i !== null)
                        }, [i]);
                    }
                });
            }
            
            return {processed: true};
        } catch (e) {
            console.error("Error applying crossfilter:", e);
            return {processed: false, error: e.toString()};
        }
    },
    
    /**
     * Set up event listeners for all interactive charts
     * This function will be called when the page loads
     */
    setupChartListeners: function() {
        // Set up MutationObserver to detect new charts and chart updates
        if (!window._chartObserver) {
            const observerCallback = (mutations) => {
                // Check if mutations include plotly charts
                let hasChartChanges = false;
                for (const mutation of mutations) {
                    if (mutation.addedNodes.length) {
                        for (const node of mutation.addedNodes) {
                            if (node.nodeType === Node.ELEMENT_NODE && 
                                (node.classList.contains('js-plotly-plot') ||
                                 node.querySelector('.js-plotly-plot'))) {
                                hasChartChanges = true;
                                break;
                            }
                        }
                    } else if (mutation.type === 'attributes' && 
                               mutation.target.classList.contains('js-plotly-plot')) {
                        hasChartChanges = true;
                        break;
                    }
                }
                
                // If chart changes detected, attach event handlers
                if (hasChartChanges) {
                    console.log('Chart changes detected, reattaching event handlers...');
                    window.dash_clientside.interactivity._setupAllChartHandlers();
                }
            };
            
            // Create and start the observer
            window._chartObserver = new MutationObserver(observerCallback);
            window._chartObserver.observe(document.body, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['data-dash-is-loading']
            });
            
            console.log('Chart observer initialized');
        }
        
        // Initial setup of chart handlers
        this._setupAllChartHandlers();
        
        // Also add a Plotly.afterPlot callback for charts that are already loaded
        if (window.Plotly && typeof window.Plotly.afterPlot === 'function') {
            const existingAfterPlot = window.Plotly.afterPlot;
            window.Plotly.afterPlot = function() {
                existingAfterPlot.apply(this, arguments);
                window.dash_clientside.interactivity._setupAllChartHandlers();
            };
            console.log('Plotly.afterPlot hook added');
        }
        
        return null;
    },
    
    /**
     * Set up event handlers for all charts
     * Private method used by setupChartListeners
     */
    _setupAllChartHandlers: function() {
        if (window._setupAllChartHandlersRunning) {
            console.log('Setup already in progress, skipping this call');
            return;
        }
        
        window._setupAllChartHandlersRunning = true;
        setTimeout(() => { window._setupAllChartHandlersRunning = false; }, 500);
        
        // Find all Plotly chart elements
        const chartElements = document.querySelectorAll('.js-plotly-plot');
        console.log(`Setting up handlers for ${chartElements.length} charts`);
        
        chartElements.forEach(chartElement => {
                const chartId = chartElement.id;
                if (!chartId) return;
                
                // Get chart object
                const chart = document.getElementById(chartId);
                if (!chart) return;
                
                // Wait for Plotly chart to be fully initialized
                if (!chart._fullLayout) {
                    console.log(`Chart ${chartId} is not fully initialized yet, will retry later`);
                    setTimeout(() => {
                        if (window.dash_clientside && window.dash_clientside.interactivity) {
                            window.dash_clientside.interactivity._setupAllChartHandlers();
                        }
                    }, 200);
                    return;
                }
                
                // Extract metadata if available
                let metadata = null;
                try {
                    // Try different ways to access metadata, as Plotly.js might store it differently
                    if (chart.layout && chart.layout.metadata) {
                        metadata = chart.layout.metadata;
                    } else if (chart._fullLayout && chart._fullLayout.metadata) {
                        metadata = chart._fullLayout.metadata;
                    } else if (chart.data && chart.data[0] && chart.data[0].metadata) {
                        metadata = chart.data[0].metadata;
                    }
                } catch (e) {
                    console.warn(`Could not extract metadata for chart ${chartId}:`, e);
                }
                
                // Set up click listeners for drill-down
                if (metadata && metadata.drill_down) {
                    let drillDownData;
                    
                    try {
                        // Handle both string JSON and object formats
                        if (typeof metadata.drill_down === 'string') {
                            drillDownData = JSON.parse(metadata.drill_down);
                        } else {
                            drillDownData = metadata.drill_down;
                        }
                    } catch (e) {
                        console.warn(`Invalid drill-down metadata for chart ${chartId}:`, e);
                    }
                    
                    if (drillDownData) {
                        chart.on('plotly_click', function(data) {
                            // Store click data for server-side processing
                            const clickData = {
                                chartType: drillDownData.chart_title || chartId,
                                points: data.points.map(pt => {
                                    const pointData = {
                                        x: pt.x,
                                        y: pt.y,
                                        curveNumber: pt.curveNumber,
                                        pointNumber: pt.pointNumber,
                                        data: pt.data
                                    };
                                    
                                    // Add customdata if available
                                    if (pt.customdata) {
                                        pointData.customdata = pt.customdata;
                                    }
                                    
                                    // Add text if available
                                    if (pt.text) {
                                        pointData.text = pt.text;
                                    }
                                    
                                    return pointData;
                                })
                            };
                            
                            // Update drill-down data
                            const drillDownStore = document.getElementById('drill-down-data');
                            if (drillDownStore) {
                                drillDownStore.dataset.data = JSON.stringify(clickData);
                                
                                // Update the data property for Dash callbacks
                                if ('__component' in drillDownStore && drillDownStore.__component.setProps) {
                                    drillDownStore.__component.setProps({data: JSON.stringify(clickData)});
                                }
                            }
                        });
                    }
                }
                
                // Set up crossfilter listeners
                if (metadata && metadata.crossfilter) {
                    let crossfilterData;
                    
                    try {
                        crossfilterData = metadata.crossfilter;
                    } catch (e) {
                        console.warn(`Invalid crossfilter metadata for chart ${chartId}:`, e);
                    }
                    
                    if (crossfilterData) {
                        chart.on('plotly_selected', function(eventData) {
                            if (!eventData || !eventData.points || eventData.points.length === 0) {
                                return;
                            }
                            
                            // Determine filter dimension
                            let dimension = 'x';  // Default to x dimension
                            if (crossfilterData.filter_dimensions) {
                                dimension = Object.keys(crossfilterData.filter_dimensions)[0] || 'x';
                            }
                            
                            // Create filter event data
                            const filterEvent = {
                                sourceId: chartId,
                                dimension: dimension,
                                linkedCharts: crossfilterData.linked_charts || [],
                                selectedData: {
                                    points: eventData.points.map(pt => {
                                        const pointData = {
                                            x: pt.x,
                                            y: pt.y,
                                            curveNumber: pt.curveNumber,
                                            pointNumber: pt.pointNumber
                                        };
                                        
                                        // Add customdata if available
                                        if (pt.customdata) {
                                            pointData.customdata = pt.customdata;
                                        }
                                        
                                        return pointData;
                                    })
                                }
                            };
                            
                            // Update filter event data
                            const filterStore = document.getElementById('filter-event-data');
                            if (filterStore) {
                                filterStore.dataset.data = JSON.stringify(filterEvent);
                                
                                // Update the data property for Dash callbacks
                                if ('__component' in filterStore && filterStore.__component.setProps) {
                                    filterStore.__component.setProps({data: JSON.stringify(filterEvent)});
                                }
                            }
                        });
                    }
                }
                
                // Set up zoom listeners for zoom synchronization
                chart.on('plotly_relayout', function(eventData) {
                    // Check if this is a zoom event
                    const isZoomEvent = eventData['xaxis.range'] || eventData['xaxis.range[0]'] || 
                                       eventData['yaxis.range'] || eventData['yaxis.range[0]'];
                    
                    if (!isZoomEvent) {
                        return;
                    }
                    
                    // Extract zoom ranges
                    let xRange = null;
                    let yRange = null;
                    
                    if (eventData['xaxis.range']) {
                        xRange = eventData['xaxis.range'];
                    } else if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {
                        xRange = [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']];
                    }
                    
                    if (eventData['yaxis.range']) {
                        yRange = eventData['yaxis.range'];
                    } else if (eventData['yaxis.range[0]'] && eventData['yaxis.range[1]']) {
                        yRange = [eventData['yaxis.range[0]'], eventData['yaxis.range[1]']];
                    }
                    
                    // Get metadata for zoom sync
                    let linkedCharts = [];
                    let syncX = true;
                    let syncY = false;
                    
                    if (metadata && metadata.zoom_sync) {
                        linkedCharts = metadata.zoom_sync.linked_charts || [];
                        syncX = metadata.zoom_sync.sync_x !== false;
                        syncY = metadata.zoom_sync.sync_y === true;
                    }
                    
                    // Create zoom event data
                    const zoomEvent = {
                        sourceId: chartId,
                        xRange: xRange,
                        yRange: yRange,
                        linkedCharts: linkedCharts,
                        syncX: syncX,
                        syncY: syncY
                    };
                    
                    // Update zoom event data
                    const zoomStore = document.getElementById('zoom-event-data');
                    if (zoomStore) {
                        zoomStore.dataset.data = JSON.stringify(zoomEvent);
                        
                        // Update the data property for Dash callbacks
                        if ('__component' in zoomStore && zoomStore.__component.setProps) {
                            zoomStore.__component.setProps({data: JSON.stringify(zoomEvent)});
                        }
                    }
                });
            });
            // Mark this chart as having handlers set up
            const currentVersion = chart.dataset.handlerVersion || '0';
            const newVersion = (parseInt(currentVersion) + 1).toString();
            chart.dataset.handlerVersion = newVersion;
            console.log(`Successfully set up handlers for chart ${chartId} (version ${newVersion})`);
        });
        
        return null;
    }
};

// Set up listeners when page loads
document.addEventListener('DOMContentLoaded', function() {
    if (window.dash_clientside && window.dash_clientside.interactivity) {
        console.log("Initializing interactive features...");
        window.dash_clientside.interactivity.setupChartListeners();
        
        // Immediate first fix for tab visibility issues
        console.log("Initial check for tab visibility issues...");
        fixTabVisibility();
        
        // Apply fix again after short delay to catch dynamically loaded tabs
        setTimeout(function() {
            console.log("Second check for tab visibility issues...");
            fixTabVisibility();
        }, 500);
        
        // Also setup listeners on page content changes (SPA navigation)
        // Use a more efficient MutationObserver instead of DOMSubtreeModified
        const tabObserver = new MutationObserver(function(mutations) {
            // Check if mutations include tab-related elements
            let hasTabChanges = false;
            for (const mutation of mutations) {
                if (mutation.addedNodes.length) {
                    for (const node of mutation.addedNodes) {
                        if (node.nodeType === Node.ELEMENT_NODE && 
                            (node.classList && (node.classList.contains('nav-tabs') || 
                             node.classList.contains('nav-item') || 
                             node.classList.contains('docker-tabs') ||
                             node.querySelector('.nav-tabs, .nav-item, .docker-tabs')))) {
                            hasTabChanges = true;
                            break;
                        }
                    }
                }
                
                if (hasTabChanges) break;
            }
            
            // If tab changes detected, fix visibility
            if (hasTabChanges) {
                console.log("Tab changes detected, fixing visibility...");
                fixTabVisibility();
            }
        });
        
        // Start observing the document for tab changes
        tabObserver.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['style', 'class']
        });
        
        // Apply multiple times to catch all edge cases
        const checkIntervals = [1000, 2000, 3000, 5000];
        checkIntervals.forEach(interval => {
            setTimeout(function() {
                console.log(`Scheduled check for tab visibility at ${interval}ms...`);
                fixTabVisibility();
                
                // Set up chart handlers at these intervals too
                if (window.dash_clientside && window.dash_clientside.interactivity) {
                    window.dash_clientside.interactivity._setupAllChartHandlers();
                }
                
                // Set up export buttons if available
                if (window.dash_clientside && window.dash_clientside.export && 
                    window.dash_clientside.export.setupExportButtons) {
                    window.dash_clientside.export.setupExportButtons();
                }
            }, interval);
        });
    }
});

// Centralized function to fix tab visibility
function fixTabVisibility() {
    // Fix tab items
    const tabElements = document.querySelectorAll('.docker-tabs .nav-item, .nav-tabs .nav-item, .tab-content .tab-pane');
    if (tabElements.length > 0) {
        console.log(`Found ${tabElements.length} tab elements, fixing visibility...`);
        tabElements.forEach(function(tab) {
            // Fix nav items
            if (tab.classList.contains('nav-item')) {
                tab.style.display = 'block';
                const tabLink = tab.querySelector('.nav-link');
                if (tabLink) {
                    tabLink.style.display = 'flex';
                    tabLink.style.alignItems = 'center';
                }
            }
            
            // Fix tab panes
            if (tab.classList.contains('tab-pane')) {
                // Only show if active
                if (tab.classList.contains('active')) {
                    tab.style.display = 'block';
                }
            }
        });
    }
    
    // Force re-render of tab container
    const tabContainers = document.querySelectorAll('.docker-tabs, .nav-tabs, #main-tabs');
    if (tabContainers.length > 0) {
        console.log(`Found ${tabContainers.length} tab containers, fixing layout...`);
        tabContainers.forEach(function(tabContainer) {
            tabContainer.style.display = 'flex';
            tabContainer.style.flexWrap = 'nowrap';
            tabContainer.style.overflowX = 'auto';
            tabContainer.style.width = '100%';
        });
    }
    
    // Fix the content containers
    const tabContent = document.querySelectorAll('.tab-content');
    if (tabContent.length > 0) {
        console.log(`Found ${tabContent.length} tab content containers, fixing display...`);
        tabContent.forEach(function(content) {
            content.style.display = 'block';
            content.style.width = '100%';
        });
    }
    
    // Force visibility of current active tab
    const activeTabs = document.querySelectorAll('.tab-pane.active');
    activeTabs.forEach(function(activeTab) {
        activeTab.style.display = 'block';
        activeTab.style.width = '100%';
        console.log(`Ensuring active tab is visible: ${activeTab.id}`);
        
        // Ensure charts inside tab are visible and properly sized
        const charts = activeTab.querySelectorAll('.js-plotly-plot');
        charts.forEach(function(chart) {
            if (chart && chart._fullData && chart._fullLayout && typeof Plotly !== 'undefined') {
                console.log(`Resizing chart in active tab: ${chart.id}`);
                setTimeout(function() {
                    Plotly.relayout(chart.id, {
                        autosize: true
                    });
                }, 100);
            }
        });
    });
}