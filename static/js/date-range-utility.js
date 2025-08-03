/**
 * Date Range Utility for Predictive Ordering System
 * 
 * This utility handles date range selection and initialization
 * across all pages in the application.
 */

// Default date range (last 30 days to next 30 days)
function getDefaultDateRange() {
    const today = new Date();
    
    // Start date - 30 days ago
    const startDate = new Date();
    startDate.setDate(today.getDate() - 30);
    
    // End date - 30 days in future
    const endDate = new Date();
    endDate.setDate(today.getDate() + 30);
    
    return {
        startDate: formatDate(startDate),
        endDate: formatDate(endDate)
    };
}

// Format date as YYYY-MM-DD
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// Initialize date range pickers with default values
function initDateRangePickers() {
    const dateRange = getDefaultDateRange();
    
    // Find all date pickers and set default values
    const startDateInputs = document.querySelectorAll('input[id="start-date"]');
    const endDateInputs = document.querySelectorAll('input[id="end-date"]');
    
    startDateInputs.forEach(input => {
        if (!input.value) {
            input.value = dateRange.startDate;
        }
    });
    
    endDateInputs.forEach(input => {
        if (!input.value) {
            input.value = dateRange.endDate;
        }
    });
    
    // Set up date range presets if the preset selector exists
    const datePresetSelector = document.getElementById('date-preset');
    if (datePresetSelector) {
        datePresetSelector.addEventListener('change', function() {
            applyDatePreset(this.value);
        });
    }
}

// Apply a preset date range
function applyDatePreset(preset) {
    const today = new Date();
    let startDate, endDate;
    
    switch(preset) {
        case 'last7':
            startDate = new Date();
            startDate.setDate(today.getDate() - 7);
            endDate = today;
            break;
        case 'last30':
            startDate = new Date();
            startDate.setDate(today.getDate() - 30);
            endDate = today;
            break;
        case 'last90':
            startDate = new Date();
            startDate.setDate(today.getDate() - 90);
            endDate = today;
            break;
        case 'next7':
            startDate = today;
            endDate = new Date();
            endDate.setDate(today.getDate() + 7);
            break;
        case 'next30':
            startDate = today;
            endDate = new Date();
            endDate.setDate(today.getDate() + 30);
            break;
        case 'next90':
            startDate = today;
            endDate = new Date();
            endDate.setDate(today.getDate() + 90);
            break;
        case 'full':
        default:
            const defaultRange = getDefaultDateRange();
            startDate = new Date(defaultRange.startDate);
            endDate = new Date(defaultRange.endDate);
            break;
    }
    
    // Update all date inputs with the new range
    const startDateInputs = document.querySelectorAll('input[id="start-date"]');
    const endDateInputs = document.querySelectorAll('input[id="end-date"]');
    
    startDateInputs.forEach(input => {
        input.value = formatDate(startDate);
    });
    
    endDateInputs.forEach(input => {
        input.value = formatDate(endDate);
    });
    
    // If this is triggered from a preset selector, also trigger any update functions
    const updateButton = document.querySelector('button[type="submit"]');
    if (updateButton) {
        updateButton.click();
    }
}

// Initialize date ranges when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initDateRangePickers();
});