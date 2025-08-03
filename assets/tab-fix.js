/**
 * Tab visibility enhancements for the Pizza Predictive Ordering Dashboard
 * This script provides minimal styling to ensure tab icons are properly displayed
 * while using Plotly's default UI settings
 */

// Execute when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("Tab visibility enhancement loaded");
    
    // Function to enhance tab icons and improve tab visibility
    function enhanceTabIcons() {
        // Fix tab icons to ensure they display properly
        const tabIcons = document.querySelectorAll('.nav-tabs .nav-link i, #main-tabs .nav-link i');
        tabIcons.forEach(function(icon) {
            if (icon) {
                icon.style.marginRight = '0.5rem';
                icon.style.display = 'inline-block';
            }
        });
        
        // Fix tab display issues
        const tabs = document.querySelectorAll('.nav-tabs, #main-tabs');
        tabs.forEach(function(tab) {
            tab.style.display = 'flex';
            tab.style.flexWrap = 'nowrap';
            tab.style.overflowX = 'auto';
            tab.style.width = '100%';
        });
        
        // Fix tab items
        const tabItems = document.querySelectorAll('.nav-tabs .nav-item, #main-tabs .nav-item');
        tabItems.forEach(function(item) {
            item.style.display = 'block';
            const tabLink = item.querySelector('.nav-link');
            if (tabLink) {
                tabLink.style.display = 'flex';
                tabLink.style.alignItems = 'center';
            }
        });
        
        // Fix active tab panes
        const activeTabPanes = document.querySelectorAll('.tab-pane.active');
        activeTabPanes.forEach(function(pane) {
            pane.style.display = 'block';
        });
    }
    
    // Make fixTabVisibility globally available for other scripts to call
    window.fixTabVisibility = function() {
        console.log("Global fixTabVisibility called");
        enhanceTabIcons();
        
        // Additional fixes for tab content container
        const tabContent = document.querySelector('#tab-content');
        if (tabContent) {
            tabContent.style.display = 'block';
            tabContent.style.width = '100%';
            console.log("Fixed tab content container");
        }
    }
    
    // Apply enhancements at various points to ensure they take effect
    enhanceTabIcons();
    setTimeout(enhanceTabIcons, 500);
    setTimeout(enhanceTabIcons, 2000);
    
    // Initial call to global function
    if (window.fixTabVisibility) {
        window.fixTabVisibility();
    }
    
    // Re-apply when tab elements change
    const observer = new MutationObserver(function(mutations) {
        let tabMutation = false;
        mutations.forEach(function(mutation) {
            if (mutation.target.classList && 
                (mutation.target.classList.contains('nav-tabs') || 
                 mutation.target.classList.contains('nav-item') ||
                 mutation.target.classList.contains('nav-link'))) {
                tabMutation = true;
            }
        });
        
        if (tabMutation) {
            enhanceTabIcons();
        }
    });
    
    // Watch for changes
    observer.observe(document.body, { 
        childList: true, 
        subtree: true
    });
    
    // Apply on window resize
    window.addEventListener('resize', enhanceTabIcons);
});