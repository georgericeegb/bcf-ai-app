// parcel-scoring.js - Separate module for deterministic parcel scoring
class ParcelScoring {
    constructor(projectType = 'solar') {
        this.projectType = projectType;
        
        // Deterministic scoring thresholds
        this.thresholds = {
            slope: {
                excellent: 5,    // <= 5 degrees
                good: 12,        // <= 12 degrees  
                fair: 20         // <= 20 degrees
                // > 20 degrees = poor
            },
            transmission: {
                excellent: 0.5,  // <= 0.5 miles
                good: 2,         // <= 2 miles
                fair: 5          // <= 5 miles
                // > 5 miles = poor
            },
            acreage: {
                solar: { min: 5, optimal: 50 },
                wind: { min: 50, optimal: 200 },
                battery: { min: 1, optimal: 10 },
                mixed: { min: 5, optimal: 50 }
            }
        };
    }

    /**
     * Calculate deterministic parcel category (Red/Yellow/Green system)
     * @param {Object} parcel - Parcel data object
     * @returns {string} - 'Excellent' (Green), 'Good' (Yellow), 'Fair' (Orange), 'Poor' (Red)
     */
    calculateParcelCategory(parcel) {
        const factors = this.evaluateAllFactors(parcel);
        
        // Deterministic logic: must pass ALL factors to be excellent
        if (factors.slope === 'excellent' && 
            factors.transmission === 'excellent' && 
            factors.acreage === 'excellent') {
            return 'Excellent';
        }
        
        // Good: mostly good factors, no poor factors
        const goodCount = Object.values(factors).filter(f => f === 'excellent' || f === 'good').length;
        const poorCount = Object.values(factors).filter(f => f === 'poor').length;
        
        if (goodCount >= 2 && poorCount === 0) {
            return 'Good';
        }
        
        // Fair: mixed factors
        if (poorCount <= 1) {
            return 'Fair';
        }
        
        // Poor: multiple poor factors
        return 'Poor';
    }

    /**
     * Evaluate all factors for a parcel
     * @param {Object} parcel - Parcel data
     * @returns {Object} - Factor evaluations
     */
    evaluateAllFactors(parcel) {
        return {
            slope: this.evaluateSlope(parcel),
            transmission: this.evaluateTransmission(parcel),
            acreage: this.evaluateAcreage(parcel)
        };
    }

    /**
     * Evaluate slope factor
     * @param {Object} parcel - Parcel data
     * @returns {string} - Factor rating
     */
    evaluateSlope(parcel) {
        const slope = this.extractSlopeValue(parcel);
        
        if (slope === null) return 'unknown';
        
        if (slope <= this.thresholds.slope.excellent) return 'excellent';
        if (slope <= this.thresholds.slope.good) return 'good';
        if (slope <= this.thresholds.slope.fair) return 'fair';
        return 'poor';
    }

    /**
     * Evaluate transmission distance factor
     * @param {Object} parcel - Parcel data
     * @returns {string} - Factor rating
     */
    evaluateTransmission(parcel) {
        const distance = this.extractTransmissionDistance(parcel);
        
        if (distance === null) return 'unknown';
        
        if (distance <= this.thresholds.transmission.excellent) return 'excellent';
        if (distance <= this.thresholds.transmission.good) return 'good';
        if (distance <= this.thresholds.transmission.fair) return 'fair';
        return 'poor';
    }

    /**
     * Evaluate acreage factor
     * @param {Object} parcel - Parcel data
     * @returns {string} - Factor rating
     */
    evaluateAcreage(parcel) {
        const acres = this.extractAcreageValue(parcel);
        const thresholds = this.thresholds.acreage[this.projectType] || this.thresholds.acreage.mixed;
        
        if (acres === null || acres <= 0) return 'unknown';
        
        if (acres < thresholds.min) return 'poor';
        if (acres >= thresholds.optimal) return 'excellent';
        
        // Scale between min and optimal
        const ratio = (acres - thresholds.min) / (thresholds.optimal - thresholds.min);
        return ratio >= 0.7 ? 'good' : 'fair';
    }

    /**
     * Extract slope value with multiple field name fallbacks
     * @param {Object} parcel - Parcel data
     * @returns {number|null} - Slope in degrees
     */
    extractSlopeValue(parcel) {
        const slopeFields = [
            'avg_slope_degrees',
            'avg_slope', 
            'slope_degrees',
            'slope',
            'average_slope'
        ];
        
        for (const field of slopeFields) {
            const value = parseFloat(parcel[field]);
            if (!isNaN(value)) return value;
        }
        
        return null;
    }

    /**
     * Extract transmission distance with multiple field name fallbacks
     * @param {Object} parcel - Parcel data
     * @returns {number|null} - Distance in miles
     */
    extractTransmissionDistance(parcel) {
        const distanceFields = [
            'tx_distance_miles',
            'tx_nearest_distance_miles',
            'transmission_distance',
            'tx_distance',
            'nearest_transmission_miles'
        ];
        
        for (const field of distanceFields) {
            const value = parseFloat(parcel[field]);
            if (!isNaN(value)) return value;
        }
        
        return null;
    }

    /**
     * Extract acreage value with multiple field name fallbacks
     * @param {Object} parcel - Parcel data
     * @returns {number|null} - Acreage
     */
    extractAcreageValue(parcel) {
        const acreageFields = [
            'acreage_calc',
            'acreage',
            'calculated_acreage',
            'acres',
            'size_acres'
        ];
        
        for (const field of acreageFields) {
            const value = parseFloat(parcel[field]);
            if (!isNaN(value) && value > 0) return value;
        }
        
        return null;
    }

    /**
     * Get CSS class for category
     * @param {string} category - Category name
     * @returns {string} - CSS class
     */
    getCategoryClass(category) {
        const classes = {
            'Excellent': 'category-excellent',
            'Good': 'category-good', 
            'Fair': 'category-fair',
            'Poor': 'category-poor',
            'Unknown': 'category-unknown'
        };
        
        return classes[category] || 'category-unknown';
    }

    /**
     * Get badge class for category
     * @param {string} category - Category name
     * @returns {string} - Badge CSS class
     */
    getBadgeClass(category) {
        const classes = {
            'Excellent': 'badge bg-success',
            'Good': 'badge bg-primary',
            'Fair': 'badge bg-warning text-dark',
            'Poor': 'badge bg-danger',
            'Unknown': 'badge bg-secondary'
        };
        
        return classes[category] || 'badge bg-secondary';
    }

    /**
     * Check if parcel is recommended for outreach
     * @param {Object} parcel - Parcel data
     * @returns {boolean} - Recommendation status
     */
    isRecommendedForOutreach(parcel) {
        // Check explicit recommendation first
        if (parcel.recommended_for_outreach !== undefined) {
            return Boolean(parcel.recommended_for_outreach);
        }
        
        // Calculate based on category
        const category = this.calculateParcelCategory(parcel);
        return category === 'Excellent' || category === 'Good';
    }

    /**
     * Generate summary statistics for parcels
     * @param {Array} parcels - Array of parcel objects
     * @returns {Object} - Summary statistics
     */
    generateSummaryStats(parcels) {
        const stats = {
            total: parcels.length,
            excellent: 0,
            good: 0,
            fair: 0,
            poor: 0,
            unknown: 0,
            recommended: 0
        };

        parcels.forEach(parcel => {
            const category = this.calculateParcelCategory(parcel);
            stats[category.toLowerCase()]++;
            
            if (this.isRecommendedForOutreach(parcel)) {
                stats.recommended++;
            }
        });

        return stats;
    }

    /**
     * Process parcels and add scoring data
     * @param {Array} parcels - Array of parcel objects
     * @returns {Array} - Processed parcels with scoring data
     */
    processParcelScoring(parcels) {
        return parcels.map(parcel => {
            const category = this.calculateParcelCategory(parcel);
            const factors = this.evaluateAllFactors(parcel);
            
            return {
                ...parcel,
                suitability_category: category,
                suitability_factors: factors,
                recommended_for_outreach: this.isRecommendedForOutreach(parcel),
                category_class: this.getCategoryClass(category),
                badge_class: this.getBadgeClass(category)
            };
        });
    }
}

// Export for use in main application
if (typeof window !== 'undefined') {
    window.ParcelScoring = ParcelScoring;
}

// Example usage:
// const scorer = new ParcelScoring('solar');
// const processedParcels = scorer.processParcelScoring(rawParcels);
// const stats = scorer.generateSummaryStats(processedParcels);