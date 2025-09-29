// individual-ai-analysis.js - Separate module for individual parcel AI analysis
class IndividualParcelAI {
    constructor(apiBaseUrl = '', projectType = 'solar', currentState = '', currentCounty = '') {
        this.apiBaseUrl = apiBaseUrl;
        this.projectType = projectType;
        this.currentState = currentState;
        this.currentCounty = currentCounty;
    }

    /**
     * Run AI analysis for individual parcel
     * @param {Object} parcel - Parcel data
     * @param {Function} showModal - Function to show modal
     * @param {Function} showLoading - Function to show loading
     * @param {Function} hideLoading - Function to hide loading
     */
    async analyzeParcel(parcel, showModal, showLoading, hideLoading) {
        try {
            showLoading(`Running AI analysis for ${parcel.owner || 'Unknown Owner'}...`);
    
            // Call the API endpoint
            const response = await fetch('/api/analysis/individual-parcel-analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    parcel_data: parcel,
                    county_name: this.currentCounty,
                    state: this.currentState,
                    project_type: this.projectType
                })
            });
    
            hideLoading();
    
            if (response.ok) {
                const result = await response.json();
                
                if (result.success && result.analysis && result.analysis.detailed_analysis) {
                    console.log('✅ Using REAL AI-powered analysis');
                    
                    // Show the actual AI analysis content
                    this.displayAIResults(parcel, result.analysis, showModal);
                    return;
                } else {
                    console.log('❌ API returned success=false or missing analysis');
                }
            } else {
                console.log('❌ API request failed:', response.status);
            }
            
        } catch (error) {
            hideLoading();
            console.log('❌ API request error:', error.message);
        }
        
        // Only show fallback if API completely fails
        console.log('📋 Falling back to enhanced analysis');
        this.displayEnhancedAnalysis(parcel, showModal);
    }
        
    /**
     * Display AI-powered results modal
     * @param {Object} parcel - Parcel data
     * @param {Object} analysis - AI analysis results
     * @param {Function} showModal - Function to show modal
     */
    displayAIResults(parcel, analysis, showModal) {
        const assessment = this.calculateParcelAssessment(parcel);
        
        const modalContent = `
            <div class="ai-analysis-container">
                <!-- Score Dashboard -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card text-center border-primary">
                            <div class="card-body">
                                <h2 class="text-primary">${assessment.category}</h2>
                                <p class="mb-0">Overall Rating</p>
                                <span class="badge ${assessment.badgeClass}">${assessment.category}</span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center border-success">
                            <div class="card-body">
                                <h3 class="text-success">${Math.round(parseFloat(parcel.acreage_calc || parcel.acreage || 0))}</h3>
                                <p class="mb-0">Acres</p>
                                <small class="text-muted">${assessment.projectScale}</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center border-info">
                            <div class="card-body">
                                <h3 class="text-info">${this.formatSlope(parcel.avg_slope_degrees)}</h3>
                                <p class="mb-0">Average Slope</p>
                                <small class="text-muted">${assessment.slopeRating}</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center border-warning">
                            <div class="card-body">
                                <h3 class="text-warning">${this.formatDistance(parcel.tx_distance_miles)}</h3>
                                <p class="mb-0">Grid Distance</p>
                                <small class="text-muted">${assessment.gridRating}</small>
                            </div>
                        </div>
                    </div>
                </div>
    
                <!-- FIXED: AI Analysis Content with proper formatting and scrolling -->
                <div class="card mb-3">
                    <div class="card-header bg-primary text-white">
                        <h6 class="mb-0"><i class="fas fa-robot me-2"></i>AI Development Analysis</h6>
                    </div>
                    <div class="card-body" style="max-height: 60vh; overflow-y: auto; padding: 0;">
                        <div class="ai-analysis-content" style="
                            white-space: pre-wrap;
                            line-height: 1.6;
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            font-size: 14px;
                            padding: 20px;
                            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                            border-radius: 8px;
                            margin: 15px;
                            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
                            max-height: 50vh;
                            overflow-y: auto;
                            border: 1px solid #e9ecef;
                        ">
                            ${this.formatAIAnalysisContent(analysis.detailed_analysis)}
                        </div>
                    </div>
                </div>
    
                ${this.generateTechnicalDetailsSection(parcel, assessment)}
                ${this.generateRecommendationSection(assessment)}
            </div>
        `;
        
        showModal(modalContent, 'AI-Powered Development Analysis');
    }
    
    showGenericModal(content, title) {
        const modalHtml = `
            <div class="modal fade" id="genericModal" tabindex="-1">
                <div class="modal-dialog modal-xl"> <!-- Changed to modal-xl for more space -->
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <h5 class="modal-title">${title}</h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body" style="max-height: 75vh; overflow-y: auto; padding: 20px;">
                            ${content}
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary" onclick="window.bcf.downloadAIAnalysis('${title}')">
                                <i class="fas fa-download me-2"></i>Download Report
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal
        const existingModal = document.getElementById('genericModal');
        if (existingModal) existingModal.remove();
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('genericModal'));
        modal.show();
        
        modal._element.addEventListener('hidden.bs.modal', () => {
            document.getElementById('genericModal').remove();
        });
    }
    
    // Add method to download AI analysis as text file
    downloadAIAnalysis(title) {
        const analysisContent = document.querySelector('.ai-analysis-content')?.textContent;
        if (!analysisContent) {
            alert('No analysis content to download');
            return;
        }
        
        const blob = new Blob([analysisContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${title.replace(/[^a-z0-9]/gi, '_')}_${new Date().toISOString().slice(0, 10)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Add this new method to format the AI content properly
    formatAIAnalysisContent(content) {
        if (!content) return 'Analysis content not available.';
        
        // Clean up the content and add proper formatting
        let formattedContent = content
            // Add spacing around section headers
            .replace(/(\d+\.\s*[A-Z\s]+)\n/g, '\n📋 $1\n' + '─'.repeat(50) + '\n')
            // Format property details section
            .replace(/PROPERTY DETAILS:/g, '🏡 PROPERTY DETAILS:')
            // Format executive summary
            .replace(/EXECUTIVE SUMMARY/g, '⭐ EXECUTIVE SUMMARY')
            // Format technical analysis
            .replace(/TECHNICAL ANALYSIS/g, '🔧 TECHNICAL ANALYSIS')
            // Format economic evaluation
            .replace(/ECONOMIC EVALUATION/g, '💰 ECONOMIC EVALUATION')
            // Format strategic recommendations
            .replace(/STRATEGIC RECOMMENDATIONS/g, '🎯 STRATEGIC RECOMMENDATIONS')
            // Format market context
            .replace(/MARKET CONTEXT/g, '📊 MARKET CONTEXT')
            // Add extra line breaks for readability
            .replace(/\n\n/g, '\n\n')
            // Clean up excessive line breaks
            .replace(/\n{3,}/g, '\n\n')
            // Add bullet point formatting
            .replace(/^- /gm, '• ')
            // Format key-value pairs
            .replace(/^([A-Za-z\s]+):\s*(.+)$/gm, '📌 $1: $2')
            // Add section dividers
            .replace(/═{10,}/g, '\n' + '═'.repeat(60) + '\n');
        
        return formattedContent.trim();
    }    
    /**
     * Display enhanced analysis as fallback
     * @param {Object} parcel - Parcel data
     * @param {Function} showModal - Function to show modal
     */
    displayEnhancedAnalysis(parcel, showModal) {
        const modalContent = this.generateEnhancedAnalysisModal(parcel);
        showModal(modalContent, 'Enhanced Parcel Analysis');
    }

    /**
     * Generate AI results modal content
     * @param {Object} parcel - Parcel data
     * @param {Object} analysis - AI analysis results
     * @returns {string} - HTML content
     */
    generateAIResultsModal(parcel, analysis) {
        const assessment = this.calculateParcelAssessment(parcel);
        
        // FIXED: Always generate analysis content, don't show placeholder
        const analysisContent = analysis && analysis.detailed_analysis ? 
            analysis.detailed_analysis : 
            this.generateFallbackAnalysis(parcel, assessment);
        
        return `
            <div class="ai-analysis-container">
                <!-- Score Dashboard -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card text-center border-primary">
                            <div class="card-body">
                                <h2 class="text-primary">${assessment.category}</h2>
                                <p class="mb-0">Overall Rating</p>
                                <span class="badge ${assessment.badgeClass}">${assessment.category}</span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center border-success">
                            <div class="card-body">
                                <h3 class="text-success">${Math.round(parseFloat(parcel.acreage_calc || parcel.acreage || 0))}</h3>
                                <p class="mb-0">Acres</p>
                                <small class="text-muted">${assessment.projectScale}</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center border-info">
                            <div class="card-body">
                                <h3 class="text-info">${this.formatSlope(parcel.avg_slope_degrees)}</h3>
                                <p class="mb-0">Average Slope</p>
                                <small class="text-muted">${assessment.slopeRating}</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center border-warning">
                            <div class="card-body">
                                <h3 class="text-warning">${this.formatDistance(parcel.tx_distance_miles)}</h3>
                                <p class="mb-0">Grid Distance</p>
                                <small class="text-muted">${assessment.gridRating}</small>
                            </div>
                        </div>
                    </div>
                </div>
    
                <!-- AI Analysis Content - FIXED -->
                <div class="card mb-3">
                    <div class="card-header">
                        <h6><i class="fas fa-robot me-2"></i>AI Development Analysis</h6>
                    </div>
                    <div class="card-body">
                        <div class="ai-analysis-content" style="white-space: pre-wrap; line-height: 1.6; font-family: 'Segoe UI', sans-serif;">
                            ${analysisContent}
                        </div>
                    </div>
                </div>
    
                ${this.generateTechnicalDetailsSection(parcel, assessment)}
                ${this.generateRecommendationSection(assessment)}
            </div>
        `;
    }

    /**
     * Generate enhanced analysis modal content (fallback)
     * @param {Object} parcel - Parcel data
     * @returns {string} - HTML content
     */
    generateEnhancedAnalysisModal(parcel) {
        const assessment = this.calculateParcelAssessment(parcel);
        
        return `
            <div class="enhanced-analysis-container">
                <!-- Score Dashboard -->
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="card text-center border-primary">
                            <div class="card-body">
                                <h2 class="text-primary">${assessment.category}</h2>
                                <p class="mb-0">Development Rating</p>
                                <span class="badge ${assessment.badgeClass}">${assessment.category}</span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-center border-success">
                            <div class="card-body">
                                <h3 class="text-success">${Math.round(parseFloat(parcel.acreage_calc || parcel.acreage || 0))}</h3>
                                <p class="mb-0">Acres</p>
                                <small class="text-muted">${assessment.projectScale}</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-center border-info">
                            <div class="card-body">
                                <h3 class="text-info">${this.formatSlope(parcel.avg_slope_degrees)}</h3>
                                <p class="mb-0">Average Slope</p>
                                <small class="text-muted">${assessment.slopeRating}</small>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Enhanced Analysis -->
                <div class="card mb-3">
                    <div class="card-header">
                        <h6><i class="fas fa-chart-line me-2"></i>Development Assessment</h6>
                    </div>
                    <div class="card-body">
                        <div style="line-height: 1.6;">
                            ${this.generateFallbackAnalysis(parcel, assessment)}
                        </div>
                    </div>
                </div>

                ${this.generateTechnicalDetailsSection(parcel, assessment)}
                ${this.generateRecommendationSection(assessment)}

                <div class="alert alert-info mt-3">
                    <h6><i class="fas fa-info-circle me-2"></i>Analysis Note</h6>
                    <p class="mb-0">This enhanced analysis is based on available technical data. 
                    For AI-powered market insights and detailed development strategies, 
                    the advanced AI analysis endpoint is being implemented.</p>
                </div>
            </div>
        `;
    }

    /**
     * Calculate comprehensive parcel assessment
     * @param {Object} parcel - Parcel data
     * @returns {Object} - Assessment results
     */
    calculateParcelAssessment(parcel) {
        const slope = parseFloat(parcel.avg_slope_degrees || 0);
        const txDistance = parseFloat(parcel.tx_distance_miles || 999);
        const acres = parseFloat(parcel.acreage_calc || parcel.acreage || 0);

        // Evaluate factors
        const slopeRating = this.evaluateSlope(slope);
        const gridRating = this.evaluateGridAccess(txDistance);
        const sizeRating = this.evaluateProjectSize(acres);

        // Determine overall category
        const category = this.determineOverallCategory(slopeRating, gridRating, sizeRating);
        
        return {
            category,
            slopeRating: slopeRating.label,
            gridRating: gridRating.label,
            sizeRating: sizeRating.label,
            projectScale: sizeRating.scale,
            badgeClass: this.getCategoryBadgeClass(category),
            recommendation: this.generateRecommendation(category, slopeRating, gridRating, sizeRating),
            developmentChallenges: this.identifyDevelopmentChallenges(slope, txDistance, acres),
            estimatedCapacity: this.estimateCapacity(acres)
        };
    }

    /**
     * Evaluate slope factor
     */
    evaluateSlope(slope) {
        if (slope <= 5) return { score: 4, label: 'Excellent - Minimal grading needed' };
        if (slope <= 10) return { score: 3, label: 'Very Good - Light grading' };
        if (slope <= 15) return { score: 2, label: 'Good - Moderate grading' };
        if (slope <= 25) return { score: 1, label: 'Challenging - Extensive grading' };
        return { score: 0, label: 'Poor - Major earthwork required' };
    }

    /**
     * Evaluate grid access
     */
    evaluateGridAccess(distance) {
        if (distance === 0) return { score: 4, label: 'Excellent - On transmission line' };
        if (distance <= 0.5) return { score: 4, label: 'Excellent - Very close to grid' };
        if (distance <= 1) return { score: 3, label: 'Very Good - Close to grid' };
        if (distance <= 2) return { score: 2, label: 'Good - Reasonable distance' };
        if (distance <= 5) return { score: 1, label: 'Fair - Requires longer connection' };
        return { score: 0, label: 'Poor - Very far from transmission' };
    }

    /**
     * Evaluate project size
     */
    evaluateProjectSize(acres) {
        const projectData = {
            solar: { min: 5, good: 25, excellent: 50 },
            wind: { min: 50, good: 100, excellent: 200 },
            battery: { min: 1, good: 5, excellent: 10 }
        };

        const thresholds = projectData[this.projectType] || projectData.solar;
        
        if (acres >= thresholds.excellent) {
            return { score: 4, label: 'Excellent size', scale: 'Utility Scale' };
        }
        if (acres >= thresholds.good) {
            return { score: 3, label: 'Good size', scale: 'Commercial Scale' };
        }
        if (acres >= thresholds.min) {
            return { score: 2, label: 'Adequate size', scale: 'Small Commercial' };
        }
        return { score: 1, label: 'Below optimal size', scale: 'Distributed Scale' };
    }

    /**
     * Determine overall category from factor scores
     */
    determineOverallCategory(slopeRating, gridRating, sizeRating) {
        const totalScore = slopeRating.score + gridRating.score + sizeRating.score;
        const maxScore = 12;
        
        if (totalScore >= 10) return 'Excellent';
        if (totalScore >= 8) return 'Good';
        if (totalScore >= 5) return 'Fair';
        return 'Poor';
    }

    /**
     * Generate fallback analysis text
     */

    generateFallbackAnalysis(parcel, assessment) {
        const owner = parcel.owner || parcel.mail_name || 'UNKNOWN OWNER';
        const acres = Math.round(parseFloat(parcel.acreage_calc || parcel.acreage || 0));
        const slope = parseFloat(parcel.avg_slope_degrees || 0);
        const txDistance = parseFloat(parcel.tx_distance_miles || 999);
        
        return `RENEWABLE ENERGY DEVELOPMENT ASSESSMENT

    PROPERTY: ${owner.toUpperCase()}
    LOCATION: ${this.currentCounty} County, ${this.currentState}
    PROJECT TYPE: ${this.projectType.toUpperCase()} Energy Development

    ═══════════════════════════════════════

    EXECUTIVE SUMMARY
    ${assessment.category.toUpperCase()} development potential identified based on comprehensive technical analysis of terrain, grid connectivity, and project scale factors.

    TECHNICAL EVALUATION

    🏔️ TERRAIN ANALYSIS
    • Average Slope: ${slope.toFixed(1)}° (${assessment.slopeRating})
    • Grading Requirements: ${slope <= 5 ? 'Minimal earthwork needed' : slope <= 15 ? 'Moderate site preparation required' : 'Extensive grading necessary'}
    • Construction Complexity: ${slope <= 10 ? 'Standard construction methods applicable' : 'Enhanced foundation design may be required'}

    ⚡ GRID CONNECTIVITY 
    • Transmission Distance: ${txDistance === 999 || isNaN(txDistance) ? 'Data unavailable' : `${txDistance.toFixed(2)} miles (${assessment.gridRating})`}
    • Interconnection Cost Impact: ${txDistance <= 1 ? 'Minimal - very favorable' : txDistance <= 5 ? 'Moderate - manageable costs' : 'High - significant infrastructure investment required'}

    📏 PROJECT SCALE
    • Total Acreage: ${acres} acres (${assessment.projectScale})
    • Estimated Capacity: ${assessment.estimatedCapacity}
    • Development Category: ${acres >= 100 ? 'Large-scale utility project' : acres >= 25 ? 'Medium commercial development' : 'Small to medium project'}

    DEVELOPMENT CONSIDERATIONS

    ${assessment.developmentChallenges.map(challenge => `• ${challenge}`).join('\n')}

    MARKET POSITIONING
    This parcel ranks as ${assessment.category.toLowerCase()} for ${this.projectType} development in the ${this.currentCounty} County market. ${assessment.category === 'Excellent' ? 'Priority target for immediate development consideration.' : assessment.category === 'Good' ? 'Suitable for development with standard project economics.' : assessment.category === 'Fair' ? 'Marginal economics - detailed feasibility study recommended.' : 'Significant challenges present - alternative sites may offer better returns.'}

    RECOMMENDATION
    ${assessment.recommendation}

    ═══════════════════════════════════════

    This analysis incorporates slope gradients, transmission infrastructure proximity, and parcel size optimization for ${this.projectType} energy development. For detailed financial modeling and regulatory assessment, consult with local development specialists.`;
    }

    // Also enhance the technical details section with better transmission data handling
    generateTechnicalDetailsSection(parcel, assessment) {
        const txDistance = parseFloat(parcel.tx_distance_miles);
        const txVoltage = parseFloat(parcel.tx_voltage_kv);
        
        return `
            <div class="row mt-3">
                <div class="col-md-6">
                    <h6>Property Information</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Parcel ID:</strong></td><td>${parcel.parcel_id || 'N/A'}</td></tr>
                        <tr><td><strong>Owner:</strong></td><td>${parcel.owner || parcel.mail_name || 'N/A'}</td></tr>
                        <tr><td><strong>Address:</strong></td><td>${parcel.address || 'N/A'}</td></tr>
                        <tr><td><strong>County:</strong></td><td>${parcel.county_name || this.currentCounty}</td></tr>
                        <tr><td><strong>Total Acres:</strong></td><td>${Math.round(parseFloat(parcel.acreage_calc || parcel.acreage || 0))}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Development Metrics</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Slope:</strong></td><td>${this.formatSlope(parcel.avg_slope_degrees)}</td></tr>
                        <tr><td><strong>TX Distance:</strong></td><td>${isNaN(txDistance) ? 'N/A' : this.formatDistance(txDistance)}</td></tr>
                        <tr><td><strong>TX Voltage:</strong></td><td>${isNaN(txVoltage) ? 'N/A' : `${Math.round(txVoltage)} kV`}</td></tr>
                        <tr><td><strong>Slope Category:</strong></td><td>${parcel.slope_category || 'Standard'}</td></tr>
                        <tr><td><strong>Project Scale:</strong></td><td>${assessment.projectScale}</td></tr>
                        <tr><td><strong>Capacity Est.:</strong></td><td>${assessment.estimatedCapacity}</td></tr>
                    </table>
                </div>
            </div>
        `;
    }

    /**
     * Generate recommendation text
     */
    generateRecommendation(category, slopeRating, gridRating, sizeRating) {
        if (category === 'Excellent') {
            return 'HIGHLY RECOMMENDED - This parcel shows excellent development potential with favorable technical characteristics across all factors.';
        }
        if (category === 'Good') {
            return 'RECOMMENDED - This parcel offers good development potential with manageable technical challenges.';
        }
        if (category === 'Fair') {
            return 'PROCEED WITH CAUTION - This parcel has mixed characteristics that may increase development costs.';
        }
        return 'NOT RECOMMENDED - Significant technical challenges may make this parcel uneconomical for development.';
    }

    /**
     * Identify development challenges
     */
    identifyDevelopmentChallenges(slope, txDistance, acres) {
        const challenges = [];
        
        if (slope > 15) challenges.push('Steep terrain will require extensive site preparation');
        if (txDistance > 2) challenges.push('Long distance to transmission infrastructure increases connection costs');
        if (acres < 10 && this.projectType === 'solar') challenges.push('Small parcel size may limit economies of scale');
        if (acres < 50 && this.projectType === 'wind') challenges.push('Insufficient area for optimal turbine spacing');
        
        if (challenges.length === 0) {
            challenges.push('No significant technical challenges identified');
        }
        
        return challenges;
    }

    /**
     * Estimate capacity
     */
    estimateCapacity(acres) {
        const capacityFactors = {
            solar: 0.4,   // MW per acre
            wind: 2,      // MW per acre (very rough estimate)
            battery: 5    // MW per acre
        };
        
        const factor = capacityFactors[this.projectType] || capacityFactors.solar;
        const mw = Math.round(acres * factor);
        
        return `~${mw} MW potential`;
    }

    /**
     * Helper methods for formatting
     */
    formatSlope(slope) {
        const numSlope = parseFloat(slope);
        return isNaN(numSlope) ? 'N/A' : `${numSlope.toFixed(1)}°`;
    }

    formatDistance(distance) {
        const numDistance = parseFloat(distance);
        if (isNaN(numDistance)) return 'N/A';
        if (numDistance === 0) return 'Intersects';
        return `${numDistance.toFixed(2)} mi`;
    }

    getCategoryBadgeClass(category) {
        const classes = {
            'Excellent': 'badge bg-success',
            'Good': 'badge bg-primary',
            'Fair': 'badge bg-warning text-dark',
            'Poor': 'badge bg-danger'
        };
        return classes[category] || 'badge bg-secondary';
    }

    /**
     * Generate technical details section
     */
    generateTechnicalDetailsSection(parcel, assessment) {
        return `
            <div class="row mt-3">
                <div class="col-md-6">
                    <h6>Parcel Information</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Parcel ID:</strong></td><td>${parcel.parcel_id || 'N/A'}</td></tr>
                        <tr><td><strong>Owner:</strong></td><td>${parcel.owner || 'N/A'}</td></tr>
                        <tr><td><strong>Address:</strong></td><td>${parcel.address || 'N/A'}</td></tr>
                        <tr><td><strong>County:</strong></td><td>${parcel.county_name || this.currentCounty}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Development Metrics</h6>
                    <table class="table table-sm">
                        <tr><td><strong>TX Distance:</strong></td><td>${this.formatDistance(parcel.tx_distance_miles)}</td></tr>
                        <tr><td><strong>TX Voltage:</strong></td><td>${parcel.tx_voltage_kv || 'N/A'} kV</td></tr>
                        <tr><td><strong>Slope Category:</strong></td><td>${parcel.slope_category || 'N/A'}</td></tr>
                        <tr><td><strong>Project Scale:</strong></td><td>${assessment.projectScale}</td></tr>
                    </table>
                </div>
            </div>
        `;
    }

    /**
     * Generate recommendation section
     */
    generateRecommendationSection(assessment) {
        const alertClass = {
            'Excellent': 'alert-success',
            'Good': 'alert-info',
            'Fair': 'alert-warning',
            'Poor': 'alert-danger'
        }[assessment.category] || 'alert-secondary';

        return `
            <div class="alert ${alertClass} mt-3">
                <h6><i class="fas fa-thumbs-up me-2"></i>Development Recommendation</h6>
                <p class="mb-0"><strong>${assessment.recommendation}</strong></p>
            </div>
        `;
    }
}

// Export for use in main application
if (typeof window !== 'undefined') {
    window.IndividualParcelAI = IndividualParcelAI;
}