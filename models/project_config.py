# models/project_config.py
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ProjectConfig:
    """Configuration class for renewable energy projects"""

    project_type: str = "solar"  # solar, wind, hydro, etc.
    location: str = "United States"
    analysis_level: str = "state"  # state, county, local
    variables: Dict[str, Any] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
        """Create ProjectConfig from dictionary"""
        return cls(
            project_type=data.get('project_type', 'solar'),
            location=data.get('location', 'United States'),
            analysis_level=data.get('analysis_level', 'state'),
            variables=data.get('variables', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProjectConfig to dictionary"""
        return {
            'project_type': self.project_type,
            'location': self.location,
            'analysis_level': self.analysis_level,
            'variables': self.variables
        }

    def add_variable(self, name: str, value: Any):
        """Add a variable to the project configuration"""
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value from the project configuration"""
        return self.variables.get(name, default)

    @staticmethod
    def get_tier_criteria(analysis_level: str) -> List[str]:
        """Get the specific criteria focus areas for each analysis tier"""

        tier_criteria = {
            "state": [
                "Resource Quality - Statewide renewable resource assessment",
                "Policy Environment - RPS standards, incentives, regulations",
                "Market Fundamentals - Electricity prices, utility structure, demand"
            ],
            "county": [
                "Infrastructure Proximity - Transmission, substations, transportation",
                "Land Suitability - Available land, zoning, environmental constraints",
                "Local Economics - Property taxes, development costs, labor market"
            ],
            "local": [
                "Interconnection Feasibility - Substation capacity, queue, upgrade costs",
                "Permitting Pathway - Local zoning, environmental review, community acceptance",
                "Site-Specific Resource - Measured data, micro-siting, site conditions"
            ]
        }

        return tier_criteria.get(analysis_level, tier_criteria["state"])

    @staticmethod
    def get_tier_description(analysis_level: str) -> str:
        """Get description of what each tier is used for in developer workflow"""

        descriptions = {
            "state": "Market Entry Screening - Identify promising states for renewable development",
            "county": "Site Identification - Narrow down to specific counties/regions for development",
            "local": "Project Development - Detailed site-specific analysis for go/no-go decisions"
        }

        return descriptions.get(analysis_level, "General analysis")

