# utils/transmission_region_mapper.py

"""
This module maps real-world datacenter location codes to cloud-specific transmission cost regions.

Supported providers: "gcp", "aws", "azure", and "custom"

To define a custom cost matrix:
1. Add your mapping in `location_to_custom_region`
2. Save your transmission cost CSV in: `data/transmission_costs/custom_transmission_cost_matrix.csv`
   - Format: rows and columns must match your custom region names
3. Use `cloud_provider='custom'` when initializing `DatacenterClusterManager`

Each row/col in the CSV must represent the cost per GB from origin -> destination.
"""


# GCP region mapping
location_to_gcp_region = {
    "US-NY-NYIS": "us-east1",
    "US-CAL-CISO": "us-west1",
    "US-TEX-ERCO": "us-central1",  # Approximation
    "DE-LU": "europe-west1",
    "FR": "europe-west4",
    "SG": "asia-southeast1",
    "JP-TK": "asia-northeast1",
    "IN": "asia-south1",
    "AU-NSW": "australia-southeast1",
    "BR": "southamerica-east1",
    "ZA": "africa-south1",
    "PT": "europe-west1",
    "ES": "europe-west1",
    "BE": "europe-west1",
    "CH": "europe-west4",
    "KR": "asia-northeast1",
    "CA-ON": "us-east1",  # Proxy to closest US East
    "CL-SIC": "southamerica-east1",
    "AT": "europe-west4",
    "NL": "europe-west1"
}

# AWS region mapping
location_to_aws_region = {
    "US-NY-NYIS": "us-east-1",
    "US-CAL-CISO": "us-west-1",
    "US-TEX-ERCO": "us-east-1-dwf-1",
    "DE-LU": "eu-central-1",
    "FR": "eu-west-3",
    "SG": "ap-southeast-1",
    "JP-TK": "ap-northeast-1",
    "IN": "ap-south-1",
    "AU-NSW": "ap-southeast-2",
    "BR": "sa-east-1",
    "ZA": "af-south-1",
    "PT": "eu-south-1",
    "ES": "eu-south-1",
    "BE": "eu-west-1",
    "CH": "eu-central-1",
    "KR": "ap-northeast-2",
    "CA-ON": "ca-central-1",
    "CL-SIC": "us-east-1-chl-1",
    "AT": "eu-central-1",
    "NL": "eu-west-1"
}

# AZURE region mapping
location_to_azure_region = {
    "US-NY-NYIS": "East US",
    "US-CAL-CISO": "West US",
    "US-TEX-ERCO": "South Central US",
    "DE-LU": "Germany West Central",
    "FR": "France Central",
    "SG": "Southeast Asia",
    "JP-TK": "Japan East",
    "IN": "Central India",
    "AU-NSW": "Australia East",
    "BR": "Brazil South",
    "ZA": "South Africa North",
    "PT": "Portugal North",
    "ES": "Spain Central",
    "BE": "West Europe",
    "CH": "Switzerland North",
    "KR": "Korea Central",
    "CA-ON": "Canada Central",
    "CL-SIC": "Chile North",
    "AT": "Austria East",
    "NL": "North Europe"
}

# Custom region mapping
location_to_custom_region = {
    "US-NY-NYIS": "CustomRegion1",
    "US-CAL-CISO": "CustomRegion2",
    "US-TEX-ERCO": "CustomRegion3",
    "DE-LU": "CustomRegion4",
    "FR": "CustomRegion5",
    "SG": "CustomRegion6",
    "JP-TK": "CustomRegion7",
    "IN": "CustomRegion8",
    "AU-NSW": "CustomRegion9",
    "BR": "CustomRegion10",
    "ZA": "CustomRegion11",
    # Add more mappings as needed
}

def map_location_to_region(location_code: str, provider: str):
    provider = provider.lower()
    if provider == "gcp":
        return location_to_gcp_region.get(location_code)
    elif provider == "aws":
        return location_to_aws_region.get(location_code)
    elif provider == "azure":
        return location_to_azure_region.get(location_code)
    elif provider == "custom":
        return location_to_custom_region.get(location_code)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use one of: gcp, aws, azure, custom.")


