import warnings
from utils.transmission_region_mapper import map_location_to_region

# --- Throughput (Mbps) by provider ---
# Extracted from Figure 3 in [1]
# [1] V. Persico, A. Botta, A. Montieri and A. Pescape, "A First Look at Public-Cloud Inter-Datacenter Network Performance," 2016 IEEE Global Communications Conference (GLOBECOM), Washington, DC, USA, 2016, pp. 1-7, doi: 10.1109/GLOCOM.2016.7841498.
aws_throughput = {
    'EU': {'US': 160, 'SA': 100, 'AP': 50},
    'US': {'EU': 160, 'SA': 100, 'AP': 50},
    'SA': {'EU': 75, 'US': 90, 'AP': 40},
    'AP': {'EU': 50, 'US': 50, 'SA': 45}
}

azure_throughput = {
    'EU': {'US': 250, 'SA': 120, 'AP': 60},
    'US': {'EU': 260, 'SA': 150, 'AP': 75},
    'SA': {'EU': 80, 'US': 150, 'AP': 50},
    'AP': {'EU': 60, 'US': 75, 'SA': 50}
}

# --- RTT (ms) by provider ---
# Extracted from Figure 6 in [1]
# [1] V. Persico, A. Botta, A. Montieri and A. Pescape, "A First Look at Public-Cloud Inter-Datacenter Network Performance," 2016 IEEE Global Communications Conference (GLOBECOM), Washington, DC, USA, 2016, pp. 1-7, doi: 10.1109/GLOCOM.2016.7841498.
aws_latency = {
    'EU': {'US': 70, 'SA': 180, 'AP': 190},
    'US': {'EU': 70, 'SA': 120, 'AP': 240},
    'SA': {'EU': 180, 'US': 120, 'AP': 350},
    'AP': {'EU': 190, 'US': 240, 'SA': 350}
}

azure_latency = {
    'EU': {'US': 70, 'SA': 180, 'AP': 315},
    'US': {'EU': 70, 'SA': 130, 'AP': 230},
    'SA': {'EU': 180, 'US': 130, 'AP': 350},
    'AP': {'EU': 315, 'US': 230, 'SA': 350}
}

# --- Region maps ---
# --- How each region code maps into our 4 clusters ---
aws_region_to_cluster = {
    'us-east-1':      'US',
    'us-east-2':      'US',
    'us-west-1':      'US',
    'us-east-1-dwf-1':'US',
    'ca-central-1':   'US',
    'eu-central-1':   'EU',
    'eu-west-1':      'EU',
    'eu-west-2':      'EU',
    'eu-west-3':      'EU',
    'eu-south-1':     'EU',
    'eu-central-2':   'EU',
    'ap-northeast-1': 'AP',
    'ap-northeast-2': 'AP',
    'ap-southeast-1': 'AP',
    'ap-southeast-2': 'AP',
    'ap-southeast-3': 'AP',
    'ap-south-1':     'AP',
    'sa-east-1':      'SA',
    'af-south-1':     'SA',
    'us-east-1-chl-1': 'SA',
}

azure_region_to_cluster = {
    'East US':           'US',
    'East US 2':         'US',
    'West US':           'US',
    'South Central US':  'US',
    'Canada Central':    'US',
    'West Europe':       'EU',
    'North Europe':      'EU',
    'France Central':    'EU',
    'Germany West Central':'EU',
    'Austria East':      'EU',
    'Switzerland North': 'EU',
    'Portugal North':    'EU',
    'Spain Central':     'EU',
    'Japan East':        'AP',
    'Korea Central':     'AP',
    'Australia East':    'AP',
    'Southeast Asia':    'AP',
    'Central India':     'AP',
    'Australia Southeast':'AP',
    'Brazil South':      'SA',
    'Chile North':       'SA',
    'South Africa North':'SA',
}

def region_to_cluster(provider, region_code):
    """
    Map a cloud-provider region string -> one of our 4 macro-clusters: EU, US, SA, AP.
    """
    p = provider.lower()
    if p == 'aws':
        table = aws_region_to_cluster
    else:  # 'azure'
        table = azure_region_to_cluster

    try:
        return table[region_code]
    except KeyError:
        raise ValueError(f"Unknown region '{region_code}' for provider '{provider}'")


def get_transmission_delay(src_loc, dst_loc, provider, size_GB):
    """
    Calculate the end-to-end transmission delay (seconds) for a `size_GB` transfer
    from src_loc -> dst_loc on the chosen provider.

      delay = serialization_time (size / throughput) + propagation_time (RTT)

    If provider ∉ {'aws','azure'}, we warn then fall back to AWS numbers.
    """
    p = provider.lower()
    if p not in ('aws', 'azure'):
        warnings.warn(f"Provider '{provider}' not supported for delay → defaulting to AWS")
        p = 'aws'

    # 1) Find the provider-specific region codes
    src_reg = map_location_to_region(src_loc, p)
    dst_reg = map_location_to_region(dst_loc, p)

    # 2) Convert GB to Mb
    size_Mb = size_GB * 8.0 * 1000.0  # Mb

    # 3) Map into one of {EU,US,SA,AP}
    sc = region_to_cluster(p, src_reg)
    dc = region_to_cluster(p, dst_reg)
    
    # 4) If same region, assume 1 Gbps
    if sc == dc:
        return size_Mb / 1000.0

    # 5) Lookup
    if p == 'aws':
        bw  = aws_throughput[sc][dc]   # Mbps
        rtt = aws_latency[sc][dc]      # ms
    else:
        bw  = azure_throughput[sc][dc]
        rtt = azure_latency[sc][dc]

    # 6) Compute delay (s)
    # Extracted from [1]
    return (size_Mb / bw) + (rtt / 1000.0)
