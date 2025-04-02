from datetime import datetime, timedelta
from openelectricity import OEClient
from openelectricity.types import DataMetric, UnitFueltechType, UnitStatusType

import os
os.environ["OPENELECTRICITY_API_KEY"] = "oe_3ZkCohwCqoHRL535ay6JtgTC"

# Calculate date range
end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
start_date = end_date - timedelta(days=7)

# Using context manager (recommended)
with OEClient() as client:
    # Get operating solar and wind facilities
    facilities = client.get_facilities(
        network_id=["NEM"],
        status_id=[UnitStatusType.OPERATING],
        fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
    )

    # Get network data for NEM
    response = client.get_network_data(
        network_code="NEM",
        metrics=[DataMetric.POWER, DataMetric.ENERGY],
        interval="1d",
        date_start=start_date,
        date_end=end_date,
        secondary_grouping="fueltech_group",
    )

    # Print results
    for series in response.data:
        print(f"\nMetric: {series.metric}")
        print(f"Unit: {series.unit}")

        for result in series.results:
            print(f"\n  {result.name}:")
            print(f"  Fuel Tech Group: {result.columns.fueltech_group}")
            for point in result.data:
                print(f"    {point.timestamp}: {point.value:.2f} {series.unit}")