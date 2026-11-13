"""
ROADGUARD AI - Synthetic Demonstration Dataset Generator
Generates realistic, statistically structured road accident records with distinct latent clusters.
Labeled explicitly as Synthetic Academic Demonstration Data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_accidents(num_records=3500, random_seed=42):
    np.random.seed(random_seed)

    # City focal coordinates (center lat, lon, name, state)
    corridors = [
        {"city": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "weight": 0.28},
        {"city": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "weight": 0.18},
        {"city": "Bengaluru", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "weight": 0.24},
        {"city": "New Delhi", "state": "Delhi NCT", "lat": 28.6139, "lon": 77.2090, "weight": 0.20},
        {"city": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "weight": 0.10},
    ]

    city_probs = [c["weight"] for c in corridors]
    chosen_corridors = np.random.choice(corridors, size=num_records, p=city_probs)

    records = []
    start_date = datetime(2023, 1, 1)

    # 5 latent archetypes for unsupervised discovery
    archetypes = ["expressway_severe", "monsoon_wet", "unlit_night", "urban_minor", "foggy_dawn"]
    archetype_probs = [0.22, 0.25, 0.20, 0.23, 0.10]
    chosen_archetypes = np.random.choice(archetypes, size=num_records, p=archetype_probs)

    for i in range(num_records):
        arch = chosen_archetypes[i]
        corr = chosen_corridors[i]

        # Accident ID
        acc_id = f"RG-SYN-{2023 + (i % 2)}-{10000 + i}"

        # Temporal
        days_offset = np.random.randint(0, 730)
        curr_date = start_date + timedelta(days=days_offset)
        day_of_week = curr_date.strftime("%A")

        # Geo jitter around corridors
        lat_jitter = np.random.normal(0, 0.045)
        lon_jitter = np.random.normal(0, 0.045)
        latitude = round(corr["lat"] + lat_jitter, 6)
        longitude = round(corr["lon"] + lon_jitter, 6)

        if arch == "expressway_severe":
            # High speed, multiple vehicles, severe casualties, night/dusk
            hour = np.random.choice([20, 21, 22, 23, 0, 1, 2, 3, 18, 19])
            minute = np.random.randint(0, 60)
            road_type = np.random.choice(["Expressway", "National Highway"], p=[0.65, 0.35])
            speed_limit = np.random.choice([100.0, 120.0], p=[0.7, 0.3])
            weather = np.random.choice(["Clear", "Overcast", "Windy"], p=[0.75, 0.20, 0.05])
            lighting = np.random.choice(["Dark - Street Lights On", "Dark - No Street Lighting", "Dusk/Dawn"], p=[0.5, 0.35, 0.15])
            road_surface = "Dry" if np.random.rand() > 0.15 else "Wet"
            veh_type = np.random.choice(["SUV", "Heavy Commercial (Truck)", "Four-Wheeler (Sedan)"], p=[0.4, 0.4, 0.2])
            num_vehicles = int(np.random.choice([2, 3, 4, 5], p=[0.45, 0.35, 0.15, 0.05]))
            fatalities = int(np.random.choice([1, 2, 3, 0], p=[0.45, 0.25, 0.10, 0.20]))
            injuries = int(np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1]))
            casualties = fatalities + injuries
            severity = "Fatal" if fatalities > 0 else "Serious"

        elif arch == "monsoon_wet":
            # Rain, wet surface, urban roads, slight to moderate injuries, rush hour
            hour = np.random.choice([8, 9, 10, 11, 16, 17, 18, 19, 20])
            minute = np.random.randint(0, 60)
            road_type = np.random.choice(["Urban Arterial", "Ring Road", "Local Street"], p=[0.55, 0.30, 0.15])
            speed_limit = float(np.random.choice([40.0, 50.0, 60.0], p=[0.3, 0.5, 0.2]))
            weather = np.random.choice(["Heavy Rain", "Thunderstorm", "Overcast"], p=[0.65, 0.25, 0.10])
            lighting = np.random.choice(["Daylight", "Dusk/Dawn", "Dark - Street Lights On"], p=[0.6, 0.2, 0.2])
            road_surface = np.random.choice(["Wet", "Waterlogged", "Slippery / Mud"], p=[0.7, 0.2, 0.1])
            veh_type = np.random.choice(["Two-Wheeler (Motorcycle)", "Four-Wheeler (Sedan)", "Auto-Rickshaw"], p=[0.45, 0.35, 0.20])
            num_vehicles = int(np.random.choice([1, 2, 3], p=[0.35, 0.50, 0.15]))
            fatalities = 0 if np.random.rand() > 0.04 else 1
            injuries = int(np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
            casualties = fatalities + injuries
            severity = "Fatal" if fatalities > 0 else ("Serious" if injuries >= 2 else "Slight")

        elif arch == "unlit_night":
            # Unlit roads, midnight to 4 AM, high fatality proportion, freight/motorcycles
            hour = np.random.choice([23, 0, 1, 2, 3, 4])
            minute = np.random.randint(0, 60)
            road_type = np.random.choice(["Rural Highway", "Urban Arterial", "National Highway"], p=[0.45, 0.35, 0.20])
            speed_limit = float(np.random.choice([60.0, 80.0], p=[0.6, 0.4]))
            weather = np.random.choice(["Clear", "Overcast"], p=[0.85, 0.15])
            lighting = np.random.choice(["Dark - No Street Lighting", "Dark - Street Lights Off"], p=[0.75, 0.25])
            road_surface = "Dry" if np.random.rand() > 0.1 else "Wet"
            veh_type = np.random.choice(["Heavy Commercial (Truck)", "Two-Wheeler (Motorcycle)", "Four-Wheeler (Sedan)"], p=[0.5, 0.35, 0.15])
            num_vehicles = int(np.random.choice([1, 2], p=[0.55, 0.45]))
            fatalities = int(np.random.choice([1, 2, 0], p=[0.5, 0.2, 0.3]))
            injuries = int(np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2]))
            casualties = max(1, fatalities + injuries)
            severity = "Fatal" if fatalities > 0 else "Serious"

        elif arch == "urban_minor":
            # Daylight, stop & go congestion, low speed, slight damage/injuries
            hour = np.random.choice([11, 12, 13, 14, 15, 16])
            minute = np.random.randint(0, 60)
            road_type = np.random.choice(["Local Street", "Urban Arterial", "Ring Road"], p=[0.5, 0.35, 0.15])
            speed_limit = float(np.random.choice([30.0, 40.0, 50.0], p=[0.3, 0.5, 0.2]))
            weather = "Clear" if np.random.rand() > 0.1 else "Overcast"
            lighting = "Daylight"
            road_surface = "Dry"
            veh_type = np.random.choice(["Four-Wheeler (Sedan)", "Auto-Rickshaw", "Two-Wheeler (Motorcycle)", "Bus"], p=[0.4, 0.25, 0.25, 0.1])
            num_vehicles = int(np.random.choice([2, 3], p=[0.75, 0.25]))
            fatalities = 0
            injuries = int(np.random.choice([0, 1], p=[0.45, 0.55]))
            casualties = max(1, injuries)
            severity = "Slight"

        else:  # foggy_dawn
            # Early morning winter fog, reduced visibility, multi-vehicle pileups
            hour = np.random.choice([4, 5, 6, 7])
            minute = np.random.randint(0, 60)
            road_type = np.random.choice(["National Highway", "Expressway", "Ring Road"], p=[0.5, 0.35, 0.15])
            speed_limit = float(np.random.choice([80.0, 100.0], p=[0.6, 0.4]))
            weather = np.random.choice(["Fog/Mist", "Heavy Fog"], p=[0.7, 0.3])
            lighting = np.random.choice(["Dusk/Dawn", "Dark - Street Lights On"], p=[0.7, 0.3])
            road_surface = "Wet" if np.random.rand() > 0.4 else "Dry"
            veh_type = np.random.choice(["Heavy Commercial (Truck)", "Bus", "SUV"], p=[0.45, 0.30, 0.25])
            num_vehicles = int(np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2]))
            fatalities = int(np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
            injuries = int(np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]))
            casualties = fatalities + injuries
            severity = "Fatal" if fatalities > 0 else "Serious"

        time_str = f"{hour:02d}:{minute:02d}"

        records.append({
            "Accident_ID": acc_id,
            "Date": curr_date.strftime("%Y-%m-%d"),
            "Time": time_str,
            "State": corr["state"],
            "City": corr["city"],
            "Latitude": latitude,
            "Longitude": longitude,
            "Road_Type": road_type,
            "Weather": weather,
            "Lighting_Condition": lighting,
            "Road_Surface": road_surface,
            "Vehicle_Type": veh_type,
            "Number_of_Vehicles": num_vehicles,
            "Number_of_Casualties": casualties,
            "Number_of_Fatalities": fatalities,
            "Number_of_Injuries": injuries,
            "Accident_Severity": severity,
            "Speed_Limit": speed_limit,
            "Dataset_Type": "Synthetic Academic Benchmark"
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = generate_synthetic_accidents(num_records=3500)
    out_path = "data/sample_accidents.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic accident records to {out_path}")
    print(df.head(3))
