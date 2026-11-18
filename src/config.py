"""
ROADGUARD AI - Configuration & Constants
Defines design themes, color palettes, standard schemas, and column mapping dictionaries.
"""

APP_NAME = "ROADGUARD AI"
APP_TAGLINE = "Road Safety Intelligence & Accident Hotspot Analytics"
APP_SUBTITLE = "Unsupervised Machine Learning Decision-Support System"
ACADEMIC_FOOTER = "Academic Project | Unsupervised Machine Learning | Exploratory Decision-Support"

# Color Palette (Modern Clean Light Theme)
COLOR_BG = "#F8FAFC"
COLOR_PANEL = "#FFFFFF"
COLOR_PANEL_ALT = "#F1F5F9"
COLOR_ACCENT_CYAN = "#0284C7"
COLOR_ACCENT_ORANGE = "#D97706"
COLOR_RED_SEVERE = "#DC2626"
COLOR_GREEN_MILD = "#059669"
COLOR_YELLOW_MOD = "#CA8A04"
COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#E2E8F0"
COLOR_CARD_HOVER = "#F8FAFC"

# Distinct high-contrast palette for K-Means and Agglomerative clusters
CLUSTER_COLORS = [
    "#0284C7",  # Deep Sky Blue
    "#D97706",  # Amber Warning
    "#DC2626",  # Crimson Severe
    "#059669",  # Emerald Green
    "#7C3AED",  # Royal Violet
    "#DB2777",  # Vivid Rose
    "#2563EB",  # Classic Blue
    "#0D9488",  # Deep Teal
    "#EA580C",  # Vibrant Orange
    "#9333EA",  # Purple
]

# Canonical Schema Definitions
CANONICAL_COLUMNS = {
    "Accident_ID": {"required": False, "dtype": "string", "desc": "Unique accident identifier"},
    "Date": {"required": True, "dtype": "datetime", "desc": "Date of the collision"},
    "Time": {"required": False, "dtype": "time", "desc": "Time of occurrence (HH:MM)"},
    "State": {"required": False, "dtype": "string", "desc": "State / Administrative Division"},
    "City": {"required": False, "dtype": "string", "desc": "City / Metropolitan jurisdiction"},
    "Latitude": {"required": False, "dtype": "float", "desc": "Geographical latitude (WGS84)"},
    "Longitude": {"required": False, "dtype": "float", "desc": "Geographical longitude (WGS84)"},
    "Road_Type": {"required": False, "dtype": "category", "desc": "Road classification (Expressway, Arterial, etc.)"},
    "Weather": {"required": False, "dtype": "category", "desc": "Atmospheric condition (Clear, Rain, Fog, etc.)"},
    "Lighting_Condition": {"required": False, "dtype": "category", "desc": "Ambient illumination (Daylight, Dark, Streetlights)"},
    "Road_Surface": {"required": False, "dtype": "category", "desc": "Surface condition (Dry, Wet, Icy, Flooded)"},
    "Vehicle_Type": {"required": False, "dtype": "category", "desc": "Primary vehicle involved"},
    "Number_of_Vehicles": {"required": False, "dtype": "int", "desc": "Total vehicles involved"},
    "Number_of_Casualties": {"required": False, "dtype": "int", "desc": "Total individuals impacted"},
    "Number_of_Fatalities": {"required": False, "dtype": "int", "desc": "Fatalities confirmed"},
    "Number_of_Injuries": {"required": False, "dtype": "int", "desc": "Non-fatal injuries sustained"},
    "Accident_Severity": {"required": False, "dtype": "category", "desc": "Severity class (Fatal, Serious, Slight)"},
    "Speed_Limit": {"required": False, "dtype": "float", "desc": "Posted roadway speed limit (km/h)"},
}

# Intelligent Synonyms / Aliases for Automatic Column Mapping
COLUMN_ALIASES = {
    "Accident_ID": [
        "accident_id", "accident id", "id", "collision_id", "incident_id", "report_num", 
        "case_id", "reference_num", "accident_ref", "crash_id"
    ],
    "Date": [
        "date", "accident_date", "crash_date", "incident_date", "collision_date", 
        "event_date", "timestamp", "datetime"
    ],
    "Time": [
        "time", "accident_time", "crash_time", "incident_time", "collision_time", 
        "hour_min", "time_of_day"
    ],
    "State": [
        "state", "province", "region", "state_name", "territory", "administrative_area"
    ],
    "City": [
        "city", "town", "municipality", "local_authority", "district", "borough", 
        "location", "county", "urban_area"
    ],
    "Latitude": [
        "latitude", "lat", "y", "geo_latitude", "gps_lat", "lat_coord", "point_y"
    ],
    "Longitude": [
        "longitude", "long", "lon", "lng", "x", "geo_longitude", "gps_lng", "lat_coord", "point_x"
    ],
    "Road_Type": [
        "road_type", "highway_type", "street_type", "road_class", "carriage_way", 
        "route_type", "infrastructure_type", "roadway_type"
    ],
    "Weather": [
        "weather", "weather_condition", "atmospheric_conditions", "weather_cond", 
        "climate", "meteo"
    ],
    "Lighting_Condition": [
        "lighting_condition", "light_condition", "lighting", "illumination", 
        "day_night", "light", "ambient_light"
    ],
    "Road_Surface": [
        "road_surface", "road_surface_conditions", "surface_condition", "pavement_condition", 
        "road_condition", "surface"
    ],
    "Vehicle_Type": [
        "vehicle_type", "primary_vehicle", "veh_type", "mode_of_transport", "transport_type"
    ],
    "Number_of_Vehicles": [
        "number_of_vehicles", "num_vehicles", "vehicles_involved", "veh_count", 
        "vehicle_count", "tot_vehicles", "no_of_vehicles"
    ],
    "Number_of_Casualties": [
        "number_of_casualties", "num_casualties", "casualties", "total_casualties", 
        "tot_casualties", "victims", "no_of_casualties"
    ],
    "Number_of_Fatalities": [
        "number_of_fatalities", "num_fatalities", "fatalities", "deaths", "killed", 
        "num_deaths", "fatality_count", "persons_killed", "no_of_fatalities"
    ],
    "Number_of_Injuries": [
        "number_of_injuries", "num_injuries", "injuries", "hurt", "injured", 
        "injury_count", "persons_injured", "no_of_injuries"
    ],
    "Accident_Severity": [
        "accident_severity", "severity", "crash_severity", "injury_severity", 
        "severity_level", "severity_class"
    ],
    "Speed_Limit": [
        "speed_limit", "posted_speed", "speed_zone", "max_speed", "road_speed"
    ],
}

# Default Clustering Features for ML Engine
DEFAULT_CLUSTERING_FEATURES = [
    "Hour",
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Number_of_Fatalities",
    "Number_of_Injuries",
    "Speed_Limit",
    "Severity_Numeric",
    "Is_Night",
    "Is_Weekend",
    "Is_Adverse_Weather",
]

# Standard Categorical Mapping for Severity
SEVERITY_MAPPING = {
    "slight": 1,
    "minor": 1,
    "low": 1,
    "damage only": 1,
    "moderate": 2,
    "serious": 2,
    "severe": 2,
    "major": 2,
    "fatal": 3,
    "critical": 3,
    "catastrophic": 3,
}

# Navigation Pages
PAGES = [
    "Dataset Upload Center",
    "Executive Overview",
    "Accident Hotspot Map",
    "Cluster Explorer",
    "Machine Learning Lab",
    "Pattern Analysis",
    "Location Intelligence",
    "Report Generator",
]
