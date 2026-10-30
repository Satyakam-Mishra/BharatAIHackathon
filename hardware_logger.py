import sqlite3
import psutil
import os
from datetime import datetime

DB_NAME = "inspection_logs.db"

def init_db():
    """Initializes the SQLite database for factory floor logging."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            defect_types TEXT,
            defect_count INTEGER,
            max_confidence REAL,
            llama_advice TEXT,
            cpu_temp_c REAL
        )
    """)
    conn.commit()
    conn.close()

def get_hardware_metrics():
    """Reads CPU load, RAM usage, and Raspberry Pi 5 thermal sensor."""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram_usage = psutil.virtual_memory().percent
    temp = 0.0
    
    # Read RPi 5 hardware thermal zone (falls back to 40.0C if testing on a laptop)
    thermal_file = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(thermal_file):
        with open(thermal_file, "r") as f:
            temp = round(int(f.read().strip()) / 1000.0, 1)
    else:
        temp = 42.5  # Simulated laptop fallback
        
    return {"cpu_percent": cpu_usage, "ram_percent": ram_usage, "temp_c": temp}

def log_inspection(defects, advice):
    """Logs a single PCB inspection result to SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    defect_names = ", ".join([d["defect"] for d in defects]) if defects else "Nominal"
    max_conf = max([d["confidence"] for d in defects]) if defects else 1.0
    hw = get_hardware_metrics()
    
    cursor.execute("""
        INSERT INTO inspections (timestamp, defect_types, defect_count, max_confidence, llama_advice, cpu_temp_c)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        defect_names,
        len(defects),
        max_conf,
        advice,
        hw["temp_c"]
    ))
    conn.commit()
    conn.close()