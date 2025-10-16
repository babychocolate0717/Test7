# agent_with_auth_english.py
# English version to avoid encoding issues
import psutil
import platform
import uuid
import getpass
import time
import json
import csv
import os
import requests
import hashlib
import hmac
from datetime import datetime, timezone, time as dtime
import subprocess
from pynput import mouse, keyboard
import threading
import sys
import socket
import yaml
from typing import Dict, Any
# --- AI 清洗所需函式庫 ---
import pandas as pd
import numpy as np
import joblib

# --- AI 清洗配置 (手動版本控制) ---
def resource_path(relative_path):
    """ 獲取資源的絕對路徑，適用於開發環境和 PyInstaller 打包後 """
    try:
        # PyInstaller 建立一個暫存資料夾，並將路徑儲存在 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # 在開發環境中，使用目前的檔案路徑
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

# 【重要】請根據您訓練腳本的輸出，手動設定此處的 ID！
ACTIVE_MODEL_ID = "20251012_164538"

# 定義模型路徑 (使用新的 resource_path 函數)
MODEL_DIR = resource_path(os.path.join("..", "models", "cleaning_models"))

# 定義模型路徑 (Agent 檔案在 agent/，模型檔案在 ../models/cleaning_models)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # ← 只往上跳一層
MODEL_DIR = os.path.join(BASE_DIR, "models", "cleaning_models")

ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, f"anomaly_model_{ACTIVE_MODEL_ID}.joblib")
IMPUTER_MODEL_PATH = os.path.join(MODEL_DIR, f"imputer_model_{ACTIVE_MODEL_ID}.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_{ACTIVE_MODEL_ID}.joblib") 

# 硬性規則限制 (與 cleaning-api 的邏輯保持一致)
POWER_LIMITS = {
    "cpu_power_watt": (0.0, 150.0), 
    "gpu_power_watt": (0.0, 350.0),
    "system_power_watt": (0.0, 500.0) 
}
FLOAT_FIELDS = [
    "gpu_usage_percent", "gpu_power_watt", "cpu_power_watt",
    "memory_used_mb", "disk_read_mb_s", "disk_write_mb_s", "system_power_watt"
]

# 載入 AI 模型
ANOMALY_MODEL = None
IMPUTER_MODEL = None
SCALER = None

#print(f"🔍 檢查模型路徑...")
#print(f"   BASE_DIR: {BASE_DIR}")
#print(f"   MODEL_DIR: {MODEL_DIR}")
#print(f"   ACTIVE_MODEL_ID: {ACTIVE_MODEL_ID}")
#print(f"   Anomaly model path: {ANOMALY_MODEL_PATH}")
#print(f"   檔案是否存在: {os.path.exists(ANOMALY_MODEL_PATH)}")

try:
    if ACTIVE_MODEL_ID == "YOUR_MODEL_ID_HERE":
        print("⚠️ Model ID 尚未設定")
        raise FileNotFoundError("Model ID not configured")
    
    if not os.path.exists(ANOMALY_MODEL_PATH):
        print(f"⚠️ 模型檔案不存在: {ANOMALY_MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {ANOMALY_MODEL_PATH}")
    
    print("📦 載入 Anomaly Model...")
    ANOMALY_MODEL = joblib.load(ANOMALY_MODEL_PATH)
    print("📦 載入 Imputer Model...")
    IMPUTER_MODEL = joblib.load(IMPUTER_MODEL_PATH)
    print("📦 載入 Scaler...")
    SCALER = joblib.load(SCALER_PATH)
    
    print(f"✅ AI Models (ID: {ACTIVE_MODEL_ID}) Loaded Successfully for Local Cleaning.")
    
except FileNotFoundError as e:
    print(f"⚠️ WARNING: Model files not found: {e}")
    print(f"   Using basic rules only.")
except Exception as e:
    print(f"⚠️ WARNING: Failed to load AI models: {type(e).__name__}: {e}")
    print(f"   Model path: {ANOMALY_MODEL_PATH}")
    print(f"   Using basic rules only.")
    import traceback
    traceback.print_exc()  # 印出完整錯誤堆疊


# ---------- Configuration ----------
API_BASE_URL = "http://localhost:8000"
AUTH_SECRET_KEY = "NTCUST-ENERGY-MONITOR"
FALLBACK_TO_CSV = True

# config.yaml
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def load_config():
    """ Loads the configuration from config.yaml. """
    config_path = resource_path("config.yaml") # <--- 關鍵修改
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # This warning will now only appear if the file was not bundled correctly
        print("WARNING: config.yaml not found. Using default settings.")
        return {
            'collection_interval': 60,
            'quota': {'daily_limit_kwh': 100},
            'api_base_url': 'http://localhost:8000', # Added default for safety
            'auth_secret_key': 'NTCUST-ENERGY-MONITOR'
        }

config = load_config()
API_BASE_URL = config.get('api_base_url', API_BASE_URL)
AUTH_SECRET_KEY = config.get('auth_secret_key', AUTH_SECRET_KEY)
FALLBACK_TO_CSV = config.get('fallback_to_csv', FALLBACK_TO_CSV)


# ---------- Class Schedule ----------
class_periods = [
    ("08:10", "09:00"), ("09:10", "10:00"),
    ("10:10", "11:00"), ("11:10", "12:00"),
    ("13:25", "14:15"), ("14:20", "15:10"),
    ("15:20", "16:10"), ("16:15", "17:05")
]

def is_class_time():
    now = datetime.now().time()
    for start_str, end_str in class_periods:
        start = dtime.fromisoformat(start_str)
        end = dtime.fromisoformat(end_str)
        if start <= now <= end:
            return True
    return False

# ---------- MAC Address and Authentication ----------
def get_mac_address():
    """Get device MAC address"""
    try:
        mac = uuid.getnode()
        mac_str = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) 
                           for elements in range(0,2*6,2)][::-1])
        return mac_str.upper()
    except:
        try:
            import netifaces
            interfaces = netifaces.interfaces()
            for interface in interfaces:
                if interface != 'lo':
                    addrs = netifaces.ifaddresses(interface)
                    if netifaces.AF_LINK in addrs:
                        mac = addrs[netifaces.AF_LINK][0]['addr']
                        return mac.upper().replace('-', ':')
        except:
            pass
        
        try:
            if platform.system() == "Windows":
                result = subprocess.run(['getmac'], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for line in lines:
                    if '-' in line and len(line.split('-')) == 6:
                        return line.replace('-', ':').upper().strip()
            else:
                result = subprocess.run(['ifconfig'], capture_output=True, text=True)
        except:
            pass
    
    return "00:00:00:00:00:00"

def generate_device_certificate(mac_address, secret_key):
    """Generate device certificate"""
    return hmac.new(
        secret_key.encode(), 
        mac_address.encode(), 
        hashlib.sha256
    ).hexdigest()

def get_auth_headers():
    """Get authentication headers"""
    mac_address = get_mac_address()
    certificate = generate_device_certificate(mac_address, AUTH_SECRET_KEY)
    
    return {
        "Content-Type": "application/json",
        "MAC-Address": mac_address,
        "Device-Certificate": certificate
    }

# integrated_agent.py (替換原 DataCleaner 類別)

class AIDataCleaner:
    """
    AI 資料清洗模組 (AI Agent 的核心清洗邏輯)
    整合了硬性規則、AI 異常值標記和智慧填補。
    """
    def __init__(self, anomaly_model, imputer_model, scaler):
        self.anomaly_model = anomaly_model
        self.imputer_model = imputer_model
        self.scaler = scaler

    def clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = data.copy()

        # ================= 1. 原始固定規則：類別資料清洗 =================
        # 處理 unknown GPU 型號
        if cleaned.get("gpu_model", "").lower() == "unknown":
            cleaned["gpu_model"] = "Generic GPU"
        
        data_df = pd.DataFrame([cleaned])
        
        # ================= 2. 數值準備：型別轉換與標記 NaN =================
        for field in FLOAT_FIELDS:
            # errors='coerce' 將無法轉換的值設為 NaN
            data_df[field] = pd.to_numeric(data_df.get(field), errors='coerce')
            if data_df[field].isnull().iloc[0]:
                cleaned[field] = np.nan

        cleaned['is_ai_spike'] = False

        # --- AI 清洗流程 (三階段) ---
        if self.anomaly_model is not None and self.imputer_model is not None and self.scaler is not None:
            try:
                # 階段 A：硬性規則過濾（識別物理錯誤）
                for field, (min_val, max_val) in POWER_LIMITS.items():
                    if field in data_df.columns:
                        is_physical_error = (data_df[field] < min_val) | (data_df[field] > max_val)
                        if is_physical_error.iloc[0]:
                            data_df.loc[0, field] = np.nan 
                
                numeric_data = data_df[FLOAT_FIELDS]
                
                # 階段 B：AI 異常值偵測 (M_A) - 僅標記真實高峰
                numeric_data_filled_for_ma = numeric_data.fillna(numeric_data.mean().fillna(0.0)) 
                
                is_anomaly = self.anomaly_model.predict(numeric_data_filled_for_ma)
                
                # Record-Level Flagging：標記為 Spike (保留數值)
                if is_anomaly[0] == -1:
                    print("AI Detected: Statistical spike (retained original high value).")
                    cleaned['is_ai_spike'] = True
                
                # 階段 C：智慧填補 (M_I) - 修復 NaN
                numeric_data_scaled = self.scaler.transform(numeric_data)
                imputed_array_scaled = self.imputer_model.transform(numeric_data_scaled)
                imputed_array = self.scaler.inverse_transform(imputed_array_scaled) # 反標準化
                
                numeric_data_filled_final = pd.DataFrame(imputed_array, columns=FLOAT_FIELDS, index=numeric_data.index)
                
                # 更新 cleaned 字典 (只有錯誤值被填補了)
                for field in FLOAT_FIELDS:
                    cleaned[field] = float(numeric_data_filled_final.loc[0, field])

            except Exception as e:
                print(f"AI cleaning failed locally: {e}. Falling back to original 0.0 filling.")
                cleaned['is_ai_spike'] = False
                # 最終補救：將所有的 NaN 值補為 0.0
                for field in FLOAT_FIELDS:
                    if pd.isna(cleaned.get(field)):
                         cleaned[field] = 0.0
            
        else:
            # 如果 AI 模型未載入，執行原始程式碼的 0.0 補值邏輯
            for field in FLOAT_FIELDS:
                if pd.isna(cleaned.get(field)):
                     cleaned[field] = 0.0

        # --- 原始 Agent 邏輯：新增 system_power_kw 與 usage_pattern ---
        # 該邏輯來自原 DataCleaner 類別
        cleaned['system_power_kw'] = cleaned.get('system_power_watt', 0) / 1000
        
        # 使用模式判斷 (需使用新的長鍵名)
        total_util = cleaned.get('cpu_power_watt', 0) * 0.4 + cleaned.get('gpu_usage_percent', 0) * 0.4
        if total_util < 20:
            cleaned['usage_pattern'] = 'idle'
        elif total_util < 50:
            cleaned['usage_pattern'] = 'normal'
        elif total_util < 80:
            cleaned['usage_pattern'] = 'intensive'
        else:
            cleaned['usage_pattern'] = 'peak'
            
        return cleaned

class QuotaManager:
    """額度管理模組（簡化版）"""
    def __init__(self, config):
        self.daily_limit_kwh = config.get('quota', {}).get('daily_limit_kwh', 100)
        self.warning_threshold = config.get('quota', {}).get('warning_threshold', 0.8)
        self.critical_threshold = config.get('quota', {}).get('critical_threshold', 0.95)
        self.usage_cache = {}
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """處理配額"""
        user_id = data.get('user_id', 'default')
        
        # 簡化的使用量計算
        energy_consumed_kwh = data.get('system_power_kw', 0) * (1/60)  # 每分鐘
        
        # 獲取或初始化今日使用量
        today_key = datetime.now().strftime('%Y-%m-%d')
        if today_key not in self.usage_cache:
            self.usage_cache[today_key] = {}
        
        if user_id not in self.usage_cache[today_key]:
            self.usage_cache[today_key][user_id] = 0
            
        # 更新使用量
        self.usage_cache[today_key][user_id] += energy_consumed_kwh
        daily_usage = self.usage_cache[today_key][user_id]
        
        # 計算剩餘和百分比
        daily_remaining = self.daily_limit_kwh - daily_usage
        daily_percentage = (daily_usage / self.daily_limit_kwh) * 100
        
        # 判斷警告等級
        alert_level = 'normal'
        alert_message = ''
        
        if daily_percentage >= 100:
            alert_level = 'exceeded'
            alert_message = f'配額已超出: {daily_usage:.2f}/{self.daily_limit_kwh} kWh'
        elif daily_percentage >= self.critical_threshold * 100:
            alert_level = 'critical'
            alert_message = f'配額即將用盡: {daily_percentage:.1f}%'
        elif daily_percentage >= self.warning_threshold * 100:
            alert_level = 'warning'
            alert_message = f'配額使用警告: {daily_percentage:.1f}%'
            
        return {
            'daily_usage_kwh': daily_usage,
            'daily_remaining_kwh': daily_remaining,
            'daily_percentage': daily_percentage,
            'alert_level': alert_level,
            'alert_message': alert_message
        }

# ---------- Enhanced System Info Collection ----------
def get_enhanced_system_info():
    """Collect detailed system info for device fingerprinting"""
    try:
        system_info = {
            "cpu_model": platform.processor() or "Unknown",
            "cpu_count": psutil.cpu_count(),
            "total_memory": psutil.virtual_memory().total,
            "disk_partitions": len(psutil.disk_partitions()),
            "network_interfaces": len(psutil.net_if_addrs()),
            "platform_machine": platform.machine(),
            "platform_architecture": platform.architecture()[0]
        }
        return system_info
    except:
        return {}

# ---------- Hardware Data Collection ----------
def get_gpu_model():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.stderr:
            return "Unknown"
        return result.stdout.decode('utf-8').strip()
    except:
        return "Unknown"

def get_gpu_usage():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.stderr:
            return 0
        usage = result.stdout.decode('utf-8').strip()
        return float(usage) if usage else 0
    except:
        return 0

def get_gpu_power_watt():
    """Get GPU power consumption in Watts"""
    try:
        # Method 1: Direct from nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0 and not result.stderr:
            power = result.stdout.decode('utf-8').strip()
            if power and power != "N/A":
                return float(power)
    except:
        pass
    
    # Method 2: Estimate based on usage
    gpu_usage = get_gpu_usage()
    gpu_model = get_gpu_model().lower()
    
    # Set power ranges based on GPU model
    if "mx250" in gpu_model:
        base_power, max_power = 5.0, 25.0
    elif "rtx" in gpu_model:
        if "4090" in gpu_model:
            base_power, max_power = 50.0, 450.0
        elif "4080" in gpu_model:
            base_power, max_power = 40.0, 320.0
        elif "4070" in gpu_model:
            base_power, max_power = 30.0, 200.0
        else:
            base_power, max_power = 25.0, 250.0
    elif "gtx" in gpu_model:
        base_power, max_power = 15.0, 180.0
    else:
        base_power, max_power = 10.0, 75.0
    
    additional_power = (gpu_usage / 100.0) * (max_power - base_power)
    estimated_power = base_power + additional_power
    
    return round(estimated_power, 2)

def get_cpu_power():
    """Improved CPU power calculation"""
    cpu_percent = psutil.cpu_percent(interval=1)
    
    try:
        cpu_info = platform.processor().lower()
        cpu_count = psutil.cpu_count()
    except:
        cpu_info = ""
        cpu_count = 4
    
    # Estimate power based on CPU type and core count
    if "intel" in cpu_info:
        if "i9" in cpu_info or "xeon" in cpu_info:
            base_power = 15.0 + (cpu_count * 2)
            max_additional = 50.0 + (cpu_count * 5)
        elif "i7" in cpu_info:
            base_power = 12.0 + (cpu_count * 1.5)
            max_additional = 35.0 + (cpu_count * 4)
        elif "i5" in cpu_info:
            base_power = 10.0 + (cpu_count * 1.2)
            max_additional = 25.0 + (cpu_count * 3)
        else:
            base_power = 8.0 + (cpu_count * 1)
            max_additional = 20.0 + (cpu_count * 2.5)
    elif "amd" in cpu_info:
        if "ryzen 9" in cpu_info or "threadripper" in cpu_info:
            base_power = 15.0 + (cpu_count * 1.8)
            max_additional = 45.0 + (cpu_count * 4.5)
        elif "ryzen 7" in cpu_info:
            base_power = 12.0 + (cpu_count * 1.4)
            max_additional = 30.0 + (cpu_count * 3.5)
        elif "ryzen 5" in cpu_info:
            base_power = 10.0 + (cpu_count * 1.2)
            max_additional = 25.0 + (cpu_count * 3)
        else:
            base_power = 8.0 + (cpu_count * 1)
            max_additional = 20.0 + (cpu_count * 2.5)
    else:
        base_power = 10.0 + (cpu_count * 1)
        max_additional = 25.0 + (cpu_count * 3)
    
    additional_power = (cpu_percent / 100.0) * max_additional
    total_power = base_power + additional_power
    
    return round(total_power, 2)

def get_memory_usage():
    memory = psutil.virtual_memory()
    return memory.used / (1024 * 1024)

def get_disk_read_write_rate(interval=1):
    before = psutil.disk_io_counters()
    time.sleep(interval)
    after = psutil.disk_io_counters()

    read_rate = (after.read_bytes - before.read_bytes) / (1024 * 1024) / interval
    write_rate = (after.write_bytes - before.write_bytes) / (1024 * 1024) / interval
    return round(read_rate, 2), round(write_rate, 2)

def get_system_power(cpu, gpu, memory):
    """Improved system total power calculation"""
    # Memory power: DDR4/DDR5 ~3-4W per GB
    memory_gb = memory / 1024.0
    memory_power = memory_gb * 3.5
    
    # Base system power (motherboard, fans, SSD, network card, etc.)
    motherboard_power = 15.0
    cooling_power = 5.0
    storage_power = 5.0
    other_power = 10.0
    
    base_system_power = motherboard_power + cooling_power + storage_power + other_power
    
    # Calculate total power
    total_power = cpu + gpu + memory_power + base_system_power
    
    # PSU efficiency loss (assume 90% efficiency)
    efficiency_factor = 1.11  # 1/0.9
    
    final_power = total_power * efficiency_factor
    
    return round(final_power, 2)


def validate_power_readings(data):
    """Validate power readings for reasonableness (使用長鍵名)"""
    # 這裡的極限值已經被 AGENT 程式碼中的設定覆蓋了，但仍需統一鍵名。
    limits = {
        'cpu_power_watt': 125.0,
        'gpu_power_watt': 500.0,
        'system_power_watt': 800.0
    }
    
    warnings = []
    
    for key, limit in limits.items():
        if key in data and data[key] > limit:
            warnings.append(f"{key}: {data[key]}W -> {limit}W")
            data[key] = limit
    
    # Logic check: system power should not be less than CPU + GPU power
    min_system_power = data.get('cpu_power_watt', 0) + data.get('gpu_power_watt', 0) + 20
    if 'system_power_watt' in data and data['system_power_watt'] < min_system_power:
        warnings.append(f"system_power_watt: {data['system_power_watt']}W -> {min_system_power}W (logic adjustment)")
        data['system_power_watt'] = min_system_power
    
    if warnings:
        print(f"Power value corrections: {', '.join(warnings)}")
    
    return data

def get_timestamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def get_location():
    """Get location based on public IP address."""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            city = data.get("city", "")
            country = data.get("country", "")
            if city and country:
                return f"{city}, {country}"
    except requests.exceptions.RequestException:
        pass  # Ignore connection errors
    return "Unknown"

def get_device_info():
    return (
        str(uuid.getnode()),
        getpass.getuser(),
        "v1.4.0",
        platform.system(),
        platform.version(),
        get_location()
    )

# ---------- Data Transmission ----------
def send_to_api(data):
    """Send data to ingestion-api with device fingerprinting"""
    try:
        headers = get_auth_headers()
        
        # 這裡的鍵名已經全部更新為長鍵名，與 collect_data 的輸出一致
        api_data = {
            # Basic energy data
            "timestamp_utc": data["timestamp"],
            "gpu_model": data["gpu_model"],
            "gpu_usage_percent": data["gpu_usage_percent"],
            "gpu_power_watt": data["gpu_power_watt"],
            "cpu_power_watt": data["cpu_power_watt"],
            "memory_used_mb": data["memory_used_mb"],
            "disk_read_mb_s": data["disk_read_mb_s"],
            "disk_write_mb_s": data["disk_write_mb_s"],
            "system_power_watt": data["system_power_watt"],
            "device_id": data["device_id"],
            "user_id": data["user_id"],
            "agent_version": data["agent_version"],
            "os_type": data["os_type"],
            "os_version": data["os_version"],
            "location": data["location"],

                        
            # --- 新增：AI 清洗結果欄位映射 ---
            "is_anomaly": data.get("is_ai_spike", False), # 將 is_ai_spike 映射到 is_anomaly
            "anomaly_reason": data.get("anomaly_reason", "No reason provided."),
            # --- 結束新增 ---
            
            # Device fingerprint for security
            "cpu_model": data.get("cpu_model"),
            "cpu_count": data.get("cpu_count"),
            "total_memory": data.get("total_memory"),
            "disk_partitions": data.get("disk_partitions"),
            "network_interfaces": data.get("network_interfaces"),
            "platform_machine": data.get("platform_machine"),
            "platform_architecture": data.get("platform_architecture")
        }
        
        print(f"Sending data with device fingerprint to API...")
        # 這裡使用長鍵名，確保輸出正確
        print(f"Basic data: CPU={data['cpu_power_watt']}W, GPU={data['gpu_power_watt']}W, System={data['system_power_watt']}W")
        print(f"Device fingerprint: {data.get('cpu_model', 'Unknown')} ({data.get('cpu_count', 'Unknown')} cores)")

        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json=api_data,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display fingerprint check results
            if "fingerprint_check" in result:
                fp_result = result["fingerprint_check"]
                risk_level = fp_result.get("risk_level", "unknown")
                message = fp_result.get("message", "")
                similarity = fp_result.get("similarity_score", 0)
                
                if risk_level == "high":
                    print(f"HIGH RISK device warning: {message} (similarity: {similarity:.2f})")
                elif risk_level == "medium":
                    print(f"Medium risk alert: {message} (similarity: {similarity:.2f})")
                else:
                    print(f"Device normal: {message} (similarity: {similarity:.2f})")
            
            print(f"Data successfully sent to API: {result.get('status', 'unknown')}")
            return True
            
        elif response.status_code == 401:
            print(f"Authentication failed: {response.json().get('detail', 'Unknown auth error')}")
            return False
        elif response.status_code == 403:
            print(f"Device not authorized: {response.json().get('detail', 'Device not authorized')}")
            print(f"   Your MAC address: {get_mac_address()}")
            print(f"   Please contact admin to add this device to whitelist")
            return False
        else:
            print(f"API response error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to API: {API_BASE_URL}")
        return False
    except requests.exceptions.Timeout:
        print("API request timeout")
        return False
    except Exception as e:
        print(f"Failed to send data: {str(e)}")
        return False

# ---------- CSV Backup Storage ----------
data_buffer = []
file_count = 0
output_dir = "agent_logs"
os.makedirs(output_dir, exist_ok=True)

def save_to_csv(row):
    global data_buffer, file_count
    data_buffer.append(row)
    if len(data_buffer) >= 50:
        filename = os.path.join(output_dir, f"agent_data_{file_count}.csv")
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(data_buffer)
        print(f"CSV backup saved: {filename}")
        data_buffer = []
        file_count += 1

# ---------- Initialization and Health Check ----------
def check_api_connection():
    """Check API connection and verify device registration status"""
    try:
        # Check API health status
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("API service running normally")
        else:
            print(f"API health check abnormal: {response.status_code}")
    except:
        print(f"Cannot connect to API: {API_BASE_URL}")
        if FALLBACK_TO_CSV:
            print("Will use CSV backup mode")
        return False
    
    # Check device registration
    mac_address = get_mac_address()
    print(f"Device MAC address: {mac_address}")
    print(f"Device fingerprint function: Enabled")
    print(f"Power calculation: Optimized (smart estimation)")
    
    try:
        headers = get_auth_headers()
        response = requests.get(f"{API_BASE_URL}/admin/devices/{mac_address}", headers=headers, timeout=5)
        
        if response.status_code == 200:
            device_info = response.json()
            print(f"Device registered: {device_info['device_name']}")
            return True
        elif response.status_code == 404:
            print("Device not yet registered to whitelist, but fingerprint function still operational")
            return True
        else:
            print(f"Check device registration status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Check device registration failed: {e}")
        return False

# ---------- 主要整合類別 ----------

class IntegratedAgent:
    def __init__(self, config):
        """初始化整合後的 Agent"""
        self.config = config
        # 使用新的 AI 清洗模組，並傳入載入的模型物件
        self.cleaner = AIDataCleaner(ANOMALY_MODEL, IMPUTER_MODEL, SCALER)
        self.quota_manager = QuotaManager(config)
        
        # 狀態管理
        self.previous_data = {}
        self.change_threshold = config.get('change_threshold', 5)
        
        # 使用者活動監測
        self.user_active = False
        self._start_input_monitoring()

    def _start_input_monitoring(self):
        """在背景執行緒中啟動使用者輸入監測"""
        def on_event(x):
            self.user_active = True

        def monitor_loop():
            try:
                with mouse.Listener(on_click=on_event) as m_listener, \
                     keyboard.Listener(on_press=on_event) as k_listener:
                    m_listener.join()
                    k_listener.join()
            except Exception as e:
                print(f"輸入監控啟動失敗: {e}")
        
        input_thread = threading.Thread(target=monitor_loop, daemon=True)
        input_thread.start()

    def has_significant_change(self, new_data):
        """檢查是否有顯著變化"""
        if not self.previous_data:
            return True
        
        keys_to_check = ['cpu_power_watt', 'gpu_power_watt', 'memory_used_mb']
        
        changes = [
            key for key in keys_to_check 
            if abs(new_data.get(key, 0) - self.previous_data.get(key, 0)) > self.change_threshold
        ]
        
        if changes:
            print(f"資料變動超過閾值: {', '.join(changes)}")
            return True
        return False

    def process_and_send_data(self):
        """
        整合了資料收集、清洗、配額計算和發送的完整流程
        【注意】這段程式碼必須在方法內部，不能放在類別定義的主體中
        """
        # 1. 收集原始數據
        raw_data = self.collect_data()
        
        # 2. 驗證並修正功耗數據
        validated_data = validate_power_readings(raw_data)
        
        # 3. 清洗資料 (AI Agent 在此執行 AI 清洗)
        cleaned_data = self.cleaner.clean(validated_data)
        
        # --- 修改此處：更精確的異常原因判斷 ---
        is_anomaly = cleaned_data.get('is_ai_spike', False)
        
        if is_anomaly:
            # 如果是 AI 標記的統計異常
            anomaly_reason = "Statistical spike detected by AI (original high value retained)"
        elif cleaned_data.get("gpu_model") == "Generic GPU":
            # GPU 資料被清洗
            anomaly_reason = "GPU model unknown and replaced with generic placeholder"
            is_anomaly = False  # 這不算真正的異常
        else:
            # 正常資料
            anomaly_reason = None
            is_anomaly = False
        
        # 4. 處理配額
        quota_info = self.quota_manager.process(cleaned_data)

        # 5. 將所有資訊合併為最終數據包
        final_data = {
            **cleaned_data, 
            **quota_info, 
            "is_anomaly": is_anomaly,           # 明確設定
            "anomaly_reason": anomaly_reason    # 明確設定
        }

        # 顯示監控資訊
        print(f"\n✅ 功耗監控 - CPU: {final_data['cpu_power_watt']}W | GPU: {final_data['gpu_power_watt']}W | 系統: {final_data['system_power_watt']}W")
        print(f"   本日配額使用: {final_data.get('daily_percentage', 0):.2f}%")
        
        if is_anomaly:
            print(f"   ⚠️ AI 標記：{anomaly_reason}")
        else:
            print(f"   ✓ 正常運作")

        # 6. 嘗試發送到 API
        api_success = send_to_api(final_data)
        
        # 7. 如果 API 失敗且啟用備援，則儲存到 CSV
        if not api_success and FALLBACK_TO_CSV:
            print("API 發送失敗，使用 CSV 備援儲存")
            save_to_csv(final_data)

    def collect_data(self):
        """收集所有系統和硬體數據"""
        device_id, user_id, agent_version, os_type, os_version, location = get_device_info()
        timestamp = get_timestamp()

        gpu_model = get_gpu_model()
        gpu_usage = get_gpu_usage()
        gpu_power = get_gpu_power_watt()
        cpu_power = get_cpu_power()
        memory_used = get_memory_usage()
        disk_read, disk_write = get_disk_read_write_rate(interval=1)
        system_power = get_system_power(cpu_power, gpu_power, memory_used)
        enhanced_info = get_enhanced_system_info()

        return {
            "timestamp": timestamp, 
            "cpu_power_watt": cpu_power,
            "gpu_power_watt": gpu_power,
            "memory_used_mb": memory_used, 
            "disk_read_mb_s": disk_read, 
            "disk_write_mb_s": disk_write,
            "gpu_usage_percent": gpu_usage, 
            "gpu_model": gpu_model, 
            "system_power_watt": system_power, 
            "device_id": device_id, "user_id": user_id, "agent_version": agent_version,
            "os_type": os_type, "os_version": os_version, "location": location,
            **enhanced_info
        }

    def run(self):
        """啟動 Agent 的主循環"""
        print("整合版 Agent 啟動中...")
        print(f"API 地址: {API_BASE_URL}")
        print(f"版本: v2.0 (整合版)")
        
        api_available = check_api_connection()
        if not api_available and not FALLBACK_TO_CSV:
            print("API 不可用且未啟用 CSV 備援，程式結束")
            return
            
        print("開始監控...")
        
        while True:
            try:
                should_collect = is_class_time()
                if not should_collect and self.user_active:
                    print("偵測到使用者活動，進行一次資料收集...")
                    should_collect = True
                    self.user_active = False

                if should_collect:
                    current_snapshot = {
                        "cpu_power_watt": get_cpu_power(),
                        "gpu_power_watt": get_gpu_power_watt(),
                        "memory_used_mb": get_memory_usage()
                    }
                    
                    if self.has_significant_change(current_snapshot):
                        self.process_and_send_data()
                        self.previous_data = current_snapshot
                    else:
                        print("數據無顯著變化，跳過本次傳送...")
                
                time.sleep(self.config.get('collection_interval', 60))

            except KeyboardInterrupt:
                print("\nAgent 停止運行")
                break
            except Exception as e:
                print(f"主循環發生錯誤: {e}") 
                time.sleep(60)

# ---------- 主執行點 ----------

if __name__ == "__main__":
    # 建立 Agent 實例並傳入設定
    agent = IntegratedAgent(config)
    # 執行 Agent
    agent.run()