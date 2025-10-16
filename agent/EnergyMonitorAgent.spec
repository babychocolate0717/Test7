# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['integrated_agent.py'],
    pathex=[],
    binaries=[],
    datas=[('config.yaml', '.'), ('LibreHardwareMonitorLib.dll', '.'), ('..\\models\\cleaning_models\\anomaly_model_20251012_164538.joblib', 'models\\cleaning_models'), ('..\\models\\cleaning_models\\imputer_model_20251012_164538.joblib', 'models\\cleaning_models'), ('..\\models\\cleaning_models\\scaler_20251012_164538.joblib', 'models\\cleaning_models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('v', None, 'OPTION')],
    name='EnergyMonitorAgent',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
