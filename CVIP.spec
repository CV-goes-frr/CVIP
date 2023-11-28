# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['CVIP.py'],
    pathex=[],
    binaries=[],
    datas=[
            ('src/filters/shape_predictor_81_face_landmarks.dat', '.'),
            ('src/filters/shape_predictor_81_face_landmarks.dat', '.'),
            ('help.txt', '.')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CVIP',
    debug=False,
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
