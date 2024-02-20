# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['CVIP.py'],
    pathex=[],
    binaries=[],
    datas=[
            ('src/filters/shape_predictor_81_face_landmarks.dat', '.'),
            ('help.txt', '.')
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure,
    a.zipped_data)


exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CVIP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
