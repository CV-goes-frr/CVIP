# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
def get_mediapipe_path():
    import mediapipe
    mediapipe_path = mediapipe.__path__[0]
    return mediapipe_path


a = Analysis(
    ['CVIP.py'],
    pathex=[],
    binaries=[],
    datas=[
            ('src/filters/shape_predictor_81_face_landmarks.dat', '.'),
            ('src/filters/haarcascade_frontalface_alt2.xml', '.'),
            ('help.txt', '.')
    ],
    hiddenimports=["mediapipe", "multipledispatch"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure,
    a.zipped_data,
    cipher=block_cipher)

mediapipe_tree = Tree(get_mediapipe_path(), prefix='mediapipe', excludes=["*.pyc"])
a.datas += mediapipe_tree

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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
