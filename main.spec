# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['D:\\Documents\\Lam�\\src\\AR-Python\\Camera recognition'],
             binaries=[],
             datas=[('D:\Documents\Lam�\src\AR-Python\Camera recognition\config.yml', '.'),
             ('D:\\Documents\\Lam�\\src\\AR-Python\\Camera recognition\\shader', 'shader'),
             ('D:\\Documents\\Lam�\\src\\AR-Python\\Camera recognition\\font', 'font'),
             ('D:\\Documents\\Lam�\\src\\AR-Python\\Camera recognition\\texture', 'texture')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='AcierRA',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='run')
