#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Bert
# DATE     : August 2018
#
# Module init
#

__version__ = '1.2.5'

# Windows: Pillow must be loaded before torchvision (pulled in by onnx2torch).
# Both ship image-codec DLLs with the same base names, and Pillow then fails
# with "DLL load failed while importing _imaging" if torchvision loads first.
try:
  from PIL import Image as _dgb_pil_preload
except Exception:
  pass
