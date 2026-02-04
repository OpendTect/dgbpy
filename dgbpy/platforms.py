#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# _________________________________________________________________________
import dgbpy.keystr as dbk
from dgbpy import uikeras, uisklearn, uitorch


class uinoplfm:
    def __init__(self):
        self.platform = ('No platform', 'No Available Platform')

    def getPlatformNm(self, full=False ):
        if full:
            return self.platform
        return self.platform[0]

def get_platforms():
    mlplatform = []
    if uitorch.hasTorch():
        mlplatform.append( uitorch.getPlatformNm(True) )
    if uikeras.hasKeras():
        mlplatform.append( uikeras.getPlatformNm(True) )
    if uisklearn.hasScikit():
        mlplatform.append( uisklearn.getPlatformNm(True) )
    if not bool(mlplatform):
        mlplatform.append(uinoplfm().getPlatformNm(True) )
    return mlplatform

def get_default_platform(mllearntype=dbk.loglogtypestr):
    if mllearntype == dbk.loglogtypestr or \
      mllearntype == dbk.logclustertypestr or \
      mllearntype == dbk.seisproptypestr:
      if uisklearn.hasScikit():
        return uisklearn.getPlatformNm(True)[0]
    else:
      if uitorch.hasTorch():
        return uitorch.getPlatformNm(True)[0]
      if uikeras.hasKeras():
        return uikeras.getPlatformNm(True)[0]
    return uinoplfm().getPlatformNm(True)[0]

def getPlatformInfo( platform ):
    infos = {}
    allplatforms = [pltfrm[0] for pltfrm in get_platforms()]
    if (platform==dbk.kerasplfnm or dbk.kerasplfnm in platform) and list(platform.keys())[0] in allplatforms:
        try:
            from dgbpy.dgbkeras import get_keras_infos
            infos = get_keras_infos()
        except Exception:
            pass
    elif (platform==dbk.torchplfnm or dbk.torchplfnm in platform) and list(platform.keys())[0] in allplatforms:
        try:
            from dgbpy.dgbtorch import get_torch_infos
            infos = get_torch_infos()
        except Exception:
            pass
    return infos
