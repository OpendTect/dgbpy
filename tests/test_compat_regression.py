"""Regression tests for the numpy 2 / torch 2.6 / hdf5 2.x compatibility fixes

Copyright (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt

These cover save and load paths no other test touches. They need no example
data, so a bare environment can run them.
"""

import sys
sys.path.insert(0, '..')

import os
import json

import numpy as np
import pytest

import odpy.compat as compat
import odpy.hdf5 as odhdf5
import dgbpy.keystr as dbk
import dgbpy.hdf5 as dgbhdf5

from init_data import get_loglog_info


def test_char_encode_returns_fixed_length_bytes():
    """The '|S' dtype is what keeps files readable by OpendTect's C++ side."""
    out = compat.char_encode( 'scikit-learn' )
    assert out.dtype.kind == 'S'
    assert out.dtype.itemsize == len( 'scikit-learn' )


@pytest.mark.parametrize( 'actual,minimum,expected', [
    ('1.9.0',  '1.10', False),   # string compare would say True
    ('1.10.0', '1.9',  True),
    ('3.10.0', '3.2',  True),    # float(v[:3]) would say False
    ('0.24.2', '1.0',  False),
] )
def test_version_at_least( actual, minimum, expected ):
    assert compat.version_at_least( actual, minimum ) is expected
    got = compat._versiontuple( actual ) >= compat._versiontuple( minimum )
    assert got is expected


def test_platform_capability_checks_return_bool():
    import dgbpy.dgbkeras as dgbkeras
    import dgbpy.dgbscikit as dgbscikit
    import dgbpy.dgbtorch as dgbtorch
    for fn in (dgbkeras.hasKeras, dgbscikit.hasScikit, dgbtorch.hasTorch,
               dgbscikit.hasXGBoost):
        assert isinstance( fn(), bool )


# --- HDF5 attributes: np.chararray.encode was removed in numpy 2 -------------

def _make_info_dataset( path ):
    h5file = odhdf5.openFile( path, 'w' )
    return h5file, odhdf5.ensureHasDataset( h5file )


def test_setattr_roundtrip_preserves_fixed_length_dtype( tmp_path ):
    fnm = os.path.join( str(tmp_path), 'attrs.h5' )
    h5file, dset = _make_info_dataset( fnm )
    odhdf5.setAttr( dset, 'Model.Type', 'scikit-learn' )
    odhdf5.setAttr( dset, 'Output.Unscale', dgbhdf5.odsetBoolValue(True) )
    odhdf5.setAttr( dset, 'model_config', json.dumps({'max_depth': None}) )
    odhdf5.setArray( dset, 'Input.Shape', [1, 8, 8] )
    odhdf5.setArray( dset, 'Nr.Outputs', 3 )
    assert dset.attrs.get_id( 'Model.Type' ).dtype.kind == 'S'
    h5file.close()

    h5file = odhdf5.openFile( fnm, 'r' )
    dset = odhdf5.getInfoDataSet( h5file )
    assert odhdf5.getText( dset, 'Model.Type' ) == 'scikit-learn'
    assert odhdf5.getBoolValue( dset, 'Output.Unscale' ) is True
    assert json.loads( odhdf5.getText(dset, 'model_config') ) == {'max_depth': None}
    assert odhdf5.getIArray( dset, 'Input.Shape' ) == [1, 8, 8]
    assert odhdf5.getIntValue( dset, 'Nr.Outputs' ) == 3
    h5file.close()


def test_addinfo_writes_model_attributes( tmp_path ):
    """dgbhdf5.addInfo runs on every saveModel for every platform."""
    inpfnm = os.path.join( str(tmp_path), 'examples.h5' )
    outfnm = os.path.join( str(tmp_path), 'model.h5' )

    h5file, dset = _make_info_dataset( inpfnm )
    odhdf5.setAttr( dset, 'Type', 'Log-Log Prediction' )
    h5file.close()
    odhdf5.openFile( outfnm, 'w' ).close()

    infos = get_loglog_info()
    infos[dbk.outputunscaledictstr] = False
    for inputnm in infos[dbk.inputdictstr]:
        infos[dbk.inputdictstr][inputnm].pop( dbk.scaledictstr, None )
    dgbhdf5.addInfo( inpfnm, dbk.scikitplfnm, outfnm, infos,
                     'RandomForestClassifier' )

    h5file = odhdf5.openFile( outfnm, 'r' )
    dset = odhdf5.getInfoDataSet( h5file )
    assert odhdf5.getText( dset, 'Model.Type' ) == dbk.scikitplfnm
    assert odhdf5.getText( dset, 'Model.Class' ) == 'RandomForestClassifier'
    assert odhdf5.getText( dset, 'Type' ) == 'Log-Log Prediction'
    h5file.close()


# --- getCubeLetNames: np.chararray.startswith was removed in numpy 2 ---------

def _write_multi_item_group( fnm, groupnm, dsetnms, nrpts=4 ):
    h5file = odhdf5.openFile( fnm, 'w' )
    grp = h5file.create_group( groupnm )
    for nm in dsetnms:
        sub = grp.create_group( nm )
        sub.create_dataset( dbk.xdatadictstr,
                            data=np.zeros((nrpts, 1, 8, 8), dtype=np.float32) )
    h5file.close()


def test_getcubeletnames_prefix_branch( tmp_path ):
    fnm = os.path.join( str(tmp_path), 'multi.h5' )
    _write_multi_item_group( fnm, 'Group1',
                             ['0: Well A', '1: Well B', '0: Well C', '2: Well D'] )
    info = { dbk.filedictstr: fnm }
    got = dgbhdf5.getCubeLetNamesByGroupByItem( info, 'Group1', 'Well A', 0 )
    assert sorted( got ) == ['0: Well A', '0: Well C']
    got = dgbhdf5.getCubeLetNamesByGroupByItem( info, 'Group1', 'Nothing', 9 )
    assert got == []


def test_getcubeletnames_direct_match_branch( tmp_path ):
    fnm = os.path.join( str(tmp_path), 'direct.h5' )
    _write_multi_item_group( fnm, 'Group1', ['Well A', 'Well B'], nrpts=5 )
    info = { dbk.filedictstr: fnm }
    got = dgbhdf5.getCubeLetNamesByGroupByItem( info, 'Group1', 'Well A', 0 )
    assert got == [0, 1, 2, 3, 4]


# --- torch load paths: torch 2.6 defaults weights_only=True ------------------

torch = pytest.importorskip( 'torch', reason='torch not installed' )


class _TinyNet( torch.nn.Module ):
    def __init__( self ):
        super().__init__()
        self.fc = torch.nn.Linear( 4, 2 )

    def forward( self, x ):
        return self.fc( x )


def test_torch_supports_weights_only_matches_signature():
    import inspect
    expected = 'weights_only' in inspect.signature( torch.load ).parameters
    assert compat.torch_supports_weights_only( torch ) is expected


def test_torch_modelinfo_reads_both_formats( tmp_path ):
    import dgbpy.torch_modelinfo as tmi

    scripted = os.path.join( str(tmp_path), 'ts.pth' )
    torch.jit.script( _TinyNet() ).save( scripted )
    assert 'model_type' in tmi.model_info( scripted )

    pickled = os.path.join( str(tmp_path), 'pk.pth' )
    torch.save( _TinyNet(), pickled )
    assert 'model_type' in tmi.model_info( pickled )
