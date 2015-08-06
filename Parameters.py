from openmdao.main.api import VariableTree
from openmdao.lib.datatypes.api import Array, Bool, Float
import numpy as np

class FLORISParameters(VariableTree):
    """Container of FLORIS wake parameters"""

    # original tuning parameters
    pP = Float(1.88, iotype='in')
    ke = Float(0.065, iotype='in')
    keCorrArray = Float(0.0, iotype='in')
    keCorrCT = Float(0.0, iotype='in')
    Region2CT = Float(4.0*(1.0/3.0)*(1.0-(1.0/3.0)), iotype='in')
    kd = Float(0.15, iotype='in')
    me = Array(np.array([-0.5, 0.22, 1.0]), iotype='in')
    MU = Array(np.array([0.5, 1.0, 5.5]), iotype='in')
    initialWakeDisplacement = Float(4.5, iotype='in')
    initialWakeAngle = Float(0.0, iotype='in')


    # old parameters
    # pP = Float(1.88, iotype='in')
    # ke = Float(0.065, iotype='in')
    # keCorrArray = Float(0.0, iotype='in')
    # keCorrCT = Float(0.0, iotype='in')
    # Region2CT = Float(4.0*(1.0/3.0)*(1.0-(1.0/3.0)), iotype='in')
    # kd = Float(0.15, iotype='in')
    # me = Array(np.array([-0.5, 0.22, 1.0]), iotype='in')
    # MU = Array(np.array([0.5, 1.0, 5.5]), iotype='in')
    # initialWakeDisplacement = Float(4.5, iotype='in')
    # initialWakeAngle = Float(0.0, iotype='in')

    # initial re-tuning (may be changed again)
    # pP = Float(1.88, iotype='in')
    # ke = Float(0.065, iotype='in')
    # kd = Float(0.11, iotype='in')
    # me = Array(np.array([-0.5, 0.3, 1.0]), iotype='in')
    # MU = Array(np.array([0.5, 1.0, 10.]), iotype='in')
    # initialWakeDisplacement = Float(4.5, iotype='in')
    # initialWakeAngle = Float(1.5, iotype='in')
    #
    # CTcorrected = Bool(True, iotype='in', desc = 'CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')
    # CPcorrected = Bool(True, iotype='in', desc = 'CP factor already corrected by CCBlade calculation (assumed with approximately factor cos(yaw)^3)')
    # CP_use_pP = Bool(True, iotype='in', desc = 'allow FLORIS to correct with factor cos(yaw)^pP')
    # axialIndProvided = Bool(True, iotype='in', desc = 'CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')

    #bd = Float(-0.01, iotype='in')
    #aU = Float(5.0, iotype='in', units='deg')
    #bU = Float(1.66, iotype='in')

    # parameters for switching between old and new model

    # ke = Float(0.065, iotype='in')

    # ke correction factors, based on C_T, turbulence intensity and absolute heating rate
    # keCorrCT = Float(0.0, iotype='in')
    baselineCT = Float(4./3.*(1.-1./3.), iotype='in')

    keCorrTI = Float(0.0, iotype='in')
    baselineTI = Float(0.045, iotype='in')

    keCorrHR = Float(0.0, iotype='in') # neutral, with heating rate 0, is baseline

    keCorrHRTI = Float(0.0, iotype='in')

    # keCorrArray = Float(0.0, iotype='in')

    keSaturation = Float(0.0, iotype='in')

    # kd = Float(0.15, iotype='in')
    kdCorrYawDirection = Float(0.0, iotype='in')

    # me = Array(np.array([-0.5, 0.22, 1.0]), iotype='in')
    # MU = Array(np.array([0.5, 1.0, 5.5]), iotype='in')

    CTcorrected = Bool(True, iotype='in', desc = 'CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')
    CPcorrected = Bool(True, iotype='in', desc = 'CP factor already corrected by CCBlade calculation (assumed with approximately factor cos(yaw)^3)')
    # pP = Float(1.88, iotype='in')
    axialIndProvided = Bool(True, iotype='in', desc = 'CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')

    # initialWakeDisplacement = Float(4.5, iotype='in')

    useWakeAngle = Bool(True, iotype = 'in')
    # initialWakeAngle = Float(0.0, iotype='in')
    bd = Float(-0.01, iotype='in')

    useaUbU = Bool(False, iotype = 'in')
    aU = Float(5.0, iotype='in', units='deg')
    bU = Float(1.66, iotype='in')

    adjustInitialWakeDiamToYaw = Bool(True, iotype = 'in')

    FLORISoriginal = Bool(False, 'override all parameters and use FLORIS as original in first Wind Energy paper')
