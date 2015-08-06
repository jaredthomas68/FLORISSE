from utilities import check_gradient
from Tapenade_components import floris_wcent_wdiam, floris_overlap, floris_power
from Analytic_components import floris_adjustCtCp, floris_windframe, floris_AEP
from Parameters import FLORISParameters
import numpy as np


def test_power():

    turbineXw = np.array([1521.00978779, 1487.94926246, 2150.60113933, 2117.540614, 2747.13196554, 2780.19249087])

    nTurbs = len(turbineXw)

    comp = floris_power(nTurbines=nTurbs)

    comp.parameters = FLORISParameters()

    comp.turbineXw = turbineXw

    comp.wakeOverlapTRel = np.array([0.94882764, 0.,         0.,         0.,         0.,         0.,         0.005853,\
                            0.,         0.,         0.,         0.,         0.,         0.00603356,\
                            0.,         0.,         0.,         0.,         0.,         0.,\
                            0.94882764, 0.,         0.,         0.,         0.,         0.,         0.005853,\
                            0.,         0.,         0.,         0.,         0.,         0.00603356,\
                            0.,         0.,         0.,         0.,         0.21889896, 0.,\
                            0.94882764, 0.,         0.,         0.,         0.30408137, 0.,         0.005853,\
                            0.,         0.,         0.,         0.35018179, 0.,         0.00603356,\
                            0.,         0.,         0.,         0.,         0.21889896, 0.,\
                            0.94882764, 0.,         0.,         0.,         0.30408137, 0.,         0.005853,\
                            0.,         0.,         0.,         0.35018179, 0.,         0.00603356,\
                            0.,         0.,         0.,         0.,         0.,         0.21889896,\
                            0.94882764, 0.,         0.,         0.14035337, 0.,         0.30408137,\
                            0.005853,   0.,         0.,         0.61612477, 0.,         0.35018179,\
                            0.00603356, 0.,         0.,         0.,         0.21889896, 0.,         0.,\
                            0.94882764, 0.14035337, 0.,         0.30408137, 0.,         0.,         0.005853,\
                            0.61612477, 0.,         0.35018179, 0.,         0.,         0.00603356])

    # turbineY = np.array([2024.7, 2335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    nTurbs = comp.turbineXw.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    for turbI in range(0, nTurbs):

        rotorDiameter[turbI] = 126.4
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generator_efficiency[turbI] = 0.944
        # yaw[turbI] = 25.0
        yaw[turbI] = 0.

    comp.rotorDiameter = rotorDiameter
    comp.axialInduction = axialInduction
    comp.Ct = Ct
    comp.Cp = Cp
    comp.generator_efficiency = generator_efficiency
    comp.yaw = yaw
        # Define flow properties
    comp.wind_speed = 8.0  # m/s
    comp.air_density = 1.1716  # kg/m^3
    comp.wind_direction = 30.  # deg
    comp.verbose = False
    comp.parameters.FLORISoriginal = True

    names, errors = check_gradient(comp, fd='central', step_size=1e-6, tol=1e-6, display=True,
        show_missing_warnings=True, show_scaling_warnings=False, min_grad=1e-6, max_grad=1e6)


def test_overlap():

    turbineXw = np.array([1521.00978779, 1487.94926246, 2150.60113933, 2117.540614, 2747.13196554, 2780.19249087])

    turbineYw =  np.array([305.06623126, 682.80372167, 360.15044013,  737.88793055, 792.97213942,  415.234649])

    # turbineY = np.array([2024.7, 2335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    nTurbs = len(turbineXw)

    comp = floris_overlap(nTurbines=nTurbs)

    comp.parameters = FLORISParameters()

    comp.turbineXw = turbineXw
    comp.turbineYw = turbineYw
    comp.wakeDiametersT = np.array([ 125.66056,     124.25106585,    0.,            0.,            0.,            0.,\
                                      126.4,         127.34553102,    0.,            0.,            0.,            0.,\
                                      127.20106,     130.69786829,    0.,            0.,            0.,            0.,\
                                      125.36730185,  125.66056,       0.,            0.,            0.,            0.,\
                                      125.36730185,  126.4,           0.,            0.,            0.,            0.,\
                                      125.36730185,  127.20106,       0.,            0.,            0.,            0.,\
                                       85.47656215,   83.327628,    125.66056001,  124.25106585,    0.,            0.,\
                                      144.40631265,  145.35184368,  126.4,         127.34553102,    0.,            0.,\
                                      208.2468757,   212.54474399,  127.20106,     130.69786829,    0.,            0.,\
                                       87.6254963,    85.47656215,  125.36730185,  125.66056001,    0.,            0.,\
                                      143.46078163,  144.40631265,  125.36730185,  126.4,           0.,            0.,\
                                      203.94900741,  208.2468757,   125.36730185,  127.20106,       0.,            0.,\
                                       46.70205845,   44.5531243,    87.6254963,   85.47656215,  125.66056,\
                                      125.36730185,  161.46709428,  162.41262531,  143.46078163,  144.40631265,\
                                      126.4,         125.36730185,  285.79588311,  290.0937514,   203.94900741,\
                                      208.2468757,   127.20105999,  125.36730185,   44.5531243,    42.40419015,\
                                       85.47656215,   83.327628,    124.25106585,  125.66056,     162.41262531,\
                                      163.35815633,  144.40631265,  145.35184368,  127.34553102,  126.4,\
                                      290.0937514,   294.39161969,  208.2468757,   212.54474399,  130.69786829,\
                                      127.20105998])

    comp.wakeCentersYT = np.array([ 300.56623126,  678.30372167,  355.65044013,  733.38793055,  788.47213942,\
                              410.734649,    300.56623126,  678.30372167,  355.65044013,  733.38793055,\
                              788.47213942,  410.734649,    300.56623126,  678.30372167,  355.65044013,\
                              733.38793055,  788.47213942,  410.734649,    300.56623126,  678.30372167,\
                              355.65044013,  733.38793056,  788.47213942,  410.734649,    300.56623126,\
                              678.30372167,  355.65044013,  733.38793055,  788.47213943,  410.734649,\
                              300.56623126,  678.30372167,  355.65044013,  733.38793055,  788.47213942,\
                              410.73464902])

    # turbineY = np.array([2024.7, 2335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    for turbI in range(0, nTurbs):

        rotorDiameter[turbI] = 126.4
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generator_efficiency[turbI] = 0.944
        # yaw[turbI] = 25.0
        yaw[turbI] = 0.

    comp.rotorDiameter = rotorDiameter
    comp.axialInduction = axialInduction
    comp.Ct = Ct
    comp.Cp = Cp
    comp.generator_efficiency = generator_efficiency
    comp.yaw = yaw
        # Define flow properties
    comp.wind_speed = 8.0  # m/s
    comp.air_density = 1.1716  # kg/m^3
    comp.wind_direction = 30.  # deg
    comp.verbose = False

    names, errors = check_gradient(comp, fd='central', step_size=1e-6, tol=1e-6, display=True,
        show_missing_warnings=True, show_scaling_warnings=False, min_grad=1e-6, max_grad=1e6)


def test_wcent_wdiam():

    turbineXw = np.array([1521.00978779, 1487.94926246, 2150.60113933, 2117.540614, 2747.13196554, 2780.19249087])

    turbineYw = np.array([305.06623126, 682.80372167, 360.15044013,  737.88793055, 792.97213942,  415.234649])

    nTurbs = turbineXw.size

    comp = floris_wcent_wdiam(nTurbines=nTurbs)

    comp.parameters = FLORISParameters()

    comp.turbineXw = turbineXw
    comp.turbineYw = turbineYw

    # turbineY = np.array([2024.7, 2335.3, 1387.2, 1697.8, 2060.3, 1749.7])


    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    for turbI in range(0, nTurbs):

        rotorDiameter[turbI] = 126.4
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generator_efficiency[turbI] = 0.944
        # yaw[turbI] = 25.0
        yaw[turbI] = 0.

    comp.rotorDiameter = rotorDiameter
    comp.axialInduction = axialInduction
    comp.Ct = Ct
    comp.Cp = Cp
    comp.generator_efficiency = generator_efficiency
    comp.yaw = yaw
        # Define flow properties
    comp.wind_speed = 8.0  # m/s
    comp.air_density = 1.1716  # kg/m^3
    comp.wind_direction = 30.  # deg
    comp.verbose = False

    names, errors = check_gradient(comp, fd='central', step_size=1e-6, tol=1e-4, display=True,
        show_missing_warnings=True, show_scaling_warnings=True, min_grad=1e-6, max_grad=1e6)


def test_windframe():

    turbineXw = np.array([1521.00978779, 1487.94926246, 2150.60113933, 2117.540614, 2747.13196554, 2780.19249087])

    turbineYw =  np.array([305.06623126, 682.80372167, 360.15044013,  737.88793055, 792.97213942,  415.234649])

    # turbineY = np.array([2024.7, 2335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    nTurbs = len(turbineXw)

    comp = floris_windframe(nTurbines=nTurbs, resolution=0)

    comp.parameters = FLORISParameters()

    comp.turbineXw = turbineXw
    comp.turbineYw = turbineYw

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    for turbI in range(0, nTurbs):

        rotorDiameter[turbI] = 126.4
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generator_efficiency[turbI] = 0.944
        # yaw[turbI] = 25.0
        yaw[turbI] = 0

    comp.rotorDiameter = rotorDiameter
    comp.axialInduction = axialInduction
    comp.Ct = Ct
    comp.Cp = Cp
    comp.generator_efficiency = generator_efficiency
    comp.yaw = yaw
        # Define flow properties
    comp.wind_speed = 8.0  # m/s
    comp.air_density = 1.1716  # kg/m^3
    comp.wind_direction = 30  # deg
    comp.verbose = False

    names, errors = check_gradient(comp, fd='central', step_size=1e-6, tol=1e-4, display=True,
        show_missing_warnings=True, show_scaling_warnings=True, min_grad=1e-6, max_grad=1e6)


def test_adjustCtCp():



    # turbineY = np.array([2024.7, 2335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    nTurbs = 6
    axialInduction = np.zeros(nTurbs)+1.0/3.0
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    comp = floris_adjustCtCp(nTurbines=nTurbs)

    comp.parameters = FLORISParameters()

    for turbI in range(0, nTurbs):

        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        # yaw[turbI] = 25.0
        yaw[turbI] = 20.
    print 'Ct, Cp = ', Ct, Cp
    comp.Ct_in = Ct
    comp.Cp_in = Cp
    comp.yaw = yaw
        # Define flow properties
    # comp.parameters.CTcorrected = False
    comp.parameters.CTcorrected = True
    # comp.parameters.CPcorrected = False
    comp.parameters.CPcorrected = True

    names, errors = check_gradient(comp, fd='central', step_size=1e-8, tol=1e-6, display=True,
        show_missing_warnings=True, show_scaling_warnings=True, min_grad=1e-6, max_grad=1e6)


def test_AEP():

    # turbineY = np.array([2024.7, 2335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    nTurbs = 6
    axialInduction = np.zeros(nTurbs)+1.0/3.0
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    dirPercent = np.array([0.2, 0.25, 0.35, 0.2])
    comp = floris_AEP(nDirections=len(dirPercent))

    comp.parameters = FLORISParameters()

    comp.windrose_frequencies = dirPercent
    comp.power_directions = np.array([200., 250., 500., 200.])

    names, errors = check_gradient(comp, fd='central', step_size=1e-8, tol=1e-6, display=True,
        show_missing_warnings=True, show_scaling_warnings=True, min_grad=1e-6, max_grad=1e6)


if __name__ == '__main__':

    test_adjustCtCp()
    # test_windframe()
    # test_wcent_wdiam()
    # test_overlap()
    # test_power()
    # test_AEP()