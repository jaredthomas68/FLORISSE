# from openmdao.main.api import Assembly
from test_case_assembly_coupled import floris_assembly_opt_AEP
from FLORIS_visualization import floris_assembly
# from FLORIS_components import floris_assembly as vis_assembly
import time
import numpy as np
import math as m

from rotor_components import *
import cPickle as pickle


if __name__ == "__main__":

    rotor_diameter = 126.4
    nRows = 3.
    spacing = 5.

    optimize_position = False
    optimize_yaw = False
    use_rotor_components = True
    print 'optimize_position = ', optimize_position
    print 'optimize_yaw = ', optimize_yaw

    # for loading amalia windrose
    windrose_directions = np.arange(0, 360, 5)
    windrose_speeds = np.array([6.53163342, 6.11908394, 6.13415514, 6.0614625,  6.21344602,
                                5.87000793, 5.62161519, 5.96779107, 6.33589422, 6.4668016,
                                7.9854581,  7.6894432,  7.5089221,  7.48638098, 7.65764618,
                                6.82414044, 6.36728201, 5.95982999, 6.05942132, 6.1176321,
                                5.50987893, 4.18461796, 4.82863115, 0.,         0.,         0.,
                                5.94115843, 5.94914252, 5.59386528, 6.42332524, 7.67904937,
                                7.89618066, 8.84560463, 8.51601497, 8.40826823, 7.89479475,
                                7.86194762, 7.9242645,  8.56269962, 8.94563889, 9.82636368,
                               10.11153102, 9.71402212, 9.95233636,  10.35446959, 9.67156182,
                                9.62462527, 8.83545158, 8.18011771, 7.9372492,  7.68726143,
                                7.88134508, 7.31394723, 7.01839896, 6.82858346, 7.06213432,
                                7.01949894, 7.00575122, 7.78735165, 7.52836352, 7.21392201,
                                7.4356621,  7.54099962, 7.61335262, 7.90293531, 7.16021596,
                                7.19617087, 7.5593657,  7.03278586, 6.76105501, 6.48004694,
                                6.94716392])
    dirPercent = np.array([1.17812570e-02, 1.09958570e-02, 9.60626600e-03, 1.21236860e-02,
                           1.04722450e-02, 1.00695140e-02, 9.68687400e-03, 1.00090550e-02,
                           1.03715390e-02, 1.12172280e-02, 1.52249700e-02, 1.56279300e-02,
                           1.57488780e-02, 1.70577560e-02, 1.93535770e-02, 1.41980570e-02,
                           1.20632100e-02, 1.20229000e-02, 1.32111160e-02, 1.74605400e-02,
                           1.72994400e-02, 1.43993790e-02, 7.87436000e-03, 0.00000000e+00,
                           2.01390000e-05, 0.00000000e+00, 3.42360000e-04, 3.56458900e-03,
                           7.18957000e-03, 8.80068000e-03, 1.13583200e-02, 1.41576700e-02,
                           1.66951900e-02, 1.63125500e-02, 1.31709000e-02, 1.09153300e-02,
                           9.48553000e-03, 1.01097900e-02, 1.18819700e-02, 1.26069900e-02,
                           1.58895900e-02, 1.77021600e-02, 2.04208100e-02, 2.27972500e-02,
                           2.95438600e-02, 3.02891700e-02, 2.69861000e-02, 2.21527500e-02,
                           2.12465500e-02, 1.82861400e-02, 1.66147400e-02, 1.90111800e-02,
                           1.90514500e-02, 1.63932050e-02, 1.76215200e-02, 1.65341460e-02,
                           1.44597600e-02, 1.40370300e-02, 1.65745000e-02, 1.56278200e-02,
                           1.53459200e-02, 1.75210100e-02, 1.59702700e-02, 1.51041500e-02,
                           1.45201100e-02, 1.34527800e-02, 1.47819600e-02, 1.33923300e-02,
                           1.10562900e-02, 1.04521380e-02, 1.16201970e-02, 1.10562700e-02])



    print dirPercent.shape, windrose_speeds.shape, windrose_directions.shape

    # windrose for DOE test case
    # dirPercent = np.array([ 0.0103304391513755,0.0101152216690551,0.0099087885737683,0.00971061280229294,
    #            0.00952020862969896,0.00933712769451244,0.00916095547386126,0.00899130815027124,0.00882782982026631,
    #            0.00867019000204726,0.00882782982026631,0.00899130815027124,0.00916095547386126,0.00933712769451244,
    #            0.00952020862969896,0.00971061280229294,0.0099087885737683,0.0101152216690551,0.0103304391513755,
    #            0.0105550139155358,0.0107895697803255,0.0110347872753329,0.0112914102352243,0.011560253336063,
    #            0.0118422107345036,0.0121382660028662,0.0124495035926833,0.0127771221082802,0.0131224497328283,0.0134869622254069,
    #            0.0138723040032756,0.0142803129445484,0.0147130497004438,0.0151728325035827,0.0156622787133757,0.0161843546704882,
    #            0.0167424358660223,0.0173403800040945,0.0179826163005425,0.0186742553890249,0.0194212256045859,0.0202304433381103,
    #            0.0211100278310716,0.0220695745506658,0.023120506672126,0.0242765320057323,0.023120506672126,0.0220695745506658,
    #            0.0211100278310716,0.0202304433381103,0.0194212256045859,0.0186742553890249,0.0179826163005425,0.0173403800040945,
    #            0.0167424358660223,0.0161843546704882,0.0156622787133757,0.0151728325035827,0.0147130497004438,0.0142803129445484,
    #            0.0138723040032756,0.0134869622254069,0.0131224497328283,0.0127771221082802,0.0124495035926833,0.0121382660028662,
    #            0.0118422107345036,0.011560253336063,0.0112914102352243,0.0110347872753329,0.0107895697803255,0.0105550139155358])

    # simple small windrose for testing
    # dirPercent = np.array([.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.09, 0.05, 0.04])
    # dirPercent = np.array([0.1, 0.8, 0.1, 0.1])
    # dirPercent = np.array([0.4, 0.3, 0.3])

    # single direction
    # dirPercent = np.array([1.0])

    nDirections = len(dirPercent)

    # Set up position arrays
    points = np.arange(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter+1, step=spacing*rotor_diameter)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    # set up speed array
    # if use_rotor_components:
    #     wind_speed = 8.1    # m/s
    # else:
    #     wind_speed = 8.0    # m/s
    # windrose_speeds = np.ones_like(dirPercent)*wind_speed

    # to make pics for Dr. Ning
    # turbineX = np.arange(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter+1, step=spacing*rotor_diameter)
    # turbineY = np.array([1000, 1000])
    # dirPercent = np.array([1])
    # nDirections = dirPercent.size
    # windDirs = np.array([270.])



    # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
    # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    # print turbineX.size, turbineX
    # print turbineY.size, turbineY

    nTurbs = turbineX.size
    NREL5MWCPCT = pickle.load(open('NREL5MWCPCT.p'))
    datasize = NREL5MWCPCT.CP.size
    # position = np.zeros([nTurbs, 2])
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # resolution = 200
    # xlim = [0, 3000]
    # ylim = [0, 3000]
    # x = np.linspace(xlim[0], xlim[1], resolution)
    # y = np.linspace(ylim[0], ylim[1], resolution)
    # x, y = np.meshgrid(x, y)
    # myFloris.ws_position = np.array([x.flatten(), y.flatten()]).transpose()

    for turbI in range(0, nTurbs):

        rotorDiameter[turbI] = rotor_diameter
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2.)
        generator_efficiency[turbI] = 0.944
        # yaw[turbI] = 25.
        yaw[turbI] = 0.

    # myFloris = floris_assembly_opt(nTurbines=nTurbs, resolution=0)
    myFloris = floris_assembly_opt_AEP(nTurbines=nTurbs, nDirections=nDirections,
                                       optimize_position=optimize_position, resolution=0.0, optimize_yaw=optimize_yaw,
                                       use_rotor_components=use_rotor_components, datasize=datasize)

    # myFloris.position = position
    myFloris.turbineX = turbineX
    myFloris.turbineY = turbineY
    # myFloris.floris_windframe.turbineX = turbineX
    # myFloris.floris_windframe.turbineY = turbineY
    myFloris.rotorDiameter = rotorDiameter
    myFloris.axialInduction = axialInduction
    myFloris.generator_efficiency = generator_efficiency
    # myFloris.wind_speed = wind_speed  # m/s

    if use_rotor_components:
        # for i in range(0, nDirections):
        #     exec('myFloris.initVelocitiesTurbines_%d = np.ones_like(turbineX)*windrose_speeds[%d]' % (i, i))
        # myFloris.initVelocitiesTurbines = np.ones_like(turbineX)*windrose_speeds
        # myFloris.windSpeedToCPCT = NREL5MWCPCT
        myFloris.curve_CP = NREL5MWCPCT.CP
        myFloris.curve_CT = NREL5MWCPCT.CT
        myFloris.curve_wind_speed = NREL5MWCPCT.wind_speed
        myFloris.parameters.ke = 0.05
        myFloris.parameters.kd = 0.17
        myFloris.parameters.aU = 12.0
        myFloris.parameters.bU = 1.3
        myFloris.parameters.initialWakeAngle = 3.0
        myFloris.parameters.useaUbU = True
        myFloris.parameters.useWakeAngle = True
        myFloris.parameters.adjustInitialWakeDiamToYaw = False

    else:
        myFloris.Ct = Ct
        myFloris.Cp = Cp

    if optimize_yaw:
        # myFloris.yaw = np.tile(yaw, nDirections)
        yaw = np.tile(yaw, (nDirections, 1))
        # print yaw
        for i in range(0, nDirections):
            exec('myFloris.yaw_%d = yaw[%d]' % (i, i))
            # exec('print myFloris.yaw_%d' % direction)
            # print myFloris.yaw_0, myFloris.yaw_1, myFloris.yaw_2, myFloris.yaw_3
    else:
        myFloris.yaw = yaw
    # myFloris.yaw = yaw

    # Define flow properties
    myFloris.windrose_frequencies = dirPercent  # cw from north using direction from (standard windrose style)
    myFloris.windrose_speeds = windrose_speeds
    myFloris.air_density = 1.1716  # kg/m^3
    myFloris.verbose = False

    # final input directions should be in the order corresponding to the frequencies provided and use
    # 0 deg = E progressing ccw
    myFloris.windrose_directions = 270. - windrose_directions
    # myFloris.windrose_directions = 270. - np.arange(0.0, 360.0, 360.0/nDirections)
    # myFloris.windrose_directions = np.array([30.0])
    # myFloris.windrose_directions = 270 - windDirs
    # print myFloris.windrose_directions
    for i in range(0, nDirections):
        if myFloris.windrose_directions[i] < 0.:
            myFloris.windrose_directions[i] += 360.

    myFloris.parameters.CPcorrected = True
    myFloris.parameters.CTcorrected = True
    myFloris.parameters.axialIndProvided = False
    myFloris.parameters.FLORISoriginal = False

    # define sampling points for optimization run (GenericFlowModel)
    # resolution = 0.
    # xlim = [0, 3000]
    # ylim = [0, 3000]
    # x = np.linspace(xlim[0], xlim[1], resolution)
    # y = np.linspace(ylim[0], ylim[1], resolution)
    # x, y = np.meshgrid(x, y)
    # myFloris.ws_position = np.array([x.flatten(), y.flatten()]).transpose()
    #print myFloris.ws_position
    # run model
    print 'start FLORIS Opt. run'
    tic = time.time()
    myFloris.run()
    toc = time.time()
    print('FLORIS Opt. calculation took %.03f sec.' % (toc-tic))

    # Display returns
    filename = 'test_case_3x3_ave_speeds_powerout.txt'
    np.savetxt(filename, myFloris.power_directions, delimiter='\t')


    # TODO get this for loop to work so you can print the desired values, or figure out another way to do this
    # for i in range(0, nDirections):
    #     print 'turbine powers for %d deg. (kW): %s' % (myFloris.windrose_directions[i],
    #                                                    myFloris.floris_power_(i).wt_power_directions)
    #     print 'effective wind speeds for %d deg. (m/s): %s' % (myFloris.windrose_directions[i],
    #                                                            myFloris.floris_power_[i].velocitiesTurbines)
    #
    #     print 'wind farm power for %d deg. (kW): %s' % (myFloris.windrose_directions[i],
    #                                                     myFloris.floris_power_[i].power_directions)
    #     print 'wind turbine powers for %d deg. (kW): %s' % (myFloris.windrose_directions[i],
    #                                                         myFloris.floris_power_[i].wt_power)

    print 'power in each wind direction (kW): %s' % myFloris.power_directions

    print 'turbine X positions in wind frame (m): %s' % myFloris.turbineX
    print 'turbine Y positions in wind frame (m): %s' % myFloris.turbineY
    if optimize_yaw:
        for direction in range(0, nDirections):
            print 'yaw for wind direction %d deg. (deg.):' % myFloris.windrose_directions[direction]
            exec('print myFloris.yaw_%d' % direction)
            # exec("print 'power of each turbine for wind direction %d deg. (kW): %s' % myFloris.floris_power_%d.wt_power" % (myFloris.windrose_directions[direction], direction))
    else:
        print 'yaw (deg) = ', myFloris.yaw
        print 'power of each turbine (kW): %s' % myFloris.floris_power_0.wt_power
    print 'AEP (kWh): %s' % myFloris.AEP

    resolution = 200
    myFloris2 = floris_assembly(nTurbines=nTurbs, resolution=resolution)
    myFloris2.turbineX = myFloris.turbineX
    myFloris2.turbineY = myFloris.turbineY

    myFloris2.rotorDiameter = rotorDiameter
    myFloris2.rotorArea = np.pi*rotorDiameter*rotorDiameter/4.0
    myFloris2.axialInduction = axialInduction
    myFloris2.Ct = Ct
    myFloris2.Cp = Cp
    myFloris2.generator_efficiency = generator_efficiency
    pri_dir_ind = np.argmax(myFloris.windrose_frequencies)
    print pri_dir_ind
    if optimize_yaw:
        # myFloris2.yaw = myFloris.yaw[pri_dir_ind*nTurbs:(pri_dir_ind+1)*nTurbs]
        exec('myFloris2.yaw = myFloris.yaw_%d' % pri_dir_ind)
        # print myFloris2.yaw
    else:
        myFloris2.yaw = myFloris.yaw

    # # Define flow properties
    myFloris2.wind_speed = myFloris.windrose_speeds[np.argmax(myFloris.windrose_frequencies)]  # m/s
    myFloris2.air_density = myFloris.air_density  # kg/m^3
    myFloris2.wind_direction = myFloris.windrose_directions[np.argmax(myFloris.windrose_frequencies)]  # deg

    # print np.argmax(myFloris.windrose_frequencies)
    # myFloris2.verbose = True

    myFloris2.parameters.CPcorrected = False
    myFloris2.parameters.CTcorrected = False

    # define sampling points for final visualization (GenericFlowModel)
    xlim = [0, np.max(myFloris.turbineX)+spacing*rotor_diameter]
    ylim = [0, np.max(myFloris.turbineY)+spacing*rotor_diameter]
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    x, y = np.meshgrid(x, y)
    myFloris2.ws_position = np.array([x.flatten(), y.flatten()]).transpose()
    #print myFloris.ws_position
    # run model
    print 'start FLORIS Vis. run'
    tic = time.time()
    myFloris2.run()
    toc = time.time()
    print('FLORIS Vis. calculation took %.03f sec.' % (toc-tic))



    np.set_printoptions(linewidth=150, precision=4)

    # Display returns
    print 'Wind direction (deg., direction to, ccw from east): %s' % myFloris2.wind_direction
    # print 'turbine powers (kW): %s' % myFloris2.wt_power
    # print 'turbine powers (kW): %s' % myFloris2.wt_power
    print 'turbine powers (kW): %s' % myFloris2.wt_power
    # print 'turbine X positions in wind frame (m): %s' % myFloris2.turbineXw
    # print 'turbine Y positions in wind frame (m): %s' % myFloris2.turbineYw
    # print 'Wake center Y positions (m): %s' % myFloris2.wakeCentersYT
    # print 'Wake diameters (m): %s' % myFloris2.wakeDiametersT
    # print 'Relative wake overlap (m*m): %s' % myFloris2.wakeOverlapTRel
    print 'wind farm power (kW): %s' % myFloris2.power
    # print myFloris2.ws_array
    velocities = myFloris2.ws_array.reshape(resolution, resolution)

    import matplotlib.pyplot as plt

    fig, (ax1) = plt.subplots(nrows=1)
    im = ax1.pcolormesh(x, y, velocities, cmap='coolwarm')
    low = m.ceil(np.min(velocities))
    high = m.ceil(np.max(velocities))
    v = np.linspace(low, high, 5, endpoint=True)
    cbar = plt.colorbar(im, ticks=v, orientation='vertical', fraction=0.09)
    cbar.set_label('Wind Speed (m/s)', rotation=270, labelpad=30)
    ax1.set_aspect('equal')
    ax1.autoscale(tight=True)

    for turbI in range(0, nTurbs):
        dx = 0.5*rotorDiameter[turbI]*np.sin((myFloris2.yaw[turbI]+myFloris.windrose_directions[
            np.argmax(dirPercent)])*np.pi/180.)
        dy = 0.5*rotorDiameter[turbI]*np.cos((myFloris2.yaw[turbI]+myFloris.windrose_directions[
            np.argmax(dirPercent)])*np.pi/180.)
        plt.plot([myFloris2.turbineX[turbI]-dx, myFloris2.turbineX[turbI]+dx], [myFloris2.turbineY[turbI]+dy, myFloris2.turbineY[turbI]-dy],
                 solid_capstyle='butt', lw=4, c='k')

    plt.xlabel('Position (m)', labelpad=0)
    plt.ylabel('Position (m)', labelpad=5)

    plt.show()
