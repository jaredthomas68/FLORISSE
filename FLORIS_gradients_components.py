from openmdao.main.api import Assembly, Component
from openmdao.main.api import VariableTree
from openmdao.lib.datatypes.api import Array, Bool, Float, VarTree

from openmdao.lib.drivers.api import SLSQPdriver
import numpy as np


class FLORISParameters(VariableTree):
    """Container of FLORIS wake parameters"""

    #pP = Float(1.88, iotype='in')
    ke = Float(0.065, iotype='in')
    keCorrArray = Float(0.0, iotype='in')
    keCorrCT = Float(0.0, iotype='in')
    Region2CT = Float(4.0*(1.0/3.0)*(1.0-(1.0/3.0)), iotype='in')
    kd = Float(0.15, iotype='in')
    me = Array(np.array([-0.5, 0.22, 1.0]), iotype='in')
    MU = Array(np.array([0.5, 1.0, 5.5]), iotype='in')
    initialWakeDisplacement = Float(4.5, iotype='in')
    initialWakeAngle = Float(0.0, iotype='in')

    CTcorrected = Bool(True, iotype='in', desc = 'CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')
    CPcorrected = Bool(True, iotype='in', desc = 'CP factor already corrected by CCBlade calculation (assumed with approximately factor cos(yaw)^3)')
    CP_use_pP = Bool(True, iotype='in', desc = 'allow FLORIS to correct with factor cos(yaw)^pP')
    axialIndProvided = Bool(True, iotype='in', desc = 'CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')

    #bd = Float(-0.01, iotype='in')
    #aU = Float(5.0, iotype='in', units='deg')
    #bU = Float(1.66, iotype='in')


class floris_component(Component):
    """ Calculates the power and effective wind speed for each turbine according to the FLORIS model """

    # original variables in Pieter's OpenMDAO stand-alone version of FLORIS
    parameters = VarTree(FLORISParameters(), iotype='in')
    velocitiesTurbines = Array(iotype='out', units='m/s')
    verbose = Bool(False, iotype='in', desc='verbosity of FLORIS, False is no output')

    # input variables added so I don't have to use WISDEM while developing gradients
    # position = Array(iotype='in', desc='position of turbines in original ref. frame')
    turbineX = Array(iotype='in', desc='x positions of turbines in original ref. frame')
    turbineY = Array(iotype='in', desc='y positions of turbines in original ref. frame')
    rotorDiameter = Array(dtype='float', iotype='in', units='m', desc='rotor diameters of all turbine')
    rotorArea = Array(iotype='in', dtype='float', units='m*m', desc='rotor area of all turbines')
    axialInduction = Array(iotype='in', dtype='float', desc='axial induction of all turbines')
    Ct = Array(iotype='in', dtype='float', desc='Thrust coefficient for all turbines')
    Cp = Array(iotype='in', dtype='float', desc='power coefficient for all turbines')
    generator_efficiency = Array(iotype='in', desc='generator efficiency of all turbines')
    yaw = Array(iotype='in', desc='yaw of each turbine')
    ws_position = Array(iotype='in', units='m', desc='positions where wind speed measurements are desired')

    # Flow property variables
    wind_speed = Float(iotype='in', units='m/s', desc='free stream wind velocity')
    air_density = Float(iotype='in', units='kg/(m*m*m)', desc='air density in free stream')
    wind_direction = Float(iotype='in', units='deg', desc='overall wind direction for wind farm')

    # output variables added so I don't have to use WISDEM while developing gradients
    wt_power = Array(iotype='out', units='kW')
    power = Float(iotype='out', units='kW')
    ws_array = Array(iotype='out', units='m/s', desc='wind speed at measurement locations')

    # variables added to test individual components
    turbineXw = Array(iotype='out', dtype='float', units='m', desc='X positions of turbines in the wind direction reference frame')
    turbineYw = Array(iotype='out', dtype='float', units='m', desc='Y positions of turbines in the wind direction reference frame')
    wakeCentersYT = Array(dtype='float', iotype='out', units='m', desc='centers of the wakes at each turbine')
    wakeDiametersT = Array(dtype='float', iotype='out', units='m', desc='diameters of each of the wake zones for each of the wakes at each turbine')
    wakeOverlapTRel = Array(dtype='float', iotype='out', units='m', desc='ratios of wake overlap area per zone to rotor area')

    # p_near0 = Float(iotyp='out', desc='upwind location of diameter spline in rotor diameters')

    def execute(self):

        # create assembly
        floris = floris_assembly()
        # print dir(floris)

        # original input variables from Pieter's OpenMDAO stand-alone version of FLORIS
        floris.parameters = self.parameters
        floris.verbose = self.verbose

        # input variables added so I don't have to use WISDEM while developing gradients
        # floris.position = self.position
        floris.rotorDiameter = self.rotorDiameter
        floris.rotorArea = self.rotorArea
        floris.axialInduction = self.axialInduction
        floris.Ct = self.Ct
        floris.Cp = self.Cp
        floris.generator_efficiency = self.generator_efficiency
        #print 'generator efficiency is: ', floris.generator_efficiency
        floris.yaw = self.yaw

        # flow property variables
        floris.wind_speed = self.wind_speed
        floris.air_density = self.air_density
        floris.wind_direction = self.wind_direction


        # connections from the floris assembly to components (original)
        floris.parameters = self.parameters

        floris.verbose = self.verbose

        # floris.position = self.position
        floris.turbineX = self.turbineX
        floris.turbineY = self.turbineY
        # added connections from the floris assembly to windframe
        floris.Ct = self.Ct
        floris.ws_position = self.ws_position

        #print 'ws_position in floris_component is: ', self.ws_position

        # added connections from the floris assembly to wcent_wdiam
        floris.rotorDiameter = self.rotorDiameter
        floris.rotorArea = self.rotorArea
        floris.axialInduction = self.axialInduction
        floris.yaw = self.yaw

        # calculate
        floris.run()
        print 'calls to windframe', floris.floris_windframe.exec_count
        print 'calls to windframe derivative', floris.floris_windframe.derivative_exec_count
        # original output variables
        # self.velocitiesTurbines = floris.velocitiesTurbines

        # output variables added so I don't have to use WISDEM while developing
        self.wt_power = floris.wt_power
        self.power = floris.power
        self.ws_array = floris.ws_array

        # output variables for testing individual components of FLORIS
        # self.turbineXw = np.zeros(self.position.size)
        self.turbineXw = floris.floris_windframe.turbineXw
        self.turbineYw = floris.floris_windframe.turbineYw
        self.wakeCentersYT = floris.wakeCentersYT
        self.wakeDiametersT = floris.wakeDiametersT
        self.wakeOverlapTRel = floris.wakeOverlapTRel
        # self.p_near0 = floris_wcent_wdiam.p_near0
        self.velocitiesTurbines = floris.velocitiesTurbines


class floris_assembly(Assembly):
    """ Defines the connections between each Component used in the FLORIS model """
    #
    # original input variables in Pieter's OpenMDAO stand-alone version of FLORIS
    parameters = VarTree(FLORISParameters(), iotype='in')
    verbose = Bool(False, iotype='in', desc='verbosity of FLORIS, False is no output')

    # input variables added so I don't have to use WISDEM while developing gradients
    # position = Array(iotype='in', desc='position of turbines in original ref. frame')
    turbineX = Array(iotype='in', desc='x positions of turbines in original ref. frame')
    turbineY = Array(iotype='in', desc='y positions of turbines in original ref. frame')
    ws_position = Array(iotype='in', units='m', desc='position where you want measurements in ref. frame')
    rotorDiameter = Array(dtype='float', iotype='in', units='m', desc='rotor diameters of all turbine')
    rotorArea = Array(iotype='in', dtype='float', units='m*m', desc='rotor area of all turbines')
    axialInduction = Array(iotype='in', dtype='float', desc='axial induction of all turbines')
    Ct = Array(iotype='in', desc='Thrust coefficient for all turbines')
    Cp = Array(iotype='in', dtype='float', desc='power coefficient for all turbines')
    generator_efficiency = Array(iotype='in', dtype='float', desc='generator efficiency of all turbines')
    yaw = Array(iotype='in', desc='yaw of each turbine')

    # Flow property variables
    wind_speed = Float(iotype='in', units='m/s', desc='free stream wind velocity')
    air_density = Float(iotype='in', units='kg/(m*m*m)', desc='air density in free stream')
    wind_direction = Float(iotype='in', units='deg', desc='overall wind direction for wind farm')

    # original output variables in Pieter's OpenMDAO stand-alone version of FLORIS
    velocitiesTurbines = Array(iotype='out', units='m/s')

    # output variables added so I don't have to use WISDEM while developing gradients
    wt_power = Array(iotype='out', units='kW')
    ws_array = Array(iotype='out', units='m/s', desc='wind speed at measurement locations')

    # variables added to test individual components
    turbineXw = Array(iotype='out', units='m', desc='X positions of turbines in the wind direction reference frame')
    turbineYw = Array(iotype='out', units='m', desc='Y positions of turbines in the wind direction reference frame')
    wakeCentersYT = Array(dtype='float', iotype='out', units='m', desc='centers of the wakes at each turbine')
    wakeDiametersT = Array(dtype='float', iotype='out', units='m', desc='diameters of each of the wake zones for each of the wakes at each turbine')
    wakeOverlapTRel = Array(dtype='float', iotype='out', units='m', desc='ratio of overlap area of each zone to rotor area')

    # testing
    # p_near0 = Float(iotyp='out', desc='upwind location of diameter spline in rotor diameters')

    # final output
    power = Float(iotype='out', units='kW', desc='total windfarm power')


    def configure(self):

        # add components to floris assembly
        self.add('floris_windframe', floris_windframe())
        self.add('floris_wcent_wdiam', floris_wcent_wdiam())
        self.add('floris_overlap', floris_overlap())
        self.add('floris_power', floris_power())

        # add driver to floris assembly
        self.driver.workflow.add(['floris_windframe', 'floris_wcent_wdiam', 'floris_overlap', 'floris_power'])

        # added for gradient testing
        self.add('driver', SLSQPdriver())
        self.driver.iprint = 0
        self.driver.add_objective('-floris_power.power')
        self.driver.add_parameter('floris_windframe.turbineX', low=0., high=1500.)
        self.driver.add_parameter('floris_windframe.turbineY', low=0., high=1500.)
        # self.driver.add_parameter('wakeDiametersT', low=0., high=1500.)
        # self.driver.add_parameter('wakeCentersYT', low=0., high=1500.)

        # connect inputs to components
        self.connect('parameters', ['floris_wcent_wdiam.parameters', 'floris_power.parameters'])
        self.connect('verbose', ['floris_windframe.verbose', 'floris_wcent_wdiam.verbose', 'floris_power.verbose'])
        # self.connect('position', 'floris_windframe.position')
        self.connect('turbineX', 'floris_windframe.turbineX')
        self.connect('turbineY', 'floris_windframe.turbineY')
        self.connect('ws_position', 'floris_windframe.ws_position')
        self.connect('rotorDiameter', ['floris_wcent_wdiam.rotorDiameter', 'floris_overlap.rotorDiameter', 'floris_power.rotorDiameter'])
        self.connect('rotorArea', ['floris_overlap.rotorArea', 'floris_power.rotorArea'])
        self.connect('axialInduction', 'floris_power.axialInduction')
        self.connect('Ct', ['floris_wcent_wdiam.Ct', 'floris_power.Ct'])
        self.connect('Cp', 'floris_power.Cp')
        self.connect('generator_efficiency', 'floris_power.generator_efficiency')
        self.connect('yaw', 'floris_wcent_wdiam.yaw')
        self.connect('wind_speed', 'floris_power.wind_speed')
        self.connect('air_density', 'floris_power.air_density')
        self.connect('wind_direction', 'floris_windframe.wind_direction')


        # for satisfying the verbosity in windframe
        self.connect('Cp', 'floris_windframe.Cp')
        self.connect('Ct', 'floris_windframe.Ct')
        self.connect('yaw', 'floris_windframe.yaw')
        self.connect('wind_speed', 'floris_windframe.wind_speed')
        self.connect('axialInduction', 'floris_windframe.axialInduction')

        # ############### Connections between components ##################
        # connections from floris_windframe to floris_wcent_wdiam
        self.connect("floris_windframe.turbineXw", "floris_wcent_wdiam.turbineXw")
        self.connect("floris_windframe.turbineYw", "floris_wcent_wdiam.turbineYw")
        self.connect("floris_windframe.wsw_position", "floris_wcent_wdiam.wsw_position")

        # connections from floris_wcent_wdiam to floris_overlap
        self.connect("floris_wcent_wdiam.wakeCentersYT", "floris_overlap.wakeCentersYT")
        self.connect("floris_wcent_wdiam.wakeDiametersT", "floris_overlap.wakeDiametersT")

        # connections from floris_windframe to floris_overlap
        self.connect("floris_windframe.turbineXw", "floris_overlap.turbineXw")
        self.connect("floris_windframe.turbineYw", "floris_overlap.turbineYw")

        # connections from floris_windframe to floris_power
        self.connect('floris_windframe.turbineXw', 'floris_power.turbineXw')
        self.connect('floris_windframe.wsw_position', 'floris_power.wsw_position')


        # test
        # self.connect('floris_wcent_wdiam.p_near0', 'floris_overlap.p_near0')

        # connections from floris_wcent_wdiam to floris_power
        self.connect("floris_wcent_wdiam.wakeCentersY", "floris_power.wakeCentersY")
        self.connect("floris_wcent_wdiam.wakeDiameters", "floris_power.wakeDiameters")

        # connections from floris_overlap to floris_power
        self.connect("floris_overlap.wakeOverlapTRel", "floris_power.wakeOverlapTRel")
        # #################################################################

        # output connections
        self.connect("floris_power.velocitiesTurbines", "velocitiesTurbines")
        self.connect("floris_power.wt_power", "wt_power")
        self.connect("floris_power.power", "power")
        self.connect("floris_power.ws_array", "ws_array")

        # outputs for testing only
        self.connect("floris_windframe.turbineXw", "turbineXw")
        self.connect("floris_windframe.turbineYw", "turbineYw")
        self.connect("floris_wcent_wdiam.wakeCentersYT", "wakeCentersYT")
        self.connect("floris_wcent_wdiam.wakeDiametersT", "wakeDiametersT")
        self.connect("floris_overlap.wakeOverlapTRel", "wakeOverlapTRel")
        # self.connect("floris_wcent_wdiam.p_near0", "p_near0")


class floris_windframe(Component):
    """ Calculates the locations of each turbine in the wind direction reference frame """

    # original variables
    parameters = VarTree(FLORISParameters(), iotype='in')
    verbose = Bool(False, iotype='in', desc='verbosity of FLORIS, False is no output')
    # position = Array(iotype='in', units='m', desc='position of turbines in original ref. frame')
    turbineX = Array(iotype='in', desc='x positions of turbines in original ref. frame')
    turbineY = Array(iotype='in', desc='y positions of turbines in original ref. frame')

    # variables for verbosity
    Ct = Array(iotype='in', dtype='float')
    Cp = Array(iotype='in', dtype='float', desc='power coefficient for all turbines')
    axialInduction = Array(iotype='in', dtype='float', desc='axial induction of all turbines')
    yaw = Array(iotype='in', desc='yaw of each turbine')

    # variables for testing wind speed at various locations
    ws_position = Array(iotype='in', units='m', desc='position of desired measurements in original ref. frame')
    wsw_position = Array(iotype='out', units='m', deriv_ignore=True, desc='position of desired measurements in wind ref. frame')

    # flow property variables
    wind_speed = Float(iotype='in', units='m/s', desc='free stream wind velocity')
    wind_direction = Float(iotype='in', units='deg', desc='overall wind direction for wind farm')

    # for testing purposes only
    turbineXw = Array(iotype='out', units='m', desc='x coordinates of turbines in wind dir. ref. frame')
    turbineYw = Array(iotype='out', units='m', desc='y coordinates of turbines in wind dir. ref. frame')

    def execute(self):

        print 'entering windframe'

        Vinf = self.wind_speed
        windDirection = self.wind_direction*np.pi/180.0

        #variables to satisfy verbosity
        axialInd = self.axialInduction
        Cp = self.Cp
        Ct = self.Ct
        CTcorrected = self.parameters.CTcorrected
        yaw = self.yaw*np.pi/180

        # get rotor coefficients, and apply corrections if necesary
        # Cp = np.hstack(self.wt_layout.wt_array(attr='CP'))
        if CTcorrected == False:
            Ct = Ct * (np.cos(yaw)**2)


        if self.verbose:
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print "wind direction %s deg" % [windDirection*180.0/np.pi]
            print "free-stream wind speed %s" % Vinf
            print "axial induction turbines %s" % axialInd
            print "C_P turbines %s" % Cp
            print "C_T turbines %s" % Ct
            print "yaw turbines %s" % yaw

        # get turbine positions and velocity sampling positions
        # position = self.position
        # turbineX = position[:, 0]
        # turbineY = position[:, 1]
        turbineX = self.turbineX
        turbineY = self.turbineY
        # print turbineX, turbineY

        if self.ws_position.any():
            velX = self.ws_position[:, 0]
            velY = self.ws_position[:, 1]
        else:
            velX = np.zeros([0, 0])
            velY = np.zeros([0, 0])

        # convert to downwind-crosswind coordinates
        rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                                   (np.sin(-windDirection), np.cos(-windDirection))])
        # print 'rotation matrix = ', rotationMatrix
        turbineLocations = np.dot(rotationMatrix, np.array([turbineX, turbineY]))
        # print turbineLocations
        self.turbineXw = np.zeros(turbineX.size)
        self.turbineYw = np.zeros(turbineX.size)
        # print self.turbineXw
        self.turbineXw = turbineLocations[0]
        self.turbineYw = turbineLocations[1]

        if velX.size>0:
            locations = np.dot(rotationMatrix,np.array([velX,velY]))
            velX = locations[0]
            velY = locations[1]

        self.wsw_position = np.array([velX, velY])
        #print 'wsw_position in windframe is:', self.wsw_position
        #print 'ws_position in windframe is:', self.ws_position


    def list_deriv_vars(self):
        """specifies the inputs and outputs where derivatives are defined"""

        return('turbineX', 'turbineY'), ('turbineXw', 'turbineYw')
        # return ('turbineX', 'turbineY'), ('turbineXw', 'turbineYw', 'wsw_position')

    def provideJ(self):

        n = np.size(self.turbineX)

        windDirection = self.wind_direction*np.pi/180

        dturbineXw_dturbineX = np.zeros([n, n])
        dturbineXw_dturbineY = np.zeros([n, n])
        dturbineYw_dturbineX = np.zeros([n, n])
        dturbineYw_dturbineY = np.zeros([n, n])

        for i in range(0, n):
            dturbineXw_dturbineX[i, i] = np.cos(-windDirection)
            dturbineXw_dturbineY[i, i] = np.sin(-windDirection)
            dturbineYw_dturbineX[i, i] = np.sin(-windDirection)
            dturbineYw_dturbineY[i, i] = np.cos(-windDirection)

        # print 'Xw wrt X is %s' %dturbineXw_dturbineX
        # print 'cos(-windDirection) = %s' %np.cos(-windDirection)
        JturbineXw = np.concatenate((dturbineXw_dturbineX, dturbineXw_dturbineY), 1)
        JturbineYw = np.concatenate((dturbineYw_dturbineX, dturbineYw_dturbineY), 1)
        J = np.concatenate((JturbineXw, JturbineYw), 0)
        # print J.shape
        print 'the jacobian is %s' %J
        return J


class floris_wcent_wdiam(Component):
    """ Calculates the center and diameter of each turbine wake at each other turbine """

    parameters = VarTree(FLORISParameters(), iotype='in')
    verbose = Bool(False, iotype='in', desc='verbosity of FLORIS, False is no output')
    turbineXw = Array(iotype='in', desc='x coordinates of turbines in wind dir. ref. frame')
    turbineYw = Array(iotype='in', desc='y coordinates of turbines in wind dir. ref. frame')
    yaw = Array(iotype='in', desc='yaw of each turbine')
    rotorDiameter = Array(dtype='float', iotype='in', desc='rotor diameter of each turbine')
    Ct = Array(iotype='in', dtype='float', desc='thrust coefficient of each turbine')

    wsw_position = Array(iotype='in', units='m', desc='positions where measurements are desired in the windframe')

    wakeCentersYT = Array(iotype='out', dtype='float', desc='wake center y position at each turbine')
    wakeDiametersT = Array(iotype='out', dtype='float', desc='wake diameter of each zone of each wake at each turbine')
    wakeDiameters = Array(iotype='out', dtype='float', desc='wake diameter of each zone of each wake at each turbine')
    wakeCentersY = Array(iotype='out', units='m', desc='Y positions of wakes at measurement points')

    # p_near0 = Float(iotyp='out', desc='upwind location of diameter spline in rotor diameters')

    def execute(self):

        print 'entering wcent_wdiam'
         # rename inputs and outputs
        # pP = self.parameters.pP
        kd = self.parameters.kd
        ke = self.parameters.ke
        initialWakeDisplacement = self.parameters.initialWakeDisplacement
        initialWakeAngle = self.parameters.initialWakeAngle
        rotorDiameter = self.rotorDiameter
        Ct = self.Ct
        CTcorrected = self.parameters.CTcorrected
        keCorrCT = self.parameters.keCorrCT
        Region2CT = self.parameters.Region2CT
        me = self.parameters.me


        turbineXw = self.turbineXw
        turbineYw = self.turbineYw
        yaw = self.yaw*np.pi/180.0
        # print yaw
        nTurbines = turbineXw.size


        velX = self.wsw_position[0][:]
        nLocations = np.size(velX)


        # set spline start and end locations (0 is start location, or upwind, 1 is end location, or downwind)
        p_center0 = 0.25
        p_center1 = 0.25
        p_unity = 0.25
        p_near0 = 1
        p_near1 = np.copy(p_unity)
        p_far0 = np.copy(p_unity)
        p_mix0 = np.copy(p_unity)
        p_mix1 = 0.25

        if CTcorrected == False:
            Ct = Ct * (np.cos(yaw)**2)

        # calculate y-location of wake centers
        wakeCentersY = np.zeros((nLocations, nTurbines))
        wakeCentersYT = np.zeros((nTurbines, nTurbines))
        # print wakeCentersYT
        for turb in range(0, nTurbines):
            wakeAngleInit = 0.5 * np.sin(yaw[turb]) * Ct[turb] + initialWakeAngle*np.pi/180.0 # change: 4*a*(1-a) --> C_T, initial angle
            for loc in range(0, nLocations):  # at velX-locations
                deltax = np.maximum(velX[loc]-turbineXw[turb], 0)
                factor = (2.0*kd*deltax/rotorDiameter[turb])+1.0
                wakeCentersY[loc, turb] = turbineYw[turb]-initialWakeDisplacement # initial displacement for no yaw (positive to the right looking downstream)
                displacement = (wakeAngleInit*(15.0*(factor**4.0)+(wakeAngleInit**2.0))/((30.0*kd*(factor**5.0))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15.0+(wakeAngleInit**2.0))/(30.0*kd)) # yaw-induced deflection
                wakeCentersY[loc, turb] = wakeCentersY[loc, turb] + displacement
                # print "displacement %s" % displacement

            for turbI in range(0, nTurbines):  # at turbineX-locations
                # deltax = np.maximum(turbineXw[turbI]-turbineXw[turb], 0.0) #original
                deltax = turbineXw[turbI]-turbineXw[turb]
                factor = (2.0*kd*deltax/rotorDiameter[turb])+1.0
                if turbineXw[turb]+p_center1*rotorDiameter[turb] < turbineXw[turbI]:
                    wakeCentersYT[turbI, turb] = turbineYw[turb]
                    wakeCentersYT[turbI, turb] = wakeCentersYT[turbI, turb]-initialWakeDisplacement # initial displacement for no yaw (positive to the right looking downstream)
                    displacement = (wakeAngleInit*(15.0*(factor**4.0)+(wakeAngleInit**2.0))/((30.0*kd*(factor**5.0))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15.0+(wakeAngleInit**2.0))/(30.0*kd)) # yaw-induced wake center displacement
                    wakeCentersYT[turbI, turb] = wakeCentersYT[turbI, turb] + displacement
                elif turbineXw[turb]+p_center1*rotorDiameter[turb] >= turbineXw[turbI] \
                        >= turbineXw[turb]-p_center0*rotorDiameter[turb]:
                    # set up spline values
                    x = turbineXw[turbI]                                    # point of interest

                    x0 = turbineXw[turb]-p_center0*rotorDiameter[turb]      # start (upwind) point of spline
                    x1 = turbineXw[turb]+p_center1*rotorDiameter[turb]      # end (downwind) point of spline

                    y0 = turbineYw[turb]-initialWakeDisplacement           # y value at start point of spline
                    # print y0, x0
                    dy0 = 0.0                                              # slope of spline at start point

                    dx_1 = x1-turbineXw[turb]
                    factor_1 = (2.0*kd*dx_1/rotorDiameter[turb])+1.0

                    y1 = turbineYw[turb]
                    y1 += -initialWakeDisplacement # initial displacement for no yaw (positive to the right looking downstream)
                    displacement = (wakeAngleInit*(15.0*(factor_1**4.0)+(wakeAngleInit**2.0))/((30.0*kd*(factor_1**5.0))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15.0+(wakeAngleInit**2.0))/(30.0*kd)) # yaw-induced wake center displacement
                    y1 += displacement                                      # y value at end point of spline

                    b = 2.0*kd/rotorDiameter[turb]
                    d = b*dx_1+1.0
                    dy1_yaw = -(5/(d**2)+(wakeAngleInit**2)/(3*(d**8)))+4.0/(d**2.0)
                    dy1 = wakeAngleInit*dy1_yaw    # slope of spline at end point

                    # if turbI == 1 and turb == 0:
                    #     print x1, y1, dy1
                    # print 'right = %s' %dy1

                    # call spline function to determine center point of wake at point of interest.
                    wakeCentersYT[turbI, turb], _ = Hermite_Spline(x, x0, x1, y0, dy0, y1, dy1)
                    # if x1+2 > turbineXw[turbI] > x1-2 or x0+2 > turbineXw[turbI] > x0-2:
                    #     wakeCentersYT[turbI, turb] = 1495

                else:
                    wakeCentersYT[turbI, turb] = turbineYw[turb] - initialWakeDisplacement

        # adjust k_e to C_T, adjusted to yaw
        ke += ke + keCorrCT*(Ct-Region2CT) # FT = Ct*0.5*rho*A*(U*cos(yaw))^2, hence, thrust decreases with cos^2
                                                           #   Should ke increase directly with thrust? ==>No - Turbulence characteristics in wind-turbine wakes, A. Crespo"'*, J. Hern'andez b
        # print wakeCentersYT
        # calculate wake zone diameters at velX-locations
        wakeDiameters = np.zeros((nLocations, nTurbines, 3))
        wakeDiametersT = np.zeros((nTurbines, nTurbines, 3))
        dwakeDiametersT_dx = np.zeros((nTurbines, nTurbines, 3))

        for turb in range(0, nTurbines):
            wakeDiameter0 = rotorDiameter[turb] * np.cos(yaw[turb]) # CHANGE: initial wake diameter at rotor adjusted to yaw
            for loc in range(0, nLocations):  # at velX-locations
                deltax = velX[loc]-turbineXw[turb]
                for zone in range(0, 3):
                    wakeDiameters[loc, turb, zone] = wakeDiameter0 + 2*ke[turb]*me[zone]*np.maximum(deltax, 0)
            for turbI in range(0, nTurbines):  # at turbineX-locations
                deltax = turbineXw[turbI]-turbineXw[turb]
                # for zone in range(0, 3):
                #     wakeDiametersT[turbI, turb, zone] = np.maximum(wakeDiameter0 + 2*ke[turb]*me[zone]*deltax, 0)
                for zone in range(0, 3):
                    if zone == 0:
                        if turbineXw[turb]+p_near1*rotorDiameter[turb] < turbineXw[turbI]:
                            wakeDiametersT[turbI, turb, zone] = wakeDiameter0+2*ke[turb]*me[zone]*deltax
                            dwakeDiametersT_dx[turbI, turb, zone] = 2*ke[turb]*me[zone]
                        elif turbineXw[turb]+p_near1*rotorDiameter[turb] >= turbineXw[turbI] > turbineXw[turb]-p_unity*rotorDiameter[turb]:

                            x = turbineXw[turbI]                              # x position of interest
                            x1 = turbineXw[turb]-p_unity*rotorDiameter[turb]  # point where all zones have equal diameter
                            x2 = turbineXw[turb]+p_near1*rotorDiameter[turb]  # downwind end point of spline

                            # diameter at upwind point of spline
                            y1 = wakeDiameter0-2*ke[turb]*me[1]*p_unity*rotorDiameter[turb]
                            # derivative of diameter at upwind point of spline w.r.t downwind position
                            dy1_dx = 2*ke[turb]*me[1]

                            # diameter at downwind point of spline
                            y2 = wakeDiameter0+2*ke[turb]*me[zone]*(x2-turbineXw[turb])
                            # derivative of diameter at downwind point of spline w.r.t. downwind position
                            dy2_dx = 2*ke[turb]*me[zone]


                            # solve for the wake zone diameter and its derivative w.r.t. the downwind
                            # location at the point of interest
                            wakeDiametersT[turbI, turb, zone], dwakeDiametersT_dx[turbI, turb, zone] = Hermite_Spline\
                                (x, x1, x2, y1, dy1_dx, y2, dy2_dx)
                            # if turbI == 1 and turb == 0:
                            #     print dy2_dx
                            #     print dwakeDiametersT_dx[turbI, turb, zone]

                        elif turbineXw[turb]-p_near0*rotorDiameter[turb] <= turbineXw[turbI] <= turbineXw[turb]:

                            x = turbineXw[turbI]                            # x position of interest
                            x0 = turbineXw[turb]-p_near0*rotorDiameter[turb]  # downwind end point of spline
                            x1 = turbineXw[turb]-p_unity*rotorDiameter[turb]  # point where all zones have equal diameter

                            # diameter at upwind point of spline
                            y0 = 0
                            # derivative of diameter at upwind point of spline
                            dy0_dx = 0

                            # diameter at upwind point of spline
                            y1 = wakeDiameter0-2*ke[turb]*me[1]*p_unity*rotorDiameter[turb]
                            # derivative of diameter at upwind point of spline
                            dy1_dx = 2*ke[turb]*me[1]

                            # solve for the wake zone diameter and its derivative w.r.t. the downwind
                            # location at the point of interest
                            wakeDiametersT[turbI, turb, zone], dwakeDiametersT_dx[turbI, turb, zone] = Hermite_Spline\
                                (x, x0, x1, y0, dy0_dx, y1, dy1_dx)
                            # if turbI == 1 and turb == 0:
                            #     print dy1_dx
                            #     print dwakeDiametersT_dx[turbI, turb, zone]

                    elif zone == 1:
                        if turbineXw[turb]-p_far0*rotorDiameter[turb] < turbineXw[turbI]:
                            wakeDiametersT[turbI, turb, zone] = wakeDiameter0 + 2*ke[turb]*me[zone]*deltax
                            dwakeDiametersT_dx[turbI, turb, zone] = 2*ke[turb]*me[zone]
                        else:
                            wakeDiametersT[turbI, turb, zone] = wakeDiametersT[turbI, turb, 0]

                    elif zone == 2:

                        if turbineXw[turb]+p_mix1*rotorDiameter[turb] < turbineXw[turbI]:
                            wakeDiametersT[turbI, turb, zone] = wakeDiameter0+2*ke[turb]*me[zone]*deltax
                            dwakeDiametersT_dx[turbI, turb, zone] = 2*ke[turb]*me[zone]

                        elif turbineXw[turb]+p_mix1*rotorDiameter[turb] >= turbineXw[turbI] > \
                                        turbineXw[turb]-p_mix0*rotorDiameter[turb]:
                            x = turbineXw[turbI]                             # x position of interest
                            x0 = turbineXw[turb]-p_mix0*rotorDiameter[turb]  # downwind end point of spline
                            x1 = turbineXw[turb]+p_mix1*rotorDiameter[turb]  # point where all zones have equal diameter

                            # diameter at upwind point of spline
                            y0 = wakeDiameter0-2*ke[turb]*me[1]*p_mix0*rotorDiameter[turb]
                            # derivative of diameter at upwind point of spline w.r.t downwind position
                            dy0_dx = 2*ke[turb]*me[1]

                            # diameter at downwind point of spline
                            y1 = wakeDiameter0+2*ke[turb]*me[zone]*p_mix1*rotorDiameter[turb]
                            # derivative of diameter at downwind point of spline w.r.t. downwind position
                            dy1_dx = 2*ke[turb]*me[zone]

                            # solve for the wake zone diameter and its derivative w.r.t. the downwind
                            # location at the point of interest
                            wakeDiametersT[turbI, turb, zone], dwakeDiametersT_dx[turbI, turb, zone] = Hermite_Spline\
                                (x, x0, x1, y0, dy0_dx, y1, dy1_dx)

                        else:
                            wakeDiametersT[turbI, turb, zone] = wakeDiametersT[turbI, turb, 0]


        self.wakeCentersYT = wakeCentersYT
        self.wakeDiametersT = wakeDiametersT
        self.wakeDiameters = wakeDiameters
        self.wakeCentersY = wakeCentersY
        # print self.wakeCentersYT

        # testing
        # self.p_near0 = p_near0

    #
    # def list_deriv_vars(self):
    #
    #     return ('')
    #
    # def provideJ(self):
    #
    #
    #     return J


class floris_overlap(Component):
    """ Calculates the overlap between each turbine rotor and the existing turbine wakes """
    turbineXw = Array(iotype='in', units='m', desc='X positions of turbines wrt the wind direction')
    turbineYw = Array(iotype='in', units='m', desc='Y positions of turbines wrt the wind direction')
    rotorDiameter = Array(iotype='in', units='m', desc='diameters of all turbine rotors')
    wakeDiametersT = Array(iotype='in', units='m', desc='diameters of all turbines wake zones')
    wakeCentersYT = Array(iotype='in', units='m', desc='Y positions of all wakes at each turbine')
    rotorArea = Array(iotype='in', units='m*m', desc='Area of each turbine rotor')

    wakeOverlapTRel = Array(iotype='out', desc='relative wake zone overlap to rotor area')

    # p_near0 = Float(iotype='in', desc='upwind location of diameter spline in rotor diameters')

    def execute(self):
        print 'entering overlap'
        nTurbines = self.turbineYw.size
        # p_near0 = self.p_near0
        p_near0 = 1
        # calculate overlap areas at rotors
        # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
        # of turbine TURB with rotor of turbine TURBI
        # wakeOverlapT = calcOverlapAreas(self.turbineXw, self.turbineYw, self.rotorDiameter, self.wakeDiametersT, self.wakeCentersYT, p_near0)

        # make overlap relative to rotor area (maximum value should be 1)
        # wakeOverlapTRel = wakeOverlapT
        # for turb in range(0, nTurbines): # Jared: I think it would make more sense to use turbI for consistency
        #     wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/self.rotorArea[turb]

        wakeOverlapTRel = calcOverlapAreas(self.turbineXw, self.turbineYw, self.rotorDiameter, self.wakeDiametersT, self.wakeCentersYT, p_near0)

        self.wakeOverlapTRel = wakeOverlapTRel
        # print self.wakeOverlapTRel
        # print '_'


class floris_power(Component):
    """ Calculates the turbine power and effective wind speed for each turbine """

    # original variables in Pieter's OpenMDAO stand-alone version of FLORIS
    parameters = VarTree(FLORISParameters(), iotype='in')
    velocitiesTurbines = Array(iotype='out', units='m/s')
    verbose = Bool(False, iotype='in', desc='verbosity of FLORIS, False is no output')

    # input variables added so I don't have to use WISDEM while developing gradients
    rotorDiameter = Array(dtype='float', iotype='in', units='m', desc='rotor diameters of all turbine')
    rotorArea = Array(iotype='in', dtype='float', units='m*m', desc='rotor area of all turbines')
    axialInduction = Array(iotype='in', dtype='float', desc='axial induction of all turbines')
    Ct = Array(iotype='in', dtype='float', desc='Thrust coefficient for all turbines')
    Cp = Array(iotype='in', dtype='float', desc='power coefficient for all turbines')
    generator_efficiency = Array(iotype='in', dtype='float', desc='generator efficiency of all turbines')
    yaw = Array(iotype='in', desc='yaw of each turbine')
    turbineXw = Array(iotype='in', dtype='float', units='m', desc='X positions of turbines in the wind direction reference frame')
    wakeCentersYT = Array(iotype='in', units='m', desc='centers of the wakes at each turbine')
    wakeDiametersT = Array(iotype='in', units='m', desc='diameters of each of the wake zones for each of the wakes at each turbine')
    wakeOverlapTRel = Array(iotype='in', units='m', desc='ratios of wake overlap area per zone to rotor area')
    wsw_position = Array(iotype='in', units='m', desc='positions where measurements are desired in the windframe')
    wakeDiameters = Array(iotype='in', units='m', desc='diameter of wake zones at measurement points')
    wakeCentersY = Array(iotype='in', units='m', desc='Y positions of wakes at measurement points')

    # Flow property variables
    wind_speed = Float(iotype='in', units='m/s', desc='free stream wind velocity')
    air_density = Float(iotype='in', units='kg/(m*m*m)', desc='air density in free stream')

    # output variables added so I don't have to use WISDEM while developing gradients
    wt_power = Array(iotype='out', units='kW')
    power = Float(iotype='out', units='kW', desc='total power output of the wind farm')
    ws_array = Array(iotype='out', units='m/s', desc='wind speed at measurement locations')

    def execute(self):
        print 'entering power'
        turbineXw = self.turbineXw
        nTurbines = turbineXw.size
        #print 'number of turbines is: ', nTurbines
        wakeOverlapTRel = self.wakeOverlapTRel
        ke = self.parameters.ke
        #print 'ke is: ', ke
        keCorrArray = self.parameters.keCorrArray
        keCorrCT = self.parameters.keCorrCT
        Region2CT = self.parameters.Region2CT
        CTcorrected = self.parameters.CTcorrected
        Ct = self.Ct
        Vinf = self.wind_speed
        turbineXw = self.turbineXw
        axialInduction = self.axialInduction
        rotorDiameter = self.rotorDiameter
        rotorArea = self.rotorArea
        rho = self.air_density
        generator_efficiency = self.generator_efficiency
        yaw = self.yaw
        Cp = self.Cp
        MU = self.parameters.MU

        velX = self.wsw_position[0][:]
        velY = self.wsw_position[1][:]
        # print 'wsw_position in power is: ', self.wsw_position
        nLocations = np.size(velX)
        print 'nlocations is', nLocations
        wakeCentersY = self.wakeCentersY
        wakeDiameters = self.wakeDiameters

        axialIndProvided = self.parameters.axialIndProvided

        # how far upwind (in rotor diameters) to calculate power (must correspond to the value in wake overlap calculations)
        p_near0 = 1


        if CTcorrected == False:
            Ct = Ct * (np.cos(yaw)**2)

        if axialIndProvided:
            axialInd = axialInduction
        else:
            axialInd = np.array([CTtoAxialInd(ct) for ct in Ct])

        # adjust k_e to C_T, adjusted to yaw
        ke = ke + keCorrCT*(Ct-Region2CT) # FT = Ct*0.5*rho*A*(U*cos(yaw))^2, hence, thrust decreases with cos^2
                                                           #   Should ke increase directly with thrust? ==>No - Turbulence characteristics in wind-turbine wakes, A. Crespo"'*, J. Hern'andez b

        # array effects with full or partial wake overlap:
        # use overlap area of zone 1 + 2 of upstream turbines to correct ke
        # Note: array effects only taken into account in calculating
        # velocity deficits, in order not to over-complicate code
        # (avoid loops in calculating overlaps)

        keArray = np.zeros(nTurbines)
        for turb in range(0, nTurbines):
            s = np.sum(wakeOverlapTRel[turb, :, 0]+wakeOverlapTRel[turb, :, 1])
            keArray[turb] = ke[turb]*(1+s*keCorrArray)

        # calculate velocities in full flow field (optional)
        self.ws_array = np.tile(Vinf, nLocations)
        #print 'nLocations in power is: ', nLocations
        for turb in range(0, nTurbines):
            #mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb]) // CHANGE: ke now only corrected with CT, which is already corrected with yaw
            for loc in range(0, nLocations):
                deltax = velX[loc] - turbineXw[turb]
                radiusLoc = abs(velY[loc]-wakeCentersY[loc, turb])
                axialIndAndNearRotor = 2*axialInd[turb]

                if deltax > 0 and radiusLoc < wakeDiameters[loc, turb, 0]/2.0:    # check if in zone 1
                    reductionFactor = axialIndAndNearRotor*\
                                      np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*keArray[turb]*(MU[0])*np.maximum(0, deltax))), 2)
                elif deltax > 0 and radiusLoc < wakeDiameters[loc, turb, 1]/2.0:    # check if in zone 2
                    reductionFactor = axialIndAndNearRotor*\
                                      np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*keArray[turb]*(MU[1])*np.maximum(0, deltax))), 2)
                elif deltax > 0 and radiusLoc < wakeDiameters[loc, turb, 2]/2.0:    # check if in zone 3
                    reductionFactor = axialIndAndNearRotor*\
                                      np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*keArray[turb]*(MU[2])*np.maximum(0, deltax))), 2)
                elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
                    reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0, deltax)/(rotorDiameter[turb]))/np.pi)
                else:
                    reductionFactor = 0
                self.ws_array[loc] *= (1-reductionFactor)
        # print 'ws_array in floris_power is: ', self.ws_array
        # find effective wind speeds at downstream turbines, then predict power downstream turbine
        self.velocitiesTurbines = np.tile(Vinf, nTurbines)

        for turbI in range(0, nTurbines):

            # find overlap-area weighted effect of each wake zone
            wakeEffCoeff = 0
            for turb in range(0, nTurbines):

                wakeEffCoeffPerZone = 0
                deltax = turbineXw[turbI] - turbineXw[turb]

                if deltax > -1*p_near0*rotorDiameter[turb] and turbI != turb:
                # if deltax > 0:
                    #mU = MU / np.cos(aU*np.pi/180 + bU*yaw[turb]) // CHANGE: ke now only corrected with CT, which is already corrected with yaw
                    for zone in range(0, 3):
                        #wakeEffCoeffPerZone = wakeEffCoeffPerZone + np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU[zone]*deltax),2.0) * wakeOverlapTRel[turbI,turb,zone]
                        wakeEffCoeffPerZone = wakeEffCoeffPerZone + np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*MU[zone]*deltax), 2.0) * wakeOverlapTRel[turbI, turb, zone]

                    wakeEffCoeff = wakeEffCoeff + np.power(axialInd[turb]*wakeEffCoeffPerZone, 2.0)

            wakeEffCoeff = (1 - 2 * np.sqrt(wakeEffCoeff))
            #print wakeEffCoeff

            # multiply the inflow speed with the wake coefficients to find effective wind speed at turbine
            self.velocitiesTurbines[turbI] *= wakeEffCoeff

        if self.verbose:
            print "wind speed at turbines %s [m/s]" % self.velocitiesTurbines
            print "rotor area %s" % rotorArea
            print "rho %s" % rho
            print "generator_efficiency %s" % generator_efficiency

        # find turbine powers
        self.wt_power = np.power(self.velocitiesTurbines, 3.0) * (0.5*rho*rotorArea*Cp) * generator_efficiency

        # # set outputs on turbine level
        # for turbI in range(0, nTurbines):
        #     turbineName = self.wt_layout.wt_names[turbI]
        #     getattr(self.wt_layout, turbineName).power = self.wt_power[turbI] # in W
        #     getattr(self.wt_layout, turbineName).wind_speed_eff = self.velocitiesTurbines[turbI]

        self.wt_power /= 1000  # in kW

        if self.verbose:
            print "powers turbines %s [kW]" % self.wt_power

        self.power = np.sum(self.wt_power)


def Hermite_Spline(x, x0, x1, y0, dy0, y1, dy1):
    """
    This function produces the y and dy values for a hermite cubic spline
    interpolating between two end points with known slopes

    :param x: x position of output y
    :param x0: x position of upwind endpoint of spline
    :param x1: x position of downwind endpoint of spline
    :param y0: y position of upwind endpoint of spline
    :param dy0: slope at upwind endpoint of spline
    :param y1: y position of downwind endpoint of spline
    :param dy1: slope at downwind endpoint of spline

    :return: [y: y value of spline at location x, dy: slope of spline at location x]

    """

    # initialize coefficients for parametric cubic spline
    # c3 = (2*(y1))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) - (2*(y0))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) + (dy0)/(x0**2 - 2*x0*x1 + x1**2) + (dy1)/(x0**2 - 2*x0*x1 + x1**2)
    # c2 = (3*(y0)*(x0 + x1))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) - ((dy1)*(2*x0 + x1))/(x0**2 - 2*x0*x1 + x1**2) - ((dy0)*(x0 + 2*x1))/(x0**2 - 2*x0*x1 + x1**2) - (3*(y1)*(x0 + x1))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3)
    # c1 = ((dy0)*(x1**2 + 2*x0*x1))/(x0**2 - 2*x0*x1 + x1**2) + ((dy1)*(x0**2 + 2*x1*x0))/(x0**2 - 2*x0*x1 + x1**2) - (6*x0*x1*(y0))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) + (6*x0*x1*(y1))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3)
    # c0 = ((y0)*(- x1**3 + 3*x0*x1**2))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) - ((y1)*(- x0**3 + 3*x1*x0**2))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) - (x0*x1**2*(dy0))/(x0**2 - 2*x0*x1 + x1**2) - (x0**2*x1*(dy1))/(x0**2 - 2*x0*x1 + x1**2)
    #
    # # Solve for y and dy values at the given point
    # y = c3*x**3 + c2*x**2 + c1*x + c0
    # dy = c3*3*x**3 + c2*2*x**2 + c1*x
    # print dy
    # print 'y = %s' %y

    # initialize coefficients for parametric cubic spline
    c3 = (2*(y1))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) - (2*(y0))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) + (dy0)/(x0**2 - 2*x0*x1 + x1**2) + (dy1)/(x0**2 - 2*x0*x1 + x1**2)
    c2 = (3*(y0)*(x0 + x1))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) - ((dy1)*(2*x0 + x1))/(x0**2 - 2*x0*x1 + x1**2) - ((dy0)*(x0 + 2*x1))/(x0**2 - 2*x0*x1 + x1**2) - (3*(y1)*(x0 + x1))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3)
    c1 = ((dy0)*(x1**2 + 2*x0*x1))/(x0**2 - 2*x0*x1 + x1**2) + ((dy1)*(x0**2 + 2*x1*x0))/(x0**2 - 2*x0*x1 + x1**2) - (6*x0*x1*(y0))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) + (6*x0*x1*(y1))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3)
    c0 = ((y0)*(- x1**3 + 3*x0*x1**2))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) - ((y1)*(- x0**3 + 3*x1*x0**2))/(x0**3 - 3*x0**2*x1 + 3*x0*x1**2 - x1**3) - (x0*x1**2*(dy0))/(x0**2 - 2*x0*x1 + x1**2) - (x0**2*x1*(dy1))/(x0**2 - 2*x0*x1 + x1**2)

    # Solve for y and dy values at the given point
    y = c3*x**3 + c2*x**2 + c1*x + c0
    dy_dx = c3*3*x**2 + c2*2*x + c1


    return y, dy_dx


def CTtoAxialInd(CT):
    if CT > 0.96: # Glauert condition
        axial_induction = 0.143+np.sqrt(0.0203-0.6427*(0.889-CT))
    else:
        axial_induction = 0.5*(1-np.sqrt(1-CT))
    return axial_induction


def calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiameters,wakeCenters,p_near0):
    """calculate overlap of rotors and wake zones (wake zone location defined by wake center and wake diameter)
    turbineX,turbineY is x,y-location of center of rotor

    wakeOverlap(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake of turbine TURB with rotor of downstream turbine
    TURBI"""

    nTurbines = turbineY.size

    wakeOverlap = np.zeros((nTurbines, nTurbines, 3))

    for turb in range(0, nTurbines):
        for turbI in range(0, nTurbines):
            if turbineX[turbI] > turbineX[turb] - p_near0*rotorDiameter[turb]:
                # print 'turb = %s, ' %turb
                # print 'turbI = %s, ' %turbI
                # print 'turbineX = %s, ' %turbineX
                # print 'rotorDiameter = %s' %rotorDiameter
                # print 'wakeCenters = %s' %wakeCenters
                OVdYd = wakeCenters[turbI, turb]-turbineY[turbI]    # distance between wake center and rotor center
                OVr = rotorDiameter[turbI]/2    # rotor diameter
                for zone in range(0, 3):
                    OVR = wakeDiameters[turbI, turb, zone]/2    # wake diameter
                    OVdYd = abs(OVdYd)
                    if OVdYd != 0:
                        # calculate the distance from the wake center to the vertical line between
                        # the two circle intersection points
                        OVL = (-np.power(OVr, 2.0)+np.power(OVR, 2.0)+np.power(OVdYd, 2.0))/(2.0*OVdYd)
                    else:
                        OVL = 0

                    OVz = np.power(OVR, 2.0)-np.power(OVL, 2.0)

                    # Finish calculating the distance from the intersection line to the outer edge of the wake zone
                    if OVz > 0:
                        OVz = np.sqrt(OVz)
                    else:
                        OVz = 0

                    if OVdYd < (OVr+OVR): # if the rotor overlaps the wake zone
                        # if
                        if OVL < OVR and (OVdYd-OVL) < OVr:
                            wakeOverlap[turbI, turb, zone] = np.power(OVR, 2.0)*np.arccos(OVL/OVR) + np.power(OVr, 2.0)*np.arccos((OVdYd-OVL)/OVr) - OVdYd*OVz
                        elif OVR > OVr:
                            wakeOverlap[turbI, turb, zone] = np.pi*np.power(OVr, 2.0)
                        else:
                            wakeOverlap[turbI, turb, zone] = np.pi*np.power(OVR, 2.0)
                    else:
                        wakeOverlap[turbI, turb, zone] = 0


    for turb in range(0, nTurbines):
        for turbI in range(0, nTurbines):
            wakeOverlap[turbI, turb, 2] = wakeOverlap[turbI, turb, 2]-wakeOverlap[turbI, turb, 1]
            wakeOverlap[turbI, turb, 1] = wakeOverlap[turbI, turb, 1]-wakeOverlap[turbI, turb, 0]

    wakeOverlapTRel = wakeOverlap

    for turbI in range(0, nTurbines): # Jared: I think it would make more sense to use turbI for consistency
            wakeOverlapTRel[turbI] = wakeOverlapTRel[turbI]/(np.pi*rotorDiameter[turbI]^2/4)


    return wakeOverlapTRel