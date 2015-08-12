from openmdao.main.api import Assembly
from openmdao.lib.datatypes.api import Array, Bool, Float, VarTree
from openmdao.lib.drivers.api import FixedPointIterator, SLSQPdriver, COBYLAdriver
from pyopt_driver.pyopt_driver import pyOptDriver
from openmdao.lib.casehandlers.listcase import ListCaseIterator
from Parameters import FLORISParameters

import numpy as np

# ###########    imports for discontinuous (original) model    ########################################################
# from Original_components import floris_windframe
# from Original_components import floris_wcent_wdiam
# from Original_components import floris_overlap
# from Original_components import floris_power

# ###########    imports for smooth model with analytic gradients    ##################################################
from Analytic_components import floris_adjustCtCp
from Analytic_components import floris_windframe
from Analytic_components import AEP
from Analytic_components import dist_const

# ###########    imports for smooth model with Tapenade provided gradients    #########################################
from Tapenade_components import floris_wcent_wdiam
from Tapenade_components import floris_overlap
from Tapenade_components import floris_power

# ###########    imports for rotor modeling    ########################################################################
from rotor_components import *

# ###########    imports for Fortran components (no provided gradient)    #############################################
# from Fortran_components import floris_wcent_wdiam
# from Fortran_components import floris_overlap
# from Fortran_components import floris_power


class floris_assembly_opt_AEP(Assembly):
    """ Defines the connections between each Component used in the FLORIS model """

    # general input variables
    parameters = VarTree(FLORISParameters(), iotype='in')
    verbose = Bool(False, iotype='in', desc='verbosity of FLORIS, False is no output')
    # optimize_yaw = Bool(False, iotyp='in', desc='optimize yaw for each wind direction, False keep the input yaw values')


    # Flow property variables
    wind_speed = Float(iotype='in', units='m/s', desc='free stream wind velocity')
    air_density = Float(iotype='in', units='kg/(m*m*m)', desc='air density in free stream')

    # output
    AEP = Float(iotype='out', units='kW', desc='total windfarm AEP')

    # def __init__(self, turbineX, turbineY, yaw, resolution):
    def __init__(self, nTurbines, nDirections, nSpeeds=1, optimize_position=False, resolution=0., optimize_yaw=False,
                 use_rotor_components=False, datasize=0):

        super(floris_assembly_opt_AEP, self).__init__()

        self.nTurbines = nTurbines
        self.resolution = resolution
        self.nDirections = nDirections
        self.optimize_yaw = optimize_yaw
        self.optimize_position = optimize_position
        self.use_rotor_components = use_rotor_components
        self.datasize = datasize

        # wt_layout input variables
        self.add('rotorDiameter', Array(np.zeros(nTurbines), dtype='float', iotype='in', units='m',
                                        desc='rotor diameters of all turbine'))
        self.add('axialInduction', Array(np.zeros(nTurbines), iotype='in', dtype='float',
                                         desc='axial induction of all turbines'))
        if use_rotor_components:
            # turbine properties for ccblade and pre-calculated controller
            # self.add('windSpeedToCPCT', VarTree(windSpeedToCPCT(datasize), iotype='in', desc='pre-calculated CPCT'))
            self.add('curve_CP', Array(np.zeros(datasize), iotype='in', desc='pre-calculated CPCT'))
            self.add('curve_CT', Array(np.zeros(datasize), iotype='in', desc='pre-calculated CPCT'))
            self.add('curve_wind_speed', Array(np.zeros(datasize), iotype='in', desc='pre-calculated CPCT'))
            self.add('initVelocitiesTurbines', Array(np.zeros(nTurbines), iotype='in', units='m/s'))
            # print 'in assembly', self.windSpeedToCPCT.CP.size, self.windSpeedToCPCT.CT.size, self.windSpeedToCPCT.wind_speed.size
        else:
            self.add('Ct', Array(np.zeros(nTurbines), iotype='in', desc='Thrust coefficient for all turbines'))
            self.add('Cp', Array(np.zeros(nTurbines), iotype='in', dtype='float',
                                 desc='power coefficient for all turbines'))
        self.add('generator_efficiency', Array(np.zeros(nTurbines), iotype='in', dtype='float',
                                               desc='generator efficiency of all turbines'))
        self.add('turbineX', Array(np.zeros(nTurbines), iotype='in', dtype='float',
                                   desc='x positions of turbines in original ref. frame'))
        self.add('turbineY', Array(np.zeros(nTurbines), iotype='in', dtype='float',
                                   desc='y positions of turbines in original ref. frame'))
        if optimize_yaw:
            # self.add('yaw', Array(np.zeros(nTurbines*nDirections), iotype='in', dtype='float', \
            #          desc='yaw of each turbine for each direction'))
            for direction in range(0, nDirections):
                self.add('yaw_%d' % direction, Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                         desc='yaw of each turbine for each direction'))
        else:
            self.add('yaw', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                              desc='yaw of each turbine'))

        # windrose input variables
        self.add('windrose_frequencies', Array(np.zeros(nDirections), dtype='float', iotype='in',
                                               desc='frequencies at each direction cw from north'))
        self.add('windrose_directions', Array(np.zeros(nDirections), dtype='float', iotype='in',
                                              desc='windrose directions in degrees ccw from east'))


        # Explicitly size output arrays

        # variables added to test individual components
        self.add('turbineXw', Array(np.zeros(nTurbines), iotype='out', units='m',
                                    desc='X positions of turbines in the wind direction reference frame'))
        self.add('turbineYw', Array(np.zeros(nTurbines), iotype='out', units='m',
                                    desc='Y positions of turbines in the wind direction reference frame'))
        self.add('wakeCentersYT', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m',
                                        desc='centers of the wakes at each turbine'))
        self.add('wakeDiametersT', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m',
                                         desc='diameters of each of the wake zones for each of the \
                                         wakes at each turbine'))
        self.add('wakeOverlapTRel', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m',
                                          desc='ratio of overlap area of each zone to rotor area'))

        # standard output
        self.add('velocitiesTurbines_directions', Array(np.zeros([nDirections, nTurbines]), iotype='out', units='m/s',
                                                        dtype='float', desc='effective windspeed at each turbine \
                                                        in each direction ccw from east using direction to'))
        self.add('wt_power_directions', Array(np.zeros([nDirections, nTurbines]), iotype='out', units='kW',
                                              dtype='float', desc='power of each turbine in each direction ccw from \
                                              east using direction to'))
        self.add('power_directions', Array(np.zeros(nDirections), iotype='out', units='kW', desc='total windfarm power \
                                           in each direction ccw from east using direction to'))

    def configure(self):

        # rename options
        nTurbines = self.nTurbines
        nDirections = self.nDirections
        optimize_position = self.optimize_position
        resolution = self.resolution
        optimize_yaw = self.optimize_yaw
        use_rotor_components = self.use_rotor_components
        datasize = self.datasize

        # add driver so the workflow is not overwritten later
        if optimize_position or optimize_yaw:
            # self.add('driver', COBYLAdriver())
            # self.driver.gradient_options.force_fd = True
            self.add('driver', pyOptDriver())
            self.driver.optimizer = 'SNOPT'
            # self.driver.pyopt_diff = True

        # add AEP component first so it can be connected to
        F6 = self.add('floris_AEP', AEP(nDirections=nDirections))
        F6.missing_deriv_policy = 'assume_zero'
        self.connect('windrose_frequencies', 'floris_AEP.windrose_frequencies')
        self.connect('floris_AEP.AEP', 'AEP')
        self.connect('floris_AEP.power_directions_out', 'power_directions')

        # set up constraints
        self.add('floris_dist_const', dist_const(nTurbines=nTurbines))
        self.connect('turbineX', 'floris_dist_const.turbineX')
        self.connect('turbineY', 'floris_dist_const.turbineY')

        print 'in configure, diirections = ', self.windrose_directions

        for i in range(0, nDirections):

            print 'i = %s' % i

            # add CpCt method
            if use_rotor_components:
                # add fixed point iterator
                self.add('FPIdriver_%d' % i, FixedPointIterator())
                self.add('rotor_CPCT_%d' % i, CPCT_Interpolate(nTurbines=self.nTurbines, datasize=self.datasize))
            else:
                self.add('floris_adjustCtCp_%d' % i, floris_adjustCtCp(nTurbines=nTurbines))

            # add components of floris to assembly
            F2 = self.add('floris_windframe_%d' % i, floris_windframe(nTurbines=nTurbines,
                                                                 resolution=resolution))
            F2.missing_deriv_policy = 'assume_zero'
            self.add('floris_wcent_wdiam_%d' % i, floris_wcent_wdiam(nTurbines=nTurbines))
            F4 = self.add('floris_overlap_%d' % i, floris_overlap(nTurbines=nTurbines))
            F4.missing_deriv_policy = 'assume_zero'
            self.add('floris_power_%d' % i, floris_power(nTurbines=nTurbines))

            # connect inputs to components
            if use_rotor_components:
                # self.connect('initVelocitiesTurbines', ['rotor_CPCT_%d.wind_speed_hub' % i])
                self.connect('curve_CP', 'rotor_CPCT_%d.windSpeedToCPCT.CP' % i)
                self.connect('curve_CT', 'rotor_CPCT_%d.windSpeedToCPCT.CT' % i)
                self.connect('curve_wind_speed', 'rotor_CPCT_%d.windSpeedToCPCT.wind_speed' % i)
                # self.connect('windSpeedToCPCT', 'rotor_CPCT_%d.windSpeedToCPCT' % i)
                # self.connect('windSpeedToCPCT.CT', 'rotor_CPCT_%d.windSpeedToCPCT.CT' % i)
                # self.connect('windSpeedToCPCT.wind_speed', 'rotor_CPCT_%d.windSpeedToCPCT.wind_speed' % i)
                self.connect('parameters.pP', 'rotor_CPCT_%d.pP' % i)
                self.connect('parameters', ['floris_wcent_wdiam_%d.parameters' % i, 'floris_power_%d.parameters' % i])
            else:
                self.connect('parameters', ['floris_adjustCtCp_%d.parameters' % i, 'floris_wcent_wdiam_%d.parameters' % i,
                                        'floris_power_%d.parameters' % i])
                self.connect('Ct', 'floris_adjustCtCp_%d.Ct_in' % i)
                self.connect('Cp', 'floris_adjustCtCp_%d.Cp_in' % i)

            self.connect('verbose', ['floris_windframe_%d.verbose' % i, 'floris_wcent_wdiam_%d.verbose' % i,
                                     'floris_power_%d.verbose' % i])
            self.connect('turbineX', 'floris_windframe_%d.turbineX' % i)
            self.connect('turbineY', 'floris_windframe_%d.turbineY' % i)
            self.connect('rotorDiameter', ['floris_wcent_wdiam_%d.rotorDiameter' % i,
                                           'floris_overlap_%d.rotorDiameter' % i, 'floris_power_%d.rotorDiameter' % i])
            self.connect('axialInduction', 'floris_power_%d.axialInduction' % i)
            self.connect('generator_efficiency', 'floris_power_%d.generator_efficiency' % i)

            if optimize_yaw:
                # for j in range(0, nTurbines):
                #     self.connect('yaw[%d]' % (i*nTurbines+j), ['floris_adjustCtCp_%d.yaw[%d]' % (i, j), 'floris_wcent_wdiam_%d.yaw[%d]' % (i, j),
                #                      'floris_power_%d.yaw[%d]' % (i, j)])
                # self.connect('yaw[%d*%d:%d*%d:1]' % (i, nTurbines, i+1, nTurbines), ['floris_adjustCtCp_%d.yaw' % i, 'floris_wcent_wdiam_%d.yaw' % i,
                #                  'floris_power_%d.yaw' % i])
                if use_rotor_components:
                    self.connect('yaw_%d' % i, ['rotor_CPCT_%d.yaw' % i, 'floris_wcent_wdiam_%d.yaw' % i,
                                 'floris_power_%d.yaw' % i])
                else:
                    self.connect('yaw_%d' % i, ['floris_adjustCtCp_%d.yaw' % i, 'floris_wcent_wdiam_%d.yaw' % i,
                                 'floris_power_%d.yaw' % i])

                # print self.yaw[i*nTurbines:(i+1)*nTurbines]
            else:
                if use_rotor_components:
                    self.connect('yaw', ['rotor_CPCT_%d.yaw' % i, 'floris_wcent_wdiam_%d.yaw' % i,
                                 'floris_power_%d.yaw' % i])
                else:
                    self.connect('yaw', ['floris_adjustCtCp_%d.yaw' % i, 'floris_wcent_wdiam_%d.yaw' % i,
                                 'floris_power_%d.yaw' % i])

            self.connect('wind_speed', 'floris_power_%d.wind_speed' % i)
            self.connect('air_density', 'floris_power_%d.air_density' % i)
            self.connect('windrose_directions[%d]' % i, 'floris_windframe_%d.wind_direction' % i)

            # for satisfying the verbosity in windframe
            if optimize_yaw:
                self.connect('yaw_%d' % i, 'floris_windframe_%d.yaw' % i)
            else:
                self.connect('yaw', 'floris_windframe_%d.yaw' % i)
            if use_rotor_components:
                self.connect('rotor_CPCT_%d.CT' % i, 'floris_windframe_%d.Ct' % i)
                self.connect('rotor_CPCT_%d.CP' % i, 'floris_windframe_%d.Cp' % i)
            else:
                self.connect('floris_adjustCtCp_%d.Ct_out' % i, 'floris_windframe_%d.Ct' % i)
                self.connect('floris_adjustCtCp_%d.Cp_out' % i, 'floris_windframe_%d.Cp' % i)
            self.connect('wind_speed', 'floris_windframe_%d.wind_speed' % i)
            self.connect('axialInduction', 'floris_windframe_%d.axialInduction' % i)

            # ############### Connections between components ##################
            # connections from CtCp calculation to other components
            if use_rotor_components:
                self.connect('rotor_CPCT_%d.CT' % i, ['floris_wcent_wdiam_%d.Ct' % i, 'floris_power_%d.Ct' % i])
                self.connect('rotor_CPCT_%d.CP' % i, 'floris_power_%d.Cp' % i)
            else:
                self.connect('floris_adjustCtCp_%d.Ct_out' % i, ['floris_wcent_wdiam_%d.Ct' % i, 'floris_power_%d.Ct' % i])
                self.connect('floris_adjustCtCp_%d.Cp_out' % i, 'floris_power_%d.Cp' % i)

            # connections from floris_windframe to floris_wcent_wdiam
            self.connect('floris_windframe_%d.turbineXw' % i, 'floris_wcent_wdiam_%d.turbineXw' % i)
            self.connect('floris_windframe_%d.turbineYw' % i, 'floris_wcent_wdiam_%d.turbineYw' % i)

            # connections from floris_wcent_wdiam to floris_overlap
            self.connect('floris_wcent_wdiam_%d.wakeCentersYT' % i, 'floris_overlap_%d.wakeCentersYT' % i)
            self.connect('floris_wcent_wdiam_%d.wakeDiametersT' % i, 'floris_overlap_%d.wakeDiametersT' % i)

            # connections from floris_windframe to floris_overlap
            self.connect('floris_windframe_%d.turbineXw' % i, 'floris_overlap_%d.turbineXw' % i)
            self.connect('floris_windframe_%d.turbineYw' % i, 'floris_overlap_%d.turbineYw' % i)

            # connections from floris_windframe to floris_power
            self.connect('floris_windframe_%d.turbineXw' % i, 'floris_power_%d.turbineXw' % i)

            # connections from floris_overlap to floris_power
            self.connect('floris_overlap_%d.wakeOverlapTRel' % i, 'floris_power_%d.wakeOverlapTRel' % i)

            # connections from floris_power to floris_AEP
            self.connect('floris_power_%d.power' % i, 'floris_AEP.power_directions[%d]' % i)
            # #################################################################

            # output connections
            # self.connect('floris_power_%d.velocitiesTurbines' % i, 'velocitiesTurbines_directions[%d]' % i)
            # self.connect('floris_power_%d.wt_power' % i, 'wt_power_directions[%d]' % i)
            # self.connect('floris_power_%d.power' % i, 'power_directions[%d]' % i)

            # add to workflow
            if use_rotor_components:
                exec("self.FPIdriver_%d.workflow.add(['rotor_CPCT_%d', 'floris_windframe_%d', \
                     'floris_wcent_wdiam_%d', 'floris_overlap_%d', 'floris_power_%d'])" % (i, i, i, i, i, i))
                exec("self.FPIdriver_%d.add_parameter('rotor_CPCT_%d.wind_speed_hub', low=0., high=100.)" % (i, i))
                exec("self.FPIdriver_%d.add_constraint('rotor_CPCT_%d.wind_speed_hub = \
                      floris_power_%d.velocitiesTurbines')" % (i, i, i))
                self.driver.workflow.add('FPIdriver_%d' % i)
            else:
                self.driver.workflow.add(['floris_adjustCtCp_%d' % i, 'floris_windframe_%d' % i,
                                      'floris_wcent_wdiam_%d' % i, 'floris_overlap_%d' % i, 'floris_power_%d' % i])

        # add AEP calculations to workflow
        self.driver.workflow.add(['floris_AEP', 'floris_dist_const'])
        if optimize_position or optimize_yaw:
            # set up driver
            self.driver.iprint = 3
            self.driver.accuracy = 1.0e-12
            self.driver.maxiter = 100
            self.driver.add_objective('-floris_AEP.AEP')
            if optimize_position:
                self.driver.add_parameter('turbineX', low=7*126.4, high=np.sqrt(self.nTurbines)*7*126.4)
                self.driver.add_parameter('turbineY', low=7*126.4, high=np.sqrt(self.nTurbines)*7*126.4)
                self.driver.add_constraint('floris_dist_const.separation > 2*rotorDiameter[0]')
            if optimize_yaw:
                for direction in range(0, self.nDirections):
                    self.driver.add_parameter('yaw_%d' % direction, low=-30., high=30., scaler=1.)



class floris_assembly_opt(Assembly):
    """ Defines the connections between each Component used in the FLORIS model """

    # general input variables
    parameters = VarTree(FLORISParameters(), iotype='in')
    verbose = Bool(False, iotype='in', desc='verbosity of FLORIS, False is no output')
    include_yaw = Bool(False, iotyp='in', desc='optimize yaw for each wind direction, False keep the input yaw values')

    # Flow property variables
    wind_speed = Float(iotype='in', units='m/s', desc='free stream wind velocity')
    air_density = Float(iotype='in', units='kg/(m*m*m)', desc='air density in free stream')
    wind_direction = Float(iotype='in', units='deg', desc='overall wind direction for wind farm using deg. from E. ccw')

    # output
    power = Float(iotype='out', units='kW', desc='total windfarm power')

    # def __init__(self, turbineX, turbineY, yaw, resolution):
    def __init__(self, nTurbines, nDirections, resolution, include_yaw):

        super(floris_assembly_opt, self).__init__()

        self.nTurbines = nTurbines
        self.nDirections = nDirections
        self.resolution = resolution
        self.include_yaw = include_yaw

        # Explicitly size input arrays

        # wt_layout input variables
        self.add('rotorDiameter', Array(np.zeros(nTurbines), dtype='float', iotype='in', units='m', \
                                        desc='rotor diameters of all turbine'))
        self.add('axialInduction', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                                         desc='axial induction of all turbines'))
        self.add('Ct', Array(np.zeros(nTurbines), iotype='in', desc='Thrust coefficient for all turbines'))
        self.add('Cp', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                             desc='power coefficient for all turbines'))
        self.add('generator_efficiency', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                                               desc='generator efficiency of all turbines'))
        self.add('turbineX', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                                   desc='x positions of turbines in original ref. frame'))
        self.add('turbineY', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                                   desc='y positions of turbines in original ref. frame'))
        if include_yaw:
            self.add('yaw', Array(np.zeros(nTurbines*nDirections), iotype='in', dtype='float', \
                              desc='yaw of each turbine for each direction'))
        else:
            self.add('yaw', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                              desc='yaw of each turbine'))


        # Explicitly size output arrays

        # variables added to test individual components
        self.add('turbineXw', Array(np.zeros(nTurbines), iotype='out', units='m', \
                                    desc='X positions of turbines in the wind direction reference frame'))
        self.add('turbineYw', Array(np.zeros(nTurbines), iotype='out', units='m', \
                                    desc='Y positions of turbines in the wind direction reference frame'))
        self.add('wakeCentersYT', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m', \
                                        desc='centers of the wakes at each turbine'))
        self.add('wakeDiametersT', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m', \
                                         desc='diameters of each of the wake zones for each of the wakes at each turbine'))
        self.add('wakeOverlapTRel', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m', \
                                          desc='ratio of overlap area of each zone to rotor area'))

        # standard output
        self.add('velocitiesTurbines', Array(np.zeros(nTurbines), iotype='out', units='m/s', dtype='float'))
        self.add('wt_power', Array(np.zeros(nTurbines), iotype='out', units='kW', dtype='float'))

    def configure(self):

        # add driver so the workflow is not overwritten later
        # self.add('driver', SLSQPdriver())
        # self.driver.gradient_options.force_fd = True
        self.add('driver', pyOptDriver())
        self.driver.optimizer = 'SNOPT'
        # self.driver.pyopt_diff = True


        # add components to floris assembly
        F1 = self.add('floris_adjustCtCp', floris_adjustCtCp(nTurbines=self.nTurbines))
        F2 = self.add('floris_windframe', floris_windframe(nTurbines=self.nTurbines, resolution=self.resolution))
        F2.missing_deriv_policy = 'assume_zero'
        F3 = self.add('floris_wcent_wdiam', floris_wcent_wdiam(nTurbines=self.nTurbines))
        F4 = self.add('floris_overlap', floris_overlap(nTurbines=self.nTurbines))
        F4.missing_deriv_policy = 'assume_zero'
        F5 = self.add('floris_power', floris_power(nTurbines=self.nTurbines))


        self.driver.workflow.add(['floris_adjustCtCp', 'floris_windframe', 'floris_wcent_wdiam', 'floris_overlap', \
                                  'floris_power'])

        # connect inputs to components
        self.connect('parameters', ['floris_adjustCtCp.parameters', 'floris_wcent_wdiam.parameters', 'floris_power.parameters'])
        self.connect('verbose', ['floris_windframe.verbose', 'floris_wcent_wdiam.verbose', 'floris_power.verbose'])
        self.connect('turbineX', 'floris_windframe.turbineX')
        self.connect('turbineY', 'floris_windframe.turbineY')
        self.connect('rotorDiameter', ['floris_wcent_wdiam.rotorDiameter', 'floris_overlap.rotorDiameter', 'floris_power.rotorDiameter'])
        self.connect('axialInduction', 'floris_power.axialInduction')
        self.connect('Ct', 'floris_adjustCtCp.Ct_in')
        self.connect('Cp', 'floris_adjustCtCp.Cp_in')
        self.connect('generator_efficiency', 'floris_power.generator_efficiency')
        self.connect('yaw', ['floris_adjustCtCp.yaw', 'floris_wcent_wdiam.yaw'])
        self.connect('wind_speed', 'floris_power.wind_speed')
        self.connect('air_density', 'floris_power.air_density')
        self.connect('wind_direction', 'floris_windframe.wind_direction')

        # for satisfying the verbosity in windframe
        # self.connect('Cp', 'floris_windframe.Cp')
        # self.connect('Ct', 'floris_windframe.Ct')
        self.connect('yaw', 'floris_windframe.yaw')
        self.connect('floris_adjustCtCp.Ct_out', 'floris_windframe.Ct')
        self.connect('floris_adjustCtCp.Cp_out', 'floris_windframe.Cp')
        self.connect('wind_speed', 'floris_windframe.wind_speed')
        self.connect('axialInduction', 'floris_windframe.axialInduction')

        # ############### Connections between components ##################
        # connections from CtCp adjustment to others
        self.connect('floris_adjustCtCp.Ct_out', ['floris_wcent_wdiam.Ct', 'floris_power.Ct'])
        self.connect('floris_adjustCtCp.Cp_out', 'floris_power.Cp')


        # connections from floris_windframe to floris_wcent_wdiam
        self.connect("floris_windframe.turbineXw", "floris_wcent_wdiam.turbineXw")
        self.connect("floris_windframe.turbineYw", "floris_wcent_wdiam.turbineYw")

        # connections from floris_wcent_wdiam to floris_overlap
        self.connect("floris_wcent_wdiam.wakeCentersYT", "floris_overlap.wakeCentersYT")
        self.connect("floris_wcent_wdiam.wakeDiametersT", "floris_overlap.wakeDiametersT")

        # connections from floris_windframe to floris_overlap
        self.connect("floris_windframe.turbineXw", "floris_overlap.turbineXw")
        self.connect("floris_windframe.turbineYw", "floris_overlap.turbineYw")

        # connections from floris_windframe to floris_power
        self.connect('floris_windframe.turbineXw', 'floris_power.turbineXw')

        # connections from floris_overlap to floris_power
        self.connect("floris_overlap.wakeOverlapTRel", "floris_power.wakeOverlapTRel")
        # #################################################################

        # output connections
        self.connect("floris_power.velocitiesTurbines", "velocitiesTurbines")
        self.connect("floris_power.wt_power", "wt_power")
        self.connect("floris_power.power", "power")

        # outputs for testing only
        self.connect("floris_windframe.turbineXw", "turbineXw")
        self.connect("floris_windframe.turbineYw", "turbineYw")
        self.connect("floris_wcent_wdiam.wakeCentersYT", "wakeCentersYT")
        self.connect("floris_wcent_wdiam.wakeDiametersT", "wakeDiametersT")
        self.connect("floris_overlap.wakeOverlapTRel", "wakeOverlapTRel")

        # set up driver
        self.driver.iprint = 1
        self.driver.accuracy = 1.0e-12
        self.driver.maxiter = 100
        self.driver.add_objective('-floris_power.power')
        self.driver.add_parameter('turbineX', low=7*126.4, high=5*7*126.4)
        # self.driver.add_parameter('turbineY', low=7*126.4, high=5*7*126.4)
        # self.driver.add_parameter('yaw', low=-30., high=30., scaler=1)


class floris_assembly(Assembly):
    """ Defines the connections between each Component used in the FLORIS model """
    #
    # original input variables in Pieter's OpenMDAO stand-alone version of FLORIS
    parameters = VarTree(FLORISParameters(), iotype='in')
    verbose = Bool(False, iotype='in', desc='verbosity of FLORIS, False is no output')

    # input
    wind_direction = Float(iotype='in', units='deg', desc='wind direction to, using deg. ccw from east')

    # final output
    power = Float(iotype='out', units='kW', desc='total windfarm power')

    def __init__(self, nTurbines, resolution):
        super(floris_assembly, self).__init__()

        self.nTurbines = nTurbines
        self.resolution = resolution

        # Explicitly size input arrays

        # wt_layout input variables
        self.add('rotorDiameter', Array(np.zeros(nTurbines), dtype='float', iotype='in', units='m', \
                                        desc='rotor diameters of all turbine'))
        self.add('axialInduction', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                                         desc='axial induction of all turbines'))
        self.add('Ct', Array(np.zeros(nTurbines), iotype='in', desc='Thrust coefficient for all turbines'))
        self.add('Cp', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                             desc='power coefficient for all turbines'))
        self.add('generator_efficiency', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                                               desc='generator efficiency of all turbines'))
        self.add('turbineX', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                                   desc='x positions of turbines in original ref. frame'))
        self.add('turbineY', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                                   desc='y positions of turbines in original ref. frame'))
        self.add('yaw', Array(np.zeros(nTurbines), iotype='in', dtype='float', \
                              desc='yaw of each turbine'))

        # visualization variables
        self.add('ws_position', Array(np.zeros([resolution*resolution, 2]), iotype='in', units='m', desc='position where you want measurements in ref. frame'))


        # Explicitly size output arrays

        # variables added to test individual components
        self.add('turbineXw', Array(np.zeros(nTurbines), iotype='out', units='m', \
                                    desc='X positions of turbines in the wind direction reference frame'))
        self.add('turbineYw', Array(np.zeros(nTurbines), iotype='out', units='m', \
                                    desc='Y positions of turbines in the wind direction reference frame'))
        self.add('wakeCentersYT', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m', \
                                        desc='centers of the wakes at each turbine'))
        self.add('wakeDiametersT', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m', \
                                         desc='diameters of each of the wake zones for each of the wakes at each turbine'))
        self.add('wakeOverlapTRel', Array(np.zeros(nTurbines), dtype='float', iotype='out', units='m', \
                                          desc='ratio of overlap area of each zone to rotor area'))

        # standard output
        self.add('velocitiesTurbines', Array(np.zeros(nTurbines), iotype='out', units='m/s', dtype='float'))
        self.add('wt_power', Array(np.zeros(nTurbines), iotype='out', units='kW', dtype='float'))

        # visualization output
        self.add('ws_array', Array(np.zeros([resolution*resolution, 2]), iotype='out', units='m/s', desc='wind speed at measurement locations'))

    def configure(self):

        # add components to floris assembly
        self.add('floris_adjustCtCp', floris_adjustCtCp(self.nTurbines))
        self.add('floris_windframe', floris_windframe(self.nTurbines, self.resolution))
        self.add('floris_wcent_wdiam', floris_wcent_wdiam())
        self.add('floris_overlap', floris_overlap())
        self.add('floris_power', floris_power())

        # add driver to floris assembly
        self.driver.workflow.add(['floris_adjustCtCp', 'floris_windframe', 'floris_wcent_wdiam', 'floris_overlap', \
                                  'floris_power'])

        # added for gradient testing
        # self.add('driver', SLSQPdriver())
        # self.driver.iprint = 0
        # self.driver.add_objective('-floris_power.power')
        # self.driver.add_parameter('floris_windframe.turbineX', low=0., high=3000.)
        # self.driver.add_parameter('floris_windframe.turbineY', low=0., high=3000.)

        # connect inputs to components
        self.connect('parameters', ['floris_adjustCtCp', 'floris_wcent_wdiam.parameters', 'floris_power.parameters'])
        self.connect('verbose', ['floris_windframe.verbose', 'floris_wcent_wdiam.verbose', 'floris_power.verbose'])
        # self.connect('position', 'floris_windframe.position')
        self.connect('turbineX', 'floris_windframe.turbineX')
        self.connect('turbineY', 'floris_windframe.turbineY')
        self.connect('ws_position', 'floris_windframe.ws_position')
        self.connect('rotorDiameter', ['floris_wcent_wdiam.rotorDiameter', 'floris_overlap.rotorDiameter', 'floris_power.rotorDiameter'])
        self.connect('rotorArea', ['floris_overlap.rotorArea', 'floris_power.rotorArea'])
        self.connect('axialInduction', 'floris_power.axialInduction')
        self.connect('Cp', 'floris_adjustCtCp.Cp')
        self.connect('Ct', 'floris_adjustCtCp.Ct')
        self.connect('floris_adjustCtCp.Ct', ['floris_wcent_wdiam.Ct', 'floris_power.Ct'])
        self.connect('floris_adjustCtCp.Cp', 'floris_power.Cp')
        # self.connect('Ct', ['floris_wcent_wdiam.Ct', 'floris_power.Ct'])
        # self.connect('Cp', 'floris_power.Cp')
        self.connect('generator_efficiency', 'floris_power.generator_efficiency')
        # self.connect('yaw', ['floris_adjustCtCp.yaw', 'floris_wcent_wdiam.yaw', 'floris_power.yaw'])
        self.connect('yaw', 'floris_adjustCtCp.yaw')
        self.conmnect('floris_adjustCtCp.yaw', ['floris_wcent_wdiam.yaw', 'floris_power.yaw'])
        self.connect('wind_speed', 'floris_power.wind_speed')
        self.connect('air_density', 'floris_power.air_density')
        self.connect('wind_direction', 'floris_windframe.wind_direction')


        # for satisfying the verbosity in windframe
        self.connect('floris_adjustCtCp.Cp', 'floris_windframe.Cp')
        self.connect('floris_adjustCtCp.Ct', 'floris_windframe.Ct')
        # self.connect('Cp', 'floris_windframe.Cp')
        # self.connect('Ct', 'floris_windframe.Ct')
        # self.connect('yaw', 'floris_windframe.yaw')
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

