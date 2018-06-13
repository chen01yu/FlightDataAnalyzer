# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import operator
import re
import six

from collections import defaultdict
from copy import deepcopy
from math import ceil, copysign
from operator import itemgetter

from flightdatautilities import aircrafttables as at, units as ut
from flightdatautilities.geometry import midpoint, great_circle_distance__haversine

from analysis_engine.settings import (ACCEL_LAT_OFFSET_LIMIT,
                                      ACCEL_LON_OFFSET_LIMIT,
                                      ACCEL_NORM_OFFSET_LIMIT,
                                      AIRSPEED_THRESHOLD,
                                      BUMP_HALF_WIDTH,
                                      CLIMB_OR_DESCENT_MIN_DURATION,
                                      CONTROL_FORCE_THRESHOLD,
                                      GRAVITY_IMPERIAL,
                                      GRAVITY_METRIC,
                                      GROUNDSPEED_FOR_MOBILE,
                                      HOVER_MIN_DURATION,
                                      HYSTERESIS_FPALT,
                                      MIN_HEADING_CHANGE,
                                      NAME_VALUES_CONF,
                                      NAME_VALUES_ENGINE,
                                      NAME_VALUES_LEVER,
                                      NAME_VALUES_RANGES,
                                      HEADING_RATE_FOR_TAXI_TURNS,
                                      REVERSE_THRUST_EFFECTIVE_EPR,
                                      REVERSE_THRUST_EFFECTIVE_N1,
                                      SPOILER_DEPLOYED,
                                      VERTICAL_SPEED_FOR_LEVEL_FLIGHT)

from analysis_engine.node import (
    KeyPointValueNode, KPV, KTI, P, S, A, M, App, Section,
    aeroplane, aeroplane_only, helicopter, helicopter_only)

from analysis_engine.library import (ambiguous_runway,
                                     align,
                                     all_deps,
                                     all_of,
                                     any_of,
                                     bearings_and_distances,
                                     bump,
                                     closest_unmasked_value,
                                     clump_multistate,
                                     coreg,
                                     cycle_counter,
                                     cycle_finder,
                                     cycle_select,
                                     find_edges,
                                     find_edges_on_state_change,
                                     first_valid_parameter,
                                     first_valid_sample,
                                     hysteresis,
                                     ils_established,
                                     index_at_value,
                                     index_of_first_start,
                                     index_of_last_stop,
                                     integrate,
                                     is_index_within_slice,
                                     is_index_within_slices,
                                     lookup_table,
                                     nearest_neighbour_mask_repair,
                                     mask_inside_slices,
                                     mask_outside_slices,
                                     max_abs_value,
                                     max_continuous_unmasked,
                                     max_maintained_value,
                                     max_value,
                                     median_value,
                                     min_value,
                                     most_common_value,
                                     moving_average,
                                     repair_mask,
                                     np_ma_masked_zeros_like,
                                     np_ma_zeros_like,
                                     peak_curvature,
                                     rate_of_change_array,
                                     runs_of_ones,
                                     runway_deviation,
                                     runway_distance_from_end,
                                     runway_heading,
                                     second_window,
                                     shift_slice,
                                     shift_slices,
                                     slice_duration,
                                     slice_midpoint,
                                     slice_samples,
                                     slices_above,
                                     slices_and_not,
                                     slices_below,
                                     slices_between,
                                     slices_duration,
                                     slices_from_ktis,
                                     slices_from_to,
                                     slices_not,
                                     slices_overlap,
                                     slices_or,
                                     slices_and,
                                     slices_overlap_merge,
                                     slices_remove_overlaps,
                                     slices_remove_small_slices,
                                     slices_remove_small_gaps,
                                     string_array_to_mapped_array,
                                     trim_slices,
                                     level_off_index,
                                     valid_slices_within_array,
                                     value_at_index,
                                     vstack_params_where_state,
                                     vstack_params)


##############################################################################
# Superclasses


class FlapOrConfigurationMaxOrMin(object):
    '''
    Abstract superclass.
    '''

    @staticmethod
    def flap_or_conf_max_or_min(conflap, parameter, function, scope=None,
                                include_zero=False):
        '''
        Generic flap and configuration key point value search process.

        This will determine key point values for a parameter based on the
        provided flap or configuration parameter. The function argument
        determines what operation should be applied and can be one of the many
        library functions, e.g. ``max_value`` or ``min_value``.

        The ``scope`` argument is used to restrict the period that should be
        monitored which is essential for minimum speed checks.

        Setting the ``include_zero`` flag will detect key point values where
        the aircraft configuration is clean.

        Note: This routine does not actually create the key point values.

        :param conflap: flap or configuration, restricted to detent settings.
        :type conflap: parameter
        :param parameter: parameter to be measured at flap/conf detent.
        :type parameter: Parameter
        :param function: function to be applied to the parameter values.
        :type function: function
        :param scope: periods to restrict the search to. (optional)
        :type scope: list of slices
        :param include_zero: include zero flap detents. (default: false)
        :type include_zero: boolean
        :returns: a tuple of data to create KPVs from.
        :rtype: tuple
        '''
        assert isinstance(conflap, M), 'Expected a multi-state.'

        if scope == []:
            return []  # can't have an event if the scope is empty.

        if scope:
            scope_array = np_ma_masked_zeros_like(parameter.array)
            for valid in scope:
                a = int(valid.slice.start or 0)
                b = int(valid.slice.stop or len(scope_array)) + 1
                scope_array.mask[a:b] = False

        data = []

        for detent in conflap.values_mapping.values():

            if np.ma.is_masked(detent):
                continue
            if detent in ('0', 'Lever 0') and not include_zero:
                continue

            array = np.ma.copy(parameter.array)
            array.mask = np.ma.mask_or(parameter.array.mask, conflap.array.mask)
            array[conflap.array != detent] = np.ma.masked
            if scope:
                array.mask = np.ma.mask_or(array.mask, scope_array.mask)

            # TODO: Check logical or is sensible for all values. (Probably fine
            #       as airspeed will always be higher than max flap setting!)
            index, value = function(array)

            # Check we have a result to record. Note that most flap settings
            # will not be used in the climb, hence this is normal operation.
            if not index or not value:
                continue

            data.append((index, value, detent))

        return data


##############################################################################
# Helpers


def thrust_reversers_working(landing, pwr, tr, threshold):
    '''
    Thrust reversers are deployed and maximum engine power is over
    REVERSE_THRUST_EFFECTIVE for EPR or N1 (nominally 65% N1, 1.25% EPR).
    '''
    high_power = np.ma.masked_less(pwr.array, threshold)
    high_power_slices = np.ma.clump_unmasked(high_power)
    high_power_landing_slices = slices_and(high_power_slices, [landing.slice])
    return clump_multistate(tr.array, 'Deployed', high_power_landing_slices)


def remove_bump(airborne):
    '''
    This removes the takeoff and landing bump periods from the airborne phase
    to avoid overlap of KPV periods.
    '''
    removed = []
    hz = airborne.frequency
    for air in airborne:
        new_slice = slice(air.slice.start + BUMP_HALF_WIDTH * hz,
                          air.slice.stop - BUMP_HALF_WIDTH * hz)
        removed.append(new_slice)
    return removed

##############################################################################
# Acceleration


########################################
# Acceleration: Lateral


class AccelerationLateralMax(KeyPointValueNode):
    '''
    Maximum recorded lateral acceleration.
    '''

    units = ut.G

    @classmethod
    def can_operate(cls, available):
        '''
        This KPV has no inherent flight phase associated with it, but we can
        reasonably say that we are not interested in anything while the aircraft is
        stationary.
        '''
        return 'Acceleration Lateral Offset Removed' in available
        

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               gnd_spd=P('Groundspeed')):

        if gnd_spd:
            self.create_kpvs_within_slices(
                acc_lat.array,
                gnd_spd.slices_above(5),
                max_abs_value,
            )
        else:
            self.create_kpv(*max_abs_value(acc_lat.array))


class AccelerationLateralAtTouchdown(KeyPointValueNode):
    '''
    Lateral acceleration recorded at touchdown.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               touchdowns=KTI('Touchdown')):

        for touchdown in touchdowns:
            self.create_kpv(*bump(acc_lat, touchdown))


class AccelerationLateralDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum lateral acceleration recorded during takeoff.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               takeoff_rolls=S('Takeoff Roll')):

        self.create_kpvs_within_slices(
            acc_lat.array,
            takeoff_rolls,
            max_abs_value,
        )


class AccelerationLateralDuringLandingMax(KeyPointValueNode):
    '''
    Maximum lateral acceleration recorded during landing roll.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               landing_rolls=S('Landing Roll'),
               ldg_rwy=A('FDR Landing Runway')):

        if ambiguous_runway(ldg_rwy):
            return
        self.create_kpv_from_slices(
            acc_lat.array,
            landing_rolls,
            max_abs_value,
        )


class AccelerationLateralWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum lateral acceleration recorded while airborne.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               airborne=S('Airborne')):

        self.create_kpv_from_slices(
            acc_lat.array,
            airborne,
            max_abs_value,
        )


class AccelerationLateralWhileTaxiingStraightMax(KeyPointValueNode):
    '''
    Maximum lateral acceleration (smoothed) while taxiing straight.

    We use the smoothed lateral acceleration which removes spikes from signals due
    to uneven surfaces.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral Smoothed'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):
        '''
        Lateral acceleration while not turning is rarely an issue, so we compute
        only one KPV for taxi out and one for taxi in. The straight sections are
        identified by masking the turning phases and then testing the resulting
        data.
        '''
        acc_lat_array = mask_inside_slices(acc_lat.array, turns.get_slices())
        self.create_kpvs_within_slices(acc_lat_array, taxiing, max_abs_value)


class AccelerationLateralWhileTaxiingTurnMax(KeyPointValueNode):
    '''
    Maximum lateral acceleration (smoothed) while turning during taxi (taxi out and taxi in).

    We use the smoothed lateral acceleration which removes spikes from signals due
    to uneven surfaces.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral Smoothed'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):
        '''
        Lateral acceleration while taxiing normally occurs in turns, and leads to
        wear on the undercarriage and discomfort for passengers. In extremis this
        can lead to taxiway excursions. Lateral acceleration is used in preference
        to groundspeed as this parameter is available on older aircraft and is
        directly related to comfort.
        '''
        acc_lat_array = mask_outside_slices(acc_lat.array, turns.get_slices())
        self.create_kpvs_within_slices(acc_lat_array, taxiing, max_abs_value)


class AccelerationLateralInTurnDuringTaxiInMax(KeyPointValueNode):
    '''
    Maximum lateral acceleration (smoothed) while turning during taxi in.

    We use the smoothed lateral acceleration which removes spikes from signals due
    to uneven surfaces.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral Smoothed'),
               taxiing=S('Taxi In'),
               turns=S('Turning On Ground')):

        acc_lat_array = mask_outside_slices(acc_lat.array, turns.get_slices())
        self.create_kpvs_within_slices(acc_lat_array, taxiing, max_abs_value)


class AccelerationLateralInTurnDuringTaxiOutMax(KeyPointValueNode):
    '''
    Maximum lateral acceleration (smoothed) while turning during taxi in.

    We use the smoothed lateral acceleration which removes spikes from signals due
    to uneven surfaces.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral Smoothed'),
               taxiing=S('Taxi Out'),
               turns=S('Turning On Ground')):
        '''
        Lateral acceleration while taxiing normally occurs in turns, and leads to
        wear on the undercarriage and discomfort for passengers. In extremis this
        can lead to taxiway excursions. Lateral acceleration is used in preference
        to groundspeed as this parameter is available on older aircraft and is
        directly related to comfort.
    
        We use the smoothed lateral acceleration which removes spikes from signals due
        to uneven surfaces.
        '''
        acc_lat_array = mask_outside_slices(acc_lat.array, turns.get_slices())
        self.create_kpvs_within_slices(acc_lat_array, taxiing, max_abs_value)


class AccelerationLateralOffset(KeyPointValueNode):
    '''
    Lateral accelerometer datum offset during taxi in straight line.
    '''

    units = ut.G

    def derive(self,
               acc_lat=P('Acceleration Lateral'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):
        '''
        This KPV computes the lateral accelerometer datum offset, as for
        AccelerationNormalOffset. The more complex slicing statement ensures we
        only accumulate error estimates when taxiing in a straight line.
        '''
        straights = slices_and([s.slice for s in list(taxiing)],
            slices_not([s.slice for s in list(turns)]),)
    
        '''
        Get the unmasked data within the taxiing in a straight line phase and
        compute the average for this section(s) if there are enough samples.
        '''    
        unmasked_data = []
        for straight in straights:
            unmasked_data.extend(np.ma.compressed(acc_lat.array[straight]))
        if len(unmasked_data) > 20:
            delta = np.sum(unmasked_data) / float(len(unmasked_data))
            self.create_kpv(0, delta)
            if abs(delta) > ACCEL_LAT_OFFSET_LIMIT:
                self.warning("Acceleration Lateral offset '%s' greater than limit '%s'",
                             delta, ACCEL_LAT_OFFSET_LIMIT)            
        

class AccelerationLateralFor5SecMax(KeyPointValueNode):
    '''
    Maximum lateral acceleration recorded for the duration of 5 seconds.
    '''

    @classmethod
    def can_operate(cls, available, frame=A('Frame')):
        # The timing interval is incompatible with the 787 data rate, hence the current restriction.
        if frame and frame.value.startswith('787'):
            return False
        else:
            return all_deps(cls, available)

    units = ut.G

    def derive(self, accel_lat=P('Acceleration Lateral Offset Removed')):
        accel_lat_20 = second_window(accel_lat.array, accel_lat.frequency, 5, extend_window=True)
        self.create_kpv(*max_abs_value(accel_lat_20))


########################################
# Acceleration: Longitudinal


class AccelerationLongitudinalOffset(KeyPointValueNode):
    '''
    Longitudinal accelerometer datum offset.
    
    We use all the taxiing phases and assume that
    the accelerations and decelerations will roughly balance out over the
    duration of the taxi phase.
    '''
    
    units = ut.G

    def derive(self,
               acc_lon=P('Acceleration Longitudinal'),
               mobiles=S('Mobile'),
               fasts=S('Fast')):
        '''
        This KPV computes the longitudinal accelerometer datum offset, as for
        AccelerationNormalOffset. We use all the taxiing phase and assume that
        the accelerations and decelerations will roughly balance out over the
        duration of the taxi phase.
   
        Note: using mobile sections which are not Fast in place of taxiing in
        order to aviod circular dependancy with Taxiing, Rejected Takeoff and
        Acceleration Longitudinal Offset Removed
        '''
    
        taxis = slices_and_not(mobiles.get_slices(), fasts.get_slices())
        
        '''
        Get the unmasked data within the taxis slices provided and compute
        the average for this section(s) if there are enough samples.
        '''        
        unmasked_data = []
        for taxi in taxis:
            unmasked_data.extend(np.ma.compressed(acc_lon.array[taxi]))
        if len(unmasked_data) > 20:
            delta = np.sum(unmasked_data) / float(len(unmasked_data))        
            self.create_kpv(0, delta)
            if abs(delta) > ACCEL_LON_OFFSET_LIMIT:
                self.warning("Acceleration Longitudinal offset '%s' greater than limit '%s'",
                             delta, ACCEL_LON_OFFSET_LIMIT)            


class AccelerationLongitudinalDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum longitudinal acceleration recorded during takeoff.
    '''

    units = ut.G

    def derive(self,
               acc_lon=P('Acceleration Longitudinal Offset Removed'),
               takeoff=S('Takeoff')):
        '''
        This may be of interest where takeoff performance is an issue, though not
        normally monitored as a safety event.
        '''
        self.create_kpv_from_slices(acc_lon.array, takeoff, max_value)


class AccelerationLongitudinalDuringLandingMin(KeyPointValueNode):
    '''
    Minimum longitudinal acceleration recorded during landing.
    '''

    units = ut.G

    def derive(self,
               acc_lon=P('Acceleration Longitudinal Offset Removed'),
               landing=S('Landing')):

        self.create_kpv_from_slices(acc_lon.array, landing, min_value)


class AccelerationLongitudinalWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum longitudinal acceleration recorded while airborne.
    '''

    units = ut.G

    def derive(self,
               acc_long=P('Acceleration Longitudinal Offset Removed'),
               airborne=S('Airborne')):

        self.create_kpv_from_slices(
            acc_long.array,
            airborne,
            max_abs_value,
        )

########################################
# Acceleration: Normal


class AccelerationNormalMax(KeyPointValueNode):
    '''
    Maximum normal acceleration recorded while the aircraft was moving.
    '''

    units = ut.G

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               mobile=S('Mobile')):

        self.create_kpv_from_slices(acc_norm.array, mobile, max_value)


class AccelerationNormal20FtTo5FtMax(KeyPointValueNode):
    '''
    Maximum normal acceleration recorded between 20ft and 5ft AAL.
    '''

    can_operate = aeroplane_only

    units = ut.G

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            acc_norm.array,
            alt_aal.slices_from_to(20, 5),
            max_value,
        )


class AccelerationNormalWithFlapUpWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum normal acceleration value with flaps retracted.

    Note that this KPV uses the flap lever angle, not the flap surface angle.
    '''

    units = ut.G

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) and \
            all_of(('Acceleration Normal Offset Removed', 'Airborne'), available)

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        if '0' in flap.array.state:
            retracted = flap.array == '0'
        acc_flap_up = np.ma.masked_where(~retracted, acc_norm.array)
        self.create_kpv_from_slices(acc_flap_up, remove_bump(airborne), max_value)


class AccelerationNormalWithFlapUpWhileAirborneMin(KeyPointValueNode):
    '''
    Minimum normal acceleration value with flaps retracted.

    Note that this KPV uses the flap lever angle, not the flap surface angle.
    '''

    units = ut.G

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) and \
            all_of(('Acceleration Normal Offset Removed', 'Airborne'), available)

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        acc_flap_up = np.ma.masked_where(~retracted, acc_norm.array)
        self.create_kpv_from_slices(acc_flap_up, remove_bump(airborne),
                                    min_value)


class AccelerationNormalWithFlapDownWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum normal acceleration value with flaps extended.

    Note that this KPV uses the flap lever angle, not the flap surface angle.
    '''

    units = ut.G

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) and \
            all_of(('Acceleration Normal Offset Removed', 'Airborne'), available)

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        acc_flap_dn = np.ma.masked_where(retracted, acc_norm.array)
        self.create_kpv_from_slices(acc_flap_dn, remove_bump(airborne), max_value)


class AccelerationNormalWithFlapDownWhileAirborneMin(KeyPointValueNode):
    '''
    Minimum normal acceleration value with flaps extended.

    Note that this KPV uses the flap lever angle, not the flap surface angle.
    '''

    units = ut.G

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) and \
            all_of(('Acceleration Normal Offset Removed', 'Airborne'), available)

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        acc_flap_dn = np.ma.masked_where(retracted, acc_norm.array)
        self.create_kpv_from_slices(acc_flap_dn, remove_bump(airborne), min_value)


class AccelerationNormalAtLiftoff(KeyPointValueNode):
    '''
    This is a measure of the normal acceleration at the point of liftoff, and
    is related to the pitch rate at takeoff.
    '''

    units = ut.G

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               liftoffs=KTI('Liftoff')):

        for liftoff in liftoffs:
            self.create_kpv(*bump(acc_norm, liftoff))


class AccelerationNormalAtTouchdown(KeyPointValueNode):
    '''
    Acceleration normal recorded at touchdown.
    '''

    units = ut.G

    @classmethod
    def can_operate(cls, available):
        return all_of(('Acceleration Normal Offset Removed', 'Touchdown',),
                      available)

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               touchdowns=KTI('Touchdown'),
               touch_and_go=KTI('Touch And Go')):
        tdwns = touchdowns
        if touch_and_go:
            tdwns += touch_and_go
        for touchdown in tdwns:
                self.create_kpv(*bump(acc_norm, touchdown))


class LoadFactorThresholdAtTouchdown(KeyPointValueNode):
    '''
    A Boeing 767/757/737 specific KPV, which returns the difference between 
    acceleration normal at touchdown and load factor threshold based on roll 
    and weight. A positive value indicates the amount over the load factor 
    threshold and therefore a hard landing which needs maintenance action.
    
    MM 05-51-01 HARD LANDING OR OVERWEIGHT/HARD LANDING OR HIGH DRAG OR HIGH 
    SIDELOAD LANDING CONDITION - MAINTENANCE PRACTICES (CONDITIONAL INSPECTION)
    
    N.B. 737 and 767 manuals used to implement this algorithm. 
    No 757 publication issued, so based on 767 MM graphs.
    '''
    units = ut.G
    
    @classmethod
    def get_landing_weight(cls, series=None, model=None, mods=None):
        '''
        From the tamplate LMX000 (Maint.) Landing weight high at touchdown
        Series       Model          Modification                     Weight(KG)
        B757-200     (F),(PF)       MOD Aircraft Line Numbers 1-209       89992
        B757-200     (PCF)          MOD Aircraft Line Numbers 1-209       89811
        B757-200     (F),(PF),(PCF) None                                  95254
        B757-200     None           MOD Aircraft Line Numbers 210         95254
                                    and above
        B757-200     None           None                                  89992
        B757-300     None           None                                 101605
        B767-200     None           Post TC                              136078
        B767-25E     None           Freighter Conversion                 126098
        B767-200     None           Freighter Conversion                 128367
        B767-200     None           None                                 123377
        B767-300     None           Freighter Conversion                 147871
        B767-300     None           None                                 145149
        B767-400     (ER)           None                                 158757
        Return the weight in KG when match is found else None
        '''
        re_series = re.match(r'^B7[365]7-[2346789][05][0E]', series)
        if not re_series and series != 'B737-MAX-8':
            return None
        re_model = re.search(r'\((F|PF|PCF|ER)\)$', model)
        model_ending = re_model.group() if re_model else None

        if 'B757-200' in series and model_ending in ['(F)', '(PF)'] and \
           'MOD Aircraft Line Numbers 1-209' in mods:
            return 89992
        elif 'B757-200' in series and model_ending in ['(PCF)',] and \
             'MOD Aircraft Line Numbers 1-209' in mods:
            return 89811
        elif 'B757-200' in series and model_ending in ['(F)', '(PF)', '(PCF)']:
            return 95254
        elif 'B757-200' in series and \
             'MOD Aircraft Line Numbers 210 and above' in mods:
            return 95254
        elif 'B757-200' in series:
            return 89992
        elif 'B757-300' in series:
            return 101605
        elif 'B767-200' in series and 'Post TC' in mods:
            return 136078
        elif 'B767-25E' in series and 'Freighter Conversion' in mods:
            return 126098
        elif 'B767-200' in series and 'Freighter Conversion' in mods:
            return 128367
        elif 'B767-200' in series:
            return 123377
        elif 'B767-300' in series and 'Freighter Conversion' in mods:
            return 147871
        elif 'B767-300' in series:
            return 145149
        elif 'B767-400' in series and model_ending in ['(ER)',]:
            return 158757
        elif 'B737-600' in series:
            return 55205
        elif 'B737-700' in series and 'Increased Gross Weight' in mods:
            return 61389
        elif 'B737-700' in series:
            return 59190
        elif 'B737-800' in series and 'Increased Gross Weight' in mods:
            return 67025
        elif 'B737-800' in series:
            return 65970
        elif 'B737-900' in series and model_ending in ['(ER)']:
            return 72064
        elif 'B737-900' in series:
            return 67482
        elif 'B737-MAX-8' in series:
            return 70003
        else:
            return None

    # This KPV is specific to 767 aircraft
    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'),
                    mods=A('Modifications')):
        ac_weight = cls.get_landing_weight(series.value, model.value,
                                           mods.value)
        return ac_weight and all_deps(cls, available)

    def derive(self,
               land_vert_acc=KPV('Acceleration Normal At Touchdown'),
               roll=P('Roll'),
               tdwns=KTI('Touchdown'),
               gw_kpv=KPV('Gross Weight At Touchdown'),
               gw = P('Gross Weight'),
               model=A('Model'),
               series=A('Series'),
               mods=A('Modifications'),
               touch_and_go=KTI('Touch And Go')):
        
        mlw = self.get_landing_weight(series.value, model.value, mods.value)
        if not mlw:
            self.warning("Cannot determine landing weight. series:'%s' "
                         "model:'%s' mods:'%s'" % series.value, model.value,
                         mods.value)
            return
        # roll_repaired = repair_mask(roll.array, frequency=roll.frequency,
                                    # extrapolate=True, repair_duration=6)
        ac_type = series.value[:4]
        if ac_type == 'B767':
            weight_threshold = mlw + 1133.981  # B767 manual: 2500LB --> 1133.981KG
        elif ac_type == 'B757':
            weight_threshold = mlw + 1133.981  # Based on B767 manual
        elif ac_type == 'B737':
            weight_threshold = mlw + 454.0 # B737 manual states 1000lb = 454kg.
        
        freq_8hz = land_vert_acc.frequency == 8.0
        freq_16hz = land_vert_acc.frequency == 16.0
        for idx, tdwn in enumerate(tdwns+touch_and_go):
            # Find the maximum roll in the second prior to the touchdown.
            ratio = roll.frequency / self.frequency
            scope = slice((tdwn.index - 1) * ratio, 
                                      (tdwn.index * ratio) + 1)
            roll_tdwn = np.max(abs(roll.array[scope]))

            gw_value = [k.value for k in gw_kpv if k.index == tdwn.index]
            if not gw_value:
                self.warning("No 'Gross Weight At Touchdown' KPV, using "
                             "'Gross Weight' parameter at this touchdown "
                             "index.")
                weight = value_at_index(gw.array, tdwn.index)
            else:
                weight = gw_value[0]

            overweight = weight > weight_threshold
            
            if ac_type in ['B757', 'B767']:
                if not overweight and freq_16hz:
                    ld_factor_grph = np.ma.append(np.array([1.90, 1.90]),
                                                  np.linspace(1.90, 1.45, 5))
                elif not overweight and freq_8hz:
                    ld_factor_grph = np.ma.append(np.array([1.80, 1.80]),
                                                  np.linspace(1.80, 1.40, 5))
                elif overweight and freq_16hz:
                    ld_factor_grph = np.ma.append(np.array([1.55,]),
                                                  np.linspace(1.55, 1.29, 6))
                elif overweight and freq_8hz:
                    ld_factor_grph = np.ma.append(np.array([1.50,]),
                                                  np.linspace(1.50, 1.25, 6))
            elif ac_type in ['B737']:
                if not overweight and freq_16hz:
                    ld_factor_grph = np.ma.append(np.array([2.20, 2.20]),
                                                  np.linspace(2.20, 1.60, 5))
                elif not overweight and freq_8hz:
                    ld_factor_grph = np.ma.append(np.array([2.10, 2.10]),
                                                  np.linspace(2.10, 1.55, 5))
                elif overweight and freq_16hz:
                    ld_factor_grph = np.ma.append(np.array([1.85,]),
                                                  np.linspace(1.85, 1.43, 6))
                elif overweight and freq_8hz:
                    ld_factor_grph = np.ma.append(np.array([1.80,]),
                                                  np.linspace(1.80, 1.40, 6))
            else:
                continue

            # Use roll_tdwn as the index for ld_factor_grph
            load_factor = value_at_index(ld_factor_grph, roll_tdwn,
                                         interpolate=True)
            # At 6 deg the graph drops to zero. 
            if roll_tdwn > 6.0:
                load_factor = 0.0
                
            self.create_kpv(tdwn.index, load_factor)


class AccelerationNormalMinusLoadFactorThresholdAtTouchdown(KeyPointValueNode):
    '''
    A Boeing 767/757/737 specific KPV, which returns the difference between 
    acceleration normal at touchdown and load factor threshold based on roll 
    and weight. A positive value indicates the amount over the load factor 
    threshold and therefore a hard landing which needs maintenance action.
    
    MM 05-51-01 HARD LANDING OR OVERWEIGHT/HARD LANDING OR HIGH DRAG OR HIGH 
    SIDELOAD LANDING CONDITION - MAINTENANCE PRACTICES (CONDITIONAL INSPECTION)
    
    N.B. 737 and 767 manuals used to implement this algorithm. 
    No 757 publication issued, so based on 767 MM graphs.
    '''
    units = ut.G

    # This KPV is specific to 767 aircraft
    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'),
                    mods=A('Modifications')):
        return all_deps(cls, available)

    def derive(self,
               land_vert_acc=KPV('Acceleration Normal At Touchdown'),
               load_factors=KPV('Load Factor Threshold At Touchdown')):
        
        for idx, load_factor in enumerate(load_factors):
            delta = land_vert_acc[idx].value - load_factor.value
            self.create_kpv(load_factor.index, delta)


class AccelerationNormalLiftoffTo35FtMax(KeyPointValueNode):
    '''
    Maximum normal acceleration recorded between liftoff and 35ft AAL.
    '''

    can_operate = aeroplane_only

    units = ut.G

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               liftoffs=S('Liftoff'),
               takeoffs=S('Takeoff')):

        slices = []
        for liftoff in liftoffs:
            takeoff = takeoffs.get(containing_index=liftoff.index)
            if not takeoff:
                self.warning('Liftoff not found within Takeoff phase. '
                             'Altitude data may be missing/invalid at Takeoff.')
                continue
            slices.append(slice(liftoff.index, takeoff[0].stop_edge))
        self.create_kpvs_within_slices(acc_norm.array, slices, max_value)


class AccelerationNormalOffset(KeyPointValueNode):
    '''
    Normal acceleration datum offset.

    This allows for offsets that are sometimes found in these sensors which remain in service
    although outside the permitted accuracy of the signal.
    '''

    units = ut.G

    @classmethod
    def can_operate(cls, available):
        return all_of(('Acceleration Normal', 'Taxiing'), available)
    
    def derive(self,
               acc_norm=P('Acceleration Normal'),
               taxiing = S('Taxiing')):
        
        '''
        Get the unmasked data within the taxiing slices provided and compute
        the average for this section(s) if there are enough samples.
        '''        
        unmasked_data = []
        for taxi in taxiing.get_slices():
            unmasked_data.extend(np.ma.compressed(acc_norm.array[taxi]))
        if len(unmasked_data) > 20:
            delta = np.sum(unmasked_data) / float(len(unmasked_data)) - 1.0
            self.create_kpv(0, delta + 1.0)
            if abs(delta) > ACCEL_NORM_OFFSET_LIMIT:
                self.warning("Acceleration Normal offset '%s' greater than limit '%s'",
                             delta, ACCEL_NORM_OFFSET_LIMIT)             


class AccelerationNormalWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum recorded normal acceleration while airborne.
    '''

    units = ut.G

    def derive(self, accel_norm=P('Acceleration Normal Offset Removed'),
               airborne=S('Airborne')):
        self.create_kpvs_within_slices(
            accel_norm.array,
            airborne.get_slices(),
            max_value)


class AccelerationNormalWhileAirborneMin(KeyPointValueNode):
    '''
    Minimum recorded normal acceleration while airborne.
    '''

    units = ut.G

    def derive(self, accel_norm=P('Acceleration Normal Offset Removed'),
               airborne=S('Airborne')):
        self.create_kpvs_within_slices(
            accel_norm.array,
            airborne.get_slices(),
            min_value
        )


##############################################################################
# Airspeed


########################################
# Airspeed: General


class AirspeedMax(KeyPointValueNode):
    '''
    Maximum recorded airspeed.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(air_spd.array, airborne, max_value)


class AirspeedAt8000FtDescending(KeyPointValueNode):
    '''
    Airspeed recorded when crossing 8000ft STD during descent. 
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std_desc=S('Altitude When Descending')):
        '''
        Refactor to be a formatted name node if multiple Airspeed At Altitude
        KPVs are required. Could depend on either Altitude When Climbing or
        Altitude When Descending, but the assumption is that we'll have both.
        '''
        self.create_kpvs_at_ktis(air_spd.array,
                                 alt_std_desc.get(name='8000 Ft Descending'))


class AirspeedDuringCruiseMax(KeyPointValueNode):
    '''
    Maximum recorded airspeed during cruise.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               cruises=S('Cruise')):

        self.create_kpvs_within_slices(air_spd.array, cruises, max_value)


class AirspeedDuringCruiseMin(KeyPointValueNode):
    '''
    Minimum recorded airspeed during cruise.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               cruises=S('Cruise')):

        self.create_kpvs_within_slices(air_spd.array, cruises, min_value)


class AirspeedGustsDuringFinalApproach(KeyPointValueNode):
    '''
    Airspeed variation during final approach - between 30ft and 10ft radio.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               gnd_spd=P('Groundspeed'),
               alt_rad=P('Altitude Radio'),
               airborne=S('Airborne')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        Excursions - Landing (Lateral). Gusts during flare/final approach. This
        is tricky. Try Speed variation >15kt 30RA to 10RA. KPV looks at peak to
        peak values to get change in airspeed. Event uses interpolated RALT
        samples and looks at the airspeed samples that fall between RALT = 30ft
        and 10ft. DW suggested that the airspeed samples should also be
        interpolated in order to be able to estimate airspeed as to close to the
        ends of the RALT range as possible.
        '''
        _, fin_apps = slices_from_to(alt_rad.array, 30, 10)
        descents = slices_and(airborne.get_slices(), fin_apps)
        for descent in descents:
            # Ensure we encompass the range of interest.
            scope = slice(descent.start - 5, descent.stop + 5)
            # We'd like to use groundspeed to compute the wind gust, but
            # variations in airspeed are a suitable backstop.
            if gnd_spd:
                headwind = air_spd.array[scope] - gnd_spd.array[scope]
            else:
                headwind = air_spd.array[scope] - air_spd.array[scope][0]
            # Precise indexing is used as this is only a short segment. Note
            # that the _idx values are floating point interpolations of the
            # radio altimeter signal, and the headwind array is also
            # interpolated.
            idx_start = index_at_value(alt_rad.array, 30.0, scope)
            idx_stop = index_at_value(alt_rad.array, 10.0, scope)

            # This condition can arise in some corrupt data cases, or for a
            # go-around with a minimum between 30ft and 10ft.
            if idx_start is None or idx_stop is None:
                continue

            new_app = shift_slice(descent, -scope.start)
            if new_app is None:
                continue  # not enough data worthy of a slice

            peak = max_value(headwind, new_app,
                             start_edge=idx_start - scope.start,
                             stop_edge=idx_stop - scope.start)
            trough = min_value(headwind, new_app,
                               start_edge=idx_start - scope.start,
                               stop_edge=idx_stop - scope.start)
            if peak.value and trough.value:
                value = peak.value - trough.value
                index = ((peak.index + trough.index) / 2.0) + scope.start
                self.create_kpv(index, value)


########################################
# Airspeed: Climbing


class AirspeedAtLiftoff(KeyPointValueNode):
    'Airspeed at liftoff.'

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               liftoffs=KTI('Liftoff')):
        '''
        A 'Tailwind At Liftoff' KPV would complement this KPV when used for 'Speed
        high at takeoff' events.
        '''
        self.create_kpvs_at_ktis(air_spd.array, liftoffs)


class AirspeedAt35FtDuringTakeoff(KeyPointValueNode):
    '''
    Airspeed at 35ft AAL during takeoff.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               takeoffs=S('Takeoff')):

        for takeoff in takeoffs:
            index = takeoff.stop_edge  # Takeoff ends at 35ft!
            value = value_at_index(air_spd.array, index)
            self.create_kpv(index, value)


class Airspeed35To1000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed between 35ft and 1000ft AAL during initial climb.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               initial_climb=S('Initial Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 35, 1000)
        alt_climb_sections = valid_slices_within_array(alt_band, initial_climb)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_climb_sections,
            max_value)


class Airspeed35To1000FtMin(KeyPointValueNode):
    '''
    Minimum airspeed between 35ft and 1000ft AAL during initial climb.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               initial_climb=S('Initial Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 35, 1000)
        alt_climb_sections = valid_slices_within_array(alt_band, initial_climb)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_climb_sections,
            min_value,
        )


class Airspeed1000To5000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed between 1000ft and 5000ft AAL during climb.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               climbs=S('Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 1000, 5000)
        alt_climb_sections = valid_slices_within_array(alt_band, climbs)
        self.create_kpvs_within_slices(air_spd.array, alt_climb_sections,
                                       max_value)

    
class Airspeed5000To8000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed between 5000ft AAL and 8000ft STD (smoothed) during climb.
    '''
    units = ut.KT

    def derive(self, air_spd=P('Airspeed'), 
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_std=P('Altitude STD Smoothed'),
               climbs=S('Climb')):
        for climb in climbs:
            aal = np.ma.clump_unmasked(np.ma.masked_less(
                alt_aal.array[climb.slice], 5000.0))
            std = np.ma.clump_unmasked(np.ma.masked_greater(
                alt_std.array[climb.slice], 8000.0))
            scope = shift_slices(slices_and(aal, std), climb.slice.start)
            self.create_kpv_from_slices(air_spd.array, scope, max_value)


class Airspeed5000To10000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed between 5000ft AAL and 10000ft STD (smoothed) during climb.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_std=P('Altitude STD Smoothed'),
               climbs=S('Climb')):
        '''
        Airspeed from 5000ft above the airfield to a pressure altitude of 10000ft.
        As we are only interested in the climbing phase, this is used as the
        normal slices_from_to will not work with two parameters.
        '''
        for climb in climbs:
            aal = np.ma.clump_unmasked(
                np.ma.masked_less(alt_aal.array[climb.slice], 5000.0))
            std = np.ma.clump_unmasked(np.ma.masked_greater(
                alt_std.array[climb.slice], 10000.0))
            scope = shift_slices(slices_and(aal, std), climb.slice.start)
            self.create_kpv_from_slices(air_spd.array, scope, max_value)


class Airspeed1000To8000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed between 1000ft AAL and 8000ft STD (smoothed) during climb.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_std=P('Altitude STD Smoothed'),
               climbs=S('Climb')):
        '''
        Airspeed from 1000ft above the airfield to a pressure altitude of 8000ft.
        As we are only interested in the climbing phase, this is used as the
        normal slices_from_to will not work with two parameters.
        '''
        for climb in climbs:
            aal = np.ma.clump_unmasked(np.ma.masked_less(alt_aal.array[climb.slice], 1000.0))
            std = np.ma.clump_unmasked(np.ma.masked_greater(alt_std.array[climb.slice], 8000.0))
            scope = shift_slices(slices_and(aal, std), climb.slice.start)
            self.create_kpv_from_slices(
                air_spd.array,
                scope,
                max_value
            )


class Airspeed8000To10000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed between 8000ft and 10000ft STD (smoothed) during climb.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std=P('Altitude STD Smoothed'),
               climb=S('Climb')):

        alt_band = np.ma.masked_outside(alt_std.array, 8000, 10000)
        alt_climb_sections = valid_slices_within_array(alt_band, climb)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_climb_sections,
            max_value,
        )


class Airspeed3000FtToTopOfClimbMax(KeyPointValueNode):
    '''
    Maximum recorded airspeed between 3000ft AAL and Top of Climb.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               tocs=KTI('Top Of Climb')):
        toc = tocs.get_first()

        if not toc:
            return

        index_at_3000ft = index_at_value(alt_aal.array, 3000,
                                         _slice=slice(toc.index, None, -1))

        if not index_at_3000ft:
            # Top Of Climb below 3000ft?
            return

        self.create_kpv(*max_value(air_spd.array,
                                   _slice=slice(index_at_3000ft, toc.index)))


class Airspeed3000FtToTopOfClimbMin(KeyPointValueNode):
    '''
    Minimum recorded airspeed between 3000ft and Top of Climb.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               tocs=KTI('Top Of Climb')):
        # TODO: Should we be able to handle multiple Top of Climbs?
        toc = tocs.get_first()

        if not toc:
            return

        index_at_3000ft = index_at_value(alt_aal.array, 3000,
                                         _slice=slice(toc.index, None, -1))

        if not index_at_3000ft:
            # Top Of Climb below 3000ft?
            return

        self.create_kpv(*min_value(air_spd.array,
                                   _slice=slice(index_at_3000ft, toc.index)))

########################################
# Airspeed: Descending


class Airspeed10000To5000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during descent between 10000ft STD (smoothed) and 5000ft AAL.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_std=P('Altitude STD Smoothed'),
               descends=S('Descent')):

        for descend in descends:
            std = np.ma.clump_unmasked(
                np.ma.masked_greater(alt_std.array[descend.slice], 10000.0))
            aal = np.ma.clump_unmasked(
                np.ma.masked_less(alt_aal.array[descend.slice], 5000.0))
            scope = shift_slices(slices_and(aal, std), descend.slice.start)
            self.create_kpv_from_slices(air_spd.array, scope, max_value)


class Airspeed10000To8000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during descent between 10000ft STD (smoothed) and 5000ft STD (smoothed).
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std=P('Altitude STD Smoothed'),
               descent=S('Descent')):

        alt_band = np.ma.masked_outside(alt_std.array, 10000, 8000)
        alt_descent_sections = valid_slices_within_array(alt_band, descent)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            max_value,
        )


class Airspeed8000To5000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during descent between 8000ft STD (smoothed) and 5000ft AAL.
    '''
    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_std=P('Altitude STD Smoothed'),
               descends=S('Descent')):
        # As we are only interested in the descending phase, this is used as
        # the normal slices_from_to will not work with two parameters.
        for descend in descends:
            std = np.ma.clump_unmasked(np.ma.masked_greater(alt_std.array[descend.slice], 8000.0))
            aal = np.ma.clump_unmasked(np.ma.masked_less(alt_aal.array[descend.slice], 5000.0))
            scope = shift_slices(slices_and(aal, std), descend.slice.start)
            self.create_kpv_from_slices(
                air_spd.array,
                scope,
                max_value
            )


class Airspeed5000To3000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during descent between 5000ft AAL to 3000ft AAL.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               descent=S('Descent')):

        alt_band = np.ma.masked_outside(alt_aal.array, 5000, 3000)
        alt_descent_sections = valid_slices_within_array(alt_band, descent)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            max_value,
        )


class Airspeed3000To1000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during descent between 3000ft AAL and 1000ft AAL.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        # TODO: Include level flight once Sections use intervals.
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(3000, 1000),
            max_value,
        )




class Airspeed1000To500FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during the final approach between 1000ft AAL and 500ft AAL.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Airspeed']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Final Approach'])
        return all_of(required, available)

    def derive(self,
               air_spd=P('Airspeed'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               final_app=S('Final Approach'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_band = np.ma.masked_outside(alt_agl.array, 1000, 500)
            alt_descent_sections = valid_slices_within_array(alt_band, descending)
            self.create_kpvs_within_slices(
                air_spd.array,
                alt_descent_sections,
                max_value,
                min_duration=HOVER_MIN_DURATION,
                freq=air_spd.frequency)
        else:
            alt_band = np.ma.masked_outside(alt_aal.array, 1000, 500)
            alt_descent_sections = valid_slices_within_array(alt_band, final_app)
            self.create_kpvs_within_slices(
                air_spd.array,
                alt_descent_sections,
                max_value)


class Airspeed1000To500FtMin(KeyPointValueNode):
    '''
    Minimum airspeed during the final approach between 1000ft AAL and 500ft AAL.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               final_app=S('Final Approach')):

        alt_band = np.ma.masked_outside(alt_aal.array, 1000, 500)
        alt_descent_sections = valid_slices_within_array(alt_band, final_app)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            min_value,
        )


class Airspeed500To100FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during the final approach between 500ft AAL and 100ft AAL.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent')):

        alt_band = np.ma.masked_outside(alt_agl.array, 500, 100)
        alt_descent_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            max_value,
            min_duration=HOVER_MIN_DURATION,
            freq=air_spd.frequency)


class Airspeed500To100FtMin(KeyPointValueNode):
    '''
    Minimum airspeed during the final approach between 500ft AAL and 100ft AAL.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 500, 100)
        alt_descent_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            min_value,
            min_duration=HOVER_MIN_DURATION,
            freq=air_spd.frequency)


class Airspeed100To20FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during the final approach and landing between 100 AAL and 20ft AAL.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 20)
        alt_descent_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            max_value,
            min_duration=HOVER_MIN_DURATION,
            freq=air_spd.frequency)


class Airspeed100To20FtMin(KeyPointValueNode):
    '''
    Minimum airspeed during the final approach and landing between 100 AAL and 20ft AAL.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 20)
        alt_descent_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            min_value,
            min_duration=HOVER_MIN_DURATION,
            freq=air_spd.frequency)


class Airspeed500To20FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during the final approach and landing between 500ft AAL and 20ft AAL.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Airspeed']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descent'])
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               air_spd=P('Airspeed'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_band = np.ma.masked_outside(alt_agl.array, 500, 20)
            alt_descent_sections = valid_slices_within_array(alt_band, descending)
            self.create_kpvs_within_slices(
                air_spd.array,
                alt_descent_sections,
                max_value,
                min_duration=HOVER_MIN_DURATION,
                freq=air_spd.frequency)
        else:
            # TODO: Include level flight once Sections use intervals.
            self.create_kpvs_within_slices(
                air_spd.array,
                alt_aal.slices_from_to(500, 20),
                max_value)


class Airspeed500To20FtMin(KeyPointValueNode):
    '''
    Minimum airspeed during the final approach and landing between 500ft AAL and 20ft AAL.
    '''

    units = ut.KT

    def derive(self, air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        # TODO: Include level flight once Sections use intervals.
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class Airspeed500To50FtMedian(KeyPointValueNode):
    '''
    Median value of the recorded airspeed from 500ft AAL to 20ft AAL.
    '''

    units = ut.KT

    def derive(self, air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        # TODO: Include level flight once Sections use intervals.
        # TODO: Round to the nearest Integer value if using for Airspeed
        # Selected (pilots select whole numbers!)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(500, 50),
            median_value,
        )


class Airspeed500To50FtMedianMinusAirspeedSelected(KeyPointValueNode):
    '''
    Median of the differences between airspeed and selected speed between 500ft and 50ft AAL.
    '''

    units = ut.KT

    def derive(self, spd_selected=P('Airspeed Selected'),
               spds_500_to_50=KPV('Airspeed 500 To 50 Ft Median')):
        for spd_500_to_50 in spds_500_to_50:
            spd_sel = value_at_index(spd_selected.array, spd_500_to_50.index)
            if spd_sel is not None:
                self.create_kpv(spd_500_to_50.index,
                                spd_500_to_50.value - spd_sel)


class Airspeed20FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum airspeed between 20ft AGL and touchdown.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_agl.slices_to_kti(20, touchdowns),
            max_value,
        )


class Airspeed2NMToOffshoreTouchdown(KeyPointValueNode):
    '''
    Airspeed 2NM from offshore touchdown. Helicopter only.
    '''

    units = ut.KT

    name = 'Airspeed 2 NM To Touchdown'

    can_operate = helicopter_only

    def derive(self, airspeed=P('Airspeed'), dtts=P('Distance To Touchdown'),
               touchdown=KTI('Offshore Touchdown')):

        for tdwn in touchdown:
            dist_to_touchdown = dtts.get_previous(tdwn.index, name='2.0 NM To Touchdown')
            if dist_to_touchdown:
                self.create_kpvs_at_ktis(airspeed.array, [dist_to_touchdown])


class AirspeedAbove500FtMin(KeyPointValueNode):
    '''
    Minimum airspeed above 500ft AGL during standard approaches (helicopter only)
    '''

    units = ut.KT

    can_operate = helicopter_only
                                        
    def derive(self, air_spd= P('Airspeed'), alt_agl=P('Altitude AGL For Flight Phases'),
               approaches=App('Approach Information')):
        
        app_slices = []
        for approach in approaches:
            if approach.type in ['SHUTTLING','AIRBORNE_RADAR']:
                app_slices.append(approach.slice)
        
        for alt_agl_slice in alt_agl.slices_above(500):
            if any([slices_overlap(app_slice, alt_agl_slice) for app_slice in app_slices]):
                continue
            else:
                self.create_kpv_from_slices(air_spd.array,[alt_agl_slice], min_value)
                        
        
class AirspeedAbove500FtMinOffshoreSpecialProcedure(KeyPointValueNode):
    '''
    Minimum airspeed above 500ft AGL during a Shuttling Approach, ARDA
    or AROA (helicopter only)
    '''

    units = ut.KT
    can_operate = helicopter_only
    def derive(self, air_spd= P('Airspeed'), alt_agl=P('Altitude AGL For Flight Phases'),
               approaches=App('Approach Information')):
        
        app_slices = []
        for approach in approaches:
            if approach.type in ['SHUTTLING','AIRBORNE_RADAR']:
                app_slices.append(approach.slice)
        
        for alt_agl_slice in alt_agl.slices_above(500):
            for app_slice in app_slices:
                if slices_overlap(app_slice, alt_agl_slice):
                    self.create_kpv_from_slices(air_spd.array,[alt_agl_slice], min_value)
         

class AirspeedAt200FtDuringOnshoreApproach(KeyPointValueNode):
    '''
    Approach airspeed at 200ft AGL (helicopter only)
    '''

    units = ut.KT
    can_operate = helicopter_only

    def derive(self, air_spd=P('Airspeed'), alt_agl=P('Altitude AGL For Flight Phases'),
               approaches=App('Approach Information'), offshore=M('Offshore')):
        for approach in approaches:
            # check if landed/lowest point of approach is Offshore. May trigger incorrectly
            # close to coast. Will be able to improve once we have implemented Rig locations
            if value_at_index(offshore.array, approach.slice.stop, interpolate=False) == 'Offshore':
                continue

            index = index_at_value(alt_agl.array, 200, 
                                   _slice=slice(approach.slice.stop, approach.slice.start, -1))
            if not index:
                continue
            value = value_at_index(air_spd.array, index)
            self.create_kpv(index, value)


class AirspeedAtTouchdown(KeyPointValueNode):
    '''
    Airspeed measurement at the point of Touchdown.
    '''

    units = ut.KT

    def derive(self, air_spd=P('Airspeed'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(air_spd.array, touchdowns)


class AirspeedMinsToTouchdown(KeyPointValueNode):
    '''
    IAS at various time intervals before touchdown.
    '''

    # TODO: Review and improve this technique of building KPVs on KTIs.
    from analysis_engine.key_time_instances import MinsToTouchdown

    NAME_FORMAT = 'Airspeed ' + MinsToTouchdown.NAME_FORMAT
    NAME_VALUES = MinsToTouchdown.NAME_VALUES
    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               mtt_kti=KTI('Mins To Touchdown')):

        for mtt in mtt_kti:
            # XXX: Assumes that the number will be the first part of the name:
            time = int(mtt.name.split(' ')[0])
            self.create_kpv(mtt.index, air_spd.array[mtt.index], time=time)


class AirspeedNMToThreshold(KeyPointValueNode):
    '''
    Airspeed at distances in nautical miles to runway threshold.
    '''

    # TODO: Review and improve this technique of building KPVs on KTIs.
    from analysis_engine.key_time_instances import DistanceFromThreshold

    NAME_FORMAT = 'Airspeed ' + DistanceFromThreshold.NAME_FORMAT
    NAME_VALUES = DistanceFromThreshold.NAME_VALUES
    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               dtt_kti=KTI('Distance From Threshold')):

        for dtt in dtt_kti:
            # XXX: Assumes that the number will be the first part of the name:
            distance = int(dtt.name.split(' ')[0])
            self.create_kpv(dtt.index, air_spd.array[dtt.index], distance=distance)



class AirspeedAtAPGoAroundEngaged(KeyPointValueNode):
    '''
    Airspeed at which AP Go Around mode has been engaged. Helicopter only.
    '''

    name = 'Airspeed At AP Go Around Engaged'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, air_spd=P('Airspeed'), airs=S('Airborne'),
               ap_mode=M('AP Pitch Mode (1)')):

        sections = slices_and(airs.get_slices(),
                              clump_multistate(ap_mode.array, 'Go Around'))
        for section in sections:
            index = section.start
            value = air_spd.array[index]
            self.create_kpv(index, value)


class AirspeedWhileAPHeadingEngagedMin(KeyPointValueNode):
    '''
    Minimum recorded airspeed at which AP Heading mode was engaged. Helicopter only.
    '''

    name = 'Airspeed While AP Heading Engaged Min'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, air_spd=P('Airspeed'), airs=S('Airborne'),
               ap_mode=M('AP Roll-Yaw Mode (1)')):

        heads = clump_multistate(ap_mode.array, 'Heading')
        if heads:
            sections = slices_and(airs.get_slices(), heads)
            self.create_kpv_from_slices(air_spd.array, sections, min_value)


class AirspeedWhileAPVerticalSpeedEngagedMin(KeyPointValueNode):
    '''
    Minimum recorded airspeed at which AP VS mode was engaged. Helicopter only.
    '''

    name = 'Airspeed While AP Vertical Speed Engaged Min'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, air_spd=P('Airspeed'), airs=S('Airborne'),
               ap_mode=M('AP Collective Mode (1)')):

        vss = clump_multistate(ap_mode.array, 'V/S')
        if vss:
            sections = slices_and(airs.get_slices(), vss)
            self.create_kpv_from_slices(air_spd.array, sections, min_value)


class AirspeedAtAPUpperModesEngaged(KeyPointValueNode):
    '''
    Airspeed at initial climb in which any of the following AP upper 
    modes are first engaged:
    - AP (1) Heading Selected Mode Engaged
    - AP (2) Heading Selected Mode Engaged
    - AP (1) Vertical Speed Mode Engaged
    - AP (2) Vertical Speed Mode Engaged
    - AP (1) Altitude Preselect Mode Engaged
    - AP (2) Altitude Preselect Mode Engaged
    - AP (1) Airspeed Mode Engaged
    - AP (2) Airspeed Mode Engaged

    (S92 helicopters only)
    '''
    name = 'Airspeed At AP Upper Modes Engaged'
    units = ut.KT

    @classmethod
    # This KPV is specific to the S92 helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        is_s92 = ac_type == helicopter and family and family.value == 'S92'
        return is_s92 and all_deps(cls, available)  

    def derive(self,
               air_spd=P('Airspeed'),
               ap_1_hdg=M('AP (1) Heading Selected Mode Engaged'),
               ap_2_hdg=M('AP (2) Heading Selected Mode Engaged'),
               ap_1_alt=M('AP (1) Altitude Preselect Mode Engaged'),
               ap_2_alt=M('AP (2) Altitude Preselect Mode Engaged'),
               ap_1_vrt=M('AP (1) Vertical Speed Mode Engaged'),
               ap_2_vrt=M('AP (2) Vertical Speed Mode Engaged'),
               ap_1_air=M('AP (1) Airspeed Mode Engaged'),
               ap_2_air=M('AP (2) Airspeed Mode Engaged'),
               climb=S('Initial Climb')):
        mode_state='Engaged'
        ap_modes = vstack_params_where_state(
            (ap_1_hdg, mode_state), (ap_2_hdg, mode_state),
            (ap_1_alt, mode_state), (ap_2_alt, mode_state),
            (ap_1_vrt, mode_state), (ap_2_vrt, mode_state),
            (ap_1_air, mode_state), (ap_2_air, mode_state),
        ).any(axis=0)
        ap_slices = slices_and(climb.get_slices(), runs_of_ones(ap_modes))
        for s in ap_slices:
            self.create_kpv(s.start, air_spd.array[s.start])


class AirspeedTrueAtTouchdown(KeyPointValueNode):
    '''
    Airspeed True at the point of Touchdown.

    This KPV relates to groundspeed at touchdown to illustrate headwinds and
    tailwinds. We also have 'Tailwind 100 Ft To Touchdown Max' to cater for
    safety event triggers.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed True'),
               touchdowns=KTI('Touchdown')):

        #  Note: low frequency airspeed true can reduce to 0 the sample
        #  following touchdown. By masking values less than 1kt
        #  index_at_value will return the last recorded value before
        #  touchdown. So long as that value is not also masked. This has
        #  proven more accurate than interpolating between last recorded
        #  value and 0.

        array = np.ma.masked_less(air_spd.array, 1)
        self.create_kpvs_at_ktis(array, touchdowns)


class AirspeedReferenceVariationMax(KeyPointValueNode):
    '''
    Maximum difference between the Airspeed Reference which is in the AFR or
    recorded on the aircraft and that of the Airspeed Reference Lookup
    calculated from tables.
    '''

    units = ut.KT

    def derive(self,
               spd_ref_rec=P('Airspeed Reference'),
               spd_ref_tbl=P('Airspeed Reference Lookup'),
               apps=S('Approach And Landing')):

        self.create_kpv_from_slices(
            spd_ref_rec.array - spd_ref_tbl.array,
            apps.get_slices(),
            max_abs_value,
        )


class V2VariationMax(KeyPointValueNode):
    '''
    Maximum difference between the V2 which is in the AFR or recorded on the
    aircraft and that of the V2 Lookup calculated from tables.
    '''

    units = ut.KT

    def derive(self,
               v2_rec=P('V2'),
               v2_tbl=P('V2 Lookup')):

        # Q: Use Airspeed Selected?
        self.create_kpv_from_slices(
            v2_rec.array - v2_tbl.array,
            [slice(0, len(v2_rec.array))],
            max_abs_value,
        )


class V2AtLiftoff(KeyPointValueNode):
    '''
    V2 IAS at liftoff. Recorded value is used if available, if not, data from
    AFR will be used.

    For Airbus aircraft, if auto speed control is enabled, V2 is not
    recorded and AFR is not available, selected speed value at the 
    beginning of takeoff roll is used. 
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available, afr_v2=A('AFR V2'),
                    manufacturer=A('Manufacturer')):
        '''
        Some other aircraft types record multiple parameters in the same location
        within data frames. We need to select only the data that we are interested
        in, i.e. the V2 values.
    
        Reference was made to the following documentation to assist with the
        development of this algorithm:
    
        - A320 Flight Profile Specification
        - A321 Flight Profile Specification
        '''
        afr = all_of((
            'AFR V2',
            'Liftoff',
            'Climb Start',
        ), available) and afr_v2 and afr_v2.value >= AIRSPEED_THRESHOLD

        airbus = all_of((
            'Airspeed Selected',
            'Speed Control',
            'Liftoff',
            'Climb Start',
            'Manufacturer',
        ), available) and manufacturer and manufacturer.value == 'Airbus'

        embraer = all_of((
            'V2-Vac',
            'Liftoff',
            'Climb Start',
        ), available)

        v2 = all_of((
            'V2',
            'Liftoff',
            'Climb Start',
        ), available)

        return v2 or afr or airbus or embraer

    def derive(self,
               v2=P('V2'),
               v2_vac=A('V2-Vac'),
               spd_sel=P('Airspeed Selected'),
               spd_ctl=P('Speed Control'),
               afr_v2=A('AFR V2'),
               liftoffs=KTI('Liftoff'),
               climb_starts=KTI('Climb Start'),
               manufacturer=A('Manufacturer')):

        # Determine interesting sections of flight which we want to use for V2.
        # Due to issues with how data is recorded, use five superframes before
        # liftoff until the start of the climb:
        starts = deepcopy(liftoffs)
        for start in starts:
            start.index = max(start.index - 5 * 64 * self.frequency, 0)
        phases = slices_from_ktis(starts, climb_starts)

        # 1. Use recorded value (if available):
        if v2:
            for phase in phases:
                index = liftoffs.get_last(within_slice=phase).index
                if v2.frequency >= 0.125:
                    v2_liftoff = closest_unmasked_value(
                        v2.array, index, start_index=phase.start,
                        stop_index=phase.stop)
                    if v2_liftoff:
                        self.create_kpv(index, v2_liftoff.value)
                else:
                    value = most_common_value(v2.array[phase])
                    self.create_kpv(index, value)
            return

        # 2. Use value provided in achieved flight record (if available):
        if afr_v2 and afr_v2.value >= AIRSPEED_THRESHOLD:
            for phase in phases:
                index = liftoffs.get_last(within_slice=phase).index
                value = round(afr_v2.value)
                if value is not None:
                    self.create_kpv(index, value)
            return

        # 3. Derive parameter for Embraer 170/190:
        if v2_vac:
            for phase in phases:
                value = most_common_value(v2_vac.array[phase])
                index = liftoffs.get_last(within_slice=phase).index
                if value is not None:
                    self.create_kpv(index, value)
            return

        # 4. Derive parameter for Airbus:
        if manufacturer and manufacturer.value == 'Airbus':
            spd_sel.array[spd_ctl.array == 'Manual'] = np.ma.masked
            for phase in phases:
                value = most_common_value(spd_sel.array[phase])
                index = liftoffs.get_last(within_slice=phase).index
                if value is not None:
                    self.create_kpv(index, value)
            return


class V2LookupAtLiftoff(KeyPointValueNode):
    '''
    V2 IAS (derived) at liftoff.

    Derived from weight and flap (lever detents).
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_series=A('Engine Series'), engine_type=A('Engine Type')):

        core = all_of((
            'Liftoff',
            'Climb Start',
            'Model',
            'Series',
            'Family',
            'Engine Type',
            'Engine Series',
        ), available)

        flap = any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

        attrs = (model, series, family, engine_type, engine_series)
        return core and flap and lookup_table(cls, 'v2', *attrs)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               weight_liftoffs=KPV('Gross Weight At Liftoff'),
               liftoffs=KTI('Liftoff'),
               climb_starts=KTI('Climb Start'),
               model=A('Model'),
               series=A('Series'),
               family=A('Family'),
               engine_type=A('Engine Type'),
               engine_series=A('Engine Series')):
        '''
        Takeoff Safety Speed (V2) can be derived for different aircraft.
    
        In cases where values cannot be derived solely from recorded parameters, we
        can make use of a look-up table to determine values for velocity speeds.
    
        For V2, looking up a value requires the weight and flap (lever detents)
        at liftoff.
    
        Flap is used as the first dependency to avoid interpolation of flap detents
        when flap is recorded at a lower frequency than airspeed.
        '''
        # Determine interesting sections of flight which we want to use for V2.
        # Due to issues with how data is recorded, use five superframes before
        # liftoff until the start of the climb:
        starts = deepcopy(liftoffs)
        for start in starts:
            start.index = max(start.index - 5 * 64 * self.hz, 0)
        phases = slices_from_ktis(starts, climb_starts)

        # Initialise the velocity speed lookup table:
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'v2', *attrs)

        for phase in phases:

            if weight_liftoffs:
                weight_liftoff = weight_liftoffs.get_first(within_slice=phase)
                index, weight = weight_liftoff.index, weight_liftoff.value
            else:
                index, weight = liftoffs.get_first(within_slice=phase).index, None

            if index is None:
                continue

            detent = (flap_lever or flap_synth).array[index]

            try:
                index = liftoffs.get_last(within_slice=phase).index
                value = table.v2(detent, weight)
                self.create_kpv(index, value)
            except (KeyError, ValueError) as error:
                self.warning("Error in '%s': %s", self.name, error)
                # Where the aircraft takes off with flap settings outside the
                # documented V2 range, we need the program to continue without
                # raising an exception, so that the incorrect flap at takeoff
                # can be detected.
                continue


class AirspeedSelectedAtLiftoff(KeyPointValueNode):
    '''
    Selected airspeed at liftoff.
    '''

    units = ut.KT

    def derive(self,
               spd_sel=P('Airspeed Selected'),
               liftoffs=KTI('Liftoff'),
               climb_starts=KTI('Climb Start')):

        starts = deepcopy(liftoffs)
        for start in starts:
            start.index = max(start.index - 5 * 64 * self.hz, 0)
        phases = slices_from_ktis(starts, climb_starts)
        for phase in phases:
            this_lift = liftoffs.get_last(within_slice=phase)
            index = None
            if this_lift:
                index = this_lift.index
            if spd_sel.frequency >= 0.125 and index:
                spd_sel_liftoff = closest_unmasked_value(
                    spd_sel.array, index, start_index=phase.start,
                    stop_index=phase.stop)
                value = spd_sel_liftoff.value if spd_sel_liftoff else None
            else:
                value = most_common_value(spd_sel.array[phase])
            if value:
                self.create_kpv(index, value)
            else:
                self.warning("%s is entirely masked within %s", spd_sel.__class__.__name__, phase)

########################################
# Airspeed: Minus V2


class AirspeedMinusV2AtLiftoff(KeyPointValueNode):
    '''
    Airspeed difference from V2 at the point of Liftoff. A positive value
    measured ensures an operational speed margin above V2.
    '''

    name = 'Airspeed Minus V2 At Liftoff'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(spd_v2.array, liftoffs)


class AirspeedMinusV2At35FtDuringTakeoff(KeyPointValueNode):
    '''
    Airspeed difference from V2 at the 35ft AGL (end of Takeoff phase). A
    positive value measured ensures an operational speed margin above V2.
    '''

    name = 'Airspeed Minus V2 At 35 Ft During Takeoff'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               takeoffs=S('Takeoff')):

        for takeoff in takeoffs:
            index = takeoff.stop_edge  # Takeoff ends at 35ft!
            value = value_at_index(spd_v2.array, index)
            self.create_kpv(index, value)


class AirspeedMinusV235ToClimbAccelerationStartMin(KeyPointValueNode):
    '''
    Minimum airspeed difference from V2 between 35ft AAL to Climb Acceleration Start
    if available, otherwise to 1000ft AAL (end of initial climb).
    '''

    name = 'Airspeed Minus V2 35 To Climb Acceleration Start Min'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               climbs=S('Initial Climb'),
               climb_accel_start=KTI('Climb Acceleration Start')):
        init_climb = climbs.get_first()
        if init_climb:
            if len(climb_accel_start):
                _slice = slice(init_climb.start_edge,
                               climb_accel_start.get_first().index + 1)
            else:
                _slice = init_climb.slice
    
            self.create_kpvs_within_slices(spd_v2.array, (_slice,), min_value)


class AirspeedMinusV235ToClimbAccelerationStartMax(KeyPointValueNode):
    '''
    Maximum airspeed difference from V2 between 35ft AAL and Climb Acceleration Start
    if available, otherwise to 1000ft AAL (end of initial climb).
    '''

    name = 'Airspeed Minus V2 35 To Climb Acceleration Start Max'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               climbs=S('Initial Climb'),
               climb_accel_start=KTI('Climb Acceleration Start')):
        init_climb = climbs.get_first()
        if init_climb:
            if len(climb_accel_start):
                _slice = slice(init_climb.start_edge,
                               climb_accel_start.get_first().index + 1)
            else:
                _slice = init_climb.slice
    
            self.create_kpvs_within_slices(spd_v2.array, (_slice,), max_value)


class AirspeedMinusV2For3Sec35ToClimbAccelerationStartMin(KeyPointValueNode):
    '''
    Minimum airspeed difference from V2 between 35ft AAL to Climb Acceleration Start
    if available, otherwise to 1000ft AAL (end of initial climb)
    '''

    name = 'Airspeed Minus V2 For 3 Sec 35 To Climb Acceleration Start Min'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2 For 3 Sec'),
               climbs=S('Initial Climb'),
               climb_accel_start=KTI('Climb Acceleration Start')):
        init_climb = climbs.get_first()
        if init_climb:
            if len(climb_accel_start):
                _slice = slice(init_climb.slice.start,
                               climb_accel_start.get_first().index + 1)
            else:
                _slice = init_climb.slice
    
            self.create_kpvs_within_slices(spd_v2.array, (_slice,), min_value)


class AirspeedMinusV2For3Sec35ToClimbAccelerationStartMax(KeyPointValueNode):
    '''
    Maximum airspeed difference from V2 between 35ft AAL to Climb Acceleration Start
    if available, otherwise to 1000ft AAL (end of initial climb)
    '''

    name = 'Airspeed Minus V2 For 3 Sec 35 To Climb Acceleration Start Max'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2 For 3 Sec'),
               climbs=S('Initial Climb'),
               climb_accel_start=KTI('Climb Acceleration Start')):
        init_climb = climbs.get_first()
        if init_climb:
            if len(climb_accel_start):
                _slice = slice(init_climb.slice.start,
                               climb_accel_start.get_first().index + 1)
            else:
                _slice = init_climb.slice
    
            self.create_kpvs_within_slices(spd_v2.array, (_slice,), max_value)


class AirspeedMinusV235To1000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed difference from V2 between 35ft AAL to 1000ft AAL.
    '''

    name = 'Airspeed Minus V2 35 To 1000 Ft Max'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_v2.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class AirspeedMinusV235To1000FtMin(KeyPointValueNode):
    '''
    Minimum airspeed difference from V2 between 35ft AAL to 1000ft AAL. A positive
    value measured ensures an operational speed margin above V2.
    '''

    name = 'Airspeed Minus V2 35 To 1000 Ft Min'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_v2.array,
            alt_aal.slices_from_to(35, 1000),
            min_value,
        )


class AirspeedMinusV2For3Sec35To1000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed difference from V2 (for at least 3 seconds) between 35ft AAL 
    and 1000ft AAL.
    '''

    name = 'Airspeed Minus V2 For 3 Sec 35 To 1000 Ft Max'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2 For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_v2.array,
            trim_slices(alt_aal.slices_from_to(35, 1000), 3, self.frequency,
                        hdf_duration),
            max_value,
        )


class AirspeedMinusV2For3Sec35To1000FtMin(KeyPointValueNode):
    '''
    Minimum airspeed difference from V2 (for at least 3 seconds) from 35ft AAL 
    to 1000ft AAL. A positive value measured ensures an operational speed margin
    above V2.
    '''

    name = 'Airspeed Minus V2 For 3 Sec 35 To 1000 Ft Min'
    units = ut.KT

    def derive(self,
               spd_v2=P('Airspeed Minus V2 For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_v2.array,
            trim_slices(alt_aal.slices_from_to(35, 1000), 3, self.frequency,
                        hdf_duration),
            min_value,
        )


########################################
# Airspeed: Minus Minimum Airspeed


class AirspeedMinusMinimumAirspeedAbove10000FtMin(KeyPointValueNode):
    '''
    Minimum difference between airspeed and the minimum airspeed above 10,000
    ft (STD smoothed). A positive value measured ensures that the aircraft is 
    above the speed limit below which there is a reduced manoeuvring capability.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed Minus Minimum Airspeed'),
               alt_std=P('Altitude STD Smoothed')):

        self.create_kpvs_within_slices(air_spd.array,
                                       alt_std.slices_above(10000),
                                       min_value)


class AirspeedMinusMinimumAirspeed35To10000FtMin(KeyPointValueNode):
    '''
    Minimum difference between airspeed and the minimum airspeed from 35ft AAL 
    to 10000ft (STD smoothed). A positive value measured ensures that the 
    aircraft is above the speed limit below which there is a reduced 
    manoeuvring capability.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed Minus Minimum Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_std=P('Altitude STD Smoothed'),
               init_climbs=S('Initial Climb'),
               climbs=S('Climb')):
        std = np.ma.clump_unmasked(np.ma.masked_greater(alt_std.array, 10000.0))
        aal = np.ma.clump_unmasked(np.ma.masked_less(alt_aal.array, 35.0))
        alt_bands = slices_and(std, aal)
        combined_climb = slices_or(climbs.get_slices(),
                                   init_climbs.get_slices())
        scope = slices_and(alt_bands, combined_climb)
        self.create_kpv_from_slices(
            air_spd.array,
            scope,
            min_value
        )


class AirspeedMinusMinimumAirspeed10000To50FtMin(KeyPointValueNode):
    '''
    Minimum difference between airspeed and the minimum airspeed between 
    10000ft (STD smoothed) and 50ft AAL. A positive value measured ensures 
    that the aircraft is above thespeed limit below which there is a 
    reduced manoeuvring capability.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed Minus Minimum Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_std=P('Altitude STD Smoothed'),
               descents=S('Descent')):
        std = np.ma.clump_unmasked(np.ma.masked_greater(alt_std.array, 10000.0))
        aal = np.ma.clump_unmasked(np.ma.masked_less(alt_aal.array, 50.0))
        alt_bands = slices_and(std, aal)
        scope = slices_and(alt_bands, descents.get_slices())
        self.create_kpv_from_slices(
            air_spd.array,
            scope,
            min_value
        )


class AirspeedMinusMinimumAirspeedDuringGoAroundMin(KeyPointValueNode):
    '''
    Minimum difference between airspeed and the minimum airspeed during
    go-around and the climbout phase. A positive value measured ensures that
    the aircraft is above the speed limit below which there is a reduced
    manoeuvring capability.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed Minus Minimum Airspeed'),
               phases=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(air_spd.array, phases, min_value)


########################################
# Airspeed: Minus Airspeed Selected


class AirspeedMinusAirspeedSelectedFor3Sec500To20FtMax(KeyPointValueNode):
    '''
    Maximum difference between IAS and selected airspeed during approach
    between 500 ft AAL and 20 ft AAL.

    For aircraft with 'Airspeed Selected (FMS)', this parameter will be used
    instead of 'Airspeed Selected'.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Altitude AAL For Flight Phases', 'HDF Duration'),
                      available) and \
               any_of(('Airspeed Minus Airspeed Selected For 3 Sec',
                       'Airspeed Minus Airspeed Selected (FMS) For 3 Sec'),
                      available)

    def derive(self,
               spd_sel=P('Airspeed Minus Airspeed Selected For 3 Sec'),
               spd_fms=P('Airspeed Minus Airspeed Selected (FMS) For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        spd_rel = spd_fms or spd_sel
        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(500, 20), 3, self.frequency,
                        hdf_duration),
            max_value,
        )


class AirspeedMinusAirspeedSelectedFor3Sec500To20FtMin(KeyPointValueNode):
    '''
    Minimum difference between IAS and selected airspeed during approach
    between 500 ft AAL and 20 ft AAL.

    For aircraft with 'Airspeed Selected (FMS)', this parameter will be used
    instead of 'Airspeed Selected'.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Altitude AAL For Flight Phases', 'HDF Duration'),
                      available) and \
               any_of(('Airspeed Minus Airspeed Selected For 3 Sec',
                       'Airspeed Minus Airspeed Selected (FMS) For 3 Sec'),
                      available)

    def derive(self,
               spd_sel=P('Airspeed Minus Airspeed Selected For 3 Sec'),
               spd_fms=P('Airspeed Minus Airspeed Selected (FMS) For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        spd_rel = spd_fms or spd_sel
        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(500, 20), 3, self.frequency,
                        hdf_duration),
            min_value,
        )


class AirspeedMinusAirspeedSelectedFor3Sec1000To500FtMax(KeyPointValueNode):
    '''
    Maximum difference between IAS and selected airspeed during approach
    for 3 sec between 1000 ft AAL and 500 ft AAL.

    For aircraft with 'Airspeed Selected (FMS)', this parameter will be used
    instead of 'Airspeed Selected'.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Altitude AAL For Flight Phases', 'HDF Duration'),
                      available) and \
               any_of(('Airspeed Minus Airspeed Selected For 3 Sec',
                       'Airspeed Minus Airspeed Selected (FMS) For 3 Sec'),
                      available)

    def derive(self,
               spd_sel=P('Airspeed Minus Airspeed Selected For 3 Sec'),
               spd_fms=P('Airspeed Minus Airspeed Selected (FMS) For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        spd_rel = spd_fms or spd_sel
        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(1000, 500), 3, self.frequency,
                        hdf_duration),
            max_value,
        )


class AirspeedMinusAirspeedSelectedFor3Sec1000To500FtMin(KeyPointValueNode):
    '''
    Minimum difference between IAS and selected airspeed during approach
    for 3 sec between 1000 ft AAL and 500 ft AAL.

    For aircraft with 'Airspeed Selected (FMS)', this parameter will be used
    instead of 'Airspeed Selected'.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Altitude AAL For Flight Phases', 'HDF Duration'),
                      available) and \
               any_of(('Airspeed Minus Airspeed Selected For 3 Sec',
                       'Airspeed Minus Airspeed Selected (FMS) For 3 Sec'),
                      available)

    def derive(self,
               spd_sel=P('Airspeed Minus Airspeed Selected For 3 Sec'),
               spd_fms=P('Airspeed Minus Airspeed Selected (FMS) For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        spd_rel = spd_fms or spd_sel
        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(1000, 500), 3, self.frequency,
                        hdf_duration),
            min_value,
        )


########################################
# Airspeed: Relative


class AirspeedRelativeAtTouchdown(KeyPointValueNode):
    '''
    Airspeed Relative at touchdown.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(spd_rel.array, touchdowns)


class AirspeedRelative1000To500FtMax(KeyPointValueNode):
    '''
    Maximum airspeed relative between 1000ft and 500ft AAL.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class AirspeedRelative1000To500FtMin(KeyPointValueNode):
    '''
    Minimum airspeed relative between 1000ft and 500ft AAL.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class AirspeedRelative500To20FtMax(KeyPointValueNode):
    '''
    Maximum airspeed relative between 500ft and 20ft AAL.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class AirspeedRelative500To20FtMin(KeyPointValueNode):
    '''
    Minimum airspeed relative between 500ft and 20ft AAL.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedRelative20FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum airspeed relative between 20ft AAL and touchdown.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_to_kti(20, touchdowns),
            max_value,
        )


class AirspeedRelative20FtToTouchdownMin(KeyPointValueNode):
    '''
    Minimum airspeed relative between 20ft AAL and touchdown.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_to_kti(20, touchdowns),
            min_value,
        )


class AirspeedRelativeFor3Sec1000To500FtMax(KeyPointValueNode):
    '''
    Maximum airspeed relative for 3 seconds between 1000ft and 500ft AAL.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(1000, 500), 3, self.frequency,
                        hdf_duration),
            max_value,
        )


class AirspeedRelativeFor3Sec1000To500FtMin(KeyPointValueNode):
    '''
    Minimum airspeed relative for 3 seconds between 1000ft and 500ft AAL.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(1000, 500), 3, self.frequency,
                        hdf_duration),
            min_value,
        )


class AirspeedRelativeFor3Sec500To20FtMax(KeyPointValueNode):
    '''
    Maximum airspeed relative for 3 seconds between 500ft and 20ft AAL.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(500, 20), 3, self.frequency,
                        hdf_duration),
            max_value,
        )


class AirspeedRelativeFor3Sec500To20FtMin(KeyPointValueNode):
    '''
    Minimum airspeed relative for 3 seconds between 500ft and 20ft AAL.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(500, 20), 3, self.frequency,
                        hdf_duration),
            min_value,
        )


class AirspeedRelativeFor3Sec20FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum airspeed relative for 3 seconds between 20ft AAL and touchdown.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_to_kti(20, touchdowns), 3,
                        self.frequency, hdf_duration),
            max_value,
        )


class AirspeedRelativeFor3Sec20FtToTouchdownMin(KeyPointValueNode):
    '''
    Minimum airspeed relative for 3 seconds between 20ft AAL and touchdown.
    '''

    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_to_kti(20, touchdowns), 3,
                        self.frequency, hdf_duration),
            min_value,
        )


########################################
# Airspeed Minus VLS

class AirspeedMinusVLSAtTouchdown(KeyPointValueNode):
    '''
    Airspeed minus VLS at touchdown.
    '''

    name = 'Airspeed Minus VLS At Touchdown'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(spd_rel.array, touchdowns)


class AirspeedMinusVLS1000To500FtMax(KeyPointValueNode):
    '''
    Maximum airspeed minus VLS between 1000ft and 500ft AAL.
    '''

    name = 'Airspeed Minus VLS 1000 To 500 Ft Max'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class AirspeedMinusVLS1000To500FtMin(KeyPointValueNode):
    '''
    Minimum airspeed minus VLS between 1000ft and 500ft AAL.
    '''

    name = 'Airspeed Minus VLS 1000 To 500 Ft Min'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class AirspeedMinusVLS500To20FtMax(KeyPointValueNode):
    '''
    Maximum airspeed minus VLS between 500ft and 20ft AAL.
    '''

    name = 'Airspeed Minus VLS 500 To 20 Ft Max'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class AirspeedMinusVLS500To20FtMin(KeyPointValueNode):
    '''
    Minimum airspeed minus VLS between 500ft and 20ft AAL.
    '''

    name = 'Airspeed Minus VLS 500 To 20 Ft Min'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedMinusVLSFor3Sec1000To500FtMax(KeyPointValueNode):
    '''
    Maximum airspeed minus VLS for 3 seconds between 1000ft and 500ft AAL.
    '''

    name = 'Airspeed Minus VLS For 3 Sec 1000 To 500 Ft Max'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(1000, 500), 3, self.frequency,
                        hdf_duration),
            max_value,
        )


class AirspeedMinusVLSFor3Sec1000To500FtMin(KeyPointValueNode):
    '''
    Minimum airspeed minus VLS for 3 seconds between 1000ft and 500ft AAL.
    '''

    name = 'Airspeed Minus VLS For 3 Sec 1000 To 500 Ft Min'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(1000, 500), 3, self.frequency,
                        hdf_duration),
            min_value,
        )


class AirspeedMinusVLSFor3Sec500To20FtMax(KeyPointValueNode):
    '''
    Maximum airspeed minus VLS for 3 seconds between 500ft and 20ft AAL.
    '''

    name = 'Airspeed Minus VLS For 3 Sec 500 To 20 Ft Max'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(500, 20), 3, self.frequency,
                        hdf_duration),
            max_value,
        )


class AirspeedMinusVLSFor3Sec500To20FtMin(KeyPointValueNode):
    '''
    Minimum airspeed minus VLS for 3 seconds between 500ft and 20ft AAL.
    '''

    name = 'Airspeed Minus VLS For 3 Sec 500 To 20 Ft Min'
    units = ut.KT

    def derive(self,
               spd_rel=P('Airspeed Minus VLS For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            spd_rel.array,
            trim_slices(alt_aal.slices_from_to(500, 20), 3, self.frequency,
                        hdf_duration),
            min_value,
        )


########################################
# Airspeed: Flap


class AirspeedWithFlapMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Maximum airspeed recorded with each flap setting.

    The KPV name includes the source parameter used for the flap measurement
    taken:

    - 'Flap': based on the Flap Lever as recorded or a Flap Lever (Sythetic)
      generated from other available sources (for safety investigations).
    - 'Flap Including Transition': Flap setting includes the transition periods
      into and out of the flap detented position (for maintenance
      investigations on the more cautious side).
    - 'Flap Excluding Transition': Flap only where the detented flap position
      has been reached, excluding the transition periods (for maintenance
      investigations).
    '''

    NAME_FORMAT = 'Airspeed With %(parameter)s %(flap)s Max'
    NAME_VALUES = NAME_VALUES_LEVER.copy()
    NAME_VALUES.update({
        'parameter': [
            'Flap',
            'Flap Including Transition',
            'Flap Excluding Transition',
        ],
    })
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
            'Flap Including Transition',
            'Flap Excluding Transition',
        ), available) and all_of(('Airspeed', 'Fast'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               flap_inc_trans=M('Flap Including Transition'),
               flap_exc_trans=M('Flap Excluding Transition'),
               scope=S('Fast'),
               flap_angle=P('Flap Angle'),
               manufacturer=A('Manufacturer')
               ):
        
        # Masking single values that cause invalid events to trigger when 
        # not picked up by the cleanser
        for unmasked_slice in np.ma.clump_unmasked(airspeed.array):
            if unmasked_slice.stop - unmasked_slice.start < 2:
                airspeed.array[unmasked_slice.start] = np.ma.masked

        # We want to use flap lever detents if they are available, but we need
        # to ensure that the parameter is called flap with the name hack below:
        flap_avail = flap_lever or flap_synth
        if flap_avail:
            flap_avail.name = 'Flap'

        for flap in (flap_avail, flap_inc_trans, flap_exc_trans):
            if not flap:
                continue
            
            # Changing the current 'Flap Including Transition' parameter to monitor transitions 
            # to a reading of 0.1 degrees for Boeing - there is a requirement for an inspection
            # if airspeed is high for each flap setting, including the transition period, until 
            # the flap reaches the next position
            
            if manufacturer.value == 'Boeing' and flap.name == 'Flap Including Transition' and \
               flap_angle and len(flap.array) == len(flap_angle.array):
                
                initial_flap = deepcopy(flap.array)
                flap_array = flap.array
                transition = False
                for i in range(0, len(flap_array)-1):
                    if initial_flap[i] is not np.ma.masked and initial_flap[i+1] is not np.ma.masked and flap_angle.array[i+1] is not np.ma.masked:
                        if initial_flap.raw[i] > initial_flap.raw[i+1] and flap_angle.array[i+1] > initial_flap.raw[i+1] + 0.1:
                            transition = True
                            flap_array.raw[i+1] = flap_array.raw[i]
                            continue
                        if transition:
                            if initial_flap.raw[i] == initial_flap.raw[i+1] and flap_angle.array[i+1] > initial_flap.raw[i+1] + 0.1:
                                flap_array.raw[i+1] = flap_array.raw[i]
                            else:
                                transition = False
                data = self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope)
                for index, value, detent in data:
                    self.create_kpv(index, value, parameter=flap.name, flap=detent)
            
            else:
                # Fast scope traps flap changes very late on the approach and
                # raising flaps before 80 kt on the landing run.
                data = self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope)
                for index, value, detent in data:
                    self.create_kpv(index, value, parameter=flap.name, flap=detent)


class AirspeedWithFlapMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Minimum airspeed recorded with each Flap Lever setting.

    Based on Flap Lever for safety based investigations which primarily
    depend upon the pilot actions, rather than maintenance investigations
    which depend upon the actual flap surface position. Uses Flap Lever if
    available otherwise falls back to Flap Lever (Synthetic) which is
    established from other sources.
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)s Min'
    NAME_VALUES = NAME_VALUES_LEVER

    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Airspeed', 'Airborne'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               scope=S('Airborne')):

        # Airborne scope avoids deceleration on the runway "corrupting" the
        # minimum airspeed with landing flap.
        flap = flap_lever or flap_synth
        data = self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope)
        for index, value, detent in data:
            self.create_kpv(index, value, flap=detent)


class AirspeedWithFlapAndSlatExtendedMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Maximum airspeed recorded with Slat Extended and without Flap extended.

    Based upon Flap and Slat surface positions for maintenance based
    investigations where the results may be compared to Slat limiting speeds.
    Measurements which include Flap transitions can be found in other
    measurements.

    The KPV name includes the source parameter used for the flap measurement
    taken:

    - 'Flap Including Transition': Flap (and Slat) setting includes the
      transition periods  into and out of the flap detented position
      (for maintenance investigations on the more cautious side).
    - 'Flap Excluding Transition': Flap (and Slat) taken only where the
      detented flap position has been reached, excluding the transition
      periods (for maintenance investigations).
    '''

    NAME_FORMAT = 'Airspeed With %(parameter)s %(flap)s And Slat Extended Max'
    NAME_VALUES = {
        'parameter': [
            'Flap Including Transition',
            'Flap Excluding Transition',
        ],
        'flap': ['0'],
    }
    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        exc = all_of((
            'Flap Excluding Transition',
            'Slat Excluding Transition',
        ), available)
        inc = all_of((
            'Flap Including Transition',
            'Slat Including Transition',
        ), available)
        return (exc or inc) and all_of(('Airspeed', 'Fast'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               flap_exc_trsn=M('Flap Excluding Transition'),
               flap_inc_trsn=M('Flap Including Transition'),
               slat_exc_trsn=M('Slat Excluding Transition'),
               slat_inc_trsn=M('Slat Including Transition'),
               fast=S('Fast')):

        pairs = (flap_inc_trsn, slat_inc_trsn), (flap_exc_trsn, slat_exc_trsn)
        for flap, slat in pairs:
            # Fast scope traps flap changes very late on the approach and
            # raising flaps before 80 kt on the landing run.
            #
            # We take the intersection of the fast slices and the slices where
            # the slat was extended.
            array = np.ma.array(slat.array != '0', mask=slat.array.mask, dtype=int)
            scope = slices_and(fast.get_slices(), runs_of_ones(array))
            scope = S(items=[Section('', s, s.start, s.stop) for s in scope])

            data = self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope,
                                                include_zero=True)
            for index, value, detent in data:
                if not detent == '0':
                    continue  # skip as only interested when flap is retracted.
                self.create_kpv(index, value, parameter=flap.name, flap=detent)


class AirspeedWithFlapIncludingTransition20AndSlatFullyExtendedMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Maximum airspeed recorded with Slat Fully Extended and with Flap 20
    extended. Created specifically for the B777 and B787 which have a
    commanded slat only movement between flap 20 and 30

    Based upon Flap and Slat surface positions for maintenance based
    investigations where the results may be compared to Slat limiting speeds.
    Measurements which include Flap transitions can be found in other
    measurements.

    The KPV name includes the source parameter used for the flap measurement
    taken:

    - 'Flap Including Transition': Flap (and Slat) setting includes the
      transition periods  into and out of the flap detented position
      (for maintenance investigations on the more cautious side).
    - 'Flap Excluding Transition': Flap (and Slat) taken only where the
      detented flap position has been reached, excluding the transition
      periods (for maintenance investigations).
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        slat_only_transition = family and family.value in ('B777',
                                                           'B787')
        inc = all_of((
            'Flap Including Transition',
            'Slat Including Transition',
            'Airspeed',
            'Fast'
        ), available)
        return slat_only_transition and inc

    def derive(self,
               airspeed=P('Airspeed'),
               flap=M('Flap Including Transition'),
               slat=M('Slat Including Transition'),
               fast=S('Fast'),
               family=A('Family'),
               series=A('Series')):

        slat_fully_ext_value = max(six.iteritems(at.get_slat_map(family=family.value, series=series.value)), key=operator.itemgetter(0))[1]
        # Fast scope traps flap changes very late on the approach and
        # raising flaps before 80 kt on the landing run.
        #
        # We take the intersection of the fast slices and the slices where
        # the slat was extended.
        array = np.ma.array(slat.array == slat_fully_ext_value, mask=slat.array.mask, dtype=int)
        scope = slices_and(fast.get_slices(), runs_of_ones(array))
        scope = S(items=[Section('', s, s.start, s.stop) for s in scope])
        data = self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope,
                                            include_zero=True)
        for index, value, detent in data:
            if detent == '20':
                self.create_kpv(index, value)


class AirspeedWithFlapDuringClimbMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Maximum airspeed recorded during the Climb phase with each Flap setting.

    The KPV name includes the source parameter used for the flap measurement
    taken:

    - 'Flap': based on the Flap Lever as recorded or a Flap Lever (Sythetic)
      generated from other available sources (for safety investigations).
    - 'Flap Including Transition': Flap setting includes the transition periods
      into and out of the flap detented position (for maintenance
      investigations on the more cautious side).
    - 'Flap Excluding Transition': Flap only where the detented flap position
      has been reached, excluding the transition periods (for maintenance
      investigations).
    '''

    NAME_FORMAT = 'Airspeed With %(parameter)s %(flap)s During Climb Max'
    NAME_VALUES = NAME_VALUES_LEVER.copy()
    NAME_VALUES.update({
        'parameter': [
            'Flap',
            'Flap Including Transition',
            'Flap Excluding Transition',
        ],
    })
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
            'Flap Including Transition',
            'Flap Excluding Transition',
        ), available) and all_of(('Airspeed', 'Climb'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               flap_inc_trans=M('Flap Including Transition'),
               flap_exc_trans=M('Flap Excluding Transition'),
               scope=S('Climb')):

        # We want to use flap lever detents if they are available, but we need
        # to ensure that the parameter is called flap with the name hack below:
        flap_avail = flap_lever or flap_synth
        if flap_avail:
            flap_avail.name = 'Flap'

        for flap in (flap_avail, flap_inc_trans, flap_exc_trans):
            if not flap:
                continue
            # Fast scope traps flap changes very late on the approach and
            # raising flaps before 80 kt on the landing run.
            data = self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope)
            for index, value, detent in data:
                self.create_kpv(index, value, parameter=flap.name, flap=detent)


class AirspeedWithFlapDuringClimbMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Minimum airspeed recorded during the Climb phase with each Flap setting.

    Based on Flap Lever for safety based investigations which primarily
    depend upon the pilot actions, rather than maintenance investigations
    which depend upon the actual flap surface position. Uses Flap Lever if
    available otherwise falls back to Flap Lever (Synthetic) which is
    established from other sources.
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)s During Climb Min'
    NAME_VALUES = NAME_VALUES_LEVER

    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Airspeed', 'Climb'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               scope=S('Climb')):

        flap = flap_lever or flap_synth
        data = self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope)
        for index, value, detent in data:
            self.create_kpv(index, value, flap=detent)


class AirspeedWithFlapDuringDescentMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Maximum airspeed recorded during the Descent phase (down to the start of
    flare) with each Flap setting.

    The KPV name includes the source parameter used for the flap measurement
    taken:

    - 'Flap': based on the Flap Lever as recorded or a Flap Lever (Sythetic)
      generated from other available sources (for safety investigations).
    - 'Flap Including Transition': Flap setting includes the transition periods
      into and out of the flap detented position (for maintenance
      investigations on the more cautious side).
    - 'Flap Excluding Transition': Flap only where the detented flap position
      has been reached, excluding the transition periods (for maintenance
      investigations).
    '''

    NAME_FORMAT = 'Airspeed With %(parameter)s %(flap)s During Descent Max'
    NAME_VALUES = NAME_VALUES_LEVER.copy()
    NAME_VALUES.update({
        'parameter': [
            'Flap',
            'Flap Including Transition',
            'Flap Excluding Transition',
        ],
    })
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
            'Flap Including Transition',
            'Flap Excluding Transition',
        ), available) and all_of(('Airspeed', 'Descent'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               flap_inc_trans=M('Flap Including Transition'),
               flap_exc_trans=M('Flap Excluding Transition'),
               scope=S('Descent')):

        # We want to use flap lever detents if they are available, but we need
        # to ensure that the parameter is called flap with the name hack below:
        flap_avail = flap_lever or flap_synth
        if flap_avail:
            flap_avail.name = 'Flap'

        for flap in (flap_avail, flap_inc_trans, flap_exc_trans):
            if not flap:
                continue
            # Fast scope traps flap changes very late on the approach and
            # raising flaps before 80 kt on the landing run.
            data = self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope)
            for index, value, detent in data:
                self.create_kpv(index, value, parameter=flap.name, flap=detent)


class AirspeedWithFlapDuringDescentMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Minimum airspeed recorded during the Descent phase (down to the start of
    flare) with each Flap setting.

    Based on Flap Lever for safety based investigations which primarily
    depend upon the pilot actions, rather than maintenance investigations
    which depend upon the actual flap surface position. Uses Flap Lever if
    available otherwise falls back to Flap Lever (Synthetic) which is
    established from other sources.
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)s During Descent Min'
    NAME_VALUES = NAME_VALUES_LEVER
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Airspeed', 'Descent To Flare'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               scope=S('Descent To Flare')):

        flap = flap_lever or flap_synth
        data = self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope)
        for index, value, detent in data:
            self.create_kpv(index, value, flap=detent)


class AirspeedMinusFlapManoeuvreSpeedWithFlapDuringDescentMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Airspeed relative to the flap manoeuvre speed during the descent phase for
    each flap setting.

    Based on Flap Lever for safety based investigations which primarily
    depend upon the pilot actions, rather than maintenance investigations
    which depend upon the actual flap surface position. Uses Flap Lever if
    available otherwise falls back to Flap Lever (Synthetic) which is
    established from other sources.
    '''

    NAME_FORMAT = 'Airspeed Minus Flap Manoeuvre Speed With Flap %(flap)s During Descent Min'
    NAME_VALUES = NAME_VALUES_LEVER
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        core = all_of((
            'Airspeed Minus Flap Manoeuvre Speed',
            'Descent To Flare',
        ), available)
        flap = any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)
        return core and flap

    def derive(self,
               airspeed=P('Airspeed Minus Flap Manoeuvre Speed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               scope=S('Descent To Flare')):

        flap = flap_lever or flap_synth
        data = self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope)
        for index, value, detent in data:
            self.create_kpv(index, value, flap=detent)


class AirspeedAtFirstFlapExtensionWhileAirborne(KeyPointValueNode):
    '''
    Airspeed measured at the point of Flap Extension while airborne.
    '''

    units = ut.KT

    def derive(self,
               airspeed=P('Airspeed'),
               ff_ext=KTI('First Flap Extension While Airborne')):

        if ff_ext:
            index = ff_ext[-1].index
            self.create_kpv(index, value_at_index(airspeed.array, index))


class AirspeedSelectedFMCMinusFlapManoeuvreSpeed1000to5000FtMin(KeyPointValueNode):
    '''
    This KPV compares Airspeed Select (FMC) and Flap Manoeuvre Speed to
    identify an unsafe situation where the Speed Select (FMC) speed is below
    the Minimum Manoeuvre Speed immediately prior to VNAV engagement.
    '''

    units = ut.KT
    name = 'Airspeed Selected (FMC) Minus Flap Manoeuvre Speed 1000 to 5000 Ft Min'

    def derive(self,
               spd_sel=P('Airspeed Selected (FMC)'),
               flap_spd=P('Flap Manoeuvre Speed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               climbs=S('Climb')):
        alt_band = np.ma.masked_outside(alt_aal.array, 1000, 5000)
        alt_climb_sections = valid_slices_within_array(alt_band, climbs)
        array = spd_sel.array - flap_spd.array

        self.create_kpvs_within_slices(array, alt_climb_sections, min_value)


########################################
# Airspeed: Landing Gear


class AirspeedWithGearDownMax(KeyPointValueNode):
    '''
    Maximum airspeed observed while the landing gear down. Only records the
    single maximum value per flight.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               gear=M('Gear Down'),
               airs=S('Airborne')):

        gear.array[gear.array != 'Down'] = np.ma.masked
        gear_downs = np.ma.clump_unmasked(gear.array)
        self.create_kpv_from_slices(
            air_spd.array, slices_and(airs.get_slices(), gear_downs), max_value)


class AirspeedWhileGearRetractingMax(KeyPointValueNode):
    '''
    Maximum airspeed observed while the landing gear was retracting.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               gear_ret=S('Gear Retracting')):

        self.create_kpvs_within_slices(air_spd.array, gear_ret, max_value)


class AirspeedWhileGearExtendingMax(KeyPointValueNode):
    '''
    Maximum airspeed observed while the landing gear was extending.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               gear_ext=S('Gear Extending')):

        self.create_kpvs_within_slices(air_spd.array, gear_ext, max_value)


class AirspeedAtGearUpSelection(KeyPointValueNode):
    '''
    Airspeed measurement at the point of Gear Up Selection.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               gear_up_sel=KTI('Gear Up Selection')):

        self.create_kpvs_at_ktis(air_spd.array, gear_up_sel)


class AirspeedAtGearDownSelection(KeyPointValueNode):
    '''
    Airspeed measurement at the point of Gear Down Selection
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               gear_dn_sel=KTI('Gear Down Selection')):

        self.create_kpvs_at_ktis(air_spd.array, gear_dn_sel)


class MainGearOnGroundToNoseGearOnGroundDuration(KeyPointValueNode):
    '''
    The time duration between the main gear touching the ground and the nose
    gear touching the ground.
    '''

    units = ut.SECOND

    def derive(self, gog=P('Gear On Ground'), gogn=P('Gear (N) On Ground'),
               landings=S('Landing')):

        for landing in landings:
            gog_index = index_of_first_start(gog.array == 'Ground',
                                             landing.slice)
            gogn_index = index_of_first_start(gogn.array == 'Ground',
                                              landing.slice)
            if gogn_index is None or gog_index is None:
                self.warning("Gears not detected touching ground during landing phase")
            else:
                self.create_kpv(gog_index,
                                (gogn_index - gog_index) / self.frequency)


########################################
# Airspeed: Conf


class AirspeedWithConfigurationMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Maximum airspeed recorded for each configuration. Airbus only KPV.

    Conf settings (for all aircraft models) include:
    %(conf)s
    ''' % NAME_VALUES_CONF

    NAME_FORMAT = 'Airspeed With Configuration %(conf)s Max'
    NAME_VALUES = NAME_VALUES_CONF.copy()
    units = ut.KT

    def derive(self,
               airspeed=P('Airspeed'),
               conf=M('Configuration'),
               scope=S('Fast')):
        
        # Masking single values that cause invalid events to trigger when 
        # not picked up by the cleanser
        for unmasked_slice in np.ma.clump_unmasked(airspeed.array):
            if unmasked_slice.stop - unmasked_slice.start < 2:
                airspeed.array[unmasked_slice.start] = np.ma.masked        

        # Fast scope traps configuration changes very late on the approach and
        # before 80 kt on the landing run.
        data = self.flap_or_conf_max_or_min(conf, airspeed, max_value, scope)
        for index, value, detent in data:
            self.create_kpv(index, value, conf=detent)


class AirspeedRelativeWithConfigurationDuringDescentMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Minimum airspeed relative to the approach speed during from Descent to
    start of Flare for each aircraft Configuration  (established from external
    surface positions). Airbus only KPV.

    Conf settings (for all aircraft models) include:
    %(conf)s
    ''' % NAME_VALUES_CONF

    can_operate = aeroplane_only

    NAME_FORMAT = 'Airspeed Relative With Configuration %(conf)s During Descent Min'
    NAME_VALUES = NAME_VALUES_CONF.copy()
    units = ut.KT

    def derive(self,
               airspeed=P('Airspeed Relative'),
               conf=M('Configuration'),
               scope=S('Descent To Flare')):

        data = self.flap_or_conf_max_or_min(conf, airspeed, min_value, scope)
        for index, value, detent in data:
            self.create_kpv(index, value, conf=detent)


########################################
# Airspeed: Speedbrakes


class AirspeedWithSpeedbrakeDeployedMax(KeyPointValueNode):
    '''
    Maximum airspeed measured while Speedbrake parameter is in excess of %sdeg
    ''' % SPOILER_DEPLOYED

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               spdbrk=P('Speedbrake')):

        spdbrk.array[spdbrk.array.data < SPOILER_DEPLOYED] = np.ma.masked
        spoiler_deployeds = np.ma.clump_unmasked(spdbrk.array)
        self.create_kpv_from_slices(
            air_spd.array, spoiler_deployeds, max_value)


########################################
# Airspeed: Thrust Reversers


class AirspeedWithThrustReversersDeployedMin(KeyPointValueNode):
    '''
    Minimum true airspeed measured with Thrust Reversers deployed and the
    maximum of either engine's EPR measurements above %.2f%% or N1 measurements
    above %d%%
    ''' % (REVERSE_THRUST_EFFECTIVE_EPR, REVERSE_THRUST_EFFECTIVE_N1)

    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Airspeed True', 'Thrust Reversers', 'Landing'),
                      available) and \
            any_of(('Eng (*) EPR Max', 'Eng (*) N1 Max'), available)

    def derive(self,
               air_spd=P('Airspeed True'),
               tr=M('Thrust Reversers'),
               eng_epr=P('Eng (*) EPR Max'),  # must come before N1 where available
               eng_n1=P('Eng (*) N1 Max'),
               landings=S('Landing')):

        for landing in landings:
            if eng_epr:
                power = eng_epr
                threshold = REVERSE_THRUST_EFFECTIVE_EPR
            else:
                power = eng_n1
                threshold = REVERSE_THRUST_EFFECTIVE_N1
            high_rev = thrust_reversers_working(landing, power, tr, threshold)
            self.create_kpvs_within_slices(air_spd.array, high_rev, min_value)


class AirspeedAtThrustReversersSelection(KeyPointValueNode):
    '''
    IAS at thrust reversers selection.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               tr=M('Thrust Reversers'),
               landings=S('Landing')):

        slices = [s.slice for s in landings]  # TODO: use landings.get_slices()
        # TODO: Replace with positive state rather than "Not Stowed"
        to_scan = clump_multistate(tr.array, 'Stowed', slices, condition=False)
        self.create_kpv_from_slices(air_spd.array, to_scan, max_value)


########################################
# Airspeed: Other


class AirspeedVacatingRunway(KeyPointValueNode):
    '''
    True airspeed while vacating the runway.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed True'),
               off_rwy=KTI('Landing Turn Off Runway')):
        '''
        True airspeed while vacating the runway, which is extended below the
        minimum range of the indicated airspeed specifically for this type of
        event. See the derived parameter for details of how groundspeed or
        acceleration data is used to cover the landing phase.
        '''
        self.create_kpvs_at_ktis(air_spd.array, off_rwy)


class AirspeedDuringRejectedTakeoffMax(KeyPointValueNode):
    '''
    Maximum airspeed during RTO.

    Although useful, please use Groundspeed During Rejected Takeoff Max.

    For most aircraft the Airspeed sensors are not able to record accurately
    below 60 knots, meaning lower speed RTOs may be missed. The Groundspeed
    version will work off the Longitudinal accelerometer if Groundspeed is
    not recorded.
    '''

    units = ut.KT

    def derive(self, air_spd=P('Airspeed'), rtos=S('Rejected Takeoff')):
        #NOTE: Use 'Groundspeed During Rejected Takeoff Max' in preference
        self.create_kpvs_within_slices(air_spd.array, rtos, max_value)


class AirspeedBelow10000FtDuringDescentMax(KeyPointValueNode):
    '''
    Maximum airspeed measured below 10,000ft pressure altitude (ouside of
    USA) or below 10,000ft QNH (inside of USA where airport identified)
    during the Descent phase of flight.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std=P('Altitude STD Smoothed'),
               alt_qnh=P('Altitude QNH'),
               ldg_apt=A('FDR Landing Airport'),
               descent=S('Descent')):
        '''
        Outside the USA 10,000 ft relates to flight levels, whereas FAA regulations
        (and possibly others we don't currently know about) relate to height above
        sea level (QNH) hence the options based on landing airport location.
    
        In either case, we apply some hysteresis to prevent nuisance retriggering
        which can arise if the aircraft is sitting on the 10,000ft boundary.
        '''
        country = None
        if ldg_apt.value:
            country = ldg_apt.value.get('location', {}).get('country')

        alt = alt_qnh.array if country == 'United States' else alt_std.array
        alt = hysteresis(alt, HYSTERESIS_FPALT)

        height_bands = np.ma.clump_unmasked(np.ma.masked_greater(alt, 10000))
        descent_bands = slices_and(height_bands, descent.get_slices())
        self.create_kpvs_within_slices(air_spd.array, descent_bands, max_value)


class AirspeedTopOfDescentTo10000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed measured from Top of Descent down to 10,000ft pressure
    altitude (ouside of USA) or below 10,000ft QNH (inside of USA where
    airport identified) during the Descent phase of flight.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std=P('Altitude STD Smoothed'),
               alt_qnh=P('Altitude QNH'),
               ldg_apt=A('FDR Landing Airport'),
               descent=S('Descent')):
        '''
        Outside the USA 10,000 ft relates to flight levels, whereas FAA regulations
        (and possibly others we don't currently know about) relate to height above
        sea level (QNH) hence the options based on landing airport location.
    
        In either case, we apply some hysteresis to prevent nuisance retriggering
        which can arise if the aircraft is sitting on the 10,000ft boundary.
        '''
        country = None
        if ldg_apt.value:
            country = ldg_apt.value.get('location', {}).get('country')

        alt = alt_qnh.array if country == 'United States' else alt_std.array
        if np.ma.clump_unmasked(alt):
            alt = hysteresis(alt, HYSTERESIS_FPALT)
    
            height_bands = np.ma.clump_unmasked(np.ma.masked_less(repair_mask(alt),
                                                                  10000))
            descent_bands = slices_and(height_bands, descent.get_slices())
            self.create_kpvs_within_slices(air_spd.array, descent_bands, max_value)


class AirspeedTopOfDescentTo4000FtMax(KeyPointValueNode):
    '''
    Maximum airspeed measured from Top of Descent down to 4,000ft pressure
    altitude (ouside of USA) or below 10,000ft QNH (inside of USA where
    airport identified) during the Descent phase of flight.

    Outside the USA 4,000 ft relates to flight levels, whereas FAA regulations
    (and possibly others we don't currently know about) relate to height above
    sea level (QNH) hence the options based on landing airport location.

    In either case, we apply some hysteresis to prevent nuisance retriggering
    which can arise if the aircraft is sitting on the 4,000ft boundary.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std=P('Altitude STD Smoothed'),
               alt_qnh=P('Altitude QNH'),
               ldg_apt=A('FDR Landing Airport'),
               descent=S('Descent')):

        country = None
        if ldg_apt.value:
            country = ldg_apt.value.get('location', {}).get('country')

        alt = alt_qnh.array if country == 'United States' else alt_std.array
        if np.ma.clump_unmasked(alt):
            alt = hysteresis(alt, HYSTERESIS_FPALT)

            height_bands = np.ma.clump_unmasked(np.ma.masked_less(repair_mask(alt),
                                                                  4000))
            descent_bands = slices_and(height_bands, descent.get_slices())
            self.create_kpvs_within_slices(air_spd.array, descent_bands, max_value)


class AirspeedTopOfDescentTo4000FtMin(KeyPointValueNode):
    '''
    Minimum airspeed measured from Top of Descent down to 4,000ft pressure
    altitude (ouside of USA) or below 10,000ft QNH (inside of USA where
    airport identified) during the Descent phase of flight.

    Outside the USA 4,000 ft relates to flight levels, whereas FAA regulations
    (and possibly others we don't currently know about) relate to height above
    sea level (QNH) hence the options based on landing airport location.

    In either case, we apply some hysteresis to prevent nuisance retriggering
    which can arise if the aircraft is sitting on the 4,000ft boundary.
    '''

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std=P('Altitude STD Smoothed'),
               alt_qnh=P('Altitude QNH'),
               ldg_apt=A('FDR Landing Airport'),
               descent=S('Descent')):

        country = None
        if ldg_apt.value:
            country = ldg_apt.value.get('location', {}).get('country')

        alt = alt_qnh.array if country == 'United States' else alt_std.array       
        
        if np.ma.clump_unmasked(alt):
            alt = hysteresis(alt, HYSTERESIS_FPALT)

            height_bands = np.ma.clump_unmasked(np.ma.masked_less(repair_mask(alt),
                                                                  4000))
            descent_bands = slices_and(height_bands, descent.get_slices())
            self.create_kpvs_within_slices(air_spd.array, descent_bands, min_value)


class AirspeedDuringLevelFlightMax(KeyPointValueNode):
    '''
    Maximum airspeed recorded during level flight (less than %sfpm).
    ''' % VERTICAL_SPEED_FOR_LEVEL_FLIGHT

    units = ut.KT

    def derive(self,
               air_spd=P('Airspeed'),
               lvl_flt=S('Level Flight')):

        for section in lvl_flt:
            self.create_kpv(*max_value(air_spd.array, section.slice))


class AirspeedAboveFL200Max(KeyPointValueNode):
    '''
    Maximum airspeed above FL200 (Alt STD Smoothed)
    '''
    name = 'Airspeed Above FL200 Max'
    units = ut.KT

    def derive(self, air_spd= P('Airspeed'), alt=P('Altitude STD Smoothed')):
        
        alt.array = nearest_neighbour_mask_repair(alt.array)
        self.create_kpvs_within_slices(air_spd.array,
                                       alt.slices_above(20000), max_value)    
        
class AirspeedAboveFL200Min(KeyPointValueNode):
    '''
    Minimum airspeed above FL200 (Alt STD Smoothed)
    '''
    name = 'Airspeed Above FL200 Min'
    units = ut.KT

    def derive(self, air_spd= P('Airspeed'), alt=P('Altitude STD Smoothed')):
        
        alt.array = nearest_neighbour_mask_repair(alt.array)
        self.create_kpvs_within_slices(air_spd.array,
                                       alt.slices_above(20000), min_value)  
        

##############################################################################
# Airspeed Autorotation
class AirspeedDuringAutorotationMax(KeyPointValueNode):
    '''
    Maximum airspeed during autorotation (helicopter only)
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, airspeed=P('Airspeed'), phase=S('Autorotation')):
        self.create_kpvs_within_slices(airspeed.array, phase, max_value)


class AirspeedDuringAutorotationMin(KeyPointValueNode):
    '''
    Minimum airspeed during autorotation (helicopter only)
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, airspeed=P('Airspeed'), phase=S('Autorotation')):
        self.create_kpvs_within_slices(airspeed.array, phase, min_value)


##############################################################################
# Alpha Floor


class AlphaFloorDuration(KeyPointValueNode):
    '''
    Duration for which A.FLOOR was active.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):

        return 'Airborne' in available and any_of(('Alpha Floor', 'FMA AT Information'), available)

    def derive(self,
               alpha_floor=M('Alpha Floor'),
               autothrottle_info=M('FMA AT Information'),
               airs=S('Airborne')):

        combined = vstack_params_where_state(
            (alpha_floor, 'Engaged'),
            (autothrottle_info, 'Alpha Floor'),
        ).any(axis=0)
        air_alpha_floor = slices_and(airs.get_slices(), runs_of_ones(combined))
        self.create_kpvs_from_slice_durations(air_alpha_floor, self.hz)


##############################################################################
# Angle of Attack


class AOADuringGoAroundMax(KeyPointValueNode):
    '''
    Maximum Angle Of Attack during Go Around.
    '''

    name = 'AOA During Go Around Max'
    units = ut.DEGREE

    def derive(self,
               aoa=P('AOA'),
               go_arounds=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(aoa.array, go_arounds, max_value)


class AOAWithFlapMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Maximum Angle Of Attack with all flap angles.
    '''

    NAME_FORMAT = 'AOA With Flap %(flap)s Max'
    NAME_VALUES = NAME_VALUES_LEVER
    name = 'AOA With Flap Max'
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Loss of Control. Pitch/Angle of Attack vs stall angles"
    
        This is an adaptation of the airspeed algorithm, used to determine peak
        AOA vs flap. It may not be possible to obtain stalling angle of attack
        figures to set event thresholds, but a threshold based on in-service data
        may suffice.
        '''
        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('AOA', 'Airborne'), available)

    def derive(self,
               aoa=P('AOA'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               scope=S('Airborne')):

        # Airborne scope avoids triggering during the takeoff or landing runs.
        flap = flap_lever or flap_synth
        data = self.flap_or_conf_max_or_min(flap, aoa, max_value, scope,
                                            include_zero=True)
        for index, value, detent in data:
            self.create_kpv(index, value, flap=detent)


class AOAWithFlapDuringClimbMax(KeyPointValueNode):
    '''
    Maximum Angle of Attack During Climb with flaps extended.
    '''

    name = 'AOA With Flap During Climb Max'
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return (all_of(('AOA', 'Climbing'), available) and
                any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available))

    def derive(self,
               aoa=P('AOA'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               climbs=S('Climbing')):
        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        aoa_flap = np.ma.masked_where(retracted, aoa.array)
        self.create_kpvs_within_slices(aoa_flap, climbs, max_value)


class AOAWithFlapDuringDescentMax(KeyPointValueNode):
    '''
    Maximum Angle of Attack During Descent with flaps extended.
    '''

    name = 'AOA With Flap During Descent Max'
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return (all_of(('AOA', 'Descending'), available) and
                any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available))

    def derive(self,
               aoa=P('AOA'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               descends=S('Descending')):
        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        aoa_flap = np.ma.masked_where(retracted, aoa.array)
        self.create_kpvs_within_slices(aoa_flap, descends, max_value)


##############################################################################
class ThrustReversersDeployedDuration(KeyPointValueNode):
    '''
    Measure the duration (secs) which the thrust reverses were deployed for.
    0 seconds represents no deployment at landing.
    '''

    units = ut.SECOND

    def derive(self, tr=M('Thrust Reversers'), landings=S('Landing')):
        for landing in landings:
            tr_in_ldg = tr.array[landing.slice]
            dur_deployed = np.ma.sum(tr_in_ldg == 'Deployed') / tr.frequency
            dep_start = find_edges_on_state_change('Deployed', tr_in_ldg)
            if dur_deployed and dep_start:
                index = dep_start[0] + landing.slice.start
            else:
                index = landing.slice.start
            self.create_kpv(index, dur_deployed)


class ThrustReversersDeployedDuringFlightDuration(KeyPointValueNode):
    '''
    Measure the duration (secs) which the thrust reverses were deployed for
    in flight. 0 seconds represents no deployment during flight.
    '''

    units = ut.SECOND

    def derive(self, tr=M('Thrust Reversers'), airs=S('Airborne')):
        for air in airs:
            tr_in_air = tr.array[air.slice]
            dur_deployed = np.ma.sum(tr_in_air == 'Deployed') / tr.frequency
            dep_start = find_edges_on_state_change('Deployed', tr_in_air)
            if dur_deployed and dep_start:
                index = dep_start[0] + air.slice.start
            else:
                index = air.slice.start
            self.create_kpv(index, dur_deployed)


class ThrustReversersCancelToEngStopDuration(KeyPointValueNode):
    '''
    Measure the duration (secs) between the thrust reversers being cancelled and
    the engines being shutdown.

    The scope is limited to the engine running period to avoid spurious
    indications of thrust reverser operation while the engine is not running,
    as can happen on some aircraft types.
    '''

    units = ut.SECOND

    def derive(self, tr=M('Thrust Reversers'),
               eng_starts=KTI('Eng Start'),
               eng_stops=KTI('Eng Stop')):
        try:
            start = eng_starts.get_first().index
        except AttributeError:
            start = 0
        try:
            stop = eng_stops.get_last().index
        except AttributeError:
            # If engine did not stop, there is no period between thrust
            # reversers being cancelled and the engine stop
            return
        cancels = find_edges_on_state_change(
            'Deployed', tr.array[start:stop], change='leaving')
        if cancels:
            # TRs were cancelled before engine stopped
            cancel_index = cancels[-1] + start
            eng_stop_index = eng_stops.get_next(cancel_index).index
            self.create_kpv(eng_stop_index,
                            (eng_stop_index - cancel_index) / self.frequency)


class TouchdownToThrustReversersDeployedDuration(KeyPointValueNode):
    '''
    Duration between touchdown and thrust reversers deployment.
    '''

    units = ut.SECOND

    def derive(self,
               tr=M('Thrust Reversers'),
               landings=S('Landing'),
               touchdowns=KTI('Touchdown')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Landing (Lateral) Reverse thrust delay - time delay.
        Selection more than 3sec after main wheel t/d."
    
        Note: 3 second threshold may be applied to derive an event from this KPV.
        '''
        for landing in landings:
            # Only interested in first opening of reversers on this landing:
            deploys = clump_multistate(tr.array, 'Deployed', landing.slice)
            try:
                deployed = deploys[0].start
            except IndexError:
                continue
            touchdown = touchdowns.get_first(within_slice=landing.slice)
            if not touchdown:
                continue
            self.create_kpv(deployed, (deployed - touchdown.index) / tr.hz)


class TouchdownToSpoilersDeployedDuration(KeyPointValueNode):
    '''
    Duration between touchdown and spoilers deployment.
    '''

    units = ut.SECOND

    def derive(self, brake=M('Speedbrake Selected'),
               lands=S('Landing'), tdwns=KTI('Touchdown')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Landing (Lateral) Late spoiler deployment - time delay".
        '''
        deploys = find_edges_on_state_change('Deployed/Cmd Up', brake.array, phase=lands)
        for land in lands:
            for deploy in deploys:
                if not is_index_within_slice(deploy, land.slice):
                    continue
                for tdwn in tdwns:
                    if not is_index_within_slice(tdwn.index, land.slice):
                        continue
                    self.create_kpv(deploy, (deploy - tdwn.index) / brake.hz)


class TrackDeviationFromRunway1000To500Ft(KeyPointValueNode):
    '''
    Track deviation from the runway centreline from 1000 to 500 feet.

    Helps establishing the stable criteria for IFR below 1000ft.

    Includes large deviations recoreded when aircraft turns onto runway at
    altitudes below 1000ft.
    '''

    units = ut.DEGREE

    def derive(self,
               track_dev=P('Track Deviation From Runway'),
               alt_aal=P('Altitude AAL')):

        alt_bands = alt_aal.slices_from_to(1000, 500)
        self.create_kpvs_within_slices(
            track_dev.array,
            alt_bands,
            max_abs_value,
        )


class TrackDeviationFromRunway500To300Ft(KeyPointValueNode):
    '''
    Track deviation from the runway centreline from 500 to 300 feet.

    Helps establishing the stable criteria for VFR below 500ft.

    Includes large deviations recorded when aircraft turns onto runway at
    altitudes below 500ft, but should be stable by 300ft.
    '''

    units = ut.DEGREE

    def derive(self,
               track_dev=P('Track Deviation From Runway'),
               alt_aal=P('Altitude AAL')):

        alt_bands = alt_aal.slices_from_to(500, 300)
        self.create_kpvs_within_slices(
            track_dev.array,
            alt_bands,
            max_abs_value,
        )


class TrackDeviationFromRunway300FtToTouchdown(KeyPointValueNode):
    '''
    Track deviation from the runway centreline from 300 to 0 feet.

    Helps establishing the FAA stable criteria for a late roll onto runway
    heading.
    '''

    units = ut.DEGREE

    def derive(self,
               track_dev=P('Track Deviation From Runway'),
               alt_aal=P('Altitude AAL')):

        alt_bands = alt_aal.slices_from_to(300, 0)
        self.create_kpvs_within_slices(
            track_dev.array,
            alt_bands,
            max_abs_value,
        )


##############################################################################
# TOGA Usage


class TOGASelectedDuringFlightDuration(KeyPointValueNode):
    '''
    Duration for which TOGA was selected in flight (excluding go-around).
    '''

    name = 'TOGA Selected During Flight Not Go Around Duration'
    units = ut.SECOND

    def derive(self,
               toga=M('Takeoff And Go Around'),
               go_arounds=S('Go Around And Climbout'),
               airborne=S('Airborne')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Loss of Control - Unexpected TOGA power selection in flight (except for
        a go-around)"
    
        Note: This covers the entire airborne phase excluding go-arounds.
        '''
        to_scan = slices_and(
            [s.slice for s in airborne],
            slices_not(
                [s.slice for s in go_arounds],
                begin_at=airborne[0].slice.start,
                end_at=airborne[-1].slice.stop,
            ),
        )
        self.create_kpvs_where(toga.array == 'TOGA', toga.hz,
                               phase=to_scan, exclude_leading_edge=True)


class TOGASelectedDuringGoAroundDuration(KeyPointValueNode):
    '''
    Duration for which TOGA was selected during go-around.
    '''

    name = 'TOGA Selected During Go Around Duration'
    units = ut.SECOND

    def derive(self, toga=M('Takeoff And Go Around'),
               go_arounds=S('Go Around And Climbout')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Loss of Control - TOGA power selection in flight (Go-arounds need to be
        kept as a separate case)."
    
        Align to Takeoff And Go Around for most accurate state change indices.
        '''
        self.create_kpvs_where(toga.array == 'TOGA',
                               toga.hz, phase=go_arounds)


##############################################################################


class LiftoffToClimbPitchDuration(KeyPointValueNode):
    '''
    Time between liftoff and reaching climb pitch (10deg).

    An endpoint of a minute after liftoff was added to avoid triggering well
    after the period of interest, and a pre-liftoff extension included for
    cases which rotate quickly and reach 10deg before liftoff !
    '''

    units = ut.SECOND

    def derive(self, pitch=P('Pitch'), lifts=KTI('Liftoff')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Take-Off (Longitudinal), Slow climb out after rotation and
        slow rotation."
    
        This KPV originally used a threshold of 12.5 deg nose up, as suggested by
        the CAA, however it was found that some corporate operators do not
        achieve this attitude, so a lower threshold of 10deg was adopted.
        '''
        for lift in lifts:
            pitch_up_idx = index_at_value(pitch.array, 10.0,
                                          _slice=slice(lift.index - 5 * pitch.hz,
                                                       lift.index + 60.0 * pitch.hz))
            if pitch_up_idx:
                duration = (pitch_up_idx - lift.index) / pitch.hz
                self.create_kpv(pitch_up_idx, duration)


##############################################################################
# Landing Gear


##################################
# Braking


class BrakeTempDuringTaxiInMax(KeyPointValueNode):
    '''
    Maximum temperature of any brake during taxi in.
    '''

    units = ut.CELSIUS

    def derive(self, brakes=P('Brake (*) Temp Max'), taxiin=S('Taxi In')):
        self.create_kpvs_within_slices(brakes.array, taxiin, max_value)


class BrakeTempAfterTouchdownDelta(KeyPointValueNode):
    '''
    Difference in the average brake temperature after touchdown.
    '''

    units = ut.CELSIUS

    def derive(self, brakes=P('Brake (*) Temp Avg'), touchdowns=S('Touchdown')):
        touchdown = touchdowns.get_last().index
        max_temp_idx = np.ma.argmax(brakes.array[touchdown:]) + touchdown
        max_temp = value_at_index(brakes.array, max_temp_idx)
        min_temp = np.ma.min(brakes.array[touchdown:max_temp_idx + 1])
        self.create_kpv(max_temp_idx, max_temp - min_temp)


class BrakePressureInTakeoffRollMax(KeyPointValueNode):
    '''
    Maximum brake pressure during takeoff roll.

    This is to capture scenarios where the brake is accidentally used when
    using the rudder (dragging toes on pedals)."
    '''

    units = None  # FIXME

    def derive(self, bp=P('Brake Pressure'),
               rolls=S('Takeoff Roll Or Rejected Takeoff')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Take off (Lateral)". Primary Brake pressure during ground
        roll. Could also be applicable to longitudinal excursions on take-off.
        '''
        self.create_kpvs_within_slices(bp.array, rolls, max_value)


# TODO: Consider renaming this as 'delayed' implies it is already late!
class DelayedBrakingAfterTouchdown(KeyPointValueNode):
    '''
    Duration of braking after the aircraft has touched down.

    An event using this KPV can be used for detecting delayed braking. The KPV
    measures the time of deceleration between V-10 kt and V-60 kt where V is
    the ground speed at touchdown.
    '''

    units = ut.SECOND

    def derive(self,
               lands=S('Landing'),
               gs=P('Groundspeed'),
               tdwns=KTI('Touchdown')):
        '''
        Reverse thrust is usually applied after the main gear touches down,
        possibly along with the autobrake, to reduce the speed of the aircraft. If
        the deceleration of the aircraft is slow, it is a possible indication of
        delay in use of reverse thrust.
    
        Reference was made to the following documentation to assist with the
        development of this algorithm:
    
        - A320 Flight Profile Specification
        - A321 Flight Profile Specification
        '''
        for land in lands:
            for tdwn in tdwns.get(within_slice=land.slice):
                gs_td = value_at_index(gs.array, tdwn.index)
                if gs_td is None:
                    continue
                minus_10 = index_at_value(gs.array, gs_td - 10.0, land.slice)
                minus_60 = index_at_value(gs.array, gs_td - 60.0, land.slice)
                if minus_10 is None or minus_60 is None:
                    continue
                self.create_kpv(minus_60, (minus_60 - minus_10) / gs.hz)


class AutobrakeRejectedTakeoffNotSetDuringTakeoff(KeyPointValueNode):
    '''
    Duration where the Autobrake Selected RTO parameter is not in "Selected"
    state during takeoff phase.
    '''

    units = ut.SECOND

    def derive(self,
               ab_rto=M('Autobrake Selected RTO'),
               takeoff=S('Takeoff Roll Or Rejected Takeoff')):

        # In order to avoid false positives, so we assume masked values are
        # Selected.
        not_selected = (ab_rto.array != 'Selected').filled(False)
        self.create_kpvs_where(
            not_selected,
            ab_rto.hz, phase=takeoff)


##############################################################################
# Altitude


########################################
# Altitude: General


class AltitudeMax(KeyPointValueNode):
    '''
    Maximum pressure altitude (STD smoothed) recorded during flight.
    '''

    units = ut.FT

    def derive(self,
               alt_std=P('Altitude STD Smoothed'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(alt_std.array, airborne, max_value)


class AltitudeDuringGoAroundMin(KeyPointValueNode):
    '''
    The minimum altitude above the local airfield level during the go-around.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               go_arounds=S('Go Around And Climbout')):
        '''
        Note: Was defined as the altitude above the local airfield level at the
        minimum altitude point of the go-around, but this was confusing as this
        is not the lowest altitude point if the go-around occurs over uneven
        ground.
        '''
        self.create_kpvs_within_slices(alt_aal.array, go_arounds, min_value)


class HeightAtGoAround(KeyPointValueNode):
    '''
    The radio altitude at the point of the go-around.
    '''

    units = ut.FT

    def derive(self,
               alt_rad=P('Altitude Radio'),
               go_arounds=KTI('Go Around')):

        self.create_kpvs_at_ktis(alt_rad.array, go_arounds)


class AltitudeOvershootAtSuspectedLevelBust(KeyPointValueNode):
    '''
    Measures the altitude overshoot through an altitude at which the aircraft subsequently maintains level flight.
    '''

    units = ut.FT

    def derive(self, alt_std=P('Altitude STD Smoothed'), alt_aal=P('Altitude AAL')):
        '''
        FDS refined this KPV as part of the UK CAA Significant Seven programme.
        
        "Airborne Conflict (Mid-Air Collision) Level Busts (>300ft from an
        assigned level) It would be useful if this included overshoots of cleared
        level, i.e. a reversal of more than 300ft".
        
        Undershoots under 3000 ft are excluded due to inconsistent Go Around
        behaviour.
        
        Q: Could we compare against Altitude Selected to know if the aircraft should
        be climbing or descending?
        '''
        bust_min = 300  # ft
        bust_samples = 3 * 60 * self.frequency  # 3 mins # + 1 min to account for late level flight stabilisation.

        alt_diff = np.ma.abs(np.ma.diff(alt_std.array)) < (2 * alt_std.hz)

        for idx, val in zip(*cycle_finder(alt_std.array, min_step=300)):

            if self and (idx - self[-1].index) < bust_samples:
                # avoid duplicates
                continue

            fwd_slice = slice(idx, idx + bust_samples)
            rev_slice = slice(idx, idx - bust_samples, -1)

            for min_bust_val in (val + bust_min, val - bust_min):
                # check bust value is exceeded before and after
                fwd_idx = index_at_value(alt_std.array, min_bust_val, _slice=fwd_slice)
                if not fwd_idx:
                    continue

                rev_idx = index_at_value(alt_std.array, min_bust_val, _slice=rev_slice)
                if not rev_idx:
                    continue

                fwd_slice = slice(fwd_idx, fwd_slice.stop)
                rev_slice = slice(rev_idx, rev_slice.stop, -1)

                lvl_off_vals = []
                for bust_slice in (fwd_slice, rev_slice):
                    # find level off indices
                    #alt_diff = np.ma.abs(np.ma.diff(alt_std.array[bust_slice])) < (2 * alt_std.hz)
                    try:
                        lvl_off_val = alt_std.array[bust_slice.start + ((bust_slice.step or 1) * np.ma.where(alt_diff[bust_slice])[0][0])]
                    except IndexError:
                        continue
                    lvl_off_vals.append(val - lvl_off_val)

                if not lvl_off_vals:
                    continue

                lvl_off_val = min(lvl_off_vals, key=lambda x: abs(x))

                if val < 3000 and lvl_off_val < val:
                    # Undershoots under 3000 ft are excluded due to inconsistent Go Around behaviour.
                    self.info('Overshoot not detected: Undershoot below 3000ft')
                    continue

                if abs(lvl_off_val) > val * 0.9:
                    # Ignore lvl_off_val more than 90% of val as indicates
                    # short hop flights
                    self.info('Overshoot not detected: Exceeds 90% of height')
                    continue

                max_level_off_samples = 60 * self.frequency  # 1 minuete
                level_off_slices = slices_remove_small_slices(slices_remove_small_gaps(runs_of_ones(alt_diff)), count=max_level_off_samples)
                # check hasn't been level for over max_level_off_samples
                if is_index_within_slices(idx, level_off_slices):
                    self.info('Overshoot not detected: Index in period of level flight')
                    continue

                self.create_kpv(idx, lvl_off_val)


class CabinAltitudeWarningDuration(KeyPointValueNode):
    '''
    The duration of the Cabin Altitude Warning signal.
    '''

    units = ut.SECOND

    def derive(self,
               cab_warn=M('Cabin Altitude Warning'),
               airborne=S('Airborne')):

        self.create_kpvs_where(cab_warn.array == 'Warning',
                               cab_warn.hz, phase=airborne)


class AltitudeDuringCabinAltitudeWarningMax(KeyPointValueNode):
    '''
    The maximum aircraft altitude when the Cabin Altitude Warning was sounding.
    '''

    units = ut.FT

    def derive(self,
               cab_warn=M('Cabin Altitude Warning'),
               airborne=S('Airborne'),
               alt=P('Altitude STD Smoothed')):

        # XXX: Grr... no test case and use of incorrect state
        # TODO: warns = runs_of_ones(cab_warn.array == 'Warning')
        warns = np.ma.clump_unmasked(np.ma.masked_equal(cab_warn.array, 0))
        air_warns = slices_and(warns, airborne.get_slices())
        self.create_kpvs_within_slices(alt.array, air_warns, max_value)


class CabinAltitudeMax(KeyPointValueNode):
    '''
    The maximum Cabin Altitude - applies on every flight.
    '''

    units = ut.FT

    def derive(self,
               cab_alt=P('Cabin Altitude'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(cab_alt.array, airborne, max_value)


class AltitudeSTDMax(KeyPointValueNode):
    '''
    Maximum altitude STD.
    '''

    name = 'Altitude STD Max'
    units = ut.FT

    def derive(self, alt_std=P('Altitude STD')):
        self.create_kpv(*max_value(alt_std.array))


class AltitudeInCruiseAverage(KeyPointValueNode):
    '''
    Average pressure altitude (not smoothed) during all the cruise phases.
    '''

    units = ut.FT

    def derive(self, alt_std=P('Altitude STD'), cruises=S('Cruise')):

        if len(cruises):
            av_array = np_ma_zeros_like(alt_std.array, mask=True)
            for cruise in [c.slice for c in cruises]:
                av_array[cruise] = alt_std.array[cruise]
            value = np.ma.mean(av_array)
            index = cruises[-1].slice.stop - 1
            self.create_kpv(index, value)


########################################
# Altitude: Helicopter


class AltitudeDensityMax(KeyPointValueNode):
    '''
    Maximum altitude density.
    '''

    units = ut.FT

    can_operate = helicopter_only

    def derive(self, alt_density=P('Altitude Density'), airborne=S('Airborne')):
        self.create_kpv_from_slices(
            alt_density.array,
            airborne.get_slices(),
            max_value
        )


class AltitudeRadioDuringAutorotationMin(KeyPointValueNode):
    '''
    Minimum altitude (radio) during autorotation.
    '''

    units = ut.FT

    can_operate = helicopter_only

    def derive(self, alt_rad=P('Altitude Radio'), autorotation=S('Autorotation')):
        self.create_kpvs_within_slices(alt_rad.array, autorotation, min_value)


class AltitudeDuringCruiseMin(KeyPointValueNode):
    '''
    Minimum altitude (AGL) recorded during cruise (helicopter only).
    '''

    units = ut.FT
    can_operate = helicopter_only

    def derive(self, alt_agl=P('Altitude AGL'), cruise=S('Cruise')):
        self.create_kpvs_within_slices(alt_agl.array, cruise, min_value)


########################################
# Altitude: Flap


class AltitudeWithFlapMax(KeyPointValueNode):
    '''
    Maximum altitude (STD smoothed) with flaps extended.
    '''

    units = ut.FT

    @classmethod
    def can_operate(cls, available):
        '''
        The exceedance being detected here is the altitude reached with flaps not
        stowed, hence any flap value greater than zero is applicable and we're not
        really interested (for the purpose of identifying the event) what flap
        setting was reached.
        '''
        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Altitude STD Smoothed', 'Airborne'), available)

    def derive(self,
               alt_std=P('Altitude STD Smoothed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        alt_flap = np.ma.masked_where(retracted, alt_std.array)
        self.create_kpvs_within_slices(alt_flap, airborne, max_value)


class AltitudeAtFlapExtension(KeyPointValueNode):
    '''
    Records the altitude at every flap extension in flight.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               flaps=KTI('Flap Extension While Airborne')):

        if flaps:
            for flap in flaps:
                value = value_at_index(alt_aal.array, flap.index)
                self.create_kpv(flap.index, value)


class AltitudeAtFirstFlapExtensionAfterLiftoff(KeyPointValueNode):
    '''
    Separates the first flap extension.
    '''

    units = ut.FT

    def derive(self, flap_exts=KPV('Altitude At Flap Extension')):
        # First Flap Extension within Airborne section should be first after
        # liftoff.
        flap_ext = flap_exts.get_first()
        if flap_ext:
            self.create_kpv(flap_ext.index, flap_ext.value)


class AltitudeAtFlapExtensionWithGearDownSelected(KeyPointValueNode):
    '''
    Altitude at flap extensions while gear is selected down (may be in
    transit) and aircraft is airborne.
    '''

    NAME_FORMAT = 'Altitude At Flap %(flap)s Extension With Gear Down Selected'
    NAME_VALUES = NAME_VALUES_LEVER
    units = ut.FT

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Altitude AAL', 'Gear Down Selected', 'Airborne'), available)

    def derive(self,
               alt_aal=P('Altitude AAL'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               gear_ext=M('Gear Down Selected'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        # Raw flap values must increase to detect extensions.
        extend = np.ma.diff(flap.array.raw) > 0

        for in_air in airborne.get_slices():
            for index in np.ma.where(extend[in_air])[0]:
                # The flap we are moving to is +1 from the diff index
                index = (in_air.start or 0) + index + 1
                if gear_ext.array[index] != 'Down':
                    continue
                value = alt_aal.array[index]
                try:
                    self.create_kpv(index, value, flap=flap.array[index])
                except:
                    # Where flap values are mapped onto bits in the recorded
                    # word (e.g. E170 family), the new flap setting may clash
                    # with the old value, giving a transient indication of
                    # both flap readings. This is a crude fix to avoid this
                    # type of error condition.
                    self.create_kpv(index, value, flap=flap.array[index + 2])


class AirspeedAtFlapExtension(KeyPointValueNode):
    '''
    Airspeed at flap extensions while the aircraft is airborne.
    '''

    NAME_FORMAT = 'Airspeed At Flap %(flap)s Extension'
    NAME_VALUES = NAME_VALUES_LEVER
    units = ut.KT

    def derive(self, flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               air_spd=P('Airspeed'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        # Raw flap values must increase to detect extensions.
        extend = np.ma.diff(flap.array.raw) > 0

        for air_down in airborne.get_slices():
            for index in np.ma.where(extend[air_down])[0]:
                # The flap we are moving to is +1 from the diff index
                index = (air_down.start or 0) + index + 1
                value = air_spd.array[index]
                try:
                    self.create_kpv(index, value, flap=flap.array[index])
                except:
                    self.create_kpv(index, value, flap=flap.array[index + 2])


class AirspeedAtFlapExtensionWithGearDownSelected(KeyPointValueNode):
    '''
    Airspeed at flap extensions while gear is down and aircraft is airborne.
    '''

    NAME_FORMAT = 'Airspeed At Flap %(flap)s Extension With Gear Down Selected'
    NAME_VALUES = NAME_VALUES_LEVER
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Airspeed', 'Gear Down Selected', 'Airborne'), available)

    def derive(self,
               air_spd=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               gear_ext=M('Gear Down Selected'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        # Raw flap values must increase to detect extensions.
        extend = np.ma.diff(flap.array.raw) > 0

        for in_air in airborne.get_slices():
            # iterate over each extension
            for index in np.ma.where(extend[in_air])[0]:
                # The flap we are moving to is +1 from the diff index
                index = (in_air.start or 0) + index + 1
                if gear_ext.array[index] != 'Down':
                    continue
                value = value_at_index(air_spd.array, index)
                try:
                    self.create_kpv(index, value, flap=flap.array[index])
                except:
                    self.create_kpv(index, value, flap=flap.array[index + 2])


class AltitudeAALCleanConfigurationMin(KeyPointValueNode):
    '''
    Minimum altitude AAL with clean configuration.
    '''

    units = ut.FT
    name = 'Altitude AAL Clean Configuration Min'

    def derive(self,
               alt_rad=P('Altitude AAL'),
               flap=M('Flap'),
               gear_retr=S('Gear Retracted')):

        alt_rad_noflap = np.ma.masked_where(flap.array != '0', alt_rad.array)
        self.create_kpvs_within_slices(alt_rad_noflap, gear_retr, min_value)


class AltitudeAtFirstFlapChangeAfterLiftoff(KeyPointValueNode):
    '''
    Altitude AAL at first flap change after liftoff.
    '''

    units = ut.FT

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Flap At Liftoff', 'Altitude AAL', 'Airborne'), available)

    def derive(self,
               alt_aal=P('Altitude AAL'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               flap_liftoff=KPV('Flap At Liftoff'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth

        # Check whether takeoff with flap 0 occurred, as it is likely the first
        # flap change will be on the approach to landing which we are not
        # interested in here.
        #
        # Note: The KPV 'Flap At Liftoff' uses 'Flap' which we expect to always
        #       have a '0' state with a value of 0.0 in this KPV. Should this
        #       change, this code will need to be updated.
        if not flap_liftoff or flap_liftoff.get_first().value == 0.0:
            return

        for air in airborne:
            change_indexes = np.ma.where(np.ma.diff(flap.array[air.slice]))[0]
            if len(change_indexes):
                # Create at first change:
                index = (air.slice.start or 0) + change_indexes[0] + 0.5
                self.create_kpv(index, value_at_index(alt_aal.array, index))


class AltitudeAtLastFlapChangeBeforeTouchdown(KeyPointValueNode):
    '''
    Altitude AAL at last flap change before touchdown.
    '''
    # TODO: Review this in comparison to AltitudeAtLastFlapRetraction

    units = ut.FT

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Altitude AAL', 'Touchdown'), available)

    def derive(self,
               alt_aal=P('Altitude AAL'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               touchdowns=KTI('Touchdown'),
               far=P('Flap Automatic Retraction')):

        flap = flap_lever or flap_synth

        for touchdown in touchdowns:
            endpoint = touchdown.index
            if far:
                # This is an aircraft with automatic flap retraction. If the
                # auto retraction happened within three seconds of the
                # touchdown, set the endpoint to three seconds before touchdown.
                delta = int(3 * flap.hz)
                if far.array.raw[endpoint - delta] == 0 and \
                        far.array.raw[endpoint + delta] == 1:
                    endpoint = endpoint - delta

            land_flap = flap.array.raw[endpoint]
            flap_move = abs(flap.array.raw - land_flap)
            rough_index = index_at_value(flap_move, 0.5, slice(endpoint, 0, -1))
            # index_at_value tries to be precise, but in this case we really
            # just want the index at the new flap setting.
            if rough_index:
                last_index = np.round(rough_index)
                alt_last = value_at_index(alt_aal.array, last_index)
                self.create_kpv(last_index, alt_last)


class AltitudeAtLastFlapSelectionBeforeTouchdown(KeyPointValueNode):
    '''
    Records the Altitude AAL when the Flap position is last set to 15, 30 and 35
    degrees before touchdown.
    '''

    NAME_FORMAT = 'Altitude At Last Flap %(flap)d Selection Before Touchdown'
    NAME_VALUES = {'flap': [15, 30, 35]}

    units = ut.FT

    @classmethod
    def can_operate(cls, available):
        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Altitude AAL', 'Touchdown'), available)

    def derive(self, alt_aal=P('Altitude AAL'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               tdwns=KTI('Touchdown')):

        flap = flap_lever or flap_synth

        last_tdwn = 0
        for tdwn in tdwns:
            for flap_pos in self.NAME_VALUES['flap']:
                flap_slices = slices_and(
                    [slice(last_tdwn, tdwn.index), ],
                    slices_between(flap.array.raw, flap_pos-1, flap_pos+1)[1]
                )
                if not flap_slices:
                    continue
                #for flap_slice in flap_slices:
                for flap_slice in reversed(flap_slices):
                    flap_start = flap_slice.start
                    if flap.array[flap_start-1] < flap.array[flap_start]:
                        self.create_kpv(flap_start,
                                        value_at_index(alt_aal.array,
                                                       flap_start),
                                        replace_values={'flap':flap_pos})
                        break
            last_tdwn = tdwn.index + 1


class AltitudeAtFirstFlapRetractionDuringGoAround(KeyPointValueNode):
    '''
    Altitude AAL at first flap retraction during GA.

    Pinpoints the flap retraction instance within the
    500ft go-around window. Create a single KPV for the first flap retraction
    within a Go Around And Climbout phase.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               flap_rets=KTI('Flap Retraction During Go Around'),
               go_arounds=S('Go Around And Climbout')):
        '''
        The exceedance being detected here is the altitude reached with flaps not
        stowed, hence any flap value greater than zero is applicable and we're not
        really interested (for the purpose of identifying the event) what flap
        setting was reached.
        '''
        for go_around in go_arounds:
            # Find the index and height at this go-around minimum:
            pit_index, pit_value = min_value(alt_aal.array, go_around.slice)
            for flap_ret in flap_rets.get_ordered_by_index(within_slice=go_around.slice):
                if flap_ret.index > pit_index:
                    # Use height between go around minimum and gear up:
                    flap_up_ht = alt_aal.array[flap_ret.index] - pit_value
                    self.create_kpv(flap_ret.index, flap_up_ht)
                    break


class AltitudeAtFirstFlapRetraction(KeyPointValueNode):
    '''
    Altitude AAL at first flap retraction.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               flap_rets=KTI('Flap Retraction While Airborne')):

        flap_ret = flap_rets.get_first()
        if flap_ret:
            self.create_kpv(flap_ret.index, alt_aal.array[flap_ret.index])


class AltitudeAtLastFlapRetraction(KeyPointValueNode):
    '''
    Altitude AAL at last flap retraction.
    '''
    # TODO: Review this in comparison to AltitudeAtLastFlapChangeBeforeTouchdown

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               flap_rets=KTI('Flap Retraction While Airborne')):

        flap_ret = flap_rets.get_last()
        if flap_ret:
            self.create_kpv(flap_ret.index, alt_aal.array[flap_ret.index])


class AltitudeAtClimbThrustDerateDeselectedDuringClimbBelow33000Ft(KeyPointValueNode):
    '''
    Specific to 787 operations. Uses Altitude AAL.
    '''

    units = ut.FT

    def derive(self, alt_aal=P('Altitude AAL'),
               derate_deselecteds=KTI('Climb Thrust Derate Deselected'),
               climbs=S('Climbing')):
        for derate_deselected in derate_deselecteds.get(within_slices=climbs.get_slices()):
            alt_aal_value = value_at_index(alt_aal.array,
                                           derate_deselected.index)
            if alt_aal_value < 33000:
                self.create_kpv(derate_deselected.index, alt_aal_value)


########################################
# Altitude: Gear


class AltitudeAtLastGearDownSelection(KeyPointValueNode):
    '''
    Altitude AAL at last gear down selection.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               gear_dn_sel=KTI('Gear Down Selection')):

        if gear_dn_sel:
            self.create_kpvs_at_ktis(alt_aal.array, [gear_dn_sel.get_last()])


class AltitudeAtGearDownSelectionWithFlapDown(KeyPointValueNode):
    '''
    Altitude AAL at gear down selection with flaps extended.
    '''

    units = ut.FT

    @classmethod
    def can_operate(cls, available):
        '''
        Inclusion of the "...WithFlap" term is intended to exclude data points
        where only the gear is down (these are exceptional occasions where gear
        has been extended with flaps up to burn extra fuel).
        '''
        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Altitude AAL', 'Gear Down Selection'), available)

    def derive(self,
               alt_aal=P('Altitude AAL'),
               gear_downs=KTI('Gear Down Selection'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        flap_dns = runs_of_ones(~retracted)
        flap_dn_gear_downs = gear_downs.get(within_slices=flap_dns)
        self.create_kpvs_at_ktis(alt_aal.array, flap_dn_gear_downs)


class AltitudeAtFirstGearUpSelection(KeyPointValueNode):
    '''
    Altitude AAL at first gear up selection.

    Gear up selections after takeoff, not following a go-around (when it is
    normal to retract gear at significant height).
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               gear_up_sel=KTI('Gear Up Selection')):

        if gear_up_sel:
            self.create_kpvs_at_ktis(alt_aal.array, [gear_up_sel.get_first()])


class AltitudeAtGearUpSelectionDuringGoAround(KeyPointValueNode):
    '''
    Altitude AAL at which gear up was selected from the point of minimum 
    altitude in the go-around. If gear up was selected before that,
    value will be zero.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               go_arounds=S('Go Around And Climbout'),
               gear_up_sel=KTI('Gear Up Selection During Go Around')):

        for go_around in go_arounds:
            # Find the index and height at this go-around minimum:
            pit_index, pit_value = min_value(alt_aal.array, go_around.slice)
            for gear_up in gear_up_sel.get(within_slice=go_around.slice):
                if gear_up.index > pit_index:
                    # Use height between go around minimum and gear up:
                    gear_up_ht = alt_aal.array[gear_up.index] - pit_value
                    self.create_kpv(gear_up.index, gear_up_ht)

                # The else condition below led to creation of a zero KPV in
                # cases where the gear was not moved, so has been deleted.
                #else:
                    ## Use zero if gear up selected before minimum height:
                    #gear_up_ht = 0.0


class AltitudeWithGearDownMax(KeyPointValueNode):
    '''
    Maximum altitude AAL with the gear down.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               gear=M('Gear Down'),
               airs=S('Airborne')):

        gear.array[gear.array != 'Down'] = np.ma.masked
        gear_downs = np.ma.clump_unmasked(gear.array)
        self.create_kpv_from_slices(
            alt_aal.array, slices_and(airs.get_slices(), gear_downs),
            max_value)


class AltitudeSTDWithGearDownMax(KeyPointValueNode):
    '''
    Maximum altitude STD with gear down.
    '''

    name = 'Altitude STD With Gear Down Max'
    units = ut.FT

    def derive(self,
               alt_std=P('Altitude STD Smoothed'),
               gear=M('Gear Down'),
               airs=S('Airborne')):
        '''
        In extreme cases, it's the pressure altitude we are interested in, not
        just the altitude above the airfield (already covered by
        "Altitude With Gear Down Max")
        '''
        gear.array[gear.array != 'Down'] = np.ma.masked
        gear_downs = np.ma.clump_unmasked(gear.array)
        self.create_kpv_from_slices(
            alt_std.array, slices_and(airs.get_slices(), gear_downs),
            max_value)


class AltitudeAtGearDownSelectionWithFlapUp(KeyPointValueNode):
    '''
    Altitude AAL at gear down selection with flaps stowed.
    '''

    units = ut.FT

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Altitude AAL', 'Gear Down Selection'), available)

    def derive(self,
               alt_aal=P('Altitude AAL'),
               gear_downs=KTI('Gear Down Selection'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        flap_ups = runs_of_ones(retracted)
        flap_up_gear_downs = gear_downs.get(within_slices=flap_ups)
        self.create_kpvs_at_ktis(alt_aal.array, flap_up_gear_downs)


########################################
# Altitude: Automated Systems


class AltitudeAtAPEngagedSelection(KeyPointValueNode):
    '''
    Altitude AAL at witch the AP has been engaged.
    '''

    name = 'Altitude At AP Engaged Selection'
    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               ap_eng=KTI('AP Engaged Selection')):

        self.create_kpvs_at_ktis(alt_aal.array, ap_eng)


class AltitudeAtAPDisengagedSelection(KeyPointValueNode):
    '''
    Altitude AAL at which AP has been disengaged.
    '''

    name = 'Altitude At AP Disengaged Selection'
    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               ap_dis=KTI('AP Disengaged Selection')):

        self.create_kpvs_at_ktis(alt_aal.array, ap_dis)


class AltitudeAtATEngagedSelection(KeyPointValueNode):
    '''
    Altitude AAL at which the AT has been engaged.

    Note: Autothrottle is normally engaged prior to takeoff, so will not
          trigger this event.
    '''

    name = 'Altitude At AT Engaged Selection'
    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               at_eng=KTI('AT Engaged Selection')):

        self.create_kpvs_at_ktis(alt_aal.array, at_eng)


class AltitudeAtATDisengagedSelection(KeyPointValueNode):
    '''
    Altitude AAL at which the AT has been disengaged.
    '''

    name = 'Altitude At AT Disengaged Selection'
    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               at_dis=KTI('AT Disengaged Selection')):

        self.create_kpvs_at_ktis(alt_aal.array, at_dis)


class AltitudeAtFirstAPEngagedAfterLiftoff(KeyPointValueNode):
    '''
    Altitude AAL at which the AP has been engaged for the first time.
    '''

    name = 'Altitude At First AP Engaged After Liftoff'
    units = ut.FT

    def derive(self,
               ap=KTI('AP Engaged'),
               alt_aal=P('Altitude AAL'),
               airborne=S('Airborne')):

        change_indexes = find_edges_on_state_change('Engaged', ap.array,
                                                    phase=airborne)
        if len(change_indexes):
            # Create at first change:
            index = change_indexes[0]
            self.create_kpv(index, value_at_index(alt_aal.array, index))


class ATEngagedAPDisengagedOutsideClimbDuration(KeyPointValueNode):
    '''
    Duration for which the AT was engaged but AP disengaged in all flight
    phases except for climb.
    '''

    name = 'AT Engaged AP Disengaged Outside Climb Duration'

    @classmethod
    def can_operate(cls, available, ac_family=A('Family')):
        '''
        Autothrottle Use
        ================
        Autothrottle use is recommended during takeoff and climb in either automatic or
        manual flight. During all other phases of flight, autothrottle use is recommended
        only when the autopilot is engaged in CMD.
    
        FCTM B737NG - AFDS guidelines 1.35
        
        Uses custom 300fpm climb criteria
        '''
        
        if ac_family and ac_family.value in ('B737 NG', 'B747', 'B757', 'B767'):
            return all_deps(cls, available)
        else:
            return False

    def derive(self,
               at_engaged=M('AT Engaged'),
               ap_engaged=M('AP Engaged'),
               vert_spd=P('Vertical Speed For Flight Phases'),
               airborne=S('Airborne'),
               takeoff=S('Takeoff')):
        '''
        Takeoff added due to climb starting up to a couple of samples after airborne
        '''
        condition = vstack_params_where_state(
            (at_engaged, 'Engaged'),
            (ap_engaged, '-'),
        ).all(axis=0)
        climbs = []
        for air in airborne:
            climbing = np.ma.masked_less(vert_spd.array[air.slice], 300) # Custom climbing criteria for this KPV
            climbing_slices = slices_remove_small_gaps(
                np.ma.clump_unmasked(climbing), time_limit=30.0, hz=vert_spd.hz)
            climbs.extend(shift_slices(climbing_slices, air.slice.start))

        not_climbing = slices_and_not(airborne.get_slices(), slices_or(climbs +
                                                                        takeoff.get_slices()))
        phases = slices_and(runs_of_ones(condition), not_climbing)
        self.create_kpvs_from_slice_durations(phases, self.frequency)


########################################
# Altitude: Mach


class AltitudeAtMachMax(KeyPointValueNode):
    '''
    Altitude STD smoothed at which Mach speed reached its maximum.
    '''

    units = ut.FT

    def derive(self,
               alt_std=P('Altitude STD Smoothed'),
               max_mach=KPV('Mach Max')):
        # Aligns altitude to mach to ensure we have the most accurate altitude
        # reading at the point of maximum mach:
        self.create_kpvs_at_kpvs(alt_std.array, max_mach)

########################################
# Altitude: On Approach

class HeightAtDistancesFromThreshold(KeyPointValueNode):
    '''
    Altitude AAL at distances from runway threshold.
    '''

    NAME_FORMAT = 'Height At %(distance)d NM From Threshold'
    NAME_VALUES = NAME_VALUES_RANGES

    units = ut.FT

    def derive(self, alt = P('Altitude AAL'),
               dist_ktis = KTI('Distance From Threshold')):

        if not dist_ktis:
            return # Empty handed; nothing we can do.
        for distance in NAME_VALUES_RANGES['distance']:
            kti = dist_ktis.get_first(name='%d NM From Threshold' % distance)
            if kti:
                self.create_kpv(kti.index,
                                value_at_index(alt.array, kti.index),
                                replace_values={'distance':distance})


class HeightAtOffsetILSTurn(KeyPointValueNode):
    '''
    Altitude AAL above threshold for approaches to runways with offset ILS
    localizer at the point where the aircraft passes outside 0.5 dots and turns
    towards the runway.
    '''
    
    name = 'Height At Offset ILS Turn'
    units = ut.FT
    
    def derive(self, alt= P('Altitude AAL'),
               apps=App('Approach Information')):
        
        for app in apps:
            if app.offset_ils:
                index = app.loc_est.stop
                value = alt.array[index]
                self.create_kpv(index, value)


class HeightAtRunwayChange(KeyPointValueNode):
    '''
    Altitude AAL above threshold for approaches where the runway 
    changes, as identified by departure from the localizer for the first 
    runway.
    '''
    
    name = 'Height At Runway Change'
    units = ut.FT
    
    def derive(self, alt= P('Altitude AAL'),
               apps=App('Approach Information')):
        
        for app in apps:
            if app.runway_change and app.loc_est:
                index = app.loc_est.stop
                value = alt.array[index]
                if value:
                    self.create_kpv(index, value)

class HeightSelectedOnApproachMin(KeyPointValueNode):
    '''
    This KPV detects the lowest altitude set on the Mode Control 
    Panel (MCP) during an approach, in case the pilot inadvertently 
    selects an altitude less than a safe approach height above 
    the airfield. 
    
    If the pilot selects zero altitude the difference is negative, but
    these are returned as KPV=0 as a setting at or below the airfield
    elevation is equally dangerous, and not more dangerous at higher 
    airports.
    '''
    
    units = ut.FT
    
    def derive(self, alt_mcp=P('Altitude Selected (MCP)'),
               apps=App('Approach Information')):
        
        for app in apps:
            sel_ht = alt_mcp.array[app.slice] - app.airport['elevation']
            sel_ht_index = np.ma.argmin(sel_ht)
            sel_ht_value = max(sel_ht[sel_ht_index], 0.0)
            self.create_kpv(sel_ht_index + app.slice.start, sel_ht_value)


##############################################################################
# Collective


class CollectiveFrom10To60PercentDuration(KeyPointValueNode):
    '''
    Collective from 10% to 60% duration.
    '''

    can_operate = helicopter_only

    name = 'Collective From 10 To 60% Duration'
    units = ut.SECOND

    def derive(self, collective=P('Collective'), rtr=S('Rotors Turning')):
        start = 10
        end = 60
        target_ranges = np.ma.clump_unmasked(np.ma.masked_outside(collective.array, start - 1, end + 1))
        valid_sections = []
        for section in target_ranges:
            if (np.ma.ptp(collective.array[max(section.start-1, 0): section.stop+1]) > end - start) and \
               (collective.array[section.start] < collective.array[section.stop]) and \
               ((section.stop - section.start) < collective.frequency*10.0):
                valid_sections.append(section)
        self.create_kpvs_from_slice_durations(slices_and(valid_sections, rtr.get_slices()),
                                              collective.frequency)


##############################################################################
# Tail Rotor

class TailRotorPedalWhileTaxiingABSMax(KeyPointValueNode):
    '''
    Maximum Absolute tail rotor pedal during ground taxi (helicopter_only).
    '''

    name = 'Tail Rotor Pedal While Taxiing ABS Max'

    can_operate = helicopter_only

    units = ut.PERCENT

    def derive(self, pedal=P('Tail Rotor Pedal'), taxiing=S('Taxiing')):
        self.create_kpvs_within_slices(pedal.array, taxiing.get_slices(),
                                       max_abs_value)


class TailRotorPedalWhileTaxiingMax(KeyPointValueNode):
    '''
    Maximum tail rotor pedal during ground taxi (helicopter_only).
    '''
    can_operate = helicopter_only

    units = ut.PERCENT

    def derive(self, pedal=P('Tail Rotor Pedal'), taxiing=S('Taxiing')):
        self.create_kpvs_within_slices(pedal.array, taxiing.get_slices(),
                                       max_value)


class TailRotorPedalWhileTaxiingMin(KeyPointValueNode):
    '''
    Minimum tail rotor pedal during ground taxi (helicopter only).
    '''
    can_operate = helicopter_only

    units = ut.PERCENT

    def derive(self, pedal=P('Tail Rotor Pedal'), taxiing=S('Taxiing')):
        self.create_kpvs_within_slices(pedal.array, taxiing.get_slices(),
                                       min_value)


##############################################################################
# Cyclic

class CyclicDuringTaxiMax(KeyPointValueNode):
    '''
    Maximum cyclic angle during taxi.
    '''

    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, cyclic=P('Cyclic Angle'), taxi=S('Taxiing'), rtr=S('Rotors Turning')):
        self.create_kpvs_within_slices(cyclic.array, slices_and(taxi.get_slices(),
                                                                rtr.get_slices()),
                                       max_value)


class CyclicLateralDuringTaxiMax(KeyPointValueNode):
    '''
    Measures the maximum lateral displacement of the cyclic from the neutral point during ground taxi phase.
    '''

    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, cyclic=P('Cyclic Lateral'), taxi=S('Taxiing'), rtr=S('Rotors Turning')):
        self.create_kpvs_within_slices(cyclic.array, slices_and(taxi.get_slices(),
                                                                rtr.get_slices()),
                                       max_abs_value)


class CyclicAftDuringTaxiMax(KeyPointValueNode):
    '''
    Measures the maximum rearward displacement of the cyclic from the neutral point during ground taxi phase.
    '''

    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, cyclic=P('Cyclic Fore-Aft'), taxi=S('Taxiing'), rtr=S('Rotors Turning')):
        np.ma.masked_greater_equal(cyclic.array, 0)
        self.create_kpvs_within_slices(cyclic.array, slices_and(taxi.get_slices(),
                                                                rtr.get_slices()),
                                       max_value)


class CyclicForeDuringTaxiMax(KeyPointValueNode):
    '''
    Measures the maximum fore ward displacement of the cyclic from the neutral point during ground taxi phase.
    '''

    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, cyclic=P('Cyclic Fore-Aft'), taxi=S('Taxiing'), rtr=S('Rotors Turning')):
        np.ma.masked_less_equal(cyclic.array, 0)
        self.create_kpvs_within_slices(cyclic.array, slices_and(taxi.get_slices(),
                                                                rtr.get_slices()),
                                       min_value)


########################################
# Stable Approach analysis


class AltitudeFirstStableDuringLastApproach(KeyPointValueNode):
    '''
    Establish first point stable during the last approach i.e. a full stop
    landing. Uses altitude AAL.

    Should the approach have not become stable, the altitude will read 0 ft,
    indicating that it was unstable all the way to touchdown.
    '''

    units = ut.FT

    def derive(self, stable=M('Stable Approach'), alt=P('Altitude AAL')):

        # no need for approaches as we can assume each approach has no masked
        # values and inbetween there will be some
        apps = np.ma.clump_unmasked(stable.array)
        if apps:
            # we're only interested in the last approach - we assume that
            # this was the one which came to a full stop
            app = apps[-1]
            index = index_of_first_start(stable.array == 'Stable', app, min_dur=2)
            if index:
                self.create_kpv(index, value_at_index(alt.array, index))
            else:
                # force an altitude of 0 feet at the end of the approach
                self.create_kpv(app.stop - 0.5, 0)


class AltitudeFirstStableDuringApproachBeforeGoAround(KeyPointValueNode):
    '''
    Establish first point (altitude AAL) stable during all but the last approach. 
    Here we assume that these approaches were followed by a Go Around (or possible a
    Touch and Go).

    Should the approach have not become stable, the altitude will read 0 ft,
    indicating that it was constantly unstable.
    '''

    units = ut.FT

    def derive(self, stable=M('Stable Approach'), alt=P('Altitude AAL')):

        # no need for approaches as we can assume each approach has no masked
        # values and inbetween there will be some
        apps = np.ma.clump_unmasked(stable.array)
        for app in apps[:-1]:
            # iterate through approaches as only one KPV is to be created per
            # approach
            index = index_of_first_start(stable.array == 'Stable', app, min_dur=2)
            if index:
                self.create_kpv(index, value_at_index(alt.array, index))
            else:
                self.create_kpv(app.stop - 0.5, 0)


class AltitudeLastUnstableDuringLastApproach(KeyPointValueNode):
    '''
    Establish last Unstable altitude AAL during the last approach i.e. a full 
    stop landing.

    Should the approach have not become stable, the altitude will read 0 ft,
    indicating that it was unstable all the way to touchdown.
    '''

    units = ut.FT

    def derive(self, stable=M('Stable Approach'), alt=P('Altitude AAL')):

        apps = np.ma.clump_unmasked(stable.array)
        if apps:
            # we're only interested in the last approach - we assume that
            # this was the one which came to a full stop
            app = apps[-1]
            index = index_of_last_stop(stable.array != 'Stable', app, min_dur=2)
            # Note: Assumed will never have an approach which is 100% Stable
            self.create_kpv(index, value_at_index(alt.array, index))


class AltitudeLastUnstableDuringLastApproachExcludingEngThrust(KeyPointValueNode):
    '''
    Establish last Unstable altitude AAL during the last approach i.e. a full 
    stop landing.
    
    Unlike 'Altitude Last Unstable During Last Approach', this uses the 
    multistae parameter 'Stable Approach Excluding Eng Thrust' which
    excludes the Engine Thrust (N1/EPR) stability stage.

    Should the approach have not become stable, the altitude will read 0 ft,
    indicating that it was unstable all the way to touchdown.
    '''

    units = ut.FT

    def derive(self, stable=M('Stable Approach Excluding Eng Thrust'),
               alt=P('Altitude AAL')):

        apps = np.ma.clump_unmasked(stable.array)
        if apps:
            # we're only interested in the last approach - we assume that
            # this was the one which came to a full stop
            app = apps[-1]
            index = index_of_last_stop(stable.array != 'Stable', app, min_dur=2)
            # Note: Assumed will never have an approach which is 100% Stable
            self.create_kpv(index, value_at_index(alt.array, index))


class AltitudeLastUnstableDuringApproachBeforeGoAround(KeyPointValueNode):
    '''
    Establish last Unstable altitude (AAL) during all but the last approach. 
    Here weassume that these approaches were followed by a Go Around (or 
    possibly a Touch and Go).

    Should the approach have not become stable, the altitude will read 0 ft,
    indicating that it was constantly unstable.
    '''

    units = ut.FT

    def derive(self, stable=M('Stable Approach'), alt=P('Altitude AAL')):

        apps = np.ma.clump_unmasked(stable.array)
        for app in apps[:-1]:
            index = index_of_last_stop(stable.array != 'Stable', app, min_dur=2)
            if index > app.stop - 1:
                # approach ended unstable
                # we were not stable so force altitude of 0 ft
                self.create_kpv(app.stop - 0.5, 0)
            else:
                self.create_kpv(index, value_at_index(alt.array, index))


class LastUnstableStateDuringLastApproach(KeyPointValueNode):
    '''
    Point where the aircraft was in an unstable state during last approach.
    '''

    units = None

    def derive(self, stable=M('Stable Approach')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
    
        Establish last Unstable state (integer representation of the "Stable"
        parameter's values_mapping) during each approach which was followed by a
        Go Around (or possibly a Touch and Go).
    
        Particuarly of interest to know the reason for instability should the
        Last Unstable condition be at a low altitude.        
        '''
        apps = np.ma.clump_unmasked(stable.array)
        if apps:
            # we're only interested in the last approach - we assume that
            # this was the one which came to a full stop
            app = apps[-1]
            index = index_of_last_stop(stable.array != 'Stable', app, min_dur=2)
            # Note: Assumed will never have an approach which is 100% Stable
            self.create_kpv(index, stable.array.raw[index])


class LastUnstableStateDuringApproachBeforeGoAround(KeyPointValueNode):
    '''
    Point where the aircraft was in an unstable state during last approach before
    go-around.
    '''

    units = None

    def derive(self, stable=M('Stable Approach')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.

        Establish last Unstable state (integer representation of the "Stable"
        parameter's values_mapping) during each approach which was followed by a
        Go Around (or possibly a Touch and Go).

        Can help to determine the reason for choosing not to land.
        '''
        apps = np.ma.clump_unmasked(stable.array)
        for app in apps[:-1]:
            index = index_of_last_stop(stable.array != 'Stable', app, min_dur=2)
            # Note: Assumed will never have an approach which is 100% Stable
            self.create_kpv(index, stable.array.raw[index])


class PercentApproachStable(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.

    Creates a KPV at 1000 ft AAL and 500 ft AAL during the approach with 
    the percent (0% to 100%) of the approach that was stable.

    Creates separate names for approaches before a Go Around (or possibly a
    Touch and Go) and those for the Last Landing (assuming a full stop
    landing)
    '''

    NAME_FORMAT = 'Percent Approach Stable Below %(altitude)d Ft %(approach)s'
    NAME_VALUES = {
        'altitude': (1000, 500),
        'approach': ('During Last Approach', 'During Approach Before Go Around'),
    }
    units = ut.PERCENT

    def derive(self, stable=M('Stable Approach'), alt=P('Altitude AAL')):

        apps = np.ma.clump_unmasked(stable.array)
        for n, app in enumerate(apps):
            if n < len(apps) - 1:
                approach_type = 'During Approach Before Go Around'
            else:
                approach_type = 'During Last Approach'

            stable_app = stable.array[app]
            alt_app = alt.array[app]
            # ensure that stability on ground does not contribute to percentage
            stable_app[alt_app <= 0] = np.ma.masked

            for level in (1000, 500):
                # mask out data above the altitude level
                stable_app[alt_app > level] = np.ma.masked
                is_stable = stable_app == 'Stable'
                percent = np.ma.sum(is_stable) / float(np.ma.count(is_stable)) * 100
                # find first stable point (if not, argmax returns 0)
                index = np.ma.argmax(is_stable) + app.start
                self.create_kpv(index, percent,
                                altitude=level, approach=approach_type)


class AltitudeAtLastAPDisengagedDuringApproach(KeyPointValueNode):
    '''
    The altitude (AAL) at which autopilot was last disengaged 
    during the cruise.
    '''

    name = 'Altitude At Last AP Disengaged During Approach'
    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               ap_dis=KTI('AP Disengaged Selection'),
               apps=App('Approach Information')):

        ktis = []
        for app in apps:
            ap_dis_kti = ap_dis.get_last(within_slice=app.slice)
            if ap_dis_kti:
                ktis.append(ap_dis_kti)
        self.create_kpvs_at_ktis(alt_aal.array, ktis)


##############################################################################
# Autopilot


class APDisengagedDuringCruiseDuration(KeyPointValueNode):
    '''
    The duration for which all autopilot channels were disengaged
    during the cruise.
    '''

    name = 'AP Disengaged During Cruise Duration'
    units = ut.SECOND

    def derive(self, ap=M('AP Engaged'), cruise=S('Cruise')):
        self.create_kpvs_where(ap.array != 'Engaged', ap.hz, phase=cruise)


##############################################################################


class ControlColumnStiffness(KeyPointValueNode):
    """
    The control force and displacement of the flying controls should follow a
    predictable relationship. This parameter is included to identify
    stiffness in the controls in flight.
    """

    units = None  # FIXME

    def derive(self,
               force=P('Control Column Force'),
               disp=P('Control Column'),
               fast=S('Fast')):

        # We only test during high speed operation to avoid "testing" the
        # full and free movements before flight.
        for speedy in fast:
            # We look for forces above a threshold to detect manual input.
            # This is better than looking for movement, as in the case of
            # stiff controls there is more force but less movement, hence
            # using a movement threshold will tend to be suppressed in the
            # cases we are looking to detect.
            push = force.array[speedy.slice]
            column = disp.array[speedy.slice]

            moves = np.ma.clump_unmasked(
                np.ma.masked_less(np.ma.abs(push),
                                  CONTROL_FORCE_THRESHOLD))
            for move in moves:
                if slice_samples(move) < 10:
                    continue
                corr, slope, off = \
                    coreg(push[move], indep_var=column[move], force_zero=True)
                if corr > 0.85:  # This checks the data looks sound.
                    when = np.ma.argmax(np.ma.abs(push[move]))
                    self.create_kpv(
                        (speedy.slice.start or 0) + move.start + when, slope)


class ControlColumnForceMax(KeyPointValueNode):
    '''
    Maximum force applied to the control column while airborne.
    '''

    units = ut.DECANEWTON

    def derive(self,
               force=P('Control Column Force'),
               fast=S('Airborne')):
        self.create_kpvs_within_slices(
            force.array, fast.get_slices(),
            max_abs_value)


class ControlWheelForceMax(KeyPointValueNode):
    '''
    Maximum force applied to the control wheel while airborne.
    '''

    units = ut.DECANEWTON

    def derive(self,
               force=P('Control Wheel Force'),
               fast=S('Airborne')):
        self.create_kpvs_within_slices(
            force.array, fast.get_slices(),
            max_abs_value)


def PreflightCheck(self, firsts, accels, disps, full_disp):
    """
    Compute the total travel of each control during the interval between first engine
    start and takeoff start of acceleration, as % of full travel for that control.
    """    
    for first in firsts:
        acc = accels.get_next(first.index)
        if acc is None or int(first.index) == int(acc.index): #avoid 0 length slice
            continue
        disps = [d.array[first.index:acc.index] for d in disps]
        ptp = max(np.ma.max(d) for d in disps) - min(np.ma.min(d) for d in disps)
        index = np.argmax(np.ma.abs(max(disps, key=lambda d: np.ma.ptp(d)))) + first.index
        # Mark the point where this control displacement was greatest.
        self.create_kpv(index, (ptp / full_disp) * 100.0)


class ElevatorPreflightCheck(KeyPointValueNode):
    """
    Point of elevator check before flight.
    """

    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'), family=A('Family')):

        if not (any_of(('Elevator', 'Elevator (L)', 'Elevator (R)'), available) and
                    all_of(('First Eng Start Before Liftoff', 'Takeoff Acceleration Start',
                            'Model', 'Series', 'Family'), available)):
            return False

        try:
            at.get_elevator_range(model.value, series.value, family.value)
        except KeyError:
            cls.warning("No Elevator range available for '%s', '%s', '%s'.",
                        model.value, series.value, family.value)
            return False

        return True

    def derive(self, elev=P('Elevator'),
               elev_l=P('Elevator (L)'),
               elev_r=P('Elevator (R)'),
               firsts=KTI('First Eng Start Before Liftoff'),
               accels=KTI('Takeoff Acceleration Start'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        disps = [e for e in (elev_l, elev_r) if e] if elev_l or elev_r else [elev]
        disp_range = at.get_elevator_range(model.value, series.value, family.value)
        full_disp = disp_range * 2 if isinstance(disp_range, (float, int)) else disp_range[1] - disp_range[0]

        PreflightCheck(self, firsts, accels, disps, full_disp)


class AileronPreflightCheck(KeyPointValueNode):
    """
    Point of aileron check before flight.
    """
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'), family=A('Family')):

        if not (any_of(('Aileron', 'Aileron (L)', 'Aileron (R)'), available) and
                all_of(('First Eng Start Before Liftoff', 'Takeoff Acceleration Start',
                        'Model', 'Series', 'Family'), available)):
            return False

        try:
            at.get_aileron_range(model.value, series.value, family.value)
        except KeyError:
            cls.warning("No Aileron range available for '%s', '%s', '%s'.",
                        model.value, series.value, family.value)
            return False

        return True

    def derive(self, ail=P('Aileron'),
               ail_l=P('Aileron (L)'),
               ail_r=P('Aileron (R)'),
               firsts=KTI('First Eng Start Before Liftoff'),
               accels=KTI('Takeoff Acceleration Start'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        disps = [a for a in (ail_l, ail_r) if a] if ail_l or ail_r else [ail]
        disp_range = at.get_aileron_range(model.value, series.value, family.value)
        full_disp = disp_range * 2 if isinstance(disp_range, (float, int)) else disp_range[1] - disp_range[0]

        PreflightCheck(self, firsts, accels, disps, full_disp)


class RudderPreflightCheck(KeyPointValueNode):
    """
    Point of rudder check before flight.
    """
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Rudder', 'First Eng Start Before Liftoff', 'Takeoff Acceleration Start', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_rudder_range(model.value, series.value, family.value)
        except KeyError:
            cls.warning("No Rudder range available for '%s', '%s', '%s'.",
                        model.value, series.value, family.value)
            return False

        return True

    def derive(self, disp=P('Rudder'),
               firsts=KTI('First Eng Start Before Liftoff'),
               accels=KTI('Takeoff Acceleration Start'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        disp_range = at.get_rudder_range(model.value, series.value, family.value)
        full_disp = disp_range * 2 if isinstance(disp_range, (float, int)) else disp_range[1] - disp_range[0]

        PreflightCheck(self, firsts, accels, [disp], full_disp)


class FlightControlPreflightCheck(KeyPointValueNode):
    '''
    sum of Elevator, Aileron and Rudder Preflight Check KPVs for use in Event
    detection
    See NTSB recommendation A-15-34.
    '''
    units = ut.PERCENT

    def derive(self, elevator=KPV('Elevator Preflight Check'),
               aileron=KPV('Aileron Preflight Check'),
               rudder=KPV('Rudder Preflight Check')):
        first_valid = elevator or aileron or rudder
        if first_valid:
            elevator_value = elevator.get_first().value if elevator else 0
            aileron_value = aileron.get_first().value if aileron else 0
            rudder_value = rudder.get_first().value if rudder else 0
            index = first_valid.get_first().index
            value = elevator_value + aileron_value + rudder_value

            self.create_kpv(index, value)


##############################################################################
# Distances: Flight

class GreatCircleDistance(KeyPointValueNode):
    '''
    Great circle distance between liftoff and touchdown points.
    '''

    units = ut.NM

    @classmethod
    def can_operate(cls, available):
        toff = all_of(('Latitude Smoothed At Liftoff', 'Longitude Smoothed At Liftoff'), available) \
            or 'FDR Takeoff Airport' in available
        ldg = all_of(('Latitude Smoothed At Touchdown', 'Longitude Smoothed At Touchdown'), available) \
            or 'FDR Landing Airport' in available
        return toff and ldg and 'Touchdown' in available

    def derive(self,
               lat_lift=KPV('Latitude Smoothed At Liftoff'),
               lon_lift=KPV('Longitude Smoothed At Liftoff'),
               toff_airport=A('FDR Takeoff Airport'),
               lat_tdwn=KPV('Latitude Smoothed At Touchdown'),
               lon_tdwn=KPV('Longitude Smoothed At Touchdown'),
               ldg_airport=A('FDR Landing Airport'),
               tdwn=KTI('Touchdown')):

        if not tdwn:
            # no point continueing
            return

        toff_lat = toff_lon = ldg_lat = ldg_lon = None

        if toff_airport and toff_airport.value:
            toff_lat = toff_airport.value.get('latitude')
            toff_lon = toff_airport.value.get('longitude')
        if lat_lift and lon_lift and (not toff_lat or not toff_lon):
            toff_lat = lat_lift.get_first().value
            toff_lon = lon_lift.get_first().value
        if not toff_lat or not toff_lon:
            # we have no takeoff coordinates so exit
            return
        if ldg_airport and ldg_airport.value:
            ldg_lat = ldg_airport.value.get('latitude')
            ldg_lon = ldg_airport.value.get('longitude')
        if lat_tdwn and lon_tdwn and (not ldg_lat or not ldg_lon):
            ldg_lat = lat_tdwn.get_last().value
            ldg_lon = lon_tdwn.get_last().value
        if not ldg_lat or not ldg_lon:
            # we have no landingcoordinates so exit
            return

        if tdwn:
            value = great_circle_distance__haversine(toff_lat, toff_lon, ldg_lat, ldg_lon, units=ut.NM)
            index = tdwn.get_last().index
            if value:
                self.create_kpv(index, value)


class DistanceTravelledDuringTurnback(KeyPointValueNode):
    '''
    This measures the distance flown over the ground during a flight which
    Lands back at the Takeoff airport only.
    '''

    units = ut.NM
    
    def derive(self, gspd=P('Groundspeed'), toff_airport=A('FDR Takeoff Airport'),
               ldg_airport=A('FDR Landing Airport'),
               loffs=KTI('Liftoff'), tdowns=KTI('Touchdown')):

        if toff_airport and ldg_airport and toff_airport.value.get('id') == ldg_airport.value.get('id'):
            loff = loffs.get_first()
            tdown = tdowns.get_last()
            dist = max(integrate(gspd.array[loff.index:tdown.index + 1],
                                 gspd.hz, scale=1.0 / 3600.0))
            self.create_kpv(tdown.index, dist)


class DistanceTravelledFollowingDiversion(KeyPointValueNode):
    '''
    This measures the distance flown over the ground following a diversion to
    a different airport than initially intended.
    '''

    can_operate = aeroplane_only

    units = ut.NM
    
    def derive(self, gspd=P('Groundspeed'), destination=P('Destination'),
               loff=KTI('Liftoff'), tdwn=KTI('Touchdown')):
        '''
        Diversions are detected from a change in the "Destination" parameter.
        '''
        # remove masked entries
        dest_repair = nearest_neighbour_mask_repair(destination.array)
        values = [(dest, index) for (dest, index) in zip(*np.unique(dest_repair, return_index=True)) if dest]
        if len(values) < 2: # We have more than one destination during flight
            return
        values = sorted(values, key=itemgetter(1))
        start_idx = loff.get_first().index
        stop_idx = tdwn.get_last().index
        start_dest = dest_repair[start_idx]
        for dest, index in values:
            if dest == start_dest:
                # Analysts requested only diversion if destination changes and remains changed.
                continue
            runs = runs_of_ones(dest_repair==dest)
            for run in runs:
                if (start_idx < run.start < stop_idx) and run.stop > stop_idx: 
                    if slice_duration(run, self.frequency) <= 64: # one superframe sample
                        continue
                    end_idx = min((stop_idx, run.stop))
                    dist = max(integrate(gspd.array[run.start:end_idx + 1],
                                                     gspd.hz, scale=1.0 / 3600.0))
                    if dist:
                        self.create_kpv(run.start, dist)
                    if stop_idx < run.stop:
                        break


##############################################################################
# Runway Distances at Takeoff


class DistanceFromLiftoffToRunwayEnd(KeyPointValueNode):
    '''
    Distance between the point of liftoff and end of the runway.
    '''

    can_operate = aeroplane_only
    units = ut.METER

    def derive(self,
               lat_lift=KPV('Latitude Smoothed At Liftoff'),
               lon_lift=KPV('Longitude Smoothed At Liftoff'),
               rwy=A('FDR Takeoff Runway')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Take-Off (Longitudinal), Runway remaining at rotation"
        '''

        if ambiguous_runway(rwy) or not lat_lift:
            return
        toff_end = runway_distance_from_end(rwy.value,
                                            lat_lift[0].value,
                                            lon_lift[0].value)
        self.create_kpv(lat_lift[0].index, toff_end)


class DistanceFromRotationToRunwayEnd(KeyPointValueNode):
    '''
    Distance between the start of rotation and runway end.
    '''

    can_operate = aeroplane_only
    units = ut.METER

    def derive(self,
               lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               rwy=A('FDR Takeoff Runway'),
               toff_rolls=S('Takeoff Roll')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Take-Off (Longitudinal), Runway remaining at rotation"
        '''
        if ambiguous_runway(rwy):
            return
        for roll in toff_rolls:
            rot_idx = roll.stop_edge
            rot_end = runway_distance_from_end(rwy.value,
                                               lat.array[rot_idx],
                                               lon.array[rot_idx])
            self.create_kpv(rot_idx, rot_end)


class DecelerationToAbortTakeoffAtRotation(KeyPointValueNode):
    '''
    Minimum deceleration required to abort takeoff at the rotation speed, in
    order to be able to stop before the runway threshold.
    '''

    units = ut.G

    def derive(self,
               lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               gspd=P('Groundspeed'),
               aspd=P('Airspeed True'),
               rwy=A('FDR Takeoff Runway'),
               toff_rolls=S('Takeoff Roll')):

        if ambiguous_runway(rwy):
            return
        if gspd:
            speed = repair_mask(gspd.array, gspd.frequency)
        else:
            speed = repair_mask(aspd.array, aspd.frequency)
        for roll in toff_rolls:
            rot_idx = roll.stop_edge
            rot_end = runway_distance_from_end(rwy.value,
                                               lat.array[rot_idx],
                                               lon.array[rot_idx])
            if rot_end:
                lift_speed = ut.convert(value_at_index(speed, rot_idx), ut.KT, ut.METER_S)
                mu = (lift_speed**2.0) / (2.0 * GRAVITY_METRIC * rot_end)
                self.create_kpv(rot_idx, mu)


"""
This KPV was sketched out following Emirates' presentation, but requires a
value for V1 which is not currently set up as a derived (or recorded)
parameter.

class DecelerationToAbortTakeoffBeforeV1(KeyPointValueNode):
    '''
    FDS developed this KPV following the 2nd EOFDM conference.
    '''

    units = ut.G

    def derive(self, lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               gspd=P('Groundspeed'),
               aspd=P('Airspeed True'),
               v1=A('V1'),
               rwy=A('FDR Takeoff Runway'),
               toff_rolls=S('Takeoff Roll')):

        if ambiguous_runway(rwy):
            return
        if gspd:
            speed=gspd.array
        else:
            speed=aspd.array
        for roll in toff_rolls:
            v1_idx = v1.value
            rot_end = runway_distance_from_end(rwy.value,
                                               lat.array[v1_idx ],
                                               lon.array[v1_idx ])

            v1_mps = ut.convert(value_at_index(speed, v1.value), ut.KT, ut.METER_S)
            mu = (v1_mps**2.0) / (2.0 * GRAVITY_METRIC * rot_end)
            self.create_kpv(vi_idx, mu)
"""


##############################################################################
# Runway Distances at Landing


class DistancePastGlideslopeAntennaToTouchdown(KeyPointValueNode):
    '''
    Distance between glideslope antenna and touchdown point.
    '''

    units = ut.METER

    def derive(self,
               lat_tdn=KPV('Latitude Smoothed At Touchdown'),
               lon_tdn=KPV('Longitude Smoothed At Touchdown'),
               tdwns=KTI('Touchdown'), rwy=A('FDR Landing Runway'),
               ils_ldgs=S('ILS Localizer Established')):

        if ambiguous_runway(rwy) or not lat_tdn or not lon_tdn:
            return
        last_tdwn = tdwns.get_last()
        if not last_tdwn:
            return
        land_idx = last_tdwn.index
        # Check we did do an ILS approach (i.e. the ILS frequency was correct etc).
        if ils_ldgs.get(containing_index=land_idx):
            # OK, now do the geometry...
            gs = runway_distance_from_end(rwy.value, point='glideslope')
            td = runway_distance_from_end(rwy.value, lat_tdn.get_last().value,
                                          lon_tdn.get_last().value)
            if gs and td:
                distance = gs - td
                self.create_kpv(land_idx, distance)


class DistanceFromRunwayStartToTouchdown(KeyPointValueNode):
    '''
    Finds the distance from the runway start location to the touchdown point.
    This only operates for the last landing, and previous touch and goes will
    not be recorded.
    '''

    units = ut.METER

    def derive(self, lat_tdn=KPV('Latitude Smoothed At Touchdown'),
               lon_tdn=KPV('Longitude Smoothed At Touchdown'),
               tdwns=KTI('Touchdown'),
               rwy=A('FDR Landing Runway')):

        if ambiguous_runway(rwy) or not lat_tdn or not lon_tdn:
            return

        distance_to_start = runway_distance_from_end(rwy.value, point='start')
        distance_to_tdn = runway_distance_from_end(rwy.value,
                                                   lat_tdn.get_last().value,
                                                   lon_tdn.get_last().value)
        distance_start_tdn = distance_to_start - distance_to_tdn
        if distance_start_tdn > -500:
            # sanity check asumes landed on runway, allows for touching down on stopway
            self.create_kpv(tdwns.get_last().index,
                            distance_start_tdn)


class DistanceFromTouchdownToRunwayEnd(KeyPointValueNode):
    '''
    Finds the distance from the touchdown point to the end of the runway
    hardstanding. This only operates for the last landing, and previous touch
    and goes will not be recorded.
    '''

    can_operate = aeroplane_only
    units = ut.METER

    def derive(self, lat_tdn=KPV('Latitude Smoothed At Touchdown'),
               lon_tdn=KPV('Longitude Smoothed At Touchdown'),
               tdwns=KTI('Touchdown'),
               rwy=A('FDR Landing Runway')):

        if ambiguous_runway(rwy) or not lat_tdn or not tdwns:
            return

        distance_to_tdn = runway_distance_from_end(rwy.value,
                                                   lat_tdn.get_last().value,
                                                   lon_tdn.get_last().value)
        self.create_kpv(tdwns.get_last().index, distance_to_tdn)


class DecelerationFromTouchdownToStopOnRunway(KeyPointValueNode):
    '''
    This determines the average level of deceleration required to stop the
    aircraft before reaching the end of the runway. It takes into account the
    length of the runway, the point of touchdown and the groundspeed at the
    point of touchdown.

    The numerical value is in units of g, and can be compared with surface
    conditions or autobrake settings. For example, if the value is 0.14 and
    the braking is "medium" (typically 0.1g) it is likely that the aircraft
    will overrun the runway if the pilot relies upon wheel brakes alone.

    The value will vary during the deceleration phase, but the highest value
    was found to arise at or very shortly after touchdown, as the aerodynamic
    and rolling drag at high speed normally exceed this level. Therefore for
    simplicity we just use the value at touchdown.
    '''

    units = ut.G

    can_operate = aeroplane_only

    def derive(self,
               gspd=P('Groundspeed'),
               tdwns=S('Touchdown'),
               landings=S('Landing'),
               lat_tdn=KPV('Latitude Smoothed At Touchdown'),
               lon_tdn=KPV('Longitude Smoothed At Touchdown'),
               rwy=A('FDR Landing Runway'),
               ils_gs_apps=S('ILS Glideslope Established'),
               ils_loc_apps=S('ILS Localizer Established'),
               precise=A('Precise Positioning')):

        if ambiguous_runway(rwy):
            return
        index = tdwns.get_last().index
        for landing in landings:
            if not is_index_within_slice(index, landing.slice):
                continue

            # Was this an ILS approach where the glideslope was captured?
            ils_approach = False
            for ils_loc_app in ils_loc_apps:
                if not slices_overlap(ils_loc_app.slice, landing.slice):
                    continue
                for ils_gs_app in ils_gs_apps:
                    if slices_overlap(ils_loc_app.slice, ils_gs_app.slice):
                        ils_approach = True

            # So for captured ILS approaches or aircraft with precision location we can compute the deceleration required.
            if (precise.value or ils_approach) and lat_tdn != []:
                distance_at_tdn = \
                    runway_distance_from_end(rwy.value,
                                             lat_tdn.get_last().value,
                                             lon_tdn.get_last().value)
                kts = value_at_index(repair_mask(gspd.array), index)
                if not kts:
                    return
                speed = ut.convert(kts, ut.KT, ut.METER_S)
                mu = (speed * speed) / (2.0 * GRAVITY_METRIC * (distance_at_tdn))
                self.create_kpv(index, mu)


class DistanceFromRunwayCentrelineAtTouchdown(KeyPointValueNode):
    '''
    Distance from runway centerline at the point of touchdown.
    '''

    name = 'Distance From Runway Centreline At Touchdown'
    units = ut.METER
    can_operate = aeroplane_only

    def derive(self,
               lat_dist=P('ILS Lateral Distance'),
               tdwns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(lat_dist.array, tdwns)


class DistanceFromRunwayCentrelineFromTouchdownTo60KtMax(KeyPointValueNode):
    '''
    Maximum distance from centerline between the point of touchdown 
    and the point of the aircraft reaching 60 knots GS.
    '''

    can_operate = aeroplane_only
    name = 'Distance From Runway Centreline From Touchdown To 60 Kt Max'
    units = ut.METER

    def derive(self,
               lat_dist=P('ILS Lateral Distance'),
               lands=S('Landing'),
               gspd=P('Groundspeed'),
               tdwns=KTI('Touchdown')):
        # where corrupted data, interpolate as we're only interested in regions
        gspd_data = repair_mask(gspd.array, repair_duration=30)
        to_scan = []
        for land in lands:
            for tdwn in tdwns:
                if is_index_within_slice(tdwn.index, land.slice) and \
                   gspd_data[land.slice.stop - 1] < 60.0:
                    stop = index_at_value(gspd_data, 60.0, land.slice)
                    if stop is None:
                        # don't take the risk of measuring below 60kts as
                        # could be off the runway
                        return
                    to_scan.append(slice(tdwn.index, stop))

        self.create_kpvs_within_slices(
            lat_dist.array,
            to_scan,
            max_abs_value
        )


class RunwayHeadingTrue(KeyPointValueNode):
    '''
    Takeoff and landing runway heading.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return all_of(('FDR Takeoff Runway', 'Liftoff'), available) \
            or 'Approach Information' in available

    def derive(self,
               takeoff_runway=A('FDR Takeoff Runway'),
               liftoffs=KTI('Liftoff'),
               apps=App('Approach Information')):

        if takeoff_runway and liftoffs:
            liftoff = liftoffs.get_first()
            if liftoff:
                self.create_kpv(liftoff.index,
                                runway_heading(takeoff_runway.value))
        if apps:
            for app in apps:
                if not app.landing_runway:
                    continue
                # Q: Is the midpoint of the slice a sensible index?
                index = (app.slice.start +
                         ((app.slice.stop - app.slice.start) / 2))
                self.create_kpv(index, runway_heading(app.landing_runway))


class RunwayOverrunWithoutSlowingDuration(KeyPointValueNode):
    '''
    This determines the minimum time that the aircraft will take to reach the
    end of the runway without further braking. It takes into account the
    reducing groundspeed and distance to the end of the runway.

    The numerical value is in units of seconds.

    The value will decrease if the aircraft is not decelerating
    progressively. Therefore the lower values arise if the pilot allows the
    aircraft to roll down the runway without reducing speed. It will reflect
    the reduction in margins where aircraft roll at high speed towards
    taxiways near the end of the runway, and the value relates to the time
    available to the pilot.
    '''

    units = ut.SECOND

    can_operate = aeroplane_only

    def derive(self,
               gspd=P('Groundspeed'),
               tdwns=S('Touchdown'),
               landings=S('Landing'),
               lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               lat_tdn=KPV('Latitude Smoothed At Touchdown'),
               lon_tdn=KPV('Longitude Smoothed At Touchdown'),
               rwy=A('FDR Landing Runway'),
               ils_gs_apps=S('ILS Glideslope Established'),
               ils_loc_apps=S('ILS Localizer Established'),
               precise=A('Precise Positioning'),
               turnoff=KTI('Landing Turn Off Runway')):

        if ambiguous_runway(rwy):
            return
        last_tdwn = tdwns.get_last()
        if not last_tdwn:
            return
        scale = ut.multiplier(ut.KT, ut.METER_S)
        for landing in landings:
            if not is_index_within_slice(last_tdwn.index, landing.slice):
                continue
            # Was this an ILS approach where the glideslope was captured?
            ils_approach = False
            for ils_loc_app in ils_loc_apps:
                if not slices_overlap(ils_loc_app.slice, landing.slice):
                    continue
                for ils_gs_app in ils_gs_apps:
                    if slices_overlap(ils_loc_app.slice, ils_gs_app.slice):
                        ils_approach = True
            # When did we turn off the runway?
            last_turnoff = turnoff.get_last()
            if not last_turnoff or not is_index_within_slice(last_turnoff.index, landing.slice):
                continue
            # So the period of interest is...
            land_roll = slice(last_tdwn.index, last_turnoff.index)
            # So for captured ILS approaches or aircraft with precision location we can compute the deceleration required.
            if precise.value or ils_approach:
                speed = gspd.array[land_roll] * scale
                if precise.value:
                    _, dist_to_end = bearings_and_distances(
                        lat.array[land_roll],
                        lon.array[land_roll],
                        rwy.value['end'])
                    time_to_end = dist_to_end / speed
                else:
                    distance_at_tdn = runway_distance_from_end(
                        rwy.value, lat_tdn.get_last().value,
                        lon_tdn.get_last().value)
                    dist_from_td = integrate(gspd.array[land_roll], gspd.hz, scale=scale)
                    time_to_end = (distance_at_tdn - dist_from_td) / speed
                if len(time_to_end) == 0:
                    continue
                limit_point = np.ma.argmin(time_to_end)
                if limit_point < 0.0:  # Some error conditions lead to rogue negative results.
                    continue
                limit_time = time_to_end[limit_point]
                self.create_kpv(limit_point + last_tdwn.index, limit_time)


class DistanceOnLandingFrom60KtToRunwayEnd(KeyPointValueNode):
    '''
    Distance between the point the aircraft crosses 60kts GS and the runway end.
    '''

    can_operate = aeroplane_only
    units = ut.METER
    name = 'Distance On Landing From 60 Kt To Runway End'

    def derive(self,
               gspd=P('Groundspeed'),
               lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               tdwns=KTI('Touchdown'),
               rwy=A('FDR Landing Runway')):

        if ambiguous_runway(rwy):
            return
        last_tdwn = tdwns.get_last()
        if not last_tdwn:
            return
        land_idx = last_tdwn.index
        idx_60 = index_at_value(gspd.array, 60.0, slice(land_idx, None))
        if idx_60 and rwy.value and 'start' in rwy.value:
            # Only work out the distance if we have a reading at 60kts...
            distance = runway_distance_from_end(rwy.value,
                                                lat.array[idx_60],
                                                lon.array[idx_60])
            self.create_kpv(idx_60, distance)  # Metres


class HeadingDuringTakeoff(KeyPointValueNode):
    '''
    Aircraft heading during takeoff.
    
    We take the median heading during the takeoff roll only as this avoids
    problems when turning onto the runway or with drift just after liftoff.
    The value is "assigned" to a time midway through the takeoff roll.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return all_of(('Heading Continuous', 'Transition Hover To Flight', 'Aircraft Type'), available)
        else:
            return all_of(('Heading Continuous', 'Takeoff Roll Or Rejected Takeoff'), available)

    def derive(self,
               hdg=P('Heading Continuous'),
               takeoffs=S('Takeoff Roll Or Rejected Takeoff'),
               ac_type=A('Aircraft Type'),
               toff_helos=S('Transition Hover To Flight')):

        takeoffs = toff_helos if ac_type and ac_type.value == 'helicopter' else takeoffs

        for takeoff in takeoffs:
            if takeoff.slice.start and takeoff.slice.stop:
                index = (takeoff.slice.start + takeoff.slice.stop) / 2.0
                value = np.ma.median(hdg.array[takeoff.slice])
                # median result is rounded as
                # -1.42108547152020037174224853515625E-14 == 360.0
                # which is an invalid value for Heading
                if not np.ma.is_masked(value):
                    self.create_kpv(index, float(np.round(value.data, 8)) % 360.0)


class HeadingTrueDuringTakeoff(KeyPointValueNode):
    '''
    Aircraft true heading during takeoff.
    
    We take the median true heading during the takeoff roll only as this avoids
    problems when turning onto the runway or with drift just after liftoff.
    The value is "assigned" to a time midway through the takeoff roll.

    This KPV has been extended to accommodate helicopter transitions, so that the takeoff
    runway can be identified where the aircraft is operating at a conventional airport.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return all_of(('Heading True Continuous', 'Transition Hover To Flight', 'Aircraft Type'), available)
        else:
            return all_of(('Heading True Continuous', 'Takeoff Roll Or Rejected Takeoff'), available)

    def derive(self,
               hdg_true=P('Heading True Continuous'),
               toff_aeros=S('Takeoff Roll'),
               ac_type=A('Aircraft Type'),
               toff_helos=S('Transition Hover To Flight')):

        takeoffs = toff_aeros
        if ac_type and ac_type.value == 'helicopter':
            takeoffs = toff_helos

        for takeoff in takeoffs:
            if takeoff.slice.start and takeoff.slice.stop:
                index = (takeoff.slice.start + takeoff.slice.stop) / 2.0
                value = np.ma.median(hdg_true.array[takeoff.slice])
                # median result is rounded as
                # -1.42108547152020037174224853515625E-14 == 360.0
                # which is an invalid value for Heading
                if not np.ma.is_masked(value):
                    self.create_kpv(index, float(np.round(value.data, 8)) % 360.0)


class HeadingDuringLanding(KeyPointValueNode):
    '''
    Aircraft's heading during landing.
    
    We take the median heading during the landing roll as this avoids problems
    with drift just before touchdown and heading changes when turning off the
    runway. The value is "assigned" to a time midway through the landing phase.

    This KPV has been extended to accommodate helicopter transitions, so that the landing
    runway can be identified where the aircraft is operating at a conventional airport.
    '''

    units = ut.DEGREE

    def derive(self,
               hdg=P('Heading Continuous'),
               landings=S('Landing Roll'),
               touchdowns=KTI('Touchdown'),
               ldg_turn_off=KTI('Landing Turn Off Runway')):

        for landing in landings:
            # Check the slice is robust.
            touchdown = touchdowns.get_first(within_slice=landing.slice)
            turn_off = ldg_turn_off.get_first(within_slice=landing.slice)
            start = touchdown.index if touchdown else landing.slice.start
            stop = turn_off.index + 1 if turn_off else landing.slice.stop
            if start and stop:
                index = (start + stop) / 2.0
                value = np.ma.median(hdg.array[start:stop])
                # median result is rounded as
                # -1.42108547152020037174224853515625E-14 == 360.0
                # which is an invalid value for Heading
                if not np.ma.is_masked(value):
                    self.create_kpv(index, float(np.round(value.data, 8)) % 360.0)



class HeadingTrueDuringLanding(KeyPointValueNode):
    '''
    Aircraft's true heading during landing.
    
    We take the median heading true during the landing roll as this avoids
    problems with drift just before touchdown and heading changes when turning
    off the runway. The value is "assigned" to a time midway through the
    landing phase.

    This KPV has been extended to accommodate helicopter transitions, so that the landing
    runway can be identified where the aircraft is operating at a conventional airport.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return all_of(('Heading True Continuous', 'Transition Flight To Hover', 'Aircraft Type'), available)
        else:
            return all_of(('Heading True Continuous', 'Landing Roll'), available)

    def derive(self,
               hdg=P('Heading True Continuous'),
               land_aeros=S('Landing Roll'),
               ac_type=A('Aircraft Type'),
               land_helos=S('Transition Flight To Hover')):

        landings = land_helos if ac_type and ac_type.value == 'helicopter' else land_aeros

        for landing in landings:
            # Check the slice is robust.
            if landing.slice.start and landing.slice.stop:
                index = (landing.slice.start + landing.slice.stop) / 2.0
                value = np.ma.median(hdg.array[landing.slice])
                # median result is rounded as
                # -1.42108547152020037174224853515625E-14 == 360.0
                # which is an invalid value for Heading
                if not np.ma.is_masked(value):
                    self.create_kpv(index, float(np.round(value.data, 8)) % 360.0)


class HeadingAtLowestAltitudeDuringApproach(KeyPointValueNode):
    '''
    Heading at the lowest point reached in the approach.
    '''

    units = ut.DEGREE

    def derive(self,
               hdg=P('Heading Continuous'),
               low_points=KTI('Lowest Altitude During Approach')):

        self.create_kpvs_at_ktis(hdg.array % 360.0, low_points)


class HeadingChange(KeyPointValueNode):
    '''
    This determines the heading change made during a turn.
    '''

    units = ut.DEGREE

    def derive(self,
               hdg=P('Heading Continuous'),
               turns=S('Turning In Air')):
        '''
        This determines the heading change made during a turn, while turning\
        at over +/- HEADING_RATE_FOR_FLIGHT_PHASES in the air.        
        '''
        
        for turn in turns:
            start_hdg = hdg.array[turn.slice.start]
            stop_hdg = hdg.array[turn.slice.stop]
            dh = stop_hdg - start_hdg
            if abs(dh) > MIN_HEADING_CHANGE:
                self.create_kpv(turn.slice.stop - 1, stop_hdg - start_hdg)


class ElevatorDuringLandingMin(KeyPointValueNode):
    '''
    Minimum angle of the elevator during landing.
    '''

    units = ut.DEGREE

    def derive(self, elev=P('Elevator'), landing=S('Landing')):

        self.create_kpvs_within_slices(elev.array, landing, min_value)


##############################################################################
# Height Loss


class HeightLossLiftoffTo35Ft(KeyPointValueNode):
    '''
    Altitude loss after takeoff - between liftoff and 35ft AAL.
    
    At these low altitudes, the aircraft is in ground effect, so we use an
    inertial vertical speed to identify small height losses. This means that
    the algorithm will still work with low sample rate (or even missing)
    radio altimeters.
    '''

    can_operate = aeroplane_only

    units = ut.FT

    def derive(self,
               vs=P('Vertical Speed Inertial'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for climb in alt_aal.slices_from_to(0, 35):
            array = np.ma.masked_greater_equal(vs.array[climb], 0.0)
            drops = np.ma.clump_unmasked(array)
            for drop in drops:
                ht_loss = integrate(vs.array[drop], vs.frequency)
                # Only interested in peak value - by definition the last value:
                if ht_loss[-1]:
                    self.create_kpv(drop.stop, abs(ht_loss[-1]))


class HeightLoss35To1000Ft(KeyPointValueNode):
    '''
    Altitude loss after takeoff - between 35ft AAL and 1000ft AAL.
    '''

    can_operate = aeroplane_only

    units = ut.FT

    def derive(self,
               ht_loss=P('Descend For Flight Phases'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               init_climb=S('Initial Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 35, 1000)
        alt_climb_sections = valid_slices_within_array(alt_band, init_climb)

        for climb in alt_climb_sections:
            index, value = min_value(ht_loss.array, climb)
            # Only report a positive value where height is lost:
            if index and value < 0:
                self.create_kpv(index, abs(value))


class HeightLoss1000To2000Ft(KeyPointValueNode):
    '''
    Altitude loss after takeoff - between 1000ft AAL and 2000ft AAL.
    '''

    can_operate = aeroplane_only

    units = ut.FT

    def derive(self,
               ht_loss=P('Descend For Flight Phases'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               climbs=S('Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 1000, 2000)
        alt_climb_sections = valid_slices_within_array(alt_band, climbs)

        for climb in alt_climb_sections:
            index, value = min_value(ht_loss.array, climb)
            # Only report a positive value where height is lost:
            if index and value < 0:
                self.create_kpv(index, abs(value))


##############################################################################
# ILS


class ILSFrequencyDuringApproach(KeyPointValueNode):
    '''
    Determine the ILS frequency during approach.

    The period when the aircraft was continuously established on the ILS and
    descending to the minimum point on the approach is already defined as a
    flight phase. This KPV just picks up the frequency tuned at that point.
    '''

    name = 'ILS Frequency During Approach'
    units = ut.MHZ

    def derive(self, apps=A('Approach Information')):

        for app in apps:
            frequency = app.ils_freq
            if app.loc_est and frequency:
                # Set the KPV index to the start of this ILS approach:
                self.create_kpv(app.loc_est.start, frequency)


class ILSGlideslopeDeviation1500To1000FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the glideslope between 1500 and 
    1000 ft AAL (AGL for helicopters).
    '''

    name = 'ILS Glideslope Deviation 1500 To 1000 Ft Max'
    units = ut.DOTS

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['ILS Glideslope', 'ILS Glideslope Established']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               ils_glideslope=P('ILS Glideslope'),
               ils_ests=S('ILS Glideslope Established'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_between(1000, 1500)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            desc_ils_bands = slices_and(ils_bands, descending.get_slices())
            self.create_kpvs_within_slices(
                ils_glideslope.array,
                desc_ils_bands,
                max_abs_value,
                min_duration=HOVER_MIN_DURATION,
                freq=ils_glideslope.frequency)
        else:
            alt_bands = alt_aal.slices_from_to(1500, 1000)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            self.create_kpvs_within_slices(
                ils_glideslope.array,
                ils_bands,
                max_abs_value)


class ILSGlideslopeDeviation1000To500FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the glideslope between 1000 and 
    500 ft AAL (AGL for helicopters).
    '''

    name = 'ILS Glideslope Deviation 1000 To 500 Ft Max'
    units = ut.DOTS

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['ILS Glideslope', 'ILS Glideslope Established']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               ils_glideslope=P('ILS Glideslope'),
               ils_ests=S('ILS Glideslope Established'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_between(500, 1000)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            desc_ils_bands = slices_and(ils_bands, descending.get_slices())
            self.create_kpvs_within_slices(
                ils_glideslope.array,
                desc_ils_bands,
                max_abs_value,
                min_duration=HOVER_MIN_DURATION,
                freq=ils_glideslope.frequency)
        else:
            alt_bands = alt_aal.slices_from_to(1000, 500)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            self.create_kpvs_within_slices(
                ils_glideslope.array,
                ils_bands,
                max_abs_value)


class ILSGlideslopeDeviation500To200FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the glideslope between 500 and 
    200 ft AAL (AGL for helicopters).
    '''

    name = 'ILS Glideslope Deviation 500 To 200 Ft Max'
    units = ut.DOTS

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['ILS Glideslope', 'ILS Glideslope Established']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               ils_glideslope=P('ILS Glideslope'),
               ils_ests=S('ILS Glideslope Established'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_between(200, 500)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            desc_ils_bands = slices_and(ils_bands, descending.get_slices())
            self.create_kpvs_within_slices(
                ils_glideslope.array,
                desc_ils_bands,
                max_abs_value,
                min_duration=HOVER_MIN_DURATION,
                freq=ils_glideslope.frequency)
        else:
            alt_bands = alt_aal.slices_from_to(500, 200)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            self.create_kpvs_within_slices(
                ils_glideslope.array,
                ils_bands,
                max_abs_value)


class ILSGlideslopeNotEstablishedHighestAltitude1000To200Ft(KeyPointValueNode):
    '''
    Determine the highest altittude where the ILS glideslope first deviates (>0.5 dots or <-0.5 dots)
    within the altitude band 1000 to 200 ft
    '''
    
    name = 'ILS Glideslope Not Established Highest Altitude 1000 To 200Ft'
    units = ut.FT
    
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return all_of(('Altitude AGL','ILS Glideslope', 'ILS Localizer'), available)
        else:
            return all_of(('Altitude AAL For Flight Phases', 'ILS Glideslope', 'ILS Localizer'), available)
        
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               alt_agl=P('Altitude AGL'),
               ac_type=A('Aircraft Type'),
               approach=S('Approach'),
               ils_gs=P('ILS Glideslope'),
               ils_loc=P('ILS Localizer')):
        
        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_from_to(1000, 200)
            approach_bands = slices_and(alt_bands, approach.get_slices())
        else:
            approach_bands = alt_aal.slices_from_to(1000, 200)
        
        for approach_slice in approach_bands:
            ils_stop = None
            if ils_established(ils_loc.array, approach_slice, ils_loc.frequency):
                ils_start = ils_established(ils_gs.array, approach_slice, ils_gs.frequency)
                if ils_start:
                    if ils_start == approach_slice.start:
                        for ils_index in range(int(ils_start), int(approach_slice.stop)+1):
                            if abs(ils_gs.array[ils_index]) > 0.5 or ils_index == approach_slice.stop:
                                if ac_type and ac_type.value == 'helicopter':
                                    ils_stop = alt_agl.array[ils_index]
                                    self.create_kpv(ils_index, ils_stop)
                                else:
                                    ils_stop = alt_aal.array[ils_index]
                                    self.create_kpv(ils_index, ils_stop)
                                break
                    elif ils_start != approach_slice.start:
                        ils_stop = approach_slice.start
                        self.create_kpv(ils_stop, 1000)
            if ils_stop is None:
                self.create_kpv(approach_slice.start, 1000)


class ILSGlideslopeNotEstablishedHighestAltitude500To200Ft(KeyPointValueNode):
    '''
    Determine the highest altittude where the ILS glideslope first deviates (>0.5 dots or <-0.5 dots)
    within the altitude band 500 to 200 ft
    '''
    
    name = 'ILS Glideslope Not Established Highest Altitude 500 To 200 Ft'
    units = ut.FT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return all_of(('Altitude AGL','ILS Glideslope', 'ILS Localizer'), available)
        else:
            return all_of(('Altitude AAL For Flight Phases', 'ILS Glideslope', 'ILS Localizer'), available)
        
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               alt_agl=P('Altitude AGL'),
               ac_type=A('Aircraft Type'),
               approach=S('Approach'),
               ils_gs=P('ILS Glideslope'),
               ils_loc=P('ILS Localizer')):
        
        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_from_to(500, 200)
            approach_bands = slices_and(alt_bands, approach.get_slices())
        else:
            approach_bands = alt_aal.slices_from_to(500, 200)
            
        for approach_slice in approach_bands:
            ils_stop = None
            if ils_established(ils_loc.array, approach_slice, ils_loc.frequency):
                ils_start = ils_established(ils_gs.array, approach_slice, ils_gs.frequency)
                if ils_start:
                    if ils_start == approach_slice.start:
                        for ils_index in range(int(ils_start), int(approach_slice.stop)+1):
                            if abs(ils_gs.array[ils_index]) > 0.5 or ils_index == approach_slice.stop:
                                if ac_type and ac_type.value == 'helicopter':
                                    ils_stop = alt_agl.array[ils_index]
                                    self.create_kpv(ils_index, ils_stop)
                                else:
                                    ils_stop = alt_aal.array[ils_index]
                                    self.create_kpv(ils_index, ils_stop)
                                break
                    elif ils_start != approach_slice.start:
                        ils_stop = approach_slice.start
                        self.create_kpv(ils_stop, 500)
            if ils_stop is None:
                self.create_kpv(approach_slice.start, 500)


class ILSLocalizerNotEstablishedHighestAltitude1000To200Ft(KeyPointValueNode):
    '''
    Determine the highest altittude where the ILS localizer first deviates (>0.5 dots or <-0.5 dots)
    within the altitude band 1000 to 200 ft
    '''
    
    name = 'ILS Localizer Not Established Highest Altitude 1000 To 200 Ft'
    units = ut.FT
    
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return all_of(('Altitude AGL','ILS Localizer'), available)
        else:
            return all_of(('Altitude AAL For Flight Phases', 'ILS Localizer'), available)
        
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               alt_agl=P('Altitude AGL'),
               ac_type=A('Aircraft Type'),
               approach=S('Approach'),
               ils_loc=P('ILS Localizer')):
            
        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_from_to(1000, 200)
            approach_bands = slices_and(alt_bands, approach.get_slices())
        else:
            approach_bands = alt_aal.slices_from_to(1000, 200)
            
        for approach_slice in approach_bands:
            ils_stop = None
            ils_start = ils_established(ils_loc.array, approach_slice, ils_loc.frequency)
            if ils_start:
                if ils_start == approach_slice.start:
                    for ils_index in range(int(ils_start), int(approach_slice.stop)+1):
                        if abs(ils_loc.array[ils_index]) > 0.5 or ils_index == approach_slice.stop:
                            if ac_type and ac_type.value == 'helicopter':
                                ils_stop = alt_agl.array[ils_index]
                                self.create_kpv(ils_index, ils_stop)
                            else:
                                ils_stop = alt_aal.array[ils_index]
                                self.create_kpv(ils_index, ils_stop)
                            break
                elif ils_start != approach_slice.start:
                    ils_stop = approach_slice.start
                    self.create_kpv(ils_stop, 1000)
            if ils_stop is None:
                self.create_kpv(approach_slice.start, 1000)


class ILSLocalizerNotEstablishedHighestAltitude500To200Ft(KeyPointValueNode):
    '''
    Determine the highest altittude where the ILS localizer first deviates (>0.5 dots or <-0.5 dots)
    within the altitude band 500 to 200 ft
    '''
    
    name = 'ILS Localizer Not Established Highest Altitude 500 To 200 Ft'
    units = ut.FT
    
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return all_of(('Altitude AGL','ILS Localizer'), available)
        else:
            return all_of(('Altitude AAL For Flight Phases', 'ILS Localizer'), available)
        
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               alt_agl=P('Altitude AGL'),
               ac_type=A('Aircraft Type'),
               approach=S('Approach'),
               ils_loc=P('ILS Localizer'),
               ils_loc_ests=S('ILS Localizer Established')):
    
        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_from_to(500, 200)
            approach_bands = slices_and(alt_bands, approach.get_slices())
        else:
            approach_bands = alt_aal.slices_from_to(500, 200)
            
        for approach_slice in approach_bands:
            ils_stop = None
            ils_start = ils_established(ils_loc.array, approach_slice, ils_loc.frequency)
            if ils_start:
                if ils_start == approach_slice.start:
                    for ils_index in range(int(ils_start), int(approach_slice.stop)+1):
                        if abs(ils_loc.array[ils_index]) > 0.5 or ils_index == approach_slice.stop:
                            if ac_type and ac_type.value == 'helicopter':
                                ils_stop = alt_agl.array[ils_index]
                                self.create_kpv(ils_index, ils_stop)
                            else:
                                ils_stop = alt_aal.array[ils_index]
                                self.create_kpv(ils_index, ils_stop)
                            break
                elif ils_start != approach_slice.start:
                    ils_stop = approach_slice.start
                    self.create_kpv(ils_stop, 500)
            if ils_stop is None:
                self.create_kpv(approach_slice.start, 500)


class ILSLocalizerDeviation1500To1000FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the localizer between 1500 and 
    1000 ft AAL (AGL for helicopters).
    '''

    name = 'ILS Localizer Deviation 1500 To 1000 Ft Max'
    units = ut.DOTS

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['ILS Localizer', 'ILS Localizer Established']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               ils_localizer=P('ILS Localizer'),
               ils_ests=S('ILS Localizer Established'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_between(1000, 1500)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            desc_ils_bands = slices_and(ils_bands, descending.get_slices())
            self.create_kpvs_within_slices(
                ils_localizer.array,
                desc_ils_bands,
                max_abs_value,
                min_duration=HOVER_MIN_DURATION,
                freq=ils_localizer.frequency)
        else:
            alt_bands = alt_aal.slices_from_to(1500, 1000)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            self.create_kpvs_within_slices(
                ils_localizer.array,
                ils_bands,
                max_abs_value)


class ILSGlideslope10SecBeforeEstablished(KeyPointValueNode):
    '''
    Glideslope dot value 10 seconds from glideslope established.  
    '''

    name = 'ILS Glideslope 10 Sec Before Established'
    units = ut.DOTS
    
    can_operate = aeroplane_only
    
    def derive(self,
               ils_glideslope=P('ILS Glideslope'),
               ils_established=S('ILS Glideslope Established')):

        for established_slice in ils_established.get_slices():
            index = established_slice.start - 10*ils_glideslope.hz 
            value = value_at_index(ils_glideslope.array, index)
            self.create_kpv(index, value)



class ILSLocalizerDeviation1000To500FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the localizer between 1000 and 
    500 ft AAL (AGL for helicopters).
    '''

    name = 'ILS Localizer Deviation 1000 To 500 Ft Max'
    units = ut.DOTS

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['ILS Localizer', 'ILS Localizer Established']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               ils_localizer=P('ILS Localizer'),
               ils_ests=S('ILS Localizer Established'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_between(500, 1000)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            desc_ils_bands = slices_and(ils_bands, descending.get_slices())
            self.create_kpvs_within_slices(
                ils_localizer.array,
                desc_ils_bands,
                max_abs_value,
                min_duration=HOVER_MIN_DURATION,
                freq=ils_localizer.frequency)
        else:
            alt_bands = alt_aal.slices_from_to(1000, 500)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            self.create_kpvs_within_slices(
                ils_localizer.array,
                ils_bands,
                max_abs_value)


class ILSLocalizerDeviation500To200FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the localizer between 500 and 
    200 ft AAL (AGL for helicopters).
    '''

    name = 'ILS Localizer Deviation 500 To 200 Ft Max'
    units = ut.DOTS

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['ILS Localizer', 'ILS Localizer Established']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               ils_localizer=P('ILS Localizer'),
               ils_ests=S('ILS Localizer Established'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_bands = alt_agl.slices_between(200, 500)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            desc_ils_bands = slices_and(ils_bands, descending.get_slices())
            self.create_kpvs_within_slices(
                ils_localizer.array,
                desc_ils_bands,
                max_abs_value,
                min_duration=HOVER_MIN_DURATION,
                freq=ils_localizer.frequency)
        else:
            alt_bands = alt_aal.slices_from_to(500, 200)
            ils_bands = slices_and(alt_bands, ils_ests.get_slices())
            self.create_kpvs_within_slices(
                ils_localizer.array,
                ils_bands,
                max_abs_value)


class ILSLocalizerDeviationAtTouchdown(KeyPointValueNode):
    '''
    Lateral deviation at touchdown from Localiser. Please note that it 
    might not always be possible to determine how close to runway edge the 
    aircraft was at touchdown using localiser parameter as runway widths 
    vary and there are different localiser beam centreline error margins 
    for different ILS categories. 
    '''

    name = 'ILS Localizer Deviation At Touchdown'
    units = ut.DOTS

    def derive(self,
               ils_localizer=P('ILS Localizer'),
               ils_ests=S('ILS Localizer Established'),
               tdwns=KTI('Touchdown')):
        '''
        ILS Localizer Deviation At Touchdown Measurements at <2 deg pitch after main
        gear TD."

        The ILS Established period may not last until touchdown, so it is
        artificially extended by a minute to ensure coverage of the touchdown
        instant.
        '''
        for ils_est in ils_ests:
            for tdwn in tdwns:
                ext_end = ils_est.slice.stop + ils_localizer.frequency * 60.0
                ils_est_ext = slice(ils_est.slice.start, ext_end)
                if not is_index_within_slice(tdwn.index, ils_est_ext):
                    continue
                deviation = value_at_index(ils_localizer.array, tdwn.index)
                self.create_kpv(tdwn.index, deviation)


class IANGlidepathDeviationMax(KeyPointValueNode):
    '''
    Deviation from IAN glidepath. Altitudes in KPV name are AAL.
    '''

    NAME_FORMAT = 'IAN Glidepath Deviation %(max_alt)d To %(min_alt)s Ft Max'
    NAME_VALUES = {
        'max_alt': (1500, 1000, 500),
        'min_alt': (1000, 500, 200),
    }
    name = 'IAN Glidepath Deviation'
    units = ut.DOTS

    def derive(self,
               ian_glidepath=P('IAN Glidepath'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               ian_est=S('IAN Glidepath Established')):

        for idx in range(len(self.NAME_VALUES['max_alt'])):
            max_alt = self.NAME_VALUES['max_alt'][idx]
            min_alt = self.NAME_VALUES['min_alt'][idx]
            alt_bands = alt_aal.slices_from_to(max_alt, min_alt)

            ian_est_bands = slices_and(alt_bands, ian_est.get_slices())

            self.create_kpvs_within_slices(
                ian_glidepath.array,
                ian_est_bands,
                max_abs_value,
                max_alt=max_alt,
                min_alt=min_alt
            )


class IANFinalApproachCourseDeviationMax(KeyPointValueNode):
    '''
    Deviation from IAN final approach course. Altitudes in KPV name are AAL.
    '''

    NAME_FORMAT = 'IAN Final Approach Course Deviation %(max_alt)d To %(min_alt)s Ft Max'
    NAME_VALUES = {
        'max_alt': (1500, 1000, 500),
        'min_alt': (1000, 500, 200),
    }
    name = 'IAN Final Approach Course Deviation'
    units = ut.DOTS

    def derive(self,
               ian_final=P('IAN Final Approach Course'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               ian_est=S('IAN Final Approach Course Established')):

        for idx in range(len(self.NAME_VALUES['max_alt'])):
            max_alt = self.NAME_VALUES['max_alt'][idx]
            min_alt = self.NAME_VALUES['min_alt'][idx]

            alt_bands = alt_aal.slices_from_to(max_alt, min_alt)
            ian_est_bands = slices_and(alt_bands, ian_est.get_slices())

            self.create_kpvs_within_slices(
                ian_final.array,
                ian_est_bands,
                max_abs_value,
                max_alt=max_alt,
                min_alt=min_alt
            )


##############################################################################


class IsolationValveOpenAtLiftoff(KeyPointValueNode):
    '''
    Isolation valve state at liftoff.
    '''

    units = None

    def derive(self,
               isol=M('Isolation Valve Open'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(isol.array.raw, liftoffs, suppress_zeros=True)


class PackValvesOpenAtLiftoff(KeyPointValueNode):
    '''
    Pack valves state at liftoff.
    '''

    units = None

    def derive(self,
               pack=M('Pack Valves Open'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(pack.array.raw, liftoffs, suppress_zeros=True)


##############################################################################
# Latitude/Longitude


########################################
# Helpers


def calculate_runway_midpoint(rwy):
    '''
    Attempts to calculate the runway midpoint data provided in the AFR.

    1. If there are no runway start coordinates, use the runway end coordinates
    2. If there are no runway end coordinates, use the runway start coordinates
    3. Attempt to calculate the midpoint of the great circle path between them.
    '''
    rwy_s = rwy.get('start', {})
    rwy_e = rwy.get('end', {})
    lat_s = rwy_s.get('latitude')
    lat_e = rwy_e.get('latitude')
    lon_s = rwy_s.get('longitude')
    lon_e = rwy_e.get('longitude')
    if lat_s is None or lon_s is None:
        return (lat_e, lon_e)
    if lat_e is None or lon_e is None:
        return (lat_s, lon_s)
    return midpoint(lat_s, lon_s, lat_e, lon_e)


########################################
# Latitude/Longitude @ Takeoff/Landing


class LatitudeAtTouchdown(KeyPointValueNode):
    '''
    Aircraft's latitude (non-smoothed) at Touchdown.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        '''
        The position of the landing is recorded in the form of KPVs as this is
        used in a number of places. From the touchdown moments, the raw latitude
        and longitude data is used to create the *AtTouchdown parameters, and these
        are in turn used to compute the landing attributes.
       
        Once the landing attributes (especially the runway details) are known,
        the positional data can be smoothed using ILS data or (if this is a
        non-precision approach) the known touchdown aiming point. With more
        accurate positional data the touchdown point can be computed more
        accurately.
       
        Note: Cannot use smoothed position as this causes circular dependancy.
        '''
        return 'Touchdown' in available and any_of(('Latitude',
                                                    'Latitude (Coarse)',
                                                    'AFR Landing Runway',
                                                    'AFR Landing Airport'),
                                                   available)

    def derive(self,
               lat=P('Latitude'),
               tdwns=KTI('Touchdown'),
               land_afr_apt=A('AFR Landing Airport'),
               land_afr_rwy=A('AFR Landing Runway'),
               lat_c=P('Latitude (Coarse)'),
               ac_type=A('Aircraft Type'),
               land_helos=KTI('Enter Transition Flight To Hover')):
        '''
        Note that Latitude (Coarse) is a superframe parameter with poor
        resolution recorded on some FDAUs. Keeping it at the end of the list
        of parameters means that it will be aligned to a higher sample rate
        rather than dragging other parameters down to its sample rate. See
        767 Delta data frame.
        '''
        # 1. Attempt to use latitude parameter if available:
        if lat:
            if ac_type and ac_type.value == 'helicopter' and land_helos:
                self.create_kpvs_at_ktis(lat.array, land_helos)
            else:
                self.create_kpvs_at_ktis(lat.array, tdwns)
            return

        if lat_c:
            for tdwn in tdwns:
                # Touchdown may be masked for Coarse parameter.
                self.create_kpv(
                    tdwn.index,
                    closest_unmasked_value(lat_c.array, tdwn.index).value)
            return

        value = None

        # 2a. Attempt to use latitude of runway midpoint:
        if value is None and land_afr_rwy:
            lat_m, lon_m = calculate_runway_midpoint(land_afr_rwy.value)
            value = lat_m

        # 2b. Attempt to use latitude of airport:
        if value is None and land_afr_apt:
            value = land_afr_apt.value.get('latitude')

        if value is not None:
            self.create_kpv(tdwns[-1].index, value)
            return

        # XXX: Is there something else we can do here other than fail?
        raise Exception('Unable to determine a latitude at touchdown.')


class LongitudeAtTouchdown(KeyPointValueNode):
    '''
    Aircraft's longitude (non-smoothed) at Touchdown.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):
        return 'Touchdown' in available and any_of(('Longitude',
                                                    'Longitude (Coarse)',
                                                    'AFR Landing Runway',
                                                    'AFR Landing Airport'),
                                                   available)

    def derive(self,
               lon=P('Longitude'),
               tdwns=KTI('Touchdown'),
               land_afr_apt=A('AFR Landing Airport'),
               land_afr_rwy=A('AFR Landing Runway'),
               lon_c=P('Longitude (Coarse)'),
               ac_type=A('Aircraft Type'),
               land_helos=KTI('Exit Transition Flight To Hover')):
        '''
        See note relating to coarse latitude and longitude under Latitude At Touchdown
        '''
        # 1. Attempt to use longitude parameter if available:
        if lon:
            if ac_type and ac_type.value == 'helicopter' and land_helos:
                self.create_kpvs_at_ktis(lon.array, land_helos)
            else:
                self.create_kpvs_at_ktis(lon.array, tdwns)
            return

        if lon_c:
            for tdwn in tdwns:
                # Touchdown may be masked for Coarse parameter.
                self.create_kpv(
                    tdwn.index,
                    closest_unmasked_value(lon_c.array, tdwn.index).value,
                )
            return

        value = None

        # 2a. Attempt to use longitude of runway midpoint:
        if value is None and land_afr_rwy:
            lat_m, lon_m = calculate_runway_midpoint(land_afr_rwy.value)
            value = lon_m

        # 2b. Attempt to use longitude of airport:
        if value is None and land_afr_apt:
            value = land_afr_apt.value.get('longitude')

        if value is not None:
            self.create_kpv(tdwns[-1].index, value)
            return

        # XXX: Is there something else we can do here other than fail?
        raise Exception('Unable to determine a longitude at touchdown.')


class LatitudeAtLiftoff(KeyPointValueNode):
    '''
    Aircraft's latitude (non-smoothed) at Liftoff.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = 'Exit Transition Hover To Flight' if ac_type and ac_type.value == 'helicopter' else 'Liftoff'
        return required in available and any_of(('Latitude',
                                                  'Latitude (Coarse)',
                                                  'AFR Takeoff Runway',
                                                  'AFR Takeoff Airport'),
                                                 available)

    def derive(self,
               lat=P('Latitude'),
               liftoffs=KTI('Liftoff'),
               toff_afr_apt=A('AFR Takeoff Airport'),
               toff_afr_rwy=A('AFR Takeoff Runway'),
               lat_c=P('Latitude (Coarse)'),
               ac_type=A('Aircraft Type'),
               toff_helos=KTI('Exit Transition Hover To Flight')):
        '''
        Note that Latitude Coarse is a superframe parameter with poor
        resolution recorded on some FDAUs. Keeping it at the end of the list
        of parameters means that it will be aligned to a higher sample rate
        rather than dragging other parameters down to its sample rate. See
        767 Delta data frame.
        '''
        ktis = liftoffs
        if ac_type and ac_type.value == 'helicopter':
            # If the helicopter transitioned cleanly this may be a better definition of the
            # point of takeoff, certainly when the transition took place over a runway.
            if toff_helos:
                ktis = toff_helos
        # 1. Attempt to use latitude parameter if available:
        if lat:
            if ktis:
                self.create_kpvs_at_ktis(lat.array, ktis)
                return

        if lat_c:
            for lift in ktis:
                # Touchdown may be masked for Coarse parameter.
                self.create_kpv(
                    lift.index,
                    closest_unmasked_value(lat_c.array, lift.index).value,
                )
            return

        value = None

        # 2a. Attempt to use latitude of runway midpoint:
        if toff_afr_rwy:
            lat_m, lon_m = calculate_runway_midpoint(toff_afr_rwy.value)
            value = lat_m

        # 2b. Attempt to use latitude of airport:
        if value is None and toff_afr_apt:
            value = toff_afr_apt.value.get('latitude')

        if value is not None:
            first = ktis.get_first()
            if first:
                self.create_kpv(first.index, value)
            return

        # XXX: Is there something else we can do here other than fail?
        # raise Exception('Unable to determine a latitude at liftoff.')
        value = None

class LongitudeAtLiftoff(KeyPointValueNode):
    '''
    Aircraft's longitude (non-smoothed) at Liftoff.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = 'Liftoff'
        if ac_type and ac_type.value == 'helicopter':
            required = 'Exit Transition Hover To Flight'

        return required in available and any_of(('Longitude',
                                                  'Longitude (Coarse)',
                                                  'AFR Takeoff Runway',
                                                  'AFR Takeoff Airport'),
                                                 available)

    def derive(self,
               lon=P('Longitude'),
               liftoffs=KTI('Liftoff'),
               toff_afr_apt=A('AFR Takeoff Airport'),
               toff_afr_rwy=A('AFR Takeoff Runway'),
               lon_c=P('Longitude (Coarse)'),
               ac_type=A('Aircraft Type'),
               toff_helos=KTI('Exit Transition Hover To Flight')):
        '''
        See note relating to coarse latitude and longitude under Latitude At Takeoff
        '''
        ktis = liftoffs
        if ac_type and ac_type.value == 'helicopter':
            # If the helicopter transitioned cleanly this may be a better definition of the
            # point of takeoff, certainly when the transition took place over a runway.
            if toff_helos:
                ktis = toff_helos
        # 1. Attempt to use longitude parameter if available:
        if lon:
            if ktis:
                self.create_kpvs_at_ktis(lon.array, ktis)
                return

        if lon_c:
            for lift in liftoffs:
                # Touchdown may be masked for Coarse parameter.
                self.create_kpv(
                    lift.index,
                    closest_unmasked_value(lon_c.array, lift.index).value,
                )
            return

        value = None

        # 2a. Attempt to use longitude of runway midpoint:
        if value is None and toff_afr_rwy:
            lat_m, lon_m = calculate_runway_midpoint(toff_afr_rwy.value)
            value = lon_m

        # 2b. Attempt to use longitude of airport:
        if value is None and toff_afr_apt:
            value = toff_afr_apt.value.get('longitude')

        if value is not None:
            first = ktis.get_first()
            if first:
                self.create_kpv(first.index, value)
            return

        # XXX: Is there something else we can do here other than fail?
        # raise Exception('Unable to determine a longitude at liftoff.')
        value = None

class LatitudeOffBlocks(KeyPointValueNode):
    '''
    Aircraft's latitude (non-smoothed) off blocks.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return 'Off Blocks' in available and any_of(
            ('Latitude', 'Latitude (Coarse)'), available)

    def derive(self,
               lat=P('Latitude'),
               off_blocks=KTI('Off Blocks'),
               lat_c=P('Latitude (Coarse)')):
        '''
        Note that Latitude Coarse is a superframe parameter with poor
        resolution recorded on some FDAUs. Keeping it at the end of the list
        of parameters means that it will be aligned to a higher sample rate
        rather than dragging other parameters down to its sample rate. See
        767 Delta data frame.
        '''
        if lat:
            self.create_kpvs_at_ktis(lat.array, off_blocks)
            return
        if lat_c:
            self.create_kpvs_at_ktis(lat_c.array, off_blocks)
            return


class LongitudeOffBlocks(KeyPointValueNode):
    '''
    Aircraft's longitude (non-smoothed) off blocks.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return 'Off Blocks' in available and any_of(
            ('Longitude', 'Longitude (Coarse)',), available)

    def derive(self,
               lon=P('Longitude'),
               off_blocks=KTI('Off Blocks'),
               lon_c=P('Longitude (Coarse)')):
        '''
        See note relating to coarse latitude and longitude under Latitude At Takeoff
        '''
        if lon:
            self.create_kpvs_at_ktis(lon.array, off_blocks)
            return
        if lon_c:
            self.create_kpvs_at_ktis(lon_c.array, off_blocks)
            return

class LatitudeAtTakeoffAccelerationStart(KeyPointValueNode):
    '''
    Latitude (non-smoothed) At Takeoff Acceleration Start.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return 'Takeoff Acceleration Start' in available and any_of(
            ('Latitude', 'Latitude (Coarse)'), available)

    def derive(self,
               lat=P('Latitude'),
               toff_accel=KTI('Takeoff Acceleration Start'),
               lat_c=P('Latitude (Coarse)')):
        '''
        Note that Latitude Coarse is a superframe parameter with poor
        resolution recorded on some FDAUs. Keeping it at the end of the list
        of parameters means that it will be aligned to a higher sample rate
        rather than dragging other parameters down to its sample rate. See
        767 Delta data frame.
        '''
        if lat:
            self.create_kpvs_at_ktis(lat.array, toff_accel)
            return
        if lat_c:
            self.create_kpvs_at_ktis(lat_c.array, toff_accel)
            return


class LongitudeAtTakeoffAccelerationStart(KeyPointValueNode):
    '''
    Longitude (non-smoothed) At Takeoff Acceleration Start.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return 'Takeoff Acceleration Start' in available and any_of(
            ('Longitude', 'Longitude (Coarse)',), available)

    def derive(self,
               lon=P('Longitude'),
               toff_accel=KTI('Takeoff Acceleration Start'),
               lon_c=P('Longitude (Coarse)')):
        '''
        See note relating to coarse latitude and longitude under Latitude At Takeoff
        '''
        if lon:
            self.create_kpvs_at_ktis(lon.array, toff_accel)
            return
        if lon_c:
            self.create_kpvs_at_ktis(lon_c.array, toff_accel)
            return


########################################
# Latitude/Longitude @ Liftoff/Touchdown


class LatitudeSmoothedAtTouchdown(KeyPointValueNode):
    '''
    Latitude (smoothed) at Touchdown.
    '''

    units = ut.DEGREE

    def derive(self, lat=P('Latitude Smoothed'), tdwns=KTI('Touchdown')):
        '''
        The position of the landing is recorded in the form of KPVs as this is
        used in a number of places. From the touchdown moments, the raw latitude
        and longitude data is used to create the *AtTouchdown parameters, and these
        are in turn used to compute the landing attributes.
     
        Once the landing attributes (especially the runway details) are known,
        the positional data can be smoothed using ILS data or (if this is a
        non-precision approach) the known touchdown aiming point. With more
        accurate positional data the touchdown point can be computed more
        accurately.
        '''
        self.create_kpvs_at_ktis(lat.array, tdwns)


class LongitudeSmoothedAtTouchdown(KeyPointValueNode):
    '''
    Longitude (smoothed) at Touchdown.
    '''

    units = ut.DEGREE

    def derive(self, lon=P('Longitude Smoothed'), tdwns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(lon.array, tdwns)


class LatitudeSmoothedAtLiftoff(KeyPointValueNode):
    '''
    Latitude (smoothed) at Liftoff.
    '''

    units = ut.DEGREE

    def derive(self, lat=P('Latitude Smoothed'), liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(lat.array, liftoffs)


class LongitudeSmoothedAtLiftoff(KeyPointValueNode):
    '''
    Longitude (smoothed) at Liftoff.
    '''

    units = ut.DEGREE

    def derive(self, lon=P('Longitude Smoothed'), liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(lon.array, liftoffs)


#########################################
# Latitude/Longitude @ Lowest Point on approach. Used to identify airport
# and runway, so that this works for both landings and aborted approaches /
# go-arounds.

class LatitudeAtLowestAltitudeDuringApproach(KeyPointValueNode):
    '''
    Latitude (non-smoothed) at lowest altitude (AAL) during approach.
    '''

    units = ut.DEGREE

    def derive(self,
               lat=P('Latitude Prepared'),
               low_points=KTI('Lowest Altitude During Approach')):
        '''
        Note: Cannot use smoothed position as this causes circular dependancy.
        '''
        self.create_kpvs_at_ktis(lat.array, low_points)


class LongitudeAtLowestAltitudeDuringApproach(KeyPointValueNode):
    '''
    Longitude (non-smoothed) at lowest altitude (AAL) during approach.
    '''

    units = ut.DEGREE

    def derive(self,
               lon=P('Longitude Prepared'),
               low_points=KTI('Lowest Altitude During Approach')):
        '''
        Note: Cannot use smoothed position as this causes circular dependancy.
        '''
        self.create_kpvs_at_ktis(lon.array, low_points)


##############################################################################
# Mach


########################################
# Mach: General


class MachMax(KeyPointValueNode):
    '''
    Maximum Mach speed reached while airborne.
    '''

    units = ut.MACH

    def derive(self,
               mach=P('Mach'),
               airs=S('Airborne')):

        self.create_kpvs_within_slices(mach.array, airs, max_value)


class MachDuringCruiseAvg(KeyPointValueNode):
    '''
    Average Mach speed during cruise.
    '''

    units = ut.MACH

    def derive(self,
               mach=P('Mach'),
               cruises=S('Cruise')):

        for _slice in cruises.get_slices():
            self.create_kpv(_slice.start + (_slice.stop - _slice.start) // 2,
                            np.ma.mean(mach.array[_slice]))


class MachAboveFL200Max(KeyPointValueNode):
    '''
    Maximum Mach speed above FL200 (Alt STD Smoothed)
    '''

    name = 'Mach Above FL200 Max'
    units = ut.MACH

    def derive(self,
               mach=P('Mach'),
               alt=P('Altitude STD Smoothed')):
        
        alt.array = nearest_neighbour_mask_repair(alt.array)
        self.create_kpvs_within_slices(mach.array, 
                                       alt.slices_above(20000), max_value)
        
class MachAboveFL200Min(KeyPointValueNode):
    '''
    Minimum Mach speed above FL200 (Alt STD Smoothed)
    '''

    name = 'Mach Above FL200 Min'
    units = ut.MACH

    def derive(self,
               mach=P('Mach'),
               alt=P('Altitude STD Smoothed')):
        
        alt.array = nearest_neighbour_mask_repair(alt.array)
        self.create_kpvs_within_slices(mach.array, 
                                       alt.slices_above(20000), min_value)
########################################
# Mach: Flap


class MachWithFlapMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    Maximum value of Mach for each flap detent.

    Note that this KPV uses the flap lever angle, not the flap surface angle.
    '''

    NAME_FORMAT = 'Mach With Flap %(flap)s Max'
    NAME_VALUES = NAME_VALUES_LEVER

    units = ut.MACH

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Mach', 'Fast'), available)

    def derive(self,
               mach=P('Mach'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               scope=S('Fast')):

        # Fast scope traps flap changes very late on the approach and raising
        # flaps before 80kn on the landing run.
        flap = flap_lever or flap_synth
        data = self.flap_or_conf_max_or_min(flap, mach, max_value, scope)
        for index, value, detent in data:
            self.create_kpv(index, value, flap=detent)


########################################
# Mach: Landing Gear


class MachWithGearDownMax(KeyPointValueNode):
    '''
    Maximum Mach speed reached with Gear Down selected.
    '''

    units = ut.MACH

    def derive(self,
               mach=P('Mach'),
               gear=M('Gear Down'),
               airs=S('Airborne')):

        gear.array[gear.array != 'Down'] = np.ma.masked
        gear_downs = np.ma.clump_unmasked(gear.array)
        self.create_kpv_from_slices(
            mach.array, slices_and(airs.get_slices(), gear_downs),
            max_value)


class MachWhileGearRetractingMax(KeyPointValueNode):
    '''
    Maximum Mach speed reached while landing gear was retracting.
    '''

    units = ut.MACH

    def derive(self,
               mach=P('Mach'),
               gear_ret=S('Gear Retracting')):

        self.create_kpvs_within_slices(mach.array, gear_ret, max_value)


class MachWhileGearExtendingMax(KeyPointValueNode):
    '''
    Maximum Mach speed reached while landing gear was extending.
    '''

    units = ut.MACH

    def derive(self,
               mach=P('Mach'),
               gear_ext=S('Gear Extending')):

        self.create_kpvs_within_slices(mach.array, gear_ext, max_value)


##############################################################################
# Magnetic Variation


class MagneticVariationAtTakeoffTurnOntoRunway(KeyPointValueNode):
    '''
    Magnetic variation at turn onto runway.
    '''

    units = ut.DEGREE

    def derive(self,
               mag_var=P('Magnetic Variation'),
               takeoff_turn_on_rwy=KTI('Takeoff Turn Onto Runway')):

        self.create_kpvs_at_ktis(mag_var.array, takeoff_turn_on_rwy)


class MagneticVariationAtLandingTurnOffRunway(KeyPointValueNode):
    '''
    Magnetic variation at runway turnoff after landing.
    '''

    units = ut.DEGREE

    def derive(self,
               mag_var=P('Magnetic Variation'),
               landing_turn_off_rwy=KTI('Landing Turn Off Runway')):

        self.create_kpvs_at_ktis(mag_var.array, landing_turn_off_rwy)


##############################################################################
# Engine Transients

class EngGasTempOverThresholdDuration(KeyPointValueNode):
    '''
    Measures the duration Gas Temp is over the Takeoff/MCP power rating
    '''

    NAME_FORMAT = 'Eng Gas Temp Over %(period)s Duration'
    NAME_VALUES = {'period': ['Takeoff Power', 'MCP', 'Go Around Power']}
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series'), eng_type=A('Engine Type'), mods=A('Modifications')):
        try:
            at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        except KeyError:
            cls.warning("No engine thresholds available for '%s', '%s', '%s'.",
                        eng_series.value, eng_type.value, mods.value)
            return False
        except AttributeError:
            return False

        return any_of((
            'Eng (1) Gas Temp',
            'Eng (2) Gas Temp',
            'Eng (3) Gas Temp',
            'Eng (4) Gas Temp'
        ), available) and any_of((
            'Takeoff 5 Min Rating',
            'Maximum Continuous Power',
            'Go Around 5 Min Rating',
        ), available)

    def derive(self,
               eng1=M('Eng (1) Gas Temp'),
               eng2=M('Eng (2) Gas Temp'),
               eng3=M('Eng (3) Gas Temp'),
               eng4=M('Eng (4) Gas Temp'),
               takeoff=S('Takeoff 5 Min Rating'),
               mcp=S('Maximum Continuous Power'),
               go_around=S('Go Around 5 Min Rating'),
               eng_series=A('Engine Series'),
               eng_type=A('Engine Type'),
               mods=A('Modifications')):

        eng_thresholds = at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        # Lookup takeoff/mcp values
        mcp_value = eng_thresholds.get('Gas Temp', {}).get('mcp')
        takeoff_value = eng_thresholds.get('Gas Temp', {}).get('takeoff')

        phase_thresholds = [
            (self.NAME_VALUES['period'][0], takeoff, takeoff_value),
            (self.NAME_VALUES['period'][1], mcp, mcp_value),
            (self.NAME_VALUES['period'][2], go_around, takeoff_value)
        ]

        engines = [e for e in (eng1, eng2, eng3, eng4) if e]

        # iterate over phases
        for name, phase, threshold in phase_thresholds:
            if threshold is None or phase is None:
                # No threshold for this parameter in this phase.
                continue
            threshold_slices = []
            # iterate over engines
            for eng in engines:
                # create slices where eng above thresold
                threshold_slices += slices_above(eng.array, threshold)[1]
            # Only interested in exceedances within period.
            phase_slices = slices_and(phase.get_slices(), threshold_slices)
            # Remove overlapping slices keeping longer
            self.create_kpvs_from_slice_durations(
                slices_remove_overlaps(phase_slices),
                self.frequency, period=name)


class EngN1OverThresholdDuration(KeyPointValueNode):
    '''
    Measures the duration N1 is over the Takeoff/MCP power rating
    '''

    NAME_FORMAT = 'Eng N1 Over %(period)s Duration'
    NAME_VALUES = {'period': ['Takeoff Power', 'MCP', 'Go Around Power']}
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series'), eng_type=A('Engine Type'), mods=A('Modifications')):
        try:
            at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        except KeyError:
            cls.warning("No engine thresholds available for '%s', '%s', '%s'.",
                        eng_series.value, eng_type.value, mods.value)
            return False

        return any_of((
            'Eng (1) N1',
            'Eng (2) N1',
            'Eng (3) N1',
            'Eng (4) N1'
        ), available) and any_of((
            'Takeoff 5 Min Rating',
            'Maximum Continuous Power',
            'Go Around 5 Min Rating',
        ), available)

    def derive(self,
               eng1=M('Eng (1) N1'),
               eng2=M('Eng (2) N1'),
               eng3=M('Eng (3) N1'),
               eng4=M('Eng (4) N1'),
               takeoff=S('Takeoff 5 Min Rating'),
               mcp=S('Maximum Continuous Power'),
               go_around=S('Go Around 5 Min Rating'),
               eng_series=A('Engine Series'),
               eng_type=A('Engine Type'),
               mods=A('Modifications')):

        eng_thresholds = at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        # Lookup takeoff/mcp values
        mcp_value = eng_thresholds.get('N1', {}).get('mcp')
        takeoff_value = eng_thresholds.get('N1', {}).get('takeoff')

        phase_thresholds = [
            (self.NAME_VALUES['period'][0], takeoff, takeoff_value),
            (self.NAME_VALUES['period'][1], mcp, mcp_value),
            (self.NAME_VALUES['period'][2], go_around, takeoff_value)
        ]

        engines = [e for e in (eng1, eng2, eng3, eng4) if e]

        # iterate over phases
        for name, phase, threshold in phase_thresholds:
            if threshold is None or phase is None:
                # No threshold for this parameter in this phase.
                continue
            threshold_slices = []
            # iterate over engines
            for eng in engines:
                # create slices where eng above thresold
                threshold_slices += slices_above(eng.array, threshold)[1]
            # Only interested in exceedances within period.
            phase_slices = slices_and(phase.get_slices(), threshold_slices)
            # Remove overlapping slices keeping longer
            self.create_kpvs_from_slice_durations(
                slices_remove_overlaps(phase_slices),
                self.frequency, period=name)


class EngN2OverThresholdDuration(KeyPointValueNode):
    '''
    Measures the duration N2 is over the Takeoff/MCP power rating
    '''

    NAME_FORMAT = 'Eng N2 Over %(period)s Duration'
    NAME_VALUES = {'period': ['Takeoff Power', 'MCP', 'Go Around Power']}
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series'), eng_type=A('Engine Type'), mods=A('Modifications')):
        try:
            at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        except KeyError:
            cls.warning("No engine thresholds available for '%s', '%s', '%s'.",
                        eng_series.value, eng_type.value, mods.value)
            return False

        return any_of((
            'Eng (1) N2',
            'Eng (2) N2',
            'Eng (3) N2',
            'Eng (4) N2'
        ), available) and any_of((
            'Takeoff 5 Min Rating',
            'Maximum Continuous Power',
            'Go Around 5 Min Rating',
        ), available)

    def derive(self,
               eng1=M('Eng (1) N2'),
               eng2=M('Eng (2) N2'),
               eng3=M('Eng (3) N2'),
               eng4=M('Eng (4) N2'),
               takeoff=S('Takeoff 5 Min Rating'),
               mcp=S('Maximum Continuous Power'),
               go_around=S('Go Around 5 Min Rating'),
               eng_series=A('Engine Series'),
               eng_type=A('Engine Type'),
               mods=A('Modifications')):

        eng_thresholds = at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        # Lookup takeoff/mcp values
        mcp_value = eng_thresholds.get('N2', {}).get('mcp')
        takeoff_value = eng_thresholds.get('N2', {}).get('takeoff')

        phase_thresholds = [
            (self.NAME_VALUES['period'][0], takeoff, takeoff_value),
            (self.NAME_VALUES['period'][1], mcp, mcp_value),
            (self.NAME_VALUES['period'][2], go_around, takeoff_value)
        ]

        engines = [e for e in (eng1, eng2, eng3, eng4) if e]

        # iterate over phases
        for name, phase, threshold in phase_thresholds:
            if threshold is None or phase is None:
                # No threshold for this parameter in this phase.
                continue
            threshold_slices = []
            # iterate over engines
            for eng in engines:
                # create slices where eng above thresold
                threshold_slices += slices_above(eng.array, threshold)[1]
            # Only interested in exceedances within period.
            phase_slices = slices_and(phase.get_slices(), threshold_slices)
            # Remove overlapping slices keeping longer
            self.create_kpvs_from_slice_durations(
                slices_remove_overlaps(phase_slices),
                self.frequency, period=name)


class EngNpOverThresholdDuration(KeyPointValueNode):
    '''
    Measures the duration propeller speed is over the Takeoff/MCP power rating
    '''

    NAME_FORMAT = 'Eng Np Over %(period)s Duration'
    NAME_VALUES = {'period': ['Takeoff Power', 'MCP', 'Go Around Power']}
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series'), eng_type=A('Engine Type'), mods=A('Modifications')):
        try:
            at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        except KeyError:
            cls.warning("No engine thresholds available for '%s', '%s', '%s'.",
                        eng_series.value, eng_type.value, mods.value)
            return False

        return any_of((
            'Eng (1) Np',
            'Eng (2) Np',
            'Eng (3) Np',
            'Eng (4) Np'
        ), available) and any_of((
            'Takeoff 5 Min Rating',
            'Maximum Continuous Power',
            'Go Around 5 Min Rating',
        ), available)

    def derive(self,
               eng1=M('Eng (1) Np'),
               eng2=M('Eng (2) Np'),
               eng3=M('Eng (3) Np'),
               eng4=M('Eng (4) Np'),
               takeoff=S('Takeoff 5 Min Rating'),
               mcp=S('Maximum Continuous Power'),
               go_around=S('Go Around 5 Min Rating'),
               eng_series=A('Engine Series'),
               eng_type=A('Engine Type'),
               mods=A('Modifications')):

        eng_thresholds = at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        # Lookup takeoff/mcp values
        mcp_value = eng_thresholds.get('Np', {}).get('mcp')
        takeoff_value = eng_thresholds.get('Np', {}).get('takeoff')

        phase_thresholds = [
            (self.NAME_VALUES['period'][0], takeoff, takeoff_value),
            (self.NAME_VALUES['period'][1], mcp, mcp_value),
            (self.NAME_VALUES['period'][2], go_around, takeoff_value)
        ]

        engines = [e for e in (eng1, eng2, eng3, eng4) if e]

        # iterate over phases
        for name, phase, threshold in phase_thresholds:
            if threshold is None or phase is None:
                # No threshold for this parameter in this phase.
                continue
            threshold_slices = []
            # iterate over engines
            for eng in engines:
                # create slices where eng above thresold
                threshold_slices += slices_above(eng.array, threshold)[1]
            # Only interested in exceedances within period.
            phase_slices = slices_and(phase.get_slices(), threshold_slices)
            # Remove overlapping slices keeping longer
            self.create_kpvs_from_slice_durations(
                slices_remove_overlaps(phase_slices),
                self.frequency, period=name)


class EngTorqueOverThresholdDuration(KeyPointValueNode):
    '''
    Measures the duration Torque is over the Takeoff/MCP power rating
    '''

    NAME_FORMAT = 'Eng Torque Over %(period)s Duration'
    NAME_VALUES = {'period': ['Takeoff Power', 'MCP', 'Go Around Power']}
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series'), eng_type=A('Engine Type'),
                    mods=A('Modifications'), ac_type=A('Aircraft Type')):
        try:
            at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        except KeyError:
            cls.warning("No engine thresholds available for '%s', '%s', '%s'.",
                        eng_series.value, eng_type.value, mods.value)
            return False
        except AttributeError:
            return False

        base = any_of((
            'Eng (1) Torque',
            'Eng (2) Torque',
            'Eng (3) Torque',
            'Eng (4) Torque'
        ), available) and any_of((
            'Takeoff 5 Min Rating',
            'Maximum Continuous Power',
            'Go Around 5 Min Rating',
        ), available)

        # Additional requirement for Helicopters only
        heli_additions = False if ac_type == helicopter and 'All Engines Operative' not in available else True

        return base and heli_additions

    def derive(self,
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque'),
               takeoff=S('Takeoff 5 Min Rating'),
               mcp=S('Maximum Continuous Power'),
               go_around=S('Go Around 5 Min Rating'),
               eng_series=A('Engine Series'),
               eng_type=A('Engine Type'),
               mods=A('Modifications'),
               all_eng=M('All Engines Operative')):

        eng_thresholds = at.get_engine_map(eng_series.value, eng_type.value, mods.value)
        # Lookup takeoff/mcp values
        mcp_value = eng_thresholds.get('Torque', {}).get('mcp')
        takeoff_value = eng_thresholds.get('Torque', {}).get('takeoff')

        phase_thresholds = [
            (self.NAME_VALUES['period'][0], takeoff, takeoff_value),
            (self.NAME_VALUES['period'][1], mcp, mcp_value),
            (self.NAME_VALUES['period'][2], go_around, takeoff_value)
        ]

        engines = [e for e in (eng1, eng2, eng3, eng4) if e]

        # iterate over phases
        for name, phase, threshold in phase_thresholds:
            if threshold is None or phase is None:
                # No threshold for this parameter in this phase.
                continue
            threshold_slices = []
            # iterate over engines
            for eng in engines:
                # create slices where eng above thresold
                threshold_slices += slices_above(eng.array, threshold)[1]
            # Only interested in exceedances within period.
            # Brief exceedances and gaps are removed to reduce KPV count.
            phase_slices = slices_and(phase.get_slices(), threshold_slices)
            phase_slices = slices_remove_small_slices(phase_slices, 2, eng1.hz)
            phase_slices = slices_remove_overlaps(phase_slices)
            phase_slices = slices_remove_small_gaps(phase_slices, 3, eng1.hz)
            # Helicopters only
            if all_eng:
                phase_slices = slices_and(runs_of_ones(all_eng.array == 'AEO'), phase_slices)
            # Remove overlapping slices keeping longer
            self.create_kpvs_from_slice_durations(
                phase_slices, self.frequency, period=name)


class EngTorqueLimitExceedanceWithOneEngineInoperativeDuration(KeyPointValueNode):
    '''
    Measures the duration of a Torque exceedance taken from engine lookup tables.
    
    Note: only true exceedances (value and duration) are measured.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series'), eng_type=A('Engine Type'),
                    mods=A('Modifications'), ac_type=A('Aircraft Type')):
        if ac_type != helicopter:
            return False
        try:
            at.get_engine_map(eng_series.value, eng_type.value, mods.value, restriction='single')
        except KeyError:
            cls.warning("No engine thresholds available for '%s', '%s', '%s'.",
                        eng_series.value, eng_type.value, mods.value)
            return False

        return any_of((
            'Eng (1) Torque',
            'Eng (2) Torque',
            'Eng (3) Torque',
            'Eng (4) Torque'
        ), available) and 'One Engine Inoperative' in available

    def derive(self,
               eng1=M('Eng (1) Torque'),
               eng2=M('Eng (2) Torque'),
               eng3=M('Eng (3) Torque'),
               eng4=M('Eng (4) Torque'),
               one_eng=M('One Engine Inoperative'),
               eng_series=A('Engine Series'),
               eng_type=A('Engine Type'),
               mods=A('Modifications')):

        # Lookup Thresholds
        eng_thresholds = at.get_engine_map(eng_series.value, eng_type.value, mods.value, restriction='single')
        torq_thresholds = eng_thresholds.get('Torque', {})

        oei_phases = runs_of_ones(one_eng.array == 'OEI')
        engines = [e for e in (eng1, eng2, eng3, eng4) if e]

        order = ('continuous', 'low', 'hi', 'transient')

        prev_threshold = torq_thresholds.get('continuous')
        for name in order:
            if not torq_thresholds.get(name):
                # No threshold for name
                continue

            duration, threshold = torq_thresholds[name]
            if not duration or threshold == prev_threshold:
                # No duration or threshold is the same as previous
                prev_threshold = threshold
                continue

            threshold_slices = []
            # iterate over engines
            for eng in engines:
                threshold_slices += slices_between(eng.array, prev_threshold, threshold)[1]

            prev_threshold = threshold

            if not threshold_slices:
                # No threshold exceedances
                continue

            phase_slices = slices_and(threshold_slices, oei_phases)
            # Brief exceedances and overlaps are removed to reduce KPV count.
            phase_slices = slices_remove_overlaps(phase_slices)
            phase_slices = slices_remove_small_slices(phase_slices, duration, eng.hz)
            if phase_slices:
                self.info("Creating Torque OEI kpvs for '%s'", name)
                self.create_kpvs_from_slice_durations(phase_slices, self.frequency)


class EngTorqueExceeding100Percent(KeyPointValueNode):
    '''
    Measures the duration of Eng (*) Torque Avg exceeding 100.
    '''

    units = ut.SECOND
    can_operate = helicopter_only

    def derive(self, avg_torque = P('Eng (*) Torque Avg')):
        threshold_slices = slices_above(avg_torque.array, 100)[1]
        threshold_slices = slices_remove_small_slices(threshold_slices, 2, avg_torque.hz)
        self.create_kpvs_from_slice_durations(threshold_slices, self.frequency)


class EngTorqueExceeding110Percent(KeyPointValueNode):
    '''
    Measures the duration of Eng (*) Torque Avg exceeding 110.
    '''

    units = ut.SECOND
    can_operate = helicopter_only

    def derive(self, avg_torque = P('Eng (*) Torque Avg')):
        threshold_slices = slices_above(avg_torque.array, 110)[1]
        threshold_slices = slices_remove_small_slices(threshold_slices, 2, avg_torque.hz)
        self.create_kpvs_from_slice_durations(threshold_slices, self.frequency)


##############################################################################
# Engine Bleed


class EngBleedValvesAtLiftoff(KeyPointValueNode):
    '''
    Eng Bleed Open is "Open" for any engine bleed open, and
    only shows "Closed" if all engine bleed valves are closed.
    '''

    units = None

    @classmethod
    def can_operate(cls, available):
        return all_of((
            'Eng Bleed Open',
            'Liftoff',
        ), available)

    def derive(self,
               liftoffs=KTI('Liftoff'),
               bleed=M('Eng Bleed Open')):

        self.create_kpvs_at_ktis(bleed.array == 'Open', liftoffs)


##############################################################################
# Engine EPR


class EngEPRDuringApproachMax(KeyPointValueNode):
    '''
    Maximum recorded engine EPR during approach.
    '''

    name = 'Eng EPR During Approach Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               approaches=S('Approach')):

        self.create_kpv_from_slices(eng_epr_max.array, approaches, max_value)


class EngEPRDuringApproachMin(KeyPointValueNode):
    '''
    Minimum recorded engine EPR during approach.
    '''

    name = 'Eng EPR During Approach Min'
    units = None

    def derive(self,
               eng_epr_min=P('Eng (*) EPR Min'),
               approaches=S('Approach')):

        self.create_kpv_from_slices(eng_epr_min.array, approaches, min_value)


class EngEPRDuringTaxiMax(KeyPointValueNode):
    '''
    Maximum recorded engine EPR during taxi.
    '''

    name = 'Eng EPR During Taxi Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_epr_max.array, taxiing, max_value)


class EngEPRDuringTaxiOutMax(KeyPointValueNode):
    '''
    Minimum recorded engine EPR during taxi out.
    '''

    name = 'Eng EPR During Taxi Out Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               taxiing=S('Taxi Out')):

        self.create_kpv_from_slices(eng_epr_max.array, taxiing, max_value)


class EngEPRDuringTaxiInMax(KeyPointValueNode):
    '''
    Maximum recorded engine EPR during taxi in.
    '''

    name = 'Eng EPR During Taxi In Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               taxiing=S('Taxi In')):

        self.create_kpv_from_slices(eng_epr_max.array, taxiing, max_value)


class EngEPRDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum recorded engine EPR value for 5 minutes 
    after the start of takeoff acceleration. 
    '''

    name = 'Eng EPR During Takeoff 5 Min Rating Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_epr_max.array, ratings, max_value)


class EngEPRFor5SecDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum engine EPR recorded for at least 5 seconds in the period 
    between the start of takeoff acceleration and 5 minutes after.
    '''

    name = 'Eng EPR For 5 Sec During Takeoff 5 Min Rating Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               ratings=S('Takeoff 5 Min Rating')):

        array = eng_epr_max.array
        if eng_epr_max.frequency >= 1.0:
            array = second_window(eng_epr_max.array, eng_epr_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngTPRDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum recorded engine TPR value for 5 minutes 
    after the start of takeoff acceleration. 
    '''

    name = 'Eng TPR During Takeoff 5 Min Rating Max'
    units = None

    def derive(self,
               eng_tpr_limit=P('Eng TPR Limit Difference'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_tpr_limit.array, ratings, max_value)


class EngTPRFor5SecDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum engine TPR recorded for at least 5 seconds in the period 
    between the start of takeoff acceleration and 5 minutes after.
    '''

    name = 'Eng TPR For 5 Sec During Takeoff 5 Min Rating Max'
    units = None

    def derive(self,
               eng_tpr_limit=P('Eng TPR Limit Difference'),
               ratings=S('Takeoff 5 Min Rating')):

        array = eng_tpr_limit.array
        if eng_tpr_limit.frequency >= 1.0:
            array = second_window(eng_tpr_limit.array, eng_tpr_limit.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngEPRDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum recorded engine EPR value for 5 minutes after a go around.
    '''

    name = 'Eng EPR During Go Around 5 Min Rating Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_epr_max.array, ratings, max_value)


class EngEPRFor5SecDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum engine EPR recorded for at least 5 seconds in the period 
    between the start of the go around and 5 minutes after.
    '''

    name = 'Eng EPR For 5 Sec During Go Around 5 Min Rating Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               ratings=S('Go Around 5 Min Rating')):

        array = eng_epr_max.array
        if eng_epr_max.frequency >= 1.0:
            array = second_window(eng_epr_max.array, eng_epr_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngTPRDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum recorded engine TPR value for 5 minutes after a go around.
    '''

    name = 'Eng TPR During Go Around 5 Min Rating Max'
    units = None

    def derive(self,
               eng_tpr_limit=P('Eng TPR Limit Difference'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_tpr_limit.array, ratings, max_value)


class EngTPRFor5SecDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum engine TPR recorded for at least 5 seconds in the period 
    between the start of the go around and 5 minutes after.
    '''

    name = 'Eng TPR For 5 Sec During Go Around 5 Min Rating Max'
    units = None

    def derive(self,
               eng_tpr_limit=P('Eng TPR Limit Difference'),
               ratings=S('Go Around 5 Min Rating')):

        array = eng_tpr_limit.array
        if eng_tpr_limit.frequency >= 1.0:
            array = second_window(eng_tpr_limit.array, eng_tpr_limit.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngEPRDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum EPR value during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force and the aircraft is airborne.
    '''

    name = 'Eng EPR During Maximum Continuous Power Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               mcp=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(eng_epr_max.array, mcp, max_value)


class EngEPRFor5SecDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum engine EPR recorded for at least 5 seconds in the period 
    of maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force and the aircraft is airborne.
    '''

    name = 'Eng EPR For 5 Sec During Maximum Continuous Power Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               ratings=S('Maximum Continuous Power')):

        array = eng_epr_max.array
        if eng_epr_max.frequency >= 1.0:
            array = second_window(eng_epr_max.array, eng_epr_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngTPRDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum EPR value during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force and the aircraft is airborne.
    '''

    name = 'Eng TPR During Maximum Continuous Power Max'
    units = None

    def derive(self,
               eng_tpr_max=P('Eng (*) TPR Max'),
               mcp=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(eng_tpr_max.array, mcp, max_value)


class EngTPRFor5SecDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum TPR recorded for at least 5 seconds during maximum
    continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force and the aircraft is airborne.
    '''

    name = 'Eng TPR For 5 Sec During Maximum Continuous Power Max'
    units = None

    def derive(self,
               eng_tpr_max=P('Eng (*) TPR Max'),
               ratings=S('Maximum Continuous Power')):

        array = eng_tpr_max.array
        if eng_tpr_max.frequency >= 1.0:
            array = second_window(eng_tpr_max.array, eng_tpr_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngEPR500To50FtMax(KeyPointValueNode):
    '''
    Maximum EPR recorded between 500 and 50ft AAL.
    '''

    name = 'Eng EPR 500 To 50 Ft Max'
    units = None

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_epr_max.array,
            alt_aal.slices_from_to(500, 50),
            max_value,
        )


class EngEPR500To50FtMin(KeyPointValueNode):
    '''
    Minimum EPR recorded between 500ft and 50ft AAL
    '''

    name = 'Eng EPR 500 To 50 Ft Min'
    units = None

    def derive(self,
               eng_epr_min=P('Eng (*) EPR Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_epr_min.array,
            alt_aal.slices_from_to(500, 50),
            min_value,
        )


class EngEPRFor5Sec500To50FtMin(KeyPointValueNode):
    '''
    Minimum EPR recorded for at least 5 seconds between
    500ft and 50ft AAL.
    '''

    name = 'Eng EPR For 5 Sec 500 To 50 Ft Min'
    units = None

    def derive(self,
               eng_epr_min=P('Eng (*) EPR Min For 5 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            eng_epr_min.array,
            trim_slices(alt_aal.slices_from_to(500, 50), 5, self.frequency,
                        hdf_duration),
            min_value,
        )


class EngEPRFor5Sec1000To500FtMin(KeyPointValueNode):
    '''
    Minimum EPR recorded for at least 5 seconds between
    1000ft and 500ft AAL.
    '''

    name = 'Eng EPR For 5 Sec 1000 To 500 Ft Min'
    units = None

    def derive(self,
               eng_epr_min=P('Eng (*) EPR Min For 5 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            eng_epr_min.array,
            trim_slices(alt_aal.slices_from_to(1000, 500), 5, self.frequency,
                        hdf_duration),
            min_value,
        )


class EngEPRAtTOGADuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum EPR recorded at takeoff with TOGA engaged.
    '''

    name = 'Eng EPR At TOGA During Takeoff Max'
    units = None

    def derive(self,
               toga=M('Takeoff And Go Around'),
               eng_epr_max=P('Eng (*) EPR Max'),
               takeoff=S('Takeoff')):

        indexes = find_edges_on_state_change('TOGA', toga.array,
                                             change='entering', phase=takeoff)
        for index in indexes:
            # Measure at known state instead of interpolated transition
            index = ceil(index)
            value = value_at_index(eng_epr_max.array, index)
            self.create_kpv(index, value)


class EngTPRAtTOGADuringTakeoffMin(KeyPointValueNode):
    '''
    Minimum EPR recorded at takeoff with TOGA engaged.
    '''

    name = 'Eng TPR At TOGA During Takeoff Min'
    units = None

    def derive(self,
               toga=M('Takeoff And Go Around'),
               eng_tpr_max=P('Eng (*) TPR Min'),
               takeoff=S('Takeoff')):
        '''
        Originally coded for 787, but the event has been disabled since it lacks a
        limit.

        Align to Takeoff And Go Around for most accurate state change indices.
        '''
        indexes = find_edges_on_state_change('TOGA', toga.array,
                                             change='entering', phase=takeoff)
        for index in indexes:
            # Measure at known state instead of interpolated transition
            index = ceil(index)
            value = value_at_index(eng_tpr_max.array, index)
            self.create_kpv(index, value)


class EngEPRExceedEPRRedlineDuration(KeyPointValueNode):
    '''
    Originally coded for B777, returns the duration EPR is above EPR Redline
    for each engine
    '''

    name = 'Eng EPR Exceeded EPR Redline Duration'
    units = ut.SECOND

    def derive(self,
               eng_1_epr=P('Eng (1) EPR'),
               eng_1_epr_red=P('Eng (1) EPR Redline'),
               eng_2_epr=P('Eng (2) EPR'),
               eng_2_epr_red=P('Eng (2) EPR Redline'),
               eng_3_epr=P('Eng (3) EPR'),
               eng_3_epr_red=P('Eng (3) EPR Redline'),
               eng_4_epr=P('Eng (4) EPR'),
               eng_4_epr_red=P('Eng (4) EPR Redline')):

        eng_eprs = (eng_1_epr, eng_2_epr, eng_3_epr, eng_4_epr)
        eng_epr_reds = (eng_1_epr_red, eng_2_epr_red, eng_3_epr_red, eng_4_epr_red)

        eng_groups = zip(eng_eprs, eng_epr_reds)
        for eng_epr, eng_epr_red in eng_groups:
            if eng_epr and eng_epr_red:
                epr_diff = eng_epr.array - eng_epr_red.array
                self.create_kpvs_where(epr_diff > 0, self.hz)


##############################################################################
# Engine Fire


class EngFireWarningDuration(KeyPointValueNode):
    '''
    Duration that the any of the Engine Fire Warnings are active.
    '''

    units = ut.SECOND

    def derive(self, eng_fire=M('Eng (*) Fire'), airborne=S('Airborne')):
        self.create_kpvs_where(eng_fire.array == 'Fire',
                               eng_fire.hz, phase=airborne)


##############################################################################
# APU On


class APUOnDuringFlightDuration(KeyPointValueNode):
    '''
    Duration of APU On during flight.
    '''
    name = 'APU On During Flight Duration'
    units = ut.SECOND

    def derive(self,
               apu=P('APU On'),
               airborne=S('Airborne')):
        '''
        If APU On started before takeoff, we want to record the last index of the
        state duration.

        If APU On started in air, we want to record the first index of the state
        duration (default behaviour of create_kpvs_where()).
        '''
        self.create_kpvs_where(apu.array == 'On', apu.hz, phase=airborne)
        for kpv in list(self):
            for in_air in airborne:
                last_index = kpv.index + kpv.value * apu.hz
                if kpv.index == in_air.slice.start:
                    # if APU On was On during liftoff, we want to use the index
                    # of the last sample
                    kpv.index = last_index


##############################################################################
# APU Fire


class APUFireWarningDuration(KeyPointValueNode):
    '''
    Duration that the any of the APU Fire Warnings are active.
    '''

    name = 'APU Fire Warning Duration'
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return ('APU Fire',) in [available] or \
               ('Fire APU Single Bottle System',
                'Fire APU Dual Bottle System') in [available]

    def derive(self, fire=M('APU Fire'),
               single_bottle=M('Fire APU Single Bottle System'),
               dual_bottle=M('Fire APU Dual Bottle System')):

        if fire:
            self.create_kpvs_where(fire.array == 'Fire', fire.hz)
        else:
            hz = (single_bottle or dual_bottle).hz
            apu_fires = vstack_params_where_state((single_bottle, 'Fire'),
                                                  (dual_bottle, 'Fire'))

            self.create_kpvs_where(apu_fires.any(axis=0), hz)


##############################################################################
# Engine Gas Temperature


class EngGasTempDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum EGT between the start of takeoff acceleration and 5 mintues after.
    '''

    units = ut.CELSIUS

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_egt_max.array, ratings, max_value)


class EngGasTempFor5SecDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum EGT recorded for at least 5 seconds between the start of takeoff
    acceleration and 5 minutes after.
    '''

    units = ut.CELSIUS

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Takeoff 5 Min Rating')):

        array = eng_egt_max.array
        if eng_egt_max.frequency >= 1.0:
            array = second_window(eng_egt_max.array, eng_egt_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngGasTempDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum EGT between the start of the go around and 5 minutes after.
    '''

    units = ut.CELSIUS

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_egt_max.array, ratings, max_value)


class EngGasTempFor5SecDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum EGT recorded for at least 5 seconds between the start of
    go around and 5 minutes after.
    '''

    units = ut.CELSIUS

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Go Around 5 Min Rating')):

        array = eng_egt_max.array
        if eng_egt_max.frequency >= 1.0:
            array = second_window(eng_egt_max.array, eng_egt_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngGasTempDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum EGT recorded during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force. 
    '''

    units = ut.CELSIUS

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               mcp=S('Maximum Continuous Power')):
        '''
        So, by collecting all the high power
        periods and inverting these from the start of the first airborne section to
        the end of the last, we have the required periods of flight.
        '''
        self.create_kpvs_within_slices(eng_egt_max.array, mcp, max_value)


class EngGasTempFor5SecDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum EGT recorded for at least 5 seconds during maximum 
    continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force. 
    '''

    units = ut.CELSIUS

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Maximum Continuous Power')):

        array = eng_egt_max.array
        if eng_egt_max.frequency >= 1.0:
            array = second_window(eng_egt_max.array, eng_egt_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngGasTempDuringMaximumContinuousPowerForXMinMax(KeyPointValueNode):
    '''
    Maximum EGT during maximum continuous power for specified amount of time.
    
    We assume maximum continuous power applies whenever takeoff or go-around
    power settings are not in force. 
    '''

    NAME_FORMAT = 'Eng Gas Temp During Maximum Continuous Power For %(minutes)d Min Max'
    NAME_VALUES = {'minutes': [3, 5]}
    align_frequency = 1
    units = ut.CELSIUS

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               airborne=S('Airborne')):

        if not airborne:
            return
        high_power_ratings = to_ratings.get_slices() + ga_ratings.get_slices()
        max_cont_rating = slices_not(
            high_power_ratings,
            begin_at=min(air.slice.start for air in airborne),
            end_at=max(air.slice.stop for air in airborne),
        )
        for minutes in self.NAME_VALUES['minutes']:
            seconds = minutes * 60
            self.create_kpvs_within_slices(
                second_window(eng_egt_max.array.astype(int), eng_egt_max.hz, seconds),
                max_cont_rating,
                max_value,
                minutes=minutes,
            )


class EngGasTempDuringEngStartMax(KeyPointValueNode):
    '''
    One key point value for maximum engine gas temperature at engine start.

    Note that for three spool engines, the N3 value is used to detect
    running, while for two spool engines N2 is the highest spool speed so is
    used.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        egt = any_of(['Eng (1) Gas Temp',
                      'Eng (2) Gas Temp',
                      'Eng (3) Gas Temp',
                      'Eng (4) Gas Temp'], available)
        n3 = any_of(('Eng (1) N3',
                     'Eng (2) N3',
                     'Eng (3) N3',
                     'Eng (4) N3'), available)
        n2 = any_of(('Eng (1) N2',
                     'Eng (2) N2',
                     'Eng (3) N2',
                     'Eng (4) N2'), available)
        n1 = any_of(('Eng (1) N1',
                     'Eng (2) N1',
                     'Eng (3) N1',
                     'Eng (4) N1'), available)
        if ac_type == helicopter:
            return egt and n1
        else:
            return egt and (n3 or n2)

    units = ut.CELSIUS

    def derive(self,
               eng_1_egt=P('Eng (1) Gas Temp'),
               eng_2_egt=P('Eng (2) Gas Temp'),
               eng_3_egt=P('Eng (3) Gas Temp'),
               eng_4_egt=P('Eng (4) Gas Temp'),
               eng_1_n3=P('Eng (1) N3'),
               eng_2_n3=P('Eng (2) N3'),
               eng_3_n3=P('Eng (3) N3'),
               eng_4_n3=P('Eng (4) N3'),
               eng_1_n2=P('Eng (1) N2'),
               eng_2_n2=P('Eng (2) N2'),
               eng_3_n2=P('Eng (3) N2'),
               eng_4_n2=P('Eng (4) N2'),
               eng_1_n1=P('Eng (1) N1'),
               eng_2_n1=P('Eng (2) N1'),
               eng_3_n1=P('Eng (3) N1'),
               eng_4_n1=P('Eng (4) N1'),
               eng_starts=KTI('Eng Start'),
               ac_type=A('Aircraft Type')):

        eng_egts = (eng_1_egt, eng_2_egt, eng_3_egt, eng_4_egt)
        eng_powers = (eng_1_n3 or eng_1_n2,
                      eng_2_n3 or eng_2_n2,
                      eng_3_n3 or eng_3_n2,
                      eng_4_n3 or eng_4_n2)

        if ac_type == helicopter:
            eng_powers = (eng_1_n1, eng_2_n1, eng_3_n1, eng_4_n1)

        eng_groups = enumerate(zip(eng_egts, eng_powers), start=1)

        search_duration = 5 * 60 * self.frequency

        for eng_number, (eng_egt, eng_power) in eng_groups:
            if not eng_egt or not eng_power or eng_egt.frequency < 0.25:
                # Where the egt is in a superframe, let's give up now:
                continue

            eng_start_name = eng_starts.format_name(number=eng_number)
            eng_number_starts = eng_starts.get(name=eng_start_name)

            for eng_start in eng_number_starts:
                # Search for 10 minutes for level off.
                start = eng_start.index
                stop = start + search_duration
                eng_start_slice = slice(start, stop)

                level_off = level_off_index(eng_power.array, self.frequency, 10, 1,
                                            _slice=eng_start_slice)

                if level_off is not None:
                    eng_start_slice = slice(start, level_off)

                self.create_kpv(*max_value(eng_egt.array,
                                           _slice=eng_start_slice))


class EngGasTempDuringEngStartForXSecMax(KeyPointValueNode):
    '''
    One key point value for maximum engine gas temperature at engine start for
    all engines. The value is taken from the engine with the largest value.
    '''

    NAME_FORMAT = 'Eng Gas Temp During Eng Start For %(seconds)d Sec Max'
    NAME_VALUES = {'seconds': [5, 10, 40]}
    align_frequency = 1
    units = ut.CELSIUS

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               eng_n2_min=P('Eng (*) N2 Min'),
               toff_turn_rwy=KTI('Takeoff Turn Onto Runway')):

        # We never see engine start if data started after aircraft is airborne:
        if not toff_turn_rwy:
            return

        # Where the egt is in a superframe, let's give up now:
        if eng_egt_max.frequency < 0.25:
            return

        # Extract the index for the first turn onto the runway:
        fto_idx = toff_turn_rwy.get_first().index

        # Mask out sections with N2 > 60%, i.e. all engines running:
        n2_data = eng_n2_min.array[0:fto_idx]
        n2_data[n2_data > 60.0] = np.ma.masked

        # Engines are already running at start of data:
        if np.ma.count(n2_data) == 0:
            return

        chunks = np.ma.clump_unmasked(n2_data)

        for seconds in self.NAME_VALUES['seconds']:
            # Remove chunks of data that are too small to clip:
            slices = slices_remove_small_slices(chunks, seconds, eng_egt_max.hz)
            if not slices:
                continue
            # second_window is more accurate than clip and much faster
            # shh... we'll add one so that second_window will work!
            dur = seconds + 1 if seconds % 2 else seconds
            array = second_window(eng_egt_max.array.astype(int), eng_egt_max.hz, dur)
            self.create_kpvs_within_slices(array, slices, max_value, seconds=seconds)


class EngGasTempDuringFlightMin(KeyPointValueNode):
    '''
    Minimum gas temperature recorded during flight. The event will then be computed
    later, testing against a suitable minimum value for a running engine.

    Note that the gas temperature can increase on an engine run down.
    '''

    units = ut.CELSIUS

    def derive(self,
               eng_egt_min=P('Eng (*) Gas Temp Min'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(
            eng_egt_min.array,
            airborne,
            min_value,
        )


class EngGasTempExceededEngGasTempRedlineDuration(KeyPointValueNode):
    '''
    Originally coded for B777, returns the duration Gas Temp is above Gas
    Temp Redline for each engine
    '''

    name = 'Eng Gas Temp Exceeded Eng Gas Temp Redline Duration'
    units = ut.SECOND

    def derive(self,
               eng_1_egt=P('Eng (1) Gas Temp'),
               eng_1_egt_red=P('Eng (1) Gas Temp Redline'),
               eng_2_egt=P('Eng (2) Gas Temp'),
               eng_2_egt_red=P('Eng (2) Gas Temp Redline'),
               eng_3_egt=P('Eng (3) Gas Temp'),
               eng_3_egt_red=P('Eng (3) Gas Temp Redline'),
               eng_4_egt=P('Eng (4) Gas Temp'),
               eng_4_egt_red=P('Eng (4) Gas Temp Redline')):

        eng_egts = (eng_1_egt, eng_2_egt, eng_3_egt, eng_4_egt)
        eng_egt_reds = (eng_1_egt_red, eng_2_egt_red, eng_3_egt_red, eng_4_egt_red)

        eng_groups = zip(eng_egts, eng_egt_reds)
        for eng_egt, eng_egt_red in eng_groups:
            if eng_egt and eng_egt_red:
                egt_diff = eng_egt.array - eng_egt_red.array
                self.create_kpvs_where(egt_diff > 0, self.hz)


class EngGasTempAboveNormalMaxLimitDuringTakeoff5MinRatingDuration(KeyPointValueNode):
    '''
    Total duration Engine Gas Temperature is above maintenance limit During
    Takeoff. Limit depends on type of engine.
    '''

    NAME_FORMAT = 'Eng (%(number)d) Gas Temp Above Normal Max Limit During Takeoff 5 Min Rating Duration'
    NAME_VALUES = NAME_VALUES_ENGINE
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series')):
        gas_temps = any_of(('Eng (%d) Gas Temp' % n for n in range(1, 5)), available)
        engine_series = eng_series and eng_series.value == 'CFM56-3'

        return gas_temps and engine_series and 'Takeoff 5 Min Rating' in available

    def derive(self,
               eng1=P('Eng (1) Gas Temp'),
               eng2=P('Eng (2) Gas Temp'),
               eng3=P('Eng (3) Gas Temp'),
               eng4=P('Eng (4) Gas Temp'),
               takeoffs=S('Takeoff 5 Min Rating'),
               eng_series=A('Engine Series')):
        '''
        TODO: extend for engines other than CFM56-3
        '''
        limit = 930
        for eng_num, eng in enumerate((eng1, eng2, eng3, eng4), start=1):
            if eng is None:
                continue  # Engine is not available on this aircraft.
            egt_limit_exceeded = runs_of_ones(eng.array > limit)
            egt_takeoff = slices_and(egt_limit_exceeded, takeoffs.get_slices())
            if egt_takeoff:
                index = egt_takeoff[0].start
                value = slices_duration(egt_takeoff, self.hz)
                self.create_kpv(index, value, number=eng_num, limit=limit)


class EngGasTempAboveNormalMaxLimitDuringMaximumContinuousPowerDuration(KeyPointValueNode):
    '''
    Total duration Engine Gas Temperature is above maintenance limit During
    Maximum Continuous Power. Limit depends on type of engine.
    '''

    NAME_FORMAT = 'Eng (%(number)d) Gas Temp Above Normal Max Limit During Maximum Continuous Power Duration'
    NAME_VALUES = NAME_VALUES_ENGINE
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series')):
        gas_temps = any_of(('Eng (%d) Gas Temp' % n for n in range(1, 5)), available)
        engine_series = eng_series and eng_series.value == 'CFM56-3'

        return gas_temps and engine_series and ('Maximum Continous Power' in available)

    def derive(self,
               eng1=P('Eng (1) Gas Temp'),
               eng2=P('Eng (2) Gas Temp'),
               eng3=P('Eng (3) Gas Temp'),
               eng4=P('Eng (4) Gas Temp'),
               mcp=S('Maximum Continous Power')):

        slices = mcp.get_slices()

        limit = 895
        for eng_num, eng in enumerate((eng1, eng2, eng3, eng4), start=1):
            if eng is None:
                continue  # Engine is not available on this aircraft.
            egt_limit_exceeded = runs_of_ones(eng.array > limit)
            egt_mcp = slices_and(egt_limit_exceeded, slices)
            if egt_mcp:
                index = min(egt_mcp, key=operator.attrgetter('start')).start
                value = slices_duration(egt_mcp, self.hz)
                self.create_kpv(index, value, number=eng_num, limit=limit)


##############################################################################
# Engine N1


class EngN1DuringTaxiMax(KeyPointValueNode):
    '''
    Maximum N1 of all Engines while taxiing.
    '''

    name = 'Eng N1 During Taxi Max'
    units = ut.PERCENT
    
    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               taxi_out=S('Taxi Out'),
               taxi_in=S('Taxi In'),
               taxiing=S('Taxiing')):
        '''
        ignores last 20 seconds of taxi out before takeoff in order to prevent  
        false event triggers in case of immediate takeoff
        '''
        
        if (len(taxi_out.get_slices()) > 0) and taxi_out[0].slice.stop - (20 * taxi_out.hz) - taxi_out[0].slice.start > 0:
            #check if taxi out is longer than 20 seconds
            taxi_slices = [slice(taxi_out[0].slice.start, taxi_out[0].slice.stop - 
                                     (20 * taxi_out.hz))] + taxi_in.get_slices()
            self.create_kpv_from_slices(eng_n1_max.array, taxi_slices, max_value)
            
        elif len(taxiing.get_slices()) > 0:
            self.create_kpv_from_slices(eng_n1_max.array, taxiing, max_value)

class EngN1DuringTaxiOutMax(KeyPointValueNode):
    '''
    Maximum N1 of all Engines during taxi out.
    '''

    name = 'Eng N1 During Taxi Out Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               taxiing=S('Taxi Out')):
        
        if (len(taxiing.get_slices()) > 0) and (taxiing[0].slice.start, taxiing[0].slice.stop - (20 * taxiing.hz) > 0):
            taxi_slices = [slice(taxiing[0].slice.start, taxiing[0].slice.stop - 
                                 (20 * taxiing.hz))]
            self.create_kpv_from_slices(eng_n1_max.array, taxi_slices, max_value)
        

class EngN1DuringTaxiInMax(KeyPointValueNode):
    '''
    Maximum N1 of all Engines during taxi in.
    '''

    name = 'Eng N1 During Taxi In Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               taxiing=S('Taxi In')):

        self.create_kpv_from_slices(eng_n1_max.array, taxiing, max_value)

class EngN1DuringApproachMax(KeyPointValueNode):
    '''
    Maximum N1 of all engines during approach phase.
    '''

    name = 'Eng N1 During Approach Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               approaches=S('Approach')):

        self.create_kpv_from_slices(eng_n1_max.array, approaches, max_value)


class EngN1DuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N1 between the start of takeoff acceleration and 5 minutes after.
    '''

    name = 'Eng N1 During Takeoff 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n1_max.array, ratings, max_value)


class EngN1For5SecDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N1 recorded for at least 5 seconds between the start
    of takeoff acceleration and 5 minutes after.
    '''

    name = 'Eng N1 For 5 Sec During Takeoff 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(
            second_window(eng_n1_max.array, eng_n1_max.frequency, 5),
            ratings, max_value)


class EngN1DuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N1 recorded between the start of go around and 5 minutes after.
    '''

    name = 'Eng N1 During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n1_max.array, ratings, max_value)


class EngN1For5SecDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N1 recorded for at least 5 seconds between the start
    of go around and 5 minutes after.
    '''

    name = 'Eng N1 For 5 Sec During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(
            second_window(eng_n1_max.array, eng_n1_max.frequency, 5),
            ratings, max_value)


class EngN1DuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum N1 recorded during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    name = 'Eng N1 During Maximum Continuous Power Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               mcp=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(eng_n1_max.array, mcp, max_value)


class EngN1For5SecDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum N1 recorded for at least 5 seconds during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    name = 'Eng N1 For 5 Sec During Maximum Continuous Power Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(
            second_window(eng_n1_max.array, eng_n1_max.frequency, 5),
            ratings, max_value)


class EngAPRDuration(KeyPointValueNode):
    '''
    Duration for which APR was active.
    '''
    
    name = 'Eng APR Duration'
    units = ut.SECOND
    
    @classmethod
    def can_operate(cls, available):
        return 'Mobile' in available and \
               all_of(('FADEC (L) APR Active',
                       'FADEC (R) APR Active',), available) or \
               all_of(('Eng (1) ATTCS Armed',
                       'Eng (1) ATTCS Enabled',
                       'Eng (1) ATTCS Triggered',
                       'Eng (2) ATTCS Armed',
                       'Eng (2) ATTCS Enabled',
                       'Eng (2) ATTCS Triggered',), available)
        
    def derive(self,
               mobiles=S('Mobile'),
               eng1_apr=P('FADEC (L) APR Active'),
               eng2_apr=P('FADEC (R) APR Active'),
               eng1_attcs_arm=P('Eng (1) ATTCS Armed'),
               eng1_attcs_enabled=P('Eng (1) ATTCS Enabled'),
               eng1_attcs_trigger=P('Eng (1) ATTCS Triggered'),
               eng2_attcs_arm=P('Eng (2) ATTCS Armed'),
               eng2_attcs_enabled=P('Eng (2) ATTCS Enabled'),
               eng2_attcs_trigger=P('Eng (2) ATTCS Triggered'),):

        for mobile in mobiles:
            if eng1_apr and eng2_apr:
                apr_active = slices_or(runs_of_ones(eng1_apr.array[mobile.slice]),
                                       runs_of_ones(eng2_apr.array[mobile.slice]))
    
                apr_active = slices_remove_small_gaps(
                    apr_active, time_limit=600, hz=eng1_apr.hz)
    
                # APR activation is a rare occurence, if we catch more than 5
                # it's most likely an error.
                self.create_kpvs_from_slice_durations(
                    shift_slices(apr_active[:5], mobile.slice.start), eng1_apr.hz)
    
            else:
                # APR is active if ATTCS Armed, Triggered and Enabled are true.
                eng1_apr_active = runs_of_ones(
                    np.logical_and(np.logical_and(eng1_attcs_arm.array[mobile.slice],
                                                  eng1_attcs_enabled.array[mobile.slice]),
                                   eng1_attcs_trigger.array[mobile.slice]))
    
                eng2_apr_active = runs_of_ones(
                    np.logical_and(np.logical_and(eng2_attcs_arm.array[mobile.slice],
                                                  eng2_attcs_enabled.array[mobile.slice]),
                                   eng2_attcs_trigger.array[mobile.slice]))
    
                apr_active = slices_or(eng1_apr_active, eng2_apr_active)
    
                apr_active = slices_remove_small_gaps(
                    apr_active, time_limit=600, hz=eng1_attcs_arm.hz)
    
                # APR activation is a rare occurence, if we catch more than 5
                # it's most likely an error.
                self.create_kpvs_from_slice_durations(
                    shift_slices(apr_active[:5], mobile.slice.start), eng1_attcs_arm.hz)


class EngN1CyclesDuringFinalApproach(KeyPointValueNode):
    '''
    Numer of N1 cycles during final approach phase.
    '''

    name = 'Eng N1 Cycles During Final Approach'
    units = ut.CYCLES
    
    def derive(self,
               eng_n1_avg=P('Eng (*) N1 Avg'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                eng_n1_avg.array[fin_app.slice],
                5.0, 10.0, eng_n1_avg.hz,
                fin_app.slice.start,
            ))


class EngN1500To50FtMax(KeyPointValueNode):
    '''
    Maximum N1 between 500ft and 50ft AAL.
    '''

    name = 'Eng N1 500 To 50 Ft Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_n1_max.array,
            alt_aal.slices_from_to(500, 50),
            max_value,
        )


class EngN1500To50FtMin(KeyPointValueNode):
    '''
    Minimum N1 between 500ft and 50ft AAL.
    '''

    name = 'Eng N1 500 To 50 Ft Min'
    units = ut.PERCENT

    def derive(self,
               eng_n1_min=P('Eng (*) N1 Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_n1_min.array,
            alt_aal.slices_from_to(500, 50),
            min_value,
        )


class EngN1For5Sec500To50FtMin(KeyPointValueNode):
    '''
    Minimum N1 recorded for at least 5 seconds between
    500ft and 50 ft AAL.
    '''

    name = 'Eng N1 For 5 Sec 500 To 50 Ft Min'
    units = ut.PERCENT
    
    @classmethod
    def can_operate(self, available):
        return all_of(('Eng (*) N1 Min', 'Altitude AAL For Flight Phases', 'HDF Duration'), available)

    def derive(self,
               eng_n1_min_param=P('Eng (*) N1 Min'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        alt_slices = trim_slices(alt_aal.slices_from_to(500, 50), 5, self.frequency, hdf_duration)
        
        if eng_n1_min_param:
            for alt_slice in alt_slices:
                array = eng_n1_min_param.array[alt_slice]
                samples = 5 * eng_n1_min_param.frequency
                if len(array) - samples < 0:
                    continue
                
                # For each 5 seconds window, get the sum of differences between 
                # each value and the maximum value of the array
                sliding_window = np.lib.stride_tricks.as_strided(
                    array.max() - array, shape=(len(array) - samples, samples),
                    strides=array.strides * 2)
                max_difference = np.sum(sliding_window, axis=-1).tolist()
                
                # The index of the maximum value of the max_difference array will also be
                # the index of the sliding window containing the most minimum values for this array
                start_slice_index = index_at_value(np.array(max_difference), max(max_difference))
                index, value = max_value(max(array) - sliding_window[start_slice_index])
                
                if index is not None:
                    self.create_kpv(start_slice_index + index + alt_slice.start, value)


class EngN1For5Sec1000To500FtMin(KeyPointValueNode):
    '''
    Minimum N1 recorded for at least 5 seconds between 
    1000ft and 500ft AAL.
    '''

    name = 'Eng N1 For 5 Sec 1000 To 500 Ft Min'
    units = ut.PERCENT

    def derive(self,
               eng_n1_min=P('Eng (*) N1 Min For 5 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               duration=A('HDF Duration')):

        hdf_duration = duration.value * self.frequency if duration else None
        self.create_kpvs_within_slices(
            eng_n1_min.array,
            trim_slices(alt_aal.slices_from_to(1000, 500), 5, self.frequency,
                        hdf_duration),
            min_value,
        )


class EngN1WithThrustReversersInTransitMax(KeyPointValueNode):
    '''
    Maximum N1 recorded with thrust reversers in transit.
    '''

    name = 'Eng N1 With Thrust Reversers In Transit Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_avg=P('Eng (*) N1 Avg'),
               tr=M('Thrust Reversers'),
               landings=S('Landing')):

        slices = [s.slice for s in landings]
        slices = clump_multistate(tr.array, 'In Transit', slices)
        self.create_kpv_from_slices(eng_n1_avg.array, slices, max_value)


class EngN1WithThrustReversersDeployedMax(KeyPointValueNode):
    '''
    Maximum N1 recorded with thrust reversers deployed.
    '''

    name = 'Eng N1 With Thrust Reversers Deployed Max'
    units = ut.PERCENT

    def derive(self,
               eng_n1_avg=P('Eng (*) N1 Avg'),
               tr=M('Thrust Reversers'),
               landings=S('Landing')):

        slices = [s.slice for s in landings]
        slices = clump_multistate(tr.array, 'Deployed', slices)
        self.create_kpv_from_slices(eng_n1_avg.array, slices, max_value)


# NOTE: Was named 'Eng N1 Cooldown Duration'.
# TODO: Similar KPV for duration between engine under 60 percent and engine shutdown
class EngN1Below60PercentAfterTouchdownDuration(KeyPointValueNode):
    '''
    Max duration N1 below 60% after Touchdown for engine cooldown. Using 60%
    allows for cooldown after use of Reverse Thrust.

    Evaluated for each engine to account for single engine taxi-in.

    Note: Assumes that all Engines are recorded at the same frequency.
    '''

    NAME_FORMAT = 'Eng (%(number)d) N1 Below 60 Percent After Touchdown Duration'
    NAME_VALUES = NAME_VALUES_ENGINE
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return all((
            any_of(('Eng (%d) N1' % n for n in range(1, 5)), available),
            'Eng Stop' in available,
            'Touchdown' in available,
        ))

    def derive(self,
               engines_stop=KTI('Eng Stop'),
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1'),
               tdwn=KTI('Touchdown')):

        if not tdwn:
            return
        for eng_num, eng in enumerate((eng1, eng2, eng3, eng4), start=1):
            if eng is None:
                continue  # Engine is not available on this aircraft.
            eng_stop = engines_stop.get(name='Eng (%d) Stop' % eng_num)
            if not eng_stop:
                # XXX: Should we measure until the end of the flight anyway?
                # (Probably not.)
                self.debug('Engine %d did not stop on this flight, cannot '
                           'measure KPV', eng_num)
                continue
            last_tdwn_idx = tdwn.get_last().index
            last_eng_stop_idx = eng_stop[-1].index
            if last_tdwn_idx > last_eng_stop_idx:
                self.debug('Engine %d was stopped before last touchdown', eng_num)
                continue
            eng_array = repair_mask(eng.array)
            eng_below_60 = np.ma.masked_greater(eng_array, 60)
            # Measure duration between final touchdown and engine stop:
            touchdown_to_stop_slice = max_continuous_unmasked(
                eng_below_60, slice(last_tdwn_idx, last_eng_stop_idx))
            if touchdown_to_stop_slice:
                # TODO: Future storage of slice: self.slice = touchdown_to_stop_slice
                touchdown_to_stop_duration = (
                    touchdown_to_stop_slice.stop - touchdown_to_stop_slice.start) / self.hz
                self.create_kpv(touchdown_to_stop_slice.start,
                                touchdown_to_stop_duration, number=eng_num)
            else:
                # Create KPV of 0 seconds:
                self.create_kpv(last_eng_stop_idx, 0.0, number=eng_num)


class EngN1AtTOGADuringTakeoff(KeyPointValueNode):
    '''
    Engine N1 with TOGA active during takeoff and go around.
    '''

    name = 'Eng N1 At TOGA During Takeoff'
    units = ut.PERCENT

    def derive(self,
               toga=M('Takeoff And Go Around'),
               eng_n1=P('Eng (*) N1 Min'),
               takeoff=S('Takeoff')):
        '''
        Align to Takeoff And Go Around for most accurate state change indices.
        '''
        indexes = find_edges_on_state_change('TOGA', toga.array, change='entering', phase=takeoff)
        for index in indexes:
            # Measure at known state instead of interpolated transition
            index = ceil(index)
            value = value_at_index(eng_n1.array, index)
            self.create_kpv(index, value)


class EngN154to72PercentWithThrustReversersDeployedDurationMax(KeyPointValueNode):
    '''
    Duration for which the engine N1 has been between 54% and 72% with thrust
    reversers deployed.
    
    KPV created at customer request following Service Bullitin from Rolls Royce
    (TAY-72-A1771).
    From EASA PROPOSAL TO ISSUE AN AIRWORTHINESS DIRECTIVE, PAD No.: 13-100:
    Applicability: Tay 620-15 and Tay 620-15/20 engines, all serial numbers.
    These engines are known to be installed on, but not limited to, Fokker F28
    Mark 0070 and Mark 0100 series aeroplanes.
    '''

    NAME_FORMAT = 'Eng (%(number)d) N1 54 To 72 Percent With Thrust Reversers Deployed Duration Max'
    NAME_VALUES = NAME_VALUES_ENGINE
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, eng_series=A('Engine Series')):
        engine_series = eng_series and eng_series.value == 'Tay 620'
        return all((
            any_of(('Eng (%d) N1' % n for n in cls.NAME_VALUES['number']), available),
            'Thrust Reversers' in available,
            engine_series,
        ))

    def derive(self, eng1_n1=P('Eng (1) N1'), eng2_n1=P('Eng (2) N1'),
               eng3_n1=P('Eng (3) N1'), eng4_n1=P('Eng (4) N1'),
               tr=M('Thrust Reversers'), eng_series=A('Engine Series')):

        eng_n1_list = (eng1_n1, eng2_n1, eng3_n1, eng4_n1)
        reverser_deployed = np.ma.where(tr.array == 'Deployed', tr.array, np.ma.masked)
        for eng_num, eng_n1 in enumerate(eng_n1_list, 1):
            if not eng_n1:
                continue
            n1_range = np.ma.masked_outside(eng_n1.array, 54, 72)
            n1_range.mask = n1_range.mask | reverser_deployed.mask
            max_slice = max_continuous_unmasked(n1_range)
            if max_slice:
                self.create_kpvs_from_slice_durations((max_slice,),
                                                      eng_n1.frequency,
                                                      number=eng_num)


class EngNp82To90PercentDurationMax(KeyPointValueNode):
    '''
    Duration for which the engine Np has been between 82% and 90%.
    
    Specifically for the Jetstrean 41, where propellor blade fatigue can
    be exacurbated in this speed band.
    '''

    NAME_FORMAT = 'Eng (%(number)d) Np 82 To 90 Percent Duration Max'
    NAME_VALUES = NAME_VALUES_ENGINE

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available, ac_series=A('Series'),
                       ):
        ac_concerned = ac_series and ac_series.value == 'Jetstream 41'
        return all_of(('Eng (1) Np', 'Eng (2) Np'), available) and ac_concerned

    def derive(self, eng1_np=P('Eng (1) Np'), eng2_np=P('Eng (2) Np')):

        eng_np_list = (eng1_np, eng2_np)
        for eng_num, eng_np in enumerate(eng_np_list, 1):
            if not eng_np:
                continue
            np_range = np.ma.masked_outside(eng_np.array, 82, 90)
            max_slice = max_continuous_unmasked(np_range)
            if max_slice:
                self.create_kpvs_from_slice_durations((max_slice,),
                                                      eng_np.frequency,
                                                      number=eng_num)


class EngN1ExceededN1RedlineDuration(KeyPointValueNode):
    '''
    Originally coded for B777, returns the duration N1 is above N1 Redline
    for each engine
    '''

    name = 'Eng N1 Exceeded N1 Redline Duration'
    units = ut.SECOND

    def derive(self,
               eng_1_n1=P('Eng (1) N1'),
               eng_1_n1_red=P('Eng (1) N1 Redline'),
               eng_2_n1=P('Eng (2) N1'),
               eng_2_n1_red=P('Eng (2) N1 Redline'),
               eng_3_n1=P('Eng (3) N1'),
               eng_3_n1_red=P('Eng (3) N1 Redline'),
               eng_4_n1=P('Eng (4) N1'),
               eng_4_n1_red=P('Eng (4) N1 Redline')):

        eng_n1s = (eng_1_n1, eng_2_n1, eng_3_n1, eng_4_n1)
        eng_n1_reds = (eng_1_n1_red, eng_2_n1_red, eng_3_n1_red, eng_4_n1_red)

        eng_groups = zip(eng_n1s, eng_n1_reds)
        for eng_n1, eng_n1_redline in eng_groups:
            if eng_n1 and eng_n1_redline:
                n1_diff = eng_n1.array - eng_n1_redline.array
                self.create_kpvs_where(n1_diff > 0, self.hz)


##############################################################################
# Engine N2


class EngN2DuringTaxiMax(KeyPointValueNode):
    '''
    Maximum recorded N2 during taxi.
    '''

    name = 'Eng N2 During Taxi Max'
    units = ut.PERCENT

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_n2_max.array, taxiing, max_value)


class EngN2DuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N2 recorded between the start of takeoff acceleration
    and 5 minutes after.
    '''

    name = 'Eng N2 During Takeoff 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n2_max.array, ratings, max_value)


class EngN2For5SecDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N2 recorded for at least 5 seconds between the start 
    of takeoff acceleration and 5 minutes after. 
    '''

    name = 'Eng N2 For 5 Sec During Takeoff 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(
            second_window(eng_n2_max.array, eng_n2_max.frequency, 5),
            ratings, max_value)


class EngN2DuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N2 recorded between the start of go around and 5 minutes after.
    '''

    name = 'Eng N2 During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n2_max.array, ratings, max_value)


class EngN2For5SecDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N2 recorded for at least 5 seconds between the start 
    of a go around and 5 minutes after.
    '''

    name = 'Eng N2 For 5 Sec During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(
            second_window(eng_n2_max.array, eng_n2_max.frequency, 5),
            ratings, max_value)


class EngN2DuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum N2 recorded during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    name = 'Eng N2 During Maximum Continuous Power Max'
    units = ut.PERCENT

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               mcp=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(eng_n2_max.array, mcp, max_value)


class EngN2DuringMaximumContinuousPowerMin(KeyPointValueNode):
    '''
    Minimum N2 recorded during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    name = 'Eng N2 During Maximum Continuous Power Min'
    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self,
               eng_n2_min=P('Eng (*) N2 Min'),
               mcp=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(eng_n2_min.array, mcp, min_value)


class EngN2For5SecDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum N2 recorded for at least 5 seconds during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    name = 'Eng N2 For 5 Sec During Maximum Continuous Power Max'
    units = ut.PERCENT

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(
            second_window(eng_n2_max.array, eng_n2_max.frequency, 5),
            ratings, max_value)


class EngN2CyclesDuringFinalApproach(KeyPointValueNode):
    '''
    Numer of N2 cycles during final approach phase.
    '''

    name = 'Eng N2 Cycles During Final Approach'
    units = ut.CYCLES

    def derive(self,
               eng_n2_avg=P('Eng (*) N2 Avg'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                eng_n2_avg.array[fin_app.slice],
                10.0, 10.0, eng_n2_avg.hz,
                fin_app.slice.start,
            ))


class EngN2ExceededN2RedlineDuration(KeyPointValueNode):
    '''
    Originally coded for B777, returns the duration N2 is above N2 Redline
    for each engine
    '''

    name = 'Eng N2 Exceeded N2 Redline Duration'
    units = ut.SECOND

    def derive(self,
               eng_1_n2=P('Eng (1) N2'),
               eng_1_n2_red=P('Eng (1) N2 Redline'),
               eng_2_n2=P('Eng (2) N2'),
               eng_2_n2_red=P('Eng (2) N2 Redline'),
               eng_3_n2=P('Eng (3) N2'),
               eng_3_n2_red=P('Eng (3) N2 Redline'),
               eng_4_n2=P('Eng (4) N2'),
               eng_4_n2_red=P('Eng (4) N2 Redline')):

        eng_n2s = (eng_1_n2, eng_2_n2, eng_3_n2, eng_4_n2)
        eng_n2_reds = (eng_1_n2_red, eng_2_n2_red, eng_3_n2_red, eng_4_n2_red)

        eng_groups = zip(eng_n2s, eng_n2_reds)
        for eng_n2, eng_n2_redline in eng_groups:
            if eng_n2 and eng_n2_redline:
                n2_diff = eng_n2.array - eng_n2_redline.array
                self.create_kpvs_where(n2_diff > 0, self.hz)


##############################################################################
# Engine N3


class EngN3DuringTaxiMax(KeyPointValueNode):
    '''
    Maximum N3 recorded during taxi.
    '''

    name = 'Eng N3 During Taxi Max'
    units = ut.PERCENT

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_n3_max.array, taxiing, max_value)


class EngN3DuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N3 recorded between the start of takeoff acceleration
    and 5 minutes after.
    '''

    name = 'Eng N3 During Takeoff 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n3_max.array, ratings, max_value)


class EngN3For5SecDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N3 recorded for at least 5 seconds between the start 
    of takeoff acceleration and 5 minutes after.
    '''

    name = 'Eng N3 For 5 Sec During Takeoff 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(
            second_window(eng_n3_max.array, eng_n3_max.frequency, 5),
            ratings, max_value)


class EngN3DuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N3 recorded between the start of go around and 5 minutes after.
    '''

    name = 'Eng N3 During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n3_max.array, ratings, max_value)


class EngN3For5SecDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum N3 recorded for at least 5 seconds between the start 
    of a go around and 5 minutes after.
    '''

    name = 'Eng N3 For 5 Sec During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(
            second_window(eng_n3_max.array, eng_n3_max.frequency, 5),
            ratings, max_value)


class EngN3DuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum N3 recorded during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    name = 'Eng N3 During Maximum Continuous Power Max'
    units = ut.PERCENT

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               mcp=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(eng_n3_max.array, mcp, max_value)


class EngN3For5SecDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum N3 recorded for at least 5 seconds during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    name = 'Eng N3 For 5 Sec During Maximum Continuous Power Max'
    units = ut.PERCENT

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(
            second_window(eng_n3_max.array, eng_n3_max.frequency, 5),
            ratings, max_value)


class EngN3ExceededN3RedlineDuration(KeyPointValueNode):
    '''
    Originally coded for B777, returns the duration N3 is above N3 Redline
    for each engine
    '''

    name = 'Eng N3 Exceeded N3 Redline Duration'
    units = ut.SECOND

    def derive(self,
               eng_1_n3=P('Eng (1) N3'),
               eng_1_n3_red=P('Eng (1) N3 Redline'),
               eng_2_n3=P('Eng (2) N3'),
               eng_2_n3_red=P('Eng (2) N3 Redline'),
               eng_3_n3=P('Eng (3) N3'),
               eng_3_n3_red=P('Eng (3) N3 Redline'),
               eng_4_n3=P('Eng (4) N3'),
               eng_4_n3_red=P('Eng (4) N3 Redline')):

        eng_n3s = (eng_1_n3, eng_2_n3, eng_3_n3, eng_4_n3)
        eng_n3_reds = (eng_1_n3_red, eng_2_n3_red, eng_3_n3_red, eng_4_n3_red)

        eng_groups = zip(eng_n3s, eng_n3_reds)
        for eng_n3, eng_n3_redline in eng_groups:
            if eng_n3 and eng_n3_redline:
                n3_diff = eng_n3.array - eng_n3_redline.array
                self.create_kpvs_where(n3_diff > 0, self.hz)


##############################################################################
# Engine Np


class EngNpDuringClimbMin(KeyPointValueNode):
    '''
    Minimum Np recorded during climb. 
    '''

    name = 'Eng Np During Climb Min'
    units = ut.PERCENT

    def derive(self,
               eng_np_min=P('Eng (*) Np Min'),
               climbs=S('Climbing')):

        self.create_kpv_from_slices(eng_np_min.array, climbs, min_value)


class EngNpDuringTaxiMax(KeyPointValueNode):
    '''
    Maximum Np recorded during taxi.
    '''

    name = 'Eng Np During Taxi Max'
    units = ut.PERCENT

    def derive(self,
               eng_np_max=P('Eng (*) Np Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_np_max.array, taxiing, max_value)


class EngNpDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum Np recorded between the start of takeoff acceleration
    and 5 minutes after.
    '''

    name = 'Eng Np During Takeoff 5 Min Rating Max'
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Eng (*) Np Max', 'Takeoff 5 Min Rating'), available)

    def derive(self,
               eng_np_max=P('Eng (*) Np Max'),
               takeoffs=S('Takeoff 5 Min Rating'),
               go_arounds=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_np_max.array, takeoffs, max_value)
        self.create_kpvs_within_slices(eng_np_max.array, go_arounds, max_value)


class EngNpFor5SecDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum Np recorded for at least 5 seconds between the start 
    of takeoff acceleration and 5 minutes after.
    '''

    name = 'Eng Np For 5 Sec During Takeoff 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_np_max=P('Eng (*) Np Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(
            second_window(eng_np_max.array, eng_np_max.frequency, 5),
            ratings, max_value)


class EngNpDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum Np recorded between the start of a go around and 
    5 minutes after.
    '''

    name = 'Eng Np During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_np_max=P('Eng (*) Np Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_np_max.array, ratings, max_value)


class EngNpFor5SecDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum Np recorded for at least 5 seconds between the start 
    of a go around and 5 minutes after.
    '''

    name = 'Eng Np For 5 Sec During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_np_max=P('Eng (*) Np Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(
            second_window(eng_np_max.array, eng_np_max.frequency, 5),
            ratings, max_value)


class EngNpDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum Np recorded during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    name = 'Eng Np During Maximum Continuous Power Max'
    units = ut.PERCENT

    def derive(self,
               eng_np_max=P('Eng (*) Np Max'),
               mcp=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(eng_np_max.array, mcp, max_value)


class EngNpForXSecDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum Np recorded for at least 5 and 20 seconds during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    NAME_FORMAT = 'Eng Np For %(seconds)d Sec During Maximum Continuous Power Max'
    NAME_VALUES = {'seconds': [5, 20]}
    units = ut.PERCENT

    def derive(self,
               eng_np_max=P('Eng (*) Np Max'),
               ratings=S('Maximum Continuous Power')):

        for seconds in self.NAME_VALUES['seconds']:
            self.create_kpvs_within_slices(
                second_window(eng_np_max.array, eng_np_max.frequency, seconds),
                ratings, max_value, seconds=seconds)


##############################################################################
# Engine Throttles


class ThrottleReductionToTouchdownDuration(KeyPointValueNode):
    '''

    Records the duration from touchdown until Throttle leaver is reduced in
    seconds, negative seconds indicates throttle reduced before touchdown.

    The algorithm has been adapted to use the peak curvature technique, scanning 
    from 5 seconds before the start of the landing (passing 50ft) to the 
    minimum throttle setting prior to application of reverse thrust.
    '''

    can_operate = aeroplane_only

    units = ut.SECOND

    def derive(self,
               tla=P('Throttle Levers'),
               landings=S('Landing'),
               touchdowns=KTI('Touchdown'),
               eng_n1=P('Eng (*) N1 Avg'),
               frame=A('Frame')):
        '''
        The original algorithm used reduction through 18deg throttle angle, but
        in cases where little power is being applied it was found that the
        throttle lever may not reach this setting. Also, this implies an
        aircraft-dependent threshold which would be difficult to maintain, and
        requires consistent throttle lever sensor rigging which may not be
        reliable on some types.
        '''
        delta = 5 * tla.hz
        for landing in landings:
            for touchdown in touchdowns.get(within_slice=landing.slice):
                # Seek the throttle reduction before thrust reverse is applied:
                scope = slice(landing.slice.start - delta, landing.slice.stop)
                dn1 = rate_of_change_array(eng_n1.array[scope], eng_n1.hz)
                dtla = rate_of_change_array(tla.array[scope], tla.hz)
                peak_decel = np.ma.argmax(dn1 * dtla)
                reduced_scope = slice(scope.start, scope.start + peak_decel)
                # Now see where the power is reduced:
                reduce_idx = peak_curvature(tla.array, reduced_scope,
                                            curve_sense='Convex', gap=1, ttp=3)
                if reduce_idx:
                    self.create_kpv(reduce_idx, (reduce_idx - touchdown.index) / tla.hz)


################################################################################
# Engine Vib Broadband


class EngVibBroadbandMax(KeyPointValueNode):
    '''
    Maximum recorded Engine Vibration Broadband.
    '''

    units = None

    def derive(self, eng_vib_max=P('Eng (*) Vib Broadband Max')):
        self.create_kpv(*max_value(eng_vib_max.array))


##############################################################################
# Engine Oil Pressure


class EngOilPressMax(KeyPointValueNode):
    '''
    Maximum oil pressure during first engine start until the last engine has
    stopped.
    '''

    units = ut.PSI

    def derive(self, oil_press=P('Eng (*) Oil Press Max'),
               eng_start=KTI('First Eng Fuel Flow Start'),
               eng_stop=KTI('Last Eng Fuel Flow Stop')):
        
        # if engine start and stop KTIs are available, we use them
        if eng_start and eng_stop:
            self.create_kpvs_within_slices(
                oil_press.array,
                [slice(eng_start[0].index, eng_stop[-1].index),],
                max_value
            )
        # if they're not available we simply look up the max value and 
        # rely on the masking algorithm
        else:
            index, value = max_value(oil_press.array)
            self.create_kpv(index, value)


class EngOilPressFor60SecDuringCruiseMax(KeyPointValueNode):
    '''
    Maximum oil pressure during the cruise for a 60 second period of flight.

    High oil pressure in cruise is an indication of clogging orifices /
    restriction in the oil supply lines to the aft sump due to oil cokeing
    (carbon accumulation). Oil Supply Line clogging will elevate the oil
    pressure and the result is decreased oil flow to the aft sump. The effect
    is reduced cooling/lubrication of the bearings and hardwear.
    '''

    units = ut.PSI

    def derive(self, oil_press=P('Eng (*) Oil Press Max'),
               cruise=S('Cruise')):
        press = second_window(oil_press.array, oil_press.hz, 60,
                              extend_window=True)
        self.create_kpvs_within_slices(press, cruise, max_value)


class EngOilPressMin(KeyPointValueNode):
    '''
    Minimum recorded engine oil pressure while airborne.
    
    Only in flight to avoid zero pressure readings for stationary engines.
    '''

    units = ut.PSI

    def derive(self, oil_press=P('Eng (*) Oil Press Min'),
               airborne=S('Airborne')):
        '''
        Extended to ignore cases where all data is zero, or single sample values are zero.
        The problem is that some low sample rate, low pressure engines can have an erroneous
        zero value that is below the rate limit, so is not detected by current spike detection.
        '''
        for air in [a.slice for a in airborne]:
            min_p = np.ma.min(oil_press.array[air])
            if min_p:
                # The minimum is non-zero, so let's use that.
                self.create_kpvs_within_slices(oil_press.array, [air], min_value)
            else:
                non_zero_press = np.ma.masked_equal(oil_press.array[air], min_p)
                non_zero_count = np.ma.count(non_zero_press)
                air_count = slice_duration(air, 1.0)
                if air_count-non_zero_count == 1:
                    # Only a single corrupt sample, so repair this
                    repair_press = repair_mask(non_zero_press, repair_duration=1.0)
                    index = np.ma.argmin(repair_press)
                    value = repair_press[index]
                elif non_zero_count > 1:
                    # Some data was non-zero
                    index = np.ma.argmin(oil_press.array[air])
                    value = oil_press.array[air][index]
                else:
                    continue
                self.create_kpv(index + air.start, value)


class EngOilPressWarningDuration(KeyPointValueNode):
    '''
    For aircraft with engine oil pressure warning indications, this measures
    the duration of either engine warning.
    '''

    units = ut.SECOND

    def derive(self,
               oil_press_warn=P('Eng (*) Oil Press Warning'),
               airborne=S('Airborne')):

        self.create_kpvs_where(oil_press_warn.array == 'Warning',
                               frequency=oil_press_warn.frequency,
                               phase=airborne)


class EngOilPressLowRedlineExceededDuration(KeyPointValueNode):
    '''
    For aircraft with Engine Oil Press Low Redline Exceeded paramater, this
    keypoint value measures the duration of the warning for any engine.
    '''

    units = ut.SECOND

    def derive(self, press_low_1=M('Eng (1) Oil Press Low Redline Exceeded'),
               press_low_2=M('Eng (2) Oil Press Low Redline Exceeded')):

        press_low = vstack_params_where_state((press_low_1, 'Exceeded'),
                                              (press_low_2, 'Exceeded'))
        self.create_kpvs_where(press_low.any(axis=0) == True,
                               frequency=press_low_1.frequency)


##############################################################################
# Engine Oil Quantity


class EngOilQtyMax(KeyPointValueNode):
    '''
    Maximum engine oil quantity while airborne.
    '''

    units = ut.QUART

    def derive(self,
               oil_qty=P('Eng (*) Oil Qty Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(oil_qty.array, airborne, max_value)


class EngOilQtyMin(KeyPointValueNode):
    '''
    Minimum engine oil quantity while airborne.
    '''

    units = ut.QUART

    def derive(self,
               oil_qty=P('Eng (*) Oil Qty Min'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(oil_qty.array, airborne, min_value)


class EngOilQtyDuringTaxiInMax(KeyPointValueNode):
    '''
    Maximum engine oil quantity during taxi in.
    '''

    NAME_FORMAT = "Eng (%(number)s) Oil Qty During Taxi In Max"
    NAME_VALUES = NAME_VALUES_ENGINE
    units = ut.QUART

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Eng (1) Oil Qty',
            'Eng (2) Oil Qty',
            'Eng (3) Oil Qty',
            'Eng (4) Oil Qty'
        ), available) and 'Taxi In' in available

    def derive(self,
               oil_qty1=P('Eng (1) Oil Qty'),
               oil_qty2=P('Eng (2) Oil Qty'),
               oil_qty3=P('Eng (3) Oil Qty'),
               oil_qty4=P('Eng (4) Oil Qty'),
               taxi_in=S('Taxi In')):

        oil_qty_list = (oil_qty1, oil_qty2, oil_qty3, oil_qty4)
        for number, oil_qty in enumerate(oil_qty_list, start=1):
            if oil_qty:
                self.create_kpvs_within_slices(oil_qty.array, taxi_in,
                                               max_value, number=number)


class EngOilQtyDuringTaxiOutMax(KeyPointValueNode):
    '''
    Maximum engine oil quantity during taxi out.
    '''

    NAME_FORMAT = "Eng (%(number)s) Oil Qty During Taxi Out Max"
    NAME_VALUES = NAME_VALUES_ENGINE
    units = ut.QUART

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Eng (1) Oil Qty',
            'Eng (2) Oil Qty',
            'Eng (3) Oil Qty',
            'Eng (4) Oil Qty'
        ), available) and 'Taxi Out' in available

    def derive(self,
               oil_qty1=P('Eng (1) Oil Qty'),
               oil_qty2=P('Eng (2) Oil Qty'),
               oil_qty3=P('Eng (3) Oil Qty'),
               oil_qty4=P('Eng (4) Oil Qty'),
               taxi_out=S('Taxi Out')):

        oil_qty_list = (oil_qty1, oil_qty2, oil_qty3, oil_qty4)
        for number, oil_qty in enumerate(oil_qty_list, start=1):
            if oil_qty:
                self.create_kpvs_within_slices(oil_qty.array, taxi_out,
                                               max_value, number=number)


##############################################################################
# Engine Oil Temperature


class EngOilTempMax(KeyPointValueNode):
    '''
    Maximum recorded engine oil temperature while airborne.
    '''

    units = ut.CELSIUS

    def derive(self,
               oil_temp=P('Eng (*) Oil Temp Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(oil_temp.array, airborne, max_value)


class EngOilTempForXMinMax(KeyPointValueNode):
    '''
    Maximum oil temperature sustained for X minutes.
    '''
    NAME_FORMAT = 'Eng Oil Temp For %(minutes)d Min Max'
    NAME_VALUES = {'minutes': [15, 20, 45]}
    name = 'Eng Oil Temp For X Min Max'
    units = ut.CELSIUS
    align_frequency = 1

    def derive(self, oil_temp=P('Eng (*) Oil Temp Max')):

        # Some aircraft don't have oil temperature sensors fitted. This trap
        # may be superceded by masking the Eng (*) Oil Temp Max parameter in
        # future:
        if oil_temp.array.mask.all():
            return

        for minutes in self.NAME_VALUES['minutes']:
            oil_sustained = oil_temp.array.astype(int)
            oil_sustained = second_window(oil_sustained, self.hz, minutes * 60)
            if not oil_sustained.mask.all():
                self.create_kpv(*max_value(oil_sustained), minutes=minutes)


##############################################################################
# Engine Torque


class EngTorqueDuringTaxiMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded during taxi phase.
    '''

    units = ut.PERCENT

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_trq_max.array, taxiing, max_value)


class EngTorqueWithOneEngineInoperativeMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded with one engine inoperative.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               airborne=S('Airborne'),
               one_eng=M('One Engine Inoperative')):

        phases = slices_and(runs_of_ones(one_eng.array == 'OEI'), airborne.get_slices())
        self.create_kpvs_within_slices(eng_trq_max.array, phases, max_value)


class EngTorqueDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded between the start of takeoff
    acceleration and 5 minutes after.
    '''

    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        heli = ac_type == helicopter and all_of(('Eng (*) Torque Max', 'Takeoff 5 Min Rating', 'All Engines Operative'), available)
        aero = ac_type == aeroplane and all_of(('Eng (*) Torque Max', 'Takeoff 5 Min Rating'), available)

        return heli or aero

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Takeoff 5 Min Rating'),
               all_eng=M('All Engines Operative')):
        phases = ratings.get_slices()
        if all_eng:
            phases = slices_and(runs_of_ones(all_eng.array == 'AEO'), phases)
        self.create_kpvs_within_slices(eng_trq_max.array, phases, max_value)


class EngTorqueFor5SecDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded for at least 5 seconds between the 
    start of takeoff acceleration and 5 minutes after.
    '''

    units = ut.PERCENT

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Takeoff 5 Min Rating')):

        array = eng_trq_max.array
        if eng_trq_max.frequency >= 1.0:
            array = second_window(eng_trq_max.array, eng_trq_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngTorque65KtsTo35FtMin(KeyPointValueNode):
    '''
    KPV designed in accordance with ATR72 FCOM

    Looks for the minimum Eng Torque between the aircraft reaching 65 kts
    during takeoff and it it reaching an altitude of 35 ft (end of takeoff)
    '''

    units = ut.PERCENT

    def derive(self,
               eng_trq_min=P('Eng (*) Torque Min'),
               airspeed=P('Airspeed'),
               takeoffs=S('Takeoff')):

        for takeoff in takeoffs:
            start = index_at_value(airspeed.array, 65, _slice=takeoff.slice,
                                   endpoint='nearest')
            self.create_kpvs_within_slices(eng_trq_min.array,
                                           [slice(start, takeoff.slice.stop)],
                                           min_value)


class EngTorqueDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum engine torque between the start of a go around and 5 minutes after.
    '''

    name = 'Eng Torque During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_trq_max.array, ratings, max_value)


class EngTorqueFor5SecDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded for at least 5 seconds between the 
    start of a go around and 5 minutes after.
    '''

    name = 'Eng Torque For 5 Sec During Go Around 5 Min Rating Max'
    units = ut.PERCENT

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Go Around 5 Min Rating')):

        array = eng_trq_max.array
        if eng_trq_max.frequency >= 1.0:
            array = second_window(eng_trq_max.array, eng_trq_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngTorqueDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded during maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        heli = ac_type == helicopter and all_of(('Eng (*) Torque Max', 'Maximum Continuous Power', 'All Engines Operative'), available)
        aero = ac_type == aeroplane and all_of(('Eng (*) Torque Max', 'Maximum Continuous Power'), available)

        return heli or aero

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               mcp=S('Maximum Continuous Power'),
               all_eng=M('All Engines Operative')):
        phases = mcp.get_slices()
        if all_eng:
            phases = slices_and(runs_of_ones(all_eng.array == 'AEO'), phases)
        self.create_kpvs_within_slices(eng_trq_max.array, phases, max_value)


class EngTorqueFor5SecDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded for at least 5 seconds during 
    maximum continuous power.
    
    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force.
    '''

    units = ut.PERCENT

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Maximum Continuous Power')):

        array = eng_trq_max.array
        if eng_trq_max.frequency >= 1.0:
            array = second_window(eng_trq_max.array, eng_trq_max.frequency, 5, extend_window=True)
        self.create_kpvs_within_slices(array, ratings, max_value)


class EngTorqueAbove90KtsMax(KeyPointValueNode):
    '''
    Maximum engine torque where the indicate airspeed is above 90 kts. (helicopter only)

    Some Helicopters have Torque restictions above 90 kts to limit flight control loads
    at high speed thereby preserving dynamic component replacement times.

    '''

    name = 'Eng Torque Above 90 Kts Max'

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, eng=P('Eng (*) Torque Max'), air_spd=P('Airspeed')):
        slices_no_gaps = slices_remove_small_gaps(air_spd.slices_above(90), time_limit=10, hz=air_spd.hz)

        self.create_kpvs_within_slices(
            eng.array,
            slices_no_gaps,
            max_value
            )


class EngTorqueAbove100KtsMax(KeyPointValueNode):
    '''
    Maximum engine torque where the indicate airspeed is above 100 kts. (helicopter only)

    Some Helicopters have Torque restictions above 100 kts to limit flight control loads
    at high speed thereby preserving dynamic component replacement times.
    (taken from S92 type certificate)
    '''

    name = 'Eng Torque Above 100 Kts Max'

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, eng=P('Eng (*) Torque Max'), air_spd=P('Airspeed')):
        self.create_kpvs_within_slices(
            eng.array,
            air_spd.slices_above(100),
            max_value
        )



class EngTorque500To50FtMax(KeyPointValueNode):
    '''
    Maximum engine torque between 500ft and 50ft AAL.
    '''

    units = ut.PERCENT

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_trq_max.array,
            alt_aal.slices_from_to(500, 50),
            max_value,
        )


class EngTorque500To50FtMin(KeyPointValueNode):
    '''
    Minimum engine torque between 500ft and 50ft AAL.
    '''

    units = ut.PERCENT

    def derive(self,
               eng_trq_min=P('Eng (*) Torque Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_trq_min.array,
            alt_aal.slices_from_to(500, 50),
            min_value,
        )


class EngTorqueWhileDescendingMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded during descent phase.
    '''

    units = ut.PERCENT

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               descending=S('Descending')):

        self.create_kpv_from_slices(eng_trq_max.array, descending, max_value)


class EngTorque7FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum engine torque between 7ft AAL and touchdown.
    '''

    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, eng_type=A('Engine Propulsion')):
        turbo_prop = eng_type and eng_type.value == 'PROP'
        required_params = all_of(['Eng (*) Torque Max',
                                  'Altitude AAL For Flight Phases',
                                  'Touchdown'], available)
        return turbo_prop and required_params

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            eng_trq_max.array,
            alt_aal.slices_to_kti(7, touchdowns),
            max_value,
        )


##############################################################################
# Torque

class TorqueAsymmetryWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum torque asymmetry while airborne.
    '''

    units = ut.PERCENT

    def derive(self, torq_asym=P('Torque Asymmetry'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(torq_asym.array, airborne.get_slices(), max_value)


##############################################################################
# Engine Vibrations (N*)


class EngVibN1Max(KeyPointValueNode):
    '''
    Maximum engine vibration on N1 while airborne.
    '''

    name = 'Eng Vib N1 Max'
    units = None

    def derive(self,
               eng_vib_n1=P('Eng (*) Vib N1 Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_n1.array, airborne, max_value)


class EngVibN2Max(KeyPointValueNode):
    '''
    Maximum engine vibration on N2 while airborne.
    '''

    name = 'Eng Vib N2 Max'
    units = None

    def derive(self,
               eng_vib_n2=P('Eng (*) Vib N2 Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_n2.array, airborne, max_value)


class EngVibN3Max(KeyPointValueNode):
    '''
    Maximum engine vibration on N3 while airborne.
    '''

    name = 'Eng Vib N3 Max'
    units = None

    def derive(self,
               eng_vib_n3=P('Eng (*) Vib N3 Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_n3.array, airborne, max_value)


# Engine Vibrations (Filters)


class EngVibAMax(KeyPointValueNode):
    '''
    Maximum engine vibration on A while airborne.
    '''

    name = 'Eng Vib A Max'
    units = None

    def derive(self,
               eng_vib_a=P('Eng (*) Vib A Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_a.array, airborne, max_value)


class EngVibBMax(KeyPointValueNode):
    '''
    Maximum engine vibration on B while airborne.
    '''

    name = 'Eng Vib B Max'
    units = None

    def derive(self,
               eng_vib_b=P('Eng (*) Vib B Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_b.array, airborne, max_value)


class EngVibCMax(KeyPointValueNode):
    '''
    Maximum engine vibration on C while airborne.
    '''

    name = 'Eng Vib C Max'
    units = None

    def derive(self,
               eng_vib_c=P('Eng (*) Vib C Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_c.array, airborne, max_value)


class EngVibNpMax(KeyPointValueNode):
    '''
    Maximum engine vibration on Np while airborne.
    '''

    name = 'Eng Vib Np Max'
    units = None

    def derive(self,
               eng_vib_np=P('Eng (*) Vib Np Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_np.array, airborne, max_value)


##############################################################################
# Engine: Warnings

# Chip Detection

class EngChipDetectorWarningDuration(KeyPointValueNode):
    '''
    Duration that any of the Engine Chip Detector Warnings are active.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        chips = any_of(('Eng (1) Chip Detector',
                        'Eng (2) Chip Detector',
                        'Eng (1) Chip Detector (1)',
                        'Eng (2) Chip Detector (1)',
                        'Eng (1) Chip Detector (2)',
                        'Eng (2) Chip Detector (2)'), available)
        return chips and 'Eng (*) Any Running' in available

    def derive(self,
               eng_1_chip=M('Eng (1) Chip Detector'),
               eng_2_chip=M('Eng (2) Chip Detector'),
               eng_1_chip_1=M('Eng (1) Chip Detector (1)'),
               eng_2_chip_1=M('Eng (2) Chip Detector (1)'),
               eng_1_chip_2=M('Eng (1) Chip Detector (2)'),
               eng_2_chip_2=M('Eng (2) Chip Detector (2)'),
               any_run=M('Eng (*) Any Running')):
        state = 'Chip Detected'
        combined = vstack_params_where_state(
            (eng_1_chip, state),
            (eng_2_chip, state),
            (eng_1_chip_1, state),
            (eng_2_chip_1, state),
            (eng_1_chip_2, state),
            (eng_2_chip_2, state),
        ).any(axis=0)

        running = any_run.array == 'Running'
        comb_run = combined & running
        self.create_kpvs_from_slice_durations(runs_of_ones(comb_run), self.hz)


class GearboxChipDetectorWarningDuration(KeyPointValueNode):
    '''
    Duration that any Gearbox Chip Detector Warning is active.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        chips = any_of(('EGB (1) Chip Detector',
                        'EGB (2) Chip Detector',
                        'MGB Chip Detector',
                        'MGB Front Chip Detector',
                        'MGB Sump Chip Detector',
                        'MGB Epicyclic Chip Detector',
                        'MGB (Fore) Chip Detector',
                        'MGB (Aft) Chip Detector',
                        'IGB Chip Detector',
                        'TGB Chip Detector',
                        'CGB Chip Detector',
                        'Rotor Shaft Chip Detector'), available)
        return chips and 'Eng (*) Any Running' in available

    def derive(self,
               eng_1_chip=M('EGB (1) Chip Detector'),
               eng_2_chip=M('EGB (2) Chip Detector'),
               mgb_chip=M('MGB Chip Detector'),
               mgb_front_chip=M('MGB Front Chip Detector'),
               mgb_sump_chip=M('MGB Sump Chip Detector'),
               mgb_epicyclic_chip=M('MGB Epicyclic Chip Detector'),
               mgb_fore_chip=M('MGB (Fore) Chip Detector'),
               mgb_aft_chip=M('MGB (Aft) Chip Detector'),
               igb_chip=M('IGB Chip Detector'),
               tgb_chip=M('TGB Chip Detector'),
               cgb_chip=M('CGB Chip Detector'),
               rotor_shaft_chip=M('Rotor Shaft Chip Detector'), # not gearbox but only found on Chinook
               any_run=M('Eng (*) Any Running')):

        state = 'Chip Detected'
        combined = vstack_params_where_state(
            (eng_1_chip, state),
            (eng_2_chip, state),
            (mgb_chip, state),
            (mgb_front_chip, state),
            (mgb_sump_chip, state),
            (mgb_epicyclic_chip, state),
            (mgb_fore_chip, state),
            (mgb_aft_chip, state),
            (igb_chip, state),
            (tgb_chip, state),
            (cgb_chip, state),
            (rotor_shaft_chip, state),
        ).any(axis=0)

        running = any_run.array == 'Running'
        comb_run = combined & running
        self.create_kpvs_from_slice_durations(runs_of_ones(comb_run), self.hz)


##############################################################################
# Engine Shutdown


class EngShutdownDuringFlightDuration(KeyPointValueNode):
    '''
    This KPV measures the duration the engines are not all running while
    airborne - i.e. Expected engine shutdown during flight.
    '''

    units = ut.SECOND

    def derive(self,
               eng_running=P('Eng (*) All Running'),
               airborne=S('Airborne')):
        '''
        Based upon "Eng (*) All Running" which uses the best of the available N2
        and Fuel Flow to determine whether the engines are all running.
        '''
        eng_off = eng_running.array == 'Not Running'
        for air in airborne:
            slices = runs_of_ones(eng_off[air.slice],
                                  min_samples=4 * self.frequency)
            slices = shift_slices(slices, air.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency, mark='start')


class EngRunningDuration(KeyPointValueNode):
    '''
    Measure the duration each engine was running for. Will create multiple
    measurements for each time the engine was running. If you have more than
    one measurement, this implies engine run-ups.
    '''

    units = ut.SECOND
    NAME_FORMAT = 'Eng (%(engnum)d) Running Duration'
    NAME_VALUES = {'engnum': [1, 2, 3, 4]}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, eng1=M('Eng (1) Running'), eng2=M('Eng (2) Running'),
               eng3=M('Eng (3) Running'), eng4=M('Eng (4) Running')):
        for engnum, eng in enumerate([eng1, eng2, eng3, eng4], start=1):
            if eng is None:
                continue
            # remove gaps in data shorter than 10s
            array = nearest_neighbour_mask_repair(
                eng.array, repair_gap_size=10 * self.frequency)
            # min 4s duration
            slices = runs_of_ones(
                array == 'Running', min_samples=4 * self.frequency)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start', engnum=engnum)


class EngStartTimeMax(KeyPointValueNode):
    '''
    This computes the longest engine cranking time.
    '''

    units = ut.SECOND

    @classmethod
    # This KPV is intended for use on the Embraer Phenom, 
    # but may be extended to other types at a later date.
    
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        is_phenom = ac_type == aeroplane and family and family.value == 'Phenom 300'
        return is_phenom and all_deps(cls, available)
    
    def derive(self, eng1_n2=P('Eng (1) N2'), eng2_n2=P('Eng (2) N2'),
               station=S('Stationary'), taxi_out=S('Taxi Out')):

        def start_time(array, hz):
            try:
                # Skip past any trailing non-zeros resulting from flight splitting.
                first_non_zero = np.where(array==0)[0][0]
                # Start counting from the first non-zero value
                # (the N2 recording is a solid zero before start)
                non_zeros = np.nonzero(array[first_non_zero:])[0]
                begin = non_zeros[0] + first_non_zero
                # Then find when the differential first drops below zero
                startup_count = index_at_value(np.ediff1d(array[begin:]), 0.0)
                # Check it's sensible
                if begin > 0 and array[begin + startup_count] > 50.0:
                    return begin, startup_count / hz
                else:
                    return None, None
            except:
                return None, None

        hz = eng1_n2.frequency
       
        for taxi in taxi_out.get_slices():

            eng_1_index, eng_1_time = start_time(eng1_n2.array[:taxi.stop], hz)
            eng_2_index, eng_2_time = start_time(eng2_n2.array[:taxi.stop], hz)

            '''
            print (eng_1_index, eng_1_time)
            print (eng_2_index, eng_2_time )
            import matplotlib.pyplot as plt
            plt.plot(eng1_n2.array[:taxi.stop])
            plt.plot(eng2_n2.array[:taxi.stop])
            plt.show()
            '''

            if eng_1_time and eng_2_time and eng_1_time > eng_2_time or \
               eng_1_time and not eng_2_time:
                self.create_kpv(eng_1_index, eng_1_time)
            elif eng_2_time:
                self.create_kpv(eng_2_index, eng_2_time)
    
    
##############################################################################


class SingleEngineDuringTaxiInDuration(KeyPointValueNode):
    '''
    Amount of time in seconds for which only one engine was running during
    taxi in.
    '''

    units = ut.SECOND

    def derive(self,
               all_run=M('Eng (*) All Running'),
               any_run=M('Eng (*) Any Running'),
               taxi=S('Taxi In')):

        some_running = all_run.array ^ any_run.array
        self.create_kpvs_where(some_running == 1, all_run.hz, phase=taxi)


class SingleEngineDuringTaxiOutDuration(KeyPointValueNode):
    '''
    Amount of time in seconds for which only one engine was running during
    taxi out.
    '''

    units = ut.SECOND

    def derive(self,
               all_run=M('Eng (*) All Running'),
               any_run=M('Eng (*) Any Running'),
               taxi=S('Taxi Out')):

        some_running = all_run.array ^ any_run.array
        self.create_kpvs_where(some_running == 1, all_run.hz, phase=taxi)


##############################################################################
# Gearbox Oil

class MGBOilTempMax(KeyPointValueNode):
    '''
    Find the Max temperature for the main gearbox oil.
    '''
    units = ut.CELSIUS
    name = 'MGB Oil Temp Max'

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        aircraft = ac_type == helicopter
        gearbox = any_of(('MGB Oil Temp', 'MGB (Fwd) Oil Temp',
                          'MGB (Aft) Oil Temp'), available)
        airborne = 'Airborne' in available
        return aircraft and gearbox and airborne

    def derive(self, mgb=P('MGB Oil Temp'), mgb_fwd=P('MGB (Fwd) Oil Temp'),
               mgb_aft=P('MGB (Aft) Oil Temp'), airborne=S('Airborne')):
        gearboxes = vstack_params(mgb, mgb_fwd, mgb_aft)
        gearbox = np.ma.max(gearboxes, axis=0)
        self.create_kpvs_within_slices(gearbox, airborne, max_value)


class MGBOilPressMax(KeyPointValueNode):
    '''
    Find the Maximum main gearbox oil pressure.
    '''
    units = ut.PSI
    name = 'MGB Oil Press Max'

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        aircraft = ac_type == helicopter
        gearbox = any_of(('MGB Oil Press', 'MGB (Fwd) Oil Press',
                          'MGB (Aft) Oil Press'), available)
        airborne = 'Airborne' in available
        return aircraft and gearbox and airborne

    def derive(self, mgb=P('MGB Oil Press'), mgb_fwd=P('MGB (Fwd) Oil Press'),
               mgb_aft=P('MGB (Aft) Oil Press'), airborne=S('Airborne')):
        gearboxes = vstack_params(mgb, mgb_fwd, mgb_aft)
        gearbox = np.ma.max(gearboxes, axis=0)
        self.create_kpvs_within_slices(gearbox, airborne, max_value)


class MGBOilPressMin(KeyPointValueNode):
    '''
    Find the Minimum main gearbox oil pressure.
    '''
    units = ut.PSI
    name = 'MGB Oil Press Min'

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        aircraft = ac_type == helicopter
        gearbox = any_of(('MGB Oil Press', 'MGB (Fwd) Oil Press',
                          'MGB (Aft) Oil Press'), available)
        airborne = 'Airborne' in available
        return aircraft and gearbox and airborne

    def derive(self, mgb=P('MGB Oil Press'), mgb_fwd=P('MGB (Fwd) Oil Press'),
               mgb_aft=P('MGB (Aft) Oil Press'), airborne=S('Airborne')):
        gearboxes = vstack_params(mgb, mgb_fwd, mgb_aft)
        gearbox = np.ma.min(gearboxes, axis=0)
        self.create_kpvs_within_slices(gearbox, airborne, min_value)


class MGBOilPressLowDuration(KeyPointValueNode):
    '''
    Duration of the gearbox oil pressure low warning.
    '''
    units = ut.SECOND
    name = 'MGB Oil Press Low Duration'

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        aircraft = ac_type == helicopter
        gearbox = any_of(('MGB Oil Press Low', 'MGB Oil Press Low (1)',
                          'MGB Oil Press Low (2)'), available)
        airborne = 'Airborne' in available
        return aircraft and gearbox and airborne

    def derive(self, mgb=M('MGB Oil Press Low'),
               mgb1=M('MGB Oil Press Low (1)'),
               mgb2=M('MGB Oil Press Low (2)'),
               airborne=S('Airborne')):
        hz = (mgb or mgb1 or mgb2).hz
        gearbox = vstack_params_where_state((mgb, 'Low Press'),
                                            (mgb1, 'Low Press'),
                                            (mgb2, 'Low Press'))
        self.create_kpvs_where(gearbox.any(axis=0), hz, phase=airborne)


class CGBOilTempMax(KeyPointValueNode):
    '''
    Find the Max temperature for the combining gearbox oil.
    '''
    units = ut.CELSIUS
    name = 'CGB Oil Temp Max'
    can_operate = helicopter_only

    def derive(self, cgb=P('CGB Oil Temp'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(cgb.array, airborne, max_value)


class CGBOilPressMax(KeyPointValueNode):
    '''
    Find the Maximum combining gearbox oil pressure.
    '''
    units = ut.PSI
    name = 'CGB Oil Press Max'
    can_operate = helicopter_only

    def derive(self, cgb=P('CGB Oil Press'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(cgb.array, airborne, max_value)


class CGBOilPressMin(KeyPointValueNode):
    '''
    Find the Minimum combining gearbox oil pressure.
    '''
    units = ut.PSI
    name = 'CGB Oil Press Min'
    can_operate = helicopter_only

    def derive(self, cgb=P('CGB Oil Press'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(cgb.array, airborne, min_value)


##############################################################################
class EventMarkerPressed(KeyPointValueNode):
    '''
    The time duration in seconds that the event marker parameter is active while the aircraft is airborne.
    '''

    units = ut.SECOND

    def derive(self, event=P('Event Marker'), airs=S('Airborne')):

        pushed = np.ma.clump_unmasked(np.ma.masked_equal(event.array, 0))
        events_in_air = slices_and(pushed, airs.get_slices())
        for event_in_air in events_in_air:
            if event_in_air:
                duration = (event_in_air.stop - event_in_air.start) / \
                    event.frequency
                index = (event_in_air.stop + event_in_air.start) / 2.0
                self.create_kpv(index, duration)


class HeightOfBouncedLanding(KeyPointValueNode):
    '''
    This measures the peak height of the bounced landing.

    Bounced landing phase is established by looking for the maximum height
    after touching the ground while still going fast.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               bounced_ldg=S('Bounced Landing')):

        self.create_kpvs_within_slices(alt_aal.array, bounced_ldg, max_value)


##############################################################################
# Heading


class HeadingVariationAbove80KtsAirspeedDuringTakeoff(KeyPointValueNode):
    '''
    Heading variation during takeoff above 80 knots IAS.
    
    Initial heading based on the median of heading between 60kts and 80kts IAS.
    End of takeoff in this case is either the moment when the rotation rate reaches
    1.5 deg/s or, if recorded the moment where nosewheel stops touching the ground.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        heading = any_of(('Heading True Continuous', 'Heading Continuous'), available)
        return ac_type == A('Aircraft Type', 'aeroplane') and \
               heading and all_of(('Airspeed', 'Pitch Rate', 'Takeoff'), available)
        
    units = ut.DEGREE

    def derive(self,
               nosewheel=P('Gear (N) On Ground'),
               head_true=P('Heading True Continuous'),
               head_mag=P('Heading Continuous'),
               airspeed=P('Airspeed'),
               pitch_rate=P('Pitch Rate'),
               toffs=S('Takeoff')):
        '''
        FDS originally developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Take off (Lateral). Heading changes on runway before rotation
        commenced. During rotation on some types, the a/c may be allowed to
        weathercock into wind." The heading deviation was measured as the largest deviation
        from the runway centreline between 80kts airspeed and 5 deg nose pitch up, at which
        time the weight is clearly coming off the mainwheels (we avoid using weight on
        nosewheel as this is often not recorded).
    
        This was often misinterpreted by analysts and customers who thought it was relating
        to heading deviations up to the start of rotation. Therefore the event was revised thus:
    
        1. The heading to be based on aircraft heading which is the median aircraft heading
        from the start of valid airspeed above 60kts to 80 kts.
        2. The end of the event will be at a rotation rate of 1.5 deg/sec or, where recorded,
        the last recorded moment of nosewheel on the ground.
    
        Previously named "HeadingDeviationFromRunwayAbove80KtsAirspeedDuringTakeoff"
        '''         
        for toff in toffs:
            begin = index_at_value(airspeed.array, 80.0, _slice=toff.slice)
            if not begin:
                self.warning(
                    "'%s' did not transition through 80 kts in '%s' slice '%s'.",
                    airspeed.name, toffs.name, toff.slice)
                continue
            spd = np.ma.masked_less(airspeed.array, 60)
            first_spd_idx = first_valid_sample(spd[toff.slice.start:ceil(begin) + 1])[0] + toff.slice.start
            # Pick first heading parameter with valid data in phase.
            head = first_valid_parameter(head_true, head_mag, phases=(slice(first_spd_idx, ceil(begin)),))
            if head is None:
                # We have no valid heading to use.
                self.warning(
                    "No valid heading data identified in takeoff slice '%s'.",
                    toff.slice)
                continue
            datum_heading = np.ma.median(head.array[first_spd_idx:ceil(begin)])

            end = None
            if nosewheel:
                end = index_at_value(nosewheel.array.data, 0.0, _slice=toff.slice)
            if not end or end < begin: # Fallback
                end = index_at_value(pitch_rate.array, 1.5, _slice=toff.slice)
            if not end or end < begin:
                self.warning(
                    "No end condition identified in takeoff slice '%s'.",
                    toff.slice)
                continue

            scan = slice(ceil(begin), ceil(end))
            # The data to test is extended to include aligned endpoints for the
            # 80kt and 1.5deg conditions. This also reduces the computational load as
            # we don't have to work out the deviation from the takeoff runway for all the flight.
            to_test = np.ma.concatenate([
                np.ma.array([value_at_index(head.array, begin) or
                             closest_unmasked_value(head.array, begin).value]),
                head.array[scan],
                np.ma.array([value_at_index(head.array, end) or
                             closest_unmasked_value(head.array, end).value])])
            dev = to_test - datum_heading
            index, value = max_abs_value(dev, slice(0,len(dev)))

            # The true index is offset and may be either of the special end conditions.
            if index==0:
                true_index = begin
            elif index==len(to_test)-1:
                true_index = end
            else:
                true_index = index + begin

            self.create_kpv(true_index, value)


class HeadingDeviationFromRunwayAtTOGADuringTakeoff(KeyPointValueNode):
    '''
    Heading deviation from the runway during takeoff when TOGA activated. 
    '''

    name = 'Heading Deviation From Runway At TOGA During Takeoff'
    units = ut.DEGREE

    can_operate = aeroplane_only

    def derive(self,
               toga=M('Takeoff And Go Around'),
               head=P('Heading True Continuous'),
               takeoff=S('Takeoff'),
               rwy=A('FDR Takeoff Runway')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Take off (Lateral). TOGA pressed before a/c aligned."

        Align to Takeoff And Go Around for most accurate state change indices.
        '''
        if ambiguous_runway(rwy):
            return
        indexes = find_edges_on_state_change('TOGA', toga.array, phase=takeoff)
        for index in indexes:
            # Measure at known state instead of interpolated transition
            index = ceil(index)
            brg = value_at_index(head.array, index)
            if brg in (None, np.ma.masked):
                self.warning("Heading True Continuous is masked at index '%s'", index)
                continue
            dev = runway_deviation(brg, rwy.value)
            self.create_kpv(index, dev)


class HeadingDeviationFromRunwayAt50FtDuringLanding(KeyPointValueNode):
    '''
    Heading deviation from runway at 50ft AAL during landing.
    '''

    units = ut.DEGREE

    can_operate = aeroplane_only

    def derive(self,
               head=P('Heading True Continuous'),
               landings=S('Landing'),
               rwy=A('FDR Landing Runway')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Take off (Lateral). Crosswind. Could look at the difference
        between a/c heading and R/W heading at 50ft."
        '''
        if ambiguous_runway(rwy):
            return
        # Only have runway details for final landing.
        land = landings[-1]
        # By definition, landing starts at 50ft.
        brg = closest_unmasked_value(head.array, land.start_edge).value
        dev = runway_deviation(brg, rwy.value)
        self.create_kpv(land.start_edge, dev)


class HeadingDeviationFromRunwayDuringLandingRoll(KeyPointValueNode):
    '''
    Heading deviation from runway during landing roll.
    '''

    units = ut.DEGREE

    can_operate = aeroplane_only

    def derive(self,
               head=P('Heading True Continuous'),
               land_rolls=S('Landing Roll'),
               rwy=A('FDR Landing Runway')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Landing (Lateral) Heading changes on runways."
        '''
        if ambiguous_runway(rwy):
            return

        final_landing = land_rolls[-1].slice
        dev = runway_deviation(head.array, rwy.value)
        self.create_kpv_from_slices(dev, [final_landing], max_abs_value)


class HeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxStandardApproach(KeyPointValueNode):
    '''
    Maximum heading variation (PTP) 1.5 to 1.0 NM from touchdown. (helicopter only)
    '''

    units = ut.DEGREE

    name = 'Heading Variation 1.5 NM To 1.0 NM From Offshore Touchdown Max Standard Approach'

    can_operate = helicopter_only

    def derive(self, heading=P('Heading Continuous'),
               dtts=KTI('Distance To Touchdown'),
               offshore_twn=KTI('Offshore Touchdown'),
               approaches=App('Approach Information')):
        
        for approach in approaches:
            if approach.type in ['APPROACH', 'LANDING', 'GO_AROUND', 'TOUCH_AND_GO', 'RIG']:
                for tdwn in offshore_twn:
                    if is_index_within_slice(tdwn.index, slice(approach.slice.start, approach.slice.stop+10)):
                        start_kti = dtts.get_previous(tdwn.index, name='1.5 NM To Touchdown')
                        stop_kti = dtts.get_previous(tdwn.index, name='1.0 NM To Touchdown')
                        if start_kti and stop_kti:
                            phase = slice(start_kti.index, stop_kti.index+1)
                            heading_delta = np.ma.ptp(heading.array[phase])
                            self.create_kpv(phase.stop-1, heading_delta)


class HeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxSpecialProcedure(KeyPointValueNode):
    '''
    Maximum heading variation (PTP) 1.5 to 1.0 NM from touchdown. (helicopter only)
    '''

    units = ut.DEGREE

    name = 'Heading Variation 1.5 NM To 1.0 NM From Offshore Touchdown Max Special Procedure'

    can_operate = helicopter_only

    def derive(self, heading=P('Heading Continuous'),
               dtts=KTI('Distance To Touchdown'),
               offshore_twn=KTI('Offshore Touchdown'),
               approaches=App('Approach Information')):
        
        for approach in approaches:
            if approach.type in ['SHUTTLING', 'AIRBORNE_RADAR']:
                for tdwn in offshore_twn:
                    if is_index_within_slice(tdwn.index, slice(approach.slice.start, approach.slice.stop+10)):
                        start_kti = dtts.get_previous(tdwn.index, name='1.5 NM To Touchdown')
                        stop_kti = dtts.get_previous(tdwn.index, name='1.0 NM To Touchdown')
                        if start_kti and stop_kti:
                            phase = slice(start_kti.index, stop_kti.index+1)
                            heading_delta = np.ma.ptp(heading.array[phase])
                            self.create_kpv(phase.stop-1, heading_delta)
                                            

class HeadingVariation300To50Ft(KeyPointValueNode):
    '''
    Heading variation between 300ft and 50ft AAL during landing.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Heading Continuous']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               head=P('Heading Continuous'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            alt_band = np.ma.masked_outside(alt_agl.array, 50, 300)
            alt_app_sections = valid_slices_within_array(alt_band, descending)
            for band in alt_app_sections:
                if slice_duration(band, head.frequency) < HOVER_MIN_DURATION:
                    continue
                dev = np.ma.ptp(head.array[band])
                self.create_kpv(band.stop, dev)
        else:
            for band in alt_aal.slices_from_to(300, 50, threshold=0.25):
                # Trap for zero length slices
                if slice_duration(band, head.frequency) < 1.0:
                    continue
                dev = np.ma.ptp(head.array[band])
                self.create_kpv(band.stop, dev)


class HeadingVariation500To50Ft(KeyPointValueNode):
    '''
    Heading variation between 500ft and 50ft AAL during landing.
    '''

    units = ut.DEGREE

    def derive(self,
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for band in alt_aal.slices_from_to(500, 50):
            dev = np.ma.ptp(head.array[band])
            self.create_kpv(band.stop, dev)

class HeadingVariation800To50Ft(KeyPointValueNode):
    '''
    Heading variation between 800ft AAL and 50ft AAL.
    '''

    units = ut.DEGREE

    def derive(self,
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for band in alt_aal.slices_from_to(800, 50):
            dev = np.ma.ptp(head.array[band])
            self.create_kpv(band.stop, dev)

class HeadingVariationAbove100KtsAirspeedDuringLanding(KeyPointValueNode):
    '''
    Heading variation above 100kts IAS during landing.
    '''

    units = ut.DEGREE

    can_operate = aeroplane_only

    def derive(self,
               head=P('Heading Continuous'),
               airspeed=P('Airspeed'),
               alt=P('Altitude AAL For Flight Phases'),
               lands=S('Landing')):
        '''
        For landing the Altitude AAL is used to detect start of landing to avoid
        variation from the use of different aircraft recording configurations.
        '''
        for land in lands:
            begin = index_at_value(alt.array, 1.0, _slice=land.slice)
            end = index_at_value(airspeed.array, 100.0, _slice=land.slice)
            if begin is None or begin > end:
                # Corrupt landing slices or landed below 100kts. Can happen!
                break
            else:
                head_dev = np.ma.ptp(head.array[begin:end + 1])
                self.create_kpv((begin + end) / 2, head_dev)


class HeadingVariationTouchdownPlus4SecTo60KtsAirspeed(KeyPointValueNode):
    '''
    Heading variation between the point of 4 seconds after touchdown
    and the point where the aircraft reaches 60kts IAS during landing roll.
    '''

    units = ut.DEGREE

    can_operate = aeroplane_only

    def derive(self,
               head=P('Heading Continuous'),
               airspeed=P('Airspeed True'),
               tdwns=KTI('Touchdown')):
        '''
        Maximum difference in Magnetic Heading.

        The final turnoff is ignored, as this may arise above 60kt IAS at a rapid
        exit turnoff. The highest variation from the mean heading is marked as
        the point of interest.
       
        Airspeed True is used as this includes short term inertial corrections
        that make this more reliable than indicated airspeed which can drop out
        around 60 kts on some types.
        '''
        for tdwn in tdwns:
            begin = tdwn.index + 4.0 * head.frequency
            end = index_at_value(airspeed.array, 60.0, slice(begin, None), endpoint='nearest')
            if end:
                # We have a meaningful slice to examine.
                to_scan = head.array[begin:end + 1]
                if not np.ma.count(to_scan):
                    continue
                # Correct for rounding down of array index at first data point.
                to_scan[0] = value_at_index(head.array, begin)
                indexes, values = cycle_finder(to_scan)
                # If the final sample is due to a turnoff, remove this before
                # examining the wanderings.
                if indexes[-1] >= len(to_scan) - 1:
                    indexes = indexes[:-1]
                    values = values[:-1]
                # The overall deviation is...
                dev = np.ma.ptp(values)
                # Which happened at...
                wander = np.ma.abs(to_scan[:indexes[-1]] - np.ma.average(to_scan[:indexes[-1]]))
                # check if there is any wandering
                if len(wander):
                    index = np.ma.argmax(wander)
                    # Create the KPV.
                    self.create_kpv(begin + index, dev)


class HeadingVacatingRunway(KeyPointValueNode):
    '''
    Aircraft heading at the point it starts vacating the runway.
    '''

    units = ut.DEGREE

    def derive(self,
               head=P('Heading Continuous'),
               off_rwys=KTI('Landing Turn Off Runway')):
        '''
        Heading vacating runway is only used to try to identify handed
        runways in the absence of better information. See Approaches node.
        '''
        # To save taking modulus of the entire array, we'll do this in stages.
        for off_rwy in off_rwys:
            # We try to extend the index by five seconds to make a clear
            # heading change. The KTI is at the point of turnoff at which
            # moment the heading change can be very small.
            index = min(off_rwy.index + 5, len(head.array) - 1)
            value = head.array[index] % 360.0
            self.create_kpv(index, value)


class HeadingRateWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum rate of heading change while airborne.
    '''

    units = ut.DEGREE_S

    def derive(self, heading_rate=P('Heading Rate'), airborne=P('Airborne')):
        self.create_kpvs_within_slices(heading_rate.array, airborne.get_slices(), max_abs_value)


class TrackVariation100To50Ft(KeyPointValueNode):
    '''
    Checking the variation in track angle during the latter stages of the descent. 
    Helicopter only. Uses Altitude AGL.
    '''

    name = 'Track Variation 100 To 50 Ft'
    units = ut.DEGREE_S

    can_operate = helicopter_only

    def derive(self, track=P('Track'),
               alt_agl=P('Altitude AGL')):

        # The threshold applied here ensures that the altitude passes through this range and does not
        # just dip into the range, as might happen for a light aircraft or helicopter flying at 100ft.
        for band in alt_agl.slices_from_to(100, 50, threshold=1.0):
            dev = np.ma.ptp(track.array[band])
            self.create_kpv(band.stop, dev)



##############################################################################
# Height


class HeightMinsToTouchdown(KeyPointValueNode):
    '''
    Altitude AAL at N minutes to touchdown.
    '''

    # TODO: Review and improve this technique of building KPVs on KTIs.
    from analysis_engine.key_time_instances import MinsToTouchdown

    NAME_FORMAT = 'Height ' + MinsToTouchdown.NAME_FORMAT
    NAME_VALUES = MinsToTouchdown.NAME_VALUES
    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               mtt_kti=KTI('Mins To Touchdown')):

        for mtt in mtt_kti:
            # XXX: Assumes that the number will be the first part of the name:
            time = int(mtt.name.split(' ')[0])
            self.create_kpv(mtt.index, alt_aal.array[mtt.index], time=time)


##############################################################################
# Flap


class FlapAtFirstMovementAfterEngineStart(KeyPointValueNode):
    '''
    Records the flap setting at the point that the aircraft starts moving.
    '''
    units = ut.DEGREE

    def derive(self, flap=P('Flap'), gnd_spd=P('Groundspeed Signed'),
               eng_start=KTI('First Eng Start Before Liftoff'),
               liftoff=KTI('Liftoff')):
        gndspd = mask_outside_slices(gnd_spd.array,
                                     [slice(eng_start.get_first().index,
                                            liftoff.get_first().index), ])
        # Slice any of forward movement above GROUNDSPEED_FOR_MOBILE.
        # This will filter out pushback and tows. It may not filtered all tows.
        movement = runs_of_ones(gndspd > GROUNDSPEED_FOR_MOBILE)
        # If there is more than one slice, this may indicate a period in which
        # the aircraft has stopped, possibly because of a tow or taxi.
        #
        # Attempt to detect tows (from example flight, speed does not go 
        # beyond 4 kt). Filter out any slices where the groundspeed
        # does not go above 4 kt and take the first slice. This will be slice
        # where the aircraft is moving under its own power.
        moving = [s for s in movement if (max_value(gndspd, s)).value > 4.0]
        if moving:
            self.create_kpv(moving[0].start, flap.array[moving[0].start])


class FlapAtLiftoff(KeyPointValueNode):
    '''
    Flap angle measured at liftoff.

    Note that this KPV uses the flap surface angle, not the flap lever angle.
    '''

    units = ut.DEGREE

    def derive(self, flap=M('Flap'), liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(flap.array, liftoffs, interpolate=False)


class FlapAtTouchdown(KeyPointValueNode):
    '''
    Flap angle measured at touchdown.

    Note that this KPV uses the flap surface angle, not the flap lever angle.
    '''

    units = ut.DEGREE

    def derive(self, flap=M('Flap'), touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(flap.array, touchdowns, interpolate=False)


class FlapAtGearDownSelection(KeyPointValueNode):
    '''
    Flap angle at gear down selection.

    Note that this KPV uses the flap surface angle, not the flap lever angle.
    '''

    units = ut.DEGREE

    def derive(self, flap=M('Flap'), gear_dn_sel=KTI('Gear Down Selection')):

        self.create_kpvs_at_ktis(flap.array, gear_dn_sel, interpolate=False)


class FlapAtGearUpSelectionDuringGoAround(KeyPointValueNode):
    '''
    Flap angle at gear up selection during go around.

    Note that this KPV uses the flap surface angle, not the flap lever angle.
    '''
    units = ut.DEGREE

    def derive(self, flap=M('Flap'),
               gear_up_sel=KTI('Gear Up Selection During Go Around')):
        self.create_kpvs_at_ktis(flap.array, gear_up_sel, interpolate=False)


class FlapWithGearUpMax(KeyPointValueNode):
    '''
    Maximum flap angle while the landing gear is up.

    Note that this KPV uses the flap surface angle, not the flap lever angle.
    '''

    units = ut.DEGREE

    def derive(self, flap=M('Flap'), gear=M('Gear Down')):

        gear_up = np.ma.masked_equal(gear.array.raw, gear.array.state['Down'])
        gear_up_slices = np.ma.clump_unmasked(gear_up)
        self.create_kpvs_within_slices(flap.array, gear_up_slices, max_value)


class FlapWithSpeedbrakeDeployedMax(KeyPointValueNode):
    '''
    Maximum flap angle while the speedbrakes are deployed.

    Note that this KPV uses the flap surface angle, not the flap lever angle.
    '''

    units = ut.DEGREE
    
    @classmethod
    def can_operate(cls, available):
        return all_of(('Flap Including Transition', 'Speedbrake Selected',
                      'Airborne', 'Landing'), available) or \
               all_of(('Flap Including Transition', 'Speedbrake Selected',
                      'Airborne', 'Landing', 'Landing Attitude Mode Delta CL'), 
                      available)
    def derive(self,
               flap=M('Flap Including Transition'),
               spd_brk=M('Speedbrake Selected'),
               airborne=S('Airborne'),
               landings=S('Landing'),
               lam=P('Landing Attitude Mode Delta CL'),):

        deployed = spd_brk.array == 'Deployed/Cmd Up'
        deployed = mask_outside_slices(deployed, airborne.get_slices())
        deployed = mask_inside_slices(deployed, landings.get_slices())
        if lam:
            deployed = np.ma.masked_where(lam.array!=0, deployed)
        deployed_slices = runs_of_ones(deployed)
        self.create_kpv_from_slices(flap.array, deployed_slices, max_value)


class FlapAt1000Ft(KeyPointValueNode):
    '''
    Flap setting at 1000ft AAL on approach.

    Note that this KPV uses the flap surface angle, not the flap lever angle.
    '''

    units = ut.DEGREE

    def derive(self, flap=M('Flap'), gates=KTI('Altitude When Descending')):

        for gate in gates.get(name='1000 Ft Descending'):
            self.create_kpv(gate.index, flap.array.raw[gate.index])


class FlapAt500Ft(KeyPointValueNode):
    '''
    Flap setting at 500ft AAL on approach.

    Note that this KPV uses the flap surface angle, not the flap lever angle.
    '''

    units = ut.DEGREE

    def derive(self, flap=M('Flap'), gates=KTI('Altitude When Descending')):

        for gate in gates.get(name='500 Ft Descending'):
            self.create_kpv(gate.index, flap.array.raw[gate.index])


class GearDownToLandingFlapConfigurationDuration(KeyPointValueNode):
    '''
    Duration between Gear Down selection and Landing Flap Configuration
    selection.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family'),
                    engine_type=A('Engine Type'), engine_series=A('Engine Series')):
        flap_lever = any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available)
        required = all_of(('Gear Down Selection', 'Approach And Landing'), available)
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(cls, 'vref', *attrs) or lookup_table(cls, 'vapp', *attrs)
        return flap_lever and required and table

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               gear_dn_sel=KTI('Gear Down Selection'),
               approaches=S('Approach And Landing'),
               model=A('Model'), series=A('Series'), family=A('Family'),
               engine_type=A('Engine Type'), engine_series=A('Engine Series')):
        '''
        Landing Flap Configurations are sourced from the aircraft table
        '''
        attrs = (model, series, family, engine_type, engine_series)
        table = lookup_table(self, 'vref', *attrs) or lookup_table(self, 'vapp', *attrs)
        detents = table.vref_detents or table.vapp_detents

        flap_lever = flap_lever or flap_synth

        for approach in approaches:
            # Assume Gear Down is selected before lowest point of descent.
            last_gear_dn = gear_dn_sel.get_last(within_slice=approach.slice)
            if not last_gear_dn:
                # gear down was selected before approach? kpv should not be triggered
                continue

            landing_flap_changes = []
            for valid_setting in detents:
                landing_flap_changes.extend(find_edges_on_state_change(
                    valid_setting,
                    flap_lever.array,
                    phase=[approach.slice],
                ))

            if not landing_flap_changes:
                if flap_lever.array[slice_midpoint(approach.slice)] in detents:
                    # create kpv if landing flap configuration is for entire approach
                    self.create_kpv(approach.slice.start,
                                    (approach.slice.start - last_gear_dn.index) / self.frequency)
                continue

            flap_idx = sorted(landing_flap_changes)[0]
            diff = flap_idx - last_gear_dn.index

            self.create_kpv(flap_idx, diff / self.frequency)


##############################################################################


class FlareDuration20FtToTouchdown(KeyPointValueNode):
    '''
    This is the time from the last time the aircraft passed 
    through 20ft AAL to the moment of touchdown.
    
    The Altitude Radio reference is included to make sure this KPV is not
    computed if there is no radio height reference. With small turboprops, we
    have seen 40ft pressure altitude difference between the point of
    touchdown and the landing roll, so trying to measure this 20ft to
    touchdown difference is impractical.
    
    '''

    can_operate = aeroplane_only

    units = ut.SECOND

    def derive(self,
               alt_aal=P('Altitude AAL For Flight Phases'),
               tdowns=KTI('Touchdown'),
               lands=S('Landing'),
               ralt=P('Altitude Radio')):

        for tdown in tdowns:
            this_landing = lands.get_surrounding(tdown.index)
            if this_landing:
                # Scan backwards from touchdown to the start of the landing
                # which is defined as 50ft, so will include passing through
                # 20ft AAL.
                aal_at_tdown = value_at_index(alt_aal.array, tdown.index)
                idx_20 = index_at_value(alt_aal.array, aal_at_tdown + 20.0,
                                        _slice=slice(tdown.index,
                                                     this_landing[0].start_edge,
                                                     -1))
                if not idx_20:
                    # why not?
                    raise ValueError("Did not cross 20ft before touchdown point - sounds unlikely")
                self.create_kpv(
                    tdown.index,
                    (tdown.index - idx_20) / alt_aal.frequency)


class FlareDistance20FtToTouchdown(KeyPointValueNode):
    '''
    This is the distance travelled from the last time the aircraft passed 
    through 20ft AAL to the point of touchdown, in metres.
    '''
    
    #TODO: Write a test for this function with less than one second between 20ft and touchdown, using interval arithmetic.
    #NAX_1_LN-DYC_20120104234127_22_L3UQAR___dev__sdb.001.hdf5
    
    can_operate = aeroplane_only

    units = ut.METER

    def derive(self,
               alt_aal=P('Altitude AAL For Flight Phases'),
               tdowns=KTI('Touchdown'),
               lands=S('Landing'),
               gspd=P('Groundspeed')):

        scale = ut.multiplier(ut.KT, ut.METER_S)
        for tdown in tdowns:
            this_landing = lands.get_surrounding(tdown.index)
            if this_landing:
                idx_20 = index_at_value(
                    alt_aal.array, 20.0,
                    _slice=slice(ceil(tdown.index), this_landing[0].slice.start - 1, -1))
                # Integrate returns an array, so we need to take the max
                # value to yield the KTP value.
                if idx_20:
                    dist = max(integrate(gspd.array[idx_20:tdown.index + 1], gspd.hz, scale=scale))
                    self.create_kpv(tdown.index, dist)


##############################################################################
# Fuel Quantity


class FuelQtyAtLiftoff(KeyPointValueNode):
    '''
    Fuel quantity at liftoff in kg.
    '''

    units = ut.KG

    def derive(self,
               fuel_qty=P('Fuel Qty'),
               liftoffs=KTI('Liftoff')):

        try:
            self.create_kpvs_at_ktis(
                moving_average(repair_mask(fuel_qty.array), 19), liftoffs)
        except ValueError:
            self.warning("KPV '%s' will not be created because '%s' array "
                         "could not be repaired.", self.name, fuel_qty.name)
            return


class FuelQtyAtTouchdown(KeyPointValueNode):
    '''
    Fuel quantity at touchdown in kg.
    '''

    units = ut.KG

    def derive(self,
               fuel_qty=P('Fuel Qty'),
               touchdowns=KTI('Touchdown')):

        try:
            self.create_kpvs_at_ktis(
                moving_average(repair_mask(fuel_qty.array), 19), touchdowns)
        except ValueError:
            self.warning("KPV '%s' will not be created because '%s' array "
                         "could not be repaired.", self.name, fuel_qty.name)
            return


class FuelQtyWingDifferenceMax(KeyPointValueNode):
    '''
    Maximum difference between fuel quantity in wing tanks. Measured in kg.
    
    If the value is positive that indicates more fuel in the right wing's tank,
    if negative - in the left wing's tank.
    '''
    def derive(self, left_wing=P('Fuel Qty (L)'), right_wing=P('Fuel Qty (R)'),
               airbornes=S('Airborne')):

        diff = right_wing.array - left_wing.array
        #value = max_abs_value(diff)
        self.create_kpv_from_slices(
            diff,
            airbornes.get_slices(),
            max_abs_value
        )

class FuelQtyWingDifference787Max(KeyPointValueNode):
    '''
    Maximum proportion of the 787 permitted imbalance.
    
    If the value is positive that indicates more fuel in the right wing's tank,
    if negative - in the left wing's tank.
    '''

    @classmethod
    def can_operate(cls, available, frame=A('Frame')):
        if frame and frame.value.startswith('787'):
            return all_deps(cls, available)
        else:
            return False

    units = ut.PERCENT

    def derive(self, left_wing=P('Fuel Qty (L)'), right_wing=P('Fuel Qty (R)'),
               airbornes=S('Airborne')):

        diff = right_wing.array - left_wing.array
        total = right_wing.array + left_wing.array
        xp = [38500, 66100] # For these total fuel weights in the wings...
        yp = [2300, 1300] # these are the permitted imbalance levels.
        imbalance_limit = np.interp(total, xp, yp)
        imbalance_percent = (diff/imbalance_limit) * 100.0
        # Second_window needs a time window that is a binary power of the sample rate
        # so we use 32 seconds, in place of the specified 30 seconds.
        # Note that second_window ignores masks, so the quantization of 787 fuel data
        # is not a problem using this technique.
        self.create_kpv_from_slices(
            second_window(imbalance_percent, left_wing.frequency, 32.0),
            airbornes.get_slices(),
            max_abs_value)


class FuelQtyLowWarningDuration(KeyPointValueNode):
    '''
    Measures the duration of the Fuel Quantity Low warning discretes.
    '''

    units = ut.SECOND

    def derive(self, warning=M('Fuel Qty (*) Low')):

        self.create_kpvs_where(warning.array == 'Warning', warning.hz)


class FuelJettisonDuration(KeyPointValueNode):
    '''
    Duration for which the fuel jettison nozzle valve was open.
    '''

    units = ut.SECOND

    def derive(self,
               jet=P('Fuel Jettison Nozzle'),
               airborne=S('Airborne')):

        self.create_kpvs_where(jet.array == 'Disagree', jet.hz, phase=airborne)


##############################################################################
# Groundspeed


class GroundspeedWithGearOnGroundMax(KeyPointValueNode):
    '''
    The highest groundspeed with gear on ground.
    For fixed wing, at liftoff or touchdown. 
    For helicopters with wheeled undercarriage, the point of highest taxi speed.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed Signed'),
               gear=M('Gear On Ground')):

        self.create_kpvs_within_slices(
            gnd_spd.array,
            runs_of_ones(gear.array == 'Ground'),
            max_value)


class GroundspeedWhileTaxiingStraightMax(KeyPointValueNode):
    '''
    Maximum recorded groundspeed while taxiing straight.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed Signed'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):
        '''
        Groundspeed while not turning is rarely an issue, so we compute only one
        KPV for taxi out and one for taxi in. The straight sections are identified
        by masking the turning phases and then testing the resulting data.
        '''
        gnd_spd_array = mask_inside_slices(gnd_spd.array, turns.get_slices())
        self.create_kpvs_within_slices(gnd_spd_array, taxiing, max_value)


class GroundspeedInStraightLineDuringTaxiInMax(KeyPointValueNode):
    '''
    Maximum groundspeed recorded while taxiing in straight line
    during taxi in.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed Signed'),
               taxiing=S('Taxi In'),
               turns=S('Turning On Ground')):

        gnd_spd_array = mask_inside_slices(gnd_spd.array, turns.get_slices())
        self.create_kpvs_within_slices(gnd_spd_array, taxiing, max_value)


class GroundspeedInStraightLineDuringTaxiOutMax(KeyPointValueNode):
    '''
    Maximum groundspeed recorded while taxiing in straight line
    during taxi out.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed Signed'),
               taxiing=S('Taxi Out'),
               turns=S('Turning On Ground')):

        gnd_spd_array = mask_inside_slices(gnd_spd.array, turns.get_slices())
        self.create_kpvs_within_slices(gnd_spd_array, taxiing, max_value)


class GroundspeedWhileTaxiingTurnMax(KeyPointValueNode):
    '''
    Maximum groundspeed detected during turn while taxiing.
    
    The rate of change of heading used to detect a turn during taxi is %.2f
    degrees per second.
    ''' % HEADING_RATE_FOR_TAXI_TURNS

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed Signed'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):

        gnd_spd_array = mask_outside_slices(gnd_spd.array, turns.get_slices())
        self.create_kpvs_within_slices(gnd_spd_array, taxiing, max_value)


class GroundspeedInTurnDuringTaxiOutMax(KeyPointValueNode):
    '''
    Maximum groundspeed detected while turning during taxi out.
    
    The rate of change of heading used to detect a turn during taxi is %.2f
    degrees per second.
    ''' % HEADING_RATE_FOR_TAXI_TURNS

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed Signed'),
               taxiing=S('Taxi Out'),
               turns=S('Turning On Ground')):

        gnd_spd_array = mask_outside_slices(gnd_spd.array, turns.get_slices())
        self.create_kpvs_within_slices(gnd_spd_array, taxiing, max_value)


class GroundspeedInTurnDuringTaxiInMax(KeyPointValueNode):
    '''
    Maximum groundspeed detected while turning during taxi in.
    
    The rate of change of heading used to detect a turn during taxi is %.2f
    degrees per second.
    ''' % HEADING_RATE_FOR_TAXI_TURNS

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed Signed'),
               taxiing=S('Taxi In'),
               turns=S('Turning On Ground')):

        gnd_spd_array = mask_outside_slices(gnd_spd.array, turns.get_slices())
        self.create_kpvs_within_slices(gnd_spd_array, taxiing, max_value)


class GroundspeedDuringRejectedTakeoffMax(KeyPointValueNode):
    '''
    Measures the maximum Groundspeed during a rejected takeoff. If
    Groundspeed is not recorded, we estimate it by integrating the
    Longitudinal Acceleration.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return 'Rejected Takeoff' in available and any_of(
            ('Acceleration Longitudinal Offset Removed',
             'Groundspeed Signed'), available)

    def derive(self,
               # Accel is first dependency as maximum recoding frequency
               accel=P('Acceleration Longitudinal Offset Removed'),
               gnd_spd=P('Groundspeed Signed'),
               rtos=S('Rejected Takeoff')):
        if gnd_spd:
            self.create_kpvs_within_slices(gnd_spd.array, rtos, max_value)
            return
        # Without groundspeed, we only calculate an estimated Groundspeed for RTOs.
        scale = ut.convert(GRAVITY_IMPERIAL, ut.FPS, ut.KT)
        for rto_slice in rtos.get_slices():
            spd = integrate(accel.array[rto_slice], accel.frequency, scale=scale)
            index, value = max_value(spd)
            self.create_kpv(rto_slice.start + index, value)


class GroundspeedAtLiftoff(KeyPointValueNode):
    '''
    The groundspeed at the point of liftoff.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(gnd_spd.array, liftoffs)


class GroundspeedAtTouchdown(KeyPointValueNode):
    '''
    The groundspeed at the point of touchdown.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(gnd_spd.array, touchdowns)


class Groundspeed20FtToTouchdownMax(KeyPointValueNode):
    '''
    The highest groundspeed during the flare manoeuvre.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Groundspeed'),
               alt_agl=P('Altitude AGL'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_agl.slices_to_kti(20, touchdowns),
            max_value,
        )


class Groundspeed20SecToOffshoreTouchdownMax(KeyPointValueNode):
    '''
    Find the maximum groundspeed 20 seconds from the point of an offshore touchdown.
    (helicopters only)
    '''
    units = ut.KT
    
    can_operate = helicopter_only
    
    def derive(self, groundspeed=P('Groundspeed'),
               touchdown=KTI('Offshore Touchdown'),
               secs_tdwn=KTI('Secs To Touchdown')):

        #idx_to_tdwn = \
            #[s.index for s in secs_tdwn if s.name == '20 Secs To Touchdown']
        #idx_at_tdwn = [t.index for t in touchdown]
        
        #if idx_to_tdwn and idx_at_tdwn:
            #_slice = [slice(a, b) for a, b in zip(idx_to_tdwn, idx_at_tdwn)]
            #self.create_kpvs_within_slices(groundspeed.array, _slice,
                                           #max_value)
        for tdwn in touchdown:
            secs_to_touchdown = secs_tdwn.get_previous(tdwn.index)
            if secs_to_touchdown:
                self.create_kpv(*max_value(groundspeed.array,
                                           _slice=slice(secs_to_touchdown.index, tdwn.index+1),
                                           start_edge=secs_to_touchdown.index,
                                           stop_edge=tdwn.index))


class Groundspeed0_8NMToOffshoreTouchdownSpecialProcedure(KeyPointValueNode):
    '''
    Groundspeed at 0.8 NM away from an offshore touchdown during a Shuttling Approach, 
    ARDA or AROA. (helicopters only)
    '''

    name = 'Groundspeed 0.8 NM To Offshore Touchdown Special Procedure'

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, groundspeed=P('Groundspeed'), 
               dtts=KTI('Distance To Touchdown'), touchdown=KTI('Offshore Touchdown'),
               approaches=App('Approach Information')):
        
        for approach in approaches:
            if approach.type in ['SHUTTLING','AIRBORNE_RADAR']:
                for tdwn in touchdown:
                    if is_index_within_slice(tdwn.index, slice(approach.slice.start, approach.slice.stop+10)):
                        dist_to_touchdown = dtts.get_previous(tdwn.index, name='0.8 NM To Touchdown')
                        if dist_to_touchdown:
                            self.create_kpvs_at_ktis(groundspeed.array, [dist_to_touchdown])
                                    
        
class Groundspeed0_8NMToOffshoreTouchdownStandardApproach(KeyPointValueNode):
    '''
    Groundspeed at 0.8 NM away from an offshore touchdown during a standard 
    approach. (helicopters only)
    '''

    name = 'Groundspeed 0.8 NM To Offshore Touchdown Standard Approach'
    units = ut.KT
    can_operate = helicopter_only

    def derive(self, groundspeed=P('Groundspeed'), 
               dtts=KTI('Distance To Touchdown'), touchdown=KTI('Offshore Touchdown'),
               approaches=App('Approach Information')):
        
        for approach in approaches:
            if approach.type in ['APPROACH', 'LANDING', 'GO_AROUND', 'TOUCH_AND_GO', 'RIG']:
                for tdwn in touchdown:
                    if is_index_within_slice(tdwn.index, slice(approach.slice.start, approach.slice.stop+10)):
                        dist_to_touchdown = dtts.get_previous(tdwn.index, name='0.8 NM To Touchdown')
                        if dist_to_touchdown:
                            self.create_kpvs_at_ktis(groundspeed.array, [dist_to_touchdown])

                            
class GroundspeedVacatingRunway(KeyPointValueNode):
    '''
    Groundspeed at the point of turning off the landing runway. May be quite high for runways with Rapid Exit Turnoffs.
    '''

    units = ut.KT

    can_operate = aeroplane_only
    
    def derive(self,
               gnd_spd=P('Groundspeed Signed'),
               off_rwy=KTI('Landing Turn Off Runway')):

        self.create_kpvs_at_ktis(gnd_spd.array, off_rwy)


class GroundspeedAtTOGA(KeyPointValueNode):
    '''
    This KPV measures the groundspeed at the point of TOGA selection.
    '''

    name = 'Groundspeed At TOGA'
    units = ut.KT

    def derive(self,
               toga=M('Takeoff And Go Around'),
               gnd_spd=P('Groundspeed'),
               takeoffs=S('Takeoff')):
        '''
        Align to Takeoff And Go Around for most accurate state change indices.

        Note: Takeoff phase is used as this includes turning onto the runway
        whereas Takeoff Roll only starts after the aircraft is accelerating.
        '''
        indexes = find_edges_on_state_change('TOGA', toga.array, phase=takeoffs)
        for index in indexes:
            # Measure at known state instead of interpolated transition
            index = ceil(index)
            self.create_kpv(index, value_at_index(gnd_spd.array, index))


class GroundspeedWithThrustReversersDeployedMin(KeyPointValueNode):
    '''
    Minimum groundspeed measured with Thrust Reversers deployed and the maximum
    of either engine's EPR measurements above %.2f%% or N1 measurements
    above %d%%
    ''' % (REVERSE_THRUST_EFFECTIVE_EPR, REVERSE_THRUST_EFFECTIVE_N1)

    align = False
    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return all_of(('Groundspeed', 'Thrust Reversers', 'Landing'),
                      available) and \
            any_of(('Eng (*) EPR Max', 'Eng (*) N1 Max'), available)

    def derive(self,
               gnd_spd=P('Groundspeed'),
               tr=M('Thrust Reversers'),
               eng_epr=P('Eng (*) EPR Max'),
               eng_n1=P('Eng (*) N1 Max'),
               landings=S('Landing'),
               recorded_n1=P('Eng (1) N1')):

        if eng_epr and eng_epr.frequency >= (recorded_n1.frequency if recorded_n1 else 0):
            power = eng_epr
            threshold = REVERSE_THRUST_EFFECTIVE_EPR
        else:
            power = eng_n1
            threshold = REVERSE_THRUST_EFFECTIVE_N1

        power.array = align(power, gnd_spd)
        power.frequency = gnd_spd.frequency
        tr.array = align(tr, gnd_spd)
        tr.frequency = gnd_spd.frequency
        aligned_landings = landings.get_aligned(gnd_spd)

        for landing in aligned_landings:
            # handle difference in frequencies
            high_rev = thrust_reversers_working(landing, power, tr, threshold)
            self.create_kpvs_within_slices(gnd_spd.array, high_rev, min_value)

class GroundspeedStabilizerOutOfTrimDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum Groundspeed turing takeoff roll when the stabilizer is out of trim.
    '''

    units = ut.KT

    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Groundspeed', 'Stabilizer', 'Takeoff Roll Or Rejected Takeoff', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_stabilizer_limits(model.value, series.value, family.value)
        except KeyError:
            cls.warning("No stabilizer limits available for '%s', '%s', '%s'.",
                        model.value, series.value, family.value)
            return False

        return True

    def derive(self,
               gnd_spd=P('Groundspeed'),
               stab=P('Stabilizer'),
               takeoff_roll=S('Takeoff Roll Or Rejected Takeoff'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        stab_fwd, stab_aft = at.get_stabilizer_limits(model.value, series.value, family.value)

        masked_in_trim = np.ma.masked_inside(stab.array, stab_fwd, stab_aft)
        # Masking groundspeed where stabilizer is in trim - we don't want the
        # KPV to be created when condition is not met (stabilizer out of trim)
        gspd_masked = np.ma.array(gnd_spd.array, mask=masked_in_trim.mask)
        self.create_kpvs_within_slices(gspd_masked, takeoff_roll, max_value)


class GroundspeedSpeedbrakeHandleDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum Groundspeed turing takeoff roll when the speedbrake handle is over
    limit.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed'),
               spdbrk=P('Speedbrake Handle'),
               spdbrk_selected=M('Speedbrake Selected'),
               takeoff_roll=S('Takeoff Roll Or Rejected Takeoff')):

        SPEEDBRAKE_HANDLE_LIMIT = 2.0
        masked_in_range = np.ma.masked_less_equal(spdbrk.array,
                                                  SPEEDBRAKE_HANDLE_LIMIT)
        
        #Masking only the portion of the array where the speedbrake is not deployed
        masked_in_range_not_deployed = np.ma.masked_not_equal(spdbrk_selected.array, 2)
        
        #Keeping unmasked only the portions of the array where the speedbrake handle is over
        #limit and the speedbrake is deployed
        final_mask = np.logical_and(masked_in_range, masked_in_range_not_deployed)

        # Masking groundspeed where speedbrake is within limit.
        # WARNING: in this particular case we don't want the KPV to be created
        # when the condition (speedbrake handle limit exceedance) is not met.
        gspd_masked = np.ma.array(gnd_spd.array, mask=final_mask.mask)
        self.create_kpvs_within_slices(gspd_masked, takeoff_roll, max_value)


class GroundspeedSpeedbrakeDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum Groundspeed turing takeoff roll when the speedbrake handle is over
    limit.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed'),
               spdbrk=P('Speedbrake'),
               takeoff_roll=S('Takeoff Roll Or Rejected Takeoff')):

        SPEEDBRAKE_LIMIT = 39

        masked_in_range = np.ma.masked_less_equal(spdbrk.array,
                                                  SPEEDBRAKE_LIMIT)

        # Masking groundspeed where speedbrake is within limit.
        # WARNING: in this particular case we don't want the KPV to be created
        # when the condition (speedbrake limit exceedance) is not met.
        gspd_masked = np.ma.array(gnd_spd.array, mask=masked_in_range.mask)
        self.create_kpvs_within_slices(gspd_masked, takeoff_roll, max_value)


class GroundspeedFlapChangeDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum Groundspeed turing takeoff roll when the flaps are being changed.
    limit.
    '''

    units = ut.KT

    def derive(self,
               gnd_spd=P('Groundspeed'),
               flap=M('Flap'),
               takeoff_roll=S('Takeoff Roll Or Rejected Takeoff')):

        flap_changes = np.ma.ediff1d(flap.array, to_begin=0)
        masked_in_range = np.ma.masked_equal(flap_changes, 0)
        gspd_masked = np.ma.array(gnd_spd.array, mask=masked_in_range.mask)
        self.create_kpvs_within_slices(gspd_masked, takeoff_roll, max_value)


class GroundspeedBelow15FtFor20SecMax(KeyPointValueNode):
    '''
    Maximum groundspeed below 15ft AAL recorded for at least 20 seconds 
    while airborne.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, gnd_spd=P('Groundspeed'), alt_aal=P('Altitude AAL For Flight Phases'), airborne=S('Airborne')):
        gspd_20_sec = second_window(gnd_spd.array, self.frequency, 20)
        height_bands = slices_and(airborne.get_slices(),
                                  slices_below(alt_aal.array, 15)[1])
        self.create_kpv_from_slices(gspd_20_sec, height_bands, max_value)


class GroundspeedWhileAirborneWithASEOff(KeyPointValueNode):
    '''
    Measures the maximum groundspeed during flight with stability systems disengaged.
    '''

    name = 'Groundspeed While Airborne With ASE Off'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, gnd_spd=P('Groundspeed'), ase=M('ASE Engaged'), airborne=S('Airborne')):
        sections = clump_multistate(ase.array, 'Engaged', airborne.get_slices(), False)
        self.create_kpvs_within_slices(gnd_spd.array, sections, max_value)


class GroundspeedWhileHoverTaxiingMax(KeyPointValueNode):
    '''
    Maximum groundspeed reached durng hover taxi. 
    '''

    name = 'Groundspeed While Hover Taxiing Max'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, gnd_spd=P('Groundspeed'), hover_taxi=S('Hover Taxi')):
        self.create_kpvs_within_slices(gnd_spd.array, hover_taxi.get_slices(), max_value)


class GroundspeedWithZeroAirspeedFor5SecMax(KeyPointValueNode):
    '''
    Maximum groundspeed recorded for at least 5 seconds with airspeed at 0.
    '''

    align_frequency = 2
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, wind_spd=P('Wind Speed'), wind_dir=P('Wind Direction'),
               gnd_spd=P('Groundspeed'), heading=P('Heading'),
               airborne=S('Airborne')):

        rad_scale = np.radians(1.0)
        headwind = gnd_spd.array + wind_spd.array * np.ma.cos((wind_dir.array-heading.array)*rad_scale)
        if np.ma.count(headwind):
            zero_airspeed = slices_and(airborne.get_slices(),
                                    slices_below(headwind, 0)[1])
            zero_airspeed = slices_remove_small_slices(zero_airspeed, time_limit=5,
                                                      hz=self.frequency)
            self.create_kpvs_within_slices(gnd_spd.array, zero_airspeed, max_value)


class GroundspeedBelow100FtMax(KeyPointValueNode):
    '''
    Maximum groundspeed below 100ft AGL (helicopter only)
    '''
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, gnd_spd=P('Groundspeed'), alt_agl=P('Altitude AGL For Flight Phases'),
               airborne=S('Airborne')):
        alt_slices = slices_and(airborne.get_slices(),
                                alt_agl.slices_below(100))
        self.create_kpvs_within_slices(gnd_spd.array,
                                       alt_slices,
                                       max_value)

##############################################################################
# Law


class AlternateLawDuration(KeyPointValueNode):
    '''
    The time in seconds that one or more alternate law parameters are active.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Alternate Law',
            'Pitch Alternate Law',
            'Roll Alternate Law',
        ), available) and 'Airborne' in available

    def derive(self,
               alternate_law=M('Alternate Law'),
               pitch_alternate_law=M('Pitch Alternate Law'),
               roll_alternate_law=M('Roll Alternate Law'),
               airborne=S('Airborne')):

        combined = vstack_params_where_state(
            (alternate_law, 'Engaged'),
            (pitch_alternate_law, 'Engaged'),
            (roll_alternate_law, 'Engaged'),
        ).any(axis=0)
        comb_air = mask_outside_slices(combined, airborne.get_slices())
        self.create_kpvs_from_slice_durations(runs_of_ones(comb_air), self.hz)


class DirectLawDuration(KeyPointValueNode):
    '''
    The time in seconds that one or more direct law parameters are active.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):

        return any_of((
            'Direct Law',
            'Pitch Direct Law',
            'Roll Direct Law',
        ), available) and 'Airborne' in available

    def derive(self,
               direct_law=M('Direct Law'),
               pitch_direct_law=M('Pitch Direct Law'),
               roll_direct_law=M('Roll Direct Law'),
               airborne=S('Airborne')):

        combined = vstack_params_where_state(
            (direct_law, 'Engaged'),
            (pitch_direct_law, 'Engaged'),
            (roll_direct_law, 'Engaged'),
        ).any(axis=0)
        comb_air = mask_outside_slices(combined, airborne.get_slices())
        self.create_kpvs_from_slice_durations(runs_of_ones(comb_air), self.hz)


##############################################################################
# Pitch


class PitchAfterFlapRetractionMax(KeyPointValueNode):
    '''
    Maximum pitch reached with flaps retracted.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Pitch', 'Airborne'), available)

    def derive(self,
               pitch=P('Pitch'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        scope = []
        for air in airborne:
            slices = runs_of_ones(retracted[air.slice])
            if not slices:
                continue
            scope.append(slice(air.slice.start + slices[0].start, air.slice.stop))
        self.create_kpvs_within_slices(pitch.array, scope, max_value)


class PitchAtLiftoff(KeyPointValueNode):
    '''
    Pitch at the point of liftoff.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(pitch.array, liftoffs)


class PitchAtTouchdown(KeyPointValueNode):
    '''
    Pitch at the point of touchdown.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(pitch.array, touchdowns)


class PitchAt35FtDuringClimb(KeyPointValueNode):
    '''
    Pitch at 35ft AAL during initial climb.
    '''
    can_operate = aeroplane_only

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL'),
               climbs=S('Initial Climb')):

        for climb in climbs:
            value = value_at_index(pitch.array, climb.start_edge)
            self.create_kpv(climb.start_edge, value)


class PitchAbove1000FtMin(KeyPointValueNode):
    '''
    Minimum Pitch above 1000ft AAL in flight.
    '''
    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt.slices_above(1000), min_value)


class PitchAbove1000FtMax(KeyPointValueNode):
    '''
    Maximum Pitch above 1000ft AAL in flight.
    '''
    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt=P('Altitude AAL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt.slices_above(1000), max_value)


class PitchBelow1000FtMax(KeyPointValueNode):
    '''
    Maximum Pitch below 1000ft AGL in flight (helicopter_only).
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt=P('Altitude AGL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt.slices_below(1000), max_value)


class PitchBelow1000FtMin(KeyPointValueNode):
    '''
    Minimum Pitch below 1000ft AGL in flight (helicopter_only).
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt=P('Altitude AGL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt.slices_below(1000), min_value)


class PitchBelow5FtMax(KeyPointValueNode):
    '''
    Maximum Pitch below 5ft AGL in flight (helicopter_only). 
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt_agl=P('Altitude AGL'),
               airborne=S('Airborne')):
        slices = slices_and(airborne.get_slices(), alt_agl.slices_below(5))
        self.create_kpvs_within_slices(pitch.array, slices, max_value)


class Pitch5To10FtMax(KeyPointValueNode):
    '''
    Maximum Pitch ascending 5 to 10ft AGL in flight (helicopter_only). 
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt_agl=P('Altitude AGL'),
               airborne=S('Airborne')):
        slices = slices_and(airborne.get_slices(),
                            alt_agl.slices_from_to(5, 10))
        self.create_kpvs_within_slices(pitch.array, slices, max_value)


class Pitch10To5FtMax(KeyPointValueNode):
    '''
    Maximum Pitch descending 10 to 5ft AGL in flight (helicopter_only). 
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt_agl=P('Altitude AGL'),
               airborne=S('Airborne')):
        slices = slices_and(airborne.get_slices(),
                            alt_agl.slices_from_to(10, 5))
        self.create_kpvs_within_slices(pitch.array, slices, max_value)


class PitchTakeoffMax(KeyPointValueNode):
    '''
    Maximum pitch during takeoff phase of flight.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               takeoffs=S('Takeoff')):

        self.create_kpvs_within_slices(pitch.array, takeoffs, max_value)


class Pitch35ToClimbAccelerationStartMin(KeyPointValueNode):
    '''
    Minimum pitch between 35ft AAL and climb acceleration start altitude.
    
    Will use Climb Acceleration Start if available, otherwise we
    fallback to 1000ft AAL (end of initial climb)
    '''

    can_operate = aeroplane_only

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               climbs=S('Initial Climb'),
               climb_accel_start=KTI('Climb Acceleration Start')):
        init_climb = climbs.get_first()
        if init_climb:
            if len(climb_accel_start):
                _slice = slice(init_climb.slice.start,
                               climb_accel_start.get_first().index + 1)
            else:
                _slice = init_climb.slice
    
            self.create_kpvs_within_slices(pitch.array, (_slice,), min_value)


class Pitch35ToClimbAccelerationStartMax(KeyPointValueNode):
    '''
    Maximum pitch between 35ft AAL and climb acceleration start altitude.
    
    Will use Climb Acceleration Start if available, otherwise we
    fallback to 1000ft AAL (end of initial climb)
    '''

    can_operate = aeroplane_only

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               climbs=S('Initial Climb'),
               climb_accel_start=KTI('Climb Acceleration Start')):
        init_climb = climbs.get_first()
        if init_climb:
            if len(climb_accel_start):
                _slice = slice(init_climb.slice.start,
                               climb_accel_start.get_first().index + 1)
            else:
                _slice = init_climb.slice
    
            self.create_kpvs_within_slices(pitch.array, (_slice,), max_value)


class Pitch35To400FtMax(KeyPointValueNode):
    '''
    Maximum pitch between 35ft and 400ft AAL during initial climb.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               climbs=S('Initial Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 35, 400)
        alt_climb_sections = valid_slices_within_array(alt_band, climbs)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_climb_sections,
            max_value,
        )


class Pitch35To400FtMin(KeyPointValueNode):
    '''
    Minimum pitch between 35ft and 400ft AAL during initial climb.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               climbs=S('Initial Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 35, 400)
        alt_climb_sections = valid_slices_within_array(alt_band, climbs)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_climb_sections,
            min_value,
        )


class Pitch400To1000FtMax(KeyPointValueNode):
    '''
    Maximum pitch between 400ft and 1000ft AAL during initial climb.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               climbs=S('Initial Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 400, 1000)
        alt_climb_sections = valid_slices_within_array(alt_band, climbs)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_climb_sections,
            max_value,
        )


class Pitch400To1000FtMin(KeyPointValueNode):
    '''
    Minimum pitch between 400ft and 1000ft AAL during initial climb.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               climbs=S('Initial Climb')):

        alt_band = np.ma.masked_outside(alt_aal.array, 400, 1000)
        alt_climb_sections = valid_slices_within_array(alt_band, climbs)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_climb_sections,
            min_value,
        )


class Pitch1000To500FtMax(KeyPointValueNode):
    '''
    Maximum pitch 1000ft to 500ft AAL (AGL for helicopters). 
    Applies to final approach phase for fixed wing aircraft and 
    descent for helicopters.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Pitch']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL For Flight Phases', 'Descent'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Final Approach'])
        return all_of(required, available)

    def derive(self,
               pitch=P('Pitch'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               fin_app=S('Final Approach'),
               # helicopter
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_band = np.ma.masked_outside(alt_agl.array, 1000, 500)
            alt_app_sections = valid_slices_within_array(alt_band, descending)
            self.create_kpvs_within_slices(
                pitch.array,
                alt_app_sections,
                max_value,
                min_duration=HOVER_MIN_DURATION,
                freq=pitch.frequency)
        else:
            alt_band = np.ma.masked_outside(alt_aal.array, 1000, 500)
            alt_app_sections = valid_slices_within_array(alt_band, fin_app)
            self.create_kpvs_within_slices(
                pitch.array,
                alt_app_sections,
                max_value)


class Pitch1000To500FtMin(KeyPointValueNode):
    '''
    Minimum pitch 1000ft to 500ft AAL (AGL for helicopters). 
    Applies to final approach phase for fixed wing aircraft and 
    descent for helicopters.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Pitch']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descent'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Final Approach'])
        return all_of(required, available)

    def derive(self,
               pitch=P('Pitch'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               fin_app=S('Final Approach'),
               # helicopter
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_band = np.ma.masked_outside(alt_agl.array, 1000, 500)
            alt_app_sections = valid_slices_within_array(alt_band, descending)
            self.create_kpvs_within_slices(
                pitch.array,
                alt_app_sections,
                min_value,
                min_duration=HOVER_MIN_DURATION,
                freq=pitch.frequency)
        else:
            alt_band = np.ma.masked_outside(alt_aal.array, 1000, 500)
            alt_app_sections = valid_slices_within_array(alt_band, fin_app)
            self.create_kpvs_within_slices(
                pitch.array,
                alt_app_sections,
                min_value)


class Pitch500To50FtMax(KeyPointValueNode):
    '''
    Maximum pitch 500ft to 50ft AAL (AGL for helicopters). 
    Applies to final approach phase for fixed wing aircraft and 
    descent for helicopters.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Pitch']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Final Approach'])
        return all_of(required, available)

    def derive(self,
               pitch=P('Pitch'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               fin_app=S('Final Approach'),
               # helicopter
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_band = np.ma.masked_outside(alt_agl.array, 50, 500)
            alt_app_sections = valid_slices_within_array(alt_band, descending)
            self.create_kpvs_within_slices(
                pitch.array,
                alt_app_sections,
                max_value,
                min_duration=HOVER_MIN_DURATION,
                freq=pitch.frequency)
        else:
            alt_band = np.ma.masked_outside(alt_aal.array, 500, 50)
            alt_app_sections = valid_slices_within_array(alt_band, fin_app)
            self.create_kpvs_within_slices(
                pitch.array,
                alt_app_sections,
                max_value)


class Pitch500To50FtMin(KeyPointValueNode):
    '''
    Minimum pitch 500ft to 50ft AAL (AGL for helicopters). 
    Applies to final approach phase for fixed wing aircraft and 
    descent for helicopters.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Pitch']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Final Approach'])
        return all_of(required, available)

    def derive(self,
               pitch=P('Pitch'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               fin_app=S('Final Approach'),
               # helicopter
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descending'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            alt_band = np.ma.masked_outside(alt_agl.array, 50, 500)
            alt_app_sections = valid_slices_within_array(alt_band, descending)
            self.create_kpvs_within_slices(
                pitch.array,
                alt_app_sections,
                min_value,
                min_duration=HOVER_MIN_DURATION,
                freq=pitch.frequency)
        else:
            alt_band = np.ma.masked_outside(alt_aal.array, 500, 50)
            alt_app_sections = valid_slices_within_array(alt_band, fin_app)
            self.create_kpvs_within_slices(
                pitch.array,
                alt_app_sections,
                min_value)


class Pitch500To100FtMax(KeyPointValueNode):
    '''
    Maximum pitch 500ft to 100ft AGL. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 500)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_app_sections,
            max_value,
            min_duration=HOVER_MIN_DURATION,
            freq=pitch.frequency)


class Pitch500To100FtMin(KeyPointValueNode):
    '''
    Minimum pitch 500ft to 100ft AGL. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 500)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_app_sections,
            min_value,
            min_duration=HOVER_MIN_DURATION,
            freq=pitch.frequency)


class Pitch500To20FtMin(KeyPointValueNode):
    '''
    Minimum pitch 500ft to 20ft AAL.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class Pitch500To7FtMax(KeyPointValueNode):
    '''
    Maximum pitch 500ft to 7ft AAL. 
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(500, 7),
            max_value,
        )


class Pitch500To7FtMin(KeyPointValueNode):
    '''
    Minimum pitch 500ft to 7ft AAL. 
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(500, 7),
            min_value,
        )


class Pitch100To20FtMax(KeyPointValueNode):
    '''
    Maximum pitch 100ft to 20ft AAL. 
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 20, 100)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_app_sections,
            max_value,
            min_duration=HOVER_MIN_DURATION, # TODO: check where this came from.
            freq=pitch.frequency)


class Pitch100To20FtMin(KeyPointValueNode):
    '''
    Minimum pitch 100ft to 20ft AGL. Helicopter only. 
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 20, 100)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_app_sections,
            min_value,
            min_duration=HOVER_MIN_DURATION, # TODO: check where this came from.
            freq=pitch.frequency)


class Pitch50FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum pitch 50ft AAL (AGL for helicopters) to touchdown.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Pitch', 'Touchdown']
        if ac_type and ac_type.value == 'helicopter':
            required.append('Altitude AGL')
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               pitch=P('Pitch'),
               touchdowns=KTI('Touchdown'),
               ac_type=A('Aircraft Type'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL')):

        self.create_kpvs_within_slices(
            pitch.array,
            (alt_aal or alt_agl).slices_to_kti(50, touchdowns),
            max_value)


class Pitch50FtToTouchdownMin(KeyPointValueNode):
    '''
    Minimum pitch 50ft AGL to touchdown. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_agl.slices_to_kti(50, touchdowns),
            min_value,
        )


class Pitch20FtToTouchdownMin(KeyPointValueNode):
    '''
    Minimum pitch 20ft AAL (AGL for helicopters) to touchdown.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Pitch']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Touchdown'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Touchdown'])
        return all_of(required, available)

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_agl=P('Altitude AGL'),
               touchdowns=KTI('Touchdown'),
               ac_type=A('Aircraft Type')):

        alt = alt_aal
        if ac_type and ac_type.value == 'helicopter':
            alt = alt_agl

        self.create_kpvs_within_slices(
            pitch.array,
            alt.slices_to_kti(20, touchdowns),
            min_value,
        )


class Pitch20FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum pitch 20ft AAL (AGL for helicopters) to touchdown.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Pitch']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Touchdown'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Touchdown'])
        return all_of(required, available)

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_agl=P('Altitude AGL'),
               touchdowns=KTI('Touchdown'),
               ac_type=A('Aircraft Type')):

        alt = alt_aal
        if ac_type and ac_type.value == 'helicopter':
            alt = alt_agl

        self.create_kpvs_within_slices(
            pitch.array,
            alt.slices_to_kti(20, touchdowns),
            max_value,
        )


class Pitch7FtToTouchdownMin(KeyPointValueNode):
    '''
    Minimum pitch 7ft AAL to touchdown.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_to_kti(7, touchdowns),
            min_value,
        )


class Pitch7FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum pitch 7ft AAL to touchdown.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_to_kti(7, touchdowns),
            max_value,
        )


class PitchCyclesDuringFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of pitch attitude that exceed 3 deg in
    pitch from peak to peak and with a maximum cycle period of 10 seconds
    during the final approach phase.
    '''

    units = ut.CYCLES

    def derive(self,
               pitch=P('Pitch'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                pitch.array[fin_app.slice],
                3.0, 10.0, pitch.hz,
                fin_app.slice.start,
            ))


class PitchDuringGoAroundMax(KeyPointValueNode):
    '''
    Maximum pitch angle reached during go around.
    '''

    units = ut.DEGREE

    def derive(self,
               pitch=P('Pitch'),
               go_arounds=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(pitch.array, go_arounds, max_value)


class PitchOnGroundMax(KeyPointValueNode):
    '''
    Pitch attitude maximum to check for sloping ground operation.
    '''
    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, pitch=P('Pitch'), coll=P('Collective'),
               grounded=S('Grounded'), on_deck=S('On Deck')):
        '''
        The collective parameter ensures this is not the attitude during liftoff.
        '''
        my_slices = slices_and_not(grounded.get_slices(), on_deck.get_slices())
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(my_slices, low_coll)
        self.create_kpvs_within_slices(pitch.array,
                                       my_slices,
                                       max_value)


class PitchOnDeckMax(KeyPointValueNode):
    '''
    Pitch attitude maximum during operation on a moving deck.
    '''
    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, pitch=P('Pitch'), coll=P('Collective'), on_deck=S('On Deck')):
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(on_deck.get_slices(), low_coll)
        self.create_kpvs_within_slices(pitch.array,
                                       my_slices,
                                       max_value)


class PitchOnGroundMin(KeyPointValueNode):
    '''
    Minimum pitch on the ground (or deck).
    '''
    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, pitch=P('Pitch'), coll=P('Collective'), grounded=S('Grounded'), on_deck=S('On Deck')):
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(on_deck.get_slices(), low_coll)
        my_slices = slices_and_not(grounded.get_slices(), my_slices)
        self.create_kpvs_within_slices(pitch.array,
                                       my_slices,
                                       min_value)


class PitchOnDeckMin(KeyPointValueNode):
    '''
    Minimum pitch while on deck.
    '''
    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, pitch=P('Pitch'), coll=P('Collective'), on_deck=S('On Deck')):
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(on_deck.get_slices(), low_coll)
        self.create_kpvs_within_slices(pitch.array,
                                       my_slices,
                                       min_value)


class PitchWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum pitch while airborne.
    '''

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(pitch.array, airborne, max_value)


class PitchWhileAirborneMin(KeyPointValueNode):
    '''
    Minimum pitch while airborne.
    '''

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(pitch.array, airborne, min_value)


class PitchTouchdownTo60KtsAirspeedMax(KeyPointValueNode):
    '''
    Maximum pitch recorded between the point of touchdown and the point
    where aircraft reached 60kts IAS during landing rollout.
    '''

    units = ut.DEGREE

    can_operate = aeroplane_only
    
    def derive(self,
               pitch=P('Pitch'),
               airspeed=P('Airspeed'),
               touchdowns=KTI('Touchdown')):
        
        for tdwn in touchdowns:
            stop = index_at_value(airspeed.array.data, 60,
                                  slice(tdwn.index, None), endpoint='nearest')
            self.create_kpv(*max_value(pitch.array,
                                       slice(tdwn.index, stop)))


class PitchAboveFL200Max(KeyPointValueNode):
    '''
    Maximum pitch angle above FL200 (Alt STD Smoothed)
    '''
    name = 'Pitch Above FL200 Max'
    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt=P('Altitude STD Smoothed')):
        
        alt.array = nearest_neighbour_mask_repair(alt.array)
        self.create_kpvs_within_slices(pitch.array,
                                       alt.slices_above(20000), max_value)        
    
class PitchAboveFL200Min(KeyPointValueNode):
    '''
    Minimum pitch angle above FL200 (Alt STD Smoothed)
    '''
    name = 'Pitch Above FL200 Min'
    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt=P('Altitude STD Smoothed')):
        
        alt.array = nearest_neighbour_mask_repair(alt.array)
        self.create_kpvs_within_slices(pitch.array,
                                       alt.slices_above(20000), min_value)    
    
##############################################################################
# Pitch Rate
class PitchRateWhileAirborneMax(KeyPointValueNode):
    '''
    Maximum pitch rate while airborne.
    '''

    units = ut.DEGREE_S

    def derive(self, pitch_rate=P('Pitch Rate'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(pitch_rate.array, airborne, max_abs_value)


class PitchRate35To1000FtMax(KeyPointValueNode):
    '''
    Maximum pitch rate between 35ft and 1000ft AAL.
    '''

    units = ut.DEGREE_S

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class PitchRate35ToClimbAccelerationStartMax(KeyPointValueNode):
    '''
    Maximum pitch rate between 35ft AAL and climb acceleration start altitude.
    
    If Climb Acceleration Start is not available it will fallback to 1000ft AAL.
    '''

    can_operate = aeroplane_only

    units = ut.DEGREE_S

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               climbs=S('Initial Climb'),
               climb_accel_start=KTI('Climb Acceleration Start')):

        init_climb = climbs.get_first()
        if init_climb:
            if len(climb_accel_start):
                _slice = slice(init_climb.slice.start,
                               climb_accel_start.get_first().index + 1)
            else:
                _slice = init_climb.slice
    
            self.create_kpvs_within_slices(pitch_rate.array, (_slice,), max_value)


class PitchRate20FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum pitch rate 20ft AAL to touchdown.
    '''

    units = ut.DEGREE_S

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_to_kti(20, touchdowns),
            max_value,
        )


class PitchRate20FtToTouchdownMin(KeyPointValueNode):
    '''
    Minimum pitch rate 20ft AAL to touchdown.
    '''

    units = ut.DEGREE_S

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_to_kti(20, touchdowns),
            min_value,
        )


class PitchRate2DegPitchTo35FtMax(KeyPointValueNode):
    '''
    Maximum recorded pitch rate between the point where pitch 
    reaches 2deg and aircraft crosses 35ft AAL.
    '''

    units = ut.DEGREE_S

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               two_deg_pitch_to_35ft=S('2 Deg Pitch To 35 Ft')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            two_deg_pitch_to_35ft,
            max_value,
        )


class PitchRate2DegPitchTo35FtMin(KeyPointValueNode):
    '''
    Minimum recorded pitch rate between the point where pitch 
    reaches 2deg and aircraft crosses 35ft AAL.
    '''

    units = ut.DEGREE_S

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               two_deg_pitch_to_35ft=S('2 Deg Pitch To 35 Ft')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            two_deg_pitch_to_35ft,
            min_value,
        )



class PitchRateTouchdownTo60KtsAirspeedMax(KeyPointValueNode):
    '''
    Maximum pitch rate at point of touchdown until airspeed reaches 60 kts.
    '''

    units = ut.DEGREE_S

    can_operate = aeroplane_only
    
    def derive(self, pitchrate=P('Pitch Rate'), airspeed=P('Airspeed'),
               touchdowns=KTI('Touchdown')):
        for tdwn in touchdowns:
            stop = index_at_value(airspeed.array.data, 60, 
                                  slice(tdwn.index, None), endpoint='nearest')
            self.create_kpv(*max_value(pitchrate.array,
                                       slice(tdwn.index, stop)))


##############################################################################
# Vertical Speed (Rate of Climb/Descent) Helpers


def vert_spd_phase_max_or_min(obj, vrt_spd, phases, function):
    '''
    Vertical Speed (Rate of Climb/Descent) Helper
    '''
    for phase in phases:
        duration = phase.slice.stop - phase.slice.start
        if duration > CLIMB_OR_DESCENT_MIN_DURATION:
            index, value = function(vrt_spd.array, phase.slice)
            obj.create_kpv(index, value)


##############################################################################
# Rate of Climb


class RateOfClimbMax(KeyPointValueNode):
    '''
    Maximum recorded rate of climb.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               climbing=S('Climbing')):
        '''
        In cases where the aircraft does not leave the ground, we get a descending
        phase that equates to an empty list, which is not iterable.
        '''        
        vrt_spd.array[vrt_spd.array < 0] = np.ma.masked
        vert_spd_phase_max_or_min(self, vrt_spd, climbing, max_value)


class RateOfClimb35ToClimbAccelerationStartMin(KeyPointValueNode):
    '''
    Minimum recorded rate of climb between 35ft AAL and the altitude of
    climb acceleration start.
    
    Will use Climb Acceleration Start if available, otherwise we
    fallback to 1000ft
    Note: The minimum Rate of Climb could be negative in this phase.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               climbs=S('Initial Climb'),
               climb_accel_start=KTI('Climb Acceleration Start')):
        init_climb = climbs.get_first()
        if init_climb:
            if len(climb_accel_start):
                _slice = slice(init_climb.slice.start,
                               climb_accel_start.get_first().index + 1)
            else:
                _slice = init_climb.slice
    
            self.create_kpvs_within_slices(vrt_spd.array, (_slice,), min_value)


class RateOfClimb35To1000FtMin(KeyPointValueNode):
    '''
    Minimum recorded rate of climb between 35ft and 1000ft AAL.
    Note: The minimum Rate of Climb could be negative in this phase.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               climbs=S('Initial Climb')):
        self.create_kpvs_within_slices(vrt_spd.array, climbs, min_value)


class RateOfClimbBelow10000FtMax(KeyPointValueNode):
    '''
    Maximum recorded rate of climb below 10000ft STD.
    '''

    # Q: Should this exclude go-around and climb out as defined below?

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude STD Smoothed'),
               airborne=S('Airborne')):
        vrt_spd.array[vrt_spd.array < 0] = np.ma.masked
        self.create_kpv_from_slices(
            vrt_spd.array,
            slices_and(alt_aal.slices_from_to(0, 10000),
                       [s.slice for s in airborne]),
            max_value,
        )


class RateOfClimbDuringGoAroundMax(KeyPointValueNode):
    '''
    Maximum recorded rate of climb during go around.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               go_arounds=S('Go Around And Climbout')):
        vrt_spd.array[vrt_spd.array < 0] = np.ma.masked
        self.create_kpvs_within_slices(vrt_spd.array, go_arounds, max_value)


class RateOfClimbAtHeightBeforeLevelFlight(KeyPointValueNode):
    '''
    Rate of climb at various altitudes before level off.
    Uses altitude STD smoothed.
    '''

    NAME_FORMAT = 'Rate Of Climb At %(altitude)d Ft Before Level Off'
    NAME_VALUES = {'altitude': [2000, 1000]}

    units = ut.FPM

    def derive(self, vert_spd=P('Vertical Speed'),
               heights=KTI('Altitude Before Level Flight When Climbing')):

        # TODO: Mask vert spd below 0?
        for altitude in self.NAME_VALUES['altitude']:
            ktis = heights.get(name='%d Ft Before Level Flight Climbing'
                               % altitude)
            for kti in ktis:
                value = value_at_index(vert_spd.array, kti.index)
                self.create_kpv(kti.index, value,
                                replace_values={'altitude': altitude})


##############################################################################
# Rate of Descent


class RateOfDescentMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent).
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               descending=S('Descending')):
        vrt_spd.array[vrt_spd.array > 0] = np.ma.masked
        vert_spd_phase_max_or_min(self, vrt_spd, descending, min_value)


class RateOfDescentTopOfDescentTo10000FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    the point of TOD and 10000ft STD Smoothed.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude STD Smoothed'),
               descents=S('Descent')):

        alt_band = np.ma.masked_less(alt_aal.array, 10000)
        alt_descent_sections = valid_slices_within_array(alt_band, descents)
        self.create_kpvs_within_slices(vrt_spd.array, alt_descent_sections, min_value)


class RateOfDescentBelow10000FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) below
    10000ft STD Smoothed.

    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Airborne Conflict (Mid-Air Collision) Excessive rates of climb/descent
    (>3,000FPM) within a TMA (defined as < 10,000ft)"
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_std=P('Altitude STD Smoothed'),
               descents=S('Descent')):
        alt_band = np.ma.masked_outside(alt_std.array, 0, 10000)
        alt_descent_sections = valid_slices_within_array(alt_band, descents)
        self.create_kpv_from_slices(
            vrt_spd.array,
            alt_descent_sections,
            min_value
        )


class RateOfDescent10000To5000FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    10000ft and 5000ft STD Smoothed.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_std=P('Altitude STD Smoothed'),
               descent=S('Descent')):

        alt_band = np.ma.masked_outside(alt_std.array, 10000, 5000)
        alt_descent_sections = valid_slices_within_array(alt_band, descent)
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_descent_sections,
            min_value
        )


class RateOfDescent5000To3000FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    5000ft and 3000ft AAL.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               descent=S('Descent')):

        alt_band = np.ma.masked_outside(alt_aal.array, 5000, 3000)
        # maximum RoD must be a big negative value; mask all positives
        alt_band[vrt_spd.array > 0] = np.ma.masked
        alt_descent_sections = valid_slices_within_array(alt_band, descent)
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_descent_sections,
            min_value
        )

class RateOfDescentAbove3000FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) above
    3000ft AAL during Descent phase.
    '''

    units = ut.FPM
    
    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               descents=S('Descent')):
        
        alt_band = np.ma.masked_inside(alt_aal.array, 0, 3000)
        alt_descent_sections = valid_slices_within_array(alt_band, descents)
        
        self.create_kpv_from_slices(
            vrt_spd.array,
            alt_descent_sections,
            min_value
        )


class RateOfDescent3000To2000FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    3000ft and 2000ft AAL.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(3000, 2000),
            min_value
        )


class RateOfDescent2000To1000FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    2000ft and 1000ft AAL.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(2000, 1000),
            min_value
        )


class RateOfDescent1000To500FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    1000ft and 500ft AAL.
    '''

    units = ut.FPM

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Vertical Speed']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descent'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Final Approach'])
        return all_of(required, available)

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               ac_type=A('Aircraft Type'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               fin_app=S('Final Approach'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               descending=S('Descent')):

        alt_band = np.ma.masked_outside((alt_aal or alt_agl).array, 1000, 500)
        # maximum RoD must be a big negative value; mask all positives
        alt_band[vrt_spd.array > 0] = np.ma.masked
        alt_app_sections = valid_slices_within_array(alt_band, fin_app or descending)
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_app_sections,
            min_value,
            min_duration=5.0,
            freq=vrt_spd.frequency)


class RateOfDescent100To20FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    100ft and 20ft AAL.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               alt_agl=P('Altitude AGL'),
               descending=S('Descent')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 20)
        # maximum RoD must be a big negative value; mask all positives
        alt_band[vrt_spd.array > 0] = np.ma.masked
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_app_sections,
            min_value,
            min_duration=5.0,
            freq=vrt_spd.frequency)


class RateOfDescent500To100FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    500ft and 100ft AAL.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               alt_agl=P('Altitude AGL'),
               descending=S('Descent')):

        alt_band = np.ma.masked_outside(alt_agl.array, 500, 100)
        # maximum RoD must be a big negative value; mask all positives
        alt_band[vrt_spd.array > 0] = np.ma.masked
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_app_sections,
            min_value,
            min_duration=5.0,
            freq=vrt_spd.frequency)


class RateOfDescent500To50FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    500ft and 50ft AAL.
    '''

    units = ut.FPM

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Vertical Speed']
        if ac_type and ac_type.value == 'helicopter':
            required.extend(['Altitude AGL', 'Descending'])
        else:
            required.extend(['Altitude AAL For Flight Phases', 'Final Approach'])
        return all_of(required, available)

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               ac_type=A('Aircraft Type'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               fin_app=S('Final Approach'),
               # helicopter
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descending')):

        alt_band = np.ma.masked_outside((alt_aal or alt_agl).array, 500, 50)
        # maximum RoD must be a big negative value; mask all positives
        alt_band[vrt_spd.array > 0] = np.ma.masked
        alt_app_sections = valid_slices_within_array(alt_band, fin_app or descending)
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_app_sections,
            min_value,
            min_duration=5.0,
            freq=vrt_spd.frequency)


class RateOfDescent20FtToTouchdownMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between 20ft
    and touchdown. Uses altitude AGL. Helicopter only.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               touchdowns=KTI('Touchdown'),
               alt_agl=P('Altitude AGL')):
        '''
        The ground effect compressibility makes the normal pressure altitude
        based vertical speed meaningless, so we use the more complex inertial
        computation to give accurate measurements within ground effect.
        '''
        # maximum RoD must be a big negative value; mask all positives
        vrt_spd.array[vrt_spd.array > 0] = np.ma.masked
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_agl.slices_to_kti(20, touchdowns),
            min_value,
        )


class RateOfDescent50FtToTouchdownMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between 20ft
    and touchdown. Uses altitude AGL/AAL.
    '''

    units = ut.FPM

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Vertical Speed Inertial', 'Touchdown']
        if ac_type and ac_type.value == 'helicopter':
            required.append('Altitude AGL')
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               touchdowns=KTI('Touchdown'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL')):
        # maximum RoD must be a big negative value; mask all positives
        vrt_spd.array[vrt_spd.array > 0] = np.ma.masked
        self.create_kpvs_within_slices(
            vrt_spd.array,
            (alt_aal or alt_agl).slices_to_kti(50, touchdowns),
            min_value,
        )


class RateOfDescentAtTouchdown(KeyPointValueNode):
    '''
    Rate of descent recorded at touchdown.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               tdns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(vrt_spd.array, tdns)


class RateOfDescentDuringGoAroundMax(KeyPointValueNode):
    '''
    Maximum recorded rate of descent during a go around.
    '''

    units = ut.FPM

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               go_arounds=S('Go Around And Climbout')):
        vrt_spd.array[vrt_spd.array > 0] = np.ma.masked
        self.create_kpvs_within_slices(vrt_spd.array, go_arounds, min_value)


class RateOfDescentBelow80KtsMax(KeyPointValueNode):
    '''
    Returns the highest single rate of descent for all periods below 80kts on a single descent.

    This avoids multiple triggers on descents flown around 80kts.
    '''

    units = ut.FPM

    def derive(self, vrt_spd=P('Vertical Speed'), air_spd=P('Airspeed'), descending=S('Descending')):
        # minimum RoD must be a small negative value; mask all positives
        vrt_spd.array[vrt_spd.array > 0] = np.ma.masked
        for descent in descending:
            to_scan = air_spd.array[descent.slice]
            if np.ma.count(to_scan):
                slow_bands = shift_slices(
                    slices_remove_small_slices(
                        slices_below(to_scan, 80)[1], time_limit=5.0, hz=air_spd.frequency
                    ), descent.slice.start)
                self.create_kpv_from_slices(vrt_spd.array, slow_bands, min_value)


class RateOfDescentBelow500FtMax(KeyPointValueNode):
    '''
    Returns the highest single rate of descent for all periods below 500ft AGL.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descending')):
        height_bands = slices_and(descending.get_slices(),
                                  slices_below(alt_agl.array, 500)[1])
        self.create_kpvs_within_slices(vrt_spd.array, height_bands, min_value,
            min_duration=HOVER_MIN_DURATION, freq=vrt_spd.frequency)



class RateOfDescentBelow30KtsWithPowerOnMax(KeyPointValueNode):
    '''
    Rate of descent below 30kts IAS with engine power on max.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self, vrt_spd=P('Vertical Speed Inertial'), air_spd=P('Airspeed'), descending=S('Descending'),
               power=P('Eng (*) Torque Avg')):
        speed_bands = slices_and(descending.get_slices(),
                                  slices_below(air_spd.array, 30)[1])
        speed_bands = slices_and(speed_bands,
                                 slices_above(power.array, 20.0)[1])
        self.create_kpvs_within_slices(vrt_spd.array, speed_bands, min_value)


class RateOfDescentAtHeightBeforeLevelFlight(KeyPointValueNode):
    '''
    Rate of descent at various altitudes before level off.
    '''

    NAME_FORMAT = 'Rate Of Descent At %(altitude)d Ft Before Level Off'
    NAME_VALUES = {'altitude': [2000, 1000]}

    units = ut.FPM

    def derive(self, vert_spd=P('Vertical Speed'),
               heights=KTI('Altitude Before Level Flight When Descending')):

        # TODO: Mask vert spd above 0?
        for altitude in self.NAME_VALUES['altitude']:
            ktis = heights.get(name='%d Ft Before Level Flight Descending'
                               % altitude)
            for kti in ktis:
                value = value_at_index(vert_spd.array, kti.index)
                self.create_kpv(kti.index, value,
                                replace_values={'altitude': altitude})


class VerticalSpeedAtAltitude(KeyPointValueNode):
    '''
    Approach vertical speed at 500 and 300 Ft
    '''
    NAME_FORMAT = 'Vertical Speed At %(altitude)d Ft'
    NAME_VALUES = {'altitude': [500, 300]}
    units = ut.FPM
    can_operate = helicopter_only

    def derive(self, vert_spd=P('Vertical Speed'), alt_agl=P('Altitude AGL'),
               approaches=S('Approach')):
        for approach in approaches:
            for altitude in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_agl.array, altitude,
                                       approach.slice, 'nearest')
                if not index:
                    continue
                value = value_at_index(vert_spd.array, index)
                if value:
                    self.create_kpv(index, value, altitude=altitude)


##############################################################################
# Roll


class RollLiftoffTo20FtMax(KeyPointValueNode):
    '''
    Roll attitude extremes just after liftoff.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               airs=S('Airborne')):
        '''
        Airborne term included to ensure aircraft with poor altimetry do not spawn lots of KPVs.
        '''
        my_slices = slices_and(alt_aal.slices_from_to(1, 20), airs.get_slices())
        self.create_kpvs_within_slices(
            roll.array,
            my_slices,
            max_abs_value,
        )


class Roll20To400FtMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle between 20ft and 400ft AAL.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(20, 400),
            max_abs_value,
        )


class Roll400To1000FtMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle between 400ft and 1000ft AAL.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(400, 1000),
            max_abs_value
        )
        

class RollAbove1000FtMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle above 1000ft AAL.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_above(1000),
            max_abs_value,
        )


class Roll1000To300FtMax(KeyPointValueNode):
    '''
    Maximum roll between 1000ft and 300ft AAL during final approach.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fin_app=S('Final Approach')):

        alt_band = np.ma.masked_outside(alt_aal.array, 1000, 300)
        alt_app_sections = valid_slices_within_array(alt_band, fin_app)
        self.create_kpvs_within_slices(
            roll.array,
            alt_app_sections,
            max_abs_value,
        )


class Roll1000To500FtMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle between 1000ft and 500ft AAL.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fin_app=S('Final Approach')):

        alt_band = np.ma.masked_outside(alt_aal.array, 1000, 500)
        alt_app_sections = valid_slices_within_array(alt_band, fin_app)
        self.create_kpvs_within_slices(
            roll.array,
            alt_app_sections,
            max_abs_value,
        )


class Roll300To20FtMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle between 300ft and 20ft AAL.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(300, 20),
            max_abs_value,
        )


class Roll100To20FtMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle between 100ft and 20ft AAL.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases'), descending=S('Descent')):
        alt_band = np.ma.masked_outside(alt_agl.array, 100, 20)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            roll.array,
            alt_app_sections,
            max_abs_value,
            min_duration=HOVER_MIN_DURATION,
            freq=roll.frequency,
        )


class Roll20FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle between 20ft AAL and 
    the point of touchdown.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Roll', 'Touchdown']
        if ac_type and ac_type.value == 'helicopter':
            required.append('Altitude AGL')
        else:
            required.append('Altitude AAL For Flight Phases')
        return all_of(required, available)

    def derive(self,
               roll=P('Roll'),
               touchdowns=KTI('Touchdown'),
               # aeroplane
               alt_aal=P('Altitude AAL For Flight Phases'),
               # helicopter
               alt_agl=P('Altitude AGL')):

        self.create_kpvs_within_slices(
            roll.array,
            (alt_aal or alt_agl).slices_to_kti(20, touchdowns),
            max_abs_value,
        )


class Roll500FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle between 500ft AAL and 
    the point of touchdown.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_to_kti(500, touchdowns),
            max_abs_value,
        )


class RollAbove300FtMax(KeyPointValueNode):
    '''
    Maximum Roll above 300ft AGL in flight (helicopter_only).
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases')):
        _, height_bands = slices_above(alt_agl.array, 300)
        self.create_kpvs_within_slices(roll.array, height_bands, max_abs_value)


class RollBelow300FtMax(KeyPointValueNode):
    '''
    Maximum Roll below 300ft AGL in flight (helicopter_only).
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases'),
               airborne=S('Airborne')):
        alt_slices = slices_and(airborne.get_slices(),
                                slices_below(alt_agl.array, 300)[1])
        self.create_kpvs_within_slices(roll.array, alt_slices, max_abs_value)


class RollWithAFCSDisengagedMax(KeyPointValueNode):
    '''
    Maximum roll whilst AFCS 1 and AFCS 2 are disengaged.
    '''
    units = ut.DEGREE
    name = 'Roll With AFCS Disengaged Max'
    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), afcs1=M('AFCS (1) Engaged'),
               afcs2=M('AFCS (2) Engaged')):
        afcs = vstack_params_where_state((afcs1, 'Engaged'),
                                         (afcs2, 'Engaged')).any(axis=0)

        afcs_slices = np.ma.clump_unmasked(np.ma.masked_equal(afcs, 1))
        self.create_kpvs_within_slices(roll.array, afcs_slices, max_abs_value)


class RollAbove500FtMax(KeyPointValueNode):
    '''
    Maximum Roll above 500ft AGL in flight (helicopter only).
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases')):
        height_bands = slices_above(alt_agl.array, 500)[1]
        self.create_kpvs_within_slices(roll.array, height_bands, max_abs_value)


class RollBelow500FtMax(KeyPointValueNode):
    '''
    Maximum Roll below 500ft AGL in flight (helicopter only).
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases')):
        height_bands = slices_below(alt_agl.array, 500)[1]
        self.create_kpvs_within_slices(roll.array, height_bands, max_abs_value)


class RollOnGroundMax(KeyPointValueNode):
    '''
    Roll attitude on firm ground or a solid rig platform
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), coll=P('Collective'), grounded=S('Grounded'), on_deck=S('On Deck')):
        my_slices = slices_and_not(grounded.get_slices(), on_deck.get_slices())
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(my_slices, low_coll)
        self.create_kpvs_within_slices(roll.array,
                                       my_slices,
                                       max_abs_value)


class RollOnDeckMax(KeyPointValueNode):
    '''
    Roll attitude on moving deck
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), coll=P('Collective'), on_deck=S('On Deck')):

        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(on_deck.get_slices(), low_coll)
        self.create_kpvs_within_slices(roll.array,
                                       my_slices,
                                       max_abs_value)


class RollCyclesExceeding5DegDuringFinalApproach(KeyPointValueNode):
    '''
    Counts the number of cycles of roll attitude that exceed 5 deg from
    peak to peak and with a maximum cycle period of 10 seconds during the
    final approach phase.
    '''

    units = ut.CYCLES

    def derive(self,
               roll=P('Roll'),
               fin_apps=S('Final Approach')):
        '''
        The algorithm counts each half-cycle, so an "N" figure would give a value
        of 1.5 cycles.
        '''
        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                roll.array[fin_app.slice],
                5.0, 10.0, roll.hz,
                fin_app.slice.start,
            ))


class RollCyclesExceeding15DegDuringFinalApproach(KeyPointValueNode):
    '''
    Counts the number of cycles of roll attitude that exceed 15 deg from
    peak to peak and with a maximum cycle period of 10 seconds during the
    final approach phase.
    '''

    units = ut.CYCLES

    def derive(self,
               roll=P('Roll'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                roll.array[fin_app.slice],
                15.0, 10.0, roll.hz,
                fin_app.slice.start,
            ))


class RollCyclesExceeding5DegDuringInitialClimb(KeyPointValueNode):
    '''
    Counts the number of cycles of roll attitude that exceed 5 deg from
    peak to peak and with a maximum cycle period of 10 seconds during the
    Initial Climb phase.
    '''

    units = ut.CYCLES

    def derive(self,
               roll=P('Roll'),
               initial_climbs=S('Initial Climb')):

        for climb in initial_climbs:
            self.create_kpv(*cycle_counter(
                roll.array[climb.slice],
                5.0, 10.0, roll.hz,
                climb.slice.start,
            ))


class RollCyclesExceeding15DegDuringInitialClimb(KeyPointValueNode):
    '''
    Counts the number of cycles of roll attitude that exceed 15 deg from
    peak to peak and with a maximum cycle period of 10 seconds during the
    Initial Climb phase.
    '''

    units = ut.CYCLES

    def derive(self,
               roll=P('Roll'),
               initial_climbs=S('Initial Climb')):

        for climb in initial_climbs:
            self.create_kpv(*cycle_counter(
                roll.array[climb.slice],
                15.0, 10.0, roll.hz,
                climb.slice.start,
            ))


class RollCyclesNotDuringFinalApproach(KeyPointValueNode):
    '''
    Counts the number of roll cycles between the point of liftoff
    and the start of final approach.
    Each cycle is defined as a change in bank angle of more 
    than 5 degrees and back again within 10 seconds. 
    '''

    units = ut.CYCLES

    def derive(self,
               roll=P('Roll'),
               airborne=S('Airborne'),
               fin_apps=S('Final Approach'),
               landings=S('Landing')):
        '''    
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Loss of Control - PIO. CAA limit > 20 deg total variation side to side".
    
        FDS cautioned 20 deg was excessive and evaluated different levels over 10
        sec time period with a view to settling the levels for production use.
        Having run a hundred sample flights using thresholds from 2 to 20 deg, 5
        deg was selected on the basis that this balanced enough data for trend
        analysis (a KPV was recorded for about one flight in three) without
        excessive counting of minor cycles. It was also convenient that this
        matched the existing threshold used by FDS for final approach analysis.
    
        Note: The algorithm counts each half-cycle, so an "N" figure would give a
        value of 1.5 cycles.
        '''
        not_fas = slices_and_not(airborne.get_slices(), fin_apps.get_slices())
        # TODO: Fix this:
        #not_fas = slices_and_not(not_fas.get_slices(), landings.get_slices())
        for not_fa in not_fas:
            self.create_kpv(*cycle_counter(
                roll.array[not_fa],
                5.0, 10.0, roll.hz,
                not_fa.start,
            ))


class RollAtLowAltitude(KeyPointValueNode):
    '''
    Roll angle below 600ft radio.
    Below 600ft, bank must not exceed 10% of A/C height for more than 5 sec.
    '''

    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt_rad=P('Altitude Radio')):
        '''
        The dlc phase used here identifies all low level operations below
        INITIAL_APPROACH_THRESHOLD. This includes takeoffs and landings. Running
        this KPV to cover these periods is not a problem and might identify
        bizzare takeoff or landing cases, hence these are not removed.
        '''
        ten_pc = 0.1

        lows = np.ma.clump_unmasked(
            np.ma.masked_outside(alt_rad.array, 50.0, 600.0))
        for low in lows:
            # Only compute the ratio for the short period below 600ft
            ratio = roll.array[low] / alt_rad.array[low]
            # We will work out bank angle periods exceeding 10% and 5 sec.
            banks = np.ma.clump_unmasked(
                np.ma.masked_less(np.ma.abs(ratio), ten_pc))
            banks = slices_remove_small_slices(banks,
                                               time_limit=5.0,
                                               hz=roll.frequency)
            for bank in banks:
                # Mark the largest roll attitude exceeding the limit.
                peak = max_abs_value(ratio[bank])
                peak_roll = roll.array[low][bank][peak.index]
                threshold = copysign(
                    alt_rad.array[low][bank][peak.index] * ten_pc,
                    peak_roll)
                index = peak.index + low.start + bank.start
                value = peak_roll - threshold
                self.create_kpv(index, value)


class RollLeftBelow6000FtAltitudeDensityBelow60Kts(KeyPointValueNode):
    '''
    Roll left angle below 6000ft Altitude Density and below 60kts IAS.
    '''

    units = ut.DEGREE

    @classmethod
    # This KPV is specific to the AS330 Puma helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'), family=A('Family')):
        is_puma = ac_type == helicopter and family and family.value == 'Puma'
        return is_puma and all_deps(cls, available)

    def derive(self, roll=P('Roll'), alt=P('Altitude Density'), airspeed=P('Airspeed'), airborne=S('Airborne')):
        # Roll left must be negative value; mask all positives
        roll_array = np.ma.masked_greater_equal(roll.array, 0)

        scope = slices_and(slices_below(alt.array, 6000)[1],
                           slices_below(airspeed.array, 60)[1])
        scope = slices_and(scope, airborne.get_slices())

        self.create_kpvs_within_slices(
            roll_array,
            scope,
            min_value,
        )


class RollLeftBelow8000FtAltitudeDensityAbove60Kts(KeyPointValueNode):
    '''
    Roll left angle below 8000ft Altitude Density and above 60kts IAS.
    '''

    units = ut.DEGREE

    @classmethod
    # This KPV is specific to the AS330 Puma helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'), family=A('Family')):
        is_puma = ac_type == helicopter and family and family.value == 'Puma'
        return is_puma and all_deps(cls, available)

    def derive(self, roll=P('Roll'), alt=P('Altitude Density'), airspeed=P('Airspeed'), airborne=S('Airborne')):
        # Roll left must be negative value; mask all positives
        roll_array = np.ma.masked_greater_equal(roll.array, 0)

        scope = slices_and(slices_below(alt.array, 8000)[1],
                           slices_above(airspeed.array, 60)[1])
        scope = slices_and(scope, airborne.get_slices())

        self.create_kpvs_within_slices(
            roll_array,
            scope,
            min_value,
        )


class RollLeftAbove6000FtAltitudeDensityBelow60Kts(KeyPointValueNode):
    '''
    Roll left angle above 6000ft Altitude Density and below 60kts IAS.
    '''

    units = ut.DEGREE

    @classmethod
    # This KPV is specific to the AS330 Puma helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'), family=A('Family')):
        is_puma = ac_type == helicopter and family and family.value == 'Puma'
        return is_puma and all_deps(cls, available)

    def derive(self, roll=P('Roll'), alt=P('Altitude Density'), airspeed=P('Airspeed'), airborne=S('Airborne')):
        # Roll left must be negative value; mask all positives
        roll_array = np.ma.masked_greater_equal(roll.array, 0)

        scope = slices_and(slices_between(alt.array, 6000, 8000)[1],
                           slices_below(airspeed.array, 60)[1])
        scope = slices_and(scope, airborne.get_slices())

        self.create_kpvs_within_slices(
            roll_array,
            scope,
            min_value,
        )


class RollLeftAbove8000FtAltitudeDensityAbove60Kts(KeyPointValueNode):
    '''
    Roll left angle above 8000ft Altitude Density and above 60kts IAS.
    '''

    units = ut.DEGREE

    @classmethod
    # This KPV is specific to the AS330 Puma helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'), family=A('Family')):
        is_puma = ac_type == helicopter and family and family.value == 'Puma'
        return is_puma and all_deps(cls, available)

    def derive(self, roll=P('Roll'), alt=P('Altitude Density'), airspeed=P('Airspeed'), airborne=S('Airborne')):
        # Roll left must be negative value; mask all positives
        roll_array = np.ma.masked_greater_equal(roll.array, 0)

        scope = slices_and(slices_above(alt.array, 8000)[1],
                           slices_above(airspeed.array, 60)[1])
        scope = slices_and(scope, airborne.get_slices())

        self.create_kpvs_within_slices(
            roll_array,
            scope,
            min_value,
        )


class RollRateMax(KeyPointValueNode):
    '''
    Maximum recorded roll rate. Helicopter only.
    '''

    units = ut.DEGREE_S

    can_operate = helicopter_only

    def derive(self, rr=P('Roll Rate'), airs=S('Airborne')):

        for air in airs:
            cycles = cycle_finder(rr.array[air.slice], min_step=5.0)
            for index in cycles[0][1:-1]:
                roll_rate = rr.array[index]
                if abs(roll_rate) > 5.0:
                    self.create_kpv(index+air.slice.start, roll_rate)


class RollAboveFL200Max(KeyPointValueNode):
    '''
    Maximum bank angle above FL200  (Alt STD Smoothed) (absolute value)
    '''
    
    name = 'Roll Above FL200 Max'
    units = ut.DEGREE

    def derive(self,
               roll=P('Roll'),
               alt=P('Altitude STD Smoothed')):

        alt.array = nearest_neighbour_mask_repair(alt.array)
        self.create_kpvs_within_slices(
            roll.array,
            alt.slices_above(20000),
            max_abs_value,
        )


class RollRateMaxAboveLimitAtTouchdown(KeyPointValueNode):
    '''
    Maximum value of roll rate limit exceedance at touchdown.
    '''
    units = ut.DEGREE_S
    
    def derive(self,
               appr_ldg=S('Approach And Landing'),
               limit=P('Roll Rate At Touchdown Limit'),
               roll_rate=P('Roll Rate For Touchdown'),
               accel_normal=P('Acceleration Normal'),):
        '''
        Roll rate exceedance within +/-2 seconds of peak acceleration 
        during Approach And Landing phase.
        '''
        max_acc_indices = [np.argmax(accel_normal.array[window.start:window.stop]) + \
                           window.start for window in appr_ldg.get_slices()]
        
        for i in max_acc_indices:
            window_start = i - (2*self.hz)
            window_end = i + (2*self.hz) + 1 # +1 so that the number of indices is equal on both sides of peak acceleration
            rr_over_limit = abs(roll_rate.array[window_start:window_end]) - \
                            limit.array[window_start:window_end]
            
            if rr_over_limit.any() > 0:
                self.create_kpv(np.argmax(rr_over_limit)+window_start, max(rr_over_limit))


##############################################################################
# Rotor


class RotorSpeedDuringAutorotationAbove108KtsMin(KeyPointValueNode):
    '''
    Minimum recorded rotor speed during autorotation above 108kts IAS.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), air_spd=P('Airspeed'), autorotation=S('Autorotation')):
        speed_bands = slices_and(autorotation.get_slices(),
                                  slices_above(air_spd.array, 108)[1])
        self.create_kpvs_within_slices(nr.array, speed_bands, min_value)


class RotorSpeedDuringAutorotationBelow108KtsMin(KeyPointValueNode):
    '''
    Minimum recorded rotor speed during autorotation below 108kts IAS.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), air_spd=P('Airspeed'), autorotation=S('Autorotation')):
        speed_bands = slices_and(autorotation.get_slices(),
                                  slices_below(air_spd.array, 108)[1])
        self.create_kpvs_within_slices(nr.array, speed_bands, min_value)


class RotorSpeedDuringAutorotationMax(KeyPointValueNode):
    '''
    Maximum recorded rotor speed during autorotation.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), autorotation=S('Autorotation')):
        self.create_kpvs_within_slices(nr.array, autorotation.get_slices(), max_value)


class RotorSpeedDuringAutorotationMin(KeyPointValueNode):
    '''
    Minimum rotor speed during autorotion. (helicopter only)
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), autorotation=S('Autorotation')):
        self.create_kpvs_within_slices(nr.array, autorotation.get_slices(),
                                       min_value)


class RotorSpeedWhileAirborneMax(KeyPointValueNode):
    '''
    This excludes autorotation, so is maximum rotor speed with power applied.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), airborne=S('Airborne'), autorotation=S('Autorotation')):
        self.create_kpv_from_slices(nr.array,
                                    slices_and_not(airborne.get_slices(),
                                                  autorotation.get_slices()),
                                    max_value)


class RotorSpeedWhileAirborneMin(KeyPointValueNode):
    '''
    This excludes autorotation, so is minimum rotor speed with power applied.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), airborne=S('Airborne'), autorotation=S('Autorotation')):
        self.create_kpv_from_slices(nr.array,
                                    slices_and_not(airborne.get_slices(),
                                                   autorotation.get_slices()),
                                    min_value)


class RotorSpeedWithRotorBrakeAppliedMax(KeyPointValueNode):
    '''
    Maximum rotor speed recorded with rotor brake applied.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), rotor_brake=P('Rotor Brake Engaged')):
        nr_array = np.ma.masked_less(nr.array, 1) # not interested if Rotor is not turning.
        slices = clump_multistate(rotor_brake.array, 'Engaged')
        # Synthetic minimum duration to ensure two samples needed to trigger.
        self.create_kpvs_within_slices(nr_array, slices, max_value, min_duration=1, freq=1)


class RotorsRunningDuration(KeyPointValueNode):
    '''
    Duration for which the rotors were running.
    '''

    units = ut.SECOND

    can_operate = helicopter_only

    def derive(self, rotors=M('Rotors Running')):
        running = runs_of_ones(rotors.array == 'Running')
        if running:
            value = slices_duration(running, rotors.frequency)
            self.create_kpv(running[-1].stop, value)


class RotorSpeedDuringMaximumContinuousPowerMin(KeyPointValueNode):
    '''
    Minimum rotor speed during maximum continuous power phase.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), mcp=S('Maximum Continuous Power'), autorotation=S('Autorotation')):
        self.create_kpv_from_slices(nr.array,
                                    slices_and_not(mcp.get_slices(),
                                                   autorotation.get_slices()),
                                    min_value)


class RotorSpeed36To49Duration(KeyPointValueNode):
    '''
    Duration in which rotor speed in running between 36% and 49%. 
    '''

    units = ut.SECOND

    @classmethod
    # This KPV is specific to the S92 helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        is_s92 = ac_type == helicopter and family and family.value == 'S92'
        return is_s92 and all_deps(cls, available)   

    def derive(self, nr=P('Nr')):
        self.create_kpvs_from_slice_durations(
            slices_between(nr.array, 36, 49)[1], nr.frequency)


class RotorSpeed56To67Duration(KeyPointValueNode):
    '''
    Duration in which rotor speed in running between 56% and 67%. 
    '''

    units = ut.SECOND

    @classmethod
    # This KPV is specific to the S92 helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        is_s92 = ac_type == helicopter and family and family.value == 'S92'
        return is_s92 and all_deps(cls, available)   

    def derive(self, nr=P('Nr')):
        self.create_kpvs_from_slice_durations(
            slices_between(nr.array, 56, 67)[1], nr.frequency)


class RotorSpeedAt6PercentCollectiveDuringEngStart(KeyPointValueNode):
    '''
    During the engines starting the collective the needs to be high, but as the
    rotor speed starts to increase the colective needs to be lowered. This KPV
    records the value Nr when the collective is fully lowered (<6% collective).
    '''

    units = ut.PERCENT

    @classmethod
    # This KPV is specific to the S92 helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        is_s92 = ac_type == helicopter and family and family.value == 'S92'
        return is_s92 and all_deps(cls, available)

    def derive(self, nr=P('Nr'), collective=P('Collective'),
               firsts=KTI('First Eng Fuel Flow Start')):
        for first in firsts:
            if value_at_index(collective.array, first.index) > 50:
                # ensure we are looking at rotor start and not rotors already
                # running at start of data
                index = index_at_value(collective.array, 6,
                                             slice(first.index, None))
                value = value_at_index(nr.array, index)
                self.create_kpv(index, value)
            else:
                continue


##############################################################################
# Rudder


class RudderDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum rudder angle recorded during takeoff (including RTO).
    '''

    units = ut.DEGREE

    def derive(self,
               rudder=P('Rudder'),
               to_rolls=S('Takeoff Roll Or Rejected Takeoff')):

        self.create_kpvs_within_slices(rudder.array, to_rolls, max_abs_value)


class RudderCyclesAbove50Ft(KeyPointValueNode):
    '''
    Counts the number of rudder cycles whilst the aircraft is above 50 ft AAL. 
    Each cycle is defined as a change in rudder deflection of more 
    than 2 degrees and back again within 2 seconds.
    '''

    units = ut.CYCLES

    def derive(self,
               rudder=P('Rudder'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Landing (Lateral) Rudder kick/oscillations Often there
        during landing, therefore need to determine what is abnormal, which may
        be difficult."
    
        Looks for sharp rudder reversal. Excludes operation below 50ft as this is
        normal use of the rudder to kick off drift. Uses the standard cycle
        counting process but looking for only one pair of half-cycles.
    
        The threshold used to be 6.5 deg, derived from a manufacturer's document,
        but this did not provide meaningful results in routine operations, so the
        threshold was reduced to 2 deg over 2 seconds.
        '''
        for above_50 in alt_aal.slices_above(50.0):
            self.create_kpv(*cycle_counter(
                rudder.array[above_50],
                2.0, 2.0, rudder.hz,
                above_50.start,
            ))


class RudderReversalAbove50Ft(KeyPointValueNode):
    '''
    Measures the single largest reversal of rudder deflection within a 3 second 
    period whilst the aircraft is above 50 ft AAL.
    '''

    units = ut.DEGREE

    def derive(self,
               rudder=P('Rudder'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for above_50 in alt_aal.slices_above(50.0):
            self.create_kpv(*cycle_select(
                rudder.array[above_50],
                1.0, 3.0, rudder.hz,
                above_50.start,
            ))


class RudderPedalForceMax(KeyPointValueNode):
    '''
    Maximum rudder pedal force (irrespective of which foot is used).
    '''
    units = ut.DECANEWTON

    def derive(self,
               force=P('Rudder Pedal Force'),
               fast=S('Fast')):
        self.create_kpvs_within_slices(
            force.array, fast.get_slices(),
            max_abs_value)


class RudderPedalMax(KeyPointValueNode):
    '''
    Temporary KPV to help determine full range of rudder pedal movement
    '''
    units = ut.DEGREE

    def derive(self, pedal=P('Rudder Pedal')):
        index, value = max_value(pedal.array)
        self.create_kpv(index, value)


class RudderPedalMin(KeyPointValueNode):
    '''
    Temporary KPV to help determine full range of rudder pedal movement
    '''
    units = ut.DEGREE

    def derive(self, pedal=P('Rudder Pedal')):
        index, value = min_value(pedal.array)
        self.create_kpv(index, value)


##############################################################################
# Speedbrake


class SpeedbrakeDeployed1000To20FtDuration(KeyPointValueNode):
    '''
    Duration for which the speedbrake was deployed between 1000ft and 20ft AAL.
    '''

    units = ut.SECOND

    def derive(self,
               spd_brk=M('Speedbrake Selected'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for descent in alt_aal.slices_from_to(1000, 20):
            array = spd_brk.array[descent] == 'Deployed/Cmd Up'
            slices = shift_slices(runs_of_ones(array), descent.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


class AltitudeWithSpeedbrakeDeployedDuringFinalApproachMin(KeyPointValueNode):
    '''
    Minimum Altitude (AAL) when speedbrake is deployed during the final approach.
    '''

    units = ut.FT

    def derive(self, alt_aal=P('Altitude AAL'),
               spd_brk=M('Speedbrake Selected'),
               fin_app=S('Final Approach')):
        slices = clump_multistate(spd_brk.array, 'Deployed/Cmd Up',
                                  fin_app.get_slices())
        self.create_kpvs_within_slices(alt_aal.array, slices, min_value)


class SpeedbrakeDeployedWithFlapDuration(KeyPointValueNode):
    '''
    Duration for which the speedbrake was deployed with flaps extended.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and all_of(('Speedbrake Selected', 'Airborne'), available)

    def derive(self,
               spd_brk=M('Speedbrake Selected'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        for air in airborne:
            spd_brk_dep = spd_brk.array[air.slice] == 'Deployed/Cmd Up'
            array = spd_brk_dep & ~retracted[air.slice]
            slices = shift_slices(runs_of_ones(array), air.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


class SpeedbrakeDeployedWithPowerOnDuration(KeyPointValueNode):
    '''
    Each time the aircraft is flown with high power and the speedbrakes open,
    something unusual is happening. We record the duration this happened for.

    The threshold for high power is 60% N1. This aligns with the Airbus AFPS
    and other flight data analysis programmes.
    '''

    units = ut.SECOND

    def derive(self, spd_brk=M('Speedbrake Selected'),
               power=P('Eng (*) N1 Avg'), alt_aal=S('Altitude AAL For Flight Phases')):
        power_on_percent = 60.0
        airborne = np.ma.clump_unmasked(np.ma.masked_less(alt_aal.array, 50))  # only interested when airborne
        for air in airborne:
            spd_brk_dep = spd_brk.array[air] == 'Deployed/Cmd Up'
            high_power = power.array[air] >= power_on_percent
            array = spd_brk_dep & high_power
            slices = shift_slices(runs_of_ones(array), air.start)
            slices = slices_remove_small_gaps(slices, 30) # multiple occurences within 30s of each other are treated as one
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


class SpeedbrakeDeployedDuringGoAroundDuration(KeyPointValueNode):
    '''
    Duration for which the speedbrake was deployed during a go around.
    '''

    units = ut.SECOND

    def derive(self,
               spd_brk=M('Speedbrake Selected'),
               go_arounds=S('Go Around And Climbout')):

        deployed = spd_brk.array == 'Deployed/Cmd Up'
        for go_around in go_arounds:
            array = deployed[go_around.slice]
            slices = shift_slices(runs_of_ones(array),
                                  go_around.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


##############################################################################
# Warnings: Stall, Stick Pusher/Shaker

class StallWarningDuration(KeyPointValueNode):
    '''
    Duration the Stall Warning was active while airborne.
    '''

    units = ut.SECOND

    def derive(self, stall_warning=M('Stall Warning'), airs=S('Airborne')):
        self.create_kpvs_where(stall_warning.array == 'Warning',
                               stall_warning.hz, phase=airs)


class StickPusherActivatedDuration(KeyPointValueNode):
    '''
    Duration for which the stick pusher was active.
    '''

    units = ut.SECOND

    def derive(self, stick_pusher=M('Stick Pusher'), airs=S('Airborne')):
        # TODO: Check that this triggers correctly as stick push events are probably
        #       single samples.
        self.create_kpvs_where(stick_pusher.array == 'Push',
                               stick_pusher.hz, phase=airs)


class StickShakerActivatedDuration(KeyPointValueNode):
    '''
    Duration for which the stick shaker was active.
    '''

    units = ut.SECOND

    def derive(self, stick_shaker=M('Stick Shaker'), airs=S('Airborne')):
        self.create_kpvs_where(stick_shaker.array == 'Shake',
                               stick_shaker.hz, phase=airs, min_duration=1.0)


class OverspeedDuration(KeyPointValueNode):
    '''
    Duration the Overspeed Warning was active.
    '''

    units = ut.SECOND

    def derive(self, overspeed=M('Overspeed Warning'), airs=S('Airborne')):
        self.create_kpvs_where(overspeed.array == 'Overspeed',
                               self.frequency, phase=airs)


class StallFaultCautionDuration(KeyPointValueNode):
    '''
    Duration in which either the left or right stall fault caution is raised.
    '''
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        stall_fault = any_of(('Stall (L) Fault Caution',
                              'Stall (R) Fault Caution'), available)
        airborne = 'Airborne' in available
        return stall_fault and airborne

    def derive(self, stall_l=M('Stall (L) Fault Caution'),
               stall_r=M('Stall (L) Fault Caution'), airborne=S('Airborne')):
        stall_fault_caution=vstack_params_where_state(
            (stall_l, 'Caution'),
            (stall_r, 'Caution')
        ).any(axis=0)
        comb_air = mask_outside_slices(stall_fault_caution,
                                       airborne.get_slices())
        self.create_kpvs_from_slice_durations(runs_of_ones(comb_air), self.hz)


class CruiseSpeedLowDuration(KeyPointValueNode):
    '''
    Duration in which the 'Cruise Speed Low' warning is raised.
    '''
    units = ut.SECOND

    def derive(self, spd=M('Cruise Speed Low'), airborne=S('Airborne')):
        self.create_kpvs_where(spd.array == 'Low', spd.frequency, airborne)


class DegradedPerformanceCautionDuration(KeyPointValueNode):
    '''
    Duration in which the 'Degraded Performance Caution' is raised.
    '''
    units = ut.SECOND

    def derive(self, caution=M('Degraded Performance Caution'),
               airborne=S('Airborne')):
        self.create_kpvs_where(caution.array == 'Caution',
                               caution.frequency, airborne)


class AirspeedIncreaseAlertDuration(KeyPointValueNode):
    '''
    Duration in which the 'Airspeed Increase Alert' is raised.
    '''
    units = ut.SECOND

    def derive(self, alert=M('Airspeed Increase Alert'),
               airborne=S('Airborne')):
        self.create_kpvs_where(alert.array == 'Alert', alert.frequency,
                               airborne)


class AirspeedBelowMinimumAirspeedMin(KeyPointValueNode):
    '''
    Duration in which the Airspeed is below Minimum Airspeed in clean
    configuration. 
    
    If no Minimum Airspeed parameter, KPV will use the Flap and 
    Flap Manoeuvre Speed
    '''
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        core = all_of(['Airspeed', 'Airborne', 'Flap'], available)
        min_spd = any_of(['Minimum Airspeed', 'Flap Manoeuvre Speed'],
                         available)
        return core and min_spd

    def derive(self,
               air_spd=P('Airspeed'),
               min_spd=P('Minimum Airspeed'),
               flap = M('Flap'),
               f_m_spd = P('Flap Manoeuvre Speed'),
               airborne = S('Airborne')):
        mspd = min_spd or f_m_spd
        if mspd.name == 'Minimum Airspeed':
            to_test = air_spd.array - mspd.array
            spd_slices = runs_of_ones(to_test < 0)
        else:
            # Flap Manoeuvre Speed masks data above 20,000 ft STD, need to use
            # raw data
            to_test = air_spd.array - mspd.array.data
            spd_slices = runs_of_ones(to_test < 0)
        slices = slices_and(
            clump_multistate(flap.array, '0', airborne.get_slices()),
            spd_slices
        )
        # self.create_kpvs_from_slice_durations(slices, air_spd.frequency)
        self.create_kpv_from_slices(to_test, slices, min_value)


##############################################################################
# Tail Clearance


class TailClearanceDuringTakeoffMin(KeyPointValueNode):
    '''
    Minimum tail clearance during takeoff.
    '''

    units = ut.FT

    def derive(self,
               alt_tail=P('Altitude Tail'),
               takeoffs=S('Takeoff')):

        self.create_kpvs_within_slices(alt_tail.array, takeoffs, min_value)


class TailClearanceDuringLandingMin(KeyPointValueNode):
    '''
    Minimum tail clearance during landing.
    '''

    units = ut.FT

    def derive(self,
               alt_tail=P('Altitude Tail'),
               landings=S('Landing')):

        self.create_kpvs_within_slices(alt_tail.array, landings, min_value)


class TailClearanceDuringGoAroundMin(KeyPointValueNode):
    '''
    '''

    units = ut.FT

    def derive(self,
               alt_tail=P('Altitude Tail'),
               go_arounds=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(alt_tail.array, go_arounds, min_value)


class TailClearanceDuringApproachMin(KeyPointValueNode):
    '''
    This finds abnormally low tail clearance during the approach down to 100ft.
    It searches for the minimum angular separation between the flightpath and
    the terrain, so a 500ft clearance at 2500ft AAL is considered more
    significant than 500ft at 1500ft AAL. The value stored is the tail
    clearance. A matching KTI will allow these to be located on the approach
    chart.
    '''

    units = ut.FT

    def derive(self,
               alt_aal=P('Altitude AAL'),
               alt_tail=P('Altitude Tail'),
               dtl=P('Distance To Landing')):

        for desc_slice in alt_aal.slices_from_to(3000, 100):
            angle_array = alt_tail.array[desc_slice] \
                / ut.convert(dtl.array[desc_slice], ut.NM, ut.FT)
            index, value = min_value(angle_array)
            if index:
                sample = index + desc_slice.start
                self.create_kpv(sample, alt_tail.array[sample])


##############################################################################
# Terrain Clearance


class TerrainClearanceAbove3000FtMin(KeyPointValueNode):
    '''
    Minimum terrain clearance above 3000ft AAL. 
    '''

    units = ut.FT

    def derive(self,
               alt_rad=P('Altitude Radio'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Controlled Flight Into Terrain (CFIT) At/Below Minimum terrain clearance
        on approach/departure >3000ft AFE and <1000ft AGL"

        Solution: Compute minimum terrain clearance while Altitude AAL over 3000ft.
        Note: For most flights, Altitude Radio will be over 2,500ft at this time,
        so masked, hence no kpv will be created.
        '''
        self.create_kpvs_within_slices(
            alt_rad.array,
            alt_aal.slices_above(3000),
            min_value,
        )


##############################################################################
# Tailwind


class TailwindLiftoffTo100FtMax(KeyPointValueNode):
    '''
    Maximum tailwind between the point of liftoff and 100ft AAL.
    Note: a negative tailwind is a headwind
    '''

    units = ut.KT

    def derive(self,
               tailwind=P('Tailwind'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        FDS developed this KPV to support the UK CAA Significant Seven programme.
        "Excursions - Take-Off (Longitudinal), Tailwind - Needs to be recorded just
        after take-off.

        CAA comment: Some operators will have purchased (AFM) a 15kt tailwind limit
        for take-off. But this should only be altered to 15 kt if it has been
        purchased.
        '''
        self.create_kpvs_within_slices(
            tailwind.array,
            alt_aal.slices_from_to(0, 100),
            max_value,
        )


class Tailwind100FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum tailwind between the 100ft AAL and the point of touchdown.
    Note: a negative tailwind is a headwind
    '''

    units = ut.KT

    def derive(self,
               tailwind=P('Tailwind'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            tailwind.array,
            alt_aal.slices_to_kti(100, touchdowns),
            max_value,
        )


class TailwindDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum tailwind from first valid sample of Airspeed to lift off.
    '''

    can_operate = aeroplane_only

    units = ut.DEGREE

    def derive(self,
               tailwind=P('Tailwind'),
               airspeed=P('Airspeed True'),
               liftoffs=KTI('Liftoff'),
               toffs=S('Takeoff'),
               ):

        for toff in toffs:
            spd = np.ma.masked_less(airspeed.array, 60)
            first_spd_idx = first_valid_sample(spd[toff.slice])[0]
            if first_spd_idx:
                first_spd_idx = first_spd_idx + toff.slice.start
                liftoff = liftoffs.get_first(within_slice=toff.slice)
                if liftoff:
                    self.create_kpvs_within_slices(tailwind.array,
                                                   (slice(first_spd_idx, liftoff.index),),
                                                   max_value)
            else:
                self.warning('No Valid Airspeed True in takeof phase?')


##############################################################################
# Warnings: Master Caution/Warning


class MasterWarningDuration(KeyPointValueNode):
    '''
    Duration for which the Master Warning was active.

    On some types nuisance recordings arise before first engine start, hence
    the added condition for engine running.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):

        return 'Master Warning' in available

    def derive(self,
               warning=M('Master Warning'),
               any_engine=M('Eng (*) Any Running'),
               family=A('Family'),
               airborne=S('Airborne')):
        
        if family and family.value in 'AW139':
            self.create_kpvs_where(warning.array == 'Warning', warning.hz, phase=airborne)
        
        elif any_engine:
            self.create_kpvs_where(np.ma.logical_and(warning.array == 'Warning',
                                                     any_engine.array == 'Running'),
                                   warning.hz)
        else:
            self.create_kpvs_where(warning.array == 'Warning', warning.hz)


class MasterWarningDuringTakeoffDuration(KeyPointValueNode):
    '''
    Duration for which Master Warning was active during takeoff.
    '''

    units = ut.SECOND

    def derive(self,
               warning=M('Master Warning'),
               takeoff_rolls=S('Takeoff Roll Or Rejected Takeoff')):

        self.create_kpvs_where(warning.array == 'Warning',
                               warning.hz, phase=takeoff_rolls)


class MasterCautionDuringTakeoffDuration(KeyPointValueNode):
    '''
    Duration for which Master Caution was active during takeoff.
    '''

    units = ut.SECOND

    def derive(self,
               caution=M('Master Caution'),
               takeoff_rolls=S('Takeoff Roll Or Rejected Takeoff')):

        self.create_kpvs_where(caution.array == 'Caution',
                               caution.hz, phase=takeoff_rolls)


##############################################################################
# Taxi In


class TaxiInDuration(KeyPointValueNode):
    '''
    Duration of taxi in.
    '''

    units = ut.SECOND

    def derive(self,
               taxi_ins=S('Taxi In')):

        # TODO: Support midpoint in self.create_kpvs_within_slices()!
        for taxi_in in taxi_ins:
            self.create_kpv(slice_midpoint(taxi_in.slice),
                            slice_duration(taxi_in.slice, self.frequency))


##############################################################################
# Taxi Out


class TaxiOutDuration(KeyPointValueNode):
    '''
    Duration of taxi out.
    '''

    units = ut.SECOND

    def derive(self,
               taxi_outs=S('Taxi Out')):

        # TODO: Support midpoint in self.create_kpvs_within_slices()!
        for taxi_out in taxi_outs:
            self.create_kpv(slice_midpoint(taxi_out.slice),
                            slice_duration(taxi_out.slice, self.frequency))


##############################################################################
# Warnings: Terrain Awareness & Warning System (TAWS)


class TAWSAlertDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Alert is triggered.
    '''

    name = 'TAWS Alert Duration'
    units = ut.SECOND

    def derive(self, taws_alert=M('TAWS Alert'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_alert.array == 'Alert',
                               taws_alert.hz, phase=airborne)


class TAWSWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Warning is triggered.
    '''

    name = 'TAWS Warning Duration'
    units = ut.SECOND

    def derive(self, taws_warning=M('TAWS Warning'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_warning.array == 'Warning',
                               taws_warning.hz, phase=airborne)


class TAWSGeneralWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS General Warning is triggered.
    '''

    name = 'TAWS General Warning Duration'
    units = ut.SECOND

    def derive(self, taws_general=M('TAWS General'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_general.array == 'Warning',
                               taws_general.hz, phase=airborne)


class TAWSSinkRateWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Sink Rate Warning is triggered.
    '''

    name = 'TAWS Sink Rate Warning Duration'
    units = ut.SECOND

    def derive(self, taws_sink_rate=M('TAWS Sink Rate'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_sink_rate.array == 'Warning',
                               taws_sink_rate.hz, phase=airborne)


class TAWSTooLowFlapWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Too Low Flaps Warning is triggered.
    '''

    name = 'TAWS Too Low Flap Warning Duration'
    units = ut.SECOND

    def derive(self, taws_too_low_flap=M('TAWS Too Low Flap'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_too_low_flap.array == 'Warning',
                               taws_too_low_flap.hz, phase=airborne)


class TAWSTerrainWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Terrain Warning is triggered.
    '''

    name = 'TAWS Terrain Warning Duration'
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return ('Airborne' in available and
                any_of(('TAWS Terrain', 'TAWS Terrain Warning'), available))

    def derive(self, taws_terrain=M('TAWS Terrain'),
               taws_terrain_warning=M('TAWS Terrain Warning'),
               airborne=S('Airborne')):
        hz = (taws_terrain or taws_terrain_warning).hz
        taws_terrains = vstack_params_where_state(
            (taws_terrain, 'Warning'),
            (taws_terrain_warning, 'Warning')).any(axis=0)
        self.create_kpvs_where(taws_terrains, hz, phase=airborne)


class TAWSTerrainPullUpWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Terrain, Pull Up Warning is triggered.
    '''

    name = 'TAWS Terrain Pull Up Warning Duration'
    units = ut.SECOND

    def derive(self, taws_terrain_pull_up=M('TAWS Terrain Pull Up'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_terrain_pull_up.array == 'Warning',
                               taws_terrain_pull_up.hz, phase=airborne)


class TAWSTerrainClearanceFloorAlertDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Terrain Clearance, Floor Alert is triggered.
    '''

    name = 'TAWS Terrain Clearance Floor Alert Duration'
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return 'Airborne' in available and any_of(
            ['TAWS Terrain Clearance Floor Alert',
             'TAWS Terrain Clearance Floor Alert (2)'], available)

    def derive(self,
               taws_terrain_clearance_floor_alert=
               M('TAWS Terrain Clearance Floor Alert'),
               taws_terrain_clearance_floor_alert_2=
               M('TAWS Terrain Clearance Floor Alert (2)'),
               airborne=S('Airborne')):

        hz = (taws_terrain_clearance_floor_alert or 
              taws_terrain_clearance_floor_alert_2).hz

        taws_terrain_alert = vstack_params_where_state(
            (taws_terrain_clearance_floor_alert, 'Alert'),
            (taws_terrain_clearance_floor_alert_2, 'Alert')).any(axis=0)

        self.create_kpvs_where(taws_terrain_alert, hz, phase=airborne)


class TAWSGlideslopeWarning1500To1000FtDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Glideslope Warning is triggered 
    between 1500 and 1000ft AAL.
    '''

    name = 'TAWS Glideslope Warning 1500 To 1000 Ft Duration'
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return 'Altitude AAL For Flight Phases' in available and\
               any_of(['TAWS Glideslope', 'TAWS Glideslope Alert'], available)

    def derive(self,
               taws_glideslope=M('TAWS Glideslope'),
               taws_alert=M('TAWS Glideslope Alert'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        taws_gs = vstack_params_where_state(
            (taws_glideslope, 'Warning'),
            (taws_alert, 'Warning')
            ).any(axis=0)

        phases = slices_and(runs_of_ones(taws_gs),
                            alt_aal.slices_from_to(1500, 1000))

        self.create_kpvs_from_slice_durations(phases, self.frequency)


class TAWSGlideslopeWarning1000To500FtDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Glideslope Warning is triggered 
    between 1000 and 500ft AAL.
    '''

    name = 'TAWS Glideslope Warning 1000 To 500 Ft Duration'
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return 'Altitude AAL For Flight Phases' in available and\
               any_of(['TAWS Glideslope', 'TAWS Glideslope Alert'], available)

    def derive(self,
               taws_glideslope=M('TAWS Glideslope'),
               taws_alert=M('TAWS Glideslope Alert'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        taws_gs = vstack_params_where_state(
            (taws_glideslope, 'Warning'),
            (taws_alert, 'Warning')
        ).any(axis=0)

        phases = slices_and(runs_of_ones(taws_gs),
                            alt_aal.slices_from_to(1000, 500))

        self.create_kpvs_from_slice_durations(phases, self.frequency)


class TAWSGlideslopeWarning500To200FtDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Glideslope Warning is triggered 
    between 500 and 200ft AAL.
    '''

    name = 'TAWS Glideslope Warning 500 To 200 Ft Duration'
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return 'Altitude AAL For Flight Phases' in available and\
               any_of(['TAWS Glideslope', 'TAWS Glideslope Alert'], available)


    def derive(self,
               taws_glideslope=M('TAWS Glideslope'),
               taws_alert=M('TAWS Glideslope Alert'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        taws_gs = vstack_params_where_state(
            (taws_glideslope, 'Warning'),
            (taws_alert, 'Warning')
        ).any(axis=0)

        phases = slices_and(runs_of_ones(taws_gs),
                            alt_aal.slices_from_to(500, 200))

        self.create_kpvs_from_slice_durations(phases, self.frequency)


class TAWSTooLowTerrainWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Too Low Terrain Warning was triggered.
    '''

    name = 'TAWS Too Low Terrain Warning Duration'
    units = ut.SECOND

    def derive(self, taws_too_low_terrain=M('TAWS Too Low Terrain'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_too_low_terrain.array == 'Warning',
                               taws_too_low_terrain.hz, phase=airborne)


class TAWSTooLowGearWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Too Low Gear Warning was triggered.
    '''

    name = 'TAWS Too Low Gear Warning Duration'
    units = ut.SECOND

    def derive(self, taws_too_low_gear=M('TAWS Too Low Gear'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_too_low_gear.array == 'Warning',
                               taws_too_low_gear.hz, phase=airborne)


class TAWSPullUpWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Pull Up Warning was triggered.
    '''

    name = 'TAWS Pull Up Warning Duration'
    units = ut.SECOND

    def derive(self, taws_pull_up=M('TAWS Pull Up'), airborne=S('Airborne')):
        self.create_kpvs_where(taws_pull_up.array == 'Warning',
                               taws_pull_up.hz, phase=airborne)


class TAWSDontSinkWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Don't Sink Warning was triggered.
    '''

    name = 'TAWS Dont Sink Warning Duration'
    units = ut.SECOND

    def derive(self, taws_dont_sink=M('TAWS Dont Sink'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_dont_sink.array == 'Warning',
                               taws_dont_sink.hz, phase=airborne)


class TAWSCautionObstacleDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Caution Obstacle was triggered.
    '''

    name = 'TAWS Caution Obstacle Duration'
    units = ut.SECOND

    def derive(self, taws_caution_obstacle=M('TAWS Caution Obstacle'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_caution_obstacle.array == 'Caution',
                               taws_caution_obstacle.hz, phase=airborne)


class TAWSCautionTerrainDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Caution Terrain was triggered.
    Sourced from GPWS.
    '''

    name = 'TAWS Caution Terrain Duration'
    units = ut.SECOND

    def derive(self, taws_caution_terrain=M('TAWS Caution Terrain'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_caution_terrain.array == 'Caution',
                               taws_caution_terrain.hz, phase=airborne)


class TAWSTerrainCautionDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Terrain Caution was triggered.
    TAWS Terrain Caution is sourced from EGPWS.
    '''

    name = 'TAWS Terrain Caution Duration'
    units = ut.SECOND

    def derive(self, taws_terrain_caution=M('TAWS Terrain Caution'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_terrain_caution.array == 'Caution',
                               taws_terrain_caution.hz, phase=airborne)


class TAWSFailureDuration(KeyPointValueNode):
    '''
    Duration for which the TAWS was reporting failure.
    '''

    name = 'TAWS Failure Duration'
    units = ut.SECOND

    def derive(self, taws_failure=M('TAWS Failure'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_failure.array == 'Failed',
                               taws_failure.hz, phase=airborne)


class TAWSObstacleWarningDuration(KeyPointValueNode):
    '''
    Duration of TAWS Obstacle Warning.
    '''

    name = 'TAWS Obstacle Warning Duration'
    units = ut.SECOND

    def derive(self, taws_obstacle_warning=M('TAWS Obstacle Warning'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_obstacle_warning.array == 'Warning',
                               taws_obstacle_warning.hz, phase=airborne)


class TAWSPredictiveWindshearDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Predictive Windshear was triggered.
    '''

    name = 'TAWS Predictive Windshear Duration'
    units = ut.SECOND

    def derive(self, taws_pw=M('TAWS Predictive Windshear'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_pw.array == 'Warning',
                               taws_pw.hz, phase=airborne)


class TAWSTerrainAheadDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Terrain Ahead was triggered.
    '''

    name = 'TAWS Terrain Ahead Duration'
    units = ut.SECOND

    def derive(self, taws_terrain_ahead=M('TAWS Terrain Ahead'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_terrain_ahead.array == 'Warning',
                               taws_terrain_ahead.hz, phase=airborne)


class TAWSTerrainAheadPullUpDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Terrain Ahead, Pull Up was triggered.
    '''

    name = 'TAWS Terrain Ahead Pull Up Duration'
    units = ut.SECOND

    def derive(self, taws_terrain_ahead_pu=M('TAWS Terrain Ahead Pull Up'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_terrain_ahead_pu.array == 'Warning',
                               taws_terrain_ahead_pu.hz, phase=airborne)


class TAWSWindshearWarningBelow1500FtDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Windshear Warning was triggered 
    below 1500ft AAL.
    '''

    name = 'TAWS Windshear Warning Below 1500 Ft Duration'
    units = ut.SECOND

    def derive(self, taws_windshear=M('TAWS Windshear Warning'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fasts=S('Fast')):
        fast_below_1500 = slices_and(fasts.get_slices(), alt_aal.slices_below(1500))
        fast_below_1500 = [_slice for _slice in fast_below_1500 if _slice.stop - _slice.start >= 1]
        self.create_kpvs_where(taws_windshear.array == 'Warning',
                               taws_windshear.hz,
                               fast_below_1500)


class TAWSWindshearCautionBelow1500FtDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Windshear Caution was triggered 
    below 1500ft AAL.
    '''

    name = 'TAWS Windshear Caution Below 1500 Ft Duration'
    units = ut.SECOND

    def derive(self, taws_windshear=M('TAWS Windshear Caution'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fasts=S('Fast')):
        fast_below_1500 = slices_and(fasts.get_slices(), alt_aal.slices_below(1500))
        fast_below_1500 = [_slice for _slice in fast_below_1500 if _slice.stop - _slice.start >= 1]
        self.create_kpvs_where(taws_windshear.array == 'Caution',
                               taws_windshear.hz,
                               fast_below_1500)


class TAWSWindshearSirenBelow1500FtDuration(KeyPointValueNode):
    '''
    The duration for which the TAWS Windshear Siren was triggered 
    below 1500ft AAL.
    '''

    name = 'TAWS Windshear Siren Below 1500 Ft Duration'
    units = ut.SECOND

    def derive(self, taws_windshear=M('TAWS Windshear Siren'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fasts=S('Fast')):
        fast_below_1500 = slices_and(fasts.get_slices(), alt_aal.slices_below(1500))
        fast_below_1500 = [_slice for _slice in fast_below_1500 if _slice.stop - _slice.start >= 1]
        self.create_kpvs_where(taws_windshear.array == 'Siren',
                               taws_windshear.hz,
                               fast_below_1500)


class TAWSUnspecifiedDuration(KeyPointValueNode):
    '''
    The duration for which the 'unspecified' TAWS Warning is active.
    '''

    name = 'TAWS Unspecified Duration'
    units = ut.SECOND

    def derive(self, taws_unspecified=M('TAWS Unspecified'),
               airborne=S('Airborne')):
        self.create_kpvs_where(taws_unspecified.array == 'Warning',
                               taws_unspecified.hz, phase=airborne)

##############################################################################
# Warnings: Traffic Collision Avoidance System (TCAS)


class TCASTAWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TCAS TA Warning was active.

    One second warnings are commonplace around airports, hence the 2 second
    minimum threshold.
    '''

    name = 'TCAS TA Warning Duration'
    units = ut.SECOND

    def derive(self, tcas_ta=M('TCAS TA'),
               airs=S('Airborne')):

        for air in airs:
            ras_local = tcas_ta.array[air.slice] == 'TA'

            ras_slices = shift_slices(runs_of_ones(ras_local), air.slice.start)
            self.create_kpvs_from_slice_durations(ras_slices, self.frequency,
                                                  min_duration=2.0,
                                                  mark='start')


class TCASRAWarningDuration(KeyPointValueNode):
    '''
    The duration for which the TCAS RA Warning was active.
    '''

    name = 'TCAS RA Warning Duration'
    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return any_of(('TCAS RA', 'TCAS Combined Control'), available) \
            and 'Airborne' in available

    def derive(self, tcas_ra=M('TCAS RA'),
               tcas=M('TCAS Combined Control'),
               airs=S('Airborne')):

        for air in airs:
            if tcas_ra:
                ras_local = tcas_ra.array[air.slice] == 'RA'
            else:
                # If the RA is not recorded separately:
                ras_local = tcas.array[air.slice].any_of('Drop Track',
                                                         'Altitude Lost',
                                                         'Up Advisory Corrective',
                                                         'Down Advisory Corrective',
                                                         'Preventive',
                                                         ignore_missing=True)

            ras_slices = shift_slices(runs_of_ones(ras_local), air.slice.start)
            # Where data is corrupted, single samples are a common source of error
            # time_limit rejects single samples, but 2+ sample events are retained.
            ras_slices = slices_remove_small_slices(ras_slices, time_limit=1)
            self.create_kpvs_from_slice_durations(ras_slices, self.frequency,
                                                  mark='start')


class TCASRAReactionDelay(KeyPointValueNode):
    '''
    This measures the time taken for the pilot to react, determined by the onset
    of the first major change in normal acceleration after the RA started.
    '''

    name = 'TCAS RA Reaction Delay'
    units = ut.SECOND

    def derive(self, acc=P('Acceleration Normal Offset Removed'),
               tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        acc_array = repair_mask(acc.array, repair_duration=None)
        for air in airs:
            ras_local = tcas.array[air.slice].any_of('Drop Track',
                                                     'Altitude Lost',
                                                     'Up Advisory Corrective',
                                                     'Down Advisory Corrective',
                                                     'Preventive',
                                                     ignore_missing=True)
            ras = shift_slices(runs_of_ones(ras_local), air.slice.start)
            # Assume that the reaction takes place during the TCAS RA period:
            for ra in ras:
                if np.ma.count(acc_array[ra]) == 0:
                    continue
                i, p = cycle_finder(acc_array[ra] - 1.0, 0.15)
                # i, p will be None if the data is too short or invalid and so
                # no cycles can be found.
                if i is None:
                    continue
                indexes = np.array(i)
                peaks = np.array(p)
                # Look beyond 2 seconds to find slope from point of initiation.
                slopes = np.ma.where(indexes > 17, abs(peaks / indexes), 0.0)
                start_to_peak = slice(ra.start, ra.start + i[np.argmax(slopes)])
                peek_curvature_ix = peak_curvature(
                    acc_array, _slice=start_to_peak, curve_sense='Bipolar')
                if peek_curvature_ix is not None:
                    react_index = peek_curvature_ix - ra.start
                    self.create_kpv(ra.start + react_index,
                                    react_index / acc.frequency)


class TCASRAInitialReactionStrength(KeyPointValueNode):
    '''
    This measures the strength of the first reaction to the RA, in g per second.
    Most importantly, this is positive if the reaction is in the same sense as
    the Resolution Advisory (up for up or down for down) but negative in sign if
    the action is in the opposite direction to the RA.
    '''

    name = 'TCAS RA Initial Reaction Strength'
    units = None  # FIXME

    def derive(self, acc=P('Acceleration Normal Offset Removed'),
               tcas=M('TCAS Combined Control'), airs=S('Airborne')):

        for air in airs:
            ras_local = tcas.array[air.slice].any_of('Drop Track',
                                                     'Altitude Lost',
                                                     'Up Advisory Corrective',
                                                     'Down Advisory Corrective',
                                                     'Preventive',
                                                     ignore_missing=True)
            ras = shift_slices(runs_of_ones(ras_local), air.slice.start)
            # We assume that the reaction takes place during the TCAS RA
            # period.
            for ra in ras:
                if np.ma.count(acc.array[ra]) == 0:
                    continue
                i, p = cycle_finder(acc.array[ra] - 1.0, 0.1)
                if i is None:
                    continue
                # Convert to Numpy arrays for ease of arithmetic
                indexes = np.array(i)
                peaks = np.array(p)
                slopes = np.ma.where(indexes > 17, abs(peaks / indexes), 0.0)
                s_max = np.argmax(slopes)

                # So we look for the steepest slope to the peak, which
                # ignores little early peaks or slightly high later peaks.
                # From inspection of many traces, this is the best way to
                # distinguish the peak of interest.
                if s_max == 0:
                    slope = peaks[0] / indexes[0]
                else:
                    slope = (peaks[s_max] - peaks[s_max - 1]) / \
                        (indexes[s_max] - indexes[s_max - 1])
                # Units of g/sec:
                slope *= acc.frequency

                if tcas.array[ra.start] == 5:
                    # Down advisory, so negative is good.
                    slope = -slope
                self.create_kpv(ra.start, slope)


class TCASRAToAPDisengagedDuration(KeyPointValueNode):
    '''
    Time between the onset of the RA and disconnection of the autopilot.

    Since the pilot's initial action should be to disengage the autopilot,
    this duration is another indication of pilot reaction time.
    '''

    name = 'TCAS RA To AP Disengaged Duration'
    units = ut.SECOND

    def derive(self,
               ap_offs=KTI('AP Disengaged Selection'),
               tcas=M('TCAS Combined Control'),
               airs=S('Airborne')):

        for air in airs:
            ras_local = tcas.array[air.slice].any_of('Drop Track',
                                                     'Altitude Lost',
                                                     'Up Advisory Corrective',
                                                     'Down Advisory Corrective',
                                                     'Preventive',
                                                     ignore_missing=True)
            ras = shift_slices(runs_of_ones(ras_local), air.slice.start)
            # Assume that the reaction takes place during the TCAS RA period:
            for ra in ras:
                ap_off = ap_offs.get_next(ra.start, within_slice=ra)
                if not ap_off:
                    continue
                index = ap_off.index
                duration = (index - ra.start) / self.frequency
                self.create_kpv(index, duration)


class TCASFailureDuration(KeyPointValueNode):
    '''
    Duration for which the TCAS system was reporting failure.
    '''

    name = 'TCAS Failure Duration'
    units = ut.SECOND

    def derive(self, tcas_failure=M('TCAS Failure'),
               airborne=S('Airborne')):
        self.create_kpvs_where(tcas_failure.array == 'Failed',
                               tcas_failure.hz, phase=airborne)


##############################################################################
# Warnings: Takeoff Configuration


class TakeoffConfigurationWarningDuration(KeyPointValueNode):
    '''
    Duration for which the takeoff configuration warning was active.
    '''

    units = ut.SECOND

    def derive(self, takeoff_warn=M('Takeoff Configuration Warning'),
               movement_starts=S('Movement Start'),
               liftoffs=S('Liftoff')):
        if not liftoffs:
            return
        movement_start = movement_starts.get_first()
        self.create_kpvs_where(
            takeoff_warn.array == 'Warning',
            takeoff_warn.hz,
            phase=slice(movement_start.index if movement_start else None,
                        liftoffs.get_first().index),
        )


class TakeoffConfigurationFlapWarningDuration(KeyPointValueNode):
    '''
    Duration for which the takeoff configuration - flap warning was active.
    '''

    units = ut.SECOND

    def derive(self, takeoff_warn=M('Takeoff Configuration Flap Warning'),
               movement_starts=S('Movement Start'),
               liftoffs=S('Liftoff')):
        if not liftoffs:
            return
        movement_start = movement_starts.get_first()
        self.create_kpvs_where(
            takeoff_warn.array == 'Warning',
            takeoff_warn.hz,
            phase=[slice(movement_start.index if movement_start else None,
                         liftoffs.get_first().index)],
        )


class TakeoffConfigurationParkingBrakeWarningDuration(KeyPointValueNode):
    '''
    Duration for which the takeoff configuration - parking brake warning was active.
    '''

    units = ut.SECOND

    def derive(self,
               takeoff_warn=M('Takeoff Configuration Parking Brake Warning'),
               movement_starts=S('Movement Start'),
               liftoffs=S('Liftoff')):
        if not liftoffs:
            return
        movement_start = movement_starts.get_first()
        self.create_kpvs_where(
            takeoff_warn.array == 'Warning',
            takeoff_warn.hz,
            phase=[slice(movement_start.index if movement_start else None,
                         liftoffs.get_first().index)],
        )


class TakeoffConfigurationSpoilerWarningDuration(KeyPointValueNode):
    '''
    Duration for which the takeoff configuration - spoilers warning was active.
    '''

    units = ut.SECOND

    def derive(self,
               takeoff_warn=M('Takeoff Configuration Spoiler Warning'),
               movement_starts=S('Movement Start'),
               liftoffs=S('Liftoff')):
        if not liftoffs:
            return
        movement_start = movement_starts.get_first()
        self.create_kpvs_where(
            takeoff_warn.array == 'Warning',
            takeoff_warn.hz,
            phase=[slice(movement_start.index if movement_start else None,
                         liftoffs.get_first().index)],
        )


class TakeoffConfigurationStabilizerWarningDuration(KeyPointValueNode):
    '''
    Duration for which the takeoff configuration - stabilizer warning was active.
    '''

    units = ut.SECOND

    def derive(self,
               takeoff_warn=M('Takeoff Configuration Stabilizer Warning'),
               movement_starts=S('Movement Start'),
               liftoffs=S('Liftoff')):
        if not liftoffs:
            return
        movement_start = movement_starts.get_first()
        self.create_kpvs_where(
            takeoff_warn.array == 'Warning',
            takeoff_warn.hz,
            phase=[slice(movement_start.index if movement_start else None,
                         liftoffs.get_first().index)],
        )


##############################################################################
# Warnings: Takeoff Configuration


class LandingConfigurationGearWarningDuration(KeyPointValueNode):
    '''
    Duration for which the landing configuration - gear warning was active.
    '''

    units = ut.SECOND

    def derive(self,
               landing_cfg_warn=M('Landing Configuration Gear Warning'),
               airs=S('Airborne')):
        self.create_kpvs_where(landing_cfg_warn.array == 'Warning',
                               landing_cfg_warn.hz, phase=airs)


class LandingConfigurationSpeedbrakeCautionDuration(KeyPointValueNode):
    '''
    Duration for which the landing configuration - speedbrake caution was active.
    '''

    units = ut.SECOND

    def derive(self,
               landing_cfg_caution=M(
                   'Landing Configuration Speedbrake Caution'),
               airs=S('Airborne')):
        self.create_kpvs_where(landing_cfg_caution.array == 'Caution',
                               landing_cfg_caution.hz, phase=airs)


##############################################################################
# Warnings: Smoke


class SmokeWarningDuration(KeyPointValueNode):
    '''
    Duration for which the Smoke Warning was active.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return 'Smoke Warning' in available

    def derive(self, smoke_warning=M('Smoke Warning')):
        self.create_kpvs_where(smoke_warning.array == 'Smoke', self.hz)


##############################################################################
# Throttle


class ThrottleCyclesDuringFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of throttle lever movement that exceed
    10 deg peak to peak and with a maximum cycle period of 14 seconds during
    the final approach phase.
    '''

    units = ut.CYCLES

    def derive(self, levers=P('Throttle Levers'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                levers.array[fin_app.slice],
                10.0, 10.0, levers.hz,
                fin_app.slice.start,
            ))


class ThrottleLeverAtLiftoff(KeyPointValueNode):
    '''
    Angle of the Throttle Levers at liftoff
    '''

    units = ut.DEGREE

    def derive(self, levers=P('Throttle Levers'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(levers.array, liftoffs)


class ThrottleLeverVariationAbove80KtsToTakeoff(KeyPointValueNode):
    '''
    KPV indicating any throttle lever variation during the takeoff phase as
    airspeed is increases above 80 Kt. The value will by the delta between
    throttle lever change and the throttle lever at 80 Kt airspeed.
    '''

    units = ut.DEGREE

    def derive(self, levers=P('Throttle Levers'), speed=P('Airspeed'),
               takeoffs=S('Takeoff Roll Or Rejected Takeoff')):
        for takeoff in takeoffs:
            index_80kt = index_at_value(speed.array, 80, takeoff.slice)
            lever_80kt = value_at_index(levers.array, index_80kt)
            index, value = max_abs_value(levers.array - lever_80kt,
                                         slice(index_80kt, takeoff.slice.stop))
            self.create_kpv(index, value)


##############################################################################
# Thrust Asymmetry


class ThrustAsymmetryDuringTakeoffMax(KeyPointValueNode):
    '''
    Maximum thrust asymmetry during takeoff.
    '''

    units = ut.PERCENT

    def derive(self, ta=P('Thrust Asymmetry'),
               takeoff_rolls=S('Takeoff Roll Or Rejected Takeoff')):

        self.create_kpvs_within_slices(ta.array, takeoff_rolls, max_value)


class ThrustAsymmetryDuringFlightMax(KeyPointValueNode):
    '''
    Maximum thrust asymmetry while airborne.
    '''

    units = ut.PERCENT

    def derive(self, ta=P('Thrust Asymmetry'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(ta.array, airborne, max_value)


class ThrustAsymmetryDuringGoAroundMax(KeyPointValueNode):
    '''
    Maximum thrust asymmetry during a go around.
    '''

    units = ut.PERCENT

    def derive(self, ta=P('Thrust Asymmetry'),
               go_arounds=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(ta.array, go_arounds, max_value)


class ThrustAsymmetryDuringApproachMax(KeyPointValueNode):
    '''
    Peak thrust asymmetry on approach. 
    '''

    units = ut.PERCENT

    def derive(self, ta=P('Thrust Asymmetry'),
               approaches=S('Approach')):

        self.create_kpvs_within_slices(ta.array, approaches, max_value)


class ThrustAsymmetryWithThrustReversersDeployedMax(KeyPointValueNode):
    '''
    Peak thrust asymmetry with thrust reversers deployed.
    '''

    units = ut.PERCENT

    def derive(self, ta=P('Thrust Asymmetry'), tr=M('Thrust Reversers'),
               mobile=S('Mobile')):
        # Note: Inclusion of the 'Mobile' phase ensures use of thrust reverse
        #       late on the landing run is included, but corrupt data at engine
        #       start etc. should be rejected.
        # Note: Use not 'Stowed' as 'In Transit' implies partially 'Deployed':
        slices = clump_multistate(tr.array, 'Stowed', mobile.get_slices(),
                                  condition=False)
        # This KPV can trigger many times if the thrust reverser signal
        # toggles. This has been seen to happen after electrical power loss,
        # and as it is not possible for the thrust reversers to deploy and
        # retract within 2 seconds, small slices are removed here.
        slices = slices_remove_small_slices(slices, time_limit=2, hz=ta.hz)
        self.create_kpvs_within_slices(ta.array, slices, max_value)


class ThrustAsymmetryDuringApproachDuration(KeyPointValueNode):
    '''
    Durations of thrust asymmetry over 10%.
    '''

    units = ut.SECOND

    def derive(self, ta=P('Thrust Asymmetry'), approaches=S('Approach')):
        for approach in approaches:
            asymmetry = np.ma.masked_less(ta.array[approach.slice], 10.0)
            slices = np.ma.clump_unmasked(asymmetry)
            slices = shift_slices(slices, approach.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency)


class ThrustAsymmetryWithThrustReversersDeployedDuration(KeyPointValueNode):
    '''
    Durations of thrust asymmetry over 10% with reverse thrust operating.
    '''

    units = ut.SECOND

    def derive(self,
               ta=P('Thrust Asymmetry'),
               tr=M('Thrust Reversers'),
               mobile=S('Mobile')):

        # Note: Inclusion of the 'Mobile' phase ensures use of thrust reverse
        #       late on the landing run is included, but corrupt data at engine
        #       start etc. should be rejected.
        slices = [s.slice for s in mobile]
        # Note: Use not 'Stowed' as 'In Transit' implies partially 'Deployed':
        slices = clump_multistate(tr.array, 'Stowed', slices, condition=False)
        for slice_ in slices:
            asymmetry = np.ma.masked_less(ta.array[slice_], 10.0)
            slices = np.ma.clump_unmasked(asymmetry)
            slices = slices_remove_small_slices(slices, time_limit=2, hz=ta.hz)
            slices = shift_slices(slices, slice_.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency)


##############################################################################


class TouchdownToElevatorDownDuration(KeyPointValueNode):
    '''
    Time between touchdown and at least 10 degrees elevator angle change.
    '''

    units = ut.SECOND

    def derive(self,
               airspeed=P('Airspeed'),
               elevator=P('Elevator'),
               tdwns=KTI('Touchdown'),
               lands=S('Landing')):
        '''
        Originally introduced to monitor pilot actions on landing, the first
        version of this algorithm triggered only on -14deg elevator setting, to
        suit the Boeing 757 aircraft type.
    
        This was amended to monitor the time to maximum elevator down should the
        14 deg threshold not be met, in any case requiring at least 10 deg change
        in elevator to indicate a significant removal of lift.
        '''
        for land in lands:
            for tdwn in tdwns:
                if not is_index_within_slice(tdwn.index, land.slice):
                    continue
                to_scan = slice(tdwn.index, land.slice.stop)

                index_elev = index_at_value(elevator.array, -14.0, to_scan)
                if index_elev:
                    t_14 = (index_elev - tdwn.index) / elevator.frequency
                    self.create_kpv(index_elev, t_14)

                else:
                    index_min = tdwn.index + np.ma.argmin(elevator.array[to_scan])
                    if index_min > tdwn.index + 2:
                        # Worth having a look
                        if np.ma.ptp(elevator.array[tdwn.index:index_min]) > 10.0:
                            t_min = (index_min - tdwn.index) / elevator.frequency
                            self.create_kpv(index_min, t_min)
                    else:
                        # Nothing useful to do.
                        pass


class TouchdownTo60KtsDuration(KeyPointValueNode):
    '''
    Time between the point of touchdown and aircraft reaching 60kts IAS.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        return all_of(('Airspeed', 'Touchdown'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               groundspeed=P('Groundspeed'),
               tdwns=KTI('Touchdown')):
        if groundspeed:
            speed = groundspeed.array
            freq = groundspeed.frequency
        else:
            speed = airspeed.array
            freq = airspeed.frequency

        for tdwn in tdwns:
            index_60kt = index_at_value(speed, 60.0, slice(tdwn.index, None))
            if index_60kt:
                t__60kt = (index_60kt - tdwn.index) / freq
                self.create_kpv(index_60kt, t__60kt)

class TouchdownToPitch2DegreesAbovePitchAt60KtsDuration(KeyPointValueNode):
    '''
    Return the duration between Touchdown and a point at which the pitch is 2
    degrees more than the pitch when the aircraft has reached an airspeed
    of 60 Kts.
    '''
    units = ut.SECOND

    can_operate = aeroplane_only

    def derive(self, pitch=P('Pitch'), airspeed=P('Airspeed'),
               tdwns=KTI('Touchdown')):
        for tdwn in tdwns:
            # get index where airspeed is 60 Kts
            air_spd_end = index_at_value(airspeed.array.data, 60,
                                         slice(tdwn.index, None))
            pitch_at_60 = value_at_index(pitch.array, air_spd_end)
            if not pitch_at_60:
                continue
            pitch_ref = pitch_at_60 + 2
            stop = index_at_value(pitch.array, pitch_ref,
                                  slice(air_spd_end, tdwn.index, -1))
            if stop:
                duration = (stop - tdwn.index) / pitch.frequency
                self.create_kpv(stop, duration)


##############################################################################
# Turbulence


class TurbulenceDuringApproachMax(KeyPointValueNode):
    '''
    Turbulence, measured as the Root Mean Squared (RMS) of the Vertical
    Acceleration, during Approach.
    '''

    units = ut.G

    def derive(self,
               turbulence=P('Turbulence'),
               approaches=S('Approach')):

        self.create_kpvs_within_slices(turbulence.array, approaches, max_value)


class TurbulenceDuringCruiseMax(KeyPointValueNode):
    '''
    Turbulence, measured as the Root Mean Squared (RMS) of the Vertical
    Acceleration, while in Cruise.
    '''

    units = ut.G

    def derive(self,
               turbulence=P('Turbulence'),
               cruises=S('Cruise')):

        self.create_kpvs_within_slices(turbulence.array, cruises, max_value)


class TurbulenceDuringFlightMax(KeyPointValueNode):
    '''
    Turbulence, measured as the Root Mean Squared (RMS) of the Vertical
    Acceleration, while Airborne.
    '''

    units = ut.G

    def derive(self,
               turbulence=P('Turbulence'),
               airborne=S('Airborne')):
        for air in airborne.get_slices():
            # Restrict airborne a little to ensure doesn't trigger at touchdown
            self.create_kpvs_within_slices(
                turbulence.array, [slice(air.start + 5, air.stop - 5)], max_value)


##############################################################################
# Wind


class WindSpeedAtAltitudeDuringDescent(KeyPointValueNode):
    '''
    Wind speed at various altitudes (all AAL) during descent.
    '''

    NAME_FORMAT = 'Wind Speed At %(altitude)d Ft During Descent'
    NAME_VALUES = {'altitude': [2000, 1500, 1000, 500, 100, 50]}
    units = ut.KT

    def derive(self,
               alt_aal=P('Altitude AAL For Flight Phases'),
               wind_spd=P('Wind Speed')):
        '''
        Note: We align to Altitude AAL for cosmetic reasons; alignment to wind
          speed leads to slightly misaligned KPVs for wind speed, which looks
          wrong although is arithmetically "correct".
        '''

        for descent in alt_aal.slices_from_to(2100, 0):
            for altitude in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, altitude, descent)
                if not index:
                    continue
                value = value_at_index(wind_spd.array, index)
                if value:
                    self.create_kpv(index, value, altitude=altitude)


class WindDirectionAtAltitudeDuringDescent(KeyPointValueNode):
    '''
    Wind direction at altitudes (all AAL) during descent.
    '''

    NAME_FORMAT = 'Wind Direction At %(altitude)d Ft During Descent'
    NAME_VALUES = {'altitude': [2000, 1500, 1000, 500, 100, 50]}
    units = ut.DEGREE

    def derive(self,
               alt_aal=P('Altitude AAL For Flight Phases'),
               wind_dir=P('Wind Direction Continuous')):

        for descent in alt_aal.slices_from_to(2100, 0):
            for altitude in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, altitude, descent)
                if not index:
                    continue
                # Check direction not masked before using % 360:
                value = value_at_index(wind_dir.array, index)
                if value:
                    self.create_kpv(index, value % 360.0, altitude=altitude)


class WindAcrossLandingRunwayAt50Ft(KeyPointValueNode):
    '''
    Wind across landing runway at 50ft AAL.
    '''

    units = ut.KT

    def derive(self,
               walr=P('Wind Across Landing Runway'),
               landings=S('Landing')):
        for landing in landings:
            index = landing.slice.start  # Landing starts at 50ft!
            value = walr.array[index]
            if value is not None:
                self.create_kpv(index, value)


class WindSpeedInCriticalAzimuth(KeyPointValueNode):
    '''
    Maximum relative windspeed when wind blowing into tail rotor.
    The critical direction is helicopter type-specific.
    '''

    align_frequency = 2
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, wind_spd=P('Wind Speed'), wind_dir=P('Wind Direction'),
               tas=P('Airspeed True'), heading=P('Heading'),
               airborne=S('Airborne')):

        # Puma AS330 critical arc is the port quarter
        min_arc = 180
        max_arc = 270

        rad_scale = np.radians(1.0)
        headwind = tas.array + wind_spd.array * np.ma.cos((wind_dir.array-heading.array)*rad_scale)
        sidewind = wind_spd.array * np.ma.sin((wind_dir.array-heading.array)*rad_scale)

        app_dir = np.arctan2(sidewind, headwind)/rad_scale%360
        critical_dir = np.ma.masked_outside(app_dir, min_arc, max_arc)
        app_speed = np.ma.sqrt(sidewind*sidewind + headwind*headwind)
        critical_speed = np.ma.array(data=app_speed.data, mask=critical_dir.mask)

        self.create_kpvs_within_slices(critical_speed, airborne, max_value)



##############################################################################
# Weight


class GrossWeightAtLiftoff(KeyPointValueNode):
    '''
    Gross weight of the aircraft at liftoff.

    We use smoothed gross weight data for better accuracy.
    '''

    units = ut.KG

    def derive(self, gw=P('Gross Weight Smoothed'), liftoffs=KTI('Liftoff')):
        try:
            # TODO: Things to consider related to gross weight:
            #       - Does smoothed gross weight need to be repaired?
            #       - What should the duration be? Vref Lookup uses 130...
            #       - Should we extrapolate values as we do for Vref Lookup?
            array = repair_mask(gw.array, repair_duration=None)
        except ValueError:
            self.warning("KPV '%s' will not be created because '%s' array "
                         "could not be repaired.", self.name, gw.name)
            return
        self.create_kpvs_at_ktis(array, liftoffs)


class GrossWeightAtTouchdown(KeyPointValueNode):
    '''
    Gross weight of the aircraft at touchdown.

    We use smoothed gross weight data for better accuracy.
    '''

    units = ut.KG

    def derive(self, gw=P('Gross Weight Smoothed'), 
               touchdowns=KTI('Touchdown'),
               touch_and_go=KTI('Touch And Go')):
        try:
            # TODO: Things to consider related to gross weight:
            #       - Does smoothed gross weight need to be repaired?
            #       - What should the duration be? Vref Lookup uses 130...
            #       - Should we extrapolate values as we do for Vref Lookup?
            array = repair_mask(gw.array, repair_duration=None)
        except ValueError:
            self.warning("KPV '%s' will not be created because '%s' array "
                         "could not be repaired.", self.name, gw.name)
            return
        self.create_kpvs_at_ktis(array, (touchdowns+touch_and_go))


class GrossWeightConditionalAtTouchdown(KeyPointValueNode):
    '''
    Gross weight of the aircraft at touchdown if certain criteria are met.
    Requested for Airbus aircraft which have maintenance actions for
    overweight landings only if hi rate of descent or hi g at landing

    We use smoothed gross weight data for better accuracy.
    '''

    units = ut.KG

    @classmethod
    def can_operate(cls, available, manufacturer=A('Manufacturer')):
        required_params = ('Gross Weight At Touchdown',
                           'Acceleration Normal At Touchdown',
                           'Rate Of Descent At Touchdown')
        return all_of(required_params, available) \
            and manufacturer and manufacturer.value == 'Airbus'

    def derive(self, gw_kpv=KPV('Gross Weight At Touchdown'),
               acc_norm_kpv=KPV('Acceleration Normal At Touchdown'),
               rod_kpv=KPV('Rate Of Descent At Touchdown')):

        acc_norm_limit = 1.7
        vrt_spd_limit = -360 # negative as rate of descent

        if not gw_kpv:
            self.warning('No Gross Weight At Touchdown KPVs')
            return
        
        for gw in gw_kpv:
            acc = [k for k in acc_norm_kpv if abs(int(gw.index) - int(k.index)) <= 1 ]
            hi_g = acc and acc[0].value and acc[0].value > acc_norm_limit

            rod = [k for k in rod_kpv if abs(int(gw.index) - int(k.index)) <= 1 ]
            hi_rod = rod and rod[0].value and rod[0].value < vrt_spd_limit # less than as descending

            if hi_g or hi_rod: # if either condition matched create KPV.
                self.create_kpv(gw.index, gw.value)            


class GrossWeightDelta60SecondsInFlightMax(KeyPointValueNode):
    '''
    Measure the maximum change of gross weight over a one minute window. This is
    primarily to detect manual adjustments of the aircraft gross wieght during
    the flight by the crew. In order to capture lots of little adjustments
    together, a one minute window is used.
    '''

    align_frequency = 1  # force to 1Hz for 60 second measurements
    units = ut.KG

    def derive(self, gross_weight=P('Gross Weight'), airborne=S('Airborne')):
        # use the recorded un-smoothed gross weight measurements
        gw_repaired = repair_mask(gross_weight.array)
        weight_diff = np.ma.ediff1d(gw_repaired[::60])
        for in_air in airborne:
            # working in 60th samples
            in_air_start = (in_air.slice.start / 60.0) if in_air.slice.start else None
            in_air_stop = (in_air.slice.stop / 60.0) if in_air.slice.stop else None
            in_air_slice = slice(in_air_start, in_air_stop)
            max_diff = max_abs_value(weight_diff, _slice=in_air_slice)
            if max_diff.index is None:
                continue
            # narrow down the index to the maximum change in this region of flight
            max_diff_slice = slice(max_diff.index * 60,
                                   (max_diff.index + 1) * 60)
            index, _ = max_abs_value(gw_repaired, _slice=max_diff_slice)
            self.create_kpv(index, max_diff.value)


##############################################################################
# Dual Input


class DualInputWarningDuration(KeyPointValueNode):
    '''
    Duration of dual input warning discrete parameter as recorded on aircraft.

    We only look for dual input from the start of the first takeoff roll until
    the end of the last landing roll. This KPV is used to detect occurrences of
    dual input by either pilot irrespective of who was flying.

    This does not include looking at dual inputs during a rejected takeoff.
    '''

    units = ut.SECOND

    def derive(self,
               dual=M('Dual Input Warning'),
               takeoff_rolls=S('Takeoff Roll'),
               landing_rolls=S('Landing Roll')):

        start = takeoff_rolls.get_first().slice.start
        stop = landing_rolls.get_last().slice.stop
        phase = slice(start, stop)
        condition = dual.array == 'Dual'
        self.create_kpvs_where(condition, dual.hz, phase)


class DualInputAbove200FtDuration(KeyPointValueNode):
    '''
    Duration of dual input above 200 ft AAL.

    We only look for dual input from the start of the first takeoff roll until
    the end of the last landing roll. This KPV is used to detect occurrences of
    dual input above 200 ft AAL by either pilot irrespective of who was flying.

    Makes use of Dual Input parameter which is derived from at least 2.0
    degrees sidestick deflection by non-pilot flying for a minimum of 3
    seconds.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''

    units = ut.SECOND

    def derive(self,
               dual=M('Dual Input'),
               alt_aal=P('Altitude AAL')):
        phase = alt_aal.slices_above(200)
        condition = dual.array == 'Dual'
        self.create_kpvs_where(condition, dual.hz, phase)


class DualInputBelow200FtDuration(KeyPointValueNode):
    '''
    Duration of dual input below 200 ft AAL.

    We only look for dual input from the start of the first takeoff roll until
    the end of the last landing roll. This KPV is used to detect occurrences of
    dual input below 200 ft AAL by either pilot irrespective of who was flying.

    Makes use of Dual Input parameter which is derived from at least 2.0
    degrees sidestick deflection by non-pilot flying for a minimum of 3
    seconds.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''

    units = ut.SECOND

    def derive(self,
               dual=M('Dual Input'),
               alt_aal=P('Altitude AAL'),
               takeoff_rolls=S('Takeoff Roll'),
               landing_rolls=S('Landing Roll')):

        start = takeoff_rolls.get_first().slice.start
        stop = landing_rolls.get_last().slice.stop
        phase = slices_and([slice(start, stop)], alt_aal.slices_below(200))
        condition = dual.array == 'Dual'
        self.create_kpvs_where(condition, dual.hz, phase)


class DualInputByCaptDuration(KeyPointValueNode):
    '''
    Duration of dual input by the captain with first officer flying.

    We only look for dual input from the start of the first takeoff roll until
    the end of the last landing roll. This KPV is used to detect occurrences of
    dual input by the captain when the first officer was the pilot flying.

    Makes use of Dual Input parameter which is derived from at least 2.0
    degrees sidestick deflection by non-pilot flying for a minimum of 3
    seconds.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''

    units = ut.SECOND

    def derive(self,
               dual=M('Dual Input'),
               pilot=M('Pilot Flying'),
               takeoff_rolls=S('Takeoff Roll'),
               landing_rolls=S('Landing Roll')):

        start = takeoff_rolls.get_first().slice.start
        stop = landing_rolls.get_last().slice.stop
        phase = slice(start, stop)
        condition = (dual.array == 'Dual') & (pilot.array == 'First Officer')
        self.create_kpvs_where(condition, dual.hz, phase)


class DualInputByFODuration(KeyPointValueNode):
    '''
    Duration of dual input by the first officer with captain flying.

    We only look for dual input from the start of the first takeoff roll until
    the end of the last landing roll. This KPV is used to detect occurrences of
    dual input by the first officer when the captain was the pilot flying.

    Makes use of Dual Input parameter which is derived from at least 2.0
    degrees sidestick deflection by non-pilot flying for a minimum of 3
    seconds.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''

    name = 'Dual Input By FO Duration'
    units = ut.SECOND

    def derive(self,
               dual=M('Dual Input'),
               pilot=M('Pilot Flying'),
               takeoff_rolls=S('Takeoff Roll'),
               landing_rolls=S('Landing Roll')):

        start = takeoff_rolls.get_first().slice.start
        stop = landing_rolls.get_last().slice.stop
        phase = slice(start, stop)
        condition = (dual.array == 'Dual') & (pilot.array == 'Captain')
        self.create_kpvs_where(condition, dual.hz, phase)


class DualInputByCaptMax(KeyPointValueNode):
    '''
    Maximum sidestick angle of captain whilst first officer flying.

    We only look for dual input from the start of the first takeoff roll until
    the end of the last landing roll. This KPV is used to detect occurrences of
    dual input by the captain when the first officer was the pilot flying.

    Makes use of Dual Input parameter which is derived from at least 2.0
    degrees sidestick deflection by non-pilot flying for a minimum of 3
    seconds.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''

    units = ut.DEGREE

    def derive(self,
               stick_capt=P('Sidestick Angle (Capt)'),
               dual=M('Dual Input'),
               pilot=M('Pilot Flying'),
               takeoff_rolls=S('Takeoff Roll'),
               landing_rolls=S('Landing Roll')):

        start = takeoff_rolls.get_first().slice.start
        stop = landing_rolls.get_last().slice.stop
        phase = slice(start, stop)
        condition = runs_of_ones((dual.array == 'Dual') & (pilot.array == 'First Officer'))
        dual_input_phases = slices_and((phase,), condition)
        self.create_kpvs_within_slices(
            stick_capt.array,
            dual_input_phases,
            max_value
        )


class DualInputByFOMax(KeyPointValueNode):
    '''
    Maximum sidestick angle of first officer whilst captain flying.

    We only look for dual input from the start of the first takeoff roll until
    the end of the last landing roll. This KPV is used to detect occurrences of
    dual input by the first officer when the captain was the pilot flying.

    Makes use of Dual Input parameter which is derived from at least 2.0
    degrees sidestick deflection by non-pilot flying for a minimum of 3
    seconds.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''

    name = 'Dual Input By FO Max'
    units = ut.DEGREE

    def derive(self,
               stick_fo=P('Sidestick Angle (FO)'),
               dual=M('Dual Input'),
               pilot=M('Pilot Flying'),
               takeoff_rolls=S('Takeoff Roll'),
               landing_rolls=S('Landing Roll')):

        start = takeoff_rolls.get_first().slice.start
        stop = landing_rolls.get_last().slice.stop
        phase = slice(start, stop)
        condition = runs_of_ones((dual.array == 'Dual') & (pilot.array == 'Captain'))
        dual_input_phases = slices_and((phase,), condition)
        self.create_kpvs_within_slices(
            stick_fo.array,
            dual_input_phases,
            max_value
        )


##############################################################################


class HoldingDuration(KeyPointValueNode):
    '''
    Duration for which the aircraft was in the holding pattern.
    '''

    units = ut.SECOND

    def derive(self, holds=S('Holding')):

        self.create_kpvs_from_slice_durations(holds, self.hz, mark='end')


##### TODO: Implement!
####class ControlForcesTimesThree(KeyPointValueNode):
####    def derive(self, x=P('Not Yet')):
####        return NotImplemented


##############################################################################


# NOTE: Python class name restriction: '2 Deg Pitch To 35 Ft Duration'
class TwoDegPitchTo35FtDuration(KeyPointValueNode):
    '''
    Time taken for aircraft to reach 35ft after rotating to 2 degrees pitch.
    '''

    name = '2 Deg Pitch To 35 Ft Duration'
    units = ut.SECOND

    def derive(self, two_deg_pitch_to_35ft=S('2 Deg Pitch To 35 Ft')):

        self.create_kpvs_from_slice_durations(
            two_deg_pitch_to_35ft,
            self.frequency,
            mark='midpoint',
        )


class LastFlapChangeToTakeoffRollEndDuration(KeyPointValueNode):
    '''
    Time between the last flap change during takeoff roll and the end of
    takeoff roll.

    The idea is that the flaps should not be changed when the aircraft has
    started accelerating down the runway.

    We detect the last change of flap selection during the takeoff roll phase
    and calculate the time between this instant and the end of takeoff roll.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) \
            and 'Takeoff Roll Or Rejected Takeoff' in available

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               rolls=S('Takeoff Roll Or Rejected Takeoff')):

        flap = flap_lever or flap_synth
        for roll in rolls:
            changes = find_edges(flap.array.raw, roll.slice, 'all_edges')
            if changes:
                roll_end = roll.slice.stop
                last_change = changes[-1]
                time_from_liftoff = (roll_end - last_change) / self.frequency
                self.create_kpv(last_change, time_from_liftoff)


class AirspeedMinusVMOMax(KeyPointValueNode):
    '''
    Maximum value of Airspeed relative to the Maximum Operating Speed (VMO).
    '''

    name = 'Airspeed Minus VMO Max'
    units = ut.KT

    @classmethod
    def can_operate(cls, available):

        return any_of(('VMO', 'VMO Lookup'), available) \
            and all_of(('Airborne', 'Airspeed'), available)

    def derive(self,
               airspeed=P('Airspeed'),
               vmo_record=P('VMO'),
               vmo_lookup=P('VMO Lookup'),
               airborne=S('Airborne')):
        '''
        Values of VMO are taken from recorded or derived values if available,
        otherwise we fall back to using a value from a lookup table.
     
        We also check to ensure that we have some valid samples in any recorded or
        derived parameter, otherwise, again, we fall back to lookup tables. To
        avoid issues with small samples of invalid data, we check that the area of
        data we are interested in has no masked values.
        '''
        phases = airborne.get_slices()

        vmo = first_valid_parameter(vmo_record, vmo_lookup, phases=phases)

        if vmo is None:
            self.array = np_ma_masked_zeros_like(airspeed.array)
            return

        self.create_kpvs_within_slices(
            airspeed.array - vmo.array,
            airborne,
            max_value,
        )


class MachMinusMMOMax(KeyPointValueNode):
    '''
    Maximum value of Mach relative to the Maximum Operating Mach (MMO).
    '''

    name = 'Mach Minus MMO Max'
    units = ut.MACH

    @classmethod
    def can_operate(cls, available):

        return any_of(('MMO', 'MMO Lookup'), available) \
            and all_of(('Airborne', 'Mach'), available)

    def derive(self,
               mach=P('Mach'),
               mmo_record=P('MMO'),
               mmo_lookup=P('MMO Lookup'),
               airborne=S('Airborne')):
        '''     
        Values of MMO are taken from recorded or derived values if available,
        otherwise we fall back to using a value from a lookup table.
      
        We also check to ensure that we have some valid samples in any recorded or
        derived parameter, otherwise, again, we fall back to lookup tables. To
        avoid issues with small samples of invalid data, we check that the area of
        data we are interested in has no masked values.
        '''
        phases = airborne.get_slices()

        mmo = first_valid_parameter(mmo_record, mmo_lookup, phases=phases)

        if mmo is None:
            self.array = np_ma_masked_zeros_like(mach.array)
            return

        self.create_kpvs_within_slices(
            mach.array - mmo.array,
            airborne,
            max_value,
        )


########################################
# Aircraft Energy


class KineticEnergyAtRunwayTurnoff(KeyPointValueNode):
    '''
    Kinetic energy measured at runway turnoff.
    '''

    units = ut.MJ

    def derive(self, turn_off=KTI('Landing Turn Off Runway'),
               kinetic_energy=P('Kinetic Energy')):

        self.create_kpvs_at_ktis(kinetic_energy.array, turn_off)


class AircraftEnergyWhenDescending(KeyPointValueNode):
    '''
    Aircraft Energy when Descending
    '''

    from analysis_engine.key_time_instances import AltitudeWhenDescending

    NAME_FORMAT = 'Aircraft Energy at %(height)s'
    NAME_VALUES = {'height': AltitudeWhenDescending.names()}

    units = ut.MJ

    def derive(self, aircraft_energy=P('Aircraft Energy'),
               altitude_when_descending=KTI('Altitude When Descending')):

        for altitude in altitude_when_descending:
            value = value_at_index(aircraft_energy.array, altitude.index)
            self.create_kpv(altitude.index, value, height=altitude.name)


class TakeoffRatingDuration(KeyPointValueNode):
    '''
    Duration for which takeoff power is in use. Measured between the start
    of the takeoff roll and 5 minutes after.
    '''

    def derive(self, toffs=S('Takeoff 5 Min Rating')):
        '''
        '''
        self.create_kpvs_from_slice_durations(
            toffs,
            self.frequency,
            mark='end'
        )


##############################################################################
# Temperature

class SATMax(KeyPointValueNode):
    '''
    Maximum recorded SAT.
    '''

    name = 'SAT Max'
    units = ut.CELSIUS

    def derive(self, sat=P('SAT')):
        self.create_kpv(*max_value(sat.array))


class SATMin(KeyPointValueNode):
    '''
    Minimum recorded SAT.
    '''

    name = 'SAT Min'
    units = ut.CELSIUS

    can_operate = helicopter_only

    def derive(self, 
               sat=P('SAT'),
               family=A('Family'),
               rotors_turning=S('Rotors Turning')):
        
        if family and family.value == 'S92':
            self.create_kpvs_within_slices(sat.array, rotors_turning, min_value)
        else:
            self.create_kpv(*min_value(sat.array))


class SATRateOfChangeMax(KeyPointValueNode):
    '''
    Peak rate of increase of SAT - specific to offshore helicopter operations to detect
    transit though gas plumes.
    '''

    name = 'SAT Rate Of Change Max'
    units = ut.CELSIUS

    can_operate = helicopter_only

    def derive(self, sat=P('SAT'), airborne=S('Airborne')):
        width = None if sat.frequency <= 0.25 else 4
        sat_roc = rate_of_change_array(sat.array, sat.frequency, width=width)
        self.create_kpv_from_slices(sat_roc, airborne, max_value)


##############################################################################
# Cruise Guide Indicator
class CruiseGuideIndicatorMax(KeyPointValueNode):
    '''
    Maximum CGI reading throughtout the whole record. (helicopter only)
    '''
    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, cgi=P('Cruise Guide'), airborne=S('Airborne')):
        self.create_kpv_from_slices(cgi.array, airborne, max_abs_value)


##############################################################################
# Training Mode
class TrainingModeDuration(KeyPointValueNode):
    '''
    Specific to the S92 helicopter, FADEC training mode used.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        # S92A case
        if ('Training Mode' in available) and \
           not(any_of(('Eng (1) Training Mode', 'Eng (2) Training Mode'), available)):
            return True
        # H225 case
        elif all_of(('Eng (1) Training Mode', 'Eng (2) Training Mode'), available) and \
            ('Training Mode' not in available) :
            return True
        # No other cases operational yet.
        else:
            return False

    def derive(self, trg=P('Training Mode'),
               trg1=P('Eng (1) Training Mode'),
               trg2=P('Eng (2) Training Mode'),
               ):

        # S92A case
        if trg:
            trg_slices = runs_of_ones(trg.array)
            frequency = trg.frequency
        else:
            # H225 case
            trg_slices = slices_or(runs_of_ones(trg1.array), runs_of_ones(trg2.array))
            frequency = trg1.frequency

        self.create_kpvs_from_slice_durations(trg_slices,
                                              frequency,
                                              min_duration=2.0,
                                              mark='start')

##############################################################################
# Hover height
class HoverHeightDuringOnshoreTakeoffMax(KeyPointValueNode):
    '''
    Maximum hover height, to monitor for safe hover operation.
    '''

    units = ut.FT

    can_operate = helicopter_only

    def derive(self, rad_alt=P('Altitude Radio'), offshore=M('Offshore'), hover=S('Hover'), toff=S('Takeoff')):
        phases = slices_and(runs_of_ones(offshore.array == 'Onshore'), hover.get_slices())
        for phase in phases:
            if toff.get(within_slice=phase, within_use='any'):
                self.create_kpvs_within_slices(rad_alt.array, [phase], max_value)


class HoverHeightDuringOffshoreTakeoffMax(KeyPointValueNode):
    '''
    Maximum hover height, to monitor for safe hover operation.
    '''

    units = ut.FT

    can_operate = helicopter_only

    def derive(self, rad_alt=P('Altitude Radio'), offshore=M('Offshore'), hover=S('Hover'), toff=S('Takeoff')):
        phases = slices_and(runs_of_ones(offshore.array == 'Offshore'), hover.get_slices())
        for phase in phases:
            if toff.get(within_slice=phase, within_use='any'):
                self.create_kpvs_within_slices(rad_alt.array, [phase], max_value)


##############################################################################
# Drift
class DriftAtTouchdown(KeyPointValueNode):
    '''
    Returns the value of drift at touchdown. 
    '''

    units = ut.DEGREE

    def derive(self, drift=P('Drift'), touchdown=KTI('Touchdown')):
        self.create_kpvs_at_ktis(drift.array, touchdown)
    
        
##############################################################################
# Engine Derate calculations

class EngN1TakeoffDerate(KeyPointValueNode):
    '''
    Specific to CFM56-5B engines
    '''
    units = ut.PERCENT
    
    name = 'Eng N1 Takeoff Derate'
    
    @classmethod
    def can_operate(cls, available, engine_series=A('Engine Series')):
        return engine_series and engine_series.value == 'CFM56-5B' and all_deps(cls, available)

    def derive(self, eng_n1=P('Eng (*) N1 Avg'),
               tat=P('TAT'), mach=P('Mach'),
               sage_toffs=KTI('SAGE Takeoff')):

        for toff in sage_toffs:
            index = toff.index
            if mach.array[index] > 0.3:
                # takeoff point mis-identified; ignore
                continue
            n1 = eng_n1.array[index]
            if n1 < 85 or n1 > 100:
                self.warning("Engine N1 '%s' outside range in takeoff derate computation", n1)
                continue
            theta = (tat.array[index] + 273.15) / 288.15
            corr_n1 = n1 / np.sqrt(theta)
            derate = 100.0 - corr_n1
            self.create_kpv(index, derate)


class EngThrustTakeoffDerate(KeyPointValueNode):
    '''
    Specific to CFM56-5B engines
    '''

    units = ut.LBF

    def derive(self, mach=P('Mach'),
               n1_derates=KPV('Eng N1 Takeoff Derate')):

        def corrected_parameter_power_management(corr_n1, mach_toff):
            # See blog article about curve fitting
            fnk = -94514.4 \
                -20501.84 * mach_toff \
                -7.7018523 * corr_n1**2.0 \
                + 2027.295 * corr_n1 \
                -0.01571593 * mach_toff * corr_n1
            return fnk

        for n1 in n1_derates:
            index = n1.index
            corr_n1 = 100.0 - n1.value
            mach_toff = mach.array[index]
            fnk_90 = corrected_parameter_power_management(90.0, mach_toff)
            fnk = corrected_parameter_power_management(corr_n1, mach_toff)
            derate = 100.0 * (1.0 - fnk / fnk_90)
            self.create_kpv(index, derate)

    
class EngTakeoffFlexTemp(KeyPointValueNode):
    '''
    '''

    units = ut.CELSIUS
    
    @classmethod
    def can_operate(cls, available):
        return all_of(['Flex Temp', 'SAGE Takeoff'], available) or \
               all_of(['Eng (1) Flex Temp', 'Eng (2) Flex Temp', 'SAGE Takeoff'], available)
    
    def derive(self, sage_toffs=KTI('SAGE Takeoff'),
               flex=P('Flex Temp'),
               flex_1=P('Eng (1) Flex Temp'), flex_2=P('Eng (2) Flex Temp'),
               ):
        
        if flex:
            self.create_kpvs_at_ktis(flex.array, sage_toffs)
        elif flex_1 and flex_2:
            for toff in sage_toffs:
                index = toff.index
                value = (flex_1.array[index] + flex_2.array[index]) / 2.0
                self.create_kpv(index, value)


class EngNpMaxDuringTakeoff(KeyPointValueNode):
    '''
    The maximum value of Eng (*) Np Max during Takeoff 5 Min Rating phase
    maintained for the specified duration
    '''
    
    NAME_FORMAT = 'Eng (*) Np Max During Takeoff %(seconds)d Sec'
    NAME_VALUES = {'seconds': [5, 20]}
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return all_of(('Takeoff 5 Min Rating', 'Eng (*) Np Max'), available)
    
    def derive(self, takeoffs=S('Takeoff 5 Min Rating'),
               eng_np_max=P('Eng (*) Np Max'),
               go_arounds=S('Go Around 5 Min Rating')):
        
        for duration in self.NAME_VALUES['seconds']:
            for takeoff in takeoffs.get_slices():
                arrays = eng_np_max.array[takeoff]
                samples = int(duration * eng_np_max.frequency)
                if len(arrays) > 0:
                    index, value = max_maintained_value(arrays, samples, takeoff)
                    if index is not None and value is not None:
                        self.create_kpv(index, value, seconds=duration)

        if go_arounds: 
            for duration in self.NAME_VALUES['seconds']:
                for go_around in go_arounds.get_slices():
                    arrays = eng_np_max.array[go_around]
                    samples = int(duration * eng_np_max.frequency)
                    if len(arrays) > 0:
                        index, value = max_maintained_value(arrays, samples, go_around)
                        if index is not None and value is not None:
                            self.create_kpv(index, value, seconds=duration)


class EngTorqueMaxDuringTakeoff(KeyPointValueNode):
    '''
    The maximum value of Eng (*) Torque Max during Takeoff 5 Min Rating phase
    maintained for the specified duration
    '''
    
    NAME_FORMAT = 'Eng (*) Torque Max During Takeoff %(durations)s'
    NAME_VALUES = {'durations': ['10 Sec', '20 Sec', '5 Min']}
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return all_of(('Eng (*) Torque Max', 'Takeoff 5 Min Rating'), available)
    
    def derive(self, eng_torq_max=P('Eng (*) Torque Max'),
               takeoffs=S('Takeoff 5 Min Rating'),
               go_arounds=S('Go Around 5 Min Rating')):
        
        seconds = np.array([10, 20, 300])
        for samples, duration in zip(seconds, self.NAME_VALUES['durations']):
            for takeoff in takeoffs.get_slices():
                arrays = eng_torq_max.array[takeoff]
                if len(arrays) > 0:
                    index, value = max_maintained_value(arrays, samples, takeoff)
                    if index is not None and value is not None:
                        self.create_kpv(index, value, durations=duration)
        if go_arounds:
            for samples, duration in zip(seconds, self.NAME_VALUES['durations']):
                for go_around in go_arounds.get_slices():
                    arrays = eng_torq_max.array[go_around]
                    if len(arrays) > 0:
                        index, value = max_maintained_value(arrays, samples, go_around)
                        if index is not None and value is not None:
                            self.create_kpv(index, value, durations=duration)


class EngTorqueMaxDuringMaximumContinuousPower(KeyPointValueNode):
    '''
    The maximum value of Eng (*) Torque Max during Maximum Continous Power phase 
    maintained for the specified duration
    '''
    
    NAME_FORMAT = 'Eng (*) Torque Max During Maximum Continuous Power %(durations)s'
    NAME_VALUES = {'durations': ['10 Sec', '20 Sec', '5 Min', '10 Min']}
    units = ut.PERCENT
    
    @classmethod
    def can_operate(cls, available):
        return all_of(('Eng (*) Torque Max', 'Maximum Continuous Power'), available)
    
    def derive(self,
               eng_torq_max=P('Eng (*) Torque Max'),
               ratings=S('Maximum Continuous Power')):
        
        seconds = np.array([10, 20, 300, 600])
        for samples, duration in zip(seconds, self.NAME_VALUES['durations']):
            for mcp in ratings.get_slices():
                arrays = eng_torq_max.array[mcp]
                if len(arrays) > 0:
                    index, value = max_maintained_value(arrays, samples, mcp)
                    if index is not None and value is not None:
                        self.create_kpv(index, value, durations=duration)
