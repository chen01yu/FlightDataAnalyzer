import numpy as np
from scipy.ndimage import filters
from scipy.signal import medfilt

from flightdatautilities import units as ut

from analysis_engine import settings

from analysis_engine.library import (
    all_deps,
    all_of,
    any_of,
    bearing_and_distance,
    cycle_finder,
    find_low_alts,
    filter_slices_duration,
    first_order_washout,
    first_valid_sample,
    heading_diff,
    index_at_value,
    is_index_within_slices,
    is_index_within_slice,
    is_slice_within_slice,
    last_valid_sample,
    max_value,
    moving_average,
    nearest_neighbour_mask_repair,
    peak_curvature,
    rate_of_change,
    rate_of_change_array,
    repair_mask,
    runs_of_ones,
    shift_slice,
    shift_slices,
    slices_above,
    slices_and,
    slices_and_not,
    slices_below,
    slices_from_to,
    slices_not,
    slices_or,
    slices_overlap,
    slices_remove_small_gaps,
    slices_remove_small_slices,
)

from analysis_engine.node import (
    A, App, FlightPhaseNode, P, S, KTI, KPV, M,
    aeroplane, aeroplane_only, helicopter, helicopter_only)

from analysis_engine.settings import (
    AIRBORNE_THRESHOLD_TIME,
    AIRSPEED_THRESHOLD,
    BOUNCED_LANDING_THRESHOLD,
    BOUNCED_MAXIMUM_DURATION,
    GROUNDSPEED_FOR_MOBILE,
    HEADING_RATE_FOR_FLIGHT_PHASES_FW,
    HEADING_RATE_FOR_FLIGHT_PHASES_RW,
    HEADING_RATE_FOR_MOBILE,
    HEADING_RATE_FOR_TAXI_TURNS,
    HEADING_TURN_OFF_RUNWAY,
    HEADING_TURN_ONTO_RUNWAY,
    HOLDING_MAX_GSPD,
    HOLDING_MIN_TIME,
    HYSTERESIS_FPALT_CCD,
    ILS_CAPTURE,
    INITIAL_CLIMB_THRESHOLD,
    LANDING_ROLL_END_SPEED,
    LANDING_THRESHOLD_HEIGHT,
    ROTORSPEED_THRESHOLD,
    TAKEOFF_ACCELERATION_THRESHOLD,
    VERTICAL_SPEED_FOR_CLIMB_PHASE,
    VERTICAL_SPEED_FOR_DESCENT_PHASE,

    AIRBORNE_THRESHOLD_TIME_RW,
    AUTOROTATION_SPLIT,
    HOVER_GROUNDSPEED_LIMIT,
    HOVER_HEIGHT_LIMIT,
    HOVER_MIN_DURATION,
    HOVER_MIN_HEIGHT,
    HOVER_TAXI_HEIGHT,
    LANDING_COLLECTIVE_PERIOD,
    LANDING_HEIGHT,
    LANDING_TRACEBACK_PERIOD,
    TAKEOFF_PERIOD,
    ROTOR_TRANSITION_ALTITUDE,
    ROTOR_TRANSITION_SPEED_LOW,
    ROTOR_TRANSITION_SPEED_HIGH,
)


class Airborne(FlightPhaseNode):
    '''
    Periods where the aircraft is in the air, includes periods where on the
    ground for short periods (touch and go).

    TODO: Review whether depending upon the "dips" calculated by Altitude AAL
    would be more sensible as this will allow for negative AAL values longer
    than the remove_small_gaps time_limit.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT'):
            return False
        else:
            return 'Altitude AAL For Flight Phases' in available

    def derive(self,
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast')):
        # Remove short gaps in going fast to account for aerobatic manoeuvres
        speedy_slices = slices_remove_small_gaps(fast.get_slices(),
                                                 time_limit=60, hz=fast.frequency)
    
        # Just find out when altitude above airfield is non-zero.
        for speedy in speedy_slices:
            # Stop here if the aircraft never went fast.
            if speedy.start is None and speedy.stop is None:
                break
    
            start_point = speedy.start or 0
            stop_point = speedy.stop or len(alt_aal.array)
            # Restrict data to the fast section (it's already been repaired)
            working_alt = alt_aal.array[start_point:stop_point]
    
            # Stop here if there is inadequate airborne data to process.
            if working_alt is None or np.ma.ptp(working_alt)==0.0:
                continue
            airs = slices_remove_small_gaps(
                    np.ma.clump_unmasked(np.ma.masked_less_equal(working_alt, 1.0)),
                    time_limit=40, # 10 seconds was too short for Herc which flies below 0  AAL for 30 secs.
                    hz=alt_aal.frequency)
            # Make sure we propogate None ends to data which starts or ends in
            # midflight.
            for air in airs:
                begin = air.start
                if begin + start_point == 0: # Was in the air at start of data
                    begin = None
                end = air.stop
                if end + start_point >= len(alt_aal.array): # Was in the air at end of data
                    end = None
                if begin is None or end is None:
                    self.create_phase(shift_slice(slice(begin, end),
                                                      start_point))
                else:
                    duration = end - begin
                    if (duration / alt_aal.hz) > AIRBORNE_THRESHOLD_TIME:
                        self.create_phase(shift_slice(slice(begin, end),
                                                          start_point))


class Autorotation(FlightPhaseNode):
    '''
    Look for at least 1% difference between the highest power turbine speed
    and the rotor speed.
    This is bound to happen in a descent, and we define the autorotation
    period as from the initial onset
    to the final establishment of normal operation.

    Note: For Autorotation KPV: Detect maximum Nr during the Autorotation phase.
    '''

    can_operate = helicopter_only

    def derive(self, max_n2=P('Eng (*) N2 Max'),
               nr=P('Nr'), descs=S('Descending')):
        for desc in descs:
            # Look for split in shaft speeds.
            delta = nr.array[desc.slice] - max_n2.array[desc.slice]
            split = np.ma.masked_less(delta, AUTOROTATION_SPLIT)
            split_ends = np.ma.clump_unmasked(split)
            if split_ends:
                self.create_phase(shift_slice(slice(split_ends[0].start,
                                                    split_ends[-1].stop ),
                                              desc.slice.start))


class GoAroundAndClimbout(FlightPhaseNode):
    '''
    We already know that the Key Time Instance has been identified at the
    lowest point of the go-around, and that it lies below the 3000ft
    approach thresholds. The function here is to expand the phase 500ft before
    to the first level off after (up to 2000ft above minimum altitude).

    Uses find_low_alts to exclude level offs and level flight sections, therefore
    approach sections may finish before reaching 2000 ft above the go around.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type'), ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return False
        correct_seg_type = seg_type and seg_type.value not in ('GROUND_ONLY', 'NO_MOVEMENT')
        return 'Altitude AAL For Flight Phases' in available and correct_seg_type

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               level_flights=S('Level Flight')):
        # Find the ups and downs in the height trace.
        level_flights = level_flights.get_slices() if level_flights else None
        low_alt_slices = find_low_alts(
            alt_aal.array, alt_aal.frequency, 3000,
            start_alt=500, stop_alt=2000,
            level_flights=level_flights,
            relative_start=True,
            relative_stop=True,
        )
        dlc_slices = []
        for low_alt in low_alt_slices:
            if (alt_aal.array[low_alt.start] and
                alt_aal.array[low_alt.stop - 1]):
                dlc_slices.append(low_alt)

        self.create_phases(dlc_slices)


class Holding(FlightPhaseNode):
    """
    Holding is a process which involves multiple turns in a short period,
    normally in the same sense. We therefore compute the average rate of turn
    over a long period to reject short turns and pass the entire holding
    period.

    Note that this is the only function that should use "Heading Increasing"
    as we are only looking for turns, and not bothered about the sense or
    actual heading angle.
    """

    can_operate = aeroplane_only

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               hdg=P('Heading Increasing'),
               lat=P('Latitude Smoothed'), lon=P('Longitude Smoothed')):
        _, height_bands = slices_from_to(alt_aal.array, 20000, 3000)
        # Three minutes should include two turn segments.
        turn_rate = rate_of_change(hdg, 3 * 60)
        for height_band in height_bands:
            if height_band.start >= height_band.stop:
                continue
            # We know turn rate will be positive because Heading Increasing only
            # increases.
            turn_bands = np.ma.clump_unmasked(
                np.ma.masked_less(turn_rate[height_band], 0.5))
            hold_bands=[]
            for turn_band in shift_slices(turn_bands, height_band.start):
                # Reject short periods and check that the average groundspeed was
                # low. The index is reduced by one sample to avoid overruns, and
                # this is fine because we are not looking for great precision in
                # this test.
                hold_sec = turn_band.stop - turn_band.start
                if (hold_sec > HOLDING_MIN_TIME*alt_aal.frequency):
                    start = turn_band.start
                    stop = turn_band.stop - 1
                    _, hold_dist = bearing_and_distance(
                        lat.array[start], lon.array[start],
                        lat.array[stop], lon.array[stop])
                    if ut.convert(hold_dist / hold_sec, ut.METER_S, ut.KT) < HOLDING_MAX_GSPD:
                        hold_bands.append(turn_band)

            self.create_phases(hold_bands)


class EngHotelMode(FlightPhaseNode):
    '''
    Some turbo props use the Engine 2 turbine to provide power and air whilst
    the aircraft is on the ground, a brake is applied to prevent the
    propellers from rotating
    '''

    @classmethod
    def can_operate(cls, available, family=A('Family'), ac_type=A('Aircraft Type')):
        return ac_type == aeroplane and all_deps(cls, available) and family.value in ('ATR-42', 'ATR-72') # Not all aircraft with Np will have a 'Hotel' mode


    def derive(self, eng2_np=P('Eng (2) Np'),
               eng1_n1=P('Eng (1) N1'),
               eng2_n1=P('Eng (2) N1'),
               groundeds=S('Grounded'),
               prop_brake=M('Propeller Brake')):
        pos_hotel = (eng2_n1.array > 45) & (eng2_np.array <= 0) & (eng1_n1.array < 40) & (prop_brake.array == 'On')
        hotel_mode = slices_and(runs_of_ones(pos_hotel), groundeds.get_slices())
        self.create_phases(hotel_mode)


class ApproachAndLanding(FlightPhaseNode):
    '''
    Approaches from 3000ft to lowest point in the approach (where a go around
    is performed) or down to and including the landing phase.

    Uses find_low_alts to exclude level offs and level flight sections, therefore
    approach sections may start below 3000 ft.

    Q: Suitable to replace this with BottomOfDescent and working back from
    those KTIs rather than having to deal with GoAround AND Landings?
    '''
    # Force to remove problem with desynchronising of approaches and landings
    # (when offset > 0.5)
    align_offset = 0


    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'START_ONLY'):
            return False
        elif ac_type == helicopter:
            return all_of(('Approach', 'Landing'), available)
        else:
            return 'Altitude AAL For Flight Phases' in available

    def _derive_aircraft(self, alt_aal, level_flights, landings):
        # Prepare to extract the slices
        level_flights = level_flights.get_slices() if level_flights else None

        low_alt_slices = slices_remove_small_slices(find_low_alts(
            alt_aal.array, alt_aal.frequency, 3000,
            stop_alt=0,
            level_flights=level_flights,
            cycle_size=500.0), 5, alt_aal.hz)

        for low_alt in low_alt_slices:
            if not alt_aal.array[low_alt.start]:
                # Exclude Takeoff.
                continue

            # Include the Landing period
            if landings:
                landing = landings.get_last()
                if is_index_within_slice(landing.slice.start, low_alt):
                    low_alt = slice(low_alt.start, landing.slice.stop)
            self.create_phase(low_alt)

    def _derive_helicopter(self, apps, landings):
        phases = []
        for new_phase in apps:
            phases = slices_or(phases, [new_phase.slice])
        for new_phase in landings:
            # The phase is extended to make sure we enclose the endpoint
            # so that the touchdown point is included in this phase.
            phases = slices_or(phases, [slice(new_phase.slice.start, new_phase.slice.stop+2)])
        self.create_phases(phases)

    def derive(self,
               ac_type=A('Aircraft Type'),
               # aircraft
               alt_aal=P('Altitude AAL For Flight Phases'),
               level_flights=S('Level Flight'),
               # helicopter
               apps=S('Approach'),
               # shared
               landings=S('Landing')):

        if ac_type == helicopter:
            self._derive_helicopter(apps, landings)
        else:
            self._derive_aircraft(alt_aal, level_flights, landings)


class Approach(FlightPhaseNode):
    """
    This separates out the approach phase excluding the landing.

    Includes all approaches such as Go Arounds, but does not include any
    climbout afterwards.

    Landing starts at 50ft, therefore this phase is until 50ft.
    Uses find_low_alts to exclude level offs and level flight sections.
    """
    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type'), ac_type=A('Aircraft Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'START_ONLY'):
            return False
        elif ac_type == helicopter:
            return all_of(('Altitude AGL', 'Altitude STD'), available)
        else:
            return 'Altitude AAL For Flight Phases' in available

    def _derive_aircraft(self, alt_aal, level_flights, landings):
        level_flights = level_flights.get_slices() if level_flights else None
        low_alts = find_low_alts(alt_aal.array, alt_aal.frequency, 3000,
                                 start_alt=3000, stop_alt=50,
                                 stop_mode='descent',
                                 level_flights=level_flights)
        for low_alt in low_alts:
            # Select landings only.
            if alt_aal.array[low_alt.start] and \
               alt_aal.array[low_alt.stop] and \
               alt_aal.array[low_alt.start] > alt_aal.array[low_alt.stop]:
                self.create_phase(low_alt)

    def _derive_helicopter(self, alt_agl, alt_std):
        _, apps = slices_from_to(alt_agl.array, 500, 100, threshold=1.0)
        for app, next_app in zip(apps, apps[1:]+[slice(None)]):
            begin = peak_curvature(alt_std.array,
                                   _slice=slice(app.start, app.start - 300 * alt_std.frequency, -1),
                                   curve_sense='Convex')
            end = index_at_value(alt_agl.array, 0.0,
                                 _slice=slice(app.stop, next_app.start),
                                 endpoint='first_closing')
            if begin and end:
                self.create_phase(slice(begin, end))

    def derive(self,
               ac_type=A('Aircraft Type'),
               # aircraft
               alt_aal=P('Altitude AAL For Flight Phases'),
               level_flights=S('Level Flight'),
               landings=S('Landing'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               alt_std=P('Altitude STD')):
        if ac_type == helicopter:
            self._derive_helicopter(alt_agl, alt_std)
        else:
            self._derive_aircraft(alt_aal, level_flights, landings)


class BouncedLanding(FlightPhaseNode):
    '''
    TODO: Review increasing the frequency for more accurate indexing into the
    altitude arrays.

    Q: Should Airborne be first so we align to its offset?
    '''
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               airs=S('Airborne'),
               fast=S('Fast')):
        for speedy in fast:
            for air in airs:
                if slices_overlap(speedy.slice, air.slice):
                    start = air.slice.stop
                    stop = speedy.slice.stop
                    if (stop - start) / self.frequency > BOUNCED_MAXIMUM_DURATION:
                        # duration too long to be a bounced landing!
                        # possible cause: Touch and go.
                        continue
                    elif stop < start:
                        # Possible condition for helicopters
                        continue
                    elif start == len(alt_aal.array):
                        # Mid-flight segments.
                        continue
                    elif start == stop:
                        stop += 1

                    scan = alt_aal.array[start:stop]
                    ht = max(scan)
                    if ht > BOUNCED_LANDING_THRESHOLD:
                        #TODO: Input maximum BOUNCE_HEIGHT check?
                        up = np.ma.clump_unmasked(np.ma.masked_less_equal(scan,
                                                                          0.0))
                        self.create_phase(
                            shift_slice(slice(up[0].start, up[-1].stop), start))


class ClimbCruiseDescent(FlightPhaseNode):
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               airs=S('Airborne')):
        for air in airs:
            try:
                alts = repair_mask(alt_std.array[air.slice], repair_duration=None)
            except:
                # Short segments may be wholly masked. We ignore these.
                continue
            # We squash the altitude signal above 10,000ft so that changes of
            # altitude to create a new flight phase have to be 10 times
            # greater; 500ft changes below 10,000ft are significant, while
            # above this 5,000ft is more meaningful.
            alt_squash = np.ma.where(
                alts > 10000, (alts - 10000) / 10.0 + 10000, alts)
            pk_idxs, pk_vals = cycle_finder(alt_squash,
                                            min_step=HYSTERESIS_FPALT_CCD)

            if pk_vals is not None:
                n = 0
                pk_idxs += air.slice.start or 0
                n_vals = len(pk_vals)
                while n < n_vals - 1:
                    pk_val = pk_vals[n]
                    pk_idx = pk_idxs[n]
                    next_pk_val = pk_vals[n + 1]
                    next_pk_idx = pk_idxs[n + 1]
                    if pk_val > next_pk_val:
                        # descending
                        self.create_phase(slice(None, next_pk_idx))
                        n += 1
                    else:
                        # ascending
                        # We are going upwards from n->n+1, does it go down
                        # again?
                        if n + 2 < n_vals:
                            if pk_vals[n + 2] < next_pk_val:
                                # Hurrah! make that phase
                                self.create_phase(slice(pk_idx,
                                                        pk_idxs[n + 2]))
                                n += 2
                        else:
                            self.create_phase(slice(pk_idx, None))
                            n += 1


"""
class CombinedClimb(FlightPhaseNode):
    '''
    Climb phase from liftoff or go around to top of climb
    '''
    def derive(self,
               toc=KTI('Top Of Climb'),
               ga=KTI('Go Around'),
               lo=KTI('Liftoff'),
               touchdown=KTI('Touchdown')):

        end_list = [x.index for x in toc.get_ordered_by_index()]
        start_list = [y.index for y in [lo.get_first()] + ga.get_ordered_by_index()]
        assert len(start_list) == len(end_list)

        slice_idxs = zip(start_list, end_list)
        for slice_tuple in slice_idxs:
            self.create_phase(slice(*slice_tuple))
"""


class Climb(FlightPhaseNode):
    '''
    This phase goes from 1000 feet (top of Initial Climb) in the climb to the
    top of climb
    '''
    def derive(self,
               toc=KTI('Top Of Climb'),
               eot=KTI('Climb Start')): # AKA End Of Initial Climb
        # First we extract the kti index values into simple lists.
        toc_list = []
        for this_toc in toc:
            toc_list.append(this_toc.index)

        # Now see which follows a takeoff
        for this_eot in eot:
            eot = this_eot.index
            # Scan the TOCs
            closest_toc = None
            for this_toc in toc_list:
                if (eot < this_toc and
                    ((closest_toc and this_toc < closest_toc)
                     or
                     closest_toc is None)):
                    closest_toc = this_toc
            # Build the slice from what we have found.
            if ((closest_toc - eot) / self.hz) <= 2:
                continue # skip brief climbs
            self.create_phase(slice(eot, closest_toc))


class Climbing(FlightPhaseNode):
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airs=S('Airborne')):
        # Climbing is used for data validity checks and to reinforce regimes.
        for air in airs:
            climbing = np.ma.masked_less(vert_spd.array[air.slice],
                                         VERTICAL_SPEED_FOR_CLIMB_PHASE)
            climbing_slices = slices_remove_small_gaps(
                np.ma.clump_unmasked(climbing), time_limit=30.0, hz=vert_spd.hz)
            self.create_phases(shift_slices(climbing_slices, air.slice.start))


class Cruise(FlightPhaseNode):
    def derive(self,
               ccds=S('Climb Cruise Descent'),
               tocs=KTI('Top Of Climb'),
               tods=KTI('Top Of Descent'),
               air_spd=P('Airspeed')):
        # We may have many phases, tops of climb and tops of descent at this
        # time.
        # The problem is that they need not be in tidy order as the lists may
        # not be of equal lengths.

        # ensure traveling greater than 50 kts in cruise
        # scope = slices_and(slices_above(air_spd.array, 50)[1], ccds.get_slices())
        scope = ccds.get_slices()
        for ccd in scope:
            toc = tocs.get_first(within_slice=ccd)
            if toc:
                begin = toc.index
            else:
                begin = ccd.start

            tod = tods.get_last(within_slice=ccd)
            if tod:
                end = tod.index
            else:
                end = ccd.stop

            # Some flights just don't cruise. This can cause headaches later
            # on, so we always cruise for at least one second !
            if None not in(end, begin) and end < begin + 1:
                end = begin + 1

            self.create_phase(slice(begin,end))


class InitialCruise(FlightPhaseNode):
    '''
    This is a period from five minutes into the cruise lasting for 30
    seconds, and is used to establish average conditions for fuel monitoring
    programmes.
    '''

    align_frequency = 1.0
    align_offset = 0.0

    can_operate = aeroplane_only

    def derive(self, cruises=S('Cruise')):
        if not cruises:
            return
        cruise = cruises[0].slice
        if cruise.stop - cruise.start > 330:
            self.create_phase(slice(cruise.start+300, cruise.start+330))

"""
class CombinedDescent(FlightPhaseNode):
    def derive(self,
               tod_set=KTI('Top Of Descent'),
               bod_set=KTI('Bottom Of Descent'),
               liftoff=KTI('Liftoff'),
               touchdown=KTI('Touchdown')):

        end_list = [x.index for x in bod_set.get_ordered_by_index()]
        start_list = [y.index for y in tod_set.get_ordered_by_index()]
        assert len(start_list) == len(end_list)

        slice_idxs = zip(start_list, end_list)
        for slice_tuple in slice_idxs:
            self.create_phase(slice(*slice_tuple))
"""

class Descending(FlightPhaseNode):
    """
    Descending faster than 500fpm towards the ground
    """
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airs=S('Airborne')):
        # Vertical speed limits of 500fpm gives good distinction with level
        # flight.
        for air in airs:
            descending = np.ma.masked_greater(vert_spd.array[air.slice],
                                              VERTICAL_SPEED_FOR_DESCENT_PHASE)
            desc_slices = slices_remove_small_slices(np.ma.clump_unmasked(descending))
            self.create_phases(shift_slices(desc_slices, air.slice.start))


class Descent(FlightPhaseNode):
    def derive(self,
               tod_set=KTI('Top Of Descent'),
               bod_set=KTI('Bottom Of Descent')):

        start_list = [y.index for y in tod_set.get_ordered_by_index()]
        end_list = [x.index for x in bod_set.get_ordered_by_index()]
        assert len(start_list) == len(end_list)

        slice_idxs = zip(start_list, end_list)
        for slice_tuple in slice_idxs:
            self.create_phase(slice(*slice_tuple))


class DescentToFlare(FlightPhaseNode):
    '''
    Descent phase down to 50ft.
    '''

    def derive(self,
            descents=S('Descent'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        #TODO: Ensure we're still in the air
        for descent in descents:
            end = index_at_value(alt_aal.array, 50.0, descent.slice)
            if end is None:
                end = descent.slice.stop
            self.create_phase(slice(descent.slice.start, end))


class DescentLowClimb(FlightPhaseNode):
    '''
    Finds where the aircaft descends below the INITIAL_APPROACH_THRESHOLD and
    then climbs out again - an indication of a go-around.

    TODO: Consider refactoring this based on the Bottom Of Descent KTIs and
    just check the altitude at each BOD.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type'), ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return False
        else:
            correct_seg_type = seg_type and seg_type.value not in ('GROUND_ONLY', 'NO_MOVEMENT')
            return 'Altitude AAL For Flight Phases' in available and correct_seg_type

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               level_flights=S('Level Flight')):
        level_flights = level_flights.get_slices() if level_flights else None
        low_alt_slices = find_low_alts(alt_aal.array, alt_aal.frequency,
                                       500,
                                       3000,
                                       level_flights=level_flights)
        for low_alt in low_alt_slices:
            if (alt_aal.array[low_alt.start] and
                alt_aal.array[low_alt.stop - 1]):
                self.create_phase(low_alt)


class Fast(FlightPhaseNode):
    '''
    Data will have been sliced into single flights before entering the
    analysis engine, so we can be sure that there will be only one fast
    phase. This may have masked data within the phase, but by taking the
    notmasked edges we enclose all the data worth analysing.

    Therefore len(Fast) in [0,1]

    TODO: Discuss whether this assertion is reliable in the presence of air data corruption.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return 'Nr' in available
        else:
            return 'Airspeed' in available

    def derive(self, airspeed=P('Airspeed'), rotor_speed=P('Nr'),
               ac_type=A('Aircraft Type')):
        """
        Did the aircraft go fast enough to possibly become airborne?

        # We use the same technique as in index_at_value where transition of
        # the required threshold is detected by summing shifted difference
        # arrays. This has the particular advantage that we can reject
        # excessive rates of change related to data dropouts which may still
        # pass the data validation stage.
        value_passing_array = (airspeed.array[0:-2]-AIRSPEED_THRESHOLD) * \
            (airspeed.array[1:-1]-AIRSPEED_THRESHOLD)
        test_array = np.ma.masked_outside(value_passing_array, 0.0, -100.0)
        """

        if ac_type == helicopter:
            nr = repair_mask(rotor_speed.array, repair_duration=600,
                             raise_entirely_masked=False)
            fast = np.ma.masked_less(nr, ROTORSPEED_THRESHOLD)
            fast_slices = np.ma.clump_unmasked(fast)
        else:
            ias = repair_mask(airspeed.array, repair_duration=600,
                              raise_entirely_masked=False)
            fast = np.ma.masked_less(ias, AIRSPEED_THRESHOLD)
            fast_slices = np.ma.clump_unmasked(fast)
            fast_slices = slices_remove_small_gaps(fast_slices, time_limit=30,
                                                   hz=self.frequency)

        self.create_phases(fast_slices)


class FinalApproach(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               airs=S('Airborne')):
        # Airborne dependancy added as we should not be approaching if never airborne
        self.create_phases(alt_aal.slices_from_to(1000, 50))


class GearExtending(FlightPhaseNode):
    '''
    Gear extending and retracting are section nodes, as they last for a
    finite period. Based on the Gear Red Warnings.

    For some aircraft no parameters to identify the transit are recorded, so
    a nominal period is included in Gear Down Selected Calculations to
    allow for exceedance of gear transit limits.
    '''

    def derive(self, down_in_transit=M('Gear Down In Transit'), airs=S('Airborne')):

        in_transit = (down_in_transit.array == 'Extending')
        gear_extending = slices_and(runs_of_ones(in_transit), airs.get_slices())
        self.create_phases(gear_extending)


class GearExtended(FlightPhaseNode):
    '''
    Simple phase translation of the Gear Down parameter.
    '''
    def derive(self, gear_down=M('Gear Down')):
        repaired = repair_mask(gear_down.array, gear_down.frequency,
                               repair_duration=120, extrapolate=True,
                               method='fill_start')
        gear_dn = runs_of_ones(repaired == 'Down')
        # remove single sample changes from this phase
        # note: order removes gaps before slices for Extended
        slices_remove_small_bits = lambda g: slices_remove_small_slices(
            slices_remove_small_gaps(g, count=2), count=2)
        self.create_phases(slices_remove_small_bits(gear_dn))


class GearRetracting(FlightPhaseNode):
    '''
    Gear extending and retracting are section nodes, as they last for a
    finite period. Based on the Gear Red Warnings.

    For some aircraft no parameters to identify the transit are recorded, so
    a nominal period is included in Gear Up Selected Calculations to
    allow for exceedance of gear transit limits.
    '''

    def derive(self, up_in_transit=M('Gear Up In Transit'), airs=S('Airborne')):

        in_transit = (up_in_transit.array == 'Retracting')
        gear_retracting = slices_and(runs_of_ones(in_transit), airs.get_slices())
        self.create_phases(gear_retracting)


class GearRetracted(FlightPhaseNode):
    '''
    Simple phase translation of the Gear Down parameter to show gear Up.
    '''
    def derive(self, gear_up=M('Gear Up')):
        repaired = repair_mask(gear_up.array, gear_up.frequency,
                               repair_duration=120, extrapolate=True,
                               method='fill_start')
        gear_up = runs_of_ones(repaired == 'Up')
        # remove single sample changes from this phase
        # note: order removes slices before gaps for Retracted
        slices_remove_small_bits = lambda g: slices_remove_small_gaps(
            slices_remove_small_slices(g, count=2), count=2)
        self.create_phases(slices_remove_small_bits(gear_up))


class Hover(FlightPhaseNode):
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type == helicopter and \
               all_of(('Altitude AGL', 'Airborne', 'Groundspeed'), available)

    def derive(self, alt_agl=P('Altitude AGL'),
               airs=S('Airborne'),
               gspd=P('Groundspeed'),
               trans_hfs=S('Transition Hover To Flight'),
               trans_fhs=S('Transition Flight To Hover')):

        low_flights = []
        hovers = []

        for air in airs:
            lows = slices_below(alt_agl.array[air.slice], HOVER_HEIGHT_LIMIT)[1]
            for low in lows:
                if np.ma.min(alt_agl.array[shift_slice(low, air.slice.start)]) <= HOVER_MIN_HEIGHT:
                    low_flights.extend([shift_slice(low, air.slice.start)])
        slows = slices_below(gspd.array, HOVER_GROUNDSPEED_LIMIT)[1]
        low_flights = slices_and(low_flights, slows)
        # Remove periods identified already as transitions.
        for low_flight in low_flights:
            if trans_fhs:
                for trans_fh in trans_fhs:
                    if slices_overlap(low_flight, trans_fh.slice):
                        low_flight = slice(trans_fh.slice.stop, low_flight.stop)

            if trans_hfs:
                for trans_hf in trans_hfs:
                    if slices_overlap(low_flight, trans_hf.slice):
                        low_flight = slice(low_flight.start, trans_hf.slice.start)

            hovers.extend([low_flight])

        # Exclude transition periods and trivial periods of operation.
        self.create_phases(filter_slices_duration(hovers, HOVER_MIN_DURATION, frequency=alt_agl.frequency))


class HoverTaxi(FlightPhaseNode):
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type == helicopter and \
               all_of(('Altitude AGL', 'Airborne', 'Hover'), available)

    def derive(self, alt_agl=P('Altitude AGL'),
               airs=S('Airborne'),
               hovers=S('Hover'),
               trans_hfs=S('Transition Hover To Flight'),
               trans_fhs=S('Transition Flight To Hover')):

        low_flights = []
        air_taxis = []
        taxis = []

        if airs:
            for air in airs:
                lows = slices_below(alt_agl.array[air.slice], HOVER_TAXI_HEIGHT)[1]
                taxis = shift_slices(lows, air.slice.start)
        # Remove periods identified already as transitions.
        if taxis:
            for taxi in slices_and_not(taxis, [h.slice for h in hovers]):
                if trans_fhs:
                    for trans_fh in trans_fhs:
                        if slices_overlap(taxi, trans_fh.slice):
                            taxi = slice(trans_fh.slice.stop, taxi.stop)

                if trans_hfs:
                    for trans_hf in trans_hfs:
                        if slices_overlap(taxi, trans_hf.slice):
                            taxi = slice(taxi.start, trans_hf.slice.start)

                air_taxis.extend([taxi])

        self.create_phases(air_taxis)


def scan_ils(beam, ils_dots, height, scan_slice, frequency,
             hdg=None, hdg_ldg=None, duration=10):
    '''
    Scans ils dots and returns last slice where ils dots fall below ILS_CAPTURE and remain below 2.5 dots
    if beam is glideslope slice will not extend below 200ft.

    :param beam: 'localizer' or 'glideslope'
    :type beam: str
    :param ils_dots: localizer deviation in dots
    :type ils_dots: Numpy array
    :param height: Height above airfield
    :type height: Numpy array
    :param scan_slice: slice to be inspected
    :type scan_slice: python slice
    :param frequency: input signal sample rate
    :type frequency: float
    :param hdg: aircraft heaing
    :type hdg: Numpy array
    :param hdg_ldg: Heading on the landing roll
    :type hdg_ldg: list fo Key Point Values
    :param duration: Minimum duration for the ILS to be established
    :type duration: float, default = 10 seconds.
    '''
    if beam not in ['localizer', 'glideslope']:
        raise ValueError('Unrecognised beam type in scan_ils')

    if beam=='localizer' and hdg_ldg:
        # Restrict the scan to approaches facing the runway This restriction
        # will be carried forward into the glideslope calculation by the
        # restricted duration of localizer established, hence does not need
        # to be repeated.
        hdg_landing = None
        for each_ldg in hdg_ldg:
            if is_index_within_slice(each_ldg.index, scan_slice):
                hdg_landing = each_ldg.value
                break

        if hdg_landing:
            diff = np.ma.abs(heading_diff(hdg[scan_slice] % 360, hdg_landing))
            facing = shift_slices(
                np.ma.clump_unmasked(np.ma.masked_greater(diff, 90.0)),
                scan_slice.start)
            scan_slice = slices_and([scan_slice], facing)[-1]


    if np.ma.count(ils_dots[scan_slice]) < duration*frequency:
        # less than duration seconds of valid data within slice
        return None

    # Find the range of valid ils dots withing scan slice
    valid_ends = np.ma.flatnotmasked_edges(ils_dots[scan_slice])
    if valid_ends is None:
        return None
    valid_slice = slice(*(valid_ends+scan_slice.start))
    if np.ma.count(ils_dots[valid_slice])/float(len(ils_dots[valid_slice])) < 0.4:
        # less than 40% valid data within valid data slice
        return None

    # get abs of ils dots as its used everywhere and repair small masked periods
    ils_abs = repair_mask(np.ma.abs(ils_dots), frequency=frequency, repair_duration=5)

    # ----------- Find loss of capture

    last_valid_idx, last_valid_value = last_valid_sample(ils_abs[scan_slice])

    if last_valid_value < 2.5:
        # finished established ? if established in first place
        ils_lost_idx = scan_slice.start + last_valid_idx + 1
    else:
        # find last time went below 2.5 dots
        last_25_idx = index_at_value(ils_abs, 2.5, slice(scan_slice.stop, scan_slice.start, -1))
        if last_25_idx is None:
            # never went below 2.5 dots
            return None
        else:
            # Round down to nearest integer
            ils_lost_idx = int(last_25_idx)

    if beam == 'glideslope':
        # If Glideslope find index of height last passing 200ft and use the
        # smaller of that and any index where the ILS was lost
        idx_200 = index_at_value(height, 200, slice(scan_slice.stop,
                                                scan_slice.start, -1),
                             endpoint='closing')
        if idx_200 is not None:
            ils_lost_idx = min(ils_lost_idx, idx_200) + 1

        if np.ma.count(ils_dots[scan_slice.start:ils_lost_idx]) < 5:
            # less than 5 valid values within remaining section
            return None

    # ----------- Find start of capture

    # Find where to start scanning for the point of "Capture", Look for the
    # last time we were within 2.5dots
    scan_start_idx = index_at_value(ils_abs, 2.5, slice(ils_lost_idx-1, scan_slice.start-1, -1))

    first_valid_idx, first_valid_value = first_valid_sample(ils_abs[scan_slice.start:ils_lost_idx])

    ils_capture_idx = None
    if scan_start_idx or (first_valid_value > ILS_CAPTURE):
        # Look for first instance of being established
        if not scan_start_idx:
            scan_start_idx = index_at_value(ils_abs, ILS_CAPTURE, slice(scan_slice.start, ils_lost_idx))
        if scan_start_idx is None:
            # didnt start established and didnt move within 2.5 dots
            return None
        half_dot = np.ma.masked_greater(ils_abs, 0.5)
        est = np.ma.clump_unmasked(half_dot)
        est_slices = slices_and(est, (slice(scan_start_idx, ils_lost_idx),))
        est_slices = slices_remove_small_slices(est_slices, duration, hz=frequency)
        if est_slices:
            ils_capture_idx = est_slices[0].start
        else:
            return None
    elif first_valid_value < ILS_CAPTURE:
        # started established
        ils_capture_idx = scan_slice.start + first_valid_idx
    if first_valid_idx is None:
        # no valid data
        return None

    if ils_capture_idx is None or ils_lost_idx is None:
        return None
    else:
        # OK, we have seen an ILS signal, but let's make sure we did respond
        # to it. The test here is to make sure that we didn't just pass
        # through the beam (L>R or R>L or U>D or D>U) without making an
        # effort to correct the variation.
        ils_slice = slice(ils_capture_idx, ils_lost_idx)
        width = 5.0
        if frequency < 0.5:
            width = 10.0
        ils_rate = rate_of_change_array(ils_dots[ils_slice], frequency, width=width, method='regression')
        top = max(ils_rate)
        bottom = min(ils_rate)
        if top*bottom > 0.0:
            # the signal never changed direction, so went straight through
            # the beam without getting established...
            return None
        else:
            # Hurrah! We did capture the beam
            return ils_slice


class IANFinalApproachCourseEstablished(FlightPhaseNode):
    name = 'IAN Final Approach Established'

    def derive(self,
               ian_final=P('IAN Final Approach Course'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               apps=S('Approach Information'),
               ils_freq=P('ILS Frequency'),
               app_src_capt=M('Displayed App Source (Capt)'),
               app_src_fo=M('Displayed App Source (FO)')):

        def create_ils_phases(slices):
            for _slice in slices:
                ils_slice = scan_ils('localizer', ian_final.array, alt_aal.array,
                                     _slice, ian_final.frequency)
                if ils_slice is not None:
                    self.create_phase(ils_slice)

        # Displayed App Source required to ensure that IAN is being followed
        in_fmc = (app_src_capt.array == 'FMC') | (app_src_fo.array == 'FMC')
        ian_final.array[~in_fmc] = np.ma.masked

        for app in apps:
            if app.loc_est:
                # Mask IAN data for approaches where ILS is established
                ian_final.array[app.slice] = np.ma.masked
                continue

            if not np.ma.count(ian_final.array[app.slice]):
                # No valid IAN Final Approach data for this approach.
                continue
            valid_slices = np.ma.clump_unmasked(ian_final.array[app.slice])
            valid_slices = slices_remove_small_gaps(valid_slices, count=5)
            last_valid_slice = shift_slice(valid_slices[-1], app.slice.start)
            create_ils_phases([last_valid_slice])


class IANGlidepathEstablished(FlightPhaseNode):
    name = 'IAN Glidepath Established'

    def derive(self,
               ian_glidepath=P('IAN Glidepath'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               apps=App('Approach Information'),
               app_src_capt=P('Displayed App Source (Capt)'),
               app_src_fo=P('Displayed App Source (FO)')):

        def create_ils_phases(slices):
            for _slice in slices:
                ils_slice = scan_ils('glideslope', ian_glidepath.array,
                                     alt_aal.array, _slice,
                                     ian_glidepath.frequency)
                if ils_slice is not None:
                    self.create_phase(ils_slice)

        # Assumption here is that Glidepath is not as tightly coupled with
        # Final Approach Course as Glideslope is to Localiser

        # Displayed App Source required to ensure that IAN is being followed
        in_fmc = (app_src_capt.array == 'FMC') | (app_src_fo.array == 'FMC')
        ian_glidepath.array[~in_fmc] = np.ma.masked

        for app in apps:
            if app.gs_est:
                # Mask IAN data for approaches where ILS is established
                ian_glidepath.array[app.slice] = np.ma.masked
                continue

            if not np.ma.count(ian_glidepath.array[app.slice]):
                # No valid ian glidepath data for this approach.
                continue
            valid_slices = np.ma.clump_unmasked(ian_glidepath.array[app.slice])
            valid_slices = slices_remove_small_gaps(valid_slices, count=5)
            last_valid_slice = shift_slice(valid_slices[-1], app.slice.start)
            create_ils_phases([last_valid_slice])


class ILSLocalizerEstablished(FlightPhaseNode):
    name = 'ILS Localizer Established'

    def derive(self, apps=App('Approach Information')):
        for app in apps:
            if app.loc_est:
                self.create_phase(app.loc_est)

'''
class ILSApproach(FlightPhaseNode):
    name = "ILS Approach"
    """
    Where a Localizer Established phase exists, extend the start and end of
    the phase back to 3 dots (i.e. to beyond the view of the pilot which is
    2.5 dots) and assign this to ILS Approach phase. This period will be used
    to determine the range for the ILS display on the web site and for
    examination for ILS KPVs.
    """
    def derive(self, ils_loc = P('ILS Localizer'),
               ils_loc_ests = S('ILS Localizer Established')):
        # For most of the flight, the ILS will not be valid, so we scan only
        # the periods with valid data, ignoring short breaks:
        locs = np.ma.clump_unmasked(repair_mask(ils_loc.array))
        for loc_slice in locs:
            for ils_loc_est in ils_loc_ests:
                est_slice = ils_loc_est.slice
                if slices_overlap(loc_slice, est_slice):
                    before_established = slice(est_slice.start, loc_slice.start, -1)
                    begin = index_at_value(np.ma.abs(ils_loc.array),
                                                     3.0,
                                                     _slice=before_established)
                    end = est_slice.stop
                    self.create_phase(slice(begin, end))
'''


class ILSGlideslopeEstablished(FlightPhaseNode):
    name = "ILS Glideslope Established"

    def derive(self, apps=App('Approach Information')):
        for app in apps:
            if app.gs_est:
                self.create_phase(app.gs_est)


class InitialApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               app_lands=S('Approach')):
        for app_land in app_lands:
            # We already know this section is below the start of the initial
            # approach phase so we only need to stop at the transition to the
            # final approach phase.
            ini_app = np.ma.masked_where(alt_AAL.array[app_land.slice]<1000,
                                         alt_AAL.array[app_land.slice])
            if not len(ini_app):
                continue
            phases = np.ma.clump_unmasked(ini_app)
            for phase in phases:
                begin = phase.start
                pit = np.ma.argmin(ini_app[phase]) + begin
                if ini_app[pit] < ini_app[begin] :
                    self.create_phases(shift_slices([slice(begin, pit)],
                                                    app_land.slice.start))


class InitialClimb(FlightPhaseNode):
    '''
    Phase from end of Takeoff (35ft) to start of climb (1000ft)
    '''
    def derive(self,
               takeoffs=S('Takeoff'),
               climb_starts=KTI('Climb Start'),
               tocs=KTI('Top Of Climb'),
               alt=P('Altitude STD')):
        
        # if max alt is above 1000 ft we don't need to consider the ToC
        # point in our calculations
        if alt and np.ma.max(alt.array) > 1000:
            to_scan = [[t.stop_edge, 'takeoff'] for t in takeoffs] + \
                [[c.index, 'climb'] for c in climb_starts]            
        else:
            to_scan = [[t.stop_edge, 'takeoff'] for t in takeoffs] + \
                [[c.index, 'climb'] for c in climb_starts]+ \
                [[c.index, 'climb'] for c in tocs]
        from operator import itemgetter
        to_scan = sorted(to_scan, key=itemgetter(0))
        for i in range(len(to_scan)-1):
            if to_scan[i][1]=='takeoff' and to_scan[i+1][1]=='climb':
                begin = to_scan[i][0]
                end = to_scan[i+1][0]
                self.create_phase(slice(begin, end), begin=begin, end=end)


class LevelFlight(FlightPhaseNode):
    '''
    '''
    def derive(self,
               airs=S('Airborne'),
               vrt_spd=P('Vertical Speed For Flight Phases')):

        for air in airs:
            limit = settings.VERTICAL_SPEED_FOR_LEVEL_FLIGHT
            level_flight = np.ma.masked_outside(vrt_spd.array[air.slice], -limit, limit)
            level_slices = np.ma.clump_unmasked(level_flight)
            level_slices = slices_remove_small_slices(level_slices,
                                                      time_limit=settings.LEVEL_FLIGHT_MIN_DURATION,
                                                      hz=vrt_spd.frequency)
            self.create_phases(shift_slices(level_slices, air.slice.start))



class StraightAndLevel(FlightPhaseNode):
    '''
    Building on Level Flight, this checks for straight flight. We use heading
    rate as more sensitive than roll attitude and sticking to the core three
    parameters.
    '''
    def derive(self,
               levels=S('Level Flight'),
               hdg=P('Heading')):

        for level in levels:
            limit = settings.HEADING_RATE_FOR_STRAIGHT_FLIGHT
            rot = rate_of_change_array(hdg.array[level.slice], hdg.frequency, width=30)
            straight_flight = np.ma.masked_outside(rot, -limit, limit)
            straight_slices = np.ma.clump_unmasked(straight_flight)
            straight_and_level_slices = slices_remove_small_slices(
                straight_slices, time_limit=settings.LEVEL_FLIGHT_MIN_DURATION,
                hz=hdg.frequency)
            self.create_phases(shift_slices(straight_and_level_slices, level.slice.start))


class Grounded(FlightPhaseNode):
    '''
    Includes start of takeoff run and part of landing run.
    Was "On Ground" but this name conflicts with a recorded 737-6 parameter name.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return all_of(('Airborne', 'Airspeed'), available)
        else:
            return 'HDF Duration' in available

    def _derive_aircraft(self, speed, hdf_duration, air):
        data_end = hdf_duration.value * self.frequency if hdf_duration else None
        if air:
            gnd_phases = slices_not(air.get_slices(), begin_at=0, end_at=data_end)
            if not gnd_phases:
                # Either all on ground or all in flight.
                median_speed = np.ma.median(speed.array)
                if median_speed > AIRSPEED_THRESHOLD:
                    gnd_phases = [slice(None,None,None)]
                else:
                    gnd_phases = [slice(0,data_end,None)]
        else:
            # no airborne so must be all on ground
            gnd_phases = [slice(0,data_end,None)]

        self.create_phases(gnd_phases)

    def _derive_helicopter(self, air, airspeed):
        '''
        Needed for AP Engaged KPV.
        '''
        all_data = slice(0, len(airspeed.array))
        self.create_sections(slices_and_not([all_data], air.get_slices()))

    def derive(self,
               ac_type=A('Aircraft Type'),
               # aircraft
               speed=P('Airspeed'),
               hdf_duration=A('HDF Duration'),
               # helicopter
               airspeed=P('Airspeed'),
               # shared
               air=S('Airborne')):
        if ac_type == helicopter:
            self._derive_helicopter(air, airspeed)
        else:
            self._derive_aircraft(speed, hdf_duration, air)


class OnDeck(FlightPhaseNode):
    '''
    Flight phase for helicopters that land on the deck of a moving vessel.

    Testing for motion will separate moving vessels from stationary decks, chosen as a better
    option than testing the location against Google Earth for land/sea.

    Also, movement was not practical as helicopters taxi at similar speeds to a ship sailing!

    Note that this qualifies Grounded which is still asserted when On Deck.
    '''

    can_operate = helicopter_only

    def derive(self, gnds=S('Grounded'),
               pitch=P('Pitch'), roll=P('Roll')):

        decks = []
        for gnd in gnds:
            # The fourier transform for pitching motion...
            p = pitch.array[gnd.slice]
            n = float(len(p)) # Scaling the result to be independet of data length.
            fft_p = np.abs(np.fft.rfft(p - moving_average(p))) / n

            # similarly for roll
            r = roll.array[gnd.slice]
            fft_r = np.abs(np.fft.rfft(r - moving_average(r))) / n

            # What was the maximum harmonic seen?
            fft_max = np.ma.max(fft_p + fft_r)

            # Values of less than 0.1 were on the ground, and 0.34 on deck for the one case seen to date.
            if fft_max > 0.2:
                decks.append(gnd.slice)
        if decks:
            self.create_sections(decks)


class Taxiing(FlightPhaseNode):
    '''
    This finds the first and last signs of movement to provide endpoints to
    the taxi phases.

    If groundspeed is available, only periods where the groundspeed is over
    5kts are considered taxiing.

    With all mobile and moving periods identified, we then remove all the
    periods where the aircraft is either airborne, taking off, landing or
    carrying out a rejected takeoff. What's left are the taxiing on the
    ground periods.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type')):
        ground_only = seg_type and seg_type.value == 'GROUND_ONLY' and \
            'Mobile' in available
        default = all_of(('Mobile', 'Takeoff', 'Landing', 'Airborne'), available)
        return default or ground_only

    def derive(self, mobiles=S('Mobile'), gspd=P('Groundspeed'),
               toffs=S('Takeoff'), lands=S('Landing'),
               rtos=S('Rejected Takeoff'),
               airs=S('Airborne')):
        # XXX: There should only be one Mobile slice.
        if gspd:
            # Limit to where Groundspeed is above GROUNDSPEED_FOR_MOBILE.
            taxiing_slices = np.ma.clump_unmasked(np.ma.masked_less
                                                  (np.ma.abs(gspd.array),
                                                   GROUNDSPEED_FOR_MOBILE))
            taxiing_slices = slices_and(mobiles.get_slices(), taxiing_slices)
        else:
            taxiing_slices = mobiles.get_slices()

        if toffs:
            taxiing_slices = slices_and_not(taxiing_slices, toffs.get_slices())
        if lands:
            taxiing_slices = slices_and_not(taxiing_slices, lands.get_slices())
        if rtos:
            taxiing_slices = slices_and_not(taxiing_slices, rtos.get_slices())
        if airs:
            taxiing_slices = slices_and_not(taxiing_slices, airs.get_slices())

        self.create_phases(taxiing_slices)


class Mobile(FlightPhaseNode):
    '''
    This finds the first and last signs of movement to provide endpoints to
    the taxi phases. As Heading Rate is derived directly from heading, this
    phase is guaranteed to be operable for very basic aircraft.
    '''

    @classmethod
    def can_operate(cls, available):
        return 'Heading Rate' in available

    def derive(self,
               rot=P('Heading Rate'),
               gspd=P('Groundspeed'),
               airs=S('Airborne'),
               #power=P('Eng (*) Any Running'),
               ):

        turning = np.ma.masked_less(np.ma.abs(rot.array), HEADING_RATE_FOR_MOBILE)
        movement = np.ma.flatnotmasked_edges(turning)
        start, stop = movement if movement is not None else (None, None)

        if gspd is not None:
            moving = np.ma.masked_less(np.ma.abs(gspd.array), GROUNDSPEED_FOR_MOBILE)
            mobile = np.ma.flatnotmasked_edges(moving)
            if mobile is not None:
                start = min(start, mobile[0]) if start else mobile[0]
                stop = max(stop, mobile[1]) if stop else mobile[1]

        if airs and airs is not None:
            start = min(start, airs[0].slice.start) if start is not None else airs[0].slice.start
            stop = max(stop, airs[-1].slice.stop) if stop else airs[-1].slice.stop

        self.create_phase(slice(start, stop))


class Stationary(FlightPhaseNode):
    """
    Phases of the flight when the aircraft remains stationary.

    This is useful in fuel monitoring.
    """
    def derive(self,
               gspd=P('Groundspeed')):
        not_moving = runs_of_ones(gspd.array < GROUNDSPEED_FOR_MOBILE)
        self.create_phases(slices_remove_small_gaps(not_moving, time_limit=5, hz=gspd.frequency))


class Landing(FlightPhaseNode):
    '''
    This flight phase starts at 50 ft in the approach and ends as the
    aircraft turns off the runway. Subsequent KTIs and KPV computations
    identify the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and thereby make sure the 50ft startpoint is exact.
    '''
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT'):
            return False
        elif ac_type == helicopter:
            return all_of(('Altitude AGL', 'Collective', 'Airborne'), available)
        else:
            return 'Altitude AAL For Flight Phases' in available

    def _derive_aircraft(self, head, alt_aal, fast, mobile):
        phases = []
        for speedy in fast:
            # See takeoff phase for comments on how the algorithm works.

            # AARRGG - How can we check if this is at the end of the data
            # without having to go back and test against the airspeed array?
            # TODO: Improve endpoint checks. DJ
            # Answer:
            #  duration=A('HDF Duration')
            #  array_len = duration.value * self.frequency
            #  if speedy.slice.stop >= array_len: continue

            if (speedy.slice.stop is None or \
                speedy.slice.stop >= len(alt_aal.array) - 2):
                break

            landing_run = speedy.slice.stop + 2
            datum = head.array[landing_run]

            first = landing_run - (300 * alt_aal.frequency)
            # Limit first to be the latest of 5 mins or maximum altitude
            # during fast slice to account for short flights
            first = max(first, max_value(alt_aal.array, _slice=slice(speedy.slice.start, landing_run)).index)+2
            landing_begin = index_at_value(alt_aal.array,
                                           LANDING_THRESHOLD_HEIGHT,
                                           slice(landing_run, first, -1))
            if landing_begin is None:
                # we are not able to detect a landing threshold height,
                # therefore invalid section
                continue

            # The turn off the runway must lie within eight minutes of the
            # landing. (We did use 5 mins, but found some landings on long
            # runways where the turnoff did not happen for over 6 minutes
            # after touchdown).
            last = landing_run + (480 * head.frequency)

            # A crude estimate is given by the angle of turn
            landing_end = index_at_value(np.ma.abs(head.array-datum),
                                         HEADING_TURN_OFF_RUNWAY,
                                         slice(landing_run, last))
            if landing_end is None:
                # The data ran out before the aircraft left the runway so use
                # end of mobile or remainder of data.
                landing_end = mobile.get_last().slice.stop if mobile else len(head.array)-1

            # ensure any overlap with phases are ignored (possibly due to
            # data corruption returning multiple fast segments)
            new_phase = [slice(landing_begin, landing_end)]
            phases = slices_or(phases, new_phase)
        self.create_phases(phases)

    def _derive_helicopter(self, alt_agl, coll, airs):
        phases = []
        for air in airs:
            tdn = air.stop_edge
            # Scan back to find either when we descend through LANDING_HEIGHT or had peak hover height.
            to_scan = tdn - alt_agl.frequency*LANDING_TRACEBACK_PERIOD
            landing_begin = index_at_value(alt_agl.array, LANDING_HEIGHT,
                                           _slice=slice(tdn, to_scan , -1),
                                           endpoint='exact')

            # Scan forwards to find lowest collective shortly after touchdown.
            to_scan = tdn + coll.frequency*LANDING_COLLECTIVE_PERIOD
            landing_end = tdn  + np.ma.argmin(coll.array[tdn:to_scan])
            if landing_begin and landing_end:
                new_phase = [slice(landing_begin, landing_end)]
                phases = slices_or(phases, new_phase)
        self.create_phases(phases)

    def derive(self,
               ac_type=A('Aircraft Type'),
               # aircraft
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast'),
               mobile=S('Mobile'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               coll=P('Collective'),
               airs=S('Airborne')):
        if ac_type == helicopter:
            self._derive_helicopter(alt_agl, coll, airs)
        else:
            self._derive_aircraft(head, alt_aal, fast, mobile)


class LandingRoll(FlightPhaseNode):
    '''
    FDS developed this node to support the UK CAA Significant Seven
    programme. This phase is used when computing KPVs relating to the
    deceleration phase of the landing.

    "CAA to go with T/D to 60 knots with the T/D defined as less than 2 deg
    pitch (after main gear T/D)."

    The complex index_at_value ensures that if the aircraft does not flare to
    2 deg, we still capture the highest attitude as the start of the landing
    roll, and the landing roll starts as the aircraft passes 2 deg the last
    time, i.e. as the nosewheel comes down and not as the flare starts.
    '''
    @classmethod
    def can_operate(cls, available):
        return 'Landing' in available and any_of(('Airspeed True', 'Groundspeed'), available)

    def derive(self, pitch=P('Pitch'), gspd=P('Groundspeed'),
               aspd=P('Airspeed True'), lands=S('Landing')):
        if gspd:
            speed = gspd.array
        else:
            speed = aspd.array
        for land in lands:
            # Airspeed True on some aircraft do not record values below 61
            end = index_at_value(speed, LANDING_ROLL_END_SPEED, land.slice)
            if end is None:
                # due to masked values, use the land.stop rather than
                # searching from the end of the data
                end = land.slice.stop
            begin = None
            if pitch:
                begin = index_at_value(pitch.array, 2.0,
                                       slice(end,land.slice.start,-1),
                                       endpoint='nearest')
            if begin is None:
                # due to masked values, use land.start in place
                begin = land.slice.start

            self.create_phase(slice(begin, end), begin=begin, end=end)


class RejectedTakeoff(FlightPhaseNode):
    '''
    Rejected Takeoff based on Acceleration Longitudinal Offset Removed exceeding
    the TAKEOFF_ACCELERATION_THRESHOLD and not being followed by a liftoff.
    
    Note: We cannot use Liftoff, Taxi Out or Airborne in this computation in case
    the rejected takeoff was followed by a taxi back to stand.
    '''
    
    @classmethod
    def can_operate(cls, available):
        return all_of(('Eng (*) All Running', 'Acceleration Longitudinal Offset Removed', 'Grounded'), available)

    def derive(self, accel_lon=P('Acceleration Longitudinal Offset Removed'),
               eng_running=M('Eng (*) All Running'),
               groundeds=S('Grounded'),
               takeoffs=S('Takeoff Roll'),
               eng_n1=P('Eng (*) N1 Max'),
               toff_acc=KTI('Takeoff Acceleration Start')):

        # We need all engines running to be a realistic attempt to get airborne
        runnings = runs_of_ones(eng_running.array=='Running')
        hz = accel_lon.frequency
        # If Takeoff Acceleration Start KTI exists, include only slices before this KTI
        # and shorten the slice containing the it.
        if toff_acc:
            toff_idx = toff_acc.get_first().index
            running_on_grounds = slices_and(runnings, [slice(runnings[0].start, toff_idx),])
        else:
            # We ignore the last slice in groundeds as this is usually the during the taxi in
            running_on_grounds = slices_and(runnings, groundeds.get_slices()[:-1])

        if eng_n1 is not None:
            accel_above_thres = runs_of_ones(repair_mask(accel_lon.array, frequency=hz, repair_duration=None) >= TAKEOFF_ACCELERATION_THRESHOLD)
            n1_max_above_50 = runs_of_ones(repair_mask(eng_n1.array, frequency=hz, repair_duration=None) > 50)
            # list of potential RTO's which may include the takeoff as well.
            potential_rtos = slices_and(accel_above_thres, n1_max_above_50)
            potential_rtos = slices_remove_small_gaps(potential_rtos, hz=hz)
            rto_list=[]
            for rto in potential_rtos:
                for running_on_ground in running_on_grounds:
                    # The RTO slice can only be within the 'Grounded' phase. 
                    # If RTO slice size changes (decreases) when AND'd with
                    # running_on_ground this Acceleration/N1 Max combination
                    # should be the part of the takeoff. 
                    if slices_and([rto], [running_on_ground]) == [rto]:
                        if len(rto_list) > 0 and (rto.start - rto_list[-1].stop)/hz < 60.0:
                            continue
                        rto_list.append(rto)
            #potential_rtos = slices_remove_small_slices(potential_rtos, count=1)
            if rto_list:
                self.create_phases(rto_list)
        else:
            # Not sure this is needed
            #if takeoffs is not None:
                #running_on_grounds = slices_and_not(running_on_grounds, takeoffs.get_slices())
            for running_on_ground in running_on_grounds:
                accel_lon_ground = accel_lon.array[running_on_ground]
                accel_lon_slices = runs_of_ones(accel_lon_ground >= TAKEOFF_ACCELERATION_THRESHOLD)

                trough_index = 0
                for peak in accel_lon_slices:
                    if trough_index and peak.start < trough_index:
                        continue
                    # Look for the deceleration characteristic of a rejected takeoff.
                    trough_index = index_at_value(accel_lon_ground, -TAKEOFF_ACCELERATION_THRESHOLD/2.0, _slice=slice(peak.stop, None))
                    # trough_index will be None for every takeoff. Then, if it looks like a rejection, 
                    # we check the two accelerations happened fairly close together.
                    if trough_index and \
                                   (trough_index - peak.start)/accel_lon.hz < 60.0:
                        self.create_phase(slice(peak.start + running_on_ground.start,
                                                trough_index + running_on_ground.start))


class RotorsTurning(FlightPhaseNode):
    '''
    Used to suppress nuisance warnings on the ground.

    Note: Rotors Running is the Multistate parameter, while Rotors Turning is the flight phase.
    '''

    can_operate = helicopter_only

    def derive(self, rotors=M('Rotors Running')):
        self.create_sections(runs_of_ones(rotors.array == 'Running'))


class Takeoff(FlightPhaseNode):
    """
    This flight phase starts as the aircraft turns onto the runway and ends
    as it climbs through 35ft. Subsequent KTIs and KPV computations identify
    the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and make sure the 35ft endpoint is exact.
    """

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'STOP_ONLY'):
            return False
        else:
            return all_of(('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast', 'Airborne'), available)

    def derive(self,
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast'),
               airs=S('Airborne'), # If never airborne, didnt takeoff.
               ):
        # Note: This algorithm works across the entire data array, and
        # not just inside the speedy slice, so the final indexes are
        # absolute and not relative references.

        for speedy in fast:
            # This basic flight phase cuts data into fast and slow sections.

            # We know a takeoff should come at the start of the phase,
            # however if the aircraft is already airborne, we can skip the
            # takeoff stuff.
            if not speedy.slice.start:
                break

            # The aircraft is part way down its takeoff run at the start of
            # the section.
            takeoff_run = speedy.slice.start

            #-------------------------------------------------------------------
            # Find the start of the takeoff phase from the turn onto the runway.

            # The heading at the start of the slice is taken as a datum for now.
            datum = head.array[takeoff_run]

            # Track back to the turn
            # If he took more than 5 minutes on the runway we're not interested!
            first = max(0, takeoff_run - (300 * head.frequency))
            # Repair small gaps incase transition is masked.
            # XXX: This could be optimized by repairing and calling abs on
            # the 5 minute window of the array. Shifting the index manually
            # will be less pretty than using index_at_value.
            head_abs_array = np.ma.abs(repair_mask(
                head.array, frequency=head.frequency, repair_duration=30) - datum)
            takeoff_begin = index_at_value(head_abs_array,
                                           HEADING_TURN_ONTO_RUNWAY,
                                           slice(takeoff_run, first, -1))

            # Where the data starts in line with the runway, default to the
            # start of the data
            if takeoff_begin is None:
                takeoff_begin = first

            #-------------------------------------------------------------------
            # Find the end of the takeoff phase as we climb through 35ft.

            # If it takes more than 5 minutes, he's certainly not doing a normal
            # takeoff !
            last = takeoff_run + (300 * alt_aal.frequency)
            # Limit last to be the earliest of 5 mins or maximum altitude
            # during fast slice to account for short flights
            last = min(last, max_value(alt_aal.array, _slice=slice(takeoff_run, speedy.slice.stop)).index)
            takeoff_end = index_at_value(alt_aal.array, INITIAL_CLIMB_THRESHOLD,
                                         slice(last, takeoff_run, -1))

            if takeoff_end <= 0:
                # catches if None or zero
                continue

            #-------------------------------------------------------------------
            # Create a phase for this takeoff
            self.create_phase(slice(takeoff_begin, takeoff_end))


class TakeoffRoll(FlightPhaseNode):
    '''
    Sub-phase originally written for the correlation tests but has found use
    in the takeoff KPVs where we are interested in the movement down the
    runway, not the turnon or liftoff.

    If pitch is not avaliable to detect rotation we use the end of the takeoff.
    '''

    @classmethod
    def can_operate(cls, available):
        return all_of(('Takeoff', 'Takeoff Acceleration Start'), available)

    def derive(self, toffs=S('Takeoff'),
               acc_starts=KTI('Takeoff Acceleration Start'),
               pitch=P('Pitch')):
        for toff in toffs:
            begin = toff.slice.start # Default if acceleration term not available.
            if acc_starts: # We don't bother with this for data validation, hence the conditional
                acc_start = acc_starts.get_last(within_slice=toff.slice)
                if acc_start:
                    begin = acc_start.index
            chunk = slice(begin, toff.slice.stop)
            if pitch:
                pwo = first_order_washout(pitch.array[chunk], 3.0, pitch.frequency)
                two_deg_idx = index_at_value(pwo, 2.0)
                if two_deg_idx is None:
                    roll_end = toff.slice.stop
                    self.warning('Aircraft did not reach a pitch of 2 deg or Acceleration Start is incorrect')
                else:
                    roll_end = two_deg_idx + begin
                self.create_phase(slice(begin, roll_end))
            else:
                self.create_phase(chunk)


class TakeoffRollOrRejectedTakeoff(FlightPhaseNode):
    '''
    For monitoring configuration warnings especially, this combines actual
    and rejected takeoffs into a single phase node for convenience.
    '''
    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               trolls=S('Takeoff Roll'),
               rtoffs=S('Rejected Takeoff'),
               helo_toffs=S('Transition Hover To Flight')):
        self.create_phases(
            [s.slice for n in (trolls, rtoffs, helo_toffs) if n for s in n],
            name= "Takeoff Roll Or Rejected Takeoff")


class TakeoffRotation(FlightPhaseNode):
    '''
    This is used by correlation tests to check control movements during the
    rotation and lift phases.
    '''

    can_operate = aeroplane_only

    align_frequency = 1

    def derive(self, lifts=S('Liftoff')):
        if not lifts:
            return
        lift_index = lifts.get_first().index
        start = lift_index - 10
        end = lift_index + 15
        self.create_phase(slice(start, end))


class TakeoffRotationWow(FlightPhaseNode):
    '''
    Used by correlation tests which need to use only the rotation period while the mainwheels are on the ground. Specifically, AOA.
    '''
    name = 'Takeoff Rotation WOW'

    can_operate = aeroplane_only

    def derive(self, toff_rots=S('Takeoff Rotation')):
        for toff_rot in toff_rots:
            self.create_phase(slice(toff_rot.slice.start,
                                    toff_rot.slice.stop-15))


################################################################################
# Takeoff/Go-Around Ratings


class Takeoff5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally a maximum of
    5 minutes from the start of takeoff.

    For all aeroplanes we use the Takeoff Acceleration Start to indicate the
    start of the Takeoff 5 Minute Rating

    For turbo prop aircraft we look for NP stabalising at least 5% less than
    liftoff NP

    For Jet aircraft we look for 5 minutes following Takeoff Acceleration Start.
    '''
    align_frequency = 1

    @classmethod
    def can_operate(cls, available, eng_type=A('Engine Propulsion'), ac_type=A('Aircraft Type')):
        if eng_type and eng_type.value == 'PROP':
            return all_of(('Takeoff Acceleration Start', 'Liftoff', 'Eng (*) Np Avg', 'Engine Propulsion', 'HDF Duration'), available)
        elif ac_type == helicopter:
            return all_of(('Liftoff', 'HDF Duration'), available)
        else:
            return all_of(('Takeoff Acceleration Start', 'HDF Duration'), available)

    def get_metrics(self, angle):
        window_sizes = [2,4,8,16,32]
        metrics = np.ma.array([1000000] * len(angle))
        for l in window_sizes:
            maxy = filters.maximum_filter1d(angle, l)
            miny = filters.minimum_filter1d(angle, l)
            m = (maxy - miny) / l
            metrics = np.minimum(metrics, m)

        metrics = medfilt(metrics,3)
        metrics = 200.0 * metrics

        return metrics

    def derive(self, toffs=KTI('Takeoff Acceleration Start'),
               lifts=KTI('Liftoff'),
               eng_np=P('Eng (*) Np Avg'),
               duration=A('HDF Duration'),
               eng_type=A('Engine Propulsion'),
               ac_type=A('Aircraft Type')):
        '''
        '''
        five_minutes = 300 * self.frequency
        max_idx = duration.value * self.frequency
        if eng_type and eng_type.value == 'PROP':
            filter_median_window = 11
            enp_filt = medfilt(eng_np.array, filter_median_window)
            enp_filt = np.ma.array(enp_filt)
            g = self.get_metrics(enp_filt)
            enp_filt.mask = g > 40
            flat_slices = np.ma.clump_unmasked(enp_filt)
            for accel_start in toffs:
                rating_end = toff_slice_avg = None
                if not lifts.get_next(accel_start.index):
                    continue
                toff_idx = lifts.get_next(accel_start.index).index
                for flat in flat_slices:
                    if is_index_within_slice(toff_idx, flat):
                        toff_slice_avg = np.ma.average(enp_filt[flat])
                    elif toff_slice_avg is not None:
                        flat_avg = np.ma.average(enp_filt[flat])
                        if abs(toff_slice_avg - flat_avg) >= 5:
                            rating_end = flat.start
                            break
                    else:
                        continue
                if rating_end is None:
                    rating_end = accel_start.index + (five_minutes)
                self.create_phase(slice(accel_start.index, min(rating_end, max_idx)))
        elif ac_type == helicopter:
            
            start_idx = end_idx = 0
            for lift in lifts:
                start_idx = start_idx or lift.index
                end_idx = lift.index + five_minutes
                next_lift = lifts.get_next(lift.index)
                if next_lift and next_lift.index < end_idx:
                    end_idx = next_lift.index + five_minutes
                    continue
                self.create_phase(slice(start_idx, min(end_idx, max_idx)))
                start_idx = 0
        else:
            for toff in toffs:
                self.create_phase(slice(toff.index, min(toff.index + five_minutes, max_idx)))


# TODO: Write some unit tests!
class GoAround5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally 5 minutes from
    the start of takeoff. Also applies in the case of a go-around.
    '''
    align_frequency = 1

    def derive(self, gas=S('Go Around And Climbout'), tdwn=S('Touchdown')):
        '''
        We check that the computed phase cannot extend beyond the last
        touchdown, which may arise if a go-around was detected on the final
        approach.
        '''
        for ga in gas:
            startpoint = ga.slice.start
            endpoint = ga.slice.start + 300
            if tdwn:
                endpoint = min(endpoint, tdwn[-1].index)
            if startpoint < endpoint:
                self.create_phase(slice(startpoint, endpoint))


class MaximumContinuousPower(FlightPhaseNode):
    '''
    '''

    align_frequency = 1

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return all_of(('Airborne', 'Takeoff 5 Min Rating'), available)
        else:
            return all_deps(cls, available)

    def derive(self,
               airborne=S('Airborne'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating')):

        ga_slices = ga_ratings.get_slices() if ga_ratings else []
        ratings = to_ratings.get_slices() + ga_slices
        mcp = slices_and_not(airborne.get_slices(), ratings)
        self.create_phases(mcp)


################################################################################


class TaxiIn(FlightPhaseNode):
    """
    This takes the period from the end of landing to either the last engine
    stop after touchdown or the end of the mobile section.
    """
    def derive(self, gnds=S('Mobile'), lands=S('Landing'),
               last_eng_stops=KTI('Last Eng Stop After Touchdown')):
        land = lands.get_last()
        if not land:
            return
        for gnd in gnds:
            if slices_overlap(gnd.slice, land.slice):
                # Mobile may or may not stop before Landing for helicopters.
                taxi_start = min(gnd.slice.stop, land.slice.stop)
                taxi_stop = max(gnd.slice.stop, land.slice.stop)
                # Use Last Eng Stop After Touchdown if available.
                if last_eng_stops:
                    last_eng_stop = last_eng_stops.get_next(taxi_start)
                    if last_eng_stop and last_eng_stop.index > taxi_start:
                        taxi_stop = min(last_eng_stop.index,
                                        taxi_stop)
                if taxi_start != taxi_stop:
                    self.create_phase(slice(taxi_start, taxi_stop),
                                      name="Taxi In")


class TaxiOut(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in.
    """
    @classmethod
    def can_operate(cls, available):
        return all_of(('Mobile', 'Takeoff'), available)

    def derive(self, gnds=S('Mobile'), toffs=S('Takeoff'),
               first_eng_starts=KTI('First Eng Start Before Liftoff')):
        if toffs:
            toff = toffs[0]
            for gnd in gnds:
                # If takeoff starts at begining of data there was no taxi out phase
                if slices_overlap(gnd.slice, toff.slice) and toff.slice.start > 1:
                    taxi_start = gnd.slice.start + 1
                    taxi_stop = toff.slice.start - 1
                    if first_eng_starts:
                        first_eng_start = first_eng_starts.get_next(taxi_start)
                        if first_eng_start and first_eng_start.index < taxi_stop:
                            taxi_start = max(first_eng_start.index,
                                             taxi_start)
                    if taxi_stop > taxi_start:
                        self.create_phase(slice(taxi_start, taxi_stop),
                                          name="Taxi Out")


class TransitionHoverToFlight(FlightPhaseNode):
    '''
    The pilot normally makes a clear nose down pitching motion to initiate the
    transition from the hover, and with airspeed built, will raise the nose and
    initiate a clear climb to mark the end of the transition phase and start of the climb.
    '''

    can_operate = helicopter_only

    def derive(self, alt_agl=P('Altitude AGL'),
               ias=P('Airspeed'),
               airs=S('Airborne'),
               pitch_rate=P('Pitch Rate')):
        for air in airs:
            lows = np.ma.clump_unmasked(np.ma.masked_greater(alt_agl.array[air.slice],
                                                             ROTOR_TRANSITION_ALTITUDE))
            for low in lows:
                trans_slices = slices_from_to(ias.array[air.slice][low],
                                              ROTOR_TRANSITION_SPEED_LOW,
                                              ROTOR_TRANSITION_SPEED_HIGH,
                                              threshold=1.0)[1]
                if trans_slices:
                    for trans in trans_slices:
                        base = air.slice.start + low.start
                        ext_start = base  + trans.start - 20*ias.frequency
                        if alt_agl.array[ext_start]==0.0:
                            trans_start = index_at_value(ias.array, 0.0,
                                                         _slice=slice(base+trans.start, ext_start, -1),
                                                         endpoint='first_closing')
                        else:
                            trans_start = np.ma.argmin(pitch_rate.array[ext_start:base+trans.start]) + ext_start
                        self.create_phase(slice(trans_start, trans.stop+base))


class TransitionFlightToHover(FlightPhaseNode):
    '''
    Forward flight to hover transitions are weakly defined from a flight parameter
    perspective, so we only reply upon airspeed changes.
    '''

    can_operate = helicopter_only

    def derive(self, alt_agl=P('Altitude AGL'),
               ias=P('Airspeed'),
               airs=S('Airborne'),
               pitch_rate=P('Pitch Rate')):
        for air in airs:
            trans_slices = slices_from_to(ias.array[air.slice],
                                          ROTOR_TRANSITION_SPEED_HIGH,
                                          ROTOR_TRANSITION_SPEED_LOW,
                                          threshold=1.0)[1]

            if trans_slices:
                for trans in shift_slices(trans_slices, air.slice.start):
                    trans_end = index_at_value(ias.array, 0.0,
                                                 _slice=slice(trans.stop, trans.stop+20*ias.frequency),
                                                 endpoint='first_closing')
                    self.create_phase(slice(trans.start, trans_end+1))


class TurningInAir(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- HEADING_RATE_FOR_FLIGHT_PHASES in the air
    """
    def derive(self, rate_of_turn=P('Heading Rate'),
               airborne=S('Airborne'),
               ac_type=A('Aircraft Type')):

        if ac_type == helicopter:
            rate = HEADING_RATE_FOR_FLIGHT_PHASES_RW
        else:
            rate = HEADING_RATE_FOR_FLIGHT_PHASES_FW

        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array), -rate, rate)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, air.slice)
                    for air in airborne]):
                # If the slice is within any airborne section.
                self.create_phase(turn_slice, name="Turning In Air")


class TurningOnGround(FlightPhaseNode):
    """
    Turning on ground is computed during the two taxi phases. This\
    avoids\ high speed turnoffs where the aircraft may be travelling at high\
    speed\ at, typically, 30deg from the runway centreline. The landing\
    phase\ turnoff angle is nominally 45 deg, so avoiding this period.

    Rate of Turn is greater than +/- HEADING_RATE_FOR_TAXI_TURNS (%.2f) on the ground
    """ % HEADING_RATE_FOR_TAXI_TURNS
    def derive(self, rate_of_turn=P('Heading Rate'), taxi=S('Taxiing')): # Q: Use Mobile?
        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array),
                                      -HEADING_RATE_FOR_TAXI_TURNS,
                                      HEADING_RATE_FOR_TAXI_TURNS)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, txi.slice)
                    for txi in taxi]):
                self.create_phase(turn_slice, name="Turning On Ground")


# NOTE: Python class name restriction: '2DegPitchTo35Ft' not permitted.
class TwoDegPitchTo35Ft(FlightPhaseNode):
    '''
    '''

    name = '2 Deg Pitch To 35 Ft'

    def derive(self, takeoff_rolls=S('Takeoff Roll'), takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            for takeoff_roll in takeoff_rolls:
                if not is_slice_within_slice(takeoff_roll.slice, takeoff.slice):
                    continue

                if takeoff.slice.stop - takeoff_roll.slice.stop > 1:
                    self.create_section(slice(takeoff_roll.slice.stop, takeoff.slice.stop),
                                    begin=takeoff_roll.stop_edge,
                                    end=takeoff.stop_edge)
                else:
                    self.warning('%s not created as slice less than 1 sample' % self.name)


class ShuttlingApproach(FlightPhaseNode):
    '''
    Flight phase for the shuttling approach
    '''
    
    def derive(self, approaches=App('Approach Information')):
        for approach in approaches:
            if approach.type == 'SHUTTLING':
                self.create_section(approach.slice, name='Shuttling Approach')
               
                
class AirborneRadarApproach(FlightPhaseNode):
    '''
    Flight phase for airborne radar approaches (ARDA/AROA)
    '''
    
    def derive(self, approaches=App('Approach Information')):
        for approach in approaches:
            if approach.type == 'AIRBORNE_RADAR':
                self.create_section(approach.slice, name='Airborne Radar Approach')
