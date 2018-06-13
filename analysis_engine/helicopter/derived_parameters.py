# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from flightdatautilities import units as ut
from analysis_engine.node import (
    A, App, DerivedParameterNode, KPV, KTI, M, P, S
)
from analysis_engine.library import (
    bearings_and_distances,
    cycle_finder,
    integrate,
    np_ma_masked_zeros_like,
)


class ApproachRange(DerivedParameterNode):
    '''
    This is the range to the touchdown point for both ILS and visual
    approaches including go-arounds. The reference point is the ILS Localizer
    antenna where the runway is so equipped, or the end of the runway where
    no ILS is available.

    The array is masked where no data has been computed, and provides
    measurements in metres from the reference point where the aircraft is on
    an approach.

    A simpler function is provided for helicopter operations as they may
    not - in fact normally do not - have a runway to land on.
    '''

    units = ut.METER

    def derive(self,
               alt_aal=P('Altitude AAL'),
               lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               tdwns=KTI('Touchdown')):
        app_range = np_ma_masked_zeros_like(alt_aal.array)

        #Helicopter compuation does not rely on runways!
        stop_delay = 10 # To make sure the helicopter has stopped moving

        for tdwn in tdwns:
            end = tdwn.index
            endpoint = {'latitude': lat.array[end], 'longitude': lon.array[end]}
            try:
                begin = tdwns.get_previous(end).index+stop_delay
            except:
                begin = 0
            this_leg = slice(begin, end+stop_delay)
            _, app_range[this_leg] = bearings_and_distances(lat.array[this_leg],
                                                            lon.array[this_leg],
                                                            endpoint)
        self.array = app_range


class AltitudeADH(DerivedParameterNode):
    '''
    Altitude Above Deck Height

    The rate of descent will differ between the radio and pressure calculations significantly for a period
    as the aircraft comes over the deck. We test for these large differences and substitute the pressure rate
    of descent for just those samples, then reconstruct the altitude trace by integrating the corrected differential.
    '''
    name = 'Altitude ADH'

    units = ut.FT

    def derive(self, rad=P('Altitude Radio'),
               hdot=P('Vertical Speed'),
               ):

        def seek_deck(rad, hdot, min_idx, rad_hz):

            def one_direction(rad, hdot, sence, rad_hz):
                # Stairway to Heaven is getting a bit old. Getting with the times?
                b_diffs = hdot/60
                r_diffs = np.ma.ediff1d(rad, to_begin=b_diffs[0])
                diffs = np.ma.where(np.ma.abs(r_diffs-b_diffs)>6.0, b_diffs, r_diffs)
                height = integrate(diffs,
                                   frequency=rad_hz,
                                   direction=sence,
                                   repair=False)
                return height

            height_from_rig = np_ma_masked_zeros_like(rad)
            if len(rad[:min_idx]) > 0:
                height_from_rig[:min_idx] = one_direction(rad[:min_idx], hdot[:min_idx], "backwards", rad_hz)
            if len(rad[min_idx:]) > 0:
                height_from_rig[min_idx:] = one_direction(rad[min_idx:], hdot[min_idx:], "forwards", rad_hz)

            '''
            # And we are bound to want to know the rig height somewhere, so here's how to work that out.
            rig_height = rad[0]-height_from_rig[0]
            # I checked this and it seems pretty consistent.
            # See Library\Projects\Helicopter FDM\Algorithm Development\Rig height estimates from Bond initial test data.xlsx
            #lat=hdf['Latitude'].array[app_slice][-1]
            #lon=hdf['Longitude'].array[app_slice][-1]
            #print(lat, lon, rig_height)
            '''
            return height_from_rig

        self.array = np_ma_masked_zeros_like(rad.array)
        rad_peak_idxs, rad_peak_vals = cycle_finder(rad.array, min_step=150.0)

        if len(rad_peak_idxs)<4:
            return

        slice_idxs = list(zip(rad_peak_idxs[:-2], rad_peak_idxs[1:-1],
                              rad_peak_idxs[2:], rad_peak_vals[1:]))
        for slice_idx in slice_idxs[1:-1]:
            this_deck_slice = slice(slice_idx[0]+1, slice_idx[2]-1)
            if slice_idx[3] > 5.0:
                # We didn't land in this period
                continue
            else:
                self.array[this_deck_slice] = seek_deck(rad.array[this_deck_slice],
                                                        hdot.array[this_deck_slice],
                                                        slice_idx[1]-slice_idx[0],
                                                        rad.frequency)
                '''
                import matplotlib.pyplot as plt
                plt.plot(rad.array[this_deck_slice])
                plt.plot(self.array[this_deck_slice])
                plt.show()
                plt.clf()
                '''
