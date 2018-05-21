import unittest
import numpy as np

from analysis_engine.helicopter.flight_phase import (
    Airborne,
    Takeoff,
)

from analysis_engine.node import (A, App, ApproachItem, KTI,
                                  KeyTimeInstance, KPV, KeyPointValue, M,
                                  Parameter, P, S, Section, SectionNode, load)

from analysis_engine.test_utils import buildsection


class TestAirborne(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        node = Airborne
        available = ('Altitude AAL For Flight Phases', 'Fast')
        self.assertFalse(node.can_operate(available,
                                          seg_type=A('Segment Type', 'START_AND_STOP')))
        available = ('Altitude Radio', 'Altitude AGL', 'Gear On Ground', 'Rotors Turning')
        self.assertTrue(node.can_operate(available,
                                         seg_type=A('Segment Type', 'START_AND_STOP')))

    def test_airborne_helicopter_basic(self):
        gog = M(name='Gear On Ground',
                array=np.ma.array([0]*3+[1]*5+[0]*30+[1]*5, dtype=int),
                frequency=1,
                offset=0,
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([2.0]*4+[0.0]*3+[20.0]*30+[0.0]*6, dtype=float))
        rtr = buildsection('Rotors Turning', 0, 40)
        node = Airborne()
        node.derive(agl, agl, gog, rtr)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 3.5)
        self.assertEqual(node[0].slice.stop, 36.95)

    def test_airborne_helicopter_short(self):
        gog = M(name='Gear On Ground',
                array=np.ma.array([0]*3+[1]*5+[0]*10+[1]*5, dtype=int),
                frequency=1,
                offset=0,
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([2.0, 0.0, 0.0]+[0.0]*4+[20.0]*10+[0.0]*6, dtype=float))
        rtr = buildsection('Rotors Turning', 0, 40)
        node = Airborne()
        node.derive(agl, agl, gog, rtr)
        self.assertEqual(len(node), 1)

    def test_airborne_helicopter_radio_refinement(self):
        '''
        Confirms that the beginning and end are trimmed to match the radio signal,
        not the (smoothed) AGL data.
        '''
        gog = M(name='Gear On Ground',
                array=np.ma.array([0]*3+[1]*5+[0]*10+[1]*5, dtype=int),
                frequency=1,
                offset=0,
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([0.0]*6+[20.0]*12+[0.0]*5, dtype=float))
        rad = P(name='Altitude Radio',
                array=np.ma.array([0.0]*7+[10.0]*10+[0.0]*6, dtype=float))
        rtr = buildsection('Rotors Turning', 0, 40)
        node = Airborne()
        node.derive(rad, agl, gog, rtr)
        self.assertEqual(node[0].start_edge, 6.1)
        self.assertEqual(node[0].stop_edge, 16.9)

    def test_airborne_helicopter_overlap(self):
        gog = M(name='Gear On Ground',
                array=np.ma.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=int),
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 2, 0], dtype=float),
                frequency=0.2)
        rtr = buildsection('Rotors Turning', 0, 40)
        node = Airborne()
        node.derive(agl, agl, gog, rtr)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 3.2)
        self.assertEqual(node[0].slice.stop, 6)
        self.assertEqual(node[1].slice.start, 8)
        self.assertEqual(node[1].slice.stop, 10.5)

    def test_airborne_helicopter_cant_fly_without_rotor_turning(self):
        gog = M(name='Gear On Ground',
                array=np.ma.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=int),
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 2, 0], dtype=float),
                frequency=0.2)
        rtr = buildsection('Rotors Turning', 0, 0)
        node = Airborne()
        node.derive(agl, agl, gog, rtr)
        self.assertEqual(len(node), 0)


class TestTakeoff(unittest.TestCase):
    def test_can_operate(self):
        # Airborne dependency added to avoid trying to derive takeoff when
        # aircraft's dependency
        available = ('Heading Continuous', 'Altitude AAL For Flight Phases',
                     'Fast', 'Airborne')
        seg_type = A('Segment Type', 'START_AND_STOP')
        #seg_type.value = 'START_ONLY'
        self.assertFalse(Takeoff.can_operate(available, seg_type=seg_type))
        available = ('Altitude AGL', 'Liftoff')
        self.assertTrue(Takeoff.can_operate(available, seg_type=seg_type))

    # TODO: Create testcases for helicopters. All testcases covered aeroplanes
    @unittest.skip('No helicopter testcases prior to split.')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')
