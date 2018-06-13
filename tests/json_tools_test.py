import simplejson
import unittest

from collections import OrderedDict
from copy import deepcopy
from dateutil.parser import parse

from analysis_engine.json_tools import (
    get_node_class,
    json_to_node,
    json_to_process_flight,
    jsondict_to_node,
    node_to_json,
    node_to_jsondict,
    process_flight_to_json,
    process_flight_to_nodes,
    sort_dict,
)
from analysis_engine.node import (
    ApproachNode,
    DerivedParameterNode,
    FlightPhaseNode,
    KeyPointValueNode,
    KeyTimeInstance,
    KeyTimeInstanceNode,
    MultistateDerivedParameterNode,
    SectionNode,
)


'''
TODO:

 - Expand test data with more node types.
'''


KTI_NAME = 'Altitude When Climbing'

KTI_JSONDICT = {
    "__class__": "KeyTimeInstance",
    "datetime": {
        "type": "datetime",
        "value": "2014-04-12T14:47:56.813991+00:00",
        },
    "index": 419.81399082568805,
    "latitude": 16.137181286511073,
    "longitude": -22.888522004372511,
    "name": "35 Ft Climbing",
}

KTI = KeyTimeInstance(
    KTI_JSONDICT['index'],
    KTI_JSONDICT['name'],
    parse(KTI_JSONDICT['datetime']['value']),
    KTI_JSONDICT['latitude'],
    KTI_JSONDICT['longitude'],
)

PROCESS_FLIGHT_JSON = {
    'approach': {},
    'flight': {},
    'kpv': {},
    'kti': {KTI_NAME: [sort_dict(KTI_JSONDICT)]},
    'phases': {},
    'version': '0.1',
}

PROCESS_FLIGHT = {
    'approach': {},
    'flight': {},
    'kpv': {},
    'kti': {KTI_NAME: [KTI]},
    'phases': {},
}


class TestJsonTools(unittest.TestCase):
    
    def test_json_to_node(self):
        self.assertEqual(json_to_node(simplejson.dumps(KTI_JSONDICT.copy())), KTI)
    
    def test_json_to_process_flight(self):
        process_flight_json = deepcopy(PROCESS_FLIGHT_JSON)
        self.assertEqual(json_to_process_flight(simplejson.dumps(process_flight_json)), PROCESS_FLIGHT)
        # incompatible or missing version does not load
        process_flight_json['version'] = '0.4'
        self.assertEqual(json_to_process_flight(simplejson.dumps(process_flight_json)), {})
        del process_flight_json['version']
        self.assertEqual(json_to_process_flight(simplejson.dumps(process_flight_json)), {})
        
    
    def test_jsondict_to_node(self):
        # TODO: Other types.
        self.assertEqual(jsondict_to_node(KTI_JSONDICT.copy()), KTI)
    
    def test_get_node_class(self):
        self.assertEqual(get_node_class('ApproachNode'), ApproachNode)
        self.assertEqual(get_node_class('DerivedParameterNode'), DerivedParameterNode)
        self.assertEqual(get_node_class('FlightPhaseNode'), FlightPhaseNode)
        self.assertEqual(get_node_class('KeyPointValueNode'), KeyPointValueNode)
        self.assertEqual(get_node_class('KeyTimeInstanceNode'), KeyTimeInstanceNode)
        self.assertEqual(get_node_class('MultistateDerivedParameterNode'), MultistateDerivedParameterNode)
        self.assertEqual(get_node_class('SectionNode'), SectionNode)
    
    def test_node_to_json(self):
        self.assertEqual(node_to_json(KTI), simplejson.dumps(sort_dict(KTI_JSONDICT)))
    
    def test_node_to_jsondict(self):
        self.assertEqual(node_to_jsondict(KTI), sort_dict(KTI_JSONDICT))
    
    def test_process_flight_to_json(self):
        self.assertEqual(process_flight_to_json(deepcopy(PROCESS_FLIGHT)), simplejson.dumps(sort_dict(PROCESS_FLIGHT_JSON), indent=2))
    
    def test_process_flight_to_nodes(self):
        nodes = process_flight_to_nodes(PROCESS_FLIGHT)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(list(nodes.keys()), [KTI_NAME])
        self.assertIn(KTI_NAME, nodes)
        node = nodes[KTI_NAME]
        self.assertTrue(isinstance(node, KeyTimeInstanceNode))
        self.assertEqual(len(node), 1)
        for name in ('index', 'latitude', 'longitude', 'name'):
            self.assertEqual(getattr(node[0], name), KTI_JSONDICT[name])
    
    def test_sort_dict(self):
        unsorted = {
            'a': 1,
            'x': 4,
            'c': 3,
            'b': 2,
            'z': 6,
            'y': 5,
        }
        sorted = OrderedDict((
            ('a', 1),
            ('b', 2),
            ('c', 3),
            ('x', 4),
            ('y', 5),
            ('z', 6),
        ))
        self.assertEqual(sort_dict(unsorted), sorted)

