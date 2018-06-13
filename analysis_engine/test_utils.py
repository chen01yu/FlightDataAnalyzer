'''
Three little routines to make building Sections for testing easier.
'''
from analysis_engine.node import (
    Section,
    SectionNode,
    KeyTimeInstance,
    KTI,
)

def builditem(name, begin, end, start_edge=None, stop_edge=None):
    '''
    This code more accurately represents the aligned section values, but is
    not suitable for test cases where the data does not get aligned.

    if begin is None:
        ib = None
    else:
        ib = int(begin)
        if ib < begin:
            ib += 1
    if end is None:
        ie = None
    else:
        ie = int(end)
        if ie < end:
            ie += 1

    :param begin: index at start of section
    :param end: index at end of section
    '''
    slice_end = end if end is None else end + 1
    return Section(name, slice(begin, slice_end, None), start_edge or begin, stop_edge or end)


def buildsection(name, *args):
    '''
    A little routine to make building Sections for testing easier.

    :param name: name for a test Section
    :returns: a SectionNode populated correctly.

    Example: land = buildsection('Landing', 100, 120)
    '''
    return SectionNode(name, items=[builditem(name, *args)])


def buildsections(name, *args):
    '''
    Like buildsection, this is used to build SectionNodes for test purposes.

    lands = buildsections('name',[from1,to1],[from2,to2])

    Example of use:
    approach = buildsections('Approach', [80,90], [100,110])None
    '''
    return SectionNode(name, items=[builditem(name, *a) for a in args])


def build_kti(name, *args):
    return KTI(items=[KeyTimeInstance(a, name) for a in args if a])
