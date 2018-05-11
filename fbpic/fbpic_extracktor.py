# Copyright 2018 Samuel Yoffe
# University of Strathclyde, Glasgow (UK)
# Extract particle tracks from FBPIC simulation data.
# (Currently written for integer particle IDs).

import h5py
import numpy as np
import multiprocessing as mp

from random import sample
from glob import glob
from scipy.constants import m_e, c
from os import path, makedirs


class Trajectory:
    """
    Class for a single particle trajectory. The `Trajectory` class contains:
    - `ID`, the particle's tag/ID
    - `labels`, a list of trajectory properties, e.g. ['x', 'y', 'z', 'px', 'py', 'pz', 'ene', 'ct']
    - `data`, a dict with keys specified in `labels`, with its element being a list containing the corresponding data
    """

    def __init__(self, tag):
        """
        Initialise the trajectory by setting the ID and creating the data dict.

        Parameters
        ----------
        tag: int or str
            The particle's tag/ID
        """
        self.ID = tag
        self.labels = ['x', 'y', 'z', 'px', 'py', 'pz', 'ene', 'ct']
        self.data = {l: [] for l in self.labels}
        self.mc = m_e * c

    def add_point(self, t, point):
        """
        Add a data point to the trajectory. Kinetic energy (units mc^2, i.e. (\gamma - 1)) calculated from momentum.

        Parameters
        ----------
        t: float
            The time
        point: tuple
            Tuple containing the position and momentum, (x, y, z, px, py, pz)
        """
        for i, val in enumerate(point):
            self.data[self.labels[i]].append(val)
        self.data['ct'].append(c*t)
        self.data['ene'].append(np.sqrt(1.0 + (self.data['px'][-1]**2
                                               + self.data['py'][-1]**2
                                               + self.data['pz'][-1]**2) / self.mc**2) - 1)


class TrajectoryList:
    """
    Class to store and manipulate a collection of `Trajectory` objects. The class contains:
    - `N`, the number of trajectories stored in the collection; and
    - `track`, a dict of `Trajectory` objects, with particle ID used as key.
    """
    def __init__(self):
        """
        Initialise an empty trajectory list.
        """
        self.N = 0
        self.track = {}

    def resize(self, trajectories):
        """
        Resize the trajectory list. Kinetic energy (units mc^2, i.e. (\gamma - 1)) calculated from momentum.

        Parameters
        ----------
        trajectories: List[int]
            A list of particle IDs for which a trajectory should be created.
        """
        self.N = len(trajectories)
        self.track = {i: Trajectory(i) for i in trajectories}


def test_condition(t, pos, mom):
    """
    The condition used to test if a particle should be added to the list of tracked IDs.

    Parameters
    ----------
    t: float
        The current time (in seconds)
    pos: tuple(float, float, float)
        A tuple containing the position (x, y, z) (m)
    mom: tuple(float, float, float)
        A tuple containing the momentum components (px, py, pz) (kg m/s)

    Returns
    -------
    boolean
    """
    x, y, z = pos
    px, py, pz = mom
    if pz > 200 * m_e * c:
        return True
    return False


def read_file(filename):
    """
    Read particles from an FBPIC HDF5 data file and find particle IDs satisfying `test_condition(t, pos, mom) == True`.

    Parameters
    ----------
    filename: str
        The HDF5 file to process

    Returns
    -------
    A `set()` of valid particle IDs
    """
    tags = set()
    with h5py.File(filename, 'r') as df:
        ts = list(df['data'])[0]
        data = df['data/{}'.format(ts)]
        time = data.attrs['time']

        total = len(data['particles/electrons/id'])
        selected = sample(range(total), int(rfrac * total))
        for idx in selected:
            position = (data['particles/electrons/position/'+x][idx] for x in data['particles/electrons/position'])
            momentum = (data['particles/electrons/momentum/'+x][idx] for x in data['particles/electrons/momentum'])
            if test_condition(time, position, momentum):
                tags.add(data['particles/electrons/id'][idx])
    return tags


def partial_generator(data, n):
    """
    Create a generator to split the particle data into chunks to be processed simultaneously.

    Parameters
    ----------
    data: HDF5 object
        The HDF5 data (= f['data/step'])
    n: int
        The number of elements to return

    Returns
    -------
    An iterable generator
    """
    for i in range(0, len(data['particles/electrons/position/x']), n):
        yield zip(data['particles/electrons/position/x'][i:i + n],
                  data['particles/electrons/position/y'][i:i + n],
                  data['particles/electrons/position/z'][i:i + n],
                  data['particles/electrons/momentum/x'][i:i + n],
                  data['particles/electrons/momentum/y'][i:i + n],
                  data['particles/electrons/momentum/z'][i:i + n],
                  data['particles/electrons/id'][i:i + n])


def find_track_points(data):
    """
    Collect a list of tracked particles at this timestep.

    Parameters
    ----------
    data: list of particles
        Output from partial_generator, a list of particles to check. Each particle is a tuple:
        (x,y,z,px,py,pz,id)

    Returns
    -------
    A list of the particles which are being tracked.
    """
    points = []
    for point in data:
        if point[-1] in tracks.track:
            points.append(point)
    return points


def read_tags(tagsfile):
    """
    Read a list of particle IDs (tags) to be tracked from file.

    Parameters
    ----------
    tagsfile: text file
        Text file listing particle IDs to track.

    Returns
    -------
    A numpy array containing the ID tags.
    """
    print('>>> Reading tags from \"{}\"'.format(tagsfile))
    tags = np.loadtxt(tagsfile, comments=['#', '!'], dtype=int)
#    with open(tagsfile, 'r') as tf:
#        tags = [int(tag) for tag in tf.read().split()]
    print('\r\033[1A\033[0;32m>>>\033[0m')
    print('    Processed {} tags'.format(len(tags)))
    return tags


def find_tracks_from_files(inputfiles):
    """
    Find tracks for particles satisfying `test_condition` by searching through the list of HDF5 files.

    Parameters
    ----------
    inputfiles: list of file names

    Returns
    -------
    A list of particle IDs which are to be tracked.
    """
    tags = set()
    nfiles = len(inputfiles)

    print('>>> Extracting IDs')
    with mp.Pool(threads) as wp:
        for prog, results in enumerate(wp.imap_unordered(read_file, inputfiles), 1):
            tags |= results
            comp = int(bars * prog / nfiles)
            print('\r\033[2K    File {}/{}: '.format(prog + 1, nfiles)
                  + '|\033[42m' + ' ' * comp + '\033[0m' + '-' * (bars - comp)
                  + '|  {:.0%}'.format(prog / nfiles), end='', flush=True)
    print('\r\033[1A\033[0;32m>>>\033[0m')
    print('\r\033[2K    {} tracks extracted from {} files'.format(len(tags), len(inputfiles)))

    return tags


def get_tracks_from_files(inputfiles):
    """
    Extract the trajectory (from the input data files) for each particle whose ID has been selected (either from an
    input tags file, or satisfying `test_condition`

    Parameters
    ----------
    inputfiles: list of HDF5 file names

    """
    nfiles = len(inputfiles)
    print('>>> Extracting tracks')
    total_points = 0
    for fidx, f in enumerate(inputfiles):
        with h5py.File(f, 'r') as df:
            fcomp = int(bars * fidx / nfiles)
            print('\r\033[2K    File {:>{}}/{}:'.format(fidx + 1, len(str(nfiles)), nfiles) +
                  '  |\033[42m' + ' ' * fcomp + '\033[0m' + '-' * (bars - fcomp) +
                  '| {:>4.0%}'.format(fidx / nfiles), end='', flush=True)
            ts = int(list(df['data'])[0])
            data = df['data/{}'.format(ts)]
            time = data.attrs['time']

            with mp.Pool(threads) as wp:
                howmany = len(data['particles/electrons/id'])
                pieces = threads
                split = int(howmany / pieces) if howmany > pieces else 1
                pieces = round(howmany / split + 0.5)

                for prog, points in enumerate(wp.imap_unordered(find_track_points, partial_generator(data, split)), 1):
                    for point in points:
                        tracks.track[point[-1]].add_point(time, point[:-1])
                    total_points += len(points)
                    comp = int(bars * prog / pieces)
                    print('\r\033[60C|\033[44m' + ' ' * comp + '\033[0m' + '-' * (bars - comp)
                          + '|  {:.0%}'.format(prog / pieces), end='', flush=True)
    print('\r\033[1A\033[0;32m>>>\033[0m')
    print('\r\033[2K    {} points processed'.format(total_points))


def write_output(outputfile):
    """
    Write the particle trajectories to an HDF5 file. (Formatted to match the output for trajectories from Osiris.)

    Parameters
    ----------
    outputfile: a name for the output HDF5 file

    """
    print('>>> Creating HDF5 file \"{}\"'.format(outputfile))
    refresh = int(tracks.N/100)
    with h5py.File(outputfile, 'w') as hf:
        hf.attrs['NAME'] = 'electrons'
        hf.attrs['QUANTS'] = np.array(['{:<16}'.format(l) for l in ['ct', 'x', 'y', 'z', 'px', 'py', 'pz', 'ene']],
                                      dtype='|S16')
        fcount = 0
        for _, tr in tracks.track.items():
            g = hf.create_group(str(tr.ID))
            #      hf['{}'.format(tr.ID)].attrs['tag'] = [int(0),int(tr.ID)]
            for l in tr.data:
                g.create_dataset(l, data=tr.data[l])
            if fcount % refresh == 0:
                comp = int(bars*fcount/tracks.N)
                print('\r\033[2K    |\033[42m'+' '*comp+'\033[0m'+'-'*(bars-comp) +
                      '|  {:.0%}'.format(fcount/tracks.N), end='', flush=True)
            fcount += 1
        print('\r\033[1A\033[0;32m>>>\033[0m')
        print('\r\033[2K    Wrote {} tracks'.format(len(hf)))


def extrackt(inputfiles, outputfile, tagsfile=''):
    """
    The main control function: selects whether to search for tags or use input file, then finds the tracks from the data.

    Parameters
    ----------
    inputfiles: list of HDF5 data file names

    outputfile: output filename for the tracks

    tagsfile: optional list of particle IDs to track.

    """
    odir = path.dirname(outputfile)
    if not path.isdir(odir) and odir != '':
        makedirs(odir)
    if path.exists(outputfile):
        try:
            hf = h5py.File(outputfile, 'w')
            hf.close()
        except OSError:
            print('\033[1;31m>>>\033[0m Output file \"{}\" cannot be opened for writing'.format(outputfile))
            exit()

    if tagsfile:
        track_ids = read_tags(tagsfile)
    else:
        track_ids = find_tracks_from_files(inputfiles)

    tracks.resize(track_ids)

    get_tracks_from_files(inputfiles)

    write_output(outputfile)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', help='path to HDF5 data', default='diags/hdf5', type=str, required=False)
    parser.add_argument('-o', help='HDF5 tracks output file', default='diags/tracks/tracks.h5',
                        type=str, required=False)
    parser.add_argument('--tags', help='tag file', default='', type=str, required=False)
    parser.add_argument('--threads', help='number of threads to use {}'.format(mp.cpu_count()), default=mp.cpu_count(),
                        type=int, required=False)
    parser.add_argument('--bars', help='number of bars to use when measuring progress', default=25, type=int,
                        required=False)
    parser.add_argument('--rfrac', help='random fraction of particles to sample', default=0.001, type=float,
                        required=False)

    args = parser.parse_args()

    input_files = sorted(glob(path.join(args.l, "*.h5")))
    output_file = args.o
    tags_file = args.tags

    threads = args.threads
    bars = args.bars
    rfrac = args.rfrac

    tracks = TrajectoryList()
    extrackt(input_files, output_file, tags_file)
