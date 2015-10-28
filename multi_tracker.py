# multi_tracker.py Jesse Wallace September 2015
# Tracks multiple bees using Kalman filters

import argparse
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.spatial.distance import cdist
from sklearn.utils.linear_assignment_ import linear_assignment
import sys
assert Axes3D   # Hack to stop pyflakes throwing W0611 imported but unused error


def get_log_kernel(sigma, show_wireframe=False):
    '''
    Generate Laplacian of a Gaussian kernel.
    Args:
        sigma - float
        radius - integer number of fields between centre of kernel and edge of
                kernel. The resulting kernel will then be of shape
                (2 * radius + 1, 2 * radius + 1)
    Returns:
        Laplacian of a Gaussian kernel.
    '''
    radius = int(sigma * 2)
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = np.float64(i - (radius))
            y = np.float64(j - (radius))
            frac = (x ** 2 + y ** 2) / (2 * sigma ** 2)
            kernel[i, j] = (frac - 1) * np.exp(-frac) / (np.pi * sigma ** 4)

    if show_wireframe:
        x, y = np.meshgrid(range(kernel.shape[0]), range(kernel.shape[1]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(x, y, kernel)
        plt.show()

    return kernel


# @profile
def process_frame(frame, bee_number, log_kernel, roi=[0, 0, -1, -1],
                  roi_mask=None, scale=1.0, thresh_kernel_size=101):
    '''
    Processes frame to find bee locations.
    Args:
        frame - image
        bee_number - integer number of bees
        log_kernel - Laplacian of a Gaussian kernal to convolve image with
        roi - list containing top left and bottom right coordinates (indices)
                of region of interest. By default, whole frame.
        scale - float <= 1.0 scaling factor for frames.
    Returns:
        List of intermediate process images sorted in revers order (index 0 is
        final processed image).
    '''
    # Set region of interest and change to grayscale
    if scale == 1.0:
        frameroi = cv2.cvtColor(frame[roi[0]:roi[2], roi[1]:roi[3]],
                                cv2.COLOR_BGR2GRAY)
    else:
        froi = cv2.cvtColor(frame[roi[0]:roi[2], roi[1]:roi[3]],
                            cv2.COLOR_BGR2GRAY)
        frameroi = cv2.resize(froi, (0, 0), fx=scale, fy=scale)

    h, w = frameroi.shape
    if roi_mask is None:
        circlemask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circlemask, (w / 2, h / 2), int(0.55 * min(h, w)), 255, -1)
    else:
        circlemask = roi_mask

    # Apply Laplacian of Gaussian convolution and linearly transform pixels into
    # range [0.0, 1.0]
    p = cv2.filter2D(frameroi, cv2.CV_32F, log_kernel)
    pmax = np.amax(p)
    pmin = np.amin(p)
    norm = (p - pmin) * (1 / (pmax - pmin))
    norm8 = cv2.convertScaleAbs(norm, alpha=255.0)

    # Threshold smoothed 8 bit image and erode to generate mask
    a_thresh = cv2.adaptiveThreshold(norm8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, thresh_kernel_size, -35)
    a_threshroi = cv2.bitwise_and(a_thresh, circlemask)
    mask = cv2.erode(a_threshroi, np.ones((3, 3)), iterations=1).astype(bool)

    # Threshold smoothed image
    # ret1, thresh = cv2.threshold(normalised, 0.70, 1.0, cv2.THRESH_TOZERO)
    # ret2, mask = cv2.threshold(normalised, 0.70, 1.0, cv2.THRESH_BINARY_INV)

    # Subtract the dilation to find local maxima, then linearly transform pixels
    # into range [0.0, 1.0]. The result is an image with values 1.0 at maxima
    # (bees), 0.0 at a contour line around the bees, and 0.5 background.
    l_max = np.array((norm == cv2.dilate(norm, np.ones((3, 3)))) * mask,
                     dtype=np.float32)

    return [l_max, mask, a_threshroi, norm8, norm, frameroi, frame]


def reassign(assignment, n, costs, max_dist, weights):
    '''
    Reassignment for non-linear assignment case. This tries to reassign
    assignments which have been removed for being not close enough.
    Args:
        assignment - nx2 array assignment as it stands
        n - number of objects to assign
        costs - cost matrix
        max_dist - cost threshold
        weights - for determining preference to assign to observations
    Returns:
        updated nx2 assignment
    '''
    l = assignment
    if len(l) < n:
        unassigned_set = set(range(n)) - set(l[:, 0])
        count_dict = {key: 0 for key in range(costs.shape[1])}
        for i in unassigned_set:
            try:
                if np.amin(costs[i, :]) < max_dist:
                    # Set up list of possible assignments
                    sort_cost_idx = np.argsort(costs[i, :])
                    poss_assgn_idx = sort_cost_idx[costs[i, :][sort_cost_idx] <
                                                   max_dist]
                    unassigned = True

                    # Try to assign to an unassigned observation close to
                    # prediction
                    for idx in poss_assgn_idx:
                        if count_dict[idx] == 0:
                            extra_assignment = np.array([i, idx])
                            count_dict[idx] = 1
                            unassigned = False
                            break
                        else:
                            continue

                    # Assign all remaining Kalman filters to observation with
                    # highest weight close to prediction. This addressess three
                    # bee interactions but is unstable in four bee interactions.
                    if unassigned:
                        assignment_idx = np.argmax(weights[poss_assgn_idx])
                        extra_assignment = np.array([i, assignment_idx])
                        count_dict[assignment_idx] += 1

                    # Append reassignment
                    l = np.vstack([l, extra_assignment])
            except ValueError:
                break
    return l


def assign(observed, predicted, max_dist):
    '''
    Assigns observed locations of bees based on predicted locations using the
    Hungarian algorithm.
    Args:
        observed - numpy array with shape (3, n) containing observed coordinates
                   and a third row containing weights for assignment in the case
                   where more than 2 bees are coassigned and then split.
        predicted - numpy array with first two lines containing predicted coords
        max_dist - maximum distance between predicted and observed
    Returns:
        an nx3 array of indices (predicted_index, observed_index, non_linear)
    '''
    costs = cdist(predicted[0:2, :].transpose(), observed[0:2, :].transpose(),
                  'euclidean')
    l = linear_assignment(costs)
    rm_list = []

    # Remove assignments which make the distance between observed and predicted
    # greater than max_dist
    for row in range(l.shape[0]):
        if costs[l[row, 0], l[row, 1]] > max_dist \
                and (predicted[0:2, ] != 0).all():
            rm_list.append(row)
    l = np.delete(l, rm_list, axis=0)

    l = reassign(l, predicted.shape[1], costs, max_dist,
                 observed[2, :])

    # Calculate co-assignments
    l = np.hstack([l, np.zeros((l.shape[0], 1), dtype=l.dtype)])
    u, u_counts = np.unique(l[:, 1], return_counts=True)
    for i in range(l.shape[0]):
        u_idx = np.where(u == l[i, 1])[0]
        l[i, 2] = u_counts[u_idx]
    return l


class MultiKalman:
    '''
    Stores list of Kalman filters for multiple bee tracking
    '''
    def __init__(self, bee_number, max_dist, reset_time):
        self.tracks = []
        self.found_dict = {}
        self.time_dict = {}
        self.track_dict = {}
        self.last_track = 0
        self.max_dist = max_dist
        self.reset_time = reset_time
        self.prev_assignment = np.vstack([[i, i, 0] for i in range(bee_number)])
        # Define Kalman filter
        self.transitionMatrix = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 0.9, 0],
                                          [0, 0, 0, 0.9]], dtype=np.float32)
        self.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], dtype=np.float32)
        self.processNoiseCov = np.array([[0.1, 0, 0, 0],
                                         [0, 0.1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], dtype=np.float32)
        for i in range(bee_number):
            self.tracks.append(cv2.KalmanFilter(4, 2, type=cv2.CV_32F))
            self.tracks[i].transitionMatrix = self.transitionMatrix.copy()
            self.tracks[i].measurementMatrix = self.measurementMatrix.copy()
            self.tracks[i].processNoiseCov = self.processNoiseCov.copy()
            self.found_dict[self.tracks[i]] = False
            self.track_dict[self.tracks[i]] = self.last_track
            self.last_track += 1

    def re_init(self, i):
        '''
        Reinitialises given kalman filter in self.tracks.
        Args:
            i - index of Kalman filter in self.tracks to reinitialise
        '''
        # Delete time dictionary entry
        del self.time_dict[self.tracks[i]]

        # Reinitialise
        self.tracks[i] = cv2.KalmanFilter(4, 2, type=cv2.CV_32F)
        self.tracks[i].transitionMatrix = self.transitionMatrix.copy()
        self.tracks[i].measurementMatrix = self.measurementMatrix.copy()
        self.tracks[i].processNoiseCov = self.processNoiseCov.copy()
        self.found_dict[self.tracks[i]] = False
        self.track_dict[self.tracks[i]] = self.last_track
        self.last_track += 1

    def predict(self, time):
        '''
        Calls kf.predict() for each kf in self.tracks, provided the maximum time
        allowable between observations has not been met.
        Args:
            time - Current time in movie in secs
        '''
        for i in range(len(self.tracks)):
            # Reinitialise if maximum time between observations has passed
            if self.found_dict[self.tracks[i]]:
                if time - self.time_dict[self.tracks[i]] > self.reset_time:
                    self.re_init(i)

            # Calculate velocity and update self.time_dict with current time
            if self.found_dict[self.tracks[i]]:
                dt = time - self.time_dict[self.tracks[i]]
                self.tracks[i].transitionMatrix[0, 2] = dt
                self.tracks[i].transitionMatrix[1, 3] = dt
                # self.time_dict[self.tracks[i]] = time
        # Call kf.predict() on each Kalman Filter in self.tracks
        return np.hstack(kf.predict() for kf in self.tracks).astype(np.float32)

    def correct(self, unassigned_pos, predicted_pos, current_time):
        '''Calls KalmanFilter.correct on each kalman filter in self.track with
        an appropriate assignment of observations to each kalman filter.
        '''
        assignment = assign(unassigned_pos, predicted_pos, self.max_dist)
        for index in assignment:
            i0, i1 = index[0], index[1]
            if not self.found_dict[self.tracks[i0]]:
                self.found_dict[self.tracks[i0]] = True
                self.tracks[i0].statePost = np.array([[unassigned_pos[0, i1]],
                                                      [unassigned_pos[1, i1]],
                                                      [0],
                                                      [0]], dtype=np.float32)
                self.tracks[i0].statePre = np.array([[unassigned_pos[0, i1]],
                                                     [unassigned_pos[1, i1]],
                                                     [0],
                                                     [0]], dtype=np.float32)
            self.time_dict[self.tracks[i0]] = current_time
            self.tracks[i0].correct(unassigned_pos[:2, i1])

        # Split all tracks which have few co-assignments in the current frame
        # than they did in the previous frame.
        prev_nonlin = self.prev_assignment[self.prev_assignment[:, 2] > 1, :]
        for prev_assgn in prev_nonlin:
            check_assgn = assignment[assignment[:, 0] == prev_assgn[0], :]
            for assgn in check_assgn:
                if assgn[2] < prev_assgn[2]:
                    kf_idx = assgn[0]
                    self.track_dict[self.tracks[kf_idx]] = self.last_track
                    self.last_track += 1

        self.prev_assignment = assignment

    def write_coords(self, current_time, out_file, scale_factor=1.0):
        '''
        Writes output coordinates to csv file
        Args:
            current_time - current capture time
            out_file - file to output data to
            scale_factor - float coordinates are multiplied by this before
                           writing.
        Format:
            time,[trackNumber,y,x] * number of tracks
        '''
        out_list = [current_time]
        for kf in self.tracks:
            out_list.extend([self.track_dict[kf], kf.statePost[0, 0],
                             kf.statePost[1, 0]])
        out_file.write(('%f' + ',%i,%f,%f' * len(self.tracks) + '\n')
                       % tuple(out_list))


def print_process_header(filename, cap, total_frames, fps, quiet):
    '''
    Prints header for process_video
    Args:
        filename - name of file
        cap - cv2.VideoCapture instance
        total_frames - integer number of frames
        fps - frames per second of movie file
        quiet - if True, do nothing and return None
    Returns: None
    '''
    if quiet:
        return None
    else:
        print '{}\nDimensions: {}x{}    Number of frames: {}    FPS: {}'.format(
            filename,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames,
            fps)
        return None


def show_progress(done_frames, total_frames, time_at_last_call, frames):
    '''
    Displays a progress bar and ETA.
    Args:
        done_frames - integer number of frames processed
        total_frames - total number of frames in movie file
        time_at_last_call - time at last iteration
        frames - number of frames processed since last call
    Returns:
        time at this iteration
    '''
    if done_frames > total_frames:
        return time.clock()

    complete = (100 * done_frames) / total_frames
    toc = time.clock()
    fps = frames / (toc - time_at_last_call)
    eta = (total_frames-done_frames)/int(fps)
    em, es = divmod(eta, 60)
    eh, em = divmod(em, 60)
    sys.stdout.write(
        '\r|{:<50}|{:>3d}% FPS{:>6.1f} ETA {:02d}:{:02d}:{:02d}'.format(
            '-'*(complete/2), complete, fps, eh, em, es))
    sys.stdout.flush()
    if done_frames == total_frames:
        print

    return time.clock()


def draw_points(image, mkf,
                colors=((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 180, 180),
                        (180, 180, 0), (180, 0, 180), (0, 0, 0),
                        (127, 127, 127))):
    '''
    Draws tracking points on image
    Args:
        image - input image
        mkf - MultiKalman instance for process
    Returns:
        drawn on image
    '''
    if len(image.shape) == 2:    # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    col_int = 0
    for kf in mkf.tracks:
        pt1 = tuple(kf.statePre.flat[:2])[::-1]
        pt2 = tuple(kf.statePost.flat[:2])[::-1]
        image = cv2.line(image, pt1, pt2, colors[col_int], thickness=5)
        col_int += 1

    return image


def display_frame(p_list, show_index, name, draw_kalman=False, mkf=None):
    '''
    Displays resulting frame and defines a keypress command interface for
    which frame to show.
    Args:
        p_list - list of possible frames to display
        show_index - index of frame to display
        draw_kalman - boolean, determines if tracking points are displayed,
                      requires a mkf as an argument
        mkf - MultiKalman instance for process
    Returns:
        new_show_index, new_draw_kalman
    '''
    if p_list[show_index].dtype == bool:
        image = p_list[show_index].astype(np.float32)
    else:
        image = p_list[show_index]
    if draw_kalman and show_index != len(p_list) - 1:
        assert isinstance(mkf, MultiKalman)
        cv2.imshow(name, draw_points(image, mkf))
    else:
        cv2.imshow(name, image)
    keypress = cv2.waitKey(1)
    if keypress & 0xFF == ord('q'):
        return -1, -1
    elif keypress in [ord('%i' % i) for i in range(len(p_list))]:
        show_index = int(chr(keypress))
    elif keypress == ord('k'):
        draw_kalman = not draw_kalman
    return show_index, draw_kalman


def get_thresh_kernel_size(roi, scale):
    '''
    Calculates appropriate size for adaptive threshold kernel based on size of
    region of interest
    Args:
        roi - list of length 4 containing region of interest
    Returns:
        odd integer kernel size
    '''
    if roi == [0, 0, -1, -1]:
        thresh_k_size = int(scale * 101)
    else:
        thresh_k_size = int(scale * (roi[2] - roi[0] + roi[3] - roi[1]) / 10)
    thresh_k_size = thresh_k_size + (thresh_k_size + 1) % 2

    return thresh_k_size


def get_roi_mask(roi, scale):
    '''
    If roi is well defined, returns a circle mask.
    '''
    if -1 in roi:
        return None
    else:
        h, w = roi[2] - roi[0], roi[3] - roi[1]
        circlemask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circlemask, (w / 2, h / 2), int(0.55 * min(h, w)), 255, -1)
        roi_mask = cv2.resize(circlemask, (0, 0), fx=scale, fy=scale)

    return roi_mask


def get_out_filepath(filename, scale, outpath=''):
    '''
    '''
    if outpath == '':
        out_filename = filename[:filename.find('.')] + '%s-traj.csv' % scale
    else:
        filebsenm = os.path.basename(filename)
        out_filebsenm = filebsenm[:filebsenm.find('.')] + '-%s-traj.csv' % scale
        out_filename = os.path.join(outpath, out_filebsenm)

    return out_filename


# @profile
def process_video(filename, bee_number, s, roi=[0, 0, -1, -1], scale=1.0,
                  show_video=0, discard=0, fps=25.0, duration=(60 * 60 * 1000),
                  max_dist=50, reset_time=0.5, quiet=False, outpath=''):
    '''Processes video by running process_frame for each frame in video.
    Args:
        filename - string, name of video to be processed
        bee_number - integer, number of bees to be expected
        s - int or float, sigma value for laplacian of gaussian kernel
        roi - list containing top left and bottom right coordinates (indices)
                of region of interest. By default, whole frame.
        scale - float <= 1.0 scale frames by scaling factor
        show_video - int, index of process_frame output to display. if out of
                    range, no video will be displayed.
        discard - int, number of frames to discard at start of movie (if camera
                    takes a while to correct exposure)
        fps - float frames per second of video, defaults to 25.0
        duration - duration of video in milliseconds, defaults to 1 hour
        max_dist - int distance threshold for assigning observations to bees
        reset_time - float time in seconds before unassigned Kalman filter is
                     reinitialised
    '''
    show_index = show_video
    draw_kalman = True

    circlemask = get_roi_mask(roi, scale)
    k = get_log_kernel(s)
    mkf = MultiKalman(bee_number, max_dist, reset_time)
    cap = cv2.VideoCapture(filename)
    thresh_k_size = get_thresh_kernel_size(roi, scale)

    total_frames = int(fps * duration / 1000)
    print_process_header(filename, cap, total_frames, fps, quiet)

    done_frames = 0
    for i in range(discard):
        cap.read()
        done_frames += 1

    out_filename = get_out_filepath(filename, scale, outpath=outpath)
    out = open(out_filename, 'w')

    start_time = time.clock()
    tictoc = time.clock()

    while 1:

        # Read next frame, process and detect. End loop if movie complete.
        ret, frame = cap.read()
        if not ret:
            break
        done_frames += 1
        p = process_frame(frame, bee_number, k, roi=roi, roi_mask=circlemask,
                          scale=scale, thresh_kernel_size=thresh_k_size)

        # Update Kalman filters with tracking observations
        detected_coords = np.array(np.where(p[0] == 1.0), dtype=np.float32)
        detected_weights = []
        for coord in detected_coords.transpose():
            detected_weights.append(p[1][coord[0], coord[1]])

        observed = np.vstack((detected_coords, detected_weights))
        capture_time = done_frames / fps
        pred_coords = mkf.predict(capture_time)
        mkf.correct(observed, pred_coords, capture_time)

        # Display the resulting frame if show_video is in range, and change
        # which video is displayed on button press.
        if show_video in range(len(p)):
            show_index, draw_kalman = display_frame(p, show_index, filename,
                                                    draw_kalman, mkf)
            if (show_index, draw_kalman) == (-1, -1):
                break

        # Show percentage complete
        if (done_frames) % 100 == 0 and not quiet:
            tictoc = show_progress(done_frames, total_frames, tictoc, 100)

        # Output to csv file
        mkf.write_coords(capture_time, out)
        last_time = time.clock()

    # Finalise
    tot_time = last_time - start_time
    m, s = divmod(int(tot_time), 60)
    h, m = divmod(m, 60)
    ave_fps = done_frames / tot_time
    if not quiet:
        print_done((filename, out_filename, done_frames, ave_fps, h, m, s))
    if show_video in range(len(p)):
        cv2.destroyAllWindows()

    cap.release()
    out.close()
    return filename, out_filename, done_frames, ave_fps, h, m, s


def print_done(tup):
    '''
    Prints summary
    Args:
        tup - tuple containing:
            filename, out_filename, done_frames, fps, h, m, s:
    '''
    print '{} -> {}\n{} Frames, {:.1f} FPS, {:02d}:{:02d}:{:02d}'.format(
        tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6])


def parse_conditions(path_tup):
    '''
    Parses conditions file.
    Args:
        path_tup - path of conditions file, movie directory
    Returns:
        list of movie filenames, dictionary with filenames as keys and
        bee_number as values
    '''
    b_ref = {'1': 1, '2': 2, '3': 2, '4': 4}
    csv_file = open(path_tup[0], 'r')
    csv_reader = csv.reader(csv_file)
    movie_list = []
    b = {}
    r = {}
    for line in csv_reader:
        movie = path_tup[1] + line[0]
        b[movie] = b_ref[line[1]]
        r[movie] = [int(line[i]) for i in range(2, 6)]
        movie_list.append(movie)

    return movie_list, b, r


def create_dir(path):
    '''
    '''
    if len(path) > 0:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


def main():
    # Set up command line interface
    parser = argparse.ArgumentParser(description='Process multiple bee movies.')

    parser.add_argument('-b', default=4, type=int, required=False,
                        metavar='BeeNumber',
                        help='Specifies number of bees in video.')

    parser.add_argument('-M', default=1, type=int, help='''Default 1. Non-1
                        value enables multiprocessing to process multiple files
                        in parallel. 0 Enables maximum processors.
                        Enables quiet mode. Not recommended to use in
                        conjunction with 'v'.''')

    parser.add_argument('-s', default=16, type=float, required=False,
                        metavar='Sigma',
                        help='''Sigma value for laplacian of gaussian kernel.
                        (default 16)''')

    parser.add_argument('-r', nargs=4, default=[0, 0, -1, -1], type=int,
                        required=False, metavar=('r0', 'c0', 'r1', 'c1'),
                        help='''Region of interest (default whole frame).
                        Coordinates given are the top right and bottom left
                        corners of the square region of interest.''')

    parser.add_argument('-S', default=1.0, type=float, required=False,
                        metavar='ScaleFactor', help='''Scale video down for
                        faster processing. Note, this will also scale the -s
                        parameter.''')

    parser.add_argument('-v', action='store_true',
                        help='''Display video. By default nothing shown.
                        Keypress commands during processing:
                        <0 - 6> changes which processing step is displayed.
                        <k>     toggles display of Kalman filter.
                        <q>     quits and stop processing.''')

    parser.add_argument('-q', action='store_true',
                        help='Only print completion line')

    parser.add_argument('-d', default=0, type=int, required=False,
                        metavar='DiscardFrames',
                        help='Discard the first d frames.')

    parser.add_argument('-f', default=25.0, type=float, required=False,
                        metavar='FPS',
                        help='Frames per second of movie file (default 25.0)')

    parser.add_argument('-D', default=(60 * 60 * 1000), type=int,
                        metavar='Duration', required=False,
                        help='Duration of video in milliseconds (default 1hr)')

    parser.add_argument('-m', default=25, type=int, required=False,
                        metavar='MaxDistance',
                        help='''Distance threshold for assigning observations to
                        bees (default 25).''')

    parser.add_argument('-t', default=0.5, type=float, required=False,
                        metavar='MaxTimeOut',
                        help='''Time threshold for reinitialising unassigned
                        Kalman filters (default 0.5 seconds).''')

    parser.add_argument('-c', default=['', ''], type=str, required=False,
                        nargs=2, metavar=('ConditionsCSV', 'MovieDir'),
                        help='''Input Conditions CSV file. If this is set
                        then MovieFiles, -r and -b will be ignored.''')

    parser.add_argument('-o', default='', type=str, required=False,
                        metavar='Path', help='''Path to output trajectory csv
                        files to. (Default is same directory as movies)''')

    parser.add_argument('MovieFiles', type=str, nargs='*',
                        help='The paths of each movie file to be processed.')

    args = parser.parse_args()

    print 'Process started at %s' % time.strftime('%c')

    # Create directory
    create_dir(args.o)

    # Run process_video
    if args.v:
        vid_index = 5
    else:
        vid_index = -1

    if len(args.c[0]) > 0:
        movie_files, b, r = parse_conditions(args.c)
    else:
        movie_files = args.MovieFiles
        r = {}
        b = {}
        for filename in movie_files:
            b[filename] = args.b
            r[filename] = args.r

    sigma = args.s * args.S

    s_time = time.clock()
    if args.M == 1:
        for filename in movie_files:
            print 'Processing %s' % filename
            process_video(filename, b[filename], sigma, roi=r[filename],
                          scale=args.S, show_video=vid_index, discard=args.d,
                          max_dist=args.m * args.S, fps=args.f,
                          duration=args.D, reset_time=args.t, quiet=args.q,
                          outpath=args.o)
    else:
        # Multiple processes will be used
        if args.M == 0:
            p = Pool()
        else:
            p = Pool(args.M)
        for filename in movie_files:
            print 'Processing %s' % filename
            p.apply_async(process_video,
                          args=(filename, b[filename], sigma),
                          kwds={'roi': r[filename], 'scale': args.S,
                                'show_video': vid_index,
                                'discard': args.d, 'fps': args.f,
                                'duration': args.D, 'max_dist': args.m * args.S,
                                'reset_time': args.t, 'quiet': True,
                                'outpath': args.o},
                          callback=print_done)

        # Close processes when done
        p.close()
        p.join()

    t = time.clock() - s_time
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    print 'Done... Processed {} files in {:02d}:{:02d}:{:02d}'.format(
        len(movie_files), h, m, s)

if __name__ == '__main__':
    main()
