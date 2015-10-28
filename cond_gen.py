import argparse
import csv
import cv2
import datetime as dt
import os


def get_date(filename, time_offset):
    '''
    Args:
        filename - name in format raspberrypiNN-yyyy-mm-dd-HH-MM-SS.h264
    Returns:
        N_int, datetime object
    '''
    d = filename[11:-5].split('-')
    for i in range(len(d)):
        d[i] = int(d[i])

    dtime = dt.datetime(d[1], d[2], d[3], d[4], d[5], d[6])
    return d[0], (dtime - dt.timedelta(hours=time_offset)).date()


def get_dict(conditions_file):
    '''
    Gets dictionary of dictionaries.
    Args:
        conditions_file - path of conditions file
    Returns:
        dictionary indexed by [cameraname][date]
    '''
    in_file = open(conditions_file, 'r')
    in_csv = csv.reader(in_file)
    header = in_csv.next()
    in_dict = {}
    for title in header[1:]:
        in_dict[title] = {}

    for line in in_csv:
        date = dt.datetime.strptime('%010s' % line[0], '%d/%m/%Y').date()
        for i in range(len(line) - 1):
            in_dict[header[i + 1]][date] = line[i + 1]

    in_file.close()
    return in_dict


def get_roi(movie, roi_tup=(0, 0, -1, -1)):
    '''
    Defines interface for determining region of interest for a movie.
    Args:
        movie - string containing path of movie file
        roi_tup - tuple (r0, c0, r1, c1) which gives initial estimate
    Returns:
        r0, c0, r1, c1
    '''
    title_str = 'q:quit e:nextframe wsad:move rf:scale ' + movie
    cap = cv2.VideoCapture(movie)
    h, w = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    r0, c0, r1, c1 = roi_tup
    if (r1 > h) or (c1 > w) or (-1 in (r1, c1)):
        r1 = min(h, w)
        c1 = min(h, w)

    ret, frame = cap.read()

    while ret:
        cv2.imshow(title_str, frame[r0:r1, c0:c1])
        keypress = cv2.waitKey(0)
        if keypress == ord('q'):
            break
        elif keypress == ord('e'):
            # Next frame
            ret, frame = cap.read()
        elif keypress == ord('w'):
            if r0 > 0:
                r0 -= 1
                r1 -= 1
        elif keypress == ord('s'):
            if r0 < r1 - 1:
                r0 += 1
                r1 += 1
            if r1 > h:
                r1 -= 1
                c1 -= 1
        elif keypress == ord('a'):
            if c0 > 0:
                c0 -= 1
                c1 -= 1
        elif keypress == ord('d'):
            if c0 < c1 - 1:
                c0 += 1
                c1 += 1
            if c1 >= w:
                c1 -= 1
                r1 -= 1
        elif keypress == ord('r'):
            if r1 > r0 + 1 and c1 > c0 + 1:
                r1 -= 1
                c1 -= 1
        elif keypress == ord('f'):
            if r1 < h and c1 < w:
                r1 += 1
                c1 += 1

    cv2.destroyAllWindows()
    cap.release()
    return r0, c0, r1, c1


def write_cond_file(movie_dir, cond_file, in_dict, time_offset, roi=True):
    '''
    Writes a csv file in the format:
        movie_filename, condition
    Args:
        movie_dir - directory where movies are stored
        cond_file - path output condition file
        in_dict - dictionary returned from get_dict()
    '''
    if not (cond_file is None):
        out_file = open(cond_file, 'w')

    prev_d = 0, 0
    r0 = c0 = 0
    r1 = c1 = -1
    for f in sorted(os.listdir(movie_dir)):
        try:
            if f[0:11] == 'raspberrypi' and f[-5:] == '.h264':
                d = get_date(f, time_offset)
                if roi:
                    if d != prev_d:
                        r0, c0, r1, c1 = get_roi(os.path.join(movie_dir, f),
                                                 roi_tup=(r0, c0, r1, c1))
                        prev_d = d

                if cond_file is None:
                    print '%s,%s,%s,%s,%s,%s' % (f, in_dict[f[0:13]][d[1]],
                                                 r0, c0, r1, c1)
                else:
                    out_file.write('%s,%s,%s,%s,%s,%s\n' % (
                        f, in_dict[f[0:13]][d[1]], r0, c0, r1, c1))
        except KeyError:
            continue

    if not (cond_file is None):
        out_file.close()


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Generate condition reference')

    parser.add_argument('-t', default=0, type=int, required=False,
                        metavar='HourOffset',
                        help='''Sets a later cut-off time for determining date
                        of video. Required when filming overnight with single
                        condition.''')

    parser.add_argument('-o', default=None, type=str, required=False,
                        metavar='OutputFile',
                        help='Write output to file. Default just prints.')

    parser.add_argument('-r', action='store_true', help='''Write regions of
                        interest (manual input)''')

    parser.add_argument('ConditionsFile', type=str,
                        help='The path of the conditions csv file.')

    parser.add_argument('MovieDir', type=str,
                        help='Directory containing movie files')

    args = parser.parse_args()

    # Create dictionary of dictionaries
    in_dict = get_dict(args.ConditionsFile)

    # Generate output
    write_cond_file(args.MovieDir, args.o, in_dict, args.t, args.r)

if __name__ == '__main__':
    main()
