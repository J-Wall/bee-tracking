# Fits a Gaussian Hidden Markov Model to determine underlying behavioural
# regimes.

from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from post_process import subsample, calculate_velocity


def sub_calc(df, subsample_factor):
    '''
    Convenience function to subsample and calculate velocity
    Args:
        df - input DataFrame
        subsample_factor - factor to pass to subsample
    Returns:
        subsampled and velocity-calculated DataFrame
    '''
    df1 = subsample(df, subsample_factor)
    calculate_velocity(df1)
    return df1


def get_features(df, features=['speed', 'rotation']):
    '''
    Extract feature matrix from trajectory DataFrame for HMM fitting
    Args:
        df - trajectory DataFrame
        features - features to include. Default ['speed', 'rotation']
    Returns:
        X, lengths
        X - features matrix
        lengths - lengths of samples
    '''
    feature_list = []
    for traj in df.index.unique():
        feature_list.append(df.loc[traj][features].values[2:])

    return (np.vstack(feature_list), np.array([len(a) for a in feature_list]))


def fit_hmm(df, n_components, features=['speed', 'rotation'],
            **kwargs):
    '''
    Fits a Gaussian HMM to the velocity data
    Args:
        df - dataframe containing positional data to be processed
        n_components - number of hidden states
        features - features to use in model fitting
        **kwargs passed to GaussianHMM
    Returns:
        model
    '''
    X, lengths = get_features(df, features=features)
    model = GaussianHMM(n_components, **kwargs)
    model.fit(X, lengths=lengths)

    return model


def decode_states(df, model, features=['speed', 'rotation']):
    '''
    Decode each trajectory and add a 'state' column to df (inplace).
    Args:
        df - trajectory DataFrame
        model - model to decode with
        features - features used for model fitting
    Returns:
        DataFrame indexed by 'traj' with values 'logprob' (logprob of path)
        'state' columns is added to df in place.
    '''
    lnp_list = []
    df['state'] = np.nan
    for traj in df.index.unique():
        lnp, states = model.decode(df.loc[traj][features].values[2:])
        lnp_list.append([traj, lnp])
        df.loc[traj]['state'].iloc[2:] = states

    lnp_df = pd.DataFrame(lnp_list, columns=['traj', 'lnp']).set_index('traj')

    return lnp_df


def thresh_decode(df, threshold, feature='speed'):
    '''
    Simplified one parameter model to determine states.
    Args:
        df - trajectory DataFrame to threshold
        threshold - value to threshold at
        feature - feature to threshold over. Default: 'speed'
    Returns:
        None. column 'thresh' is added to df in place.
    '''
    df['thresh'] = np.nan
    df.thresh[df[feature] > threshold] = 1
    df.thresh[df[feature] <= threshold] = 0

    return None


def features_from_csv(path, features=['speed', 'rotation'], subsample_factor=1):
    '''
    Load trajectories from csv, subsample and get_features
    Args:
        path - path of trajectory DataFrame
        features - features to get
        subsample_factor - subsample data
    Returns:
        X, lengths, trajectory DataFrame
    '''
    print 'Loading %s' % path
    df = pd.read_csv(path, index_col='traj')
    print 'Subsampling... Factor: %d' % subsample_factor
    df1 = sub_calc(df, subsample_factor)
    print 'Getting features...'
    return get_features(df1, features=features)


def fit_from_csv(path, n_components=2, subsample_factor=1,
                 features=['speed', 'rotation'], **kwargs):
    '''
    Load trajectories from csv, subsample, and fit HMM
    Args:
        path - path of csv file
        n_components - number of hidden states (default 2)
        subsample_factor - subsample data
        features - columns to fit data to
        **kwargs passed to GaussianHMM
    Returns:
        model, DataFrame
    '''
    print 'Loading %s' % path
    df = pd.read_csv(path, index_col='traj')
    print 'Subsampling... Factor: %d' % subsample_factor
    df1 = sub_calc(df, subsample_factor)
    print 'Fitting model...'
    model = fit_hmm(df1, n_components, features=features, **kwargs)
    return model, df1


def decode_from_csv(path, model, subsample_factor=1,
                    features=['speed', 'rotation']):
    '''
    Load trajectories from csv, subsample, and fit HMM
    Args:
        path - path of csv file
        model - HMM model which has been fit
        subsample_factor - subsample data
        features - columns which model has been fit to
    Returns:
        df - trajectory DataFrame
        lnp_df - log probability DataFrame
    '''
    print 'Loading %s' % path
    df = pd.read_csv(path, index_col='traj')
    print 'Subsampling... Factor: %d' % subsample_factor
    df1 = sub_calc(df, subsample_factor)
    print 'Running Viterbi algorithm...'
    lnp_df = decode_states(df1, model, features=features)
    return df1, lnp_df


def count_states(df):
    '''
    Convenience function to count states in decoded DataFrame.
    Args:
        df - decoded DataFrame
    Returns:
        array with shape (n_components, )
    '''
    return np.bincount(df.state.dropna().astype(np.int64))


def fit_and_decode(traj_data, concat_fit=True, n_components=2,
                   subsample_factor=1, features=['speed', 'rotation'],
                   plot_figs=False, n_bees=None, **kwargs):
    '''
    Fits model to paths in traj_data, then applies model to decode states.
    Args:
        traj_data - list of paths of training dataset (trajectory csv)
        concat_fit - Default True. fit model to concatenated data
        n_components - number of hidden states
        subsample_factor - subsample factor to apply to all files
        features - columns to fit model to
        plot_figs - False or path str. plot figs to a PDF file. Must specify
                    n_bees.
        n_bees - number of bees (required for plotting)
        **kwargs passed to GaussianHMM
    Returns:
        model - fitted model
        lnp_df_list - list of log probabilities of ML paths through HMM
        state_counts - array of shape (len(apply_to), n_components)
    '''
    if isinstance(plot_figs, str):
        pdf_file = PdfPages(plot_figs)

    if concat_fit is True:
        print 'Fitting model...'
        model = fit_batch(traj_data, n_components=n_components,
                          subsample_factor=subsample_factor,
                          features=features, **kwargs)

    lnp_df_list = []
    state_counts_list = []
    for path in traj_data:

        if concat_fit is True:
            print 'Loading %s' % path
            df = pd.read_csv(path, index_col='traj')
            print 'Subsampling... Factor: %d' % subsample_factor
            df = sub_calc(df, subsample_factor)

        elif concat_fit is False:
            model, df = fit_from_csv(path, n_components=n_components,
                                     subsample_factor=subsample_factor,
                                     features=features, **kwargs)

        print 'Decoding...'
        lnp_df = decode_states(df, model, features=features)
        lnp_df_list.append(lnp_df)
        state_counts_list.append(count_states(df))
        if isinstance(plot_figs, str):
            print 'Producing figure...'
            fig = plot_states(df, n_bees=n_bees)
            fig.suptitle(path)
            pdf_file.savefig()
            plt.close(fig)

    if isinstance(plot_figs, str):
        pdf_file.close()
    return model, lnp_df_list, np.vstack(state_counts_list)


def fit_batch(traj_data, n_components=2, subsample_factor=1,
              features=['speed', 'rotation'], **kwargs):
    '''
    Fits model to concatenated traj_data
    Args:
        traj_data - list of paths of training dataset (trajectory csv)
        n_components - number of hidden states
        subsample_factor - subsample factor to apply to all files
        features - columns to fit model to
        **kwargs passed to GaussianHMM
    Returns:
        model - fitted model
    '''
    # Concatenate data
    feature_list = []
    lengths_list = []
    for path in traj_data:
        X, l = features_from_csv(path, features=features,
                                 subsample_factor=subsample_factor)
        feature_list.append(X)
        lengths_list.append(l)
    print 'Concatenating features...'
    X = np.vstack(feature_list)
    l = np.hstack(lengths_list)

    # Fit HMM
    print 'Fitting model...'
    model = GaussianHMM(n_components, **kwargs)
    model.fit(X, lengths=l)

    return model


def decode_batch(traj_data, model, subsample_factor=1,
                 features=['speed', 'rotation']):
    '''
    Calls decode_from_csv on a batch of files
    Args:
        traj_data - list of trajectory csv paths
        model - fitted HMM to decode data with
        subsample_factors - subsample data
        features - features to decode (must be same as features fit for HMM)
    Returns:
        df_list, lnp_df_list
    '''
    df_list = []
    lnp_df_list = []
    for path in traj_data:
        df, lnp_df = decode_from_csv(path, model,
                                     subsample_factor=subsample_factor,
                                     features=features)
        df_list.append(df)
        lnp_df.append(df)

    return df_list, lnp_df_list


def get_state_times(df, n_bees=1, state_col='state'):
    '''
    Gets xranges and yranges for broken_barh
    Args:
        df - trajectory DataFrame which has been decoded.
    Returns:
    xranges - indexed by bee, then state
    yranges - indexed by bee
    state_col - 'state' or 'thresh'
    '''
    assert state_col in {'state', 'thresh'}
    df1 = df.dropna()[['t', state_col]]
    components = df1[state_col].unique()
    components.sort()
    if state_col == 'state':
        yranges = [(2. * bee, 1.) for bee in range(n_bees)]
    elif state_col == 'thresh':
        yranges = [(-2., 1.)]
    bees = [{state: np.array((0, 2)) for state in components}
            for bee in range(n_bees)]

    # get bee_dict (assigns traj to 'bees')
    if n_bees == 1:
        bee_dict = {traj: 0 for traj in df1.index.unique()}
    else:
        curr_traj = [df1.index.unique()[bee] for bee in range(n_bees)]
        curr_endt = [df1.loc[traj].iloc[-1].t for traj in curr_traj]
        bee_dict = {curr_traj[i]: i for i in range(n_bees)}
        for traj in df1.index.unique()[n_bees:]:
            replace_idx = np.argmin(curr_endt)
            curr_traj[replace_idx] = traj
            curr_endt[replace_idx] = df1.loc[traj].iloc[-1].t
            bee_dict[traj] = replace_idx

    # update bees and yranges
    for traj in df1.index.unique():
        t = df1.loc[traj].t.values
        s = df1.loc[traj][state_col].values
        starti = np.hstack((0, np.where(s[:-1] != s[1:])[0] + 1))
        widths = np.hstack((t[starti[1:]] - t[starti[:-1]],
                            t[-1] - t[starti[-1]]))
        startt = t[starti]
        comp_a = s[starti]
        traj_xranges = np.column_stack((startt, widths))
        for state in components:
            bees[bee_dict[traj]][state] = np.vstack(
                (bees[bee_dict[traj]][state], traj_xranges[comp_a == state, :]))

    return bees, yranges, components


def plot_states(df, n_bees=1, colors=('red', 'blue', 'yellow', 'green'),
                plot_thresh_model=False):
    '''
    Produces figure with plots of position, and state for decoded DataFrame.
    Args:
        df - trajectory Dataframe which has been decoded.
        n_bees - number of bees
        colors - colors to represent each state in.
        plot_thresh_model - Boolean of whether to plot thresh model below HMM
    Returns:
        figure
    '''
    xranges, yranges, states = get_state_times(df, n_bees=n_bees)
    fig = plt.figure(figsize=(20, 10))
    ax_x = fig.add_subplot(711)
    ax_y = fig.add_subplot(712, sharex=ax_x, sharey=ax_x)
    ax_speed = fig.add_subplot(713, sharex=ax_x)
    ax_angle = fig.add_subplot(714, sharex=ax_x)
    ax_rotation = fig.add_subplot(715, sharex=ax_x)
    ax_d_mid = fig.add_subplot(716, sharex=ax_x)
    ax_s = fig.add_subplot(717, sharex=ax_x)

    for traj in df.index.unique():
        ax_x.plot(df.loc[traj].t.iloc[2:], df.loc[traj].x.iloc[2:])
        ax_y.plot(df.loc[traj].t.iloc[2:], df.loc[traj].y.iloc[2:])
        ax_speed.plot(df.loc[traj].t.iloc[2:], df.loc[traj].speed.iloc[2:])
        ax_angle.plot(df.loc[traj].t.iloc[2:], df.loc[traj].angle.iloc[2:])
        ax_rotation.plot(df.loc[traj].t.iloc[2:],
                         df.loc[traj].rotation.iloc[2:])
        ax_d_mid.plot(df.loc[traj].t.iloc[2:], df.loc[traj].d_mid.iloc[2:])
    for row in range(n_bees):
        for state in states:
            ax_s.broken_barh(xranges[row][state], yranges[row],
                             facecolors=colors[int(state)], edgecolors='none')

    if plot_thresh_model is True:
        t_xrang, t_yrang, t_states = get_state_times(df, state_col='thresh')
        for state in t_states:
            ax_s.broken_barh(t_xrang[0][state], t_yrang[0],
                             facecolors=colors[int(state)], edgecolors='none')

    ax_x.set_xlim(0.0, df.t.max())
    ax_x.set_ylabel('x-coordinate\n(mm)', rotation=0, ha='right', va='center')
    ax_y.set_ylabel('y-coordinate\n(mm)', rotation=0, ha='right', va='center')
    ax_speed.set_ylabel('speed\n(mm/s)', rotation=0, ha='right', va='center')
    ax_angle.set_ylabel('movement\nangle (rad)', rotation=0, ha='right',
                        va='center')
    ax_rotation.set_ylabel('rotation\n(rad/s)', rotation=0, ha='right',
                           va='center')
    ax_d_mid.set_ylabel('distance from\ncentre (mm)', rotation=0, ha='right',
                        va='center')
    ax_s.set_ylabel('movement\nregime\nmodels', rotation=0, ha='right',
                    va='center')
    if plot_thresh_model is True:
        ax_s.set_yticks([-1.5] + [2 * bee + 0.5 for bee in range(n_bees)])
        ax_s.set_ylim(-2.5, 2. * n_bees - 0.5)
        ax_s.set_yticklabels(['Thresh'] + ['HMM' for bee in range(n_bees)])
    else:
        ax_s.set_yticks([2 * bee + 0.5 for bee in range(n_bees)])
        ax_s.set_ylim(-0.5, 2. * n_bees - 0.5)
        ax_s.set_yticklabels(['HMM' for bee in range(n_bees)])
    ax_s.set_xlabel('time (s)')
    for ax in fig.axes[:-1]:
        plt.setp(ax.get_yticklabels()[0], visible=False)
        plt.setp(ax.get_yticklabels()[-1], visible=False)

    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.subplots_adjust(hspace=0)
    return fig


def state_props(data, thresh, feature='speed', subsample_factor=1,
                trange=(0., 86400.), bins=144):
    '''
    Simeseries of proportion of bees who are active in each timebin for
    each condition.
    Args:
        data - list of lists of trajectory data. First index is for different
                conditions.
        thresh - threshold values
        feature - feature to threshold
        subsample_factor - subsample trajectory data
        trange - time range to plot. default 24hours
        bins - number of bins - default 144 (10min if 24h range)
    Returns:
        time_array, list of prop_1 arrays
    '''
    prop_1_list = []
    for cond in range(len(data)):
        df_list = []
        for path in data[cond]:
            print 'Loading %s' % path
            df = pd.read_csv(path, index_col='traj')
            print 'Subsampling... Factor: %d' % subsample_factor
            df = sub_calc(df, subsample_factor)
            print 'Thresholding...'
            thresh_decode(df, thresh, feature=feature)
            df_list.append(df)

        print 'Calculating timeseries...'
        t0_times = np.hstack([df1.t[df1.thresh == 0] for df1 in df_list])
        t1_times = np.hstack([df1.t[df1.thresh == 1] for df1 in df_list])
        h_t0, b = np.histogram(t0_times, range=trange, bins=bins)
        h_t1, b = np.histogram(t1_times, range=trange, bins=bins)
        prop_1 = h_t1.astype(np.float64) / (h_t0 + h_t1)
        prop_1_list.append(prop_1)

    time_array = b[:-1] + 0.5 * (b[1:] - b[:-1])

    return time_array, prop_1_list


def main():
    pass

if __name__ == '__main__':
    main()
