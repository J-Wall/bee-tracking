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
        X, lengths, df1
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


def fit_and_decode(traj_data, n_components=2,
                   subsample_factor=1, features=['speed', 'rotation'],
                   plot_figs=False, n_bees=None, **kwargs):
    '''
    Fits model to each traj_data, then applies model to decode states.
    Args:
        traj_data - list of paths of training dataset (trajectory csv)
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
    lnp_df_list = []
    state_counts_list = []
    for path in traj_data:
        model, df = fit_from_csv(path, n_components=n_components,
                                 subsample_factor=subsample_factor,
                                 features=features, **kwargs)
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


def get_state_times(df, n_bees=1):
    '''
    Gets xranges and yranges for broken_barh
    Args:
        df - trajectory DataFrame which has been decoded.
    Returns:
    xranges - indexed by bee, then state
    yranges - indexed by bee
    components - hidden states
    '''
    df1 = df.dropna()[['t', 'state']]
    components = df1.state.unique()
    components.sort()
    yranges = [(2. * bee, (2. * bee) + 1.) for bee in range(n_bees)]
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
        s = df1.loc[traj].state.values
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


def plot_states(df, n_bees=1, colors=('red', 'blue', 'yellow', 'green')):
    '''
    Produces figure with plots of position, and state for decoded DataFrame.
    Args:
        df - trajectory Dataframe which has been decoded.
        n_bees - number of bees
        colors - colors to represent each state in.
    Returns:
        figure
    '''
    xranges, yranges, states = get_state_times(df, n_bees=n_bees)
    fig = plt.figure(figsize=(20, 10))
    ax_x = fig.add_subplot(311)
    ax_y = fig.add_subplot(312, sharex=ax_x, sharey=ax_x)
    ax_s = fig.add_subplot(313, sharex=ax_x)

    for traj in df.index.unique():
        ax_x.plot(df.loc[traj].t.iloc[2:], df.loc[traj].x.iloc[2:])
        ax_y.plot(df.loc[traj].t.iloc[2:], df.loc[traj].y.iloc[2:])
    for row in range(n_bees):
        for state in states:
            ax_s.broken_barh(xranges[row][state], yranges[row],
                             facecolors=colors[int(state)], edgecolors='none')

    ax_x.set_xlim(0.0, df.t.max())
    ax_x.set_title('x-coordinate')
    plt.setp(ax_x.get_xticklabels(), visible=False)
    ax_y.set_title('y-coordinate')
    plt.setp(ax_y.get_xticklabels(), visible=False)
    ax_s.set_title('ML path through HMM')
    ax_s.set_ylim(-0.5, 2. * n_bees - 0.5)
    ax_s.set_yticks([2 * bee + 0.5 for bee in range(n_bees)])
    ax_s.set_yticklabels(['' for bee in range(n_bees)])
    ax_s.set_xlabel('time (s)')

    return fig


def main():
    pass

if __name__ == '__main__':
    main()
