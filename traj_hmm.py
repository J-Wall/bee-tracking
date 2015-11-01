# Fits a Gaussian Hidden Markov Model to determine underlying behavioural
# regimes.

from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from post_process import subsample, calculate_velocity


def get_features(df, sub_sample=1, features=['speed', 'rotation']):
    '''
    Extract feature matrix from trajectory DataFrame for HMM fitting
    Args:
        df - trajectory DataFrame
        sub_sample - subsample df. Default, don't subsample
        features - features to include. Default ['speed', 'rotation']
    Returns:
        X, lengths, df1
        X - features matrix
        lengths - lengths of samples
        df1 - subsampled df
    '''
    df1 = subsample(df, 5)
    l = np.array([len(df1.loc[traj]) for traj in df1.index.unique()])
    trajs = df1.index.unique()[l >= 4]  # Too short trajectories not be included
    calculate_velocity(df1)
    feature_list = []
    for traj in trajs:
        feature_list.append(df1.loc[traj][features].values[2:])

    return (np.vstack(feature_list), np.array([len(a) for a in feature_list]),
            df1.loc[trajs])


def fit_hmm(df, n_components, sub_sample=1, features=['speed', 'rotation'],
            **kwargs):
    '''
    Fits a Gaussian HMM to the velocity data
    Args:
        df - dataframe containing positional data to be processed
        n_components - number of hidden states
        sub_sample - subsample DataFrame
        features - features to use in model fitting
        **kwargs passed to GaussianHMM
    Returns:
        model, df1
    '''
    X, lengths, df1 = get_features(df, sub_sample=sub_sample, features=features)
    model = GaussianHMM(n_components, **kwargs)
    model.fit(X, lengths=lengths)

    return model, df1


def decode_states(df, model, features=['speed', 'rotation']):
    '''
    Decode each trajectory and add a 'state' column to df (inplace).
    Args:
        df - trajectory DataFrame
        model - model to decode with
        features - features used for model fitting
    Returns:
        DataFrame indexed by 'traj' with values 'logprob' (logprob of path)
    '''
    lnp_list = []
    df['state'] = np.nan
    for traj in df.index.unique():
        lnp, states = model.decode(df.loc[traj][features].values[2:])
        lnp_list.append([traj, lnp])
        df.loc[traj]['state'].iloc[2:] = states

    lnp_df = pd.DataFrame(lnp_list, columns=['traj', 'lnp']).set_index('traj')

    return lnp_df


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
    fig = plt.figure()
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
