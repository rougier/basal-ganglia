# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines


# -----------------------------------------------------------------------------
def figure_H_P(records, GPi, title, filename, save=True, show=False):

    n_session = records.shape[0]
    n_block = records.shape[1]
    # n_trial = records.shape[2]

    figsize = (3*n_block, 6)
    plt.figure(figsize=figsize, facecolor="w")
    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(width=1)

    index = np.array([0, 1])
    width = 1
    color = 'r'
    bar_kw = {'width': 0.95, 'linewidth': 0, 'zorder': 5}
    err_kw = {'zorder': 10, 'fmt': 'none', 'linewidth': 0,
              'elinewidth': 1, 'ecolor': 'k'}

    def histogram(X, mean, sigma, color, alpha):
        plt.bar(X, mean, alpha=alpha, color=color, **bar_kw)
        _, caps, _ = plt.errorbar(X+width/2.0, mean, sigma, **err_kw)
        for cap in caps:
            cap.set_markeredgewidth(1)

    for i in range(n_block):
        color = 'b'
        if not GPi[i]:
            color = 'r'

        P1 = np.squeeze(records["best"][:, i, :25])
        P1 = P1.mean(axis=-1)
        P2 = np.squeeze(records["best"][:, i, -25:])
        P2 = P2.mean(axis=-1)
        mean = P1.mean(), P2.mean()
        std = P1.std(), P2.std()
        histogram(index-width+i*2.5, mean, std, color, 0.45)

    plt.xticks([])
    plt.xlim(-1.5, n_block*2)

    X, Y = np.array([[-1.125, 1.125], [-0.125, -0.125]])
    for i in range(n_block):
        ax.add_line(lines.Line2D(X, Y, lw=1, color='k', clip_on=False))
        X += 2.5

    if len(GPi) == 2:
        s1 = ["OFF", "ON"][GPi[0]]
        s2 = ["OFF", "ON"][GPi[1]]
        plt.xticks([-0.5, 0.0, 0.5, 2.0, 2.5, 3.0],
                   ["25 first\ntrials",
                    "\n\n\nDay 1 (GPi %s)\n" % s1, "25 last\ntrials",
                    "25 first\ntrials",
                    "\n\n\nDay 2 (GPi %s)\n" % s2, "25 last\ntrials"])
    else:
        s1 = ["OFF", "ON"][GPi[0]]
        s2 = ["OFF", "ON"][GPi[1]]
        s3 = ["OFF", "ON"][GPi[2]]
        plt.xticks([-0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 4.5, 5.0, 5.5],
                   ["25 first\ntrials",
                    "\n\n\nDay 1 (GPi %s)\n" % s1, "25 last\ntrials",
                    "25 first\ntrials",
                    "\n\n\nDay 2 (GPi %s)\n" % s2, "25 last\ntrials",
                    "25 first\ntrials",
                    "\n\n\nDay 3 (GPi %s)\n" % s3, "25 last\ntrials"])

    # Custom function to draw the diff bars
    # http://stackoverflow.com/questions/11517986/...
    # ...indicating-the-statistically-significant-difference-in-bar-graph
    def label_diff(X1, X2, Y, text):
        # x = (X1+X2)/2.0
        y = 1.15*Y
        props = {'connectionstyle': 'bar', 'arrowstyle': '-',
                 'shrinkA': 25, 'shrinkB': 25, 'lw': 1}
        ax.annotate(text, xy=((X2+X1)/2., y+0.15), zorder=10, ha='center')
        ax.annotate('', xy=(X1, y), xytext=(X2, y), arrowprops=props)
    # label_diff(-0.5,0.5, 0.85, '***')
    label_diff(0.5, 2.0, 0.825, '***')

    plt.ylim(0.0, 1.2)
    plt.ylabel("Ratio of optimum trials")
    plt.yticks([0, .25, .5, .75, 1.0])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')

    plt.title("%s (model, N=%d)" % (title, n_session))

    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=10)
        os.system("pdfcrop %s %s" % (filename, filename))
    if show:
        plt.show()


# -----------------------------------------------------------------------------
def figure_H_RT(records, GPi, title, filename, save=True, show=False):

    n_session = records.shape[0]
    n_block = records.shape[1]
    # n_trial = records.shape[2]

    figsize = (3*n_block, 6)
    plt.figure(figsize=figsize, facecolor="w")
    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(width=1)

    index = np.array([0, 1])
    width = 1
    color = 'r'
    bar_kw = {'width': 0.95, 'linewidth': 0, 'zorder': 5}
    err_kw = {'zorder': 10, 'fmt': 'none', 'linewidth': 0,
              'elinewidth': 1, 'ecolor': 'k'}

    def histogram(X, mean, sigma, color, alpha):
        plt.bar(X, mean, alpha=alpha, color=color, **bar_kw)
        _, caps, _ = plt.errorbar(X+width/2.0, mean, sigma, **err_kw)
        for cap in caps:
            cap.set_markeredgewidth(1)

    for i in range(n_block):
        color = 'b'
        if not GPi[i]:
            color = 'r'

        P1 = np.squeeze(records["RT"][:, i, :25])
        P1 = P1.mean(axis=-1)
        P2 = np.squeeze(records["RT"][:, i, -25:])
        P2 = P2.mean(axis=-1)
        mean = P1.mean(), P2.mean()
        std = P1.std(), P2.std()
        histogram(index-width+i*2.5, mean, std, color, 0.45)

    plt.xticks([])
    plt.xlim(-1.5, n_block*2)

    X, Y = np.array([[-1.125, 1.125], [-0.125, -0.125]])
    for i in range(n_block):
        ax.add_line(lines.Line2D(X, Y, lw=1, color='k', clip_on=False))
        X += 2.5

    if len(GPi) == 2:
        s1 = ["OFF", "ON"][GPi[0]]
        s2 = ["OFF", "ON"][GPi[1]]
        plt.xticks([-0.5, 0.0, 0.5, 2.0, 2.5, 3.0],
                   ["25 first\ntrials", "\n\n\nDay 1 (GPi %s)\n" % s1,
                    "25 last\ntrials",
                    "25 first\ntrials", "\n\n\nDay 2 (GPi %s)\n" % s2,
                    "25 last\ntrials"])
    else:
        s1 = ["OFF", "ON"][GPi[0]]
        s2 = ["OFF", "ON"][GPi[1]]
        s3 = ["OFF", "ON"][GPi[2]]
        plt.xticks([-0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 4.5, 5.0, 5.5],
                   ["25 first\ntrials", "\n\n\nDay 1 (GPi %s)\n" % s1,
                    "25 last\ntrials",
                    "25 first\ntrials", "\n\n\nDay 2 (GPi %s)\n" % s2,
                    "25 last\ntrials",
                    "25 first\ntrials", "\n\n\nDay 3 (GPi %s)\n" % s3,
                    "25 last\ntrials"])

    plt.ylim(0.0, 1.0)
    plt.ylabel("Reaction time (s)")
    plt.yticks([0, .25, .5, .75, 1.0])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')

    plt.title("%s (model, N=%d)" % (title, n_session))

    plt.tight_layout()
    if save:
        plt.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
    if show:
        plt.show()


# -----------------------------------------------------------------------------
def figure_P(records, GPi, title, filename, save=True, show=False):

    sliding_window = 10
    n_session = records.shape[0]
    n_block = records.shape[1]
    n_trial = records.shape[2]
    figsize = (5*n_block, 4)

    plt.figure(figsize=figsize, facecolor="w")
    ax = plt.subplot(111)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="out")

    # alpha = 0.1

    X = 10+np.arange(n_trial-10)

    for i in range(n_block):
        P = np.squeeze(records["best"][:, i, :])
        color = 'b'
        if not GPi[i]:
            color = 'r'

        running_mean = np.zeros((n_session, n_trial-10))
        for j in range(n_session):
            for k in range(10, n_trial):
                imin, imax = max(k+1-sliding_window, 0), k+1
                running_mean[j, k-10] = P[j, imin:imax].mean()

        M = np.mean(running_mean, axis=0)
        S = np.std(running_mean, axis=0)
        # V = np.var(running_mean, axis=0)
        M1, M2 = M+S, M-S

        plt.plot(X, M, c=color, lw=2)
        plt.fill_between(X, M1, M2, color=color, alpha=0.10)
        plt.plot(X, M1, c=color, lw=.5, alpha=.25)
        plt.plot(X, M2, c=color, lw=.5, alpha=.25)

        # global_mean = np.zeros(n_trial)
        # local_mean = np.zeros(n_trial)

        # for j in range(n_session):
        #     for k in range(n_trial):
        #         imin, imax = max(k+1-sliding_window,0), k+1
        #         global_mean[k] = P[:,imin:imax].mean()
        #         local_mean[k] = P[j,imin:imax].mean()
        #     plt.plot(X, local_mean, c=color, lw=1, alpha=alpha)
        # plt.plot(X, global_mean, c=color, lw=2)
        ax.axvspan(X[1]-11, X[1]-1, color='.975')
        X += n_trial

    plt.xticks([])
    X, Y = np.array([[1, n_trial-1], [-0.025, -0.025]])
    for i in range(n_block):
        ax.add_line(lines.Line2D(X, Y, lw=1, color='k', clip_on=False))
        text = "Day %d, GPi %s, %d trials" % (i+1,
                                              ["OFF", "ON"][GPi[i]], n_trial)
        ax.text((X[0]+X[1])/2, -0.075, text, ha="center", va="top")
        if i < n_block-1:
            ax.axvline(X[1]+1, linewidth=0.5, c='k', alpha=.75)

        X += n_trial

    plt.ylabel("Instantaneous performance\n(sliding window of %d trials)"
               % sliding_window)
    plt.title("%s (model, N=%d)" % (title, n_session))
    plt.xlim(0, n_block*n_trial)
    plt.ylim(0, 1.1)

    if save:
        plt.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
    if show:
        plt.show()


# -----------------------------------------------------------------------------
def figure_P_all(records, GPi, title, filename, save=True, show=False):

    n_session = records.shape[0]
    n_block = records.shape[1]
    n_trial = records.shape[2]
    figsize = (6*n_block, 12)
    plt.figure(figsize=figsize, facecolor="w")

    # Individual trials
    # -------------------------------------------------------------------------
    ax = plt.subplot(3, 1, 1)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="out", color='w')

    X = np.arange(n_block*n_trial)
    R = records["best"].reshape(n_session, n_block*n_trial)
    for i in range(len(records)):
        X1 = X[np.argwhere(R[i])]
        X2 = X[np.argwhere(1-R[i])]
        plt.scatter(1+X1, [1+i, ]*len(X1),
                    s=10, color='k', edgecolor='k', lw=.1)
        plt.scatter(1+X2, [1+i, ]*len(X2),
                    s=10, color='w', edgecolor='k', lw=.1)

    ticks = np.arange(n_trial/2, n_block*n_trial, n_trial)
    labels = ["Day %d, GPi %s" %
              (i+1, ["OFF", "ON"][GPi[i]])
              for i in range(n_block)]
    plt.xticks(ticks, labels)
    plt.yticks(1+np.arange(n_session), fontsize=8)

    for i in range(0, n_block-1):
        ax.axvline((i+1)*n_trial, linewidth=0.5, c='k', alpha=.5)

    x1, x2 = 1, 25
    plt.axvspan(x1, x2, facecolor='0.95', edgecolor="none", zorder=-10)
    plt.plot([x1, x2], [0.5, 0.5], linewidth=1.5, color="0.25")
    plt.text(x1+(x2-x1)/2, 0, 'D1 / 25 first trials', fontsize=8, color="0.25",
             va="top", ha='center', transform=ax.transData)

    x1, x2 = 121-25, 121-1
    plt.axvspan(x1, x2, facecolor='0.95', edgecolor="none", zorder=-10)
    plt.plot([x1, x2], [0.5, 0.5], linewidth=1.5, color="0.25")
    plt.text(x1+(x2-x1)/2, 0, 'D1 / 25 last trials', fontsize=8, color="0.25",
             va="top", ha='center', transform=ax.transData)

    x1, x2 = 120+1-1, 120+25-1
    plt.axvspan(x1, x2, facecolor='0.95', edgecolor="none", zorder=-10)
    plt.plot([x1, x2], [0.5, 0.5], linewidth=1.5, color="0.25")
    plt.text(x1+(x2-x1)/2, 0, 'D2 / 25 first trials', fontsize=8, color="0.25",
             va="top", ha='center', transform=ax.transData)

    x1, x2 = 120+121-25-1, 120+121-1-1
    plt.axvspan(x1, x2, facecolor='0.95', edgecolor="none", zorder=-10)
    plt.plot([x1, x2], [0.5, 0.5], linewidth=1.5, color="0.25")
    plt.text(x1+(x2-x1)/2, 0, 'D2 / 25 last trials', fontsize=8, color="0.25",
             va="top", ha='center', transform=ax.transData)

    
    # plt.title("Individual performance")
    plt.ylabel("Session #")
    plt.xlim(0, n_block*n_trial+1)
    plt.ylim(0.5, len(records)+.5)

    # Instantaneous performance
    # -------------------------------------------------------------------------
    sliding_window = 10
    n_session = records.shape[0]
    n_block = records.shape[1]
    n_trial = records.shape[2]
    figsize = (5*n_block, 4)

    ax = plt.subplot(312)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="out", color="w")

    X = 10+np.arange(n_trial-10)
    for i in range(n_block):
        P = np.squeeze(records["best"][:, i, :])
        color = ["red","blue"][GPi[i]]

        running_mean = np.zeros((n_session, n_trial-10))
        for j in range(n_session):
            for k in range(10, n_trial):
                imin, imax = max(k+1-sliding_window, 0), k+1
                running_mean[j, k-10] = P[j, imin:imax].mean()

        M = np.mean(running_mean, axis=0)
        S = np.std(running_mean, axis=0)
        M1, M2 = M+S, M-S
        plt.plot(X, M, c=color, lw=2)
        plt.fill_between(X, M1, M2, color=color, alpha=0.10)
        plt.plot(X, M1, c=color, lw=.5, alpha=.25)
        plt.plot(X, M2, c=color, lw=.5, alpha=.25)
        ax.axvspan(X[1]-11, X[1]-1, color='.975')
        X += n_trial

    ticks = np.arange(n_trial/2, n_block*n_trial, n_trial)
    labels = ["Day %d, GPi %s" %
              (i+1, ["OFF", "ON"][GPi[i]])
              for i in range(n_block)]
    plt.xticks(ticks, labels)

    for i in range(0, n_block-1):
        ax.axvline((i+1)*n_trial, linewidth=0.5, c='k', alpha=.5)

    plt.ylabel("Mean success rate (sliding window)")
    plt.xlim(0, n_block*n_trial)
    plt.ylim(0, 1.1)

    # Histogram
    # -------------------------------------------------------------------------
    ax = plt.subplot(325)
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(width=1)

    index = np.array([0, 1])
    width = 1
    color = 'r'
    bar_kw = {'width': 0.95, 'linewidth': 0, 'zorder': 5}
    err_kw = {'zorder': 10, 'fmt': 'none', 'linewidth': 0,
              'elinewidth': 1, 'ecolor': 'k'}

    def histogram(X, mean, sigma, color, alpha):
        plt.bar(X, mean, alpha=alpha, color=color, **bar_kw)
        _, caps, _ = plt.errorbar(X+width/2.0, mean, sigma, **err_kw)
        for cap in caps:
            cap.set_markeredgewidth(1)

    for i in range(n_block):
        color = ["red","blue"][GPi[i]]
        
        P1 = np.squeeze(records["best"][:, i, :25])
        P1 = P1.mean(axis=-1)
        P2 = np.squeeze(records["best"][:, i, -25:])
        P2 = P2.mean(axis=-1)
        mean = P1.mean(), P2.mean()
        std = P1.std(), P2.std()
        histogram(index-width+i*2.5, mean, std, color, 0.45)

    plt.xticks([])
    plt.xlim(-1.5, n_block*2)

    X, Y = np.array([[-1.125, 1.125], [-0.175, -0.175]])
    for i in range(n_block):
        ax.add_line(lines.Line2D(X, Y, lw=1, color='k', clip_on=False))
        X += 2.5

    s1 = ["OFF", "ON"][GPi[0]]
    s2 = ["OFF", "ON"][GPi[1]]
    plt.xticks([-0.5, 0.0, 0.5, 2.0, 2.5, 3.0],
               ["25 first\ntrials",
                "\n\n\nDay 1, GPi %s\n" % s1, "25 last\ntrials",
                "25 first\ntrials",
                "\n\n\nDay 2, GPi %s\n" % s2, "25 last\ntrials"])

    plt.ylim(0.0, 1.2)
    plt.ylabel("Mean success rate (block)")
    plt.yticks([0, .25, .5, .75, 1.0])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')

    # Mean values
    # -------------------------------------------------------------------------
    ax = plt.subplot(326)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    #ax.yaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="out", color="white")

    X = np.arange(n_trial)
    for i in range(n_block):
        V = np.squeeze(records["value"][:, i, :])
        color = ["red","blue"][GPi[i]]

        for j in [0, 1]:
            M = V[..., j].mean(axis=0)
            S = V[..., j].std(axis=0)
            plt.plot(X, M, color=color, lw=2)
            plt.fill_between(X, M-S, M+S, color=color, alpha=0.10)
            plt.plot(X, M-S, c=color, lw=.5, alpha=.25)
            plt.plot(X, M+S, c=color, lw=.5, alpha=.25)
            if i == n_block-1:
                print("V=%.2f ± %.2f" % (M[-1], S[-1]))
        X += n_trial

    ticks = np.arange(n_trial/2, n_block*n_trial, n_trial)
    labels = ["Day %d, GPi %s" %
              (i+1, ["OFF", "ON"][GPi[i]])
              for i in range(n_block)]
    plt.xticks(ticks, labels)

    for i in range(0, n_block-1):
        ax.axvline((i+1)*n_trial, linewidth=0.5, c='k', alpha=.5)

    plt.ylabel("Internal value of stimuli")
    plt.xlim(0, n_block*n_trial)
    plt.ylim(0, 1.2)
    plt.yticks([0, .25, .5, .75, 1.0])
    
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    if show:
        plt.show()


    
# -----------------------------------------------------------------------------
def figure_P_individual(records, GPi, title, filename, save=True, show=False):

    n_session = records.shape[0]
    n_block = records.shape[1]
    n_trial = records.shape[2]
    figsize = (6*n_block, 5)

    plt.figure(figsize=figsize, facecolor="w")
    ax = plt.subplot(111)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="out", color='w')

    X = np.arange(n_block*n_trial)
    # Y = np.ones(n_block*n_trial)
    # C = np.ones((n_block*n_trial, 3))
    # R = records["best"].reshape(n_session, n_block*n_trial, 1)
    R = records["best"].reshape(n_session, n_block*n_trial)

    for i in range(len(records)):
        X1 = X[np.argwhere(R[i])]
        X2 = X[np.argwhere(1-R[i])]
        plt.scatter(1+X1, [1+i, ]*len(X1),
                    s=10, color='k', edgecolor='k', lw=.1)
        plt.scatter(1+X2, [1+i, ]*len(X2),
                    s=10, color='w', edgecolor='k', lw=.1)
        # plt.scatter(X, Y+i, s=2.5, color=C*(1-R[i])*.85)

    ticks = np.arange(n_trial/2, n_block*n_trial, n_trial)
    labels = ["Day %d, GPi %s, %d trials" %
              (i+1, ["OFF", "ON"][GPi[i]], n_trial)
              for i in range(n_block)]
    plt.xticks(ticks, labels)
    plt.yticks(1+np.arange(n_session), fontsize=8)

    for i in range(0, n_block-1):
        ax.axvline((i+1)*n_trial, linewidth=0.5, c='k', alpha=.5)

    x1,x2 = 1, 25
    plt.axvspan(x1, x2, facecolor='0.95', edgecolor="none", zorder=-10)
    plt.plot([x1, x2], [0.5, 0.5], linewidth=1.5, color="0.5")
    plt.text(x1+(x2-x1)/2, 0, '25 first trials', fontsize=10, color="0.5",
             va="top", ha='center', transform=ax.transData)

    x1,x2 = 121-25, 121-1
    plt.axvspan(x1, x2, facecolor='0.95', edgecolor="none", zorder=-10)
    plt.plot([x1, x2], [0.5, 0.5], linewidth=1.5, color="0.5")
    plt.text(x1+(x2-x1)/2, 0, '25 last trials', fontsize=10, color="0.5",
             va="top", ha='center', transform=ax.transData)

    

    plt.ylabel("Session #")
    plt.title("Individual performance")
    plt.xlim(0, n_block*n_trial+1)
    plt.ylim(0.5, len(records)+.5)
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    if show:
        plt.show()


# -----------------------------------------------------------------------------
def figure_RT_individual(records, GPi, title, filename, save=True, show=False):

    n_session = records.shape[0]
    n_block = records.shape[1]
    n_trial = records.shape[2]
    figsize = (6*n_block, 6)

    plt.figure(figsize=figsize, facecolor="w")
    ax = plt.subplot(111)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="out")

    X = np.arange(n_block*n_trial)
    R = records["RT"].reshape(n_session, n_block*n_trial, 1)

    RT_max = R.ravel().max()
    R = 0.75 * R/RT_max
    for i in range(len(records)):
        plt.bar(left=X, height=R[i], bottom=i+1, width=1.0,
                color='k', edgecolor='w', linewidth=.5)

    ticks = np.arange(n_trial/2, n_block*n_trial, n_trial)
    labels = ["Day %d, GPi %s, %d trials" %
              (i+1, ["OFF", "ON"][GPi[i]], n_trial)
              for i in range(n_block)]
    plt.xticks(ticks, labels)
    for i in range(0, n_block-1):
        ax.axvline((i+1)*n_trial, linewidth=0.5, c='k', alpha=.5)

    plt.ylabel("Session #")
    plt.title("Individual (normalized) response time")
    plt.xlim(0, n_block*n_trial)
    plt.ylim(0, len(records)+1)
    plt.tight_layout()

    if save:
        plt.savefig(filename)
    if show:
        plt.show()


# -----------------------------------------------------------------------------
def figure_RT(records, GPi, title, filename, save=True, show=False):

    n_session = records.shape[0]
    n_block = records.shape[1]
    n_trial = records.shape[2]
    figsize = (5*n_block, 4)

    plt.figure(figsize=figsize, facecolor="w")
    ax = plt.subplot(111)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="out")

    alpha = 0.05
    X = np.arange(n_trial)
    for i in range(n_block):
        RT = np.squeeze(records["RT"][:, i, :])
        color = 'b'
        if not GPi[i]:
            color = 'r'
        for j in range(n_session):
            plt.scatter(X, RT[j], 20, color=color, alpha=alpha)
        plt.plot(X, RT.mean(axis=0), color=color, lw=2)
        X += n_trial

    plt.xticks([])
    X, Y = np.array([[1, n_trial-1], [-0.025, -0.025]])
    for i in range(n_block):
        ax.add_line(lines.Line2D(X, Y, lw=1, color='k', clip_on=False))
        text = "Day %d, GPi %s, %d trials" % (
            i+1, ["OFF", "ON"][GPi[i]], n_trial)
        ax.text((X[0]+X[1])/2, -0.075, text, ha="center", va="top")
        if i < n_block-1:
            ax.axvline(X[1]+1, linewidth=0.5, c='k', alpha=.75)
        X += n_trial

    plt.ylabel("Reaction time (s)", fontsize=14)
    plt.title("%s (model, N=%d)" % (title, n_session))
    plt.xlim(0, n_block*n_trial)
    plt.ylim(0, 2.0)

    if save:
        plt.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
    if show:
        plt.show()


# -----------------------------------------------------------------------------
def figure_V(records, GPi, title, filename, save=True, show=False):

    n_session = records.shape[0]
    n_block = records.shape[1]
    n_trial = records.shape[2]
    figsize = (5*n_block, 4)

    plt.figure(figsize=figsize, facecolor="w")
    ax = plt.subplot(111)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="out")

    # alpha = 0.05
    X = np.arange(n_trial)
    for i in range(n_block):
        V = np.squeeze(records["value"][:, i, :])

        color = 'b'
        if not GPi[i]:
            color = 'r'

        for j in [0, 1]:
            M = V[..., j].mean(axis=0)
            S = V[..., j].std(axis=0)
            plt.plot(X, M, color=color, lw=2)
            plt.fill_between(X, M-S, M+S, color=color, alpha=0.10)
            plt.plot(X, M-S, c=color, lw=.5, alpha=.25)
            plt.plot(X, M+S, c=color, lw=.5, alpha=.25)

            if i == n_block-1:
                # plt.text(X[-1]+1, M[-1], "%.2f ± %.2f" % (M[-1],S[-1]),
                # ha="left", va="center", color="r")
                print("V=%.2f ± %.2f" % (M[-1], S[-1]))
        X += n_trial

    plt.xticks([])
    X, Y = np.array([[1, n_trial-1], [-0.025, -0.025]])
    for i in range(n_block):
        ax.add_line(lines.Line2D(X, Y, lw=1, color='k', clip_on=False))
        text = "Day %d, GPi %s, %d trials" % (
            i+1, ["OFF", "ON"][GPi[i]], n_trial)
        ax.text((X[0]+X[1])/2, -0.075, text, ha="center", va="top")
        if i < n_block-1:
            ax.axvline(X[1]+1, linewidth=0.5, c='k', alpha=.75)
        X += n_trial

    plt.ylabel("Estimated internal value of stimuli", fontsize=14)
    plt.title("%s (model, N=%d)" % (title, n_session))
    plt.xlim(0, n_block*n_trial)
    plt.ylim(0, 1.0)

    if save:
        plt.savefig(filename)
        os.system("pdfcrop %s %s" % (filename, filename))
    if show:
        plt.show()
